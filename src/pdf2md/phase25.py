"""Phase 2.5: post-conversion figure rescue.

After Phase 2 writes the .qmd, two things can still go wrong:
(a) Leftover FIG_N tokens the regex didn't catch (bare parens, partial HTML attrs)
(b) Detected figures the converter never referenced at all

Both are resolved deterministically, no LLM cost:
Phase 2.5a rewrites leftover FIG_N tokens.
Phase 2.5b places each unreferenced figure at its transcribed caption (the
converter drops the ![](FIG_n) token near a long document's end but still writes
the caption as text), appending any whose caption never reached the body.
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# ── 2.5a: Deterministic leftover-token resolution ──────────────────────────

# Catch any remaining FIG_N reference in any form:
#   (FIG_2), (FIG_2), src="FIG_15, FIG_3:, etc.
_BARE_FIG_RE = re.compile(r'(?<![a-zA-Z])FIG_(\d+)(?![/\w-])')


def resolve_leftover_fig_tokens(qmd_text: str, figures: list,
                                 media_dirname: str, stem: str) -> tuple:
    """Catch FIG_N tokens that slipped through the main resolver.
    Returns (new_text, count_resolved)."""
    by_num = {}
    for f in figures:
        fid = f.get('fig_id', '')
        m = re.match(r'FIG_(\d+)', fid)
        if m:
            by_num[int(m.group(1))] = f

    resolved = 0

    def _repl(m):
        nonlocal resolved
        num = int(m.group(1))
        fig = by_num.get(num)
        if fig and fig.get('file'):
            resolved += 1
            cap = fig.get('caption') or f'Figure {num}'
            # Escape brackets for markdown image syntax
            cap = cap.replace('[', '\\[').replace(']', '\\]')
            return f'![{cap}]({media_dirname}/{fig["file"]})'
        return m.group(0)  # leave unknown tokens as-is

    new_text = _BARE_FIG_RE.sub(_repl, qmd_text)
    if resolved:
        log.info('Phase 2.5a: resolved %d leftover FIG_N token(s)', resolved)
    return new_text, resolved


# ── 2.5b: Deterministic caption-anchored placement of unreferenced figures ──
#
# When the converter drops a figure near the END of a long document (attention
# drift), it still transcribes the figure's CAPTION as body text — it just omits
# the ![](FIG_n) token. So the figure belongs exactly where its caption already
# sits. Rewrite that caption line in place as an image reference. No LLM: the old
# per-figure LLM rescue saw only the first 8000 chars of the .qmd, so a page-42
# figure was placed into beginning-of-document context and got cover-page text
# inserted after the frontmatter (duplicated captions, garbage).

_CAP_LABEL_RE = re.compile(r'^\s*(?:fig(?:ure)?\.?|table)\s*\d+', re.I)
_CAP_NUM_RE = re.compile(r'(?:fig(?:ure)?\.?|table)\s*(\d+)', re.I)


def _place_figures_by_caption(qmd_text, unreferenced, media_dirname):
    """Rewrite each unreferenced figure's transcribed caption line into an image
    reference, in place. Returns (new_text, placed_fig_ids). A caption line is the
    body line that starts with the figure's label ('Figure 10 …') and best matches
    the detected caption text — a threshold on shared tokens keeps a cross-reference
    sentence ('see Figure 10') from being mistaken for the caption."""
    from .verify.textutil import normalize

    lines = qmd_text.split('\n')
    cand = [i for i, ln in enumerate(lines)
            if _CAP_LABEL_RE.match(ln) and not ln.lstrip().startswith('![')]
    placed, used = [], set()
    for fig in unreferenced:
        cap = (fig.get('caption') or '').strip()
        cap_tokens = set(normalize(cap).split())
        if len(cap_tokens) < 4:
            continue                        # too short to anchor confidently
        nm = _CAP_NUM_RE.match(cap)
        num = nm.group(1) if nm else None
        best_i, best_score = None, 0
        for i in cand:
            if i in used:
                continue
            lm = _CAP_NUM_RE.match(lines[i].strip())
            if num and lm and lm.group(1) != num:
                continue                    # a different figure number
            score = len(cap_tokens & set(normalize(lines[i]).split()))
            if score > best_score:
                best_i, best_score = i, score
        if best_i is not None and best_score >= max(4, len(cap_tokens) // 2):
            cap_line = lines[best_i].strip()
            alt = cap_line.replace('[', '\\[').replace(']', '\\]')
            lines[best_i] = f'![{alt}]({media_dirname}/{fig["file"]})'
            used.add(best_i)
            placed.append(fig.get('fig_id', ''))
    return '\n'.join(lines), placed


def _append_unplaced_figures(qmd_text, unplaced, media_dirname):
    """Safety net for figures whose caption never made it into the body: append the
    image at the END of the document (near where these late figures belong) rather
    than dropping the content or splicing it mid-text. Returns (new_text, count)."""
    if not unplaced:
        return qmd_text, 0
    blocks = []
    for fig in unplaced:
        cap = (fig.get('caption')
               or f"Figure {fig.get('fig_id', '').replace('FIG_', '')}").strip()
        alt = cap.replace('[', '\\[').replace(']', '\\]')
        blocks.append(f'![{alt}]({media_dirname}/{fig["file"]})')
    return qmd_text.rstrip() + '\n\n' + '\n\n'.join(blocks) + '\n', len(blocks)


def run_phase25(qmd_path: Path, detections_path: Path,
                working_pdf: Path | None = None,
                api_key: str = '',
                rescue_model: str = 'google/gemini-2.5-flash',
                timeout: int = 120) -> dict:
    """Run Phase 2.5 figure rescue on a completed .qmd.

    Args:
        qmd_path: path to the .qmd produced by Phase 2
        detections_path: path to detections.json from Phase 1
        working_pdf: the chrome-stripped working PDF (for page image extraction)
        api_key: OpenRouter API key (needed for 2.5b)
        rescue_model: model for 2.5b insertion calls (default: flash, fast+cheap)

    Returns summary dict with counts.
    """
    if not qmd_path.exists():
        return {'error': f'{qmd_path} not found'}

    detections = json.loads(detections_path.read_text()) if detections_path.exists() else {}
    figures = detections.get('figures', [])
    if not figures:
        return {'resolved_2_5a': 0, 'inserted_2_5b': 0, 'note': 'no figures to rescue'}

    stem = qmd_path.stem
    media_dirname = f'{stem}-media'
    qmd_text = qmd_path.read_text(encoding='utf-8')

    # 2.5a: catch leftover tokens
    new_text, resolved = resolve_leftover_fig_tokens(qmd_text, figures,
                                                      media_dirname, stem)
    if resolved:
        qmd_path.write_text(new_text, encoding='utf-8')
        qmd_text = new_text

    # 2.5b: find and insert unreferenced figures
    # Unreferenced = in detections but the image file is not referenced in the
    # visible (non-comment) portion of the qmd. HTML comments (<!-- ... -->) may
    # list unreferenced figures with their file paths — exclude those.
    visible_text = re.sub(r'<!--.*?-->', '', qmd_text, flags=re.DOTALL)
    unreferenced = []
    for fig in figures:
        fid = fig.get('fig_id', '')
        ffile = fig.get('file', '')
        # A figure is placed if either its FIG_id token is resolved OR its
        # image file is referenced in visible text
        if fid and ffile:
            if fid in visible_text or ffile in visible_text:
                continue
            unreferenced.append(fig)

    inserted, appended = 0, 0
    if unreferenced:
        log.info('Phase 2.5b: %d unreferenced figure(s) to place', len(unreferenced))
        new_text, placed = _place_figures_by_caption(qmd_text, unreferenced, media_dirname)
        placed_set = set(placed)
        remaining = [f for f in unreferenced if f.get('fig_id') not in placed_set]
        new_text, appended = _append_unplaced_figures(new_text, remaining, media_dirname)
        if placed or appended:
            qmd_path.write_text(new_text, encoding='utf-8')
        inserted = len(placed)
        log.info('Phase 2.5b: %d placed at caption, %d appended (caption not in body)',
                 inserted, appended)

    return {
        'resolved_2_5a': resolved,
        'unreferenced_count': len(unreferenced),
        'inserted_2_5b': inserted,
        'appended_2_5b': appended,
        'cost_usd': 0.0,                    # deterministic — no LLM calls
    }
