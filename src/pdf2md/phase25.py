"""Phase 2.5: post-conversion figure rescue.

After Phase 2 writes the .qmd, two things can still go wrong:
(a) Leftover FIG_N tokens the regex didn't catch (bare parens, partial HTML attrs)
(b) Detected figures the converter never referenced at all

Phase 2.5a resolves (a) deterministically (no LLM cost).
Phase 2.5b sends a focused LLM call per unreferenced figure to insert it.
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


# ── 2.5b: LLM-driven insertion of unreferenced figures ─────────────────────

_UNREFERENCED_PROMPT = """You are given a section of a converted markdown document and an image of a figure that was detected on a specific page but NOT placed by the previous converter.

Your task: insert this figure at the MOST appropriate location in the markdown text. The figure appeared on the page shown; use the surrounding text to determine where it belongs.

Rules:
- Insert the image reference as: ![Figure N: brief caption](IMAGE_PATH)
- Place it near the text that discusses or references this figure
- Do NOT modify any existing figures or tables
- Keep all existing formatting intact
- Return the COMPLETE section with the figure inserted, not just the insertion

IMAGE_PATH will be replaced with the actual path."""


def _extract_page_image(working_pdf: Path, page_num: int, out_dir: Path,
                         stem: str, fig_id: str) -> Path | None:
    """Extract a single page as a PNG for the LLM to see."""
    try:
        import fitz
        doc = fitz.open(str(working_pdf))
        if page_num < 0 or page_num >= doc.page_count:
            doc.close()
            return None
        page = doc[page_num]
        # Render at 150 DPI for a reasonable size/quality balance
        pix = page.get_pixmap(dpi=150)
        img_path = out_dir / f'{stem}-media' / f'_rescue_{fig_id}.png'
        img_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(img_path))
        doc.close()
        return img_path
    except Exception as e:
        log.warning('Could not extract page %d for %s: %s', page_num, fig_id, e)
        return None


def _insert_unreferenced_figures(qmd_path: Path, unreferenced: list,
                                  working_pdf: Path, api_key: str,
                                  model: str = 'google/gemini-2.5-flash',
                                  timeout: int = 120) -> int:
    """For each unreferenced figure, send a focused LLM call to insert it.
    Sends the page image via image_url so the model can see the figure.
    Returns count of successfully inserted figures."""
    if not unreferenced:
        return 0

    import base64
    from .llm_client import _post_with_retries, _model_max_tokens

    qmd_text = qmd_path.read_text(encoding='utf-8')
    media_dirname = f'{qmd_path.stem}-media'
    out_dir = qmd_path.parent
    inserted = 0

    for fig in unreferenced:
        fig_id = fig.get('fig_id', '')
        page = fig.get('page', 0)
        page_idx = page - 1 if page > 0 else 0

        img_path = _extract_page_image(working_pdf, page_idx, out_dir,
                                        qmd_path.stem, fig_id)
        if not img_path:
            continue

        img_b64 = base64.b64encode(img_path.read_bytes()).decode()
        img_uri = f'data:image/png;base64,{img_b64}'

        cap = fig.get('caption') or f'Figure {fig_id.replace("FIG_", "")}'
        img_ref = f'{media_dirname}/{fig["file"]}'

        # Send a focused prompt with just the page image and context
        user_prompt = (
            f'A figure was detected on page {page} of the source document but '
            f'was not placed in the converted markdown below.\n\n'
            f'Insert this figure reference at the most appropriate location:\n'
            f'![{cap}]({img_ref})\n\n'
            f'Return ONLY the markdown section (a few paragraphs) with the figure '
            f'inserted. Do NOT return the entire document.\n\n'
            f'--- MARKDOWN ---\n{qmd_text[:8000]}\n--- END ---'
        )

        payload = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': 'You insert figures into markdown documents at the correct location. Return only the modified section, not the full document.'},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': user_prompt},
                    {'type': 'image_url', 'image_url': {'url': img_uri}},
                ]},
            ],
            'max_tokens': min(_model_max_tokens(model), 4096),
        }

        try:
            text, _usage = _post_with_retries(
                api_key=api_key, payload=payload,
                label=f'rescue-{fig_id}', timeout=timeout
            )
            # Refresh visible_text (may have been modified by previous insertions)
            visible_text = re.sub(r'<!--.*?-->', '', qmd_text, flags=re.DOTALL)
            if text and len(text) > 50:
                # Skip if the figure image is already in visible qmd text
                # (exclude HTML comments which list unreferenced figures)
                if fig.get('file', '') in visible_text:
                    log.info('Phase 2.5b: %s already placed (dedup)', fig_id)
                    continue
                # Append after the frontmatter as a best-effort placement
                fm_end = qmd_text.find('\n---\n\n')
                if fm_end < 0:
                    fm_end = qmd_text.find('\n---\n')
                if fm_end > 0:
                    fm_end = qmd_text.find('\n', fm_end + 5) + 1
                else:
                    fm_end = 0
                qmd_text = (qmd_text[:fm_end] +
                           f'\n\n<!-- Rescue-inserted {fig_id} -->\n\n{text.strip()}\n' +
                           qmd_text[fm_end:])
                inserted += 1
                log.info('Phase 2.5b: inserted %s via LLM', fig_id)
        except Exception as e:
            log.warning('Phase 2.5b: failed to insert %s: %s', fig_id, e)
            continue

    if inserted:
        qmd_path.write_text(qmd_text, encoding='utf-8')

    return inserted


# ── Main entry point ────────────────────────────────────────────────────────

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

    inserted = 0
    if unreferenced and working_pdf and working_pdf.exists() and api_key:
        log.info('Phase 2.5b: %d unreferenced figure(s) to rescue via LLM',
                 len(unreferenced))
        inserted = _insert_unreferenced_figures(
            qmd_path, unreferenced, working_pdf, api_key,
            model=rescue_model, timeout=timeout
        )
    elif unreferenced:
        log.info('Phase 2.5b: %d unreferenced figure(s) — skipping LLM rescue '
                 '(no API key or working PDF)', len(unreferenced))

    return {
        'resolved_2_5a': resolved,
        'unreferenced_count': len(unreferenced),
        'inserted_2_5b': inserted,
    }
