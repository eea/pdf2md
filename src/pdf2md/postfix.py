import json, logging, re, tempfile
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)
from .llm_client import _post_with_retries

_MIN_TABLE_PCT = 0.70
_MIN_FRAGMENT_LEN = 8
_REPAIR_MODEL = 'google/gemini-2.5-flash'

# recovered tables are anchored in place (like missing text), never appended: a marker
# is dropped on the surviving text above the table's source position, then filled
_TBL_MARKER_TMPL = '<!--pdf2md-tbl-{}-->'
_TBL_MARKER_RE = re.compile(r'<!--pdf2md-tbl-\d+-->')


def _body_pdf(out_dir, stem):
    """The chrome/footer-stripped copy the .qmd was actually derived from, when it
    survived the run — so anchor context is body text only, matching what the
    conversion saw. Falls back to the original source.pdf (with the heuristic chrome
    filter) for improve-only / replay runs where the working copy was cleaned up."""
    working = Path(out_dir) / '{}.working.pdf'.format(stem)
    source = Path(out_dir) / '{}.source.pdf'.format(stem)
    return working if working.exists() else source


# code-block recovery tuning
_CODE_BLOCK_MIN_MONO_SPANS = 2   # a block needs this many mono spans to count as code
_CODE_BLOCK_MONO_RATIO = 0.6     # ...and this share of its spans must be monospaced
_CODE_GROUP_GAP_PT = 40.0        # vertical gap (pt) that splits two code groups on a page
_CODE_MIN_CHARS = 12             # ignore groups smaller than this (stray inline glyphs)
_CODE_PROBE_MIN = 8              # min probe length before an in-.qmd presence check counts


def run_postfix(qmd_path, verify_results, out_dir, *, api_key=None, passes=1, meta=None):
    summary = {'postfixes_applied': [], 'cost_usd': 0.0}
    if passes <= 0 or not verify_results:
        return summary

    verify_by_name = {r.name: r for r in verify_results}

    # Pass 1: header bleed in tables
    table_check = verify_by_name.get('table_coverage')
    if table_check and table_check.status in ('warn', 'fail'):
        fixed = _strip_header_bleed(qmd_path)
        if fixed:
            summary['postfixes_applied'].append(
                'header_bleed: stripped {} header fragment(s)'.format(fixed))

    # Pass 1.5: code-block recovery (deterministic, no LLM)
    code_check = verify_by_name.get('code_block_presence')
    if code_check and code_check.status in ('warn', 'fail'):
        recovered = _recover_code_blocks(qmd_path, out_dir)
        if recovered:
            summary['postfixes_applied'].append(
                'code_blocks: recovered {} code block(s) from source PDF'.format(recovered))

    # Pass 1.8: hyperlink recovery (deterministic, no LLM). The href lives in a PDF
    # annotation the model never sees, so this is the only way those links can survive.
    link_check = verify_by_name.get('link_preservation')
    if link_check and link_check.status in ('warn', 'fail'):
        n_in, _ = _recover_links(qmd_path, out_dir)
        if n_in:
            summary['postfixes_applied'].append(
                'links: {} restored inline'.format(n_in))

    # Pass 1.9: heading restore (deterministic, no LLM)
    head_check = verify_by_name.get('heading_hierarchy')
    if head_check and head_check.status in ('warn', 'fail'):
        n_head = _postfix_headings(qmd_path, out_dir)
        if n_head:
            summary['postfixes_applied'].append(
                'headings: restored {} missing heading(s) from source outline'.format(n_head))

    # Pass 1.95: footnote rescue (deterministic, no LLM). An orphaned [^n] definition
    # (no matching [^n] reference) is dropped by Quarto — common for footnotes marking
    # table cells, whose {=html} raw block can't carry a [^n] link. Recover the lost
    # text as a visible ^n^ table-note in place.
    fn_check = verify_by_name.get('footnote_placement')
    if fn_check and fn_check.status in ('warn', 'fail'):
        n_fn = _postfix_footnotes(qmd_path, out_dir)
        if n_fn:
            summary['postfixes_applied'].append(
                'footnotes: recovered {} dropped note(s) as table-note(s)'.format(n_fn))

    # Pass 2: missing text rescue
    text_check = verify_by_name.get('text_coverage')
    if text_check and text_check.status in ('warn', 'fail') and api_key:
        rescued, items, repair_cost = _postfix_missing_text(qmd_path, out_dir, api_key, text_check)
        summary['cost_usd'] += repair_cost   # calls are billed even when nothing lands
        if rescued:
            summary['postfixes_applied'].append(
                'missing_text: {} items recovered from {} pages'.format(items, rescued))
            summary['items_recovered'] = items

    # Pass 3: missing-table recovery (deterministic, no LLM). Runs LAST — after text and
    # heading recovery — so a table in a collapsed region can anchor on the prose those
    # passes just restored, instead of declining for want of surviving context. This is
    # the sequential-repair principle: each pass builds on the previous one's output.
    # Not gated on the check's status: whole tables can be absent while the aggregate
    # still reads ok, because a missing table's boilerplate is supplied by its siblings.
    if table_check:
        n_tbl, tbl_cost = _recover_missing_tables(qmd_path, out_dir, api_key=api_key)
        summary['cost_usd'] += tbl_cost
        if n_tbl:
            summary['postfixes_applied'].append(
                'tables: re-emitted {} missing table(s) from source'.format(n_tbl))

    # Pass 4: focused-crop repair of PARTIALLY-mangled tables (substitutive). Pass 3
    # only recovers fully-missing tables — one that survived thin is invisible to it.
    # Also not gated on status: the aggregate can read ok while one table sits at 0%.
    if api_key and table_check and table_check.status != 'skipped':
        n_crop, crop_cost = _repair_thin_tables(qmd_path, out_dir, api_key)
        summary['cost_usd'] += crop_cost
        if n_crop:
            summary['postfixes_applied'].append(
                'tables: re-converted {} thin table(s) from focused crops'.format(n_crop))

    # Pass 5: structural cleanup of value-complete but mangled tables (empty filler
    # columns, or prose wrapped as a table). Runs after Pass 4 — a table Pass 4 already
    # rebuilt from a crop is clean and won't re-flag. Invisible to the coverage gate,
    # so it's driven by the .qmd's own table structure, not the check status.
    if api_key:
        n_mang, mang_cost = _repair_mangled_tables(qmd_path, out_dir, api_key)
        summary['cost_usd'] += mang_cost
        if n_mang:
            summary['postfixes_applied'].append(
                'tables: cleaned {} structurally-mangled table(s)'.format(n_mang))

    # Pass 5.5: strip leaked cover-page and TOC noise from the body (deterministic).
    # Runs after the text passes so a recovered cover paragraph is dropped too.
    qmd_text = qmd_path.read_text(encoding='utf-8')
    stripped_text, n_strip = _strip_front_matter_noise(qmd_text)
    if n_strip:
        qmd_path.write_text(stripped_text, encoding='utf-8')
        summary['postfixes_applied'].append(
            'front-matter: removed {} cover/TOC block(s)'.format(n_strip))

    # Pass 6: fence code the converter left unfenced (deterministic, no LLM). Runs LAST
    # so any code the earlier passes recovered into the body is fenced too — else its
    # `$` signs render as inline math.
    qmd_text = qmd_path.read_text(encoding='utf-8')
    fenced_text, n_fenced = _fence_unfenced_code(qmd_text)
    if n_fenced:
        qmd_path.write_text(fenced_text, encoding='utf-8')
        summary['postfixes_applied'].append(
            'code: fenced {} unfenced code block(s)'.format(n_fenced))

    # Final cleanup: strip author-facing postfix BREADCRUMB comments. They never render
    # (HTML comments) but clutter the .qmd source, and the report already lists every
    # repair. NOT touched: the <!--pdf2md-…--> functional markers — if one survives it
    # means content wasn't recovered, which must stay visible rather than be hidden.
    qmd_text = qmd_path.read_text(encoding='utf-8')
    cleaned = re.sub(r'(?m)^[ \t]*<!-- postfix:[^\n]*-->[ \t]*\n?', '', qmd_text)
    cleaned = re.sub(r'<!-- figures detected in Phase 1.*?-->\n?', '', cleaned, flags=re.DOTALL)
    # raw text-layer equation dumps render as '?' boxes; drop them (the proper $$…$$
    # LaTeX the converter emitted alongside is what should render)
    cleaned, n_tofu = _strip_raw_math_lines(cleaned)
    if n_tofu:
        summary['postfixes_applied'].append(
            'math: removed {} unrenderable raw-equation line(s)'.format(n_tofu))
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Quarto auto-numbers sections, so a manual number in the heading text renders doubled
    cleaned, n_headnum = _strip_heading_numbers(cleaned)
    if n_headnum:
        summary['postfixes_applied'].append(
            'headings: removed manual section number from {} heading(s) '
            '(Quarto auto-numbers)'.format(n_headnum))
    # markdown safety: a pipe table glued to a heading/caption won't render as a table
    cleaned, n_tblblank = _ensure_pipe_table_blanks(cleaned)
    if cleaned != qmd_text:
        qmd_path.write_text(cleaned, encoding='utf-8')
    if n_tblblank:
        summary['postfixes_applied'].append(
            'tables: separated {} table(s) glued to a caption/heading'.format(n_tblblank))

    # Re-verify
    if summary['postfixes_applied']:
        try:
            from .verify import VerifyContext, run_verify, overall_status, write_report
            stem = qmd_path.stem
            det_path = out_dir / 'detections.json'
            detections = json.loads(det_path.read_text()) if det_path.exists() else {'figures': []}
            ctx = VerifyContext(
                run_dir=out_dir,
                original_pdf=out_dir / '{}.source.pdf'.format(stem),
                working_pdf=out_dir / '{}.working.pdf'.format(stem),
                qmd_path=qmd_path,
                qmd_text=qmd_path.read_text(encoding='utf-8'),
                detections=detections,
                media_dir=out_dir / '{}-media'.format(stem),
                rendered_pdf=None,
            )
            results = run_verify(ctx)
            report_meta = dict(meta or {})
            report_meta["postfixes"] = summary["postfixes_applied"]
            report_meta["cost_repair"] = summary["cost_usd"]
            write_report(results, out_dir, meta=report_meta)
            summary['verify_after'] = overall_status(results)
            tc = next((r for r in results if r.name == 'text_coverage'), None)
            tbl = next((r for r in results if r.name == 'table_coverage'), None)
            summary['coverage_after'] = {
                'text': tc.metric if tc else None,
                'text_effective': (tc.detail or {}).get('effective') if tc else None,
                'text_recovered': (tc.detail or {}).get('recovered', 0) if tc else 0,
                'table': tbl.metric if tbl else None,
            }
        except Exception as e:
            log.warning('Re-verify after postfix failed: %s', e)

    return summary


def _strip_header_bleed(qmd_path):
    text = qmd_path.read_text()
    if not text:
        return 0

    tables = re.findall(r'<table>(.*?)</table>', text, re.DOTALL)
    if len(tables) < 3:
        return 0

    table_cell_words = []
    for tbl in tables:
        cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', tbl, re.IGNORECASE)
        all_words = []
        for cell in cells:
            clean = re.sub(r'<[^>]+>', ' ', cell)
            words = clean.lower().split()
            all_words.extend(words)
        table_cell_words.append(all_words)

    ngram_counter = Counter()
    for words in table_cell_words:
        seen = set()
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                if len(ngram) >= _MIN_FRAGMENT_LEN:
                    seen.add(ngram)
        for ng in seen:
            ngram_counter[ng] += 1

    threshold = max(3, int(len(tables) * _MIN_TABLE_PCT))
    bleed_fragments = [ng for ng, count in ngram_counter.items() if count >= threshold]
    if not bleed_fragments:
        return 0

    removed = 0
    for fragment in bleed_fragments:
        pattern = re.compile(re.escape(fragment), re.IGNORECASE)
        # Only strip within <table> blocks; leave body prose alone
        n = 0
        def _strip_table_content(m):
            nonlocal n
            tbl = m.group(1)
            tbl_new, c = pattern.subn('', tbl)
            n += c
            return f'<table>{tbl_new}</table>'
        new_text = re.sub(r'<table>(.*?)</table>', _strip_table_content, text, flags=re.DOTALL)
        if n > 0:
            text = new_text
            removed += 1
            log.info('postfix: stripped %s from %d table locations', fragment, n)

    if removed:
        qmd_path.write_text(text)
    return removed


def _guess_code_lang(text):
    """Best-effort language tag from cheap content hints; '' if unsure."""
    if '#!/' in text or re.search(r'^\s*\$ ', text, re.MULTILINE):
        return 'bash'
    if re.search(r'^\s*(def|class|import|from)\b', text, re.MULTILINE):
        return 'python'
    return ''


def _recover_code_blocks(qmd_path, out_dir):
    """Deterministically recover monospaced/code listings from the source PDF.

    Scans the source PDF for runs of monospaced text, groups vertically-adjacent
    monospaced blocks per page, and re-inserts any group not already present in the
    .qmd as a fenced code block. No LLM calls. Returns the number of blocks inserted.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.warning('code-block recovery skipped: PyMuPDF (fitz) not available')
        return 0

    from .verify.checks.code_block_presence import _span_is_mono, _qmd_code_blocks

    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0

    # 1-3. Collect monospaced blocks (bbox + text) from every page.
    groups = []  # list of raw code strings, preserving line breaks
    doc = fitz.open(str(source_pdf))
    try:
        for pno in range(doc.page_count):
            mono_blocks = []  # (y0, y1, text)
            for block in doc[pno].get_text('dict').get('blocks', []):
                block_lines, mono_spans, total_spans = [], 0, 0
                for line in block.get('lines', []):
                    parts = []
                    for span in line.get('spans', []):
                        txt = span.get('text', '')
                        if not txt.strip():
                            continue
                        total_spans += 1
                        if _span_is_mono(span):
                            mono_spans += 1
                        parts.append(txt)
                    if parts:
                        block_lines.append(''.join(parts))
                # a block is "code" when it is mostly monospaced
                if (mono_spans >= _CODE_BLOCK_MIN_MONO_SPANS
                        and mono_spans >= total_spans * _CODE_BLOCK_MONO_RATIO):
                    bbox = block.get('bbox', (0, 0, 0, 0))
                    mono_blocks.append((bbox[1], bbox[3], '\n'.join(block_lines)))

            # 4. Group consecutive mono blocks that sit close together vertically.
            mono_blocks.sort(key=lambda b: b[0])
            cur, last_y1 = [], None
            for y0, y1, text in mono_blocks:
                if last_y1 is not None and (y0 - last_y1) > _CODE_GROUP_GAP_PT:
                    groups.append('\n'.join(cur))
                    cur = []
                cur.append(text)
                last_y1 = y1
            if cur:
                groups.append('\n'.join(cur))
    finally:
        doc.close()

    # Fast exit: no monospaced text anywhere in the source.
    if not groups:
        return 0

    qmd_text = qmd_path.read_text(encoding='utf-8')
    existing = _qmd_code_blocks(qmd_text)
    norm_qmd = re.sub(r'\s+', ' ', qmd_text)

    # 5-8. Build fences for groups that are non-trivial and not already in the .qmd.
    new_blocks = []
    for group in groups:
        lines = [l for l in group.splitlines() if l.strip()]
        if not lines or sum(len(l) for l in lines) < _CODE_MIN_CHARS:
            continue
        # Edge case: skip if the .qmd already covers this code (avoid duplicates).
        probe = re.sub(r'\s+', ' ', max(lines, key=len).strip())
        if len(probe) >= _CODE_PROBE_MIN and probe in norm_qmd:
            continue
        lang = _guess_code_lang(group)
        new_blocks.append('```{}\n{}\n```'.format(lang, group.rstrip('\n')))

    if not new_blocks:
        return 0

    payload = ('<!-- postfix: code-block recovery -->\n\n'
               + '\n\n'.join(new_blocks) + '\n')

    # 7. Insert before the "## Recovered Technical Details" section, else at the end.
    marker = '## Recovered Technical Details'
    idx = qmd_text.find(marker)
    if idx != -1:
        # step back over an immediately-preceding postfix comment, if any
        head = qmd_text[:idx].rstrip()
        comment = head.rfind('<!-- postfix:')
        if comment != -1 and not head[comment:].startswith('<!-- postfix: code-block'):
            idx = comment
        qmd_text = qmd_text[:idx].rstrip() + '\n\n' + payload + '\n' + qmd_text[idx:]
    else:
        qmd_text = qmd_text.rstrip() + '\n\n' + payload

    qmd_path.write_text(qmd_text)
    for block in new_blocks:
        first = block.splitlines()[1] if len(block.splitlines()) > 1 else ''
        log.info('postfix: recovered code block (%d lines) starting %r',
                 block.count('\n') - 1, first[:60])
    log.info('postfix: code-block recovery inserted %d block(s) (%d already present)',
             len(new_blocks), existing)
    return len(new_blocks)


def _safe_boundary(text, pos):
    """First blank-line boundary at/after `pos` that is not inside a code fence, so an
    insert never lands mid-table or mid-fence. None when no safe boundary remains."""
    while True:
        nl = text.find('\n\n', pos)
        if nl == -1:
            return None
        cand = nl + 2
        if text.count('```', 0, cand) % 2 == 0:
            return cand
        pos = cand


_ANCHOR_WINDOW = 20000   # chars; how far apart two hits can be and still be "the same place"


def _bracketed_insertion_point(qmd_text, page_lines, pno, look=6):
    """Where does missing page `pno` belong? Immediately after the nearest EARLIER page
    whose text actually survived into the .qmd.

    The gap page cannot anchor itself — its text is the text that went missing — so we
    walk backwards to the closest page that did survive. `after` (the next surviving
    page forward) is used only as a sanity bound: if the two disagree, the anchor is
    unreliable and we decline rather than guess.
    """
    before = None
    for p in range(pno - 1, max(-1, pno - look - 1), -1):
        before = _insertion_point(qmd_text, page_lines.get(p, []))
        if before is not None:
            break
    if before is None:
        return None
    for p in range(pno + 1, pno + look + 1):
        after = _insertion_point(qmd_text, page_lines.get(p, []))
        if after is not None:
            # a later page must sit after the gap; if it doesn't, we've mis-anchored
            return before if after >= before else None
    return before


def _insertion_point(qmd_text, anchor_lines):
    """Where does a source page's content belong in the .qmd? Where that page's
    surviving lines CLUSTER — content-anchored, so it does not depend on headings the
    converter may have reworded. Clustering (not the last hit) matters: a single
    coincidental match elsewhere would otherwise drag the insert to the wrong part of
    the document. None when nothing survived to anchor on.
    """
    hay = qmd_text.lower()
    hits = []
    for ln in anchor_lines:
        s = ' '.join(ln.split())
        if len(s) < 40:
            continue
        probe = s[:60].lower()
        i = hay.find(probe)
        # the probe must be UNIQUE in the .qmd: a heading also appears in the table of
        # contents, and running chrome appears on every page — matching either pins the
        # insert to the top of the document (observed).
        if i != -1 and hay.find(probe, i + 1) == -1:
            hits.append(i + len(probe))
    if not hits:
        return None
    # pick the densest cluster, then insert after its last member
    best_hit, best_n = hits[0], 0
    for h in hits:
        n = sum(1 for x in hits if abs(x - h) <= _ANCHOR_WINDOW)
        if n > best_n:
            best_n, best_hit = n, h
    cluster_end = max(x for x in hits if abs(x - best_hit) <= _ANCHOR_WINDOW)
    return _safe_boundary(qmd_text, cluster_end)


def _drop_already_present(recovered, qmd_text):
    """Keep only paragraphs not already in the .qmd, so re-insertion never duplicates."""
    from .verify.textutil import qmd_to_plain, shingles, tokens
    qsh = shingles(tokens(qmd_to_plain(qmd_text)))
    keep = []
    for para in recovered.split('\n\n'):
        st = tokens(para)
        if len(st) < 8:
            continue
        sh = shingles(st)
        if sh and len(sh & qsh) / len(sh) >= 0.5:
            continue
        keep.append(para.strip())
    return '\n\n'.join(keep)


# a line repeating on this many source pages is running chrome, not content
_CHROME_LINE_MIN_PAGES = 5


def _strip_chrome_lines(recovered, line_freq):
    """Drop running headers/footers and bare page numbers from recovered page text.

    The page re-convert sees the raw page incl. its chrome; without this, every
    repaired page re-inserts the document title line and page number (observed:
    17× the running header after a 31-page repair)."""
    from .verify.textutil import normalize
    keep = []
    for ln in recovered.splitlines():
        n = normalize(ln)
        if n and (line_freq.get(n, 0) >= _CHROME_LINE_MIN_PAGES
                  or re.fullmatch(r'\d{1,4}', n)):
            continue
        keep.append(ln)
    return '\n'.join(keep)


_TABLE_ABSENT_MIN = 0.5    # recover when this share of a table's distinctive values is gone
_TABLE_MIN_DISTINCTIVE = 4  # ignore tables with too little unique data to judge
_TABLE_MIN_ROWS = 2
_TABLE_MIN_COLS = 2


def _ensure_pipe_table_blanks(text):
    """A pipe table must be preceded by a blank line or Markdown parses its header as
    part of the paragraph above and the table never renders. The converter (and
    re-emitted tables) sometimes glue the header row straight onto a heading or a
    'Table N' caption. Insert the missing blank. Returns (new_text, n_fixed)."""
    lines = text.split('\n')
    out, n = [], 0
    for i, ln in enumerate(lines):
        is_header = (re.match(r'^\s*\|.*\|\s*$', ln)
                     and i + 1 < len(lines)
                     and re.match(r'^\s*\|[-: |]+\|\s*$', lines[i + 1]))
        if is_header and out and out[-1].strip() and not re.match(r'^\s*\|', out[-1]):
            out.append('')
            n += 1
        out.append(ln)
    return '\n'.join(out), n


_HEADING_NUM_RE = re.compile(r'^(#{1,6})\s+\d+(?:\.\d+)*\.?\s+(\S.*)$')
_FENCE_RE = re.compile(r'^\s*(```|~~~)')
# Mathematical Alphanumeric Symbols block: 𝐴 𝑄 𝜃 … — the codepoints PDF text
# extraction uses for equation glyphs. Real math uses ASCII inside $…$, so several of
# these on a line marks a raw text-layer equation dump.
_MATH_ALNUM_RE = re.compile(r'[\U0001D400-\U0001D7FF]')


def _strip_raw_math_lines(text):
    """Drop lines dominated by Mathematical Alphanumeric Symbols. The converter sometimes
    leaves the PDF's raw text-layer equation in the body IN ADDITION to a proper $$…$$
    LaTeX version; the render font has no glyphs for that block, so it shows as '?' boxes
    (tofu). Three-plus such codepoints on one line is an unambiguous garbled-dump signal
    (legitimate prose and LaTeX never use them). Returns (new_text, n_removed)."""
    out, n = [], 0
    for ln in text.split('\n'):
        if len(_MATH_ALNUM_RE.findall(ln)) >= 3:
            n += 1
            continue
        out.append(ln)
    return '\n'.join(out), n


def _strip_heading_numbers(text):
    """Quarto auto-numbers sections (the render template sets number-sections: true), so
    a manual '1.'/'2.3' prefix in the heading TEXT renders doubled ('1 1. Introduction').
    Strip the leading section number from ATX headings, keeping the title. Fence-aware so
    a raw '#set …' line inside a ```{=typst} block is never mistaken for a heading. A
    number-only heading ('## 5') has no title after the number, so it's left untouched."""
    out, in_fence, n = [], False, 0
    for ln in text.split('\n'):
        if _FENCE_RE.match(ln):
            in_fence = not in_fence
        elif not in_fence:
            m = _HEADING_NUM_RE.match(ln)
            if m:
                out.append('{} {}'.format(m.group(1), m.group(2)))
                n += 1
                continue
        out.append(ln)
    return '\n'.join(out), n


# ── Strip cover-page and TOC noise the converter left in the body ───────────────
# The template regenerates a title page from the frontmatter (title/subtitle/date/
# version), so the source cover's body text (contact, disclaimer, produced-by) is
# redundant, and the printed TOC/list-of-figures is rebuilt by Quarto. The converter
# is told to drop both but often leaks them — as cover paragraphs and a run-on line of
# section numbers. Remove them by unambiguous signature; tables, headings and real
# prose are kept (measured: a false "Copernicus Land Monitoring Service" phrase match
# on body prose is why detection is anchor-based, not keyword-anywhere).

_COVER_ANCHOR = re.compile(
    r'^\s*(contact|produced by|disclaimer|project officer|lead service providers?|'
    r'document version|document date)\b', re.I)
_COVER_PHRASE = re.compile(r'\ball rights reserved\b', re.I)
_TOC_DUMP = re.compile(r'\bcontents\b.*\d|\blist of (figures|tables)\b', re.I)
# the change-log section is front matter, not a numbered chapter (absent from the
# bookmark outline), so the converter wrongly renders it as a section before "1
# Introduction". Drop its heading and the issue-history table that follows.
_CHANGE_LOG_RE = re.compile(r'^\s*#+\s*document change log\b', re.I)


def _is_toc_dump(block):
    if _TOC_DUMP.search(block):
        return True
    toks = block.split()
    if len(toks) < 8:
        return False
    nums = sum(1 for t in toks if re.fullmatch(r'\d+(\.\d+)*\.?', t))
    return nums / len(toks) > 0.4          # a run-on of section numbers = a TOC dump


def _strip_front_matter_noise(text):
    """Remove leaked cover-page paragraphs and printed TOC/list dumps. Returns
    (new_text, n_removed). Splits the body into blank-line blocks (code fences kept
    intact) and drops a block only when it STARTS with a cover anchor, contains
    'all rights reserved', or is a section-number run-on."""
    m = re.match(r'^(---\n.*?\n---\n)', text, re.DOTALL)
    head = m.group(1) if m else ''
    body = text[len(head):]

    blocks, cur, infence = [], [], False
    for ln in body.split('\n'):
        if ln.lstrip().startswith('```'):
            infence = not infence
        if not ln.strip() and not infence:
            blocks.append('\n'.join(cur))
            cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append('\n'.join(cur))

    kept, removed = [], 0
    drop_next_table = False
    for b in blocks:
        s = b.strip()
        if not s:
            continue
        first = s.split('\n', 1)[0]
        if drop_next_table:
            drop_next_table = False
            if re.match(r'^\s*\|', s):          # the change-log table (separate block)
                removed += 1
                continue
        if _CHANGE_LOG_RE.match(first):
            removed += 1
            if not re.search(r'(?m)^\s*\|', b):  # heading not glued to its table
                drop_next_table = True
            continue
        if _COVER_ANCHOR.match(first) or _COVER_PHRASE.search(s) or _is_toc_dump(s):
            removed += 1
            continue
        kept.append(b)
    if not removed:
        return text, 0
    return head + '\n\n'.join(kept) + '\n', removed


# ── Fence code the converter left unfenced ──────────────────────────────────────
# A shell/script listing emitted as raw markdown renders badly: Quarto pairs its `$`
# signs as inline math, so `${var}` and `$(cmd)` mangle into equations (measured: a
# whole bash processing script rendered as broken math on one ATBD). Wrap runs of
# unfenced code so `$` stays literal. Conservative: a run must be dense with code and
# carry at least two UNAMBIGUOUS shell signals, so prose is never wrapped.

_CODE_STRONG = re.compile(r'#!|\$\{|\$\(|;\s*do\b|\bdone\b|\bfi\b|\bthen\b|\besac\b')
_CODE_WEAK = re.compile(
    r'^\s*(for|while|if|elif|else|case|function|do|then|fi|done|esac|continue|break|'
    r'return|exit|local)\b'
    r'|^\s*[A-Za-z_][\w-]*=(?!=)'                       # VAR=... assignment
    r'|^\s*set -[eux]'
    r'|\b(mkdir|echo|find|rm|cp|mv|seq|export|read|awk|sed|grep|cat|sort|uniq'
    r'|gdal\w+|ogr\w+)\b'
    r'|^\s*#\s')                                        # shell comment


def _code_line_score(line):
    """2 = an unambiguous code line, 1 = a weak code signal, 0 = blank, -1 = prose."""
    if not line.strip():
        return 0
    if _CODE_STRONG.search(line):
        return 2
    if _CODE_WEAK.search(line):
        return 1
    return -1


def _fence_unfenced_code(text):
    """Wrap runs of unfenced code in ```` ```bash ````. A run is a maximal block of
    non-fence lines that are code-or-blank, bounded by prose; it is fenced only when it
    has >= 2 strong shell signals and >= 3 code lines. Returns (new_text, n_fenced)."""
    lines = text.split('\n')
    inside, infence = False, []
    for ln in lines:
        if ln.lstrip().startswith('```'):
            infence.append(True)
            inside = not inside
        else:
            infence.append(inside)

    blocks, i, n = [], 0, len(lines)
    while i < n:
        if infence[i] or _code_line_score(lines[i]) <= 0:
            i += 1
            continue
        j = i
        while j < n and not infence[j] and _code_line_score(lines[j]) >= 0:
            j += 1
        blk = lines[i:j]
        strong = sum(1 for l in blk if _CODE_STRONG.search(l))
        code = sum(1 for l in blk if _code_line_score(l) > 0)
        if strong >= 2 and code >= 3:
            end = j
            while end > i and not lines[end - 1].strip():
                end -= 1                                # trim trailing blanks
            blocks.append((i, end))
        i = j

    if not blocks:
        return text, 0
    for s, e in sorted(blocks, reverse=True):
        lines[s:e] = ['```bash'] + lines[s:e] + ['```']
    return '\n'.join(lines), len(blocks)


def _grid_to_markdown(rows):
    """Render extracted source cells as a Markdown grid. Values are copied verbatim."""
    width = max(len(r) for r in rows)
    out = []
    for i, r in enumerate(rows):
        cells = [(c or '').replace('\n', ' ').replace('|', '\\|').strip() for c in r]
        cells += [''] * (width - len(cells))
        out.append('| ' + ' | '.join(cells) + ' |')
        if i == 0:
            out.append('|' + '---|' * width)
    return '\n'.join(out)


_LINK_ANCHOR_MIN = 10        # shorter anchor text is too ambiguous to link safely


def _safe_to_inline(text, pos):
    """True when `pos` is in ordinary prose — not inside a code fence, an HTML table,
    or an existing markdown link, where injecting a link would corrupt the markup."""
    before = text[:pos]
    if before.count('```') % 2:
        return False
    if before.count('<table') > before.count('</table'):
        return False
    tail = text[max(0, pos - 2):pos]
    return '[' not in tail


def _recover_links(qmd_path, out_dir):
    """Restore hyperlink targets the converter never saw.

    A PDF keeps the href in a link ANNOTATION, not in the page text, so the model cannot
    reproduce it — measured across 6 documents: not one URL reached the .qmd that was not
    already visible as text. We hold them exactly, so re-attach a target inline when its
    anchor text occurs exactly once (provably unambiguous). Links we can't place that way
    are left alone — verify's link_preservation still reports them — rather than dumped
    into a synthetic 'Source links' section that isn't in the source. Returns
    (n_inlined, n_not_inlinable).
    """
    try:
        import fitz
    except ImportError:
        return 0, 0
    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0

    from .verify.checks.link_preservation import _uri_in_qmd

    qmd = qmd_path.read_text(encoding='utf-8')
    qmd_lower = qmd.lower()

    pairs, seen = [], set()
    doc = fitz.open(str(source_pdf))
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            for l in page.get_links():
                uri = (l.get('uri') or '').strip()
                if not uri or uri in seen:
                    continue
                seen.add(uri)
                # "present" uses the SAME contiguous match as the verify check — else a
                # URL wrapped mid-string reads as present here (whitespace-stripped) yet
                # missing to the check, so it's neither restored nor counted (measured:
                # 2 reference DOIs on one ATBD were broken across lines and lost this way)
                if _uri_in_qmd(uri, qmd_lower):
                    continue            # already present (contiguously), nothing to restore
                # DOI fallback: a malformed doubled DOI ("https://doi.org/https:/doi.org/
                # 10.x/…" — the PDF's own defect, with fitz collapsing one slash) never
                # matches contiguously against the reference's un-collapsed copy, so it
                # gets re-listed as a phantom "source link". Match on the bare DOI core,
                # which is immune to the prefix mangling and unique enough to prove presence.
                doi = re.search(r'10\.\d{4,}/\S+', uri)
                if doi and doi.group(0).lower().rstrip('.,;)') in qmd_lower:
                    continue
                anchor = ' '.join(page.get_textbox(l['from']).split())
                anchor = anchor.strip(' .,;:)（(')   # keep punctuation outside the link
                pairs.append((uri, anchor))
    finally:
        doc.close()
    if not pairs:
        return 0, 0

    inlined, dropped = 0, 0
    for uri, anchor in pairs:
        # inline only when the anchor is real prose occurring exactly once (provably
        # unambiguous). A truncated-URL anchor or an ambiguous one can't be placed.
        usable = (len(anchor) >= _LINK_ANCHOR_MIN
                  and not anchor.lower().startswith(('http', 'www.', 'mailto:')))
        if usable and qmd.count(anchor) == 1:
            pos = qmd.find(anchor)
            if _safe_to_inline(qmd, pos):
                qmd = qmd[:pos] + '[{}]({})'.format(anchor, uri) + qmd[pos + len(anchor):]
                inlined += 1
                continue
        dropped += 1        # can't inline; we no longer append a synthetic "Source
                            # links" section — verify still reports these as missing
    if inlined:
        qmd_path.write_text(qmd, encoding='utf-8')
    log.info('postfix: restored %d link(s) inline, %d not inlinable', inlined, dropped)
    return inlined, dropped


_TABLE_LLM_MIN_KEEP = 0.98   # LLM rendering is used only if it keeps ~all source values


def _table_values(rows, normalize):
    """Every word token in a table's cells — the ground truth a rendering must preserve."""
    out = set()
    for r in rows:
        for c in r:
            if c:
                out |= set(normalize(c).split())
    return out


def _llm_table_markdown(api_key, doc, pno, rows, normalize):
    """Re-convert a table region for STRUCTURE (merged cells, real header, caption), then
    verify it against the deterministic cell values.

    The model's known failure mode is silently dropping data — which is what caused these
    gaps in the first place. find_tables gives the exact values for free, so we can take
    the model's better structure without trusting its completeness: if any value is lost,
    the caller keeps the plain deterministic grid instead. Returns (markdown, cost) with
    markdown None when verification fails.
    """
    prompt = (
        'Below is the text of page {} of a technical document containing a table.\n'
        'Convert THE TABLE to a clean Markdown table. Preserve every value exactly as '
        'written. Include the table caption if one is present, as a line above the '
        'table. Join cell text that the extraction split across lines, and do not emit '
        'empty filler columns. Do not summarise, do not omit any row or value, and do '
        'not add commentary.\nOutput only the caption line (if any) and the table.\n\n{}'
        .format(pno + 1, doc[pno].get_text()[:3000])
    )
    try:
        response, usage = _post_with_retries(
            api_key=api_key,
            payload={'model': _REPAIR_MODEL,
                     'messages': [{'role': 'user', 'content': prompt}],
                     'max_tokens': 4096},
            label='postfix-table-p{}'.format(pno + 1), timeout=120,
        )
    except RuntimeError as e:
        log.warning('Table re-conversion p%d failed: %s', pno + 1, e)
        return None, 0.0
    cost = (usage or {}).get('cost', 0.0)
    if not response:
        return None, cost
    need = _table_values(rows, normalize)
    got = set(normalize(response).split())
    kept = len(need & got) / len(need) if need else 1.0
    if kept < _TABLE_LLM_MIN_KEEP:
        log.info('postfix: p%d LLM table kept only %.0f%% of values — using exact grid',
                 pno + 1, 100 * kept)
        return None, cost
    return response.strip(), cost


def _scan_source_tables(source_pdf):
    """find_tables scan shared by the table passes.

    Returns (tables, clean_lines): tables = [(pno, bbox, rows, ctx_tokens)] where ctx
    is the chrome-free text preceding the table (crossing the page boundary, so a
    page-top table still has its caption/paragraph to anchor on); clean_lines maps
    pno -> [(y, txt)] chrome-free substantial lines. Running headers/footers are
    excluded from context via a digit-insensitive repeats-across-pages signature —
    they were stripped from the .qmd, so an anchor containing them can never match."""
    import fitz
    from .verify.textutil import normalize, tokens as _tok

    def _chrome_sig(txt):
        return re.sub(r'\d+', '#', normalize(txt))

    doc = fitz.open(str(source_pdf))
    try:
        # first pass: all substantial lines per page (15-char floor keeps captions like
        # "Table 5. …" while dropping stray glyphs)
        page_lines = {}
        for pno in range(doc.page_count):
            lines = []
            for b in doc[pno].get_text("dict").get("blocks", []):
                for ln in b.get("lines", []):
                    txt = "".join(s["text"] for s in ln["spans"]).strip()
                    if len(txt) >= 15:
                        lines.append((ln["bbox"][1], txt))
            lines.sort()
            page_lines[pno] = lines

        total_pages = doc.page_count or 1
        seen = defaultdict(set)
        for pno, lines in page_lines.items():
            for _y, txt in lines:
                seen[_chrome_sig(txt)].add(pno)
        chrome = {s for s, ps in seen.items() if len(ps) >= max(3, total_pages * 0.5)}

        tables, clean_lines = [], {}
        prev_tail = []
        for pno in range(doc.page_count):
            plines = [(y, txt) for y, txt in page_lines[pno] if _chrome_sig(txt) not in chrome]
            clean_lines[pno] = plines
            try:
                found = doc[pno].find_tables().tables
            except Exception:               # noqa: BLE001 — one bad page must not abort
                found = []
            for t in found:
                rows = [r for r in t.extract() if any(c for c in r)]
                if not rows:
                    continue
                above = [txt for y, txt in plines if y < t.bbox[1]]
                pre = (prev_tail + above)[-10:]
                tables.append((pno, tuple(t.bbox), rows, _tok(' '.join(pre))))
            prev_tail = [txt for _, txt in plines]
    finally:
        doc.close()
    return tables, clean_lines


# ── Focused-crop table repair (substitutive) ────────────────────────────────────
# Whole-doc conversion partially mangles dense tables (attention dilution over ~80
# pages), and the additive Pass-3 recovery only fires on FULLY-missing tables — a
# table that survived thin is invisible to it. Measured on 20 tables / 5 docs: a
# focused 300-dpi crop of just the table region lifts them 81%→93% avg (44→100 on
# the worst). Transcription is noisy run-to-run, so a guard keeps the incumbent
# unless the crop is strictly better — the pass can only improve, never corrupt.

_TBL_CROP_DPI = 300
_TBL_CROP_MAX_COV = 0.9        # incumbent at/above this is left alone
_TBL_CROP_MIN_DISTINCT = 8     # fewer distinctive values = sliver, not worth a call
_TBL_CROP_INCUMBENT_MIN = 0.3  # below this the table is "missing" (Pass 3's job)
_TBL_CROP_INSERT_MIN = 0.7     # a vision-found table must transcribe this well to insert
_TBL_CROP_SYS = (
    'You transcribe a data table to Markdown. Preserve EVERY cell value exactly as '
    'written; never omit a row or summarize; join wrapped cell text. If the table has '
    'merged/spanning cells, emit a raw HTML <table> instead of a pipe table. '
    'Output ONLY the table, no commentary.')


def _tbl_md_tokens(md):
    """Normalized word tokens of a markdown/HTML table blob — the common coin all
    coverage comparisons in this pass are made in."""
    from .verify.textutil import normalize
    txt = re.sub(r'<[^>]+>', ' ', md)
    txt = txt.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    return set(normalize(txt).split())


def _qmd_table_spans(qmd_text):
    """(start, end, token_set) of each replaceable .qmd table block: ```{=html}```
    fences containing a <table>, and pipe-table line runs outside those fences."""
    raw = []
    for m in re.finditer(r'```\{=html\}\n.*?\n```', qmd_text, re.DOTALL):
        if '<table' in m.group(0).lower():
            raw.append((m.start(), m.end()))
    fenced = list(raw)
    for m in re.finditer(r'(?m)^(?:\|[^\n]*\|[ \t]*\n?)+', qmd_text):
        if not any(s <= m.start() < e for s, e in fenced):
            raw.append((m.start(), m.end()))
    return [(s, e, _tbl_md_tokens(qmd_text[s:e])) for s, e in sorted(raw)]


def _crop_replace_ok(dist, new_toks, block_toks, src_toks):
    """Replace only when strictly safer: the crop keeps every distinctive value the
    incumbent block already has, adds at least one more, AND the block is mostly
    THIS table — never overwrite a block that merged other content (e.g. the other
    pages of a multi-page table, which a single-page crop cannot supply)."""
    new_hit = dist & new_toks
    inc_hit = dist & block_toks
    if not (inc_hit <= new_hit and len(new_hit) > len(inc_hit)):
        return False
    # alien share 0.25: a replaced block may carry at most a quarter of content that
    # is not this table's — replacing deletes that content, so keep the ceiling low
    # (0.5 allowed losing up to half a merged block; measured -1.4pt on one doc)
    return len(block_toks - src_toks) <= 0.25 * len(block_toks)


def _crop_table_md(api_key, doc, pno, bbox, est_chars, system=None, user=None):
    """One focused vision call: render the table bbox at 300 dpi and transcribe it.
    max_tokens is PROPORTIONAL to the table's own text volume — ~est_chars/4 source
    tokens with 4x headroom gives cap ≈ est_chars — clamped to [2000, 16000]: big
    enough that no legitimate table clips, small enough that a repetition runaway
    fails fast and cheap. `system`/`user` override the prompt (the structural pass
    passes a variant that may return prose)."""
    import base64
    import fitz
    from .llm_client import call_vision
    from .cost import usage_cost

    png = doc[pno].get_pixmap(clip=fitz.Rect(*bbox), dpi=_TBL_CROP_DPI).tobytes('png')
    uri = 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')
    cap = max(2000, min(16000, int(est_chars)))
    try:
        md, usage = call_vision(
            api_key=api_key, model=_REPAIR_MODEL,
            system_instruction=system or _TBL_CROP_SYS,
            user_prompt=user or 'Transcribe the table in this cropped image.',
            image_data_uris=[uri], timeout=120, max_tokens=cap,
            response_format=None, return_usage=True)
    except RuntimeError as e:               # truncation / API failure → decline
        log.warning('table-crop p%d: %s', pno + 1, e)
        return None, 0.0
    md = md.strip()
    md = re.sub(r'^```[^\n]*\n', '', md)
    md = re.sub(r'\n```$', '', md).strip()
    if '<table' in md.lower():
        md = '```{=html}\n' + md + '\n```'
    return (md or None), usage_cost(usage)


def _repair_thin_tables(qmd_path, out_dir, api_key):
    """Re-convert partially-mangled tables from focused crops, in place.

    Region set = find_tables grids ∪ vision-detected excluded_tables (borderless
    grids find_tables can't see, from the detections.json sidecar). Each region is
    scored by its DISTINCTIVE values against the .qmd's table blocks; a thin
    incumbent (< _TBL_CROP_MAX_COV) is re-converted from a 300-dpi crop and
    replaced only if _crop_replace_ok holds. A vision-only region with no incumbent
    is inserted at its anchored position when the crop transcribes well. One retry
    per region (transcription is noisy). Returns (n_changed, llm_cost)."""
    import fitz
    from .verify.textutil import normalize

    stem = qmd_path.stem
    source_pdf = _body_pdf(out_dir, stem)
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    try:
        found, clean_lines = _scan_source_tables(source_pdf)
    except Exception as e:                  # noqa: BLE001 — repair must never abort
        log.warning('Table-crop repair: could not scan source: %s', e)
        return 0, 0.0

    # unified region list: (pno, bbox, src_toks, ctx_tokens, est_chars, origin)
    regions = []
    for pno, bbox, rows, ctx in found:
        ncols = max((len(r) for r in rows), default=0)
        if ncols >= 45 or len(rows) * ncols >= 2500:
            continue                        # oversized → cropped as a figure upstream
        cells = [normalize(c) for r in rows for c in r if c]
        stoks = set(t for c in cells for t in c.split())
        regions.append((pno, bbox, stoks, ctx,
                        sum(len(c) + 1 for c in cells), 'grid'))

    det = out_dir / 'detections.json'
    if det.exists():
        try:
            others = json.loads(det.read_text(encoding='utf-8')).get('other_detections', [])
        except Exception:                   # noqa: BLE001
            others = []
        grid_boxes = defaultdict(list)
        for pno, bbox, _s, _c, _e, _o in regions:
            grid_boxes[pno].append(bbox)
        doc = fitz.open(str(source_pdf))
        try:
            for r in others:
                if r.get('rtype') != 'table' or not r.get('bbox'):
                    continue
                pno, bbox = int(r['page']), tuple(r['bbox'])
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
                       for b in grid_boxes.get(pno, [])):
                    continue                # find_tables already owns this region
                if not (0 <= pno < doc.page_count):
                    continue
                clip = doc[pno].get_text(clip=fitz.Rect(*bbox))
                stoks = set(normalize(clip).split())
                above = [t for y, t in clean_lines.get(pno, []) if y < bbox[1]]
                from .verify.textutil import tokens as _tok
                regions.append((pno, bbox, stoks, _tok(' '.join(above[-10:])),
                                len(clip), 'vision'))
        finally:
            doc.close()

    if not regions:
        return 0, 0.0
    df = Counter()
    for _p, _b, stoks, _c, _e, _o in regions:
        for t in stoks:
            df[t] += 1
    spans = _qmd_table_spans(qmd_text)
    words, starts, ends = _qmd_word_offsets(qmd_text)

    doc = fitz.open(str(source_pdf))
    edits, used, cost = [], set(), 0.0
    try:
        for pno, bbox, stoks, ctx, est_chars, origin in regions:
            dist = {t for t in stoks if df[t] <= 2}
            if len(dist) < _TBL_CROP_MIN_DISTINCT:
                log.debug('table-crop p%d %s: skip, %d distinctive value(s)',
                          pno + 1, origin, len(dist))
                continue
            best, inc_cov = None, 0.0
            for sp in spans:
                c = len(dist & sp[2]) / len(dist)
                if c > inc_cov:
                    best, inc_cov = sp, c
            if inc_cov >= _TBL_CROP_MAX_COV:
                log.debug('table-crop p%d %s: skip, incumbent %.0f%%',
                          pno + 1, origin, 100 * inc_cov)
                continue

            def _score(m):
                return len(dist & _tbl_md_tokens(m)) / len(dist) if m else 0.0
            md, c1 = _crop_table_md(api_key, doc, pno, bbox, est_chars)
            cost += c1
            if _score(md) <= inc_cov:       # noisy — one retry before declining
                md2, c2 = _crop_table_md(api_key, doc, pno, bbox, est_chars)
                cost += c2
                if _score(md2) > _score(md):
                    md = md2
            if not md:
                log.debug('table-crop p%d %s: decline, no usable transcription',
                          pno + 1, origin)
                continue
            new_toks = _tbl_md_tokens(md)

            if best is not None and inc_cov >= _TBL_CROP_INCUMBENT_MIN:
                s, e, btoks = best
                if (s, e) in used:
                    log.debug('table-crop p%d %s: decline, block already claimed',
                              pno + 1, origin)
                    continue
                if _crop_replace_ok(dist, new_toks, btoks, stoks):
                    used.add((s, e))
                    if qmd_text[s:e].endswith('\n') and not md.endswith('\n'):
                        md += '\n'
                    edits.append((s, e, md))
                    log.info('table-crop p%d %s: replacing block (%.0f%% -> %.0f%% '
                             'of %d distinctive values)', pno + 1, origin,
                             100 * inc_cov, 100 * _score(md), len(dist))
                else:
                    log.debug('table-crop p%d %s: guard declined (inc %.0f%%, '
                              'crop %.0f%%)', pno + 1, origin,
                              100 * inc_cov, 100 * _score(md))
            elif origin == 'vision' and len(dist & new_toks) / len(dist) >= _TBL_CROP_INSERT_MIN:
                at = _anchor_by_context(words, ends, ctx)
                if at is None:
                    log.debug('table-crop p%d vision: decline, no unique anchor', pno + 1)
                    continue                # no unique anchor → decline, never append
                safe = _safe_boundary(qmd_text, at)
                if safe is None:
                    continue
                edits.append((safe, safe, md + '\n\n'))
                log.info('table-crop p%d vision: inserting missing table (%.0f%% of '
                         '%d distinctive values)', pno + 1, 100 * _score(md), len(dist))
            else:
                log.debug('table-crop p%d %s: decline (inc %.0f%% below replace floor, '
                          'crop %.0f%%)', pno + 1, origin, 100 * inc_cov, 100 * _score(md))
    finally:
        doc.close()

    if not edits:
        return 0, cost
    for s, e, md in sorted(edits, reverse=True):
        qmd_text = qmd_text[:s] + md + qmd_text[e:]
    qmd_path.write_text(qmd_text, encoding='utf-8')
    log.info('postfix: re-converted %d thin table(s) from focused crops', len(edits))
    return len(edits), cost


# ── Structural table cleanup (value-complete but mangled grids) ──────────────────
# Some tables keep all their VALUES (so they pass Pass 4's coverage gate) yet render
# badly: whole-doc conversion pads a wide table with empty filler columns and splits
# headers, or wraps a bordered prose callout in `| paragraph |  |` table syntax. The
# value-coverage metric is blind to this. Detect it structurally (high empty-cell
# ratio), re-convert from a focused crop that self-corrects to a clean grid OR to
# prose, and replace only when structure improves AND no source value is lost.

_MANGLE_EMPTY_RATIO = 0.35     # a pipe table with more blank cells than this is suspect
_STRUCT_IMPROVE = 0.15         # the re-conversion must beat the old empty-ratio by this
_NUM_KEEP = 0.9                # the rebuild must retain this share of the table's numbers
_TBL_CLEAN_SYS = (
    'You transcribe a table region from an image to Markdown. Preserve EVERY value '
    'exactly. Emit a CLEAN grid: exactly one column per real column, NO empty filler '
    'columns, and join any header split across lines. If the region is actually '
    'running prose or a note (not tabular data), return it as plain paragraph text, '
    'NOT a table. Output ONLY the result.')
_PIPE_BLOCK_RE = re.compile(r'(?m)(?:^\|[^\n]*\|[ \t]*\n?)+')
_NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')


def _numbers(text):
    """Numeric tokens (integers/decimals) in raw text — the table's actual DATA. Unlike
    word tokens, numbers are stable across formatting, so they are the reliable signal
    that a rebuild didn't drop a row or value (a split header like MAXV→M,AXV is not a
    number, so fixing the split doesn't read as data loss)."""
    return set(_NUM_RE.findall(text))


def _pipe_empty_ratio(block):
    """Fraction of a pipe table's data cells that are blank (divider rows ignored)."""
    cells = []
    for ln in block.splitlines():
        s = ln.strip()
        if not (s.startswith('|') and s.endswith('|')):
            continue
        if set(s) <= set('|-: '):           # header/body divider row
            continue
        cells += [c.strip() for c in s.strip('|').split('|')]
    return (sum(1 for c in cells if not c) / len(cells)) if cells else 0.0


def _repair_mangled_tables(qmd_path, out_dir, api_key):
    """Re-convert value-complete but structurally-mangled pipe tables from focused
    crops. Replaces a flagged table only when the re-conversion (a) improves structure
    — a much lower empty-cell ratio, or it turns out to be prose — AND (b) loses no
    source distinctive value (coverage does not drop vs the mangled version). Worst
    case is keeping the original, so the pass can only improve. Returns (n, llm_cost)."""
    import fitz
    from .verify.textutil import normalize

    stem = qmd_path.stem
    source_pdf = _body_pdf(out_dir, stem)
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    flagged = [(m.start(), m.end(), m.group(0)) for m in _PIPE_BLOCK_RE.finditer(qmd_text)
               if m.group(0).count('\n') >= 3
               and _pipe_empty_ratio(m.group(0)) > _MANGLE_EMPTY_RATIO]
    if not flagged:
        return 0, 0.0

    try:
        src = _scan_source_tables(source_pdf)[0]        # (pno, bbox, rows, ctx)
    except Exception as e:                              # noqa: BLE001
        log.warning('Mangled-table repair: could not scan source: %s', e)
        return 0, 0.0
    regions = []
    for pno, bbox, rows, _ctx in src:
        toks = {t for r in rows for c in r if c for t in normalize(c).split()}
        est = sum(len(c) + 1 for r in rows for c in r if c)
        if toks:
            regions.append((pno, bbox, toks, est))
    if not regions:
        return 0, 0.0

    doc = fitz.open(str(source_pdf))
    edits, cost = [], 0.0
    try:
        for s, e, block in flagged:
            btoks = _tbl_md_tokens(block)
            # locate the source region this mangled table came from (best overlap)
            best = max(regions, key=lambda r: len(btoks & r[2]), default=None)
            if best is None or len(btoks & best[2]) < 5:
                continue                                # can't confidently place it
            pno, bbox, _rtoks, est = best
            old_ratio = _pipe_empty_ratio(block)
            new_md, c = _crop_table_md(api_key, doc, pno, bbox, est * 3,
                                       system=_TBL_CLEAN_SYS,
                                       user='Transcribe the table in this cropped image '
                                            'to a clean Markdown table, or to prose if '
                                            'it is not tabular.')
            cost += c
            if not new_md:
                continue
            new_is_prose = '|' not in new_md and '<table' not in new_md.lower()
            # structure must improve: prose always beats a `| prose |` table; a rebuilt
            # grid must shed its empty filler columns. (The coverage metric now excludes
            # prose-callout regions, so un-tabling them no longer drags the score.)
            if new_is_prose:
                structure_better = True
            else:
                structure_better = _pipe_empty_ratio(new_md) <= old_ratio - _STRUCT_IMPROVE
            if not structure_better:
                continue
            # value preservation on the DATA (numbers), block↔new: robust to the split-
            # header formatting that made a coverage-vs-find_tables guard reject the fix
            old_nums = _numbers(block)
            if old_nums and len(old_nums & _numbers(new_md)) / len(old_nums) < _NUM_KEEP:
                continue                                # would drop values → decline
            # prose replacing a table needs blank-line separation to stay its own block
            body = ('\n' + new_md + '\n') if new_is_prose else new_md
            tail = '\n' if block.endswith('\n') and not body.endswith('\n') else ''
            edits.append((s, e, body + tail))
            log.info('table-clean p%d: %s (empty %.0f%%%s, numbers kept)', pno + 1,
                     'un-tabled to prose' if new_is_prose else 'rebuilt grid',
                     100 * old_ratio,
                     '->prose' if new_is_prose else '->%.0f%%' % (100 * _pipe_empty_ratio(new_md)))
    finally:
        doc.close()

    if not edits:
        return 0, cost
    for s, e, md in sorted(edits, reverse=True):
        qmd_text = qmd_text[:s] + md + qmd_text[e:]
    qmd_path.write_text(qmd_text, encoding='utf-8')
    log.info('postfix: cleaned %d structurally-mangled table(s)', len(edits))
    return len(edits), cost


def _recover_missing_tables(qmd_path, out_dir, api_key=None):
    """Re-emit source tables whose data never reached the .qmd. Returns (count, llm_cost).

    Values come straight from PyMuPDF find_tables, so they are exact; the LLM is only
    (optionally) asked for better STRUCTURE, and its rendering is kept solely when every
    source value survives. Detection uses each table's DISTINCTIVE values (tokens rare
    in the source itself). Shared boilerplate ('small', '%', column headers) proves
    nothing, because sibling tables supply it — which is how 13 pages of missing tables
    still scored 77-85% on token-bag matching.
    """
    try:
        import fitz
    except ImportError:
        return 0, 0.0
    from .verify.checks.table_coverage import _qmd_grids, _tokens_of
    from .verify.textutil import normalize

    stem = qmd_path.stem
    source_pdf = _body_pdf(out_dir, stem)   # prefer the chrome-stripped body copy
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    qtoks = set()
    for g in _qmd_grids(qmd_text):
        qtoks |= _tokens_of(g)

    tables = [(pno, rows, ctx)
              for pno, _bbox, rows, ctx in _scan_source_tables(source_pdf)[0]]

    def toks_of(rows):
        out = set()
        for r in rows:
            for c in r:
                if c:
                    out |= set(normalize(c).split())
        return out

    src_df = Counter()
    for _p, rows, _c in tables:
        for t in toks_of(rows):
            src_df[t] += 1

    blocks = []                             # (pno, md, label, ctx_tokens)
    llm_cost = [0.0]
    for pno, rows, ctx in tables:
        if len(rows) < _TABLE_MIN_ROWS or max(len(r) for r in rows) < _TABLE_MIN_COLS:
            continue                        # degenerate segmentation — not a real grid
        distinctive = {t for t in toks_of(rows) if src_df[t] <= 2}
        if len(distinctive) < _TABLE_MIN_DISTINCTIVE:
            continue
        absent = distinctive - qtoks
        if len(absent) / len(distinctive) < _TABLE_ABSENT_MIN:
            continue
        md, src_label = None, 'exact grid'
        if api_key:
            doc2 = fitz.open(str(source_pdf))
            try:
                md, c = _llm_table_markdown(api_key, doc2, pno, rows, normalize)
                llm_cost[0] += c
            finally:
                doc2.close()
            if md:
                src_label = 'structured'
        if not md:
            md = _grid_to_markdown(rows)
        blocks.append((pno, md, src_label, ctx))

    if not blocks:
        return 0, llm_cost[0]

    # anchor each recovered table on the surviving text above its source position and
    # drop a marker there; decline (never append) when there is no unique anchor
    words, starts, ends = _qmd_word_offsets(qmd_text)
    planned = []                            # (safe_offset, tid, pno, md, label)
    tid = 0
    for pno, md, label, ctx in blocks:
        at = _anchor_by_context(words, ends, ctx)   # dynamic: grow until unique
        if at is None:
            continue                        # no unique anchor at any length → decline
        safe = _safe_boundary(qmd_text, at)
        if safe is None:
            continue
        tid += 1
        planned.append((safe, tid, pno, md, label))

    declined = len(blocks) - len(planned)
    if not planned:
        if declined:
            log.info('postfix: %d missing table(s) had no anchor — left for manual '
                     '(not appended)', declined)
        return 0, llm_cost[0]

    # place markers back-to-front (offsets stay valid), then fill by token
    for safe, tid, _pno, _md, _label in sorted(planned, key=lambda p: p[0], reverse=True):
        qmd_text = qmd_text[:safe] + _TBL_MARKER_TMPL.format(tid) + '\n\n' + qmd_text[safe:]
    for _safe, tid, pno, md, label in planned:
        block = ('<!-- postfix: table recovered in place (source p{}, {}) -->\n\n{}'
                 .format(pno + 1, label, md))
        marker = _TBL_MARKER_TMPL.format(tid)
        if marker in qmd_text:
            qmd_text = qmd_text.replace(marker, block, 1)

    leftover = _TBL_MARKER_RE.findall(qmd_text)
    if leftover:
        log.warning('postfix: %d table marker(s) unfilled — stripped', len(leftover))
        qmd_text = _TBL_MARKER_RE.sub('', qmd_text)

    qmd_path.write_text(qmd_text, encoding='utf-8')
    log.info('postfix: recovered %d missing table(s) in place from the source PDF%s',
             len(planned),
             '' if not declined else ' (%d unanchorable, left for manual)' % declined)
    return len(planned), llm_cost[0]


_WINDOW_RE = re.compile(r'<<<PAGE\s+(\d+)>>>')


def _convert_window(api_key, doc, pno, span=1):
    """Convert pages pno-span … pno+span in ONE call. Returns ({page_idx: markdown}, cost).

    Two reasons for the window rather than the bare page:
      * sentences spanning a page break convert coherently;
      * the neighbours give us a splice key. Matching the MODEL'S rendering of a
        neighbouring page against the .qmd works far better than matching raw PDF text,
        because the .qmd is itself model-rendered — raw lines differ by joined line
        breaks, normalised punctuation and markdown escaping, so they rarely match.
    """
    pages = [p for p in range(pno - span, pno + span + 1) if 0 <= p < doc.page_count]
    if not pages:
        return {}, 0.0
    body = '\n\n'.join('<<<PAGE {}>>>\n{}'.format(p + 1, doc[p].get_text()[:2200])
                       for p in pages)
    prompt = (
        'Below are consecutive pages of a technical document, each introduced by a '
        '<<<PAGE n>>> marker. Convert the BODY PROSE of each page to clean Markdown.\n'
        'Reproduce it faithfully and in full: do not summarise, do not add commentary '
        'or headings of your own, and do not invent anything. Omit page headers and '
        'footers, figure labels, and table contents.\n'
        'Emit the same <<<PAGE n>>> markers, each followed by that page\'s prose.\n\n'
        + body
    )
    try:
        response, usage = _post_with_retries(
            api_key=api_key,
            payload={'model': _REPAIR_MODEL,
                     'messages': [{'role': 'user', 'content': prompt}],
                     'max_tokens': 8192},
            label='postfix-window-p{}'.format(pno + 1), timeout=180,
        )
    except RuntimeError as e:
        log.warning('Postfix window p%d failed: %s', pno + 1, e)
        return {}, 0.0
    cost = (usage or {}).get('cost', 0.0)
    if not response:
        return {}, cost
    chunks = _WINDOW_RE.split(response)
    out = {}
    for i in range(1, len(chunks) - 1, 2):
        try:
            out[int(chunks[i]) - 1] = chunks[i + 1].strip()
        except ValueError:
            continue
    return out, cost


def _postfix_headings(qmd_path, out_dir):
    """Restore source-outline headings the converter dropped. Deterministic:
    the PDF bookmark outline gives (level, title, page); each missing heading is
    inserted right before the first line of its own section content that
    survived into the .qmd. No anchor found → skipped, never guessed."""
    import fitz
    from .verify.textutil import normalize
    from .verify.checks.heading_hierarchy import _fuzzy, _qmd_headings, _similar

    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0
    qmd_text = qmd_path.read_text(encoding='utf-8')
    qmd_keys = [_fuzzy(t) for t in _qmd_headings(qmd_text)]

    doc = fitz.open(str(source_pdf))
    try:
        toc = doc.get_toc(simple=True)          # [[level, title, 1-based page], …]
        if not toc:
            return 0
        missing = [(lvl, title, pg - 1) for lvl, title, pg in toc
                   if normalize(title)
                   and not any(_similar(_fuzzy(normalize(title)), q) for q in qmd_keys)]
        if not missing:
            return 0

        hay = qmd_text.lower()

        def _section_anchor(title, pno):
            """Start-of-line offset in the .qmd of the first unique line that
            follows the heading in the source (same page, then the next)."""
            seen_title = False
            for p in (pno, pno + 1):
                if not (0 <= p < doc.page_count):
                    continue
                for ln in doc[p].get_text().splitlines():
                    s = ' '.join(ln.split())
                    if not seen_title and normalize(s) and normalize(title) in normalize(s):
                        seen_title = True
                        continue            # the heading line itself is not an anchor
                    if len(s) < 40:
                        continue
                    probe = s[:60].lower()
                    i = hay.find(probe)
                    if i != -1 and hay.find(probe, i + 1) == -1:
                        return qmd_text.rfind('\n', 0, i) + 1
            return None

        plans = []
        for lvl, title, pno in missing:
            at = _section_anchor(title, pno)
            if at is not None:
                # drop manual section numbers, matching the converter's own style
                clean = re.sub(r'^\s*[\d.]+\s*', '', title).strip() or title.strip()
                plans.append((at, '#' * max(1, min(6, lvl)) + ' ' + clean))

        for at, heading in sorted(plans, reverse=True):
            qmd_text = qmd_text[:at] + heading + '\n\n' + qmd_text[at:]
        if plans:
            qmd_path.write_text(qmd_text, encoding='utf-8')
            log.info('postfix: restored %d missing heading(s) from the source outline '
                     '(%d unanchorable, skipped)', len(plans), len(missing) - len(plans))
        return len(plans)
    finally:
        doc.close()


# ── Marker-based missing-text repair ────────────────────────────────────────────
# Instead of re-converting whole damaged pages (expensive: ~90% of the generated
# output is content already present, then discarded), we diff the source against
# the .qmd, anchor each gap on the SURVIVING sentence right before it, drop a
# placeholder marker there, batch every gap's source text into ONE LLM call to
# clean-format it, then fill each marker. Cost scales with what's actually missing,
# not document length. A fidelity guard falls back to raw text if the model drifts.

_MARKER_TMPL = '<!--pdf2md-repair-{}-->'
_MARKER_RE = re.compile(r'<!--pdf2md-repair-\d+-->')
_GAP_RE = re.compile(r'<<<GAP\s+(\d+)>>>')
_MIN_INJECT_TOKENS = 8       # a shorter run is a fragment, not a body sentence
_SOLID_TOKENS = 6            # a sentence long enough to be a trustworthy anchor; shorter
#                             survivors ("Pol.", "J.") don't break a gap, so a collapsed
#                             region (mostly missing, sparse fragment survivors) stays one
#                             gap anchored on the solid sentence before it — not appended
_ANCHOR_PROBE = 8            # tokens of the before/after sentence used to locate it
_ANCHOR_PROBE_LONG = 15      # escalated probe when the short one isn't unique (Tier 2a)
_DEDUP_OVERLAP = 0.6         # a gap whose text is already this present is a false miss


def _clean_raw(text):
    """Minimal deterministic cleanup for raw PDF text (fallback when the LLM drifts):
    join hyphenated line breaks, collapse whitespace. No reflow, no reformat."""
    text = re.sub(r'-\s*\n\s*', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _qmd_word_offsets(qmd_text):
    """Lowercased alnum words of the .qmd with each word's start and end char offset —
    the coordinate space anchor sentences are located in."""
    words, starts, ends = [], [], []
    for m in re.finditer(r'[a-z0-9]+', qmd_text.lower()):
        words.append(m.group())
        starts.append(m.start())
        ends.append(m.end())
    return words, starts, ends


def _find_unique(words, probe):
    """Index of the UNIQUE consecutive occurrence of `probe` (a token list) in the
    .qmd word stream. None if absent or ambiguous — the anchor never guesses."""
    n = len(probe)
    if not n:
        return None
    hit = -1
    for i in range(len(words) - n + 1):
        if words[i:i + n] == probe:
            if hit != -1:
                return None                 # not unique
            hit = i
    return hit if hit != -1 else None


def _locate_after(words, ends, probe):
    """Char offset just past a unique occurrence of `probe`. None if absent/ambiguous."""
    i = _find_unique(words, probe)
    return ends[i + len(probe) - 1] if i is not None else None


def _anchor_by_context(words, ends, ctx, cap=40):
    """Dynamic minimal-unique anchor: grow the probe outward from the TAIL of `ctx`
    (the text right before where the content belongs) until that suffix is UNIQUE in
    the .qmd, and return the char offset just after it. A short suffix that repeats
    gets disambiguated by adding more preceding context. None if no suffix up to
    `cap` tokens is unique — absent (reworded/missing context) or irreducibly
    repeated — in which case the caller declines rather than guess."""
    n = len(ctx)
    if not n:
        return None
    for k in range(min(_ANCHOR_PROBE, n), min(cap, n) + 1):
        i = _find_unique(words, ctx[-k:])
        if i is not None:
            return ends[i + k - 1]
    return None


def _safe_boundary_before(text, pos):
    """Last blank-line boundary at/before `pos` that is not inside a code fence — so a
    marker inserted BEFORE the following sentence lands at a paragraph start, never
    mid-block. None when no safe boundary precedes `pos`."""
    cut = text.rfind('\n\n', 0, pos)
    while cut != -1:
        cand = cut + 2
        if text.count('```', 0, cand) % 2 == 0:
            return cand
        cut = text.rfind('\n\n', 0, cut)
    return None


def _anchor_gap(units, present, before_idx, after_idx, words, starts, ends, qmd_text):
    """Find a safe insertion offset for a gap, escalating through precision-preserving
    tiers, stopping at the first that yields a UNIQUE anchor. Every tier matches a real
    word run, so placement stays exact; when none is unique we return None and decline
    rather than guess.

      Tier 1  : tail of the surviving sentence right before the gap (short probe)
      Tier 2a : same, longer probe (more often unique)
      Tier 2b : the two surviving sentences before the gap, combined (short sentences
                become distinctive together)
      Tier 2c : head of the surviving sentence AFTER the gap — insert before it
                (rescues gaps whose leading context wasn't unique, incl. top-of-doc)
    """
    def _order_ok(at):
        # the following surviving sentence, if locatable, must sit after `at`
        if after_idx is None:
            return True
        i = _find_unique(words, units[after_idx][2][:_ANCHOR_PROBE])
        return i is None or ends[i] > at

    # Tiers 1 / 2a / 2b — anchor AFTER a surviving sentence preceding the gap
    if before_idx is not None:
        combos = [units[before_idx][2]]
        if before_idx - 1 >= 0 and present[before_idx - 1]:
            combos.append(units[before_idx - 1][2] + units[before_idx][2])
        for toks in combos:
            for n in (_ANCHOR_PROBE, _ANCHOR_PROBE_LONG):
                i = _find_unique(words, toks[-min(n, len(toks)):])
                if i is None:
                    continue
                at = ends[i + min(n, len(toks)) - 1]
                if _order_ok(at):
                    safe = _safe_boundary(qmd_text, at)
                    if safe is not None:
                        return safe

    # Tier 2c — anchor BEFORE the surviving sentence following the gap
    if after_idx is not None:
        atk = units[after_idx][2]
        for n in (_ANCHOR_PROBE, _ANCHOR_PROBE_LONG):
            i = _find_unique(words, atk[:min(n, len(atk))])
            if i is None:
                continue
            safe = _safe_boundary_before(qmd_text, starts[i])
            if safe is not None:
                return safe
    return None


def _faithful(raw, md):
    """True if the model's rendering preserved the source (didn't summarise, drop,
    or rewrite): most of the source's 4-gram shingles survive in the output."""
    from .verify.textutil import shingles, tokens
    rs = shingles(tokens(raw))
    if not rs:
        return True
    return len(rs & shingles(tokens(md))) / len(rs) >= 0.5


def _llm_convert_gaps(api_key, gaps, model=_REPAIR_MODEL, char_budget=12000):
    """Convert each gap's raw source text to clean Markdown in as few calls as the
    output budget allows. gaps: [(gap_id, raw_text)]. Returns ({gap_id: md}, cost)."""
    out, total_cost = {}, 0.0

    def _flush(batch):
        nonlocal total_cost
        if not batch:
            return
        body = '\n\n'.join('<<<GAP {}>>>\n{}'.format(g, _clean_raw(r)) for g, r in batch)
        prompt = (
            'The excerpts below were dropped from a technical document during an '
            'earlier PDF-to-Markdown conversion. Convert EACH excerpt to clean '
            'Markdown, faithfully and in full: reproduce the wording verbatim; do '
            'NOT summarise, translate, reorder, add commentary, or invent anything. '
            'Fix only mechanical artefacts (hyphenation, broken spacing). Keep each '
            '<<<GAP n>>> marker on its own line immediately before its excerpt.\n\n'
            + body)
        try:
            resp, usage = _post_with_retries(
                api_key=api_key,
                payload={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
                         'max_tokens': 8192},
                label='repair-gaps', timeout=180,
            )
        except RuntimeError as e:
            log.warning('gap-convert batch failed: %s', e)
            return
        total_cost += (usage or {}).get('cost', 0.0)
        if not resp:
            return
        parts = _GAP_RE.split(resp)
        for i in range(1, len(parts) - 1, 2):
            try:
                out[int(parts[i])] = parts[i + 1].strip()
            except ValueError:
                continue

    batch, size = [], 0
    for g, r in gaps:
        if batch and size + len(r) > char_budget:
            _flush(batch)
            batch, size = [], 0
        batch.append((g, r))
        size += len(r)
    _flush(batch)
    return out, total_cost


def _build_units(source_pdf):
    """Ordered source prose units: [(page, raw_sentence, norm_tokens)] in reading
    order, with running headers/footers and TOC leaders dropped."""
    from .verify.textutil import normalize, pdf_lines, split_sentences, tokens
    from .verify.checks.text_coverage import _join_wrapped, _TOC_LEADER_RE

    lines = pdf_lines(source_pdf, exclude_boxes_by_page={})
    total_pages = (max((p for p, _ in lines), default=-1) + 1) or 1

    # digit-insensitive chrome signature: "Page | 9" and "Page | 15" collapse to one
    # key, so numbered running headers/footers are recognised as chrome (an
    # exact-text match would treat each numbered variant as unique and leak it in)
    def _sig(t):
        return re.sub(r'\d+', '#', normalize(t))

    seen = defaultdict(set)
    for pno, txt in lines:
        seen[_sig(txt)].add(pno)
    chrome = {s for s, ps in seen.items() if len(ps) >= max(3, total_pages * 0.5)}

    by_page = defaultdict(list)
    for pno, txt in lines:
        n = normalize(txt)
        if not n or _sig(txt) in chrome or _TOC_LEADER_RE.search(txt):
            continue                        # chrome, blank, or table-of-contents leader
        by_page[pno].append(txt)

    units = []
    for pno in sorted(by_page):
        for sent in split_sentences(_join_wrapped(by_page[pno])):
            tk = tokens(sent)
            if tk:
                units.append((pno, sent.strip(), tk))
    return units


def _region_gaps(units, present):
    """Group missing sentences into gaps bounded by SOLID surviving sentences.

    A gap spans everything between two solid survivors: missing sentences AND tiny
    non-anchorable fragments ("Pol.", "J.") alike. So a heavily collapsed region — a
    references section that mostly failed to convert, leaving only fragments — becomes
    ONE region gap anchored on the solid sentence before it, recovered in place at the
    region boundary rather than appended. Each gap's `run` lists the MISSING units only;
    surviving fragments stay where they are (no duplication). Returns
    [(before_idx, run_of_missing_indices, after_idx)]."""
    def _solid(k):
        return present[k] and len(units[k][2]) >= _SOLID_TOKENS

    gaps, i, n = [], 0, len(units)
    while i < n:
        if _solid(i):
            i += 1
            continue
        j = i
        while j < n and not _solid(j):
            j += 1
        run = [k for k in range(i, j) if not present[k]]
        if run and any(len(units[k][2]) >= _MIN_INJECT_TOKENS for k in run):
            gaps.append((i - 1 if i > 0 else None, run, j if j < n else None))
        i = j
    return gaps


_SUP_DIGITS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


def _postfix_footnotes(qmd_path, out_dir):
    """Rescue orphaned footnote definitions whose content Quarto would silently drop.

    A `[^n]:` definition with no matching `[^n]` reference is 'orphaned', and Quarto
    renders an orphaned definition to NOTHING — its text is lost. This is unavoidable
    for footnotes that annotate table cells: complex tables are emitted as ```{=html}```
    raw blocks, and inside a raw block a `[^n]` renders literally (never links), so the
    mark stays a <sup>n</sup> and the definition is left orphaned.

    Fix, deterministic and no LLM: rewrite each such definition — it already sits right
    after its table — into a visible note line `^n^ text`, the conventional table-note.
    The <sup>n</sup> mark stays put in the cell. A definition is converted only when its
    digit actually appears as a superscript mark (<sup>n</sup> or unicode ⁿ) in the doc;
    a mark-less definition (e.g. a model-invented one) is left untouched and logged, so
    we never emit a note whose number points at nothing."""
    from .verify.checks.footnote_placement import _QMD_REF, _QMD_DEF

    qmd_text = qmd_path.read_text(encoding='utf-8')
    ref_ids = set(_QMD_REF.findall(qmd_text))
    orphaned = [fid for fid in dict.fromkeys(_QMD_DEF.findall(qmd_text))
                if fid not in ref_ids]
    if not orphaned:
        return 0

    lines = qmd_text.split('\n')
    converted, skipped = 0, 0
    for fid in orphaned:
        if not fid.isdigit():
            skipped += 1
            continue
        # the mark must exist as <sup>n</sup> or a unicode superscript, else it dangles
        has_mark = (re.search(r'<sup>\s*{}\s*</sup>'.format(re.escape(fid)), qmd_text)
                    or fid.translate(_SUP_DIGITS) in qmd_text)
        if not has_mark:
            skipped += 1
            continue
        prefix = '[^{}]:'.format(fid)
        for i, ln in enumerate(lines):
            if ln.startswith(prefix):
                # a def with an indented continuation line would become a code block if
                # de-prefixed — rare; leave those for a human rather than mangle them
                nxt = lines[i + 1] if i + 1 < len(lines) else ''
                if nxt[:4] == '    ' and nxt.strip():
                    skipped += 1
                    break
                lines[i] = '^{}^{}'.format(fid, ln[len(prefix):])
                converted += 1
                break
    if not converted:
        return 0
    qmd_path.write_text('\n'.join(lines), encoding='utf-8')
    log.info('postfix: recovered %d dropped footnote(s) as visible table-note(s)%s',
             converted, ' (%d left, no in-doc mark)' % skipped if skipped else '')
    return converted


def _postfix_missing_text(qmd_path, out_dir, api_key, text_check):
    """Recover dropped body sentences by anchoring each gap on the surviving
    sentence before it and filling a placeholder marker with an LLM-cleaned (raw
    fallback) rendering of the gap's source text. Returns (pages, items, cost)."""
    from .verify.textutil import qmd_to_plain, shingles, tokens

    stem = qmd_path.stem
    source_pdf = _body_pdf(out_dir, stem)   # prefer the chrome-stripped body copy
    if not source_pdf.exists():
        return 0, 0, 0.0

    qmd_text = qmd_path.read_text(encoding='utf-8')
    qmd_sh = shingles(tokens(qmd_to_plain(qmd_text)))

    try:
        units = _build_units(source_pdf)
    except Exception as e:                  # noqa: BLE001 — repair must never abort
        log.warning('Missing-text repair: could not read source: %s', e)
        return 0, 0, 0.0
    if not units:
        return 0, 0, 0.0

    def _present(tk):
        sh = shingles(tk)
        return bool(sh) and len(sh & qmd_sh) / len(sh) >= 0.5

    present = [_present(tk) for _, _, tk in units]
    gaps = _region_gaps(units, present)
    if not gaps:
        return 0, 0, 0.0

    words, starts, ends = _qmd_word_offsets(qmd_text)
    planned = []                            # (safe_offset, gap_id, page, raw_text)
    gid = 0
    for before_idx, run, after_idx in gaps:
        safe = _anchor_gap(units, present, before_idx, after_idx,
                           words, starts, ends, qmd_text)
        if safe is None:                    # no unique anchor in any tier → decline
            continue
        raw = ' '.join(units[k][1] for k in run)
        # dedup guard: if the gap text is already largely present, the diff mis-flagged
        # it — do not inject a duplicate
        gsh = shingles(tokens(raw))
        if gsh and len(gsh & qmd_sh) / len(gsh) >= _DEDUP_OVERLAP:
            continue
        gid += 1
        planned.append((safe, gid, units[run[0]][0], raw))

    if not planned:
        return 0, 0, 0.0

    # place every marker first (back-to-front so offsets stay valid), then fill by
    # token — so the fill is order-independent and no marker can shift another
    for safe, gid, _pno, _raw in sorted(planned, key=lambda p: p[0], reverse=True):
        marker = _MARKER_TMPL.format(gid)
        qmd_text = qmd_text[:safe] + marker + '\n\n' + qmd_text[safe:]

    rendered, cost = _llm_convert_gaps(api_key, [(gid, raw) for _, gid, _, raw in planned])

    pages, items = set(), 0
    for _safe, gid, pno, raw in planned:
        content = rendered.get(gid)
        if not content or not _faithful(raw, content):
            content = _clean_raw(raw)       # model drifted or dropped it → raw fallback
        block = ('<!-- postfix: recovered in place (source p{}) -->\n\n{}'
                 .format(pno + 1, content))
        marker = _MARKER_TMPL.format(gid)
        if marker in qmd_text:
            qmd_text = qmd_text.replace(marker, block, 1)
            pages.add(pno)
            items += 1

    # safety sweep: no unfilled marker may ever reach the output
    leftover = _MARKER_RE.findall(qmd_text)
    if leftover:
        log.warning('postfix: %d repair marker(s) unfilled — stripped', len(leftover))
        qmd_text = _MARKER_RE.sub('', qmd_text)

    if not items:
        return 0, 0, cost
    qmd_path.write_text(qmd_text, encoding='utf-8')
    log.info('postfix: recovered %d missing passage(s) from %d page(s) via marker '
             'anchoring (batched %s)', items, len(pages), _REPAIR_MODEL)
    return len(pages), items, cost
