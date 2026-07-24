import json, logging, re, tempfile
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)
from .llm_client import _post_with_retries

_MIN_TABLE_PCT = 0.70
_MIN_FRAGMENT_LEN = 8
_REPAIR_MODEL = 'google/gemini-2.5-flash'

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

    # Pass 1.7: missing-table recovery (deterministic, no LLM). Not gated on the check's
    # status: whole tables can be absent while the aggregate still reads ok, because a
    # missing table's boilerplate is supplied by its siblings.
    if table_check:
        n_tbl, tbl_cost = _recover_missing_tables(qmd_path, out_dir, api_key=api_key)
        summary['cost_usd'] += tbl_cost
        if n_tbl:
            summary['postfixes_applied'].append(
                'tables: re-emitted {} missing table(s) from source'.format(n_tbl))

    # Pass 1.8: hyperlink recovery (deterministic, no LLM). The href lives in a PDF
    # annotation the model never sees, so this is the only way those links can survive.
    link_check = verify_by_name.get('link_preservation')
    if link_check and link_check.status in ('warn', 'fail'):
        n_in, n_list = _recover_links(qmd_path, out_dir)
        if n_in or n_list:
            summary['postfixes_applied'].append(
                'links: {} restored inline, {} listed'.format(n_in, n_list))

    # Pass 1.9: heading restore (deterministic, no LLM)
    head_check = verify_by_name.get('heading_hierarchy')
    if head_check and head_check.status in ('warn', 'fail'):
        n_head = _postfix_headings(qmd_path, out_dir)
        if n_head:
            summary['postfixes_applied'].append(
                'headings: restored {} missing heading(s) from source outline'.format(n_head))

    # Pass 2: missing text rescue
    text_check = verify_by_name.get('text_coverage')
    if text_check and text_check.status in ('warn', 'fail') and api_key:
        rescued, items, repair_cost = _postfix_missing_text(qmd_path, out_dir, api_key, text_check)
        summary['cost_usd'] += repair_cost   # calls are billed even when nothing lands
        if rescued:
            summary['postfixes_applied'].append(
                'missing_text: {} items recovered from {} pages'.format(items, rescued))
            summary['items_recovered'] = items

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
    already visible as text. We hold them exactly, so re-attach them: inline when the
    anchor text occurs exactly once (provably unambiguous), otherwise in a 'Source links'
    list so no citation is lost. Returns (n_inlined, n_listed).
    """
    try:
        import fitz
    except ImportError:
        return 0, 0
    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0

    qmd = qmd_path.read_text(encoding='utf-8')
    despaced = re.sub(r'\s+', '', qmd.lower())

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
                if re.sub(r'\s+', '', uri.rstrip('/').lower()) in despaced:
                    continue            # already present, nothing to restore
                anchor = ' '.join(page.get_textbox(l['from']).split())
                anchor = anchor.strip(' .,;:)（(')   # keep punctuation outside the link
                pairs.append((uri, anchor))
    finally:
        doc.close()
    if not pairs:
        return 0, 0

    inlined, listed = 0, []
    for uri, anchor in pairs:
        # a truncated URL as its own anchor is not prose — list it rather than guess
        usable = (len(anchor) >= _LINK_ANCHOR_MIN
                  and not anchor.lower().startswith(('http', 'www.', 'mailto:')))
        if usable and qmd.count(anchor) == 1:
            pos = qmd.find(anchor)
            if _safe_to_inline(qmd, pos):
                qmd = qmd[:pos] + '[{}]({})'.format(anchor, uri) + qmd[pos + len(anchor):]
                inlined += 1
                continue
        listed.append((uri, anchor))

    if listed:
        def _label(a, u):
            # a truncated copy of the URL is not a useful label — show the URL alone
            if not a or a.lower().startswith(('http', 'www.', 'mailto:')):
                return ''
            return '{} — '.format(a)

        rows = '\n'.join('- {}{}'.format(_label(a, u), u) for u, a in listed)
        qmd = (qmd.rstrip() + '\n\n<!-- postfix: source links recovered from PDF '
               'annotations -->\n\n## Source links\n\n' + rows + '\n')
    qmd_path.write_text(qmd, encoding='utf-8')
    log.info('postfix: restored %d link(s) inline, listed %d', inlined, len(listed))
    return inlined, len(listed)


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
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    qtoks = set()
    for g in _qmd_grids(qmd_text):
        qtoks |= _tokens_of(g)

    doc = fitz.open(str(source_pdf))
    try:
        tables = []
        for pno in range(doc.page_count):
            try:
                for t in doc[pno].find_tables().tables:
                    rows = [r for r in t.extract() if any(c for c in r)]
                    if rows:
                        tables.append((pno, rows))
            except Exception:               # noqa: BLE001 — one bad page must not abort
                continue
    finally:
        doc.close()

    def toks_of(rows):
        out = set()
        for r in rows:
            for c in r:
                if c:
                    out |= set(normalize(c).split())
        return out

    src_df = Counter()
    for _p, rows in tables:
        for t in toks_of(rows):
            src_df[t] += 1

    added = 0
    blocks = []
    llm_cost = [0.0]
    for pno, rows in tables:
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
        blocks.append((pno, md, src_label))
        added += 1

    if not blocks:
        return 0, llm_cost[0]
    parts = [qmd_text.rstrip()]
    for pno, md, src_label in blocks:
        parts.append('<!-- postfix: table recovered from source p{} ({}) -->\n\n{}'
                     .format(pno + 1, src_label, md))
    qmd_path.write_text('\n\n'.join(parts) + '\n', encoding='utf-8')
    log.info('postfix: re-emitted %d missing table(s) from the source PDF', added)
    return added, llm_cost[0]


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




def _postfix_missing_text(qmd_path, out_dir, api_key, text_check):
    import fitz
    from .verify.textutil import normalize, pdf_lines, split_sentences, tokens
    from .verify.textutil import qmd_to_plain, shingles

    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0, 0.0

    qmd_text = qmd_path.read_text(encoding='utf-8')

    # Find pages with missing sentences
    lines = pdf_lines(source_pdf, exclude_boxes_by_page={})
    sentence_pages = defaultdict(set)
    for pno, txt in lines:
        for sent in split_sentences(txt):
            if len(tokens(sent)) >= 5:
                sentence_pages[normalize(sent)].add(pno)

    qmd_tokens = tokens(qmd_to_plain(qmd_text))
    qmd_shingles = shingles(qmd_tokens)

    missing_by_page = defaultdict(int)
    for sent, pages in sentence_pages.items():
        stoks = tokens(sent)
        if len(stoks) <= 7:
            continue
        sh = shingles(stoks)
        if not sh:
            continue
        if len(sh & qmd_shingles) / len(sh) < 0.5:
            for p in pages:
                missing_by_page[p] += 1

    if not missing_by_page:
        return 0, 0, 0.0

    # no cap: every page with missing sentences gets a repair pass — cost is one
    # cheap flash call per damaged page, bounded by damage, not document length
    log.info('postfix: %d page(s) carry missing text, repairing all of them',
             len(missing_by_page))

    # ascending page order so earlier inserts don't invalidate later anchors
    repair_pages = sorted(missing_by_page)

    # Anchor only on lines unique in the source. Running headers/footers repeat on
    # every page, and str.find returns their FIRST hit, which would drag every insert
    # to the top of the document (observed).
    line_freq = Counter(normalize(t) for _, t in lines)
    page_lines = defaultdict(list)
    for pno, txt in lines:
        if line_freq[normalize(txt)] == 1:
            page_lines[pno].append(txt)

    doc = fitz.open(str(source_pdf))
    try:
        llm_cost = 0.0
        plans = []
        repaired = 0
        n_items = 0
        for pno in repair_pages:
            if pno >= doc.page_count:
                continue
            if not doc[pno].get_text().strip():
                continue
            window, cost = _convert_window(api_key, doc, pno)
            llm_cost += cost
            recovered = _strip_chrome_lines(window.get(pno, '').strip(), line_freq)
            recovered = _drop_already_present(recovered, qmd_text)
            if not recovered:
                continue

            # Anchor on the model's OWN rendering of the preceding page — it matches the
            # (model-rendered) .qmd far better than raw PDF text. Fall back to raw-text
            # bracketing, then to appending, rather than guessing a location.
            neighbour = [l.strip() for l in window.get(pno - 1, '').splitlines()
                         if len(l.strip()) >= 40]
            at = _insertion_point(qmd_text, neighbour) if neighbour else None
            if at is None:
                at = _bracketed_insertion_point(qmd_text, page_lines, pno)
            # Plan only — positions resolve against the ORIGINAL text. Editing as we go
            # made each insert anchor onto the previous one and chain to the end.
            plans.append((at, pno, recovered))

        # Apply back-to-front so earlier offsets stay valid.
        for at, pno, recovered in sorted(
                plans, key=lambda t: (t[0] if t[0] is not None else len(qmd_text)),
                reverse=True):
            block = ('<!-- postfix: recovered in place (source p{}) -->\n\n{}\n\n'
                     .format(pno + 1, recovered))
            if at is None:
                # no trustworthy anchor — append rather than guess a location
                qmd_text = qmd_text.rstrip() + '\n\n' + block
            else:
                qmd_text = qmd_text[:at] + block + qmd_text[at:]
            repaired += 1
            n_items += len([p for p in recovered.split('\n\n') if p.strip()])

        if not repaired:
            return 0, 0, llm_cost
        qmd_path.write_text(qmd_text, encoding='utf-8')
        log.info('postfix: re-inserted prose from %d page(s) in place via %s',
                 repaired, _REPAIR_MODEL)
        return repaired, n_items, llm_cost
    except Exception as e:                  # noqa: BLE001 — repair must never abort
        log.warning('Missing-text repair failed: %s', e)
        return 0, 0, 0.0
    finally:
        doc.close()
