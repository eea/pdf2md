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
    source_pdf = _body_pdf(out_dir, stem)   # prefer the chrome-stripped body copy
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    qtoks = set()
    for g in _qmd_grids(qmd_text):
        qtoks |= _tokens_of(g)

    from .verify.textutil import tokens as _tok

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

        # running headers/footers were stripped from the .qmd during conversion, so they
        # must be kept OUT of the anchor context or it will never match. Digit-insensitive
        # signature, dropped when it repeats across pages (same detector as _build_units).
        total_pages = doc.page_count or 1
        seen = defaultdict(set)
        for pno, lines in page_lines.items():
            for _y, txt in lines:
                seen[_chrome_sig(txt)].add(pno)
        chrome = {s for s, ps in seen.items() if len(ps) >= max(3, total_pages * 0.5)}

        # second pass: tables with chrome-free preceding context (across the page
        # boundary, so a page-top table still has its caption/paragraph to anchor on)
        tables = []
        prev_tail = []
        for pno in range(doc.page_count):
            plines = [(y, txt) for y, txt in page_lines[pno] if _chrome_sig(txt) not in chrome]
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
                tables.append((pno, rows, _tok(' '.join(pre))))
            prev_tail = [txt for _, txt in plines]
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
