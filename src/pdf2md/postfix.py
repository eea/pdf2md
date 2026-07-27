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


def _cov(results, name):
    r = next((r for r in results if r.name == name), None)
    return r.metric if r else None


def _verify_now(qmd_path, out_dir):
    """Re-run verify on the current .qmd (local, no LLM). None when verify fails —
    the loop then keeps its previous results rather than aborting the repair."""
    from .verify import VerifyContext, run_verify
    stem = qmd_path.stem
    try:
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
        return run_verify(ctx)
    except Exception as e:                  # noqa: BLE001 — repair must never abort
        log.warning('Re-verify during repair loop failed: %s', e)
        return None


def _deterministic_pass(qmd_path, out_dir, verify_results, api_key):
    """Step 1 — deterministic fixes: header bleed, code blocks, tables, links,
    headings. €0 except the optional LLM table structuring, which is bounded by
    how many tables are actually absent. Returns (fixes, cost_usd)."""
    fixes, cost = [], 0.0
    verify_by_name = {r.name: r for r in verify_results}

    table_check = verify_by_name.get('table_coverage')
    if table_check and table_check.status in ('warn', 'fail'):
        fixed = _strip_header_bleed(qmd_path)
        if fixed:
            fixes.append('header_bleed: stripped {} header fragment(s)'.format(fixed))

    code_check = verify_by_name.get('code_block_presence')
    if code_check and code_check.status in ('warn', 'fail'):
        recovered = _recover_code_blocks(qmd_path, out_dir)
        if recovered:
            fixes.append('code_blocks: recovered {} code block(s) from source PDF'
                         .format(recovered))

    # Not gated on the check's status: whole tables can be absent while the aggregate
    # still reads ok, because a missing table's boilerplate is supplied by its siblings.
    if table_check:
        n_tbl, tbl_cost = _recover_missing_tables(qmd_path, out_dir, api_key=api_key)
        cost += tbl_cost
        if n_tbl:
            fixes.append('tables: re-emitted {} missing table(s) from source'.format(n_tbl))

    # The href lives in a PDF annotation the model never sees, so this is the only
    # way those links can survive.
    link_check = verify_by_name.get('link_preservation')
    if link_check and link_check.status in ('warn', 'fail'):
        n_in, n_list = _recover_links(qmd_path, out_dir)
        if n_in or n_list:
            fixes.append('links: {} restored inline, {} listed'.format(n_in, n_list))

    head_check = verify_by_name.get('heading_hierarchy')
    if head_check and head_check.status in ('warn', 'fail'):
        n_head = _postfix_headings(qmd_path, out_dir)
        if n_head:
            fixes.append('headings: restored {} missing heading(s) from source outline'
                         .format(n_head))

    artifacts_check = verify_by_name.get('artifacts')
    if artifacts_check and artifacts_check.status in ('warn', 'fail'):
        n_artifacts = _strip_artifacts(qmd_path)
        if n_artifacts:
            fixes.append('artifacts: stripped {} leftover Office/reference artifact(s)'
                         .format(n_artifacts))

    return fixes, cost


def _llm_text_pass(qmd_path, out_dir, api_key, verify_results):
    """Step 2 — LLM missing-text rescue, only on pages verify still flags.
    Returns (fixes, cost_usd, items_recovered)."""
    text_check = next((r for r in verify_results if r.name == 'text_coverage'), None)
    if not (text_check and text_check.status in ('warn', 'fail') and api_key):
        return [], 0.0, 0
    rescued, items, repair_cost = _postfix_missing_text(qmd_path, out_dir, api_key,
                                                        text_check)
    fixes = []
    if rescued:
        fixes.append('missing_text: {} items recovered from {} pages'
                     .format(items, rescued))
    # calls are billed even when nothing lands
    return fixes, repair_cost, (items if rescued else 0)


def _vision_patch_pass(qmd_path, out_dir, api_key, verify_results):
    """Step 3 — vision patches on pages still flagged after the cheap steps.

    A vision model compares each flagged source page image against the .qmd and
    proposes exact old→new string patches (review.inspect_pages); unambiguous ones
    are applied. Judged from the source page + .qmd text alone — no mid-loop
    Quarto render, whose stale output would mislead the model. Returns
    (fixes, cost_usd)."""
    if not api_key:
        return [], 0.0
    from .review import _apply_patches, _flagged_pages, inspect_pages
    flagged = _flagged_pages(verify_results)
    if not flagged:
        return [], 0.0
    source_pdf = out_dir / '{}.source.pdf'.format(qmd_path.stem)
    if not source_pdf.exists():
        return [], 0.0
    defects = inspect_pages(source_pdf, None, flagged, api_key, qmd_path=qmd_path)
    cost = sum((d.get('cost_usd') or 0.0) for d in defects)
    applied = _apply_patches(qmd_path, defects) if defects else 0
    fixes = []
    if applied:
        fixes.append('vision: applied {} patch(es) across {} flagged page(s)'
                     .format(applied, len(defects)))
    return fixes, cost


def _fmt_cov(v):
    return '—' if v is None else '{:.1f}%'.format(v)


def _write_repair_report(out_dir, stem, iterations, stop_reason, total_cost):
    """repair_report.md — coverage summary table (Before / per-iteration / Final)
    plus short per-iteration notes. Per-page patch history belongs in
    review_llm.md, not here."""
    md = ['# Iterative Repair — {}'.format(stem), '']
    if not iterations:
        md.append('No repair iterations ran.')
    else:
        first, last = iterations[0], iterations[-1]
        md += ['| Iteration | Fixes applied | Cost (USD) | Text coverage | Table coverage |',
               '|---|---|---|---|---|',
               '| Before | — | — | {} | {} |'.format(
                   _fmt_cov(first['text_cov_before']),
                   _fmt_cov(first.get('table_cov_before')))]
        for it in iterations:
            md.append('| {} | {} | {:.4f} | {} | {} |'.format(
                it['iteration'], len(it['fixes']), it['cost'],
                _fmt_cov(it['text_cov_after']),
                _fmt_cov(it.get('table_cov_after'))))
        md.append('| Final | {} | {:.4f} | {} | {} |'.format(
            sum(len(it['fixes']) for it in iterations), total_cost,
            _fmt_cov(last['text_cov_after']),
            _fmt_cov(last.get('table_cov_after'))))
        md += ['', 'Stopped: {}.'.format(stop_reason)]
        md += ['', '## Iteration notes', '']
        for it in iterations:
            md.append('- Iteration {}: {}'.format(
                it['iteration'], '; '.join(it['fixes']) or 'no fixes applied'))
    path = Path(out_dir) / 'repair_report.md'
    path.write_text('\n'.join(md) + '\n', encoding='utf-8')
    log.info('Wrote %s (%d iteration(s))', path.name, len(iterations))
    return path


def run_repair_loop(qmd_path, out_dir, verify_results, api_key, max_iterations=3,
                    *, meta=None):
    """Iterative repair: detect errors → fix what we can → re-detect → repeat.

    Each iteration runs, in order:
      1. deterministic fixes (€0): header bleed, code blocks, tables, links, headings
      2. LLM missing-text rescue on pages verify still flags
      3. vision patches on pages still flagged after steps 1–2 (re-detected first)
      4. re-verify → measure coverage
    and loops back while coverage improved and iterations remain. Stops on: a clean
    verify, an iteration that applied nothing, no coverage improvement, or
    max_iterations — whichever hits first.

    Writes repair_report.md (iteration summary table + notes) and refreshes verify_report.md
    with the final post-repair state. Returns a cumulative summary dict.
    """
    summary = {'iterations': [], 'postfixes_applied': [], 'cost_usd': 0.0,
               'items_recovered': 0, 'stop_reason': ''}
    if max_iterations <= 0 or not verify_results:
        return summary

    from .verify import overall_status

    results = verify_results
    reverified = False
    stop_reason = 'max iterations ({}) reached'.format(max_iterations)
    for iteration in range(1, max_iterations + 1):
        stat = {'iteration': iteration, 'fixes': [], 'cost': 0.0,
                'text_cov_before': _cov(results, 'text_coverage'),
                'text_cov_after': None,
                'table_cov_before': _cov(results, 'table_coverage'),
                'table_cov_after': None}

        # 1. deterministic fixes — run every iteration, they are (near) free
        fixes, cost = _deterministic_pass(qmd_path, out_dir, results, api_key)

        # 2. LLM missing-text rescue on pages still flagged
        f2, c2, items = _llm_text_pass(qmd_path, out_dir, api_key, results)
        fixes += f2
        cost += c2
        summary['items_recovered'] += items

        # 3. vision patches on pages STILL flagged after 1–2 → re-detect first so
        #    the vision step never spends on a page the cheap steps already fixed
        mid = results
        if fixes:
            fresh = _verify_now(qmd_path, out_dir)
            if fresh is not None:
                mid, reverified = fresh, True
        f3, c3 = _vision_patch_pass(qmd_path, out_dir, api_key, mid)
        fixes += f3
        cost += c3

        stat['fixes'] = fixes
        stat['cost'] = round(cost, 6)
        summary['cost_usd'] += cost

        if not fixes:
            stat['text_cov_after'] = stat['text_cov_before']
            stat['table_cov_after'] = stat['table_cov_before']
            summary['iterations'].append(stat)
            results = mid
            stop_reason = 'iteration {} applied no fixes'.format(iteration)
            break

        summary['postfixes_applied'] += ['[iter {}] {}'.format(iteration, f)
                                         for f in fixes]

        # 4. re-verify → measure this iteration's effect
        if f3:
            fresh = _verify_now(qmd_path, out_dir)
            if fresh is not None:
                mid, reverified = fresh, True
        results = mid
        stat['text_cov_after'] = _cov(results, 'text_coverage')
        stat['table_cov_after'] = _cov(results, 'table_coverage')
        summary['iterations'].append(stat)
        log.info('repair iteration %d: %d fix(es), $%.4f, text %s → %s',
                 iteration, len(fixes), cost,
                 _fmt_cov(stat['text_cov_before']), _fmt_cov(stat['text_cov_after']))

        # 5–6. cutoff: stop when verify is clean or coverage stopped improving
        if overall_status(results) == 'ok':
            stop_reason = 'verify ok after iteration {}'.format(iteration)
            break
        before, after = stat['text_cov_before'], stat['text_cov_after']
        if before is not None and after is not None and after <= before:
            stop_reason = 'no coverage improvement in iteration {}'.format(iteration)
            break

    summary['stop_reason'] = stop_reason
    try:
        summary['report'] = str(_write_repair_report(
            out_dir, qmd_path.stem, summary['iterations'], stop_reason,
            summary['cost_usd']))
    except Exception as e:                  # noqa: BLE001 — reporting is best-effort
        log.warning('Could not write repair_report.md: %s', e)

    if reverified:
        try:
            from .verify import write_report
            report_meta = dict(meta or {})
            report_meta['postfixes'] = summary['postfixes_applied']
            report_meta['cost_repair'] = summary['cost_usd']
            write_report(results, out_dir, meta=report_meta)
        except Exception as e:              # noqa: BLE001
            log.warning('Could not refresh verify_report.md after repair: %s', e)
        summary['verify_after'] = overall_status(results)
        tc = next((r for r in results if r.name == 'text_coverage'), None)
        tbl = next((r for r in results if r.name == 'table_coverage'), None)
        summary['coverage_after'] = {
            'text': tc.metric if tc else None,
            'text_effective': (tc.detail or {}).get('effective') if tc else None,
            'text_recovered': (tc.detail or {}).get('recovered', 0) if tc else 0,
            'table': tbl.metric if tbl else None,
        }

    return summary


_ARTIFACT_PATTERNS = (
    re.compile(r"[Ee]rror!\s*Reference source not found\.?"),
    re.compile(r"[Ee]rror!\s*Bookmark not defined\.?"),
    re.compile(r"[Ee]rror!\s*Hyperlink reference not valid\.?"),
    re.compile(r"#REF!"),
)


def _strip_artifacts(qmd_path):
    """Remove leftover Word/Office field-code artifacts (e.g. "Error! Reference
    source not found.", "#REF!") that sometimes survive PDF conversion. Free —
    pure regex substitution, no LLM. Returns the number of replacements made."""
    text = qmd_path.read_text(encoding='utf-8')
    n = 0
    for pat in _ARTIFACT_PATTERNS:
        text, count = pat.subn('[missing reference]', text)
        n += count
    if n:
        qmd_path.write_text(text, encoding='utf-8')
    return n


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

# standalone page-number lines: "12", "Page 12", "12 of 345", "12 / 345"
_PAGE_NUMBER_RE = re.compile(r'(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?')


def strip_chrome_qmd(qmd_path, out_dir):
    """Strip running headers/footers and page numbers from the final .qmd.

    Deterministic, zero LLM cost — the end-of-pipeline complement to 1:1
    conversion: verify and review compare the chrome-complete output, then this
    removes the chrome the reader doesn't want. Chrome lines are identified from
    the SOURCE PDF two ways:
      * geometry — text lines whose centre falls inside a
        marginchrome.detect_running_chrome() band (catches per-page variants
        like "Page 12", digits and all);
      * frequency — lines whose normalized text repeats on
        >= _CHROME_LINE_MIN_PAGES source pages (catches chrome the band
        detector missed, same rule as _strip_chrome_lines).
    A .qmd line is dropped when its normalized text matches a chrome signature
    or is a bare page-number line. YAML frontmatter, headings and code-fence
    interiors are never touched (a running header often equals the document
    title, which legitimately survives as a heading). Returns lines removed.
    """
    from .marginchrome import detect_running_chrome
    from .verify.textutil import _rect_center_in, normalize

    try:
        import fitz
    except ImportError:
        log.warning('chrome strip skipped: PyMuPDF (fitz) not available')
        return 0
    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0

    regions = detect_running_chrome(source_pdf)

    chrome_sigs = set()
    line_freq = Counter()
    doc = fitz.open(str(source_pdf))
    try:
        for pno in range(doc.page_count):
            boxes = regions.get(pno, [])
            for block in doc[pno].get_text('dict').get('blocks', []):
                for line in block.get('lines', []):
                    txt = ''.join(s['text'] for s in line['spans']).strip()
                    n = normalize(txt)
                    if not n:
                        continue
                    line_freq[n] += 1
                    if boxes and _rect_center_in(line['bbox'], boxes):
                        chrome_sigs.add(n)
    finally:
        doc.close()

    chrome_sigs |= {n for n, c in line_freq.items() if c >= _CHROME_LINE_MIN_PAGES}
    if not chrome_sigs:
        return 0

    text = qmd_path.read_text(encoding='utf-8')
    lines = text.split('\n')

    # leave YAML frontmatter untouched
    body_start = 0
    if lines and lines[0].strip() == '---':
        for i in range(1, len(lines)):
            if lines[i].strip() == '---':
                body_start = i + 1
                break

    kept = lines[:body_start]
    removed = 0
    in_fence = False
    for ln in lines[body_start:]:
        s = ln.strip()
        if s.startswith('```'):
            in_fence = not in_fence
        elif not in_fence and s and not s.startswith('#'):
            n = normalize(s)
            if n and (n in chrome_sigs or _PAGE_NUMBER_RE.fullmatch(n)):
                removed += 1
                continue
        kept.append(ln)

    if not removed:
        return 0

    # collapse the blank-line runs the removals leave behind (never inside fences)
    cleaned, prev_blank, in_fence = [], False, False
    for ln in kept:
        if ln.strip().startswith('```'):
            in_fence = not in_fence
        blank = not in_fence and not ln.strip()
        if blank and prev_blank:
            continue
        prev_blank = blank
        cleaned.append(ln)

    qmd_path.write_text('\n'.join(cleaned), encoding='utf-8')
    log.info('postfix: chrome strip removed %d running header/footer/page-number '
             'line(s) from the .qmd', removed)
    return removed


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
    from .verify.textutil import normalize, qmd_to_plain

    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0.0
    qmd_text = qmd_path.read_text(encoding='utf-8')

    qtoks = set()
    for g in _qmd_grids(qmd_text):
        qtoks |= _tokens_of(g)
    # Count BODY-PROSE tokens as "already present" too. find_tables sometimes reads
    # a block of running prose as a 2-column grid; if we only compare against existing
    # .qmd *grids*, that prose looks "absent" and gets re-emitted as a duplicate
    # single-column table at EOF (observed: MRVPP ATBD "table recovered" blocks that
    # duplicated section text). Using normalize().split() keeps the same token space
    # as `distinctive` below, so the comparison is apples-to-apples.
    qtoks |= set(normalize(qmd_to_plain(qmd_text)).split())

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
