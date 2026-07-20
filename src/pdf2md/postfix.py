import json, logging, re, tempfile
from collections import Counter, defaultdict
from pathlib import Path

log = logging.getLogger(__name__)
from .llm_client import _post_with_retries

_MIN_TABLE_PCT = 0.70
_MIN_FRAGMENT_LEN = 8
_MAX_REPAIR_PAGES = 5
_REPAIR_MODEL = 'google/gemini-2.5-flash'

# code-block recovery tuning
_CODE_BLOCK_MIN_MONO_SPANS = 2   # a block needs this many mono spans to count as code
_CODE_BLOCK_MONO_RATIO = 0.6     # ...and this share of its spans must be monospaced
_CODE_GROUP_GAP_PT = 40.0        # vertical gap (pt) that splits two code groups on a page
_CODE_MIN_CHARS = 12             # ignore groups smaller than this (stray inline glyphs)
_CODE_PROBE_MIN = 8              # min probe length before an in-.qmd presence check counts


def run_postfix(qmd_path, verify_results, out_dir, *, api_key=None, passes=1):
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

    # Pass 2: missing text rescue
    text_check = verify_by_name.get('text_coverage')
    if text_check and text_check.status in ('warn', 'fail') and api_key:
        rescued, items, repair_cost = _postfix_missing_text(qmd_path, out_dir, api_key, text_check)
        if rescued:
            summary['postfixes_applied'].append(
                'missing_text: {} items recovered from {} pages'.format(items, rescued))
            summary['items_recovered'] = items
            summary['cost_usd'] += repair_cost

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
            write_report(results, out_dir)
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


def _postfix_missing_text(qmd_path, out_dir, api_key, text_check):
    import fitz
    from .verify.textutil import normalize, pdf_lines, split_sentences, tokens
    from .verify.textutil import qmd_to_plain, shingles
    import urllib.request, json

    stem = qmd_path.stem
    source_pdf = out_dir / '{}.source.pdf'.format(stem)
    if not source_pdf.exists():
        return 0, 0, 0.0

    qmd_text = qmd_path.read_text(encoding='utf-8')

    # Find pages with most missing sentences
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

    worst = sorted(missing_by_page.items(), key=lambda x: -x[1])[:_MAX_REPAIR_PAGES]
    worst = [(p, c) for p, c in worst if c >= 1]
    if not worst:
        return 0, 0, 0.0

    repair_pages = [p for p, _ in worst]

    # Extract text from worst pages
    doc = fitz.open(str(source_pdf))
    page_texts = []
    for pno in repair_pages:
        if pno < doc.page_count:
            text = doc[pno].get_text()
            page_texts.append('--- Page {} ---\n{}'.format(pno + 1, text[:2000]))
    doc.close()

    context_text = '\n\n'.join(page_texts)
    prompt = (
        'Below is text extracted from pages of a technical document '
        'that was partially lost during conversion. Extract and list every specific '
        'technical detail: URLs, parameter values, accuracy percentages, file naming '
        'conventions, version numbers, abbreviations with expansions, and reference '
        'numbers. Format each as a bullet point. Only include details that are '
        'clearly present in the source text below. Do not invent or guess anything.\n\n'
        '{}'.format(context_text)
    )

    try:
        try:
            response, llm_usage = _post_with_retries(
                api_key=api_key,
                payload={'model': _REPAIR_MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 4096},
                label='postfix-repair', timeout=120,
            )
        except RuntimeError as e:
            if 'truncat' in str(e).lower() or 'output-token' in str(e).lower():
                log.warning('Postfix repair truncated: %s', e)
                return 0, 0, 0.0
            raise
        llm_cost = (llm_usage or {}).get('cost', 0.0)
        if not response:
            return 0, 0, 0.0

        pages_str = ', '.join(str(p+1) for p in repair_pages)
        qmd_text += '\n\n'
        qmd_text += '<!-- postfix: missing-text recovery -->\n\n'
        qmd_text += '## Recovered Technical Details\n\n'
        qmd_text += '> The following details were present in the source document '
        qmd_text += 'but partially lost during automated conversion. '
        qmd_text += 'Recovered from pages: {}.\n\n'.format(pages_str)
        qmd_text += response + '\n'

        qmd_path.write_text(qmd_text)
        log.info('postfix: recovered missing text from %d page(s) via %s',
                 len(repair_pages), _REPAIR_MODEL)
        n_items = len([l for l in response.split(chr(10)) if l.strip().startswith("*")])
        return len(repair_pages), n_items, llm_cost
    except Exception as e:
        log.warning('Missing-text repair failed: %s', e)
        return 0, 0, 0.0
