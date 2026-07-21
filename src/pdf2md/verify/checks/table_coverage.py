"""Check: did each source table's data survive into the .qmd?

Compares source table grids (PyMuPDF find_tables) against .qmd table grids at the
token-bag level, per table, reporting per-table coverage. Catches dropped tables,
rows, or cells.
"""

import math
import re
from collections import Counter

from .. import CheckResult, Finding, register
from ..textutil import normalize, top_level_html_tables

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

_GLOBAL_MIN = 0.85   # WARN if token-weighted coverage across all tables drops below this
_CELL_HIT = 0.7      # a table is "thin" below this fraction of its words matched
_MIN_TOKENS = 12     # ignore find_tables slivers as diagnostics (a 3-word piece isn't a table)
_UNION_MAX = 4       # a source table may span at most this many .qmd tables
_UNION_MIN_GAIN = 0.05   # ...and each must add >=5% of its tokens to join the union


# Oversized tables (>= these) are cropped as figures, not transcribed, so they must NOT
# count here — else a correctly-cropped 58-col matrix reads as 0% coverage and tanks the
# score.
_OVERSIZE_COLS = 45
_OVERSIZE_CELLS = 2500


def _is_oversized(rows: list) -> bool:
    """A table cropped as a figure by Phase 1 (too wide/large to transcribe)."""
    ncols = max((len(r) for r in rows), default=0)
    return ncols >= _OVERSIZE_COLS or len(rows) * ncols >= _OVERSIZE_CELLS


def _source_grids(pdf_path) -> list:
    """Extract table grids, excluding running-header tables."""
    from collections import Counter

    grids_raw = []
    doc = fitz.open(str(pdf_path))
    total_pages = doc.page_count
    try:
        for pno in range(total_pages):
            try:
                for tab in doc[pno].find_tables().tables:
                    rows = tab.extract()
                    if _is_oversized(rows):
                        continue
                    cells = [normalize(c) for row in rows for c in row if c and normalize(c)]
                    if cells:
                        fp = " ".join(cells[:2])
                        grids_raw.append((fp, cells))
            except Exception:
                continue
    finally:
        doc.close()

    fp_counts = Counter(fp for fp, _ in grids_raw)
    threshold = max(3, total_pages * 0.5)
    running = {fp for fp, c in fp_counts.items() if c >= threshold}

    return [cells for fp, cells in grids_raw if fp not in running]
def _html_table_grids(qmd_text: str) -> list:
    """One token-bag per top-level HTML table, nested-table text included. Coverage is
    token-overlap, so per-cell granularity isn't needed, and collapsing nested tables
    avoids the truncation a per-<td> regex suffers on nesting."""
    grids = []
    for tbl in top_level_html_tables(qmd_text):
        text = re.sub(r"<[^>]+>", " ", tbl)  # strip all tags, nested included
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        norm = normalize(text)
        if norm:
            grids.append([norm])
    return grids


def _qmd_grids(qmd_text: str) -> list:
    """Parse .qmd tables (pipe AND raw-HTML) into flat normalized cell lists."""
    grids, cur = [], []
    for line in qmd_text.splitlines():
        s = line.strip()
        is_row = s.startswith("|") and s.endswith("|")
        is_divider = is_row and set(s) <= set("|-: ")
        if is_row and not is_divider:
            cells = [normalize(c) for c in s.strip("|").split("|") if normalize(c)]
            cur.extend(cells)
        elif cur and not is_row:
            grids.append(cur)
            cur = []
    if cur:
        grids.append(cur)
    grids.extend(_html_table_grids(qmd_text))
    return grids


def _tokens_of(cells) -> set:
    """All word tokens across a table's cells."""
    toks = set()
    for c in cells:
        toks.update(c.split())
    return toks


def _best_match(src_cells, qmd_grids):
    """Pick the .qmd table with the best TOKEN overlap with the source table.

    Token overlap rather than exact cell strings: complex tables have big
    multi-line cells that PyMuPDF concatenates but the HTML re-segments into
    <br>/<li>/nested tables, so the cell strings differ while the words match.
    Returns (best_grid, token_coverage), coverage being the fraction of the
    SOURCE table's tokens that appear in the matched .qmd table.
    """
    src_toks = _tokens_of(src_cells)
    if not src_toks:
        return None, 1.0
    best, best_score = None, -1.0
    for g in qmd_grids:
        score = len(src_toks & _tokens_of(g)) / len(src_toks)
        if score > best_score:
            best, best_score = g, score
    return best, best_score


def _token_weights(qmd_tok_sets: list) -> dict:
    """IDF-style token weights. A token occurring in many .qmd tables proves nothing —
    bare numbers ('475') and units ('small', '%') recur everywhere, so a table that never
    converted can still collect them from unrelated tables and score as covered
    (measured: 13 pages of missing tables still scored 97.5%). Rare tokens are the real
    evidence a specific table survived.
    """
    df = Counter()
    for s in qmd_tok_sets:
        for t in s:
            df[t] += 1
    n = max(1, len(qmd_tok_sets))
    # A token absent from every .qmd table is maximally rare, so it must carry the
    # HIGHEST weight — it is the strongest evidence a table did not convert. Returning
    # it as the default matters: weighting it 1.0 while present tokens scored 2-4
    # inflated coverage instead of exposing gaps.
    return {t: math.log(1 + n / (1 + c)) for t, c in df.items()}, math.log(1 + n)


def _w(tok, weights, default):
    return default if weights is None else weights.get(tok, default)


def _weigh(toks, weights, default):
    return sum(_w(t, weights, default) for t in toks)


def _union_match(src_toks: set, qmd_tok_sets: list, weights: dict = None,
                 wdefault: float = 1.0) -> tuple:
    """Score a source table against the best-matching SET of .qmd tables.

    The converter legitimately re-segments tables — splitting one source table across
    several .qmd tables and merging others — so scoring against a single best match
    counts present content as missing (measured: 91% single-match vs 98% actual on a
    131-page manual). Greedily add whichever .qmd table contributes the most as-yet
    unmatched tokens, while it contributes meaningfully. Bounded by _UNION_MAX and
    _UNION_MIN_GAIN so this stays "spans a few tables", not "appears anywhere".
    Returns (matched_tokens, coverage, n_tables_used).
    """
    if not src_toks:
        return set(), 1.0, 0
    total_w = _weigh(src_toks, weights, wdefault) or 1.0
    remaining, matched, used = set(src_toks), set(), 0
    for _ in range(_UNION_MAX):
        best_toks, best_gain = None, 0.0
        for q in qmd_tok_sets:
            gain = _weigh(remaining & q, weights, wdefault)
            if gain > best_gain:
                best_toks, best_gain = q, gain
        if best_toks is None or best_gain / total_w < _UNION_MIN_GAIN:
            break
        hit = remaining & best_toks
        matched |= hit
        remaining -= hit
        used += 1
        if not remaining:
            break
    return matched, _weigh(matched, weights, wdefault) / total_w, used


@register
class TableCoverageCheck:
    name = "table_coverage"

    def applicable(self, ctx) -> bool:
        return _FITZ_AVAILABLE and ctx.original_pdf and ctx.original_pdf.exists()

    def run(self, ctx) -> CheckResult:
        src_grids = _source_grids(ctx.original_pdf)
        if not src_grids:
            return CheckResult(self.name, "ok", "no source tables detected")
        qmd_grids = _qmd_grids(ctx.qmd_text)

        # Two numbers, because the converter re-segments tables:
        #   content — tokens found across the few .qmd tables the source table spans.
        #             This is cell FIDELITY and drives the status.
        #   aligned — tokens found in a SINGLE best-matching .qmd table. A structure
        #             signal: much lower than content means heavy split/merge.
        # Token-WEIGHTED, so a find_tables sliver can't trip a warn while a big table
        # losing half its cells does.
        qmd_tok_sets = [_tokens_of(g) for g in qmd_grids]
        weights, wdefault = _token_weights(qmd_tok_sets)
        per_table = []           # (i, content, src_cells, matched_tokens, n_src_tokens)
        total_toks = content_toks = aligned_toks = 0
        for i, src in enumerate(src_grids, 1):
            stoks = _tokens_of(src)
            ntoks = len(stoks)
            matched, content, _used = _union_match(stoks, qmd_tok_sets, weights, wdefault)
            wt = _weigh(stoks, weights, wdefault) or 1.0
            aligned = max((_weigh(stoks & q, weights, wdefault) / wt for q in qmd_tok_sets),
                          default=0.0) if ntoks else 1.0
            total_toks += ntoks
            content_toks += content * ntoks
            aligned_toks += aligned * ntoks
            per_table.append((i, content, src, matched, ntoks))

        weighted = (content_toks / total_toks) if total_toks else 1.0
        aligned_w = (aligned_toks / total_toks) if total_toks else 1.0
        simple_avg = sum(t[1] for t in per_table) / len(per_table)
        status = "warn" if weighted < _GLOBAL_MIN else "ok"

        # diagnostics for substantial-but-thin tables only (slivers filtered). Severity
        # tracks the overall verdict: FYI when coverage is fine, the warn detail when not.
        sev = "warn" if status == "warn" else "info"
        findings = []
        for i, content, src, matched, ntoks in per_table:
            if ntoks < _MIN_TOKENS or content >= _CELL_HIT:
                continue
            missing = []
            for c in dict.fromkeys(src):
                ct = c.split()
                if ct and sum(t in matched for t in ct) / len(ct) < 0.5:
                    missing.append(c[:80])
            findings.append(Finding(
                f"table {i}: {round(100 * content)}% of words matched"
                + (f"; e.g. missing {missing[:3]}" if missing else ""),
                sev, f"table {i}"))

        wpct = round(100 * weighted, 1)
        apct = round(100 * aligned_w, 1)
        avg = round(100 * simple_avg, 1)
        return CheckResult(
            self.name, status,
            f"{len(src_grids)} source table(s); word coverage {wpct}% "
            f"(simple avg {avg}%; {apct}% single-table aligned)"
            + (f"; {len(findings)} substantial table(s) below {int(_CELL_HIT*100)}%"
               if findings else ""),
            metric=wpct, findings=findings,
            detail={"content": wpct, "aligned": apct},
        )
