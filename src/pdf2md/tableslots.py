"""Table placeholders: keep tables OUT of the whole-doc conversion.

Whole-doc conversion partially mangles dense tables (measured 61-89% coverage);
a focused per-table conversion reaches ~94-100%. So Phase 1 paints a grey TBL_k
box over every substantial table region (same mechanism as FIG_n boxes), the
whole-doc pass emits only a marker per box, and Pass 2 fills each marker from a
focused 300-dpi crop. A slot whose marker the model skipped is rescued by
anchoring on the text that precedes the table in the source; a crop that fails
its fidelity gate falls back to the deterministic find_tables grid. Measured on
the ice ATBD: tables 89.0 -> 98.7% coverage, conversion cost -42% (tables are
the bulkiest output, and dropping them from the whole-doc response also removes
the long-generation pressure that mangles them in the first place).
"""

import logging
import re
from collections import Counter

from .regions import Region

log = logging.getLogger(__name__)

_MARKER_RE = re.compile(r'<!--pdf2md-tblslot-(?:TBL_)?(\d+)[^>]*-->')
_MARKER_LINE_RE = re.compile(r'<!--pdf2md-tblslot-[^>]*-->\n?')

_FIDELITY_MIN = 0.6      # crop must hit this share of the slot's distinctive values
# The printed grid is the authority when it can prove itself: read every cell by
# clipping text to its rectangle, then check what share of the region's words landed
# in a cell. Measured over 78 PDFs / 1365 grids the result is bimodal — 95.9% capture
# every word, 1.8% (partially ruled tables, e.g. confusion matrices) capture ~60%,
# nothing in between. So accept only a complete read and let the rest go to vision.
_GRID_MIN_COV = 0.999
# find_tables follows STROKED rules only. A table that divides its columns by colour
# instead leaves two columns inside one detected cell, and the read then places both
# columns' text in one cell — with every word still present, so coverage alone says
# 1.000. Catch it by asking whether a fill boundary splits a cell into two parts that
# BOTH hold text: an inset row background or a highlight never does, a missed column
# rule always does. Measured over the corpus this is the difference between 75.1% of
# slots accepted (unchecked, some misplaced) and 48.6% accepted and all sound.
_FILL_SPAN_MIN = 0.8       # a divider runs (nearly) the full extent of the cell
_FILL_INSET_PT = 3.0       # ignore an edge this close to the cell border
_MIN_DISTINCT = 8        # fewer distinctive values = sliver, leave it inline
_VISION_MIN_CHARS = 60   # a vision region with less clip text is noise

# Appended to the conversion system instruction when slots exist. The box-count
# contract matters: without it the model silently skipped 10/22 boxes (55%
# marker emission); with it, 19/22 (86%) — the anchored rescue covers the rest.
PROMPT_SECTION = """

TABLE PLACEHOLDERS (overrides the TABLES section for grey boxes):
In this PDF every data table has additionally been replaced by a grey box
labelled TBL_k (same style as the FIG boxes). At each such box output EXACTLY
this line, alone on its own line:

    <!--pdf2md-tblslot-TBL_k-->

Use the SAME number shown in the box. Do not describe the box and do not invent
a table. A visible caption line near the box (e.g. "Table 5: ...") is normal
text — transcribe it just before the marker line. The TABLES section still
applies to any table that appears as a real table (not a grey box).
BOX COUNT CONTRACT: this document contains EXACTLY {nfig} FIG boxes and EXACTLY
{ntbl} TBL boxes. Your output MUST reference every FIG box once and contain
every TBL marker once — a skipped box is a critical failure. Before finishing,
verify both counts."""


def _center_in(bbox, boxes) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def build_table_slots(working_pdf, figure_regions, excluded_regions):
    """Compute the table regions to placeholder.

    Returns (slots, box_regions): `slots` are JSON-ready dicts persisted in the
    detections sidecar ({slot, page, bbox, rows, ctx, dist, est_chars}); the
    Regions carry the TBL_k labels for placeholder injection.

    Regions = substantial find_tables grids + vision-detected excluded tables
    (dedup'd). A region that overlaps a figure box is left inline: boxing it
    would bury the figure's own placeholder (in-cell figures), and a focused
    crop of such a table would lose the image anyway.
    """
    import fitz
    from .postfix import _scan_source_tables
    from .verify.textutil import normalize, tokens

    grids, clean_lines = _scan_source_tables(working_pdf)
    fig_boxes = {}
    for r in figure_regions:
        fig_boxes.setdefault(r.page, []).append(r.bbox)

    def _figure_clash(pno, bbox):
        boxes = fig_boxes.get(pno, [])
        return any(_center_in(fb, [bbox]) for fb in boxes) or _center_in(bbox, boxes)

    cand = []
    for pno, bbox, rows, ctx in grids:
        ncols = max((len(r) for r in rows), default=0)
        if ncols >= 45 or len(rows) * ncols >= 2500 or len(rows) < 2 or ncols < 2:
            continue                        # oversized (cropped as figure) or sliver
        if _figure_clash(pno, bbox):
            continue
        toks = {t for r in rows for c in r if c for t in normalize(c).split()}
        cand.append([pno, bbox, rows, toks, ctx])

    doc = fitz.open(str(working_pdf))
    try:
        grid_boxes = {}
        for pno, bbox, _r, _t, _c in cand:
            grid_boxes.setdefault(pno, []).append(bbox)
        for r in excluded_regions:
            pno, bbox = r.page, tuple(r.bbox)
            if _center_in(bbox, grid_boxes.get(pno, [])) or _figure_clash(pno, bbox):
                continue
            clip = doc[pno].get_text(clip=fitz.Rect(*bbox))
            if len(clip) < _VISION_MIN_CHARS:
                continue
            above = [t for y, t in clean_lines.get(pno, []) if y < bbox[1]]
            cand.append([pno, bbox, None, set(normalize(clip).split()),
                         tokens(' '.join(above[-10:]))])
    finally:
        doc.close()

    df = Counter(t for c in cand for t in c[3])
    cand = [c for c in cand
            if len({t for t in c[3] if df[t] <= 2}) >= _MIN_DISTINCT]
    cand.sort(key=lambda c: (c[0], c[1][1], c[1][0]))

    slots, box_regions = [], []
    for k, (pno, bbox, rows, toks, ctx) in enumerate(cand, 1):
        dist = sorted({t for t in toks if df[t] <= 2})
        est = sum(len(t) + 1 for t in toks) * 3
        slots.append({"slot": k, "page": pno, "bbox": list(bbox), "rows": rows,
                      "ctx": ctx, "dist": dist, "est_chars": est})
        box_regions.append(Region(page=pno, bbox=tuple(bbox), rtype="figure",
                                  fig_id=f"TBL_{k}"))
    log.info("table slots: %d region(s) placeholdered", len(slots))
    return slots, box_regions


def fill_table_slots(text, slots, working_pdf, api_key):
    """Fill every TBL marker from a focused crop; rescue markerless slots by anchor.

    Returns (text, report) with report = {filled, rescued, fallbacks, lost, cost}.
    A crop below the fidelity gate falls back to the deterministic find_tables
    grid, so a marker never ships unfilled; a slot with neither marker nor a
    unique anchor is counted `lost` (Pass 3/4 repair remains as the net below).
    """
    import fitz
    from .postfix import (_anchor_by_context, _crop_table_md, _grid_to_markdown,
                          _qmd_word_offsets, _safe_boundary, _tbl_md_tokens)
    from .verify.textutil import tokens as _tok

    fill_cache = {}

    def _page_fills(page):
        if page.number not in fill_cache:
            fill_cache[page.number] = [
                fitz.Rect(d['rect']) for d in page.get_drawings()
                if 'f' in d['type'] and d.get('fill')
                and fitz.Rect(d['rect']).width >= 15 and fitz.Rect(d['rect']).height >= 8]
        return fill_cache[page.number]

    def _wrongly_merged(page, cell, fills):
        """True if a fill edge splits this cell into two parts that both hold text."""
        for fill in fills:
            if (fill & cell).is_empty:
                continue
            if fill.height >= _FILL_SPAN_MIN * cell.height:
                for x in (fill.x0, fill.x1):
                    if cell.x0 + _FILL_INSET_PT < x < cell.x1 - _FILL_INSET_PT and \
                            page.get_text(clip=fitz.Rect(cell.x0, cell.y0, x, cell.y1)).strip() and \
                            page.get_text(clip=fitz.Rect(x, cell.y0, cell.x1, cell.y1)).strip():
                        return True
            if fill.width >= _FILL_SPAN_MIN * cell.width:
                for y in (fill.y0, fill.y1):
                    if cell.y0 + _FILL_INSET_PT < y < cell.y1 - _FILL_INSET_PT and \
                            page.get_text(clip=fitz.Rect(cell.x0, cell.y0, cell.x1, y)).strip() and \
                            page.get_text(clip=fitz.Rect(cell.x0, y, cell.x1, cell.y1)).strip():
                        return True
        return False

    def _grid_read(pno, bbox):
        """(markdown, coverage) for the printed grid over this region.

        Returns (None, cov) when the grid cannot be trusted: either it misses text
        (coverage) or a cell holds two columns that a colour boundary divides.
        """
        page = doc[pno]
        rect = fitz.Rect(*bbox)
        region = set(_tok(page.get_text(clip=rect)))
        if not region:
            return None, 0.0
        try:
            # lines_strict follows the printed rules; the default strategy splits a
            # wrapped line into its own row (57 rows for a 27-row table)
            found = page.find_tables(strategy='lines_strict').tables
        except Exception:                   # noqa: BLE001 — fall through to vision
            return None, 0.0
        for t in found:
            box = fitz.Rect(t.bbox)
            overlap = box & rect
            if overlap.is_empty or overlap.get_area() < 0.5 * min(box.get_area(),
                                                                 rect.get_area()):
                continue
            words, rects = set(), []
            for row in t.rows:
                for cell in row.cells:
                    if cell:
                        cr = fitz.Rect(cell)
                        rects.append(cr)
                        words |= set(_tok(page.get_text(clip=cr)))
            cov = len(words & region) / len(region)
            fills = _page_fills(page)
            if any(_wrongly_merged(page, cr, fills) for cr in rects):
                log.debug('grid declined on page %d: a colour boundary divides a cell', pno + 1)
                return None, cov
            rows = [r for r in t.extract() if any(c for c in r)]
            return (_grid_to_markdown(rows) if rows else None), cov
        return None, 0.0

    report = {"filled": 0, "rescued": 0, "fallbacks": 0, "lost": 0, "cost": 0.0,
              "from_grid": 0}
    if not slots:
        return _MARKER_LINE_RE.sub('', text), report
    by_num = {s["slot"]: s for s in slots}

    doc = fitz.open(str(working_pdf))
    try:
        def _convert(s):
            # the grid first: free, deterministic, and it cannot mangle a long table
            # the way a token-limited generation does — but only when it proves it
            # captured every word of the region
            grid_md, cov = _grid_read(s["page"], tuple(s["bbox"]))
            if grid_md and cov >= _GRID_MIN_COV:
                report["from_grid"] += 1
                return grid_md
            dist = set(s["dist"]) or {"_"}
            md, c = _crop_table_md(api_key, doc, s["page"], tuple(s["bbox"]),
                                   s["est_chars"])
            report["cost"] += c
            if not (md and len(dist & _tbl_md_tokens(md)) / len(dist) >= _FIDELITY_MIN):
                report["fallbacks"] += 1
                md = _grid_to_markdown(s["rows"]) if s["rows"] else md
            return md

        filled = set()
        out, pos = [], 0
        for m in _MARKER_RE.finditer(text):
            k = int(m.group(1))
            s = by_num.get(k)
            out.append(text[pos:m.start()])
            pos = m.end()
            if s is None or k in filled:    # invented or duplicated marker → drop it
                continue
            md = _convert(s)
            if md:
                out.append(md)
                filled.add(k)
                report["filled"] += 1
        out.append(text[pos:])
        text = _MARKER_LINE_RE.sub('', ''.join(out))

        # rescue: slots the model never markered — insert at the unique anchor of
        # the text that precedes the table in the source; decline (never append)
        words, _starts, ends = _qmd_word_offsets(text)
        inserts = []
        for s in slots:
            if s["slot"] in filled:
                continue
            at = _anchor_by_context(words, ends, s["ctx"]) if s["ctx"] else None
            safe = _safe_boundary(text, at) if at is not None else None
            md = _convert(s) if safe is not None else None
            if md and safe is not None:
                inserts.append((safe, md))
                report["rescued"] += 1
            else:
                report["lost"] += 1
        for safe, md in sorted(inserts, reverse=True):
            text = text[:safe] + md + '\n\n' + text[safe:]
    finally:
        doc.close()

    if report["lost"]:
        log.warning("table slots: %d slot(s) had no marker and no anchor — left to "
                    "the repair passes", report["lost"])
    log.info("table slots: %d filled, %d rescued, %d from the printed grid, "
             "%d fallback grid(s), $%.4f", report["filled"], report["rescued"],
             report["from_grid"], report["fallbacks"], report["cost"])
    return text, report
