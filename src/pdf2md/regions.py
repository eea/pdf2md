"""Figure-region geometry: the deterministic half of the figure pipeline (no LLM).

Refines a coarse detector box to the tight ink extent, renders a region to PNG
(encoding-agnostic), content-hash names it, assigns FIG_<n> in reading order, and
writes the detections.json sidecar.

bboxes are PDF points, top-left origin (PyMuPDF page space), as (x0, y0, x1, y1).
"""

import hashlib
import json
import logging
from collections import namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import fitz  # PyMuPDF
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False

log = logging.getLogger(__name__)

DEFAULT_FIGURE_DPI = 300
_REFINE_PAD_PT = 4.0       # padding added around the snapped ink bbox, in points
_REFINE_OVERLAP = 0.30     # min (intersection / rect-area) to treat a graphic as part of the figure
_MIN_GRAPHIC_PT = 3.0      # ignore rects thinner than this (page rules, underlines)

# A PDF stores an image as a byte stream plus a filter (/DCTDecode = JPEG bytes,
# verbatim). Re-rendering that region instead decodes, resamples to the target dpi
# and re-compresses — on a real 226-page guide that cost 71x the embedded size,
# because the sources are 96-dpi JPEGs and the output was 300-dpi lossless PNG.
# So: copy the stream when the region IS one stored image, else render at the
# resolution the content actually has.
_PASS_TOL_PT = 1.5          # geometry slop matching an image placement to a region
_PASS_ASPECT_TOL = 0.02     # max relative aspect mismatch (catches clipped/squashed)
_PASS_EXT = {"jpeg", "png"}          # codecs Quarto/Typst embed directly
_PASS_CS = {"DeviceRGB", "DeviceGray"}  # ICC/CMYK render instead, so the profile applies
_RASTER_ONLY_COVER = 0.9    # image coverage above which a region counts as all-raster
_JPEG_QUALITY = 92          # fallback encode for regions containing a photo

_Ink = namedtuple("_Ink", "coverage has_text has_vector")


@dataclass
class Region:
    """One detected illustration region on a page."""
    page: int                      # 0-based page index
    bbox: tuple                    # (x0, y0, x1, y1) in PDF points, top-left origin
    rtype: str = "figure"          # figure | table | chrome
    confidence: float = 1.0
    caption: str = ""
    fig_id: str = ""               # assigned in reading order, e.g. "FIG_1"
    md5: str = ""                  # set after render
    file: str = ""                 # media-relative filename, set after render
    origin: str = "detector"       # detector | oversized-table


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _graphic_rects(page) -> list:
    """All vector-drawing and embedded-image rects, minus thin ones (page rules,
    underlines) that would otherwise stretch a figure box."""
    rects = []
    for d in page.get_drawings():
        r = d.get("rect")
        if r:
            rects.append(fitz.Rect(r))
    for img in page.get_images(full=True):
        try:
            rects.extend(fitz.Rect(r) for r in page.get_image_rects(img[0]))
        except Exception:
            continue
    return [
        r for r in rects
        if not r.is_empty and r.width >= _MIN_GRAPHIC_PT and r.height >= _MIN_GRAPHIC_PT
    ]


def refine_bbox(page, bbox: tuple, pad: float = _REFINE_PAD_PT) -> tuple:
    """Snap a coarse detector box to the tight extent of the figure's graphics.

    Unions the full rect of every drawing/image overlapping the coarse box by at least
    _REFINE_OVERLAP of its own area. Full rect (not the clamped intersection) so it can
    both tighten loose boxes and grow to recover a sub-panel the detector clipped. The
    overlap test keeps page backgrounds and adjacent figures out. Falls back to the
    coarse box when nothing qualifies.
    """
    coarse = fitz.Rect(bbox) & page.rect          # clamp to page
    if coarse.is_empty:
        return tuple(page.rect)

    graphic = fitz.Rect()                          # empty; grows by union
    for r in _graphic_rects(page):
        inter = r & coarse
        if inter.is_empty:
            continue
        inter_area = inter.width * inter.height
        rect_area = max(r.width * r.height, 1e-6)
        if inter_area >= _REFINE_OVERLAP * rect_area:
            graphic |= r                           # full rect; may extend beyond coarse

    snapped = coarse if graphic.is_empty else graphic
    snapped = fitz.Rect(
        snapped.x0 - pad, snapped.y0 - pad, snapped.x1 + pad, snapped.y1 + pad
    ) & page.rect
    return tuple(snapped)


def _inner(bbox) -> "fitz.Rect":
    """The region minus refine_bbox's padding — the area the figure really occupies."""
    r = fitz.Rect(bbox)
    return r + (_REFINE_PAD_PT, _REFINE_PAD_PT, -_REFINE_PAD_PT, -_REFINE_PAD_PT)


def _region_ink(page, bbox: tuple, drawings=None) -> _Ink:
    """What a region is made of: raster coverage, live text, local vector art.

    One scan feeds every downstream decision. `drawings` lets the caller hoist
    page.get_drawings() out of a per-region loop.
    """
    region = fitz.Rect(bbox)
    inner = _inner(bbox)
    area = region.get_area()

    covered = 0.0
    for info in page.get_image_info():
        shown = fitz.Rect(info["bbox"]) & region
        if not shown.is_empty:
            covered += shown.get_area()

    has_text = bool(page.get_text("text", clip=inner).strip()) if not inner.is_empty else False

    has_vector = False
    for d in (page.get_drawings() if drawings is None else drawings):
        r = fitz.Rect(d["rect"])
        if r.width < _MIN_GRAPHIC_PT and r.height < _MIN_GRAPHIC_PT:
            continue
        # only marks local to the figure count; a page-spanning rect is background,
        # and the figure is painted over it anyway
        if region.contains(r) and not (r & inner).is_empty:
            has_vector = True
            break

    return _Ink(covered / area if area > 0 else 0.0, has_text, has_vector)


def raw_region_image(doc, page, bbox: tuple, ink: _Ink):
    """Original encoded bytes for a region that IS one stored image.

    Returns (bytes, ext) to copy out verbatim — same pixels, same codec, no
    resampling, no re-compression — or None when the stream alone doesn't
    reproduce what the page shows: labels or vector art on top, several images,
    a rotated/clipped placement, a soft mask, a colour profile, or a codec
    Quarto can't embed. Every bail falls through to rendering.
    """
    if ink.has_text or ink.has_vector:
        return None
    region = fitz.Rect(bbox)
    inner = _inner(bbox)
    if inner.is_empty:
        return None

    hits = [i for i in page.get_image_info(xrefs=True)
            if not (fitz.Rect(i["bbox"]) & region).is_empty]
    if len(hits) != 1:
        return None                      # composite, or no raster at all
    info = hits[0]
    xref = info.get("xref") or 0
    if not xref:
        return None                      # inline image, no stream to copy

    rect = fitz.Rect(info["bbox"])
    grow = (-_PASS_TOL_PT, -_PASS_TOL_PT, _PASS_TOL_PT, _PASS_TOL_PT)
    if not (rect + grow).contains(inner):
        return None                      # doesn't fill the region
    if not (region + grow).contains(rect):
        return None                      # spills out; the crop means something

    a, b, c, d, e, f = info["transform"]
    if abs(b) > 1e-3 or abs(c) > 1e-3 or a <= 0 or d <= 0:
        return None                      # rotated or mirrored placement
    # the transform places the whole image; a smaller reported bbox means the page
    # shows only part of it — a crop the stream can't express, at any aspect ratio
    implied = fitz.Rect(e, f, e + a, f + d)
    if abs(implied.width - rect.width) > 1.0 or abs(implied.height - rect.height) > 1.0:
        return None

    if info.get("cs-name") not in _PASS_CS:
        return None                      # ICC/CMYK/indexed: let rendering apply it

    try:
        base = doc.extract_image(xref)
    except Exception as exc:             # noqa: BLE001 — one bad xref is not fatal
        log.debug("xref %d not extractable (%s); rendering instead", xref, exc)
        return None
    if base["ext"] not in _PASS_EXT or base.get("smask") or info.get("has-mask"):
        return None                      # exotic codec, or alpha we would have to bake in

    if base["height"] and rect.height:
        native = base["width"] / base["height"]
        placed = rect.width / rect.height
        if abs(native - placed) / native > _PASS_ASPECT_TOL:
            return None                  # squashed placement

    return base["image"], base["ext"]


def _render_dpi(page, bbox: tuple, ink: _Ink, dpi: int) -> int:
    """Cap the render DPI at the region's own resolution.

    Only for regions that are nothing but raster. Upsampling a 96-dpi photo to 300
    invents no detail and multiplies the file — but a region carrying text or vector
    art keeps the full DPI, because those are resolution-free in the source and it is
    the rasterization, not the photo, that decides how sharp the labels come out.
    """
    if ink.coverage < _RASTER_ONLY_COVER or ink.has_text or ink.has_vector:
        return dpi
    region = fitz.Rect(bbox)
    native = 0.0
    for info in page.get_image_info():
        placed = fitz.Rect(info["bbox"])
        if placed.width <= 0 or (placed & region).is_empty:
            continue
        native = max(native, info["width"] / placed.width * 72.0)
    if native <= 0:
        return dpi
    return max(72, min(dpi, int(round(native))))


def _render(page, bbox: tuple, dpi: int, ink: _Ink) -> tuple:
    """Rasterize a region: JPEG when it holds a photo, PNG for pure line art.

    PNG on photographic content costs ~8x and buys nothing — it only preserves the
    source JPEG's own artifacts exactly. PNG matters where edges are hard.
    """
    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=dpi)
    if ink.coverage > 0:
        return pix.tobytes("jpeg", jpg_quality=_JPEG_QUALITY), "jpeg"
    return pix.tobytes("png"), "png"


def extract_region(doc, page, bbox: tuple, dpi: int = DEFAULT_FIGURE_DPI,
                   drawings=None) -> tuple:
    """(bytes, ext, how) for one region — verbatim stream if possible, else rendered."""
    ink = _region_ink(page, bbox, drawings)
    raw = raw_region_image(doc, page, bbox, ink)
    if raw:
        return raw[0], raw[1], "verbatim stream"
    used = _render_dpi(page, bbox, ink, dpi)
    data, ext = _render(page, bbox, used, ink)
    return data, ext, f"rendered {used} dpi {ext}"


def render_region(page, bbox: tuple, dpi: int = DEFAULT_FIGURE_DPI) -> bytes:
    """Render a page rectangle to PNG bytes at the given DPI (encoding-agnostic)."""
    pix = page.get_pixmap(clip=fitz.Rect(bbox), dpi=dpi)
    return pix.tobytes("png")


def _reading_order(regions: list) -> list:
    """Sort regions top-to-bottom, left-to-right by page then bbox."""
    return sorted(regions, key=lambda r: (r.page, round(r.bbox[1], 1), round(r.bbox[0], 1)))


def materialize_figures(
    pdf_path: Path,
    regions: list,
    media_dir: Path,
    dpi: int = DEFAULT_FIGURE_DPI,
    refine: bool = True,
) -> list:
    """Render every figure region to media_dir, assigning FIG_<n> in reading order.

    Mutates and returns the figure-only regions with bbox refined and fig_id / md5 /
    file populated. Non-figure regions are ignored.
    """
    if not _FITZ_AVAILABLE:
        raise RuntimeError("PyMuPDF (fitz) is required for region rendering.")
    media_dir.mkdir(parents=True, exist_ok=True)

    figures = _reading_order([r for r in regions if r.rtype == "figure"])
    doc = fitz.open(str(pdf_path))
    verbatim = total = 0
    cached_page, drawings = None, None
    try:
        for n, reg in enumerate(figures, start=1):
            page = doc[reg.page]
            if refine:
                reg.bbox = refine_bbox(page, reg.bbox)
            if cached_page != reg.page:     # figures arrive in page order
                cached_page, drawings = reg.page, page.get_drawings()
            data, ext, how = extract_region(doc, page, reg.bbox, dpi, drawings)
            reg.md5 = _md5(data)
            reg.fig_id = f"FIG_{n}"
            reg.file = f"img-{reg.md5}.{ext}"
            (media_dir / reg.file).write_bytes(data)
            verbatim += how == "verbatim stream"
            total += len(data)
            log.info(
                "%s ← page %d region %s (%d bytes, %s)",
                reg.fig_id, reg.page + 1,
                tuple(round(v, 1) for v in reg.bbox), len(data), how,
            )
        log.info(
            "Materialized %d figure(s): %d verbatim stream cop%s, %d rendered — %.1f MB",
            len(figures), verbatim, "y" if verbatim == 1 else "ies",
            len(figures) - verbatim, total / 1e6,
        )
    finally:
        doc.close()
    return figures


def write_sidecar(path: Path, figures: list, others: list = None,
                  cover: dict = None, table_slots: list = None) -> None:
    """Write detections.json.

    `figures` are the materialized FIG_<n> regions. `others` are non-figure detections
    (tables/chrome), recorded so the Phase-1 review can catch a figure misclassified as
    a table. `cover` is the optional ``{"is_cover": bool, "fields": {...}}`` block.
    `table_slots` are the TBL_<k> placeholder regions Pass 2 fills from focused crops.
    """
    payload = {
        "figures": [asdict(r) for r in figures],
        "other_detections": [asdict(r) for r in (others or [])],
    }
    if table_slots:
        payload["table_slots"] = table_slots
    if cover is not None:
        payload["cover"] = cover
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info(
        "Wrote detections sidecar %s (%d figures, %d other)",
        path.name, len(figures), len(others or []),
    )
