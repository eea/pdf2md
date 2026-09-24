"""Table-slot fill: markers filled from crops, markerless slots rescued by anchor."""
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pdf2md.postfix as postfix  # noqa: E402
from pdf2md.tableslots import fill_table_slots  # noqa: E402

GOOD_MD = "| Alpha | Beta |\n|---|---|\n| valuone | valutwo |"

ANCHOR_PROSE = ("the quick brown fox jumps over the lazy dog tonight and "
                "settles down for the night.")

TEXT = (
    "# Doc\n\nIntro paragraph.\n\n"
    "<!--pdf2md-tblslot-TBL_1-->\n\n"
    + ANCHOR_PROSE + "\n\nTail paragraph.\n"
)


def _slot(k, ctx=None, rows=None):
    return {"slot": k, "page": 0, "bbox": [0, 0, 100, 100], "rows": rows,
            "ctx": ctx or [], "dist": ["valuone", "valutwo"], "est_chars": 500}


def _fake_pdf(tmp_path):
    import fitz
    p = tmp_path / "w.pdf"
    d = fitz.open(); d.new_page(); d.save(str(p)); d.close()
    return p


def test_fill_marker_rescue_and_lost(tmp_path, monkeypatch):
    monkeypatch.setattr(postfix, "_crop_table_md", lambda *a, **k: (GOOD_MD, 0.001))
    from pdf2md.verify.textutil import tokens
    slots = [
        _slot(1),                                # has a marker → filled in place
        _slot(2, ctx=tokens(ANCHOR_PROSE)),      # no marker, unique anchor → rescued
        _slot(3),                                # no marker, no ctx → lost
    ]
    out, rep = fill_table_slots(TEXT, slots, _fake_pdf(tmp_path), "k")
    assert rep["filled"] == 1 and rep["rescued"] == 1 and rep["lost"] == 1
    assert "pdf2md-tblslot" not in out           # no marker survives
    assert out.count(GOOD_MD) == 2
    # rescued table landed after the anchor prose, not appended at the end
    assert out.index(ANCHOR_PROSE) < out.rindex(GOOD_MD) < out.index("Tail paragraph.")


def test_fill_falls_back_to_grid_on_bad_crop(tmp_path, monkeypatch):
    monkeypatch.setattr(postfix, "_crop_table_md",
                        lambda *a, **k: ("| junk | junk |", 0.001))
    rows = [["Alpha", "Beta"], ["valuone", "valutwo"]]
    out, rep = fill_table_slots(TEXT, [_slot(1, rows=rows)], _fake_pdf(tmp_path), "k")
    assert rep["fallbacks"] == 1 and rep["filled"] == 1
    assert "valuone" in out and "junk" not in out   # deterministic grid won


def test_invented_marker_is_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(postfix, "_crop_table_md", lambda *a, **k: (GOOD_MD, 0.0))
    text = "A.\n\n<!--pdf2md-tblslot-TBL_9-->\n\nB.\n"
    out, rep = fill_table_slots(text, [_slot(1)], _fake_pdf(tmp_path), "k")
    assert "pdf2md-tblslot" not in out and rep["filled"] == 0


def _table_pdf(tmp_path, ruled_divider: bool):
    """A 2x2 table. The column split is a stroked rule or only a colour change."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(); page = doc.new_page()
    outer = fitz.Rect(50, 50, 250, 130)
    page.draw_rect(outer, color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(50, 90), fitz.Point(250, 90), color=(0, 0, 0), width=1)
    if ruled_divider:
        page.draw_line(fitz.Point(150, 50), fitz.Point(150, 130), color=(0, 0, 0), width=1)
    else:                                   # the boundary exists only as a fill
        page.draw_rect(fitz.Rect(50, 50, 150, 130), color=None, fill=(1, 0.85, 0.85))
    page.insert_text((60, 75), "Alpha")
    page.insert_text((160, 75), "Beta")
    page.insert_text((60, 115), "Gamma")
    page.insert_text((160, 115), "Delta")
    out = tmp_path / ("ruled.pdf" if ruled_divider else "coloured.pdf")
    doc.save(str(out)); doc.close()
    return out, tuple(outer)


def _read(pdf, bbox):
    """Drive fill_table_slots' grid reader without touching the network."""
    from pdf2md.tableslots import fill_table_slots
    captured = {}
    slot = {"slot": 1, "page": 0, "bbox": list(bbox), "rows": None, "ctx": [],
            "dist": ["alpha"], "est_chars": 100}

    def _no_vision(*a, **k):
        captured["vision"] = True
        return None, 0.0

    import pdf2md.postfix as pf
    real = pf._crop_table_md
    pf._crop_table_md = _no_vision
    try:
        text, report = fill_table_slots("before\n\n<!--pdf2md-tblslot-TBL_1-->\n\nafter\n",
                                        [slot], pdf, api_key=None)
    finally:
        pf._crop_table_md = real
    return text, report, captured


def test_grid_used_when_the_divider_is_a_printed_rule(tmp_path):
    pdf, bbox = _table_pdf(tmp_path, ruled_divider=True)
    text, report, captured = _read(pdf, bbox)
    assert report["from_grid"] == 1            # deterministic path, no vision call
    assert "vision" not in captured
    assert "<td>Alpha</td>" in text and "<td>Beta</td>" in text


def test_grid_reads_a_colour_divided_table(tmp_path):
    """The failure that made PA21's nomenclature table a 7-column misread: the
    Level 1/Level 2 boundary is a fill edge, and lines_strict discards fill-only
    paths, so both columns landed in one cell. Feeding those edges back via
    add_lines separates them, so the table is read rather than sent to vision."""
    pdf, bbox = _table_pdf(tmp_path, ruled_divider=False)
    text, report, captured = _read(pdf, bbox)
    assert report["from_grid"] == 1
    assert "vision" not in captured
    # HTML so merged cells survive and the colour pass has a <td> to style
    assert "<td>Alpha</td>" in text and "<td>Beta</td>" in text


def test_grid_declined_when_text_escapes_the_cells(tmp_path):
    """Coverage gate: text inside the region but outside every cell would be lost
    by a rebuild, so the grid must decline regardless of how clean it looks."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open(); page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 250, 110), color=(0, 0, 0), width=1)
    page.draw_line(fitz.Point(150, 50), fitz.Point(150, 110), color=(0, 0, 0), width=1)
    page.insert_text((60, 75), "Alpha")
    page.insert_text((160, 75), "Beta")
    page.insert_text((60, 135), "stray text below the grid")   # inside region, no cell
    out = tmp_path / "escaped.pdf"
    doc.save(str(out)); doc.close()
    _text, report, captured = _read(out, (50, 50, 250, 150))
    assert report["from_grid"] == 0
    assert captured.get("vision") is True
