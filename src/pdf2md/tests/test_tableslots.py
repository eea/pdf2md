"""Table-slot fill: markers filled from crops, markerless slots rescued by anchor."""
import sys
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
