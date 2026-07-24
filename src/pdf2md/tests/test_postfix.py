"""Postfix in-place recovery: where recovered prose gets put back.

The converter intermittently drops a passage (LLM non-determinism). Postfix detects
it and recovers the prose from the source PDF — these tests cover putting it back
*in flow* rather than in an end-of-document appendix.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md.postfix import (  # noqa: E402
    _drop_already_present, _insertion_point, _safe_boundary,
)

QMD = (
    "# Doc\n\n"
    "The accuracy of delineation was evaluated applying sampling protocols here.\n\n"
    "Next block of prose follows on from that one.\n\n"
    "```{=html}\n<table><tr><td>cell</td></tr></table>\n```\n\n"
    "Tail paragraph.\n"
)


# ── boundary safety ─────────────────────────────────────────────────────────────

def test_safe_boundary_lands_between_blocks():
    at = _safe_boundary(QMD, 0)
    assert QMD[:at].endswith("\n\n")


def test_safe_boundary_never_lands_inside_a_fence():
    inside = QMD.find("<table>")
    at = _safe_boundary(QMD, inside)
    # either no boundary remains, or the one chosen has balanced fences before it
    assert at is None or QMD.count("```", 0, at) % 2 == 0


# ── anchoring ───────────────────────────────────────────────────────────────────

def test_insertion_point_follows_the_surviving_anchor():
    at = _insertion_point(QMD, [
        "The accuracy of delineation was evaluated applying sampling protocols here."])
    assert at is not None
    assert QMD[:at].endswith("\n\n")
    # inserts after the anchor, not before it
    assert at > QMD.index("The accuracy of delineation")


def test_insertion_point_none_when_nothing_survived():
    # no text from the page made it into the .qmd → caller falls back to appending
    assert _insertion_point(QMD, [
        "A completely absent sentence that is certainly long enough to probe with"]) is None


def test_insertion_point_ignores_a_lone_distant_match():
    # regression: taking the LAST match dragged inserts to the end of the document
    # when one line coincidentally recurred late (observed: source p83 landed at 99%).
    body = ("Alpha paragraph about sampling strata and their thresholds here.\n\n"
            "Beta paragraph about delineation accuracy of the polygons here.\n\n"
            "Gamma paragraph about positional offset of the input data here.\n\n")
    filler = "".join(f"Filler paragraph number {i} with enough words to pad.\n\n"
                     for i in range(400))
    late = "Alpha paragraph about sampling strata and their thresholds here.\n\n"
    qmd = body + filler + late
    at = _insertion_point(qmd, [
        "Alpha paragraph about sampling strata and their thresholds here.",
        "Beta paragraph about delineation accuracy of the polygons here.",
        "Gamma paragraph about positional offset of the input data here.",
    ])
    # must anchor on the cluster at the top, not the lone recurrence at the bottom
    assert at is not None and at < len(body) + 200


def test_insertion_point_ignores_short_anchors():
    # short lines are too ambiguous to anchor on
    assert _insertion_point(QMD, ["Doc", "Tail"]) is None


# ── de-duplication ──────────────────────────────────────────────────────────────

def test_drop_already_present_removes_existing_prose():
    dup = "The accuracy of delineation was evaluated applying sampling protocols here."
    assert _drop_already_present(dup, QMD) == ""


def test_drop_already_present_keeps_novel_prose():
    novel = "This paragraph introduces entirely new material about sampling strata."
    assert "entirely new material" in _drop_already_present(novel, QMD)


def test_drop_already_present_skips_tiny_fragments():
    assert _drop_already_present("too short", QMD) == ""


def test_postfix_headings_restores_missing_chapter(tmp_path):
    import fitz
    from pdf2md.postfix import _postfix_headings
    body_line = ("This chapter describes the water cover duration product in detail "
                 "for the pan-European area.")
    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 80), "4 Water Cover Duration (WCD)")
    p.insert_text((72, 120), body_line)
    doc.set_toc([[1, "4 Water Cover Duration (WCD)", 1]])
    src = tmp_path / "d.source.pdf"
    doc.save(str(src)); doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text(f"## Overview\n\nSome intro.\n\n{body_line}\n", encoding="utf-8")
    n = _postfix_headings(qmd, tmp_path)
    assert n == 1
    out = qmd.read_text()
    # inserted as H1, section number stripped, right before its section body
    assert "# Water Cover Duration (WCD)\n" in out
    assert out.index("Water Cover Duration") < out.index(body_line)


def test_postfix_headings_skips_unanchorable(tmp_path):
    import fitz
    from pdf2md.postfix import _postfix_headings
    doc = fitz.open(); doc.new_page()
    doc.set_toc([[1, "Ghost chapter", 1]])
    src = tmp_path / "d.source.pdf"; doc.save(str(src)); doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text("## Other\n\nUnrelated text entirely.\n", encoding="utf-8")
    assert _postfix_headings(qmd, tmp_path) == 0


def test_strip_chrome_lines_drops_running_header_and_page_number():
    from pdf2md.postfix import _strip_chrome_lines
    from pdf2md.verify.textutil import normalize
    freq = {normalize("CLC+ Backbone Product Specification and User Manual"): 130}
    text = ("CLC+ Backbone Product Specification and User Manual\n"
            "96\n"
            "Genuine prose that appears only on this page of the document.")
    out = _strip_chrome_lines(text, freq)
    assert out == "Genuine prose that appears only on this page of the document."


# ── window recovery: page-marker parsing ────────────────────────────────────────

from pdf2md.postfix import _WINDOW_RE  # noqa: E402


def _parse_window(response):
    chunks = _WINDOW_RE.split(response)
    out = {}
    for i in range(1, len(chunks) - 1, 2):
        out[int(chunks[i]) - 1] = chunks[i + 1].strip()
    return out


def test_window_markers_split_pages():
    out = _parse_window(
        "<<<PAGE 83>>>\nProse of page eighty three.\n\n"
        "<<<PAGE 84>>>\nProse of page eighty four.\n\n"
        "<<<PAGE 85>>>\nProse of page eighty five.")
    assert sorted(out) == [82, 83, 84]
    assert out[83] == "Prose of page eighty four."


def test_window_tolerates_preamble_and_spacing():
    out = _parse_window("Here you go:\n<<<PAGE  7>>>\n  Seven prose.  \n<<<PAGE 8>>>\nEight prose.")
    assert out[6] == "Seven prose." and out[7] == "Eight prose."


def test_insertion_point_skips_ambiguous_anchors():
    # regression: a heading also present in the table of contents matched the TOC copy
    # and pinned inserts to the top of the document. Ambiguous probes must be skipped.
    toc = "## Contents\n\nMethodological approach and sampling design overview\n\n"
    body = ("Filler body paragraph with sufficient length to be a real block.\n\n"
            "Methodological approach and sampling design overview\n\n"
            "Trailing body paragraph that follows the duplicated heading text.\n\n")
    qmd = toc + body
    # the duplicated line alone is ambiguous -> no anchor at all
    assert _insertion_point(qmd, [
        "Methodological approach and sampling design overview"]) is None
    # a unique line still anchors fine
    at = _insertion_point(qmd, [
        "Trailing body paragraph that follows the duplicated heading text."])
    assert at is not None and at > len(toc)


# ── deterministic table recovery ────────────────────────────────────────────────

from pdf2md.postfix import _grid_to_markdown  # noqa: E402


def test_grid_to_markdown_renders_header_and_rows():
    md = _grid_to_markdown([["a", "b"], ["1", "2"]])
    assert md.splitlines() == ["| a | b |", "|---|---|", "| 1 | 2 |"]


def test_grid_to_markdown_pads_ragged_rows():
    # find_tables emits ragged rows for merged cells; every row must keep the grid width
    md = _grid_to_markdown([["a", "b", "c"], ["1"]])
    assert all(l.count("|") == 4 for l in md.splitlines())


def test_grid_to_markdown_escapes_pipes_and_newlines():
    md = _grid_to_markdown([["x|y", "p\nq"], ["1", "2"]])
    assert r"x\|y" in md          # a literal pipe must not break the grid
    assert "p q" in md and "\n" not in md.split("|---")[0].strip("\n")


def test_grid_to_markdown_copies_values_verbatim():
    md = _grid_to_markdown([["No. of samples", "1438"], ["size classes", "475 small"]])
    assert "1438" in md and "475 small" in md


# ── hyperlink recovery ──────────────────────────────────────────────────────────

from pdf2md.postfix import _safe_to_inline  # noqa: E402


def test_safe_to_inline_in_plain_prose():
    t = "Some prose mentioning the Convention here.\n"
    assert _safe_to_inline(t, t.index("Convention"))


def test_not_safe_inside_a_code_fence():
    t = "intro\n\n```\ncode Convention here\n```\n"
    assert not _safe_to_inline(t, t.index("Convention"))


def test_not_safe_inside_an_html_table():
    t = "intro\n\n<table><tr><td>Convention</td></tr></table>\n"
    assert not _safe_to_inline(t, t.index("Convention"))


def test_safe_again_after_table_closes():
    t = "<table><tr><td>x</td></tr></table>\n\nProse Convention follows.\n"
    assert _safe_to_inline(t, t.index("Convention"))


def test_not_safe_inside_an_existing_link():
    t = "see [Convention](http://x) for details"
    assert not _safe_to_inline(t, t.index("Convention"))
