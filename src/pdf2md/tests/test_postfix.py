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
