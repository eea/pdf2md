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
