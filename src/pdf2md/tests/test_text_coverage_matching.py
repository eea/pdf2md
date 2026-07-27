"""Phase 4: short-line matching in text_coverage.

A short correspondence line (`X → class Y`) present in the .qmd must NOT be
reported missing; a genuinely-absent short line still must be; long paragraphs
keep using shingle matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md.verify.checks.text_coverage import (  # noqa: E402
    _join_wrapped, _short_line_covered, _TOC_LEADER_RE)
from pdf2md.verify.textutil import tokens  # noqa: E402


def test_wrapped_lines_join_into_one_sentence():
    lines = ["In some cases, especially when steep terrain is involved,",
             "pre-processing aiming to level illumination effects is applied."]
    assert _join_wrapped(lines) == (
        "In some cases, especially when steep terrain is involved, "
        "pre-processing aiming to level illumination effects is applied.")


def test_heading_stays_separate():
    assert _join_wrapped(["4.2.2 Results", "The validation confirms the trends."]) == \
        "4.2.2 Results\nThe validation confirms the trends."


def test_trailing_unpunctuated_buffer_kept():
    assert _join_wrapped(["a final line without punctuation"]) == \
        "a final line without punctuation"


def test_toc_leader_matches():
    assert _TOC_LEADER_RE.search("2.1 Scope of the Document ........... 12")
    assert not _TOC_LEADER_RE.search("An ordinary sentence ending in 2021.")


def _index(qmd_plain: str):
    qtoks = tokens(qmd_plain)
    positions = {}
    for i, t in enumerate(qtoks):
        positions.setdefault(t, []).append(i)
    return qtoks, positions


def test_present_short_arrow_line_is_covered():
    # the .qmd renders the correspondence line (arrow is stripped by normalize)
    qmd = "Some intro text. Allotment gardens class 1.4. More text follows here."
    qtoks, pos = _index(qmd)
    line = tokens("Allotment gardens class 1.4")
    assert _short_line_covered(line, qtoks, pos)


def test_present_with_one_extra_token_still_covered():
    # a footnote superscript sneaks a token into the middle — windowed match tolerates it
    qmd = "Allotment gardens ¹ class 1.4 are complexes of land parcels."
    qtoks, pos = _index(qmd)
    line = tokens("Allotment gardens class 1.4")
    assert _short_line_covered(line, qtoks, pos)


def test_genuinely_absent_short_line_reported_missing():
    qmd = "This document is about urban land cover classes and nomenclature."
    qtoks, pos = _index(qmd)
    line = tokens("Zebra crossings map to class 9.9")
    assert not _short_line_covered(line, qtoks, pos)


def test_scattered_words_not_falsely_covered():
    # the needle tokens all exist in the doc but far apart — must NOT count as covered
    qmd = ("Allotment plots appear early. " + "filler " * 40 +
           "gardens of a different kind. " + "filler " * 40 + "class system. ")
    qtoks, pos = _index(qmd)
    line = tokens("Allotment gardens class 1.4")
    assert not _short_line_covered(line, qtoks, pos)

# ── classifier used for both strict and effective coverage ──────────────────────

from pdf2md.verify.checks.text_coverage import _classify, _index as _tc_index, _ld_split  # noqa: E402


def _st(s):
    return _ld_split(tokens(s))


def test_classify_covered_when_in_order():
    idx = _tc_index("The annual precipitation across the northern region exceeded historical averages.")
    assert _classify(_st("The annual precipitation across the northern region exceeded"), idx) == "covered"


def test_classify_missing_when_absent():
    idx = _tc_index("The annual precipitation across the northern region exceeded historical averages.")
    assert _classify(_st("Spacecraft telemetry showed anomalous readings during orbital insertion burn"),
                     idx) == "missing"


def test_effective_recovers_gap_via_appendix():
    # a sentence absent from the body but present in the recovery appendix flips
    # missing (strict, appendix stripped) → covered (effective, appendix included)
    sentence = "Groundwater recharge rates declined sharply across the karst plateau last decade."
    body = "# Doc\n\nUnrelated intro paragraph about surface runoff and evaporation totals.\n"
    appendix = f"\n\n<!-- postfix: missing-text recovery -->\n\n{sentence}\n"
    strict = _tc_index(body)                 # appendix stripped upstream
    effective = _tc_index(body + appendix)
    assert _classify(_st(sentence), strict) == "missing"
    assert _classify(_st(sentence), effective) == "covered"


# ── structural markers are notation, not content ────────────────────────────────

from pdf2md.verify.checks.text_coverage import _strip_structural_markers  # noqa: E402


def test_strips_pdf_letter_o_bullet():
    # Word exports render list bullets as a literal 'o'; the converter emits a markdown
    # bullet. Same meaning, so the 'o' must not count as a missing word.
    assert _strip_structural_markers("o 21: pure needle leaved >75 %") == "21: pure needle leaved >75 %"


def test_strips_symbol_and_dash_bullets():
    assert _strip_structural_markers("• bullet item here") == "bullet item here"
    assert _strip_structural_markers("- dash item here") == "dash item here"


def test_strips_source_section_numbering():
    # Quarto numbers headings itself, so the source's numbering is dropped by design
    assert _strip_structural_markers("3.3.2.2.2 Workflow") == "Workflow"


def test_keeps_a_leading_year():
    # requires a dotted number, so real data opening with a year is untouched
    assert _strip_structural_markers("2018 Land cover map") == "2018 Land cover map"


def test_keeps_ordinary_prose_starting_with_o():
    assert _strip_structural_markers("output of the process") == "output of the process"
