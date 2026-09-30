"""Postfix in-place recovery: where recovered prose gets put back.

The converter intermittently drops a passage (LLM non-determinism). Postfix detects
it and recovers the prose from the source PDF — these tests cover putting it back
*in flow* rather than in an end-of-document appendix.
"""
import json
import sys
from pathlib import Path

import pytest

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


def test_build_units_drops_numbered_running_footer(tmp_path):
    import fitz
    from pdf2md.postfix import _build_units
    doc = fitz.open()
    bodies = [f"Distinct paragraph number {w} about the processing chain for stage {w}."
              for w in ("alpha", "beta", "gamma", "delta")]
    for i, body in enumerate(bodies):        # footer number differs; body differs per page
        p = doc.new_page()
        p.insert_text((72, 60), f"Page | {i + 1}")
        p.insert_text((72, 120), body)
    src = tmp_path / "d.pdf"
    doc.save(str(src)); doc.close()
    units = _build_units(src)
    joined = " ".join(u[1] for u in units)
    assert "Page |" not in joined                          # numbered footer gone
    assert "processing chain" in joined                    # bodies kept


def test_build_units_drops_figure_internal_text(tmp_path):
    """Class codes stamped over a satellite image are part of the picture. Without the
    exclusion the repair pass re-injects them as a stray line of digits (PA21: a line
    reading "1110 1110 1110 2110 …" that appears nowhere in the source prose)."""
    import fitz
    from pdf2md.postfix import _build_units, _figure_boxes
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "The delineation rules for urban fabric are described here.")
    page.insert_text((200, 400), "1110")          # inside the figure
    page.insert_text((260, 430), "2110")          # inside the figure
    src = tmp_path / "d.pdf"
    doc.save(str(src)); doc.close()
    fig_box = {0: [(150.0, 350.0, 450.0, 500.0)]}

    joined = " ".join(u[1] for u in _build_units(src))
    assert "1110" in joined                       # today's behaviour: leaks in
    joined = " ".join(u[1] for u in _build_units(src, fig_box))
    assert "1110" not in joined and "2110" not in joined
    assert "delineation rules" in joined          # body prose untouched


def test_figure_boxes_reads_detections(tmp_path):
    from pdf2md.postfix import _figure_boxes
    (tmp_path / "detections.json").write_text(json.dumps({"figures": [
        {"file": "a.png", "page": 3, "bbox": [1, 2, 3, 4]},
        {"file": "b.png", "page": 3, "bbox": [5, 6, 7, 8]},
        {"file": "c.png", "page": 9, "bbox": None},        # unmaterialised: skipped
    ]}), encoding="utf-8")
    assert _figure_boxes(tmp_path) == {3: [(1, 2, 3, 4), (5, 6, 7, 8)]}
    assert _figure_boxes(tmp_path / "nope") == {}          # missing file is not fatal


def test_locate_after_unique_anchor():
    from pdf2md.postfix import _qmd_word_offsets, _locate_after
    qmd = "Intro line.\n\nThe fox jumps over the lazy dog here.\n\nTrailing text.\n"
    words, starts, ends = _qmd_word_offsets(qmd)
    at = _locate_after(words, ends, ["the", "lazy", "dog"])
    assert at is not None and "dog" in qmd[:at] and qmd[at:].lstrip().startswith("here")


def test_locate_after_declines_ambiguous():
    from pdf2md.postfix import _qmd_word_offsets, _locate_after
    qmd = "the same words here and the same words there"
    words, starts, ends = _qmd_word_offsets(qmd)
    assert _locate_after(words, ends, ["the", "same", "words"]) is None   # appears twice


def test_body_pdf_prefers_working_copy(tmp_path):
    from pdf2md.postfix import _body_pdf
    (tmp_path / "d.source.pdf").write_bytes(b"src")
    assert _body_pdf(tmp_path, "d").name == "d.source.pdf"          # fallback
    (tmp_path / "d.working.pdf").write_bytes(b"work")
    assert _body_pdf(tmp_path, "d").name == "d.working.pdf"          # preferred


def test_anchor_by_context_grows_until_unique():
    # a short suffix that repeats ("the report") is disambiguated by adding more of
    # the preceding context, keeping placement exact
    from pdf2md.postfix import _qmd_word_offsets, _anchor_by_context
    qmd = ("intro the report says one thing here.\n\n"
           "the distinctive alpha beta gamma prelude to the report says another.\n")
    words, starts, ends = _qmd_word_offsets(qmd)
    ctx = ["distinctive", "alpha", "beta", "gamma", "prelude", "to", "the", "report", "says"]
    at = _anchor_by_context(words, ends, ctx)
    assert at is not None and qmd[:at].count("the report says") == 2  # landed at 2nd, unique run


def test_region_gaps_absorbs_fragments_into_one_collapsed_gap():
    # a collapsed references block: two solid body sentences bracket a region that is
    # mostly missing with only tiny surviving fragments — must be ONE gap, not many
    from pdf2md.postfix import _region_gaps
    def u(text):
        toks = text.lower().split()
        return (0, text, toks)
    units = [
        u("This is a solid body sentence before the references section here"),  # 0 solid present
        u("First missing reference entry with plenty of words to inject here"),  # 1 missing
        u("Pol"),                                                                # 2 tiny fragment present
        u("Second missing reference entry also long enough to be injected now"),  # 3 missing
        u("J"),                                                                  # 4 tiny fragment present
        u("Third missing reference entry likewise carrying many words indeed"),  # 5 missing
        u("Another solid body sentence after the references block appears now"),  # 6 solid present
    ]
    present = [True, False, True, False, True, False, True]
    gaps = _region_gaps(units, present)
    assert len(gaps) == 1                       # one region gap, not three
    before, run, after = gaps[0]
    assert before == 0 and after == 6           # bounded by the solid survivors
    assert run == [1, 3, 5]                      # only the MISSING units; fragments (2,4) skipped


def test_find_unique_returns_start_index():
    from pdf2md.postfix import _qmd_word_offsets, _find_unique
    qmd = "alpha beta gamma delta epsilon"
    words, starts, ends = _qmd_word_offsets(qmd)
    i = _find_unique(words, ["gamma", "delta"])
    assert i == 2 and starts[i] == qmd.index("gamma")


def test_safe_boundary_before_finds_paragraph_start():
    from pdf2md.postfix import _safe_boundary_before
    text = "First para.\n\nSecond para here.\n\nThird."
    pos = text.index("Third")
    at = _safe_boundary_before(text, pos)
    assert text[at:].startswith("Third")


def test_anchor_gap_uses_after_sentence_when_before_ambiguous():
    # before-sentence is non-unique ("see below"); after-sentence is distinctive,
    # so Tier 2c anchors the gap right before it (and rescues a top-of-doc gap)
    from pdf2md.postfix import _qmd_word_offsets, _anchor_gap
    qmd = ("see below\n\nThe distinctive concluding paragraph about ice cover.\n")
    words, starts, ends = _qmd_word_offsets(qmd)
    units = [(0, "see below", ["see", "below"]),
             (0, "the missing sentence text here", ["the", "missing", "sentence", "text", "here"]),
             (0, "The distinctive concluding paragraph about ice cover.",
              ["the", "distinctive", "concluding", "paragraph", "about", "ice", "cover"])]
    present = [True, False, True]
    at = _anchor_gap(units, present, 0, 2, words, starts, ends, qmd)
    # a duplicate "see below" makes the before-anchor ambiguous → falls to Tier 2c
    qmd2 = "see below\n\n" + qmd
    w2, s2, e2 = _qmd_word_offsets(qmd2)
    at2 = _anchor_gap(units, present, 0, 2, w2, s2, e2, qmd2)
    assert at2 is not None and qmd2[at2:].lstrip().startswith("The distinctive")


def test_faithful_accepts_verbatim_rejects_rewrite():
    from pdf2md.postfix import _faithful
    src = "Regions with scarce vegetation typically indicate high human influence."
    assert _faithful(src, "Regions with scarce vegetation typically indicate high human influence.")
    assert not _faithful(src, "Something completely different about unrelated topics entirely.")


def test_clean_raw_joins_hyphenation():
    from pdf2md.postfix import _clean_raw
    assert _clean_raw("topo-\ngraphic normal-\nisation applied") == "topographic normalisation applied"


def test_gap_convert_parses_labelled_response(monkeypatch):
    import pdf2md.postfix as pf
    def fake_post(**kw):
        return ("<<<GAP 1>>>\nFirst recovered.\n\n<<<GAP 2>>>\nSecond recovered.",
                {"cost": 0.004})
    monkeypatch.setattr(pf, "_post_with_retries", fake_post)
    out, cost = pf._llm_convert_gaps("k", [(1, "first raw"), (2, "second raw")])
    assert out == {1: "First recovered.", 2: "Second recovered."} and cost == 0.004


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


# ── footnote table-note rescue ──────────────────────────────────────────────────

_FN_QMD = (
    "# Doc\n\n"
    "Body ref links here.[^4]\n\n"
    "```{=html}\n"
    "<table><tr><td>Resampling via GdalWarp<sup>1</sup> in UTM<sup>5</sup>.</td></tr></table>\n"
    "```\n\n"
    "[^1]: GDAL 3.8.3 package.\n"
    "[^5]: WGS84/UTM projection note.\n"
    "[^4]: A real linked footnote.\n"
    "[^9]: A definition whose mark was never emitted.\n"
)


def test_postfix_footnotes_converts_orphaned_intable_defs(tmp_path):
    from pdf2md.postfix import _postfix_footnotes
    qmd = tmp_path / "d.qmd"
    qmd.write_text(_FN_QMD, encoding="utf-8")
    n = _postfix_footnotes(qmd, tmp_path)
    out = qmd.read_text(encoding="utf-8")
    assert n == 2                                   # only [^1] and [^5] (have <sup> marks)
    assert "^1^ GDAL 3.8.3 package." in out         # def rewritten to a visible note
    assert "^5^ WGS84/UTM projection note." in out
    assert "[^1]:" not in out and "[^5]:" not in out
    assert "<sup>1</sup>" in out and "<sup>5</sup>" in out   # marks left in the cell
    assert "[^4]:" in out and "[^4]" in out         # linked footnote untouched
    assert "[^9]: A definition" in out              # mark-less orphan left alone


def test_collapse_doubled_doi():
    from pdf2md.postfix import _collapse_doubled_doi
    # double-slash (as the LLM transcribes it) and fitz's single-slash variant both collapse
    for src in ("<https://doi.org/https://doi.org/10.1016/j.rse.2020.111685>",
                "https://doi.org/https:/doi.org/10.1002/ecm.1337"):
        out, n = _collapse_doubled_doi(src)
        assert n == 1 and "doi.org/https" not in out
        assert out.count("doi.org/") == 1
    # a clean DOI is untouched
    out, n = _collapse_doubled_doi("https://doi.org/10.3390/rs9121271")
    assert n == 0


def test_despace_url():
    from pdf2md.postfix import _despace_url
    qmd = "Ref. http://www.mdpi.com/2072- 4292/10/4/635 end."
    out, fixed = _despace_url("http://www.mdpi.com/2072-4292/10/4/635", qmd)
    assert fixed and "http://www.mdpi.com/2072-4292/10/4/635 end." in out
    assert "2072- 4292" not in out
    # already contiguous -> no change
    out2, fixed2 = _despace_url("http://www.mdpi.com/2072-4292/10/4/635", out)
    assert not fixed2 and out2 == out


def test_recover_links_collapses_doubled_doi_in_body(tmp_path):
    """A doubled DOI the converter transcribed into the body is collapsed to a valid,
    matchable link — repaired in the OUTPUT, so verify then finds it on its own."""
    import fitz
    from pdf2md.postfix import _recover_links
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "See reference.")
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 68, 200, 82),
                      "uri": "https://doi.org/https:/doi.org/10.1016/j.rse.2020.111685"})
    doc.save(str(tmp_path / "d.source.pdf"))
    doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Bolton (2020). <https://doi.org/https://doi.org/"
                   "10.1016/j.rse.2020.111685>\n", encoding="utf-8")
    repaired, dropped = _recover_links(qmd, tmp_path)
    body = qmd.read_text(encoding="utf-8")
    assert "<https://doi.org/10.1016/j.rse.2020.111685>" in body   # collapsed in place
    assert "doi.org/https" not in body and dropped == 0
    assert "## Source links" not in body


def test_recover_links_despaces_wrapped_url(tmp_path):
    """A URL broken by a line-wrap space is de-spaced in place, using the clean
    annotation URI as the search key (no boundary guessing)."""
    import fitz
    from pdf2md.postfix import _recover_links
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "reference line")
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 68, 260, 82),
                      "uri": "http://www.mdpi.com/2072-4292/10/4/635"})
    doc.save(str(tmp_path / "d.source.pdf"))
    doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Jönsson et al. http://www.mdpi.com/2072- 4292/10/4/635\n",
                   encoding="utf-8")
    repaired, dropped = _recover_links(qmd, tmp_path)
    body = qmd.read_text(encoding="utf-8")
    assert "http://www.mdpi.com/2072-4292/10/4/635" in body         # space closed
    assert "2072- 4292" not in body and repaired == 1 and dropped == 0


def test_recover_links_no_synthetic_section_for_uninlinable(tmp_path):
    """A link that can't be placed inline (anchor is a bare URL, not present in the body)
    is left to verify — we no longer append a synthetic '## Source links' section."""
    import fitz
    from pdf2md.postfix import _recover_links
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "https://example.org/report")
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 68, 300, 82),
                      "uri": "https://example.org/report"})
    doc.save(str(tmp_path / "d.source.pdf"))
    doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Body text with no link.\n", encoding="utf-8")
    inlined, dropped = _recover_links(qmd, tmp_path)
    assert inlined == 0 and dropped == 1
    assert "## Source links" not in qmd.read_text(encoding="utf-8")


def test_recover_links_inlines_unambiguous_anchor(tmp_path):
    """Inline restoration still works: a real, once-occurring anchor gets its href."""
    import fitz
    from pdf2md.postfix import _recover_links
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "the Copernicus portal")
    page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(72, 68, 250, 82),
                      "uri": "https://land.copernicus.eu/"})
    doc.save(str(tmp_path / "d.source.pdf"))
    doc.close()
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Data comes from the Copernicus portal for Europe.\n", encoding="utf-8")
    inlined, dropped = _recover_links(qmd, tmp_path)
    assert inlined == 1
    assert "[the Copernicus portal](https://land.copernicus.eu/)" in qmd.read_text()


def test_postfix_footnotes_noop_when_all_linked(tmp_path):
    from pdf2md.postfix import _postfix_footnotes
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Text with a ref.[^1]\n\n[^1]: linked note.\n", encoding="utf-8")
    assert _postfix_footnotes(qmd, tmp_path) == 0


# ── focused-crop table repair ───────────────────────────────────────────────────

def test_qmd_table_spans_finds_pipe_and_html_blocks():
    from pdf2md.postfix import _qmd_table_spans
    qmd = (
        "Intro text.\n\n"
        "| A | B |\n|---|---|\n| x1 | y1 |\n\n"
        "Prose between.\n\n"
        "```{=html}\n<table><tr><td>alpha</td><td>beta</td></tr></table>\n```\n\n"
        "Tail.\n"
    )
    spans = _qmd_table_spans(qmd)
    assert len(spans) == 2
    (s1, e1, t1), (s2, e2, t2) = spans
    assert "x1" in t1 and "y1" in t1
    assert "alpha" in t2 and "beta" in t2
    assert qmd[s2:e2].startswith("```{=html}") and qmd[s2:e2].endswith("```")


def test_crop_replace_guard():
    from pdf2md.postfix import _crop_replace_ok
    dist = {"v1", "v2", "v3", "v4"}
    src = dist | {"header", "unit"}
    block = {"v1", "v2", "header"}          # incumbent holds 2 of 4 values
    better = {"v1", "v2", "v3", "header"}   # keeps both, gains one → ok
    assert _crop_replace_ok(dist, better, block, src)
    lossy = {"v1", "v3", "v4"}              # gains two but LOSES v2 → decline
    assert not _crop_replace_ok(dist, lossy, block, src)
    nogain = {"v1", "v2"}                   # keeps but adds nothing → decline
    assert not _crop_replace_ok(dist, nogain, block, src)
    merged_block = block | {"other%d" % i for i in range(10)}  # mostly alien content
    assert not _crop_replace_ok(dist, better, merged_block, src)  # multi-page merge
    third_alien = block | {"o1", "o2"}      # ~40% alien: above the 0.25 ceiling
    assert not _crop_replace_ok(dist, better, third_alien, src)


# ── structural table cleanup ────────────────────────────────────────────────────

def test_pipe_empty_ratio():
    from pdf2md.postfix import _pipe_empty_ratio
    clean = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    mangled = "| a |  | b |  |  |\n|---|---|---|---|---|\n| 1 |  | 2 |  |  |\n"
    assert _pipe_empty_ratio(clean) == 0.0
    assert _pipe_empty_ratio(mangled) > 0.5


def test_repair_mangled_declines_when_no_source(tmp_path):
    # no source pdf → cannot re-crop → no-op, never raises
    from pdf2md.postfix import _repair_mangled_tables
    q = tmp_path / "d.qmd"
    q.write_text("| x |  |  |\n|---|---|---|\n| prose here |  |  |\n", encoding="utf-8")
    assert _repair_mangled_tables(q, tmp_path, api_key="k") == (0, 0.0)


def test_repair_mangled_flags_only_high_empty(monkeypatch, tmp_path):
    # a clean table must not be flagged; verify the detector gate, not the LLM path
    import pdf2md.postfix as pf
    from pdf2md.postfix import _PIPE_BLOCK_RE, _pipe_empty_ratio, _MANGLE_EMPTY_RATIO
    clean = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n"
    flagged = [b.group(0) for b in _PIPE_BLOCK_RE.finditer(clean)
               if b.group(0).count("\n") >= 3
               and _pipe_empty_ratio(b.group(0)) > _MANGLE_EMPTY_RATIO]
    assert flagged == []


def test_numbers_extraction():
    from pdf2md.postfix import _numbers
    n = _numbers("| SOSD | 0.99 | -0.01 | 4.4×10⁷ | 99.54 |")
    assert "0.99" in n and "-0.01" in n and "99.54" in n


# ── unfenced-code fencing ───────────────────────────────────────────────────────

def test_fence_unfenced_shell_script():
    from pdf2md.postfix import _fence_unfenced_code
    qmd = ("# Method\n\nWe run the following processing script.\n\n"
           "#!/usr/bin/env bash\nset -euo pipefail\n"
           'OUTDIR="./out"\nfor p in "${PARAMS[@]}"; do\n'
           '  f="$(mktemp)"\n  echo "$p"\ndone\n\n'
           "The output is then aggregated.\n")
    out, n = _fence_unfenced_code(qmd)
    assert n == 1
    assert "```bash\n#!/usr/bin/env bash" in out
    assert out.count("```") == 2                      # exactly one fenced block
    assert "We run the following" in out and "The output is then" in out  # prose untouched


def test_fence_leaves_prose_and_existing_fences_alone():
    from pdf2md.postfix import _fence_unfenced_code
    qmd = ("Normal prose about $x$ and costs of $5 and a price.\n\n"
           "```bash\necho already fenced ${VAR}\n```\n\n"
           "More prose with one ${stray} token but no other code signals here.\n")
    out, n = _fence_unfenced_code(qmd)
    assert n == 0 and out == qmd                       # nothing to do → unchanged


def test_fence_needs_two_strong_signals():
    from pdf2md.postfix import _fence_unfenced_code
    # a single ${x} in a prose paragraph must not trigger fencing
    qmd = "Set the variable ${HOME} in your shell profile before running.\n"
    out, n = _fence_unfenced_code(qmd)
    assert n == 0


def test_breadcrumb_comments_stripped_but_markers_kept():
    import re
    qmd = ("# Doc\n\n<!-- postfix: recovered in place (source p4) -->\n\n"
           "Recovered sentence here.\n\n"
           "<!-- figures detected in Phase 1 but not placed\n   FIG_9: x -> y\n-->\n\n"
           "More text.\n\n<!--pdf2md-repair-3-->\n\n"  # functional marker: keep
           "Tail.\n\n<!-- postfix: table 2 re-rendered from crop -->\n\n"
           "## Results\n\n- item\n")
    cleaned = re.sub(r'(?m)^[ \t]*<!-- postfix:[^\n]*-->[ \t]*\n?', '', qmd)
    cleaned = re.sub(r'<!-- figures detected in Phase 1.*?-->\n?', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    assert '<!-- postfix:' not in cleaned          # breadcrumbs gone
    assert 'figures detected in Phase 1' not in cleaned
    assert '<!--pdf2md-repair-3-->' in cleaned      # functional marker kept
    assert 'Recovered sentence here.' in cleaned and '## Results' in cleaned


def test_strip_front_matter_noise():
    from pdf2md.postfix import _strip_front_matter_noise
    qmd = ("---\ntitle: X\n---\n"
           "## DOCUMENT CHANGE LOG\n\n"
           "| Issue | Date |\n|---|---|\n| 1.0 | 2025 |\n\n"
           "1. Contents 1 1.1 1.2 2 2.1 3 3.1 List of figures Figure 6.\n\n"
           "Contact:\nCLMS\nProject Officer: Someone\n\n"
           "Disclaimer:\nAll Rights Reserved.\n\n"
           "# Introduction\n\nWe deliver phenology products to users.\n")
    out, n = _strip_front_matter_noise(qmd)
    assert n == 3                    # TOC, Contact, Disclaimer — cover/TOC only
    assert "Contact:" not in out and "Disclaimer:" not in out
    assert "1. Contents 1 1.1" not in out
    # policy: keep the change log (convert as much as possible) — only cover/TOC dropped
    assert "DOCUMENT CHANGE LOG" in out and "| Issue | Date |" in out
    assert "# Introduction" in out and "We deliver phenology" in out    # kept


def test_unnumber_frontmatter_headings():
    from pdf2md.postfix import _unnumber_frontmatter_headings
    qmd = ("## Document Change Log\n\n| Issue | Date |\n\n"
           "## Introduction\n\n### Revision History\n")
    out, n = _unnumber_frontmatter_headings(qmd)
    assert n == 2
    assert "## Document Change Log {.unnumbered}" in out
    assert "### Revision History {.unnumbered}" in out
    assert "## Introduction\n" in out               # body heading untouched
    # idempotent: already-attributed heading is left alone
    out2, n2 = _unnumber_frontmatter_headings(out)
    assert n2 == 0 and out2 == out


def test_strip_front_matter_keeps_body_prose_mentioning_services():
    from pdf2md.postfix import _strip_front_matter_noise
    qmd = ("---\nt: x\n---\n# Intro\n\nWithin this framework, the Copernicus Land "
           "Monitoring Service (CLMS) and the European Environment Agency (EEA) "
           "deliver products.\n")
    out, n = _strip_front_matter_noise(qmd)
    assert n == 0 and "Within this framework" in out   # body prose untouched


def test_escape_text_underscores():
    from pdf2md.postfix import _escape_text_underscores
    src = r"$$\text{DVI} = \text{NIR_R} - \text{RED_R}, \quad \text{Eq. 2}$$"
    out, n = _escape_text_underscores(src)
    assert n == 2                                   # NIR_R and RED_R
    assert r"\text{NIR\_R}" in out and r"\text{RED\_R}" in out
    assert r"\text{DVI}" in out                      # untouched groups stay
    # a real subscript OUTSIDE \text{} must not be escaped
    src2 = r"$\text{MINV}_{\text{extrema}}$"
    out2, n2 = _escape_text_underscores(src2)
    assert n2 == 0 and out2 == src2
    # idempotent: already-escaped stays put
    out3, n3 = _escape_text_underscores(out)
    assert n3 == 0 and out3 == out


def test_wrap_labeled_equations():
    from pdf2md.postfix import _wrap_labeled_equations
    src = "NDSI= (Pgreen-PSWIR1)/(Pgreen +PSWIR1). {#eq:eq2}\n"
    out, n = _wrap_labeled_equations(src)
    assert n == 1
    assert out.startswith("$$ NDSI= (Pgreen-PSWIR1)/(Pgreen +PSWIR1). $$ {#eq-eq2}")
    # already-math or unlabelled lines untouched
    for keep in ("$$\\text{x}=1$$ {#eq-1}\n", "Normal prose with an = sign.\n",
                 "See section {#sec-3}.\n"):
        o, k = _wrap_labeled_equations(keep)
        assert k == 0 and o == keep


def test_strip_raw_math_lines():
    from pdf2md.postfix import _strip_raw_math_lines
    qmd = ("Prose before.\n\n"
           "\U0001D444\U0001D438= \U0001D451\U0001D450+ (1) ∙ \U0001D43A cos(\U0001D703), Eq. 4\n\n"
           "$$\\text{G} = \\text{G_0}, \\quad \\text{Eq. 4}$$\n\n"
           "Normal prose with an italic $x$ and one stray \U0001D45D symbol stays.\n")
    out, n = _strip_raw_math_lines(qmd)
    assert n == 1                                  # only the tofu equation line
    assert "cos(" not in out                        # tofu line gone
    assert "$$\\text{G}" in out                     # proper LaTeX kept
    assert "Normal prose" in out                    # 1 stray math char -> kept


def test_fix_midtable_separators():
    from pdf2md.postfix import _fix_midtable_separators
    # SAME width -> spurious continuation separator dropped, one table kept
    cont = ("| RD | Reference document |\n| RLIE | River and Lake Ice Extent |\n"
            "| :--- | :--- |\n| S1 | Sentinel-1 |\n")
    out, n = _fix_midtable_separators(cont)
    assert n == 1 and "| :--- | :--- |" not in out
    assert "| RLIE | River and Lake Ice Extent |\n| S1 | Sentinel-1 |" in out
    # DIFFERENT width -> new table's header glued on, split with a blank line
    glued = ("| Approved by: | J D | MAG | 2022 | |\n| Document reference : | COSIMS |\n"
             "| :--- | :--- |\n| Edition.Revision : | 2.6 |\n")
    out2, n2 = _fix_midtable_separators(glued)
    assert n2 == 1
    assert "| Approved by: | J D | MAG | 2022 | |\n\n| Document reference : | COSIMS |" in out2
    # a normal table (header+separator preceded by blank) is untouched
    ok = "text\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    out3, n3 = _fix_midtable_separators(ok)
    assert n3 == 0 and out3 == ok


def test_strip_heading_numbers():
    from pdf2md.postfix import _strip_heading_numbers
    qmd = ("## 1. Introduction\n\n### 1.4.1 Applicable documents\n\n"
           "text\n\n"
           "```{=typst}\n#set text(size: 9pt)\n```\n\n"
           "## 3D reconstruction\n\n## 5\n")
    out, n = _strip_heading_numbers(qmd)
    assert n == 2
    assert "## Introduction" in out and "### Applicable documents" in out
    assert "#set text(size: 9pt)" in out          # raw-typst line untouched
    assert "## 3D reconstruction" in out          # not a section number
    assert "## 5\n" in out                         # number-only heading kept as-is


def test_ensure_pipe_table_blanks():
    from pdf2md.postfix import _ensure_pipe_table_blanks
    # header glued to a caption/heading -> blank inserted; already-spaced table untouched
    qmd = ("## Document Change Log\n| Issue | Date |\n|---|---|\n| 1.0 | 2025 |\n\n"
           "Table 1 (Continued).\n| A | B |\n|---|---|\n| x | y |\n\n"
           "Fine table below.\n\n| C | D |\n|---|---|\n| 1 | 2 |\n")
    out, n = _ensure_pipe_table_blanks(qmd)
    assert n == 2                                       # two glued tables fixed
    assert "## Document Change Log\n\n| Issue | Date |" in out
    assert "Table 1 (Continued).\n\n| A | B |" in out
    assert out.count("| C | D |") == 1                 # already-spaced one untouched
    # idempotent
    out2, n2 = _ensure_pipe_table_blanks(out)
    assert n2 == 0


def test_ensure_list_blanks():
    from pdf2md.postfix import _ensure_list_blanks
    # a list glued to the paragraph above is folded into it by Pandoc and renders as
    # literal `*` text (PA21 "This category includes:")
    qmd = ("**This category includes:**\nVery dense Urban:\n"
           "*   Built-up areas\n*   Buildings and roads\n\n"
           "## Heading\n*   heading closes the block, no blank needed\n\n"
           "Spaced already:\n\n*   untouched\n")
    out, n = _ensure_list_blanks(qmd)
    assert n == 1                                      # only the glued list
    assert "Very dense Urban:\n\n*   Built-up areas" in out
    assert "## Heading\n*   heading closes" in out    # heading untouched
    assert out.count("*   untouched") == 1
    out2, n2 = _ensure_list_blanks(out)
    assert n2 == 0 and out2 == out                     # idempotent


def test_ensure_list_blanks_skips_front_matter_and_fences():
    from pdf2md.postfix import _ensure_list_blanks
    # YAML `- ` entries are not Markdown lists; a fenced block is verbatim
    qmd = ("---\nformat:\n  - pdf\n  - html\n---\n"
           "Intro line:\n```\ncode:\n- not a list\n```\n")
    out, n = _ensure_list_blanks(qmd)
    assert n == 0 and out == qmd


def test_ensure_list_blanks_leaves_list_continuations(tmp_path):
    from pdf2md.postfix import _ensure_list_blanks
    # an indented continuation line inside a list must not split the list
    qmd = ("*   Industrial, commercial, public units\n"
           "    $\\rightarrow$ class 1.1.2\n"
           "*   Allotment gardens\n")
    out, n = _ensure_list_blanks(qmd)
    assert n == 0 and out == qmd


def _colour_pdf(tmp_path):
    """A one-page PDF with two filled cells and their text, like a nomenclature table."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page()
    page.draw_rect(fitz.Rect(50, 50, 250, 90), color=None, fill=(1, 1, 0))      # #FFFF00
    page.draw_rect(fitz.Rect(50, 90, 250, 130), color=None, fill=(0, 0.5, 0))   # #008000
    page.insert_text((60, 75), "Managed grassland")
    page.insert_text((60, 115), "Natural woodland")
    out = tmp_path / "t.pdf"
    doc.save(str(out)); doc.close()
    return out


def _two_page_table_pdf(tmp_path, same_page=False):
    """A table printed across two pages, each repeating its header — or, when
    same_page is set, two sibling tables sharing a header on ONE page."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    def _draw(page, y, first):
        page.insert_text((60, y), "Level 1")
        page.insert_text((160, y), "Level 2")
        page.insert_text((60, y + 20),
                         "1. Urban fabric areas" if first else "4. Grassland areas")
        page.insert_text((160, y + 20),
                         "1.1 industrial units" if first else "4.1 managed grassland")
    if same_page:
        page = doc.new_page(); _draw(page, 100, True); _draw(page, 300, False)
    else:
        _draw(doc.new_page(), 100, True); _draw(doc.new_page(), 100, False)
    out = tmp_path / ("same.pdf" if same_page else "split.pdf")
    doc.save(str(out)); doc.close()
    return out


_COLS = '<colgroup>\\n<col style="width: 50%">\\n<col style="width: 50%">\\n</colgroup>\\n'
_T1 = ("```{=html}\n<table>\n" + _COLS + "  <tr><td>Level 1</td><td>Level 2</td></tr>\n"
       "  <tr><td>1. Urban fabric areas</td><td>1.1 industrial units</td></tr>\n</table>\n```")
_T2 = ("```{=html}\n<table>\n" + _COLS + "  <tr><td>Level 1</td><td>Level 2</td></tr>\n"
       "  <tr><td>4. Grassland areas</td><td>4.1 managed grassland</td></tr>\n</table>\n```")


def test_merges_a_table_split_across_two_source_pages(tmp_path):
    """One source table printed per page with a repeated header becomes one slot per
    page, so the document ends up with two tables where the source has one."""
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path)
    out, n = _merge_continued_tables(_T1 + "\n\n" + _T2 + "\n", pdf)
    assert n == 1
    assert out.count("<table>") == 1
    assert "1. Urban fabric areas" in out and "4. Grassland areas" in out  # no rows lost
    assert out.count("<td>Level 1</td>") == 1                # repeated header dropped
    # neither <colgroup> survives: the first was measured on half the table and a
    # second spliced mid-table makes the markup invalid (Pandoc then silently drops
    # the whole table). run_postfix re-stamps one against the joined columns.
    assert out.count("<colgroup>") == 0
    assert _merge_continued_tables(out, pdf)[1] == 0         # idempotent


def test_merges_tables_whose_tag_carries_attributes(tmp_path):
    """The grid reader records the sampled font size on the opening tag; a merge
    pattern that insisted on a bare "<table>" silently stopped matching and the split
    table stayed split."""
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path)
    a = _T1.replace("<table>", '<table style="font-size:8pt">')
    b = _T2.replace("<table>", '<table style="font-size:8pt">')
    out, n = _merge_continued_tables(a + "\n\n" + b + "\n", pdf)
    assert n == 1 and out.count("<table") == 1
    assert 'font-size:8pt' in out                     # the size survives the join


def test_merges_across_a_footnote_definition(tmp_path):
    """The converter drops a footnote definition wherever its reference fell, which
    is nondeterministic — one run put it between the two halves of the nomenclature
    table and blocked the merge. It is carried past the join, not treated as content
    that separates the halves."""
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path)
    note = "[^2]: https://example.org/MAESWorkingPaper2013.pdf"
    out, n = _merge_continued_tables(_T1 + "\n\n" + note + "\n\n" + _T2 + "\n", pdf)
    assert n == 1
    assert out.count("<table>") == 1
    assert note in out                                   # the footnote survives
    assert out.index("</table>") < out.index(note)       # …below the joined table


def test_prose_between_halves_still_blocks_the_merge(tmp_path):
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path)
    out, n = _merge_continued_tables(
        _T1 + "\n\nA sentence of real prose.\n\n" + _T2 + "\n", pdf)
    assert n == 0 and out.count("<table>") == 2


def test_sibling_tables_on_one_page_are_not_merged(tmp_path):
    """Same header, adjacent in the document, but printed side by side on ONE page:
    the page test is the only thing separating these from a continuation."""
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path, same_page=True)
    out, n = _merge_continued_tables(_T1 + "\n\n" + _T2 + "\n", pdf)
    assert n == 0 and out.count("<table>") == 2


def test_tables_separated_by_prose_are_not_merged(tmp_path):
    from pdf2md.postfix import _merge_continued_tables
    pdf = _two_page_table_pdf(tmp_path)
    out, n = _merge_continued_tables(_T1 + "\n\nSome paragraph.\n\n" + _T2 + "\n", pdf)
    assert n == 0 and out.count("<table>") == 2


def test_colorize_tables_samples_cell_colours(tmp_path):
    from pdf2md.postfix import _colorize_tables
    pdf = _colour_pdf(tmp_path)
    qmd = ("<table><tr><td>Managed grassland</td></tr>"
           "<tr><td>Natural woodland</td></tr></table>")
    out, n = _colorize_tables(qmd, pdf)
    assert n == 2
    assert '<td style="background-color:#FFFF00">Managed grassland</td>' in out
    assert '<td style="background-color:#008000">Natural woodland</td>' in out
    # idempotent: a second pass reports nothing and changes nothing
    again, n2 = _colorize_tables(out, pdf)
    assert n2 == 0 and again == out


def test_colorize_tables_merges_and_respects_existing_style(tmp_path):
    from pdf2md.postfix import _colorize_tables
    pdf = _colour_pdf(tmp_path)
    qmd = ('<table><tr><td style="text-align:center">Managed grassland</td></tr>'
           '<tr><td style="background-color:#123456">Natural woodland</td></tr></table>')
    out, n = _colorize_tables(qmd, pdf)
    assert n == 1                                        # only the uncoloured one
    assert 'style="text-align:center; background-color:#FFFF00"' in out
    assert 'background-color:#123456' in out             # model's own colour kept

def test_colorize_tables_leaves_unmatched_cells_alone(tmp_path):
    from pdf2md.postfix import _colorize_tables
    pdf = _colour_pdf(tmp_path)
    qmd = "<table><tr><td>Managed grassland</td><td>Not in the PDF at all</td></tr></table>"
    out, n = _colorize_tables(qmd, pdf)
    assert n == 1 and "<td>Not in the PDF at all</td>" in out


def test_colorize_tables_noop_without_tables_or_pdf(tmp_path):
    from pdf2md.postfix import _colorize_tables
    assert _colorize_tables("no tables here", tmp_path / "missing.pdf") == ("no tables here", 0)
    assert _colorize_tables("<table><tr><td>x</td></tr></table>",
                            tmp_path / "missing.pdf") == ("<table><tr><td>x</td></tr></table>", 0)


def test_group_image_rows_builds_grid_from_source_geometry():
    from pdf2md.postfix import _group_image_rows
    # four photos, two per row in the source -> layout-ncol=2, captions preserved
    figs = [{"file": "a.jpeg", "page": 18, "bbox": [100, 100, 300, 300]},
            {"file": "b.jpeg", "page": 18, "bbox": [400, 110, 600, 310]},
            {"file": "c.jpeg", "page": 18, "bbox": [100, 400, 300, 600]},
            {"file": "d.jpeg", "page": 18, "bbox": [400, 410, 600, 610]}]
    qmd = ("Text above.\n"
           "![Cap A](media/a.jpeg)\n![Cap B](media/b.jpeg)\n"
           "![Cap C](media/c.jpeg)\n![Cap D](media/d.jpeg)\n")
    out, n = _group_image_rows(qmd, figs)
    assert n == 1
    assert "::: {layout-ncol=2}" in out and out.rstrip().endswith(":::")
    # each image is its own block now, so Pandoc keeps the alt text as a caption
    assert "![Cap A](media/a.jpeg)\n\n![Cap B](media/b.jpeg)" in out


def test_lone_image_keeps_its_source_width():
    """With no width attribute Typst stretches the image to the text column: a photo
    printed 210pt wide rendered at 441pt, 2.1x too big and at 44 dpi instead of 92."""
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "a.jpeg", "page": 68, "bbox": [196.0, 342.5, 405.8, 502.0]}]  # 210pt
    out, n = _group_image_rows("Text.\n![Vineyard](m/a.jpeg)\nMore text.\n", figs)
    assert n == 1 and "![Vineyard](m/a.jpeg){width=210pt}" in out
    again, n2 = _group_image_rows(out, figs)          # attrs present -> left alone
    assert n2 == 0 and again == out


def test_full_width_image_gets_no_explicit_width():
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "b.jpeg", "page": 1, "bbox": [60.0, 100.0, 560.0, 400.0]}]    # 500pt
    out, _ = _group_image_rows("Text.\n![Wide](m/b.jpeg)\nMore.\n", figs)
    assert "![Wide](m/b.jpeg)" in out and "width=" not in out


def test_grid_images_are_not_given_widths():
    """Inside a layout div the column already constrains the image, and a width would
    fight Quarto's own proportioning."""
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "a.jpeg", "page": 1, "bbox": [50, 100, 250, 300]},
            {"file": "b.jpeg", "page": 1, "bbox": [300, 100, 500, 300]}]
    out, _ = _group_image_rows("![A](m/a.jpeg)\n![B](m/b.jpeg)\n", figs)
    assert "layout-ncol=2" in out and "width=" not in out


def test_image_with_existing_attrs_is_left_alone():
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "a.jpeg", "page": 1, "bbox": [0, 0, 100, 100]}]
    out, _ = _group_image_rows('Text.\n![A](m/a.jpeg){width=80%}\nMore.\n', figs)
    assert "{width=80%}" in out and "pt}" not in out


def test_group_image_rows_stacks_when_source_stacked():
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "a.jpeg", "page": 3, "bbox": [100, 100, 300, 200]},
            {"file": "b.jpeg", "page": 3, "bbox": [100, 300, 300, 400]}]
    qmd = "![Cap A](m/a.jpeg)\n![Cap B](m/b.jpeg)\n"
    out, n = _group_image_rows(qmd, figs)
    assert n == 1 and "layout-ncol" not in out        # one per row in the source
    assert "![Cap A](m/a.jpeg)\n\n![Cap B](m/b.jpeg)" in out


def test_group_image_rows_unglues_a_lone_image_from_surrounding_text():
    from pdf2md.postfix import _group_image_rows
    # an image glued to the line below is inline too -> caption dropped
    qmd = "![Schematic view of a lagoon.](m/a.jpeg)\nLagoon\nSand bank\n"
    out, n = _group_image_rows(qmd, [])
    assert n == 1
    assert "![Schematic view of a lagoon.](m/a.jpeg)\n\nLagoon" in out
    again, n2 = _group_image_rows(out, [])
    assert n2 == 0 and again == out


def test_group_image_rows_sizes_lone_images_and_is_idempotent():
    """A lone image already in its own block still needs its source width — without it
    Typst stretches it to the full column (see test_lone_image_keeps_its_source_width).
    Everything else about the block is left as it is."""
    from pdf2md.postfix import _group_image_rows
    figs = [{"file": "a.jpeg", "page": 1, "bbox": [0, 0, 10, 10]},
            {"file": "b.jpeg", "page": 1, "bbox": [20, 0, 30, 10]}]
    lone = "Para.\n\n![Only one](m/a.jpeg)\n\nMore text.\n"
    out, n = _group_image_rows(lone, figs)
    assert n == 1 and out == lone.replace("(m/a.jpeg)", "(m/a.jpeg){width=10pt}")
    settled, n_again = _group_image_rows(out, figs)
    assert n_again == 0 and settled == out
    grid, _ = _group_image_rows("![A](m/a.jpeg)\n![B](m/b.jpeg)\n", figs)
    again, n2 = _group_image_rows(grid, figs)
    assert n2 == 0 and again == grid


def test_group_image_rows_without_detections_defaults_to_stacking():
    from pdf2md.postfix import _group_image_rows
    qmd = "![A](m/a.jpeg)\n![B](m/b.jpeg)\n"
    out, n = _group_image_rows(qmd, [])
    assert n == 1 and "layout-ncol" not in out and "![A](m/a.jpeg)\n\n![B](m/b.jpeg)" in out


def test_run_postfix_accepts_repair_model_kwarg(tmp_path):
    # plumbing: run_postfix takes repair_model; with passes=0 it returns early, no LLM
    from pdf2md.postfix import run_postfix
    s = run_postfix(tmp_path / "d.qmd", [], tmp_path, passes=0,
                    repair_model="google/gemini-2.5-flash")
    assert s == {'postfixes_applied': [], 'cost_usd': 0.0}


def test_cli_parser_has_split_model_flags():
    from pdf2md.app_cli import _build_parser, resolve_aux_model
    a = _build_parser().parse_args(
        ["d.pdf", "--figure-model", "google/x", "--repair-model", "google/y"])
    assert a.figure_model == "google/x" and a.repair_model == "google/y"
    # resolve: CLI value wins; unset with no config key -> None (pipeline falls back)
    assert resolve_aux_model("google/x", "figure_model") == "google/x"
    assert resolve_aux_model(None, "no_such_key_zzz") is None
