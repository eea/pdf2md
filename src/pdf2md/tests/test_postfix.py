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


# ── end-of-pipeline chrome strip (--strip-chrome) ───────────────────────────────

import pytest  # noqa: E402

from pdf2md.postfix import strip_chrome_qmd  # noqa: E402

_HEADER = "ACME Corp Annual Report"


def _chrome_source_pdf(out_dir, stem, n_pages=6):
    """Source PDF with a running header, per-page page number, and unique body."""
    fitz = pytest.importorskip("fitz", reason="PyMuPDF required")
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 25), _HEADER, fontsize=9)
        page.insert_text((50, 400),
                         f"Genuine paragraph {i} that appears only on this page.",
                         fontsize=11)
        page.insert_text((290, 830), str(i + 1), fontsize=9)
    path = out_dir / f"{stem}.source.pdf"
    doc.save(str(path))
    doc.close()
    return path


def test_strip_chrome_qmd_drops_headers_and_page_numbers(tmp_path):
    _chrome_source_pdf(tmp_path, "d")
    qmd = tmp_path / "d.qmd"
    qmd.write_text(
        "---\ntitle: T\n---\n\n"
        f"# {_HEADER}\n\n"          # heading: must survive (doc title)
        "Body prose one.\n\n"
        f"{_HEADER}\n\n"            # running header: dropped
        "3\n\n"                     # bare page number: dropped
        "Page 4\n\n"                # page-number variant: dropped
        "```\n"
        f"{_HEADER}\n42\n"          # fence interior: untouched
        "```\n\n"
        "Body prose two.\n",
        encoding="utf-8")
    removed = strip_chrome_qmd(qmd, tmp_path)
    assert removed == 3
    text = qmd.read_text(encoding="utf-8")
    lines = text.split("\n")
    assert f"# {_HEADER}" in lines                # heading kept
    assert lines.count(_HEADER) == 1              # only the fence-interior copy left
    assert "3" not in lines and "Page 4" not in lines
    assert "42" in lines                          # fence interior kept
    assert "Body prose one." in lines and "Body prose two." in lines
    assert "title: T" in lines                    # frontmatter untouched
    assert "\n\n\n" not in text                   # removals leave no blank runs


def test_strip_chrome_qmd_no_chrome_evidence_is_a_no_op(tmp_path):
    # Two pages of unique prose: no running chrome, so even a bare-digit line
    # in the .qmd must be presumed content and kept.
    fitz = pytest.importorskip("fitz", reason="PyMuPDF required")
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 400), f"Only page {i} has this sentence.", fontsize=11)
    doc.save(str(tmp_path / "d.source.pdf"))
    doc.close()
    qmd = tmp_path / "d.qmd"
    before = "Prose.\n\n7\n\nMore prose.\n"
    qmd.write_text(before, encoding="utf-8")
    assert strip_chrome_qmd(qmd, tmp_path) == 0
    assert qmd.read_text(encoding="utf-8") == before


def test_strip_chrome_qmd_without_source_pdf_is_a_no_op(tmp_path):
    qmd = tmp_path / "d.qmd"
    qmd.write_text("Prose.\n", encoding="utf-8")
    assert strip_chrome_qmd(qmd, tmp_path) == 0


# ── iterative repair loop ───────────────────────────────────────────────────────

from pdf2md import postfix as pf  # noqa: E402
from pdf2md.postfix import run_repair_loop  # noqa: E402
from pdf2md.verify import CheckResult  # noqa: E402


def _tc(metric, status="warn"):
    """A minimal verify-results list carrying just text_coverage."""
    return [CheckResult("text_coverage", status, "s", metric=metric)]


def _quiet_passes(monkeypatch, det=None, verify_seq=None):
    """Stub the three fix steps and re-verify. `det` is a callable returning the
    deterministic pass result; `verify_seq` a list of results consumed in order."""
    monkeypatch.setattr(pf, "_deterministic_pass",
                        det or (lambda *a: ([], 0.0)))
    monkeypatch.setattr(pf, "_llm_text_pass", lambda *a: ([], 0.0, 0))
    monkeypatch.setattr(pf, "_vision_patch_pass", lambda *a: ([], 0.0))
    if verify_seq is not None:
        seq = list(verify_seq)
        monkeypatch.setattr(pf, "_verify_now", lambda *a: seq.pop(0))


def _qmd(tmp_path):
    q = tmp_path / "doc.qmd"
    q.write_text("body\n", encoding="utf-8")
    return q


def test_repair_loop_iterates_while_coverage_improves(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch,
                  det=lambda *a: (["fix"], 0.0),
                  verify_seq=[_tc(92.0), _tc(94.0), _tc(96.0)])
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=3)
    assert len(s["iterations"]) == 3
    assert "max iterations" in s["stop_reason"]
    covs = [(i["text_cov_before"], i["text_cov_after"]) for i in s["iterations"]]
    assert covs == [(90.0, 92.0), (92.0, 94.0), (94.0, 96.0)]
    # every applied fix is recorded with its iteration
    assert s["postfixes_applied"] == ["[iter 1] fix", "[iter 2] fix", "[iter 3] fix"]
    assert s["verify_after"] == "warn"
    assert s["coverage_after"]["text"] == 96.0


def test_repair_loop_stops_when_verify_is_clean(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch,
                  det=lambda *a: (["fix"], 0.0),
                  verify_seq=[_tc(100.0, status="ok")])
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=3)
    assert len(s["iterations"]) == 1
    assert "verify ok" in s["stop_reason"]
    assert s["verify_after"] == "ok"


def test_repair_loop_stops_without_improvement(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch,
                  det=lambda *a: (["fix"], 0.0),
                  verify_seq=[_tc(90.0)])       # same coverage as before
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=3)
    assert len(s["iterations"]) == 1
    assert "no coverage improvement" in s["stop_reason"]


def test_repair_loop_stops_when_nothing_to_fix(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch)                  # every step returns no fixes
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=3)
    assert len(s["iterations"]) == 1
    assert "applied no fixes" in s["stop_reason"]
    assert s["postfixes_applied"] == []
    assert s["cost_usd"] == 0.0
    assert "verify_after" not in s              # never re-verified


def test_repair_loop_zero_iterations_is_a_no_op(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch)
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=0)
    assert s["iterations"] == [] and s["postfixes_applied"] == []
    assert not (tmp_path / "repair_report.md").exists()


def test_repair_loop_accumulates_cost_and_items(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_deterministic_pass", lambda *a: (["det"], 0.01))
    monkeypatch.setattr(pf, "_llm_text_pass",
                        lambda *a: (["missing_text: 2 items"], 0.02, 2))
    monkeypatch.setattr(pf, "_vision_patch_pass",
                        lambda *a: (["vision: 1 patch"], 0.03))
    monkeypatch.setattr(pf, "_verify_now", lambda *a: _tc(91.0))
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=1)
    assert abs(s["cost_usd"] - 0.06) < 1e-9
    assert s["items_recovered"] == 2
    assert s["iterations"][0]["fixes"] == ["det", "missing_text: 2 items",
                                           "vision: 1 patch"]


def test_repair_loop_writes_repair_report(tmp_path, monkeypatch):
    _quiet_passes(monkeypatch,
                  det=lambda *a: (["header_bleed: stripped 2"], 0.0),
                  verify_seq=[_tc(95.0), _tc(95.0)])   # 2nd iter: no improvement
    s = run_repair_loop(_qmd(tmp_path), tmp_path, _tc(90.0), "key",
                        max_iterations=2)
    report = (tmp_path / "repair_report.md").read_text(encoding="utf-8")
    assert s["report"] == str(tmp_path / "repair_report.md")
    assert "| Iteration |" in report            # the improvement table
    assert "header_bleed: stripped 2" in report
    assert "90.0%" in report and "95.0%" in report
    assert "Stopped:" in report
