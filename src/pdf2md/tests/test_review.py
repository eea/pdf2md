"""Vision patch step (review.py): flagged-page identification, prompts, and
patch application. Orchestration now lives in postfix.run_repair_loop (tested
in test_postfix.py).

No network and no fitz needed — the LLM calls are exercised only through the
pure helpers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md.review import (  # noqa: E402
    _apply_patches, _build_prompt, _flagged_pages, inspect_pages,
)
from pdf2md.verify import CheckResult, Finding  # noqa: E402


def _cr(name, status, findings=None, detail=None):
    return CheckResult(name=name, status=status, summary="s",
                       findings=findings or [], detail=detail)


# ── _flagged_pages ──────────────────────────────────────────────────────────────

def test_flagged_pages_collects_pages_from_clusters():
    detail = {"clusters": [{"pages": [3, 5]}, {"pages": [7, 7]}]}
    r = _cr("text_coverage", "warn", detail=detail)
    assert _flagged_pages([r]) == {3: ["text_coverage"], 4: ["text_coverage"],
                                    5: ["text_coverage"], 7: ["text_coverage"]}


def test_flagged_pages_collects_pages_from_thin_tables():
    detail = {"thin": [{"page": 2}, {"page": 5}]}
    r = _cr("table_coverage", "warn", detail=detail)
    assert _flagged_pages([r]) == {2: ["table_coverage"], 5: ["table_coverage"]}


def test_flagged_pages_collects_src_pages():
    detail = {"src_pages": [1, 10]}
    r = _cr("text_coverage", "warn", detail=detail)
    assert _flagged_pages([r]) == {1: ["text_coverage"], 10: ["text_coverage"]}


def test_flagged_pages_parses_finding_location():
    f = Finding(message="", location="table 3, p8")
    r = _cr("table_coverage", "warn", findings=[f])
    assert _flagged_pages([r]) == {8: ["table_coverage"]}


def test_flagged_pages_ignores_ok_checks():
    detail = {"src_pages": [5]}
    ok_r = _cr("text_coverage", "ok", detail=detail)
    assert _flagged_pages([ok_r]) == {}


def test_flagged_pages_returns_empty_for_no_pages():
    r = _cr("math_presence", "warn")
    assert _flagged_pages([r]) == {}


def test_flagged_pages_merges_checks_on_same_page():
    d1 = {"src_pages": [2]}
    d2 = {"thin": [{"page": 2}]}
    a = _cr("text_coverage", "warn", detail=d1)
    b = _cr("table_coverage", "warn", detail=d2)
    f = Finding(message="", location="p2")
    c = _cr("link_preservation", "warn", findings=[f])
    pages = _flagged_pages([a, b, c])
    assert set(pages[2]) == {"text_coverage", "table_coverage", "link_preservation"}


# ── prompts ─────────────────────────────────────────────────────────────────────

def test_prompt_names_page_issues_and_qmd():
    text = _build_prompt(5, ["text_coverage", "footnote_placement"],
                         "SOME QMD BODY")
    assert "page 5" in text
    assert "text_coverage" in text
    assert "footnote_placement" in text
    assert "SOME QMD BODY" in text


def test_prompt_asks_for_patches():
    text = _build_prompt(1, ["table_coverage"], "qmd")
    assert "patches" in text
    assert "old_string" in text
    assert "new_string" in text
    assert "false_positives" in text


def test_prompt_notes_missing_render():
    text = _build_prompt(1, ["table_coverage"], "qmd", has_rendered=False)
    assert "unavailable" in text


# ── inspect_pages ───────────────────────────────────────────────────────────────

def test_inspect_pages_without_source_pdf_returns_nothing(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("body\n", encoding="utf-8")
    out = inspect_pages(tmp_path / "doc.source.pdf", None,
                        {1: ["text_coverage"]}, "key", qmd_path=qmd)
    assert out == []


# ── _apply_patches ──────────────────────────────────────────────────────────────

def _defect(page, *patches):
    return {"page": page, "issues": ["text_coverage"], "verdict": "issues",
            "false_positives": [],
            "patches": [dict(severity="moderate", check="text_coverage",
                             location="", description="d",
                             old_string=o, new_string=n,
                             applied=False, apply_note="")
                        for o, n in patches],
            "cost_usd": 0.0}


def test_apply_patches_replaces_unique_match(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("alpha beta gamma\n", encoding="utf-8")
    defects = [_defect(1, ("beta", "BETA"))]
    assert _apply_patches(qmd, defects) == 1
    assert qmd.read_text() == "alpha BETA gamma\n"
    assert defects[0]["patches"][0]["applied"] is True


def test_apply_patches_supports_deletion(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("keep DROP keep\n", encoding="utf-8")
    assert _apply_patches(qmd, [_defect(1, ("DROP ", ""))]) == 1
    assert qmd.read_text() == "keep keep\n"


def test_apply_patches_skips_missing_old_string(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("alpha beta\n", encoding="utf-8")
    defects = [_defect(1, ("nope", "x"))]
    assert _apply_patches(qmd, defects) == 0
    assert qmd.read_text() == "alpha beta\n"
    assert "not found" in defects[0]["patches"][0]["apply_note"]


def test_apply_patches_skips_ambiguous_old_string(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("dup text dup text\n", encoding="utf-8")
    defects = [_defect(1, ("dup text", "x"))]
    assert _apply_patches(qmd, defects) == 0
    assert "ambiguous" in defects[0]["patches"][0]["apply_note"]


def test_apply_patches_counts_across_pages(tmp_path):
    qmd = tmp_path / "doc.qmd"
    qmd.write_text("one two three\n", encoding="utf-8")
    defects = [_defect(1, ("one", "1")), _defect(2, ("three", "3"))]
    assert _apply_patches(qmd, defects) == 2
    assert qmd.read_text() == "1 two 3\n"
