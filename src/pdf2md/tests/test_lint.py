#!/usr/bin/env python3
"""Tests for the emit-time lint gate (lint.py) — deterministic, no LLM.

Fixtures mirror the structural hazards observed in the MRVPP ATBD conversion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md import lint  # noqa: E402

FRONT = "---\ntitle: T\nsubtitle: S\ncategory: products\ndate: '2020-01-01'\n---\n\n"


def _write(tmp_path, body, *, media=None):
    """Write a .qmd (with frontmatter) and optional media files; return its path."""
    d = tmp_path / "doc"
    (d / "doc-media").mkdir(parents=True)
    for name in media or []:
        (d / "doc-media" / name).write_bytes(b"\x89PNG\r\n")
    p = d / "doc.qmd"
    p.write_text(FRONT + body, encoding="utf-8")
    return p


# --- sanitizers -----------------------------------------------------------

def test_unwraps_nested_image():
    text = "![](![A caption](doc-media/x.png))\n"
    out, fixes = lint.sanitize(text)
    assert out == "![A caption](doc-media/x.png)\n"
    assert [i.code for i in fixes] == ["nested_image"]


def test_strips_page_footers_even_inside_fence():
    text = "```bash\necho hi\nVersion 5.0, Issue 2.0 Page 26 of 46\n```\n\nPage 1 of 46\n"
    out, fixes = lint.sanitize(text)
    assert "Page 26 of 46" not in out
    assert "Page 1 of 46" not in out
    assert "echo hi" in out
    assert fixes and fixes[0].code == "page_footer" and fixes[0].count == 2


def test_fixes_doubled_url_scheme():
    text = "See [DOI](https://doi.org/https:/doi.org/10.1016/j.rse.2020.111685).\n"
    out, _ = lint.sanitize(text)
    assert "https://doi.org/10.1016/j.rse.2020.111685" in out
    assert "https:/doi.org" not in out.replace("https://doi.org", "")


def test_removes_stray_fence_after_frontmatter():
    p_body = "```\n\nRegular paragraph that should survive.\n"
    text = FRONT + p_body
    out, fixes = lint.sanitize(text)
    assert [i.code for i in fixes] == ["stray_fence"]
    assert "Regular paragraph that should survive." in out
    assert out.count("```") == 0


def test_leaves_balanced_language_fence_alone():
    text = FRONT + "```python\nprint(1)\n```\n"
    out, fixes = lint.sanitize(text)
    assert out == text
    assert fixes == []


# --- gate (check) ---------------------------------------------------------

def test_missing_media_is_a_failure(tmp_path):
    p = _write(tmp_path, "![ok](doc-media/present.png)\n\n![gone](doc-media/absent.png)\n",
               media=["present.png"])
    res = lint.lint_qmd(p)
    assert not res.ok
    codes = [i.code for i in res.failures]
    assert "missing_media" in codes


def test_present_media_passes(tmp_path):
    p = _write(tmp_path, "![ok](doc-media/present.png)\n", media=["present.png"])
    res = lint.lint_qmd(p)
    assert res.ok
    assert res.failures == []


def test_unbalanced_code_fence_fails(tmp_path):
    # a language-tagged opener with no close: sanitize won't touch it, gate must fail
    p = _write(tmp_path, "text\n\n```python\nprint(1)\n")
    res = lint.lint_qmd(p)
    assert not res.ok
    assert any(i.code == "unbalanced_code_fence" for i in res.failures)


def test_media_ref_inside_code_is_ignored(tmp_path):
    # an image path shown inside a code sample is not a real reference
    p = _write(tmp_path, "```md\n![x](doc-media/not-real.png)\n```\n")
    res = lint.lint_qmd(p)
    assert res.ok


def test_lint_qmd_writes_sanitized_output(tmp_path):
    p = _write(tmp_path, "![](![C](doc-media/x.png))\n", media=["x.png"])
    res = lint.lint_qmd(p)
    assert res.changed
    assert p.read_text(encoding="utf-8").rstrip().endswith("![C](doc-media/x.png)")
    assert res.ok
