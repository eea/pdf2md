"""Tests for the --template frontmatter-injection feature.

These import only prompt.py and resolve.py, which are free of PEP 604 union
syntax, so they run on Python 3.9 too. The pass2/app-level behavior (gating to
qmd, _find_quarto) is covered in test_find_quarto.py, guarded to 3.10+.
"""

import logging

from pdf2md.prompt import (
    extract_template_frontmatter,
    inject_template_frontmatter,
    parse_prompt_file,
)
from pdf2md.resolve import normalize_frontmatter

QMD_PROMPT = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "prompt_templates" / "convert_prompt_qmd.md"
)


def _write_template(tmp_path, body="title: \"T\"\nauthor: \"A\"\nyear: 2020"):
    p = tmp_path / "tmpl.qmd"
    p.write_text(f"---\n{body}\n---\n\n# Body ignored\nsome prose\n", encoding="utf-8")
    return p


# ── extract_template_frontmatter ────────────────────────────────────────────

def test_extract_reads_yaml_body(tmp_path):
    p = _write_template(tmp_path)
    body = extract_template_frontmatter(p)
    assert 'author: "A"' in body
    assert "# Body ignored" not in body  # body dropped, only the YAML block


def test_extract_missing_file_returns_empty(tmp_path):
    assert extract_template_frontmatter(tmp_path / "nope.qmd") == ""


def test_extract_no_frontmatter_returns_empty(tmp_path):
    p = tmp_path / "plain.qmd"
    p.write_text("# Just a heading, no YAML\n", encoding="utf-8")
    assert extract_template_frontmatter(p) == ""


# ── inject_template_frontmatter against the real prompt ─────────────────────

def test_inject_replaces_frontmatter_and_keeps_output_section(tmp_path):
    system_instruction, _ = parse_prompt_file(QMD_PROMPT)
    assert "FRONTMATTER:" in system_instruction
    assert "OUTPUT:" in system_instruction

    p = _write_template(tmp_path)
    out = inject_template_frontmatter(system_instruction, p)

    # template's keys are now in the prompt
    assert 'author: "A"' in out
    assert "year: 2020" in out
    # the OUTPUT: section survived (regression: the old \Z fallback could eat it)
    assert "OUTPUT:" in out
    # the original default-frontmatter guidance was replaced
    assert "containing exactly these keys" not in out


def test_inject_none_and_empty_are_noops(tmp_path):
    system_instruction, _ = parse_prompt_file(QMD_PROMPT)
    assert inject_template_frontmatter(system_instruction, None) == system_instruction

    empty = tmp_path / "empty.qmd"
    empty.write_text("# no yaml here\n", encoding="utf-8")
    assert inject_template_frontmatter(system_instruction, empty) == system_instruction


def test_inject_safe_fail_when_no_following_header(tmp_path):
    """If FRONTMATTER: isn't followed by an ALL-CAPS section header, make NO
    change rather than swallowing the rest of the prompt to end-of-string."""
    si = "FRONTMATTER:\nfill in the values.\n\nthen just prose, no header follows."
    p = _write_template(tmp_path)
    assert inject_template_frontmatter(si, p) == si


# ── normalize_frontmatter(keep_template_fields=True) ────────────────────────

def test_keep_template_fields_preserves_existing_frontmatter():
    text = '---\ntitle: "T"\nauthor: "A"\n---\n\nbody\n'
    out = normalize_frontmatter(text, keep_template_fields=True)
    assert out == text  # untouched: no forced category/date


def test_keep_template_fields_warns_and_prepends_when_missing(caplog):
    with caplog.at_level(logging.WARNING):
        out = normalize_frontmatter("just body, no frontmatter\n",
                                    keep_template_fields=True)
    assert out.startswith("---\n---\n\n")
    assert any("no frontmatter" in r.message for r in caplog.records)
