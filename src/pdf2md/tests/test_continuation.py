"""Tests for long-output continuation (llm_client).

A document whose Markdown exceeds one response's token cap is delivered across
continuation calls: the whole PDF stays in context every call, the output-so-far
is fed back, and seams are trimmed to a clean block boundary so a table/fence is
never split. See llm_client continuation helpers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md import llm_client  # noqa: E402
from pdf2md.llm_client import (  # noqa: E402
    _dedup_seam,
    _trim_to_block_boundary,
    call_openrouter,
)


# ── seam trimming ──────────────────────────────────────────────────────────────

def test_trim_clean_prose_boundary():
    text = "# Title\n\nPara one is complete.\n\nPara two is half writ"
    safe, carry = _trim_to_block_boundary(text)
    assert safe == "# Title\n\nPara one is complete.\n\n"
    assert carry == "Para two is half writ"


def test_trim_never_cuts_inside_open_fence():
    # blank line falls *inside* an unterminated code fence — must roll back past it
    text = "Intro para.\n\n```python\ncode line 1\n\ncode line 2 unfinished"
    safe, carry = _trim_to_block_boundary(text)
    assert safe == "Intro para.\n\n"
    assert "```" not in safe          # the open fence stays in carry, regenerated whole


def test_trim_never_cuts_inside_open_html_table():
    text = ("Before.\n\n<table>\n<tr><td>a</td></tr>\n\n<tr><td>b half")
    safe, carry = _trim_to_block_boundary(text)
    assert safe == "Before.\n\n"
    assert "<table>" not in safe


def test_trim_closed_table_is_a_valid_boundary():
    text = ("<table>\n<tr><td>a</td></tr>\n</table>\n\nnext block started")
    safe, carry = _trim_to_block_boundary(text)
    assert safe.endswith("</table>\n\n")
    assert carry == "next block started"


def test_trim_no_boundary_returns_whole():
    text = "one giant unbroken block with no blank line at all and no fences"
    safe, carry = _trim_to_block_boundary(text)
    assert safe == text and carry == ""


# ── seam dedup ─────────────────────────────────────────────────────────────────

def test_dedup_strips_repeated_tail():
    acc = "... end of section about hydrology and soil moisture content here."
    cont = "soil moisture content here. Then the new paragraph continues onward."
    out = _dedup_seam(acc, cont)
    assert out == " Then the new paragraph continues onward."


def test_dedup_leaves_clean_continuation_untouched():
    acc = "First installment ends cleanly.\n\n"
    cont = "## Second Section\n\nBrand new content."
    assert _dedup_seam(acc, cont) == cont


# ── end-to-end continuation loop ───────────────────────────────────────────────

def _fake_poster(segments):
    """Return a _post_with_retries stand-in that yields queued (content, finish)
    pairs and records each call's messages so we can assert the PDF is resent."""
    calls = []

    def fake(*, api_key, payload, label, timeout, stream=False, on_delta=None,
             allow_truncation=False):
        calls.append(payload["messages"])
        content, finish = segments[len(calls) - 1]
        return content, {"cost": 0.10, "completion_tokens": 100}, finish

    return fake, calls


def test_two_truncations_then_stop_concatenates(monkeypatch):
    # segment 1 truncates mid second block; the partial block is discarded and
    # regenerated whole in segment 2; segment 3 finishes.
    segments = [
        ("# Doc\n\nBlock A done.\n\nBlock B was cut halfw", "length"),
        ("Block B in full now.\n\nBlock C done.\n\nBlock D cut", "length"),
        ("Block D in full now.\n", "stop"),
    ]
    fake, calls = _fake_poster(segments)
    monkeypatch.setattr(llm_client, "_post_with_retries", fake)

    text, usage = call_openrouter(
        api_key="k", model="google/gemini-2.5-pro", engine="native",
        system_instruction="sys", user_prompt="convert",
        file_data="data:application/pdf;base64,AAAA", filename="doc.pdf",
        return_usage=True,
    )

    # every block present, the truncated half-blocks replaced by their full form
    for block in ("Block A done.", "Block B in full now.", "Block C done.",
                  "Block D in full now."):
        assert block in text
    assert "halfw" not in text and "Block D cut" not in text
    # cost accrues across all three calls
    assert abs(usage["cost"] - 0.30) < 1e-9
    # the PDF (file part) is resent on every call, and continuations add a prefill
    assert len(calls) == 3
    for msgs in calls:
        assert any(part.get("type") == "file"
                   for m in msgs if isinstance(m["content"], list)
                   for part in m["content"])
    assert calls[1][-2]["role"] == "assistant"   # output-so-far fed back
    assert calls[1][-1]["role"] == "user"        # + continue instruction


def test_single_pass_stop_makes_one_call(monkeypatch):
    fake, calls = _fake_poster([("# Short doc\n\nAll done.\n", "stop")])
    monkeypatch.setattr(llm_client, "_post_with_retries", fake)

    text = call_openrouter(
        api_key="k", model="google/gemini-2.5-pro", engine="native",
        system_instruction="sys", user_prompt="convert",
        file_data="data:application/pdf;base64,AAAA", filename="doc.pdf",
    )
    assert text == "# Short doc\n\nAll done.\n"
    assert len(calls) == 1


def test_cap_keeps_partial_and_does_not_raise(monkeypatch):
    # model never stops — loop must hit the cap, keep what it has, not raise
    fake, calls = _fake_poster([("chunk %d.\n\nmore" % i, "length")
                                for i in range(llm_client._MAX_CONTINUATIONS)])
    monkeypatch.setattr(llm_client, "_post_with_retries", fake)

    text = call_openrouter(
        api_key="k", model="google/gemini-2.5-pro", engine="native",
        system_instruction="sys", user_prompt="convert",
        file_data="data:application/pdf;base64,AAAA", filename="doc.pdf",
    )
    assert len(calls) == llm_client._MAX_CONTINUATIONS
    assert "chunk 0." in text
