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
    _build_continue_message,
    _dedup_seam,
    _heading_outline,
    _output_tail,
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
    # the PDF (file part) is resent on every call
    assert len(calls) == 3
    for msgs in calls:
        assert any(part.get("type") == "file"
                   for m in msgs if isinstance(m["content"], list)
                   for part in m["content"])
    # a continuation adds exactly ONE user turn (tail + anchor), no assistant prefill
    cont = calls[1][-1]
    assert cont["role"] == "user"
    assert not any(m["role"] == "assistant" for m in calls[1])
    assert "Block A done." in cont["content"]        # the tail is fed back


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


# ── progress-map anchor (heading outline + page count) ──────────────────────────

def test_heading_outline_lists_all_headings_in_order():
    out = _heading_outline("# One\n\ntext\n\n## 7.3 Two\n\nmore\n\n### Deep\n\nx")
    assert out == "# One\n## 7.3 Two\n### Deep"


def test_heading_outline_empty_when_none():
    assert _heading_outline("plain prose, no structure at all") == ""


def test_heading_outline_collapses_when_huge():
    body = "".join(f"## Section {i}\n\ntext\n\n" for i in range(300))
    out = _heading_outline(body, max_lines=200)
    assert "earlier sections omitted" in out
    assert "## Section 299" in out          # the recent tail is kept
    assert out.count("\n") <= 200


def test_continue_message_carries_progress_and_page_count():
    msg = _build_continue_message("# Intro\n## Methods", "…tail text…", total_pages=131)
    assert "131 pages" in msg                       # page-count anchor
    assert "## Methods" in msg                        # progress outline
    assert "…tail text…" in msg                       # exact resume anchor
    assert "NOT finished" in msg or "not conclude" in msg.lower()  # anti-early-stop


# ── bounded tail ────────────────────────────────────────────────────────────────

def test_output_tail_bounded_and_block_aligned():
    text = "HEAD\n\n" + "a" * 10000 + "\n\ntail block here"
    tail = _output_tail(text, max_chars=500)
    assert len(tail) <= 500
    assert tail == "tail block here"   # advanced past the mid-block cut to a clean boundary


def test_output_tail_returns_whole_when_short():
    assert _output_tail("short doc", max_chars=8000) == "short doc"


def test_continuation_input_stays_bounded(monkeypatch):
    # THE point of tail-only: as total output grows across many chunks, the fed-back
    # continuation message must NOT grow with it (else we're back to the context wall).
    segments = [("blk%d\n\n" % i + "z" * 5000 + "\n\nmore", "length") for i in range(6)]
    segments.append(("final block.\n", "stop"))
    fake, calls = _fake_poster(segments)
    monkeypatch.setattr(llm_client, "_post_with_retries", fake)

    call_openrouter(
        api_key="k", model="google/gemini-2.5-pro", engine="native",
        system_instruction="sys", user_prompt="convert",
        file_data="data:application/pdf;base64,AAAA", filename="doc.pdf",
    )
    cont_sizes = [len(msgs[-1]["content"]) for msgs in calls[1:]]
    assert cont_sizes
    # every continuation message is bounded (~tail + fixed prompt overhead)…
    assert max(cont_sizes) < llm_client._TAIL_CHARS + 2000
    # …and does not creep upward as accumulated output balloons
    assert max(cont_sizes) - min(cont_sizes) < 2000


# ── runaway repetition guard ─────────────────────────────────────────────────────

from pdf2md.llm_client import _excise_loops, _loop_cut  # noqa: E402


def test_loop_cut_finds_active_line_loop():
    head = "# Doc\n\nReal paragraph one here.\n\nReal paragraph two here.\n\n"
    loop = "\n".join(["<td>-</td>"] * 300)
    cut = _loop_cut(head + loop)
    assert cut is not None
    # keeps at most one unit past the loop start
    assert len(head) <= cut <= len(head) + len("<td>-</td>\n") + 1


def test_loop_cut_finds_single_line_dash_loop():
    # regression: a real runaway was one giant line of dashes, no newlines at all
    head = "intro paragraph text.\n"
    cut = _loop_cut(head + "---|" * 2000)
    assert cut is not None and cut <= len(head) + 8


def test_loop_cut_none_on_normal_text():
    text = "# Doc\n\n" + "\n\n".join(f"Distinct paragraph number {i}." for i in range(120))
    assert _loop_cut(text) is None


def test_loop_cut_tolerates_legitimate_small_repeats():
    text = ("x" * 900) + "\n\n" + "\n".join(["<td>-</td>"] * 8) + "\n\nafter " + "y" * 900
    assert _loop_cut(text) is None


def test_excise_removes_recovered_midfile_loop_keeps_tail():
    # a loop that self-terminated: run of identical lines followed by good content
    head = "before content.\n"
    loop = "\n".join(["<td>-</td>"] * 200) + "\n"
    tail = "after content that must survive.\n"
    cleaned, removed = _excise_loops(head + loop + tail)
    assert removed > 0
    assert "after content that must survive." in cleaned
    assert cleaned.count("<td>-</td>") == 1          # one unit kept


def test_excise_shrinks_intra_line_loop():
    ln = "start " + "---|" * 5000
    cleaned, removed = _excise_loops("a\n" + ln + "\nb")
    assert removed > 10000
    assert "a\n" in cleaned and "\nb" in cleaned


def test_excise_leaves_normal_text_untouched():
    text = "# Doc\n\n" + "\n\n".join(f"Paragraph {i} with content." for i in range(50))
    cleaned, removed = _excise_loops(text)
    assert removed == 0 and cleaned == text
