"""Model-refusal detection: recitation/content-filter blocks and silent token burn."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md import llm_client  # noqa: E402


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _fake_post(payload, calls):
    def post(url, headers=None, json=None, timeout=None, stream=False):
        calls.append(1)
        return _Resp(payload)
    return post


def test_recitation_finish_fails_fast_without_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client.requests, "post", _fake_post({
        "choices": [{"message": {"content": ""}, "finish_reason": None,
                     "native_finish_reason": "RECITATION"}],
        "usage": {"completion_tokens": 1200},
    }, calls))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="guardrails"):
        llm_client._post_with_retries(api_key="k", payload={"messages": []},
                                      label="doc.pdf", timeout=5)
    assert len(calls) == 1          # a refusal must not be retried


def test_empty_with_token_burn_mentions_guardrails(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client.requests, "post", _fake_post({
        "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        "usage": {"completion_tokens": 2968},
    }, calls))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as e:
        llm_client._post_with_retries(api_key="k", payload={"messages": []},
                                      label="doc.pdf", timeout=5)
    msg = str(e.value)
    assert "2968" in msg and "guardrails" in msg
    assert len(calls) == 1          # a deterministic budget burn is not retried


def test_plain_empty_keeps_old_message(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_client.requests, "post", _fake_post({
        "choices": [{"message": {"content": ""}, "finish_reason": None}],
        "usage": {},
    }, calls))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError) as e:
        llm_client._post_with_retries(api_key="k", payload={"messages": []},
                                      label="doc.pdf", timeout=5)
    assert "guardrails" not in str(e.value)


# ── pre-flight model fit check ──────────────────────────────────────────────────

from pdf2md.llm_client import check_model_fit  # noqa: E402

_LIMITS = {"_meta": {
    "text-only/model": {"modalities": ["text"], "context_length": 200_000},
    "small-ctx/model": {"modalities": ["text", "file"], "context_length": 131_072},
    "google/gemini-2.5-flash": {"modalities": ["file", "image", "text"],
                                "context_length": 1_048_576},
}}


def _levels(notes):
    return [n["level"] for n in notes]


def test_fit_rejects_model_without_file_input():
    notes = check_model_fit("text-only/model", pages=10, limits=_LIMITS)
    assert "error" in _levels(notes)
    assert "cannot read PDFs" in notes[0]["msg"]


def test_fit_rejects_doc_too_long_for_context():
    # 300 pages * ~800 tok ≈ 240k > 131k window
    notes = check_model_fit("small-ctx/model", pages=300, limits=_LIMITS)
    assert any(n["level"] == "error" and "context window" in n["msg"] for n in notes)


def test_fit_happy_path_short_doc_is_silent():
    assert check_model_fit("google/gemini-2.5-flash", pages=30, limits=_LIMITS) == []


def test_fit_warns_anthropic_over_100_pages():
    notes = check_model_fit("anthropic/claude-sonnet-4.5", pages=131, limits=_LIMITS)
    assert any(n["level"] == "warn" and "100 pages" in n["msg"] for n in notes)


def test_fit_long_doc_gets_guardrail_info_note():
    notes = check_model_fit("google/gemini-2.5-flash", pages=131, limits=_LIMITS)
    assert any(n["level"] == "info" and "guardrails" in n["msg"] for n in notes)


def test_fit_unknown_model_only_generic_notes():
    # no metadata -> no capability claims, just the page-count note when relevant
    assert check_model_fit("someone/new-model", pages=30, limits=_LIMITS) == []
    notes = check_model_fit("someone/new-model", pages=150, limits=_LIMITS)
    assert _levels(notes) == ["info"]


# ── API-key diagnosis (what the 'no key' failure tells the user) ────────────────

from pdf2md.app_cli import describe_key_sources  # noqa: E402


def test_key_diag_env_not_exported(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_FILE", raising=False)
    assert "not exported in this shell" in describe_key_sources()


def test_key_diag_malformed_env_value(monkeypatch):
    # a redacted or mispasted value is worse than none: it silently resolves to no key
    monkeypatch.setenv("OPENROUTER_API_KEY", "***REDACTED***")
    assert "doesn't look like an OpenRouter key" in describe_key_sources()


def test_key_diag_key_file_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(tmp_path / "nope.txt"))
    assert "missing file" in describe_key_sources()


def test_key_diag_valid_env_is_quiet(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc123")
    d = describe_key_sources()
    assert "not exported" not in d and "doesn't look like" not in d
