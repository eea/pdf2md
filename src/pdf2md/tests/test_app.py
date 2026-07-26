#!/usr/bin/env python3
"""Tests for the production orchestrator (app.py) — mocked LLM/quarto."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md import app  # noqa: E402
from pdf2md.verify import CheckResult  # noqa: E402


def _make_pdf(path: Path):
    import fitz
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    doc.save(str(path))
    doc.close()
    return path


def _stub_phases(monkeypatch, *, verify_status="ok", phase2_raises=False, render_ok=True):
    """Replace the LLM/quarto/verify steps with fast local stubs."""
    def fake_phase1(pdf, out_dir, **kw):
        return {"figures": 2, "cost_usd": {"cover": 0.01, "detect": 0.20},
                "cover": {"is_cover": True,
                          "fields": {"title": "T", "subtitle": "S", "date": "2020-01-01", "version": "v1"}}}

    def fake_phase2(out_dir, **kw):
        if phase2_raises:
            raise RuntimeError("convert blew up")
        (out_dir / f"{out_dir.name}.qmd").write_text("---\ntitle: T\n---\nbody\n")
        return {"cost_usd": 1.00}

    def fake_render(out_dir, stem):
        if render_ok:
            (out_dir / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
        return render_ok, "render log"

    def fake_verify(out_dir, stem, meta=None):
        return [
            CheckResult("figure_placement", "ok", "2/2"),
            CheckResult("text_coverage", verify_status, "coverage", metric=97.3),
            CheckResult("table_coverage", "ok", "tables", metric=99.0),
        ]

    monkeypatch.setattr(app, "run_phase1", fake_phase1)
    monkeypatch.setattr(app, "run_phase2", fake_phase2)
    monkeypatch.setattr(app, "_render", fake_render)
    monkeypatch.setattr(app, "_run_verify", fake_verify)
    monkeypatch.setattr(app, "_ensure_scaffolding", lambda out_root: out_root.mkdir(parents=True, exist_ok=True))


class TestConvertOne:
    def test_happy_path(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k", do_render=True)
        assert r.status == "ok"
        assert r.figures == 2
        assert r.qmd.name == "doc.qmd"
        assert r.pdf_out and r.pdf_out.name == "doc.pdf"
        assert r.verify_status == "ok"
        assert r.cover["title"] == "T"
        # cost accumulated across phases (cover 0.01 + detect 0.20 + convert 1.00)
        assert abs(r.cost_usd - 1.21) < 1e-9
        assert r.phase_cost == {"cover": 0.01, "detect": 0.20, "convert": 1.00,
                                "rescue": 0.0, "repair": 0.0}

    def test_verify_warn_sets_warn(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch, verify_status="warn")
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k")
        assert r.status == "warn" and r.verify_status == "warn"

    def test_phase2_failure_is_captured_not_raised(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch, phase2_raises=True)
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k")
        assert r.status == "fail" and "blew up" in r.error

    def test_render_failure_is_warn(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch, render_ok=False)
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k", do_render=True)
        assert r.status == "warn" and "render failed" in r.error

    def test_no_render_no_verify_honored(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        called = {"render": False, "verify": False}
        monkeypatch.setattr(app, "_render", lambda *a: called.__setitem__("render", True) or (True, ""))
        monkeypatch.setattr(app, "_run_verify", lambda *a: called.__setitem__("verify", True) or [])
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k", do_render=False, do_verify=False)
        assert r.status == "ok"
        assert called["render"] is False and called["verify"] is False
        assert r.pdf_out is None and r.verify_status == ""

    def test_skip_if_exists(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        out_root = tmp_path / "out"
        (out_root / "doc").mkdir(parents=True)
        (out_root / "doc" / "doc.qmd").write_text("existing")
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, out_root, api_key="k")
        assert r.status == "ok" and r.resumed is True and "skipped" in r.error

    def test_resume_only_processes_unconverted(self, tmp_path, monkeypatch):
        # simulate a cancelled batch: one doc already has its .qmd, one doesn't
        _stub_phases(monkeypatch)
        converted = {"n": 0}
        orig_p2 = app.run_phase2

        def counting_p2(out_dir, **kw):
            converted["n"] += 1
            return orig_p2(out_dir, **kw)
        monkeypatch.setattr(app, "run_phase2", counting_p2)

        out_root = tmp_path / "out"
        (out_root / "done").mkdir(parents=True)
        (out_root / "done" / "done.qmd").write_text("---\ntitle: T\n---\n")
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        _make_pdf(inbox / "done.pdf")     # already converted → must be skipped
        _make_pdf(inbox / "todo.pdf")     # fresh → must be converted
        results = app.convert_batch(inbox, out_root, api_key="k")
        by = {r.stem: r for r in results}
        assert by["done"].resumed is True
        assert by["todo"].resumed is False
        assert converted["n"] == 1        # only the unconverted file ran Phase 2


class TestConvertBatch:
    def test_batch_continues_on_failure(self, tmp_path, monkeypatch):
        # first file fails in phase2, second succeeds
        calls = {"n": 0}

        def fake_phase2(out_dir, **kw):
            calls["n"] += 1
            if out_dir.name == "bad":
                raise RuntimeError("boom")
            (out_dir / f"{out_dir.name}.qmd").write_text("---\ntitle: T\n---\n")
            return {"cost_usd": 0.5}
        _stub_phases(monkeypatch)
        monkeypatch.setattr(app, "run_phase2", fake_phase2)

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        _make_pdf(inbox / "bad.pdf")
        _make_pdf(inbox / "good.pdf")
        results = app.convert_batch(inbox, tmp_path / "out", api_key="k")
        assert len(results) == 2
        by = {r.stem: r.status for r in results}
        assert by["bad"] == "fail" and by["good"] in ("ok", "warn")


def _fixed_estimate(usd):
    return lambda *a, **k: {
        "expected_usd": usd, "low_usd": usd * 0.5, "high_usd": usd * 2.0,
        "pages": 5, "candidate_pages": 1, "text_chars": 100,
        "breakdown": {"cover": 0.0, "detect": 0.0, "convert": usd}, "calibrated": True,
    }


class TestBudget:
    def test_per_file_skip_when_estimate_over_limit(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        monkeypatch.setattr(app, "estimate_file", _fixed_estimate(5.0))
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k", max_cost_per_file=2.0)
        assert r.status == "skip"
        assert r.qmd is None                 # never converted → no work, no spend
        assert r.est_usd == 5.0
        assert "use --allow-over-budget" in r.error

    def test_allow_over_budget_overrides_gate(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        monkeypatch.setattr(app, "estimate_file", _fixed_estimate(5.0))
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k",
                            max_cost_per_file=2.0, allow_over_budget=True)
        assert r.status in ("ok", "warn")    # converted despite the estimate
        assert r.qmd is not None

    def test_no_gate_when_limit_unset(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        monkeypatch.setattr(app, "estimate_file", _fixed_estimate(99.0))
        pdf = _make_pdf(tmp_path / "doc.pdf")
        r = app.convert_one(pdf, tmp_path / "out", api_key="k")  # no max_cost_per_file
        assert r.status in ("ok", "warn")

    def test_batch_total_backstop_skips_remaining(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)                 # phase1 0.21 + convert 1.00 actual per file
        monkeypatch.setattr(app, "estimate_file", _fixed_estimate(3.0))
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        _make_pdf(inbox / "a.pdf")
        _make_pdf(inbox / "b.pdf")
        # total=4: a passes (0+3<=4) and actually spends ~1.21; then b: 1.21+3>4 → skip
        results = app.convert_batch(inbox, tmp_path / "out", api_key="k", max_cost_total=4.0)
        by = {r.stem: r.status for r in results}
        assert by["a"] in ("ok", "warn")
        assert by["b"] == "skip"
        assert "batch budget" in next(r.error for r in results if r.stem == "b")


class TestLargeDocWarning:
    def _est(self, text_chars, pages):
        return {"expected_usd": 1.0, "low_usd": 0.5, "high_usd": 2.0, "pages": pages,
                "candidate_pages": 0, "text_chars": text_chars,
                "breakdown": {"cover": 0.0, "detect": 0.0, "convert": 1.0}, "calibrated": True}

    def test_warns_when_text_exceeds_context_budget(self, tmp_path, monkeypatch, caplog):
        import logging
        _stub_phases(monkeypatch)
        pdf = _make_pdf(tmp_path / "doc.pdf")
        # 4M chars ≈ 1M tokens of text > 0.7 * 1M window → warn
        with caplog.at_level(logging.WARNING):
            app.convert_one(pdf, tmp_path / "out", api_key="k",
                            estimate=self._est(text_chars=4_000_000, pages=1400))
        assert any("very large" in r.message and "context window" in r.message
                   for r in caplog.records)

    def test_no_warning_for_normal_doc(self, tmp_path, monkeypatch, caplog):
        import logging
        _stub_phases(monkeypatch)
        pdf = _make_pdf(tmp_path / "doc.pdf")
        with caplog.at_level(logging.WARNING):
            app.convert_one(pdf, tmp_path / "out", api_key="k",
                            estimate=self._est(text_chars=120_000, pages=40))
        assert not any("very large" in r.message for r in caplog.records)


class TestValidateKey:
    def _mock(self, monkeypatch, *, code=None):
        import urllib.error, urllib.request
        from pdf2md import llm_client

        class _Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"data": {"label": "ok"}}'

        def _open(req, timeout=10):
            if code:
                raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)
            return _Resp()
        monkeypatch.setattr(urllib.request, "urlopen", _open)
        return llm_client

    def test_valid_key_ok(self, monkeypatch):
        lc = self._mock(monkeypatch)
        ok, _ = lc.validate_key("sk-or-good")
        assert ok

    def test_401_rejected_with_friendly_message(self, monkeypatch):
        lc = self._mock(monkeypatch, code=401)
        ok, msg = lc.validate_key("sk-or-bad")
        assert not ok and "rejected" in msg.lower() and "setup" in msg.lower()

    def test_402_reports_credits(self, monkeypatch):
        lc = self._mock(monkeypatch, code=402)
        ok, msg = lc.validate_key("sk-or-broke")
        assert not ok and "credit" in msg.lower()

    def test_network_error_does_not_block(self, monkeypatch):
        import urllib.request
        from pdf2md import llm_client
        def _boom(req, timeout=10): raise OSError("no network")
        monkeypatch.setattr(urllib.request, "urlopen", _boom)
        ok, _ = llm_client.validate_key("sk-or-x")
        assert ok            # flaky network must not block a valid run


class TestScaffolding:
    def test_ensure_scaffolding_creates_project(self, tmp_path):
        out_root = tmp_path / "out"
        app._ensure_scaffolding(out_root)
        assert (out_root / "_quarto.yml").exists()
        assert (out_root / "_typst.yml").exists()
        assert (out_root / "_meta").exists()    # symlink to render assets

class TestStripChrome:
    """1:1 conversion is the default; --strip-chrome opts into chrome removal."""

    def _spy_phase1(self, monkeypatch, seen):
        def spy(pdf, out_dir, **kw):
            seen.update(kw)
            return {"figures": 0, "cost_usd": {"cover": 0.0, "detect": 0.0},
                    "cover": None}
        monkeypatch.setattr(app, "run_phase1", spy)

    def test_phase1_keeps_chrome_by_default(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        seen = {}
        self._spy_phase1(monkeypatch, seen)
        r = app.convert_one(_make_pdf(tmp_path / "doc.pdf"), tmp_path / "out", api_key="k")
        assert r.status == "ok"
        assert seen["do_strip_chrome"] is False

    def test_strip_chrome_strips_in_phase1_and_postfixes_qmd(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        seen = {}
        self._spy_phase1(monkeypatch, seen)
        verifies = {"n": 0}
        orig_verify = app._run_verify

        def counting_verify(out_dir, stem, meta=None):
            verifies["n"] += 1
            return orig_verify(out_dir, stem, meta=meta)
        monkeypatch.setattr(app, "_run_verify", counting_verify)
        monkeypatch.setattr(app, "strip_chrome_qmd", lambda qmd, out_dir: 2)

        r = app.convert_one(_make_pdf(tmp_path / "doc.pdf"), tmp_path / "out",
                            api_key="k", strip_chrome=True)
        assert seen["do_strip_chrome"] is True
        assert any(p.startswith("chrome:") for p in r.postfixes_applied)
        assert verifies["n"] == 2       # main verify + post-strip re-verify

    def test_no_qmd_postfix_when_nothing_stripped(self, tmp_path, monkeypatch):
        _stub_phases(monkeypatch)
        monkeypatch.setattr(app, "strip_chrome_qmd", lambda qmd, out_dir: 0)
        r = app.convert_one(_make_pdf(tmp_path / "doc.pdf"), tmp_path / "out",
                            api_key="k", strip_chrome=True)
        assert not any(p.startswith("chrome:") for p in r.postfixes_applied)


class TestCliChromeFlags:
    def _parse(self, argv):
        from pdf2md.app_cli import _build_parser
        return _build_parser().parse_args(argv)

    def test_strip_chrome_defaults_off(self):
        args = self._parse(["doc.pdf"])
        assert args.strip_chrome is False

    def test_strip_chrome_flag(self):
        args = self._parse(["doc.pdf", "--strip-chrome"])
        assert args.strip_chrome is True

    def test_keep_headers_still_accepted_as_noop(self):
        args = self._parse(["doc.pdf", "--keep-headers"])
        assert args.keep_headers is True and args.strip_chrome is False


class TestCliPostfixFlags:
    def _parse(self, argv):
        from pdf2md.app_cli import _build_parser
        return _build_parser().parse_args(argv)

    def test_postfix_defaults_to_three_iterations(self):
        assert self._parse(["doc.pdf"]).postfix == 3

    def test_postfix_takes_an_iteration_budget(self):
        assert self._parse(["doc.pdf", "--postfix", "5"]).postfix == 5

    def test_no_postfix_disables_the_loop(self):
        assert self._parse(["doc.pdf", "--no-postfix"]).postfix == 0

    def test_review_flag_is_gone(self, capsys):
        import pytest
        with pytest.raises(SystemExit):
            self._parse(["doc.pdf", "--review"])
