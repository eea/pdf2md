#!/usr/bin/env python3
"""Tests for the pre-flight cost estimator (estimate.py) — all local, no LLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pdf2md import estimate  # noqa: E402


def _make_pdf(path: Path, pages: int = 1, text: str = ""):
    import fitz
    doc = fitz.open()
    for _ in range(pages):
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def _write_sidecars(out_dir: Path, *, pages, candidate, detect, convert):
    out_dir.mkdir(parents=True)
    (out_dir / "phase1.json").write_text(json.dumps({
        "pages_total": pages, "pages_candidate": candidate,
        "cost_usd": {"cover": 0.0, "detect": detect}, "figures": 0,
    }))
    (out_dir / "result.json").write_text(json.dumps({
        "phase_cost": {"cover": 0.0, "detect": detect, "convert": convert},
    }))


class TestCalibration:
    def test_seed_when_no_history(self, tmp_path):
        calib = estimate.load_calibration(tmp_path)
        assert calib["n_calibration_docs"] == 0
        assert calib["detect_usd_per_candidate"] == estimate.SEED_DETECT_USD_PER_CANDIDATE
        assert calib["convert_usd_per_page"] == estimate.SEED_CONVERT_USD_PER_PAGE

    def test_learns_per_unit_from_sidecars(self, tmp_path):
        # one doc: 10 pages, 4 candidates, detect $0.40, convert $1.00
        _write_sidecars(tmp_path / "doc", pages=10, candidate=4, detect=0.40, convert=1.00)
        calib = estimate.load_calibration(tmp_path)
        assert calib["n_calibration_docs"] == 1
        assert abs(calib["detect_usd_per_candidate"] - 0.10) < 1e-9   # 0.40 / 4
        assert abs(calib["convert_usd_per_page"] - 0.10) < 1e-9       # 1.00 / 10

    def test_ignores_dirs_without_sidecars(self, tmp_path):
        (tmp_path / "_meta").mkdir()
        (tmp_path / "empty").mkdir()
        calib = estimate.load_calibration(tmp_path)
        assert calib["n_calibration_docs"] == 0

    def test_calibrates_from_result_json_alone(self, tmp_path):
        # phase1.json is deleted by cleanup; calibration must still work from the
        # `est` block that survives inside result.json
        d = tmp_path / "doc"
        d.mkdir()
        (d / "result.json").write_text(json.dumps({
            "est": {"pages": 10, "candidate_pages": 4},
            "phase_cost": {"detect": 0.40, "convert": 1.00},
        }))
        calib = estimate.load_calibration(tmp_path)
        assert calib["n_calibration_docs"] == 1
        assert abs(calib["detect_usd_per_candidate"] - 0.10) < 1e-9
        assert abs(calib["convert_usd_per_page"] - 0.10) < 1e-9

    def test_skips_zero_spend_runs(self, tmp_path):
        # a failed/skipped doc (0 cost) must not drag the per-unit average to zero
        good = tmp_path / "good"; good.mkdir()
        (good / "result.json").write_text(json.dumps({
            "est": {"pages": 10, "candidate_pages": 4},
            "phase_cost": {"detect": 0.40, "convert": 1.00}}))
        failed = tmp_path / "failed"; failed.mkdir()
        (failed / "result.json").write_text(json.dumps({
            "est": {"pages": 20, "candidate_pages": 8},
            "phase_cost": {"detect": 0.0, "convert": 0.0}}))
        calib = estimate.load_calibration(tmp_path)
        assert calib["n_calibration_docs"] == 1                       # only the good one
        assert abs(calib["convert_usd_per_page"] - 0.10) < 1e-9

    def test_seeds_are_realistic_for_current_model(self):
        # guardrail: the old seeds were ~15-25x too high; keep them near real costs
        assert estimate.SEED_DETECT_USD_PER_CANDIDATE < 0.005
        assert estimate.SEED_CONVERT_USD_PER_PAGE < 0.005


class TestEstimateFile:
    def test_structure_and_band(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=3)
        calib = {"cover_usd": 0.005, "detect_usd_per_candidate": 0.02,
                 "convert_usd_per_page": 0.04, "n_calibration_docs": 1}
        e = estimate.estimate_file(pdf, calib)
        assert e["pages"] == 3
        assert e["low_usd"] <= e["expected_usd"] <= e["high_usd"]
        # convert term dominates: 3 pages * 0.04 = 0.12 (+ cover 0.005, no candidates)
        assert abs(e["breakdown"]["convert"] - 0.12) < 1e-9
        assert e["calibrated"] is True

    def test_uses_seed_calib_when_none(self, tmp_path):
        pdf = _make_pdf(tmp_path / "doc.pdf", pages=2)
        e = estimate.estimate_file(pdf, out_root=tmp_path)   # empty → seed
        assert e["calibrated"] is False
        assert e["expected_usd"] > 0