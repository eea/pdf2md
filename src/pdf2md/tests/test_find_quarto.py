"""Resolution order for _find_quarto: config override > PATH > install candidates.

Requires Python 3.10+ (app.py uses PEP 604 unions); the package floor is >=3.10.
"""

import json

import pdf2md.app as app


def _fake_quarto(tmp_path):
    q = tmp_path / "quarto"
    q.write_text("#!/bin/sh\n")
    return q


def test_config_override_wins_over_path(tmp_path, monkeypatch):
    cfg_quarto = _fake_quarto(tmp_path)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"quarto_path": str(cfg_quarto)}))
    monkeypatch.setattr(app, "CONFIG_FILE", cfg)
    # PATH also has one, but config must win
    monkeypatch.setattr(app.shutil, "which", lambda name: "/usr/bin/quarto")
    assert app._find_quarto() == str(cfg_quarto)


def test_falls_back_to_path_when_no_config(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CONFIG_FILE", tmp_path / "absent.json")
    monkeypatch.setattr(app.shutil, "which", lambda name: "/opt/bin/quarto")
    assert app._find_quarto() == "/opt/bin/quarto"


def test_bad_config_path_is_ignored(tmp_path, monkeypatch):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"quarto_path": "/does/not/exist/quarto"}))
    monkeypatch.setattr(app, "CONFIG_FILE", cfg)
    monkeypatch.setattr(app.shutil, "which", lambda name: "/opt/bin/quarto")
    # config value doesn't exist on disk → skip it, use PATH
    assert app._find_quarto() == "/opt/bin/quarto"
