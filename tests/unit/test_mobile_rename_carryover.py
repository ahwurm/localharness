"""The two things that carry over from the `web` name on their own (CHANGELOG, Unreleased): a config
file that still says `web:` configures the mobile plugin until 0.17.0, said once per file in the log;
and the token folder a release named `web/` is renamed `mobile/` once, so a paired phone stays paired.
`localharness web` itself is gone — `test_mobile_plugin_e2e` pins that."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from localharness.channels.mobile import auth, push
from localharness.config import loader as config_loader


def test_a_web_section_is_read_as_mobile_and_said_once(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("version: '1'\nweb:\n  public_url: https://box.example\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="localharness.config.loader"):
        data = config_loader._load_yaml_file(cfg)
        again = config_loader._load_yaml_file(cfg)
    assert data == {"version": "1", "mobile": {"public_url": "https://box.example"}} == again
    assert caplog.text.count("`web:` in") == 1 and "is now `mobile:`" in caplog.text


def test_mobile_values_win_over_a_leftover_web_section(tmp_path: Path):
    cfg = tmp_path / "overrides.yaml"
    cfg.write_text("web:\n  public_url: https://old.example\n  enabled: false\n"
                   "mobile:\n  public_url: https://new.example\n", encoding="utf-8")
    assert config_loader._load_yaml_file(cfg) == {
        "mobile": {"public_url": "https://new.example", "enabled": False}}


def test_the_web_state_folder_is_renamed_mobile_once(tmp_path: Path):
    legacy = tmp_path / "web"
    legacy.mkdir()
    (legacy / "token").write_text("t\n", encoding="utf-8")
    assert auth.token_path(tmp_path) == tmp_path / "mobile" / "token"
    assert (tmp_path / "mobile" / "token").read_text(encoding="utf-8") == "t\n" and not legacy.exists()
    assert push.vapid_path(tmp_path) == tmp_path / "mobile" / "vapid.pem"
    assert push.subscriptions_path(tmp_path) == tmp_path / "mobile" / "push-subscriptions.json"


def test_a_mobile_folder_is_never_overwritten_by_a_leftover_web_one(tmp_path: Path):
    (tmp_path / "mobile").mkdir()
    (tmp_path / "mobile" / "token").write_text("new\n", encoding="utf-8")
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "token").write_text("old\n", encoding="utf-8")
    assert auth.token_path(tmp_path).read_text(encoding="utf-8") == "new\n"
    assert (tmp_path / "web").is_dir()  # left for the owner: nothing of theirs is deleted
