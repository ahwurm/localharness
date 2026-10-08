"""What carries over from the `web` name on its own (CHANGELOG, 0.16.1): the token folder a release
named `web/` is renamed `mobile/` once, so a paired phone stays paired. A config file that still
says `web:` was read as `mobile:` until 0.17.0; since 0.17.1 the loader leaves it alone, so it is
refused as a section nothing owns, like any other. `localharness web` itself is gone, not aliased."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.channels.mobile import auth, push
from localharness.cli.app import app
from localharness.config import loader as config_loader


def test_web_is_not_a_command():
    result = CliRunner().invoke(app, ["web"])
    assert result.exit_code == 2 and "No such command 'web'" in result.output


def test_a_web_section_is_no_longer_read_as_mobile(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("version: '1'\nweb:\n  public_url: https://box.example\n", encoding="utf-8")
    assert config_loader._load_yaml_file(cfg) == {"version": "1", "web": {"public_url": "https://box.example"}}


def test_a_stale_web_section_is_refused_as_a_section_nothing_owns(tmp_path: Path):
    """The 0.17.0 sunset, kept in 0.17.1: a file that still says `web:` fails to load naming the
    file, the line and the fix, as any section no core key and no installed plugin owns does."""
    from localharness.config.loader import ConfigLoader, ConfigValidationError

    g, ws = tmp_path / "g", tmp_path / "proj" / ".localharness"
    g.mkdir()
    ws.mkdir(parents=True)
    (g / "config.yaml").write_text(
        "version: '1'\nprovider:\n  provider_type: vllm\n  base_url: http://127.0.0.1:9/v1\n"
        "  default_model: test-model\nweb:\n  public_url: https://box.example\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g, local_config_dir=ws).load_harness()
    assert exc.value.path == str(g / "config.yaml")
    assert "web (line 6): not a LocalHarness setting" in str(exc.value), str(exc.value)


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
