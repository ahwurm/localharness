"""Write hygiene for the files LocalHarness puts in a config dir: mode bits and backup count.

Two pre-release findings, one seam. Both are about the config files themselves rather than their
contents, so they sit beside `test_config_migrate.py` rather than inside it.

C3 — every config file this package writes is owner-only. `config.yaml` carries
`provider.api_key`, and it and its agent/division siblings carry the deny list and ask rule sets a
session is gated by. `Path.write_text` creates at 0666 & ~umask, so on the usual 022 umask `init`
and `migrate` were both landing 0664: any other account on the box could read the key and read the
policy. The overlay's atomic write was already 0600, but by accident of `NamedTemporaryFile`
rather than by statement.

C4 — migrate's timestamped backups accumulated forever (eleven beside one config.yaml on the
machine this was found on). Each is a full copy of the same secret, and the pile buries the one
backup someone actually wants. `BACKUP_RETENTION` bounds them, and the pruning is scoped so
tightly that nothing a human named themselves can be caught by it.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.migrate import (
    BACKUP_INFIX,
    BACKUP_RETENTION,
    BACKUP_STAMP_FORMAT,
)
from localharness.config.models import PermissionConfig
from localharness.config.overlay import CONFIG_FILE_MODE, atomic_write_overlay

from tests.unit.test_config_migrate import OLD_7, _write_agent_file, _write_config

runner = CliRunner()

posix_only = pytest.mark.skipif(
    os.name != "posix", reason="POSIX mode bits; Windows access control is the directory ACL"
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _backups(config_dir: Path, name: str = "config.yaml") -> list[Path]:
    return sorted(config_dir.glob(f"{name}{BACKUP_INFIX}*"))


# --- C3: mode bits ----------------------------------------------------------

@posix_only
@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_init_writes_a_config_only_its_owner_can_read(mock_client_cls, mock_detect, tmp_path):
    """The real `init` command end to end — the file it leaves behind holds `provider.api_key`."""
    from localharness.provider.detector import DetectorResult

    from tests.unit.test_init_cmd import _make_capability_result

    mock_detect.return_value = DetectorResult(
        found=True, provider_type="llamacpp", base_url="http://localhost:8080/v1",
        models=["qwen"], suggested_model="qwen", probe_duration_ms=1.0,
    )
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    mock_client_cls.return_value = client

    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output

    config_file = tmp_path / "config.yaml"
    assert "api_key" in config_file.read_text(), "the premise: the file carries a secret"
    assert _mode(config_file) == CONFIG_FILE_MODE


@posix_only
def test_migrate_leaves_the_config_its_sidecar_and_the_backups_owner_only(tmp_path):
    """Every file migrate writes, not just the one it was aimed at.

    The backup matters as much as the rewrite: it is a byte-for-byte copy of the pre-migration
    config, so leaving it at the umask default would hand out exactly the secret the tightened
    file no longer exposes.
    """
    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    agent = _write_agent_file(tmp_path, "scout", allow_patterns=[])
    for f in (cfg, agent):
        f.chmod(0o644)  # what a pre-fix install, or a stray umask, leaves behind

    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    for written in (cfg, agent, *_backups(tmp_path), *_backups(tmp_path / "agents", "scout.yaml")):
        assert _mode(written) == CONFIG_FILE_MODE, f"{written} is readable beyond its owner"


@posix_only
def test_the_overlay_atomic_write_is_owner_only_by_statement(tmp_path):
    """The overlay was 0600 only because `NamedTemporaryFile` happens to create at 0600. Now it
    is stated — including over a file that already exists at a looser mode."""
    overlay = tmp_path / "overrides.yaml"
    overlay.write_text("agent: {}\n", encoding="utf-8")
    overlay.chmod(0o644)

    atomic_write_overlay(overlay, {"agent": {"temperature": 0.3}})

    assert _mode(overlay) == CONFIG_FILE_MODE
    assert yaml.safe_load(overlay.read_text())["agent"]["temperature"] == 0.3


def test_restricting_a_file_that_cannot_be_chmodded_is_not_an_error(tmp_path, monkeypatch):
    """The write has already succeeded by the time the mode is tightened, so a chmod that fails
    (exotic FUSE and network mounts refuse it outright) must not fail `init`."""
    from localharness.config import overlay as overlay_mod

    def _boom(*args, **kwargs):
        raise OSError("chmod not supported")

    monkeypatch.setattr(overlay_mod.os, "chmod", _boom)
    target = tmp_path / "config.yaml"
    target.write_text("version: '1'\n", encoding="utf-8")

    overlay_mod.restrict_config_file(target)  # must not raise


# --- C4: backup retention ---------------------------------------------------

def _fake_backups(config_dir: Path, count: int, name: str = "config.yaml") -> list[Path]:
    """`count` backups of `name`, oldest first, named exactly the way migrate names them."""
    made = []
    for day in range(1, count + 1):
        stamp = f"2020{day:02d}01-000000"
        # the stamp must be one migrate itself could have written
        from datetime import datetime

        datetime.strptime(stamp, BACKUP_STAMP_FORMAT)
        path = config_dir / f"{name}{BACKUP_INFIX}{stamp}"
        path.write_bytes(b"old\n")
        made.append(path)
    return made


def test_migrate_keeps_only_the_newest_backups(tmp_path):
    """Eleven copies of a config.yaml is what the owner's machine actually had."""
    cfg = _write_config(tmp_path, OLD_7)
    stale = _fake_backups(tmp_path, BACKUP_RETENTION + 3)

    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output

    kept = _backups(tmp_path)
    assert len(kept) == BACKUP_RETENTION
    # the one this migration just wrote is the newest and is never a pruning candidate
    fresh = [p for p in kept if p not in stale]
    assert len(fresh) == 1 and fresh[0].read_bytes() != b"old\n"
    # oldest first out; the survivors are the tail of the stale list plus the fresh one
    assert set(kept) - {fresh[0]} == set(stale[-(BACKUP_RETENTION - 1):])
    assert cfg.exists()


def test_pruning_never_touches_a_file_it_did_not_write(tmp_path):
    """The scope of a delete-files-in-the-user's-config-dir routine, pinned.

    A backup a human named themselves does not parse as a timestamp, and a backup of a DIFFERENT
    file is not a sibling of the one being rewritten. Neither is migrate's to remove.
    """
    _write_config(tmp_path, OLD_7)
    _fake_backups(tmp_path, BACKUP_RETENTION + 3)
    hand_kept = tmp_path / f"config.yaml{BACKUP_INFIX}before-the-upgrade"
    hand_kept.write_text("mine\n", encoding="utf-8")
    other_file = tmp_path / f"notes.yaml{BACKUP_INFIX}20200101-000000"
    other_file.write_text("unrelated\n", encoding="utf-8")

    assert runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)]).exit_code == 0

    assert hand_kept.read_text() == "mine\n"
    assert other_file.read_text() == "unrelated\n"


def test_the_start_seam_prunes_on_the_same_terms(tmp_path):
    """`start`'s auto-migrate is the path a user hits without asking for it, so it is the one
    that actually grows the pile — it runs the same engine and must prune the same way."""
    from localharness.cli.start_cmd import _auto_migrate_deny_defaults

    cfg = _write_config(tmp_path, OLD_7)
    _fake_backups(tmp_path, BACKUP_RETENTION + 3)

    _auto_migrate_deny_defaults(cfg)

    assert len(_backups(tmp_path)) == BACKUP_RETENTION
    assert set(PermissionConfig().deny_patterns).issubset(
        set(yaml.safe_load(cfg.read_text())["org"]["permissions"]["deny_patterns"])
    )


def test_a_sidecar_keeps_its_own_backup_budget(tmp_path):
    """Retention is per FILE: pruning config.yaml's backups must not count, or touch, an agent
    file's — they are different undo histories that happen to share a directory tree."""
    _write_config(tmp_path, PermissionConfig().deny_patterns)
    _write_agent_file(tmp_path, "scout", allow_patterns=[])
    agent_stale = _fake_backups(tmp_path / "agents", BACKUP_RETENTION - 1, "scout.yaml")
    config_stale = _fake_backups(tmp_path, BACKUP_RETENTION + 3)

    assert runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)]).exit_code == 0

    # the agent's history was under budget, so every one of its backups survives + the new one
    assert len(_backups(tmp_path / "agents", "scout.yaml")) == BACKUP_RETENTION
    assert all(p.exists() for p in agent_stale)
    # config.yaml was over budget and was pruned independently
    assert not config_stale[0].exists()
