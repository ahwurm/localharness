"""Tests for `localharness config migrate`.

Regression cover for the v0.9.1 upgrade gap (#15): `localharness init` bakes the
fully-resolved `org.permissions.deny_patterns` into config.yaml, so a later growth of
the shipped default deny list never reaches an existing install. `config migrate`
additively syncs the missing shipped defaults into an existing config — appending only,
never removing/reordering the user's own entries, and never touching any other key.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.models import PermissionConfig

runner = CliRunner()

# The real v0.9.0 shipped default deny list (7 patterns), including the historically
# broken `bash_exec(sudo:*)` — the glob required a literal colon after `sudo`, so it
# matched no real command until v0.9.1 corrected it to `bash_exec(*sudo *)`.
OLD_7 = [
    "write(*/.env)",
    "write(*/secrets*)",
    "write(*/config.yaml)",
    "write(*/agents/*.yaml)",
    "bash_exec(sudo:*)",
    "bash_exec(rm -rf *)",
    "bash_exec(chmod 777 *)",
]


def _write_config(config_dir: Path, deny: list[str]) -> Path:
    """Write a valid v0.9.0-shaped config.yaml with the given deny_patterns."""
    cfg = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "api_key": "none",
            "default_model": "Qwen/Qwen3-30B",
            "available_models": ["Qwen/Qwen3-30B"],
            "supports_function_calling": True,
            "timeout_seconds": 600.0,
        },
        "org": {
            "name": "default",
            "default_model": "Qwen/Qwen3-30B",
            "default_temperature": 0.6,
            "default_max_tokens": 4096,
            "permissions": {
                "mode": "auto",
                "deny_patterns": list(deny),
                "budget": {
                    "max_actions": 100,
                    "max_duration_minutes": 30.0,
                    "kill_file": "~/.localharness/KILL",
                },
            },
            "log_level": "info",
        },
    }
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "config.yaml"
    path.write_text(
        yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    return path


def _run(config_dir: Path, *args: str):
    return runner.invoke(
        app, ["config", "migrate", "--config-dir", str(config_dir), *args]
    )


def _deny(config_file: Path) -> list[str]:
    return yaml.safe_load(config_file.read_text())["org"]["permissions"]["deny_patterns"]


def test_migrate_adds_missing_shipped_defaults(tmp_path):
    """v0.9.0 config → migrate brings it to a superset incl. all 24 current defaults."""
    cfg = _write_config(tmp_path, OLD_7)
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    new_deny = _deny(cfg)
    shipped = PermissionConfig().deny_patterns
    # every shipped default is now present (superset incl. all 24)
    assert set(shipped).issubset(set(new_deny))
    # additive: the user's original 7 stay, in order, at the front
    assert new_deny[: len(OLD_7)] == OLD_7
    # the historically-broken sudo pattern STAYS (additive never removes)
    assert "bash_exec(sudo:*)" in new_deny
    # the corrected sudo pattern ARRIVES
    assert "bash_exec(*sudo *)" in new_deny
    # exactly the missing shipped defaults were appended, nothing else
    expected_missing = [p for p in shipped if p not in OLD_7]
    assert len(expected_missing) > 0
    assert new_deny == OLD_7 + expected_missing


def test_migrate_preserves_custom_user_pattern(tmp_path):
    """A user-added custom deny pattern survives the migration verbatim."""
    user = OLD_7 + ["bash_exec(*curl*)"]
    cfg = _write_config(tmp_path, user)
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    new_deny = _deny(cfg)
    assert "bash_exec(*curl*)" in new_deny
    # the user's whole list is preserved, in order, at the front
    assert new_deny[: len(user)] == user
    assert set(PermissionConfig().deny_patterns).issubset(set(new_deny))


def test_migrate_leaves_other_keys_semantically_unchanged(tmp_path):
    """Only org.permissions.deny_patterns changes; every other key re-parses identically."""
    cfg = _write_config(tmp_path, OLD_7)
    before = yaml.safe_load(cfg.read_text())
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    after = yaml.safe_load(cfg.read_text())
    before["org"]["permissions"].pop("deny_patterns")
    after["org"]["permissions"].pop("deny_patterns")
    assert after == before


def test_migrate_dry_run_writes_nothing(tmp_path):
    """--dry-run reports what would change but leaves the file byte-identical, no backup."""
    cfg = _write_config(tmp_path, OLD_7)
    before = cfg.read_bytes()
    result = _run(tmp_path, "--dry-run")
    assert result.exit_code == 0, result.output

    assert cfg.read_bytes() == before
    assert list(tmp_path.glob("config.yaml.bak-*")) == []
    # still reports the patterns it WOULD add
    assert "bash_exec(*sudo *)" in result.output


def test_migrate_writes_backup_with_premigration_bytes(tmp_path):
    """A real run writes a timestamped backup equal to the pre-migration bytes."""
    cfg = _write_config(tmp_path, OLD_7)
    before = cfg.read_bytes()
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output

    backups = list(tmp_path.glob("config.yaml.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before
    # and the live file actually changed
    assert cfg.read_bytes() != before


def test_migrate_is_idempotent(tmp_path):
    """A second run is a no-op: reports up to date, adds nothing, writes no new backup."""
    cfg = _write_config(tmp_path, OLD_7)
    first = _run(tmp_path)
    assert first.exit_code == 0, first.output
    after_first = cfg.read_bytes()
    backups_after_first = sorted(tmp_path.glob("config.yaml.bak-*"))
    assert len(backups_after_first) == 1

    second = _run(tmp_path)
    assert second.exit_code == 0, second.output
    assert "up to date" in second.output.lower()
    # no change, no new backup
    assert cfg.read_bytes() == after_first
    assert sorted(tmp_path.glob("config.yaml.bak-*")) == backups_after_first


def test_migrate_missing_config_fails(tmp_path):
    """No config.yaml → clear failure pointing at init, non-zero exit."""
    result = _run(tmp_path)
    assert result.exit_code != 0
    assert "init" in result.output.lower()


def test_migrate_broken_yaml_fails(tmp_path):
    """Unparseable config.yaml → non-zero exit, nothing written."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.yaml").write_text("{ broken: [unterminated\n", encoding="utf-8")
    result = _run(tmp_path)
    assert result.exit_code != 0
    assert list(tmp_path.glob("config.yaml.bak-*")) == []
