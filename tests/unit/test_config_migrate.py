"""Tests for `localharness config migrate`.

Regression cover for the v0.9.1 upgrade gap (#15): `localharness init` bakes the
fully-resolved `org.permissions.deny_patterns` into config.yaml, so a later growth of
the shipped default deny list never reaches an existing install. `config migrate`
additively syncs the missing shipped defaults into an existing config — appending only,
never removing/reordering the user's own entries, and never touching any other key.
"""
from __future__ import annotations

import warnings
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


_ABSENT = object()


def _write_config(config_dir: Path, deny: list[str], *, allow_patterns=_ABSENT) -> Path:
    """Write a valid v0.9.0-shaped config.yaml with the given deny_patterns.

    `allow_patterns` reproduces the key every pre-v0.14 `init` wrote (D2); left out by default,
    because most of this file is about the deny list.
    """
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
    if allow_patterns is not _ABSENT:
        cfg["org"]["permissions"]["allow_patterns"] = allow_patterns
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
    for d in (before, after):
        d["org"]["permissions"].pop("deny_patterns")
        # the migration's own revision stamp is bookkeeping, not a user key
        d["org"]["permissions"].pop("defaults_revision", None)
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


# --------------------------------------------------------------------------- #
# Revision-stamping semantics (redesign: the sync auto-applies on first start
# after a package upgrade; the stamp is what makes auto-apply safe).
# --------------------------------------------------------------------------- #

def test_migrate_stamps_current_defaults_revision(tmp_path):
    """After migrating, the config is stamped at the current shipped defaults revision."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    cfg = _write_config(tmp_path, OLD_7)
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    data = yaml.safe_load(cfg.read_text())
    assert data["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION


def test_absent_revision_key_reads_as_zero_and_plans_migration(tmp_path):
    """A config with no defaults_revision key = revision 0 → a migration is planned."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
    from localharness.config.migrate import plan

    cfg = _write_config(tmp_path, OLD_7)
    data = yaml.safe_load(cfg.read_text())
    assert "defaults_revision" not in data["org"]["permissions"]

    p = plan(data)
    assert p is not None
    assert p.from_revision == 0
    assert p.to_revision == CURRENT_DEFAULTS_REVISION
    assert len(p.added) > 0


def test_migrate_respects_deliberate_removal_after_stamp(tmp_path):
    """THE load-bearing property: a default deleted AFTER the config is stamped current is
    never re-added — subsequent migrations/startups add nothing and never nag. This is why
    auto-apply is safe."""
    cfg = _write_config(tmp_path, OLD_7)
    assert _run(tmp_path).exit_code == 0  # brings config to current revision + full list

    # user deliberately deletes a shipped default
    data = yaml.safe_load(cfg.read_text())
    removed = "bash_exec(*poweroff*)"
    assert removed in data["org"]["permissions"]["deny_patterns"]
    data["org"]["permissions"]["deny_patterns"].remove(removed)
    cfg.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
    before = cfg.read_bytes()

    # migrate again → removal respected: nothing re-added, up to date, zero writes
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "up to date" in result.output.lower()
    assert removed not in _deny(cfg)
    assert cfg.read_bytes() == before


# --------------------------------------------------------------------------- #
# Startup auto-apply seam (first `start` after a package upgrade folds it in).
# Driven directly the way existing start_cmd tests import helpers.
# --------------------------------------------------------------------------- #

def test_start_auto_migrate_applies_stamps_and_backs_up_once_then_quiet(tmp_path):
    """The `start` seam runs the same engine: a stale config is migrated, stamped, and
    backed up exactly once; a second call is a no-op (zero writes, zero new backups)."""
    from localharness.cli.start_cmd import _auto_migrate_deny_defaults
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    cfg = _write_config(tmp_path, OLD_7)
    _auto_migrate_deny_defaults(cfg)

    data = yaml.safe_load(cfg.read_text())
    assert set(PermissionConfig().deny_patterns).issubset(
        set(data["org"]["permissions"]["deny_patterns"])
    )
    assert data["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1
    after_first = cfg.read_bytes()

    # second call: already at current revision → no-op
    _auto_migrate_deny_defaults(cfg)
    assert cfg.read_bytes() == after_first
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1


def test_start_auto_migrate_failure_does_not_raise(tmp_path, monkeypatch):
    """A migration failure must NEVER block startup — the seam swallows it and returns."""
    from localharness.cli import start_cmd
    from localharness.config import migrate as migrate_mod

    cfg = _write_config(tmp_path, OLD_7)
    before = cfg.read_bytes()

    def _boom(*a, **k):
        raise OSError("simulated write failure")

    monkeypatch.setattr(migrate_mod, "apply", _boom)

    # must not raise out of startup
    start_cmd._auto_migrate_deny_defaults(cfg)
    # config untouched, no partial backup
    assert cfg.read_bytes() == before
    assert list(tmp_path.glob("config.yaml.bak-*")) == []


# --------------------------------------------------------------------------- #
# D2 — the v0.14 removal of `org.permissions.allow_patterns`.
#
# Every pre-v0.14 `localharness init` wrote `allow_patterns: []`. v0.14 removed the key, which
# made those configs unloadable — and `config migrate`, the documented repair, failed on the
# same validator before it could write. These pin both halves: an old-init-shaped config loads
# through the model, and migrate strips the key and writes.
# --------------------------------------------------------------------------- #

def _perms(config_file: Path) -> dict:
    return yaml.safe_load(config_file.read_text())["org"]["permissions"]


def test_an_old_init_config_loads_through_the_real_loader(tmp_path):
    """The exact shape `init` wrote before v0.14: mode `auto` + `allow_patterns: []` + denies.

    Through `HarnessConfig`, not a bare `PermissionConfig` — this is what `start` does, and it
    is what failed with "Value error, permissions.allow_patterns was removed in v0.14".
    """
    from localharness.config.models import HarnessConfig

    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        loaded = HarnessConfig.model_validate(yaml.safe_load(cfg.read_text()))
    assert loaded.org.permissions.mode == "guarded"
    assert loaded.org.permissions.deny_patterns == OLD_7


def test_migrate_strips_an_empty_allow_patterns_and_writes(tmp_path):
    """The repair path: migrate removes the key, the result validates, the file is written."""
    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "allow_patterns" not in _perms(cfg)
    assert "permissions.allow_patterns" in result.output
    # and the migrated file is loadable — the thing that was broken
    from localharness.config.models import HarnessConfig

    HarnessConfig.model_validate(yaml.safe_load(cfg.read_text()))


def test_migrate_strips_a_populated_allow_patterns_and_names_the_entries(tmp_path):
    """A non-empty list is a real loosening intent this release does not honor. It is removed
    too — migrate is the ONLY way past a validator that otherwise blocks every start — and the
    output says the entries were never honored rather than dropping them silently."""
    cfg = _write_config(tmp_path, OLD_7, allow_patterns=["bash_exec(*)", "write(*)"])
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "allow_patterns" not in _perms(cfg)
    assert "bash_exec(*)" in result.output and "never honored" in result.output


def test_migrate_repairs_the_dead_key_even_when_already_stamped_current(tmp_path):
    """Revision-gating alone would refuse: `allow_patterns` blocks loading at ANY stamp."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    cfg = _write_config(tmp_path, PermissionConfig().deny_patterns, allow_patterns=[])
    data = yaml.safe_load(cfg.read_text())
    data["org"]["permissions"]["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "allow_patterns" not in _perms(cfg)


def test_migrate_carries_the_embedded_chmod_fix_into_an_old_config(tmp_path):
    r"""#159: `bash_exec(chmod 777 *)` is anchored, so `find . -exec chmod 777 {} \;` passed it.

    The embedded form must actually REACH an old install — which it could not while the same
    config's `allow_patterns` made migrate refuse to write.
    """
    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "bash_exec(*chmod 777*)" in cfg.read_text()


def test_migrate_leaves_other_keys_unchanged_when_stripping_the_dead_key(tmp_path):
    """The removal is surgical: nothing else in the file moves."""
    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    before = yaml.safe_load(cfg.read_text())
    before["org"]["permissions"].pop("allow_patterns")
    assert _run(tmp_path).exit_code == 0

    after = yaml.safe_load(cfg.read_text())
    for d in (before, after):
        d["org"]["permissions"].pop("deny_patterns")
        d["org"]["permissions"].pop("defaults_revision", None)
    assert after == before


def test_start_auto_migrate_repairs_the_dead_key_before_the_loader_reads_it(tmp_path):
    """`start` runs the same engine before loading, so an upgrading user never sees the error."""
    from localharness.cli.start_cmd import _auto_migrate_deny_defaults

    cfg = _write_config(tmp_path, OLD_7, allow_patterns=[])
    _auto_migrate_deny_defaults(cfg)
    assert "allow_patterns" not in _perms(cfg)
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1


# --------------------------------------------------------------------------- #
# R10: the dead-key repair must not smuggle the deny defaults back in, and it
# has to reach the agent/division files `write_agent` stamped the same way.
# --------------------------------------------------------------------------- #

def test_repairing_the_dead_key_at_the_current_stamp_re_adds_nothing(tmp_path):
    """R10: the removal-respect promise survives the repair path.

    A config stamped current still gets a plan when `allow_patterns` is present — but that plan
    is the removal and NOTHING else. Re-appending a shipped default the user deliberately
    deleted, just because the dead key happened to be in the same file, contradicts the whole
    reason auto-apply is safe to run unattended."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
    from localharness.config.migrate import plan

    deny = list(PermissionConfig().deny_patterns)
    deleted = deny.pop()  # a shipped default the user removed on purpose
    cfg = _write_config(tmp_path, deny, allow_patterns=[])
    data = yaml.safe_load(cfg.read_text())
    data["org"]["permissions"]["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    p = plan(yaml.safe_load(cfg.read_text()))
    assert p is not None
    assert p.added == [], "a stamped-current config must not have defaults re-added"
    assert p.removed_allow_patterns == []

    assert _run(tmp_path).exit_code == 0
    assert "allow_patterns" not in _perms(cfg)
    assert deleted not in _deny(cfg), "the deleted default came back"


def _write_agent_file(config_dir: Path, name: str, *, allow_patterns, subdir: str = "agents") -> Path:
    """An agent/division file exactly as pre-v0.14 `write_agent` wrote it: the model's own
    serialization, `allow_patterns: []` and all."""
    perms: dict = {"mode": "guarded", "deny_patterns": []}
    if allow_patterns is not _ABSENT:
        perms["allow_patterns"] = allow_patterns
    body: dict = {"name": name, "role": f"{name} agent.", "permissions": perms}
    if subdir == "divisions":
        body = {"name": name, "description": f"{name} division.", "permissions": body["permissions"]}
    path = config_dir / subdir / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def test_migrate_strips_the_dead_key_from_an_agent_file_that_only_warns(tmp_path):
    """R10: `write_agent` stamped every agent file with `allow_patterns: []` too, and migrate
    never looked there — so the file warned on every single load with no repair path."""
    from localharness.config.loader import ConfigLoader

    _write_config(tmp_path, PermissionConfig().deny_patterns)
    agent = _write_agent_file(tmp_path, "scout", allow_patterns=[])

    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "scout.yaml" in result.output
    assert "allow_patterns" not in yaml.safe_load(agent.read_text())["permissions"]
    assert len(list((tmp_path / "agents").glob("scout.yaml.bak-*"))) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert ConfigLoader(config_dir=tmp_path).load_agent("scout").name == "scout"


def test_migrate_strips_a_populated_dead_key_from_an_agent_file_and_names_it(tmp_path):
    """A populated `allow_patterns` in an agent file does not warn — it fails validation, so the
    agent cannot be loaded at all. Migrate removes it and says the entries were never honored."""
    from localharness.config.loader import ConfigLoader

    _write_config(tmp_path, PermissionConfig().deny_patterns)
    agent = _write_agent_file(tmp_path, "scout", allow_patterns=["bash_exec(*)"])

    result = _run(tmp_path)
    assert result.exit_code == 0, result.output
    assert "scout.yaml" in result.output and "bash_exec(*)" in result.output
    assert "allow_patterns" not in yaml.safe_load(agent.read_text())["permissions"]
    assert ConfigLoader(config_dir=tmp_path).load_agent("scout").name == "scout"


def test_migrate_strips_the_dead_key_from_a_division_file(tmp_path):
    """Divisions sit on the same cascade (`loader._global_layer_permission`) and were written by
    the same serializer, so they carry the same dead key."""
    _write_config(tmp_path, PermissionConfig().deny_patterns)
    division = _write_agent_file(tmp_path, "research", allow_patterns=[], subdir="divisions")

    assert _run(tmp_path).exit_code == 0
    assert "allow_patterns" not in yaml.safe_load(division.read_text())["permissions"]


def test_a_clean_tree_plans_nothing(tmp_path):
    """No config work, no agent file carrying the dead key → None, zero writes, zero backups."""
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
    from localharness.config.migrate import load_plan

    cfg = _write_config(tmp_path, PermissionConfig().deny_patterns)
    data = yaml.safe_load(cfg.read_text())
    data["org"]["permissions"]["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _write_agent_file(tmp_path, "scout", allow_patterns=_ABSENT)

    _original, p = load_plan(cfg)
    assert p is None

    before = cfg.read_bytes()
    result = _run(tmp_path)
    assert result.exit_code == 0
    assert "up to date" in result.output.lower()
    assert cfg.read_bytes() == before
    assert list(tmp_path.glob("**/*.bak-*")) == []


def test_start_auto_migrate_repairs_an_agent_file_and_says_so(tmp_path):
    """The startup receipt names the sidecar files too — a silent file rewrite is not a receipt."""
    import io
    from contextlib import redirect_stdout

    from localharness.cli.start_cmd import _auto_migrate_deny_defaults

    cfg = _write_config(tmp_path, PermissionConfig().deny_patterns)
    data = yaml.safe_load(cfg.read_text())
    from localharness.config.defaults import CURRENT_DEFAULTS_REVISION

    data["org"]["permissions"]["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    agent = _write_agent_file(tmp_path, "scout", allow_patterns=[])

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _auto_migrate_deny_defaults(cfg)
    out = buffer.getvalue()

    assert "allow_patterns" not in yaml.safe_load(agent.read_text())["permissions"]
    assert "scout.yaml" in out
    assert list(tmp_path.glob("config.yaml.bak-*")) == [], "config.yaml needed nothing — leave it"
