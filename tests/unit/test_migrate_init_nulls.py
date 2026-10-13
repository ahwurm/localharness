"""The init-nulls migration step: init's three dumped values become null, once, under their own stamp.

Older `init` runs wrote `max_actions: 100`, `max_tool_output_chars: 32000` and
`tool_result_evict_threshold_chars: 8000` as if the person had chosen them; they now default to
"none, follow the window". The step nulls exactly those values (equality on a plain int) in
config.yaml and in every agent/division file, under `org.permissions.init_nulls_revision`, with one
backup per file, and touches neither the deny list nor the machine fingerprint.
"""
from __future__ import annotations

import copy
import io
from contextlib import redirect_stdout
from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.cli.start_cmd import _auto_migrate_deny_defaults
from localharness.config import migrate, overlay, trust
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION, CURRENT_INIT_NULLS_REVISION
from localharness.config.models import HarnessConfig, PermissionConfig

runner = CliRunner()

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m",
            "api_key": "none"}
SERVER = {"runtime": "llamacpp", "launch": "binary", "binary": "/x/llama-server",
          "model": "/x/m.gguf", "port": 8081}
ORG_NULLED = ("org.permissions.budget.max_actions", "org.context.max_tool_output_chars",
              "org.context.tool_result_evict_threshold_chars")
NAME = "scout"


def _v017(*, max_actions=100, output=32000, evict=8000, deny=None, stamp=None, **extra) -> dict:
    """A 0.17-shaped config: defaults_revision current, init's three numbers dumped, no new stamp."""
    perms = {"defaults_revision": CURRENT_DEFAULTS_REVISION,
             "deny_patterns": list(PermissionConfig().deny_patterns if deny is None else deny),
             "budget": {"max_actions": max_actions}}
    if stamp is not None:
        perms["init_nulls_revision"] = stamp
    return {"version": "1", "provider": dict(PROVIDER),
            "org": {"permissions": perms,
                    "context": {"max_tool_output_chars": output,
                                "tool_result_evict_threshold_chars": evict}},
            **extra}


def _write(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return cfg


def _sidecar(tmp_path: Path, directory: str, name: str, data: dict) -> Path:
    path = tmp_path / directory / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _pre_v014_agent() -> dict:
    """What pre-v0.14 `write_agent` dumped whole: the dead key and init's three numbers."""
    return {"name": NAME, "role": "scout", "model": "inherit",
            "permissions": {"mode": "auto", "budget": {"max_actions": 100}, "allow_patterns": []},
            "context": {"max_tool_output_chars": 32000, "tool_result_evict_threshold_chars": 8000}}


def _migrate(cfg: Path):
    original, plan = migrate.load_plan(cfg)
    assert plan is not None
    return plan, migrate.apply(cfg, original, plan)


def _disk(cfg: Path) -> dict:
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))


def _start_receipt(cfg: Path) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _auto_migrate_deny_defaults(cfg)
    return buffer.getvalue()


# ----------------------------------------------------------------------------- config.yaml


def test_a_017_config_gets_exactly_the_three_values_nulled_and_its_own_stamp(tmp_path):
    cfg = _write(tmp_path, _v017(server=dict(SERVER)))
    data = yaml.safe_load(cfg.read_text())
    before = copy.deepcopy(data)

    plan = migrate.plan(data)

    assert plan is not None and plan.init_nulls_due and plan.nulled == ORG_NULLED
    assert plan.config_unchanged is False and plan.defaults_unchanged is True
    assert data == before  # the input is never mutated
    _migrate(cfg)
    on_disk = _disk(cfg)
    org = on_disk["org"]
    assert org["permissions"]["budget"]["max_actions"] is None
    assert org["context"] == {"max_tool_output_chars": None, "tool_result_evict_threshold_chars": None}
    assert org["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert org["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    assert org["permissions"]["deny_patterns"] == before["org"]["permissions"]["deny_patterns"]
    assert on_disk["server"] == before["server"]
    HarnessConfig.model_validate(on_disk)  # a nulled config validates


def test_a_hand_set_number_a_bool_a_float_or_a_string_stays(tmp_path):
    for kept in (150, True, 100.0, "100"):
        plan = migrate.plan(_v017(max_actions=kept, output=32001, evict=8000.0))
        budget = plan.updated["org"]["permissions"]["budget"]
        assert budget["max_actions"] == kept and type(budget["max_actions"]) is type(kept)
        assert plan.updated["org"]["context"] == {"max_tool_output_chars": 32001,
                                                  "tool_result_evict_threshold_chars": 8000.0}
        assert plan.nulled == ()


def test_it_runs_once_under_its_stamp_and_never_on_content_alone(tmp_path):
    cfg = _write(tmp_path, _v017())
    _migrate(cfg)
    assert migrate.plan(_disk(cfg)) is None

    stamped = _v017(stamp=CURRENT_INIT_NULLS_REVISION)  # still holds 100: the person's, now
    assert migrate.plan(stamped) is None
    assert migrate.plan(_v017(stamp=True)) is not None  # a bool is not a revision
    assert migrate.plan(_v017(stamp=0)) is not None


def test_a_due_config_holding_none_of_the_values_gets_a_stamp_only_rewrite(tmp_path):
    cfg = _write(tmp_path, _v017(max_actions=150, output=20000, evict=5000))
    plan, _backups = _migrate(cfg)
    assert plan.nulled == () and plan.init_nulls_due
    org = _disk(cfg)["org"]
    assert org["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert org["permissions"]["budget"]["max_actions"] == 150
    assert org["context"]["max_tool_output_chars"] == 20000


def test_a_deny_default_the_person_deleted_stays_deleted(tmp_path):
    deny = list(PermissionConfig().deny_patterns)
    deleted = deny.pop()
    cfg = _write(tmp_path, _v017(deny=deny))
    plan, _backups = _migrate(cfg)
    assert plan.added == [] and plan.from_revision == plan.to_revision
    assert deleted not in _disk(cfg)["org"]["permissions"]["deny_patterns"]
    assert _disk(cfg)["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION


# ------------------------------------------------------------------------------- sidecars


def test_a_pre_v014_agent_gets_one_plan_one_write_and_one_backup_equal_to_the_original(tmp_path):
    cfg = _write(tmp_path, _v017())
    agent = _sidecar(tmp_path, "agents", NAME, _pre_v014_agent())
    original = agent.read_bytes()

    plan, backups = _migrate(cfg)

    (sidecar,) = plan.sidecars
    assert sidecar.removed == [] and sidecar.nulled == (
        "permissions.budget.max_actions", "context.max_tool_output_chars",
        "context.tool_result_evict_threshold_chars")
    agent_backups = sorted((tmp_path / "agents").glob("scout.yaml.bak-*"))
    assert len(agent_backups) == 1 and agent_backups[0].read_bytes() == original
    assert agent_backups[0] in backups
    written = _disk(agent)
    assert "allow_patterns" not in written["permissions"]
    assert written["permissions"]["budget"]["max_actions"] is None
    assert written["context"] == {"max_tool_output_chars": None, "tool_result_evict_threshold_chars": None}


def test_a_sidecar_with_only_max_actions_has_no_dead_key_and_a_division_is_walked_too(tmp_path):
    cfg = _write(tmp_path, _v017())
    agent = _sidecar(tmp_path, "agents", NAME, {"name": NAME, "permissions": {"budget": {"max_actions": 100}}})
    division = _sidecar(tmp_path, "divisions", "research",
                        {"name": "research", "permissions": {"budget": {"max_actions": 100}}})

    plan, _backups = _migrate(cfg)

    assert [s.removed for s in plan.sidecars] == [None, None]
    assert [s.path for s in plan.sidecars] == [agent, division]
    for path in (agent, division):
        assert _disk(path)["permissions"]["budget"]["max_actions"] is None


def test_a_sidecar_with_only_the_dead_key_is_unchanged_behaviour_when_the_step_is_not_due(tmp_path):
    cfg = _write(tmp_path, _v017(stamp=CURRENT_INIT_NULLS_REVISION))
    agent = _sidecar(tmp_path, "agents", NAME, _pre_v014_agent())

    plan, _backups = _migrate(cfg)

    (sidecar,) = plan.sidecars
    assert sidecar.removed == [] and sidecar.nulled == ()
    assert plan.config_unchanged is True and list(tmp_path.glob("config.yaml.bak-*")) == []
    written = _disk(agent)
    assert "allow_patterns" not in written["permissions"]
    assert written["permissions"]["budget"]["max_actions"] == 100  # not due: init's number is left


def test_a_sidecar_without_any_of_it_is_left_alone(tmp_path):
    cfg = _write(tmp_path, _v017())
    quiet = _sidecar(tmp_path, "agents", "quiet", {"name": "quiet", "permissions": {"budget": {"max_actions": 150}}})
    before = quiet.read_bytes()

    plan, _backups = _migrate(cfg)

    assert plan.sidecars == () and quiet.read_bytes() == before
    assert list((tmp_path / "agents").glob("*.bak-*")) == []


# --------------------------------------------------------------------------- the fingerprint


def test_the_machine_snapshot_and_fingerprint_are_equal_before_and_after(tmp_path):
    cfg = _write(tmp_path, _v017(server=dict(SERVER)))
    _sidecar(tmp_path, "agents", NAME, _pre_v014_agent())
    before = trust.machine_snapshot(tmp_path)

    _migrate(cfg)

    after = trust.machine_snapshot(tmp_path)
    assert after == before and trust.fingerprint(after) == trust.fingerprint(before)
    on_disk = _disk(cfg)
    assert on_disk["server"] == SERVER  # no launch or launch-model key written
    assert set(on_disk) == {"version", "provider", "org", "server"}


# ------------------------------------------------------------------------------ the receipts


def test_the_start_receipt_names_each_file_with_the_clause_and_not_the_deny_sentence(tmp_path):
    cfg = _write(tmp_path, _v017(server=dict(SERVER)))
    _sidecar(tmp_path, "agents", "orchestrator", _pre_v014_agent())

    out = " ".join(_start_receipt(cfg).split())

    assert migrate.INIT_NULLS_CLAUSE + " (config.yaml)" in out
    assert migrate.INIT_NULLS_CLAUSE + " (agents/orchestrator.yaml)" in out
    assert "cleaned allow_patterns from orchestrator.yaml" in out
    assert "Security defaults updated" not in out and "Config updated" in out
    assert _start_receipt(cfg) == ""  # once


def test_a_sidecar_nulled_without_a_dead_key_gets_the_clause_and_no_cleaning_note(tmp_path):
    cfg = _write(tmp_path, _v017())
    _sidecar(tmp_path, "divisions", "research", {"name": "research", "permissions": {"budget": {"max_actions": 100}}})

    out = " ".join(_start_receipt(cfg).split())

    assert migrate.INIT_NULLS_CLAUSE + " (divisions/research.yaml)" in out
    assert "cleaned allow_patterns" not in out


def test_a_stamp_only_rewrite_prints_no_clause(tmp_path):
    cfg = _write(tmp_path, _v017(max_actions=150, output=20000, evict=5000))

    out = _start_receipt(cfg)

    assert "Config updated" in out and "cleared init's" not in out
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION


def test_config_migrate_lists_the_nulled_keys_and_prints_the_per_file_clause(tmp_path):
    cfg = _write(tmp_path, _v017())
    _sidecar(tmp_path, "agents", NAME, {"name": NAME, "permissions": {"budget": {"max_actions": 100}}})

    dry = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    plain = " ".join(dry.output.split())
    assert "org.permissions.budget.max_actions" in plain and "permissions.budget.max_actions → null in" in plain
    assert "empty — nothing was carried" not in plain
    assert "init_nulls_revision" not in cfg.read_text()  # a dry run writes nothing

    real = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert real.exit_code == 0, real.output
    plain = " ".join(real.output.split())
    assert migrate.INIT_NULLS_CLAUSE + " (config.yaml)" in plain
    assert migrate.INIT_NULLS_CLAUSE + " (agents/scout.yaml)" in plain
    assert "cleaned" not in plain
    again = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert "up to date" in again.output.lower()


# ------------------------------------------------------------------- safety edges (Nyquist)


def test_an_unparseable_or_non_mapping_sidecar_is_skipped_and_the_rest_still_migrates(tmp_path):
    cfg = _write(tmp_path, _v017())
    broken = tmp_path / "agents" / "broken.yaml"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_bytes(b"name: broken\npermissions: {budget: {max_actions: 100\n  :: [")
    listed = tmp_path / "agents" / "listed.yaml"
    listed.write_bytes(b"- max_actions: 100\n")
    good = _sidecar(tmp_path, "agents", NAME, {"name": NAME, "permissions": {"budget": {"max_actions": 100}}})
    broken_before, listed_before = broken.read_bytes(), listed.read_bytes()

    plan, _backups = _migrate(cfg)

    assert [s.path for s in plan.sidecars] == [good]
    assert broken.read_bytes() == broken_before and listed.read_bytes() == listed_before
    assert list((tmp_path / "agents").glob("broken.yaml.bak-*")) == []
    assert list((tmp_path / "agents").glob("listed.yaml.bak-*")) == []
    assert _disk(good)["permissions"]["budget"]["max_actions"] is None
    assert _disk(cfg)["org"]["permissions"]["budget"]["max_actions"] is None


def test_an_unparseable_config_raises_and_writes_nothing(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(b"org: {permissions: {budget: {max_actions: 100\n  :: [")
    before = cfg.read_bytes()
    import pytest

    with pytest.raises(migrate.MigrationError):
        migrate.load_plan(cfg)
    assert cfg.read_bytes() == before
    assert list(tmp_path.glob("config.yaml.bak-*")) == []


def test_an_owner_changed_sidecar_value_stays_while_inits_siblings_are_nulled(tmp_path):
    cfg = _write(tmp_path, _v017())
    agent = _sidecar(tmp_path, "agents", NAME, {
        "name": NAME, "permissions": {"budget": {"max_actions": 150}},
        "context": {"max_tool_output_chars": 32000, "tool_result_evict_threshold_chars": 9000}})

    plan, _backups = _migrate(cfg)

    (sidecar,) = plan.sidecars
    assert sidecar.nulled == ("context.max_tool_output_chars",)
    written = _disk(agent)
    assert written["permissions"]["budget"]["max_actions"] == 150
    assert written["context"] == {"max_tool_output_chars": None, "tool_result_evict_threshold_chars": 9000}


def test_two_starts_make_one_backup_per_file_and_the_second_changes_nothing(tmp_path):
    cfg = _write(tmp_path, _v017())
    agent = _sidecar(tmp_path, "agents", NAME, _pre_v014_agent())
    cfg_original, agent_original = cfg.read_bytes(), agent.read_bytes()

    _start_receipt(cfg)
    after_first = (cfg.read_bytes(), agent.read_bytes())
    _start_receipt(cfg)

    assert (cfg.read_bytes(), agent.read_bytes()) == after_first
    cfg_backups = list(tmp_path.glob("config.yaml.bak-*"))
    agent_backups = list((tmp_path / "agents").glob(f"{NAME}.yaml.bak-*"))
    assert len(cfg_backups) == 1 and cfg_backups[0].read_bytes() == cfg_original
    assert len(agent_backups) == 1 and agent_backups[0].read_bytes() == agent_original


def test_a_rewrite_that_fails_mid_write_leaves_the_original_bytes_and_no_stray_backup(tmp_path, monkeypatch):
    """The rewrite is atomic (WR-01), so a write that dies part-way leaves the original bytes where
    they were and the temp file is gone. The backup made for it is removed again (WR-20): it is no
    undo for an unchanged file, and kept it would pile up once per failed start. This used to pin
    that the backup survived the failure."""
    import pytest

    cfg = _write(tmp_path, _v017())
    original = cfg.read_bytes()

    def dying_fsync(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(overlay.os, "fsync", dying_fsync)
    raw, plan = migrate.load_plan(cfg)
    with pytest.raises(migrate.PartialMigrationError) as caught:  # WR-02: names what failed
        migrate.apply(cfg, raw, plan)
    assert caught.value.failed == cfg

    assert list(tmp_path.glob("config.yaml.bak-*")) == []
    assert cfg.read_bytes() == original
    assert not list(tmp_path.glob(".config.yaml.*.tmp"))


def test_a_migrated_config_that_fails_validation_writes_nothing(tmp_path):
    import pytest

    data = _v017()
    data["org"]["permissions"]["budget"]["max_duration_minutes"] = -5  # invalid before and after
    cfg = _write(tmp_path, data)
    before = cfg.read_bytes()
    raw, plan = migrate.load_plan(cfg)
    with pytest.raises(migrate.MigrationError):
        migrate.apply(cfg, raw, plan)
    assert cfg.read_bytes() == before and list(tmp_path.glob("config.yaml.bak-*")) == []


# ------------------------------------------- WR-02: a partial failure never skips a file for good

import os  # noqa: E402

import pytest  # noqa: E402

_not_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                               reason="root writes a read-only file")


def _locked_world(tmp_path: Path) -> tuple[Path, Path, Path]:
    cfg = _write(tmp_path, _v017())
    ok = _sidecar(tmp_path, "agents", "a_ok", {"name": "a_ok", "permissions": {"budget": {"max_actions": 100}}})
    locked = _sidecar(tmp_path, "agents", "b_locked",
                      {"name": "b_locked", "permissions": {"budget": {"max_actions": 100}}})
    locked.chmod(0o444)
    return cfg, ok, locked


@_not_root
def test_a_failed_sidecar_is_recorded_as_owed_and_retried_alone(tmp_path):
    # CR-20 changed this pin: config.yaml used to be left byte-for-byte unwritten. CR-30 changed it
    # again: the stamp is no longer held back (that re-ran the match over config.yaml at every
    # start); it moves, and the failed file is recorded in the pending file instead.
    cfg, ok, locked = _locked_world(tmp_path)
    original, plan = migrate.load_plan(cfg)

    with pytest.raises(migrate.PartialMigrationError) as caught:
        migrate.apply(cfg, original, plan)

    assert caught.value.failed == locked and caught.value.written == (ok, cfg)
    assert [p for p, _ in caught.value.failures] == [locked]
    assert len(caught.value.backups) == 2
    on_disk = _disk(cfg)["org"]
    assert on_disk["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert migrate.read_init_nulls_pending(tmp_path) == ("agents/b_locked.yaml",)
    assert on_disk["permissions"]["budget"]["max_actions"] is None  # config.yaml's own values did
    assert _disk(ok)["permissions"]["budget"]["max_actions"] is None
    assert _disk(locked)["permissions"]["budget"]["max_actions"] == 100

    # The next run still sees the locked file; once it is writable it is nulled and stamped.
    _, again = migrate.load_plan(cfg)
    assert again is not None and [s.path for s in again.sidecars] == [locked]
    locked.chmod(0o644)
    _migrate(cfg)
    assert _disk(locked)["permissions"]["budget"]["max_actions"] is None
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
    assert migrate.load_plan(cfg)[1] is None


def test_config_yaml_is_written_after_every_sidecar(tmp_path):
    cfg = _write(tmp_path, _v017())
    side = _sidecar(tmp_path, "agents", NAME, {"name": NAME, "permissions": {"budget": {"max_actions": 100}}})
    _, backups = _migrate(cfg)
    assert [b.name.split(migrate.BACKUP_INFIX)[0] for b in backups] == [side.name, cfg.name]


@_not_root
def test_the_start_warning_names_what_was_rewritten_and_its_advice_holds(tmp_path):
    import contextlib

    cfg, ok, locked = _locked_world(tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    plain = " ".join(err.getvalue().split())
    assert "Could not finish updating the config" in plain
    assert str(locked) in plain and f"rewrote {ok}" in plain
    assert "Run 'localharness config migrate' to finish" in plain

    locked.chmod(0o644)
    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "up to date" not in result.output.lower()  # the advice is true: there was work left
    assert _disk(locked)["permissions"]["budget"]["max_actions"] is None


@_not_root
def test_config_migrate_names_what_it_wrote_and_never_says_refusing(tmp_path):
    _, ok, locked = _locked_world(tmp_path)
    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    plain = " ".join(result.output.split())
    assert result.exit_code == 1
    assert f"Rewrote {ok}" in plain and "Could not finish" in plain
    assert "Refusing to write" not in plain


# ------------------------------------------- IN-03: a stamp-only rewrite says why it happened


def test_a_stamp_only_rewrite_says_it_recorded_the_stamp(tmp_path):
    cfg = _write(tmp_path, _v017(max_actions=150, output=20000, evict=5000))
    out = " ".join(_start_receipt(cfg).split())
    assert " ".join(migrate.INIT_NULLS_STAMP_CLAUSE.split()) in out


def test_a_rewrite_that_cleared_values_does_not_also_claim_stamp_only(tmp_path):
    cfg = _write(tmp_path, _v017())
    out = " ".join(_start_receipt(cfg).split())
    assert "nothing of init's to clear" not in out


def test_config_migrate_dry_run_and_help_announce_the_init_nulls_step(tmp_path):
    _write(tmp_path, _v017(max_actions=150, output=20000, evict=5000))
    dry = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "org.permissions.init_nulls_revision →" in " ".join(dry.output.split())
    real = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert " ".join(migrate.INIT_NULLS_STAMP_CLAUSE.split()) in " ".join(real.output.split())
    helped = runner.invoke(app, ["config", "migrate", "--help"])
    assert "init_nulls_revision" in " ".join(helped.output.split())


# ------------------------------------------- WR-20: a file that keeps failing gains no backups


@_not_root
def test_a_sidecar_that_keeps_failing_gains_no_backup_however_often_it_is_tried(tmp_path):
    import contextlib

    cfg, _ok, locked = _locked_world(tmp_path)
    for _ in range(3):  # three starts
        with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            _auto_migrate_deny_defaults(cfg)
    assert list(locked.parent.glob(f"{locked.name}{migrate.BACKUP_INFIX}*")) == []
    assert _disk(locked)["permissions"]["budget"]["max_actions"] == 100


@_not_root
def test_an_unwritable_directory_is_refused_before_any_backup(tmp_path):
    agents = tmp_path / "agents"
    agents.mkdir()
    agent = agents / "scout.yaml"
    agent.write_text("name: scout\n", encoding="utf-8")
    agents.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            migrate._write_with_backup(agent, agent.read_bytes(), {"name": "other"})
        assert [p.name for p in agents.iterdir()] == ["scout.yaml"]
    finally:
        agents.chmod(0o755)


# ------------------------------------------- CR-20: an unwritable sidecar never holds config.yaml back


def _behind_world(tmp_path: Path) -> tuple[Path, Path]:
    """config.yaml at defaults revision 0 with 2 of the shipped deny patterns, plus a 0444 agent
    file still holding init's `max_actions: 100`."""
    shipped = list(PermissionConfig().deny_patterns)
    data = _v017(deny=shipped[:2])
    data["org"]["permissions"]["defaults_revision"] = 0
    cfg = _write(tmp_path, data)
    locked = _sidecar(tmp_path, "agents", "locked",
                      {"name": "locked", "permissions": {"budget": {"max_actions": 100}}})
    locked.chmod(0o444)
    return cfg, locked


@_not_root
def test_an_unwritable_sidecar_never_holds_back_the_deny_defaults(tmp_path):
    import contextlib

    cfg, locked = _behind_world(tmp_path)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)

    perms = _disk(cfg)["org"]["permissions"]
    assert perms["deny_patterns"] == list(PermissionConfig().deny_patterns)  # every default landed
    assert perms["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    # CR-30 changed this pin (it was "the stamp stays due"): the stamp moves, `locked` is owed.
    assert perms["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert migrate.read_init_nulls_pending(tmp_path) == ("agents/locked.yaml",)
    assert "Could not finish updating the config" in " ".join(err.getvalue().split())

    # The file is retried, not skipped: the next plan still names it, and once it is writable the
    # step finishes and stamps.
    _, again = migrate.load_plan(cfg)
    assert again is not None and [s.path for s in again.sidecars] == [locked]
    locked.chmod(0o644)
    _migrate(cfg)
    assert _disk(locked)["permissions"]["budget"]["max_actions"] is None
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert migrate.load_plan(cfg)[1] is None


@_not_root
def test_a_sidecar_that_keeps_failing_costs_config_yaml_no_rewrite_per_start(tmp_path):
    import contextlib

    cfg, locked = _behind_world(tmp_path)
    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    after_first = cfg.read_bytes()
    for _ in range(2):
        with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            _auto_migrate_deny_defaults(cfg)
    assert cfg.read_bytes() == after_first
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1
    assert list((tmp_path / "agents").glob("locked.yaml.bak-*")) == []


@_not_root
def test_every_sidecar_is_tried_even_after_one_fails(tmp_path):
    cfg = _write(tmp_path, _v017())
    first = _sidecar(tmp_path, "agents", "a_locked",
                     {"name": "a_locked", "permissions": {"budget": {"max_actions": 100}}})
    first.chmod(0o444)
    later = _sidecar(tmp_path, "divisions", "z_later",
                     {"name": "z_later", "permissions": {"budget": {"max_actions": 100}}})
    original, plan = migrate.load_plan(cfg)
    with pytest.raises(migrate.PartialMigrationError) as caught:
        migrate.apply(cfg, original, plan)
    assert caught.value.written == (later, cfg)
    assert _disk(later)["permissions"]["budget"]["max_actions"] is None


@_not_root
def test_a_failed_dead_key_only_sidecar_still_lets_the_init_nulls_stamp_move(tmp_path):
    """Only a file the init-nulls step clears holds that stamp back; the dead key is found on
    content alone, so a file failing only that repair is retried without holding anything."""
    cfg = _write(tmp_path, _v017())
    dead = _sidecar(tmp_path, "agents", "dead",
                    {"name": "dead", "permissions": {"allow_patterns": []}})
    dead.chmod(0o444)
    original, plan = migrate.load_plan(cfg)
    with pytest.raises(migrate.PartialMigrationError):
        migrate.apply(cfg, original, plan)
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    _, again = migrate.load_plan(cfg)
    assert again is not None and [s.path for s in again.sidecars] == [dead]


# ------------------------------- CR-30: a hand-set value survives while an owed sidecar stays locked


@_not_root
def test_a_hand_set_100_survives_every_start_while_a_sidecar_stays_unwritable(tmp_path):
    """The owner sets `max_actions: 100` by hand, in config.yaml and in an agent file that was
    written, after the first start. However many starts follow while another agent file stays
    read-only, both values stay; the locked file is still cleared once it can be written."""
    import contextlib

    cfg, ok, locked = _locked_world(tmp_path)
    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)  # start 1: the step runs, `locked` is owed
    assert _disk(ok)["permissions"]["budget"]["max_actions"] is None

    data = _disk(cfg)
    data["org"]["permissions"]["budget"]["max_actions"] = 100  # the owner's own choice now
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    ok.write_text(yaml.safe_dump({"name": "a_ok", "permissions": {"budget": {"max_actions": 100}}},
                                 sort_keys=False), encoding="utf-8")
    config_bytes = cfg.read_bytes()

    for _ in range(3):  # starts 2..4, `locked` still read-only
        err = io.StringIO()
        with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
            _auto_migrate_deny_defaults(cfg)
        assert "b_locked.yaml" in err.getvalue()  # still retried, and said
        assert _disk(cfg)["org"]["permissions"]["budget"]["max_actions"] == 100
        assert _disk(ok)["permissions"]["budget"]["max_actions"] == 100
        assert cfg.read_bytes() == config_bytes  # not even rewritten

    locked.chmod(0o644)
    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    assert _disk(locked)["permissions"]["budget"]["max_actions"] is None  # owed, so cleared once
    assert _disk(cfg)["org"]["permissions"]["budget"]["max_actions"] == 100
    assert _disk(ok)["permissions"]["budget"]["max_actions"] == 100
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
    assert migrate.load_plan(cfg)[1] is None

    # And never again: a 100 the owner now writes into the formerly locked file stays too.
    locked.write_text(yaml.safe_dump({"name": "b_locked",
                                      "permissions": {"budget": {"max_actions": 100}}}),
                      encoding="utf-8")
    assert migrate.load_plan(cfg)[1] is None


@_not_root
def test_an_owed_file_fixed_by_hand_drops_off_the_pending_list(tmp_path):
    """A listed file that no longer holds init's values is struck off, so a 100 the owner writes
    into it later is never matched."""
    import contextlib

    cfg, _ok, locked = _locked_world(tmp_path)
    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    locked.chmod(0o644)
    locked.write_text(yaml.safe_dump({"name": "b_locked",
                                      "permissions": {"budget": {"max_actions": 150}}}),
                      encoding="utf-8")
    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
    assert _disk(locked)["permissions"]["budget"]["max_actions"] == 150
    locked.write_text(yaml.safe_dump({"name": "b_locked",
                                      "permissions": {"budget": {"max_actions": 100}}}),
                      encoding="utf-8")
    assert migrate.load_plan(cfg)[1] is None


def test_the_pending_list_names_only_files_under_the_sidecar_dirs(tmp_path):
    """A line naming anything else is inert: only globbed agents/divisions files are matched."""
    cfg = _write(tmp_path, _v017(stamp=CURRENT_INIT_NULLS_REVISION, max_actions=150))
    other = tmp_path / "elsewhere.yaml"
    other.write_text(yaml.safe_dump({"permissions": {"budget": {"max_actions": 100}}}),
                     encoding="utf-8")
    (tmp_path / migrate.INIT_NULLS_PENDING_FILE).write_text("elsewhere.yaml\n../x.yaml\n",
                                                           encoding="utf-8")
    original, plan = migrate.load_plan(cfg)
    assert plan is not None and plan.sidecars == () and plan.config_unchanged
    assert migrate.apply(cfg, original, plan) == []
    assert _disk(other)["permissions"]["budget"]["max_actions"] == 100
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()


# ------------------------------- WR-30: a partial migration's receipt names config.yaml's changes


@_not_root
def test_a_partial_receipt_names_every_change_config_yaml_got(tmp_path):
    """config.yaml IS written on the partial path, so its clauses (the cleared budget value, the
    bind_all key that keeps a server on the network) are said there, as on the success path; the
    locked file's own clause is not, because it was not written."""
    import contextlib

    data = _v017()
    data["server"] = {"runtime": "vllm", "launch": "binary", "binary": "/usr/bin/vllm", "model": "m",
                      "port": 8081}
    cfg = _write(tmp_path, data)
    locked = _sidecar(tmp_path, "agents", "locked",
                      {"name": "locked", "permissions": {"budget": {"max_actions": 100}}})
    locked.chmod(0o444)
    err = io.StringIO()
    with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)
    text = " ".join(err.getvalue().split())
    assert "Could not finish updating the config" in text
    assert " ".join(migrate.INIT_NULLS_CLAUSE.split()) + " (config.yaml)" in text
    assert "server.bind_all: true" in text and "reachable from other machines" in text
    assert "(agents/locked.yaml)" not in text
    assert "continuing; config.yaml was updated." in text


@_not_root
def test_an_unreadable_sidecar_is_neither_tried_nor_owed(tmp_path):
    """IN-32: what the docstrings now say. A file migrate cannot read is not in the plan and not
    recorded as owed; the stamp moves, so once readable it keeps init's value (a cap stays)."""
    cfg = _write(tmp_path, _v017())
    hidden = _sidecar(tmp_path, "agents", "hidden",
                      {"name": "hidden", "permissions": {"budget": {"max_actions": 100}}})
    hidden.chmod(0o000)
    try:
        original, plan = migrate.load_plan(cfg)
        assert plan is not None and hidden not in [s.path for s in plan.sidecars]
        migrate.apply(cfg, original, plan)
        assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
        assert migrate.read_init_nulls_pending(tmp_path) == ()
    finally:
        hidden.chmod(0o644)
    assert _disk(hidden)["permissions"]["budget"]["max_actions"] == 100
    assert migrate.load_plan(cfg)[1] is None


# ------------------------- SEC-01: an unwritable config.yaml clears no agent file's init values


def _read_only_config(tmp_path: Path, kind: str, data: dict) -> tuple[Path, "callable"]:
    """config.yaml the migration cannot write: a 0444 file, or a link into a read-only directory
    (the store-link shape). Returns the config path and a callable that makes it writable."""
    if kind == "file":
        cfg = _write(tmp_path, data)
        cfg.chmod(0o444)
        return cfg, lambda: cfg.chmod(0o644)
    store = tmp_path / "store"
    store.mkdir()
    real = store / "config.yaml"
    real.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    store.chmod(0o555)
    cfg = tmp_path / "config.yaml"
    cfg.symlink_to(real)
    return cfg, lambda: store.chmod(0o755)


@_not_root
@pytest.mark.parametrize("kind", ["file", "store_link"])
def test_an_owner_set_100_is_never_cleared_twice_while_config_yaml_is_read_only(tmp_path, kind):
    """SEC-01: config.yaml cannot be written, so the init-nulls stamp cannot move. Clearing an
    agent file anyway would match it again at the next start and clear a `100` the owner put back.
    No agent file is cleared until config.yaml can be written; then the step runs once, whole."""
    import contextlib

    data = _v017()
    data["org"]["permissions"]["defaults_revision"] = 0  # deny defaults owed as well
    cfg, make_writable = _read_only_config(tmp_path, kind, data)
    agent = _sidecar(tmp_path, "agents", "a",
                     {"name": "a", "permissions": {"budget": {"max_actions": 100}}})
    try:
        for _ in range(2):  # start 1, then (after the owner re-sets 100) start 2
            err = io.StringIO()
            with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
                _auto_migrate_deny_defaults(cfg)
            text = " ".join(err.getvalue().split())
            assert "Could not finish updating the config" in text  # the deny defaults did not land
            assert "continuing with the current config." in text
            assert "(agents/a.yaml)" not in text  # no clearing is claimed
            assert _disk(agent)["permissions"]["budget"]["max_actions"] == 100
            assert not list(agent.parent.glob(f"a.yaml{migrate.BACKUP_INFIX}*"))
            assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
            agent.write_text(yaml.safe_dump({"name": "a",
                                             "permissions": {"budget": {"max_actions": 100}}}),
                             encoding="utf-8")  # the owner sets 100 (again) by hand
        assert _disk(cfg)["org"]["permissions"].get("init_nulls_revision") is None
    finally:
        make_writable()

    with contextlib.redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        _auto_migrate_deny_defaults(cfg)  # config.yaml writable: the step runs once and stamps
    perms = _disk(cfg)["org"]["permissions"]
    assert perms["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
    assert perms["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    assert _disk(agent)["permissions"]["budget"]["max_actions"] is None
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
    assert migrate.load_plan(cfg)[1] is None
    agent.write_text(yaml.safe_dump({"name": "a", "permissions": {"budget": {"max_actions": 100}}}),
                     encoding="utf-8")
    assert migrate.load_plan(cfg)[1] is None  # and never again


@_not_root
def test_a_read_only_config_yaml_still_lets_a_sidecar_lose_the_dead_key_but_not_inits_values(tmp_path):
    """The dead key's removal is found on content and needs no stamp, so it still lands; init's
    values in the same file wait for the run that can record the stamp."""
    import contextlib

    cfg, make_writable = _read_only_config(tmp_path, "file", _v017())
    agent = _sidecar(tmp_path, "agents", NAME, _pre_v014_agent())
    try:
        err = io.StringIO()
        with contextlib.redirect_stderr(err), redirect_stdout(io.StringIO()):
            _auto_migrate_deny_defaults(cfg)
        text = " ".join(err.getvalue().split())
        on_disk = _disk(agent)
        assert migrate.DEAD_KEY not in on_disk["permissions"]
        assert on_disk["permissions"]["budget"]["max_actions"] == 100
        assert on_disk["context"]["max_tool_output_chars"] == 32000
        assert f"cleaned {migrate.DEAD_KEY} from {agent.name}" in text
        assert f"(agents/{agent.name})" not in text
    finally:
        make_writable()
    _migrate(cfg)
    assert _disk(agent)["permissions"]["budget"]["max_actions"] is None
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == CURRENT_INIT_NULLS_REVISION
