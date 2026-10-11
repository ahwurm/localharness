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
from localharness.config import migrate, trust
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
    cfg = _write(tmp_path, _v017(stamp=CURRENT_INIT_NULLS_REVISION - 1))
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


def test_a_rewrite_that_fails_mid_write_leaves_the_original_bytes_in_a_backup(tmp_path, monkeypatch):
    """The backup lands before the file is touched: a write that dies part-way still has an undo."""
    import pytest

    cfg = _write(tmp_path, _v017())
    original = cfg.read_bytes()
    real_write_text = Path.write_text

    def dying_write(self, data, *a, **kw):
        if self == cfg:
            real_write_text(self, data[: len(data) // 2], *a, **kw)
            raise OSError("disk full")
        return real_write_text(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", dying_write)
    raw, plan = migrate.load_plan(cfg)
    with pytest.raises(OSError):
        migrate.apply(cfg, raw, plan)

    (backup,) = tmp_path.glob("config.yaml.bak-*")
    assert backup.read_bytes() == original


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
