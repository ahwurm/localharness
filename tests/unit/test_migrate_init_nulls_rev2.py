"""init-nulls revision 2: the temperature `init` dumped (`org.default_temperature: 0.6`) becomes null.

`init` wrote `org.default_temperature: 0.6` into every config.yaml as if the person had chosen it;
the value was never applied, and with the temperature rung switched on it would start overriding the
model's own sampling. Revision 2 nulls exactly that float, once, in config.yaml and in the global
overrides.yaml, with the same backup, receipt and read-only retry revision 1 has. Revision 2's key
is scoped to the `org:` section: an agent or division file's own `temperature` is never touched, and
a config stamped at revision 1 clears nothing revision 1 cleared before.
"""
from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.cli.start_cmd import _auto_migrate_deny_defaults
from localharness.config import defaults, migrate
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION, CURRENT_INIT_NULLS_REVISION
from localharness.config.models import HarnessConfig, PermissionConfig

runner = CliRunner()

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m",
            "api_key": "none"}
CLAUSE = ("; org.default_temperature nulled (it was never applied; the model's own sampling now "
          "governs — set org.default_temperature: 0.6 to restore)")
_not_root = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                               reason="root writes a read-only file")


def _config(*, stamp=1, temperature=0.6, max_actions=100, **extra) -> dict:
    """A 0.18-shaped config: revision 1 already run (so `max_actions` is the owner's), init's 0.6 still in."""
    perms = {"defaults_revision": CURRENT_DEFAULTS_REVISION,
             "deny_patterns": list(PermissionConfig().deny_patterns),
             "budget": {"max_actions": max_actions}}
    if stamp is not None:
        perms["init_nulls_revision"] = stamp
    return {"version": "1", "provider": dict(PROVIDER),
            "org": {"default_temperature": temperature, "permissions": perms,
                    "context": {"max_tool_output_chars": 32000,
                                "tool_result_evict_threshold_chars": 8000}},
            **extra}


def _write(tmp_path: Path, data: dict) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return cfg


def _put(tmp_path: Path, rel: str, data: dict) -> Path:
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _migrate(cfg: Path):
    original, plan = migrate.load_plan(cfg)
    assert plan is not None
    return plan, migrate.apply(cfg, original, plan)


def _disk(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _start_receipt(cfg: Path) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _auto_migrate_deny_defaults(cfg)
    return " ".join(buffer.getvalue().split())


def _backups(directory: Path, name: str) -> list[Path]:
    return sorted(directory.glob(f"{name}.bak-*"))


# ------------------------------------------------------------------------------- the constants


def test_the_revision_is_two_and_the_unused_temperature_constant_is_gone():
    assert CURRENT_INIT_NULLS_REVISION == 2
    assert not hasattr(defaults, "DEFAULT_" + "TEMPERATURE")


def test_the_clause_text_is_verbatim_and_the_keys_are_revision_keyed():
    assert migrate.INIT_NULLS_TEMPERATURE_CLAUSE == CLAUSE
    assert (2, "org", ("default_temperature",), 0.6) in migrate.INIT_NULL_KEYS
    assert [k[0] for k in migrate.INIT_NULL_KEYS if k[1] == "any"] == [1, 1, 1]
    assert migrate.init_nulls_clauses(()) == ""
    assert migrate.init_nulls_clauses(("org.default_temperature",)) == CLAUSE
    both = migrate.init_nulls_clauses(("org.permissions.budget.max_actions", "org.default_temperature"))
    assert both == migrate.INIT_NULLS_CLAUSE + CLAUSE


# ----------------------------------------------------------------------------------- config.yaml


def test_a_revision_1_config_gets_only_the_temperature_nulled_and_the_stamp_moved(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1, max_actions=100))

    plan, _backups_written = _migrate(cfg)

    assert plan.nulled == ("org.default_temperature",) and plan.init_nulls_due
    org = _disk(cfg)["org"]
    assert org["default_temperature"] is None
    assert org["permissions"]["budget"]["max_actions"] == 100  # the owner's, set after revision 1
    assert org["context"]["max_tool_output_chars"] == 32000
    assert org["permissions"]["init_nulls_revision"] == 2
    assert len(_backups(tmp_path, "config.yaml")) == 1
    HarnessConfig.model_validate(_disk(cfg))


def test_an_unstamped_config_gets_all_four_values_nulled(tmp_path):
    cfg = _write(tmp_path, _config(stamp=None))

    plan, _ = _migrate(cfg)

    assert plan.nulled == ("org.permissions.budget.max_actions", "org.context.max_tool_output_chars",
                           "org.context.tool_result_evict_threshold_chars", "org.default_temperature")
    org = _disk(cfg)["org"]
    assert org["default_temperature"] is None and org["permissions"]["budget"]["max_actions"] is None
    assert org["permissions"]["init_nulls_revision"] == 2


@pytest.mark.parametrize("kept", [0.5, 0.60000001, 1, True, "0.6"])
def test_any_other_value_stays_as_written(tmp_path, kept):
    cfg = _write(tmp_path, _config(stamp=1, temperature=kept))
    before = _disk(cfg)["org"]["default_temperature"]

    plan, _ = _migrate(cfg)

    assert plan.nulled == () and plan.init_nulls_due  # a stamp-only rewrite
    after = _disk(cfg)["org"]
    assert after["default_temperature"] == before and type(after["default_temperature"]) is type(before)
    assert after["permissions"]["init_nulls_revision"] == 2


def test_a_config_stamped_2_is_not_touched(tmp_path):
    cfg = _write(tmp_path, _config(stamp=2))
    before = cfg.read_bytes()

    assert migrate.load_plan(cfg)[1] is None
    assert cfg.read_bytes() == before and _backups(tmp_path, "config.yaml") == []


def test_the_config_cmd_receipt_names_the_temperature_clause_and_not_revision_1s(tmp_path):
    _write(tmp_path, _config(stamp=1))

    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])

    out = " ".join(result.output.split())
    assert result.exit_code == 0, result.output
    assert CLAUSE + " (config.yaml)" in out
    assert migrate.INIT_NULLS_CLAUSE not in out
    assert _disk(tmp_path / "config.yaml")["org"]["default_temperature"] is None


def test_the_first_start_receipt_names_the_temperature_clause(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))

    out = _start_receipt(cfg)

    assert CLAUSE + " (config.yaml)" in out and migrate.INIT_NULLS_CLAUSE not in out
    assert _start_receipt(cfg) == ""  # once


# ------------------------------------------------------------------------- agents and divisions


@pytest.mark.parametrize("stamp", [1, None])
def test_an_agent_and_a_division_temperature_stay_byte_identical(tmp_path, stamp):
    cfg = _write(tmp_path, _config(stamp=stamp))
    agent = _put(tmp_path, "agents/x.yaml", {"name": "x", "temperature": 0.6})
    division = _put(tmp_path, "divisions/y.yaml", {"name": "y", "temperature": 0.6})
    before = (agent.read_bytes(), division.read_bytes())

    plan, _ = _migrate(cfg)

    assert plan.sidecars == ()
    assert (agent.read_bytes(), division.read_bytes()) == before
    assert _backups(tmp_path / "agents", "x.yaml") == [] and _backups(tmp_path / "divisions", "y.yaml") == []


def test_a_revision_1_config_clears_nothing_revision_1_cleared_in_a_sidecar(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    agent = _put(tmp_path, "agents/x.yaml", {"name": "x", "permissions": {"budget": {"max_actions": 100}}})
    before = agent.read_bytes()

    plan, _ = _migrate(cfg)

    assert plan.sidecars == () and agent.read_bytes() == before


def test_an_unstamped_config_still_clears_revision_1s_values_in_a_sidecar(tmp_path):
    cfg = _write(tmp_path, _config(stamp=None))
    agent = _put(tmp_path, "agents/x.yaml", {"name": "x", "temperature": 0.6,
                                             "permissions": {"budget": {"max_actions": 100}}})

    _migrate(cfg)

    written = _disk(agent)
    assert written["permissions"]["budget"]["max_actions"] is None and written["temperature"] == 0.6


@_not_root
def test_a_sidecar_owed_since_revision_1_is_still_retried_when_revision_2_is_due(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    locked = _put(tmp_path, "agents/locked.yaml", {"name": "locked", "permissions": {"budget": {"max_actions": 100}}})
    (tmp_path / migrate.INIT_NULLS_PENDING_FILE).write_text("agents/locked.yaml\n", encoding="utf-8")

    _migrate(cfg)

    assert _disk(locked)["permissions"]["budget"]["max_actions"] is None
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()


# ------------------------------------------------------------------------------- overrides.yaml


def test_overrides_yaml_holding_the_dumped_temperature_is_nulled_with_a_backup_and_the_clause(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6, "default_model": "m"},
                                                  "agent": {"temperature": 0.6}})
    original = overrides.read_bytes()

    plan, _ = _migrate(cfg)

    assert [s.path for s in plan.sidecars] == [overrides]
    written = _disk(overrides)
    assert written["org"] == {"default_temperature": None, "default_model": "m"}
    assert written["agent"] == {"temperature": 0.6}  # the overlay's agent section is never a target
    (backup,) = _backups(tmp_path, "overrides.yaml")
    assert backup.read_bytes() == original
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == 2


def test_the_first_start_receipt_names_overrides_yaml(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})

    out = _start_receipt(cfg)

    assert CLAUSE + " (config.yaml)" in out and CLAUSE + " (overrides.yaml)" in out


def test_the_config_cmd_receipt_names_overrides_yaml(tmp_path):
    _write(tmp_path, _config(stamp=1))
    _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})

    result = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert CLAUSE + " (overrides.yaml)" in " ".join(result.output.split())


@pytest.mark.parametrize("content", [
    {"org": {"default_temperature": 0.7}},
    {"org": {"default_model": "m"}},
    {"agent": {"temperature": 0.6}},
    {"org": {"permissions": {"budget": {"max_actions": 100}}}},  # revision 1 never targeted the overlay
])
def test_overrides_yaml_without_the_dumped_value_is_not_rewritten(tmp_path, content):
    cfg = _write(tmp_path, _config(stamp=None))
    overrides = _put(tmp_path, "overrides.yaml", content)
    before = overrides.read_bytes()

    plan, _ = _migrate(cfg)

    assert all(s.path != overrides for s in plan.sidecars)
    assert overrides.read_bytes() == before and _backups(tmp_path, "overrides.yaml") == []


def test_an_unparsable_overrides_yaml_is_not_rewritten(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    overrides = tmp_path / "overrides.yaml"
    overrides.write_text("org: [unclosed\n", encoding="utf-8")
    before = overrides.read_bytes()

    _migrate(cfg)

    assert overrides.read_bytes() == before and _backups(tmp_path, "overrides.yaml") == []


def test_a_config_stamped_2_never_touches_overrides_yaml(tmp_path):
    cfg = _write(tmp_path, _config(stamp=2))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})
    before = overrides.read_bytes()

    assert migrate.load_plan(cfg)[1] is None
    assert overrides.read_bytes() == before


@_not_root
def test_a_read_only_overrides_yaml_is_owed_and_retried_on_the_next_run(tmp_path):
    cfg = _write(tmp_path, _config(stamp=1))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})
    overrides.chmod(0o444)
    original, plan = migrate.load_plan(cfg)

    with pytest.raises(migrate.PartialMigrationError) as caught:
        migrate.apply(cfg, original, plan)

    assert caught.value.failed == overrides and cfg in caught.value.written
    assert _disk(cfg)["org"]["default_temperature"] is None
    assert _disk(cfg)["org"]["permissions"]["init_nulls_revision"] == 2
    assert migrate.read_init_nulls_pending(tmp_path) == ("overrides.yaml",)
    assert _disk(overrides)["org"]["default_temperature"] == 0.6

    overrides.chmod(0o644)
    again, retry = migrate.load_plan(cfg)
    assert retry is not None and [s.path for s in retry.sidecars] == [overrides] and retry.config_unchanged
    migrate.apply(cfg, again, retry)
    assert _disk(overrides)["org"]["default_temperature"] is None
    assert not (tmp_path / migrate.INIT_NULLS_PENDING_FILE).exists()
    assert migrate.load_plan(cfg)[1] is None


@_not_root
def test_a_hand_set_max_actions_in_a_retried_overrides_yaml_is_not_cleared(tmp_path):
    cfg = _write(tmp_path, _config(stamp=2))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6,
                                                         "permissions": {"budget": {"max_actions": 100}}}})
    (tmp_path / migrate.INIT_NULLS_PENDING_FILE).write_text("overrides.yaml\n", encoding="utf-8")

    _migrate(cfg)

    org = _disk(overrides)["org"]
    assert org["default_temperature"] is None and org["permissions"]["budget"]["max_actions"] == 100


# ------------------------------------------------------------------------------- a re-init (WR-07)


def _reinit(tmp_path: Path, *, force: bool = True) -> str:
    """`init`'s write over `tmp_path` with the stamp current (what a re-init writes); its output."""
    from rich.console import Console

    from localharness.cli import init_cmd
    from localharness.config.models import OrgConfig

    out = io.StringIO()
    original_console = init_cmd.console
    init_cmd.console = Console(file=out, width=400)
    try:
        init_cmd._write_harness(tmp_path, HarnessConfig.model_validate({
            "version": "1", "provider": dict(PROVIDER),
            "org": OrgConfig(permissions=PermissionConfig(
                init_nulls_revision=CURRENT_INIT_NULLS_REVISION)).model_dump()}), True, force=force)
    finally:
        init_cmd.console = original_console
    return " ".join(out.getvalue().split())


@pytest.mark.parametrize("stamp", [None, 1])
def test_a_reinit_below_revision_2_nulls_the_dumped_temperature_in_overrides_yaml(tmp_path, stamp):
    _write(tmp_path, _config(stamp=stamp))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6, "default_model": "m"}})
    original = overrides.read_bytes()

    out = _reinit(tmp_path, force=False)

    assert _disk(overrides)["org"] == {"default_temperature": None, "default_model": "m"}
    (backup,) = _backups(tmp_path, "overrides.yaml")
    assert backup.read_bytes() == original
    assert f"Updated overrides.yaml{CLAUSE} — backup at {backup}" in out
    later = migrate.load_plan(tmp_path / "config.yaml")[1]  # nothing about overrides.yaml is left owed
    assert later is None or (all(s.path != overrides for s in later.sidecars) and not later.init_nulls_due)
    assert migrate.read_init_nulls_pending(tmp_path) == ()


def test_a_force_reinit_that_resets_saved_choices_keeps_the_temperature_nulled(tmp_path):
    _write(tmp_path, _config(stamp=1))
    overrides = _put(tmp_path, "overrides.yaml", {
        "org": {"default_temperature": 0.6}, "provider": {"default_model": "saved"}})

    _reinit(tmp_path, force=True)

    assert (_disk(overrides).get("org") or {}).get("default_temperature") is None


def test_a_reinit_over_a_stamp_2_config_leaves_overrides_yaml_alone(tmp_path):
    _write(tmp_path, _config(stamp=2))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})
    before = overrides.read_bytes()

    _reinit(tmp_path, force=False)

    assert overrides.read_bytes() == before  # a 0.6 set after revision 2 ran is the person's


def test_a_reinit_leaves_revision_1_keys_in_overrides_yaml_alone(tmp_path):
    _write(tmp_path, _config(stamp=None))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6,
                                                         "permissions": {"budget": {"max_actions": 100}}}})

    _reinit(tmp_path, force=False)

    org = _disk(overrides)["org"]
    assert org["default_temperature"] is None and org["permissions"]["budget"]["max_actions"] == 100


@_not_root
def test_a_reinit_over_a_read_only_overrides_yaml_warns_and_owes_it(tmp_path):
    _write(tmp_path, _config(stamp=1))
    overrides = _put(tmp_path, "overrides.yaml", {"org": {"default_temperature": 0.6}})
    overrides.chmod(0o444)
    before = overrides.read_bytes()

    out = _reinit(tmp_path, force=False)

    assert f"Could not update {overrides}" in out and "org.default_temperature: 0.6" in out
    assert overrides.read_bytes() == before and _backups(tmp_path, "overrides.yaml") == []
    assert migrate.read_init_nulls_pending(tmp_path) == ("overrides.yaml",)
    overrides.chmod(0o644)
    _migrate(tmp_path / "config.yaml")  # the next start's migration retries it alone
    assert _disk(overrides)["org"]["default_temperature"] is None
