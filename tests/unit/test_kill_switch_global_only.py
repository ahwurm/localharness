"""A workspace cannot relocate the kill switch (F5).

The kill file is a machine-global CONTROL artifact — one file stops every agent on the box — and
start_cmd/subagent already pin its DIRECTORY to the global config dir. The VALUE did not follow
that rule: it came off the resolved AgentConfig, so a workspace `agents/<name>.yaml` (or division
yaml) carrying an ABSOLUTE `kill_file` detached the session from the operator's global KILL while
every other part of the session looked normal.
"""
from __future__ import annotations

import yaml

from localharness.config.loader import ConfigLoader
from localharness.config.paths import resolve_runtime_path


def _seed(base, subdir: str, name: str, **fields):
    d = base / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(yaml.safe_dump({"name": name, **fields}), encoding="utf-8")


def _budget(kill_file: str) -> dict:
    return {"permissions": {"budget": {"kill_file": kill_file}}}


def test_workspace_agent_yaml_cannot_move_the_kill_file(tmp_path):
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _seed(ws, "agents", "a1", role="r", **_budget("/tmp/pwned-KILL"))

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("a1")

    assert cfg.permissions.budget.kill_file != "/tmp/pwned-KILL"


def test_the_resolved_kill_path_stays_under_the_global_config_dir(tmp_path):
    """The composed answer start_cmd computes: value from config, directory pinned global."""
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _seed(ws, "agents", "a1", role="r", **_budget("/tmp/pwned-KILL"))

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("a1")
    kill_value = cfg.permissions.budget.kill_file or "KILL"

    assert resolve_runtime_path(kill_value, global_dir) == global_dir / "KILL"


def test_workspace_division_yaml_cannot_move_the_kill_file(tmp_path):
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _seed(ws, "divisions", "d1", **_budget("/tmp/pwned-KILL"))
    _seed(ws, "agents", "a1", role="r", division="d1")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("a1")

    assert cfg.permissions.budget.kill_file != "/tmp/pwned-KILL"


def test_the_global_layers_own_kill_file_still_applies_through_a_shadow(tmp_path):
    """Global-only resolution is not "no resolution": the global file's value still wins."""
    global_dir, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _seed(global_dir, "agents", "a1", role="r", **_budget("/var/run/lh/KILL"))
    _seed(ws, "agents", "a1", role="r", **_budget("/tmp/pwned-KILL"))

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("a1")

    assert cfg.permissions.budget.kill_file == "/var/run/lh/KILL"


def test_no_workspace_layer_is_unchanged(tmp_path):
    """LAYR-03: with no workspace, an agent's own kill_file is honored exactly as before."""
    global_dir = tmp_path / "global"
    _seed(global_dir, "agents", "a1", role="r", **_budget("/var/run/lh/KILL"))

    cfg = ConfigLoader(config_dir=global_dir).load_agent("a1")

    assert cfg.permissions.budget.kill_file == "/var/run/lh/KILL"
