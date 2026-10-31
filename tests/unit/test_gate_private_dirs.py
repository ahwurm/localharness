"""The gate's deny class on an agent's task and session folders: refused in every mode, keyed to the path class."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate_types import GateSettings, Grant, ToolMeta, Verdict
from localharness.agent.verdict import GateContext, evaluate

SETTINGS = GateSettings()
WRITE_META = ToolMeta(destructive=True, group="fs.write")
MODES = ("guarded", "trusted", "auto", "unattended")


def no_grants(workspace: Path, klass: str, key: str) -> Grant | None:
    return None


def make_ctx(workspace: Path, **kw) -> GateContext:
    kw.setdefault("boundary", workspace)
    kw.setdefault("grants", no_grants)
    kw.setdefault("has_review_surface", True)
    kw.setdefault("mode", "guarded")
    return GateContext(workspace=workspace, **kw)


@pytest.fixture
def ws(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def write(ctx: GateContext, path: Path | str, tool: str = "write"):
    params = {"path": str(path), "content": "x"} if tool == "write" else {"path": str(path), "old_string": "a", "new_string": "b"}
    return evaluate(tool, params, WRITE_META, ctx, SETTINGS)


@pytest.mark.parametrize("mode", MODES)
def test_a_write_to_a_tasks_plan_is_denied_in_every_mode_and_names_the_plan_command(ws, mode):
    plan = ws.parent / ".localharness" / "agents" / "a" / "tasks" / "t1" / "plan.md"
    plan.parent.mkdir(parents=True)
    result = write(make_ctx(ws, mode=mode), plan)
    assert result.verdict is Verdict.DENY
    assert "/task plan" in result.reason


def test_a_write_to_the_agents_own_config_is_not_denied(ws):
    config = ws.parent / ".localharness" / "agents" / "a" / "config.yaml"
    assert write(make_ctx(ws), config).verdict is not Verdict.DENY
