"""A project's agent files can only tighten permissions — trusted or not — and an untrusted project
loses its MCP servers (orchestrator ruling R5: never "drop the whole block", which would loosen).

Every PermissionConfig, AskConfig and BudgetConfig field is classified here as narrowed (a project
value is held to the machine's or the shipped one), machine-only (a project value is dropped) or
inert (budgets and bookkeeping). A field added to those models without a classification fails the
first test, so a new loosening surface cannot arrive unreviewed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import DEFAULT_MODE, GateSettings
from localharness.config.loader import (
    ASK_GLOBAL_ONLY_FIELDS,
    ASK_TIGHTEN_ONLY_FIELDS,
    ConfigLoader,
    _gate_default,
)
from localharness.config.models import (
    ASK_TO_GATE_FIELD,
    AgentConfig,
    AskConfig,
    BudgetConfig,
    PermissionConfig,
)

NARROWED = {"mode", "deny_patterns", "workspace_root", "ask.network_hosts"} | {
    "ask." + f for f in ASK_TIGHTEN_ONLY_FIELDS}
MACHINE = {"ask." + f for f in ASK_GLOBAL_ONLY_FIELDS} | {"budget.kill_file"}
INERT = {"budget.max_actions", "budget.max_tool_calls", "budget.max_duration_minutes",
         "defaults_revision"}
EVIL = {"name": "evil", "transport": "stdio", "command": "/bin/echo", "args": ["pwned"]}


@pytest.fixture
def g(monkeypatch) -> Path:
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    return Path(os.environ["LOCALHARNESS_HOME"])


@pytest.fixture
def ws(tmp_path) -> Path:
    workspace = tmp_path / "proj" / ".localharness"
    (workspace / "agents").mkdir(parents=True)
    return workspace


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _gate_value(agent: AgentConfig, field: str):
    return getattr(agent.permissions.ask.to_gate_settings(), ASK_TO_GATE_FIELD.get(field, field))


def test_every_permissions_key_is_classified():
    every = ({f for f in PermissionConfig.model_fields if f not in ("ask", "budget")}
             | {"ask." + f for f in AskConfig.model_fields}
             | {"budget." + f for f in BudgetConfig.model_fields})

    assert not (NARROWED & MACHINE or NARROWED & INERT or MACHINE & INERT)
    assert every == NARROWED | MACHINE | INERT, (
        f"unclassified: {sorted(every - NARROWED - MACHINE - INERT)}; "
        f"stale: {sorted((NARROWED | MACHINE | INERT) - every)}")


def _loosening(key: str):
    """(the project file's `permissions` block, a check on the loaded agent that it was held)."""
    if key == "mode":
        return {"mode": "unattended"}, lambda a: a.permissions.mode == DEFAULT_MODE
    if key == "deny_patterns":
        shipped = PermissionConfig().deny_patterns
        return {"deny_patterns": []}, lambda a: set(shipped) <= set(a.permissions.deny_patterns)
    if key == "workspace_root":
        return {"workspace_root": "/"}, lambda a: a.permissions.workspace_root is None
    if key == "ask.network_hosts":  # the machine switched it on (see _machine_layer)
        return {"ask": {"network_hosts": False}}, lambda a: a.permissions.ask.network_hosts is True
    if key == "budget.kill_file":
        shipped = BudgetConfig().kill_file
        return {"budget": {"kill_file": "/tmp/elsewhere"}}, (
            lambda a: a.permissions.budget.kill_file == shipped)
    field = key.removeprefix("ask.")
    shipped = _gate_default(field)
    if field in ASK_TIGHTEN_ONLY_FIELDS:
        return {"ask": {field: []}}, lambda a: set(shipped) <= set(_gate_value(a, field))
    value = 9999.0 if field == "timeout_s" else ["zz-project-added"]
    return {"ask": {field: value}}, lambda a: _gate_value(a, field) == shipped


def _machine_layer(g: Path) -> None:
    text = (g / "config.yaml").read_text(encoding="utf-8")
    (g / "config.yaml").write_text(
        text + "org:\n  permissions:\n    ask:\n      network_hosts: true\n", encoding="utf-8")


@pytest.mark.parametrize("trusted", [True, False], ids=["trusted", "untrusted"])
@pytest.mark.parametrize("key", sorted(NARROWED | MACHINE))
def test_a_project_loosening_is_held_back_whether_trusted_or_not(g, ws, key, trusted):
    _machine_layer(g)
    perms, held = _loosening(key)
    _write(ws / "agents" / "worker.yaml", {"name": "worker", "role": "R", "permissions": perms})

    agent = ConfigLoader(config_dir=g, local_config_dir=ws,
                         project_trusted=trusted).load_agent("worker")

    assert held(agent), f"{key} loosened the project's agent: {agent.permissions!r}"


def test_the_shipped_ask_baseline_is_the_gates_own_default():
    """The baseline the narrowing and the machine record compare against is GateSettings itself."""
    for field in ASK_TIGHTEN_ONLY_FIELDS | ASK_GLOBAL_ONLY_FIELDS:
        gate = ASK_TO_GATE_FIELD.get(field, field)
        assert _gate_default(field) == getattr(GateSettings(), gate)


# --------------------------------------------------------------------------- MCP servers


def test_an_untrusted_projects_agent_file_starts_no_server_and_says_which(g, ws):
    path = _write(ws / "agents" / "worker.yaml", {"name": "worker", "role": "R",
                                                 "tools": {"mcp_servers": [EVIL]}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws, project_trusted=False)

    assert loader.load_agent("worker").tools.mcp_servers == []
    assert loader.project_mcp_skipped == {str(path): ["evil"]}


def test_a_trusted_projects_agent_file_keeps_its_servers(g, ws):
    _write(ws / "agents" / "worker.yaml", {"name": "worker", "role": "R",
                                          "tools": {"mcp_servers": [EVIL]}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws, project_trusted=True)

    assert [s.name for s in loader.load_agent("worker").tools.mcp_servers] == ["evil"]
    assert loader.project_mcp_skipped == {}


def test_the_machines_own_agent_file_keeps_its_servers_in_an_untrusted_project(g, ws):
    _write(g / "agents" / "mine.yaml", {"name": "mine", "role": "R",
                                        "tools": {"mcp_servers": [EVIL]}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws, project_trusted=False)

    assert [s.name for s in loader.load_agent("mine").tools.mcp_servers] == ["evil"]
    assert loader.project_mcp_skipped == {}


@pytest.mark.parametrize("trusted", [True, False])
def test_a_project_overlay_of_a_built_in_agent_follows_the_same_rule(g, ws, trusted):
    path = _write(ws / "agents" / "explore.yaml", {"tools": {"mcp_servers": [EVIL]}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws, project_trusted=trusted)

    agent = loader.overlay_builtin_config("explore", AgentConfig(name="explore", role="R"))

    assert [s.name for s in agent.tools.mcp_servers] == (["evil"] if trusted else [])
    assert loader.project_mcp_skipped == ({} if trusted else {str(path): ["evil"]})


def test_the_strip_names_a_server_without_a_name(g, ws):
    path = _write(ws / "agents" / "worker.yaml", {"name": "worker", "role": "R", "tools": {
        "mcp_servers": [{"transport": "stdio", "command": "/bin/true"}]}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws, project_trusted=False)

    assert loader.load_agent("worker").tools.mcp_servers == []
    assert loader.project_mcp_skipped == {str(path): ["?"]}
