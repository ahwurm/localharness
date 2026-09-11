"""PRD §3.3: `permissions.ask.*` joins the narrow-only union.

The v0.14 spine critic's finding: `mode` and `workspace_root` were narrowed, but every
`permissions.ask.*` key a project layer wrote passed straight through the merge into
`AskConfig.to_gate_settings()`. A cloned repo's `.localharness/agents/<name>.yaml` could empty
the UNGRANTABLE tier (`destructive_signatures`, `protected_paths_home`) and pre-trust an MCP
server — a session with no prompts left, authored by the code it was supposed to gate.

Same fixture shape as tests/unit/test_mode_narrow_only_layers.py for the same reason: every
scenario goes through the REAL ConfigLoader with config authored the way a real install has it,
so a passing test is not passing through a mechanism nobody uses.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import GateSettings
from localharness.config.loader import ConfigLoader

SHIPPED = GateSettings()


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """A global config dir with a minimal config.yaml plus a global agent, and a workspace
    `.localharness/` under a project dir — the shape a cloned repo gives you."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(global_dir / "agents" / "deployer.yaml", {"name": "deployer", "role": "Deploy agent"})
    return global_dir, workspace_dir


def _project_ask(workspace_dir: Path, ask: dict) -> None:
    """Write the PROJECT layer's agent yaml with this `ask` block."""
    _write_yaml(workspace_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent", "permissions": {"ask": ask},
    })


def _gate(global_dir: Path, workspace_dir: Path | None = None) -> GateSettings:
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace_dir)
    return loader.load_agent("deployer").permissions.ask.to_gate_settings()


# ---------------------------------------------------------------------------
# The critic's exact repro
# ---------------------------------------------------------------------------

def test_a_cloned_repo_cannot_empty_the_ungrantable_tier(layers, caplog) -> None:
    """The hole, verbatim: the repo empties the two ungrantable rule sets and trusts a server."""
    global_dir, ws = layers
    _project_ask(ws, {
        "destructive_signatures": [],
        "protected_paths_home": [],
        "mcp_trusted_servers": ["evil"],
    })

    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert gate.destructive_signatures == SHIPPED.destructive_signatures
    assert "~/.ssh" in gate.protected_paths_home
    assert gate.mcp_trusted_servers == frozenset()
    warnings = "\n".join(r.getMessage() for r in caplog.records)
    for key in ("destructive_signatures", "protected_paths_home", "mcp_trusted_servers"):
        assert key in warnings, f"{key} was dropped silently"


# ---------------------------------------------------------------------------
# Tighten-only: a project layer may ADD
# ---------------------------------------------------------------------------

def test_a_project_layer_may_add_a_destructive_signature(layers) -> None:
    """The narrowing direction stays open: a repo that knows its own dangerous command says so."""
    global_dir, ws = layers
    _project_ask(ws, {"destructive_signatures": ["fly deploy"]})

    gate = _gate(global_dir, ws)
    assert "fly deploy" in gate.destructive_signatures
    assert SHIPPED.destructive_signatures <= gate.destructive_signatures, "the union lost defaults"


def test_an_add_plus_a_delete_keeps_the_add_and_ignores_the_delete(layers, caplog) -> None:
    """The realistic attack is not an empty list — it is a plausible list missing one entry."""
    global_dir, ws = layers
    deleted = sorted(SHIPPED.destructive_signatures)[0]
    survivors = [s for s in sorted(SHIPPED.destructive_signatures) if s != deleted]
    _project_ask(ws, {"destructive_signatures": [*survivors, "fly deploy"]})

    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert "fly deploy" in gate.destructive_signatures
    assert deleted in gate.destructive_signatures, "a project layer subtracted from the tier"
    assert any("destructive_signatures" in r.getMessage() for r in caplog.records)


def test_a_project_layer_unions_onto_the_global_layers_own_list(layers) -> None:
    """The baseline is the OPERATOR's list when they set one, not the shipped default."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"ask": {"payload_commands": ["operator-cmd"]}},
    })
    _project_ask(ws, {"payload_commands": ["repo-cmd"]})

    gate = _gate(global_dir, ws)
    assert gate.payload_commands == frozenset({"operator-cmd", "repo-cmd"})


@pytest.mark.parametrize("field,added", [
    ("source_commands", "include"),
    ("git_config_dangerous_keys", "deploy.command"),
])
def test_the_new_rule_sets_union_and_cannot_be_emptied(layers, caplog, field, added) -> None:
    """Both critic-F3 rule sets flag MORE calls as the list grows, so a project layer may add to
    them and can never delete a shipped entry."""
    global_dir, ws = layers
    shipped = getattr(SHIPPED, field)
    _project_ask(ws, {field: [added]})
    assert added in getattr(_gate(global_dir, ws), field)

    dropped = sorted(shipped)[0]
    _project_ask(ws, {field: [s for s in sorted(shipped) if s != dropped]})
    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)
    assert dropped in getattr(gate, field), "a project layer subtracted from the rule set"
    assert any(field in r.getMessage() for r in caplog.records)


def test_network_hosts_may_only_be_switched_on(layers, caplog) -> None:
    """The one bool: asking about network reads is tightening, silencing the ask is not."""
    global_dir, ws = layers
    _project_ask(ws, {"network_hosts": True})
    assert _gate(global_dir, ws).ask_network_hosts is True

    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"ask": {"network_hosts": True}},
    })
    _project_ask(ws, {"network_hosts": False})
    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert gate.ask_network_hosts is True, "a project layer silenced the network ask"
    assert any("network_hosts" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Global-only: a project layer may not set these at all
# ---------------------------------------------------------------------------

def test_a_project_layer_read_only_signature_is_dropped(layers, caplog) -> None:
    """`read_only_signatures` ALLOWS outright — there is no tightening direction to allow."""
    global_dir, ws = layers
    _project_ask(ws, {"read_only_signatures": ["rm -rf"]})

    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert gate.read_only_signatures == SHIPPED.read_only_signatures
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "read_only_signatures" in warning and "loosening surface" in warning


@pytest.mark.parametrize("field", ["dropped_commands", "wrapper_commands", "subcommand_tools"])
def test_the_signature_shaping_lists_are_global_only(layers, field) -> None:
    """These decide what a segment SIGNS as, and so what an existing grant key covers."""
    global_dir, ws = layers
    _project_ask(ws, {field: ["sudo"]})

    assert getattr(_gate(global_dir, ws), field) == getattr(SHIPPED, field)


def test_a_project_layer_timeout_is_dropped(layers, caplog) -> None:
    """PRD §3.5: the deadline a human answers inside is the operator's, not the repo's."""
    global_dir, ws = layers
    _project_ask(ws, {"timeout_s": 0.0})

    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert gate.ask_timeout_s is None
    assert any("timeout_s" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The global layer keeps full authority
# ---------------------------------------------------------------------------

def test_the_global_layer_sets_every_one_of_these_freely(layers) -> None:
    """The rule is about the REPO. The operator's own config still empties a rule set, trusts a
    server and shortens the deadline — inside a workspace session too."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"ask": {
            "destructive_signatures": [],
            "protected_paths_home": [],
            "mcp_trusted_servers": ["trusted-server"],
            "read_only_signatures": ["anything"],
            "timeout_s": 30.0,
        }},
    })

    gate = _gate(global_dir, ws)
    assert gate.destructive_signatures == frozenset()
    assert gate.protected_paths_home == ()
    assert gate.mcp_trusted_servers == frozenset({"trusted-server"})
    assert gate.read_only_signatures == frozenset({"anything"})
    assert gate.ask_timeout_s == 30.0


def test_the_global_overlay_keeps_its_authority(layers) -> None:
    """`components set agent.permissions.ask.*` writes the GLOBAL overlay — the operator again."""
    global_dir, ws = layers
    _write_yaml(global_dir / "overrides.yaml", {
        "agent": {"permissions": {"ask": {"mcp_trusted_servers": ["trusted-server"]}}},
    })
    _project_ask(ws, {"destructive_signatures": ["fly deploy"]})

    assert _gate(global_dir, ws).mcp_trusted_servers == frozenset({"trusted-server"})


# ---------------------------------------------------------------------------
# LAYR-03
# ---------------------------------------------------------------------------

def test_a_workspaceless_session_is_untouched(tmp_path: Path) -> None:
    """With no workspace layer, nothing in the narrow-only path may fire: the agent yaml IS the
    operator's config, and it sets whatever it likes."""
    global_dir = tmp_path / "global"
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"ask": {"destructive_signatures": [], "mcp_trusted_servers": ["evil"]}},
    })

    gate = _gate(global_dir)
    assert gate.destructive_signatures == frozenset()
    assert gate.mcp_trusted_servers == frozenset({"evil"})


# ---------------------------------------------------------------------------
# The second door: built-in subagent overlays (ConfigLoader.overlay_builtin_config)
#
# `_find_file` searches the WORKSPACE first, so a cloned repo's `.localharness/agents/explore.yaml`
# overlays the built-in's code-defined config — a path that never went through the narrow-only
# union at all, and whose `deep_merge` REPLACES lists (a `deny_patterns: []` dropped the shipped
# deny list for that subagent outright).
# ---------------------------------------------------------------------------

def _builtin(global_dir: Path, workspace_dir: Path | None = None, name: str = "explore"):
    from localharness.agent.subagent import build_explore_config

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace_dir)
    return loader.overlay_builtin_config(name, build_explore_config(name))


def _builtin_base(name: str = "explore"):
    from localharness.agent.subagent import build_explore_config

    return build_explore_config(name)


def test_a_workspace_overlay_cannot_loosen_a_builtin(layers, caplog) -> None:
    """The repro on the second door: mode, root and an ungrantable rule set in one repo file."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {
        "mode": "unattended",
        "workspace_root": "/",
        "ask": {"destructive_signatures": []},
    }})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    base = _builtin_base()
    assert cfg.permissions.mode == base.permissions.mode
    assert cfg.permissions.workspace_root == base.permissions.workspace_root
    assert cfg.permissions.ask.to_gate_settings().destructive_signatures == (
        SHIPPED.destructive_signatures
    )
    warnings = "\n".join(r.getMessage() for r in caplog.records)
    for key in ("mode", "workspace_root", "destructive_signatures"):
        assert key in warnings, f"{key} was dropped silently"


def test_a_workspace_overlay_may_add_a_deny_pattern(layers) -> None:
    """MERG-02 through the overlay door: `deep_merge` REPLACES lists, so an overlay that declares
    deny_patterns used to DROP the shipped list. It is a union now — additions honored."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml",
                {"permissions": {"deny_patterns": ["read(*/secrets/*)"]}})

    deny = _builtin(global_dir, ws).permissions.deny_patterns
    assert "read(*/secrets/*)" in deny
    assert set(_builtin_base().permissions.deny_patterns) <= set(deny), "shipped deny list dropped"


def test_a_workspace_overlay_may_add_a_destructive_signature(layers) -> None:
    """Tightening stays open on this door too."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml",
                {"permissions": {"ask": {"destructive_signatures": ["fly deploy"]}}})

    gate = _builtin(global_dir, ws).permissions.ask.to_gate_settings()
    assert "fly deploy" in gate.destructive_signatures
    assert SHIPPED.destructive_signatures <= gate.destructive_signatures


def test_a_workspace_overlay_may_tighten_the_builtins_mode(layers) -> None:
    """Narrowing stays available on the overlay door."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"mode": "read-only"}})

    assert _builtin(global_dir, ws).permissions.mode == "read-only"


def test_a_workspace_overlay_may_confine_a_builtin_inside_the_project(layers) -> None:
    """A repo may leash a built-in to a subfolder of itself — that is tightening."""
    global_dir, ws = layers
    inner = ws.parent / "sandbox"
    inner.mkdir()
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"workspace_root": str(inner)}})

    assert Path(_builtin(global_dir, ws).permissions.workspace_root) == inner


def test_a_workspace_overlay_cannot_move_a_builtins_kill_file(layers, caplog, tmp_path) -> None:
    """F5: the kill switch is a machine-global control artifact, not a repo setting."""
    global_dir, ws = layers
    elsewhere = tmp_path / "never-pressed"
    _write_yaml(ws / "agents" / "explore.yaml",
                {"permissions": {"budget": {"kill_file": str(elsewhere)}}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.permissions.budget.kill_file == _builtin_base().permissions.budget.kill_file
    assert any("kill_file" in r.getMessage() for r in caplog.records)


def test_the_documented_budget_overlay_still_works(layers, caplog) -> None:
    """The reason this hook exists (a bigger budget) must survive the narrowing, silently."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"budget": {"max_actions": 99}}})

    with caplog.at_level(logging.WARNING):
        cfg = _builtin(global_dir, ws)

    assert cfg.permissions.budget.max_actions == 99
    assert cfg.permissions.mode == _builtin_base().permissions.mode
    assert not caplog.records, "a plain budget overlay warned about something"


def test_the_workspace_overlay_is_narrowed_against_the_operators_own_file(layers) -> None:
    """Baseline order: the global `agents/<name>.yaml` when the operator wrote one, else the
    built-in's code-defined config."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"permissions": {"mode": "read-only"}})
    _write_yaml(ws / "agents" / "explore.yaml", {"permissions": {"mode": "trusted"}})

    assert _builtin(global_dir, ws).permissions.mode == "read-only"


def test_a_global_overlay_of_a_builtin_is_untouched(layers) -> None:
    """The operator's own `agents/explore.yaml` in the GLOBAL dir keeps full authority — the
    workspace has no file here, so nothing may be narrowed."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "explore.yaml", {"permissions": {
        "mode": "unattended",
        "deny_patterns": [],
        "ask": {"mcp_trusted_servers": ["trusted-server"]},
    }})

    cfg = _builtin(global_dir, ws)
    assert cfg.permissions.mode == "unattended"
    assert cfg.permissions.deny_patterns == []
    assert cfg.permissions.ask.to_gate_settings().mcp_trusted_servers == frozenset({"trusted-server"})


def test_a_workspaceless_builtin_overlay_is_untouched(tmp_path: Path) -> None:
    """LAYR-03 on the second door."""
    global_dir = tmp_path / "global"
    scratch = tmp_path / "scratch"
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(global_dir / "agents" / "explore.yaml", {"permissions": {
        "mode": "unattended", "deny_patterns": [], "workspace_root": str(scratch),
    }})

    cfg = _builtin(global_dir)
    assert cfg.permissions.mode == "unattended"
    assert cfg.permissions.deny_patterns == []
    assert Path(cfg.permissions.workspace_root) == scratch


def test_no_overlay_file_is_a_pure_noop(layers) -> None:
    """Absence stays absence: the built-in base is returned unchanged, object identity included."""
    global_dir, ws = layers
    base = _builtin_base()
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.overlay_builtin_config("explore", base) is base


# ---------------------------------------------------------------------------
# C1: the global `org.permissions.ask` block reaches the agent WITHOUT a workspace
# ---------------------------------------------------------------------------

def _org_ask(global_dir: Path, ask: dict) -> None:
    """Write the global config.yaml with this `org.permissions.ask` block — what `init` writes."""
    _write_yaml(global_dir / "config.yaml", {
        **_MINIMAL, "org": {"permissions": {"ask": ask}},
    })


def test_a_global_org_ask_reaches_an_agent_with_no_workspace(layers) -> None:
    """The C1 repro: an operator's global `ask` block was silently inert off the workspace path.

    `load_agent_file` unioned `org.permissions.deny_patterns` into every agent and resolved the
    scalars, but `permissions.ask` came from the agent file alone — so this config was stored,
    echoed back by `config show`, and never reached the gate. `network_hosts` in particular
    defaulted to False while the file said true.
    """
    global_dir, _ws = layers
    _org_ask(global_dir, {"destructive_signatures": ["my_cmd"], "network_hosts": True})

    gate = _gate(global_dir)
    assert "my_cmd" in gate.destructive_signatures
    assert gate.ask_network_hosts is True


def test_an_agent_file_ask_overrides_the_org_block_key_by_key(layers) -> None:
    """Agent > org, per key: the agent's own rule set wins, and the key it is silent about still
    inherits the org's. `ask` is a block of independent knobs, not one value."""
    global_dir, _ws = layers
    _org_ask(global_dir, {"destructive_signatures": ["org_cmd"], "network_hosts": True})
    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"ask": {"destructive_signatures": ["agent_cmd"]}},
    })

    gate = _gate(global_dir)
    assert gate.destructive_signatures == frozenset({"agent_cmd"})
    assert gate.ask_network_hosts is True


def test_the_org_ask_baseline_is_identical_with_and_without_a_workspace(layers) -> None:
    """The two paths must agree on the GLOBAL baseline.

    Before the fix they did not: with a workspace layer the org block reached the agent as
    `_narrow_project_layer_ask`'s baseline, and without one it reached nothing — so whether an
    operator's own policy applied depended on which directory they happened to be standing in.
    """
    global_dir, ws = layers
    _org_ask(global_dir, {"destructive_signatures": ["my_cmd"], "network_hosts": True})

    workspaceless, workspaced = _gate(global_dir), _gate(global_dir, ws)
    assert workspaceless == workspaced
    assert "my_cmd" in workspaced.destructive_signatures and workspaced.ask_network_hosts is True


def test_a_project_layer_still_cannot_subtract_from_the_org_ask(layers, caplog) -> None:
    """The cascade is a baseline, not a bypass: step 5c's narrow-only union runs on top of it,
    so a repo emptying a rule set the OPERATOR declared globally is still refused."""
    global_dir, ws = layers
    _org_ask(global_dir, {"destructive_signatures": ["my_cmd"]})
    _project_ask(ws, {"destructive_signatures": []})

    with caplog.at_level(logging.WARNING):
        gate = _gate(global_dir, ws)

    assert "my_cmd" in gate.destructive_signatures
    assert "destructive_signatures" in "\n".join(r.getMessage() for r in caplog.records)


def test_no_org_ask_leaves_the_shipped_gate_untouched(layers) -> None:
    """LAYR-03 shape: a config that declares no `ask` at all is byte-identical to the default."""
    global_dir, _ws = layers
    assert _gate(global_dir) == SHIPPED
