"""MERG-02: deny rules UNION across layers — safety accumulates, it never subtracts.

A workspace may ADD denials. It may never remove one the global layer imposed, and it may never
drop `PermissionConfig`'s shipped security defaults by declaring a list of its own.

Every scenario here authors its org config inside `config.yaml`'s `org:` section and writes NO
standalone `org.yaml`. That is deliberate: `init` has only ever written org config embedded in
`config.yaml`, nothing in `src/` writes the standalone file at all, so the embedded shape is the
only one that proves the criterion for a real installation. A test built on a hand-authored
`org.yaml` passes through a mechanism no user has.

The last two tests carry MERG-01's agent-facing half: a workspace `org.context` block must reach
the agent that actually runs, merged per key against the global one — otherwise the merged
`HarnessConfig` reports one number while the running agent uses another.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.agent.permissions import PermissionEvaluator
from localharness.config.loader import ConfigLoader
from localharness.core.types import ToolCall


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
    """The two config bases: a global dir holding a minimal valid `config.yaml` and an agent, and
    a workspace `.localharness/` under a project. Every workspace file is optional."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(
        global_dir / "agents" / "deployer.yaml",
        {"name": "deployer", "role": "Deploy agent"},
    )
    return global_dir, workspace_dir


def _deny_patterns(global_dir: Path, workspace_dir: Path) -> list[str]:
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace_dir)
    return loader.load_agent("deployer").permissions.deny_patterns


# ---------------------------------------------------------------------------
# The union itself
# ---------------------------------------------------------------------------

def test_global_only_deny_is_enforced_inside_a_workspace_session(layers) -> None:
    """MERG-02's criterion, in both directions, in one scenario.

    The global layer's deny still applies inside a workspace session, and the workspace's own
    deny applies too. Neither layer's declaration erases the other's.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(ws / "config.yaml", {"org": {"permissions": {"deny_patterns": ["bash_exec(*wget *)"]}}})

    patterns = _deny_patterns(global_dir, ws)
    assert "bash_exec(*curl *)" in patterns, "a global-only deny vanished inside a workspace session"
    assert "bash_exec(*wget *)" in patterns, "the workspace's own deny never reached the agent"


def test_a_workspace_cannot_subtract_a_global_deny(layers) -> None:
    """An explicit empty list in the workspace contributes nothing and removes nothing.

    This is the shape `deep_merge` would get wrong on its own: lists are REPLACED wholesale, so a
    workspace `deny_patterns: []` would have deleted the global org's denials outright.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(ws / "config.yaml", {"org": {"permissions": {"deny_patterns": []}}})

    assert "bash_exec(*curl *)" in _deny_patterns(global_dir, ws)


def test_a_workspace_deny_declaration_does_not_drop_the_shipped_defaults(layers) -> None:
    """The fail-open case that matters most: the global layer declares NO org permissions.

    The baseline here comes from `load_org()` — with no standalone `org.yaml` it returns
    `OrgConfig()`, whose `PermissionConfig` carries the shipped security defaults, and the
    layered union is added ON TOP of those rather than replacing them. Under a plain
    `deep_merge`, the workspace's one-element list would have become the WHOLE deny list and
    `bash_exec(*sudo *)` would have silently stopped being denied.
    """
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"permissions": {"deny_patterns": ["bash_exec(*wget *)"]}}})

    patterns = _deny_patterns(global_dir, ws)
    assert "bash_exec(*wget *)" in patterns
    assert "bash_exec(*sudo *)" in patterns, "a workspace declaration replaced the shipped baseline"


def test_overrides_layers_also_contribute_denies(layers) -> None:
    """All FOUR sources feed the union, not just the two config.yamls.

    `overrides.yaml` is a real place org config lives (it is what `components set` writes), so a
    union that only read config.yaml would enforce a partial list.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "overrides.yaml",
        {"org": {"permissions": {"deny_patterns": ["bash_exec(*nc *)"]}}},
    )
    _write_yaml(
        ws / "overrides.yaml",
        {"org": {"permissions": {"deny_patterns": ["bash_exec(*ftp *)"]}}},
    )

    patterns = _deny_patterns(global_dir, ws)
    assert "bash_exec(*nc *)" in patterns, "the global overrides layer contributed nothing"
    assert "bash_exec(*ftp *)" in patterns, "the workspace overrides layer contributed nothing"


# ---------------------------------------------------------------------------
# Enforcement — the composition the agent loop performs
# ---------------------------------------------------------------------------

def test_enforcement_denies_a_global_only_pattern_in_a_workspace_session(layers) -> None:
    """The two objects `agent/loop.py` composes: the evaluator, and load_agent()'s `permissions`.

    This is the enforcement chain minus the loop's plumbing — a resolved list that never reaches
    `PermissionEvaluator` denies nothing, and an evaluator with the wrong list is worse than
    useless. The negative half is what stops this from being a test that passes because
    everything is denied.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"permissions": {"deny_patterns": ["bash_exec(*curl *)"]}}},
    )
    _write_yaml(ws / "config.yaml", {"org": {"name": "proj"}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("deployer")

    result = PermissionEvaluator().evaluate(
        ToolCall(name="bash_exec", arguments={"command": "curl https://example.com"}), cfg.permissions
    )
    assert result.denied is True
    assert "bash_exec(*curl *)" in result.reason

    allowed = PermissionEvaluator().evaluate(
        ToolCall(name="bash_exec", arguments={"command": "echo hello"}), cfg.permissions
    )
    assert allowed.denied is False, "an unrelated call was denied — this list denies everything"


def test_loop_composes_the_evaluator_with_load_agents_permissions() -> None:
    """The structural half of the test above.

    The test above proves the evaluator denies the pattern when handed `load_agent()`'s
    permissions. This proves the loop is what hands it those — if `agent/loop.py` ever stops
    passing `self._config.permissions`, the test above becomes a test of an object nobody uses,
    and nothing else would notice.
    """
    loop_py = Path(__file__).resolve().parents[2] / "src" / "localharness" / "agent" / "loop.py"
    source = loop_py.read_text(encoding="utf-8")
    assert source, f"read nothing from {loop_py}"
    call = "self._permissions.evaluate(tool_call, self._config.permissions)"
    assert source.count(call) >= 2, (
        f"{loop_py} no longer composes the evaluator with load_agent()'s permissions at both "
        "call sites — the enforcement test above is now testing an unused object"
    )


# ---------------------------------------------------------------------------
# MERG-01's agent-facing half: the org context: block
# ---------------------------------------------------------------------------

def test_workspace_org_context_reaches_the_agent(layers) -> None:
    """A workspace `org.context` value must reach the RUNNING agent, not just the merged config.

    `_raw_org_context()` used to read the global config.yaml alone, so this returned 40000 while
    `load_harness()` reported 12345 — the merged view and the agent would have disagreed about
    the context window.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"context": {"max_context_tokens": 40000}}},
    )
    _write_yaml(ws / "config.yaml", {"org": {"context": {"max_context_tokens": 12345}}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("deployer")
    assert cfg.context.max_context_tokens == 12345


def test_workspace_org_context_falls_back_to_global_per_key(layers) -> None:
    """Per-key merge, not wholesale block replacement.

    The workspace names one key; the global layer still governs every key the workspace is silent
    about. Replacing the whole `context:` block would reset `preserve_last_n_messages` to its
    schema default and silently undo a global tuning.
    """
    global_dir, ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {
            **_MINIMAL,
            "org": {"context": {"max_context_tokens": 40000, "preserve_last_n_messages": 12}},
        },
    )
    _write_yaml(ws / "config.yaml", {"org": {"context": {"max_context_tokens": 12345}}})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("deployer")
    assert cfg.context.max_context_tokens == 12345
    assert cfg.context.preserve_last_n_messages == 12, "the workspace replaced the whole block"


# ---------------------------------------------------------------------------
# The overlay layer's own deny rules (badmood wave 2).
# ---------------------------------------------------------------------------

def test_overlay_agent_deny_patterns_reach_enforcement(layers) -> None:
    """`agent.permissions.deny_patterns` in overrides.yaml must survive to the loaded agent.

    The overlay is layered UNDER the resolved agent dict (step 5b), and that dict already carries
    `permissions.deny_patterns` — the org/division/agent union — so deep_merge replaced the
    overlay's list wholesale and the rules a user wrote through `components set` were stored,
    confirmed and never enforced. Deny is union-only in every other layer; the overlay is not an
    exception.
    """
    global_dir, ws = layers
    (global_dir / "overrides.yaml").write_text(
        yaml.dump({"agent": {"permissions": {"deny_patterns": ["bash_exec(*overlay-only*)"]}}}),
        encoding="utf-8",
    )

    patterns = _deny_patterns(global_dir, ws)
    assert "bash_exec(*overlay-only*)" in patterns
    # Union, not replacement: the shipped defaults are still there underneath it.
    assert len(patterns) > 1, "the overlay's list replaced the shipped defaults"


def test_overlay_agent_deny_patterns_never_subtract(layers) -> None:
    """An empty overlay list removes nothing — same physics as every other layer (MERG-02)."""
    global_dir, ws = layers
    before = _deny_patterns(global_dir, ws)
    (global_dir / "overrides.yaml").write_text(
        yaml.dump({"agent": {"permissions": {"deny_patterns": []}}}),
        encoding="utf-8",
    )
    assert _deny_patterns(global_dir, ws) == before


def test_overlay_agent_deny_patterns_dedupe(layers) -> None:
    """Restating a rule the agent already denies adds no duplicate row."""
    global_dir, ws = layers
    _write_yaml(
        global_dir / "agents" / "deployer.yaml",
        {"name": "deployer", "role": "Deploy agent",
         "permissions": {"deny_patterns": ["bash_exec(*shared*)"]}},
    )
    (global_dir / "overrides.yaml").write_text(
        yaml.dump({"agent": {"permissions": {"deny_patterns": ["bash_exec(*shared*)"]}}}),
        encoding="utf-8",
    )
    patterns = _deny_patterns(global_dir, ws)
    assert patterns.count("bash_exec(*shared*)") == 1
