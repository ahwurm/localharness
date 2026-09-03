"""CONF-01 — the confinement leash comes free with the workspace layer.

`permissions.workspace_root` is opt-in filesystem confinement for write/edit/bash_exec, and until
this plan it had zero workspace-awareness: `load_agent()` copied the agent's own `permissions`
block and nothing later touched the key, so a workspace session's file tools were unconfined
unless the user had hand-configured a root. The layer already names the folder the work lives in,
so the default now follows from it — when a workspace applies and nothing else set a root,
`workspace_root` defaults to the folder CONTAINING `.localharness/` (the project root, NOT the
dotdir). Explicit config still wins, from either source that can set it: the agent's own yaml or
the user overlay's `agent:` section. With no workspace layer the injection is inert and `None`
still means UNCONFINED — the shipped contract in models.py, which file-write capability depends on.

Scope, stated plainly: these tests exercise the LOADER by constructing `ConfigLoader` with a named
workspace layer. Nothing here drives a real session, and no test here proves the value reaches the
Write/Edit/BashExec instances — `register_builtin_tools(..., workspace_root=...)` at
start_cmd.py:799 is that seam, and 41-06 owns proving it end to end.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader

_MINIMAL_HARNESS = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


def _seed_agent(base: Path, name: str, **fields) -> Path:
    """Write `{base}/agents/{name}.yaml`. Discovery keys on the STEM, kept equal to `name:` here."""
    d = base / "agents"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, "role": "Test role", **fields}), encoding="utf-8")
    return path


@pytest.fixture
def layers(tmp_path: Path):
    """Phase 39's tmp-tree shape: a global config dir, and a workspace `.localharness/` under a
    project folder. Returns (global_dir, project_root, workspace).

    The loader is constructed DIRECTLY with `local_config_dir=` — no chdir, no discovery. This
    plan's subject is the loader's injection; whether a real session arrives at this workspace is
    39-04/39-05's question, already proven there, and 41-06 re-proves it for confinement.
    """
    global_dir = tmp_path / "global"
    project = tmp_path / "proj"
    workspace = project / ".localharness"
    (global_dir / "agents").mkdir(parents=True)
    (workspace / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.safe_dump(_MINIMAL_HARNESS), encoding="utf-8")
    return global_dir, project, workspace


# ---------------------------------------------------------------------------
# 1-2. The default, and the half of it that is easy to get wrong
# ---------------------------------------------------------------------------

def test_workspace_session_defaults_the_root_to_the_project(layers):
    """A workspace applies, the agent yaml says nothing about permissions → the project root.

    This is CONF-01's whole claim: the leash comes free with the layer. Before this, the same
    load returned None and the agent's write/edit/bash tools were unconfined.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root == str(project), (
        "CONF-01: with a workspace layer and no explicit root, workspace_root must default to the "
        f"folder CONTAINING .localharness/ — expected {str(project)!r}, got "
        f"{cfg.permissions.workspace_root!r}"
    )


def test_the_default_is_the_parent_not_the_dotdir(layers):
    """Asserted SEPARATELY from test 1 so it cannot hide behind that equality.

    `ConfigLoader._local_dir` IS the `.localharness/` directory (both discovery entry points
    return the dotdir), so `str(self._local_dir)` is the plausible wrong answer: it type-checks,
    it is a real directory, and confinement would still "work" — it would just leash every agent
    inside the config folder instead of the project. This assertion is written for exactly that
    regression, so it reddens on its own rather than depending on test 1's fixture shape.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")
    root = cfg.permissions.workspace_root

    assert root != str(workspace), (
        "CONF-01 says the folder CONTAINING .localharness/, not the dotdir itself — got the "
        f"workspace dir {root!r}, which would confine every agent inside the config folder"
    )
    assert not str(root).endswith(".localharness"), (
        f"the default must be the project root; {root!r} ends in .localharness"
    )
    assert Path(root) == workspace.parent


# ---------------------------------------------------------------------------
# 3-4. Explicit config still wins — from BOTH sources that can set the key
# ---------------------------------------------------------------------------

def test_explicit_agent_yaml_root_is_not_overwritten(layers):
    """The agent's own yaml is the primary source; the default only fills a GAP.

    A user who confined an agent to a scratch dir (the harness's own evals do exactly this) must
    not silently have that widened to the whole project just because a workspace exists.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder", permissions={"workspace_root": "/explicit/root"})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root == "/explicit/root", (
        "an explicitly configured workspace_root must survive the workspace default — got "
        f"{cfg.permissions.workspace_root!r}"
    )
    assert cfg.permissions.workspace_root != str(project), (
        "the project-root default must NOT have replaced the agent's explicit confinement"
    )


def test_overlay_agent_section_root_is_not_overwritten(layers):
    """The user overlay's `agent:` section is the second source, and the reason placement matters.

    The injection sits AFTER the 5b overlay merge. Injected BEFORE it, the overlay would be
    deep-merged UNDERNEATH our default and lose — the user's `components set agent.*` value would
    be silently discarded in any workspace. Both halves are asserted: the overlay value present
    AND the project root absent, because "the overlay value is there" passes when both are.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")
    (global_dir / "overrides.yaml").write_text(
        yaml.safe_dump({"agent": {"permissions": {"workspace_root": "/from/overlay"}}}),
        encoding="utf-8",
    )

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root == "/from/overlay", (
        "the overlay's agent.permissions.workspace_root is explicit config and must win over the "
        f"workspace default — got {cfg.permissions.workspace_root!r}"
    )
    assert cfg.permissions.workspace_root != str(project), (
        "the project-root default must be ABSENT here; asserting only that the overlay value is "
        "present would pass an implementation that injected before the overlay merge"
    )


# ---------------------------------------------------------------------------
# 5. No workspace layer — the unconfined contract is untouched
# ---------------------------------------------------------------------------

def test_no_workspace_layer_leaves_the_root_unset(layers):
    """LAYR-03: a workspace-less load must be byte-identical to before this plan.

    `None` means UNCONFINED and that is deliberate (models.py) — file-write capability is a core
    product feature. A default that leaked into global-only sessions would silently confine every
    existing user to whatever directory their config lives beside.
    """
    global_dir, _project, _workspace = layers
    _seed_agent(global_dir, "builder")

    cfg = ConfigLoader(config_dir=global_dir).load_agent("builder")

    assert cfg.permissions.workspace_root is None, (
        "with no workspace layer nothing may set a root — None still means UNCONFINED, got "
        f"{cfg.permissions.workspace_root!r}"
    )


# ---------------------------------------------------------------------------
# 6. Why no per-subagent wiring is needed
# ---------------------------------------------------------------------------

def test_every_agent_from_one_workspace_loader_gets_the_root(layers):
    """Two different agent names, ONE workspace-aware loader, both confined to the project.

    This is the mechanism that makes per-subagent confinement wiring unnecessary: `start` hands
    the subagent dispatch `load_agent=lambda n: loader.load_agent(n, bypass_cache=True)` — the
    same loader and the same injection that produced the root agent's config. A default that
    only applied to the first-loaded agent would confine the orchestrator and leave every
    delegated agent unconfined.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")
    _seed_agent(workspace, "reporter")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace)
    builder = loader.load_agent("builder", bypass_cache=True)
    reporter = loader.load_agent("reporter", bypass_cache=True)

    assert builder.permissions.workspace_root == str(project)
    assert reporter.permissions.workspace_root == str(project), (
        "every agent loaded through a workspace-aware loader must get the project root — the "
        "subagent dispatch reuses this loader, so a first-agent-only default would leave "
        "delegated agents unconfined"
    )
