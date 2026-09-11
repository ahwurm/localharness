"""The confinement leash does NOT come free with the workspace layer (v0.14.1 removed 5c).

`permissions.workspace_root` is opt-in filesystem confinement for write/edit/bash_exec. v0.13
(CONF-01) made a workspace layer fill it in automatically with the folder CONTAINING
`.localharness/`, on the reasoning that the layer already names where the work lives. A live
end-to-end run showed what that actually bought: with `auto` as the mode,
`bash_exec("cat > /tmp/notes/x")` ran and `write(path="/tmp/notes/x")` was hard-refused — same
file, same session, opposite answers. The refusal came from the per-tool leash
(`tools/base.Tool._outside_workspace`), which 5c had switched on without anyone asking for it,
while the shell tool's own path string never reached it.

So the default is gone and the gate owns the boundary: it DERIVES one from where you stand
(`verdict.derive_boundary`), applies it to every tool the same way, and `auto`'s contract is that
a write outside the project runs unless it lands somewhere protected. What stays is the EXPLICIT
setting: a `workspace_root` in the agent's own yaml or the overlay's `agent:` section still
resolves exactly as it always did and is still a hard confinement (permission_denied, no prompt,
no mode that overrides it) — which is what the harness's own evals set, and now the only way the
leash switches on is that a human wrote it down.

Scope, stated plainly: these tests exercise the LOADER by constructing `ConfigLoader` with a named
workspace layer. Nothing here drives a real session; `tests/unit/test_workspace_carveouts.py` is
where the value (or its absence) is followed into the Write/Edit/BashExec instances.
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
# 1-2. No default, and neither of the two paths it used to be
# ---------------------------------------------------------------------------

def test_a_workspace_session_leaves_the_root_unset(layers):
    """A workspace applies, the agent yaml says nothing about permissions → still None.

    None means UNCONFINED (models.py), and unconfined is the honest answer here: the session is
    not unguarded, it is guarded by the ONE mechanism that sees every tool. The leash this used to
    switch on saw only three of them, which is how `bash_exec("cat > /tmp/notes/x")` ran in the
    same session that hard-refused `write(path="/tmp/notes/x")`.
    """
    global_dir, _project, workspace = layers
    _seed_agent(workspace, "builder")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root is None, (
        "a workspace layer must not invent a confinement root — the gate derives the boundary and "
        f"applies it to every tool, got {cfg.permissions.workspace_root!r}"
    )


def test_neither_the_project_nor_the_dotdir_is_injected(layers):
    """Asserted SEPARATELY from test 1 so a half-removal cannot hide behind that None.

    The two candidate values differ by a single path component and both name a real directory, so
    a re-added default — whichever of them it picked — would look plausible in any assertion that
    only checked "not the other one". This test names both and rejects both.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")
    root = cfg.permissions.workspace_root

    assert root != str(project), (
        f"the project root {root!r} was injected again — that is the leash the gate replaced"
    )
    assert root != str(workspace), (
        f"the dotdir {root!r} was injected — the wrong half of a default that should not exist"
    )
    assert root is None


# ---------------------------------------------------------------------------
# 3-4. Explicit config still wins — from BOTH sources that can set the key
# ---------------------------------------------------------------------------

def test_explicit_agent_yaml_root_is_not_overwritten(layers):
    """The agent's own yaml is the primary source, and now the ONLY one — a human wrote it down.

    A user who confined an agent to a scratch dir (the harness's own evals do exactly this) must
    not silently have that widened to the whole project just because a workspace exists.

    The explicit root lives in the GLOBAL agent yaml here. It used to live in the workspace one,
    where a root pointing outside the project is now dropped by the narrow-only union (PRD §3.3,
    tests/unit/test_mode_narrow_only_layers.py): a repo may tighten its own confinement, never
    move it outward. "Explicit config beats the default" is unchanged for every layer entitled to
    set it — which is what this asserts.
    """
    global_dir, project, workspace = layers
    _seed_agent(global_dir, "builder", permissions={"workspace_root": "/explicit/root"})

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root == "/explicit/root", (
        "an explicitly configured workspace_root must reach the session untouched by the "
        f"workspace layer — got {cfg.permissions.workspace_root!r}"
    )
    assert cfg.permissions.workspace_root != str(project), (
        "the project root must not appear here at all — neither as a replacement for the "
        "agent's explicit confinement nor as a resurrected default"
    )


def test_overlay_agent_section_root_is_not_overwritten(layers):
    """The user overlay's `agent:` section is the second place a human can write the root down.

    Both halves are asserted — the overlay value present AND the project root absent — because
    "the overlay value is there" passes just as happily when something else put a default in
    alongside it, which is exactly what 5c used to do from the line below the overlay merge.
    """
    global_dir, project, workspace = layers
    _seed_agent(workspace, "builder")
    (global_dir / "overrides.yaml").write_text(
        yaml.safe_dump({"agent": {"permissions": {"workspace_root": "/from/overlay"}}}),
        encoding="utf-8",
    )

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("builder")

    assert cfg.permissions.workspace_root == "/from/overlay", (
        "the overlay's agent.permissions.workspace_root is explicit config and must reach the "
        f"session — got {cfg.permissions.workspace_root!r}"
    )
    assert cfg.permissions.workspace_root != str(project), (
        "the project root must be ABSENT here; asserting only that the overlay value is present "
        "would pass an implementation that injected a default over it"
    )


# ---------------------------------------------------------------------------
# 5. No workspace layer — the unconfined contract is untouched
# ---------------------------------------------------------------------------

def test_no_workspace_layer_leaves_the_root_unset(layers):
    """LAYR-03: a workspace-less load never had a root and still does not.

    `None` means UNCONFINED and that is deliberate (models.py) — file-write capability is a core
    product feature. Kept as its own test because it is the case that never changed: whatever
    happens to the workspace path, a global-only session must not acquire a leash.
    """
    global_dir, _project, _workspace = layers
    _seed_agent(global_dir, "builder")

    cfg = ConfigLoader(config_dir=global_dir).load_agent("builder")

    assert cfg.permissions.workspace_root is None, (
        "with no workspace layer nothing may set a root — None still means UNCONFINED, got "
        f"{cfg.permissions.workspace_root!r}"
    )


# ---------------------------------------------------------------------------
# 6. The removal holds for every agent the loader serves
# ---------------------------------------------------------------------------

def test_no_agent_from_a_workspace_loader_is_silently_confined(layers):
    """Two different agent names, ONE workspace-aware loader, neither one leashed.

    Checked across two agents because that is the seam the removal has to hold at: `start` hands
    the subagent dispatch `load_agent=lambda n: loader.load_agent(n, bypass_cache=True)` — the
    same loader that produced the root agent's config. A default left behind on any path through
    this loader would resurrect the split answer for whichever agents took that path, and a
    subagent that refuses a write its orchestrator just made is the same bug wearing a hat.
    """
    global_dir, _project, workspace = layers
    _seed_agent(workspace, "builder")
    _seed_agent(workspace, "reporter")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace)
    builder = loader.load_agent("builder", bypass_cache=True)
    reporter = loader.load_agent("reporter", bypass_cache=True)

    assert builder.permissions.workspace_root is None
    assert reporter.permissions.workspace_root is None, (
        "every agent loaded through a workspace-aware loader is unconfined unless a human "
        "configured a root — the subagent dispatch reuses this loader"
    )
