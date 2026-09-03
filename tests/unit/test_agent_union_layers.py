"""AGNT-01 — agents resolve as a UNION by name across the two config layers.

A name present in both layers resolves to the WORKSPACE agent wholesale (no per-field merge,
v013 Risk #4); an agent that exists only globally stays loadable and stays in the roster inside
a workspace session.

The mechanism shipped in phases 38-39: `ConfigLoader._find_file()` walks `_search_bases()`
first-wins (workspace first), and `agent_yaml_paths()` deliberately lists the GLOBAL dir first so
`discover_agents()`' by-stem dict lets the workspace file win by overwrite. What did NOT exist
until this file is a proof of the collision SEMANTICS — the suite pinned the file-listing ORDER
and the None-vs-named layer question, never "and nothing bleeds through from the shadowed global
file". `deep_merge` is applied to the overlay and to org/division inheritance, so "the workspace
file wins" and "the workspace file wins WHOLESALE" are two different claims and only one of them
was written down.

The last two tests guard the write side from opposite directions. `_migrate_legacy_root_agent_yaml`
creates and DELETES files; test 6 proves it leaves a workspace's own `agents/default.yaml`
byte-identical when pointed at the global dir, and test 7 pins the `start` call site to the global
layer structurally, because the two expressions are value-identical today and no behavioral test
can tell them apart.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


def _seed_agent(base: Path, name: str, **fields) -> Path:
    """Write `{base}/agents/{name}.yaml`. The file's STEM is what discovery keys on, so it and
    the `name:` field are kept equal here — a stem/name disagreement is a different test's job."""
    d = base / "agents"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, **fields}), encoding="utf-8")
    return path


@pytest.fixture
def layers(tmp_path):
    """The two config bases, phase 39's tmp-tree shape: a global dir and a workspace
    `.localharness/` under a project. Only the GLOBAL dir gets a config.yaml — a workspace
    never makes a file mandatory, and in v0.13 its config.yaml is not read at all."""
    global_dir = tmp_path / "global"
    workspace = tmp_path / "proj" / ".localharness"
    (global_dir / "agents").mkdir(parents=True)
    (workspace / "agents").mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.safe_dump(_MINIMAL), encoding="utf-8")
    return global_dir, workspace


# ---------------------------------------------------------------------------
# The union, and what a collision means
# ---------------------------------------------------------------------------

def test_name_collision_resolves_to_the_workspace_agent_wholesale(layers):
    """The workspace file REPLACES the global one; it does not merge with it.

    `temperature` and `max_tokens` are set ONLY in the shadowed global file. If either reached
    the result, agents would be resolving per-field across layers — which would mean a workspace
    agent silently inherits half of a global agent that merely shares its name.
    """
    global_dir, workspace = layers
    _seed_agent(global_dir, "deployer", role="GLOBAL ROLE", temperature=0.11, max_tokens=1234)
    _seed_agent(workspace, "deployer", role="WORKSPACE ROLE")

    cfg = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).load_agent("deployer")

    assert cfg.role == "WORKSPACE ROLE", (
        "wholesale nearest-wins (Risk #4) — no per-field agent merge: the workspace file must be "
        f"the one that was read, got role={cfg.role!r}"
    )
    assert cfg.temperature != 0.11, (
        "wholesale nearest-wins (Risk #4) — no per-field agent merge: temperature 0.11 is set "
        "only in the SHADOWED global deployer.yaml and must not bleed into the workspace agent"
    )
    assert cfg.max_tokens != 1234, (
        "wholesale nearest-wins (Risk #4) — no per-field agent merge: max_tokens 1234 is set "
        "only in the SHADOWED global deployer.yaml and must not bleed into the workspace agent"
    )


def test_global_only_agent_remains_loadable_in_a_workspace_session(layers):
    """A workspace ADDS to the roster, it does not replace it. Opening a project must not cost
    you the agents you keep globally."""
    global_dir, workspace = layers
    _seed_agent(global_dir, "reporter", role="GLOBAL REPORTER")
    _seed_agent(workspace, "deployer", role="WORKSPACE ROLE")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace)

    assert loader.load_agent("reporter").role == "GLOBAL REPORTER"
    assert "reporter" in loader.list_agents()


def test_roster_is_the_union_of_both_layers(layers):
    """Three distinct names from two layers with one overlapping stem."""
    global_dir, workspace = layers
    _seed_agent(global_dir, "reporter", role="GLOBAL REPORTER")
    _seed_agent(global_dir, "deployer", role="GLOBAL ROLE")
    _seed_agent(workspace, "deployer", role="WORKSPACE ROLE")
    _seed_agent(workspace, "builder", role="WORKSPACE BUILDER")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace)

    assert {a["name"] for a in loader.discover_agents()} == {"reporter", "deployer", "builder"}
    assert loader.list_agents() == ["builder", "deployer", "reporter"]


def test_roster_collision_entry_is_the_workspace_file(layers):
    """The delegation/subagent half of AGNT-01.

    `discover_agents()` is the ONE roster `start` (start_cmd.py) and `agent list`
    (agent_cmd.py) both read since phase 38, and it is what the delegation registry is built
    from — so asserting the colliding entry carries the workspace role IS asserting what a
    subagent dispatch will see.
    """
    global_dir, workspace = layers
    _seed_agent(global_dir, "reporter", role="GLOBAL REPORTER")
    _seed_agent(global_dir, "deployer", role="GLOBAL ROLE")
    _seed_agent(workspace, "deployer", role="WORKSPACE ROLE")
    _seed_agent(workspace, "builder", role="WORKSPACE BUILDER")

    roster = ConfigLoader(config_dir=global_dir, local_config_dir=workspace).discover_agents()

    entry = [a for a in roster if a["name"] == "deployer"]
    assert len(entry) == 1, f"one stem must yield one roster entry, got {entry}"
    assert entry[0]["role"] == "WORKSPACE ROLE", (
        "wholesale nearest-wins (Risk #4) — no per-field agent merge: the roster entry for a "
        "colliding stem is the WORKSPACE file, whole"
    )


def test_discover_agents_is_the_only_roster_source_in_src():
    """One union means one implementation. Phase 38 collapsed three private agent-discovery
    loops into `ConfigLoader`; a fourth would silently reintroduce a roster that knows nothing
    about the workspace layer.

    Rooted at `__file__` and asserting it scanned something (39-01's rule: a CWD-relative scan
    that visited zero files reports green and proves nothing).
    """
    src_root = Path(__file__).resolve().parents[2] / "src" / "localharness"
    files = sorted(src_root.rglob("*.py"))
    assert len(files) >= 50, f"source scan visited only {len(files)} files — it scanned the wrong root"

    offenders = []
    for f in files:
        rel = f.relative_to(src_root).as_posix()
        if rel == "config/loader.py":
            continue
        lines = f.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if 'glob("*.yaml")' not in line:
                continue
            # The directory expression is often built a line or two above the glob, so judge the
            # small window, not just the matching line.
            window = "\n".join(lines[max(0, i - 3) : i + 1])
            if "agents" in window:
                offenders.append(f"{rel}:{i + 1}: {line.strip()}")

    assert offenders == [], (
        "a second agent-discovery implementation appeared — the roster union lives in "
        f"ConfigLoader.agent_yaml_paths() and nowhere else: {offenders}"
    )


# ---------------------------------------------------------------------------
# The write side: the legacy rename is global-only, guarded from both directions
# ---------------------------------------------------------------------------

def test_legacy_rename_never_touches_a_workspace_agents_dir(layers):
    """The migration WRITES and DELETES. Pointed at the global dir it must do its job there and
    leave an identically-named workspace file exactly as the user wrote it — the harness does not
    rewrite files in a project folder it was not asked to modify."""
    from localharness.cli import start_cmd

    global_dir, workspace = layers
    _seed_agent(global_dir, "default", role="R")
    ws_legacy = _seed_agent(workspace, "default", role="R")
    ws_bytes = ws_legacy.read_bytes()

    start_cmd._migrate_legacy_root_agent_yaml(global_dir / "agents")

    assert (global_dir / "agents" / "orchestrator.yaml").exists(), "the global migration did not run"
    assert not (global_dir / "agents" / "default.yaml").exists()
    assert (workspace / "agents" / "default.yaml").exists(), (
        "the migration deleted a workspace file it was never handed"
    )
    assert not (workspace / "agents" / "orchestrator.yaml").exists(), (
        "the migration wrote into a workspace it was never handed"
    )
    assert ws_legacy.read_bytes() == ws_bytes, "the workspace agent file was rewritten"


def test_start_points_the_legacy_rename_at_the_global_layer():
    """A STRUCTURAL pin, in the shape of `test_server_dir_global_pin.py` (39-03).

    `cfg_path` and `global_config_dir(config_dir)` are value-identical TODAY by construction, so
    no behavioral test can distinguish them — reverting the call site leaves test 6 above green.
    The pin exists for the day they diverge: a future change to what `cfg_path` means must not be
    able to hand a file-deleting migration a user's project directory.
    """
    start_cmd_py = (
        Path(__file__).resolve().parents[2] / "src" / "localharness" / "cli" / "start_cmd.py"
    )
    text = start_cmd_py.read_text(encoding="utf-8")
    assert len(text) > 0, f"read nothing from {start_cmd_py} — the structural scan has no subject"

    assert '_migrate_legacy_root_agent_yaml(global_config_dir(config_dir) / "agents")' in text, (
        "start's legacy-rename call site lost its global pin"
    )
    assert "_migrate_legacy_root_agent_yaml(cfg_path" not in text, (
        "the legacy rename is back on cfg_path — the layer that learns to follow a workspace"
    )
    assert "GLOBAL-ONLY, by contract" in text, (
        "the migration's docstring no longer states which layer it may be handed"
    )
