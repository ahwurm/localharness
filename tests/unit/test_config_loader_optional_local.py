"""ConfigLoader's local layer is now opt-in (phase 39, LAYR-02/LAYR-03).

Before this, `_local_dir` defaulted to the literal `./.localharness` and was checked FIRST — so a
stray directory in whatever CWD the process happened to start in outranked an explicit
--config-dir, which LAYR-02 calls a full replacement. That peek is gone: `_local_dir` is None
unless a caller names a workspace layer. This is a deliberate, desirable behavior change, not a
regression; these tests are its record.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader, ConfigNotFoundError


def _seed(dirpath: Path, subdir: str, name: str, **fields) -> Path:
    """Write `{dirpath}/{subdir}/{name}.yaml`. Divisions forbid `role`, agents want it, so the
    caller names the fields (the plan's fixed-`role` helper cannot seed a division)."""
    d = dirpath / subdir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"name": name, **fields}), encoding="utf-8")
    return path


@pytest.fixture
def layers(tmp_path):
    """The two config bases: a global dir and a workspace `.localharness/` under a project."""
    return tmp_path / "global", tmp_path / "ws" / ".localharness"


# ---------------------------------------------------------------------------
# Inert when no workspace layer is named
# ---------------------------------------------------------------------------

def test_explicit_config_dir_is_a_full_replacement_no_cwd_peek(layers, tmp_path, monkeypatch):
    """THE regression this plan exists for: a stray `./.localharness` no longer outranks
    --config-dir. Before phase 39 `load_agent("ghost")` silently returned the CWD's file."""
    global_dir, ws = layers
    _seed(ws, "agents", "ghost", role="from a directory that merely happened to be the CWD")
    _seed(global_dir, "agents", "real", role="the named layer")
    monkeypatch.chdir(tmp_path / "ws")

    loader = ConfigLoader(config_dir=global_dir)

    assert "ghost" not in loader.list_agents()
    assert loader.list_agents() == ["real"]
    with pytest.raises(ConfigNotFoundError):
        loader.load_agent("ghost")


def test_local_dir_is_none_without_a_named_workspace(layers):
    global_dir, _ = layers
    assert ConfigLoader(config_dir=global_dir)._local_dir is None


def test_readers_run_clean_with_no_workspace_layer(layers, tmp_path, monkeypatch):
    """Every `_local_dir` consumer tolerates None and reports only the global dir's contents."""
    global_dir, ws = layers
    _seed(global_dir, "agents", "alpha", role="global alpha")
    _seed(global_dir, "divisions", "research", description="global division")
    _seed(ws, "agents", "ghost", role="unreachable")
    _seed(ws, "divisions", "phantom", description="unreachable")
    monkeypatch.chdir(tmp_path / "ws")

    loader = ConfigLoader(config_dir=global_dir)

    assert loader.list_agents() == ["alpha"]
    assert loader.list_divisions() == ["research"]
    assert [p.name for p in loader.agent_yaml_paths()] == ["alpha.yaml"]
    assert [a["name"] for a in loader.discover_agents()] == ["alpha"]
    validated = [p for p, _ in loader.validate_all()]
    assert all(str(global_dir) in p for p in validated), validated
    assert not any("ghost" in p or "phantom" in p for p in validated), validated


def test_searched_list_names_only_real_bases(layers):
    """A ConfigNotFoundError names the paths actually searched — one base, or two with a
    workspace, workspace first (it is checked first, so it wins)."""
    global_dir, ws = layers

    with pytest.raises(ConfigNotFoundError) as bare:
        ConfigLoader(config_dir=global_dir).load_agent("nope")
    assert bare.value.searched_paths == [str(global_dir / "agents" / "nope.yaml")]

    with pytest.raises(ConfigNotFoundError) as layered:
        ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_agent("nope")
    assert layered.value.searched_paths == [
        str(ws / "agents" / "nope.yaml"),
        str(global_dir / "agents" / "nope.yaml"),
    ]


def test_division_searched_list_tracks_the_same_bases(layers):
    global_dir, ws = layers

    with pytest.raises(ConfigNotFoundError) as bare:
        ConfigLoader(config_dir=global_dir).load_division("nope")
    assert bare.value.searched_paths == [str(global_dir / "divisions" / "nope.yaml")]

    with pytest.raises(ConfigNotFoundError) as layered:
        ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_division("nope")
    assert layered.value.searched_paths == [
        str(ws / "divisions" / "nope.yaml"),
        str(global_dir / "divisions" / "nope.yaml"),
    ]


# ---------------------------------------------------------------------------
# Two-tier precedence, unchanged, whenever a workspace IS named
# ---------------------------------------------------------------------------

def test_named_workspace_agent_wins_over_the_global_one(layers):
    """`load_agent` merges org/division defaults, so assert on the field we set: `role`."""
    global_dir, ws = layers
    _seed(global_dir, "agents", "shared", role="global role")
    _seed(ws, "agents", "shared", role="workspace role")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.load_agent("shared").role == "workspace role"


def test_named_workspace_division_wins_over_the_global_one(layers):
    global_dir, ws = layers
    _seed(global_dir, "divisions", "research", description="global division")
    _seed(ws, "divisions", "research", description="workspace division")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.load_division("research").description == "workspace division"


def test_agent_yaml_paths_lists_global_first_then_workspace(layers):
    """Global-first is load-bearing: `discover_agents` keys a dict by stem, so the workspace file
    must be listed LAST to win by overwrite."""
    global_dir, ws = layers
    _seed(global_dir, "agents", "beta", role="global beta")
    _seed(global_dir, "agents", "alpha", role="global alpha")
    _seed(ws, "agents", "alpha", role="workspace alpha")
    _seed(ws, "agents", "gamma", role="workspace gamma")

    paths = ConfigLoader(config_dir=global_dir, local_config_dir=ws).agent_yaml_paths()

    assert paths[0] == global_dir / "agents" / "alpha.yaml"
    assert [str(p) for p in paths] == [
        str(global_dir / "agents" / "alpha.yaml"),
        str(global_dir / "agents" / "beta.yaml"),
        str(ws / "agents" / "alpha.yaml"),
        str(ws / "agents" / "gamma.yaml"),
    ]


def test_discover_agents_gives_the_workspace_dict_for_a_colliding_stem(layers):
    global_dir, ws = layers
    _seed(global_dir, "agents", "shared", role="global role")
    _seed(ws, "agents", "shared", role="workspace role")

    agents = ConfigLoader(config_dir=global_dir, local_config_dir=ws).discover_agents()

    assert len(agents) == 1
    assert agents[0]["role"] == "workspace role"


def test_list_readers_union_both_named_layers(layers):
    global_dir, ws = layers
    _seed(global_dir, "agents", "alpha", role="global alpha")
    _seed(global_dir, "divisions", "research", description="global division")
    _seed(ws, "agents", "gamma", role="workspace gamma")
    _seed(ws, "divisions", "labs", description="workspace division")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.list_agents() == ["alpha", "gamma"]
    assert loader.list_divisions() == ["labs", "research"]
    validated = [p for p, _ in loader.validate_all()]
    assert any("gamma.yaml" in p for p in validated), validated
    assert any("labs.yaml" in p for p in validated), validated


def test_a_named_workspace_needs_no_cwd(layers, tmp_path, monkeypatch):
    """The workspace layer is whatever the caller named — never whatever the CWD happens to be."""
    global_dir, ws = layers
    _seed(ws, "agents", "only-here", role="workspace only")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.list_agents() == ["only-here"]
    assert loader.load_agent("only-here").role == "workspace only"
