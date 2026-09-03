"""Microagent resolution across the two config layers (AGNT-02).

the SPECIFIC beats the GENERAL: `microagents/*.md` files union across the global and workspace
layers, and a stem collision resolves to the workspace file WHOLESALE — the workspace file
replaces the global one of the same name rather than being appended to it, exactly as a
same-named agent resolves.

Scope: RESOLUTION only. `ContextConfig.microagents` is declared-only today and the
keyword-triggered injection mechanism is deliberately out of scope for v0.13.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.config.loader import ConfigLoader


def _write_md(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """The two config bases: a global dir and a workspace `.localharness/` under a project."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    global_dir.mkdir(parents=True)
    workspace_dir.mkdir(parents=True)
    return global_dir, workspace_dir


def test_global_only_microagents_resolve(layers) -> None:
    global_dir, ws = layers
    expected = _write_md(global_dir / "microagents" / "style.md", "global style")

    found = ConfigLoader(config_dir=global_dir, local_config_dir=ws).discover_microagents()

    assert found == {"style": expected}


def test_workspace_only_microagents_resolve(layers) -> None:
    global_dir, ws = layers
    expected = _write_md(ws / "microagents" / "deploy.md", "workspace deploy")

    found = ConfigLoader(config_dir=global_dir, local_config_dir=ws).discover_microagents()

    assert found == {"deploy": expected}


def test_union_contains_both_layers(layers) -> None:
    global_dir, ws = layers
    _write_md(global_dir / "microagents" / "style.md", "global style")
    _write_md(ws / "microagents" / "deploy.md", "workspace deploy")

    found = ConfigLoader(config_dir=global_dir, local_config_dir=ws).discover_microagents()

    assert set(found) == {"style", "deploy"}


def test_stem_collision_resolves_to_the_workspace_file(layers) -> None:
    global_dir, ws = layers
    _write_md(global_dir / "microagents" / "style.md", "global body")
    ws_file = _write_md(ws / "microagents" / "style.md", "workspace body")

    found = ConfigLoader(config_dir=global_dir, local_config_dir=ws).discover_microagents()

    assert found["style"].read_text(encoding="utf-8") == "workspace body"
    assert found["style"] == ws_file
    assert str(found["style"]).startswith(str(ws))


def test_microagent_paths_lists_global_first(layers) -> None:
    """The ordering contract the stem-keyed overwrite depends on. Without it the collision
    silently inverts and the global file would win."""
    global_dir, ws = layers
    g_file = _write_md(global_dir / "microagents" / "style.md", "global body")
    ws_file = _write_md(ws / "microagents" / "style.md", "workspace body")

    paths = ConfigLoader(config_dir=global_dir, local_config_dir=ws).microagent_paths()

    assert paths == [g_file, ws_file]


def test_no_microagents_dir_is_empty_not_an_error(layers) -> None:
    global_dir, ws = layers

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.microagent_paths() == []
    assert loader.discover_microagents() == {}


def test_non_md_files_are_ignored(layers) -> None:
    global_dir, ws = layers
    md = _write_md(global_dir / "microagents" / "style.md", "global style")
    _write_md(global_dir / "microagents" / "notes.txt", "not a microagent")

    loader = ConfigLoader(config_dir=global_dir, local_config_dir=ws)

    assert loader.microagent_paths() == [md]
    assert set(loader.discover_microagents()) == {"style"}
