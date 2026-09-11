"""PRD §3.3: `permissions.mode` and `permissions.workspace_root` join the narrow-only union.

The deny list has had this property since MERG-02 (tests/unit/test_deny_union_layers.py): a
workspace may ADD safety and never subtract it. Mode and workspace_root did NOT — `loader.py`
took whatever the highest-priority layer said, so a cloned repo's `.localharness/` could hand
itself `mode: unattended` (never ask again) or a confinement root pointing at `$HOME`.

Same fixture shape as the deny-union file for the same reason: every scenario authors config the
way a real install has it, so a passing test is not passing through a mechanism nobody uses.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import DEFAULT_MODE
from localharness.config.loader import ConfigLoader


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
    """A global config dir with a minimal config.yaml, and a workspace `.localharness/` under a
    project dir. The derivable boundary is therefore `tmp_path/proj`."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(global_dir / "agents" / "deployer.yaml", {"name": "deployer", "role": "Deploy agent"})
    return global_dir, workspace_dir


def _permissions(global_dir: Path, workspace_dir: Path):
    loader = ConfigLoader(config_dir=global_dir, local_config_dir=workspace_dir)
    return loader.load_agent("deployer").permissions


# ---------------------------------------------------------------------------
# mode: a project layer may only RAISE strictness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("looser", ["unattended", "trusted", "auto"])
def test_a_workspace_cannot_loosen_the_session_mode(layers, looser, caplog) -> None:
    """The whole point of the spine: a repo cannot switch the gate off for the session it runs in.

    `auto` is in the list because the legacy spelling maps to `guarded` — a workspace that says
    `auto` against a global `read-only` must be narrowed like any other loosening value, not wave
    through as an unrecognized string.
    """
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": "read-only"}})
    _write_yaml(ws / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": looser}})

    with caplog.at_level(logging.WARNING):
        perms = _permissions(global_dir, ws)

    assert perms.mode == "read-only", "a workspace loosened the session mode"
    assert any("permissions.mode" in r.getMessage() for r in caplog.records)


def test_the_warning_names_both_values(layers, caplog) -> None:
    """A dropped setting that says nothing is its own bug report: the user must be able to see
    WHICH value was ignored and what won."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": "unattended"}})

    with caplog.at_level(logging.WARNING):
        _permissions(global_dir, ws)

    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "unattended" in warning and DEFAULT_MODE in warning


def test_a_workspace_may_tighten_the_session_mode(layers) -> None:
    """The narrowing direction stays open — a project that wants to be read-only gets to say so."""
    global_dir, ws = layers
    _write_yaml(ws / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": "read-only"}})

    assert _permissions(global_dir, ws).mode == "read-only"


def test_the_global_layer_may_still_loosen_its_own_mode(layers) -> None:
    """The rule is about the REPO, not about strictness for its own sake. The operator's own
    global config still sets whatever mode it likes, inside a workspace session too."""
    global_dir, ws = layers
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": "unattended"}})

    assert _permissions(global_dir, ws).mode == "unattended"


def test_a_workspaceless_session_is_untouched(tmp_path: Path) -> None:
    """LAYR-03: with no workspace layer, nothing in the narrow-only path may fire."""
    global_dir = tmp_path / "global"
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    _write_yaml(global_dir / "agents" / "deployer.yaml",
                {"name": "deployer", "role": "Deploy agent", "permissions": {"mode": "trusted"}})

    cfg = ConfigLoader(config_dir=global_dir).load_agent("deployer")
    assert cfg.permissions.mode == "trusted"
    assert cfg.permissions.workspace_root is None, "unconfined is still the workspace-less default"


# ---------------------------------------------------------------------------
# workspace_root: a project layer may only confine INSIDE its own project
# ---------------------------------------------------------------------------

def test_a_workspace_root_outside_the_project_is_dropped(layers, tmp_path, caplog) -> None:
    """PRD §3.1: the boundary is DERIVED, never configured. A repo pointing its own leash at
    `$HOME` (or anywhere else outside the project) is the loosening this closes."""
    global_dir, ws = layers
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _write_yaml(ws / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"workspace_root": str(outside)},
    })

    with caplog.at_level(logging.WARNING):
        perms = _permissions(global_dir, ws)

    # Dropped back to the 5c default: the project folder containing `.localharness/`.
    assert Path(perms.workspace_root) == ws.parent
    warning = "\n".join(r.getMessage() for r in caplog.records)
    assert "workspace_root" in warning and str(outside) in warning


def test_a_workspace_root_inside_the_project_is_kept(layers) -> None:
    """Narrowing stays available: a project may confine an agent to a subfolder of itself."""
    global_dir, ws = layers
    inner = ws.parent / "sandbox"
    inner.mkdir()
    _write_yaml(ws / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"workspace_root": str(inner)},
    })

    assert Path(_permissions(global_dir, ws).workspace_root) == inner


def test_the_global_layer_may_still_name_a_root_anywhere(layers, tmp_path) -> None:
    """The operator's own config is not the threat. A global `workspace_root` outside the project
    is a deliberate choice (harness-run evals point one at an isolated scratch dir) and stands."""
    global_dir, ws = layers
    scratch = tmp_path / "eval-scratch"
    scratch.mkdir()
    _write_yaml(global_dir / "agents" / "deployer.yaml", {
        "name": "deployer", "role": "Deploy agent",
        "permissions": {"workspace_root": str(scratch)},
    })

    assert Path(_permissions(global_dir, ws).workspace_root) == scratch


def test_the_5c_default_still_applies_when_nothing_sets_a_root(layers) -> None:
    """CONF-01 regression: the confinement leash that comes free with the workspace layer must
    survive the new check — it runs in the same block."""
    global_dir, ws = layers
    assert Path(_permissions(global_dir, ws).workspace_root) == ws.parent
