"""Which workspaces from OUTSIDE the current project has the user agreed to load config from
(LAYR-05).

The record lives in the GLOBAL config dir, never inside the workspace being judged — a
workspace that could vouch for itself is not a trust boundary. Keyed by the resolved
(realpath) `.localharness/` path, so a symlinked checkout and its real path are one entry
and a git worktree is a separate workspace by design (v0.13 ruling).

Only consulted for workspaces the caller has already judged to be outside the project the
user is standing in — a workspace inside your own repository loads without ever reaching this
module (owner ruling 2026-09-03). See cli/workspace.resolve_workspace_layer.

Undecided is None, not False: a declined-in-a-script session must not become a permanent
"no" — only an answered prompt records anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

WORKSPACE_TRUST_FILE = "trusted_workspaces.yaml"


def trust_store_path() -> Path:
    """Always the GLOBAL dir — resolved at call time so tests' env changes take effect."""
    return global_config_dir() / WORKSPACE_TRUST_FILE


def _key(workspace_dir: Path) -> str:
    return str(Path(workspace_dir).resolve())


def _load() -> dict:
    path = trust_store_path()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt store means "undecided", never a crashed session
        return {}
    return data if isinstance(data, dict) else {}


def is_trusted(workspace_dir: Path) -> Optional[bool]:
    """True / False / None (never asked). None is the fail-closed case for the caller."""
    entry = _load().get(_key(workspace_dir))
    if isinstance(entry, dict) and isinstance(entry.get("trusted"), bool):
        return entry["trusted"]
    return None


def record_trust(workspace_dir: Path, trusted: bool) -> None:
    """Persist a decision. Permanent by design — v0.13 ships no expiry and no `workspace trust`
    CLI verb (owner: "trust forever after"); changing an answer means hand-editing
    `~/.localharness/trusted_workspaces.yaml`, which the config spec documents."""
    data = _load()
    data[_key(workspace_dir)] = {"trusted": trusted}
    atomic_write_overlay(trust_store_path(), data)
