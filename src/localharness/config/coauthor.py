"""Per-project co-author consent (spec 14).

The user may want localharness credited as a co-author on commits the harness helps
make — a ``Co-Authored-By:`` trailer, attribution only. Consent is asked once per
project (git repo root) and remembered in the GLOBAL config dir, never inside the
workspace. The same doctrine as ``trusted_workspaces.yaml``: a workspace that could
vouch for its own consent is not a consent boundary.

Undecided is ``None``, not ``False``: a session that could not ask must not become a
permanent "no" — only an answered prompt records anything (trust.py:13-14).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from localharness.config.defaults import COAUTHOR_EMAIL, COAUTHOR_NAME
from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

COAUTHOR_CONSENT_FILE = "coauthor_consent.yaml"


def consent_store_path() -> Path:
    """Always the GLOBAL dir — resolved at call time so tests' env changes take effect."""
    return global_config_dir() / COAUTHOR_CONSENT_FILE


def _normalize_root(project_root: str) -> str:
    """Normalize a project root to a canonical key: absolute, resolved, no trailing slash.

    Mirrors trust.py's ``_key()`` (trust.py:61-62) so the same project maps to the same
    key in both stores. ``Path.resolve()`` follows symlinks, so a symlinked checkout and
    its real path are one entry (same doctrine as trust.py:5-7)."""
    return str(Path(project_root).resolve())


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt store means "undecided", never a crashed session
        return {}
    return data if isinstance(data, dict) else {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def consent(project_root: str) -> Optional[bool]:
    """True / False / None (never asked for this project).

    ``project_root`` is the git repo root (from ``git rev-parse --show-toplevel``), not
    the workspace directory. None is the fail-closed case: no record for this project,
    so no co-author and (if interactive) the question is still owed for this project."""
    entry = _load(consent_store_path())
    projects = entry.get("projects", {})
    if not isinstance(projects, dict):
        return None
    root = _normalize_root(project_root)
    p = projects.get(root)
    if isinstance(p, dict) and isinstance(p.get("co_author"), bool):
        return p["co_author"]
    return None


def record_consent(project_root: str, granted: bool) -> None:
    """Persist a decision for one project. Permanent by design — changing the answer
    means hand-editing ``~/.localharness/coauthor_consent.yaml``, exactly as with
    workspace trust. Only a human answering the prompt gets here; a session that could
    not ask records nothing."""
    data = _load(consent_store_path())
    if not isinstance(data.get("projects"), dict):
        data["projects"] = {}
    root = _normalize_root(project_root)
    data["projects"][root] = {"co_author": granted, "recorded": _now()}
    atomic_write_overlay(consent_store_path(), data)


# --------------------------------------------------------------------------- trailer

def coauthor_trailer(granted: bool) -> str | None:
    """The credit line when consent was granted, else None.

    ``Co-Authored-By: localharness <localharness.agent@gmail.com>`` — the exact trailer
    git and GitHub render as a co-author. None means 'append nothing'."""
    if not granted:
        return None
    return f"Co-Authored-By: {COAUTHOR_NAME} <{COAUTHOR_EMAIL}>"


def prepare_commit_message(message: str, granted: bool) -> str:
    """The single integration seam. If consent was granted and the message does not
    already carry the trailer, append it on its own line. Idempotent: a message that
    already has the trailer is returned unchanged."""
    trailer = coauthor_trailer(granted)
    if trailer is None:
        return message
    if trailer in message:
        return message
    body = message.rstrip("\n")
    return f"{body}\n\n{trailer}"
