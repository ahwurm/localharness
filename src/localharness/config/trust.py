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

# `start`'s offer to CREATE a workspace remembers a "no" the same way the trust question
# remembers one — asked once per directory, ever — and keeps that memory in its own file. Two
# reasons for the sibling rather than a second key inside the trust store: this is a preference,
# not a security boundary, and a file whose whole subject is "config I agreed to load" must not
# grow entries that mean something else. Same directory, same realpath keying, same atomic write.
DECLINED_OFFERS_FILE = "declined_workspace_offers.yaml"


def trust_store_path() -> Path:
    """Always the GLOBAL dir — resolved at call time so tests' env changes take effect."""
    return global_config_dir() / WORKSPACE_TRUST_FILE


def declined_offers_path() -> Path:
    """The sibling store, in the same GLOBAL dir and resolved at call time for the same reason."""
    return global_config_dir() / DECLINED_OFFERS_FILE


def _key(workspace_dir: Path) -> str:
    return str(Path(workspace_dir).resolve())


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt store means "undecided", never a crashed session
        return {}
    return data if isinstance(data, dict) else {}


def is_trusted(workspace_dir: Path) -> Optional[bool]:
    """True / False / None (never asked). None is the fail-closed case for the caller."""
    entry = _load(trust_store_path()).get(_key(workspace_dir))
    if isinstance(entry, dict) and isinstance(entry.get("trusted"), bool):
        return entry["trusted"]
    return None


def record_trust(workspace_dir: Path, trusted: bool) -> None:
    """Persist a decision. Permanent by design — v0.13 ships no expiry and no `workspace trust`
    CLI verb (owner: "trust forever after"); changing an answer means hand-editing
    `~/.localharness/trusted_workspaces.yaml`, which the config spec documents."""
    data = _load(trust_store_path())
    data[_key(workspace_dir)] = {"trusted": trusted}
    atomic_write_overlay(trust_store_path(), data)


def offer_was_declined(workspace_dir: Path) -> bool:
    """Has the user already said no to creating THIS workspace?

    False when unreadable, missing or corrupt — the fail-open direction here, deliberately, and
    the opposite of `is_trusted`'s: the worst case is one repeated question, never config loaded
    without consent. Nothing else reads this file, so it can only ever cost a prompt.
    """
    entry = _load(declined_offers_path()).get(_key(workspace_dir))
    return isinstance(entry, dict) and entry.get("declined") is True


def record_offer_decline(workspace_dir: Path) -> None:
    """Remember a "no" so the offer is asked once per directory, ever.

    Only a person answering the prompt gets here — EOF and every non-interactive path record
    nothing, exactly as an unanswered trust prompt does. Undo it by deleting the entry (or the
    file); `init --workspace` and creating the directory by hand ignore this store entirely, so a
    recorded "no" can never stand between a user and a workspace they went and asked for.
    """
    data = _load(declined_offers_path())
    data[_key(workspace_dir)] = {"declined": True}
    atomic_write_overlay(declined_offers_path(), data)
