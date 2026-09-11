"""Durable "allow always" answers, keyed by workspace realpath (PRD §3.3).

The store lives in the GLOBAL config dir — ``~/.localharness/grants.yaml`` — and NEVER in a
project tree. That is the whole security property: a cloned repo can add deny patterns, tools
and agents (things that *ask*), but it cannot ship a file that pre-approves its own
``curl … | sh``. Phase 39 already ruled this way for workspace trust
(``config/trust.py``); grants are the same threat, so they get the same shape: one YAML map
keyed by the resolved (realpath) workspace path, atomic writes, and a decision recorded only
when a human answered a prompt.

Nested workspaces inherit (the phase-39 "nested inherits" principle): a grant recorded for
``/p`` is honored in ``/p/sub``, because ``/p/sub`` is inside the project you already
answered for. The walk is upward only — a grant in a CHILD never leaks to the parent.

Provenance (``channel``, ``session_id``, ``granted_at``) is mandatory per PRD §3.3. A record
missing any of it is not a grant anyone can audit, so it is skipped with a warning rather than
honored.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from localharness.agent.gate_types import Grant
from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

log = logging.getLogger(__name__)

GRANTS_FILE = "grants.yaml"
"""PRD §3.3: the file name inside the GLOBAL config dir. Sibling of
``trust.WORKSPACE_TRUST_FILE``, same directory, same realpath keying."""

GRANTS_KEY = "grants"
DENIES_KEY = "denies"
"""The two lists a workspace entry may hold. "Never here" answers write a deny pattern into the
same file (PRD §3.3) so one file is the whole memory of what a human answered."""

GRANT_REQUIRED_FIELDS: tuple[str, ...] = ("key", "class", "granted_at", "channel", "session_id")
"""PRD §3.3 record shape ``{key, class, granted_at, channel, session_id}``. Provenance is
mandatory: every one of these must be a non-empty string or the record is invalid."""

DENY_REQUIRED_FIELDS: tuple[str, ...] = ("pattern", "added_at", "channel", "session_id")
"""Same provenance rule for a "never here" answer, with the pattern in place of the grant key."""


def grants_store_path() -> Path:
    """Where the grant store lives (PRD §3.3).

    Always the GLOBAL dir, resolved at call time so a test's ``LOCALHARNESS_DIR`` takes effect —
    identical to ``trust.trust_store_path()`` on purpose.
    """
    return global_config_dir() / GRANTS_FILE


def _now() -> str:
    """ISO-8601 UTC, the ``granted_at`` format named in ``gate_types.Grant``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _key(workspace: Path | str) -> str:
    """The realpath string a workspace is filed under (PRD §3.3; mirrors ``trust._key``)."""
    return str(Path(workspace).expanduser().resolve())


def _ancestors(workspace: Path | str) -> tuple[str, ...]:
    """The workspace key and every ancestor key, nearest first (PRD §3.3 "nested inherits")."""
    here = Path(workspace).expanduser().resolve()
    return tuple(str(p) for p in (here, *here.parents))


def _valid(record: Any, required: tuple[str, ...]) -> bool:
    """A record is usable only when every provenance field is a non-empty string (PRD §3.3)."""
    if not isinstance(record, dict):
        return False
    return all(isinstance(record.get(f), str) and record[f].strip() for f in required)


class GrantStore:
    """Read/write the global grant store (PRD §3.3).

    ``path`` defaults to :func:`grants_store_path`; the parameter exists so tests and the ACP
    adapter can point at an explicit file. Nothing here ever consults a project tree, so a
    ``.localharness/grants.yaml`` committed to a repo is inert — it is simply never opened.

    The file is re-read on every call (like ``config/trust.py``): a grant written by a second
    session is visible to this one immediately, and a corrupt file degrades to "no grants"
    rather than a crashed turn.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path is not None else None

    @property
    def path(self) -> Path:
        """Resolved at access time so an env change (tests, ``--config-dir``) is honored."""
        return self._path if self._path is not None else grants_store_path()

    # ------------------------------------------------------------------ reads

    def _load(self) -> dict:
        path = self.path
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt store means "no grants", never a crash
            log.warning("grant store %s is unreadable; treating it as empty", path)
            return {}
        return data if isinstance(data, dict) else {}

    def _records(self, data: dict, workspace_key: str, list_key: str, required: tuple[str, ...]) -> list[dict]:
        entry = data.get(workspace_key)
        if not isinstance(entry, dict):
            return []
        raw = entry.get(list_key)
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for record in raw:
            if _valid(record, required):
                out.append(record)
            else:
                log.warning(
                    "grant store %s: skipping invalid %s record for %s (missing provenance: %s)",
                    self.path, list_key, workspace_key, ", ".join(required),
                )
        return out

    def lookup(self, workspace: Path, key: str) -> Optional[Grant]:
        """The grant covering ``key`` in ``workspace``, walking ancestors (PRD §3.3).

        Nearest workspace wins; a parent's grant is inherited by a nested workspace. Returns
        None when nothing is recorded — the caller (``agent/verdict.evaluate``) then asks.
        """
        data = self._load()
        for workspace_key in _ancestors(workspace):
            for record in self._records(data, workspace_key, GRANTS_KEY, GRANT_REQUIRED_FIELDS):
                if record["key"] == key:
                    return Grant(
                        key=record["key"],
                        klass=record["class"],
                        granted_at=record["granted_at"],
                        channel=record["channel"],
                        session_id=record["session_id"],
                        workspace=workspace_key,
                    )
        return None

    def deny_patterns_for(self, workspace: Path) -> list[str]:
        """Every "never here" pattern that applies to ``workspace``, nearest first (PRD §3.3).

        These join the DENY tier, which no grant and no mode can override.
        """
        data = self._load()
        patterns: list[str] = []
        for workspace_key in _ancestors(workspace):
            for record in self._records(data, workspace_key, DENIES_KEY, DENY_REQUIRED_FIELDS):
                if record["pattern"] not in patterns:
                    patterns.append(record["pattern"])
        return patterns

    # ----------------------------------------------------------------- writes

    def _write(self, data: dict) -> None:
        atomic_write_overlay(self.path, data)

    def add(self, grant: Grant) -> None:
        """Record an "allow always" answer (PRD §3.3).

        Replaces any existing record for the same key in the same workspace, so re-answering
        refreshes provenance instead of growing the file.
        """
        data = self._load()
        workspace_key = _key(grant.workspace)
        entry = data.get(workspace_key)
        if not isinstance(entry, dict):
            entry = {}
        grants = [g for g in entry.get(GRANTS_KEY, []) if not (isinstance(g, dict) and g.get("key") == grant.key)]
        grants.append(
            {
                "key": grant.key,
                "class": grant.klass,
                "granted_at": grant.granted_at or _now(),
                "channel": grant.channel,
                "session_id": grant.session_id,
            }
        )
        entry[GRANTS_KEY] = grants
        data[workspace_key] = entry
        self._write(data)

    def add_deny(self, workspace: Path, pattern: str, *, channel: str, session_id: str) -> None:
        """Record a "never here" answer as a deny pattern (PRD §3.3).

        Deny wins forever after: the gate's DENY tier reads these before anything else, so this
        is the one answer a later "allow always" cannot undo from a prompt.
        """
        data = self._load()
        workspace_key = _key(workspace)
        entry = data.get(workspace_key)
        if not isinstance(entry, dict):
            entry = {}
        denies = [d for d in entry.get(DENIES_KEY, []) if not (isinstance(d, dict) and d.get("pattern") == pattern)]
        denies.append(
            {
                "pattern": pattern,
                "added_at": _now(),
                "channel": channel,
                "session_id": session_id,
            }
        )
        entry[DENIES_KEY] = denies
        data[workspace_key] = entry
        self._write(data)


def new_grant(*, key: str, klass: str, workspace: Path | str, channel: str, session_id: str) -> Grant:
    """Build a :class:`Grant` with the current timestamp and a realpath workspace (PRD §3.3).

    The one constructor the channels use, so provenance can never be forgotten at a call site.
    """
    return Grant(
        key=key,
        klass=klass,
        granted_at=_now(),
        channel=channel,
        session_id=session_id,
        workspace=_key(workspace),
    )
