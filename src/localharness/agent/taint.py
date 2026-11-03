"""Which agents have read pages saved from the web, and what they wrote since."""
from __future__ import annotations

import hashlib
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from localharness.agent.gate_types import TOOL_SCRIPT_UNCONFIRMED
from localharness.config.paths import WORKSPACE_DIR_NAME

log = logging.getLogger(__name__)

DOWNLOADS = (WORKSPACE_DIR_NAME, "downloads")

TAINTED_ASK_CLASSES = frozenset({
    "shell-destructive", "interpreter-inline", "code-exec", TOOL_SCRIPT_UNCONFIRMED, "edit-outside",
})

DISPATCHING: ContextVar[Optional[str]] = ContextVar("dispatching_agent", default=None)
"""The agent whose tool call is running now; a loop started inside that call is a subagent of it."""


def downloads_dir(boundary: Path) -> Path:
    return Path(boundary).joinpath(*DOWNLOADS)


def under_downloads(path: str | Path) -> bool:
    """Does `path`, symlinks resolved, sit in a downloads folder?"""
    try:
        parts = Path(path).expanduser().resolve().parts
    except (OSError, ValueError, RuntimeError):
        parts = Path(path).parts
    return any(parts[i:i + 2] == DOWNLOADS for i in range(len(parts) - 1))


def content_key(data: str | bytes) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return "sha256:" + hashlib.sha256(raw).hexdigest()[:12]


@dataclass
class _State:
    sources: dict[str, None] = field(default_factory=dict)
    written: dict[str, str] = field(default_factory=dict)
    grants: set[str] = field(default_factory=set)


class Taint:
    """Per agent: the saved pages it has read, the files it wrote since, and the exact-content grants given meanwhile."""

    def __init__(self) -> None:
        self._agents: dict[str, _State] = {}

    def note_read(self, agent_id: str, tool_name: str, paths: Iterable[Path | str]) -> bool:
        hits = [Path(p).name for p in paths if under_downloads(p)]
        if not hits:
            return False
        self._agents.setdefault(agent_id, _State()).sources.update(dict.fromkeys(hits))
        log.debug("%s read saved pages through %s: %s", agent_id, tool_name, hits)
        return True

    def note_write(self, agent_id: str, paths: Iterable[Path | str]) -> None:
        state = self._agents.get(agent_id)
        if state is None:
            return
        for path in paths:
            try:
                if Path(path).is_file():
                    state.written[os.path.realpath(path)] = content_key(Path(path).read_bytes())
            except OSError:
                continue

    def source(self, agent_id: Optional[str]) -> Optional[str]:
        state = self._agents.get(agent_id or "")
        if state is None:
            return None
        first, *rest = state.sources
        return f"{first} and {len(rest)} more" if rest else first

    def written(self, agent_id: str, path: Path | str) -> Optional[str]:
        state = self._agents.get(agent_id)
        return state.written.get(os.path.realpath(path)) if state else None

    def grant(self, agent_id: str, key: str) -> None:
        if (state := self._agents.get(agent_id)) is not None:
            state.grants.add(key)

    def grants(self, agent_id: str) -> frozenset[str]:
        state = self._agents.get(agent_id)
        return frozenset(state.grants) if state else frozenset()

    def inherit(self, parent: str, child: str) -> None:
        if (state := self._agents.get(parent)) is not None:
            self._agents[child] = _State(dict(state.sources), dict(state.written))

    def end(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)
