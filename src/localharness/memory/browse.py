"""The memory slot's transitional occupant (ROADMAP D3): browse() over this session's MemoryStore.

Phase 46 seats it from start_cmd's legacy memory block so the phone's memory screen reaches memory
only through the slot; context() and bind_subagent() stay the inherited empty ones (the legacy
memory path still feeds the prompt and the cruncher). Phase 47's memory plugin replaces it and
deletes the seating line. It is a plain object, not a bundled plugin: no manifest, never loaded."""
from __future__ import annotations

import time
from typing import Any

from localharness.memory.sqlite import USER_EDIT_PROVENANCE_PREFIX, FactQuery
from localharness.plugins.api import BrowseQuery, MemorySlotPlugin


def fact_row(f: Any) -> dict[str, Any]:
    """A fact as the phone and /memory see it — the shape web's _fact_row built, full value."""
    return {"name": f.key, "value": f.value or "", "status": f.status, "confidence": f.confidence,
            "source": f.source, "node_kind": getattr(f, "node_kind", None),
            "tags": list(f.tags or []), "updated_at": f.updated_at, "provenance": f.provenance}


class StoreBrowse(MemorySlotPlugin):
    """the transitional memory slot occupant: browse() over the session's MemoryStore"""

    def __init__(self, store: Any, router: Any = None, *, workspace_identity: str = "") -> None:
        # promote's global target is the router's one global handle — exactly what the REPL passes
        self._store, self._router, self._identity = store, router, workspace_identity

    def browse(self) -> StoreBrowse:
        return self

    async def search(self, query: BrowseQuery) -> list[dict[str, Any]]:
        facts = await self._store.query_facts(FactQuery(
            text=query.text or None, tags=list(query.tags),
            min_confidence=query.min_confidence if query.min_confidence is not None else 0.0,
            limit=query.limit, include_superseded=query.include_superseded,
            since=int(query.since.timestamp()) if query.since else None,
            until=int(query.until.timestamp()) if query.until else None))
        return [fact_row(f) for f in facts]

    async def get(self, name: str) -> dict[str, Any] | None:
        fact, history = await self._store.get_fact(name), await self._store.get_fact_history(name)
        if fact is None and not history:
            return None
        return {"fact": fact_row(fact) if fact is not None else None,
                "history": [fact_row(f) for f in history]}

    async def edit(self, name: str, content: str, origin: str = "") -> dict[str, Any]:
        """Supersede the fact's content (history kept); tags, confidence and node_kind carry
        forward — an edit that dropped them would be data loss wearing a save button."""
        current = await self._store.get_fact(name)
        if current is None:
            return {"status": "missing", "name": name}
        if content.strip() == (current.value or "").strip():
            return {"status": "unchanged", "name": name}
        await self._store.store_fact(
            key=name, value=content.strip(), tags=list(current.tags or []),
            confidence=current.confidence, source="user_edit",
            provenance=f"{USER_EDIT_PROVENANCE_PREFIX}{int(time.time())}" + (f";{origin}" if origin else ""),
            node_kind=getattr(current, "node_kind", None) or "fact")
        return {"status": "edited", "name": name}

    async def forget(self, name: str) -> bool:
        """Retire the active fact (recoverable, never deleted); False when there is none, or when a
        live turn superseded it first."""
        current = await self._store.get_fact(name)
        return current is not None and await self._store.forget_fact(current.id)
