"""StoreBrowse — the memory plugin's browse API over its store (MemoryBrowse).

MemoryPlugin.browse() returns one, built over the session's MemoryStore and RecallRouter; the phone's
memory screen and /memory reach it only through core's memory slot. A plain object, not a plugin."""
from __future__ import annotations

import time
from typing import Any

from localharness.memory.sqlite import USER_EDIT_PROVENANCE_PREFIX, FactQuery
from localharness.plugins.api import BrowseQuery


def fact_row(f: Any) -> dict[str, Any]:
    """A fact as the phone and /memory see it — the shape the mobile channel's _fact_row built, full value."""
    return {"name": f.key, "value": f.value or "", "status": f.status, "confidence": f.confidence,
            "source": f.source, "node_kind": getattr(f, "node_kind", None),
            "tags": list(f.tags or []), "updated_at": f.updated_at, "provenance": f.provenance}


class StoreBrowse:
    """the memory plugin's browse API over its store (MemoryBrowse)"""

    def __init__(self, store: Any, router: Any = None, *, workspace_identity: str = "") -> None:
        # promote's global target is the router's one global handle — exactly what the REPL passes
        self._store, self._router, self._identity = store, router, workspace_identity

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

    async def promote(self, name: str) -> dict[str, Any]:
        """Copy one active fact to the machine-global store — /memory promote's rules, reused, not
        copied (supersede and refusal live in one place)."""
        fact = await self._store.get_fact(name)
        if fact is None:
            return {"promoted": False, "message": f"No memory named '{name}' — nothing to promote."}
        from localharness.cli.memory_cmd import render_promote  # memory plugin file -> memory plugin file
        message = await render_promote(
            self._store, f"{fact.id} confirm",
            promote_target=self._router.ensure_global if self._router is not None else None,
            workspace_identity=self._identity)
        return {"promoted": message.startswith("Promoted"), "message": message}

    # plugin-internal — /memory, `localharness memory` and the bench; NOT MemoryBrowse protocol verbs
    # (PAPI-04 stays five). Thin delegates with the store's own semantics: their value is the access
    # boundary (the surfaces reach the store only through the plugin's own class), not new behaviour.

    async def list_groups(self, *, limit: int | None = None, named_only: bool = False) -> list[dict[str, Any]]:
        return await self._store.list_groups(limit=limit, named_only=named_only)

    async def recent_facts(self, limit: int = 10) -> list[Any]:
        return await self._store.recent_facts(limit)

    async def get_fact(self, key: str) -> Any:
        return await self._store.get_fact(key)

    async def get_fact_by_id(self, fact_id: int) -> Any:
        return await self._store.get_fact_by_id(fact_id)

    async def get_fact_history(self, key: str) -> list[Any]:
        return await self._store.get_fact_history(key)

    async def query_facts(self, query: FactQuery) -> list[Any]:
        return await self._store.query_facts(query)

    async def forget_fact(self, fact_id: int) -> bool:
        """Retire exactly this id (M4): False when a live turn superseded it first — never the
        newer version in its place."""
        return await self._store.forget_fact(fact_id)

    async def list_archived(self, limit: int = 50, *, key: str | None = None) -> list[Any]:
        return await self._store.list_archived(limit, key=key)

    async def count_archived(self) -> int:
        return await self._store.count_archived()

    async def restore_fact(self, fact_id: int) -> bool:
        return await self._store.restore_fact(fact_id)

    async def archive_dormant(self, **kw: Any) -> Any:
        from localharness.memory.consolidation import archive_dormant_facts
        return await archive_dormant_facts(self._store, **kw)

    async def archive_listed(self, **kw: Any) -> Any:
        from localharness.memory.consolidation import archive_listed_facts
        return await archive_listed_facts(self._store, **kw)

    async def store(self, name: str, content: str, *, confidence: float = 1.0) -> None:
        """Seed one fact exactly as `store_fact(name, content, confidence=...)` — no embedding, no LLM.
        Optional, additive, on the bundled class only (NOT on the MemoryBrowse Protocol: it is
        runtime_checkable, so a sixth member would make every five-verb browse fail isinstance)."""
        await self._store.store_fact(name, content, confidence=confidence)
