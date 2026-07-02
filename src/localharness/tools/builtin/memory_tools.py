"""Memory retrieval tools: memory_search (FTS5 over fact contents) and memory_get (full body).

These serve the full persistent-fact bodies on demand so the system prompt can inline only a
small INDEX (fact names + one-line descriptions) instead of the entire MEMORY.md every turn.
"""
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema


class MemorySearchTool(Tool):
    """Search persistent-fact contents. Uses the existing FTS5 table (facts_fts) via
    MemoryStore.query_facts — lower risk than a fresh LIKE scan because the schema already
    defines facts_fts with INSERT/UPDATE/DELETE triggers that keep it in sync."""

    def __init__(self, memory_store: Any) -> None:
        self._mem = memory_store

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="memory_search",
            description=(
                "Search your persistent memory (fact names, values, tags) for a query string. "
                "Returns matching fact names with a short snippet. Use memory_get(name) for a "
                "match's full body. The system prompt shows only an index, so search when you "
                "need detail that isn't already inlined."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms to match against fact contents.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max matches to return. Default: 10.",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["query"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(self, query: str, limit: int = 10) -> ToolResult:
        from localharness.memory.sqlite import FactQuery

        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        try:
            facts = await self._mem.query_facts(
                FactQuery(text=query, min_confidence=0.0, limit=limit)
            )
        except Exception as exc:
            return self.err(f"Memory search failed: {exc}")
        if not facts:
            return self.ok(f"No facts matched '{query}'.")
        # Reads bump STAGED counters only (RANK-04): ranking learns from use without
        # ever reordering the injected block mid-conversation.
        touch = getattr(self._mem, "touch_staged", None)
        if touch is not None:
            try:
                await touch([f.key for f in facts])
            except Exception:
                pass  # staging is best-effort; retrieval must never fail on it
        lines = []
        for f in facts:
            snippet = (f.value or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:159] + "…"
            # Critic M4: unvetted candidates must never read with the same authority as
            # confirmed facts — mark them until consolidation promotes them.
            marker = " [pending]" if "pending_consolidation" in getattr(f, "tags", []) else ""
            lines.append(f"- {f.key}{marker}: {snippet}")
        # Structure-aware retrieval (HIER-03): the FTS hit is the ENTRY POINT; the graph
        # supplies the neighborhood — a leaf hit surfaces its gist/schema context, a
        # schema hit surfaces its members. Gist routes; verbatim answers.
        nbhd = getattr(self._mem, "neighborhood", None)
        by_ids = getattr(self._mem, "get_facts_by_ids", None)
        top_id = getattr(facts[0], "id", 0)
        if nbhd is not None and by_ids is not None and top_id:
            try:
                walk = await nbhd(top_id, depth=1, limit=6)
                rel = await by_ids([nid for nid, d in walk if d > 0])
                if rel:
                    lines.append(
                        "Related (graph neighborhood of top hit): "
                        + ", ".join(f"{f.key} [{f.node_kind}]" for f in rel)
                    )
            except Exception:
                pass  # the neighborhood is enrichment; search must never fail on it
        return self.ok("\n".join(lines), match_count=len(facts))


class MemoryRememberTool(Tool):
    """Persist one durable fact (WRITE-01). Writes route through MemoryStore.store_fact —
    supersede-not-overwrite + read-back-verified; a conflicting name supersedes the old
    version (history kept, retrievable via get_fact_history)."""

    def __init__(self, memory_store: Any) -> None:
        self._mem = memory_store

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="remember",
            description=(
                "Save one durable fact to your persistent memory so future sessions can use it. "
                "Use a short stable name (e.g. 'deploy-requires-vpn') and a self-contained "
                "content sentence. Writing an existing name with new content supersedes the old "
                "version (history is kept). Use for things worth knowing NEXT session — not "
                "scratch state for the current task."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short stable fact name/key (shown in the memory index).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The fact body — self-contained, understandable without this conversation.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for grouping/decay classes.",
                    },
                },
                "required": ["name", "content"],
            },
            destructive=False,
            estimated_tokens=200,
        )

    async def _execute(self, name: str, content: str, tags: Any = None) -> ToolResult:
        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        clean_name = (name or "").strip()
        clean_content = (content or "").strip()
        if not clean_name or not clean_content:
            return self.err("Both 'name' and 'content' must be non-empty.", error_type="validation_error")
        tag_list = [str(t) for t in (tags or [])] + ["remember"]
        try:
            fact = await self._mem.store_fact(
                key=clean_name,
                value=clean_content,
                tags=tag_list,
                confidence=0.9,
                source="remember",
            )
        except Exception as exc:
            return self.err(f"Remember failed: {exc}")
        return self.ok(
            f"Remembered '{fact.key}' (read-back verified).",
            fact_key=fact.key,
        )


class MemoryGetTool(Tool):
    """Return one persistent fact's full body by its exact name/key."""

    def __init__(self, memory_store: Any) -> None:
        self._mem = memory_store

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="memory_get",
            description=(
                "Return the full body of one persistent fact by its exact name (the name shown "
                "in the memory index or returned by memory_search). Pass history=true to see "
                "the fact's full version history (superseded values are kept, never deleted)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The exact fact name/key to fetch.",
                    },
                    "history": {
                        "type": "boolean",
                        "description": "If true, return all versions newest-first (default: current only).",
                        "default": False,
                    },
                },
                "required": ["name"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(self, name: str, history: bool = False) -> ToolResult:
        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        # The explicit-request path to the past (WRITE-02's door — critic m8).
        if history and hasattr(self._mem, "get_fact_history"):
            try:
                versions = await self._mem.get_fact_history(name)
            except Exception as exc:
                return self.err(f"Memory get failed: {exc}")
            if not versions:
                return self.err(f"No fact named '{name}'.", error_type="not_found")
            lines = [
                f"[{v.status}{'' if v.status == 'active' else ''}] {v.value}"
                for v in versions
            ]
            return self.ok("\n---\n".join(lines), version_count=len(versions))
        try:
            fact = await self._mem.get_fact(name)
        except Exception as exc:
            return self.err(f"Memory get failed: {exc}")
        if fact is None:
            return self.err(f"No fact named '{name}'.", error_type="not_found")
        touch = getattr(self._mem, "touch_staged", None)
        if touch is not None:
            try:
                await touch([fact.key])
            except Exception:
                pass
        return self.ok(fact.value)
