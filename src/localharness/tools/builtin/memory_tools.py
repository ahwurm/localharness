"""Memory retrieval tools: memory_search (FTS5 over fact contents) and memory_get (full body).

These serve the full persistent-fact bodies on demand so the system prompt can inline only a
small INDEX (fact names + one-line descriptions) instead of the entire MEMORY.md every turn.
"""
from datetime import datetime, time as _dtime, timedelta
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema


def resolve_time_expr(expr: str, *, end: bool = False) -> int:
    """Resolve a memory_search time expression to LOCAL epoch seconds.

    Accepts: 'today' | 'yesterday' | 'this_week' | an ISO-8601 date or datetime.
    Date-precision values resolve to start-of-day (end=False) or end-of-day 23:59:59
    (end=True) so ``until='yesterday'`` includes all of yesterday; datetime-precision values
    are used exactly. Local timezone BY DESIGN — mirrors the injected shelf's relative-day
    labels (sqlite._relative_day_label), so "today" in search means what "today" means on the
    shelf. Raises ValueError naming the accepted grammar on anything else."""
    word = expr.strip().lower()
    today_local = datetime.now().astimezone().date()  # read the clock ONCE per call
    if word == "today":
        day = today_local
    elif word == "yesterday":
        day = today_local - timedelta(days=1)
    elif word == "this_week":
        day = today_local - timedelta(days=today_local.weekday())  # most recent Monday
    else:
        try:
            parsed = datetime.fromisoformat(expr.strip())
        except ValueError:
            raise ValueError(
                f"Unrecognized time expression {expr!r}: use today|yesterday|this_week "
                "or an ISO date/datetime like 2026-07-01 or 2026-07-01T09:30"
            ) from None
        # A bare date parses to midnight — distinguish a REAL midnight datetime from a date
        # by the presence of a time separator in the raw string (the subtle part).
        if parsed.time() != _dtime.min or "T" in expr or " " in expr.strip():
            dt = parsed if parsed.tzinfo else parsed.astimezone()  # naive → local
            return int(dt.timestamp())
        day = parsed.date()
    boundary = _dtime(23, 59, 59) if end else _dtime.min
    return int(datetime.combine(day, boundary).astimezone().timestamp())


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
                "need detail that isn't already inlined. Supports time filters — e.g. "
                "since='yesterday' answers 'what did we learn yesterday?'."
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
                    "since": {
                        "type": "string",
                        "description": (
                            "Only facts updated at/after this time. Accepts 'today', "
                            "'yesterday', 'this_week', or an ISO date/datetime "
                            "(e.g. '2026-07-01', '2026-07-01T09:30')."
                        ),
                    },
                    "until": {
                        "type": "string",
                        "description": (
                            "Only facts updated at/before this time. Same formats as 'since'; "
                            "a bare date includes the whole day."
                        ),
                    },
                },
                "required": ["query"],
            },
            destructive=False,
            estimated_tokens=400,
        )

    async def _execute(
        self, query: str, limit: int = 10, since: str | None = None, until: str | None = None
    ) -> ToolResult:
        from localharness.memory.sqlite import FactQuery

        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        since_epoch = until_epoch = None
        try:
            if since:
                since_epoch = resolve_time_expr(since, end=False)
            if until:
                until_epoch = resolve_time_expr(until, end=True)
        except ValueError as exc:
            # Readable teach-back, never an exception into the loop (must_have #4). Note:
            # error_type must be a valid ToolResult Literal — 'invalid_params' is NOT one
            # (it raises a pydantic ValidationError); 'validation_error' is the correct fit.
            return self.err(str(exc), error_type="validation_error")
        try:
            facts = await self._mem.query_facts(
                FactQuery(
                    text=query, min_confidence=0.0, limit=limit,
                    since=since_epoch, until=until_epoch,
                )
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
