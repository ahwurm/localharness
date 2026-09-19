"""Memory retrieval tools — retrieval by RESONANCE in the model's own space.

memory_search embeds the query with the subject-family model and ranks stored facts
by cosine in that learned space (memory/resonance.py) — the model IS the connection
between the present and the past. There is NO lexical, statistical, or hashing
fallback behind these tools: if the resonance engine cannot run, the tool says so
loudly (owner order: model as the connection, nothing else).

remember persists one fact as a BET: birth truth comes from the writer's measured
track record, the content is embedded at write time, and a re-sighting of an
existing claim (judged by the model, not by string rules) lands as EVIDENCE on the
existing row instead of a duplicate.
"""
import asyncio
import logging
from datetime import datetime, time as _dtime, timedelta
from typing import Any

from localharness.tools.base import Tool, ToolResult, ToolSchema

log = logging.getLogger(__name__)

# Bounded wall-clock for remember()'s same-claim model look. Machine safety, not memory
# math: the in-flight generation is cancelled at the budget (releasing the serial
# inference gate) and the write proceeds as a plain new row.
_REMEMBER_FOLD_BUDGET_S = 6.0


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


def _engine_error(exc: Exception) -> str:
    return f"Memory unavailable — the resonance engine failed: {exc}"


class MemorySearchTool(Tool):
    """Search persistent memory by meaning: the query is embedded in the subject-family
    model's space and facts rank by resonance (cosine). No FTS, no fallback."""

    def __init__(self, memory_store: Any, engine: Any = None) -> None:
        self._mem = memory_store
        self._engine = engine

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="memory_search",
            group="memory",
            description=(
                "Search your persistent memory by meaning for a query string. Returns the "
                "most resonant fact names with a short snippet. Use memory_get(name) for a "
                "match's full body. The system prompt shows only an index, so search when you "
                "need detail that isn't already inlined. Supports time filters — e.g. "
                "since='yesterday' answers 'what did we learn yesterday?'."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for — matched by meaning, not exact words.",
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
        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        if self._engine is None:
            return self.err(
                "Memory search unavailable: no resonance engine is wired. Retrieval runs "
                "in the model's representation space and has no fallback path.",
                error_type="execution_error",
            )
        since_epoch = until_epoch = None
        try:
            if since:
                since_epoch = resolve_time_expr(since, end=False)
            if until:
                until_epoch = resolve_time_expr(until, end=True)
        except ValueError as exc:
            return self.err(str(exc), error_type="validation_error")
        try:
            qvec = await asyncio.to_thread(self._engine.embed_query, query)
        except Exception as exc:
            return self.err(_engine_error(exc), error_type="execution_error")
        try:
            hits = await self._mem.resonance_search(
                qvec, limit=limit, since=since_epoch, until=until_epoch
            )
        except Exception as exc:
            return self.err(f"Memory search failed: {exc}")
        # Counts role: every retrieval moment is recorded, hits or not — what was asked
        # is as much a measurement as what answered. A trace-write failure never fails
        # the search.
        hit_ids = [f.id for f, _score in hits]
        rec = getattr(self._mem, "record_activation_trace", None)
        if rec is not None:
            try:
                await rec(stimulus=query, fired_ids=hit_ids, injected_ids=hit_ids,
                          source="memory_search")
            except Exception:
                log.warning("activation-trace write failed (memory_search)", exc_info=True)
        if not hits:
            return self.ok(f"No memories resonated with '{query}'.")
        # Reads bump STAGED counters only: ranking learns from use without ever
        # reordering the injected block mid-conversation.
        touch = getattr(self._mem, "touch_staged", None)
        if touch is not None:
            try:
                await touch([f.key for f, _score in hits])
            except Exception:
                pass  # staging is best-effort; retrieval must never fail on it
        lines = []
        for f, score in hits:
            snippet = (f.value or "").strip().replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:159] + "…"
            lines.append(f"- {f.key}: {snippet}")
        return self.ok("\n".join(lines), match_count=len(hits))


class MemoryRememberTool(Tool):
    """Persist one durable fact as a bet (memory spec, write side). Writes route through
    MemoryStore.store_fact — supersede-not-overwrite + read-back-verified; birth truth is
    the writer's measured track record; the content is embedded at write time so the fact
    can resonate with future presents."""

    def __init__(self, memory_store: Any, llm: Any = None, engine: Any = None) -> None:
        self._mem = memory_store
        # An optional text-completion LLM (LLMTextAdapter in prod) enables the same-claim
        # model look. None (tests / no model) keeps the plain write.
        self._llm = llm
        self._engine = engine

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="remember",
            group="memory",
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
                        "description": "Optional tags for grouping.",
                    },
                },
                "required": ["name", "content"],
            },
            # It mutates the store — a write, and one that supersedes an existing name. The flag
            # feeds the permission gate alone (`agent/verdict`), where it is what keeps `remember`
            # out of read-only mode; the memory group still keeps it out of every ask tier.
            destructive=True,
            estimated_tokens=200,
        )

    async def _execute(self, name: str, content: str, tags: Any = None) -> ToolResult:
        from localharness.memory import resonance as _res

        if self._mem is None:
            return self.err("No memory store available.", error_type="execution_error")
        if self._engine is None:
            return self.err(
                "Remember unavailable: no resonance engine is wired. Every memory is "
                "embedded at write time and there is no fallback path.",
                error_type="execution_error",
            )
        clean_name = (name or "").strip()
        clean_content = (content or "").strip()
        if not clean_name or not clean_content:
            return self.err("Both 'name' and 'content' must be non-empty.", error_type="validation_error")
        try:
            vec = await asyncio.to_thread(
                self._engine.embed_docs, [f"{clean_name}: {clean_content}"]
            )
        except Exception as exc:
            return self.err(_engine_error(exc), error_type="execution_error")
        blob = _res.pack(vec[0])

        # Re-sighting check (spec: "a re-sighting of an existing claim is EVIDENCE on
        # that row, never a new row"). The MODEL judges same-claim — top resonant
        # existing fact under a different name, one budget-capped yes/no look. Any
        # failure or timeout falls through to a plain new-row write.
        folded_into = None
        if self._llm is not None:
            try:
                candidates = await self._mem.resonance_search(vec[0], limit=2)
                cand = next(
                    (f for f, _s in candidates if f.key != clean_name), None
                )
                if cand is not None:
                    same = await asyncio.wait_for(
                        self._llm.complete(
                            "Do these two statements make the same claim?\n"
                            f"A: {cand.key}: {cand.value}\n"
                            f"B: {clean_name}: {clean_content}\n"
                            "Answer with exactly one word, yes or no."
                        ),
                        timeout=_REMEMBER_FOLD_BUDGET_S,
                    )
                    if isinstance(same, str) and same.strip().lower().startswith("yes"):
                        folded_into = cand
            except Exception:
                log.debug("remember same-claim look failed (non-fatal)", exc_info=True)

        tag_list = [str(t) for t in (tags or [])] + ["remember"]
        try:
            if folded_into is not None:
                fact = await self._mem.store_fact(
                    key=folded_into.key,
                    value=folded_into.value,
                    tags=folded_into.tags,
                    source="remember",
                )
                return self.ok(
                    f"Reinforced existing memory '{fact.key}' — the model judged this the "
                    "same claim (evidence added, no duplicate row).",
                    fact_key=fact.key,
                )
            fact = await self._mem.store_fact(
                key=clean_name,
                value=clean_content,
                tags=tag_list,
                source="remember",
                embedding=blob,
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
            group="memory",
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
        # Counts role: a memory_get surfaces one atom — a recall event
        # (stimulus=name, fired=injected=[that atom]). Best-effort (wrap + warn).
        rec = getattr(self._mem, "record_activation_trace", None)
        if rec is not None:
            try:
                await rec(stimulus=name, fired_ids=[fact.id], injected_ids=[fact.id],
                          source="memory_get")
            except Exception:
                log.warning("activation-trace write failed (memory_get)", exc_info=True)
        return self.ok(fact.value)
