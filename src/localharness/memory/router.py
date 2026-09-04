"""Scope-aware recall routing — the ONE read gate above scope-naive memory stores.

The ruled shape (v0.13 MEMS-02): a CALLER-LEVEL router holding two store handles, answering every
read according to `agent.memory.recall_scope` as a scoped-first labeled merge — the scoped store's
hits first, each line carrying which store it came from — while `MemoryStore` itself stays
single-store. No store learns about a second store; the router composes two of them.

READ-SIDE ONLY. There is no write verb on this object and no `__getattr__` passthrough — see
`RecallRouter`'s docstring for why that absence is the mechanism rather than a convention.

The second (machine-global) handle arrives CONSTRUCTED BUT NOT OPENED and is opened lazily, at
most once. `MemoryStore.open()` mkdirs, connects and MIGRATES — it WRITES — so a default-scope
session must never call it. That laziness is MEMS-03's precondition ("the global store is
byte-identical after a workspace session"), and it is why `ensure_global()` exists at all
instead of the handle simply being opened by whoever constructs the router.

When the knob DOES ask for the second store, the boundary is still one-way, and this is the
whole of it (v0.13 B1): the twin is opened with `owner_init=False` (no legacy adoption, no tag
seeding — `MemoryStore.open`'s docstring lists exactly what such an open does and does not
write), and the enrichment verbs below write ONLY when the store they would write to is the
primary. The honest residue: a non-owner open still creates the database file and applies
schema migrations if it is missing or behind, and a scope:global session leaves the global
store's access counts and traces un-updated by its reads.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import replace
from typing import Any

ORIGIN_WORKSPACE = "workspace"
ORIGIN_GLOBAL = "global"
SCOPE_BOTH = "both"
RECALL_SCOPES = (ORIGIN_WORKSPACE, ORIGIN_GLOBAL, SCOPE_BOTH)

# `[workspace#12]` / `[global#7]` — the composite token the injected index renders in `both`
# mode and `memory_get` accepts back. Composite because `facts.id` is AUTOINCREMENT PER
# DATABASE (sqlite.py:316-321): workspace #12 and global #12 are unrelated rows, so a bare int
# is ambiguous the moment two stores are in play. Anchored and whitespace-tolerant: a model
# that echoes the token with a stray space still resolves, a fact whose KEY merely contains
# brackets does not.
_ORIGIN_TOKEN_RE = re.compile(rf"^\s*\[({ORIGIN_WORKSPACE}|{ORIGIN_GLOBAL})#(\d+)\]\s*$")


def format_origin_token(origin: str, fact_id: int) -> str:
    """The composite handle for one row of one store: `[workspace#12]`."""
    return f"[{origin}#{int(fact_id)}]"


def parse_origin_token(token: str) -> tuple[str, int] | None:
    """('workspace', 12) for a composite token, None for an ordinary fact key."""
    m = _ORIGIN_TOKEN_RE.match(token or "")
    return (m.group(1), int(m.group(2))) if m else None


# The two composition strings for `both` mode. Named, not inlined: the merged block is an
# injected prompt surface, and a bare literal buried in an f-string is exactly the shape that
# gets reflowed by accident.
_MERGED_HEADER = "## Global Memory (merged in)"
_MERGED_PREAMBLE = (
    "Memory below is merged from TWO stores: this project's own memory first, then the "
    "machine-global memory. Every line carries an origin token — `[workspace#id]` or "
    "`[global#id]` — and passing that exact token to `memory_get` returns that fact from the "
    "store the token names.\n\n"
)

# The global block's session shelf is suppressed: the shelf is THIS project's working history,
# and the global store's own sittings are a different project's story.
_MERGED_GLOBAL_SESSION_HISTORY = 0


class RecallRouter:
    """The ONE read gate for scope-aware recall (v0.13 MEMS-02).

    Holds this session's own MemoryStore and, when a workspace layer applies, a SECOND handle
    on the machine-global store. `recall_scope` decides which one(s) a read sees; ambient
    injection and the on-demand memory tools both read through this object, so the knob cannot
    be bypassed by using a tool instead of the prompt.

    READ-SIDE ONLY, and the absence of write verbs is the mechanism: there is no `store_fact`,
    no `set_current_session`, no `__getattr__` passthrough. `remember`, the write gate, the
    predictive gate, consolidation and compaction-gist persistence all keep the raw primary
    store, so `recall_scope: global` changes what a session READS and never where it WRITES
    (42-RESEARCH Pitfall 2). A stray write call against this object raises AttributeError
    loudly instead of landing silently in the wrong database.

    The two verbs that DO write — `touch_staged` and `record_activation_trace`, the enrichment
    memory_search/memory_get look up by `getattr` — are the exception that proves it: they
    write only when the store they would write to is the primary (`_enrichment_target`). A
    non-primary store is a read-only VIEW here.

    The global handle arrives CONSTRUCTED BUT NOT OPENED and is opened lazily, at most once:
    `MemoryStore.open()` creates and migrates its memory.db, so a default-scope session must
    never call it — that is exactly MEMS-03's "the global store is byte-identical afterwards".
    """

    def __init__(
        self, primary: Any, global_store: Any = None, *, scope: str = ORIGIN_WORKSPACE
    ) -> None:
        self._primary = primary
        self._global_store = global_store          # constructed, NOT opened; None = no workspace
        self._scope = scope if scope in RECALL_SCOPES else ORIGIN_WORKSPACE
        self._global_opened = False
        self._open_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Scope + lifecycle
    # ------------------------------------------------------------------

    @property
    def scope(self) -> str:
        """The EFFECTIVE scope. With no second store there is exactly one store to read, so
        all three configured values collapse to 'workspace' HERE, in one place — that collapse
        is LAYR-03's mechanism for this subsystem, not a defence repeated at each call site."""
        return self._scope if self._global_store is not None else ORIGIN_WORKSPACE

    @property
    def configured_scope(self) -> str:
        """What config asked for, before the collapse — for doctor/diagnostics only."""
        return self._scope

    async def ensure_global(self) -> Any | None:
        """Open (once) and return the global handle, or None when this session has no second
        store. Public because `/memory promote` needs the same single handle (42-04) — one
        owner, one lifecycle, never two connections to one database file (Pitfall 6).

        Opened with `owner_init=False`: this session is not that store's owner, so the open
        performs neither the Phase-33.1 legacy adoption (a RENAME of another install's memory
        directory) nor the tag seeding. It still creates the file and applies pending schema
        migrations if the database is missing or behind — see `MemoryStore.open`'s docstring
        for exactly what a non-owner open does and does not write."""
        if self._global_store is None:
            return None
        async with self._open_lock:
            if not self._global_opened:
                await self._global_store.open(owner_init=False)
                self._global_opened = True
        return self._global_store

    async def close(self) -> None:
        """Close ONLY what this router opened. The primary belongs to the caller and is closed
        by the caller's own shutdown ordering."""
        if self._global_store is not None and self._global_opened:
            self._global_opened = False
            await self._global_store.close()

    async def _read_store(self) -> Any:
        """The single store that single-scope reads go through."""
        if self.scope == ORIGIN_GLOBAL:
            g = await self.ensure_global()
            if g is not None:
                return g
        return self._primary

    async def _store_for_origin(self, origin: str) -> Any | None:
        """Resolve a token's named store — but only when the CURRENT scope may read it. A
        `[global#7]` token pasted into a workspace-scope session resolves to None, so the knob
        is honoured even by the addressing form (MEMS-02 criterion 4)."""
        if origin == ORIGIN_WORKSPACE:
            return self._primary if self.scope in (ORIGIN_WORKSPACE, SCOPE_BOTH) else None
        if self.scope in (ORIGIN_GLOBAL, SCOPE_BOTH):
            return await self.ensure_global()
        return None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def load_context(
        self, index_mode: bool = True, max_session_history: int = 8
    ) -> Any:
        """The ambient-injection read. Same signature as `MemoryStore.load_context`.

        In `both` mode the two indexes are composed scoped-first, each block labelled with its
        origin token, under ONE merge preamble. Two fields are deliberately not exact:

        * `fact_count`'s second term is the global store's INJECTED count, not a second COUNT
          query — the field has no consumer in `src/` outside the dataclass itself (measured),
          so the hot path stays one query per store rather than buying an unread number.
        * `injected_fact_ids` carries the PRIMARY's ids ONLY (see below).

        `index_mode=False` inlines each store's whole MEMORY.md under the same header. That
        legacy dump has no per-fact lines, so it carries no origin tokens and no ids.
        """
        if self.scope != SCOPE_BOTH:
            store = await self._read_store()
            ctx = await store.load_context(
                index_mode=index_mode, max_session_history=max_session_history
            )
            # `injected_fact_ids` may only ever carry ids the PRIMARY owns: the loop records
            # the ambient trace on its own store handle, and facts.id is per-database. A
            # global-scope read therefore reports an EMPTY injected set — the already-supported
            # "empty shelf still records a row" shape (#96) — never ids from another database.
            return ctx if store is self._primary else replace(ctx, injected_fact_ids=[])

        ws_ctx = await self._primary.load_context(
            index_mode=index_mode, max_session_history=max_session_history
        )
        g = await self.ensure_global()
        if g is None:                      # unreachable while `scope` collapses; kept explicit
            return ws_ctx
        if index_mode:
            # The primary's index is rendered twice here (once inside load_context above for the
            # safety-context fields and the true fact_count, once labelled below). Two local
            # SQLite SELECTs; the alternative — reaching past load_context for guardrails,
            # division and the count — trades a measurable cost for an unmeasurable one.
            ws_md, ws_ids = await self._primary._render_memory_index_with_ids(
                max_session_history, origin_label=ORIGIN_WORKSPACE
            )
            g_md, g_ids = await g._render_memory_index_with_ids(
                _MERGED_GLOBAL_SESSION_HISTORY, origin_label=ORIGIN_GLOBAL, include_preamble=False
            )
        else:
            g_ctx = await g.load_context(
                index_mode=False, max_session_history=_MERGED_GLOBAL_SESSION_HISTORY
            )
            ws_md, ws_ids = ws_ctx.agent_memory_md, list(ws_ctx.injected_fact_ids)
            g_md, g_ids = g_ctx.agent_memory_md, []
        merged_md = f"{_MERGED_PREAMBLE}{ws_md}\n\n{_MERGED_HEADER}\n\n{g_md}"
        # guardrails/division come from ws_ctx UNCHANGED: both stores derive them from the same
        # global_base_dir, so concatenating would inject the org's safety voice twice (Pitfall 3).
        return replace(
            ws_ctx,
            agent_memory_md=merged_md,
            fact_count=ws_ctx.fact_count + len(g_ids),
            token_estimate=len(merged_md + ws_ctx.division_md + ws_ctx.guardrails_md) // 4,
            injected_fact_ids=ws_ids,
        )

    async def query_facts(self, query: Any) -> list[Any]:
        """The memory_search read. `both` is scoped-first with a key-level dedup, then the
        caller's own limit — the cut happens AFTER the merge, so this project's facts are
        never crowded out by the machine-global store's."""
        if self.scope != SCOPE_BOTH:
            return await (await self._read_store()).query_facts(query)

        primary_hits = await self._primary.query_facts(query)
        g = await self.ensure_global()
        if g is None:
            return primary_hits
        seen = {f.key for f in primary_hits}
        merged = list(primary_hits)
        for f in await g.query_facts(query):
            if f.key not in seen:   # a name in both stores resolves to THIS project's version
                merged.append(f)
        limit = int(getattr(query, "limit", 0) or 0)
        return merged[:limit] if limit > 0 else merged

    async def get_fact(self, key: str) -> Any | None:
        """The memory_get read. `key` may be a composite origin token, in which case it
        addresses one row of one NAMED store — subject to the current scope."""
        parsed = parse_origin_token(key)
        if parsed is not None:
            origin, fact_id = parsed
            store = await self._store_for_origin(origin)
            return await store.get_fact_by_id(fact_id) if store is not None else None
        if self.scope != SCOPE_BOTH:
            return await (await self._read_store()).get_fact(key)

        hit = await self._primary.get_fact(key)
        if hit is not None:
            return hit                      # a name in both stores resolves to THIS project's
        g = await self.ensure_global()
        return await g.get_fact(key) if g is not None else None

    async def get_fact_history(self, key: str) -> list[Any]:
        """The memory_get(history=True) read. In `both` mode: the primary's chain when it has
        one, otherwise the global's — NEVER spliced. A supersede chain is per-store, so a
        merged one would describe a history that never happened."""
        if self.scope != SCOPE_BOTH:
            return await (await self._read_store()).get_fact_history(key)

        chain = await self._primary.get_fact_history(key)
        if chain:
            return chain
        g = await self.ensure_global()
        return await g.get_fact_history(key) if g is not None else []

    # ------------------------------------------------------------------
    # The optional enrichment methods memory_search / memory_get getattr() for.
    # An absent one degrades SILENTLY (getattr -> None), so each must exist here or the
    # DEFAULT path quietly stops bumping staged counters and writing activation traces.
    # ------------------------------------------------------------------

    async def _enrichment_target(self) -> Any | None:
        """The ONE store an enrichment write may land on: the primary, and only the primary.

        Enrichment (staged read-counters, activation traces) is a WRITE, and the second handle
        is a store this session does not own — a read-only VIEW of another project's memory.
        Routing enrichment through `_read_store()` meant a `recall_scope: global` project
        session bumped counters and appended traces in the machine-global database (v0.13 B1),
        falsifying "no value of this knob makes a project session write global memory". None
        here means: this scope reads a store it may not write, so the enrichment is skipped.

        The honest cost, stated rather than hidden: in a scope:global session the global
        store's access counts and traces do not learn from those reads. Staleness in another
        store's ranking signal is the price of the boundary being structural.
        """
        return None if self.scope == ORIGIN_GLOBAL else self._primary

    async def touch_staged(self, keys: list[str]) -> None:
        # Keys are store-agnostic strings, so a key that lives only in the global store simply
        # matches no row on the primary.
        store = await self._enrichment_target()
        if store is not None:
            await store.touch_staged(keys)

    async def record_activation_trace(self, **kwargs: Any) -> None:
        # `both` mode is skipped deliberately: the hit list spans two databases and facts.id is
        # per-database, so there is no store this row could be written to without pointing at
        # another database's ids (Pitfall 7). `global` scope is skipped because the store that
        # was read is not this session's to write.
        if self.scope == SCOPE_BOTH:
            return
        store = await self._enrichment_target()
        if store is not None:
            await store.record_activation_trace(**kwargs)

    async def neighborhood(self, *args: Any, **kwargs: Any) -> list:
        # The tag graph is per-database; a merged hit list has no single graph to walk, so the
        # enrichment is off in `both` mode rather than walking the wrong store's edges.
        if self.scope == SCOPE_BOTH:
            return []
        return await (await self._read_store()).neighborhood(*args, **kwargs)

    async def get_facts_by_ids(self, ids: list[int]) -> list:
        if self.scope == SCOPE_BOTH:
            return []
        return await (await self._read_store()).get_facts_by_ids(ids)
