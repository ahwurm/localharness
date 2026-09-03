"""Scope-aware recall routing — the ONE read gate above scope-naive memory stores.

Implements ROADMAP amendment #6's ruled shape (v0.13 MEMS-02): a CALLER-LEVEL router that
answers every read according to `agent.memory.recall_scope`, while `MemoryStore` itself stays
single-store. No store learns about a second store; the router composes two of them.

READ-SIDE ONLY. There is no write verb on this object and no `__getattr__` passthrough — see
`RecallRouter`'s docstring for why that absence is the mechanism rather than a convention.

The second (machine-global) handle arrives CONSTRUCTED BUT NOT OPENED and is opened lazily, at
most once. `MemoryStore.open()` mkdirs, connects and MIGRATES — it WRITES — so a default-scope
session must never call it. That laziness is MEMS-03's precondition ("the global store is
byte-identical after a workspace session"), and it is why `ensure_global()` exists at all
instead of the handle simply being opened by whoever constructs the router.
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
        owner, one lifecycle, never two connections to one database file (Pitfall 6)."""
        if self._global_store is None:
            return None
        async with self._open_lock:
            if not self._global_opened:
                await self._global_store.open()
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
        """The ambient-injection read. Same signature as `MemoryStore.load_context`."""
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
        raise NotImplementedError("both-scope merge lands in 42-02 task 2")

    async def query_facts(self, query: Any) -> list[Any]:
        """The memory_search read."""
        if self.scope != SCOPE_BOTH:
            return await (await self._read_store()).query_facts(query)
        raise NotImplementedError("both-scope merge lands in 42-02 task 2")

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
        raise NotImplementedError("both-scope merge lands in 42-02 task 2")

    async def get_fact_history(self, key: str) -> list[Any]:
        """The memory_get(history=True) read."""
        if self.scope != SCOPE_BOTH:
            return await (await self._read_store()).get_fact_history(key)
        raise NotImplementedError("both-scope merge lands in 42-02 task 2")

    # ------------------------------------------------------------------
    # The optional enrichment methods memory_search / memory_get getattr() for.
    # An absent one degrades SILENTLY (getattr -> None), so each must exist here or the
    # DEFAULT path quietly stops bumping staged counters and writing activation traces.
    # ------------------------------------------------------------------

    async def touch_staged(self, keys: list[str]) -> None:
        # Reads bump staged counters on the store that was actually read. In `both` mode that
        # is the primary: keys are store-agnostic strings, and a key that lives only in the
        # global store simply matches no row here.
        await (await self._read_store()).touch_staged(keys)

    async def record_activation_trace(self, **kwargs: Any) -> None:
        # `both` mode is skipped deliberately: the hit list spans two databases and facts.id is
        # per-database, so there is no store this row could be written to without pointing at
        # another database's ids (Pitfall 7). Single-scope sessions trace exactly as before.
        if self.scope == SCOPE_BOTH:
            return
        await (await self._read_store()).record_activation_trace(**kwargs)

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
