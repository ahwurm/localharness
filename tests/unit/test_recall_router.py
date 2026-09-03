"""The recall router — the ONE read gate for scope-aware recall (Phase 42, plan 02, MEMS-02).

`RecallRouter` holds this session's own `MemoryStore` and, when a workspace layer applies, a
SECOND handle on the machine-global store. `recall_scope` decides which one(s) a read sees.
Ambient injection and the on-demand memory tools will both (42-03) read through this ONE
object, so "tools cannot bypass the knob" is a property of the SHAPE, not of per-tool checks.

Every scope assertion here is made in BOTH directions. The two stores are seeded with unique
markers (`WS-ONLY-MARKER`, `GLOBAL-ONLY-MARKER`) so "the workspace fact is present" is always
paired with "the global fact is absent" — a test that only checks the first cannot fail.

This plan ships NO caller: the router is graded before it is wired (39-01's discipline).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localharness.memory.router import (
    ORIGIN_GLOBAL,
    ORIGIN_WORKSPACE,
    RECALL_SCOPES,
    RecallRouter,
    format_origin_token,
    parse_origin_token,
)
from localharness.memory.sqlite import FactQuery, MemoryStore

AGENT = "test-agent"
WS_MARKER = "WS-ONLY-MARKER"
GLOBAL_MARKER = "GLOBAL-ONLY-MARKER"

# Copied character-for-character from sqlite.py's render (the 42-01 convention): if the
# source literal is reflowed, this constant is what turns that into a failure.
PREAMBLE = (
    "This is an INDEX, not the full memory. Each line below is one persistent fact "
    "(name: short description). Call `memory_get(name)` for a fact's full body, or "
    "`memory_search(query)` to search fact contents.\n\n"
)


def make_store(base: Path) -> MemoryStore:
    """The make_store idiom from tests/unit/test_memory_store.py:18."""
    return MemoryStore(
        agent_id=AGENT,
        division_id="test-div",
        org_id="default",
        base_dir=str(base),
    )


def db_path(base: Path) -> Path:
    return base / "agents" / AGENT / "memory.db"


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    return tmp_path / "ws"


@pytest.fixture
def gl_dir(tmp_path: Path) -> Path:
    return tmp_path / "global"


@pytest.fixture
async def primary(ws_dir: Path) -> MemoryStore:
    """The session's own store: OPEN. The router never opens it and never closes it."""
    s = make_store(ws_dir)
    await s.open()
    await s.store_fact("ws-fact", f"{WS_MARKER} body")
    yield s
    await s.close()


@pytest.fixture
async def global_store(gl_dir: Path) -> MemoryStore:
    """Seeded, then CLOSED — handed to the router exactly as production will: CONSTRUCTED,
    not opened. The router owns opening it (once) and closing it."""
    s = make_store(gl_dir)
    await s.open()
    await s.store_fact("global-fact", f"{GLOBAL_MARKER} body")
    await s.close()
    yield s
    await s.close()


@pytest.fixture
def virgin_global(tmp_path: Path) -> MemoryStore:
    """Constructed and NEVER opened — nothing exists under its base dir at all.
    Constructing a MemoryStore does zero disk I/O (sqlite.py:714-755 builds Paths and two
    writer objects); `open()` is what mkdirs, connects and MIGRATES."""
    return make_store(tmp_path / "virgin")


# ------------------------------------------------------------------ #
# 1. With no second handle the knob collapses (LAYR-03).
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("configured", RECALL_SCOPES)
async def test_no_second_store_collapses_every_scope(primary: MemoryStore, configured: str) -> None:
    """All three configured values mean the same thing when there is exactly one store to
    read. The collapse lives HERE, in one place, not as a defence repeated per call site."""
    router = RecallRouter(primary, None, scope=configured)

    assert router.scope == ORIGIN_WORKSPACE
    # ...and the configured value is still reportable, for doctor/diagnostics.
    assert router.configured_scope == configured


async def test_an_unknown_scope_value_falls_back_to_workspace(primary: MemoryStore) -> None:
    """Defence in depth behind the Literal: a value config validation never produced must
    not silently mean `both`."""
    router = RecallRouter(primary, None, scope="nonsense")
    assert router.scope == ORIGIN_WORKSPACE
    assert router.configured_scope == ORIGIN_WORKSPACE


async def test_the_knob_is_behaviourally_inert_without_a_second_store(
    primary: MemoryStore,
) -> None:
    """Every read gives the same answer under all three values — this is what "inert" means
    to a user, as opposed to what the property says."""
    seen = []
    for configured in RECALL_SCOPES:
        router = RecallRouter(primary, None, scope=configured)
        ctx = await router.load_context()
        seen.append(ctx.agent_memory_md)
        assert (await router.get_fact("ws-fact")) is not None
        assert [f.key for f in await router.query_facts(FactQuery())] == ["ws-fact"]
    assert seen[0] == seen[1] == seen[2]
    assert WS_MARKER in seen[0]


# ------------------------------------------------------------------ #
# 2. workspace scope reads the PRIMARY only — both directions.
# ------------------------------------------------------------------ #
async def test_workspace_scope_reads_only_the_workspace_store(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    router = RecallRouter(primary, global_store, scope=ORIGIN_WORKSPACE)
    assert router.scope == ORIGIN_WORKSPACE

    ctx = await router.load_context()
    assert WS_MARKER in ctx.agent_memory_md
    assert GLOBAL_MARKER not in ctx.agent_memory_md

    keys = [f.key for f in await router.query_facts(FactQuery())]
    assert keys == ["ws-fact"]

    assert (await router.get_fact("ws-fact")) is not None
    assert (await router.get_fact("global-fact")) is None

    assert [f.key for f in await router.get_fact_history("ws-fact")] == ["ws-fact"]
    assert await router.get_fact_history("global-fact") == []


# ------------------------------------------------------------------ #
# 3. global scope reads the GLOBAL store only — the mirror image.
# ------------------------------------------------------------------ #
async def test_global_scope_reads_only_the_global_store(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)
    assert router.scope == ORIGIN_GLOBAL

    ctx = await router.load_context()
    assert GLOBAL_MARKER in ctx.agent_memory_md
    assert WS_MARKER not in ctx.agent_memory_md

    keys = [f.key for f in await router.query_facts(FactQuery())]
    assert keys == ["global-fact"]

    assert (await router.get_fact("global-fact")) is not None
    assert (await router.get_fact("ws-fact")) is None

    assert [f.key for f in await router.get_fact_history("global-fact")] == ["global-fact"]
    assert await router.get_fact_history("ws-fact") == []

    await router.close()


async def test_global_scope_reports_no_injected_ids(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    """Pitfall 7: the loop records the ambient trace on ITS OWN store handle and `facts.id`
    is per-database, so a global-scope read must report an EMPTY injected set rather than
    ids belonging to another database. The text is still there — only the ids are dropped."""
    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)
    ctx = await router.load_context()

    assert ctx.injected_fact_ids == []
    assert GLOBAL_MARKER in ctx.agent_memory_md  # the read really happened

    # ...and the primary path is NOT stripped: it owns its ids.
    ws_ctx = await RecallRouter(primary, global_store, scope=ORIGIN_WORKSPACE).load_context()
    assert ws_ctx.injected_fact_ids != []

    await router.close()


# ------------------------------------------------------------------ #
# 4. Laziness — MEMS-03's precondition, asserted on the FILESYSTEM.
# ------------------------------------------------------------------ #
async def test_default_scope_never_creates_the_global_database(
    primary: MemoryStore, virgin_global: MemoryStore, tmp_path: Path
) -> None:
    """`MemoryStore.open()` mkdirs, connects and MIGRATES — it WRITES. A default-scope
    session must therefore never call it: that is exactly MEMS-03's "the global store is
    byte-identical afterwards"."""
    router = RecallRouter(primary, virgin_global, scope=ORIGIN_WORKSPACE)

    for _ in range(3):
        ctx = await router.load_context()
        await router.query_facts(FactQuery())
        await router.get_fact("ws-fact")
        await router.get_fact_history("ws-fact")
        await router.touch_staged(["ws-fact"])

    virgin = tmp_path / "virgin"
    assert not db_path(virgin).exists()
    # No -wal / -shm either: the whole tree is untouched, not merely the .db name.
    assert not virgin.exists(), sorted(p.name for p in virgin.rglob("*"))
    # The reads DID happen — otherwise "no file" passes for the wrong reason.
    assert WS_MARKER in ctx.agent_memory_md


async def test_the_global_handle_opens_exactly_once(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    """Open-once under concurrency: `ensure_global` holds a lock, so five reads racing on
    one event loop still produce ONE `open()`. Two opens would mean two aiosqlite worker
    threads on one database file (Pitfall 6)."""
    calls: list[int] = []
    real_open = global_store.open

    async def counting_open() -> None:
        calls.append(1)
        await real_open()

    global_store.open = counting_open  # type: ignore[method-assign]

    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)
    await asyncio.gather(
        router.load_context(),
        router.query_facts(FactQuery()),
        router.get_fact("global-fact"),
        router.get_fact_history("global-fact"),
        router.load_context(),
    )

    assert calls == [1]
    await router.close()


# ------------------------------------------------------------------ #
# 5. close() closes ONLY what the router opened.
# ------------------------------------------------------------------ #
async def test_close_closes_the_global_handle_and_never_the_primary(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)
    await router.load_context()
    assert global_store._db is not None

    await router.close()
    assert global_store._db is None
    await router.close()  # twice is safe

    # The primary belongs to the CALLER and is closed by the caller's own shutdown ordering.
    assert primary._db is not None
    assert (await primary.get_fact("ws-fact")) is not None


async def test_closing_a_router_that_never_opened_one_is_a_noop(
    primary: MemoryStore, virgin_global: MemoryStore, tmp_path: Path
) -> None:
    router = RecallRouter(primary, virgin_global, scope=ORIGIN_WORKSPACE)
    await router.close()
    assert not (tmp_path / "virgin").exists()
    assert primary._db is not None

    router_without_a_second_store = RecallRouter(primary, None, scope=ORIGIN_WORKSPACE)
    await router_without_a_second_store.close()
    assert primary._db is not None


async def test_ensure_global_returns_none_without_a_second_store(primary: MemoryStore) -> None:
    router = RecallRouter(primary, None, scope="both")
    assert (await router.ensure_global()) is None


# ------------------------------------------------------------------ #
# 6. No write verbs — the mechanism, not a convention.
# ------------------------------------------------------------------ #
WRITE_VERBS = (
    "store_fact",
    "set_current_session",
    "forget_fact",
    "end_session",
    "flush_memory_md",
    "create_session",
)


@pytest.mark.parametrize("verb", WRITE_VERBS)
async def test_the_router_has_no_write_verbs(
    primary: MemoryStore, global_store: MemoryStore, verb: str
) -> None:
    """`recall_scope: global` changes what a session READS and never where it WRITES
    (Pitfall 2). A stray write call raises AttributeError loudly instead of landing
    silently in the wrong database — which also rules out a `__getattr__` passthrough."""
    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)

    assert not hasattr(RecallRouter, verb)
    assert not hasattr(router, verb)
    with pytest.raises(AttributeError):
        getattr(router, verb)

    # The verb really does exist on the store — otherwise this passes on a typo.
    assert hasattr(primary, verb)


async def test_no_getattr_passthrough_at_all(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    router = RecallRouter(primary, global_store, scope=ORIGIN_GLOBAL)
    assert not hasattr(RecallRouter, "__getattr__")
    with pytest.raises(AttributeError):
        getattr(router, "fold_staged_access")  # a real MemoryStore method, absent here
    assert hasattr(primary, "fold_staged_access")


# ------------------------------------------------------------------ #
# 7. The optional enrichment methods the tools getattr() for MUST exist.
#    An absent one degrades SILENTLY (getattr -> None) — a behavior
#    regression in the DEFAULT path, not a compile error.
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "verb", ["touch_staged", "record_activation_trace", "neighborhood", "get_facts_by_ids"]
)
async def test_optional_enrichment_methods_are_present(
    primary: MemoryStore, global_store: MemoryStore, verb: str
) -> None:
    router = RecallRouter(primary, global_store, scope=ORIGIN_WORKSPACE)
    assert callable(getattr(router, verb, None)), verb


async def test_single_scope_enrichment_hits_the_store_that_was_read(
    primary: MemoryStore, global_store: MemoryStore
) -> None:
    """memory_search bumps staged read-counters and writes an activation trace through
    these. Under a single scope they must behave exactly as they do today."""
    router = RecallRouter(primary, global_store, scope=ORIGIN_WORKSPACE)
    fact = await primary.get_fact("ws-fact")
    assert fact is not None

    await router.touch_staged(["ws-fact"])
    await router.record_activation_trace(
        stimulus="what do I know", fired_ids=[fact.id], injected_ids=[fact.id], source="search"
    )
    assert [t.source for t in await primary.recent_activation_traces()] == ["search"]
    assert [f.key for f in await router.get_facts_by_ids([fact.id])] == ["ws-fact"]
    assert await router.neighborhood(fact.id, depth=1, limit=6) == [(fact.id, 0)]


# ------------------------------------------------------------------ #
# 8. The composite origin token.
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "token,expected",
    [
        ("[global#7]", (ORIGIN_GLOBAL, 7)),
        ("[workspace#12]", (ORIGIN_WORKSPACE, 12)),
        ("  [workspace#12] ", (ORIGIN_WORKSPACE, 12)),  # a model echoing with stray space
        ("plain-key", None),
        ("", None),
        ("[global#7", None),
        ("[nonsense#7]", None),
        ("[global#seven]", None),
        ("prefix [global#7]", None),  # anchored: a KEY containing brackets is not a token
        ("[global#7] suffix", None),
    ],
)
def test_parse_origin_token(token: str, expected: tuple[str, int] | None) -> None:
    assert parse_origin_token(token) == expected


def test_format_origin_token_round_trips() -> None:
    assert format_origin_token(ORIGIN_WORKSPACE, 12) == "[workspace#12]"
    assert format_origin_token(ORIGIN_GLOBAL, 7) == "[global#7]"
    assert parse_origin_token(format_origin_token(ORIGIN_GLOBAL, 7)) == (ORIGIN_GLOBAL, 7)
