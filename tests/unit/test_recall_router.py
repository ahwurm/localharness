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
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.memory.router import (
    ORIGIN_GLOBAL,
    ORIGIN_WORKSPACE,
    RECALL_SCOPES,
    SCOPE_BOTH,
    RecallRouter,
    format_origin_token,
    parse_origin_token,
)
from localharness.memory.router import _MERGED_HEADER, _MERGED_PREAMBLE
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


def make_store(base: Path, global_base: Path | None = None) -> MemoryStore:
    """The make_store idiom from tests/unit/test_memory_store.py:18.

    `global_base` is amendment #4's split: per-agent STATE may follow a workspace layer, but
    org/division SAFETY CONTEXT never does. Production gives BOTH stores the same one, which
    is why the two `load_context` calls return byte-identical division/guardrails text."""
    return MemoryStore(
        agent_id=AGENT,
        division_id="test-div",
        org_id="default",
        base_dir=str(base),
        global_base_dir=str(global_base) if global_base is not None else None,
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


# ================================================================== #
# TASK 2 — the `both` merge.
# ================================================================== #

GUARDRAILS_TEXT = "# Guardrails\nGUARDRAILS-MARKER: never exfiltrate.\n"
DIVISION_TEXT = "# Division\nDIVISION-MARKER: research division.\n"


async def _seed_sitting(store: MemoryStore, session_id: str, summary: str) -> None:
    """One CLOSED sitting — `summary IS NOT NULL` is what puts it on the shelf."""
    await store.create_session(session_id, {}, "test-model", 1000)
    await store.end_session(session_id, "complete", summary, 1, 1, 10, 10)


@pytest.fixture
async def pair(tmp_path: Path):
    """Two seeded stores that share ONE global_base_dir, handed over exactly as production
    will: the workspace store OPEN (the caller owns it), the global store CLOSED (the router
    opens it lazily).

    Seeded so the two id spaces COLLIDE — global rows 1..3 against workspace rows 1..3 — because
    an injected-ids assertion over disjoint id spaces passes for the wrong reason.
    """
    gl_dir, ws_dir = tmp_path / "global", tmp_path / "ws"
    (gl_dir / "orgs" / "default").mkdir(parents=True)
    (gl_dir / "orgs" / "default" / "GUARDRAILS.md").write_text(GUARDRAILS_TEXT, encoding="utf-8")
    (gl_dir / "divisions" / "test-div").mkdir(parents=True)
    (gl_dir / "divisions" / "test-div" / "DIVISION.md").write_text(DIVISION_TEXT, encoding="utf-8")

    g = make_store(gl_dir, global_base=gl_dir)
    await g.open()
    g_shared = await g.store_fact("shared-key", f"{GLOBAL_MARKER} global version")
    g_only = await g.store_fact("global-only", f"{GLOBAL_MARKER} solo body")
    await g.store_fact("versioned", f"{GLOBAL_MARKER} global v1")
    await _seed_sitting(g, "g-sitting", f"{GLOBAL_MARKER} sitting")
    await g.close()

    ws = make_store(ws_dir, global_base=gl_dir)
    await ws.open()
    ws_shared = await ws.store_fact("shared-key", f"{WS_MARKER} workspace version")
    await ws.store_fact("versioned", f"{WS_MARKER} v1")
    ws_versioned = await ws.store_fact("versioned", f"{WS_MARKER} v2")   # supersedes v1
    await _seed_sitting(ws, "ws-sitting", f"{WS_MARKER} sitting")

    ns = SimpleNamespace(
        ws=ws, g=g, ws_dir=ws_dir, gl_dir=gl_dir,
        ws_shared=ws_shared, ws_versioned=ws_versioned, g_shared=g_shared, g_only=g_only,
    )
    yield ns
    await ws.close()
    await g.close()


@pytest.fixture
def both(pair) -> RecallRouter:
    return RecallRouter(pair.ws, pair.g, scope=SCOPE_BOTH)


# ------------------------------------------------------------------ #
# 9. The merged index: both stores, workspace first, every line labelled.
# ------------------------------------------------------------------ #
async def test_both_merges_the_two_indexes_workspace_first(
    both: RecallRouter, pair
) -> None:
    assert both.scope == SCOPE_BOTH
    ctx = await both.load_context()
    md = ctx.agent_memory_md

    assert WS_MARKER in md and GLOBAL_MARKER in md
    # ORDER is a claim of its own: "both are present" passes with them the wrong way round.
    assert md.index(WS_MARKER) < md.index(GLOBAL_MARKER)
    assert _MERGED_HEADER in md
    assert md.index(WS_MARKER) < md.index(_MERGED_HEADER) < md.index(GLOBAL_MARKER)
    assert md.startswith(_MERGED_PREAMBLE)

    await both.close()


async def test_every_merged_fact_line_carries_an_origin_token(both: RecallRouter) -> None:
    """A bare `#12` would be ambiguous across two databases; an UNLABELLED line is worse —
    the model cannot tell which store it may address."""
    md = (await both.load_context()).agent_memory_md
    head, sep, _shelf = md.partition("### Recent Session History")
    assert sep, md   # the workspace shelf really is in there — see the next test

    bullets = [ln for ln in head.splitlines() if ln.startswith("- ")]
    assert len(bullets) == 5, bullets     # ws: shared-key + versioned; global: 3 facts
    unlabelled = [ln for ln in bullets if not re.match(r"^- \[(workspace|global)#\d+\] ", ln)]
    assert unlabelled == [], unlabelled

    await both.close()


async def test_the_global_block_repeats_neither_the_preamble_nor_the_shelf(
    both: RecallRouter,
) -> None:
    """`include_preamble=False` and `max_session_history=0` on the second render: the merged
    block owns ONE set of INDEX instructions, and the shelf is THIS project's working history —
    the global store's sittings are a different project's story."""
    md = (await both.load_context()).agent_memory_md

    assert md.count(PREAMBLE) == 1
    assert md.count("### Recent Session History") == 1
    assert f"{WS_MARKER} sitting" in md
    assert f"{GLOBAL_MARKER} sitting" not in md

    await both.close()


async def test_the_safety_context_appears_exactly_once(both: RecallRouter) -> None:
    """Pitfall 3: both stores derive division/guardrails from the SAME global_base_dir, so a
    merge that concatenated them would inject the org's safety voice twice."""
    ctx = await both.load_context()

    assert ctx.guardrails_md == GUARDRAILS_TEXT
    assert ctx.division_md == DIVISION_TEXT

    await both.close()


async def test_injected_ids_are_the_primary_s_ids_only(both: RecallRouter, pair) -> None:
    """Pitfall 7: the loop records the ambient trace on ITS OWN handle. A global id in this
    list writes a row pointing at another database's ids into a table with no FK enforcement —
    silent corruption."""
    ctx = await both.load_context()

    _ws_md, ws_rendered = await pair.ws._render_memory_index_with_ids(8)
    assert ctx.injected_fact_ids == ws_rendered

    g = await both.ensure_global()
    g_rendered = (await g._render_memory_index_with_ids(0))[1]
    # The id spaces COLLIDE — asserted, not assumed, or "no global id present" is vacuous.
    assert set(ws_rendered) & set(g_rendered)
    # ...so the discriminating claim is about the COUNT, not about membership.
    assert len(ctx.injected_fact_ids) == len(ws_rendered) < len(ws_rendered) + len(g_rendered)
    # The global facts ARE in the text — only their ids are withheld.
    assert GLOBAL_MARKER in ctx.agent_memory_md

    await both.close()


async def test_fact_count_sums_both_blocks(both: RecallRouter, pair) -> None:
    """Documented shape: the second term is the global store's INJECTED count, not a second
    COUNT query — the field has no production consumer and the hot path stays one query
    per store."""
    ctx = await both.load_context()
    ws_only = await RecallRouter(pair.ws, None).load_context()

    assert ctx.fact_count > ws_only.fact_count

    await both.close()


# ------------------------------------------------------------------ #
# 10. query_facts: scoped-first, key-level dedup.
# ------------------------------------------------------------------ #
async def test_both_query_is_scoped_first_and_dedups_by_key(both: RecallRouter) -> None:
    facts = await both.query_facts(FactQuery())
    keys = [f.key for f in facts]
    by_key = {f.key: f.value for f in facts}

    # A name in BOTH stores resolves to THIS project's version, exactly once.
    assert keys.count("shared-key") == 1
    assert WS_MARKER in by_key["shared-key"]
    assert GLOBAL_MARKER not in by_key["shared-key"]
    # Every workspace hit precedes every global hit.
    assert keys.index("global-only") > max(keys.index("shared-key"), keys.index("versioned"))
    # The global-only fact is reachable — dedup did not become "drop the global store".
    assert GLOBAL_MARKER in by_key["global-only"]

    await both.close()


async def test_both_query_honours_the_caller_s_limit(both: RecallRouter) -> None:
    """The merge happens BEFORE the cut, so a limit of 2 still yields workspace facts first."""
    facts = await both.query_facts(FactQuery(limit=2))
    assert len(facts) == 2
    assert all(WS_MARKER in f.value for f in facts)

    await both.close()


# ------------------------------------------------------------------ #
# 11. get_fact: the token addresses a named store — subject to scope.
# ------------------------------------------------------------------ #
async def test_a_global_token_resolves_under_both_scope(both: RecallRouter, pair) -> None:
    fact = await both.get_fact(format_origin_token(ORIGIN_GLOBAL, pair.g_only.id))
    assert fact is not None
    assert fact.key == "global-only"
    assert GLOBAL_MARKER in fact.value

    ws_fact = await both.get_fact(format_origin_token(ORIGIN_WORKSPACE, pair.ws_shared.id))
    assert ws_fact is not None and WS_MARKER in ws_fact.value

    await both.close()


async def test_a_global_token_is_refused_under_workspace_scope(pair) -> None:
    """MEMS-02 criterion 4, written down: the knob is honoured even by the ADDRESSING form.
    A `[global#7]` token pasted into a workspace-scope session must not reach across."""
    router = RecallRouter(pair.ws, pair.g, scope=ORIGIN_WORKSPACE)
    token = format_origin_token(ORIGIN_GLOBAL, pair.g_only.id)

    assert (await router.get_fact(token)) is None
    # The same id EXISTS in the workspace store, so this is a refusal, not a miss.
    assert (await pair.ws.get_fact_by_id(pair.g_only.id)) is not None
    # ...and the workspace token still works, so the token path is not simply broken.
    assert (await router.get_fact(format_origin_token(ORIGIN_WORKSPACE, pair.ws_shared.id)))

    await router.close()


async def test_a_workspace_token_is_refused_under_global_scope(pair) -> None:
    router = RecallRouter(pair.ws, pair.g, scope=ORIGIN_GLOBAL)
    assert (await router.get_fact(format_origin_token(ORIGIN_WORKSPACE, pair.ws_shared.id))) is None
    assert (await router.get_fact(format_origin_token(ORIGIN_GLOBAL, pair.g_only.id))) is not None
    await router.close()


async def test_both_get_fact_falls_through_to_the_global_store(both: RecallRouter) -> None:
    """A plain key: workspace first, then global."""
    shared = await both.get_fact("shared-key")
    assert shared is not None and WS_MARKER in shared.value

    solo = await both.get_fact("global-only")
    assert solo is not None and GLOBAL_MARKER in solo.value

    assert (await both.get_fact("no-such-key")) is None

    await both.close()


# ------------------------------------------------------------------ #
# 12. get_fact_history: one chain or the other, never spliced.
# ------------------------------------------------------------------ #
async def test_both_history_is_never_spliced(both: RecallRouter) -> None:
    """A supersede chain is per-store; a merged one would describe a history that never
    happened."""
    chain = await both.get_fact_history("versioned")

    assert [f.value for f in chain] == [f"{WS_MARKER} v2", f"{WS_MARKER} v1"]
    assert all(GLOBAL_MARKER not in f.value for f in chain)

    # No workspace chain -> the global one, whole.
    global_chain = await both.get_fact_history("global-only")
    assert len(global_chain) == 1
    assert GLOBAL_MARKER in global_chain[0].value

    assert await both.get_fact_history("no-such-key") == []

    await both.close()


# ------------------------------------------------------------------ #
# 13. The three cross-store-unsafe enrichments are documented no-ops.
# ------------------------------------------------------------------ #
async def test_both_writes_no_activation_trace_to_either_store(
    both: RecallRouter, pair
) -> None:
    """The hit list spans two databases and facts.id is per-database, so there is no store
    this row could be written to without pointing at another database's ids (Pitfall 7)."""
    await both.record_activation_trace(
        stimulus="anything", fired_ids=[1], injected_ids=[1], source="search"
    )

    g = await both.ensure_global()
    assert await pair.ws.recent_activation_traces() == []
    assert await g.recent_activation_traces() == []

    # The control: under a single scope the SAME call does write.
    single = RecallRouter(pair.ws, pair.g, scope=ORIGIN_WORKSPACE)
    await single.record_activation_trace(
        stimulus="anything", fired_ids=[1], injected_ids=[1], source="search"
    )
    assert len(await pair.ws.recent_activation_traces()) == 1

    await both.close()


async def test_both_disables_the_graph_enrichments(both: RecallRouter, pair) -> None:
    """The tag graph is per-database; a merged hit list has no single graph to walk."""
    assert await both.neighborhood(pair.ws_shared.id, depth=1, limit=6) == []
    assert await both.get_facts_by_ids([pair.ws_shared.id]) == []

    # The control: the same calls under a single scope return real answers.
    single = RecallRouter(pair.ws, pair.g, scope=ORIGIN_WORKSPACE)
    assert await single.neighborhood(pair.ws_shared.id, depth=1, limit=6) == [
        (pair.ws_shared.id, 0)
    ]
    assert [f.key for f in await single.get_facts_by_ids([pair.ws_shared.id])] == ["shared-key"]

    await both.close()


# ------------------------------------------------------------------ #
# 14. The legacy whole-MEMORY.md render also merges.
# ------------------------------------------------------------------ #
async def test_both_merges_the_legacy_memory_md_render(both: RecallRouter) -> None:
    """`index_mode=False` inlines each store's MEMORY.md — a dump with no per-fact lines to
    label, so the merged text carries the header and no tokens."""
    ctx = await both.load_context(index_mode=False)
    md = ctx.agent_memory_md

    assert WS_MARKER in md and GLOBAL_MARKER in md
    assert md.index(WS_MARKER) < md.index(_MERGED_HEADER) < md.index(GLOBAL_MARKER)
    assert re.search(r"\[(workspace|global)#\d+\]", md) is None, md
    assert ctx.injected_fact_ids == []

    await both.close()
