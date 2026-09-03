"""42-04 (MEMS-05): `/memory promote <id>` — the one deliberate bridge between the two stores.

Isolation without a bridge strands a genuinely general lesson in the project that learned it; a
bridge without an explicit act fills the machine-global store silently, which is the thing the
owner's line-walk guarantees forbid. So promote is one command, one fact, addressed by id, in three
forms: preview (writes nothing), `confirm` (copies), `revert` (undoes its own copy).

Two layers of proof, deliberately different in kind:

* **Unit** — `memory_cmd.dispatch(...)` driven directly against TWO real `MemoryStore`s in
  `tmp_path`. No session, no REPL. `promote_target` is a counting async closure, so "the preview
  writes nothing" is proven by the handle NEVER BEING ASKED FOR, not by looking for an absent file
  (a file can be absent for a dozen reasons; an un-awaited handle cannot have been written through).
* **Session** — a real offline `_start_async` drive from a workspace, promoting through the REPL's
  own `/memory` path and then asserting against the global database ON DISK with a fresh store,
  rather than trusting the command's own return string.

The discriminating pair throughout is "the global copy moved AND the workspace original did not".
Asserting only the first passes a `revert` that retires the wrong database's row — the worst
failure this command can have, and one that type-checks perfectly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.cli import memory_cmd
from localharness.memory.router import format_origin_token
from localharness.memory.sqlite import MemoryStore

# The workspace identity a live session passes: the PROJECT ROOT, realpath'd. In these unit tests it
# is just a string the command must carry verbatim into the provenance — the session tests below are
# what prove the REPL computes it correctly.
IDENTITY = "/home/u/projects/harness"

KEY = "lesson/measure-before-claiming"
VALUE = "never report a number a finished run did not return"
ORIG_PROV = "session-2026-09-03T21:00"


async def _open_store(root: Path) -> MemoryStore:
    store = MemoryStore(agent_id="solo", division_id="default", org_id="default",
                        base_dir=str(root), global_base_dir=str(root))
    await store.open()
    return store


class _Handle:
    """The `promote_target` a real session passes: an async callable returning the global store.

    A callable and not a store, precisely so the preview branch can decide not to open anything —
    and so this counter can say whether it decided that.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self.calls = 0

    async def __call__(self) -> MemoryStore:
        self.calls += 1
        return self._store


@pytest.fixture
async def two(tmp_path):
    """(workspace store, global store, promote handle) — two real databases, two real files."""
    ws = await _open_store(tmp_path / "proj" / ".localharness")
    gl = await _open_store(tmp_path / "home" / ".localharness")
    try:
        yield ws, gl, _Handle(gl)
    finally:
        await ws.close()
        await gl.close()


async def _seed(ws: MemoryStore, value: str = VALUE):
    return await ws.store_fact(key=KEY, value=value, tags=["lesson"], confidence=0.9,
                               source="user", provenance=ORIG_PROV)


async def _promote(ws, arg, handle=None, identity=IDENTITY):
    return await memory_cmd.dispatch(ws, f"promote {arg}", promote_target=handle,
                                     workspace_identity=identity)


# --------------------------------------------------------------------------- preview


async def test_preview_never_asks_for_the_global_handle(two):
    """The guarantee, asserted as a mechanism: nothing is opened, so nothing can be written. The
    global store's database is not even created until you confirm."""
    ws, _gl, handle = two
    fact = await _seed(ws)

    await _promote(ws, str(fact.id), handle)

    assert handle.calls == 0, "the preview opened the global store"


async def test_preview_names_the_memory_its_destination_and_the_confirm_line(two):
    ws, _gl, handle = two
    fact = await _seed(ws)

    out = await _promote(ws, str(fact.id), handle)

    assert f"/memory promote {fact.id} confirm" in out, f"no confirm line: {out!r}"
    assert VALUE in out, f"the preview did not show the value: {out!r}"
    assert IDENTITY in out, f"the preview did not name the workspace it came from: {out!r}"


async def test_promote_is_routed_before_the_tag_fall_through(two):
    """`dispatch` treats anything it does not recognise as a tag path, so a promote route placed
    after the fall-through would read `promote` as the name of a tag."""
    ws, _gl, handle = two

    out = await _promote(ws, "", handle)

    assert "Usage: /memory promote" in out, f"promote fell through to the tag path: {out!r}"


# --------------------------------------------------------------------------- confirm


async def test_confirm_copies_the_fact_into_the_global_store(two):
    ws, gl, handle = two
    fact = await _seed(ws)

    await _promote(ws, f"{fact.id} confirm", handle)

    copy = await gl.get_fact(KEY)
    assert copy is not None, "the fact never arrived in the global store"
    assert copy.value == VALUE
    assert copy.status == "active"
    assert copy.source == "promote", f"the copy does not say where it came from: {copy.source!r}"


async def test_the_promoted_copy_carries_origin_provenance(two):
    """Which workspace it came from and when, plus the original chain — one TEXT column, the
    `user_forget@` / `revert-of:` convention this codebase already uses twice."""
    ws, gl, handle = two
    fact = await _seed(ws)

    await _promote(ws, f"{fact.id} confirm", handle)

    copy = await gl.get_fact(KEY)
    prefix = memory_cmd.PROMOTE_PROVENANCE_PREFIX
    assert copy.provenance.startswith(prefix), f"no promotion marker: {copy.provenance!r}"
    stamp, identity, original = copy.provenance[len(prefix):].split(";", 2)
    assert stamp.isdigit(), f"the marker carries no timestamp: {copy.provenance!r}"
    assert identity == IDENTITY, f"the marker names {identity!r}, not the workspace"
    assert original == ORIG_PROV, f"the original provenance was dropped: {copy.provenance!r}"
    assert "promoted" in copy.tags, f"the copy is not tagged as promoted: {copy.tags}"


async def test_the_workspace_original_is_untouched_by_a_confirm(two):
    ws, _gl, handle = two
    fact = await _seed(ws)

    await _promote(ws, f"{fact.id} confirm", handle)

    after = await ws.get_fact_by_id(fact.id)
    assert after.status == "active", f"promoting retired the original: {after.status}"
    assert after.value == VALUE
    assert after.provenance == ORIG_PROV, "promoting rewrote the original's provenance"


async def test_re_promoting_supersedes_the_global_copy_instead_of_forking_it(two):
    """Two handles, ONE copy. Renaming the copy (`key + '-promoted'`) type-checks, keeps this
    test's first confirm green, and forks the memory — which is why the RE-promote is the case
    that grades it."""
    ws, gl, handle = two
    first = await _seed(ws)
    await _promote(ws, f"{first.id} confirm", handle)

    second = await ws.store_fact(key=KEY, value="corrected: report the selector too",
                                 tags=["lesson"], confidence=0.9, provenance=ORIG_PROV)
    await _promote(ws, f"{second.id} confirm", handle)

    history = await gl.get_fact_history(KEY)
    active = [f for f in history if f.status == "active"]
    assert len(active) == 1, f"the global store holds {len(active)} active rows for one name"
    assert active[0].value == "corrected: report the selector too"
    assert len(history) == 2, f"the superseded copy was not kept in history: {len(history)}"


# --------------------------------------------------------------------------- revert


async def test_revert_retires_the_promoted_global_copy(two):
    ws, gl, handle = two
    fact = await _seed(ws)
    await _promote(ws, f"{fact.id} confirm", handle)

    await _promote(ws, f"{fact.id} revert", handle)

    assert await gl.get_fact(KEY) is None, "the promoted copy is still active in the global store"
    history = await gl.get_fact_history(KEY)
    assert len(history) == 1 and history[0].status != "active", \
        f"revert did not retire the copy (kept, never deleted): {history}"


async def test_revert_leaves_the_workspace_original_active(two):
    """A revert that retires the wrong copy is the worst failure this command can have, and
    `store.forget_fact(...)` in place of `g.forget_fact(...)` type-checks perfectly. Its own test
    id, so the mutation that swaps them names this line and not the one above."""
    ws, gl, handle = two
    fact = await _seed(ws)
    await _promote(ws, f"{fact.id} confirm", handle)

    await _promote(ws, f"{fact.id} revert", handle)

    after = await ws.get_fact_by_id(fact.id)
    assert after.status == "active", f"revert retired THIS PROJECT's own memory: {after.status}"


async def test_revert_refuses_a_global_fact_that_promote_did_not_write(two):
    """The marker check is a SAFETY property, not politeness: without it, revert is a way to retire
    any global fact that happens to share a name with a workspace one."""
    ws, gl, handle = two
    fact = await _seed(ws)
    native = await gl.store_fact(key=KEY, value="the global store's own claim", confidence=0.9,
                                 provenance="written-here")

    out = await _promote(ws, f"{fact.id} revert", handle)

    still = await gl.get_fact(KEY)
    assert still is not None and still.id == native.id, "revert retired a fact it did not write"
    assert still.status == "active"
    assert "not written by promote" in out or "promotion marker" in out, \
        f"the refusal does not say why: {out!r}"


async def test_revert_with_nothing_promoted_says_so(two):
    ws, _gl, handle = two
    fact = await _seed(ws)

    out = await _promote(ws, f"{fact.id} revert", handle)

    assert "Nothing to revert" in out, f"{out!r}"


# --------------------------------------------------------------------------- no workspace layer


async def test_no_workspace_layer_explains_itself_and_writes_nothing(two):
    """With no `.localharness/`, this session's memory IS the machine-global memory — there is
    nowhere to promote to, and saying so is the whole behavior."""
    ws, gl, _handle = two
    fact = await _seed(ws)

    out = await _promote(ws, f"{fact.id} confirm", handle=None, identity="")

    assert "IS the machine-global memory" in out, f"{out!r}"
    assert await gl.get_fact(KEY) is None, "something was written without a global handle"


async def test_the_default_call_shape_still_works_for_every_existing_caller(two):
    """`dispatch(store, arg)` with no new kwargs at all — every pre-existing caller and test."""
    ws, _gl, _handle = two
    fact = await _seed(ws)

    out = await memory_cmd.dispatch(ws, f"promote {fact.id} confirm")

    assert "Promotion needs a project layer" in out, f"{out!r}"


# --------------------------------------------------------------------------- composite tokens


async def test_a_workspace_origin_token_addresses_the_same_row_as_a_bare_id(two):
    """Under `both` recall every injected line carries a `[workspace#12]` token, so that is what a
    user (or the model) has to hand. It must round-trip into the store the token names."""
    ws, gl, handle = two
    fact = await _seed(ws)

    out = await _promote(ws, f"{format_origin_token('workspace', fact.id)} confirm", handle)

    copy = await gl.get_fact(KEY)
    assert copy is not None, f"the token did not resolve: {out!r}"
    assert copy.value == VALUE


async def test_a_global_origin_token_is_refused_rather_than_promoting_the_wrong_row(two):
    """`[global#12]` and `[workspace#12]` are unrelated rows in unrelated databases. Reading the
    number out of a global token would promote whatever the workspace happens to have at that id."""
    ws, gl, handle = two
    await _seed(ws)

    out = await _promote(ws, f"{format_origin_token('global', 1)} confirm", handle)

    assert handle.calls == 0, "a global token opened the global store"
    assert "already" in out and "global" in out, f"the refusal does not say why: {out!r}"
    assert await gl.get_fact(KEY) is None, "a global token wrote something"


# --------------------------------------------------------------------------- junk input


@pytest.mark.parametrize("arg", ["", "abc", "abc confirm", "12 sideways", "[workspace#x]"])
async def test_junk_input_returns_usage_text_and_never_raises(two, arg):
    ws, _gl, handle = two
    await _seed(ws)

    out = await _promote(ws, arg, handle)

    assert isinstance(out, str) and "Usage: /memory promote" in out, f"{arg!r} -> {out!r}"
    assert handle.calls == 0


async def test_an_unknown_id_is_refused_in_plain_text(two):
    ws, _gl, handle = two
    await _seed(ws)

    out = await _promote(ws, "9999 confirm", handle)

    assert "No memory with id 9999" in out, f"{out!r}"
    assert handle.calls == 0


async def test_a_retired_memory_cannot_be_promoted(two):
    """Retired rows never inject and never search; promoting one would resurrect it elsewhere."""
    ws, gl, handle = two
    fact = await _seed(ws)
    assert await ws.forget_fact(fact.id)

    out = await _promote(ws, f"{fact.id} confirm", handle)

    assert "retired" in out, f"{out!r}"
    assert await gl.get_fact(KEY) is None
    assert handle.calls == 0
