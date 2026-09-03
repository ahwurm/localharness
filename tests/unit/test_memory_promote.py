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

# Phase 41's drive harness, imported rather than copied (42-03's precedent) — those files stay
# byte-untouched, and a drift between "how phase 41 drives a workspace session" and "how the
# promote tests do" would make the two phases' claims incomparable.
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _drive,
    _global_only_start,
    _workspace_start,
)

# The workspace identity a live session passes: the PROJECT ROOT, realpath'd. In the unit tests it
# is just a string the command must carry verbatim into the provenance — the session tests at the
# bottom are what prove the REPL computes it correctly.
IDENTITY = "/home/u/projects/harness"

KEY = "lesson/measure-before-claiming"
VALUE = "never report a number a finished run did not return"
ORIG_PROV = "session-2026-09-03T21:00"
DECOY = "GLOBAL-DECOY never trust an unmeasured claim"


async def _open_store(root: Path) -> MemoryStore:
    """A store on the same agent the drive harness runs as — so "the global database on disk" and
    "the database the session promoted into" are the same file by construction, not by luck."""
    store = MemoryStore(agent_id=AGENT, division_id="default", org_id="default",
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


# =========================================================================== the live session
#
# A real offline `_start_async` drive from inside a project. The interactive loop is replaced by
# one that writes a fact into the session's OWN store and then promotes it through the REPL's own
# `/memory` handler — so what is graded is the threading (the router's handle and the workspace
# identity reaching `dispatch`), not the renderer, which the unit tests above already own.


def _in_session(monkeypatch, seen: dict, *commands: str) -> None:
    """Seed one fact, run `/memory` commands through the REPL's real handler, capture the output.

    `self._send_info` is replaced on the INSTANCE so the text the user would have seen is captured
    without a channel double — `_handle_memory_cmd` swallows every exception, so a silent
    `/memory failed: ...` would otherwise look exactly like a promote that never ran.
    """
    async def _run(self):
        fact = await self._store.store_fact(key=KEY, value=VALUE, tags=["lesson"],
                                            confidence=0.9, provenance=ORIG_PROV)
        seen["fact"] = fact
        out: list[str] = []

        async def _capture(text, colorize=False):
            out.append(text)

        self._send_info = _capture
        for cmd in commands:
            await self._handle_memory_cmd(cmd.format(id=fact.id))
        seen["out"] = "\n".join(out)
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _run)


async def _seed_global_decoy(global_dir: Path) -> None:
    """A fact that exists ONLY in the machine-global store, before the drive."""
    gl = await _open_store(global_dir)
    try:
        await gl.store_fact(key="global/own-claim", value=DECOY, confidence=0.9,
                            provenance="written-here")
    finally:
        await gl.close()


async def _global_copy(global_dir: Path, key: str = KEY):
    """Read the global store back from DISK with a fresh handle — never trust the command's own
    return string for a claim about what was written."""
    gl = await _open_store(global_dir)
    try:
        return await gl.get_fact(key), await gl.get_fact_history(key)
    finally:
        await gl.close()


async def test_a_live_workspace_session_promotes_into_the_global_store(tmp_path, monkeypatch):
    """The end-to-end claim: `/memory promote <id> confirm` typed in a real workspace session puts
    the fact in `<global>/agents/<AGENT>/memory.db`."""
    _home, global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    seen: dict = {}
    _in_session(monkeypatch, seen, "promote {id} confirm")

    await _drive()

    assert seen.get("fact") is not None, "the stubbed loop never fired"
    assert "/memory failed" not in seen["out"], seen["out"]
    copy, _history = await _global_copy(global_dir)
    assert copy is not None, f"nothing reached the global store; the session said: {seen['out']!r}"
    assert copy.value == VALUE
    assert copy.source == "promote"


async def test_the_promoted_provenance_names_the_project_root(tmp_path, monkeypatch):
    """Which workspace it came from, in the ONE identity format this milestone uses: the project
    root, realpath'd — the same string `permissions.workspace_root` carries and the trust store
    keys on. NOT the `.localharness` dir, and not a workspace-relative name."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    seen: dict = {}
    _in_session(monkeypatch, seen, "promote {id} confirm")

    await _drive()

    copy, _history = await _global_copy(global_dir)
    assert copy is not None, f"nothing reached the global store: {seen.get('out')!r}"
    prefix = memory_cmd.PROMOTE_PROVENANCE_PREFIX
    _stamp, identity, original = copy.provenance[len(prefix):].split(";", 2)
    assert identity == str(ws.resolve().parent), \
        f"the copy says it came from {identity!r}, not the project root"
    assert identity != str(ws.resolve()), "the identity is the .localharness dir, not the project"
    assert original == ORIG_PROV, "the original provenance chain was dropped"


async def test_the_memory_window_still_browses_only_this_projects_store(tmp_path, monkeypatch):
    """The ruled v1 boundary, asserted so that widening it is a deliberate act: promote is the ONE
    verb that reaches across. `search` (and the rest of the browsing family) stays primary-only."""
    _home, global_dir, _ws = _workspace_start(tmp_path, monkeypatch)
    await _seed_global_decoy(global_dir)
    seen: dict = {}
    _in_session(monkeypatch, seen, "search never")

    await _drive()

    assert "GLOBAL-DECOY" not in seen["out"], \
        f"/memory search read the machine-global store: {seen['out']!r}"
    assert "finished run" in seen["out"], \
        f"/memory search did not even read this project's own store: {seen['out']!r}"


async def test_a_session_without_a_workspace_says_why_it_cannot_promote(tmp_path, monkeypatch):
    """With no `.localharness/` the session's memory IS the machine-global memory. The failure mode
    this guards is a promote that "succeeds" by writing a duplicate row into the same database."""
    _home, global_dir, _proj = _global_only_start(tmp_path, monkeypatch)
    seen: dict = {}
    _in_session(monkeypatch, seen, "promote {id} confirm")

    await _drive()

    assert "Promotion needs a project layer" in seen["out"], seen["out"]
    fact, history = await _global_copy(global_dir)
    assert fact is not None and fact.source != "promote", \
        f"a session with no workspace promoted into its own store: {fact}"
    assert len(history) == 1, f"promote forked the one store it had: {history}"
