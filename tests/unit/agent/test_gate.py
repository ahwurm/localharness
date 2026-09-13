"""The effectful gate: asking, remembering, failing closed (PRD §3.5, §3.6, §3.3).

The pure verdict is covered by `test_verdict.py`; these tests are about what happens AROUND
it — whether a human is awaited, whether the answer becomes durable, whether a channel that
cannot ask denies loudly, and whether the two bus events carry what the ask-rate report needs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from localharness.agent.gate import (
    ASK_TIMEOUT_TOOL_MULTIPLE,
    GATE_ERROR_REASON,
    MODE_SET_FROM_CHANNEL_LOG,
    NO_ASKER_REASON,
    PermissionGate,
    deny_fn_from,
    tool_meta_from_schema,
)
from localharness.agent.gate_types import (
    UNGRANTABLE_OBSERVATION_SUFFIX,
    Decision,
    PermissionRequest,
    ToolMeta,
)
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.grants import GrantStore, new_refusal
from localharness.config.models import PermissionConfig
from localharness.core.bus import EventBus
from localharness.core.events import (
    CANCELLED_RESOLUTION,
    PermissionAsked,
    PermissionResolved,
    PermissionStaged,
)
from localharness.tools import ToolSchema

SHELL = ToolMeta(group="shell")
WRITE = ToolMeta(group="fs.write")


def _gate(tmp_path: Path, **kw) -> PermissionGate:
    """A gate for the GUARDED suite below — the asking, remembering, event-publishing half.

    ``mode`` is pinned because v0.14.1 moved the default to ``auto`` (owner ruling 2026-09-11),
    which asks about almost nothing and writes no grants; the effectful machinery these tests
    exercise only runs when something asks. The ``auto`` semantics are covered by
    ``tests/unit/agent/test_auto_mode.py``.
    """
    workspace = kw.pop("workspace", tmp_path / "project")
    workspace.mkdir(exist_ok=True)
    return PermissionGate(
        boundary=kw.pop("boundary", workspace),
        workspace=workspace,
        grants=kw.pop("grants", GrantStore(tmp_path / "grants.yaml")),
        channel_name=kw.pop("channel_name", "test"),
        mode=kw.pop("mode", "guarded"),
        **kw,
    )


def _answer(kind: str, seen: list | None = None):
    async def _asker(request: PermissionRequest) -> Decision:
        if seen is not None:
            seen.append(request)
        return Decision(kind=kind)

    return _asker


async def _check(gate: PermissionGate, tool: str, params: dict, meta: ToolMeta = SHELL, **kw):
    return await gate.check(tool, params, meta, agent_id="a", session_id="s", **kw)


# --------------------------------------------------------------- ask and remember

@pytest.mark.asyncio
async def test_ask_then_allow_always_is_remembered(tmp_path):
    """The whole point of the spine: answer once, never asked again (PRD §3.3)."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_always", seen))
    params = {"command": "cargo build --release"}

    first = await _check(gate, "bash_exec", params)
    assert first.allowed and len(seen) == 1
    assert seen[0].klass == "shell-unfamiliar"

    second = await _check(gate, "bash_exec", params)
    assert second.allowed
    assert len(seen) == 1, "a remembered answer must not ask again"


@pytest.mark.asyncio
async def test_allow_once_does_not_write_a_grant(tmp_path):
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_once", seen))
    await _check(gate, "bash_exec", {"command": "cargo build"})
    await _check(gate, "bash_exec", {"command": "cargo build"})
    assert len(seen) == 2


@pytest.mark.asyncio
async def test_ungrantable_allow_always_never_writes_a_grant(tmp_path):
    """A destructive call asks EVERY time, even if the channel hands back allow_always."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_always", seen))
    for _ in range(2):
        outcome = await _check(gate, "bash_exec", {"command": "rm -rf build"})
        assert outcome.allowed
    assert len(seen) == 2
    assert seen[0].grantable is False
    assert gate.grants.lookup(gate.workspace, "shell-destructive", "rm -rf") is None


@pytest.mark.asyncio
async def test_reject_always_writes_a_refusal_that_then_denies_without_asking(tmp_path):
    """"Never here" wins forever after, as a negative grant on the key (PRD §3.3)."""
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    first = await _check(gate, "bash_exec", {"command": "cargo publish"})
    assert not first.allowed

    refusal = gate.grants.refused(gate.workspace, "shell-unfamiliar", "cargo publish")
    assert refusal is not None
    assert (refusal.klass, refusal.channel, refusal.session_id) == ("shell-unfamiliar", "test", "s")

    asked: list[PermissionRequest] = []
    gate.asker = _answer("allow_always", asked)  # even a yes cannot undo it
    second = await _check(gate, "bash_exec", {"command": "cargo publish"})
    assert not second.allowed
    assert "never here" in second.reason
    assert asked == []  # a refusal asks no more


@pytest.mark.asyncio
async def test_a_refusal_denies_only_the_signature_it_names(tmp_path):
    """The defect: a "never" on ``cp`` used to become ``bash_exec(*cp*)`` and ban ``scp`` too."""
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    assert not (await _check(gate, "bash_exec", {"command": "cp a b"})).allowed

    gate.asker = _answer("allow_once")
    assert (await _check(gate, "bash_exec", {"command": "scp a host:b"})).allowed
    assert (await _check(gate, "bash_exec", {"command": "cpio -o < list"})).allowed
    assert (await _check(gate, "bash_exec", {"command": "ls /srv/backup-cp"})).allowed


@pytest.mark.asyncio
async def test_reject_always_refuses_every_key_the_call_asked_about(tmp_path):
    """One prompt covers several keys, so one "never" must refuse all of them (PRD §3.3)."""
    outside = tmp_path / "outside"
    outside.mkdir()
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    command = f"cargo publish && touch {outside}/f"
    assert not (await _check(gate, "bash_exec", {"command": command})).allowed

    assert gate.grants.refused(gate.workspace, "shell-unfamiliar", "cargo publish") is not None
    assert gate.grants.refused(gate.workspace, "edit-outside", str(outside)) is not None


@pytest.mark.asyncio
async def test_a_directory_refusal_covers_the_subtree(tmp_path):
    """Directory refusals reach down exactly as directory grants do (PRD §3.1 edit-outside)."""
    outside = tmp_path / "outside"
    (outside / "y").mkdir(parents=True)
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    assert not (await _check(gate, "write", {"path": str(outside / "f")}, WRITE)).allowed

    asked: list[PermissionRequest] = []
    gate.asker = _answer("allow_always", asked)
    deeper = await _check(gate, "write", {"path": str(outside / "y" / "f")}, WRITE)
    assert not deeper.allowed
    assert "never here" in deeper.reason
    assert asked == []


@pytest.mark.asyncio
async def test_a_refusal_beats_a_grant_already_written_for_the_same_key(tmp_path):
    """A "never" is a tightening, so it wins over any grant on the key, in either order."""
    gate = _gate(tmp_path, asker=_answer("allow_always"))
    assert (await _check(gate, "bash_exec", {"command": "cargo publish"})).allowed
    assert gate.grants.lookup(gate.workspace, "shell-unfamiliar", "cargo publish") is not None

    gate.grants.add_refusal(
        new_refusal(key="cargo publish", klass="shell-unfamiliar", workspace=gate.workspace,
                    channel="test", session_id="s"))
    outcome = await _check(gate, "bash_exec", {"command": "cargo publish"})
    assert not outcome.allowed
    assert "never here" in outcome.reason


@pytest.mark.asyncio
async def test_an_ungrantable_reject_always_refuses_its_primary_key(tmp_path):
    """A destructive call asks every time, but a "never" on it still sticks (PRD §3.3)."""
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    assert not (await _check(gate, "bash_exec", {"command": "rm -rf build"})).allowed
    assert gate.grants.refused(gate.workspace, "shell-destructive", "rm -rf") is not None

    asked: list[PermissionRequest] = []
    gate.asker = _answer("allow_once", asked)
    assert not (await _check(gate, "bash_exec", {"command": "rm -rf dist"})).allowed
    assert asked == []


# ------------------------------------------------------------------- fail closed

@pytest.mark.asyncio
async def test_no_asker_denies_with_the_named_reason(tmp_path, caplog):
    gate = _gate(tmp_path, asker=None)
    outcome = await _check(gate, "bash_exec", {"command": "cargo build"})
    assert not outcome.allowed
    assert outcome.reason == NO_ASKER_REASON
    assert "permissions.mode: unattended" in caplog.text


@pytest.mark.asyncio
async def test_no_asker_warns_only_once_per_gate(tmp_path, caplog):
    gate = _gate(tmp_path, asker=None)
    for _ in range(3):
        await _check(gate, "bash_exec", {"command": "cargo build"})
    assert caplog.text.count("permissions.mode: unattended") == 1


@pytest.mark.asyncio
async def test_unattended_never_reaches_the_asker(tmp_path):
    """Bench and cron pin this mode; nothing there may block on a human (PRD §3.4)."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, mode="unattended", asker=_answer("reject_once", seen))
    outcome = await _check(gate, "bash_exec", {"command": "rm -rf build"})
    assert outcome.allowed and seen == []


@pytest.mark.asyncio
async def test_deny_tier_is_never_asked_about(tmp_path):
    seen: list[PermissionRequest] = []
    gate = _gate(
        tmp_path,
        asker=_answer("allow_always", seen),
        deny=deny_fn_from(PermissionEvaluator(), PermissionConfig()),
    )
    outcome = await _check(gate, "bash_exec", {"command": "sudo rm -rf /"})
    assert not outcome.allowed and seen == []


# ----------------------------------------------------------------------- timeout

@pytest.mark.asyncio
async def test_timeout_denies_as_reject_once_and_says_so_on_the_bus(tmp_path):
    bus = EventBus()

    async def _never(request):
        await asyncio.sleep(10)
        return Decision(kind="allow_always")

    gate = _gate(tmp_path, asker=_never, bus=bus)
    outcome = await _check(gate, "bash_exec", {"command": "cargo build"}, tool_timeout_s=0.05)
    assert not outcome.allowed
    assert "no answer" in outcome.reason
    # A sub-second deadline used to render as "no answer within 0s", which reads as a bug in
    # the gate rather than as a short timeout.
    assert "within 0.05s" in outcome.reason

    resolved = bus.history(event_types=[PermissionResolved])
    assert [e.decision for e in resolved] == ["reject_once"]
    assert resolved[0].wrote_grant is False
    assert gate.grants.refused(gate.workspace, "shell-unfamiliar", "cargo build") is None


def test_ask_timeout_derives_from_the_tool_timeout(tmp_path):
    gate = _gate(tmp_path)
    assert gate._timeout_s(30.0) == 30.0 * ASK_TIMEOUT_TOOL_MULTIPLE
    assert gate._timeout_s(None) is None


def test_configured_ask_timeout_wins_over_the_derivation(tmp_path):
    from localharness.agent.gate_types import GateSettings

    gate = _gate(tmp_path, settings=GateSettings(ask_timeout_s=7.0))
    assert gate._timeout_s(30.0) == 7.0
    assert gate._timeout_s(None) == 7.0


# ------------------------------------------------------------------------ events

@pytest.mark.asyncio
async def test_both_events_carry_the_ask_rate_fields(tmp_path):
    bus = EventBus()
    gate = _gate(tmp_path, asker=_answer("allow_always"), bus=bus, channel_name="terminal")
    await _check(gate, "bash_exec", {"command": "cargo build"})

    asked = bus.history(event_types=[PermissionAsked])
    resolved = bus.history(event_types=[PermissionResolved])
    assert len(asked) == 1 and len(resolved) == 1
    assert asked[0].klass == "shell-unfamiliar"
    assert asked[0].key == "cargo build"
    assert asked[0].channel == "terminal"
    assert resolved[0].decision == "allow_always"
    assert resolved[0].wrote_grant is True
    assert resolved[0].latency_ms is not None


@pytest.mark.asyncio
async def test_ungrantable_events_carry_no_key(tmp_path):
    """PRD §3.6: `key` is None for the classes that ask every time."""
    bus = EventBus()
    gate = _gate(tmp_path, asker=_answer("allow_once"), bus=bus)
    await _check(gate, "bash_exec", {"command": "rm -rf build"})
    assert bus.history(event_types=[PermissionAsked])[0].key is None


# ------------------------------------------------------------------------- modes

def test_set_mode_validates_the_name(tmp_path):
    gate = _gate(tmp_path)
    assert gate.set_mode("read-only") == "read-only"
    with pytest.raises(ValueError, match="unknown mode"):
        gate.set_mode("yolo")


def test_a_human_can_set_unattended_from_their_own_channel(tmp_path, caplog):
    """``/mode unattended`` is accepted, and the switch is logged rather than refused.

    Until v0.14.1 the gate raised on this, on the reasoning that a channel must not be able to
    switch the gate off. The owner met that rule from the other side (2026-09-11: "I can't swap
    my active localharness session to unattended without exiting"): the person typing ``/mode``
    in their own terminal IS the person the gate protects, and making them restart to make that
    decision was the bug. What replaces the refusal is an audit line, so a session that spent
    part of its life with the gate off says so in its own log.
    """
    gate = _gate(tmp_path)
    with caplog.at_level(logging.INFO, logger="localharness.agent.gate"):
        assert gate.set_mode("unattended", from_channel=True) == "unattended"
    assert gate.mode == "unattended"

    logged = [r for r in caplog.records if r.msg == MODE_SET_FROM_CHANNEL_LOG]
    assert len(logged) == 1
    assert logged[0].getMessage() == MODE_SET_FROM_CHANNEL_LOG % ("guarded", "unattended", "test")


@pytest.mark.asyncio
async def test_mode_is_live_on_the_shared_object(tmp_path):
    """Subagents hold the same gate, so a /mode switch has to bite mid-session (PRD §3.4)."""
    gate = _gate(tmp_path, asker=_answer("allow_once"))
    assert (await _check(gate, "write", {"path": "notes.md"}, WRITE)).allowed
    gate.set_mode("read-only", from_channel=True)
    outcome = await _check(gate, "write", {"path": "notes.md"}, WRITE)
    assert not outcome.allowed
    assert outcome.reason == "not permitted in read-only mode"


@pytest.mark.asyncio
async def test_a_keyless_request_writes_nothing_either_way(tmp_path):
    """A computed command name (`$(echo rm) -rf x`) has nothing to remember: `grantable=False`
    and `key=None`. Neither answer may become durable state — a bare `bash_exec` deny would ban
    every shell command forever, which is not what the human answered."""
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    outcome = await _check(gate, "bash_exec", {"command": "$(echo rm) -rf build"})
    assert not outcome.allowed
    assert gate.grants.refused(gate.workspace, "shell-unfamiliar", "bash_exec") is None
    assert not (tmp_path / "grants.yaml").exists()

    gate.asker = _answer("allow_always")
    assert (await _check(gate, "bash_exec", {"command": "$(echo rm) -rf build"})).allowed
    assert gate.grants.lookup(gate.workspace, "shell-unfamiliar", "bash_exec") is None


# ------------------------------------------------------------ derivation helpers

def test_tool_meta_reads_the_schema_and_the_mcp_group():
    plain = ToolSchema(name="write", description="", parameters={}, group="fs.write", destructive=True)
    assert tool_meta_from_schema(plain) == ToolMeta(destructive=True, group="fs.write")

    mcp = ToolSchema(name="srv__do", description="", parameters={}, group="mcp/srv")
    meta = tool_meta_from_schema(mcp)
    assert meta.is_mcp and meta.mcp_server == "srv"


@pytest.mark.asyncio
async def test_grant_store_is_never_read_from_the_workspace(tmp_path):
    """A repo cannot pre-approve itself: the gate's store is the global one, always."""
    workspace = tmp_path / "project"
    (workspace / ".localharness").mkdir(parents=True)
    (workspace / ".localharness" / "grants.yaml").write_text(
        f"{workspace}:\n  grants:\n"
        "    - {key: 'rm -rf', class: shell-destructive, granted_at: x, channel: c, session_id: s}\n"
    )
    gate = _gate(tmp_path, workspace=workspace, asker=None)
    assert not (await _check(gate, "bash_exec", {"command": "rm -rf /"})).allowed


# ------------------------------- the deadline is a channel property (defect D3)

class _HoldsDialog:
    """A channel with a person in front of it (terminal, Zed): PRD §3.5 "Timeout: none"."""

    channel_id = "holds"
    can_ask = True
    ask_holds_dialog = True
    has_review_surface = True

    async def ask_permission(self, request):  # pragma: no cover - replaced per test
        raise AssertionError("not called")


class _Expires:
    """A message-shaped channel (Discord): a question nobody reacts to has to expire."""

    channel_id = "expires"
    can_ask = True
    ask_holds_dialog = False
    has_review_surface = False

    async def ask_permission(self, request):  # pragma: no cover - replaced per test
        raise AssertionError("not called")


@pytest.mark.asyncio
async def test_a_channel_that_holds_the_dialog_is_never_timed_out(tmp_path):
    """Verification A defect D3: an unanswered terminal prompt auto-denied after the tool's own
    timeout. A slow answer must still be the human's answer, at any tool timeout."""
    from localharness.channels.terminal import TerminalChannel

    assert TerminalChannel.ask_holds_dialog is True

    async def _slow(request):
        await asyncio.sleep(0.05)
        return Decision(kind="allow_once")

    gate = _gate(tmp_path)
    gate.attach_channel(_HoldsDialog())
    gate.asker = _slow
    assert gate._timeout_s(0.01) is None
    outcome = await _check(gate, "bash_exec", {"command": "cargo build"}, tool_timeout_s=0.01)
    assert outcome.allowed, outcome.reason


@pytest.mark.asyncio
async def test_a_channel_that_cannot_hold_the_dialog_still_times_out(tmp_path):
    from localharness.channels.discord import DiscordChannel

    assert DiscordChannel.ask_holds_dialog is False

    async def _never(request):
        await asyncio.sleep(10)
        return Decision(kind="allow_always")

    bus = EventBus()
    gate = _gate(tmp_path, bus=bus)
    gate.attach_channel(_Expires())
    gate.asker = _never
    assert gate._timeout_s(0.05) == 0.05 * ASK_TIMEOUT_TOOL_MULTIPLE
    outcome = await _check(gate, "bash_exec", {"command": "cargo build"}, tool_timeout_s=0.05)
    assert not outcome.allowed and "no answer" in outcome.reason
    assert [e.decision for e in bus.history(event_types=[PermissionResolved])] == ["reject_once"]


def test_attach_channel_reads_the_flag_and_defaults_it_off(tmp_path):
    gate = _gate(tmp_path)
    assert gate.ask_holds_dialog is False, "the safe default is a deadline"
    gate.attach_channel(_HoldsDialog())
    assert gate.ask_holds_dialog is True
    gate.attach_channel(_Expires())
    assert gate.ask_holds_dialog is False


# --------------------------------------------------------- the verdict itself raises

@pytest.mark.asyncio
async def test_a_verdict_that_raises_denies_instead_of_escaping(tmp_path, monkeypatch, caplog):
    """The gate's own crash is a denial, not a lost turn (`GATE_ERROR_REASON`).

    A null byte in a write path used to raise ValueError out of `evaluate`, through `check`,
    into the loop — the gate failing open in the worst way: not by allowing the call but by
    taking the agent down with it.
    """
    def _boom(*args, **kwargs):
        raise ValueError("embedded null byte")

    monkeypatch.setattr("localharness.agent.gate.evaluate", _boom)
    gate = _gate(tmp_path, asker=_answer("allow_always"))
    with caplog.at_level(logging.ERROR):
        outcome = await _check(gate, "bash_exec", {"command": "ls"})
    assert outcome.allowed is False and outcome.reason == GATE_ERROR_REASON
    assert "ValueError" in caplog.text, "the traceback has to reach the log to be fixable"


@pytest.mark.asyncio
async def test_a_null_byte_path_is_decided_not_raised(tmp_path):
    """End to end through the real verdict: the argument that found this."""
    gate = _gate(tmp_path, asker=_answer("reject_once"))
    outcome = await _check(gate, "write", {"path": "/tmp/x\x00y", "content": "x"}, WRITE)
    assert outcome.allowed is False


# --------------------------------------------------------------------- cancelled

@pytest.mark.asyncio
async def test_a_cancelled_ask_still_resolves_on_the_bus(tmp_path):
    """Every PermissionAsked gets a PermissionResolved, cancellation included (PRD §3.6).

    Ctrl-C, a channel closing, a turn timeout: the awaiting task is cancelled while a human is
    still looking at the prompt. `_ask` caught TimeoutError and nothing else, so the pair stayed
    open forever and read in a trace exactly like a prompt still waiting for an answer.
    """
    bus = EventBus()

    async def _never(request):
        await asyncio.sleep(10)
        return Decision(kind="allow_always")

    gate = _gate(tmp_path, asker=_never, bus=bus, ask_holds_dialog=True)
    task = asyncio.create_task(_check(gate, "bash_exec", {"command": "cargo build"}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(bus.history(event_types=[PermissionAsked])) == 1
    resolved = bus.history(event_types=[PermissionResolved])
    assert [e.decision for e in resolved] == [CANCELLED_RESOLUTION]
    assert resolved[0].wrote_grant is False
    assert gate.grants.lookup(gate.workspace, "shell-unfamiliar", "cargo build") is None


@pytest.mark.asyncio
async def test_cancellation_is_re_raised_not_swallowed(tmp_path):
    """A cancelled turn stays cancelled — the resolve is bookkeeping, not a rescue."""
    async def _never(request):
        await asyncio.sleep(10)
        return Decision(kind="allow_always")

    gate = _gate(tmp_path, asker=_never, ask_holds_dialog=True)  # no bus: nothing to publish
    task = asyncio.create_task(_check(gate, "bash_exec", {"command": "cargo build"}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ------------------------------------------------------------------- who is asking

@pytest.mark.asyncio
async def test_a_subagents_ask_is_labelled_and_the_owners_is_not(tmp_path):
    """One gate serves the orchestrator and every child (PRD §3.4), so the line has to say
    which of them is asking — but only when it is not the session's own agent."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_once", seen), owner_agent_id="orchestrator")

    await gate.check("bash_exec", {"command": "cargo build"}, SHELL,
                     agent_id="orchestrator", session_id="s")
    await gate.check("bash_exec", {"command": "cargo test"}, SHELL,
                     agent_id="researcher", session_id="s")

    assert seen[0].agent_id == "orchestrator"
    assert not seen[0].display.startswith("[")
    assert seen[1].agent_id == "researcher"
    assert seen[1].display.startswith("[researcher] ")


@pytest.mark.asyncio
async def test_the_first_agent_to_check_becomes_the_owner(tmp_path):
    """`start_cmd` does not pass an owner yet, so the gate derives one: the orchestrator runs
    before it can dispatch anything, so the first caller is the orchestrator by construction."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_once", seen))
    assert gate.owner_agent_id is None

    await gate.check("bash_exec", {"command": "cargo build"}, SHELL,
                     agent_id="main", session_id="s")
    assert gate.owner_agent_id == "main"

    await gate.check("bash_exec", {"command": "cargo test"}, SHELL,
                     agent_id="child", session_id="s")
    assert seen[0].display.startswith("bash_exec")
    assert seen[1].display.startswith("[child] ")


@pytest.mark.asyncio
async def test_an_allowed_call_still_claims_ownership(tmp_path):
    """The owner is whoever called FIRST, not whoever asked first — otherwise an orchestrator
    whose own calls all passed would be labelled as a subagent the moment one asked."""
    gate = _gate(tmp_path, asker=_answer("allow_once"))
    await gate.check("read", {"path": "x"}, ToolMeta(group="fs.read"),
                     agent_id="main", session_id="s")
    assert gate.owner_agent_id == "main"


@pytest.mark.asyncio
async def test_the_call_id_travels_to_the_request(tmp_path):
    """The ACP adapter pairs its dialog with a `tool_call` it already knows (`request.call_id`)."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_once", seen))
    await gate.check("bash_exec", {"command": "cargo build"}, SHELL,
                     agent_id="a", session_id="s", call_id="tc-7")
    assert seen[0].call_id == "tc-7"


@pytest.mark.asyncio
async def test_a_request_built_by_hand_needs_neither_field(tmp_path):
    """Both default to None so nothing that constructs a request itself had to change."""
    request = PermissionRequest(
        tool_name="bash_exec", tool_params={}, klass="shell-unfamiliar", key="k",
        grantable=True, reason="r", display="d",
    )
    assert request.agent_id is None and request.call_id is None


# ------------------------------------------------------------------- staging (auto)

def _auto_gate(tmp_path, **kw) -> PermissionGate:
    """A gate in the DEFAULT mode, where a blocked call is parked rather than asked about.

    Kept apart from `_gate` above, which pins `guarded` to exercise the blocking ask: these
    tests are about the mode a person gets without choosing one, and the whole behaviour under
    test is that the asker is never reached.
    """
    return _gate(tmp_path, mode="auto", **kw)


HOME_DELETE = {"command": "rm -rf ~/old-notes"}
"""A call `auto` still stops: a target-scoped destructive verb pointed OUTSIDE the workspace
(`AUTO_BLACKLIST.target_scoped_verbs`). `rm -rf build` inside the project is allowed silently,
which is the point of the mode — so the fixture has to leave the boundary to stage anything."""


@pytest.mark.asyncio
async def test_auto_stages_instead_of_asking(tmp_path):
    """The owner ruling of 2026-09-12: a blacklisted call must never hold the agent loop.

    Everything in one test because they are one behaviour: nobody is awaited, the model is told
    to carry on, the call is parked under a number, and the human hears about it on the bus.
    """
    seen: list[PermissionRequest] = []
    bus = EventBus()
    gate = _auto_gate(tmp_path, asker=_answer("allow_once", seen), bus=bus, channel_name="terminal")

    outcome = await _check(gate, "bash_exec", HOME_DELETE)

    assert outcome.allowed is False
    assert seen == [], "the asker was called; a prompt held the loop"
    assert "pending #1" in outcome.reason
    assert outcome.pending is not None and outcome.pending.id == 1
    assert list(gate.pending) == [1]
    assert gate.pending[1].rendering == outcome.pending.rendering
    assert "rm -rf" in gate.pending[1].rendering

    staged = bus.history(event_types=[PermissionStaged])
    assert len(staged) == 1
    assert staged[0].pending.id == 1 and staged[0].total == 1
    assert staged[0].channel == "terminal"
    assert bus.history(event_types=[PermissionAsked]) == []


@pytest.mark.asyncio
async def test_the_same_call_again_is_the_same_pending_number(tmp_path):
    """A model that retries must fill the log, not the queue."""
    bus = EventBus()
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"), bus=bus)

    first = await _check(gate, "bash_exec", HOME_DELETE)
    second = await _check(gate, "bash_exec", HOME_DELETE)

    assert second.pending is not None and second.pending.id == first.pending.id == 1
    assert "already pending" in second.reason
    assert list(gate.pending) == [1]
    assert len(bus.history(event_types=[PermissionStaged])) == 1, "a retry re-announced itself"


@pytest.mark.asyncio
async def test_a_different_target_is_a_different_pending(tmp_path):
    """The queue is keyed by the CALL, not by the grant key: `rm -rf` is one key and two very
    different commands, and a human must never approve one by reading the other."""
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"))
    await _check(gate, "bash_exec", {"command": "rm -rf ~/notes"})
    await _check(gate, "bash_exec", {"command": "rm -rf ~/photos"})
    assert sorted(gate.pending) == [1, 2]


@pytest.mark.asyncio
async def test_approve_allows_exactly_one_re_run(tmp_path):
    """`/approve` answers the command the human read, not its class — so the ticket is spent by
    the next matching call and the one after that stages afresh."""
    bus = EventBus()
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"), bus=bus)
    await _check(gate, "bash_exec", HOME_DELETE)

    answered = await gate.approve(1)
    assert answered.id == 1 and gate.pending == {}
    resolved = bus.history(event_types=[PermissionResolved])
    assert [e.decision for e in resolved] == ["allow_once"]
    assert resolved[0].wrote_grant is False

    allowed = await _check(gate, "bash_exec", HOME_DELETE)
    assert allowed.allowed and "pending #1" in allowed.reason

    again = await _check(gate, "bash_exec", HOME_DELETE)
    assert not again.allowed and again.pending is not None and again.pending.id == 2


@pytest.mark.asyncio
async def test_approve_with_no_number_takes_the_oldest(tmp_path):
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"))
    await _check(gate, "bash_exec", {"command": "rm -rf ~/notes"})
    await _check(gate, "bash_exec", {"command": "rm -rf ~/photos"})
    assert (await gate.approve()).id == 1
    assert list(gate.pending) == [2]


@pytest.mark.asyncio
async def test_deny_clears_the_pending_and_grants_nothing(tmp_path):
    bus = EventBus()
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"), bus=bus)
    await _check(gate, "bash_exec", HOME_DELETE)

    assert (await gate.deny(1)).id == 1
    assert gate.pending == {}
    assert [e.decision for e in bus.history(event_types=[PermissionResolved])] == ["reject_once"]

    # A denial is a reject_ONCE: it clears the queue entry, it does not become a stored refusal,
    # so the next identical call is parked again rather than silently denied forever.
    again = await _check(gate, "bash_exec", HOME_DELETE)
    assert not again.allowed and again.pending is not None and again.pending.id == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["approve", "deny"])
async def test_an_unknown_pending_number_raises(tmp_path, verb):
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"))
    with pytest.raises(KeyError):
        await getattr(gate, verb)(7)
    with pytest.raises(KeyError):
        await getattr(gate, verb)()  # nothing is pending at all


@pytest.mark.asyncio
async def test_guarded_still_blocks_on_the_asker(tmp_path):
    """The mode a person CHOSE to be asked in is untouched: being asked is what it is for."""
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, asker=_answer("allow_once", seen))  # _gate pins `guarded`
    outcome = await _check(gate, "bash_exec", HOME_DELETE)
    assert outcome.allowed and len(seen) == 1
    assert gate.pending == {} and outcome.pending is None


@pytest.mark.asyncio
async def test_auto_without_an_asker_still_fails_closed(tmp_path):
    """Staging is a promise that a human can be reached. A bench or cron run cannot keep it, so
    it keeps the loud denial instead of piling up a queue nobody will ever answer."""
    gate = _auto_gate(tmp_path, asker=None)
    outcome = await _check(gate, "bash_exec", HOME_DELETE)
    assert not outcome.allowed and outcome.pending is None
    assert outcome.reason == NO_ASKER_REASON
    assert gate.pending == {}


@pytest.mark.asyncio
async def test_the_deny_tier_says_nobody_can_approve_it_and_is_not_stageable(tmp_path):
    """The owner watched the model stall in prose after a hard refusal of `rm -rf`, asking the
    human to run it. A DENY is the owner's own never-run list — no /approve reaches it."""
    gate = _auto_gate(
        tmp_path,
        asker=_answer("allow_once"),
        deny=deny_fn_from(PermissionEvaluator(), PermissionConfig()),
    )
    outcome = await _check(gate, "bash_exec", {"command": "sudo rm -rf /"})
    assert not outcome.allowed
    assert outcome.reason.endswith(UNGRANTABLE_OBSERVATION_SUFFIX)
    assert outcome.pending is None and gate.pending == {}


@pytest.mark.asyncio
async def test_an_approval_never_beats_the_deny_tier(tmp_path):
    """A ticket is spent on sight, and the never-run list still wins — the patterns may even
    have changed between the staging and the answer."""
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"))
    await _check(gate, "bash_exec", HOME_DELETE)
    await gate.approve(1)

    gate._config_deny = deny_fn_from(PermissionEvaluator(), PermissionConfig(
        deny_patterns=["bash_exec(*old-notes*)"]))
    outcome = await _check(gate, "bash_exec", HOME_DELETE)
    assert not outcome.allowed
    assert outcome.reason.endswith(UNGRANTABLE_OBSERVATION_SUFFIX)
    assert gate._approved_once == {}, "the one-shot ticket survived a hard deny"


@pytest.mark.asyncio
async def test_a_subagents_pending_says_whose_it_is(tmp_path):
    """One gate serves the orchestrator and every subagent it dispatches (PRD §3.4), and
    "approve `rm -rf`" is a different question depending on which agent asked it."""
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"), owner_agent_id="main")
    await gate.check("bash_exec", HOME_DELETE, SHELL, agent_id="worker", session_id="s")
    assert gate.pending[1].agent_label == "[worker] "
    assert gate.pending[1].rendering.startswith("bash_exec")
    assert gate.pending[1].session_id == "s"


@pytest.mark.asyncio
async def test_an_approval_ticket_never_beats_a_refusal_recorded_since(tmp_path):
    """The ticket turns an ASK into an ALLOW and nothing else: a "never here" written between
    staging and the re-run is a DENY from the verdict, and it still wins (critic, 2026-09-12)."""
    gate = _auto_gate(tmp_path, asker=_answer("allow_once"))
    staged = await _check(gate, "bash_exec", HOME_DELETE)
    request = staged.pending.request
    await gate.approve(1)

    gate.grants.add_refusal(
        new_refusal(key=request.key, klass=request.klass, workspace=gate.workspace,
                    channel="test", session_id="s"))
    outcome = await _check(gate, "bash_exec", HOME_DELETE)
    assert not outcome.allowed and "never here" in outcome.reason
    assert gate._approved_once == {}, "the ticket is spent on sight, never stockpiled"
