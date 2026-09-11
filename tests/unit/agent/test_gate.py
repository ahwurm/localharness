"""The effectful gate: asking, remembering, failing closed (PRD §3.5, §3.6, §3.3).

The pure verdict is covered by `test_verdict.py`; these tests are about what happens AROUND
it — whether a human is awaited, whether the answer becomes durable, whether a channel that
cannot ask denies loudly, and whether the two bus events carry what the ask-rate report needs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from localharness.agent.gate import (
    ASK_TIMEOUT_TOOL_MULTIPLE,
    NO_ASKER_REASON,
    PermissionGate,
    deny_fn_from,
    deny_pattern_for,
    tool_meta_from_schema,
)
from localharness.agent.gate_types import Decision, PermissionRequest, ToolMeta
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.grants import GrantStore
from localharness.config.models import PermissionConfig
from localharness.core.bus import EventBus
from localharness.core.events import PermissionAsked, PermissionResolved
from localharness.tools import ToolSchema

SHELL = ToolMeta(group="shell")
WRITE = ToolMeta(group="fs.write")


def _gate(tmp_path: Path, **kw) -> PermissionGate:
    workspace = kw.pop("workspace", tmp_path / "project")
    workspace.mkdir(exist_ok=True)
    return PermissionGate(
        boundary=kw.pop("boundary", workspace),
        workspace=workspace,
        grants=kw.pop("grants", GrantStore(tmp_path / "grants.yaml")),
        channel_name=kw.pop("channel_name", "test"),
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
    assert gate.grants.lookup(gate.workspace, "rm -rf") is None


@pytest.mark.asyncio
async def test_reject_always_writes_a_deny_the_deny_tier_then_matches(tmp_path):
    """"Never here" wins forever after — through the DENY tier, before any ask (PRD §3.3)."""
    gate = _gate(tmp_path, asker=_answer("reject_always"))
    first = await _check(gate, "bash_exec", {"command": "cargo publish"})
    assert not first.allowed

    gate.asker = _answer("allow_always")  # even a yes cannot undo it
    second = await _check(gate, "bash_exec", {"command": "cargo publish"})
    assert not second.allowed
    assert "never here" in second.reason
    assert gate.grants.deny_patterns_for(gate.workspace) == ["bash_exec(*cargo publish*)"]


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

    resolved = bus.history(event_types=[PermissionResolved])
    assert [e.decision for e in resolved] == ["reject_once"]
    assert resolved[0].wrote_grant is False
    assert gate.grants.deny_patterns_for(gate.workspace) == []


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


def test_channel_can_never_set_unattended(tmp_path):
    gate = _gate(tmp_path)
    with pytest.raises(ValueError, match="cannot be set from a channel"):
        gate.set_mode("unattended", from_channel=True)
    assert gate.mode == "guarded"
    assert gate.set_mode("unattended") == "unattended"  # config still can


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
    assert gate.grants.deny_patterns_for(gate.workspace) == []

    gate.asker = _answer("allow_always")
    assert (await _check(gate, "bash_exec", {"command": "$(echo rm) -rf build"})).allowed
    assert gate.grants.lookup(gate.workspace, "bash_exec") is None


# ------------------------------------------------------------ derivation helpers

@pytest.mark.parametrize(
    "tool,klass,key,expected",
    [
        ("bash_exec", "shell-unfamiliar", "cargo publish", "bash_exec(*cargo publish*)"),
        ("bash_exec", "shell-destructive", "rm -rf", "bash_exec(*rm -rf*)"),
        ("bash_exec", "protected-path", "/h/.ssh/id", "bash_exec(*/h/.ssh/id*)"),
        ("write", "edit-outside", "/tmp/out", "write(/tmp/out*)"),
        ("web_fetch", "network-host", "evil.example", "web_fetch(*evil.example*)"),
        ("python_exec", "code-exec", "python_exec", "python_exec"),
        ("agent", "delegate", None, "agent"),
    ],
)
def test_deny_pattern_shapes(tool, klass, key, expected):
    assert deny_pattern_for(tool, klass, key) == expected


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
