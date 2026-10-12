"""`/instructions off | on | show` typed into the REPL (spec 16, D-59, D-60).

Drives the real ``OrchestratorREPL`` with a scripted channel and a real ``ProjectInstructions``
handle on a tmp root, the store hermetic through ``instructions_home``. The texts are the shared
handler's, verbatim: no preview of the file and no diff of a change.
"""
# ruff: noqa: F811  (instructions_home is imported as a fixture and requested by name)
from __future__ import annotations

import asyncio

from localharness.config.project_records import (
    open_project_instructions,
    run_instructions_command,
)
from tests.unit.instructions_support import instructions_home, make_root  # noqa: F401
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl

INFO = {"style": "system.info"}
WINDOW = 65536


def _handle(tmp_path):
    root = make_root(tmp_path, {"CLAUDE.md": "a" * 5221 + "\n"})
    handle = open_project_instructions(
        root, sitting_id="s1", max_ratio=0.25, basis="layer", no_input=False,
        window_tokens=WINDOW,
    )
    assert handle is not None
    handle.on_delivered()  # the record the first delivery writes (D-33)
    return handle


def _repl(channel, handle):
    repl, agent, bus = _build_repl(channel)
    repl._instructions = handle
    return repl, agent, bus


def test_instructions_show_prints_the_path_size_digest_and_state_and_runs_no_turn(
        tmp_path, instructions_home):
    handle = _handle(tmp_path)
    channel = RecordingChannel(["/instructions show"])
    repl, agent, bus = _repl(channel, handle)

    asyncio.run(repl.run())

    expected = run_instructions_command(handle, "show")
    assert [(t, m) for t, m in channel.sent if "CLAUDE.md" in t] == [(expected, INFO)]
    state = handle.state
    assert repr(state.path) in expected and state.digest in expected
    assert "5.1 KB" in expected and expected.endswith(state.state)
    assert "aaaa" not in expected  # no preview of the file (D-01)
    agent.run_turn.assert_not_called()
    bus.publish.assert_not_called()


async def test_a_repl_built_without_the_keyword_says_there_is_no_file():
    channel = RecordingChannel([])
    repl, agent, _ = _build_repl(channel)
    assert repl._instructions is None

    assert await repl._handle_slash("/instructions show") is True

    assert channel.sent == [("no instructions file in this workspace", INFO)]
    agent.run_turn.assert_not_called()


# --------------------------------------------------------------------------- off and on (D-01, D-33)

class _ClearingChannel(RecordingChannel):
    """A channel with the terminal's ``clear_instructions_status`` (11-07), counting its calls."""

    def __init__(self, inputs) -> None:
        super().__init__(inputs)
        self.cleared = 0

    def clear_instructions_status(self) -> None:
        self.cleared += 1


async def test_off_answers_one_line_records_off_and_stays_off_after_a_hand_edit(
        tmp_path, instructions_home):
    handle = _handle(tmp_path)
    channel = RecordingChannel([])
    repl, agent, _ = _repl(channel, handle)

    assert await repl._handle_slash("/instructions off") is True

    assert channel.sent == [
        ("instructions off for this workspace; /instructions on to use CLAUDE.md", INFO)]
    assert handle.state.state == "off"
    assert handle.section(WINDOW) == (None, None)
    (tmp_path / "proj" / "CLAUDE.md").write_text("b" * 300 + "\n", encoding="utf-8")
    assert handle.section(WINDOW) == (None, None), "off is never re-adopted until on"
    agent.run_turn.assert_not_called()


async def test_on_uses_the_file_records_yes_and_clears_the_channels_pending_line(
        tmp_path, instructions_home):
    from tests.unit.instructions_support import load_store

    handle = _handle(tmp_path)
    channel = _ClearingChannel([])
    repl, _, _ = _repl(channel, handle)
    await repl._handle_slash("/instructions off")
    channel.sent.clear()

    assert await repl._handle_slash("/Instructions ON") is True

    assert channel.sent == [("using CLAUDE.md (5.1 KB)", INFO)]
    assert channel.cleared == 1
    assert handle.state.state == "yes"
    (record,) = (r["instructions"] for r in load_store()["roots"].values())
    assert record["state"] == "yes" and record["digest"] == handle.state.digest


async def test_off_does_not_clear_the_pending_line_and_a_channel_without_the_method_is_fine(
        tmp_path, instructions_home):
    handle = _handle(tmp_path)
    channel = _ClearingChannel([])
    repl, _, _ = _repl(channel, handle)
    await repl._handle_slash("/instructions off")
    assert channel.cleared == 0

    bare = RecordingChannel([])
    repl, _, _ = _repl(bare, handle)
    assert await repl._handle_slash("/instructions on") is True
    assert bare.sent == [("using CLAUDE.md (5.1 KB)", INFO)]


async def test_an_unknown_word_gets_the_usage_line_and_writes_nothing(tmp_path, instructions_home):
    from tests.unit.instructions_support import load_store

    handle = _handle(tmp_path)
    before = load_store()
    channel = RecordingChannel([])
    repl, _, _ = _repl(channel, handle)

    for line in ("/instructions preview", "/instructions"):
        assert await repl._handle_slash(line) is True

    assert channel.sent == [("usage: /instructions off | on | show", INFO)] * 2
    assert load_store() == before and handle.state.state != "off"


# --------------------------------------------------------------------------- session trust (D-61)

def _trust_repl(channel, handle, gate):
    repl, _, _ = _build_repl(channel)
    repl._gate = gate
    repl._instructions = handle
    return repl


async def test_the_trust_question_gets_the_handle_and_none_without_one(monkeypatch):
    from unittest.mock import MagicMock

    from localharness.cli import session_trust

    seen: list[dict] = []

    async def _spy(gate, notice=None, **kw):
        seen.append(kw)
        return "auto"

    monkeypatch.setattr(session_trust, "establish_session_trust", _spy)
    handle = MagicMock()

    await _trust_repl(RecordingChannel([]), handle, MagicMock())._establish_workspace_trust()
    await _trust_repl(RecordingChannel([]), None, MagicMock())._establish_workspace_trust()

    assert [kw["instructions"] for kw in seen[:1]] == [handle]
    assert seen[1] == {"instructions": None}


async def test_a_yes_to_the_session_question_makes_the_file_usable_through_the_handle(
        tmp_path, instructions_home):
    from localharness.agent.gate import PermissionGate
    from localharness.agent.gate_types import Decision
    from localharness.config import trust
    from localharness.config.grants import GrantStore

    handle = _handle(tmp_path)
    root = tmp_path / "proj"
    asked: list = []

    async def _asker(request):
        asked.append(request)
        return Decision(kind="allow_once")

    gate = PermissionGate(boundary=root, workspace=root, grants=GrantStore(tmp_path / "grants.yaml"),
                          mode="auto", asker=_asker, channel_name="test")
    channel = RecordingChannel([])

    await _trust_repl(channel, handle, gate)._establish_workspace_trust()

    assert len(asked) == 1 and "CLAUDE.md" in asked[0].reason
    assert trust.is_trusted_tree(root) is True
    assert handle.state.state == "yes"
