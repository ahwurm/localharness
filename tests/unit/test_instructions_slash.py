"""`/instructions off | on | show` typed into the REPL (spec 16, D-59, D-60).

Drives the real ``OrchestratorREPL`` with a scripted channel and a real ``ProjectInstructions``
handle on a tmp root, the store hermetic through ``instructions_home``. The texts are the shared
handler's, verbatim: no preview of the file and no diff of a change.
"""
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
