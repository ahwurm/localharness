"""Sleep and wake through the real session builder (0.16.2): a mobile session asked to sleep ends
through its REPL's own `read_input`, writes the exact conversation and the permission mode beside
its session log, and the next mobile session takes that file and continues the same thread as a
new sitting. A wake that fails before a session exists puts the file back; the new-chat verb
discards it.

STUBBED (`_machine`, as the other mobile e2e): the LLM probe, the tokenizer, plugin discovery is
real. NOT stubbed here: the REPL loop — the sleep lands in the real `_run_classic`, which is the
point. No model answers, so the turns are seeded into the loop the way a model's would land.
"""
from __future__ import annotations

import asyncio
import json
import stat

import pytest

from localharness.agent.loop import AgentLoop
from localharness.channels.mobile.channel import MobileChannel
from localharness.cli.repl import OrchestratorREPL
from localharness.cli.session_resume import SLEEP_ACTION, Resume, asleep_path, write_asleep
from localharness.cli.slash_commands import set_plugin_rows
from localharness.core.bus import EventBus
from tests.integration.test_image_plugin_e2e import _machine
from tests.unit.test_start_cmd import _capture_start_console

CONV = [{"role": "user", "content": "the key is under the blue mat"},
        {"role": "assistant", "content": "noted: under the blue mat"}]


@pytest.fixture(autouse=True)
def _process_state():
    yield
    set_plugin_rows(())


def _rig(tmp_path, monkeypatch, fake_home):
    """`_machine` with the REAL REPL loop put back (it stubs run() to a no-op), console captured."""
    real_run = OrchestratorREPL.run
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    monkeypatch.setattr(OrchestratorREPL, "run", real_run)
    _capture_start_console(monkeypatch)
    return asleep_path(global_dir / "agents" / "orchestrator" / "sessions")


def _spy_resume(monkeypatch) -> list:
    resumed: list = []
    real = AgentLoop.resume

    def spy(self, conversation, prior_context):
        resumed.append(([dict(m) for m in conversation], prior_context))
        return real(self, conversation, prior_context)

    monkeypatch.setattr(AgentLoop, "resume", spy)
    return resumed


async def _session(channel, after_bind, *, fresh_thread: bool = False):
    """One real mobile session; `after_bind` runs once it is bound and must end it. Returns
    `_start_async`'s answer, or raises what the build raised."""
    from localharness.cli.start_cmd import _start_async

    # trust_project: a fresh folder would otherwise ask its trust question through the channel,
    # an open question nobody here answers (and, rightly, not an idle session).
    task = asyncio.ensure_future(_start_async(
        None, False, False, None, channel_mode="mobile", mobile_channel=channel,
        fresh_thread=fresh_thread, trust_project=True))
    for _ in range(1500):
        if channel.session_id is not None or task.done():
            break
        await asyncio.sleep(0.02)
    if not task.done():
        await after_bind(channel)
    return await asyncio.wait_for(task, timeout=60)


def _record(**kw) -> Resume:
    fields = dict(action=SLEEP_ACTION, agent_name="orchestrator", conversation=tuple(CONV),
                  prior_context="PRIOR", eviction_store=None, queued=(), gate_mode="auto",
                  previous_sitting_id="earlier-sitting")
    return Resume(**(fields | kw))


def test_the_thread_sleeps_to_disk_and_wakes_in_the_next_session(tmp_path, monkeypatch, fake_home):
    path = _rig(tmp_path, monkeypatch, fake_home)
    channel = MobileChannel(bus=EventBus(), config={})
    seen: dict = {}

    async def main():
        async def talk_then_sleep(ch):
            ch._agent_loop.resume(list(CONV), "PRIOR")  # the turns a model would have produced
            ch._gate.set_mode("unattended")
            assert ch.idle(), "bound, nothing running, nobody attached, nothing open"
            ch.request_sleep()

        first = await _session(channel, talk_then_sleep)
        seen["first"] = first
        seen["file_after_first"] = json.loads(path.read_text())
        seen["mode_600"] = stat.S_IMODE(path.stat().st_mode)
        seen["where"] = (path.parent.name, path.parent.parent.name)  # the agent's session-log folder

        resumed = _spy_resume(monkeypatch)
        channel.reset_session()  # what bring-up does once a session has slept

        async def look_then_sleep(ch):
            seen["mode"] = ch._gate.mode
            seen["sleep_requested_at_bind"] = ch.sleep_requested
            seen["file_gone_while_awake"] = not path.exists()
            ch.request_sleep()

        seen["second"] = await _session(channel, look_then_sleep)
        seen["resumed"] = resumed
        seen["file_after_second"] = json.loads(path.read_text())

    asyncio.run(main())

    first, second = seen["first"], seen["second"]
    assert first is not None and first.resume.slept
    assert first.resume.conversation == tuple(CONV) and first.resume.gate_mode == "unattended"
    assert seen["file_after_first"]["conversation"] == CONV
    assert seen["file_after_first"]["previous_sitting_id"] == first.resume.previous_sitting_id
    assert seen["mode_600"] == 0o600 and seen["where"] == ("sessions", "orchestrator")

    assert seen["resumed"] == [(CONV, "PRIOR")], "the exact conversation, into the next AgentLoop"
    assert seen["mode"] == "unattended", "the permission mode as it was left"
    assert seen["sleep_requested_at_bind"] is False and seen["file_gone_while_awake"]
    assert second.resume.previous_sitting_id != first.resume.previous_sitting_id, "same thread, new sitting"
    assert seen["file_after_second"]["previous_sitting_id"] == second.resume.previous_sitting_id


def test_a_wake_that_fails_before_a_session_exists_puts_the_thread_back(tmp_path, monkeypatch, fake_home):
    path = _rig(tmp_path, monkeypatch, fake_home)
    write_asleep(path, _record())
    import localharness.tools.builtin as builtin

    def boom(*a, **kw):
        raise RuntimeError("the registry would not bind")

    monkeypatch.setattr(builtin, "bind_agent_store_tools", boom)
    channel = MobileChannel(bus=EventBus(), config={})
    with pytest.raises(RuntimeError, match="would not bind"):
        asyncio.run(_session(channel, None))
    assert json.loads(path.read_text())["conversation"] == CONV, "still asleep: the next message tries again"


def test_the_new_chat_verb_discards_a_sleeping_thread(tmp_path, monkeypatch, fake_home):
    path = _rig(tmp_path, monkeypatch, fake_home)
    write_asleep(path, _record())
    resumed = _spy_resume(monkeypatch)
    channel = MobileChannel(bus=EventBus(), config={})

    async def sleep_at_once(ch):
        ch.request_sleep()

    result = asyncio.run(_session(channel, sleep_at_once, fresh_thread=True))
    assert resumed == [], "nothing was carried in"
    on_disk = json.loads(path.read_text())
    assert on_disk["conversation"] == [] and on_disk["previous_sitting_id"] == result.resume.previous_sitting_id


def test_a_sleeping_server_holds_nothing_of_the_session(tmp_path, monkeypatch, fake_home):
    """The point of sleeping: the memory plugin, its embedding engine (the model, about 1.2 GB
    resident once loaded), the agent loop and the tool registry all become garbage once the
    session has slept and the channel has reset. Found the first time by measuring the real
    server, which stayed at 1.8 GB: the REPL's resolvers on the channel kept the whole chain."""
    import gc
    import weakref

    _rig(tmp_path, monkeypatch, fake_home)
    channel = MobileChannel(bus=EventBus(), config={})
    refs: dict = {}

    async def main():
        async def grab_then_sleep(ch):
            plugin = ch.memory_slot()._occupant
            refs.update(plugin=weakref.ref(plugin), engine=weakref.ref(plugin._engine),
                        loop=weakref.ref(ch._agent_loop), registry=weakref.ref(ch._tool_registry))
            ch.request_sleep()

        await _session(channel, grab_then_sleep)
        channel.reset_session()  # what bring-up does once the builder has slept
        gc.collect()  # and what the runner does after it

    asyncio.run(main())
    alive = {name: ref() is not None for name, ref in refs.items()}
    assert alive == {"plugin": False, "engine": False, "loop": False, "registry": False}, alive
