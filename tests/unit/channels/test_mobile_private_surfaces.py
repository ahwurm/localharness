"""A private chat leaves nothing behind through a queued line, a cache header, an answer or a notification."""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from localharness.agent.gate_types import PermissionRequest
from localharness.agent.loop import AgentLoop
from localharness.channels.mobile import push
from localharness.channels.mobile.channel import (
    PRIVATE_TAINTED_CONFIRM,
    MobileChannel,
)
from localharness.core.bus import EventBus
from localharness.core.events import TaskComplete, TurnCompleted, TurnStarted
from localharness.core.run_folder import begin, end, kept_names, note_written
from tests.unit.channels.test_mobile_artifacts import PNG, _put, _two_roots
from tests.unit.channels.test_mobile_private import _private_stack
from tests.unit.channels.test_mobile_push import _pushable
from tests.unit.channels.test_mobile_server import BEARER, JSON


def _frames(client) -> list[tuple[str, dict]]:
    rows = []
    while not client.queue.empty():
        name, _, payload = client.queue.get_nowait()
        rows.append((name, json.loads(payload)))
    return rows


async def _channel(tmp_path, *, private=False) -> MobileChannel:
    channel = MobileChannel(bus=EventBus(persist_path=tmp_path / "bus-events.jsonl"), config={})
    await channel.start()
    channel.private = private
    return channel


async def test_a_line_typed_into_another_chat_is_dropped_by_id_with_no_text_and_no_bus_event(tmp_path):
    channel = await _channel(tmp_path)
    client = channel.attach_client()
    channel.submit("private words", line_id="L1", private=True)
    channel.submit("normal words", line_id="L2")
    assert await channel.read_input() == "normal words"
    sent = [row for row in _frames(client) if row[0] == "LineDropped"]
    assert sent == [("LineDropped", {"session_id": None, "frame_type": "LineDropped", "line_id": "L1"})]
    assert "private words" not in json.dumps(sent)
    log = tmp_path / "bus-events.jsonl"
    assert not log.exists() or "LineDropped" not in log.read_text() and "private words" not in log.read_text()


async def test_a_private_chat_reads_its_own_line_and_drops_a_normal_one(tmp_path):
    channel = await _channel(tmp_path, private=True)
    client = channel.attach_client()
    channel.submit("normal words", line_id="L1", private=False)
    channel.submit("private words", line_id="L2")
    assert await channel.read_input() == "private words"
    assert [row[1]["line_id"] for row in _frames(client) if row[0] == "LineDropped"] == ["L1"]


async def test_a_dropped_line_with_no_id_sends_no_frame(tmp_path):
    channel = await _channel(tmp_path)
    client = channel.attach_client()
    channel.submit("private words", private=True)
    channel.submit("normal words")
    assert await channel.read_input() == "normal words"
    assert [row for row in _frames(client) if row[0] == "LineDropped"] == []


async def test_a_stale_line_neither_keeps_a_chat_awake_nor_counts_as_queued(tmp_path):
    channel = await _channel(tmp_path)
    channel.bind_runtime(session_id="s1", agent_id="a", session_dir=tmp_path / "s")
    channel.submit("private words", line_id="L1", private=True)
    assert channel.queued_input == 0 and not channel.has_turns() and channel.idle()
    channel.submit("normal words")
    assert channel.queued_input == 1 and channel.has_turns() and not channel.idle()


async def test_a_stop_still_leaves_a_waiting_line_of_the_chat_in_the_queue(tmp_path):
    channel = await _channel(tmp_path)
    channel.submit("normal words", line_id="L1")
    channel.request_sleep(now=True)
    with pytest.raises(EOFError):
        await channel.read_input()
    assert channel.queued_input == 1


@pytest.mark.parametrize("sent, kept", [("Abc_123-x", "Abc_123-x"), ("a" * 64, "a" * 64), ("a" * 65, None),
                                        ("bad id", None), ("", None), (7, None)])
async def test_message_keeps_a_well_formed_line_id_and_treats_any_other_as_absent(tmp_path, sent, kept):
    channel, _, client = await _private_stack(tmp_path, private=False, turns=1)
    got = await client.post("/api/sessions/s1/message", json={"text": "hi", "line_id": sent}, headers=JSON)
    assert got.status_code == 200
    _, _, line_id, private = channel._inbound.get_nowait()
    assert line_id == kept and private is False


async def test_message_and_command_queue_with_the_privacy_they_were_sent_under(tmp_path):
    channel, _, client = await _private_stack(tmp_path, private=True, turns=1)
    await client.post("/api/sessions/s1/message", json={"text": "hi", "private": True}, headers=JSON)
    await client.post("/api/sessions/s1/command", json={"text": "/help", "private": True}, headers=JSON)
    assert [item[3] for item in (channel._inbound.get_nowait(), channel._inbound.get_nowait())] == [True, True]


async def test_every_api_reply_is_no_store_while_a_chat_is_private_and_unchanged_otherwise(tmp_path):
    channel, _, client = await _private_stack(tmp_path, private=True)
    for path in ("/api/health", "/api/sessions", "/api/protocol"):
        got = await client.get(path, headers=BEARER)
        assert got.headers["cache-control"] == "no-store", path
    channel.private = False
    assert (await client.get("/api/protocol", headers=BEARER)).headers.get("cache-control") != "no-store"
    channel.private = None
    assert (await client.get("/api/protocol", headers=BEARER)).headers.get("cache-control") != "no-store"


async def test_a_private_picture_is_no_store_and_a_normal_one_keeps_the_year_long_cache(tmp_path):
    roots, server, client = await _two_roots(tmp_path)
    name = _put(roots["image"], "art-20260930-120000-aaaaaa.png")
    path = f"/api/artifacts/image/{name[:-4]}"
    normal = await client.get(path, headers=BEARER)
    assert normal.content == PNG and "immutable" in normal.headers["cache-control"]
    server.channel.private = True
    private = await client.get(path, headers=BEARER)
    assert private.content == PNG and private.headers["cache-control"] == "no-store"
    assert (await client.get("/api/artifacts", headers=BEARER)).status_code == 404
    assert server._screens()["pictures"] is False


async def test_the_incognito_switch_reads_off_in_a_private_chat_until_it_is_turned_on(tmp_path):
    _, server, client = await _private_stack(tmp_path, private=True)
    assert server._screens()["incognito"] is False
    assert (await client.post("/api/incognito", json={"on": True}, headers=JSON)).json()["screens"]["incognito"] is True


def _ask(**over):
    base = dict(tool_name="bash_exec", tool_params={"command": "rm -rf build/"}, klass="shell-destructive",
                key="bash:rm", grantable=True, reason="destructive", display="bash_exec: rm", agent_id="a1")
    return PermissionRequest(**base | over)


async def _asking(tmp_path, request, *, private, marked=False):
    channel = await _channel(tmp_path, private=private)
    channel.run = SimpleNamespace(private=private)
    channel._gate = SimpleNamespace(taint=SimpleNamespace(source=lambda agent: "a saved page" if marked else None))
    client = channel.attach_client()
    task = asyncio.ensure_future(channel.ask_permission(request))
    await asyncio.sleep(0)
    frame = next(row[1] for row in _frames(client) if row[0] == "BlockingAsk")
    return channel, task, frame


async def test_the_trust_buttons_name_what_a_private_chat_does(tmp_path):
    request = _ask(tool_name="workspace", klass="workspace-trust", key=None, grantable=False)
    for private, names in ((True, ["Trust for this run", "Not now"]), (False, ["Trust this workspace", "Not now"])):
        _, task, frame = await _asking(tmp_path, request, private=private)
        assert [o["name"] for o in frame["options"]] == names
        task.cancel()


async def test_a_private_always_is_one_tap_with_nothing_permanent_said(tmp_path):
    channel, task, frame = await _asking(tmp_path, _ask(), private=True)
    assert not any(o["confirm_required"] for o in frame["options"])
    assert channel.answer_ask(frame["request_id"], "allow_always") == {"status": "recorded", "decision": "allow_always"}
    assert (await task).kind == "allow_always"


async def test_a_private_always_in_a_tainted_class_from_a_marked_agent_keeps_the_second_tap(tmp_path):
    channel, task, frame = await _asking(tmp_path, _ask(), private=True, marked=True)
    assert [o["kind"] for o in frame["options"] if o["confirm_required"]] == ["allow_always", "reject_always"]
    first = channel.answer_ask(frame["request_id"], "allow_always")
    assert first["status"] == "confirm_required" and first["detail"] == PRIVATE_TAINTED_CONFIRM
    assert channel.answer_ask(frame["request_id"], "allow_always", first["confirm_token"])["status"] == "recorded"
    await task


@pytest.mark.parametrize("klass, marked", [("shell-new", True), ("shell-destructive", False)])
async def test_a_private_always_needs_both_a_tainted_class_and_a_marked_agent_for_the_second_tap(tmp_path, klass, marked):
    channel, task, frame = await _asking(tmp_path, _ask(klass=klass), private=True, marked=marked)
    assert channel.answer_ask(frame["request_id"], "allow_always")["status"] == "recorded"
    await task


async def test_a_normal_chat_keeps_two_taps_and_the_permanent_text(tmp_path):
    channel, task, frame = await _asking(tmp_path, _ask(), private=False)
    first = channel.answer_ask(frame["request_id"], "allow_always")
    assert first["status"] == "confirm_required" and "permanent" in first["detail"]
    task.cancel()


async def test_a_chat_that_is_not_known_private_never_gets_the_one_tap(tmp_path):
    channel = await _channel(tmp_path, private=True)
    assert channel.run is None
    client = channel.attach_client()
    task = asyncio.ensure_future(channel.ask_permission(_ask()))
    await asyncio.sleep(0)
    frame = next(row[1] for row in _frames(client) if row[0] == "BlockingAsk")
    assert channel.answer_ask(frame["request_id"], "allow_always")["status"] == "confirm_required"
    task.cancel()


def test_a_private_push_keeps_the_kind_title_and_says_nothing_of_the_chat():
    policy = push.PushPolicy()
    policy.turn_started("s1", now=0)
    done = policy.turn_finished("s1", clients_attached=0, now=999, summary="the secret plan", verdict="pass", private=True)
    assert done.title == "reply ready" and done.body == push.PRIVATE_PUSH_BODY
    assert done.tag == "turn_finished:private"
    assert done.data["session_id"] is None and done.data["url"] == "/?" and "s1" not in json.dumps(done.data)
    ask = policy.needs_you("s1", kind="blocking", request_id="r1", detail="rm secret.txt", private=True)
    assert ask.title == "permission needed" and ask.body == push.PRIVATE_PUSH_BODY
    assert ask.tag == "needs_you:private" and ask.data["session_id"] is None
    assert ask.data["url"] == "/?ask=r1" and ask.data["request_id"] == "r1"
    parked = policy.needs_you("s1", kind="parked", pending_id=4, detail="write x", private=True)
    assert parked.data["url"] == "/?pending=4" and parked.badge == 2
    assert "s1" not in json.dumps([ask.data, parked.data]) and "secret" not in ask.body


def test_a_normal_push_is_unchanged():
    policy = push.PushPolicy()
    policy.turn_started("s1", now=0)
    done = policy.turn_finished("s1", clients_attached=0, now=999, summary="fixed the parser", verdict="pass")
    assert done.body == "fixed the parser" and done.tag == "turn_finished:s1"
    assert done.data == {"class": "turn_finished", "session_id": "s1", "url": "/?session=s1"}
    ask = policy.needs_you("s1", kind="blocking", request_id="r1", detail="rm build")
    assert ask.body == "rm build" and ask.tag == "needs_you:s1" and ask.data["url"] == "/?session=s1&ask=r1"


async def _long_turn(bus, session_id="s1"):
    await bus.publish(TurnStarted(session_id=session_id, agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    await bus.publish(TurnCompleted(session_id=session_id, agent_id="orchestrator", iterations=1, elapsed_tokens=9,
                                    duration_seconds=push.TURN_PUSH_MIN_S + 30, summary="the secret plan"))


@pytest.mark.parametrize("run", [None, SimpleNamespace(private=True)], ids=["no run", "private run"])
async def test_a_turn_finishing_in_a_chat_that_is_not_known_normal_pushes_the_private_form(tmp_path, run):
    bus, channel, recorder = await _pushable(tmp_path)
    channel.run = run
    await _long_turn(bus)
    await channel.flush_push()
    assert [(p.body, p.tag, p.data["session_id"]) for p in recorder.sent] == [
        (push.PRIVATE_PUSH_BODY, "turn_finished:private", None)]


async def test_a_turn_that_finishes_after_the_chat_was_forgotten_pushes_the_private_form(tmp_path):
    bus, channel, recorder = await _pushable(tmp_path)
    await bus.publish(TurnStarted(session_id="s1", agent_id="orchestrator", task_summary="t", budget={"max_actions": 5}))
    channel.forget_chat()
    await bus.publish(TurnCompleted(session_id="s1", agent_id="orchestrator", iterations=1, elapsed_tokens=9,
                                    duration_seconds=push.TURN_PUSH_MIN_S + 30, summary="the secret plan"))
    await channel.flush_push()
    assert [(p.body, p.data["session_id"]) for p in recorder.sent] == [(push.PRIVATE_PUSH_BODY, None)]


async def test_an_event_of_another_session_pushes_the_private_form(tmp_path):
    bus, channel, recorder = await _pushable(tmp_path)
    await _long_turn(bus, session_id="old")
    await channel.flush_push()
    assert [(p.body, p.data["session_id"]) for p in recorder.sent] == [(push.PRIVATE_PUSH_BODY, None)]


async def test_a_normal_chats_turn_push_keeps_its_summary_and_session(tmp_path):
    bus, channel, recorder = await _pushable(tmp_path)
    await _long_turn(bus)
    await channel.flush_push()
    assert [(p.body, p.data["session_id"]) for p in recorder.sent] == [("the secret plan", "s1")]


async def test_a_blocking_ask_decides_from_the_bound_run(tmp_path):
    bus, channel, recorder = await _pushable(tmp_path)
    channel.run = SimpleNamespace(private=True)
    task = asyncio.ensure_future(channel.ask_permission(_ask()))
    await asyncio.sleep(0)
    await channel.flush_push()
    assert [(p.body, p.tag, p.data["session_id"]) for p in recorder.sent] == [
        (push.PRIVATE_PUSH_BODY, "needs_you:private", None)]
    task.cancel()


@pytest.fixture
def world(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "config.yaml").write_text("{}")
    ws = tmp_path / "proj"
    ws.mkdir()
    run = begin(cfg, private=True)
    run.turn = 1
    yield SimpleNamespace(ws=ws, run=run, tmp=tmp_path)
    end(run)


def _loop(world, run):
    gate = SimpleNamespace(run=run, owner_agent_id="root", boundary=world.ws, workspace=world.ws)
    loop = object.__new__(AgentLoop)
    loop._gate, loop._config = gate, SimpleNamespace(name="root")
    return loop


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")
def test_kept_names_lists_what_stays_project_relative_else_absolute_and_caps_the_list(world):
    outside = world.tmp / "elsewhere.txt"
    paths = [world.ws / f"f{i}.txt" for i in range(7)] + [outside]
    for path in paths:
        path.write_text("x")
    note_written(world.run, paths, shown=True)
    names = kept_names(world.run, world.ws)
    assert names == tuple(f"f{i}.txt" for i in range(6)) + ("and 2 more",)
    assert kept_names(world.run, world.ws / "nowhere")[0].startswith(str(world.ws))
    note_written(world.run, [outside], shown=True)
    paths[0].unlink()
    assert "f0.txt" not in kept_names(world.run, world.ws)


@pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")
def test_only_the_owner_loop_of_a_private_run_reports_kept(world):
    (world.ws / "a.txt").write_text("x")
    note_written(world.run, [world.ws / "a.txt"], shown=True)
    assert _loop(world, world.run)._kept() == ("a.txt",)
    assert _loop(world, SimpleNamespace(private=False))._kept() == ()
    assert _loop(world, None)._kept() == ()
    sub = _loop(world, world.run)
    sub._config = SimpleNamespace(name="child")
    assert sub._kept() == ()


async def test_the_last_replys_kept_list_rides_hello_and_goes_with_the_chat(tmp_path):
    bus, channel, _ = await _pushable(tmp_path)
    await bus.publish(TaskComplete(session_id="s1", agent_id="orchestrator", success=True, summary="done",
                                   duration_seconds=1.0, iterations=1, kept=("a.txt", "b.txt")))
    assert (await channel.hello(channel.attach_client(), None)).kept == ("a.txt", "b.txt")
    channel.forget_chat()
    assert (await channel.hello(channel.attach_client(), None)).kept == ()


async def test_a_subagents_reply_does_not_set_the_kept_list(tmp_path):
    bus, channel, _ = await _pushable(tmp_path)
    await bus.publish(TaskComplete(session_id="s1", agent_id="child", parent_id="orchestrator", success=True,
                                   summary="done", duration_seconds=1.0, iterations=1, kept=("z.txt",)))
    assert channel.kept == ()
