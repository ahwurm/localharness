"""`WebChannel`: the raw-event wire, the transcript contract, and the permission spine.

The permission tests are the ones that matter. The web PRD calls the gate THE load-bearing
design, and the property under test is not "a human can answer" — it is that **nothing except a
human's answer can ever produce an allow**. Each of those tests is written so that flipping
`ASK_FALLBACK_DECISION` to an allow, or deleting the server-side confirm, turns it red.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from localharness.agent.gate_types import Decision, PermissionRequest
from localharness.channels.web.channel import ALWAYS_KINDS, ASK_FALLBACK_DECISION, WebChannel
from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation, TaskComplete, TurnStarted

pytestmark = pytest.mark.asyncio


def _request(**over):
    base = dict(
        tool_name="bash_exec", tool_params={"command": "rm -rf build/"}, klass="shell-destructive",
        key=None, grantable=False, reason="destructive", display="bash_exec: rm -rf build/",
    )
    base.update(over)
    return PermissionRequest(**base)


async def _channel(bus=None) -> WebChannel:
    channel = WebChannel(bus=bus or EventBus(), config={})
    await channel.start()
    return channel


def _drain(client) -> list[tuple[str, object, dict]]:
    rows = []
    while not client.queue.empty():
        name, seq, payload = client.queue.get_nowait()
        rows.append((name, seq, json.loads(payload)))
    return rows


# ------------------------------------------------------------------ the raw wire

async def test_every_event_reaches_the_client_as_the_same_bytes_the_jsonl_gets(tmp_path):
    """WEBCH-03: byte-identical to `model_dump_json()`, with the bus seq as the SSE id."""
    bus = EventBus(persist_path=tmp_path / "events.jsonl")
    channel = await _channel(bus)
    client = channel.attach_client()

    published = await bus.publish(Action(
        agent_id="a", session_id="s", action_type="tool_call",
        tool_name="bash_exec", tool_params={"command": "ls"}, tool_call_id="call-1",
    ))
    name, seq, payload = client.queue.get_nowait()

    assert name == "Action"
    assert seq == published.seq
    assert payload == published.model_dump_json()
    # And the same bytes the session log received.
    logged = (tmp_path / "sessions" / "s.jsonl").read_text().strip()
    assert json.loads(logged) == json.loads(payload)


async def test_tool_call_id_survives_because_the_base_class_path_is_not_used(tmp_path):
    """WEBCH-35: `send_tool_call`'s signature has no `tool_call_id`, so it is never on the path."""
    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    client = channel.attach_client()

    await bus.publish(Action(agent_id="a", session_id="s", action_type="tool_call",
                             tool_name="read", tool_params={}, tool_call_id="pair-me"))
    await bus.publish(Observation(agent_id="a", session_id="s", observation_type="tool_result",
                                  tool_name="read", tool_call_id="pair-me", output="ok"))

    rows = _drain(client)
    assert [r[2]["tool_call_id"] for r in rows] == ["pair-me", "pair-me"]
    # The ABC's methods exist (it declares them) but produce nothing on the wire.
    await channel.send_tool_call("read", {}, "a")
    await channel.send_tool_result("read", "ok", False, "a")
    assert client.queue.empty()


async def test_child_events_are_forwarded_not_swallowed(tmp_path):
    """§4.2: the base adapter DROPS a child TaskComplete. That is a client rendering default —
    a filter at the wire would be invisible and impossible for a UI to undo."""
    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    client = channel.attach_client()

    await bus.publish(TaskComplete(agent_id="child", session_id="s2", parent_id="s",
                                   success=True, summary="child says hi",
                                   duration_seconds=1.0, iterations=1))
    rows = _drain(client)
    assert [r[0] for r in rows] == ["TaskComplete"]
    assert rows[0][2]["summary"] == "child says hi"
    assert rows[0][2]["parent_id"] == "s"


# ------------------------------------------------------------------ the transcript contract

async def test_stream_is_superseded_by_the_action_that_owns_it(tmp_path):
    """§4.2.1: the client never has to INFER the hand-off — StreamClosed names the seq."""
    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    channel.session_id = "s"
    client = channel.attach_client()

    await channel.on_token("hel")
    await channel.on_token("lo")
    action = await bus.publish(Action(agent_id="a", session_id="s", action_type="llm_response",
                                      content="hello", has_tool_calls=False))

    rows = _drain(client)
    kinds = [r[0] for r in rows]
    assert kinds == ["TokenDelta", "TokenDelta", "Action", "StreamClosed"]
    assert [r[2]["text"] for r in rows[:2]] == ["hel", "lo"]
    closed = rows[-1][2]
    assert closed["superseded_by_seq"] == action.seq
    assert closed["stream_id"] == rows[0][2]["stream_id"]


async def test_sse_only_frames_carry_no_sse_id(tmp_path):
    """A frame that stamped its own id would poison `Last-Event-ID` and the persisted cursor,
    because neither would then name something a replay can serve."""
    channel = await _channel()
    client = channel.attach_client()
    await channel.on_token("x")
    channel.show_reasoning = True
    await channel.on_reasoning("thinking")
    for _, seq, _ in _drain(client):
        assert seq is None


async def test_reasoning_respects_the_toggle(tmp_path):
    """WEBCH-07: the sink stays wired so /reasoning works live; the flag decides what ships."""
    channel = await _channel()
    client = channel.attach_client()
    await channel.on_reasoning("secret thought")
    assert client.queue.empty()
    channel.show_reasoning = True
    await channel.on_reasoning("visible thought")
    assert _drain(client)[0][0] == "ReasoningDelta"


# ------------------------------------------------------------------ the permission spine

async def test_a_question_nobody_answers_denies(tmp_path):
    """THE guard. The gate's deadline arrives as a cancel; it must never become an allow.

    Mutation check: setting `ASK_FALLBACK_DECISION` to an allow kind, or swallowing the
    `CancelledError` and returning a Decision, both turn this red.
    """
    channel = await _channel()
    client = channel.attach_client()
    task = asyncio.ensure_future(channel.ask_permission(_request()))
    await asyncio.sleep(0)
    assert _drain(client)[0][0] == "BlockingAsk"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ASK_FALLBACK_DECISION == "reject_once"
    expired = [r for r in _drain(client) if r[0] == "AskExpired"]
    assert expired and expired[0][2]["decision"] == "reject_once"


async def test_shutdown_with_a_question_open_denies_it(tmp_path):
    """A channel that stops mid-question must resolve it, not leave the gate awaiting forever."""
    channel = await _channel()
    task = asyncio.ensure_future(channel.ask_permission(_request()))
    await asyncio.sleep(0)
    await channel.stop()
    decision = await task
    assert isinstance(decision, Decision)
    assert not decision.allowed


async def test_an_ungrantable_question_offers_only_the_once_pair(tmp_path):
    """An "always" button on a class that asks every time by construction would be a lie."""
    channel = await _channel()
    client = channel.attach_client()
    task = asyncio.ensure_future(channel.ask_permission(_request(grantable=False)))
    await asyncio.sleep(0)
    frame = _drain(client)[0][2]
    assert [o["kind"] for o in frame["options"]] == ["allow_once", "reject_once"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_always_needs_a_second_server_checked_tap(tmp_path):
    """WEBCH-10: a SINGLE POST of `allow_always` writes no grant.

    Enforced here, not in the page: the reference page is the owner's disposable half, so
    anything else holding the bearer token — curl, a Shortcut, a future UI, a bug in the page's
    own state machine — would otherwise write a permanent, global, unrevokable grant in one
    request.
    """
    channel = await _channel()
    task = asyncio.ensure_future(
        channel.ask_permission(_request(grantable=True, key="bash:ls", klass="shell-new"))
    )
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))

    first = channel.answer_ask(request_id, "allow_always")
    assert first["status"] == "confirm_required"
    assert not task.done(), "the first tap must not settle the question"

    wrong = channel.answer_ask(request_id, "allow_always", "not-the-token")
    assert wrong["status"] == "confirm_required"
    assert not task.done()
    # ...and a wrong guess must not invalidate the token the real client is holding, or an
    # ordinary double-submit could never complete the confirm.
    assert wrong["confirm_token"] == first["confirm_token"]

    ok = channel.answer_ask(request_id, "allow_always", first["confirm_token"])
    assert ok == {"status": "recorded", "decision": "allow_always"}
    assert (await task).kind == "allow_always"


async def test_allow_once_takes_one_tap(tmp_path):
    """The second tap is only for the two kinds that write durable state."""
    channel = await _channel()
    task = asyncio.ensure_future(channel.ask_permission(_request()))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))
    assert channel.answer_ask(request_id, "allow_once") == {
        "status": "recorded", "decision": "allow_once",
    }
    assert (await task).kind == "allow_once"
    assert ALWAYS_KINDS == {"allow_always", "reject_always"}


async def test_answering_is_idempotent(tmp_path):
    """A retry on a flaky phone connection must not be able to allow something twice."""
    channel = await _channel()
    task = asyncio.ensure_future(channel.ask_permission(_request()))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))
    channel.answer_ask(request_id, "reject_once")
    await task
    again = channel.answer_ask(request_id, "allow_once")
    assert again["status"] in ("already_answered", "unknown")
    assert again.get("decision") in (None, "reject_once")


async def test_an_option_that_was_not_offered_is_refused(tmp_path):
    channel = await _channel()
    task = asyncio.ensure_future(channel.ask_permission(_request(grantable=False)))
    await asyncio.sleep(0)
    request_id = next(iter(channel._open_asks))
    assert channel.answer_ask(request_id, "allow_always")["status"] == "invalid"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_open_asks_are_state_so_a_reconnect_still_sees_them(tmp_path):
    """§5.3: the question is state, not a message — a second device renders it having never
    received the frame."""
    channel = await _channel()
    task = asyncio.ensure_future(channel.ask_permission(_request()))
    await asyncio.sleep(0)
    listed = channel.open_asks()
    assert len(listed) == 1 and listed[0]["tool_name"] == "bash_exec"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert channel.open_asks() == []


# ------------------------------------------------------------------ workspace trust (§5.6)

async def test_workspace_trust_gets_its_own_permanent_sounding_buttons(tmp_path):
    """Drawing it with the generic ungrantable pair would label a PERMANENT decision
    "Allow once" — the opposite of what it does."""
    channel = await _channel()
    client = channel.attach_client()
    task = asyncio.ensure_future(channel.ask_permission(
        _request(klass="workspace-trust", tool_name="workspace", grantable=False)
    ))
    await asyncio.sleep(0)
    options = _drain(client)[0][2]["options"]
    assert [o["kind"] for o in options] == ["allow_always", "reject_once"]
    assert options[0]["name"] == "Trust this workspace"
    # ...and it takes ONE tap: the question already says it is permanent, and it writes no grant.
    assert options[0]["confirm_required"] is False
    request_id = next(iter(channel._open_asks))
    channel.answer_ask(request_id, "allow_always")
    assert (await task).allowed


async def test_the_trust_bridge_refuses_to_ask_into_an_empty_room(tmp_path):
    """With no client attached there is nobody to answer, so asking and then WAITING is how
    bring-up hangs on a question no screen ever showed. Fail closed at once instead."""
    channel = await _channel()
    asker = channel.trust_asker()
    assert await asyncio.to_thread(asker, "trust this place?") is False


async def test_the_trust_bridge_carries_an_answer_across_the_thread_boundary(tmp_path):
    """`resolve_workspace_layer`'s asker is SYNCHRONOUS and runs on a worker thread, so the
    answer has to come back with `run_coroutine_threadsafe` — the bridge ACP documents. Skipping
    it does not fail loudly: it silently makes an outside `.localharness/` invisible forever."""
    channel = await _channel()
    channel.attach_client()
    asker = channel.trust_asker()
    answered = asyncio.ensure_future(asyncio.to_thread(asker, "trust this place?"))

    for _ in range(200):                       # let the worker thread reach the loop
        await asyncio.sleep(0.005)
        if channel._open_asks:
            break
    request_id = next(iter(channel._open_asks))
    channel.answer_ask(request_id, "allow_always")
    assert await asyncio.wait_for(answered, timeout=5) is True


async def test_a_trust_question_nobody_answers_leaves_the_workspace_untrusted(tmp_path):
    """It is the one ask with no gate behind it, so it carries its own deadline (§5.6.3)."""
    import localharness.channels.web.channel as web_channel

    channel = await _channel()
    channel.attach_client()
    original = web_channel.TRUST_ASK_TIMEOUT_S
    web_channel.TRUST_ASK_TIMEOUT_S = 0.05
    try:
        assert await channel._ask_trust("trust this place?") is False
    finally:
        web_channel.TRUST_ASK_TIMEOUT_S = original
    assert channel.open_asks() == [], "the expired question must not keep showing live buttons"


# ------------------------------------------------------------------ status, cancel, gaps

async def test_the_ticker_runs_only_while_a_turn_runs(tmp_path):
    """§5.4: an idle phone holds a silent stream."""
    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    channel.progress_source = lambda: {"phase": "writing", "elapsed": 2.0, "silent": 0.1}
    channel.tps_source = lambda: (42.0, True)
    channel.model_source = lambda: "qwen"
    client = channel.attach_client()

    assert channel._status_task is None
    await bus.publish(TurnStarted(agent_id="a", session_id="s", task_summary="go",
                                  budget={"max_actions": 5}))
    await asyncio.sleep(0.05)
    assert channel._status_task is not None and not channel._status_task.done()

    ticks = [r[2] for r in _drain(client) if r[0] == "StatusTick"]
    assert ticks and ticks[0]["tps"] == 42.0 and ticks[0]["tps_verified"] is True
    assert ticks[0]["model"] == "qwen" and ticks[0]["phase"] == "writing"

    channel._turn_running = False
    await channel._stop_status_ticker()
    assert channel._status_task is None


async def test_a_broken_instrument_reports_nothing_rather_than_raising(tmp_path):
    channel = await _channel()

    def _boom():
        raise RuntimeError("the client went away")

    channel.progress_source = _boom
    channel.tps_source = _boom
    channel.model_source = _boom
    frame = channel.status_frame()
    assert frame.phase == "waiting" and frame.tps is None and frame.model is None


async def test_cancel_emits_the_frame_the_loop_never_publishes(tmp_path):
    """WEBCH-12: a cancelled turn publishes NEITHER TurnCompleted NOR TurnFailed, so without
    this frame the phone shows a turn that never ends."""
    channel = await _channel()
    channel._turn_running = True
    client = channel.attach_client()
    assert await channel.cancel_turn() is False  # no resolver installed yet — honest False

    async def _cancel() -> bool:
        return True

    channel._cancel_resolver = _cancel
    assert await channel.cancel_turn() is True
    assert [r[0] for r in _drain(client)] == ["TurnCancelled"]
    assert channel._turn_running is False


async def test_a_swallowed_persist_failure_becomes_a_visible_gap(tmp_path):
    """WEBCH-33: `_append_jsonl` logs and returns, and `publish()` delivers anyway — so an event
    can be live-visible and permanently absent from the log a reconnect replays from."""
    channel = await _channel()
    channel._forwarded_max["s"] = 42
    assert channel.gap_against_log("s", 42) is None
    gap = channel.gap_against_log("s", 39)
    assert gap is not None and (gap.from_seq, gap.to_seq) == (40, 42)
    assert channel.gap_against_log("unknown-session", None) is None


async def test_a_client_that_falls_behind_is_told_rather_than_grown(tmp_path):
    """Overflow is VISIBLE. An unbounded queue would let a sleeping phone pin the session in
    memory; a silent drop would let it miss the answer and never know."""
    from localharness.channels.web.channel import CLIENT_QUEUE_MAX_FRAMES
    from localharness.channels.web.protocol import Notice

    channel = await _channel()
    client = channel.attach_client()
    for _ in range(CLIENT_QUEUE_MAX_FRAMES + 5):
        channel.push(Notice(text="filler"))
    assert client.lagged is True
    rows = _drain(client)
    assert rows[-1][0] == "Lagged"


# ------------------------------------------------------------------ prose and flags

async def test_slash_command_output_reaches_the_phone(tmp_path):
    """Without this the entire slash-command surface is invisible: the REPL talks to a person
    exclusively through send_message / send_error / send_renderable."""
    channel = await _channel()
    client = channel.attach_client()
    await channel.send_message("mode: guarded", metadata={"style": "system.info"})
    await channel.send_error("unknown command", detail="/help lists commands")
    rows = _drain(client)
    assert [r[0] for r in rows] == ["Notice", "Notice"]
    assert rows[0][2]["style"] == "system.info"
    assert rows[1][2]["style"] == "system.error"
    assert rows[1][2]["detail"] == "/help lists commands"


async def test_a_rich_renderable_survives_as_preformatted_text(tmp_path):
    """`/memory` hands the channel a live rich Tree. Box-drawing only survives in a <pre>."""
    from rich.tree import Tree

    channel = await _channel()
    client = channel.attach_client()
    tree = Tree("memory")
    tree.add("facts")
    await channel.send_renderable(tree)
    frame = _drain(client)[0][2]
    assert frame["preformatted"] is True and frame["style"] == "renderable"
    assert "memory" in frame["text"]


async def test_the_flags_are_the_locked_ones(tmp_path):
    """WEBCH-10. `has_review_surface=True` would silently REMOVE a gate; `ask_holds_dialog=True`
    would let one unanswered question hang a turn forever."""
    assert WebChannel.channel_id == "web"
    assert WebChannel.can_ask is True
    assert WebChannel.ask_holds_dialog is False
    assert WebChannel.has_review_surface is False
    assert WebChannel.has_display_toggles is True


async def test_the_repl_installable_resolvers_are_declared(tmp_path):
    """The REPL installs each handle ONLY if the channel already declares it as None. A channel
    that does not declare one is skipped with NO error — which fails silently, at a tap, later."""
    for name in ("_pending_resolver", "_nudge_resolver", "_cancel_resolver"):
        assert getattr(WebChannel, name, "absent") is None, name


async def test_parked_calls_travel_typed_not_as_prose(tmp_path):
    """The base default would also draw a one-line sentence, which on a typed wire is the same
    fact twice — once badgeable, once not."""
    channel = await _channel()
    client = channel.attach_client()
    await channel.on_permission_staged(object())
    assert client.queue.empty()


async def test_the_channel_subscribes_to_every_declared_event_type(tmp_path):
    """§4.2: "a new event type added anywhere in the harness reaches the phone with no channel
    change". A hand-kept subscription list is the same drift a hand-written DTO per event would
    be, one type at a time — and it WAS drifting: a replayed log carried memory-gate and
    predictive-gate events a live session never forwarded, so the two modes disagreed about what
    the wire contains."""
    from localharness.core.events import EVENT_TYPE_MAP, SurpriseScored

    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    assert len(channel._handles) == len(EVENT_TYPE_MAP)

    client = channel.attach_client()
    await bus.publish(SurpriseScored(
        agent_id="a", session_id="s", tool_call_id="c1", tool_name="read",
        score=0.9, quadrant="surprising_failure",
    ))
    assert [r[0] for r in _drain(client)] == ["SurpriseScored"]


async def test_stop_releases_every_subscription(tmp_path):
    bus = EventBus(persist_path=tmp_path / "e.jsonl")
    channel = await _channel(bus)
    before = bus.subscriber_count
    await channel.stop()
    assert bus.subscriber_count == 0 and before > 0
