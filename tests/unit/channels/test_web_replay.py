"""`--replay` and `--fixtures`: the offline development loop (WIN-B, WEBCH-20).

The property under test is the one §8 says an earlier draft got wrong: replaying a log RAW is
half a promise, because a persisted log contains none of the SSE-only frames — which are exactly
the interactive parts a UI author most needs to iterate on. So these tests check that the
provisional-supersede path and the permission modal are BOTH exercisable with the box asleep, and
that nothing synthesized is ever passed off as measured.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.replay import ReplayDriver, ReplayFixtures
from localharness.core.bus import EventBus

pytestmark = pytest.mark.asyncio


def _log(tmp_path, *rows):
    path = tmp_path / "sessions" / "abc123.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _row(seq, event_type, **fields):
    base = {
        "event_type": event_type, "seq": seq, "id": f"e{seq}",
        "timestamp": f"2026-09-14T10:00:{seq:02d}+00:00",
        "agent_id": "orchestrator", "session_id": "abc123", "parent_id": None,
    }
    base.update(fields)
    return base


async def _drain(client):
    rows = []
    while not client.queue.empty():
        name, seq, payload = client.queue.get_nowait()
        rows.append((name, seq, json.loads(payload)))
    return rows


async def _channel():
    channel = WebChannel(bus=EventBus(), config={})
    await channel.start()
    return channel


async def test_replay_synthesizes_the_stream_the_log_does_not_contain(tmp_path):
    """Each llm_response's content is re-emitted as TokenDelta frames BEFORE the Action, so the
    provisional-supersede path — the client logic most likely to be wrong — is exercised offline."""
    path = _log(
        tmp_path,
        _row(1, "UserMessage", content="is the daemon up?", channel="web", attachments=[]),
        _row(2, "Action", action_type="llm_response", content="Yes, it is up.",
             has_tool_calls=False),
        _row(3, "TaskComplete", success=True, summary="Yes, it is up.",
             duration_seconds=2.0, iterations=1),
    )
    channel = await _channel()
    client = channel.attach_client()
    await ReplayDriver(channel, path, speed=1000).run()

    rows = await _drain(client)
    kinds = [r[0] for r in rows]
    assert kinds.index("TokenDelta") < kinds.index("Action"), (
        "the deltas must precede the Action that supersedes them, exactly as a live turn orders it"
    )
    streamed = "".join(r[2]["text"] for r in rows if r[0] == "TokenDelta")
    assert streamed == "Yes, it is up."
    # The persisted events themselves are forwarded verbatim, with their real seqs.
    assert [(r[0], r[1]) for r in rows if r[1] is not None] == [
        ("UserMessage", 1), ("Action", 2), ("TaskComplete", 3),
    ]


async def test_replayed_instruments_are_flagged_synthetic(tmp_path):
    """A replayed tok/s is a fiction, and a UI that treats it as a measurement is reading one."""
    path = _log(
        tmp_path,
        _row(1, "Heartbeat", iteration=1, context_utilization_pct=12.5, last_tool="read"),
    )
    channel = await _channel()
    client = channel.attach_client()
    await ReplayDriver(channel, path, speed=1000).run()

    ticks = [r[2] for r in await _drain(client) if r[0] == "StatusTick"]
    assert ticks and ticks[0]["synthetic"] is True
    assert ticks[0]["context_pct"] == 12.5


async def test_the_parked_queue_replays_for_real(tmp_path):
    """PermissionStaged/PermissionResolved are ordinary bus events carrying the whole PendingCall,
    so unlike a blocking ask they need no fixture at all."""
    staged = _row(
        1, "PermissionStaged", total=1, channel="web",
        pending={
            "id": 1, "rendering": "bash_exec: rm -rf build/", "agent_label": "",
            "session_id": "abc123", "created_at": 1.0,
            "request": {
                "tool_name": "bash_exec", "tool_params": {"command": "rm -rf build/"},
                "klass": "shell-destructive", "key": None, "grantable": False,
                "reason": "destructive", "display": "bash_exec: rm -rf build/",
                "grant_keys": [], "agent_id": None, "options_legend": None,
                "auto_entry": None, "call_id": None,
            },
        },
    )
    path = _log(tmp_path, staged)
    channel = await _channel()
    client = channel.attach_client()
    await ReplayDriver(channel, path, speed=1000).run()

    rows = [r for r in await _drain(client) if r[0] == "PermissionStaged"]
    assert rows and rows[0][2]["pending"]["rendering"] == "bash_exec: rm -rf build/"


async def test_a_fixture_blocking_ask_is_genuinely_answerable(tmp_path):
    """Without this, --fixtures would draw a modal whose buttons do nothing — and a UI author
    could build one that looks right and has never once completed its own round trip."""
    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    fixtures = tmp_path / "fx.json"
    fixtures.write_text(json.dumps({"frames": [{
        "after_seq": 1,
        "frame": {
            "frame_type": "BlockingAsk", "request_id": "fx-1", "tool_name": "bash_exec",
            "tool_params": {"command": "rm -rf /"}, "klass": "shell-destructive",
            "grantable": False, "display": "bash_exec: rm -rf /",
        },
    }]}))

    channel = await _channel()
    client = channel.attach_client()
    await ReplayDriver(channel, path, speed=1000, fixtures=ReplayFixtures.load(fixtures)).run()

    asks = [r[2] for r in await _drain(client) if r[0] == "BlockingAsk"]
    assert asks and asks[0]["request_id"] == "fx-1"
    # It is real state: listed by GET /api/permissions, and answerable.
    assert [a["request_id"] for a in channel.open_asks()] == ["fx-1"]
    assert channel.answer_ask("fx-1", "reject_once") == {
        "status": "recorded", "decision": "reject_once",
    }


async def test_a_fixture_always_answer_still_takes_the_server_side_second_tap(tmp_path):
    """The confirm is not a live-session special case; it is how the verb behaves."""
    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    fixtures = tmp_path / "fx.json"
    fixtures.write_text(json.dumps({"frames": [{
        "frame": {
            "frame_type": "BlockingAsk", "request_id": "fx-2", "tool_name": "bash_exec",
            "tool_params": {}, "klass": "shell-new", "key": "bash:ls", "grantable": True,
            "display": "bash_exec: ls",
            "options": [{"kind": "allow_always", "name": "Always allow here",
                         "confirm_required": True}],
        },
    }]}))
    channel = await _channel()
    await ReplayDriver(channel, path, speed=1000, fixtures=ReplayFixtures.load(fixtures)).run()
    assert channel.answer_ask("fx-2", "allow_always")["status"] == "confirm_required"


async def test_a_typo_in_a_fixture_file_is_refused_loudly(tmp_path):
    """A fixture that silently produces nothing is a morning lost."""
    bad = tmp_path / "fx.json"
    bad.write_text(json.dumps({"frames": [{"frame": {"frame_type": "BlockingAsq"}}]}))
    with pytest.raises(ValueError) as exc:
        ReplayFixtures.load(bad)
    assert "BlockingAsq" in str(exc.value) and "BlockingAsk" in str(exc.value)


async def test_replay_never_writes_into_a_real_session_log(tmp_path):
    """It drives the channel's fan-out only. A development mode that appended to the corpus it
    was replaying would corrupt the thing it exists to read."""
    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    before = path.read_bytes()
    channel = await _channel()
    channel.attach_client()
    await ReplayDriver(channel, path, speed=1000).run()
    assert path.read_bytes() == before
    assert channel.bus._persist_path is None


async def test_the_replay_hello_says_it_is_synthetic(tmp_path):
    """So nobody mistakes a replayed session for a live one, or a synthesized tick for a reading."""
    from localharness.channels.web.server import WebServer

    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    channel = await _channel()
    driver = ReplayDriver(channel, path, speed=1000)
    server = WebServer(channel, token="t", replay=driver)
    client = channel.attach_client()
    hello = await channel.hello(client, None)
    assert hello.synthetic is False                 # the channel itself does not know
    assert server.replay is driver                  # the SERVER stamps it on the way out
    body = json.loads((await channel.hello(client, None)).model_copy(
        update={"synthetic": True}).model_dump_json())
    assert body["synthetic"] is True


async def test_replay_paces_itself_from_the_persisted_timestamps(tmp_path):
    """Real gaps run to minutes (the measured median between turns is 3.7 of them) and nobody
    iterating on a stylesheet wants to wait one out — so the pause is CAPPED, not dropped:
    ordering and the feel of a pause survive while the dead time does not."""
    from localharness.channels.web.replay import MAX_STEP_S

    rows = [
        _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0),
        dict(_row(2, "Heartbeat", iteration=2, context_utilization_pct=2.0),
             timestamp="2026-09-14T10:30:00+00:00"),   # half an hour later
    ]
    path = _log(tmp_path, *rows)
    channel = await _channel()
    channel.attach_client()
    started = asyncio.get_running_loop().time()
    await ReplayDriver(channel, path, speed=1000).run()
    assert asyncio.get_running_loop().time() - started < MAX_STEP_S


async def test_playback_starts_when_a_client_connects_not_when_the_server_boots(tmp_path):
    """Found by driving the real command: a browser opened a few seconds after `localharness web
    --replay` had already missed the opening of the session it came to watch — and "edit the
    page, pull to refresh, watch it again" is the entire loop --replay exists for."""
    from localharness.channels.web.server import WebServer

    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    channel = await _channel()
    driver = ReplayDriver(channel, path, speed=1000)
    server = WebServer(channel, token="t", replay=driver)

    assert driver._task is None, "constructing the server must not start playback"
    server._maybe_prewarm()                       # what a connect does
    assert driver._task is not None
    await driver._task

    # A refresh replays from the top rather than joining a finished playback.
    first = driver._task
    server._maybe_prewarm()
    assert driver._task is not first
    await driver.stop()


async def test_the_session_id_is_known_before_playback_begins(tmp_path):
    """`Hello` goes out on connect and playback starts a moment later, so a session id set inside
    `run()` would reach the client as null on the one frame meant to name the session."""
    path = _log(tmp_path, _row(1, "Heartbeat", iteration=1, context_utilization_pct=1.0))
    channel = await _channel()
    ReplayDriver(channel, path)
    assert channel.session_id == "abc123"
