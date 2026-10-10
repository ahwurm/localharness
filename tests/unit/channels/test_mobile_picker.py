"""`MobileChannel.run_picker`: the phone's picker over the existing BlockingAsk option frame (D-33).

No new wire frame and no page change: four tappable levels answered through the same
`answer_ask` / `GET /api/permissions` path a permission question uses. The properties that matter
are that nothing but an offered level can answer it, and that an unanswered picker never holds
the REPL for longer than its deadline.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from localharness.channels.mobile import channel as channel_mod
from localharness.channels.mobile.channel import (
    ASK_FALLBACK_DECISION,
    PICKER_ASK_TEXT,
    PICKER_EXPIRED_DECISION,
    PICKER_KLASS,
    MobileChannel,
)
from localharness.core.bus import EventBus

pytestmark = pytest.mark.asyncio

LEVELS = ("off", "low", "high", "max")


async def _channel() -> MobileChannel:
    channel = MobileChannel(bus=EventBus(), config={})
    await channel.start()
    return channel


def _drain(client) -> list[tuple[str, object, dict]]:
    rows = []
    while not client.queue.empty():
        name, seq, payload = client.queue.get_nowait()
        rows.append((name, seq, json.loads(payload)))
    return rows


async def _open(channel, client, *args, **kw):
    """Start run_picker as a task and return (task, the BlockingAsk payload)."""
    task = asyncio.ensure_future(channel.run_picker(*args, **kw))
    await asyncio.sleep(0)
    frames = [r for r in _drain(client) if r[0] == "BlockingAsk"]
    assert len(frames) == 1
    return task, frames[0][2]


async def test_the_picker_is_one_blocking_ask_with_a_button_per_level():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")

    assert [(o["kind"], o["name"]) for o in frame["options"]] == [(l, l) for l in LEVELS]
    assert all(o["confirm_required"] is False for o in frame["options"])
    assert frame["options_legend"] == "current: high"
    assert frame["tool_name"] == "" and frame["tool_params"] == {}
    assert frame["klass"] == PICKER_KLASS and frame["grantable"] is False
    assert frame["display"] == PICKER_ASK_TEXT == frame["reason"]
    assert "no tool call" in PICKER_ASK_TEXT
    assert [a["request_id"] for a in channel.open_asks()] == [frame["request_id"]]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_tapping_a_level_resolves_the_picker_to_it():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")
    rid = frame["request_id"]

    assert channel.answer_ask(rid, "max") == {"status": "recorded", "decision": "max"}
    assert await task == "max"
    assert channel.open_asks() == []
    assert channel.answer_ask(rid, "low")["status"] == "unknown"


async def test_a_permission_kind_is_refused_and_the_picker_stays_open():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")
    rid = frame["request_id"]

    for kind in ("allow_always", "allow_once", "reject_always", "bogus", ""):
        assert channel.answer_ask(rid, kind)["status"] == "invalid"
    assert not task.done()
    assert [a["request_id"] for a in channel.open_asks()] == [rid]
    assert channel.answer_ask(rid, "off")["status"] == "recorded"
    assert await task == "off"


async def test_a_prompt_replaces_the_text_and_is_sanitised():
    channel = await _channel()
    client = channel.attach_client()
    prompt = "Reasoning effort (/effort). This question holds no tool call.\x1b[31m"
    task, frame = await _open(channel, client, LEVELS, None, prompt=prompt)
    assert frame["display"] == frame["reason"]
    assert frame["display"].startswith("Reasoning effort (/effort).")
    assert "\x1b" not in frame["display"]
    assert frame["options_legend"] == "current: none"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_an_unanswered_picker_expires_unchanged(monkeypatch):
    monkeypatch.setattr(channel_mod, "PICKER_ASK_TIMEOUT_S", 0.05)
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")

    assert await asyncio.wait_for(task, 5.0) is None
    expired = [r for r in _drain(client) if r[0] == "AskExpired"]
    assert len(expired) == 1
    assert expired[0][2]["decision"] == PICKER_EXPIRED_DECISION == "unchanged"
    assert expired[0][2]["request_id"] == frame["request_id"]
    assert channel.open_asks() == []


async def test_cancelling_the_picker_announces_the_expiry_and_reraises():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    expired = [r for r in _drain(client) if r[0] == "AskExpired"]
    assert len(expired) == 1
    assert expired[0][2]["decision"] == PICKER_EXPIRED_DECISION
    assert expired[0][2]["request_id"] == frame["request_id"]
    assert channel.open_asks() == []


async def test_shutdown_settles_the_picker_with_no_level():
    channel = await _channel()
    client = channel.attach_client()
    task, _ = await _open(channel, client, LEVELS, "high")

    await channel.stop()
    assert ASK_FALLBACK_DECISION not in LEVELS
    assert await asyncio.wait_for(task, 5.0) is None


async def test_the_picker_needs_an_attached_client():
    channel = await _channel()
    assert channel.can_run_picker() is False
    assert await channel.run_picker(LEVELS, "high") is None
    assert channel.open_asks() == []

    client = channel.attach_client()
    assert channel.can_run_picker() is True
    channel.detach_client(client)
    assert channel.can_run_picker() is False


async def test_no_push_notification_fires_for_the_picker():
    channel = await _channel()
    rung = []

    class _Policy:
        def needs_you(self, *a, **kw):
            rung.append((a, kw))

        def resolved(self, *a, **kw):
            return None

    channel._push = SimpleNamespace(policy=_Policy())
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")
    channel.answer_ask(frame["request_id"], "low")
    assert await task == "low"
    assert rung == []


async def test_an_empty_option_list_raises():
    channel = await _channel()
    channel.attach_client()
    with pytest.raises(ValueError, match="picker needs at least one option"):
        await channel.run_picker([], None)


async def test_non_ascii_levels_round_trip():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, ("off", "máx ✓"), "off")
    assert [o["kind"] for o in frame["options"]] == ["off", "máx ✓"]
    channel.answer_ask(frame["request_id"], "máx ✓")
    assert await task == "máx ✓"


async def test_a_second_answer_after_the_first_is_not_recorded_twice():
    channel = await _channel()
    client = channel.attach_client()
    task, frame = await _open(channel, client, LEVELS, "high")
    rid = frame["request_id"]
    channel.answer_ask(rid, "low")
    assert channel.answer_ask(rid, "max")["status"] in {"unknown", "already_answered"}
    assert await task == "low"
