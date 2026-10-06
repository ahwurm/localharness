"""The terminal's picture intake: /image <path>, a dropped path, the clipboard, and the hand-off
of staged + channel-carried pictures into run_turn(images=...)."""
from __future__ import annotations

import pytest

from localharness.cli import repl as repl_mod
from localharness.core.content import image_part
from tests.unit.test_image_content import png_bytes
from tests.unit.test_repl_input_box import FakeBoxChannel, _repl


@pytest.fixture
def shot(tmp_path):
    p = tmp_path / "Screenshot 2026-10-03 002204.png"
    p.write_bytes(png_bytes(265, 51))
    return p


def _config(repl, cap: int = 4096):
    repl._agent._config.context.max_image_tokens = cap


async def test_slash_image_stages_a_file_and_the_next_line_carries_it(shot):
    repl, channel, agent = _repl()
    _config(repl)
    assert await repl._handle_slash(f'/image "{shot}"') is True
    receipt = channel.sent[-1][0]
    assert receipt.startswith("📎 Screenshot 2026-10-03 002204.png 265×51 png · ~78 tokens — attached to your next message")
    assert len(repl._staged_images) == 1

    await repl._start_user_turn("what is wrong with this header?")
    kwargs = agent.run_turn.call_args.kwargs
    assert kwargs["task"] == "what is wrong with this header?"
    assert [p["_lh"]["name"] for p in kwargs["images"]] == ["Screenshot 2026-10-03 002204.png"]
    assert repl._staged_images == []  # consumed by that turn
    event = repl._bus.publish.call_args.args[0]
    assert event.content == "what is wrong with this header?"
    assert event.attachments == ["Screenshot 2026-10-03 002204.png 265×51 png · ~78 tokens"]


async def test_a_dropped_path_is_an_attachment_not_a_sentence(shot):
    repl, channel, agent = _repl()
    _config(repl)
    # What a terminal pastes for a dragged file: the path, shell-escaped because of the spaces.
    line = str(shot).replace(" ", "\\ ")
    assert repl_mod._dropped_image_path(line) == str(shot)
    assert repl_mod._dropped_image_path(f"look at {shot}") is None
    assert repl_mod._dropped_image_path("/image notes.png") is None
    assert await repl._dispatch_input(line) is None
    agent.run_turn.assert_not_called()
    assert len(repl._staged_images) == 1 and "📎" in channel.sent[-1][0]


async def test_not_an_image_is_refused_with_the_reason(tmp_path):
    repl, channel, _ = _repl()
    _config(repl)
    fake = tmp_path / "shot.png"
    fake.write_text("not really")
    await repl._handle_slash(f"/image {fake}")
    assert channel.sent[-1][0].startswith("Not attached: not a PNG, JPEG, GIF or WebP image")
    assert repl._staged_images == []
    await repl._handle_slash("/image clear")
    assert channel.sent[-1][0] == "No images staged."


async def test_clipboard_without_a_display_says_so(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr("sys.platform", "linux")
    repl, channel, _ = _repl()
    _config(repl)
    await repl._paste_image()
    text = channel.sent[-1][0]
    assert text.startswith("Not attached: no clipboard is reachable from this session")
    assert "/image <path>" in text


async def test_channel_carried_pictures_are_fitted_and_sent_with_the_line():
    channel = FakeBoxChannel()
    channel.take_images = lambda: [image_part(png_bytes(1920, 1080), name="phone.png")]
    repl, _, agent = _repl(channel=channel)
    _config(repl, cap=4096)
    await repl._start_user_turn("from my phone")
    images = agent.run_turn.call_args.kwargs["images"]
    assert len(images) == 1 and images[0]["_lh"]["width"] == 1920


async def test_an_unfittable_picture_stops_the_line_not_silently(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "PIL", None)
    channel = FakeBoxChannel()
    channel.take_images = lambda: [image_part(png_bytes(3840, 2160), name="4k.png")]
    repl, _, agent = _repl(channel=channel)
    _config(repl, cap=4096)
    assert await repl._start_user_turn("from my phone") is None
    agent.run_turn.assert_not_called()
    assert channel.sent[-1][0].startswith("Not sent: 4k.png 3840×2160 png is ~8,162 tokens; the cap is 4,096")


async def test_a_line_typed_mid_turn_with_a_picture_staged_is_queued_not_nudged(shot):
    from tests.unit.test_repl_input_box import _pending_turn
    repl, channel, agent = _repl()
    _config(repl)
    await repl._handle_slash(f'/image "{shot}"')
    await _pending_turn(repl)
    await repl._route_during_turn("and this one", forced=False)
    agent.push_user_nudge.assert_not_called()
    assert list(repl._fifo) == ["and this one"]
    repl._turn_task.cancel()
