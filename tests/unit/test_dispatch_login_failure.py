"""A login Discord refuses ends `start --channel discord` with an error instead of waiting forever.

Before, `connect` awaited only `on_ready`; when `client.start` raised (a rejected token, the
Message Content intent off, no gateway), on_ready never fired and the channel start hung."""
from __future__ import annotations

import asyncio

import pytest

from localharness.channels.errors import ChannelStartError
from localharness.core.bus import EventBus
from localharness.dispatch.adapters.discord import DiscordAdapter
from localharness.dispatch.channel import DispatchChannel
from tests.dispatch_support import install_fake_discord

SENTINEL = "tok-SENTINEL-1234567890"


@pytest.mark.asyncio
async def test_a_refused_login_raises_and_never_names_the_token(monkeypatch):
    fake = install_fake_discord(monkeypatch)

    async def refused(self, token):
        raise RuntimeError(f"Improper token has been passed: {token}")

    monkeypatch.setattr(fake.module.Client, "start", refused)
    ch = DispatchChannel(EventBus(), {"adapter": DiscordAdapter(token=SENTINEL), "allow": {"42"}})
    with pytest.raises(ChannelStartError) as exc:
        await asyncio.wait_for(ch.start(), 5)
    assert "Discord login failed" in str(exc.value)
    assert "Improper token" in str(exc.value)
    assert SENTINEL not in str(exc.value) and SENTINEL not in repr(exc.value.__cause__ or "")
    assert ch._handles == [] and not ch._ready.is_set()
    await ch.stop()  # closing after a failed login is safe


@pytest.mark.asyncio
async def test_a_healthy_login_still_returns_once_ready(monkeypatch):
    install_fake_discord(monkeypatch)
    ch = DispatchChannel(EventBus(), {"adapter": DiscordAdapter(token="t"), "allow": {"42"}})
    await asyncio.wait_for(ch.start(), 5)
    assert ch._ready.is_set()
    await ch.stop()
