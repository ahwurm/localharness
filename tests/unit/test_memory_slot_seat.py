"""46-06 (ROADMAP D3, WEBP-03): start seats the transitional occupant — StoreBrowse over the
session's own store — on the ONE slot object that reaches both the AgentLoop and the web channel.
A real memory plugin already in the slot wins; memory off leaves the slot empty; a failed plugin
substrate still gets a fresh, seated slot so the phone keeps its memory screen. Driven through the
real `_start_async` with only the external boundaries stubbed."""
from __future__ import annotations

from typing import Any

import pytest

from localharness.memory.browse import StoreBrowse
from localharness.plugins.slot import MemorySlot
from tests.unit.test_memory_slot import _memory_plugin
from tests.unit.test_start_cmd import _stub_start_boundaries

pytestmark = pytest.mark.asyncio


async def _drive(tmp_path, monkeypatch, *, config_extra: str = "") -> dict[str, Any]:
    """Run a web session; return what the channel and the root loop held while it was live."""
    from localharness.channels.web.channel import WebChannel
    from localharness.cli.start_cmd import _start_async
    from localharness.core.bus import EventBus

    seen: dict[str, Any] = {}

    async def _repl(self):
        seen["channel_slot"] = ch.memory_slot()
        seen["loop_slot"] = ch._agent_loop._memory_slot

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=_repl)
    if config_extra:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write(config_extra)
    ch = WebChannel(bus=EventBus(), config={})
    await _start_async(None, False, False, str(tmp_path), channel_mode="web", web_channel=ch)
    seen["after"] = ch.memory_slot()
    return seen


async def test_a_web_session_binds_an_occupied_slot_when_memory_is_on(tmp_path, monkeypatch):
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert isinstance(slot, MemorySlot) and slot.occupied and slot.occupant_name == "memory"
    assert isinstance(slot.browse(), StoreBrowse)
    assert slot is seen["loop_slot"], "one slot object reaches the loop AND the channel"


async def test_memory_off_binds_an_empty_slot(tmp_path, monkeypatch):
    seen = await _drive(tmp_path, monkeypatch, config_extra="org:\n  memory_enabled: false\n")
    assert seen["channel_slot"].occupied is False
    assert seen["channel_slot"] is seen["loop_slot"]


async def test_a_seated_memory_plugin_is_not_displaced(tmp_path, monkeypatch):
    from localharness.cli.web_plugin import WebPlugin

    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS",
                        (_memory_plugin("recall"), WebPlugin))
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert slot.occupied and slot.occupant_name == "recall"
    assert not isinstance(slot.browse(), StoreBrowse)
    assert slot is seen["loop_slot"]


async def test_substrate_failure_still_seats(tmp_path, monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("substrate down")

    monkeypatch.setattr("localharness.plugins.lifecycle.start_plugins", boom)
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert slot.occupied and isinstance(slot.browse(), StoreBrowse)
    assert slot is seen["loop_slot"]
