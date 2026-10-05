"""MEMP-01, WEBP-03: in a real session the memory plugin occupies the ONE slot object that reaches
both the AgentLoop and the mobile channel; memory off leaves it empty; a replacement memory plugin
occupies it instead; two memory plugins are both refused and it stays empty. A failed plugin
substrate now leaves the slot EMPTY — memory is a plugin, so no plugin lifecycle means no memory and
no phone memory screen (46's transitional occupant, seated by start_cmd outside the lifecycle, is
gone). Driven through the real `_start_async` with only the external boundaries stubbed."""
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
    from localharness.channels.mobile.channel import MobileChannel
    from localharness.cli.start_cmd import _start_async
    from localharness.core.bus import EventBus

    seen: dict[str, Any] = {}

    async def _repl(self):
        seen["channel_slot"] = ch.memory_slot()
        seen["loop_slot"] = ch._agent_loop._memory_slot
        seen["browse"] = ch.memory_slot().browse() if ch.memory_slot() is not None else None

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=_repl)
    if config_extra:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write(config_extra)
    ch = MobileChannel(bus=EventBus(), config={})
    await _start_async(None, False, False, str(tmp_path), channel_mode="mobile", mobile_channel=ch)
    seen["after"] = ch.memory_slot()
    return seen


async def test_a_web_session_binds_the_memory_plugin_as_the_occupant(tmp_path, monkeypatch):
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert isinstance(slot, MemorySlot) and slot.occupied and slot.occupant_name == "memory"
    assert type(slot._occupant).__name__ == "MemoryPlugin"
    assert isinstance(seen["browse"], StoreBrowse)
    assert slot is seen["loop_slot"], "one slot object reaches the loop AND the channel"


async def test_memory_off_binds_an_empty_slot(tmp_path, monkeypatch):
    seen = await _drive(tmp_path, monkeypatch, config_extra="org:\n  memory_enabled: false\n")
    assert seen["channel_slot"].occupied is False
    assert seen["channel_slot"] is seen["loop_slot"]


async def test_a_replacement_memory_plugin_occupies_the_slot(tmp_path, monkeypatch):
    from localharness.cli.mobile_plugin import MobilePlugin

    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS",
                        (_memory_plugin("recall"), MobilePlugin))
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert slot.occupied and slot.occupant_name == "recall"
    assert not isinstance(seen["browse"], StoreBrowse)
    assert slot is seen["loop_slot"]


async def test_two_memory_plugins_are_refused_and_the_slot_stays_empty(tmp_path, monkeypatch):
    from localharness.cli.mobile_plugin import MobilePlugin
    from localharness.memory.plugin import MemoryPlugin

    monkeypatch.setattr("localharness.plugins.builtin.BUILTIN_PLUGINS",
                        (MobilePlugin, MemoryPlugin, _memory_plugin("recall")))
    seen = await _drive(tmp_path, monkeypatch)
    assert seen["channel_slot"].occupied is False
    assert seen["browse"] is None
    assert seen["channel_slot"] is seen["loop_slot"]


async def test_substrate_failure_leaves_the_slot_empty(tmp_path, monkeypatch):
    """Inverted from 46-06 by design: the memory plugin cannot start without the lifecycle, so the
    slot is empty and the phone's memory screen is absent — the session itself still runs."""
    async def boom(*_a, **_k):
        raise RuntimeError("substrate down")

    monkeypatch.setattr("localharness.plugins.lifecycle.start_plugins", boom)
    seen = await _drive(tmp_path, monkeypatch)
    slot = seen["channel_slot"]
    assert isinstance(slot, MemorySlot) and not slot.occupied
    assert seen["browse"] is None
    assert slot is seen["loop_slot"]
