"""The phone's memory screen against the memory PLUGIN occupant, in a real web session (rig half of
phase 47 criterion 2; the device half is the owner's).

A real `_start_async(..., channel_mode="web")` with only the external boundaries stubbed: the slot the
web channel holds is the one the memory plugin occupies through the lifecycle, and the four memory
routes answer over ASGI with the shapes phase 46 pinned (tests/unit/channels/test_web_server.py::
test_memory_list_edit_history_and_forget_roundtrip). Memory off by either key: every route 404s and
`screens.memory` is false, so the phone shows no Memory button.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.core.bus import EventBus
from localharness.memory.sqlite import USER_EDIT_PROVENANCE_PREFIX, MemoryStore
from tests.unit.channels.test_web_server import BEARER, JSON, TOKEN
from tests.unit.test_start_cmd import _stub_start_boundaries

pytestmark = pytest.mark.asyncio
ROW_KEYS = {"name", "value", "status", "confidence", "source", "node_kind", "tags", "updated_at", "provenance"}


async def _seed(config_dir) -> None:
    """A fact in the store the session will open: <config dir>/agents/orchestrator/memory.db."""
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(config_dir), global_base_dir=str(config_dir))
    await store.open()
    try:
        await store.store_fact(key="notes/searxng", value="run it on port 8888",
                               tags=["workaround"], source="remember")
    finally:
        await store.close()


async def _session(tmp_path, monkeypatch, phone, *, config_extra: str = "") -> dict[str, Any]:
    """Run a real web session; `phone(client, ch, seen)` drives the routes while it is live."""
    from localharness.cli.start_cmd import _start_async

    seen: dict[str, Any] = {}

    async def _repl(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=WebServer(ch, token=TOKEN).app),
                                     base_url="http://web.test") as client:
            await phone(client, ch, seen)

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=_repl)

    async def never(self) -> bool:  # the seeded fact is un-embedded: a session-start consolidation
        return False                # pass would load the embedding model mid-test (47-01)
    monkeypatch.setattr("localharness.memory.consolidation.ConsolidationScheduler.should_run", never)
    if config_extra:
        with (tmp_path / "config.yaml").open("a", encoding="utf-8") as f:
            f.write(config_extra)
    ch = WebChannel(bus=EventBus(), config={})
    await _start_async(None, False, False, str(tmp_path), channel_mode="web", web_channel=ch)
    return seen


async def test_phone_memory_routes_answer_through_the_plugin(tmp_path, monkeypatch):
    await _seed(tmp_path)

    async def phone(client, ch, seen):
        slot = ch.memory_slot()
        seen["occupant"] = (slot.occupied, slot.occupant_name == "memory", type(slot._occupant).__name__)
        seen["screens"] = (await client.get("/api/protocol", headers=BEARER)).json()["screens"]
        seen["list"] = await client.get("/api/memory", headers=BEARER)
        seen["search"] = await client.get("/api/memory?q=searxng", headers=BEARER)
        seen["miss"] = await client.get("/api/memory?q=no-such-thing-anywhere", headers=BEARER)
        seen["fact"] = await client.get("/api/memory/fact?name=notes/searxng", headers=BEARER)
        seen["edit"] = await client.post("/api/memory/edit", headers=JSON,
                                         json={"name": "notes/searxng", "content": "run it on port 9999"})
        seen["edited"] = (await client.get("/api/memory/fact?name=notes/searxng", headers=BEARER)).json()
        seen["forget"] = await client.post("/api/memory/forget", headers=JSON, json={"name": "notes/searxng"})
        seen["after_list"] = (await client.get("/api/memory", headers=BEARER)).json()["facts"]
        seen["after_fact"] = (await client.get("/api/memory/fact?name=notes/searxng", headers=BEARER)).json()

    seen = await _session(tmp_path, monkeypatch, phone)

    occupied, named_memory, kind = seen["occupant"]
    assert occupied and named_memory and kind == "MemoryPlugin", seen["occupant"]
    assert seen["screens"]["memory"] is True
    for key in ("list", "search"):
        got = seen[key]
        assert got.status_code == 200, got.text
        (row,) = [r for r in got.json()["facts"] if r["name"] == "notes/searxng"]
        assert set(row) == ROW_KEYS, row
        assert row["value"] == "run it on port 8888" and row["tags"] == ["workaround"]
    assert seen["miss"].json() == {"facts": []}
    fact = seen["fact"].json()
    assert set(fact) == {"name", "fact", "history"} and fact["fact"]["value"] == "run it on port 8888"
    assert seen["edit"].status_code == 200 and seen["edit"].json()["status"] == "edited"
    edited = seen["edited"]["fact"]
    assert edited["value"] == "run it on port 9999" and "workaround" in edited["tags"]
    assert edited["provenance"].startswith(USER_EDIT_PROVENANCE_PREFIX) and edited["provenance"].endswith(";web")
    assert any(f["value"] == "run it on port 8888" and f["status"] == "superseded"
               for f in seen["edited"]["history"])
    assert seen["forget"].status_code == 200 and seen["forget"].json()["status"] == "forgotten"
    assert not any(r["name"] == "notes/searxng" for r in seen["after_list"]), "forgotten, still listed"
    assert seen["after_fact"]["history"], "forget deleted the fact; it must only retire it"

    # Retired, not deleted — read back from the database the session wrote, after it closed.
    store = MemoryStore(agent_id="orchestrator", division_id="default", org_id="default",
                        base_dir=str(tmp_path), global_base_dir=str(tmp_path))
    await store.open()
    try:
        assert await store.get_fact("notes/searxng") is None
        history = await store.get_fact_history("notes/searxng")
        assert {f.value for f in history} == {"run it on port 8888", "run it on port 9999"}
    finally:
        await store.close()


@pytest.mark.parametrize("extra", ["memory:\n  enabled: false\n", "org:\n  memory_enabled: false\n"],
                         ids=["memory.enabled", "org.memory_enabled"])
async def test_phone_memory_absent_when_off(tmp_path, monkeypatch, extra):
    await _seed(tmp_path)  # a store exists on disk; memory off must still serve none of it

    async def phone(client, ch, seen):
        seen["occupied"] = ch.memory_slot().occupied
        seen["screens"] = (await client.get("/api/protocol", headers=BEARER)).json()["screens"]
        seen["routes"] = [
            await client.get("/api/memory", headers=BEARER),
            await client.get("/api/memory?q=searxng", headers=BEARER),
            await client.get("/api/memory/fact?name=notes/searxng", headers=BEARER),
            await client.post("/api/memory/edit", headers=JSON, json={"name": "notes/searxng", "content": "x"}),
            await client.post("/api/memory/forget", headers=JSON, json={"name": "notes/searxng"}),
        ]

    seen = await _session(tmp_path, monkeypatch, phone, config_extra=extra)

    assert seen["occupied"] is False
    assert seen["screens"]["memory"] is False
    for got in seen["routes"]:
        assert got.status_code == 404 and "error" in got.json(), (got.request.url, got.text)
