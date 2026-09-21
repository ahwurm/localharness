"""/api/images/{image_id} — the generated-image endpoint: auth, id shape, confinement,
caching, and the tool<->wire id contract that makes the phone actually render a picture."""
from __future__ import annotations

import httpx
import pytest

from localharness.channels.web.channel import WebChannel
from localharness.channels.web.server import WebServer
from localharness.core.bus import EventBus
from localharness.core.events import IMAGE_ID_RE
from localharness.tools.builtin import generate_image_tool as gi
from localharness.tools.builtin.generate_image_tool import GenerateImageTool, image_artifacts_dir
from localharness.tools.registry import ToolRegistry

pytestmark = pytest.mark.asyncio

TOKEN = "test-token-not-a-real-one"
BEARER = {"Authorization": f"Bearer {TOKEN}"}
GOOD_ID = "img-20260921-204500-abc123"
PNG = b"\x89PNG\r\n\x1a\n-image-endpoint-bytes"


async def _stack(tmp_path, *, with_tool=True):
    bus = EventBus(persist_path=tmp_path / "bus-events.jsonl")
    channel = WebChannel(bus=bus, config={})
    await channel.start()
    registry = ToolRegistry()
    if with_tool:
        await registry.register(GenerateImageTool(workspace_root=str(tmp_path)), scope="global")
    channel.bind_runtime(session_id="s1", agent_id="orchestrator",
                         session_dir=tmp_path / "sessions", tool_registry=registry)
    server = WebServer(channel, token=TOKEN)
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                               base_url="http://web.test")
    return channel, client


def _put_png(tmp_path, image_id=GOOD_ID):
    root = image_artifacts_dir(str(tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{image_id}.png").write_bytes(PNG)


async def test_unauthenticated_is_401(tmp_path):
    channel, client = await _stack(tmp_path)
    try:
        r = await client.get(f"/api/images/{GOOD_ID}")
        assert r.status_code == 401
    finally:
        await client.aclose()
        await channel.stop()


async def test_serves_png_with_immutable_cache(tmp_path):
    _put_png(tmp_path)
    channel, client = await _stack(tmp_path)
    try:
        r = await client.get(f"/api/images/{GOOD_ID}", headers=BEARER)
        assert r.status_code == 200
        assert r.content == PNG
        assert r.headers["content-type"] == "image/png"
        assert "immutable" in r.headers["cache-control"]
    finally:
        await client.aclose()
        await channel.stop()


async def test_module_off_is_404_even_for_a_real_looking_id(tmp_path):
    _put_png(tmp_path)  # file exists, but no generate_image tool is registered
    channel, client = await _stack(tmp_path, with_tool=False)
    try:
        r = await client.get(f"/api/images/{GOOD_ID}", headers=BEARER)
        assert r.status_code == 404
    finally:
        await client.aclose()
        await channel.stop()


@pytest.mark.parametrize("bad", [
    "notanid",
    "img-20260921-204500-ABC123",      # uppercase — not the minted shape
    "img-20260921-204500-abc123x",     # 7th hex char
    "img-2026-09-21-abc123",
    "..%2Fweb%2Ftoken",                # traversal shape: decoded after routing, refused by shape
    f"{GOOD_ID}.png",
])
async def test_malformed_ids_are_404_before_touching_disk(tmp_path, bad):
    _put_png(tmp_path)
    channel, client = await _stack(tmp_path)
    try:
        r = await client.get(f"/api/images/{bad}", headers=BEARER)
        assert r.status_code == 404
    finally:
        await client.aclose()
        await channel.stop()


async def test_wellformed_id_with_no_file_is_404(tmp_path):
    channel, client = await _stack(tmp_path)
    try:
        r = await client.get("/api/images/img-20990101-000000-eeeeee", headers=BEARER)
        assert r.status_code == 404
    finally:
        await client.aclose()
        await channel.stop()


async def test_tool_minted_ids_fullmatch_the_wire_contract(monkeypatch, tmp_path):
    """The cross-module pin: if generate_image ever changes its id shape without moving
    IMAGE_ID_RE, the loop silently stops forwarding image_id and the phone shows no image.
    This is the test that makes that a red bar instead of a mystery."""
    monkeypatch.setenv("LOCALHARNESS_COMFYUI_URL", "http://comfy.test")
    monkeypatch.setattr(gi, "_POLL_S", 0.01)

    class _Resp:
        status_code = 200
        content = PNG
        def json(self):
            return {"prompt_id": "p1"}
        def raise_for_status(self):
            pass

    class _Hist(_Resp):
        def json(self):
            return {"p1": {"status": {"status_str": "success", "completed": True},
                           "outputs": {"9": {"images": [{"filename": "f.png", "subfolder": "",
                                                         "type": "output"}]}}}}

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None): return _Resp()
        async def get(self, url, params=None):
            return _Hist() if "/history/" in url else _Resp()

    monkeypatch.setattr(gi.httpx, "AsyncClient", _Client)
    result = await GenerateImageTool(workspace_root=str(tmp_path)).run(prompt="x")
    assert result.success is True, result.error
    assert IMAGE_ID_RE.fullmatch(result.metadata["image_id"])
