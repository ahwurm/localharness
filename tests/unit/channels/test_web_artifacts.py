"""The generic artifact route `/api/artifacts/{plugin}/{id}` (PAPI-10's route half, criterion 5),
driven over the real ASGI wire: real routes, real auth, real files on disk.

Posture, the same the image-only route on the image branch had, now for any plugin: authenticated
like every GET; served only for a plugin whose root core computed and handed this session — any
other name 404s before a single filesystem call; only a core-minted id; only the core media-type
allowlist (else 415); a symlink is never served, so nothing leaves the root; an id names
one immutable file, so it is cached as immutable.
"""
from __future__ import annotations

import asyncio
import os
import re
import threading
from pathlib import Path

import pytest

from localharness.channels.web import auth
from localharness.channels.web import server as server_mod
from localharness.channels.web.channel import WebChannel
from localharness.channels.web.protocol import PROTOCOL_VERSION
from localharness.core.artifacts import artifact_root, mint_artifact_id, write_artifact
from localharness.core.bus import EventBus
from tests.unit.channels.test_web_server import BEARER, TOKEN, _stack

pytestmark = pytest.mark.asyncio

PNG = (server_mod.PACKAGED_UI_DIR / "icon-192.png").read_bytes()  # a real PNG the server ships
GIF = (b"GIF89a\x01\x00\x01\x00\x80\x00\x00\xff\xff\xff\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00"
       b",\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;")  # a 1x1 GIF: off the allowlist
IMMUTABLE = "private, max-age=31536000, immutable"


async def _served(tmp_path):
    """A session with the example plugin ON: its root is the one core computes for it."""
    root = artifact_root(tmp_path / "state", "example")
    _, channel, _, client = await _stack(tmp_path, runtime={"artifact_roots": {"example": root}})
    return root, channel, client


async def test_an_unauthenticated_get_is_refused_like_every_get(tmp_path):
    root, _, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    url = f"/api/artifacts/example/{ref.id}"
    refused = await client.get(url)
    assert refused.status_code == 401
    assert refused.json() == (await client.get("/api/health")).json()  # the refusal every GET gives
    assert (await client.get(url, headers={"Authorization": "Bearer not-the-token"})).status_code == 401
    client.cookies.set(auth.AUTH_COOKIE, TOKEN)  # the enrolment cookie: how a same-origin <img> loads
    got = await client.get(url)
    assert got.status_code == 200 and got.content == PNG


async def test_a_valid_id_is_served_with_its_type_and_the_immutable_cache_header(tmp_path):
    root, _, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    got = await client.get(f"/api/artifacts/example/{ref.id}", headers=BEARER)
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.headers["cache-control"] == IMMUTABLE
    assert got.headers["x-content-type-options"] == "nosniff"
    assert got.content == PNG


async def test_an_image_plugin_png_is_served_from_the_generic_route(tmp_path):
    """The image plugin needs no route of its own: its PNG rides the generic one."""
    root = artifact_root(tmp_path / "state", "image")
    ref = write_artifact(root, "image", PNG, "image/png")
    _, _, _, client = await _stack(tmp_path, runtime={"artifact_roots": {"image": root}})
    r = await client.get(f"/api/artifacts/image/{ref.id}", headers=BEARER)
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png"
    assert r.headers["cache-control"] == IMMUTABLE == server_mod.ARTIFACT_CACHE_CONTROL


async def test_a_plugin_that_is_not_on_404s_without_touching_any_filesystem(tmp_path, monkeypatch):
    root, _, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    (tmp_path / "state" / "artifacts" / "image").mkdir()
    (tmp_path / "state" / "artifacts" / "image" / f"{ref.id}.png").write_bytes(PNG)

    def touched(*_a, **_k):
        raise AssertionError("the route touched the filesystem for a plugin that is not on")

    def on_this_thread(real):  # the app runs on this thread; other threads keep the real call
        here = threading.get_ident()
        return lambda *a, **k: touched() if threading.get_ident() == here else real(*a, **k)

    with monkeypatch.context() as spies:  # only around the requests: teardown may stat freely
        spies.setattr(server_mod, "_find_artifact", touched)
        spies.setattr(auth, "confine", touched)
        for name in ("stat", "lstat", "listdir", "scandir"):
            spies.setattr(os, name, on_this_thread(getattr(os, name)))
        statuses = [(await client.get(f"/api/artifacts/{plugin}/{ref.id}", headers=BEARER)).status_code
                    for plugin in ("image", "Example")]
    assert statuses == [404, 404]


@pytest.mark.parametrize("bad", ["art-1", "{id}.png", "{id}.png.png", "ART-{rest}", "{id}x"])
async def test_an_id_not_of_the_core_minted_shape_404s(tmp_path, bad):
    """Each bad id names a file that IS in the root, so only the id check can refuse it."""
    root, _, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    bad = bad.format(id=ref.id, rest=ref.id[4:])
    (root / f"{bad}.png").write_bytes(PNG)
    got = await client.get(f"/api/artifacts/example/{bad}", headers=BEARER)
    assert got.status_code == 404


async def test_a_traversal_never_reaches_a_file(tmp_path):
    root, _, client = await _served(tmp_path)
    write_artifact(root, "example", PNG, "image/png")  # the root exists; the secret sits above it
    (tmp_path / "state" / "secret.png").write_bytes(PNG)
    for path in ("/api/artifacts/example/../secret", "/api/artifacts/example/..%2F..%2Fsecret",
                 "/api/artifacts/example/%2E%2E%2Fsecret"):
        got = await client.get(path, headers=BEARER)
        assert got.status_code == 404 and got.content != PNG, path


async def test_a_type_off_the_allowlist_is_415(tmp_path):
    root, _, client = await _served(tmp_path)
    artifact_id = mint_artifact_id()
    root.mkdir(parents=True)
    (root / f"{artifact_id}.gif").write_bytes(GIF)
    got = await client.get(f"/api/artifacts/example/{artifact_id}", headers=BEARER)
    assert got.status_code == 415 and got.content != GIF


async def test_an_id_that_names_two_files_is_not_served(tmp_path):
    root, _, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    (root / f"{ref.id}.jpg").write_bytes(PNG)
    assert (await client.get(f"/api/artifacts/example/{ref.id}", headers=BEARER)).status_code == 404


async def test_a_symlink_out_of_the_root_is_not_served(tmp_path):
    root, _, client = await _served(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    artifact_id = mint_artifact_id()
    root.mkdir(parents=True)
    (root / f"{artifact_id}.png").symlink_to(outside)
    got = await client.get(f"/api/artifacts/example/{artifact_id}", headers=BEARER)
    assert got.status_code == 404 and got.content != PNG


async def test_a_missing_root_a_missing_file_and_a_directory_404(tmp_path):
    root, _, client = await _served(tmp_path)
    artifact_id = mint_artifact_id()
    url = f"/api/artifacts/example/{artifact_id}"
    assert (await client.get(url, headers=BEARER)).status_code == 404  # the plugin wrote nothing yet
    root.mkdir(parents=True)
    assert (await client.get(url, headers=BEARER)).status_code == 404
    (root / f"{artifact_id}.png").mkdir()
    assert (await client.get(url, headers=BEARER)).status_code == 404


async def test_the_session_owns_the_roots(tmp_path):
    """bind_runtime takes a copy of the core-computed roots; a new session starts with none."""
    root = artifact_root(tmp_path / "state", "example")
    roots = {"example": root}
    channel = WebChannel(bus=EventBus(persist_path=tmp_path / "bus.jsonl"), config={})
    assert channel.artifact_roots() == {}
    channel.bind_runtime(session_id="s1", agent_id="orchestrator", artifact_roots=roots)
    roots["image"] = root
    assert channel.artifact_roots() == {"example": root}
    channel.reset_session()
    assert channel.artifact_roots() == {}
    channel.bind_runtime(session_id="s2", agent_id="orchestrator")
    assert channel.artifact_roots() == {}


async def test_a_reset_session_stops_serving(tmp_path):
    root, channel, client = await _served(tmp_path)
    ref = write_artifact(root, "example", PNG, "image/png")
    url = f"/api/artifacts/example/{ref.id}"
    assert (await client.get(url, headers=BEARER)).status_code == 200
    channel.reset_session()
    assert (await client.get(url, headers=BEARER)).status_code == 404


async def test_the_wire_protocol_did_not_move(tmp_path):
    """The artifact ROUTE adds no wire verb: the hello frame reports the protocol constant,
    whatever it is."""
    _, _, client = await _served(tmp_path)
    body = (await client.get("/api/protocol", headers=BEARER)).json()
    assert body["protocol_version"] == PROTOCOL_VERSION
    verb = next(v for v in body["verbs"] if v["path"] == "/api/artifacts/{plugin}/{id}")
    assert verb["method"] == "GET"


# ------------------------------------------------------------------ the gallery listing (46-05)

def _put(root, name, data=PNG):
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(data)
    return name


async def _two_roots(tmp_path, **kw):
    roots = {p: artifact_root(tmp_path / "state", p) for p in ("example", "image")}
    _, channel, server, client = await _stack(tmp_path, runtime={"artifact_roots": roots}, **kw)
    return roots, server, client


async def test_the_listing_merges_every_bound_root_newest_first(tmp_path):
    roots, _, client = await _two_roots(tmp_path)
    _put(roots["example"], "art-20260930-120000-aaaaaa.png")
    _put(roots["image"], "art-20260930-130000-aaaaaa.webp", PNG[:10])
    _put(roots["example"], "art-20260930-140000-aaaaaa.jpg")
    got = await client.get("/api/artifacts", headers=BEARER)
    assert got.status_code == 200
    body = got.json()
    assert body["truncated"] is False
    assert body["items"] == [
        {"plugin": "example", "id": "art-20260930-140000-aaaaaa", "mime": "image/jpeg", "bytes": len(PNG)},
        {"plugin": "image", "id": "art-20260930-130000-aaaaaa", "mime": "image/webp", "bytes": 10},
        {"plugin": "example", "id": "art-20260930-120000-aaaaaa", "mime": "image/png", "bytes": len(PNG)},
    ]
    assert str(tmp_path) not in got.text  # ids, never a path


async def test_the_listing_pages_sixty_at_a_time(tmp_path):
    roots, _, client = await _two_roots(tmp_path)
    ids = [f"art-20260930-120000-{i:06x}" for i in range(61)]
    for artifact_id in ids:
        _put(roots["example"], f"{artifact_id}.png", b"x")
    first = (await client.get("/api/artifacts", headers=BEARER)).json()
    assert [i["id"] for i in first["items"]] == ids[:0:-1] and first["truncated"] is True
    older = (await client.get(f"/api/artifacts?before={ids[1]}", headers=BEARER)).json()
    assert [i["id"] for i in older["items"]] == [ids[0]] and older["truncated"] is False
    assert (await client.get("/api/artifacts?before=not-an-id", headers=BEARER)).status_code == 400


async def test_the_listing_skips_everything_the_artifact_route_would_not_serve(tmp_path):
    roots, _, client = await _two_roots(tmp_path)
    root = roots["example"]
    keep = _put(root, "art-20260930-120000-000000.png")
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG)
    (root / "art-20260930-120001-000000.png").symlink_to(outside)
    (root / "art-20260930-120002-000000.png").symlink_to(tmp_path / "gone.png")  # dangling
    _put(root / "art-20260930-120003-000000", "art-20260930-120003-000001.png")  # a subfolder
    _put(root, "art-20260930-120004-bbbbbb.gif", GIF)
    _put(root, "art-20260930-120005-cccccc.PNG")
    _put(root, "notes.png")
    _put(root, "art-20260930-120006-dddddd.png")
    _put(root, "art-20260930-120006-dddddd.webp")  # an ambiguous stem
    items = (await client.get("/api/artifacts", headers=BEARER)).json()["items"]
    assert [i["id"] for i in items] == [keep.removesuffix(".png")]


async def test_the_listing_is_authed_and_404s_with_no_root_bound(tmp_path):
    _, _, client = await _two_roots(tmp_path)
    assert (await client.get("/api/artifacts")).status_code == 401
    _, _, _, bare = await _stack(tmp_path)
    assert (await bare.get("/api/artifacts", headers=BEARER)).status_code == 404


async def test_no_store_flips_the_cache_header_and_hides_the_listing(tmp_path):
    roots, server, client = await _two_roots(tmp_path, no_store=True)
    name = _put(roots["image"], "art-20260930-120000-aaaaaa.png")
    pic = await client.get(f"/api/artifacts/image/{name[:-4]}", headers=BEARER)
    assert pic.status_code == 200 and pic.content == PNG  # inline pictures still load
    assert pic.headers["cache-control"] == "no-store"
    assert (await client.get("/api/artifacts", headers=BEARER)).status_code == 404
    assert server._artifact_policy()[1:] == ("no-store", False)


async def test_one_decision_point_reads_the_cache_header_and_the_flag():
    source = Path(server_mod.__file__).read_text(encoding="utf-8")
    assert source.count("ARTIFACT_CACHE_CONTROL") == 2  # its definition + _artifact_policy
    assert source.count("self.no_store") == 2  # its assignment + _artifact_policy
    assert "channel.artifact_root(" not in source


def _invoke_web(tmp_path, monkeypatch, *args):
    """`web` with _serve faked; sync because web_cmd runs its own asyncio.run (call via a thread)."""
    from typer.testing import CliRunner

    from localharness.cli import web_cmd

    seen: dict = {}

    async def fake_serve(**kw):
        seen.update(kw)

    monkeypatch.setattr(web_cmd, "_serve", fake_serve)
    result = CliRunner().invoke(web_cmd.app, ["--config-dir", str(tmp_path), *args],
                                env={"COLUMNS": "200"})
    return result, seen


async def test_no_store_flag_reaches_the_server(tmp_path, monkeypatch):
    result, seen = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch, "--no-store")
    assert result.exit_code == 0, result.output
    assert seen["no_store"] is True
    result, seen = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch)
    assert result.exit_code == 0 and seen["no_store"] is False


async def test_no_store_help_names_its_limit(tmp_path, monkeypatch):
    result, _ = await asyncio.to_thread(_invoke_web, tmp_path, monkeypatch, "--help")
    flat = re.sub(r"[\s│]+", " ", result.output)
    assert "--no-store" in flat and "still persist" in flat


# ------------------------------------------------------------------ screens presence (46-07, v5)

async def test_screens_follow_the_slot_and_the_policy(tmp_path):
    from tests.unit.channels.test_web_server import _fake_slot

    async def screens(**kw):
        _, _, _, client = await _stack(tmp_path, **kw)
        return (await client.get("/api/protocol", headers=BEARER)).json()["screens"]

    root = {"image": artifact_root(tmp_path / "state", "image")}
    assert await screens() == {"memory": False, "pictures": False}
    on = {"memory_slot": _fake_slot(), "artifact_roots": root}
    assert await screens(runtime=dict(on)) == {"memory": True, "pictures": True}
    assert await screens(runtime=dict(on), no_store=True) == {"memory": True, "pictures": False}


async def test_the_memory_button_means_a_browse_api_exists(tmp_path):
    """screens.memory and the memory routes read one fact: an occupant with no browse API shows no
    button, rather than a button whose every tap 404s."""
    from localharness.plugins.api import MemorySlotPlugin
    from localharness.plugins.slot import MemorySlot
    from tests.unit.channels.test_web_server import _fake_slot

    slot = MemorySlot()
    slot.seat(MemorySlotPlugin(), name="no-browse")  # the base browse() answers None
    _, _, _, client = await _stack(tmp_path, runtime={"memory_slot": slot})
    assert (await client.get("/api/protocol", headers=BEARER)).json()["screens"]["memory"] is False
    assert (await client.get("/api/memory", headers=BEARER)).status_code == 404
    _, _, _, client = await _stack(tmp_path, runtime={"memory_slot": _fake_slot()})
    assert (await client.get("/api/protocol", headers=BEARER)).json()["screens"]["memory"] is True
    assert (await client.get("/api/memory", headers=BEARER)).status_code == 200


async def test_the_listing_and_the_route_share_one_ambiguity_rule(tmp_path):
    """An id the listing shows is an id the route serves, and vice versa: a stem that names an
    off-allowlist twin or a subfolder is ambiguous for both."""
    roots, _, client = await _two_roots(tmp_path)
    root = roots["example"]
    lone = "art-20260930-120000-aaaaaa"
    txt_twin, dir_twin = "art-20260930-120001-aaaaaa", "art-20260930-120002-aaaaaa"
    for artifact_id in (lone, txt_twin, dir_twin):
        _put(root, f"{artifact_id}.png")
    _put(root, f"{txt_twin}.txt", b"notes")
    (root / dir_twin).mkdir()
    listed = [i["id"] for i in (await client.get("/api/artifacts", headers=BEARER)).json()["items"]]
    served = [a for a in (lone, txt_twin, dir_twin)
              if (await client.get(f"/api/artifacts/example/{a}", headers=BEARER)).status_code == 200]
    assert listed == served == [lone]
