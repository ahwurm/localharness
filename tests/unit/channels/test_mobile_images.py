"""Image input on the mobile surface: the message route takes `images`, validates them through
`core.content.image_part`, and the channel hands the parts to the REPL via `take_images()`."""
from __future__ import annotations

import asyncio
import base64

import pytest

from tests.unit.channels.test_mobile_server import JSON, _stack
from tests.unit.test_image_content import png_bytes

pytestmark = pytest.mark.asyncio

MSG = "/api/sessions/s1/message"


def _img(data: bytes, name: str = "shot.png") -> dict:
    return {"data": base64.b64encode(data).decode(), "mime": "image/png", "name": name}


async def test_a_png_with_text_reaches_the_repl_as_text_plus_one_part(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "what is this?", "images": [_img(png_bytes(37, 11))]},
                            headers=JSON)
    assert got.status_code == 200 and got.json()["status"] == "queued"
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "what is this?"
    parts = channel.take_images()
    assert len(parts) == 1 and parts[0]["type"] == "image_url"
    assert (parts[0]["_lh"]["width"], parts[0]["_lh"]["height"], parts[0]["_lh"]["name"]) == (37, 11, "shot.png")
    assert channel.take_images() == []   # consumed once


async def test_a_non_image_payload_is_400_with_the_sniff_error(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "x", "images": [_img(b"%PDF-1.7 not a picture")]},
                            headers=JSON)
    assert got.status_code == 400
    assert got.json()["error"].startswith("not a PNG, JPEG, GIF or WebP image")
    assert channel._inbound.empty()


async def test_bad_base64_is_400(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "x", "images": [{"data": "!!!", "name": "a"}]},
                            headers=JSON)
    assert got.status_code == 400 and "base64" in got.json()["error"]


async def test_five_images_are_refused(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "x", "images": [_img(png_bytes(2, 2))] * 5},
                            headers=JSON)
    assert got.status_code == 400 and "at most 4" in got.json()["error"]
    assert channel._inbound.empty()


async def test_an_image_only_message_reads_as_its_label(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "", "images": [_img(png_bytes(40, 30), "a.png")]},
                            headers=JSON)
    assert got.status_code == 200
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "[image: 40×30 png, a.png]"
    assert len(channel.take_images()) == 1


async def test_empty_text_and_no_images_is_still_refused(tmp_path):
    _, _, _, client = await _stack(tmp_path)
    got = await client.post(MSG, json={"text": "", "images": []}, headers=JSON)
    assert got.status_code == 400 and got.json()["error"] == "text is required"


async def test_images_on_a_nudge_are_refused(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    channel._turn_running = True
    seen: list[str] = []

    async def _nudge(text: str, intent: str) -> bool:
        seen.append(text)
        return True

    channel._nudge_resolver = _nudge
    got = await client.post(MSG, json={"text": "look", "intent": "nudge",
                                       "images": [_img(png_bytes(4, 4))]}, headers=JSON)
    assert got.status_code == 400
    assert got.json()["error"] == "images cannot ride a nudge; send them as a new message"
    assert seen == [] and channel._inbound.empty()


async def test_the_message_route_takes_a_body_over_1mib_and_other_routes_do_not(tmp_path):
    _, channel, _, client = await _stack(tmp_path)
    big = png_bytes(2, 2) + b"\x00" * (2 << 20)   # a valid header, ~2 MiB of trailing bytes
    got = await client.post(MSG, json={"text": "big", "images": [_img(big)]}, headers=JSON)
    assert got.status_code == 200
    assert await asyncio.wait_for(channel.read_input(), timeout=1) == "big"
    assert channel.take_images()[0]["_lh"]["bytes"] == len(big)
    refused = await client.post("/api/sessions/s1/command", json={"text": "x" * (2 << 20)},
                                headers=JSON)
    assert refused.status_code == 400 and refused.json()["error"] == "body too large"
