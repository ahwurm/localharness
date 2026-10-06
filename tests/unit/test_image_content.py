"""core.content: image parts, the text projection, the budget formula, the cap.

Image bytes here are built in-test (header-exact PNG/GIF/WebP/JPEG) so the sniffing is proven
against the format, not against a fixture someone could swap."""
from __future__ import annotations

import struct
import zlib

import pytest

from localharness.core import content as c


def png_bytes(w: int, h: int) -> bytes:
    """A real, decodable RGB PNG of w×h (solid colour) — Pillow can open it."""
    raw = b"".join(b"\x00" + b"\x80\x40\x20" * w for _ in range(h))
    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def jpeg_header(w: int, h: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, h, w, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def test_sniff_reads_every_supported_header():
    assert c.sniff(png_bytes(265, 51)) == ("image/png", 265, 51)
    assert c.sniff(b"GIF89a" + struct.pack("<HH", 320, 200) + b"\x00" * 6) == ("image/gif", 320, 200)
    assert c.sniff(jpeg_header(1920, 1080)) == ("image/jpeg", 1920, 1080)
    vp8x = b"RIFF" + b"\x00" * 4 + b"WEBPVP8X" + b"\x00" * 4 + b"\x00" * 4 + (799).to_bytes(3, "little") + (599).to_bytes(3, "little")
    assert c.sniff(vp8x) == ("image/webp", 800, 600)
    with pytest.raises(c.ImageError, match="not a PNG, JPEG, GIF or WebP"):
        c.sniff(b"%PDF-1.4 not an image")


def test_image_tokens_is_the_measured_resize_rule():
    # Live vLLM 2026-10-06 (usage.prompt_tokens minus the same text alone): 265×51 → 78 because
    # the preprocessor upscales to 65,536 px first; 810×576 → 452 (edges round to 32: 800×576).
    assert c.image_tokens(265, 51) == 78
    assert c.image_tokens(810, 576) == 25 * 18 + 2 == 452
    assert c.image_tokens(1920, 1080) == 60 * 34 + 2 == 2042
    assert c.image_tokens(3840, 2160) == 120 * 68 + 2 == 8162
    assert c.image_tokens(1, 1) == 66  # the floor: 8×8 patches of the upscaled minimum


def test_image_part_carries_facts_privately_and_bytes_as_data_uri():
    data = png_bytes(64, 32)
    part = c.image_part(data, name="shot.png")
    assert part["type"] == "image_url"
    assert part["image_url"]["url"].startswith("data:image/png;base64,")
    assert c.image_bytes(part) == data
    assert part["_lh"]["width"] == 64 and part["_lh"]["height"] == 32 and part["_lh"]["name"] == "shot.png"
    assert c.part_tokens(part) == c.image_tokens(64, 32)
    stripped = c.strip_private([part])[0]
    assert "_lh" not in stripped
    assert c.part_meta(stripped)["width"] == 64  # re-read from the data URI when the record is gone


def test_text_projection_names_the_picture_never_hides_it():
    part = c.image_part(png_bytes(64, 32), name="shot.png")
    body = [c.text_part("what is wrong here?"), part]
    assert c.text_of(body) == "what is wrong here?\n[image: 64×32 png, shot.png]"
    assert c.text_of("plain") == "plain"
    assert c.drop_images(body).endswith("[image dropped to fit the context window: 64×32 png — attach it again if it is still needed]")
    assert c.with_text(body, "note")[-1] == c.text_part("note")
    assert c.with_text("a", "note") == "a\n\nnote"
    assert c.join_contents("a", "b") == "a\n\nb"
    assert c.join_contents(body, "b") == [*body, c.text_part("\n\n"), c.text_part("b")]
    assert c.image_parts(body) == [part] and c.image_parts("plain") == []


def test_redact_images_keeps_logs_free_of_screenshots():
    part = c.image_part(png_bytes(64, 32))
    out = c.redact_images([{"role": "user", "content": [c.text_part("hi"), part]}, {"role": "assistant", "content": "ok"}])
    url = out[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,<") and url.endswith("chars redacted>")
    assert len(url) < 80
    assert out[1] == {"role": "assistant", "content": "ok"}


def test_load_image_file_validates_by_header_and_size(tmp_path):
    good = tmp_path / "shot.png"
    good.write_bytes(png_bytes(8, 8))
    assert c.load_image_file(good)["_lh"]["name"] == "shot.png"
    fake = tmp_path / "shot.png.txt"
    fake.write_text("hello")
    with pytest.raises(c.ImageError, match="not a PNG"):
        c.load_image_file(fake)
    with pytest.raises(c.ImageError, match="is not a file|No such file"):
        c.load_image_file(tmp_path / "missing.png")
    with pytest.raises(c.ImageError, match="limit is 0 MiB"):
        c.load_image_file(good, max_bytes=4)


def test_fit_to_tokens_downscales_with_pillow_and_names_the_source_size():
    pytest.importorskip("PIL")
    part = c.image_part(png_bytes(1600, 900))  # 50×29 patches = 1452 tokens
    assert c.fit_to_tokens(part, 2000) is part
    small = c.fit_to_tokens(part, 400)
    assert c.part_tokens(small) <= 400
    assert small["_lh"]["downscaled_from"] == [1600, 900]
    assert small["_lh"]["width"] < 1600


def test_fit_to_tokens_refuses_explicitly_without_pillow(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "PIL", None)  # import PIL → ImportError
    part = c.image_part(jpeg_header(3840, 2160), name="4k.jpg")
    with pytest.raises(c.ImageError) as exc:
        c.fit_to_tokens(part, 4096)
    msg = str(exc.value)
    assert "4k.jpg 3840×2160 jpeg is ~8,162 tokens; the cap is 4,096" in msg
    assert "localharness[vision]" in msg
