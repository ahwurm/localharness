#!/usr/bin/env python3
"""Regenerate the web channel's placeholder app icons.

The icons are a PLACEHOLDER and are meant to be replaced — three stacked bars on a dark square,
which reads at 60 pixels on a home screen and pretends to be nothing more than a marker. This
script exists so they are not mystery binaries in the tree: run it and you get exactly the files
that are committed.

    python scripts/make_web_icons.py

Replacing them properly needs no build step and no code change: drop your own `icon-192.png`,
`icon-512.png` and `icon-180.png` into `src/localharness/channels/web/ui/` (and edit `icon.svg`),
and the manifest already points at them.

stdlib only — deliberately. Pillow is not a dependency of this project and adding an image
library to draw three rectangles would be the definition of paying for an abstraction nobody
asked for.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

UI_DIR = Path(__file__).resolve().parent.parent / "src" / "localharness" / "channels" / "web" / "ui"

BACKGROUND = (17, 17, 17)      # #111111 — the manifest's theme_color
BAR = (255, 255, 255)
ACCENT = (170, 102, 0)         # #a60 — the same amber the reference page badges with

SIZES = {"icon-192.png": 192, "icon-512.png": 512, "icon-180.png": 180}
"""192 and 512 are what the manifest declares; 180 is the `apple-touch-icon`, which is what iOS
actually uses for a home-screen install (it does not take an SVG)."""

# Fractions of the canvas, so every size is the same picture: three centered bars narrowing
# downward, spanning 0.28-0.72 vertically. Nothing reaches past 56% of the width, because a
# maskable icon gets cropped to a circle by some launchers and the edges are not safe.
BARS = ((0.28, 0.56), (0.45, 0.42), (0.62, 0.28))
BAR_HEIGHT = 0.10


def _png(width: int, pixels: list[list[tuple[int, int, int]]]) -> bytes:
    """A minimal RGB PNG. Filter byte 0 (None) on every scanline, one IDAT, no interlacing."""
    raw = b"".join(b"\x00" + b"".join(bytes(px) for px in row) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, width, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def render(size: int) -> bytes:
    pixels = [[BACKGROUND] * size for _ in range(size)]
    for index, (top, width) in enumerate(BARS):
        y0, y1 = int(top * size), int((top + BAR_HEIGHT) * size)
        x0 = int((1.0 - width) * size / 2)
        x1 = x0 + int(width * size)
        color = ACCENT if index == 0 else BAR
        for y in range(max(0, y0), min(size, y1)):
            for x in range(max(0, x0), min(size, x1)):
                pixels[y][x] = color
    return _png(size, pixels)


def main() -> None:
    for name, size in SIZES.items():
        target = UI_DIR / name
        target.write_bytes(render(size))
        print(f"wrote {target} ({size}x{size})")


if __name__ == "__main__":
    main()
