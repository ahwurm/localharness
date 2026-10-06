"""Message content parts: text and images.

A user turn is a plain ``str`` until a picture rides with it; then it is the OpenAI parts list
``[{"type": "text", ...}, {"type": "image_url", ...}]``. Everything that treats content as text
goes through :func:`text_of`; everything that needs the pictures goes through :func:`image_parts`.

An image part carries the bytes as a ``data:`` URI (what the server reads) plus a private ``_lh``
record — media type, pixel size, byte size, sha256 — that :func:`localharness.core.types.provider_messages`
strips before the wire, exactly like the message-level ``_lh``. The pixel size is what the
context budget charges: Qwen3-VL spends one token per 32×32 pixel patch (patch 16 × spatial merge
2) plus the vision start/end pair — measured live against vLLM's ``usage.prompt_tokens`` on
2026-10-06 (265×51 → 78 tokens, 810×576 → ~450). Other vision models differ; in vLLM mode the
exact count still comes from the server's ``/tokenize``, this formula is the budget estimate.

Images enter the harness ONLY by human action — a paste, a path, an upload, an attachment. No
tool reads one, so the model cannot pull a file into its own context through this path; the
validation here is therefore about honesty (real image, known size, under the cap), not
boundary.
"""
from __future__ import annotations

import base64
import hashlib
import math
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

Content = str | list[dict[str, Any]]

PIXELS_PER_TOKEN_EDGE = 32   # Qwen3-VL: patch 16 × merge 2
IMAGE_TOKEN_OVERHEAD = 2     # <|vision_start|> … <|vision_end|>
MIN_PIXELS = 65_536          # preprocessor_config.json size.shortest_edge: smaller images are upscaled
MAX_PIXELS = 16_777_216      # size.longest_edge: larger ones are downscaled by the preprocessor
MAX_IMAGE_BYTES = 20 * 1024 * 1024
DEFAULT_MAX_IMAGE_TOKENS = 4096   # a 2560×1440 screenshot (3,602 tokens) passes untouched
DROPPED_IMAGE_PREFIX = "[image dropped to fit the context window:"
VISION_EXTRA_HINT = (
    "install the vision extra (`uv sync --extra vision` or `pip install 'localharness[vision]'`) "
    "to downscale automatically, or crop it"
)


class ImageError(ValueError):
    """A file or paste that is not a usable image, or an image over the cap with no way to shrink it."""


# ── sniffing ──────────────────────────────────────────────────────────────────────────────

def sniff(data: bytes) -> tuple[str, int, int]:
    """``(media_type, width, height)`` read from the bytes' own header — never from a filename
    or a declared MIME type. Raises :class:`ImageError` for anything that is not PNG, JPEG, GIF
    or WebP."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        w, h = struct.unpack(">II", data[16:24])
        return "image/png", w, h
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return "image/gif", w, h
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ("image/webp", *_webp_dims(data))
    if data[:2] == b"\xff\xd8":
        return ("image/jpeg", *_jpeg_dims(data))
    raise ImageError("not a PNG, JPEG, GIF or WebP image (the file header did not match)")


def _webp_dims(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        return (1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little"))
    if chunk == b"VP8L" and len(data) >= 25:
        b = data[21:25]
        return (1 + ((b[1] & 0x3F) << 8 | b[0]),
                1 + ((b[3] & 0x0F) << 10 | b[2] << 2 | (b[1] & 0xC0) >> 6))
    if chunk == b"VP8 " and len(data) >= 30:
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    raise ImageError("WebP header is truncated or of an unknown chunk type")


def _jpeg_dims(data: bytes) -> tuple[int, int]:
    pos, n = 2, len(data)
    while pos + 4 <= n:
        if data[pos] != 0xFF:
            break
        marker = data[pos + 1]
        if marker == 0xFF:                      # fill byte
            pos += 1
            continue
        if marker in (0x01, 0xD8) or 0xD0 <= marker <= 0xD7:   # standalone markers
            pos += 2
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):   # SOFn
            if pos + 9 > n:
                break
            h, w = struct.unpack(">HH", data[pos + 5: pos + 9])
            return w, h
        pos += 2 + struct.unpack(">H", data[pos + 2: pos + 4])[0]
    raise ImageError("JPEG has no frame header (SOF) before the data ended")


# ── parts ─────────────────────────────────────────────────────────────────────────────────

def image_tokens(width: int, height: int) -> int:
    """Budget cost of one image: one token per 32×32 patch of the size the preprocessor actually
    feeds the model, plus the vision start/end pair. The resize rule is Qwen's ``smart_resize``:
    each edge rounds to a multiple of 32, then the whole image scales up to ``MIN_PIXELS`` or down
    to ``MAX_PIXELS`` keeping aspect. Verified live 2026-10-06: 265×51 → 78 (upscaled), 810×576 → 452."""
    f = PIXELS_PER_TOKEN_EDGE
    h_bar, w_bar = max(f, round(height / f) * f), max(f, round(width / f) * f)
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar, w_bar = math.floor(height / beta / f) * f, math.floor(width / beta / f) * f
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
    return (h_bar // f) * (w_bar // f) + IMAGE_TOKEN_OVERHEAD


def image_part(data: bytes, name: str = "") -> dict[str, Any]:
    """An OpenAI ``image_url`` part from raw bytes, validated by header. ``name`` is a label for
    humans (the file name, "clipboard", "attachment") and never reaches the model."""
    media, w, h = sniff(data)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media};base64,{base64.b64encode(data).decode('ascii')}"},
        "_lh": {"kind": "image", "media_type": media, "width": w, "height": h,
                "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "name": name},
    }


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def is_image_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def image_parts(content: Any) -> list[dict[str, Any]]:
    return [p for p in content if is_image_part(p)] if isinstance(content, list) else []


def image_bytes(part: dict[str, Any]) -> bytes:
    url = (part.get("image_url") or {}).get("url") or ""
    head, sep, payload = url.partition(",")
    if not sep or not head.startswith("data:"):
        raise ImageError("image part is not a data: URI (remote image URLs are not fetched)")
    return base64.b64decode(payload)


@lru_cache(maxsize=256)
def _sniff_url(url: str) -> tuple[str, int, int, int]:
    data = base64.b64decode(url.partition(",")[2])
    return (*sniff(data), len(data))


def part_meta(part: dict[str, Any]) -> dict[str, Any]:
    """The part's ``_lh`` record, or the same facts re-read from its data URI when the record
    was stripped (a wire-rendered list, a resumed session written by an older build)."""
    meta = part.get("_lh")
    if isinstance(meta, dict) and "width" in meta and "height" in meta:
        return meta
    media, w, h, size = _sniff_url((part.get("image_url") or {}).get("url") or "")
    return {"kind": "image", "media_type": media, "width": w, "height": h, "bytes": size, "name": ""}


def part_tokens(part: dict[str, Any]) -> int:
    meta = part_meta(part)
    return image_tokens(meta["width"], meta["height"])


def content_image_tokens(content: Any) -> int:
    return sum(part_tokens(p) for p in image_parts(content))


def image_label(part: dict[str, Any]) -> str:
    """What stands in for the picture wherever only text can go: ``[image: 1920×1080 png]``."""
    m = part_meta(part)
    name = f", {m['name']}" if m.get("name") else ""
    return f"[image: {m['width']}×{m['height']} {m['media_type'].split('/')[1]}{name}]"


def describe(part: dict[str, Any]) -> str:
    """One human line for an echo or an event: ``shot.png 1920×1080 png · ~2,042 tokens``."""
    m = part_meta(part)
    name = f"{m['name']} " if m.get("name") else ""
    return f"{name}{m['width']}×{m['height']} {m['media_type'].split('/')[1]} · ~{part_tokens(part):,} tokens"


# ── projections ───────────────────────────────────────────────────────────────────────────

def text_of(content: Any, label: Callable[[dict[str, Any]], str] = image_label) -> str:
    """The text projection: a ``str`` unchanged; a parts list joined, each image replaced by
    ``label(part)``. This is what every text-only consumer (summaries, memory, ledgers, search,
    logs) sees — the picture is named, never silently absent."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    pieces = [label(p) if is_image_part(p)
              else (p.get("text") or "") if isinstance(p, dict) else str(p)
              for p in content]
    return "\n".join(piece for piece in pieces if piece)


def dropped_image_label(part: dict[str, Any]) -> str:
    m = part_meta(part)
    return f"{DROPPED_IMAGE_PREFIX} {m['width']}×{m['height']} {m['media_type'].split('/')[1]} — attach it again if it is still needed]"


def drop_images(content: Any) -> str:
    """Text only, with each image replaced by an explicit "dropped" line. Used by the emergency
    floor: a model told "this screenshot" with no screenshot invents one (measured 2026-10-06),
    so the gap is named in the text that stays."""
    return text_of(content, dropped_image_label)


def as_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return list(content)
    text = text_of(content)
    return [text_part(text)] if text else []


def with_text(content: Any, text: str) -> Content:
    """Append text to content, keeping the images: ``str`` gets ``\\n\\n`` + text, a parts list
    gets one more text part."""
    if isinstance(content, list):
        return [*content, text_part(text)]
    return f"{content or ''}\n\n{text}"


def join_contents(first: Any, second: Any, sep: str = "\n\n") -> Content:
    """Two same-role bodies as one (strict-alternation templates). Both ``str`` → joined and
    stripped, as the XML downgrade always did; otherwise a parts list with the separator as text."""
    if not isinstance(first, list) and not isinstance(second, list):
        return ((first or "") + sep + (second or "")).strip()
    return [*as_parts(first), text_part(sep), *as_parts(second)]


def strip_private(content: Any) -> Any:
    """Parts without their ``_lh`` — the wire shape."""
    if not isinstance(content, list):
        return content
    return [{k: v for k, v in p.items() if k != "_lh"} if isinstance(p, dict) else p for p in content]


def redact_images(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For logs: the same messages with every data URI replaced by its size, so an HTTP-400 dump
    stays readable and a log file never carries a screenshot."""
    def redact_part(p: Any) -> Any:
        if not is_image_part(p):
            return p
        url = (p.get("image_url") or {}).get("url") or ""
        head = url.partition(",")[0]
        return {**p, "image_url": {"url": f"{head},<{len(url):,} chars redacted>"}}
    return [{**m, "content": [redact_part(p) for p in m["content"]]}
            if isinstance(m.get("content"), list) else m for m in messages]


# ── intake ────────────────────────────────────────────────────────────────────────────────

def load_image_file(path: str | Path, max_bytes: int = MAX_IMAGE_BYTES) -> dict[str, Any]:
    """An image part from a file the HUMAN named. Validated by header and size; the name is kept
    as the part's label."""
    p = Path(path).expanduser()
    try:
        p = p.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ImageError(f"{path}: {exc.strerror or exc}") from None
    if not p.is_file():
        raise ImageError(f"{p} is not a file")
    size = p.stat().st_size
    if size > max_bytes:
        raise ImageError(f"{p.name} is {size / 1024 / 1024:.1f} MiB; the limit is {max_bytes // 1024 // 1024} MiB")
    return image_part(p.read_bytes(), name=p.name)


def fit_to_tokens(part: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """The part unchanged when it costs ≤ ``max_tokens``; otherwise downscaled with Pillow to fit,
    or an :class:`ImageError` that names the cost, the cap and the fix. Never a silent resize,
    never a silent drop."""
    tokens = part_tokens(part)
    if tokens <= max_tokens:
        return part
    meta = part_meta(part)
    w, h, ext = meta["width"], meta["height"], meta["media_type"].split("/")[1]
    too_big = f"{meta.get('name') or 'image'} {w}×{h} {ext} is ~{tokens:,} tokens; the cap is {max_tokens:,} (context.max_image_tokens)"
    try:
        from PIL import Image
    except ImportError:
        raise ImageError(f"{too_big}; {VISION_EXTRA_HINT}") from None
    import io
    patches = max(1, max_tokens - IMAGE_TOKEN_OVERHEAD)
    scale = math.sqrt(patches / ((tokens - IMAGE_TOKEN_OVERHEAD) or 1))
    with Image.open(io.BytesIO(image_bytes(part))) as img:
        img.load()
        for _ in range(8):
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            if image_tokens(nw, nh) <= max_tokens:
                break
            scale *= 0.97
        else:
            raise ImageError(f"{too_big}; the cap is below the model's minimum image size "
                             f"({image_tokens(1, 1)} tokens) — raise context.max_image_tokens")
        resized = img.convert("RGB") if ext == "jpeg" else img
        resized = resized.resize((nw, nh), Image.LANCZOS)
        buf = io.BytesIO()
        if ext == "jpeg":
            resized.save(buf, format="JPEG", quality=90)
        else:
            resized.save(buf, format="PNG", optimize=True)
    out = image_part(buf.getvalue(), name=meta.get("name") or "")
    out["_lh"]["downscaled_from"] = [w, h]
    return out


def fit_all(parts: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    return [fit_to_tokens(p, max_tokens) for p in parts]
