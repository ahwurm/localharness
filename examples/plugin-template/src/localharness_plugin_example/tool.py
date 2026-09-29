"""The `example_swatch` tool: renders a solid-color PNG into the directory core assigned the plugin.

Imported only when a session asks the plugin for its tools (ExamplePlugin.tools).
"""
from __future__ import annotations

import struct
import zlib
from typing import Any

from localharness.core.artifacts import write_artifact
from localharness.plugins.api import PluginContext
from localharness.tools.base import Tool, ToolResult, ToolSchema


def solid_png(size: int, rgb: bytes) -> bytes:
    """A size x size truecolor PNG in one color (`rgb`: three bytes), standard library only."""
    raw = (b"\x00" + rgb * size) * size  # each row: filter type 0, then `size` pixels

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class SwatchTool(Tool):
    """Renders a swatch in the configured color and size, and returns its artifact id."""

    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="example_swatch",  # one namespace with core's tools: prefix yours with your plugin
            description="Render a small solid-color swatch image and return its artifact id.",
            parameters={"type": "object", "properties": {}, "required": []},
            # The four safety declarations (SAFE-01). Say what the tool does: a tool that says
            # nothing is treated as the riskiest kind on all four.
            ingest="none",            # it reads no outside content
            host="safe",              # it writes only into the directory core assigned it
            result_origin="trusted",  # its result is text this code wrote from validated settings
            gate_family=None,         # no family: asked about once per workspace in `guarded`. A
                                      # plugin you install has its family honoured only if it asks
                                      # at least as often as an undeclared tool would (SAFE-06).
        )

    async def _execute(self, **_: Any) -> ToolResult:
        root = self._ctx.paths.artifact_dir
        if root is None:
            return self.err("no artifact directory was assigned to the example plugin")
        color, size = self._ctx.config.color, self._ctx.agent_config.size
        ref = write_artifact(root, "example", solid_png(size, bytes.fromhex(color[1:])), "image/png")
        return self.ok(f"Rendered a {size}x{size} {color} swatch: artifact {ref.id}",
                       artifact=ref.model_dump())
