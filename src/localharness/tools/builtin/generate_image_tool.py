"""generate_image — text-to-image via a locally run ComfyUI server (the image plugin's tool).

Model-agnostic by construction: the model-specific part is a ComfyUI workflow TEMPLATE
(JSON with __LH_*__ placeholder values), selected by config — swapping image models is a
new template + downloaded weights, zero harness code. The repo ships one reference
template per supported model under workflows/ (first: qwen-image-2.1).

Settings are the image plugin's (`image.comfyui_url`, `image.workflow`, `image.timeout_s`), never
tool arguments — the model sees only prompt/width/height/steps/seed.

Trust model: declared ingest none / host safe / result_origin trusted (see info()). The
endpoint is a machine-level setting — POSTing to it is not untrusted ingest (nothing
attacker-controllable enters context; the result text below is harness-authored), and the
tool takes no path argument — core mints the artifact name and root, so it is not
host-dangerous either.
"""
from __future__ import annotations

import asyncio
import importlib.resources
import json
import random
import secrets
import time
from pathlib import Path
from typing import Any

import httpx

from localharness.core.artifacts import write_artifact
from localharness.core.events import ARTIFACT_MIMES
from localharness.plugins.api import PluginContext
from localharness.tools.base import Tool, ToolResult, ToolSchema

_POLL_S = 2.0  # /history poll cadence (patched down in tests)
_PLACEHOLDER_PROMPT = "__LH_PROMPT__"
_INT_PLACEHOLDERS = ("__LH_WIDTH__", "__LH_HEIGHT__", "__LH_STEPS__", "__LH_SEED__")
_PLACEHOLDER_PREFIX = "__LH_PREFIX__"
_TRANSPORT: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None  # tests set httpx.MockTransport; one seam for the tool, the CLI and doctor


def load_template(workflow: str = "") -> tuple[dict[str, Any], str]:
    """Load a workflow template (`image.workflow`; empty = the shipped qwen-image-2.1); returns
    (graph, template_name). Strips top-level entries that aren't ComfyUI nodes (no class_type) so
    templates can carry _comment keys."""
    if workflow.strip():
        p = Path(workflow.strip()).expanduser()
        text = p.read_text(encoding="utf-8")
        name = p.stem
    else:
        res = importlib.resources.files("localharness.tools.builtin") / "workflows" / "qwen-image-2.1.json"
        text = res.read_text(encoding="utf-8")
        name = "qwen-image-2.1"
    raw = json.loads(text)
    graph = {k: v for k, v in raw.items() if isinstance(v, dict) and "class_type" in v}
    if not graph:
        raise ValueError(f"workflow template '{name}' contains no ComfyUI nodes")
    return graph, name


def _fill(node: Any, values: dict[str, Any], hits: set[str]) -> Any:
    """Replace exact-match placeholder STRING VALUES anywhere in the graph (post-parse, so
    numeric slots become real ints and the template stays valid JSON)."""
    if isinstance(node, dict):
        return {k: _fill(v, values, hits) for k, v in node.items()}
    if isinstance(node, list):
        return [_fill(v, values, hits) for v in node]
    if isinstance(node, str) and node in values:
        hits.add(node)
        return values[node]
    return node


def build_graph(template: dict[str, Any], *, prompt: str, width: int, height: int,
                steps: int, seed: int, prefix: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        _PLACEHOLDER_PROMPT: prompt,
        "__LH_WIDTH__": width, "__LH_HEIGHT__": height,
        "__LH_STEPS__": steps, "__LH_SEED__": seed,
        _PLACEHOLDER_PREFIX: prefix,
    }
    hits: set[str] = set()
    graph = _fill(template, values, hits)
    if _PLACEHOLDER_PROMPT not in hits:
        raise ValueError("workflow template has no __LH_PROMPT__ slot — it would ignore the prompt")
    return graph


async def generate(base: str, graph: dict[str, Any], deadline_s: float) -> tuple[bytes, int, float]:
    """Submit, wait, download. Returns (image_bytes, images_in_result, elapsed_s) and writes
    nothing — the caller decides where the bytes go. Shared by the tool and the CLI command."""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=90.0), transport=_TRANSPORT) as client:
        resp = await client.post(f"{base}/prompt", json={"prompt": graph})
        if resp.status_code >= 400:
            raise RuntimeError(f"ComfyUI rejected the workflow (HTTP {resp.status_code}): "
                               f"{resp.text[:600]}")
        pid = resp.json().get("prompt_id")
        if not pid:
            raise RuntimeError(f"ComfyUI returned no prompt_id: {resp.text[:300]}")
        while True:
            await asyncio.sleep(_POLL_S)
            hist = (await client.get(f"{base}/history/{pid}")).json()
            entry = hist.get(pid)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError("generation failed: "
                                       f"{json.dumps(status.get('messages', []))[:600]}")
                if status.get("completed") or status.get("status_str") == "success":
                    images = [i for o in entry.get("outputs", {}).values()
                              for i in o.get("images", [])]
                    saved = [i for i in images if i.get("type", "output") == "output"] or images
                    if not saved:
                        raise RuntimeError("generation completed but produced no images")
                    first = saved[0]
                    view = await client.get(f"{base}/view", params={
                        "filename": first.get("filename", ""),
                        "subfolder": first.get("subfolder", ""),
                        "type": first.get("type", "output"),
                    })
                    view.raise_for_status()
                    return view.content, len(saved), time.monotonic() - t0
            if time.monotonic() - t0 > deadline_s:
                raise TimeoutError(f"image not finished after {int(deadline_s)}s (the first call also "
                                   "loads the model weights — retry once, or raise image.timeout_s)")


def sniff_mime(data: bytes) -> str | None:
    """The artifact mime by magic bytes — the template's save node decides the format, not us."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _snap(v: int, lo: int = 256, hi: int = 2048) -> int:
    # Diffusion latents need /8 dims; /16 is safe across model families. Snap, don't reject —
    # the actual size is reported in the result text.
    return max(lo, min(hi, (int(v) // 16) * 16))


class GenerateImageTool(Tool):
    def __init__(self, ctx: PluginContext) -> None:
        self._ctx = ctx
        # Instance bound above image.timeout_s: the tool's own deadline fires first, with its message.
        self.timeout_s = ctx.config.timeout_s + 30.0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="generate_image",
            group="image",
            # Blunt on purpose: measured twice (2026-09-28/29), the subject model's trained
            # prior ("harnesses don't have diffusion models") beat a clean tool listing and it
            # hand-built SVG screenshots instead. The description must defeat the prior.
            description=(
                "Create a picture with the locally installed image model — image generation "
                "IS available in this harness through this tool. Use it for any request for "
                "an image, picture, art, or logo; never claim image generation is "
                "unavailable, and never hand-craft SVG or screenshots instead. Takes ~half "
                "a minute and returns the saved PNG's file path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "What the image should show."},
                    "width": {"type": "integer", "description": "Width in pixels (default 1024).",
                              "default": 1024, "minimum": 256, "maximum": 2048},
                    "height": {"type": "integer", "description": "Height in pixels (default 1024).",
                               "default": 1024, "minimum": 256, "maximum": 2048},
                    "steps": {"type": "integer",
                              "description": "Sampling steps; more is slower but finer (default 30).",
                              "default": 30, "minimum": 1, "maximum": 100},
                    "seed": {"type": "integer",
                             "description": "Seed for reproducibility (default: random)."},
                },
                "required": ["prompt"],
            },
            destructive=False,
            estimated_tokens=80,
            ingest="none",          # the result text is harness-authored; nothing fetched enters context
            host="safe",            # no path argument; core names the file and its directory
            result_origin="trusted",  # a local server the machine owner configured
            gate_family=None,       # asked once per workspace in guarded, as before the plugin
        )

    async def _execute(self, prompt: str, width: int = 1024, height: int = 1024,
                       steps: int = 30, seed: int | None = None) -> ToolResult:
        if not (prompt or "").strip():
            return self.err("prompt is empty", error_type="validation_error")
        ctx = self._ctx
        base = ctx.config.comfyui_url
        if not base:
            return self.err("image generation is not set up: image.comfyui_url is empty — run "
                            "`localharness plugins enable image`")
        root = ctx.paths.artifact_dir
        if root is None:
            return self.err("no artifact directory was assigned to the image plugin")
        try:
            template, tpl_name = load_template(ctx.config.workflow)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self.err(f"workflow template unusable: {exc}", error_type="validation_error")
        w, h = _snap(width), _snap(height)
        steps = max(1, min(int(steps), 100))
        seed = random.randrange(2**31) if seed is None else abs(int(seed)) % 2**63
        try:
            graph = build_graph(template, prompt=prompt, width=w, height=h, steps=steps, seed=seed,
                                prefix=f"lh_{secrets.token_hex(4)}")  # ComfyUI-side filename only
        except ValueError as exc:
            return self.err(str(exc), error_type="validation_error")
        try:
            data, count, elapsed = await generate(base, graph, ctx.config.timeout_s)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self.err(f"ComfyUI unreachable at {base} — the image server is not running. "
                            "Start it, then run `localharness doctor` to check.")
        except httpx.HTTPError as exc:
            return self.err(f"ComfyUI request failed: {exc}")
        except (RuntimeError, TimeoutError) as exc:
            return self.err(str(exc))
        mime = sniff_mime(data)
        if mime is None:
            return self.err("ComfyUI returned a file that is not a PNG, JPEG or WEBP image — check "
                            f"the save node in template '{tpl_name}'")
        ref = write_artifact(root, "image", data, mime)
        path = (root / f"{ref.id}{ARTIFACT_MIMES[mime]}").absolute()
        note = f" ({count} images generated; saved the first)" if count > 1 else ""
        return self.ok(
            f"Image saved: artifact {ref.id}, {path} ({w}x{h}, seed {seed}, {steps} steps, "
            f"{elapsed:.1f}s){note}",
            artifact=ref.model_dump(), path=str(path), width=w, height=h, steps=steps,
            seed=seed, template=tpl_name,
        )
