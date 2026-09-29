"""generate_image — text-to-image via a locally run ComfyUI server (opt-in module).

Model-agnostic by construction: the model-specific part is a ComfyUI workflow TEMPLATE
(JSON with __LH_*__ placeholder values), selected by config — swapping image models is a
new template + downloaded weights, zero harness code. The repo ships one reference
template per supported model under workflows/ (first: qwen-image-2.1).

Config is operator env, never tool arguments — the subject model sees only
prompt/width/height/steps/seed:
  LOCALHARNESS_COMFYUI_URL        e.g. http://127.0.0.1:8188 — unset = tool not registered
  LOCALHARNESS_COMFYUI_WORKFLOW   path to a workflow template (default: shipped qwen-image-2.1)
  LOCALHARNESS_COMFYUI_TIMEOUT_S  generation deadline (default 570; first call loads weights)

Trust model (why this is in NEITHER capability bucket, see tools/capabilities.py): the
endpoint is operator-configured localhost — POSTing to it is not untrusted ingest (nothing
attacker-controllable enters context; the result text below is harness-authored), and the
tool takes no path argument — it can only write harness-named PNGs into its own artifacts
dir, so it is not host-dangerous either.
"""
from __future__ import annotations

import asyncio
import importlib.resources
import json
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from localharness.tools.base import Tool, ToolResult, ToolSchema

_POLL_S = 2.0  # /history poll cadence (patched down in tests)
_PLACEHOLDER_PROMPT = "__LH_PROMPT__"
_INT_PLACEHOLDERS = ("__LH_WIDTH__", "__LH_HEIGHT__", "__LH_STEPS__", "__LH_SEED__")
_PLACEHOLDER_PREFIX = "__LH_PREFIX__"


def comfyui_url() -> str | None:
    """Operator opt-in switch: the tool registers only when this is set (read per call so
    tests and long-lived processes see env changes, unlike an import-time snapshot)."""
    url = os.environ.get("LOCALHARNESS_COMFYUI_URL", "").strip().rstrip("/")
    return url or None


def _deadline_s() -> float:
    try:
        return float(os.environ.get("LOCALHARNESS_COMFYUI_TIMEOUT_S", "570"))
    except ValueError:
        return 570.0


def image_artifacts_dir(workspace_root: str | None) -> Path:
    """Harness-owned destination for generated PNGs. The web channel's future image endpoint
    must serve from exactly this root (single definition — keep them from drifting apart)."""
    base = Path(workspace_root).expanduser() if workspace_root else Path.home()
    return base / ".localharness" / "artifacts" / "images"


def load_template() -> tuple[dict[str, Any], str]:
    """Load the active workflow template; returns (graph, template_name). Strips top-level
    entries that aren't ComfyUI nodes (no class_type) so templates can carry _comment keys."""
    path = os.environ.get("LOCALHARNESS_COMFYUI_WORKFLOW", "").strip()
    if path:
        p = Path(path).expanduser()
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


async def generate(base: str, graph: dict[str, Any], dest_dir: Path,
                   image_id: str, deadline_s: float) -> tuple[Path, int, float]:
    """Submit, wait, download. Returns (png_path, images_in_result, elapsed_s).
    Shared by the tool and the CLI command — one client, two surfaces."""
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=90.0)) as client:
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
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest = dest_dir / f"{image_id}.png"
                    dest.write_bytes(view.content)
                    return dest, len(saved), time.monotonic() - t0
            if time.monotonic() - t0 > deadline_s:
                raise TimeoutError(f"image not finished after {int(deadline_s)}s "
                                   "(first call also loads model weights — retry once, or raise "
                                   "LOCALHARNESS_COMFYUI_TIMEOUT_S)")


def _snap(v: int, lo: int = 256, hi: int = 2048) -> int:
    # Diffusion latents need /8 dims; /16 is safe across model families. Snap, don't reject —
    # the actual size is reported in the result text.
    return max(lo, min(hi, (int(v) // 16) * 16))


class GenerateImageTool(Tool):
    timeout_s = 600.0  # outer bound; the inner deadline (default 570s) fires first with a clear message

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
        )

    async def _execute(self, prompt: str, width: int = 1024, height: int = 1024,
                       steps: int = 30, seed: int | None = None) -> ToolResult:
        if not (prompt or "").strip():
            return self.err("prompt is empty", error_type="validation_error")
        base = comfyui_url()
        if base is None:
            return self.err("image generation is not configured on this install "
                            "(set LOCALHARNESS_COMFYUI_URL)", error_type="execution_error")
        try:
            template, tpl_name = load_template()
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return self.err(f"workflow template unusable: {exc}", error_type="validation_error")
        w, h = _snap(width), _snap(height)
        steps = max(1, min(int(steps), 100))
        seed = random.randrange(2**31) if seed is None else abs(int(seed)) % 2**63
        image_id = f"img-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        try:
            graph = build_graph(template, prompt=prompt, width=w, height=h,
                                steps=steps, seed=seed, prefix=f"lh_{image_id}")
        except ValueError as exc:
            return self.err(str(exc), error_type="validation_error")
        try:
            dest, count, elapsed = await generate(
                base, graph, image_artifacts_dir(self.workspace_root), image_id, _deadline_s())
        except (httpx.ConnectError, httpx.ConnectTimeout):
            return self.err(f"ComfyUI unreachable at {base} — the image server is not running. "
                            "Start it first (see docs/reference-architectures, or run "
                            "`localharness doctor`).")
        except httpx.HTTPError as exc:
            return self.err(f"ComfyUI request failed: {exc}")
        except (RuntimeError, TimeoutError) as exc:
            return self.err(str(exc))
        note = f" ({count} images generated; saved the first)" if count > 1 else ""
        return self.ok(
            f"Image saved: {dest} ({w}x{h}, seed {seed}, {steps} steps, {elapsed:.1f}s){note}",
            image_id=image_id, path=str(dest), width=w, height=h, steps=steps,
            seed=seed, template=tpl_name,
        )
