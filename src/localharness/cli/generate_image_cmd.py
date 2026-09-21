"""`localharness generate-image` — the image module's CLI surface.

Same client and template contract as the generate_image tool (one implementation,
two surfaces); writes the PNG to the current directory or --out."""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import typer

from localharness.tools.builtin.generate_image_tool import (
    _snap,
    build_graph,
    comfyui_url,
    generate,
    load_template,
    _deadline_s,
)


def generate_image(
    prompt: str = typer.Argument(..., help="What the image should show."),
    width: int = typer.Option(1024, "--width", help="Width in pixels (snapped to /16)."),
    height: int = typer.Option(1024, "--height", help="Height in pixels (snapped to /16)."),
    steps: int = typer.Option(30, "--steps", min=1, max=100, help="Sampling steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed (default: random)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output PNG path "
                                       "(default: ./img-<stamp>.png)."),
) -> None:
    """Generate an image with the locally configured image model (ComfyUI)."""
    base = comfyui_url()
    if base is None:
        typer.echo("image module not configured: set LOCALHARNESS_COMFYUI_URL "
                   "(e.g. http://127.0.0.1:8188)", err=True)
        raise typer.Exit(code=2)
    try:
        template, tpl_name = load_template()
    except Exception as exc:  # noqa: BLE001 — bad path/JSON both end the same way for the user
        typer.echo(f"workflow template unusable: {exc}", err=True)
        raise typer.Exit(code=2)
    if out is not None and out.suffix != ".png":
        typer.echo(f"--out must end in .png (got: {out})", err=True)
        raise typer.Exit(code=2)
    w, h = _snap(width), _snap(height)
    seed = random.randrange(2**31) if seed is None else abs(int(seed)) % 2**63
    image_id = (out.stem if out is not None
                else f"img-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}")
    dest_dir = out.parent if out is not None else Path.cwd()
    graph = build_graph(template, prompt=prompt, width=w, height=h, steps=steps,
                        seed=seed, prefix=f"lh_{image_id}")
    try:
        dest, count, elapsed = asyncio.run(
            generate(base, graph, dest_dir, image_id, _deadline_s()))
    except (httpx.ConnectError, httpx.ConnectTimeout):
        typer.echo(f"ComfyUI unreachable at {base} — start the image server first "
                   "(localharness doctor checks it).", err=True)
        raise typer.Exit(code=1)
    except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    note = f" ({count} images; saved the first)" if count > 1 else ""
    typer.echo(f"saved: {dest} ({w}x{h}, seed {seed}, {steps} steps, "
               f"{elapsed:.1f}s, template {tpl_name}){note}")
