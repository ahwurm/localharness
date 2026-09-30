"""`localharness generate-image` — the image plugin's command, mounted from its manifest and imported
only when it runs. Same client and template as the generate_image tool; writes the picture to --out
or the current directory. It does not record the file as an artifact (it has no session)."""
from __future__ import annotations

import asyncio
import random
import secrets
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import typer

from localharness.tools.builtin.generate_image_tool import (
    _snap, build_graph, generate, load_template, sniff_mime,
)

app = typer.Typer(help="Make a picture with your local ComfyUI server.")


def _image_config() -> Any:
    """image.* from the same resolved plugin config a session reads, or None."""
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    settings = resolve(ConfigLoader(local_config_dir=resolve_workspace_layer(
        None, interactive=False))).settings.get("image")
    return settings.config if settings is not None else None


@app.command()
def generate_image(
    prompt: str = typer.Argument(..., help="What to draw."),
    width: int = typer.Option(1024, "--width", help="Width in pixels (snapped to /16)."),
    height: int = typer.Option(1024, "--height", help="Height in pixels (snapped to /16)."),
    steps: int = typer.Option(30, "--steps", min=1, max=100, help="Sampling steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed (default: random)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Where to save the PNG (must end in .png)."),
) -> None:
    """Make a picture with your local ComfyUI server."""
    cfg = _image_config()
    if cfg is None or not cfg.comfyui_url:
        typer.echo("image generation is not set up — run: "
                   "localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188", err=True)
        raise typer.Exit(2)
    if out is not None and out.suffix.lower() != ".png":
        typer.echo(f"--out must end in .png (got: {out})", err=True)
        raise typer.Exit(2)
    try:
        template, tpl_name = load_template(cfg.workflow)
    except (ValueError, OSError) as exc:
        typer.echo(f"workflow template unusable: {exc}", err=True)
        raise typer.Exit(2)
    dest = out or Path.cwd() / f"image-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}.png"
    w, h = _snap(width), _snap(height)
    seed = random.randrange(2**31) if seed is None else abs(int(seed)) % 2**63
    graph = build_graph(template, prompt=prompt, width=w, height=h, steps=steps, seed=seed,
                        prefix=f"lh_{secrets.token_hex(4)}")
    try:
        data, count, elapsed = asyncio.run(generate(cfg.comfyui_url, graph, cfg.timeout_s))
    except (httpx.ConnectError, httpx.ConnectTimeout):
        typer.echo(f"ComfyUI unreachable at {cfg.comfyui_url} — start it first "
                   "(localharness doctor checks it).", err=True)
        raise typer.Exit(1)
    except (httpx.HTTPError, RuntimeError, TimeoutError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)
    if sniff_mime(data) != "image/png":
        typer.echo(f"ComfyUI returned a file that is not a PNG — check the save node in template '{tpl_name}'", err=True)
        raise typer.Exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    note = f" ({count} images; saved the first)" if count > 1 else ""
    typer.echo(f"saved: {dest} ({w}x{h}, seed {seed}, {steps} steps, {elapsed:.1f}s, "
               f"template {tpl_name}){note}")
