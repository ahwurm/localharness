"""`localharness components` subapp: list, get, set with audit trail.

Phase 14 — REG-01..04. See:
  - 14-RESEARCH.md Example A for the end-to-end set flow
  - 14-CONTEXT.md for the locked CLI output format
  - 14-VALIDATION.md for the test contract
"""
from __future__ import annotations

import asyncio
import json as _json
import os
import re
from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from localharness.config.loader import ConfigLoader
from localharness.config.models import HarnessConfig
from localharness.config.overlay import (
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.registry import (
    SURFACE_FAMILIES,
    build_catalogue,
    coerce_value,
    set_value_in_dict,
)

components_app = typer.Typer(
    name="components",
    help="List, inspect, and mutate harness components (registry).",
    no_args_is_help=True,
)

console = Console()
err_console = Console(stderr=True)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _build_loader() -> ConfigLoader:
    """Honor LOCALHARNESS_HOME for hermetic tests (mirrors components_home fixture)."""
    home = os.environ.get("LOCALHARNESS_HOME")
    if home:
        return ConfigLoader(config_dir=Path(home))
    return ConfigLoader()


def _build_overlays(loader: ConfigLoader) -> dict[str, dict]:
    """Assemble overlays dict for build_catalogue's layer attribution.
    Phase 14 ships project + user layers; experiment layer added in Phase 17.
    """
    project_dict = loader.raw_harness_dict()
    user_dict = load_overlay(loader.user_overlay_path)
    return {"project": project_dict, "user": user_dict}


def _err(json_output: bool, message: str, exit_code: int = 2) -> None:
    """Print error to stderr (or JSON) and exit."""
    if json_output:
        typer.echo(_json.dumps({"error": message}), err=True)
    else:
        err_console.print(f"[bold red]Error:[/bold red] {message}")
    raise typer.Exit(code=exit_code)


def _serialize_value(value: Any) -> Any:
    """Render value for JSON output. Primitives passthrough; complex types -> repr."""
    if value is None or isinstance(value, (bool, int, float, str, list, dict)):
        return value
    return repr(value)


# ------------------------------------------------------------------ #
# list
# ------------------------------------------------------------------ #


@components_app.command("list")
def components_list(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
    layer: Optional[str] = typer.Option(
        None,
        "--layer",
        help="Filter to entries with this winning layer (default|project|user|experiment)",
    ),
) -> None:
    """List every mutable component with its current value and winning layer."""
    try:
        loader = _build_loader()
        cfg = loader.load_harness()
    except Exception as exc:
        _err(json_output, f"Failed to load config: {exc}", exit_code=2)
        return  # unreachable but satisfies type checker

    overlays = _build_overlays(loader)
    catalogue = build_catalogue(cfg, overlays=overlays)

    entries = list(catalogue.values())
    if layer is not None:
        entries = [e for e in entries if e.winning_layer == layer]

    entries.sort(key=lambda e: e.path)

    if json_output:
        payload = [
            {
                "path": e.path,
                "type": e.type_name,
                "current_value": _serialize_value(e.current_value),
                "layer": e.winning_layer,
            }
            for e in entries
        ]
        typer.echo(_json.dumps(payload, indent=2))
        return

    table = Table(title="LocalHarness Components", show_lines=False)
    table.add_column("path", style="cyan", no_wrap=True)
    table.add_column("type", style="dim")
    table.add_column("current value", overflow="fold")
    table.add_column("layer", style="green")
    for e in entries:
        table.add_row(e.path, e.type_name, repr(e.current_value), e.winning_layer)
    console.print(table)


# ------------------------------------------------------------------ #
# get
# ------------------------------------------------------------------ #


@components_app.command("get")
def components_get(
    path: str = typer.Argument(
        ...,
        help="Dot-path of the component (e.g. agent.stuck_detector.window_size)",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Print the resolved value of one component + its winning layer."""
    try:
        loader = _build_loader()
        cfg = loader.load_harness()
    except Exception as exc:
        _err(json_output, f"Failed to load config: {exc}", exit_code=2)
        return

    overlays = _build_overlays(loader)
    catalogue = build_catalogue(cfg, overlays=overlays)
    entry = catalogue.get(path)
    if entry is None:
        _err(
            json_output,
            f"Unknown path: {path!r}. Run `localharness components list` to see valid paths.",
            exit_code=2,
        )
        return

    if json_output:
        payload = {
            "path": entry.path,
            "value": _serialize_value(entry.current_value),
            "type": entry.type_name,
            "layer": entry.winning_layer,
            "default": _serialize_value(entry.default_value),
        }
        typer.echo(_json.dumps(payload))
        return

    console.print(f"{entry.path} = {entry.current_value!r}")
    console.print(f"  type:    {entry.type_name}")
    console.print(f"  layer:   {entry.winning_layer}")
    console.print(f"  default: {entry.default_value!r}")


# ------------------------------------------------------------------ #
# set
# ------------------------------------------------------------------ #

_MULTI_PATH_PATTERN = re.compile(r"[,\s;]")


@components_app.command("set")
def components_set(
    path: str = typer.Argument(
        ...,
        help="Dot-path (one only -- atomic set, no multi-path syntax)",
    ),
    value: str = typer.Argument(
        ...,
        help="New value as a string. Coerced to the path's target type.",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Mutate one component. Writes user overlay atomically; emits ComponentMutated audit event."""

    # 1. Refuse multi-path syntax (atomicity per CONTEXT.md + EXP-02 forward-compat)
    if _MULTI_PATH_PATTERN.search(path):
        _err(
            json_output,
            f"Atomic set: one path per invocation (got {path!r}). "
            f"Run `set` once per component.",
            exit_code=2,
        )
        return

    try:
        loader = _build_loader()
        cfg = loader.load_harness()
    except Exception as exc:
        _err(json_output, f"Failed to load config: {exc}", exit_code=2)
        return

    # 2. Build catalogue, resolve entry
    overlays = _build_overlays(loader)
    catalogue = build_catalogue(cfg, overlays=overlays)
    entry = catalogue.get(path)
    if entry is None:
        _err(json_output, f"Unknown path: {path!r}", exit_code=2)
        return

    # 3. Coerce CLI string to typed value
    try:
        typed_value = coerce_value(value, entry.annotation)
    except ValueError as exc:
        _err(
            json_output,
            f"Cannot coerce {value!r} to {entry.type_name}: {exc}",
            exit_code=2,
        )
        return

    before = entry.current_value

    # 4. Load existing overlay, deep-set new value, deep-merge with project, validate
    overlay_path = loader.user_overlay_path
    existing_overlay = load_overlay(overlay_path)
    new_overlay = set_value_in_dict(dict(existing_overlay), path, typed_value)

    project_dict = loader.raw_harness_dict()
    merged = deep_merge(project_dict, new_overlay)

    try:
        HarnessConfig.model_validate(merged)
    except ValidationError as exc:
        _err(
            json_output,
            f"Validation failed for {path}={typed_value!r}: {exc}",
            exit_code=2,
        )
        return

    # 5. Atomic write -- overlay does NOT touch disk until this succeeds
    try:
        atomic_write_overlay(overlay_path, new_overlay)
    except Exception as exc:
        _err(
            json_output,
            f"Failed to write overlay {overlay_path}: {exc}",
            exit_code=2,
        )
        return

    # 6. Invalidate loader cache so next read sees the new value
    loader.invalidate_cache()

    # 7. Emit ComponentMutated audit event
    audit_path = cfg.org.audit_log_path
    audit_path_resolved: Optional[Path] = None
    if audit_path:
        audit_path_resolved = Path(audit_path).expanduser()

    bus = EventBus(persist_path=audit_path_resolved)
    event = ComponentMutated(
        path=path,
        before_value=_serialize_value(before),
        after_value=_serialize_value(typed_value),
        layer="user",
        actor="cli",
    )
    try:
        asyncio.run(bus.publish(event))
    except RuntimeError:
        # Already inside an event loop (rare for CLI but possible under pytest-asyncio)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(bus.publish(event))
        finally:
            loop.close()

    # 8. Confirm
    if json_output:
        typer.echo(
            _json.dumps(
                {
                    "path": path,
                    "before": _serialize_value(before),
                    "after": _serialize_value(typed_value),
                    "layer": "user",
                }
            )
        )
    else:
        console.print(
            f"[green]set[/green] {path} = {typed_value!r} "
            f"(was: {before!r}, layer: user)"
        )

    # Reference SURFACE_FAMILIES to keep the import live (avoids unused-warning if linted)
    _ = SURFACE_FAMILIES
