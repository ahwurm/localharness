"""`localharness components` subapp: list, get, set with audit trail.

Phase 14 — REG-01..04. See:
  - 14-RESEARCH.md Example A for the end-to-end set flow
  - 14-CONTEXT.md for the locked CLI output format
  - 14-VALIDATION.md for the test contract
"""
from __future__ import annotations

import asyncio
import json as _json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from localharness.config.loader import ConfigLoader
from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import (
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.config.paths import resolve_runtime_path
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
    """ConfigLoader honoring the config-dir env chain (LOCALHARNESS_DIR > LOCALHARNESS_HOME >
    ~/.localharness). The precedence now lives in config/paths (#35), so no explicit env read
    here — a bare ConfigLoader() picks up the hermetic-test LOCALHARNESS_HOME just the same."""
    return ConfigLoader()


def _build_overlays(loader: ConfigLoader) -> dict[str, dict]:
    """Assemble overlays dict for build_catalogue's layer attribution.
    Phase 14 ships project + user layers; experiment layer added in Phase 17.
    """
    project_dict = loader.raw_harness_dict()
    user_dict = load_overlay(loader.user_overlay_path)
    return {"project": project_dict, "user": user_dict}


# ------------------------------------------------------------------ #
# Agent-scoped axes (issue #22)
#
# Agent axes live under the `agent.` namespace, which is a registry addressing
# convention — NOT a field of HarnessConfig (extra="forbid"). They validate against
# AgentConfig and are written to the user overlay's `agent:` section, the same layer
# `load_agent`/`get` read back. Mirrors autoresearch.adoption's validation split so
# both overlay-write paths share ONE contract.
# ------------------------------------------------------------------ #

_AGENT_KEY = "agent"
_AGENT_PREFIX = _AGENT_KEY + "."
# Placeholder identity so an agent-scoped overlay validates against AgentConfig (whose
# name/role are required + name is format-checked). Mirrors adoption._AGENT_VALIDATE_BASE.
_AGENT_VALIDATE_BASE = {"name": "components-validate", "role": "components-validate"}


def _validate_overlay(loader: ConfigLoader, path: str, new_overlay: dict) -> None:
    """Validate `new_overlay` against the model that OWNS `path`; raise ValidationError on failure.

    agent.* → AgentConfig (the merged `agent:` subtree); every other path → the merged
    HarnessConfig with the agent-scope `agent:` section EXCLUDED (it is not a HarnessConfig
    field). Before #22 every path validated against HarnessConfig, so agent.* always failed
    'Extra inputs are not permitted'; and once the overlay carries an `agent:` section, a later
    harness-path set would inherit that same failure unless `agent:` is excluded here too
    (mirrors load_harness's overlay handling).
    """
    if path.startswith(_AGENT_PREFIX):
        merged_agent = deep_merge(dict(_AGENT_VALIDATE_BASE), new_overlay.get(_AGENT_KEY, {}))
        AgentConfig.model_validate(merged_agent)
    else:
        harness_overlay = {k: v for k, v in new_overlay.items() if k != _AGENT_KEY}
        HarnessConfig.model_validate(deep_merge(loader.raw_harness_dict(), harness_overlay))


def _dig_dict(d: dict, dotpath: str) -> tuple[Any, bool]:
    """Walk a dot-path through a nested dict. Returns (value, True) if present, else (None, False)."""
    cur: Any = d
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def _apply_agent_overlay_values(catalogue: dict, user_overlay: dict) -> dict:
    """Reflect the user overlay's agent.* leaves in `current_value` so `get`/`list` read back
    exactly what `set agent.*` wrote (the round-trip). Winning_layer is already resolved by
    build_catalogue from the same overlay. Only paths EXPLICITLY present in the overlay are
    patched — every other agent.* axis keeps its compiled-in default (so we never leak the
    name-derived memory paths that model-validating a placeholder agent would introduce)."""
    agent_overlay = user_overlay.get(_AGENT_KEY) if isinstance(user_overlay, dict) else None
    if not isinstance(agent_overlay, dict):
        return catalogue
    for cat_path, entry in list(catalogue.items()):
        if not cat_path.startswith(_AGENT_PREFIX):
            continue
        value, found = _dig_dict(agent_overlay, cat_path[len(_AGENT_PREFIX):])
        if found:
            catalogue[cat_path] = replace(entry, current_value=value)
    return catalogue


def _build_tool_registry() -> Any:
    """Construct a ToolRegistry with builtin tools registered so the
    tools.*.description surface family appears in `components list`.

    Mirrors start_cmd.py's eager builtin registration but skips MCP / plugin
    instantiation since CLI inspection doesn't need live tool dispatch.
    Returns None on any failure (e.g. import error) — catalogue degrades
    gracefully to "no tools" rather than crashing the list command.
    """
    try:
        from localharness.tools.builtin import register_builtin_tools
        from localharness.tools.registry import ToolRegistry

        registry = ToolRegistry()
        # register_builtin_tools is async; run in a one-shot loop.
        try:
            asyncio.run(register_builtin_tools(registry))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(register_builtin_tools(registry))
            finally:
                loop.close()
        return registry
    except Exception:
        return None


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
    tool_registry = _build_tool_registry()
    catalogue = build_catalogue(cfg, overlays=overlays, tool_registry=tool_registry)
    catalogue = _apply_agent_overlay_values(catalogue, overlays["user"])

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
    tool_registry = _build_tool_registry()
    catalogue = build_catalogue(cfg, overlays=overlays, tool_registry=tool_registry)
    catalogue = _apply_agent_overlay_values(catalogue, overlays["user"])
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
    tool_registry = _build_tool_registry()
    catalogue = build_catalogue(cfg, overlays=overlays, tool_registry=tool_registry)
    catalogue = _apply_agent_overlay_values(catalogue, overlays["user"])
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

    # 4. Load existing overlay, deep-set new value, validate against the model that OWNS the path.
    #    agent.* validates against AgentConfig; everything else against the merged HarnessConfig
    #    (issue #22). No disk write happens until validation passes.
    overlay_path = loader.user_overlay_path
    existing_overlay = load_overlay(overlay_path)
    new_overlay = set_value_in_dict(dict(existing_overlay), path, typed_value)

    try:
        _validate_overlay(loader, path, new_overlay)
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

    # 7. Emit ComponentMutated audit event. Resolve the audit path against the loader's config
    # dir (#35 — a bare default 'audit.jsonl' lands under it; absolute/~ values honored as-is).
    audit_path = cfg.org.audit_log_path
    audit_path_resolved: Optional[Path] = (
        resolve_runtime_path(audit_path, loader._config_dir) if audit_path else None
    )

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
