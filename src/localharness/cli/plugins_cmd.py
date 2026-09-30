"""`localharness plugins` — see plugins, and turn them on and off (ENAB-03, PRD §4).

- `list`: every plugin — NAME / WHAT IT DOES / STATE / FROM — with the exact command that turns an
  off or available one on.
- `info NAME`: one plugin, what it adds, and every setting it owns: the dot-paths `components list`
  marks `(plugin: NAME)`, read through the same catalogue.
- `enable NAME [--set k=v …]` and `disable NAME` write `NAME.enabled` (and the settings) into ONE
  layer's overrides.yaml through the atomic overlay writer — the machine's, or with --workspace this
  project's — and never into a config.yaml. `enable NAME --set k=v` writes the same overlay as
  `enable NAME` followed by `components set NAME.k v`.

A plugin you installed is turned on and off only in machine-level settings: turning it on is your
trust grant (SAFE-06), so --workspace is refused for it, as it is for a machine-level setting.
Nothing here imports a plugin that is not enabled, except `enable NAME --set …` for a plugin you
installed: checking the values needs its settings model, and that command is the grant.

Unlike `components set`, these commands emit no ComponentMutated audit event.
"""
from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Annotated, Any, NoReturn, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from localharness.cli.components_cmd import (
    _build_layered_loader, _err, _err_config, _serialize_value,
)
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.registry import (
    LAYER_DEFAULT, LAYER_GLOBAL_CONFIG, LAYER_GLOBAL_OVERRIDES, LAYER_WORKSPACE_CONFIG,
    LAYER_WORKSPACE_OVERRIDES, coerce_value, set_value_in_dict, walk_model_fields,
)
from localharness.registry.catalogue import plugin_catalogue_rows
from localharness.registry.provenance import layered_catalogue

if TYPE_CHECKING:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.plan import PlanEntry
    from localharness.plugins.resolve import Resolution

plugins_app = typer.Typer(name="plugins", help="See, enable and disable plugins.", no_args_is_help=True)

console = Console()
err_console = Console(stderr=True)

ConfigDir = Annotated[Optional[str], typer.Option(
    "--config-dir", envvar="LOCALHARNESS_DIR", show_default=False,
    help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.")]
Json = Annotated[bool, typer.Option("--json", help="Emit JSON instead of text")]
Name = Annotated[str, typer.Argument(help="The plugin's name, as `plugins list` shows it.")]
Workspace = Annotated[bool, typer.Option(
    "--workspace", help="Write this project's overrides.yaml instead of the machine's. Plugins that "
                        "ship with LocalHarness only: a plugin you installed is switched machine-wide.")]
_BANDS = (LAYER_GLOBAL_CONFIG, LAYER_GLOBAL_OVERRIDES, LAYER_WORKSPACE_CONFIG, LAYER_WORKSPACE_OVERRIDES)


def _fail(message: str, json_output: bool = False) -> NoReturn:
    """components' `_err` (stderr, or JSON; exit 2), typed as the exit it is."""
    _err(json_output, message)
    raise AssertionError("unreachable: _err exits")


def _resolve(config_dir: Optional[str], *, json_output: bool
             ) -> tuple[Resolution, ConfigLoader, Any]:
    """(resolution, loader, workspace) for this invocation — resolved once per command."""
    from localharness.plugins.resolve import resolve  # lazy: `localharness --help` never pays for it

    loader, workspace = _build_layered_loader(config_dir, json_output=json_output)
    try:
        return resolve(loader), loader, workspace
    except Exception as exc:  # an unreadable config, reported as `components` reports it
        _err_config(json_output, exc)
        raise  # unreachable: _err_config exits


def _entry(resolution: Resolution, name: str, json_output: bool) -> PlanEntry:
    entry = resolution.plan.entry(name)
    if entry is None:
        _fail(f"Unknown plugin: {name!r}. Run `localharness plugins list` to see the plugins "
              "installed here.", json_output)
    return entry


def _warn(resolution: Resolution) -> None:
    """The resolver's warnings, on stderr — never inside the table or the JSON."""
    for warning in resolution.warnings:
        err_console.print("[yellow]⚠[/yellow] " + escape(warning), soft_wrap=True)


@plugins_app.command("list")
def plugins_list(json_output: Json = False, config_dir: ConfigDir = None) -> None:
    """Every plugin: what it does, whether it is on (and how to turn it on), and where it is from."""
    resolution, _, _ = _resolve(config_dir, json_output=json_output)
    _warn(resolution)
    entries = resolution.plan.entries
    if json_output:
        typer.echo(_json.dumps([{"name": e.name, "what_it_does": e.summary, "state": e.display,
                                 "state_kind": e.state, "from": e.source,
                                 "enable_command": e.enable_command} for e in entries], indent=2))
        return
    if not entries:
        console.print("No plugins installed — to write one, copy examples/plugin-template/.")
        return
    table = Table(box=None, header_style="bold", pad_edge=False)
    # A narrow terminal wraps cells, never cuts them (measured at 80 columns: a whole STATE column
    # left the others 5 characters wide); `info` and --json carry the enable command whole.
    for column in ("NAME", "WHAT IT DOES", "STATE", "FROM"):
        table.add_column(column, no_wrap=column == "NAME", overflow="fold")
    for e in entries:
        table.add_row(*(escape(cell) for cell in (e.name, e.summary, e.display, e.source)))
    console.print(table)


def _switch_layer(loader: ConfigLoader, name: str, bundled: bool) -> str:
    """The layer `<name>.enabled` is read from: the four for a bundled plugin, the machine's two for
    a plugin you installed (the resolver's rule)."""
    sections = loader.plugin_layers().get(name, (None,) * 4)[: 4 if bundled else 2]
    return next((band for band, section in reversed(list(zip(_BANDS, sections)))
                 if isinstance(section, dict) and "enabled" in section), LAYER_DEFAULT)


@plugins_app.command("info")
def plugins_info(name: Name, json_output: Json = False, config_dir: ConfigDir = None) -> None:
    """One plugin: its state, what it adds, and every setting it owns — the rows `components list`
    marks `(plugin: NAME)`."""
    resolution, loader, workspace = _resolve(config_dir, json_output=json_output)
    entry = _entry(resolution, name, json_output)
    _warn(resolution)
    rows = plugin_catalogue_rows(resolution)
    try:
        catalogue, _ = layered_catalogue(loader.global_config_dir, workspace, loader=loader, plugins=rows)
    except Exception as exc:
        _err_config(json_output, exc)
        raise  # unreachable: _err_config exits
    own = next((r for r in rows if r.name == name), None)
    machine = {f"{name}.{path}" for path in own.global_only} if own is not None else set()
    settings = [(e.path, e.type_name, e.current_value, e.winning_layer, e.path in machine)
                for e in sorted(catalogue.values(), key=lambda e: e.path) if e.plugin == name]
    note = None
    if not settings and name in resolution.enabled:  # nothing of it loaded to list: its switch only
        settings = [(f"{name}.enabled", "bool", resolution.enabled[name],
                     _switch_layer(loader, name, entry.bundled), not entry.bundled)]
        note = ("Its other settings are listed once it is enabled." if entry.state == "available"
                else "Its other settings are listed once it loads.")
    m = entry.manifest
    if json_output:
        typer.echo(_json.dumps({
            "name": name, "state": entry.display, "state_kind": entry.state,
            "what_it_does": entry.summary, "from": entry.source, "enable_command": entry.enable_command,
            "version": m.version if m else None, "kind": m.kind if m else None,
            "requires": list(m.requires) if m else [], "uses": list(m.uses) if m else [],
            "cli": [d.name for d in m.cli] if m else [], "slash": [d.name for d in m.slash] if m else [],
            "settings": [{"path": p, "type": t, "current_value": _serialize_value(v), "layer": layer,
                          "machine_level_only": only} for p, t, v, layer, only in settings],
            "note": note}, indent=2))
        return
    lines = [("state", entry.display), ("what it does", entry.summary), ("from", entry.source)]
    if m is not None:
        lines += [("version", m.version), ("kind", m.kind)] + [
            (label, ", ".join(values)) for label, values in (
                ("requires", m.requires), ("uses", m.uses),
                ("commands", [f"localharness {d.name}" for d in m.cli]),
                ("slash", [d.name for d in m.slash])) if values]
    console.print(escape(name), soft_wrap=True)
    for label, value in lines:
        console.print(escape(f"  {label + ':':<14}{value}"), soft_wrap=True)
    console.print("Settings:")
    table = Table(box=None, show_header=False, pad_edge=False)
    for path, type_name, value, layer, only in settings:
        table.add_row(*(escape(cell) for cell in (
            path, type_name, repr(value), f"[{layer}]" + (" (machine-level only)" if only else ""))))
    console.print(Padding(table, (0, 0, 0, 2), expand=False))
    if note:
        console.print(f"  {note}")


def _checked(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, pairs: list[str],
             to_workspace: bool, overlay: dict[str, Any]) -> dict[str, Any]:
    """Put each --set KEY=VALUE under `<name>.` in `overlay`, after checking the section the next
    start will read with the plugin's own ConfigModel; exit 2, writing nothing, on any refusal."""
    from localharness.config.plugin_sections import merge_plugin_layers
    from localharness.plugins import discovery
    from localharness.plugins.resolve import _machine_only

    name, cls = entry.name, resolution.classes.get(entry.name)
    if cls is None and not entry.bundled:
        # A plugin you installed that is not on yet: its model is in its code, and this command is
        # the trust grant (SAFE-06). A plain `enable` imports nothing.
        found = next(d for d in discovery.discover(loader.global_config_dir) if d.name == name)
        try:
            cls = discovery.load_plugin_class(found)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 — contained, as the resolver does
            _fail(f"cannot check the --set values: {name} could not be imported: "
                  f"{type(exc).__name__}: {exc}")
    model = cls.ConfigModel if cls is not None else None
    if model is None:
        _fail(f"{name} has no settings to set")
    leaves, machine = dict(walk_model_fields(model)), _machine_only(model, entry.bundled)
    values: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            _fail(f"--set takes KEY=VALUE, not {pair!r}")
        if key not in leaves:
            _fail(f"{name} has no setting {key!r} — its settings: {', '.join(sorted(leaves))}")
        if to_workspace and key in machine:
            _fail(f"{name}.{key} is a machine-level setting — set it without --workspace")
        try:
            values[key] = coerce_value(raw, leaves[key])
        except ValueError as exc:
            _fail(f"Cannot coerce {raw!r} for {name}.{key}: {exc}")
        set_value_in_dict(overlay, f"{name}.{key}", values[key])
    layers = list(loader.plugin_layers().get(name, (None,) * 4))
    # The written layer as it will read; a machine value holds in every project, so it is checked
    # with the machine's layers alone (as `components set` checks it).
    layers[1:] = [layers[1], layers[2], overlay[name]] if to_workspace else [overlay[name], None, None]
    try:
        section, _ = merge_plugin_layers(name, layers, global_only=machine,
                                         layer_files=loader.plugin_layer_files())
        model.model_validate({k: v for k, v in section.items() if k != "enabled"})
    except (Exception, SystemExit) as exc:  # noqa: BLE001 — the plugin's own validator: contained
        why = exc if isinstance(exc, ValidationError) else f"{type(exc).__name__}: {exc}"
        _fail(f"Validation failed for {name}: {why}")
    return values


def _switch(name: str, on: bool, pairs: list[str], to_workspace: bool, config_dir: Optional[str]) -> None:
    """Write `<name>.enabled: <on>`, and any --set values, into one layer's overrides.yaml."""
    resolution, loader, workspace = _resolve(config_dir, json_output=False)
    entry = _entry(resolution, name, False)
    word, verb = ("on", "enable") if on else ("off", "disable")
    if name not in resolution.enabled:  # a refused NAME (invalid, a core key, taken): its key is
        _fail(f"{name} cannot be turned {word}: {entry.reason}")  # not this plugin's to write
    if to_workspace and not entry.bundled:
        _fail(f"a plugin you installed is turned {word} only in machine-level settings — run "
              f"`localharness plugins {verb} {name}` without --workspace")
    if to_workspace and workspace is None:
        _fail("no project workspace applies here — run `localharness init --workspace` in the "
              "project first, or leave out --workspace to change the machine-level setting")
    # Each layer's overrides.yaml exactly as config/loader.py reads it.
    target = workspace / "overrides.yaml" if to_workspace else loader.user_overlay_path
    overlay = load_overlay(target)
    set_value_in_dict(overlay, f"{name}.enabled", on)
    values = _checked(resolution, loader, entry, pairs, to_workspace, overlay) if pairs else {}
    atomic_write_overlay(target, overlay)
    console.print("[green]✓[/green] " + escape(
        f"{name} {verb}d in {target} — takes effect on the next `localharness start`"), soft_wrap=True)
    for key, value in values.items():
        console.print(escape(f"  set {name}.{key} = {value!r}"), soft_wrap=True)
    project = [s["enabled"] for s in loader.plugin_layers().get(name, (None,) * 4)[2:]
               if isinstance(s, dict) and isinstance(s.get("enabled"), bool)]
    if not to_workspace and entry.bundled and project and project[-1] is not on:
        console.print("  [yellow]note:[/yellow] " + escape(
            f"this project turns {name} {'off' if on else 'on'} ({workspace}), and that still wins "
            f"here — `localharness plugins {verb} {name} --workspace` changes it for this project"),
            soft_wrap=True)


@plugins_app.command("enable")
def plugins_enable(
    name: Name,
    settings: Annotated[Optional[list[str]], typer.Option(
        "--set", metavar="KEY=VALUE",
        help="Also write one of its settings, the same as `components set NAME.KEY VALUE` "
             "(repeatable).")] = None,
    workspace: Workspace = False,
    config_dir: ConfigDir = None,
) -> None:
    """Turn a plugin on. Writes an overrides.yaml, never your config.yaml; takes effect on the next
    `localharness start`."""
    _switch(name, True, settings or [], workspace, config_dir)


@plugins_app.command("disable")
def plugins_disable(name: Name, workspace: Workspace = False, config_dir: ConfigDir = None) -> None:
    """Turn a plugin off. Writes an overrides.yaml, never your config.yaml; takes effect on the next
    `localharness start`."""
    _switch(name, False, [], workspace, config_dir)
