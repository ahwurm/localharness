"""`localharness plugins` — see plugins, and turn them on and off (ENAB-03, PRD §4).

- `list`: every plugin — NAME / WHAT IT DOES / STATE / FROM — with the exact command that turns an
  off or available one on; on stderr, each section no installed plugin owns (most often a removed
  plugin's settings), with its file, line and fix (QA-06).
- `info NAME`: one plugin, what it adds, and every setting it owns: the dot-paths `components list`
  marks `(plugin: NAME)`, read through the same catalogue.
- `enable NAME [--set k=v …]` and `disable NAME` write `NAME.enabled` (and the settings) into ONE
  layer's overrides.yaml through the atomic overlay writer — the machine's, or with --workspace this
  project's — and never into a config.yaml. `enable NAME --set k=v` writes the same overlay as
  `enable NAME` followed by `components set NAME.k v`. On a terminal, with no --set, it asks the
  plugin's declared setup questions, writes the answers the same way, and runs the plugin's doctor
  check once.

A plugin you installed is turned on and off only in machine-level settings: turning it on is your
trust grant (SAFE-06), so --workspace is refused for it, as it is for a machine-level setting.
Nothing here imports a plugin that is not enabled, except `enable NAME --set …` for a plugin you
installed: checking the values needs its settings model, and that command is the grant.

Unlike `components set`, these commands emit no ComponentMutated audit event.
"""
from __future__ import annotations

import json as _json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn, Optional

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table

from localharness.cli.components_cmd import (
    _build_layered_loader, _err, _err_config, _serialize_value, is_secret, scrub, shown,
)
from localharness.cli.workspace import _stdin_is_a_terminal
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.config.plugin_sections import unowned_hint
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
# The same literal init's scaffolded plugins/README.md links to (a PyPI install has no examples/).
_TEMPLATE_URL = "https://github.com/ahwurm/localharness/tree/main/examples/plugin-template"


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


def _where(dotpath: str, file: Path, line: Optional[int]) -> str:
    """`example:` in <file> (line 1), `agent.example` in <file> (line 5)."""
    key = dotpath if dotpath.startswith("agent.") else f"{dotpath}:"
    return f"`{key}` in {file}" + (f" (line {line})" if line else "")


def _entry(resolution: Resolution, loader: ConfigLoader, name: str, json_output: bool) -> PlanEntry:
    entry = resolution.plan.entry(name)
    if entry is None:
        # QA-06: uninstalling leaves the plugin's settings behind; say where, and what to do.
        left = [s for s in loader.unowned_sections() if s[0] in (name, f"agent.{name}")]
        if left:
            _fail(f"Unknown plugin: {name!r} — no installed plugin has that name, but its settings "
                  "are still here:\n" + "".join(f"  {_where(*s)}\n" for s in left)
                  + "Delete them, or reinstall the plugin. Run `localharness plugins list` to see "
                  "the plugins installed here.", json_output)
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
    resolution, loader, _ = _resolve(config_dir, json_output=json_output)
    _warn(resolution)
    for dotpath, file, line in loader.unowned_sections():  # QA-06: never silent about a leftover
        hint = unowned_hint(dotpath.removeprefix("agent."), under_agent=dotpath.startswith("agent."))
        err_console.print("[yellow]⚠[/yellow] " + escape(f"{_where(dotpath, file, line)}: {hint}"),
                          soft_wrap=True)
    entries = resolution.plan.entries
    if json_output:
        typer.echo(_json.dumps([{"name": e.name, "what_it_does": e.summary, "state": e.display,
                                 "state_kind": e.state, "from": e.source,
                                 "enable_command": e.enable_command} for e in entries], indent=2))
        return
    if not entries:
        console.print(escape(  # QA-13: where it looked, and a URL a PyPI install can reach
            f"No plugins found in this Python environment ({sys.prefix}) or in "
            f"{loader.global_config_dir / 'plugins'}/. To write one, copy the example plugin: "
            f"{_TEMPLATE_URL}"), soft_wrap=True)
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
    """One plugin: its state, what it adds, and every setting it owns.

    The settings are the rows `components list` marks `(plugin: NAME)`.
    """
    resolution, loader, workspace = _resolve(config_dir, json_output=json_output)
    entry = _entry(resolution, loader, name, json_output)
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
    setup_cmd = _set_spelling(name, m.setup) if m is not None and m.setup else None
    if json_output:
        typer.echo(_json.dumps({
            "name": name, "state": entry.display, "state_kind": entry.state,
            "what_it_does": entry.summary, "from": entry.source, "enable_command": entry.enable_command,
            "version": m.version if m else None, "kind": m.kind if m else None,
            "requires": list(m.requires) if m else [], "uses": list(m.uses) if m else [],
            "cli": [d.name for d in m.cli] if m else [], "slash": [d.name for d in m.slash] if m else [],
            "settings": [{"path": p, "type": t, "current_value": _serialize_value(v), "layer": layer,
                          "machine_level_only": only} for p, t, v, layer, only in settings],
            "setup_command": setup_cmd, "note": note,
            "sections": list(m.sections) if m else []}, indent=2))
        return
    lines = [("state", entry.display), ("what it does", entry.summary), ("from", entry.source)]
    if m is not None:
        lines += [("version", m.version), ("kind", m.kind)] + [
            (label, ", ".join(values)) for label, values in (
                ("requires", m.requires), ("uses", m.uses),
                ("commands", [f"localharness {d.name}" for d in m.cli]),
                ("slash", [d.name for d in m.slash])) if values]
    if setup_cmd:
        lines.append(("set up", setup_cmd))
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
    if m is not None and m.sections:  # a bundled plugin's claimed core sections (PluginManifest.sections)
        *rest, last = [f"{s}:" for s in m.sections]
        console.print(escape(f"  {', '.join(rest)} and {last} keep their pre-plugin names" if rest
                             else f"  {last} keeps its pre-plugin name"), soft_wrap=True)


def _checked(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, pairs: list[str],
             to_workspace: bool, overlay: dict[str, Any]) -> dict[str, Any]:
    """Put each --set KEY=VALUE under `<name>.` in `overlay`, after checking the section the next
    start will read with the plugin's own ConfigModel; exit 2, writing nothing, on any refusal.
    Returns each key's value as it may be SHOWN (a secret leaf masked)."""
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
            _fail(scrub(f"Cannot coerce {shown(raw, leaves[key])} for {name}.{key}: {exc}",
                        [raw] if is_secret(leaves[key]) else []))
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
        _fail(scrub(f"Validation failed for {name}: {why}",
                    [v for _, v in walk_secret_values(leaves, overlay[name])]))
    return {key: shown(value, leaves[key]) for key, value in values.items()}


def walk_secret_values(leaves: dict[str, Any], section: dict[str, Any]):
    """(key, raw value) for every secret leaf `section` holds — scrubbed from error texts."""
    for key, ann in leaves.items():
        if is_secret(ann):
            node: Any = section
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            yield key, node


def _set_spelling(name: str, setup) -> str:
    """The non-interactive enable for a plugin's setup fields: `--set key=<default or <key>>` each."""
    return f"localharness plugins enable {name} " + " ".join(
        f"--set {f.key}={f.default or '<' + f.key + '>'}" for f in setup)


def _switch(name: str, on: bool, pairs: list[str], to_workspace: bool, config_dir: Optional[str],
            ask: bool = False) -> None:
    """Write `<name>.enabled: <on>`, and any --set values, into one layer's overrides.yaml. With
    `ask` (a terminal, no --set), the plugin's declared setup questions are asked first, their
    answers written as --set values, and its doctor check run once after the write."""
    resolution, loader, workspace = _resolve(config_dir, json_output=False)
    entry = _entry(resolution, loader, name, False)
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
    setup = entry.manifest.setup if (on and entry.manifest is not None) else ()
    if ask and setup:  # the answers go through the one checked write path, as --set values
        pairs = [f"{f.key}={typer.prompt(f.prompt, default=f.default, hide_input=f.secret)}"
                 for f in setup]
    overlay = load_overlay(target)
    set_value_in_dict(overlay, f"{name}.enabled", on)
    values = _checked(resolution, loader, entry, pairs, to_workspace, overlay) if pairs else {}
    atomic_write_overlay(target, overlay)
    console.print("[green]✓[/green] " + escape(
        f"{name} {verb}d in {target} — takes effect on the next `localharness start`"), soft_wrap=True)
    for key, value in values.items():
        console.print(escape(f"  set {name}.{key} = {value}"), soft_wrap=True)
    if on and not ask and entry.manifest is not None and entry.manifest.requires_extra:
        # "takes effect on the next start" is not true while its install extra is missing; the
        # interactive path says so through _probe's row, this one re-reads the plan it just wrote
        from localharness.config.loader import ConfigLoader as _Loader
        from localharness.plugins.resolve import resolve
        after = resolve(_Loader(config_dir=loader.global_config_dir, local_config_dir=workspace)).plan.entry(name)
        if after is not None and after.state == "needs-extra":
            console.print("  [yellow]note:[/yellow] " + escape(
                f"{name} is missing its install extra — {after.reason}"), soft_wrap=True)
    project = [s["enabled"] for s in loader.plugin_layers().get(name, (None,) * 4)[2:]
               if isinstance(s, dict) and isinstance(s.get("enabled"), bool)]
    if not to_workspace and entry.bundled and project and project[-1] is not on:
        console.print("  [yellow]note:[/yellow] " + escape(
            f"this project turns {name} {'off' if on else 'on'} ({workspace}), and that still wins "
            f"here — `localharness plugins {verb} {name} --workspace` changes it for this project"),
            soft_wrap=True)
    if ask and setup:
        console.print("Checking it now:")
        _probe(name, entry.manifest.setup_help, loader, workspace)
    elif on and setup and not pairs:
        console.print(escape("  next step — give it the " + ", ".join(f.prompt for f in setup)
                             + ": " + _set_spelling(name, setup)), soft_wrap=True)


def _probe(name: str, setup_help: str, loader: ConfigLoader, workspace: Any) -> None:
    """Run the plugin's doctor check once against what was just written, and print it with doctor's
    own row printer; `setup_help` follows when it does not pass. Never fails the enable (it wrote)."""
    import asyncio

    from localharness.cli.doctor_cmd import print_plugin_row
    from localharness.config.loader import ConfigLoader as _Loader
    from localharness.plugins.api import PluginPaths
    from localharness.plugins.lifecycle import DoctorRow, _doctor_row
    from localharness.plugins.resolve import resolve

    try:
        # A fresh loader over the SAME layers: re-deriving the workspace would print its trust
        # notice (and maybe ask) a second time.
        fresh = _Loader(config_dir=loader.global_config_dir, local_config_dir=workspace)
        resolution = resolve(fresh)
        entry = resolution.plan.entry(name)
        paths = PluginPaths(global_config_dir=fresh.global_config_dir, workspace=workspace,
                            state_dir=workspace if workspace is not None else fresh.global_config_dir)
        row = (asyncio.run(_doctor_row(resolution, name, paths)) if entry.state == "on"
               else DoctorRow(name, entry.state, entry.display))
        print_plugin_row(row, [])
    except Exception as exc:  # noqa: BLE001 — the enable already wrote; the check is advice
        console.print(escape(f"  could not check it now: {type(exc).__name__}: {exc}"), soft_wrap=True)
        return
    if setup_help and (row.state != "on" or any(c.status == "fail" for c in row.checks)):
        console.print()
        console.print(escape(setup_help), soft_wrap=True)


@plugins_app.command("enable")
def plugins_enable(
    name: Name,
    settings: Annotated[Optional[list[str]], typer.Option(
        "--set", metavar="KEY=VALUE",
        help="Also write one of its settings, the same as `components set NAME.KEY VALUE` "
             "(repeatable).")] = None,
    workspace: Workspace = False,
    no_input: Annotated[bool, typer.Option(
        "--no-input", help="Never ask the plugin's setup questions: write the switch (and any --set "
                           "values) only, and print the --set spelling of the rest.")] = False,
    config_dir: ConfigDir = None,
) -> None:
    """Turn a plugin on.

    Writes an overrides.yaml, never your config.yaml; takes effect on the next `localharness start`.
    On a terminal, with no --set, it first asks the plugin's setup questions, if it declares any.
    """
    _switch(name, True, settings or [], workspace, config_dir,
            ask=not settings and not no_input and not workspace and _stdin_is_a_terminal())


@plugins_app.command("disable")
def plugins_disable(name: Name, workspace: Workspace = False, config_dir: ConfigDir = None) -> None:
    """Turn a plugin off.

    Writes an overrides.yaml, never your config.yaml; takes effect on the next `localharness start`.
    """
    _switch(name, False, [], workspace, config_dir)
