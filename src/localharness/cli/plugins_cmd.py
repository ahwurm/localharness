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
    _build_layered_loader, _err, _err_config, _serialize_value, _validate_overlay, is_secret, scrub,
    shown,
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
    from localharness.plugins.api import PluginManifest, PluginPaths, SetupField
    from localharness.plugins.lifecycle import DoctorRow
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
    if m is not None and m.agent_prompt:  # the text view only: the --json key set is pinned
        from localharness.plugins.setup import render_agent_prompt
        console.print()
        console.print(escape(render_agent_prompt(
            m.agent_prompt, _prompt_values(resolution, loader, entry, m.agent_prompt))), soft_wrap=True)


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
        _fail(scrub(f"Validation failed for {name}: {_validation_text(exc)}",
                    [v for _, v in walk_secret_values(leaves, overlay[name])]))
    return {f"{name}.{key}": shown(value, leaves[key]) for key, value in values.items()}


def _validation_text(exc: BaseException) -> str:
    """Why a value was refused, without echoing any input: each pydantic error's location and
    message — never its input_value, which can hold a whole section, keys included."""
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" if e["loc"] else e["msg"]
                         for e in exc.errors())
    return f"{type(exc).__name__}: {exc}"


def _core_secret_values(loader: ConfigLoader, overlay: dict[str, Any]) -> list[str]:
    """Every string at a SecretStr leaf of the core settings — the global config.yaml merged with
    the overlay about to be written (proposer.api_key today) — so a refusal can scrub them. The
    literal default "none" is left out: scrubbing it would mask every "none" in the refusal."""
    from localharness.config.models import HarnessConfig
    from localharness.config.overlay import deep_merge
    merged = deep_merge(loader.raw_harness_dict(), {k: v for k, v in overlay.items() if k != "agent"})
    return [v for _k, v in walk_secret_values(dict(walk_model_fields(HarnessConfig)), merged)
            if isinstance(v, str) and v and v != "none"]


def _checked_core(loader: ConfigLoader, entry: PlanEntry, pairs: list[str], to_workspace: bool,
                  overlay: dict[str, Any]) -> dict[str, str]:
    """Put each KEY=VALUE whose head is one of a bundled plugin's `sections` at its own path in
    `overlay`. These are core settings (proposer.model, not <name>.proposer.model), so they are
    checked TOGETHER against the core settings, as `components set` checks one: the global
    config.yaml merged with the overlay about to be written. Values that are only valid together
    (a proposer needs both its base_url and its model) land in one write or none; exit 2, writing
    nothing, on any refusal. Returns {path: value as it may be shown}."""
    from localharness.config.models import HarnessConfig

    if to_workspace:
        keys = ", ".join(pair.partition("=")[0] for pair in pairs)
        _fail(f"{keys}: these are machine-level settings for {entry.name} — run it without --workspace")
    try:  # the settings these are checked against must load, as `components set` requires
        loader.load_harness()
    except Exception as exc:  # noqa: BLE001 — reported as `components` reports it, exit 2
        _err_config(False, exc)
    leaves = dict(walk_model_fields(HarnessConfig))
    out: dict[str, str] = {}
    typed_secrets: list[str] = []
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            _fail(f"--set takes KEY=VALUE, not {pair!r}")
        if key not in leaves:
            _fail(f"{entry.name} has no setting {key!r}")
        if is_secret(leaves[key]):
            typed_secrets.append(raw)
        try:
            value = coerce_value(raw, leaves[key])
        except ValueError as exc:
            _fail(scrub(f"Cannot coerce {shown(raw, leaves[key])} for {key}: {exc}", typed_secrets))
        set_value_in_dict(overlay, key, value)
        out[key] = shown(value, leaves[key])
    try:
        _validate_overlay(loader, pairs[0].partition("=")[0], overlay)
    except ValueError as exc:  # a pydantic ValidationError is one
        _fail(scrub(f"Validation failed for {entry.name}: {_validation_text(exc)}",
                    typed_secrets + _core_secret_values(loader, overlay)))
    return out


def _checked_all(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, pairs: list[str],
                 to_workspace: bool, overlay: dict[str, Any]) -> dict[str, str]:
    """`_checked` for the plugin's own keys, `_checked_core` for a bundled plugin's keys under one
    of its `sections` — so a plugin with no ConfigModel and only such keys is never told it has no
    settings to set. Returns {path: value as it may be shown}."""
    m = entry.manifest
    sections = m.sections if entry.bundled and m is not None else ()
    core = [pair for pair in pairs if pair.partition("=")[0].split(".")[0] in sections]
    own = [pair for pair in pairs if pair not in core]
    values = _checked(resolution, loader, entry, own, to_workspace, overlay) if own else {}
    return values | (_checked_core(loader, entry, core, to_workspace, overlay) if core else {})


def _stored(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, key: str) -> str:
    """The value KEY holds now, as the text a question offers: "" when it holds nothing. Never
    returns a secret's value (a SecretStr reads as ""). A key under one of a bundled plugin's
    `sections` is read from the core settings, any other from the plugin's own validated ones."""
    from pydantic import SecretStr
    try:
        m = entry.manifest
        if entry.bundled and m is not None and key.split(".")[0] in m.sections:
            node: Any = loader.load_harness().model_dump()
        else:
            settings = resolution.settings.get(entry.name)
            config = settings.config if settings is not None else None
            node = config.model_dump() if config is not None else {}
        for part in key.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None or isinstance(node, SecretStr):
            return ""
        return ",".join(map(str, node)) if isinstance(node, (list, tuple)) else str(node)
    except Exception:  # noqa: BLE001 — a default to offer, never a reason to stop
        return ""


def _ask_fields(fields: tuple[SetupField, ...], stored: Any) -> list[str]:
    """Ask each setup question on the terminal; KEY=VALUE for every answer given. A question that
    is not secret offers `stored(key)`, the value it holds now, else its default; a secret one
    offers nothing, so a stored token is never shown."""
    pairs = []
    for f in fields:
        default = "" if f.secret else (stored(f.key) or f.default)
        answer = typer.prompt(f.prompt, default=default, hide_input=f.secret,
                              show_default=bool(default)).strip()
        if answer:  # an empty answer is not written: Enter on a question with nothing to offer skips it
            pairs.append(f"{f.key}={answer}")
    return pairs


def _extra_missing(manifest: PluginManifest) -> bool:
    """Is the plugin's install extra missing? Asked through the very function the resolver uses
    (resolve()'s `extra_installed` keyword default), so this step and the plan never disagree —
    and the suite's one seam, `resolve.resolve.__kwdefaults__["extra_installed"]`, covers both."""
    from localharness.plugins import plan
    from localharness.plugins.resolve import resolve
    extra = manifest.requires_extra
    installed = (getattr(resolve, "__kwdefaults__", None) or {}).get("extra_installed",
                                                                     plan.extra_installed)
    return bool(extra) and not installed(extra)


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


def _paths(loader: ConfigLoader, workspace: Any) -> PluginPaths:
    """A plugin's paths outside a session, as doctor gives them: state in the workspace, if any."""
    from localharness.plugins.api import PluginPaths
    return PluginPaths(global_config_dir=loader.global_config_dir, workspace=workspace,
                       state_dir=workspace if workspace is not None else loader.global_config_dir)


def _not_clean(row: DoctorRow, field_unanswered: bool) -> bool:
    """Is this check row short of "set up"? Not on (unconfigured, failed and the rest), a failing
    check, or a skipped one while one of the plugin's questions has no stored value."""
    return (row.state != "on" or any(c.status == "fail" for c in row.checks)
            or (field_unanswered and any(c.status == "skip" for c in row.checks)))


def _prompt_values(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry,
                   template: str) -> dict[str, str]:
    """What a coding-agent prompt is filled with: each question that is not secret, its stored
    value else its default; the config dir; and the {machine} sentence. The GPU is asked for only
    when the template names {machine}, so nvidia-smi runs only then."""
    from localharness.plugins.setup import gpu_name, machine_sentence
    fields = entry.manifest.setup if entry.manifest is not None else ()
    return {f.key: _stored(resolution, loader, entry, f.key) or f.default
            for f in fields if not f.secret} | {
        "config_dir": str(loader.global_config_dir),
        "machine": machine_sentence(gpu_name()) if "{machine}" in template else ""}


def _switch(name: str, on: bool, pairs: list[str], to_workspace: bool, config_dir: Optional[str],
            ask: bool = False) -> None:
    """Write `<name>.enabled: <on>`, and any --set values, into one layer's overrides.yaml; turning
    a plugin on, run its setup step around that write. With `ask` (a terminal, no --set): its
    questions are asked first and the answers written as --set values; after the write its setup
    action runs (after its yes/no question, when it has one) and its doctor check runs whenever it
    declares questions or an action; setup_help and the filled-in coding-agent prompt follow a
    check that is not clean. Without `ask` nothing is asked and nothing runs: the next step says
    what to run. A plugin missing its install extra is asked nothing and not checked. next_steps
    print on every outcome, and the command exits 0 whether or not the check passes: it wrote."""
    import asyncio

    from localharness.cli.doctor_cmd import print_plugin_row
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.lifecycle import DoctorRow, setup_action_rows
    from localharness.plugins.resolve import resolve
    from localharness.plugins.setup import has_setup_action, render_agent_prompt

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
    m = entry.manifest if on else None
    # A missing extra is tested directly, not through the plan state: a default-on plugin the user
    # turned off reads `off` there, and would still be asked for settings it cannot use.
    missing = m is not None and _extra_missing(m)
    setup = m.setup if m is not None else ()
    cls = resolution.classes.get(name)
    has_action = m is not None and cls is not None and has_setup_action(cls)
    if ask and setup and not missing:  # the answers go through the one checked write path, as --set values
        pairs = _ask_fields(setup, lambda key: _stored(resolution, loader, entry, key))
    overlay = load_overlay(target)
    set_value_in_dict(overlay, f"{name}.enabled", on)
    values = _checked_all(resolution, loader, entry, pairs, to_workspace, overlay) if pairs else {}
    atomic_write_overlay(target, overlay)
    console.print("[green]✓[/green] " + escape(
        f"{name} {verb}d in {target} — takes effect on the next `localharness start`"), soft_wrap=True)
    for path, value in values.items():
        console.print(escape(f"  set {path} = {value}"), soft_wrap=True)
    fresh_loader = fresh = after = None
    if m is not None:
        # What was just written, read through a NEW loader over the same layers: ConfigLoader caches
        # its harness and raw sources, and re-deriving the workspace would print its trust notice
        # (and maybe ask) a second time.
        fresh_loader = ConfigLoader(config_dir=loader.global_config_dir, local_config_dir=workspace)
        try:
            fresh = resolve(fresh_loader)
            after = fresh.plan.entry(name)
        except Exception as exc:  # noqa: BLE001 — the enable already wrote; the rest is advice
            console.print(escape(f"  could not check it now: {type(exc).__name__}: {exc}"),
                          soft_wrap=True)
    if after is not None and m.requires_extra and after.state == "needs-extra":
        missing = True  # "takes effect on the next start" is not true without its install extra
        console.print("  [yellow]note:[/yellow] " + escape(
            f"{name} is missing its install extra — {after.reason}"), soft_wrap=True)
    project = [s["enabled"] for s in loader.plugin_layers().get(name, (None,) * 4)[2:]
               if isinstance(s, dict) and isinstance(s.get("enabled"), bool)]
    if not to_workspace and entry.bundled and project and project[-1] is not on:
        console.print("  [yellow]note:[/yellow] " + escape(
            f"this project turns {name} {'off' if on else 'on'} ({workspace}), and that still wins "
            f"here — `localharness plugins {verb} {name} --workspace` changes it for this project"),
            soft_wrap=True)
    step = ask and after is not None and not missing
    paths = _paths(fresh_loader, workspace) if fresh_loader is not None else None
    if step and has_action and after.state == "on" and (
            not m.setup_action or typer.confirm(m.setup_action, default=True)):
        rows = asyncio.run(setup_action_rows(fresh, name, paths))
        if rows:
            print_plugin_row(DoctorRow(name, "on", "", rows), [])
    if step and (setup or has_action):
        console.print("Checking it now:")
        row = _probe(name, fresh, paths)
        unanswered = any(not _stored(fresh, fresh_loader, after, f.key) for f in setup)
        if row is not None and _not_clean(row, unanswered):
            if m.setup_help:  # the one place it prints
                console.print()
                console.print(escape(m.setup_help), soft_wrap=True)
            if m.agent_prompt:
                console.print()
                console.print(escape(render_agent_prompt(m.agent_prompt, _prompt_values(
                    fresh, fresh_loader, after, m.agent_prompt))), soft_wrap=True)
    elif not ask and m is not None:
        if setup and not pairs:
            console.print(escape("  next step — give it the " + ", ".join(f.prompt for f in setup)
                                 + ": " + _set_spelling(name, setup)), soft_wrap=True)
        if has_action and m.setup_action:
            console.print(escape(f"  next step — run `localharness plugins enable {name}` "
                                 f"on a terminal to answer: {m.setup_action}"), soft_wrap=True)
    if m is not None and m.next_steps:
        for line in m.next_steps.splitlines():
            console.print(escape(f"  {line}"), soft_wrap=True)


def _probe(name: str, resolution: Resolution, paths: PluginPaths) -> DoctorRow | None:
    """Run the plugin's doctor check once against `resolution` — read after the write — and print
    it with doctor's own row printer. Returns that row, or None when it could not check. Never
    fails the enable (it wrote)."""
    import asyncio

    from localharness.cli.doctor_cmd import print_plugin_row
    from localharness.plugins.lifecycle import DoctorRow, _doctor_row

    try:
        entry = resolution.plan.entry(name)
        row = (asyncio.run(_doctor_row(resolution, name, paths)) if entry.state == "on"
               else DoctorRow(name, entry.state, entry.display))
        print_plugin_row(row, [])
        return row
    except Exception as exc:  # noqa: BLE001 — the enable already wrote; the check is advice
        console.print(escape(f"  could not check it now: {type(exc).__name__}: {exc}"), soft_wrap=True)
        return None


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
