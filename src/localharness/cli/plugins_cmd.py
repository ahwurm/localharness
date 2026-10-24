"""`localharness plugins` — see plugins, and turn them on and off (ENAB-03, PRD §4).

- `list`: every plugin — NAME / WHAT IT DOES / STATE / FROM — with the exact command that turns an
  off or available one on; on stderr, each section no installed plugin owns (most often a removed
  plugin's settings), with its file, line and fix (QA-06).
- `info NAME`: one plugin, what it adds, and every setting it owns: the dot-paths `components list`
  marks `(plugin: NAME)`, read through the same catalogue.
- `enable NAME [--set k=v …]` and `disable NAME` write `NAME.enabled` (and the settings) into ONE
  layer's overrides.yaml through the atomic overlay writer — the machine's, or with --workspace this
  project's — and never into a config.yaml. `enable NAME --set k=v` writes the same overlay as
  `enable NAME` followed by `components set NAME.k v`. On a terminal, with no --set, `enable` runs
  the plugin's setup step (`_switch`): its questions, the one write, its setup action, its doctor
  check and its next steps. `session_step` runs the same step for `/plugins enable|disable NAME`,
  between the two halves of a terminal session's restart; `switch_decision` decides, before the
  session tears down, whether that restart happens at all, and `plugins_overview` is bare `/plugins`.

A plugin you installed is turned on and off only in machine-level settings: turning it on is your
trust grant (SAFE-06), so --workspace is refused for it, as it is for a machine-level setting.
Nothing here imports a plugin that is not enabled, except `enable NAME --set …` for a plugin you
installed: checking the values needs its settings model, and that command is the grant.

Unlike `components set`, these commands emit no ComponentMutated audit event.
"""
from __future__ import annotations

import asyncio
import importlib
import json as _json
import os
import shutil
import signal
import sys
import tempfile
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Mapping, NoReturn, Optional
from urllib.parse import urlsplit

import click
import typer
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from localharness.channels.base import sanitize_for_display
from localharness.cli import update_cmd
from localharness.cli.components_cmd import (
    Refusal, _build_layered_loader, _core_secret_values, _err, _err_config, _serialize_value,
    _validate_overlay, _validation_text, is_secret, redact_text, shown, walk_secret_values,
)
from localharness.cli.interview import confirm_line, prompt_line
from localharness.cli.workspace import _stdin_is_a_terminal
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.config.plugin_sections import unowned_hint
from localharness.core.private_files import OPEN_TO_OTHERS, ensure_private_dir
from localharness.registry import LAYER_DEFAULT, coerce_value, set_value_in_dict, walk_model_fields
from localharness.registry.catalogue import LAYER_BANDS, plugin_catalogue_rows
from localharness.registry.provenance import layered_catalogue

if TYPE_CHECKING:
    from localharness.cli.update_cmd import InstallRoute, Manual
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
# Asked once, in a session, before turning on a plugin the user installed: it is machine-wide trust
# in code that runs with their permissions (SAFE-06). Bundled plugins and the shell command never ask.
THIRD_PARTY_CONFIRM = ("{name} is a plugin you installed, not one that ships with LocalHarness: turning "
                       "it on lets its code run with your permissions on this machine, in every "
                       "session. Turn it on?")
# The same literal init's scaffolded plugins/README.md links to (a PyPI install has no examples/).
_TEMPLATE_URL = "https://github.com/ahwurm/localharness/tree/main/examples/plugin-template"
# A setup question for a secret that is already stored (#28): what Enter does, and how to clear it.
SECRET_KEPT = "stored key kept — type none to clear, or paste a new one"
SECRET_KEPT_FOR = "stored key for {host} kept — type none to clear, or paste a new one"
CTRLC = "Ctrl-C {does}"
CTRLC_DOES = MappingProxyType({"prompt": "cancels this enable — nothing is written", "batch": "stops the batch"})
CTRLC_IGNORED = "Ctrl-C is ignored in this session; this {step} cannot be stopped from the keyboard"
INSTALLING = "{installer} installing {spec} — output is shown only if it fails"
CONSENT = ("Enable {names}",
           "{installer} installs {spec} from your package index and runs its build steps.{with}",
           "{name}: {question}",
           "{command}")
CONSENT_WITH = " Packages added with uv tool install --with are dropped."
CONSENT_ASK = "n = enable without installing or downloading · {ctrlc}"
INSTALL_SKIPPED = ("{names}: enabled in config, package not installed{why}",
                   "To finish later, in another terminal run:",
                   "{command}",
                   "then {route}")
INSTALL_SKIPPED_WHY = MappingProxyType({
    "declined": "",
    "failed": " — the install failed (exit {code}; see the output above)",
    "not_writable": " — {dir} is not writable",
    "open": " — {dir} is open to other accounts and could not be made private",
    "no_temp": " — no temp dir under {dir}"})
ROUTE_BACK = "{rerun} under /plugins → {names}"
ENABLE_MANUAL = "{name}: enabled in config; this install must be done by hand."
TYPED_NEEDS_EXTRA = "{name}: needs {extra}."
TYPED_ROUTE = "/plugins enable {name} again"
_TERMINATE_GRACE = 5.0
_READ_SIZE = 4096
_KEPT_LINES = 30


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
    return next((band for band, section in reversed(list(zip(LAYER_BANDS, sections)))
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
    start will read with the plugin's own ConfigModel; raise Refusal, writing nothing, on any refusal.
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
            raise Refusal(f"cannot check the --set values: {name} could not be imported: "
                          f"{type(exc).__name__}: {exc}", ())
    model = cls.ConfigModel if cls is not None else None
    if model is None:
        raise Refusal(f"{name} has no settings to set", ())
    leaves, machine = dict(walk_model_fields(model)), _machine_only(model, entry.bundled)
    values: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep:
            raise Refusal(f"--set takes KEY=VALUE, not {pair!r}", ())
        if key not in leaves:
            raise Refusal(f"{name} has no setting {key!r} — its settings: {', '.join(sorted(leaves))}", ())
        if to_workspace and key in machine:
            raise Refusal(f"{name}.{key} is a machine-level setting — set it without --workspace", ())
        try:
            values[key] = coerce_value(raw, leaves[key])
        except ValueError as exc:
            raise Refusal(f"Cannot coerce {shown(raw, leaves[key])} for {name}.{key}: {exc}",
                          [raw] if is_secret(leaves[key]) else [])
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
        # Its message can quote the section it was given: scrub every layer's secrets, config.yaml's
        # included, not only the overlay's.
        raise Refusal(f"Validation failed for {name}: {_validation_text(exc)}",
                      [v for layer in layers if isinstance(layer, dict)
                       for _, v in walk_secret_values(leaves, layer)])
    return {f"{name}.{key}": shown(value, leaves[key]) for key, value in values.items()}


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
        raise Refusal(f"{keys}: these are machine-level settings for {entry.name} — run it without --workspace", ())
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
            raise Refusal(f"--set takes KEY=VALUE, not {pair!r}", ())
        if key not in leaves:
            raise Refusal(f"{entry.name} has no setting {key!r}", ())
        if is_secret(leaves[key]):
            typed_secrets.append(raw)
        try:
            value = coerce_value(raw, leaves[key])
        except ValueError as exc:
            raise Refusal(f"Cannot coerce {shown(raw, leaves[key])} for {key}: {exc}", typed_secrets)
        set_value_in_dict(overlay, key, value)
        out[key] = shown(value, leaves[key])
    try:
        _validate_overlay(loader, pairs[0].partition("=")[0], overlay)
    except ValueError as exc:  # a pydantic ValidationError is one
        raise Refusal(f"Validation failed for {entry.name}: {_validation_text(exc)}",
                      typed_secrets + _core_secret_values(loader, overlay))
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


def _stored_node(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, key: str) -> Any:
    """What KEY holds now, as the validated settings hold it (a secret as its SecretStr); None when
    nothing does. A key under one of a bundled plugin's `sections` is read from the core settings,
    any other from the plugin's own validated ones."""
    m = entry.manifest
    if entry.bundled and m is not None and key.split(".")[0] in m.sections:
        node: Any = loader.load_harness().model_dump()
    else:
        settings = resolution.settings.get(entry.name)
        config = settings.config if settings is not None else None
        node = config.model_dump() if config is not None else {}
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def _stored(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, key: str) -> str:
    """The value KEY holds now, as the text a question offers: "" when it holds nothing. Never
    returns a secret's value (a SecretStr reads as "")."""
    from pydantic import SecretStr
    try:
        node = _stored_node(resolution, loader, entry, key)
        if node is None or isinstance(node, SecretStr):
            return ""
        return ",".join(map(str, node)) if isinstance(node, (list, tuple)) else str(node)
    except Exception:  # noqa: BLE001 — a default to offer, never a reason to stop
        return ""


def _cleared(resolution: Resolution, entry: PlanEntry, key: str) -> str:
    """KEY's own default as text — what `none` at its question writes ("none" for proposer.api_key,
    "" for discord.token): a bundled plugin's `sections` key from the core settings, any other from
    the plugin's ConfigModel; "" when it has none."""
    from localharness.config.loader import _UNSET, _model_default
    from localharness.config.models import HarnessConfig
    try:
        m, parts = entry.manifest, key.split(".")
        if entry.bundled and m is not None and parts[0] in m.sections:
            value = _model_default(HarnessConfig, parts)
        else:
            value = _model_default(resolution.classes[entry.name].ConfigModel, parts)
        return "" if value is _UNSET or value is None else str(value)
    except Exception:  # noqa: BLE001 — no default to name: clearing writes empty
        return ""


def _secret_stored(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry, key: str) -> bool:
    """Does the secret KEY hold a value now — set, not empty, not its own default? Yes or no only:
    the value is never returned or shown."""
    from pydantic import SecretStr
    try:
        node = _stored_node(resolution, loader, entry, key)
        return isinstance(node, SecretStr) and node.get_secret_value() not in (
            "", _cleared(resolution, entry, key))
    except Exception:  # noqa: BLE001 — unknown reads as nothing stored: the plain question
        return False


async def ask_fields(fields: tuple[SetupField, ...], stored: Any, secret_stored: Any = lambda key: False,
                     cleared: Any = lambda key: "", *, prompt: Any) -> list[str]:
    """Ask each setup question through `prompt` (None raises click.Abort); KEY=VALUE for every answer given. A question that
    is not secret offers `stored(key)`, the value it holds now, else its default; a secret one
    offers nothing, so a stored token is never shown. When a secret is stored (`secret_stored`), its
    question says so: Enter keeps it and writes nothing (R5), `none` writes `cleared(key)` (the
    setting's own default), anything else is the new value; and when an address asked earlier in
    the same step was just changed, the question names the host the stored key was for."""
    pairs = []
    changed_from = ""
    for f in fields:
        was = "" if f.secret else stored(f.key)
        default = "" if f.secret else (was or f.default)
        text = f.prompt
        if f.secret and secret_stored(f.key):
            text += f" [{SECRET_KEPT_FOR.format(host=changed_from) if changed_from else SECRET_KEPT}]"
        answer = await prompt(sanitize_for_display(f"{text}: "), default=default, secret=f.secret)
        if answer is None:
            raise click.Abort()
        if not f.secret and f.key.split(".")[-1].endswith("url") and was and answer and answer != was:
            changed_from = urlsplit(was).hostname or ""
        if f.secret and answer.lower() == "none":
            pairs.append(f"{f.key}={cleared(f.key)}")
        elif answer:  # an empty answer is not written: Enter on a question with nothing to offer skips it
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


def extra_for(entry: PlanEntry) -> str | None:
    """The extra a bundled plugin needs and this install lacks and can provide; None otherwise."""
    m = entry.manifest
    if m is None or not entry.bundled or not m.requires_extra or not _extra_missing(m):
        return None
    return m.requires_extra if update_cmd.install_route(m.requires_extra) is not None else None


def install_route_for(entry: PlanEntry) -> InstallRoute | None:
    extra = extra_for(entry)
    return update_cmd.install_route(extra) if extra else None


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


@dataclass(frozen=True)
class StepOutcome:
    """What one plugin step did, for the session that resumes after `/plugins enable|disable`
    (cli/start_cmd): `failed_check` is the first check row that failed ("" when none did, or the
    check did not run); `skipped_check` is the first skipped row's detail when no row failed and
    the check was not clean, a plugin not set up yet (mobile's "not enrolled yet"), not a failure;
    `stopped` is True when Ctrl-C, Ctrl-D or a refusal ended the step early."""
    name: str
    on: bool
    failed_check: str = ""
    skipped_check: str = ""
    stopped: bool = False


Kind = Literal[
    "enabled", "enabled_unconfigured", "check_failed", "check_failed_noload", "check_skipped",
    "enabled_not_loading", "rerun_ok", "rerun_failed", "rerun_blocked", "disabled", "saved",
    "saved_overridden", "override", "declined", "refused", "install_skipped", "enabled_manual"]
Phase = Literal["consent", "install", "question", "question_installed", "download", "check", "edit",
                "disable"]
TailKey = Phase | Literal["install_batch", "edit_written"]
ReportKind = Literal["wrote", "note", "checking", "row", "help", "next_steps", "stays_off", "refused"]

ACTION_ENABLE = "Enable"
ACTION_INSTALL = "Install and set up"
ACTION_SHOW_INSTALL = "Show install step"
ACTION_CONFIGURE = "Configure…"
ACTION_RERUN = "Re-run setup"
ACTION_DISABLE = "Disable"
ACTION_INFO = "Info"


@dataclass
class PassProgress:
    """The one mutable holder: the producer it is handed sets these as it goes."""
    phase: Phase = "consent"
    written: bool = False
    wrote: bool = False
    answered: bool = False
    overridden: bool = False
    rerun: bool = False


@dataclass(frozen=True)
class PassResult:
    """What one pass did; `answered` holds setting keys, never the typed values."""
    kind: Kind
    step: StepOutcome
    install: Literal["done", "skipped", "none"] = "none"
    questions: Literal["answered", "skipped", "none"] = "none"
    answered: tuple[str, ...] = ()
    downloaded: bool = False
    row: DoctorRow | None = None
    loads: bool = False
    rebind: bool = False
    override: tuple[Literal["on", "off"] | None, Path] | None = None
    shown: str = ""
    detail: str = ""
    next_steps: str = ""
    help: tuple[str, ...] = ()


@dataclass(frozen=True)
class Touch:
    rebind: bool
    loads: bool
    probe: tuple[str, str] | None


@dataclass(frozen=True)
class ReloadPlan:
    names: frozenset[str]
    parked: int
    waiting: int

    @property
    def reload(self) -> bool:
        return bool(self.names) and self.parked == self.waiting == 0


@dataclass(frozen=True)
class PluginSession:
    cfg_path: Path
    workspace: Path | None
    running: frozenset[str]
    unconfigured: Mapping[str, str]
    failed: Mapping[str, str]
    touched: dict[str, Touch] = field(default_factory=dict)


def prior_touch(session: PluginSession, name: str) -> Touch:
    """The seed for a plugin nothing has touched this session."""
    return Touch(rebind=False, loads=name in session.running, probe=None)


def refused(session, name, text, install) -> PassResult:
    prior = session.touched.get(name) or prior_touch(session, name)
    return PassResult("refused", StepOutcome(name, False), install=install, loads=prior.loads, detail=text)


def loads_from_plan(session: PluginSession, name: str, prior: Touch, probe: DoctorRow | None,
                    resolution: Resolution | None = None) -> bool:
    """Will the next build load it: the plan reads `on` and the last known check did not fail."""
    if resolution is None:
        from localharness.config.loader import ConfigLoader
        from localharness.plugins.resolve import resolve
        resolution = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace))
    entry = resolution.plan.entry(name)
    if entry is None or entry.state != "on":
        return False
    if probe is not None:
        return probe.state == "on"
    return ((prior.probe is None or prior.probe[0] == "on")
            and name not in session.failed and name not in session.unconfigured)


def touch_from(session: PluginSession, name: str, result: PassResult, prior: Touch) -> Touch | None:
    """The Touch a returned pass leaves, or None when it wrote nothing."""
    if result.kind in ("refused", "declined", "rerun_blocked") or (result.kind == "override" and result.step.on):
        return None
    probe = ((result.row.state, result.row.detail) if result.row
             else ("unconfigured", "setup questions skipped") if result.questions == "skipped"
             else None if result.kind == "disabled" else prior.probe)
    return Touch(rebind=prior.rebind or result.rebind, loads=result.loads, probe=probe)


def touch_after_stop(session: PluginSession, name: str, progress: PassProgress, prior: Touch) -> Touch | None:
    """The Touch a caught stop leaves: what the pass had already written before it ended."""
    if not (progress.wrote or progress.answered):
        return None
    rebind = prior.rebind or (name in session.running and (
        progress.answered or (progress.phase == "edit" and not progress.overridden)))
    enable_written = progress.phase in ("download", "check") and not progress.rerun
    loads = loads_from_plan(session, name, prior, None) if enable_written else prior.loads
    return Touch(rebind=rebind, loads=loads, probe=prior.probe)


async def step_pending(resolution: Resolution, loader: ConfigLoader, entry: PlanEntry,
                       paths: PluginPaths) -> bool:
    """Does this plugin's setup step still have work — the first time, or while it is not set up?
    Only a plugin that declares a step (setup questions or a setup action) has one, and it does
    while its check, the row `doctor` prints, is not clean: unconfigured, failed or a failing row;
    or a skipped row while one of its questions has no stored value. Settings that do not
    validate leave nothing to check with: not set up. Asks and writes nothing."""
    from localharness.plugins.lifecycle import _doctor_row
    from localharness.plugins.setup import has_setup_action

    m, cls = entry.manifest, resolution.classes.get(entry.name)
    if m is None or cls is None or not (m.setup or has_setup_action(cls)):
        return False
    if entry.name not in resolution.settings:
        return True
    row = await _doctor_row(resolution, entry.name, paths)
    return _not_clean(row, any(not _stored(resolution, loader, entry, f.key) for f in m.setup))


def _switch(name: str, on: bool, pairs: list[str], to_workspace: bool, config_dir: Optional[str],
            ask: bool = False, in_session: bool = False) -> StepOutcome:
    """Write `<name>.enabled: <on>`, and any --set values, into one layer's overrides.yaml; turning
    a plugin on, run its setup step around that write. With `ask` (a terminal, no --set): its
    questions are asked first and the answers written as --set values; after the write its setup
    action runs (after its yes/no question, when it has one) and its doctor check runs whenever it
    declares questions or an action; setup_help and the filled-in coding-agent prompt follow a
    check that is not clean. Without `ask` nothing is asked and nothing runs: the next step says
    what to run. A plugin missing its install extra is asked nothing and not checked. next_steps
    print on every outcome, and the command exits 0 whether or not the check passes: it wrote.

    `in_session` (the /plugins restart, via session_step): the questions are asked only while the
    step is pending — a configured plugin is just switched — and the change takes effect in the
    session that resumes, so the success line does not point at the next start. Either way the
    setup action runs only while the step is pending or when answers were written in this run."""
    resolution, loader, workspace = _resolve(config_dir, json_output=False)
    entry = _entry(resolution, loader, name, False)
    state = PassState(name, on, tuple(pairs), to_workspace, ask, in_session, resolution, loader, workspace, entry)
    try:
        done = asyncio.run(_run_pass(state, prompt=partial(prompt_line, console=console),
                                     confirm=partial(confirm_line, console=console), report=shell_report))
    except Refusal as refusal:
        shell_report("refused", str(refusal))
        raise typer.Exit(code=2)
    return StepOutcome(name, on, failed_check=done.failed, skipped_check=done.skipped, stopped=done.declined)


def shell_report(kind: ReportKind, payload: Any) -> None:
    """The shell's transcript: the one place a pass's lines are drawn, every text as text."""
    from localharness.cli.doctor_cmd import print_plugin_row

    def say(text: str, lead: str = "") -> None:
        console.print(lead + escape(sanitize_for_display(text)), soft_wrap=True)

    if kind == "wrote":
        line, values = payload
        say(line, "[green]✓[/green] ")
        for path, value in values:
            say(f"  set {path} = {value}")
    elif kind == "note":
        say(payload, "  [yellow]note:[/yellow] ")
    elif kind == "checking":
        console.print("Checking it now:")
    elif kind == "row":
        print_plugin_row(payload, [])
    elif kind == "help":
        for paragraph in payload:
            console.print()
            say(paragraph)
    elif kind == "next_steps":
        blank_first, lines = payload
        if blank_first:
            console.print()
        for line in lines:
            say(f"  {line}")
    elif kind == "stays_off":
        say(f"{payload} stays off. Your conversation continues.")
    else:
        err_console.print("[bold red]Error:[/bold red] " + escape(sanitize_for_display(payload)), soft_wrap=True)


@dataclass(frozen=True)
class PassState:
    name: str
    on: bool
    pairs: tuple[str, ...]
    to_workspace: bool
    ask: bool
    in_session: bool
    resolution: Resolution
    loader: ConfigLoader
    workspace: Any
    entry: PlanEntry
    m: PluginManifest | None = None
    missing: bool = False
    setup: tuple[SetupField, ...] = ()
    cls: Any = None
    has_action: bool = False
    pending: bool = True
    target: Path | None = None
    declined: bool = False
    values: tuple[tuple[str, str], ...] = ()
    fresh_loader: ConfigLoader | None = None
    fresh: Resolution | None = None
    after: PlanEntry | None = None
    paths: PluginPaths | None = None
    rows: tuple = ()
    row: DoctorRow | None = None
    failed: str = ""
    skipped: str = ""

    @property
    def verb(self) -> str:
        return "enable" if self.on else "disable"

    @property
    def step(self) -> bool:
        return self.ask and self.after is not None and not self.missing


async def _confirmed(confirm: Any, text: str, default: bool) -> bool:
    answer = await confirm(sanitize_for_display(text), default=default)
    if answer is None:
        raise click.Abort()
    return answer


async def _run_pass(state: PassState, *, prompt: Any, confirm: Any, report: Any) -> PassState:
    state = await _prepare(state, confirm=confirm, report=report)
    if state.declined:
        return state
    state = await _questions(state, prompt=prompt)
    state = await _write(state, report=report)
    state = await _act(state, confirm=confirm, report=report)
    return await _check(state, report=report)


async def _prepare(s: PassState, *, confirm: Any, report: Any) -> PassState:
    from localharness.plugins.setup import has_setup_action

    word = "on" if s.on else "off"
    if s.name not in s.resolution.enabled:  # a refused NAME (invalid, a core key, taken): its key is
        raise Refusal(f"{s.name} cannot be turned {word}: {s.entry.reason}", ())  # not this plugin's to write
    if s.to_workspace and not s.entry.bundled:
        raise Refusal(f"a plugin you installed is turned {word} only in machine-level settings — run "
                      f"`localharness plugins {s.verb} {s.name}` without --workspace", ())
    if s.to_workspace and s.workspace is None:
        raise Refusal("no project workspace applies here — run `localharness init --workspace` in the "
                      "project first, or leave out --workspace to change the machine-level setting", ())
    if (s.in_session and s.on and s.ask and not s.entry.bundled
            and not await _confirmed(confirm, THIRD_PARTY_CONFIRM.format(name=s.name), False)):
        report("stays_off", s.name)
        return replace(s, declined=True)
    m = s.entry.manifest if s.on else None
    # A missing extra is tested directly, not through the plan state: a default-on plugin the user
    # turned off reads `off` there, and would still be asked for settings it cannot use.
    missing = m is not None and _extra_missing(m)
    cls = s.resolution.classes.get(s.name)
    has_action = m is not None and cls is not None and has_setup_action(cls)
    # Before asking, from what is stored now: does the step still have work? The shell asks every
    # question every time, as it always has; a session asks only while the step is pending.
    pending = (await step_pending(s.resolution, s.loader, s.entry, _paths(s.loader, s.workspace))
               if s.ask and m is not None and not missing and (s.in_session or has_action) else True)
    # Each layer's overrides.yaml exactly as config/loader.py reads it.
    target = s.workspace / "overrides.yaml" if s.to_workspace else s.loader.user_overlay_path
    return replace(s, m=m, missing=missing, setup=m.setup if m is not None else (), cls=cls,
                   has_action=has_action, pending=pending, target=target)


async def _questions(s: PassState, *, prompt: Any) -> PassState:
    if not (s.ask and s.setup and not s.missing and (s.pending or not s.in_session)):
        return s
    # the answers go through the one checked write path, as --set values
    pairs = await ask_fields(s.setup, lambda key: _stored(s.resolution, s.loader, s.entry, key),
                             lambda key: _secret_stored(s.resolution, s.loader, s.entry, key),
                             lambda key: _cleared(s.resolution, s.entry, key), prompt=prompt)
    return replace(s, pairs=tuple(pairs))


async def _write(s: PassState, *, report: Any) -> PassState:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    overlay = load_overlay(s.target)
    set_value_in_dict(overlay, f"{s.name}.enabled", s.on)
    values = (_checked_all(s.resolution, s.loader, s.entry, list(s.pairs), s.to_workspace, overlay)
              if s.pairs else {})
    atomic_write_overlay(s.target, overlay)
    tail = "" if s.in_session else " — takes effect on the next `localharness start`"
    shown_values = tuple((path, redact_text(value)) for path, value in values.items())
    report("wrote", (f"{s.name} {s.verb}d in {s.target}{tail}", shown_values))
    fresh_loader = fresh = after = row = None
    if s.m is not None:
        # What was just written, read through a NEW loader over the same layers: ConfigLoader caches
        # its harness and raw sources, and re-deriving the workspace would print its trust notice
        # (and maybe ask) a second time.
        fresh_loader = ConfigLoader(config_dir=s.loader.global_config_dir, local_config_dir=s.workspace)
        try:
            fresh = resolve(fresh_loader)
            after = fresh.plan.entry(s.name)
        except Exception as exc:  # noqa: BLE001 — the enable already wrote; the rest is advice
            row = could_not_check(s.name, exc)
    missing = s.missing
    if after is not None and s.m.requires_extra and after.state == "needs-extra":
        missing = True  # "takes effect on the next start" is not true without its install extra
        report("note", f"{s.name} is missing its install extra — {after.reason}")
    project = [x["enabled"] for x in s.loader.plugin_layers().get(s.name, (None,) * 4)[2:]
               if isinstance(x, dict) and isinstance(x.get("enabled"), bool)]
    if not s.to_workspace and s.entry.bundled and project and project[-1] is not s.on:
        report("note", f"this project turns {s.name} {'off' if s.on else 'on'} ({s.workspace}), and that "
                       f"still wins here — `localharness plugins {s.verb} {s.name} --workspace` "
                       "changes it for this project")
    paths = _paths(fresh_loader, s.workspace) if fresh_loader is not None else None
    return replace(s, values=shown_values, missing=missing, fresh_loader=fresh_loader, fresh=fresh,
                   after=after, paths=paths, row=row)


async def _act(s: PassState, *, confirm: Any, report: Any) -> PassState:
    from localharness.plugins.lifecycle import DoctorRow, setup_action_rows

    if not (s.step and s.has_action and s.after.state == "on" and (s.pending or s.pairs) and (
            not s.m.setup_action or await _confirmed(confirm, s.m.setup_action, True))):
        return s
    rows = _stripped(DoctorRow(s.name, "on", "", await setup_action_rows(s.fresh, s.name, s.paths))).checks
    if rows:
        report("row", DoctorRow(s.name, "on", "", rows))
    return replace(s, rows=rows)


async def _check(s: PassState, *, report: Any) -> PassState:
    from localharness.plugins.setup import render_agent_prompt

    m, row, failed, skipped, block = s.m, s.row, "", "", False
    if row is not None or (s.step and (s.setup or s.has_action)):
        report("checking", None)
        stored = row is not None
        row = row or await probe(s.name, s.fresh, s.paths)
        report("row", row)
        unanswered = not stored and any(not _stored(s.fresh, s.fresh_loader, s.after, f.key) for f in s.setup)
        not_clean = _not_clean(row, unanswered)
        # The setup action's rows count as the check's do: one that failed (a far end that does not
        # answer, a download that raised) is not set up, even when the offline doctor row passes.
        action_failed = [c for c in s.rows if c.status == "fail"]
        if not_clean or action_failed:
            # A failing row, the check's or the setup action's, is a failed check; a skipped row with
            # none failing is a plugin not set up yet (mobile's "not enrolled yet"), not a failure; a
            # check row that is not on (unconfigured, failed) has only its state to say.
            checks = row.checks if not_clean else ()
            failed = sanitize_for_display(
                row.detail if not_clean and row.state != "on" else
                next((c.detail for c in (*checks, *action_failed) if c.status == "fail"), ""))
            skipped = "" if failed else sanitize_for_display(
                next((c.detail for c in checks if c.status == "skip"), ""))
            paragraphs = tuple(p for p in (m.setup_help, redact_text(render_agent_prompt(
                m.agent_prompt, _prompt_values(s.fresh or s.resolution, s.fresh_loader or s.loader,
                                               s.after or s.entry, m.agent_prompt))) if m.agent_prompt else "") if p)
            if paragraphs:  # the one place it prints
                report("help", paragraphs)
            block = bool(paragraphs)
    elif not s.ask and m is not None:
        lines = ((f"next step — give it the {', '.join(f.prompt for f in s.setup)}: "
                  f"{_set_spelling(s.name, s.setup)}",) if s.setup and not s.pairs else ()) + (
            (f"next step — run `localharness plugins enable {s.name}` on a terminal to answer: "
             f"{m.setup_action}",) if s.has_action and m.setup_action else ())
        if lines:
            report("next_steps", (False, lines))
    if m is not None and m.next_steps:
        report("next_steps", (block, tuple(m.next_steps.splitlines())))  # a blank line ends the text to paste
    return replace(s, row=row, failed=failed, skipped=skipped)


def could_not_check(name: str, exc: Exception) -> DoctorRow:
    """The failed row for a check that could not run, its text scrubbed of credentials."""
    from localharness.plugins.lifecycle import DoctorRow
    return DoctorRow(name, "failed", f"could not check it now: {type(exc).__name__}: {redact_text(str(exc))}")


def _stripped(row: DoctorRow) -> DoctorRow:
    return replace(row, detail=redact_text(row.detail), checks=tuple(
        c.model_copy(update={"detail": redact_text(c.detail), "hint": redact_text(c.hint)}) for c in row.checks))


async def probe(name: str, resolution: Resolution, paths: PluginPaths) -> DoctorRow:
    """The plugin's doctor row read from `resolution` (after the write), or the row that says it could not."""
    from localharness.plugins.lifecycle import DoctorRow, _doctor_row

    try:
        entry = resolution.plan.entry(name)
        return _stripped(await _doctor_row(resolution, name, paths) if entry.state == "on"
                         else DoctorRow(name, entry.state, entry.display, bundled=entry.bundled))
    except Exception as exc:  # noqa: BLE001 — the enable already wrote; the check is advice
        return could_not_check(name, exc)


def manual_block(names: list[str], manual: Manual, route: str) -> tuple[str, ...]:
    """The lines that tell a person to run the install themselves, then come back by `route`."""
    joined = ", ".join(names)
    return (*([manual.prereq] if manual.prereq else []),
            "Quit localharness first, then run:" if manual.quit_first else "In another terminal run:",
            manual.command,
            "then " + ("start localharness and " if manual.quit_first else "")
            + route.format(rerun=ACTION_RERUN, names=joined, name=joined))


def _config_dir_problem(cfg: Path) -> str | None:
    ensure_private_dir(cfg)
    if not cfg.is_dir() or not os.access(cfg, os.W_OK):
        return "not_writable"
    if not update_cmd._is_windows() and cfg.stat().st_mode & OPEN_TO_OTHERS:
        return "open"
    return None


def _line(raw: bytes) -> str:
    return sanitize_for_display(raw[:_READ_SIZE].decode("utf-8", errors="replace"))


async def _read_lines(stream: Any) -> list[str]:
    """The last lines of `stream`, read in fixed chunks so no line length can raise."""
    lines: deque[str] = deque(maxlen=_KEPT_LINES)
    partial = b""
    while chunk := await stream.read(_READ_SIZE):
        *done, partial = (partial + chunk).split(b"\n")
        lines.extend(map(_line, done))
        partial = partial[:_READ_SIZE]
    if partial:
        lines.append(_line(partial))
    return list(lines)


async def _run_install(argv: list[str], cfg: Path, announce: Any) -> tuple[int, list[str]] | None:
    """Run the installer in a fresh private directory; None when no such directory could be made.
    The child never outlives this call, and neither does the directory."""
    d = None
    try:
        with suppress(OSError):
            d = tempfile.mkdtemp(prefix="install-", dir=cfg)
        if d is None:
            return None
        announce()
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=d, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        try:
            lines = await _read_lines(proc.stdout)
            await proc.wait()
        finally:
            if proc.returncode is None:
                with suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), _TERMINATE_GRACE)
                except BaseException as exc:
                    if proc.returncode is None:
                        with suppress(ProcessLookupError):
                            proc.kill()
                    await proc.wait()
                    if not isinstance(exc, TimeoutError):
                        raise
        return proc.returncode, lines
    finally:
        if d:
            shutil.rmtree(d, ignore_errors=True)


async def install_pass(session: PluginSession, extras: set[str], names: list[str], *, progress: PassProgress,
                       confirm: Any, console: Console, in_batch: bool = False) -> frozenset[str]:
    """One consent, one install of the union of `extras`, for the plugins in `names`; the names it
    covers on success, an empty set after printing why not. Prints its own blocks through `console`."""
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    route = update_cmd.install_route(*sorted(extras))
    joined = ", ".join(names)
    command = update_cmd._quote(route.argv) if route.argv else ""

    def say(*lines: str) -> None:
        for line in lines:
            console.print(Text(sanitize_for_display(line)), soft_wrap=True)

    def skipped(why: str, **fill: Any) -> frozenset[str]:
        back = (manual_block(names, route.manual, ROUTE_BACK) if route.argv is None else (
            INSTALL_SKIPPED[1], INSTALL_SKIPPED[2].format(command=command),
            INSTALL_SKIPPED[3].format(route=ROUTE_BACK.format(rerun=ACTION_RERUN, names=joined))))
        say(INSTALL_SKIPPED[0].format(names=joined, why=INSTALL_SKIPPED_WHY[why].format(**fill)), *back)
        return frozenset()

    if route.argv is None:
        return skipped("declined")
    plan = resolve(ConfigLoader(config_dir=session.cfg_path, local_config_dir=session.workspace)).plan
    asks = [(n, e.manifest.setup_action) for n in names
            if (e := plan.entry(n)) is not None and e.manifest is not None and e.manifest.setup_action]
    uv = route.argv[1:3] == ["tool", "install"]
    installer = "uv" if uv else "pip"
    head, how, per_plugin, last = CONSENT
    say(head.format(names=joined),
        how.format(installer=installer, spec=route.spec, **{"with": CONSENT_WITH if uv else ""}),
        *(per_plugin.format(name=n, question=q) for n, q in asks),
        last.format(command=command))
    ctrlc = CTRLC.format(does=CTRLC_DOES["batch" if in_batch else "prompt"])
    answer = await confirm("", default=True, tail=CONSENT_ASK.format(ctrlc=ctrlc))
    if answer is None:
        raise click.Abort()
    if not answer:
        return skipped("declined")
    progress.phase = "install"
    cfg = session.cfg_path
    if problem := _config_dir_problem(cfg):
        return skipped(problem, dir=cfg)

    def announce() -> None:
        if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
            say(CTRLC_IGNORED.format(step="install"))
        say(INSTALLING.format(installer=installer, spec=route.spec))

    if (ran := await _run_install(route.argv, cfg, announce)) is None:
        return skipped("no_temp", dir=cfg)
    code, tail = ran
    if code:
        say(*tail)
        return skipped("failed", code=code)
    importlib.invalidate_caches()
    return frozenset(names)


def session_step(action: tuple[str, str], config_dir: str | None) -> StepOutcome:
    """The step `/plugins enable|disable NAME` runs between the two halves of a terminal session's
    restart (cli/start_cmd.start_app): the same _switch the shell command runs, on the plain
    terminal with nothing else drawing on it — never inside the input box, so a typed secret never
    reaches the box's history. It never raises: Ctrl-C or Ctrl-D at a question, a refusal, or an
    error (a failed write, a plugin bug) stops the step and the session resumes anyway."""
    import click
    verb, name = action
    try:
        return _switch(name, verb == "enable", [], False, config_dir,
                       ask=_stdin_is_a_terminal(), in_session=True)
    except (click.exceptions.Abort, click.exceptions.Exit, EOFError, KeyboardInterrupt):
        console.print("Stopped. Your conversation continues.")
        return StepOutcome(name, verb == "enable", stopped=True)
    except Exception as exc:  # noqa: BLE001 — after teardown a raise would lose the conversation
        why = (f"{type(exc).__name__}: {exc}".splitlines() or [type(exc).__name__])[0]
        console.print(escape(f"{name}'s setup hit an error: {why}. Your conversation continues."),
                      soft_wrap=True)
        return StepOutcome(name, verb == "enable", stopped=True)



async def switch_decision(name: str, on: bool, running: frozenset[str], global_config_dir: Path,
                          workspace: Path | None) -> str | None:
    """Should `/plugins enable|disable NAME` restart the running session? None: yes — end the REPL,
    the step runs between the two halves. A line: no — show it and stay. A session restarts when
    the set of running plugins would change, when the plugin's flag on disk would change, or when
    its setup step still has work (its questions are asked on the plain terminal, never in the
    input box). Reads a fresh plan over the session's own layers; asks and writes nothing."""
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve
    loader = ConfigLoader(config_dir=global_config_dir, local_config_dir=workspace)
    resolution = resolve(loader)
    entry = resolution.plan.entry(name)
    word, verb = ("on", "enable") if on else ("off", "disable")
    if entry is None:
        return f"Unknown plugin: {name}. /plugins lists them."
    if name not in resolution.enabled:
        return f"{name} cannot be turned {word}: {entry.reason}"
    project = [s["enabled"] for s in loader.plugin_layers().get(name, (None,) * 4)[2:]
               if isinstance(s, dict) and isinstance(s.get("enabled"), bool)]
    if entry.bundled and project and project[-1] is not on:
        return (f"this project turns {name} {'off' if on else 'on'} ({workspace}), and that still wins "
                f"here — `localharness plugins {verb} {name} --workspace` changes it for this project")
    flag = resolution.enabled[name]
    if not on:  # on on disk, or still running here: the restart writes the flag and stops it
        return None if flag or name in running else f"{name} is already off."
    m = entry.manifest
    if m is not None and _extra_missing(m):
        return (f"{name} needs its install extra first: install `localharness[{m.requires_extra}]`, "
                f"then /plugins enable {name} again")
    if name in running and flag and not await step_pending(resolution, loader, entry,
                                                            _paths(loader, workspace)):
        return f"{name} is already on."
    return None


def plugins_overview(running: frozenset[str], global_config_dir: Path, workspace: Path | None) -> str:
    """Bare `/plugins`: every plugin, whether it runs in this session (else the plan's own state
    text), and how to switch one here."""
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve
    entries = resolve(ConfigLoader(config_dir=global_config_dir, local_config_dir=workspace)).plan.entries
    width = max((len(e.name) for e in entries), default=0)
    lines = ["Plugins:"] + [f"  {e.name.ljust(width)}  "
                            + ("running in this session" if e.name in running else e.display)
                            for e in entries]
    return "\n".join([*lines, "/plugins enable <name> turns one on here; /plugins disable <name> turns one off."])


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
