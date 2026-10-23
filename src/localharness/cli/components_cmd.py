"""`localharness components` subapp: list, get, set with audit trail."""
from __future__ import annotations

import asyncio
import json as _json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional
from urllib.parse import urlsplit

import typer
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from localharness.config.loader import ConfigLoader, ConfigNotFoundError
from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.overlay import (
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.config.paths import resolve_config_dir, resolve_runtime_path
from localharness.config.plugin_sections import core_agent_view, core_harness_view
# The secret-safe error text lives in config/ (the loader uses it too; config/ never imports cli/).
from localharness.config.redact import SECRET_MASK, holds_secret, is_secret, scrub
from localharness.config.redact import validation_text as _validation_text
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.registry import (
    SURFACE_FAMILIES,
    ComponentEntry,
    coerce_value,
    set_value_in_dict,
    walk_model_fields,
)
from localharness.registry.catalogue import (
    AGENT_PATHS_NOT_ADDRESSABLE,
    LAYER_BANDS,
    PluginRows,
    plugin_catalogue_rows,
)
from localharness.registry.provenance import display_note, layered_catalogue

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
    here — a bare ConfigLoader() picks up the hermetic-test LOCALHARNESS_HOME just the same.

    NOT workspace-aware, deliberately, and it must stay that way. Six callers OUTSIDE this module
    use it for the GLOBAL harness config — autoresearch adoption/experiment/loop, propose_cmd and
    autoresearch_cmd (x2) — and making them workspace-aware is a behavior change no requirement
    asks for, in the gate/experiment path of all places. The three `components` commands use
    `_build_layered_loader` below instead."""
    return ConfigLoader()


def _workspace_for(config_dir: Optional[str], *, json_output: bool) -> Optional[Path]:
    """The workspace layer for THIS `components` invocation, or None.

    `config_dir` is the RAW flag value. Resolving it first would erase the "was this explicit"
    signal the workspace gate is built on (39-04): an explicit --config-dir is a full replacement
    and must skip discovery entirely (LAYR-02).

    `--json` means machine output, so an undecided outside-the-project workspace stays inert
    rather than stopping to prompt on a payload stream.

    Called ONCE per invocation and its result passed around, never re-derived: for an untrusted
    or undecided workspace `resolve_workspace_layer` PRINTS a notice (and may prompt and record),
    so a second call in the same command prints that notice twice.
    """
    from localharness.cli.workspace import resolve_workspace_layer

    return resolve_workspace_layer(config_dir, interactive=False if json_output else None)


def _build_layered_loader(
    config_dir: Optional[str], *, json_output: bool
) -> tuple[ConfigLoader, Optional[Path]]:
    """(loader, workspace) for the `components` commands: global layer + the workspace layer, if
    one applies. The sibling pattern doctor/validate/agent have used since 39-05.

    Returns the workspace alongside the loader because the caller needs the same value — `set`
    tells the user which project it is standing in — and re-deriving it would double the notice
    (see `_workspace_for`).
    """
    workspace = _workspace_for(config_dir, json_output=json_output)
    return (
        ConfigLoader(
            config_dir=resolve_config_dir(config_dir),
            local_config_dir=workspace,
        ),
        workspace,
    )


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


def _validate_overlay(loader: ConfigLoader, path: str, new_overlay: dict, *,
                      plugins: Sequence[PluginRows] = ()) -> None:
    """Validate `new_overlay` against the model that OWNS `path`; raise ValueError (a pydantic
    ValidationError is one) on failure.

    agent.* → AgentConfig (the merged `agent:` subtree); every other path → the merged
    HarnessConfig with the agent-scope `agent:` section EXCLUDED (it is not a HarnessConfig
    field). Before #22 every path validated against HarnessConfig, so agent.* always failed
    'Extra inputs are not permitted'; and once the overlay carries an `agent:` section, a later
    harness-path set would inherit that same failure unless `agent:` is excluded here too
    (mirrors load_harness's overlay handling).

    A plugin's setting is then checked by that plugin's own model (ENAB-01), from the same rows the
    catalogue listed: `<name>.*` — the global config.yaml's `<name>:` merged with the new overlay's,
    minus `enabled` — against its ConfigModel; `agent.<name>.*` — the new overlay's
    `agent.<name>` — against its AgentConfigModel.
    """
    agent = path.startswith(_AGENT_PREFIX)
    if agent:
        merged_agent = deep_merge(dict(_AGENT_VALIDATE_BASE), new_overlay.get(_AGENT_KEY, {}))
        # plugin sections are validated by their plugins (ENAB-01); unknown keys are rejected at load
        AgentConfig.model_validate(core_agent_view(merged_agent))
    else:
        harness_overlay = {k: v for k, v in new_overlay.items() if k != _AGENT_KEY}
        # plugin sections are validated by their plugins (ENAB-01); unknown keys are rejected at load
        HarnessConfig.model_validate(core_harness_view(deep_merge(loader.raw_harness_dict(), harness_overlay)))
    name = path.split(".")[1 if agent else 0]
    rows = next((p for p in plugins if p.name == name), None)
    if rows is None:
        return
    if agent:
        model, section = rows.agent_config_model, (new_overlay.get(_AGENT_KEY) or {}).get(name) or {}
    else:
        merged = deep_merge(loader.raw_harness_dict().get(name) or {}, new_overlay.get(name) or {})
        model, section = rows.config_model, {k: v for k, v in merged.items() if k != "enabled"}
    if model is None:
        return
    try:
        model.model_validate(section)
    except ValidationError:
        raise
    except (Exception, SystemExit) as exc:  # the plugin's own validator: contained, nothing written
        raise ValueError(f"plugin {name}'s settings check raised {type(exc).__name__}: {exc}") from exc


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
        # escape(): the message quotes back the key and the VALUE the user typed, and rich would
        # parse `[/red]` in either and kill the command instead of explaining it — in the one
        # function whose whole job is explaining what went wrong (E cluster, badmood review).
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(str(message)), soft_wrap=True
        )
    raise typer.Exit(code=exit_code)


def _err_config(json_output: bool, exc: Exception) -> None:
    """Config load failure. #119: when the cause is a MISSING config, end with the same next
    step doctor gives — otherwise a user who ran `components list` first never learns `init`."""
    hint = " Run 'localharness init' to create it." if isinstance(exc, ConfigNotFoundError) else ""
    _err(json_output, f"Failed to load config: {exc}{hint}", exit_code=2)


def _layer_cell(e: ComponentEntry) -> str:
    """The layer a human reads; a plugin's setting also names its plugin (ENAB-04)."""
    return e.winning_layer + (f" (plugin: {e.plugin})" if e.plugin else "")


def shown(value: Any, annotation: Any = None) -> str:
    """repr() for a human, masked for a SecretStr value or a type that holds one anywhere (a typed
    endpoint list carries its key inside the text, so it is masked whole)."""
    from pydantic import SecretStr
    return repr(SECRET_MASK) if holds_secret(annotation) or isinstance(value, SecretStr) else repr(value)


_URL_WITH_AT = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"]*@[^\s'\"]*")


def _authority_unparseable(url: str) -> bool:
    parts = urlsplit(url)
    try:
        parts.port
    except ValueError:
        return True
    return ":" in parts.netloc and "@" in url[len(parts.scheme) + 3 + len(parts.netloc):]


def redact_text(text: str, secrets: Sequence[str] = ()) -> str:
    """`text` with `secrets` masked, URL userinfo stripped, and a URL with an `@` past its authority masked whole."""
    from localharness.provider.server import strip_url_credentials  # pulls httpx: not at CLI start

    stripped = strip_url_credentials(scrub(text, list(secrets)))
    return _URL_WITH_AT.sub(lambda m: SECRET_MASK if _authority_unparseable(m.group()) else m.group(), stripped)


def userinfo_present(value: Any) -> bool:
    if isinstance(value, str):
        return redact_text(value) != value
    if isinstance(value, BaseModel):
        return any(userinfo_present(getattr(value, name)) for name in type(value).model_fields)
    if isinstance(value, dict):
        return any(userinfo_present(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(userinfo_present(v) for v in value)
    return False


def workspace_file_of(loader: ConfigLoader, winning_layer: str) -> Path | None:
    """The workspace file `winning_layer` names, or None for any other band or no workspace."""
    if winning_layer not in LAYER_BANDS[2:]:
        return None
    file = loader.plugin_layer_files()[LAYER_BANDS.index(winning_layer)]
    return Path(file) if file else None


def _overlay_secrets(catalogue: dict[str, ComponentEntry], overlay: dict) -> list[str]:
    """The raw strings an overlay holds at secret leaves."""
    out = []
    for path, e in catalogue.items():
        if is_secret(e.annotation):
            node: Any = overlay
            for part in path.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            out.append(node)
    return out


def walk_secret_values(leaves: dict[str, Any], section: dict[str, Any]):
    """(key, raw value) for every secret leaf `section` holds — scrubbed from error texts."""
    for key, ann in leaves.items():
        if is_secret(ann):
            node: Any = section
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            yield key, node


def _secret_strings(values: Any) -> list[str]:
    """The values worth scrubbing: non-empty strings, without the literal default "none" (a
    proposer's "no key"), which would otherwise mask every "none" in the text."""
    return [v for v in values if isinstance(v, str) and v and v != "none"]


def _core_secret_values(loader: ConfigLoader, overlay: dict[str, Any]) -> list[str]:
    """Every string at a SecretStr leaf of the core settings — the global config.yaml merged with
    the overlay about to be written (proposer.api_key today) — so a refusal can scrub them."""
    merged = deep_merge(loader.raw_harness_dict(), {k: v for k, v in overlay.items() if k != _AGENT_KEY})
    return _secret_strings(v for _k, v in walk_secret_values(dict(walk_model_fields(HarnessConfig)), merged))


def _settings_secrets(catalogue: dict[str, ComponentEntry], loader: ConfigLoader,
                      overlay: dict[str, Any]) -> list[str]:
    """Every string at a secret leaf the catalogue knows, core and plugin, in the settings a write
    is checked against: the global config.yaml merged with the overlay about to be written. Not
    only the overlay's own: a key stored in config.yaml is in what _validate_overlay checks too."""
    return _secret_strings(_overlay_secrets(catalogue, deep_merge(loader.raw_harness_dict(), overlay)))




def _serialize_value(value: Any, secret: bool = False) -> Any:
    """Render value for JSON output: a secret leaf or SecretStr is SECRET_MASK; a model is its
    model_dump(mode="json") (which masks its own SecretStr fields); lists, tuples and dicts are
    walked; other primitives pass through; anything else is its repr."""
    from pydantic import BaseModel, SecretStr
    if secret or isinstance(value, SecretStr):
        return SECRET_MASK
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _stdin_is_a_terminal() -> bool:
    """Is a person typing at stdin? (`components set <secret> -` asks them with a hidden prompt.)"""
    return sys.stdin is not None and sys.stdin.isatty()


class Refusal(Exception):
    """A write the checked path declined; its message is already scrubbed."""

    def __init__(self, text: str, secrets: Sequence[str]) -> None:
        super().__init__(redact_text(text, secrets))


@dataclass(frozen=True)
class Written:
    path: str
    before: Any = field(repr=False)
    after: Any = field(repr=False)
    annotation: Any
    overlay_path: Path
    audit_path: Path | None


def write_value(loader: ConfigLoader, catalogue: dict[str, ComponentEntry],
                plugins: Sequence[PluginRows], path: str, raw: str) -> Written:
    """Coerce, validate and atomically write one value to the machine overlay; raises Refusal."""
    entry = catalogue.get(path)
    if entry is None:
        raise Refusal(AGENT_PATHS_NOT_ADDRESSABLE.get(path) or f"Unknown path: {path!r}", ())
    try:
        typed_value = coerce_value(raw, entry.annotation)
    except ValueError as exc:
        raise Refusal(f"Cannot coerce {shown(raw, entry.annotation)} to {entry.type_name}: {exc}",
                      [raw] if is_secret(entry.annotation) else []) from exc
    before = entry.current_value
    audit_log = loader.load_harness().org.audit_log_path
    audit_path = resolve_runtime_path(audit_log, loader._config_dir) if audit_log else None
    overlay_path = loader.user_overlay_path
    new_overlay = set_value_in_dict(dict(load_overlay(overlay_path)), path, typed_value)
    try:
        _validate_overlay(loader, path, new_overlay, plugins=plugins)
    except ValueError as exc:  # a pydantic ValidationError, or a plugin's contained validator failure
        # Never pydantic's input_value (the whole merged settings), and never a secret either file
        # holds: a plugin validator's own message can quote its section.
        why = _validation_text(exc) if isinstance(exc, ValidationError) else str(exc)
        raise Refusal(f"Validation failed for {path}={shown(typed_value, entry.annotation)}: {why}",
                      _settings_secrets(catalogue, loader, new_overlay)) from exc
    try:
        atomic_write_overlay(overlay_path, new_overlay)
    except Exception as exc:
        raise Refusal(f"Failed to write overlay {overlay_path}: {exc}", ()) from exc
    loader.invalidate_cache()
    return Written(path, before, typed_value, entry.annotation, overlay_path, audit_path)


def _side(value: Any, annotation: Any) -> Any:
    return _serialize_value(value, is_secret(annotation) or userinfo_present(value))


async def audit_value(written: Written) -> None:
    await EventBus(persist_path=written.audit_path).publish(ComponentMutated(
        path=written.path,
        before_value=_side(written.before, written.annotation),
        after_value=_side(written.after, written.annotation),
        layer="user",
        actor="cli",
    ))


# ------------------------------------------------------------------ #
# list
# ------------------------------------------------------------------ #


@components_app.command("list")
def components_list(
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
    layer: Optional[str] = typer.Option(
        None,
        "--layer",
        help=(
            "Filter to entries with this winning layer (default|global-config|"
            "global-overrides|workspace-config|workspace-overrides|experiment)"
        ),
    ),
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
) -> None:
    """List every mutable component with its current value and winning layer."""
    workspace = _workspace_for(config_dir, json_output=json_output)
    tool_registry = _build_tool_registry()
    try:
        catalogue, _overlays = layered_catalogue(
            resolve_config_dir(config_dir), workspace, tool_registry=tool_registry
        )
    except Exception as exc:
        _err_config(json_output, exc)
        return  # unreachable but satisfies type checker

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
                "plugin": e.plugin,
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
        # display_note: an empty deny list on screen is not an empty deny list at runtime.
        table.add_row(e.path, e.type_name, repr(e.current_value) + display_note(e.path, e.current_value), _layer_cell(e))
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
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
) -> None:
    """Print the resolved value of one component + its winning layer."""
    workspace = _workspace_for(config_dir, json_output=json_output)
    tool_registry = _build_tool_registry()
    try:
        catalogue, _overlays = layered_catalogue(
            resolve_config_dir(config_dir), workspace, tool_registry=tool_registry
        )
    except Exception as exc:
        _err_config(json_output, exc)
        return

    entry = catalogue.get(path)
    if entry is None:
        _err(
            json_output,
            AGENT_PATHS_NOT_ADDRESSABLE.get(path)
            or f"Unknown path: {path!r}. Run `localharness components list` to see valid paths.",
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
            "plugin": entry.plugin,
        }
        typer.echo(_json.dumps(payload))
        return

    # Values and layer bands are both data: a band renders as `[workspace-config]`, which rich
    # parses as a style tag, and a stored value can be any string. escape() the whole row.
    console.print(
        escape(f"{entry.path} = {entry.current_value!r}"
               f"{display_note(entry.path, entry.current_value)}"),
        soft_wrap=True,
    )
    console.print(escape(f"  type:    {entry.type_name}"), soft_wrap=True)
    console.print(escape(f"  layer:   {_layer_cell(entry)}"), soft_wrap=True)
    console.print(escape(f"  default: {entry.default_value!r}"), soft_wrap=True)


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
        help=(
            "New value as a string. Coerced to the path's target type. For a secret, `-` reads it "
            "without echo (a hidden prompt on a terminal, else one line of stdin), so it never sits "
            "in your shell history or `ps`."
        ),
    ),
    json_output: bool = typer.Option(False, "--json"),
    config_dir: Annotated[
        Optional[str],
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
) -> None:
    """Mutate one component. Writes the GLOBAL overrides.yaml atomically (v0.13 policy: every
    overlay writer targets the machine layer); emits a ComponentMutated audit event."""

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
        loader, workspace = _build_layered_loader(config_dir, json_output=json_output)
        loader.load_harness()
    except Exception as exc:
        _err_config(json_output, exc)
        return

    # 2. Build catalogue, resolve entry. Workspace-aware so the reported `was:` is the value the
    #    user can actually see — and so a workspace-owned path can be named as such below.
    #    The plugins are resolved ONCE: their rows build the catalogue and their models check the
    #    value below. Lazy import: resolving imports every enabled plugin, never paid at CLI start.
    tool_registry = _build_tool_registry()
    from localharness.plugins.resolve import resolve

    plugins = plugin_catalogue_rows(resolve(loader))
    catalogue, _overlays = layered_catalogue(
        resolve_config_dir(config_dir), workspace, tool_registry=tool_registry, loader=loader,
        plugins=plugins,
    )
    entry = catalogue.get(path)
    if entry is not None and is_secret(entry.annotation) and value == "-":  # never in shell history or `ps`
        value = (typer.prompt(path, hide_input=True, default="", show_default=False).strip()
                 if _stdin_is_a_terminal() else sys.stdin.readline().rstrip("\r\n"))
        if not value:
            _err(json_output, f"No value read for {path}; nothing was written.", exit_code=2)
            return

    try:
        written = write_value(loader, catalogue, plugins, path, value)
    except Refusal as refusal:
        _err(json_output, str(refusal), exit_code=2)
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(audit_value(written))
    else:  # already inside an event loop (possible under pytest-asyncio)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(audit_value(written))
        finally:
            loop.close()
    before, typed_value, overlay_path = written.before, written.after, written.overlay_path

    # 8. Confirm, and SAY WHERE IT WENT. The write target has always been the global
    #    overrides.yaml; F4 is that nothing said so at the moment it happened, so a user standing
    #    in a project could reconfigure the whole machine believing they had touched this project.
    # Spec 21 §5: the input box is built once, so a `terminal.keys` change only takes effect at
    # the next `localharness start`; the receipt says so (a human clause and a JSON field), and no
    # other path's output changes.
    applies_next_start = path == "terminal.keys" or path.startswith("terminal.keys.")
    after = typed_value
    if applies_next_start:
        # `before` is the effective list, so `after` is too: the validated model's list, which is
        # also where the other list's trimmed chord shows (spec 21 §5), not the string typed.
        try:
            after = loader.load_harness()
            for part in path.split("."):
                after = getattr(after, part)
        except Exception:  # noqa: BLE001 — the receipt falls back to what was typed
            after = typed_value
    if json_output:
        receipt = {
            "path": path,
            "before": _side(before, written.annotation),
            "after": _side(after, written.annotation),
            # `layer` is the ComponentMutated audit event's own, older vocabulary and is
            # NOT the registry band — renaming it would change a persisted log's schema.
            # `target` is the honest addition: the file that was actually written.
            "layer": "user",
            "target": str(overlay_path),
        }
        if applies_next_start:
            receipt["applies"] = "next-start"
        typer.echo(_json.dumps(receipt))
    else:
        # `layer: user` used to close this line. It is dropped, not renamed: `user` is the audit
        # EVENT's vocabulary (kept in the JSON and in ComponentMutated), and as a band name shown
        # to a human it says neither which layer nor which file — the exact ambiguity F2 removes.
        # The next line names the actual file, which is strictly more informative.
        # escape() around the whole payload: the VALUE is whatever the user typed, and a value
        # like `[dim]` renders as nothing at all — this receipt would confirm setting a key to an
        # empty string. Same lesson as the path lines, applied to values (41-06 / doctor's rows).
        console.print(
            "[green]set[/green] " + escape(redact_text(f"{path} = {shown(after, entry.annotation)} "
                                                      f"(was: {shown(before, entry.annotation)})")),
            soft_wrap=True,
        )
        # escape(): a path is data, not markup — a folder named `[old] proj` must not be parsed
        # (41-06's measured lesson, same reason doctor and validate escape theirs).
        wrote_note = "the global overrides.yaml"
        if applies_next_start:
            wrote_note += "; applies at the next start"
        console.print(f"  wrote: {escape(str(overlay_path))}  [dim]({wrote_note})[/dim]")
        if workspace is not None:
            console.print(
                "  [yellow]note:[/yellow] this is a MACHINE-WIDE setting. It applies in every "
                "project, including this one. A per-project value goes in "
                f"{escape(str(workspace / 'config.yaml'))}."
            )
            if workspace_file_of(loader, entry.winning_layer) is not None:
                # Without this the command reports success and the value a user reads back does
                # not move — the same class of silence F4 exists to remove.
                console.print(
                    f"  [yellow]note:[/yellow] {path} is set by this project "
                    f"({entry.winning_layer}), which still WINS here. The value above is now the "
                    "machine default for every other project; to change it here, edit "
                    f"{escape(str(workspace / 'config.yaml'))}."
                )

    # Reference SURFACE_FAMILIES to keep the import live (avoids unused-warning if linted)
    _ = SURFACE_FAMILIES
