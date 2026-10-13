"""localharness start command — smart routing REPL entry point."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml
from pydantic import SecretStr
from rich.console import Console
from rich.markup import escape
from rich.prompt import IntPrompt
from rich.table import Table
from rich.text import Text

from localharness.cli.theme import entity, entity_text
from localharness.cli.workspace import (
    NO_INPUT_HELP,
    MachineTrust,
    primary_server_port,
    raw_settings,
    uncovered,
    visible_text,
    write_overlay_validated,
)
from localharness.config.effort import EFFORT_UNENFORCED_LINE
from localharness.core.private_files import touch_private
from localharness.core.which import which_outside_cwd
from localharness.plugins.channels import (OWN_COMMAND, accepted_channels, channel_names,
                                           plugin_channel_names)

if TYPE_CHECKING:
    from localharness.provider.lifecycle import ServerRecovery

console = Console()
err_console = Console(stderr=True)
log = logging.getLogger(__name__)

UNKNOWN_CHANNEL_ERROR = "unknown channel {given!r}; choose one of: {known}"

OWN_COMMAND_ERROR = ("the {name} channel is served by its own command, because {why}. "
                     "Run `localharness {name}` instead of `localharness start --channel {name}`.")

FIRST_START_LEAD = "LocalHarness is not set up yet. Setting up the model server first:"

RESTARTED_LINE = "Restarted with {name} {state}. Your conversation continues."
# The two start-summary lines for a server that refused an effort request key (spec 16 §7.4, D-19, D-22);
# the first is spelled once in config/effort.py (imported above), shared with /effort show.
EFFORT_CUT_OFF_LINE = "effort: thinking-off rejected by this server; the reasoning cut is off"
# Said once when a configured temperature overrides the model's own default (spec 20 §7, D-27): the
# rung's own label names what to unset. A fixed template filled with numbers and the loader's label.
SAMPLING_LINE = ("sampling: temperature {temp} from {rung} "
                 "(the model's own is {own} — unset {rung} to use it)")
RESUMED_LINE = "Awake again, continuing the conversation that went to sleep (sitting {previous})."
WAKING_DETAIL = "waking the chat you left"


def _own_command(mobile_channel: Any, acp_channel: Any) -> dict[str, tuple[str, Any]]:
    """The ONE own-command table: name -> (why it has its own command, the channel it handed in).
    Its keys must equal plugins.channels.OWN_COMMAND (pinned by a test)."""
    return {"mobile": ("the HTTP server has to be reachable before a session exists", mobile_channel),
            "acp": ("the editor's handshake has to be answered before a session exists", acp_channel)}


def _refuse_unbuildable_channel(channel_mode: str, mobile_channel: Any = None,
                                acp_channel: Any = None) -> None:
    """Tier one: raise typer.BadParameter for an unknown `--channel`, or for one served by its own
    command that was not handed in. Reads nothing from disk, so `_start_async` and a first start's
    setup can both ask it before anything else."""
    known = channel_names()  # static manifests: no plugin is resolved, imported or loaded here
    if channel_mode not in known:
        raise typer.BadParameter(
            UNKNOWN_CHANNEL_ERROR.format(given=channel_mode, known=", ".join(sorted(known))),
            param_hint="--channel")
    own = _own_command(mobile_channel, acp_channel)
    if channel_mode in own and own[channel_mode][1] is None:
        # mobile and acp name channels _start_async cannot BUILD: the HTTP server / the editor's
        # handshake has to be up and answering before a session exists, so the channel is
        # constructed by `localharness mobile` / `localharness acp` and handed in. Without that, its
        # channel branch would fall through to the terminal — precisely the silent fallback the
        # check above exists to end, reintroduced by the same commit that ended it.
        msg = OWN_COMMAND_ERROR.format(name=channel_mode, why=own[channel_mode][0])
        if channel_mode in plugin_channel_names():
            msg += f" (if that command is missing, run `localharness plugins enable {channel_mode}`)"
        raise typer.BadParameter(msg, param_hint="--channel")


CHANNEL_PLUGIN_NOT_ON = "channel '{name}' is provided by the {plugin} plugin, which is {state} — {fix}"
CHANNEL_PLUGIN_NOT_RUNNING = "channel '{name}' is provided by the {plugin} plugin, which did not start — {reason}"
"""Tier 2 (the plugin resolved off / needs-extra / failed) and tier 3 (it resolved on but is not
running after start_plugins). Both refuse: asking for a channel and getting the terminal is the
silent fallback the tier-1 check exists to end."""


def _channel_plugin(channel_mode: str) -> str | None:
    """The bundled channel plugin whose static manifest names `channel_mode`."""
    from localharness.plugins.builtin import bundled_plugins
    return next((c.manifest.name for c in bundled_plugins() if c.manifest.kind == "channel"
                 and channel_mode in (c.manifest.channels or (c.manifest.name,))), None)


def _channel_refusal(channel_mode: str, resolution: Any) -> str | None:
    """Tier 2, read off the resolved plan: None when the channel is not a plugin channel or its
    plugin is on; otherwise the refusal, naming the plugin and the fix."""
    if channel_mode not in plugin_channel_names() - OWN_COMMAND or channel_mode in accepted_channels(resolution):
        return None
    plugin = _channel_plugin(channel_mode)
    entry = next((e for e in resolution.plan.entries if e.name == plugin), None)
    if entry is None:
        return CHANNEL_PLUGIN_NOT_ON.format(name=channel_mode, plugin=plugin, state="not loaded",
                                            fix="see `localharness plugins list`")
    fix = f"run `{entry.enable_command}`" if entry.enable_command else entry.reason
    state = "missing its install extra" if entry.state == "needs-extra" else entry.state
    return CHANNEL_PLUGIN_NOT_ON.format(name=channel_mode, plugin=entry.name, state=state, fix=fix)


MOBILE_NEEDS_ITS_OWN_COMMAND = OWN_COMMAND_ERROR.format(name="mobile", why=_own_command(None, None)["mobile"][0])
"""Shown verbatim. A refusal naming the alternatives is the difference between a typo costing a
second and a typo costing a session."""

NO_BOUNDARY_NOTICE = (
    "No project folder here, so there is no workspace boundary: every write, edit and "
    "non-read-only shell command will ask, every time. Run localharness from inside a project "
    "(a folder with a .git or .localharness in it) to get the quiet path."
)
"""PRD §3.1 / critic finding 1: when the project root resolves to $HOME or above there is no
boundary, and the honest answer is to say so once at startup rather than let the person
discover it one prompt at a time. It names the fix, because "ask every time" with no
explanation is the prompt fatigue the ask-rate SLO exists to prevent."""


def _start_problems(plugin_resolution: Any) -> list[str]:
    """The plan's problem lines the START summary shows. A BUNDLED plugin waiting for its install
    extra is not a startup problem (a terminal-only install never asked for the phone); `plugins
    list` and doctor still show it. An installed plugin's missing extra still warns: the user opted
    in. (46, ruling 5)"""
    return plugin_resolution.problems(include=lambda e: not (e.state == "needs-extra" and e.bundled))


def _first_prompt_hint(is_returning: bool) -> str:
    """The guidance shown in the first interactive input bubble (#49). First-run gets the
    full 'describe a task' hint; a returning session still gets a short '/help' reminder
    (the returning banner previously reinforced nothing)."""
    return "/help for commands." if is_returning else "Describe a task, or /help for commands."


def _hint_with_instructions(hint: str, line: str) -> str:
    """The first-prompt hint with the instructions start line riding in the same slot (D-47).

    The returning default ("/help for commands.") and an empty hint give way to the line alone;
    any other hint (the first-run text) keeps its words and the line follows after a dot."""
    if not hint or hint == _first_prompt_hint(True):
        return line
    return f"{hint} · {line}"


def _open_project_instructions(
    boundary: Path | None,
    *,
    workspace: Path | None,
    decision: bool | None,
    trust_flag: bool,
    sitting_id: str,
    max_ratio: float,
    no_input: bool,
    channel_mode: str,
    window_tokens: int,
    config_dir: str | None,
) -> Any:
    """The workspace-open hook: the one instructions handle of a session, or None (D-48, D-67, D-69).

    Scoped to the session's effective boundary (the key the gate and session trust use) and built
    for any root with a candidate file, because a session-trust yes arrives after start returns.
    None when there is no boundary, when the root was declined (``decision is False``), or when no
    candidate exists. The basis is ``layer`` for a trusted project whose workspace layer loaded,
    ``trusted`` for a trusted root without a layer or an unrecorded one under ``--trust-project``,
    else ``awaiting_trust`` (the file is unused until a yes record exists). ``no_input`` here is the
    CLI flag only: ACP's own internal no-input is not it, so Zed still uses and records the file
    (D-06). A broken store never stops a start."""
    if boundary is None or decision is False:
        return None
    if decision is True:
        basis = "layer" if workspace is not None else "trusted"
    elif trust_flag:
        basis = "trusted"
    else:
        basis = "awaiting_trust"
    cli_no_input = bool(no_input) and channel_mode != "acp"
    try:
        from localharness.config.project_records import open_project_instructions

        return open_project_instructions(
            boundary, sitting_id=sitting_id, max_ratio=max_ratio, basis=basis,
            no_input=cli_no_input, window_tokens=window_tokens, config_dir=config_dir,
        )
    except Exception:  # noqa: BLE001 — a broken store costs the instructions, never the session
        log.warning("could not open the project instructions handle", exc_info=True)
        return None


def _children_instructions(handle: Any, children: bool) -> Any:
    """What the runner hands each child loop: the handle itself, or (``instructions.children:
    false``) its mark-only view; None when there is no handle (W-1)."""
    if handle is None or children:
        return handle
    return handle.mark_only()


def _available_hint(names: list[str]) -> str:
    """ENAB-04 / PRD §4: nothing runs because it was installed, and the operator is told how to turn
    it on — once, at start."""
    if len(names) == 1:
        return (f"i 1 plugin available, not enabled: {names[0]} — run "
                f"`localharness plugins enable {names[0]}` to turn it on")
    return (f"i {len(names)} plugins available, not enabled: {', '.join(names)} — run "
            f"`localharness plugins enable <name>` to turn one on")


from localharness.cli.session_resume import (  # noqa: E402 — the handle lives beside its file form
    SLEEP_ACTION, Restart, Resume, asleep_path, take_asleep, write_asleep,
)


def _resume_status(resume: Resume, result: Any) -> str:
    """The indicator's last line after a /plugins restart: did the plugin come up, and how?"""
    verb, name = resume.action
    running = result is not None and name in result.loaded_names
    if verb == "disable":
        return f"{name}: off in this session" if not running else f"{name}: still on in this session — /plugins shows why"
    if resume.step_stopped:
        return f"{name}: {'on' if running else 'not on'} in this session — its setup stopped before it finished"
    if result is not None and name in result.unconfigured:
        return f"{name}: on, but not set up yet — run /plugins enable {name} to set it up"
    if result is not None and name in result.failed:
        return f"{name}: on, but it could not start: {result.failed[name]}"
    if resume.failed_check:
        return f"{name}: on, but its check failed: {resume.failed_check}"
    if resume.skipped_check:  # a skipped check is a plugin not set up yet, not a failure
        return f"{name}: on, but not set up yet — {resume.skipped_check}"
    if running:
        return f"{name}: on in this session"
    return f"{name}: not running in this session — /plugins shows its state"


def default_root_agent(agents: list[dict[str, Any]]) -> dict[str, Any]:
    """The discovered agent `start` opens without --agent/--subagents: 'default', else
    'orchestrator', else the first. Shared with `doctor`, so it checks the agent `start` runs."""
    return next((a for a in agents if a.get("name") == "default"),
                next((a for a in agents if a.get("name") == "orchestrator"), agents[0]))


def _agent_roster_table(agents: list[dict[str, Any]]) -> Table:
    """The --subagents picker roster, with each agent NAME in the agent entity color.

    Cells are Text spans, not markup strings: a Table renders str cells through the markup
    parser, so an agent legally named `[old] proj` used to lose its `[old]` to a swallowed
    tag (and `[/red]proj` would have raised MarkupError and killed the picker). Text makes
    the name literal by construction — the `[old] proj` lesson, applied without needing an
    escape() call around a value that also has to carry a style.

    Split out of _start_async so the roster can be rendered and asserted on directly.
    """
    table = Table(title=entity_text("agent", "Available Agents"))
    table.add_column("No.")
    table.add_column("Name")
    table.add_column("Role", style="dim")
    for i, a in enumerate(agents, start=1):
        table.add_row(str(i), entity_text("agent", a.get("name", "")), Text(a.get("role", "")))
    return table


def _managed_server_running(strategy: Any, srv: Any, global_dir: Path) -> bool:
    """Is the harness-managed server ALREADY serving? ``global_dir`` is the GLOBAL config dir, not
    whatever layer the session is otherwise reading: the pidfile/log/venv under ``<global>/vllm/``
    are machine-wide (one physical GPU daemon per box), so this liveness question must never be
    asked of a workspace layer. Asks the LIFECYCLE STRATEGY (the single
    owner of how a backend is launched/stopped/checked) instead of re-deriving liveness here:
    docker reads the CONTAINER by name — never the pidfile, whose pid is the `docker run`
    sig-proxy CLIENT, the orphan-client-pid bug; binary/ollama read `server_pid()`, correct
    there; lmstudio probes the ENDPOINT. Re-deriving it from `srv.launch` alone (the old code)
    ignored `srv.runtime`, so a managed LM Studio — which by design writes NO pidfile — always
    read False and start took the "not running" branch into LmsStrategy.activate, whose
    pre-existing-server fail-fast then raised on the server that was in fact already up.
    Called inside the autostart try, so an OSError from the pid path degrades as before."""
    return strategy.liveness(srv, global_dir).alive


def _read_session_key(harness: Any) -> str | None:
    """Read `server.api_key_file` into this session's bearer, or return the one line to print.

    Called once, after `decide_machine_trust` and `load_harness`: a withheld entry has already lost
    its `server:` section by then, and an entry the user just answered Yes (or the record already
    covers) is in it, so the file is opened only once the record covers it. Success sets
    `harness.provider.api_key` to the key — in memory only, never written, printed or logged — so
    `required_key_kwargs` yields it, both probes and every request carry it and the file rides argv
    beside the environment key. A file the byte rule refuses is recorded on the spec
    (`mark_key_file`), so the launch decision keeps the file off argv and binds loopback on every
    launch path of the session, whatever other key the harness holds, and the returned line says
    which of the two is true: `server.api_key_file <path>: <reason> — this session sends no key` when
    no other key is held, or `... — the file is not used; this session keeps sending the key it
    already holds` when a configured or environment key keeps the session keyed. The file is opened
    only when `provider.base_url` is this machine (literally loopback, no DNS): the key is sent to that
    URL, which nothing ties to `server.port`, so any other destination is refused the same way with
    the reason `provider.base_url <host:port> is not this machine, so the file is not read` (WR-06).
    Never raises for the file's sake. No server or no file: None."""
    from localharness.provider.server import (
        KeyFileError,
        key_destination,
        key_kwargs,
        mark_key_file,
        read_key_file,
    )
    spec = getattr(harness, "server", None)
    path = getattr(spec, "api_key_file", None) if spec is not None else None
    if not path:
        return None

    def refused(reason: object) -> str:
        mark_key_file(spec, True)
        # the path comes from a settings file: shown through visible_text like every file-supplied value
        head = f"server.api_key_file {visible_text(path)}: {visible_text(reason)} — "
        if key_kwargs(harness.provider.api_key):
            return head + "the file is not used; this session keeps sending the key it already holds"
        return head + "this session sends no key"

    # The key goes to provider.base_url, which nothing ties to server.port or to the host the person
    # confirmed: a destination that is not this machine never gets a key out of a file (WR-06).
    here, shown = key_destination(harness.provider.base_url)
    if not here:
        return refused(f"provider.base_url {shown} is not this machine, so the file is not read")
    try:
        key = read_key_file(path)
    except KeyFileError as exc:
        return refused(exc)
    mark_key_file(spec, False)
    harness.provider.api_key = SecretStr(key)
    return None


# ------------------------------------------------------------------ what start senses before the decision

PATH_BINARY_LINE = ("llama-server found on PATH at {path} — written to overrides.yaml as server.binary; "
                    "nothing launches it until it is confirmed")
CONFIG_INI_WRITTEN_LINE = ("config.ini ({dir}) differs from server.config_ini — written to overrides.yaml as "
                           "server.config_ini; nothing launches it until it is confirmed")
CONFIG_INI_REFUSED_LINE = "config.ini ({dir}) sets {key} (a {category} option) — not carried; a harness launch does not see it"
HOST_SOURCE_KEY = "<extra_args-host>"
"""The decline-store key prefix under which a converted `--host` offer remembers its source (WR-03)."""
HOST_UNDO_LINE = ("server.extra_args --host {host} is not applied — launched on 127.0.0.1 — "
                  "server.bind_all: true restores it (asks once)")
HOST_CONVERTED = "server.bind_all: true, from server.extra_args --host 0.0.0.0, which 0.17 applied"
PROBE_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class SensedStart:
    """What `start` learned before `decide_machine_trust` (D-08, D-16, D-17, D-20): whether this start
    is about to launch the primary, the display-only annotations for the one question, the D-17 suffix
    of an upgrade conversion (`converted`), the PATH binary this start wrote (`path_written`), the undo
    line of a `--host` that was not converted (`undo_line`) and the masked `config.ini` text this start
    wrote as `server.config_ini` (`ini_written`, D-31) with the overlay's own value it replaced
    (`ini_prior`, None when the overlay held none): a No puts that value back (CR-51)."""

    launching: bool
    annotations: dict = field(default_factory=dict)
    converted: str | None = None
    path_written: str | None = None
    undo_line: str | None = None
    ini_written: str | None = None
    ini_prior: str | None = None


@dataclass
class LaunchNotes:
    """What start's own launch prints beside its announcement: the posture ``suffix`` appended to the
    announcement's phrase and the ``lines`` printed once right after it. ``printed`` is set by a launch
    that carried them, so the caller can hand the REPL only what was not printed."""

    lines: list[str] = field(default_factory=list)
    suffix: str | None = None
    printed: bool = False


def _server_section(raw: dict) -> dict:
    value = raw.get("server")
    return value if isinstance(value, dict) else {}


def _merged_server(cfg: dict, over: dict) -> dict:
    from localharness.config.overlay import deep_merge

    return deep_merge(_server_section(cfg), _server_section(over))


def _host_of(extra_args: list) -> str | None:
    """The value of the LAST `--host X` / `--host=X` in ``extra_args``, as llama-server resolves it."""
    tokens = [str(t) for t in extra_args]
    host: str | None = None
    for i, token in enumerate(tokens):
        if token == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]
        elif token.startswith("--host="):
            host = token[len("--host="):]
    return host


def _write_path_binary(global_dir: Path, rec: dict | None, path_binary: str | None) -> str | None:
    """D-20: a PATH-found `llama-server` for a declared-only primary (merged `runtime: llamacpp`, no
    `binary` in either file, no remembered No for this path) is written to overrides.yaml as
    `server.binary` BEFORE the machine question, so it is shown as an ordinary uncovered entry and
    launched only on Yes. Never overwrites a binary. A machine with no record yet is never written to:
    its first start adopts the disk unasked, and a binary the harness found must not be adopted that
    way. Returns the path written, or None."""
    from localharness.config import trust
    from localharness.registry import set_value_in_dict

    if not path_binary or rec is None:
        return None
    cfg, over = raw_settings(global_dir)
    merged = _merged_server(cfg, over)
    if merged.get("runtime") != "llamacpp" or merged.get("binary"):
        return None
    if trust.offer_was_declined(trust.decline_key("<binary>", path_binary)):
        return None
    if not write_overlay_validated(
            global_dir, lambda d: set_value_in_dict(d, "server.binary", path_binary),
            [("server.binary", None, path_binary)],
            actor_detail="localharness start: llama-server found on PATH, written before the machine question"):
        return None
    err_console.print(PATH_BINARY_LINE.format(path=visible_text(path_binary)), markup=False, soft_wrap=True)
    return path_binary


def _after_no_remove_path_binary(global_dir: Path, path: str) -> None:
    """A terminal No on the PATH-found binary: its `server.binary` line leaves overrides.yaml, but only
    while it still equals the path the harness found (a value someone changed stays) and the record
    holds no `server.binary` for overrides.yaml (the line is an addition the harness never saw
    confirmed, not an edit of a binary the person had), and the No is remembered for that path so only
    a newly found path asks again. Keyed on the file and the path, not on this process having written
    the line (WR-02): an earlier start may have written it, and the question returns until answered."""
    from localharness.config import trust

    _, over = raw_settings(global_dir)
    rec = trust.recorded_machine(global_dir)
    recorded = rec is not None and any(
        e.get("file") == "overrides.yaml" and e.get("name") == "server.binary" for e in rec["entries"])
    if not recorded and _server_section(over).get("binary") == path:
        def mutate(overlay: dict) -> None:
            section = overlay.get("server")
            if isinstance(section, dict):
                section.pop("binary", None)
                if not section:
                    del overlay["server"]

        write_overlay_validated(
            global_dir, mutate, [("server.binary", path, None)],
            actor_detail="localharness start: a No on the PATH-found llama-server, line removed")
    trust.record_offer_decline(trust.decline_key("<binary>", path))


def _after_no_on_path_binary(global_dir: Path, machine_trust: MachineTrust, *, can_ask: bool,
                             path_binary: str | None) -> None:
    """A terminal start whose question left `server.binary` of overrides.yaml withheld was answered No
    (a start that cannot ask never gets here): drop the PATH binary's line and remember the path. It
    does not matter which start wrote the line (WR-02)."""
    if (path_binary and can_ask
            and ("launch", "server.binary") in machine_trust.withheld.get("overrides.yaml", ())):
        _after_no_remove_path_binary(global_dir, path_binary)


def _write_config_ini(global_dir: Path, rec: dict | None) -> tuple[str | None, str | None]:
    """D-31, D-57: llama-server's own `${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini`, masked by
    the one function `init` also uses, is written to overrides.yaml as `server.config_ini` BEFORE the
    machine question when it differs from the merged record's value (an existing value is overwritten,
    unlike the PATH binary: the file on disk is what a launch of this binary would have read). Only a
    merged llama.cpp primary, only a machine with a record, and only when the text was not declined
    (keyed on its digest, so a changed file asks again). A file that cannot be carried says so in one
    line and writes nothing. Returns `(text written, prior)`: the text, or None, and the value of
    overrides.yaml's own `server.config_ini` that the write replaced (None when it held none), which a
    No restores (CR-51)."""
    from localharness.cli import init_import
    from localharness.config import trust
    from localharness.registry import set_value_in_dict

    if rec is None:
        return None, None
    cfg, over = raw_settings(global_dir)
    merged = _merged_server(cfg, over)
    if merged.get("runtime") != "llamacpp":
        return None, None
    read = init_import.read_config_ini(init_import.user_config_ini())
    if read.line:
        err_console.print(read.line, markup=False, soft_wrap=True)
        return None, None
    if read.ini is None:
        return None, None
    if read.ini.refused:  # a download-identity or TLS key: nothing of the file is carried (D-56)
        from localharness.provider import server as managed_server

        key = read.ini.refused
        err_console.print(
            CONFIG_INI_REFUSED_LINE.format(dir=read.dir, key=visible_text(key),
                                           category=managed_server.excluded_option(key)),
            markup=False, soft_wrap=True)
        return None, None
    if not read.ini.kept:
        return None, None
    kept = read.ini.kept
    digest = hashlib.sha256(kept.encode("utf-8")).hexdigest()
    if trust.offer_was_declined(trust.decline_key("<config.ini>", digest)):
        return None, None
    before = merged.get("config_ini")
    if kept == before:
        # Already merged. If overrides.yaml holds it unconfirmed (an earlier start wrote it and its
        # question was never answered) it is still this start's pending entry: a No must reach it (WR-02).
        if _server_section(over).get("config_ini") != kept:
            return None, None
        held = next((e for e in rec["entries"]
                     if e.get("file") == "overrides.yaml" and e.get("name") == "server.config_ini"), None)
        if held is None:
            return kept, None
        # The record confirmed another text and an earlier start overwrote it with this one unanswered:
        # still pending, and a No puts the confirmed text back (CR-51).
        confirmed = held.get("shown")
        return (kept, confirmed) if isinstance(confirmed, str) and confirmed != kept else (None, None)
    held_by_overlay = _server_section(over).get("config_ini")
    prior = held_by_overlay if isinstance(held_by_overlay, str) else None
    if not write_overlay_validated(
            global_dir, lambda d: set_value_in_dict(d, "server.config_ini", kept),
            [("server.config_ini", before, kept)],
            actor_detail="localharness start: llama.cpp config.ini sensed, written before the machine question"):
        return None, None
    err_console.print(CONFIG_INI_WRITTEN_LINE.format(dir=read.dir), markup=False, soft_wrap=True)
    return kept, prior


def _after_no_remove_config_ini(global_dir: Path, kept: str, prior: str | None = None) -> None:
    """A terminal No on the sensed `config.ini`: its `server.config_ini` line leaves overrides.yaml, but
    only while it still equals the text the harness sensed (a value someone changed stays). When the
    write replaced a value overrides.yaml already held (`prior`: an edit of what the person had
    confirmed) that value is put back, so nothing the person set is lost and no uncovered line is left
    to withhold `server:` (CR-51). With no `prior` the line is an addition the harness never saw
    confirmed, and it is dropped unless the record holds a `server.config_ini` for overrides.yaml. The
    No is remembered for that text's digest so only a changed file asks again (D-31)."""
    from localharness.config import trust
    from localharness.registry import set_value_in_dict

    _, over = raw_settings(global_dir)
    rec = trust.recorded_machine(global_dir)
    recorded = rec is not None and any(
        e.get("file") == "overrides.yaml" and e.get("name") == "server.config_ini" for e in rec["entries"])
    if _server_section(over).get("config_ini") == kept and prior is not None:
        write_overlay_validated(
            global_dir, lambda d: set_value_in_dict(d, "server.config_ini", prior),
            [("server.config_ini", kept, prior)],
            actor_detail="localharness start: a No on the sensed config.ini, the earlier value restored")
    elif not recorded and _server_section(over).get("config_ini") == kept:
        def mutate(overlay: dict) -> None:
            section = overlay.get("server")
            if isinstance(section, dict):
                section.pop("config_ini", None)
                if not section:
                    del overlay["server"]

        write_overlay_validated(
            global_dir, mutate, [("server.config_ini", kept, None)],
            actor_detail="localharness start: a No on the sensed config.ini, line removed")
    trust.record_offer_decline(
        trust.decline_key("<config.ini>", hashlib.sha256(kept.encode("utf-8")).hexdigest()))


def _after_no_on_config_ini(global_dir: Path, machine_trust: MachineTrust, *, can_ask: bool,
                            ini_written: str | None, ini_prior: str | None = None) -> None:
    """A terminal start whose question left `server.config_ini` of overrides.yaml withheld was answered
    No (a start that cannot ask never gets here): drop the sensed line and remember the file's digest."""
    if (ini_written and can_ask
            and ("launch", "server.config_ini") in machine_trust.withheld.get("overrides.yaml", ())):
        _after_no_remove_config_ini(global_dir, ini_written, ini_prior)


def _adopt_written_bind_all(global_dir: Path) -> None:
    """Adopt `server.bind_all: true` from a read of the files just written, nothing else on disk."""
    from localharness.config import trust

    cfg, over = raw_settings(global_dir)
    now = next((e for e in trust._posture_entries({"config.yaml": cfg, "overrides.yaml": over})
                if e["name"] == "server.bind_all"), None)
    if now is not None and now["shown"].split(":")[1] == "true":
        trust.adopt_entries(global_dir, [now])


def _convert_extra_args_host(global_dir: Path, rec: dict | None, *, can_ask: bool
                            ) -> tuple[dict, str | None, str | None]:
    """D-16: what a 0.17 `server.extra_args --host` becomes. Returns `(annotations, converted,
    undo_line)`. Acts only for a merged llama.cpp or vLLM-binary record with no `bind_all` set, and
    only when the winning file's `server.extra_args` entry is covered BY VALUE by the record:

    - exactly `0.0.0.0` at the upgrade start (`launch_model` not in the record's kinds), on any
      channel: `server.bind_all: true` is written and adopted, no question, and the launch says so;
    - any other address on a terminal start: `server.bind_all: true` is written so the one question
      shows it, with its source as an annotation;
    - everything else that 0.17 would have bound beyond loopback (a hostname, a `.sock`, an uncovered
      entry, a headless start, a `0.0.0.0` after the upgrade start) writes nothing and the launch is
      on loopback with the undo line. A docker record converts nothing and says nothing."""
    from localharness.config import trust
    from localharness.provider import server as managed_server
    from localharness.registry import set_value_in_dict

    cfg, over = raw_settings(global_dir)
    merged = _merged_server(cfg, over)
    runtime = merged.get("runtime", "vllm")
    if not (runtime == "llamacpp" or (runtime == "vllm" and merged.get("launch", "binary") == "binary")):
        return {}, None, None
    winner_file, winner = ("overrides.yaml", over) if "extra_args" in _server_section(over) \
        else ("config.yaml", cfg)
    extra_args = _server_section(winner).get("extra_args")
    host = _host_of(extra_args) if isinstance(extra_args, list) else None
    kind = managed_server.classify_hosts(host) if host is not None else "loopback"
    if merged.get("bind_all") is not None:
        # Nothing to convert: a `bind_all` is set. But an offer this start's predecessor converted and
        # deferred (the server was up, so the question was quiet) still has its source (WR-03).
        return _remembered_source(over, rec, winner_file, extra_args, host, kind), None, None
    if kind == "loopback":
        return {}, None, None
    undo = HOST_UNDO_LINE.format(host=visible_text(host))
    if rec is None or kind != "address" or uncovered(
            rec["entries"], trust._launch_entries(winner_file, "server.", {"extra_args": extra_args})):
        return {}, None, undo
    upgrade = trust.LAUNCH_MODEL_KIND not in rec["kinds"]
    every_interface = {part.strip() for part in host.split(",") if part.strip()} == {"0.0.0.0"}
    if every_interface and not upgrade:  # a 0.0.0.0 that 0.17 did not apply: never converted
        return {}, None, undo
    if not every_interface and not can_ask:  # another address is offered only where it can be asked
        return {}, None, undo
    if not write_overlay_validated(
            global_dir, lambda d: set_value_in_dict(d, "server.bind_all", True),
            [("server.bind_all", None, True)],
            actor_detail=f"localharness start: server.extra_args --host {visible_text(host)} converted to "
                         "server.bind_all: true"):
        return {}, None, undo
    if every_interface:
        _adopt_written_bind_all(global_dir)
        return {}, HOST_CONVERTED, None
    # The source of this offer is remembered beside the decline store (display only: it can suppress
    # nothing and launch nothing), so a question deferred to the moment of use shows it too (WR-03).
    trust.record_offer_decline(trust.decline_key(HOST_SOURCE_KEY, host, "upgrade" if upgrade else "later"))
    note = f"(from server.extra_args --host {host}{', which 0.17 applied' if upgrade else ''})"
    return {("overrides.yaml", trust.LAUNCH_KIND, "server.bind_all"): note}, None, None


def _remembered_source(over: dict, rec: dict | None, winner_file: str, extra_args: Any,
                       host: str | None, kind: str) -> dict:
    """The diff-line source of a `server.bind_all: true` this harness converted from an
    `extra_args --host` at an earlier start and has not been asked about yet (WR-03): the same
    `(from server.extra_args --host H[, which 0.17 applied])` note the converting start made, rebuilt
    from the fact it remembered. Only while overrides.yaml still holds `bind_all: true` (a No writes
    `false`), the host is still an address, and the record still covers that `extra_args` entry by
    value; otherwise nothing is claimed about where the `true` came from."""
    from localharness.config import trust

    if (rec is None or host is None or kind != "address"
            or trust.coerce_bool(_server_section(over).get("bind_all")) is not True
            or uncovered(rec["entries"], trust._launch_entries(winner_file, "server.", {"extra_args": extra_args}))):
        return {}
    for state in ("upgrade", "later"):
        if trust.offer_was_declined(trust.decline_key(HOST_SOURCE_KEY, host, state)):
            note = f"(from server.extra_args --host {host}{', which 0.17 applied' if state == 'upgrade' else ''})"
            return {("overrides.yaml", trust.LAUNCH_KIND, "server.bind_all"): note}
    return {}


def _sensed_launching(global_dir: Path, rec: dict | None) -> bool:
    """D-08/D-27: is this start about to launch the primary? A live pidfile says a harness-launched
    server is up (False, and no connect is made). With no pidfile, only a llama.cpp record whose
    merged `bind_all` is true and uncovered needs to know, and asks ONE loopback connect (0.5 s): an
    accepted connect means a server is serving (False); refused or timed out, True. Anything else is
    True with no connect."""
    from localharness.config import trust
    from localharness.provider import server as managed_server

    try:
        if managed_server.server_pid(global_dir) is not None:
            return False
    except OSError:  # server_pid's liveness probe is POSIX-only; an unreadable pid path is "no pidfile"
        pass
    if rec is None:
        return True
    cfg, over = raw_settings(global_dir)
    merged = _merged_server(cfg, over)
    if merged.get("runtime") != "llamacpp" or trust.coerce_bool(merged.get("bind_all")) is not True:
        return True
    posture = next((e for e in trust._posture_entries({"config.yaml": cfg, "overrides.yaml": over})
                    if e["name"] == "server.bind_all"), None)
    if posture is None or posture["kind"] not in rec["kinds"] or not uncovered(rec["entries"], [posture]):
        return True
    port = primary_server_port(global_dir)
    try:
        with contextlib.closing(socket.create_connection(("127.0.0.1", port), timeout=PROBE_TIMEOUT_SECONDS)):
            return False
    except OSError:
        return True


def _sense_before_trust(global_dir: Path, *, can_ask: bool, path_binary: str | None) -> SensedStart:
    """Everything `start` senses, and the pre-question writes it makes, before `decide_machine_trust`:
    the PATH-found binary (D-20), the 0.17 `--host` conversion (D-16) and whether this start is about
    to launch (D-08). Every overlay write goes through `workspace.write_overlay_validated`."""
    from localharness.config import trust

    rec = trust.recorded_machine(global_dir)
    launching = _sensed_launching(global_dir, rec)
    # The PATH binary is written only for a start that is about to launch AND can put the question
    # that follows (WR-01): an attach-only or a headless start would otherwise rewrite overrides.yaml
    # for a binary nothing will launch, and leave a line no answer ever cleans up. The write still
    # precedes the question (D-20).
    path_written = _write_path_binary(global_dir, rec, path_binary) if launching and can_ask else None
    # llama-server's own config.ini is sensed the same way (D-31, D-57): written only by a start that is
    # about to launch and can ask, shown as an ordinary uncovered launch entry, removed on a No.
    ini_written, ini_prior = _write_config_ini(global_dir, rec) if launching and can_ask else (None, None)
    annotations, converted, undo_line = _convert_extra_args_host(global_dir, rec, can_ask=can_ask)
    if annotations:  # the conversion wrote `bind_all: true`: it is an uncovered entry now, so sense again
        launching = _sensed_launching(global_dir, rec)
    return SensedStart(launching, annotations, converted, path_written, undo_line, ini_written, ini_prior)


def _server_recovery_supplier(repl_cell: list) -> Callable[[], ServerRecovery]:
    """The one reconnect supplier `start` hands the root loop and the delegation runner (D-21).

    The REPL exists only after the loop and the runner are built, so the supplier is late-bound the
    way `get_parent_session_id` is: it answers `repl_cell[0].server_recovery()` once the cell holds a
    REPL, and until then (and for ACP, which returns before any REPL is built) one shared poll-only
    ``ServerRecovery(None, None, None)``. The shared pair is the same object on every call, so
    `ProviderLost` is still published once per death."""
    from localharness.provider.lifecycle import ServerRecovery
    shared = ServerRecovery(None, None, None)

    def _supplier() -> "ServerRecovery":
        return repl_cell[0].server_recovery() if repl_cell else shared

    return _supplier


async def _start_managed_server(srv: Any, global_dir: Path, base_url: str, launch_key: dict[str, str],
                                *, path_binary: str | None = None, autosense: Any = None,
                                endpoint_out: list | None = None,
                                notes: LaunchNotes | None = None) -> str | None:
    """Launch path 1 of the five: `start`'s down-server branch. Wait for a server that is already
    loading, or ask the launch decision BEFORE announcing anything and launch through the lifecycle
    strategy. Returns the decision's refusal line when the launch is refused (nothing stopped,
    nothing announced, nothing started) and None otherwise; raises what `activate` and `wait_ready`
    raise. ``path_binary`` is a `llama-server` found on PATH: it only reaches the refusal text.

    The posture phrase comes from the effective argv (`effective_posture`), never from the flags,
    and only on a launch this start makes — a start that only waits for a loading server attaches.

    ``autosense`` is the machine's `AutosenseConfig`, handed to `activate` so a llama.cpp launch sizes
    its window; ``endpoint_out`` collects the `LiveEndpoint` of a launch this start made (nothing is
    appended on the wait-for-a-loading-server path), which is where the caller reads the window the
    launch got and the environment names it dropped.

    ``notes`` is what start's own launch prints beside the announcement (D-17, D-16): its ``suffix``
    joins the posture phrase when the launch really binds beyond loopback, its ``lines`` follow the
    announcement once, and ``printed`` is set. The waiting and refused branches print none of it."""
    from localharness.provider import server as managed_server
    from localharness.provider.lifecycle import LAUNCH_LOCK_WAIT_LINE, launch_lock, strategy_for
    strategy = strategy_for(srv)
    label = managed_server.RUNTIME_LABELS.get(srv.runtime, srv.runtime)

    async def _wait_for_loading_server() -> None:
        console.print(f"Managed {label} is still loading — waiting...")
        # Already launched (a live pidfile / running container) — only wait for readiness. The dir is
        # the strategy's own: LM Studio writes no pidfile, so with the raw dir a server another
        # process brought up would read "exited during startup" at once.
        await managed_server.wait_ready(base_url, config_dir=strategy.recovery_config_dir(global_dir),
                                        **launch_key)

    if _managed_server_running(strategy, srv, global_dir):
        await _wait_for_loading_server()
        return None
    keyed = bool(launch_key)
    decision = managed_server.launch_decision(srv, keyed)
    if decision.refusal is not None:
        found = f" (PATH has {path_binary})" if path_binary and srv.runtime == "llamacpp" else ""
        return decision.refusal + found
    exposed, posture = managed_server.effective_posture(srv, keyed=keyed, decision=decision)
    if notes is not None and notes.suffix and exposed:
        posture = f"{posture or 'every interface'} — {notes.suffix}"

    async def _say_launch_lock_wait() -> None:
        err_console.print(LAUNCH_LOCK_WAIT_LINE, markup=False, soft_wrap=True)

    # D-27 item 1: the launch is taken under the one launch lock, so another LocalHarness process
    # that is launching the same server (the REPL's relaunch, a second `start`) cannot overwrite the
    # one pidfile. The refusal above was answered before the lock: a refused launch never touches it.
    loading = False
    async with launch_lock(global_dir, on_wait=_say_launch_lock_wait):
        if _managed_server_running(strategy, srv, global_dir):
            loading = True  # someone else brought it up while we waited for the lock: attach, never launch
        else:
            def _announce(clause: str = "") -> None:
                """The one launch line (D-23): the posture phrase first, then what was derived, then the
                load warning. Plain text, never markup: the clause is built from closed plan values."""
                inside = "; ".join(part for part in (posture, clause) if part)
                inside = f"{inside}; " if inside else ""
                console.print(
                    f"Managed {label} is not running — starting it ({inside}model load can take several minutes)...",
                    markup=False)

            def _say_notes() -> None:
                if notes is not None:
                    for line in notes.lines:
                        err_console.print(line, markup=False, soft_wrap=True)
                    notes.printed = True
                if decision.note:
                    err_console.print(decision.note, markup=False, soft_wrap=True)

            launch_kw: dict[str, Any] = {}
            if getattr(strategy, "plans_launch", False):
                # A planning strategy hands each plan it runs to `on_plan` right before that plan's launch
                # (the first plan, then a retry), so the line names the shape that really starts. The
                # notes and the decision note follow the first line only.
                said: list[bool] = []

                def on_plan(plan: Any) -> None:
                    _announce(_plan_clause(plan))
                    if not said:
                        said.append(True)
                        _say_notes()

                launch_kw = dict(on_plan=on_plan)
            else:
                # R-10e: a strategy that plans nothing keeps today's order, with no clause.
                _announce()
                _say_notes()
            # activate = serve_command → start_server → wait_ready (launch + readiness) via the lifecycle
            # strategy. The off-loop verified-stop wrapper lives in the strategy, not this launch path.
            live = await strategy.activate(srv, global_dir, base_url, autosense=autosense, **launch_kw,
                                           **launch_key)
            if endpoint_out is not None:
                endpoint_out.append(live)
    if loading:
        await _wait_for_loading_server()
    return None


def _plan_clause(plan: Any) -> str:
    """The derived clause of the launch line for the plan a launch is about to run (D-23); empty for an
    untuned launch (`None`). The undo field is `server.speculative`: `start` launches the primary."""
    if plan is None:
        return ""
    from localharness.provider import tuning

    report = getattr(plan, "report", None)
    return tuning.derived_clause(
        plan.tuned, shape=plan.shape, cause=plan.cause, dropped=plan.dropped,
        remembered_at=plan.remembered_at, context=getattr(report, "asked", None),
        undo_field="server.speculative")


_POSTURE_KEYS = ("bind_all", "require_api_key")
_BIND_ALL_ASKS = "{subject}: true not applied — launched on 127.0.0.1 for this run; a terminal start asks"
_BIND_ALL_DEFERRED = ("{subject}: true not applied — launched on 127.0.0.1 for this run; "
                      "the next terminal start that launches it asks")
_REQUIRE_KEY_ASKS = "{subject}: false not applied — launched keyed for this run; a terminal start asks"


def _posture_owner(name: str) -> "tuple[str | None, str] | None":
    """`(peer, key)` for a posture entry name: `server.<key>` is the primary (`peer` None),
    `extra_endpoints.<peer>.lifecycle.<key>` a peer (the peer is recovered by stripping the fixed
    prefix and splitting on the LAST `.lifecycle.`, so `qwen3.6` reads whole). None for any other name."""
    if name.startswith("server.") and name[len("server."):] in _POSTURE_KEYS:
        return None, name[len("server."):]
    if name.startswith("extra_endpoints."):
        peer, sep, key = name[len("extra_endpoints."):].rpartition(".lifecycle.")
        if sep and peer and key in _POSTURE_KEYS:
            return peer, key
    return None


def _dropped_launches(machine_trust: MachineTrust) -> "tuple[bool, set[str]]":
    """`(primary, peers)`: whether a withheld entry other than the two posture keys drops the primary's
    `server:` section for this run (CONTEXT D-14), and the names of the peers whose `lifecycle` a
    withheld entry other than the posture keys removes (D-13). The loader narrows a posture entry
    only when nothing else of its record is withheld, so for these a "launched on 127.0.0.1" line
    would promise a launch that does not happen."""
    from localharness.config import trust

    primary = False
    peers: set[str] = set()
    for entries in machine_trust.withheld.values():
        for kind, name in entries:
            if kind not in (trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND):
                continue
            if name.startswith("server."):
                primary = primary or name[len("server."):] not in _POSTURE_KEYS
            elif name.startswith("extra_endpoints."):
                peer, sep, key = name[len("extra_endpoints."):].rpartition(".lifecycle.")
                if sep and peer and key not in _POSTURE_KEYS:
                    peers.add(peer)
    return primary, peers


def _posture_lines(machine_trust: MachineTrust,
                   snap_shown: Mapping[tuple[str, str, str], str]) -> list[str]:
    """The withheld-posture lines `start` composes from the decision's result (D-15, TRUST-04,
    TRUST-08), for a launch to print once. A withheld `bind_all: true` is launched on 127.0.0.1 and a
    withheld `require_api_key: false` keyed, each saying the key is `not applied`, for the primary
    (`server.<key>`) and for a peer (`extra_endpoints[<name>].lifecycle.<key>`). ``snap_shown`` maps
    `(file, kind, name)` to the entry's `shown` (`<runtime>:<value>:<keyed|open>`) as the files hold it
    now. A QUIET entry is silent unless it is a llama.cpp `bind_all: true` deferred to the moment of
    use (the deferred wording): a primary entry the person's No
    remembered now holds `false`, and a peer entry quieted by its remembered No is their own answer.
    Any other withheld entry dropped `server:` and `MACHINE_WITHHELD_LINE` already named it."""
    from localharness.config import trust

    quiet = machine_trust.quiet
    dropped_primary, dropped_peers = _dropped_launches(machine_trust)
    lines: list[str] = []
    for file, entries in sorted(machine_trust.withheld.items()):
        for kind, name in sorted(entries):
            owner = _posture_owner(name) if kind == trust.LAUNCH_KIND else None
            shown = snap_shown.get((file, kind, name))
            if owner is None or shown is None:
                continue
            peer, key = owner
            if (dropped_primary if peer is None else peer in dropped_peers):
                continue  # another withheld entry dropped this launch: nothing launches, nothing is narrowed
            runtime, _, rest = shown.partition(":")
            value = rest.partition(":")[0]
            subject = f"extra_endpoints[{peer}].lifecycle.{key}" if peer is not None else f"server.{key}"
            if (file, kind, name) in quiet:
                declined = peer is not None and trust.offer_was_declined(
                    trust.decline_key("<peer>", name, shown))
                if key == "bind_all" and runtime == "llamacpp" and value == "true" and not declined:
                    lines.append(_BIND_ALL_DEFERRED.format(subject=subject))
            elif key == "bind_all" and value == "true":
                lines.append(_BIND_ALL_ASKS.format(subject=subject))
            elif key == "require_api_key" and value == "false":
                lines.append(_REQUIRE_KEY_ASKS.format(subject=subject))
    return lines


def _conversion_launch_line(harness: Any, launch_key: dict, suffix: str | None) -> str | None:
    """The D-17 line a start that did not launch hands the REPL: the posture the first launch will
    have, then the conversion's suffix. Only when that launch really binds beyond loopback (the same
    rule `_start_managed_server` applies to its announcement), else None."""
    srv = getattr(harness, "server", None)
    if srv is None or not suffix:
        return None
    from localharness.provider import server as managed_server

    keyed = bool(launch_key)
    exposed, posture = managed_server.effective_posture(
        srv, keyed=keyed, decision=managed_server.launch_decision(srv, keyed))
    return f"{posture or 'every interface'} — {suffix}" if exposed else None


PROPS_READ_DEADLINE_SECONDS = 3.0
PROPS_MAX_BYTES = 64 * 1024
PROPS_PATH_MAX = 4096
_PROPS_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _props_model_path(base_url: str, launch_key: Mapping[str, str]) -> str | None:
    """D-08: the weights path a llama.cpp server on THIS machine reports at `GET /props`
    (`model_path`), read with the same bearer `probe_window` uses. Blocking httpx: the caller runs it OFF
    the event loop. Only a loopback `base_url` is asked (a remote server's path names a file on its own
    machine and must not be filed under the key `init` uses for the local one); the read is one
    streamed GET with no redirect followed, an overall deadline and a 64 KiB cap, and only a 200 counts.
    The value must be an absolute path of at most 4096 characters with no control character. Silent on
    every failure (a timeout, an HTTP error, a body that is not JSON, a missing or unfit field all read
    None): it is stored as a path string the ranking compares to discovered files and never opens."""
    import time
    from urllib.parse import urlparse

    import httpx

    from localharness.provider.server import auth_headers

    try:
        if (urlparse(base_url).hostname or "").lower() not in _LOOPBACK_HOSTS:
            return None
        headers = auth_headers(launch_key.get("api_key")) or None
        deadline = time.monotonic() + PROPS_READ_DEADLINE_SECONDS
        chunks: list[bytes] = []
        size = 0
        with httpx.stream("GET", f"{base_url.rstrip('/').removesuffix('/v1')}/props",
                          timeout=PROPS_READ_DEADLINE_SECONDS, headers=headers,
                          follow_redirects=False) as response:
            if response.status_code != 200:
                return None
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > PROPS_MAX_BYTES or time.monotonic() > deadline:
                    return None
                chunks.append(chunk)
        body = json.loads(b"".join(chunks))
        value = body.get("model_path") if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001 — a ledger convenience: any failure is "not recorded"
        return None
    if (isinstance(value, str) and 0 < len(value) <= PROPS_PATH_MAX and os.path.isabs(value)
            and not _PROPS_CONTROL.search(value)):
        return value
    return None


def _fmt_temp(x: float) -> str:
    """A temperature as the line prints it: 0.6 -> "0.6", 1.0 -> "1.0", float32 0.8000000119 -> "0.8"."""
    return str(round(x, 3))


def _sampling_line(last: Any, *, temperature: float | None, rung: str | None,
                   server_default: float | None) -> str | None:
    """The sampling line, or None when there is nothing to say (D-27): no rung set a temperature, the
    server's default is unknown, the two agree (within 1e-3), or `last` already holds these facts."""
    if temperature is None or not rung or server_default is None:
        return None
    if abs(temperature - server_default) <= 1e-3:
        return None
    if last is not None and (last.temperature, last.rung, last.server_default) == (
            temperature, rung, server_default):
        return None
    return SAMPLING_LINE.format(temp=_fmt_temp(temperature), rung=rung, own=_fmt_temp(server_default))


def _say_sampling(probe: Any, *, temperature: float | None, rung: str | None, ledger: Path,
                  provider_type: str, model: str, say: Callable[[str], None] | None = None) -> None:
    """Print the sampling line once on change and record the facts (R-07c). `probe.sampling` None
    means the server did not say (a flaked probe, another runtime): nothing to compare, nothing
    recorded, so a good baseline is never replaced by Nones. The ledger is a convenience: a start
    never fails on it."""
    server_default = getattr(probe, "sampling", None)
    if server_default is None:
        return
    from localharness.provider.speed_stats import last_sampling, record_sampling
    try:
        last = last_sampling(ledger, provider_type, model)
    except OSError:
        last = None
    line = _sampling_line(last, temperature=temperature, rung=rung, server_default=server_default)
    if line is not None:
        (say or (lambda text: console.print(text, markup=False, soft_wrap=True)))(line)
    try:
        record_sampling(ledger, provider_type, model,
                        temperature=temperature, rung=rung, server_default=server_default)
    except OSError:
        pass  # the ledger is a convenience: a start never fails on it


def _attach_window_report(harness: Any, provider_type: str, served: int | None, trained: int | None,
                          *, harness_launched: bool) -> "tuple[Any, str | None]":
    """The window report for a start that ATTACHES to a server, and the undo kind its second line names.

    A llama.cpp record with a model file gets the fit computed beside the running server (the window
    already served counts as served plus what the free memory adds) and `trained` falls back to the
    file's own header. No record, or any other runtime, gets no fit (`mode "not-computed"`): for
    llama.cpp `window_lines` then says nothing (the no-record pointer is Phase 5's, D-34) and the
    Ollama / LM Studio single lines come from `runtime=provider_type`.

    The undo, by precedence (D-38): `"record"` when the record sets `context`; `"extra_args"` when a
    `-c` rides `extra_args`; `None` when the harness launched the incumbent and `context` is unset
    (nothing to undo); else `"hand"`, a foreign incumbent that the next harness launch replaces."""
    from localharness.provider import autosense as _auto

    srv = harness.server
    fits = free = total = weights = None
    undo: str | None = None
    mode = "not-computed"
    if srv is not None and srv.runtime == "llamacpp" and srv.model:
        fi = _auto.compute_fit(srv, harness.autosense, attached_served=served)
        fits, weights = fi.fits, fi.weights
        if fi.memory is not None:
            free, total = fi.memory.free_bytes, fi.memory.total_bytes
        trained = trained or fi.header.get("context_length")
        if srv.context is not None:
            undo, mode = "record", "record"
        elif _auto.has_ctx_arg(srv.extra_args or []):
            undo, mode = "extra_args", "extra_args"
        elif harness_launched:
            undo, mode = None, "fit"
        else:
            undo, mode = "hand", "fit"
    report = _auto.WindowReport(
        served=served, trained=trained, fits=fits, free_bytes=free, total_bytes=total,
        weights_bytes=weights, asked=None, mode=mode, attached=True,
        verified=served is not None, runtime=provider_type)
    return report, undo


def _route_memory_logs_to_file(agent_dir: Path) -> Path:
    """Interactive REPL only: send the memory subsystem's stdlib logs to a file instead of
    the terminal (#20). consolidation.py + mining.py log via `logging.getLogger(__name__)`
    and the interactive start path configures no handler, so their WARNING/EXCEPTION records
    surface through `logging.lastResort` on stderr — landing in the REPL over the input
    prompt while a background pass runs (the "something is broken" incident). Attaching a
    file handler to `localharness.memory` and setting `propagate=False` keeps full detail in
    `<agent_dir>/memory.log` and off the terminal. bench/eval use their own
    `logging.basicConfig` and never call this — non-interactive channels are untouched."""
    log_path = agent_dir / "memory.log"
    mem_log = logging.getLogger("localharness.memory")
    already = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None) == str(log_path)
        for h in mem_log.handlers
    )
    if not already:
        touch_private(log_path)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        mem_log.addHandler(handler)
    mem_log.setLevel(logging.DEBUG)   # the file keeps full detail…
    mem_log.propagate = False         # …and nothing bubbles to root/lastResort (the stderr leak)
    return log_path


async def _probe_llm(
    llm: Any, max_retries: int = 3, delay: float = 2.0
) -> tuple[bool, str | None, int | None, str | None]:
    """Probe LLM reachability with retry for cold start.

    Returns (reachable, probed_tool_call_mode, served_context_window, probe_error). On a clean
    probe error is None. On reachable=True a non-None probe_error is a DEGRADATION NOTE (tools
    rejected with HTTP 400, or probe inconclusive) for the caller's startup warnings, not a
    failure. On reachable=False mode/window are None and probe_error carries the concrete cause
    (a 404 naming the model, or a connection error) for the caller's message (#44).

    detect_capabilities() NEVER raises — it reports via CapabilityResult. The gate is
    CapabilityResult.server_reached (reachability), NOT probe_error's text: any probe the server
    ANSWERED for this model — native confirmed, tools rejected with 400, or inconclusive (answers
    arrived but no tool call was ever observed) — returns reachable=True with the probed mode.
    Only transport-level failures (connection refused, 404 not-served, timeout, auth) retry here
    (the cold-start purpose) and, exhausted, abort startup rather than proceed against a dead
    endpoint and then misattribute the cause at the TokenCounter step. The 0.12.0 gate instead
    string-matched probe_error and read the probe's own xml FALLBACK as a dead endpoint — the
    probe announced "falling back to xml", then start died on "cannot reach model" with the
    server up and serving. (probe_error is None still implies reached, for fabricated results
    predating the field.)

    The served window is the single source of truth for the effective context budget — callers
    must use it rather than the config default. Callers must feed the probed mode into LLMConfig
    rather than the stored provider.supports_function_calling flag (FIDEL-04).
    """
    import asyncio as _asyncio
    probe_error: str | None = None
    for attempt in range(max_retries):
        try:
            result = await llm.detect_capabilities()
            if result.server_reached or result.probe_error is None:
                return True, result.tool_call_mode, result.context_window, result.probe_error
            probe_error = result.probe_error
            if _classify_probe_failure(probe_error) == "unreachable":
                break  # the probe already spent its attempts on a dead endpoint; outer retries just replay it
        except Exception as exc:  # detect_capabilities shouldn't raise, but never wedge on a surprise
            probe_error = str(exc)
        if attempt < max_retries - 1:
            await _asyncio.sleep(delay)
    return False, None, None, probe_error


def _classify_probe_failure(probe_error: str | None) -> str:
    """Conservatively classify a hard probe failure for the startup message: 'unreachable' (a
    connection/timeout error — nothing is listening at the endpoint), 'unserved' (a 404 naming a
    missing model — the port IS listening), or 'unknown' (fall back to the generic message). Parses
    the openai client's own exception text, not arbitrary model output, so the token set is
    controlled; a served HTTP error BODY (e.g. a 404 body) proves the port is up, so it reads as
    'unserved', never 'unreachable'."""
    e = (probe_error or "").lower()
    if any(s in e for s in (
        "connection error", "connection refused", "connection reset", "failed to connect",
        "could not connect", "cannot connect", "timed out", "timeout", "getaddrinfo",
        "errno 111", "name or service not known", "max retries",
        # The harness's OWN vocabulary: for a local endpoint the inference gate's TCP pre-flight
        # (provider/client.py _inference_gate) raises "... unreachable (TCP connect failed) ..."
        # BEFORE any openai call, so openai's "Connection error." can no longer occur there — the
        # dominant real failure (nothing listening) was classifying as 'unknown'.
        "unreachable", "tcp connect", "connect failed",
    )):
        return "unreachable"
    if any(s in e for s in ("404", "not found", "does not exist", "no such model")):
        return "unserved"
    return "unknown"


# ContextConfig.max_context_tokens is ge=1_000 (config/models.py), so a derived bound below it
# CANNOT be written to config.yaml — the window guard must say the server is too small rather than
# print a remedy the schema rejects (the bound was once served−4_096, so a 4,096-window Ollama got
# "set max_context_tokens ≤ 0"; #145 made the bound the served window, so only a sub-1_000 server
# reaches this branch now).
MIN_CONFIGURABLE_CONTEXT_TOKENS = 1_000


def _reconcile_sole_served_model(
    resolved_model: str, live: list[str], reachable: bool, base_url: str
) -> tuple[str, str | None]:
    """Reconcile a configured model name against what the endpoint is ACTUALLY serving.

    A single-model runtime (vLLM / llama.cpp) serves exactly ONE checkpoint, so a stale
    configured name has exactly one sensible meaning — `start` should pick up what is there
    rather than fail on a name. Returns ``(model_to_use, notice_or_None)``.

    Adopts ONLY when the endpoint is reachable AND serves exactly one model AND that model
    differs from the configured one. Two-or-more served models is a REAL choice (attach-only
    providers like Ollama/LM Studio serve many) and is never guessed at — the caller's normal
    unserved-model error path handles it.

    Returns a notice rather than adopting silently: this is a benchmarking harness, and a run
    whose recorded model differs from the one actually used is a corrupted result.
    """
    if not reachable or len(live) != 1 or live[0] == resolved_model:
        return resolved_model, None
    return live[0], (
        f"Configured model {resolved_model!r} is not served at {base_url}; "
        f"using {live[0]!r} (the only model there)."
    )


def _effective_max_context(served_window: int | None, cfg_window: int) -> int:
    """Single source of truth for the context budget.

    max_context_tokens MEANS the full served window — the reply reserve is subtracted at
    runtime by agent.context.response_reserve, inside this number — so the bound here is the
    SERVED window itself, not served-minus-reserve. Subtracting it again was the double reserve
    (#145): it made a config equal to the served window (exactly what `init` writes) fail a
    guard that exists only to catch a budget LARGER than the server can serve.

    The config value is honored when it fits inside the served window; otherwise the served
    window wins. If the server didn't report a window, the config value is the only signal.
    Below-minimum results (a server too small to run at all) are the caller's to report honestly.
    """
    if not served_window:
        return cfg_window
    return min(cfg_window, served_window)


def _resolve_timeout(agent_timeout: float | None, provider_timeout: float) -> float:
    """Per-agent timeout override wins when set; otherwise the provider default.

    AgentConfig.timeout_seconds was previously never read at runtime — the start
    path always passed provider.timeout_seconds — so the per-agent override the
    reference-architecture docs tell slow-decode users to set was dead config."""
    return agent_timeout if agent_timeout is not None else provider_timeout


def _ensure_packaged_tools(config_dir: Path) -> None:
    """Install packaged helper scripts into <config-dir>/tools (idempotent).

    The frontend-designer builtin shells out to design-screenshot.js by path —
    ship it with the package so the builtin works out of the box. A missing or
    uncopyable asset must never block start; the agent reports the gap itself."""
    tools_dir = config_dir / "tools"
    dest = tools_dir / "design-screenshot.js"
    if dest.exists():
        return
    try:
        from importlib import resources
        src = resources.files("localharness").joinpath("assets", "design-screenshot.js")
        tools_dir.mkdir(parents=True, exist_ok=True)
        touch_private(dest)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        dest.chmod(0o700)  # owner-only and executable
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[yellow]⚠ could not install design-screenshot.js: {exc}[/yellow]")


# Agent discovery lives in ConfigLoader (agent_yaml_paths / discover_agents) — phase 38 (#150)
# folded start's own copy into it. start, `agent list` and the loader had three implementations
# with three different malformed-YAML behaviors; the roster below reads the only one that remains.


def _skipped_agent_file(f: Path, exc: Exception) -> None:
    """The user-visible half of discover_agents' warn-and-skip.

    ConfigLoader logs, but config/ must never import a rich console from cli/ — so the caller
    supplies the printer. This warning is load-bearing: the mint-refusal below points at it
    ("see the warnings above") before it refuses to overwrite an unparseable root agent.

    escape() around the path and the reason, markup outside — agent_cmd's twin of this function
    carries the same fix and the same reason: a project folder legally named `[/red]proj` made
    the one message whose job is naming the broken file raise MarkupError and kill the session.
    """
    err_console.print(
        "[yellow]⚠ " + escape(f"skipping unreadable agent file {f}: {exc}") + "[/yellow]",
        soft_wrap=True,
    )


def _migrate_legacy_root_agent_yaml(agents_dir: Path) -> None:
    """Phase 33.1 (ORCH-01/03): one-time root-agent rename on the global agents dir —
    agents/default.yaml -> agents/orchestrator.yaml with the name: field REWRITTEN
    (a bare file rename is not enough: discovery reads the name: key, and the legacy
    file carries name: default). The MemoryStore data directory migrates separately
    inside MemoryStore.open() (Phase 33.1 plan 01).

    GLOBAL-ONLY, by contract: this function writes and deletes files, so it is only ever handed
    `global_config_dir(...)/"agents"`. A workspace's own stray `agents/default.yaml` is left
    exactly as the user wrote it — the harness does not rewrite files in a project folder it was
    not asked to modify. There is no per-layer variant of this migration and there should not be
    one; if a workspace ever needs migrating, that is an explicit user-invoked command.

    Idempotent + crash-safe by construction:
    - no default.yaml -> no-op (fresh install, or already migrated);
    - default.yaml whose name: is not 'default' -> no-op (not the minted root);
    - orchestrator.yaml already exists:
        * parsed-equal to the would-be migration -> crash remnant of a previous run
          that died between write and unlink -> finish the job (unlink default.yaml);
        * different content -> GENUINE collision (the user has their own
          'orchestrator' agent): refuse loudly, keep the legacy root under its old
          name — never merge, never clobber;
    - normal path: write orchestrator.yaml FIRST, then unlink default.yaml (a crash
      between the two leaves both files; the remnant branch completes it next start).
    """
    legacy = agents_dir / "default.yaml"
    if not legacy.exists():
        return
    try:
        data = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
    except Exception:
        return  # unreadable legacy yaml: leave it to discovery's tolerant path
    if not isinstance(data, dict):
        return  # a list/scalar default.yaml is not an agent config — don't crash startup
    if data.get("name", "default") != "default":
        return
    data["name"] = "orchestrator"
    target = agents_dir / "orchestrator.yaml"
    if target.exists():
        try:
            existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = None
        if existing == data:
            legacy.unlink(missing_ok=True)  # crash remnant — the migrated copy is already in place
        else:
            console.print(
                "[yellow]Warning:[/yellow] cannot rename the root agent 'default' -> "
                "'orchestrator': agents/orchestrator.yaml already exists (an "
                "unrelated agent). Keeping the root under its old name 'default'. "
                "To resolve, rename your 'orchestrator' agent — note it can no "
                "longer be delegated to (the root's name is guarded)."
            )
        return
    touch_private(target)
    target.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    legacy.unlink(missing_ok=True)  # two-process first-start race: the other may have unlinked


def _migration_clauses(plan: Any, config_file: Path,
                       written: "frozenset[Path] | None" = None, *, nulls_held: bool = False) -> str:
    """The receipt's per-change clauses, rich-safe (each file-supplied part escaped). With
    `written` (a partial migration, WR-30) only the changes of the files actually rewritten are
    named; None names every change of the plan. `nulls_held` (SEC-01: config.yaml could not be
    written, so no agent file had init's values cleared) drops the per-file init-nulls clauses.

    escape(): a path under the user's config dir reaches rich as data — this receipt fires on the
    FIRST start after an upgrade, so under a markup-named config dir it crashed every session until
    the config was hand-edited."""
    from localharness.config import migrate as _migrate

    def done(path: Path) -> bool:
        return written is None or path in written

    config_done = done(config_file)
    removed_note = (
        f"; removed permissions.{_migrate.DEAD_KEY} (gone in v0.14 — grants live in the global "
        "store)" if config_done and plan.removed_allow_patterns is not None else ""
    )
    # The sidecar files are named, not counted: this receipt is the only notice a user gets that
    # an agent file on disk was rewritten, and "1 file cleaned" does not say which (finding R10).
    for sidecar in plan.sidecars:
        if sidecar.removed is not None and done(sidecar.path):
            removed_note += "; cleaned " + escape(f"{_migrate.DEAD_KEY} from {sidecar.path.name}")
    # init's own 100 / 32000 / 8000, cleared once under its own stamp: one clause per file, each
    # naming it (a stamp-only rewrite of config.yaml says nothing — no value was changed).
    if config_done and plan.nulled:
        removed_note += _migrate.init_nulls_clauses(plan.nulled) + " (config.yaml)"
    elif config_done and plan.init_nulls_due:
        removed_note += _migrate.INIT_NULLS_STAMP_CLAUSE  # IN-03: say why config.yaml was rewritten
    for sidecar in plan.sidecars:
        if sidecar.nulled and done(sidecar.path) and not nulls_held:
            removed_note += (
                _migrate.init_nulls_clauses(sidecar.nulled)
                + " (" + escape(_migrate.sidecar_label(sidecar.path)) + ")"
            )
    # A launched vLLM server that predates server.bind_all keeps listening on every interface, as
    # it did: written for the user once, said once, and doctor's row says how to undo it.
    if config_done and plan.bind_all_paths:
        removed_note += (
            "; kept the model server reachable from other machines ("
            + escape(", ".join(f"{p}.bind_all: true" for p in plan.bind_all_paths))
            + ") — `localharness doctor` says how to keep it on this machine"
        )
    return removed_note


def _auto_migrate_deny_defaults(config_file: Path) -> None:
    """First-start-after-upgrade fold-in of new shipped security defaults (issue #15).

    Runs the SAME engine as `localharness config migrate`, revision-gated: silent with zero
    writes when the config is already at the current defaults revision (the common path, and
    the removal-respect path — a deliberately-deleted default stays deleted). It also fires on
    a config — or an `agents/`/`divisions/` file — still carrying v0.14's removed
    `permissions.allow_patterns`, whatever the stamp: running BEFORE the loader is what lets that
    key be repaired instead of read, and a stamped-current config gets the removal ALONE (no deny
    default the user deleted is re-added). Best-effort: a migration failure NEVER blocks startup
    — we warn once and continue with the on-disk config.
    """
    from localharness.config import migrate as _migrate

    try:
        original, plan = _migrate.load_plan(config_file)
    except _migrate.MigrationError:
        return  # missing/unparseable → the loader below surfaces it with a proper error
    if plan is None:
        return  # already current: no output, no writes

    try:
        backups = _migrate.apply(config_file, original, plan)
    except _migrate.PartialMigrationError as exc:
        # WR-02 / CR-20: every other file WAS rewritten, config.yaml and its deny defaults
        # included; they are named with their backups, never passed over in silence, and (WR-30)
        # with the same per-change clauses the success receipt prints for the files that were
        # written: a network-exposure key added or a budget value cleared is said here too. A
        # failed file is recorded as owed (CR-30), so `config migrate` (or the next start) really
        # does try it again.
        written = "".join(
            f"; rewrote {path} (backup at {backup})" for path, backup in zip(exc.written, exc.backups))
        config_written = config_file in exc.written
        err_console.print(
            "[yellow]⚠[/yellow]  " + escape(f"Could not finish updating the config: {exc}{written}")
            + _migration_clauses(plan, config_file, frozenset(exc.written),
                                 nulls_held=exc.nulls_held) + ". "
            + "Run 'localharness config migrate' to finish; "
            + ("continuing; config.yaml was updated." if config_written
               else "continuing with the current config."),
            soft_wrap=True,
        )
        return
    except Exception as exc:
        err_console.print(
            "[yellow]⚠[/yellow]  " + escape(f"Could not auto-update security defaults: {exc}. ")
            + "Run 'localharness config migrate' to apply them; continuing with the current config.",
            soft_wrap=True,
        )
        return
    if not backups:
        return  # CR-30: only the list of owed files changed (a file on it was fixed by hand)

    removed_note = _migration_clauses(plan, config_file)
    head = (
        "Config repaired" if plan.config_unchanged
        else "Config updated" if plan.defaults_unchanged
        else (f"Security defaults updated (revision {plan.from_revision} → {plan.to_revision}): "
              f"added {len(plan.added)} deny pattern(s)")
    )
    console.print(
        f"[cyan]i[/cyan]  {head}{removed_note} — "
        + escape("backups at " + ", ".join(str(b) for b in backups)),
        soft_wrap=True,
    )



async def _announce_presence(channel: Any, *, config_dir: str | None, agent: str,
                             channel_mode: str, session_id: str, workspace: Any) -> None:
    """Register this session and warn — loudly, by name — if another already drives this agent.

    Its own function rather than a block inside `_start_async` so it can be TESTED. It was a
    block, and deleting the `set_co_tenants` line left the whole suite green while
    `GET /api/health` silently reported no co-tenants forever — a checkmark on a lie, caught in
    review rather than by a test, which is the argument for the seam.

    WARNS and never refuses (owner ruling 8): `history.jsonl` and `compact.md` take unlocked
    appends, so two live sessions on one agent can interleave and a long append can tear. That
    is a hazard to be told about, not a reason to refuse somebody their own second session.
    """
    try:
        import os

        from localharness.config import session_presence

        co_tenants = session_presence.register(
            config_dir, agent=agent, channel=channel_mode,
            session_id=session_id, workspace=workspace,
        )
        if hasattr(channel, "set_co_tenants"):
            channel.set_co_tenants(co_tenants, lambda: session_presence.live_sessions(
                config_dir, agent=agent, workspace=workspace, exclude_pid=os.getpid()))
        if co_tenants:
            await channel.send_error(session_presence.warning(co_tenants, agent=agent))
    except Exception:  # noqa: BLE001 — a lost warning never costs a session its start-up
        log.warning("session_presence_failed", exc_info=True)


async def _start_async(agent_name: str | None, verbose: bool, debug: bool, config_dir: str | None,
                       channel_mode: str = "terminal", subagents: bool = False,
                       model_override: str | None = None, list_models: bool = False,
                       no_input: bool = False, show_reasoning: bool = False,
                       acp_channel: Any = None, mobile_channel: Any = None,
                       resume: Resume | None = None, trust_project: bool = False,
                       fresh_thread: bool = False) -> Restart | None:
    """Async entry point: discover agent, wire dependencies, run REPL.

    `trust_project`: `--trust-project` — this run trusts the project it stands in, recording
    nothing (the machine's own files are decided as always).

    Returns a `Restart` when the REPL ended for `/plugins enable|disable` (a terminal session):
    start_app runs the plugin's step, then calls this again with `resume` — the conversation, the
    eviction store, the typed-ahead lines and /mode carried over, a fresh sitting id, and a compact
    indicator in place of the banner. Also when a mobile session went to sleep: the same handle,
    already written beside the session log, which the next mobile bring-up takes from disk and
    passes here as `resume` on its own. `fresh_thread` (the phone's new-chat verb) discards a
    sleeping thread instead of waking it. Otherwise None.

    `mobile_channel` is the PWA adapter (`channels/mobile.MobileChannel`) when this session is being
    driven from a phone. Passed in for the same reason `acp_channel` is — the HTTP server is up
    and serving before any session exists, so the channel outlives and precedes this call — but
    it takes the opposite exit: the web branch does NOT replace the REPL. That is the hybrid of
    the web PRD §6.0, and picking either pure shape would have cost something real. ACP's shape
    buys reachability-before-a-session and costs the REPL's slash commands, the input router, the
    pending resolver and `UserMessage` publishing — which is why an ACP session's log contains no
    user turns at all. Discord's shape buys all of those and cannot be reached before a session.
    The mobile channel is built early like ACP and driven by `OrchestratorREPL` like Discord.

    `acp_channel` is the Zed/ACP adapter (`channels/acp.AcpChannel`) when this session is being
    driven over the Agent Client Protocol. It is passed IN rather than built here because the
    protocol handshake happens long before a session exists — `initialize` and `session/new` must
    answer immediately, and only the first `session/prompt` calls this function (PRD §4). When it
    is set, it replaces the channel and, at the end, replaces the REPL: ACP turns arrive as
    JSON-RPC requests instead of lines pulled from `read_input`. Everything between — config
    layers, provider bring-up, memory, MCP, the tool registry, the permission gate, the subagent
    fleet — is the same path a terminal or Discord session takes, which is the point of passing
    the channel in rather than forking a third session builder.
    """
    # FIRST, before anything is read or built. An unknown `--channel` used to fall through to the
    # terminal SILENTLY — `--channel discrod` started an ordinary terminal session and nothing
    # anywhere said the flag had been ignored. It is refused here rather than at the CLI boundary
    # so every caller of this function gets the same answer, and before the config lookup so the
    # refusal does not depend on what happens to be on disk: a typo in the channel is a typo
    # whether or not the box has been `init`ed yet.
    _refuse_unbuildable_channel(channel_mode, mobile_channel, acp_channel)

    import time as _time
    import uuid

    from localharness.agent.context import (
        CompactionPipeline,
        ContextManager,
        TokenCounter,
        context_size_kwargs,
    )
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.channels.terminal import TerminalChannel
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.slash_commands import set_plugin_rows
    from localharness.config.loader import ConfigLoader
    from localharness.config.paths import global_config_dir, resolve_config_dir, resolve_runtime_path
    from localharness.core.bus import EventBus
    from localharness.core.agent_dir import _migrate_legacy_root_agent_dir
    from localharness.plugins.api import PluginPaths, SessionInfo
    from localharness.plugins.lifecycle import start_plugins, stop_plugins
    from localharness.plugins.slot import MemorySlot
    from localharness.plugins.resolve import resolve
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.tools.hooks import HookSystem
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    cfg_path = resolve_config_dir(config_dir)
    # GPU server state is machine-wide (C2 single-pidfile invariant, ROADMAP Standing Invariant):
    # <global>/vllm/{server.pid,serve.log,venv/}. Derived from the RAW --config-dir so it stays
    # global no matter what cfg_path comes to mean when workspace layering lands.
    server_cfg_path = global_config_dir(config_dir)
    config_file = cfg_path / "config.yaml"

    # No config → welcome message + exit
    if not config_file.exists():
        from localharness.orchestrator.router import AgentCreationFlow
        console.print(AgentCreationFlow.no_config_message())
        raise typer.Exit(0)

    # First start after a package upgrade: fold in any newer shipped security defaults
    # (revision-stamped, additive, backed up) BEFORE the config is loaded. Best-effort —
    # never blocks startup.
    _auto_migrate_deny_defaults(config_file)

    # Load harness config.
    # v0.13 LAYR-01: the nearest trusted `.localharness/` at or above CWD layers over the global
    # dir. None whenever --config-dir/LOCALHARNESS_DIR/LOCALHARNESS_HOME was explicit, nothing was
    # found, or trust was withheld — in which case this is byte-identical to v0.12 (LAYR-03).
    # The RAW flag value, not cfg_path: "was this explicit" does not survive resolution.
    from localharness.cli.workspace import (
        MCP_NOT_STARTED_LINE, TRUST_PROJECT_ENV, MachineTrust, ProjectTrust, _stdin_is_a_terminal,
        decide_machine_trust, decide_project_trust, resolve_workspace_layer, settle_startup_trust)
    interactive = False if no_input else None
    if mobile_channel is not None:
        # The mobile channel is already live and a client is already attached (its SSE connection is
        # what triggered this bring-up), so the one question a phone CAN answer at this point is
        # the one about an outside `.localharness/`. `resolve_workspace_layer`'s asker is
        # synchronous and runs on a worker thread, so the channel bridges the answer back with
        # `run_coroutine_threadsafe` — the same bridge ACP documents. Skipping it does not fail
        # loudly: it silently makes an outside workspace invisible forever, which is exactly the
        # phase-39 failure the trust dialog was added to prevent.
        workspace = await asyncio.to_thread(
            resolve_workspace_layer, config_dir, asker=mobile_channel.trust_asker()
        )
    else:
        workspace = resolve_workspace_layer(config_dir, interactive=interactive)
    # The ONE startup question (owner bar 2026-09-11). It settles trust for this workspace root
    # and, when the project has no state store at all, creates one — the moment a person is
    # demonstrably in a project and about to work in it (owner ruling 2026-09-04). Every guard
    # for "don't ask" lives inside it, including the ones that make it silent in scripts.
    #
    # It runs whether or not a layer was found, and it runs HERE rather than after the channel is
    # live, for two reasons: it is the only point before any session store is opened (so the
    # "have I been here before" check cannot count this run's own files), and a yes has to return
    # a layer that is active for THIS session. `cli/session_trust` reads the record it writes and
    # asks nothing; on a channel that cannot use stdin — Zed, Discord — this is silent and that
    # one asks instead.
    trust_flag = trust_project or os.environ.get(TRUST_PROJECT_ENV) == "1"
    settled = settle_startup_trust(config_dir, interactive=False if trust_flag else interactive)
    if workspace is None:
        workspace = settled
    # What may start, load or loosen this run? Decided once, here, before anything is loaded:
    # the project's own agent files (their MCP servers) and the machine's own (servers, embedding
    # model, loosened permissions). Asked only on a terminal, and only about what changed.
    # `--list-models` loads no agent and starts nothing, so it decides (and asks) nothing.
    can_ask = channel_mode == "terminal" and not no_input and _stdin_is_a_terminal()
    project_trust = ProjectTrust(True) if list_models else decide_project_trust(
        workspace, ask=can_ask, trust_flag=trust_flag, channel_mode=channel_mode)
    # Sensed once per start, before the machine-trust decision; this release uses it only in the
    # declared-only refusal text (D-09).
    path_llama_server = None if list_models else which_outside_cwd("llama-server")
    # Then what the start can tell before it decides (D-08, D-16, D-20): is it about to launch, is a
    # PATH binary to be shown, does a 0.17 `--host` need converting. Its writes are made here, before
    # the one question, so the question shows them as ordinary uncovered entries.
    sensed = SensedStart(True, {}) if list_models else _sense_before_trust(
        cfg_path, can_ask=can_ask, path_binary=path_llama_server)
    machine_trust = MachineTrust() if list_models else decide_machine_trust(cfg_path, ask=can_ask,
        launching=sensed.launching, annotations=sensed.annotations)
    _after_no_on_path_binary(cfg_path, machine_trust, can_ask=can_ask, path_binary=path_llama_server)
    _after_no_on_config_ini(cfg_path, machine_trust, can_ask=can_ask, ini_written=sensed.ini_written,
                            ini_prior=sensed.ini_prior)
    # What the first launch of this session prints (D-15, D-17): the withheld-posture lines and the
    # undo line. Start's own down-server launch prints them; a start that only attaches (or waits, or
    # is refused) hands them to the REPL's first launch instead (D-26).
    from localharness.config import trust as _trust_mod
    _snap_shown = {} if not machine_trust.withheld else {
        (e["file"], e["kind"], e["name"]): e["shown"]
        for e in _trust_mod.machine_snapshot(cfg_path)}
    pending = [*_posture_lines(machine_trust, _snap_shown),
               *([sensed.undo_line] if sensed.undo_line else [])]
    launch_notes = LaunchNotes(list(pending), sensed.converted)
    if machine_trust.line:
        err_console.print(machine_trust.line, markup=False, soft_wrap=True)
    loader = ConfigLoader(config_dir=cfg_path, local_config_dir=workspace,
                          project_trusted=project_trust.executables,
                          machine_withheld=machine_trust.withheld)
    if workspace is not None:
        # markup=False, like the model list above: a folder named `[old] proj` is legal
        # everywhere, and rich would either eat the bracket (naming a path that does not exist)
        # or raise MarkupError on `[/]`. `style=` alone does NOT disable markup parsing (39-04).
        # soft_wrap: rich otherwise folds a deep project path at the console width and the user
        # copies half a path out of the one line that says where their config came from (43-04).
        console.print(
            f"Workspace layer: {workspace}", style="dim", markup=False, soft_wrap=True
        )
    # v0.13 MEMS-01: WHERE THIS SESSION'S STATE LIVES. `cfg_path` keeps meaning the always-global
    # config layer — the kill file, plugins, packaged tools, the root-agent mint and the GPU daemon
    # all keep naming it. `state_dir` is where this session's WORK lands: the workspace layer when one
    # applies, else exactly `cfg_path`, so a workspace-less session is byte-identical (LAYR-03). Two
    # locals rather than one widened value: everything that must stay global says so by name, and a
    # future change to what one means cannot silently drag the other.
    state_dir = workspace if workspace is not None else cfg_path
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Cannot load config: {exc}")
        raise typer.Exit(1)
    # The machine's config dir holds keys, the phone token, grants and every session log: owner-only.
    # An older install made it at the umask; tighten it once, silently, best effort (a folder that
    # refuses is left as it is — doctor says so), and record that it happened. An explicit
    # --config-dir / LOCALHARNESS_DIR is the machine's dir for this run, so it is tightened too (R16).
    from localharness.core.private_files import PRIVATE_DIR_MODE, ensure_private_dir
    _previous_mode = ensure_private_dir(server_cfg_path)
    if _previous_mode is not None and harness.org.audit_log_path:
        from localharness.core.events import ComponentMutated
        try:
            await EventBus(persist_path=resolve_runtime_path(harness.org.audit_log_path, server_cfg_path)).publish(
                ComponentMutated(path="config_dir.mode", before_value=oct(_previous_mode),
                                 after_value=oct(PRIVATE_DIR_MODE), layer="user", actor="cli",
                                 actor_detail="localharness start made the config directory owner-only"))
        except Exception:  # noqa: BLE001 — a record that cannot be written never stops a start
            log.debug("the config dir tighten was not recorded", exc_info=True)
    from localharness.provider.server import required_key_kwargs
    # The key file is opened here and nowhere earlier: after decide_machine_trust (a withheld entry
    # has already dropped `server:`) and load_harness. `--list-models` runs no trust decision, so it
    # never opens the file. The key stays in memory (harness.provider.api_key) for this session.
    if not list_models:
        key_file_line = _read_session_key(harness)
        if key_file_line:
            err_console.print(key_file_line, markup=False, soft_wrap=True)
    # The key the launched server requires, or {} — server.require_api_key is opt-in (default off).
    launch_key = required_key_kwargs(harness.server, harness.provider.api_key)
    from localharness.tools.builtin.netguard import set_private_allowlist
    # The machine's fetch allowlist (machine-level only); the web tools read it per call.
    set_private_allowlist(harness.org.web_fetch_allow_private)

    # --list-models: a lightweight, config-only path — reuses the same live-probe `localharness
    # model` (bare) uses, exposed on `start` too since that's the command most people reach for
    # first. Never loads agents / starts a session.
    if list_models:
        from localharness.cli import model_ops
        provider = harness.provider
        try:
            live, reachable = model_ops.list_live_models(provider.base_url, **launch_key)
        except model_ops.MalformedModelListError as exc:
            err_console.print(
                f"[bold red]Error:[/bold red] the server at {provider.base_url} responded, but "
                f"the response wasn't understood ({exc})"
            )
            raise typer.Exit(2)
        configured_only = [m for m in provider.available_models if m not in live]
        if not reachable and not configured_only:
            err_console.print(
                f"[bold red]Error:[/bold red] could not reach {provider.base_url} and no "
                "configured models were found. Is the server running?"
            )
            raise typer.Exit(2)
        console.print("Models:")
        for m in live:
            mark = "  [active]" if m == provider.default_model else ""
            console.print(f"  {m}  (serving){mark}", markup=False)
        for m in configured_only:
            console.print(f"  {m}  (configured, not currently served)", markup=False)
        if not live and not configured_only:
            console.print("  (none)")
        console.print(
            "\nUse `localharness start --model <name>` for a one-off session, or "
            "`localharness model <name>` to persist a new default."
        )
        raise typer.Exit(0)

    _ensure_packaged_tools(cfg_path)

    # Discover agents (migrate the legacy root-agent YAML first so discovery reads the
    # rewritten name: field, not the stale name: default — Phase 33.1 ORCH-01/03)
    # The rename WRITES (creates orchestrator.yaml, unlinks default.yaml). It migrates the
    # mint-scaffolded ROOT agent, which stays global by phase 39's carve-out, so it names the
    # GLOBAL layer explicitly and can never be handed a workspace agents dir — a project folder
    # the user did not ask us to rewrite. Value-identical to `cfg_path` today by construction;
    # the guarantee is that a future change to what `cfg_path` means cannot drag a
    # file-deleting migration into someone's repository (39-03's pattern).
    _migrate_legacy_root_agent_yaml(global_config_dir(config_dir) / "agents")
    agents = loader.discover_agents(on_error=_skipped_agent_file)

    if agent_name:
        # --agent flag: find by name
        match = [a for a in agents if a.get("name") == agent_name]
        if not match and agent_name == "default":
            # Phase 33.1: the root agent was renamed default -> orchestrator; keep old
            # muscle memory / scripts working instead of hard-erroring (ORCH-03).
            match = [a for a in agents if a.get("name") == "orchestrator"]
            if match:
                console.print(
                    "[yellow]Note:[/yellow] the root agent was renamed 'default' -> "
                    "'orchestrator'; starting 'orchestrator'."
                )
        if not match:
            err_console.print(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found.")
            raise typer.Exit(1)
        selected_data = match[0]
    elif not agents:
        # Zero agents is NOT proof of a fresh install: discovery skips files it cannot parse, so a
        # one-character YAML typo in agents/orchestrator.yaml lands here too — and the mint below
        # used to write the 3-line default straight over it. Mint only when there is nothing to lose.
        unparsed = loader.agent_yaml_paths()
        if unparsed:
            err_console.print(
                "[bold red]Error:[/bold red] no agent could be loaded, but agent files exist: "
                f"{', '.join(str(p) for p in unparsed)}. Fix the YAML (see the warnings above) — "
                "refusing to overwrite it with the default orchestrator."
            )
            raise typer.Exit(1)
        # No agents: mint the root agent as 'orchestrator' (ORCH-01)
        console.print("[yellow]No agents configured. Creating the orchestrator (root agent)...[/yellow]")
        # Global on purpose: minting the machine's root agent into a project folder would surprise.
        agents_dir = cfg_path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        root_data = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
        import yaml as _yaml
        touch_private(agents_dir / "orchestrator.yaml")
        (agents_dir / "orchestrator.yaml").write_text(
            _yaml.dump(root_data, default_flow_style=False), encoding="utf-8"
        )
        agents = [root_data]
        selected_data = root_data
    elif len(agents) == 1 or not subagents:
        # Default: open straight to the banner. Pass --subagents to pick from the
        # multi-agent table instead.
        # Prefer 'default' if it still exists: post-migration that only happens when the
        # YAML migration REFUSED (a name collision with a user's own 'orchestrator'
        # agent), and the un-migrated legacy root must keep winning selection or the
        # user lands in a different agent and loses their memory continuity. Normal
        # installs have no default.yaml after migration, so 'orchestrator' wins.
        selected_data = default_root_agent(agents)
    else:
        # --subagents + multiple agents: show picker
        console.print(_agent_roster_table(agents))
        choice = IntPrompt.ask("Select agent", default=1)
        idx = max(1, min(choice, len(agents))) - 1
        selected_data = agents[idx]

    # Load full AgentConfig (uses ConfigLoader for inheritance). QA-16: a load error REFUSES the
    # session, exactly like a bad config.yaml above. It used to fall back to a bare
    # AgentConfig(name, role, model) without a word, which ran the agent without its mode (a
    # read-only agent ran auto), deny patterns, workspace_root, budget, MCP servers, memory and
    # context settings. No fallback is kept, because no legitimate path reaches this line without
    # a file: every agent here was read from one (discovered, or minted just above), and
    # `--agent <name>` with no file already stopped at "not found". A not-found error here means a
    # file whose `name:` is not its file name — still that file's settings, still refused.
    agent_name_str: str = selected_data.get("name", "orchestrator")
    try:
        agent_config = loader.load_agent(agent_name_str)
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"Cannot load agent '{agent_name_str}': {exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(1)
    # QA-18: an agent's own mode stands even when it is looser than the org's (no "most restrictive
    # wins"; that is the owner's call), and the session says so once, naming both.
    from localharness.agent.gate_types import MODE_STRICTNESS
    _own, _org = agent_config.permissions.mode, loader.org_mode()
    if _org is not None and MODE_STRICTNESS[_own] < MODE_STRICTNESS[_org]:
        console.print(
            f"⚠ The {agent_name_str} agent sets permissions.mode: {_own}, looser than "
            f"org.permissions.mode: {_org}. This session runs in {_own}; remove the agent's mode "
            "to use the org's.", style="yellow", markup=False, soft_wrap=True,
        )

    # --- Capability floor (P-A): sync the module flag from config. The ROOT agent's strip reads
    # tool declarations, so it runs after every global tool is registered (after step 5's plugins).
    from localharness.tools.capabilities import set_floor_enabled
    set_floor_enabled(harness.org.enforce_capability_floor)

    # Wire dependencies
    provider = harness.provider
    resolved_model = (
        agent_config.model
        if agent_config.model != "inherit"
        else provider.default_model
    )
    if model_override:
        # --model (session-only, never persisted): restricted to a model ALREADY being served.
        # A harness-managed single-model server (llama.cpp/vLLM) serves exactly one checkpoint at
        # a time; switching it means a restart, which `localharness model <name>` (persist +
        # restart) or the REPL `/model` command (live restart) already handle deliberately — this
        # flag must NOT trigger that, so it only accepts the model the managed server already
        # serves. An attach-only provider (Ollama/LM Studio) can serve several models at once, so
        # any already-live model is fair game.
        from localharness.cli import model_ops
        try:
            _live, _reachable = model_ops.list_live_models(provider.base_url, **launch_key)
        except model_ops.MalformedModelListError as exc:
            err_console.print(
                f"[bold red]Error:[/bold red] the server at {provider.base_url} responded, but "
                f"the response wasn't understood ({exc})"
            )
            raise typer.Exit(1)
        _choices = _live or list(provider.available_models)
        if model_override not in _choices:
            avail_hint = f" Available: {', '.join(_choices)}." if _choices else ""
            err_console.print(
                f"[bold red]Error:[/bold red] --model {model_override!r} is not currently "
                f"served at {provider.base_url}.{avail_hint}"
            )
            if harness.server is not None:
                err_console.print(
                    "This is a harness-managed single-model server — switching models restarts "
                    "it. Use `localharness model <name>` to persist + restart, or the REPL "
                    "`/model <name>` command for a live restart."
                )
            raise typer.Exit(1)
        resolved_model = model_override
    elif harness.server is None:
        # ATTACH mode with no --model: reconcile the configured name against the live endpoint
        # BEFORE probing. Probing a stale name burns three retries and prints the alarming
        # "falling back to xml" warning before any correction could apply; fixing the name first
        # means ONE clean probe, in the right tool-call mode. Skipped for a managed server, which
        # may legitimately not be up yet — its own launch path below covers that.
        from localharness.cli import model_ops as _mo
        try:
            _live_now, _reachable_now = _mo.list_live_models(provider.base_url, **launch_key)
        except _mo.MalformedModelListError:
            _live_now, _reachable_now = [], False
        resolved_model, _notice = _reconcile_sole_served_model(
            resolved_model, _live_now, _reachable_now, provider.base_url
        )
        if _notice:
            console.print(_notice)

    # Build initial client for the probe (tool_call_mode will be overwritten by probe result).
    _initial_cfg = LLMConfig(
        base_url=provider.base_url,
        model=resolved_model,
        api_key=provider.api_key,
        timeout_seconds=_resolve_timeout(agent_config.timeout_seconds, provider.timeout_seconds),
        queue_wait_seconds=provider.inference_queue_wait_seconds,  # #62 gate-wait ceiling
        temperature=agent_config.temperature,  # None unless a person set one (spec 20 §7)
    )
    # detect_capabilities SEEDS CapabilityResult.context_window from this config value and only
    # overwrites it from a vLLM-shape /v1/models answer (and MUTATES config.context_window as it
    # goes), so the seed captured here is the sentinel for "no window was ever server-reported".
    _seed_window = _initial_cfg.context_window
    _probe_client = LLMClient(_initial_cfg)

    # Startup probe — local LLMs may need warm-up; probe returns the real tool_call_mode (FIDEL-04)
    # AND the served context window (single source of truth for the budget).
    probe_ok, probed_mode, served_window, probe_error = await _probe_llm(_probe_client)
    # What a launch THIS start makes hands back (its LiveEndpoint: the window it got, the environment
    # names it dropped); empty when the server was already up or only loading — that start attaches.
    _launched: list = []
    _pending_warnings: list[str] = []
    if not probe_ok and harness.server is not None:
        # Harness-managed server (init guided setup) — start it instead of erroring
        # (covers reboots). If a pid is alive, it's mid-load: just wait.
        refusal: str | None = None
        try:
            refusal = await _start_managed_server(
                harness.server, server_cfg_path, provider.base_url, launch_key,
                path_binary=path_llama_server, autosense=harness.autosense, endpoint_out=_launched,
                notes=launch_notes)
            if refusal is None:
                probe_ok, probed_mode, served_window, probe_error = await _probe_llm(_probe_client)
        except (RuntimeError, TimeoutError, OSError) as exc:
            # OSError: server_pid()'s os.kill(pid, 0) liveness probe is POSIX-only semantics —
            # on Windows a stale pidfile raises a plain OSError instead of ProcessLookupError.
            # Managed-server lifecycle is POSIX-only for now; degrade to a message, not a traceback.
            from localharness.provider.server import RUNTIME_LABELS
            label = RUNTIME_LABELS.get(harness.server.runtime, harness.server.runtime)
            # escape(): `exc` can quote llama-server's own log tail (wait_ready) or a path — a
            # bracketed fragment there is not markup, and a stray `[/x]` raises MarkupError here.
            err_console.print(
                "[bold red]Error:[/bold red] " + escape(f"managed {label} failed to start: {exc}"),
                soft_wrap=True)
        if refusal is not None:
            # The one refusal line is the whole answer: no `Error:` line, no `doctor` pointer.
            err_console.print(refusal, markup=False, soft_wrap=True)
            await _probe_client.aclose()
            raise typer.Exit(1)
    # The probe client has done its job — the session's real client is rebuilt below with the
    # probe-derived mode — so release its httpx pool here (#154), ahead of the `typer.Exit(1)`
    # failure paths just after, which would else leave the sockets to GC on the way out.
    await _probe_client.aclose()
    if not probe_ok:
        # #44: name the concrete cause instead of a generic "Cannot reach model" — an unserved
        # model and a dead endpoint are different fixes — and point at the diagnostics.
        kind = _classify_probe_failure(probe_error)
        if kind == "unserved":
            avail = list(provider.available_models)
            avail_hint = f" Configured models: {', '.join(avail)}." if avail else ""
            err_console.print(
                f"[bold red]Error:[/bold red] model '{resolved_model}' is not served at "
                f"{provider.base_url} (probe: {probe_error}).{avail_hint}"
            )
        elif kind == "unreachable":
            err_console.print(
                f"[bold red]Error:[/bold red] model endpoint unreachable at "
                f"{provider.base_url} (probe: {probe_error})."
            )
        else:
            err_console.print(
                f"[bold red]Error:[/bold red] cannot reach model '{resolved_model}' at "
                f"{provider.base_url} (probe: {probe_error})."
            )
        err_console.print(
            "Run `localharness doctor` to diagnose, or `localharness model` to list served models."
        )
        raise typer.Exit(1)

    # config.yaml is the SINGLE SOURCE OF TRUTH for the window (now inheritance-resolved,
    # which kills the 61,440-in-a-131,072-world bug). We do NOT silently override it — we
    # VALIDATE against the served window and FAIL LOUD if it would 400 mid-session, so the
    # value the user sees in config.yaml is exactly what the agent runs on.
    from localharness.agent.context import clamp_response_tokens, response_reserve, probe_window
    # #132: a per-model pin IS this model's budget. Apply it BEFORE the guard so the check and the
    # session that follows run on the same number (exact name match; no map ⇒ the scalar, unchanged).
    # #137: resolve_budget is the ONE resolution `doctor` reads too — and CONFIRM an applied pin
    # here, because the only pin feedback that shipped fires on a /model swap (unreachable on a
    # single-model server), leaving the happy path silent about the budget it actually runs on.
    _pin, _pinned = agent_config.context.resolve_budget(resolved_model)
    if _pinned:
        agent_config.context.max_context_tokens = _pin
        console.print(
            f"[green]✓[/green] Context budget pinned for {resolved_model}: {_pin:,} tokens"
        )
    _cfg_window = agent_config.context.max_context_tokens
    # PROVIDER-AWARE served-window discovery for the guard. _probe_llm's window comes from
    # detect_capabilities' vLLM-shape /v1/models probe (max_model_len) — None for llama.cpp/
    # LM Studio/Ollama, so this fail-loud guard used to silently NO-OP for them (_effective_max_context
    # returns cfg unchanged on a None window). probe_window reads each runtime's real window
    # (llama.cpp /props, Ollama /api/ps, LM Studio /api/v0/models), so the guard FIRES when the
    # window is discoverable and cleanly discloses (config wins) when it isn't. ONE probe_window call
    # per start (D-03) feeds both this guard (its served window) and the window report below (its
    # served + trained). Blocking httpx → off the
    # event loop (#32); falls back to the probe's window (vLLM parity / a flaked provider probe).
    # The capability probe's window is only SERVER-REPORTED when it changed from the seed above;
    # unchanged, it is the harness's own default and must never drive a fail-loud guard (it aborted
    # every llama.cpp/Ollama/LM Studio/unknown start on a window no server ever served).
    _probe_window = served_window if served_window != _seed_window else None
    _probe = await asyncio.to_thread(
        probe_window, provider.base_url, resolved_model, provider.provider_type, **launch_key
    )
    _served = _probe.served or _probe_window
    _bound = _effective_max_context(_served, _cfg_window)
    # The window block (AUTO-02). `last` is read ONCE, before anything this start records, and one
    # verdict (`changed`) gates both the block and the #127 line below (D-35).
    from localharness.provider.autosense import window_changed, window_lines
    from localharness.provider.speed_stats import (
        default_speed_stats_path,
        last_window_report,
        record_model_path,
        record_window_report,
    )
    _ledger = default_speed_stats_path(cfg_path)
    last = last_window_report(_ledger, provider.provider_type, resolved_model)
    if _launched and _launched[0].window is not None:
        # This start launched the server: the report is what `activate` asked and the probe saw (a
        # launch carries no undo, D-36). A launch with no window (vLLM) falls to the attach report.
        report, undo, launched_here = _launched[0].window, None, True
    else:
        # Was the serving process launched by the harness (a live pidfile)? Only the undo sentence of
        # an attach report asks (D-38), and only a llama.cpp record has one; a failure reads "no".
        _harness_launched = False
        if harness.server is not None and harness.server.runtime == "llamacpp":
            try:
                from localharness.provider.lifecycle import strategy_for
                _harness_launched = _managed_server_running(
                    strategy_for(harness.server), harness.server, server_cfg_path)
            except Exception:  # noqa: BLE001 — wording only: degrade to a foreign incumbent
                _harness_launched = False
        # Off the event loop: the attach report reads the header, stats the weights and runs nvidia-smi.
        report, undo = await asyncio.to_thread(
            _attach_window_report,
            harness, provider.provider_type, _served, _probe.trained, harness_launched=_harness_launched)
        launched_here = False
        # D-08: every llama.cpp attach stores the weights path the server reports beside the alias it
        # serves, so a later `init` ranks the model that served before. Silent on any failure; a
        # launch (above), another runtime and a server that reports no path record nothing. start adds
        # no pointer to `init` here (D-42).
        if provider.provider_type == "llamacpp" and resolved_model:
            try:
                _weights = await asyncio.wait_for(
                    asyncio.to_thread(_props_model_path, provider.base_url, launch_key),
                    PROPS_READ_DEADLINE_SECONDS + 1)
            except (asyncio.TimeoutError, TimeoutError):
                _weights = None
            if _weights:
                try:
                    record_model_path(_ledger, "llamacpp", resolved_model, _weights)
                except OSError:
                    pass  # the ledger is a convenience: a start never fails on it
    changed = window_changed(last, report, harness.autosense.notify_ratio)
    # An attach to a runtime that reported nothing (no served, trained or fit number) has nothing to say.
    _nothing_known = report.served is None and report.trained is None and report.fits is None
    if launched_here or (changed and not _nothing_known):
        for line in window_lines(report, harness.autosense, undo):
            console.print(line, markup=False, soft_wrap=True)
    # Recorded whether or not it printed (D-22) — but a report that knows nothing (a probe that flaked
    # on a server still loading) must not replace a good baseline with Nones: the next healthy start
    # would see a change and say the whole block again.
    if not _nothing_known:
        try:
            record_window_report(
                _ledger, provider.provider_type, resolved_model, report.served, report.trained, report.fits)
        except OSError:
            pass  # the ledger is a convenience: a start never fails on it
    # D-27: a configured temperature that overrides the model's own default, said once on change.
    _say_sampling(
        _probe, temperature=agent_config.temperature, rung=loader.temperature_rung(agent_name_str),
        ledger=_ledger, provider_type=provider.provider_type, model=resolved_model)
    # D-25: the names the launch scrubbed from the server's environment, said again only when the
    # list changed. Names only, never values.
    # Every launch this start made records its list, an empty one included: a launch that dropped
    # nothing must reset the record, or a later repeat of an earlier drop goes unannounced.
    if _launched and harness.server is not None:
        from localharness.provider.speed_stats import last_dropped_env, record_dropped_env
        _runtime = harness.server.runtime
        _names = sorted(set(_launched[0].dropped_env))
        if _names != (last_dropped_env(_ledger, _runtime) or []):
            if _names:
                _pending_warnings.append(
                    f"launch environment dropped {', '.join(_names)}"
                    + (" — set server.env: for what llama-server should see" if _runtime == "llamacpp" else ""))
            try:
                record_dropped_env(_ledger, _runtime, _names)
            except OSError:
                pass
    # D-19/D-22: a request key this server refused in an earlier session, said again at start (the
    # probe sends no effort key, so the live set is empty here — the ledger is what remembers).
    from localharness.provider.client import EFFORT_BUDGET_KEY
    from localharness.provider.speed_stats import last_rejected_effort_keys
    try:
        _rejected_keys = last_rejected_effort_keys(_ledger, provider.provider_type, resolved_model) or []
    except OSError:
        _rejected_keys = []  # the ledger is a convenience: a start never fails on it
    for _key in _rejected_keys:
        _pending_warnings.append(EFFORT_UNENFORCED_LINE.format(key=_key))
    if "chat_template_kwargs" in _rejected_keys and EFFORT_BUDGET_KEY.get(provider.provider_type) is None:
        _pending_warnings.append(EFFORT_CUT_OFF_LINE)
    # The session agent's level: `start` runs on the primary and never reads `active_endpoint`
    # (spec 16 §7.2), so the endpoint rung is the primary's own `server.effort`.
    from localharness.config.effort import resolve_effort
    _effort_level, _effort_rung = resolve_effort(None, agent_config.effort, harness, provider.base_url)
    _effort_ratios = (harness.effort_budget.low_ratio, harness.effort_budget.high_ratio)
    # #145: a served window that cannot hold a reply reserve cannot run AT ALL — history is
    # allowed to fill the whole thing, so prompt + max_tokens overruns it on the first real turn
    # whatever the config says, and clamp_response_tokens has no reserve to fit the reply into.
    # Keyed to the SERVED window, not the resolved budget: a deliberately tiny budget on a roomy
    # server is safe (the request still fits the server) and must not be swept up in this.
    # (`_served and …` also narrows away the None/unknown case: no window read, no refusal.)
    _unrunnable = bool(_served and response_reserve(_served) <= 0)
    if _bound != _cfg_window:
        from localharness.config.defaults import DEFAULT_MAX_CONTEXT_TOKENS
        # #127: the untouched FACTORY default must never hard-block startup — on the reference
        # setup (served == default) it mathematically cannot fit under served-reserve, so the
        # guard aborted every fresh install. Default => auto-fit for this session (nothing
        # written to disk); an explicitly configured value keeps the fail-loud abort below.
        if (
            _cfg_window == DEFAULT_MAX_CONTEXT_TOKENS
            and _bound >= MIN_CONFIGURABLE_CONTEXT_TOKENS
            and not _unrunnable  # never auto-fit onto a window that cannot run
        ):
            if changed:  # D-23: said once per (runtime, model), again when the window moved
                console.print(
                    f"Context budget fitted to {_bound:,}: the factory default ({_cfg_window:,}) "
                    f"exceeds the window this server serves ({_served}). "
                    f"Set context.max_context_tokens to pin it explicitly."
                )
            agent_config.context.max_context_tokens = _bound
            _cfg_window = _bound
    # `or _unrunnable`: an explicit budget EQUAL to a too-small served window agrees with the
    # bound, so the disagreement gate alone would wave it through to a guaranteed 400.
    if _bound != _cfg_window or _unrunnable:
        # The bound printed is the SAME expression the check enforces — the old message derived its
        # own unfloored number and could tell the user to set an unsatisfiable "≤ 0".
        if _unrunnable or _bound < MIN_CONFIGURABLE_CONTEXT_TOKENS:
            err_console.print(
                f"[bold red]Error:[/bold red] the context window served for '{resolved_model}' "
                f"({_served}) is too small to run the harness: at or below 1,024 tokens nothing is "
                "left to hold the model's reply once history fits, so every request would overrun "
                f"the window (the configurable minimum is {MIN_CONFIGURABLE_CONTEXT_TOKENS}; the "
                "practical one is 1,025). Raise the window at the SERVER (llama.cpp `-c`, Ollama "
                "`OLLAMA_CONTEXT_LENGTH`, LM Studio's context length) and start again."
            )
        else:
            # #137: name the setting that actually GOVERNS this number. When the budget came from
            # a pin, "set context.max_context_tokens ≤ N" is a loop that changes nothing — the pin
            # outranks the scalar on the next start too.
            _key = (
                f"context.model_context_overrides['{resolved_model}']" if _pinned
                else "context.max_context_tokens"
            )
            err_console.print(
                f"[bold red]Error:[/bold red] {_key}={_cfg_window} exceeds the "
                f"window served for '{resolved_model}' ({_bound}). It would 400 mid-session. "
                f"Set {_key} ≤ {_bound} (under org: in config.yaml, or in the agent's yaml — it "
                f"is not a top-level key), or raise the served window at the server."
            )
        raise typer.Exit(1)

    # The session's per-reply cap — the agent's RESOLVED max_tokens (agent yaml -> division ->
    # org default_max_tokens) and nothing else. None means no rung set one, which is the default:
    # no cap is sent, the model decides when it is done, and the served window is the only bound.
    # Used by both halves of the shared reserve below (the client's request cap, and the
    # ContextManager's max_response_tokens the reserve is sized from) so the two cannot disagree.
    _output_cap = agent_config.max_tokens

    # Build the real LLMClient with the probe-derived tool_call_mode (FIDEL-04).
    # A model swap re-probes via _probe_llm before constructing the new LLMClient.
    llm_cfg = LLMConfig(
        base_url=provider.base_url,
        model=resolved_model,
        api_key=provider.api_key,
        timeout_seconds=_resolve_timeout(agent_config.timeout_seconds, provider.timeout_seconds),
        # #145: history may fill (budget - reserve), so a CONFIGURED output cap has to fit the
        # reserve or prompt + max_tokens overruns the served window and a strict server 400s.
        # The reserve grows to hold it on a normal window (response_reserve), and repl's swap
        # refit runs the same fit on the same two values so the two never disagree. No cap
        # configured (the default) => None => the request omits max_tokens entirely.
        max_tokens=clamp_response_tokens(_cfg_window, _output_cap),
        tool_call_mode=probed_mode or "native",
        queue_wait_seconds=provider.inference_queue_wait_seconds,  # #62 gate-wait ceiling
        temperature=agent_config.temperature,  # None unless a person set one (spec 20 §7)
        # D-41: the reconnect bound reaches the client the root loop and every child share; without
        # this line a real session keeps the LLMConfig default 0.0 (reconnect off).
        reconnect_wait_seconds=provider.reconnect_wait_seconds,
        provider_type=provider.provider_type,  # keys the measured-speed ledger (speed_stats)
        config_dir=cfg_path,  # …and WRITES it where the /model picker reads it (this session's dir)
    )
    llm = LLMClient(llm_cfg)

    start_time = _time.monotonic()

    # --- Startup state tracker ---
    # machine-level keys a workspace set (its config files, then its agent files)
    warnings: list[str] = [*loader.harness_warnings, *loader.agent_warnings]
    from localharness.config.cleartext import cleartext_warnings
    # a key headed over plain http to another machine: named, never refused
    warnings.extend(cleartext_warnings(harness))
    warnings.extend(_pending_warnings)
    plugins_loaded = 0
    mcp_connected = 0
    mcp_failed = 0

    # Reachable-but-degraded probe (see _probe_llm): say so in the startup summary instead of
    # blocking start — the endpoint is alive and serving the model.
    if probe_error:
        if probe_error.startswith("HTTP 400"):
            warnings.append(
                "provider rejects the `tools` param (HTTP 400) — tool calls run in taught-xml mode"
            )
        else:
            warnings.append(
                f"capability probe inconclusive ({probe_error}) — proceeding {probed_mode}; if "
                "tool calls arrive as plain text, check the runtime's tool-call parser flag "
                "(llama.cpp --jinja, vLLM --tool-call-parser)"
            )

    # --- 1. HARD requirements (abort on failure) ---
    # Phase 33.1: adopt the legacy root data dir (agents/default -> agents/orchestrator)
    # BEFORE the EventBus below materializes agents/<root>/bus-events.jsonl. The store's
    # own open()-time adoption (plan 01) refuses once agents/orchestrator/ exists, and the
    # bus's persist_path.parent.mkdir would create it first — stranding the old memories in
    # agents/default/. Doing it here preserves the single-adoption contract (open()'s call
    # then no-ops on the existing dir, while its SQL row re-key still runs).
    # `state_dir`, not `cfg_path`: this migrates the DATA tree, so it must operate on the same tree
    # the MemoryStore below opens. Do NOT confuse it with `_migrate_legacy_root_agent_yaml` (local to
    # this module), which renames a CONFIG file and is GLOBAL-ONLY by 40-02's ruling.
    _migrate_legacy_root_agent_dir(state_dir, agent_name_str)
    # One edit, five artifacts: bus-events.jsonl (below), `sessions/` (derived inside
    # EventBus.__init__ as persist_path.parent / "sessions"), memory.log (_route_memory_logs_to_file),
    # and compact.md (~line 1060). All per-agent WORK, so all follow the workspace. Anyone tempted to
    # "tidy" this back to cfg_path would silently move all five back to the global dir.
    agent_dir = state_dir / "agents" / agent_name_str
    events_path = agent_dir / "bus-events.jsonl"
    bus = EventBus(persist_path=events_path)
    # #20: keep background consolidation/mining logs off the interactive terminal (file only)
    # — otherwise their stdlib WARNING/EXCEPTION records leak onto the REPL via lastResort.
    _route_memory_logs_to_file(agent_dir)
    sitting_id = str(uuid.uuid4())  # SESS-01: one session per SITTING, minted once
    if resume is not None:  # a rebuild is a new sitting: memory's sessions.id is a primary key
        log.info("session %s resumes %s after %s %s", sitting_id, resume.previous_sitting_id,
                 *resume.action)
    asleep_file = asleep_path(events_path.parent / "sessions")
    # LLMClient built above with probe-derived tool_call_mode.

    # --- 2. Core infrastructure ---
    tool_registry = ToolRegistry()
    # git_read is rooted at the SAME boundary the gate uses below (derived from where you stand;
    # config may only narrow it - spec 17 section 7.3), so it is computed once, here, and shared.
    from localharness.agent.gate import derive_session_boundary
    from localharness.agent.verdict import narrow_boundary

    _derived = derive_session_boundary(cwd=Path.cwd(), local_dir=workspace)
    _boundary, _boundary_warning = narrow_boundary(
        _derived, getattr(agent_config.permissions, "workspace_root", None)
    )
    # workspace_root (issue #15): opt-in; None (default) leaves write/edit/bash unconfined.
    await register_builtin_tools(
        tool_registry,
        workspace_root=agent_config.permissions.workspace_root,
        boundary=_boundary,
    )

    # --- 3. Hook system (soft) ---
    hook_system: HookSystem | None = None
    try:
        hook_system = HookSystem()
        hook_system.wire_to_registry(tool_registry)
    except Exception as exc:
        warnings.append(f"hooks: {exc}")
        hook_system = None

    # --- Resource-owning window (#43) ---
    # Everything constructed from here on must be torn down by the finally below. A hard failure in
    # this window (e.g. the TokenCounter fail-loud) otherwise skips cleanup — and a plugin left
    # running (memory's aiosqlite NON-DAEMON worker thread) hangs interpreter shutdown forever.
    # Pre-bind every component the finally inspects so an early failure can't UnboundLocalError
    # past the close.
    mcp_manager = None
    plugin_resolution = None
    plugin_result = None
    session_info = None
    _exit_reason = "complete"
    restart_request: Restart | None = None
    disk_resume: Resume | None = None  # a thread taken off disk, until a session is bound
    bound = False
    try:
        if mobile_channel is not None and resume is None:
            # A thread asleep in this agent's folder wakes here, through the handle a /plugins
            # restart passes in memory. Taken inside the resource window: a wake that fails before
            # a session is bound puts it back (the finally), so the thread is not lost to a model
            # server that happened to be down.
            if fresh_thread:
                asleep_file.unlink(missing_ok=True)
            else:
                disk_resume = resume = take_asleep(asleep_file, workspace=os.getcwd())
            if resume is not None:
                log.info("session %s wakes %s from %s", sitting_id, resume.previous_sitting_id, asleep_file)
                mobile_channel.set_bringup("starting the session", detail=WAKING_DETAIL)
        # Queryable-handle tool: tool_result_get (restore evicted tool-result bodies). The
        # ContentStore is shared with the ContextManager below so eviction-writes and restore-reads
        # hit the same map. The memory tools are the memory plugin's (step 5).
        from localharness.agent.context import ContentStore
        # A rebuild keeps the store: the carried conversation's stubs name handles it holds.
        eviction_store = (resume.eviction_store if resume is not None else None) or ContentStore()
        try:
            if agent_config.context.tool_result_eviction:
                from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
                await tool_registry.register(ToolResultGetTool(eviction_store), scope="global")
        except Exception as exc:
            warnings.append(f"queryable-handle tools: {exc}")

        # Bind the root agent's store-backed verb tools (web_fetch / web_page_query / tool_result_get)
        # to the root ContentStore, so the root has ONE per-agent store (web pages + evicted bodies) and
        # children — which rebind to their OWN store in dispatch — are isolated from it.
        from localharness.tools.builtin import bind_agent_store_tools
        bind_agent_store_tools(tool_registry, eviction_store)

        # --- 5. Plugins: build the plan, run the lifecycle (soft: never stops the harness) ---
        # The plan is pure (BUILTIN_PLUGINS + discovered METADATA + this config); only ENABLED
        # third-party plugins are imported. Each stage is contained per plugin (PAPI-11): a failure
        # is named in the summary line's warnings and the session goes on. Plugins, memory included
        # (it opens its store, session row, accumulator and dreaming in its start() — each soft piece
        # fails alone; a store that will not open leaves the slot empty).
        # #150 phase 38 criterion 3: plugins follow the SESSION's config dir — resolve() discovers
        # from `loader.global_config_dir` (cfg_path, never the workspace), after the same loader
        # loaded this agent (its `agent.<name>` plugin sections are keyed on the declared name).
        channel_refusal = None
        plugin_error = None
        try:
            plugin_resolution = resolve(loader, agent_name=agent_config.name)
            warnings.extend(plugin_resolution.warnings)
            warnings.extend(_start_problems(plugin_resolution))
            # Tier 2: computed here, RAISED below the except — inside it the refusal would be
            # swallowed as a plugin warning and the session would fall through to the terminal.
            channel_refusal = _channel_refusal(channel_mode, plugin_resolution)
            if channel_refusal is None:
                session_info = SessionInfo(
                    agent_id=agent_name_str, division_id=agent_config.division or "default",
                    sitting_id=sitting_id, model=resolved_model, context_tokens=_cfg_window,
                    budget=agent_config.permissions.budget.model_dump(),
                )
                plugin_result = await start_plugins(
                    plugin_resolution, bus=bus, registry=tool_registry, hooks=hook_system, llm=llm,
                    paths=PluginPaths(global_config_dir=cfg_path, workspace=workspace,
                                      state_dir=state_dir),
                    session=session_info,
                )
                warnings.extend(plugin_result.warnings)
                warnings.extend(set_plugin_rows(plugin_result.slash_rows))
                plugins_loaded = len(plugin_result.running)
        except Exception as exc:  # noqa: BLE001 — the substrate itself failing is still not fatal
            plugin_error = f"plugins: {exc}"
            warnings.append(plugin_error)
        if channel_refusal is not None:
            # Nothing has started (start_plugins was skipped, MCP is step 6); the finally below
            # still closes the LLM client.
            raise typer.BadParameter(channel_refusal, param_hint="--channel")

        # One slot object reaches the loop AND the channel; the memory plugin occupies it when it
        # started, and it is empty with memory off or when memory failed to start.
        memory_slot = plugin_result.slot if plugin_result is not None else MemorySlot()

        # --- Capability floor (P-A) for the ROOT agent, read off DECLARATIONS (SAFE-02) ---
        # Strips every GLOBAL tool that declares — or, undeclared, defaults to — ingest: untrusted:
        # the web verbs today, any plugin tool that ingests, any tool that declares nothing. It runs
        # here, after every global tool is registered, because a declaration can only be read off a
        # registered tool. MCP tools (step 6) keep today's rule: checked by the chokepoint, not stripped.
        from localharness.tools.capabilities import apply_root_capability_floor
        _global = {s.name: s for s in tool_registry.global_schemas()}
        for _name in apply_root_capability_floor(agent_config.tools, _global.values()):
            # A plugin tool the root cannot hold is named, not silently missing. Core's web verbs
            # are stripped the same way by design (the root delegates ingestion) and say nothing.
            if (_plugin := _global[_name].source_plugin) is not None:
                _why = ("it declares ingest: untrusted" if "ingest" in _global[_name].model_fields_set
                        else "it declares no ingest, which counts as ingest: untrusted")
                warnings.append(f"capability floor: the root agent does not hold {_name} (plugin "
                                f"{_plugin}) — {_why}; delegate to an agent without host tools")

        if loader.project_mcp_skipped:
            err_console.print(MCP_NOT_STARTED_LINE.format(
                files=", ".join(loader.project_mcp_skipped), why=project_trust.why),
                markup=False, soft_wrap=True)
        # --- 6. MCP client manager (soft) ---
        mcp_manager: Any = None  # an MCPClientManager when a server is configured
        try:
            mcp_configs = agent_config.tools.mcp_servers
            if mcp_configs:
                # Imported only for a configured server: the MCP client pulls mcp, starlette and
                # uvicorn, which a stock start never needs.
                from localharness.tools.mcp import MCPClientManager
                mcp_manager = MCPClientManager(tool_registry)
                results = await mcp_manager.startup(mcp_configs)
                mcp_connected = sum(1 for v in results.values() if v > 0)
                mcp_failed = sum(1 for v in results.values() if v == 0)
        except Exception as exc:
            warnings.append(f"mcp: {exc}")

        # --- 6b. Model-aware token counter (one instance, injected everywhere) ---
        # Counts via the served model's exact tokenizer so budget gates fire at the real fraction.
        # Provider-aware (#8): vLLM ({model,prompt}->{count}) and llama.cpp ({content}->{tokens})
        # both count EXACTLY; a KNOWN exact runtime whose /tokenize is unreachable hard-fails here
        # (doctor's check 5c reports the same) rather than silently mis-metering — an approximate
        # meter is what hid the context overflows (400s). Ollama / LM Studio serve no tokenize
        # endpoint, so they count EXACTLY from the served model's own GGUF vocab (mode exact_local),
        # falling back to EXPLICIT approximate mode (cl100k x safety factor, or a byte estimate when
        # the vocabulary cannot be had) — surfaced by the warning below — only when no local GGUF
        # is reachable.
        try:
            token_counter = TokenCounter(
                base_url=provider.base_url,
                model=resolved_model,
                provider_type=provider.provider_type,
            )
        except RuntimeError as exc:
            # #44 defense-in-depth: the probe above already hard-fails an unreachable/unserved model,
            # so reaching here means the model IS served but its /tokenize is unavailable — the
            # tokenizer message is now accurate. Still point at doctor for the rare
            # case where /tokenize is a separate seam from the chat endpoint the probe reached.
            err_console.print(
                f"[bold red]Error:[/bold red] {exc}\n"
                f"Ensure the model server at {provider.base_url} exposes an exact tokenizer "
                f"(vLLM or llama.cpp /tokenize), then retry.\n"
                f"Run `localharness doctor` to diagnose."
            )
            raise typer.Exit(1)
        if token_counter.approximate:
            from localharness.agent.context import APPROX_TOKENIZE_SAFETY_FACTOR
            console.print(
                f"[yellow]⚠[/yellow]  {provider.provider_type} serves no tokenize endpoint and no "
                f"local GGUF was found for exact counting — token accounting uses a conservative "
                f"estimate (cl100k × {APPROX_TOKENIZE_SAFETY_FACTOR}): budget gates fire early rather "
                f"than overflow. Install the 'exact-tokenizer' extra + co-locate the model files (or "
                f"use vLLM/llama.cpp) for exact counts."
            )

        # --- 7. Compaction pipeline (soft) ---
        pipeline: CompactionPipeline | None = None
        # THE real wiring for ContextConfig.compaction_threshold_pct (#132-class fix): this field
        # used to be parsed but never consulted — both TokenBudget.needs_summary_compact and
        # SummaryCompactionStage hardcoded 0.80 directly. Every direct construction elsewhere
        # (tests, subagents) still defaults to 0.80; this is the one production call site that
        # threads the configured value through.
        _compaction_trigger_fraction = agent_config.context.compaction_threshold_pct / 100.0
        try:
            compact_md_path = agent_dir / "compact.md"

            from localharness.agent.context import make_compaction_summarize_fn, summarizer_input_chars
            pipeline = CompactionPipeline(
                token_counter=token_counter,
                # Read per apply from the step-8 manager (the closure never runs if this
                # construction fails): the cap follows the manager's live window.
                tool_result_cap=lambda: ctx_mgr.tool_output_cap_chars(),
                preserve_first_n=agent_config.context.preserve_first_n_messages,
                preserve_last_n=agent_config.context.preserve_last_n_messages,
                llm_summarize_fn=make_compaction_summarize_fn(  # shared, tuple-unpack tested
                    llm, summarizer_input_chars(agent_config.context.max_context_tokens),
                ),
                compact_md_path=compact_md_path,
                trigger_usage_fraction=_compaction_trigger_fraction,
            )
        except Exception as exc:
            warnings.append(f"compaction: {exc}")
            pipeline = None

        # --- 8. Context manager (with pipeline) ---
        ctx_mgr = ContextManager(
            max_context_tokens=agent_config.context.max_context_tokens,
            preserve_first_n=agent_config.context.preserve_first_n_messages,
            preserve_last_n=agent_config.context.preserve_last_n_messages,
            # Without bus= the CompactionTriggered publishes in ContextManager are dead code and
            # interactive compactions are invisible (found forensically: a real compaction fired
            # with zero telemetry). The sitting id names the session file those events go to (an
            # empty id would write sessions/.jsonl); the loop re-asserts it at the start of every turn.
            bus=bus,
            agent_id=agent_config.name,
            session_id=sitting_id,
            pipeline=pipeline,
            eviction_store=eviction_store,
            # An evicted body's origin is what its tool DECLARES (result_origin, SAFE-03).
            result_origin=tool_registry.result_origin,
            # A failed call a later identical call fixed may leave the request only for a tool
            # that declares host: safe; the manager fails closed on anything else.
            host=tool_registry.host,
            # The configured sizes and ratios, raw (an unset size stays None): the manager
            # resolves them from its live window on every use.
            **context_size_kwargs(agent_config.context),
            tool_evict_enabled=agent_config.context.tool_result_eviction,
            token_counter=token_counter,
            compaction_trigger_fraction=_compaction_trigger_fraction,
            max_response_tokens=_output_cap,
        )
        # The registry was built in step 2, before the manager existed; nothing dispatches a model
        # tool call before this line. From here its result cap follows the manager's live window.
        tool_registry.set_result_size_cap(ctx_mgr.result_cap_chars)

        # --- 9. Orchestrator layer ---
        from localharness.orchestrator.router import AgentCreationFlow
        from localharness.orchestrator.cards import AgentCardRegistry
        card_registry = AgentCardRegistry()
        for agent_data in agents:
            try:
                a_name = agent_data.get("name", "")
                a_cfg = loader.load_agent(a_name)
                card_registry.register_from_config(a_cfg)
            except Exception:
                pass  # skip agents that fail to load — non-fatal
        orchestrator = AgentCreationFlow(card_registry=card_registry)

        # --- 9a. Project instructions (spec 17; D-48, D-67, D-69) ---
        # One handle per session on the effective boundary, built above the delegation block so the
        # runner, the loop, the REPL and ACP all hold the same object. Nothing is built for a
        # declined root or a root with no instructions file.
        from localharness.config import trust as _instr_trust

        instructions_handle = _open_project_instructions(
            _boundary,
            workspace=workspace,
            decision=_instr_trust.is_trusted_tree(_boundary) if _boundary is not None else None,
            trust_flag=trust_flag,
            sitting_id=sitting_id,
            max_ratio=harness.instructions.max_ratio,
            no_input=no_input,
            channel_mode=channel_mode,
            window_tokens=ctx_mgr.max_context_tokens,
            config_dir=config_dir,
        )

        # --- 9a'. Narrowing notices (D-16, D-55, D-56, D-83, D-89) ---
        # What the project's files asked for and the loader refused, plus the oversized-file line,
        # as one set. `unseen()` runs once, after the roster load and the handle; the set is stored
        # at the summary only when a person is at a terminal (`can_ask`), so a narrowing prints at
        # the first such start, again when the rejected value changes, and on every start where
        # nobody records it. A session with no boundary prints the loader's texts every start.
        from localharness.config.project_records import replace_notices, unseen

        # The built-in subagents' workspace overlays are evaluated here, up front, so what they
        # reject joins the set printed at the summary; a dispatch re-reads the file and tells the
        # channel only about a rejection this set did not hold (D-56, D-57). A malformed overlay is
        # an explicit error at dispatch and is skipped here.
        from localharness.agent.subagent import builtin_base_config
        from localharness.config.project_records import DispatchNoticeSink

        _overlay_notices: list = []
        for _builtin_name in ("explore", "web-researcher", "search-verifier"):
            try:
                loader.overlay_builtin_config(
                    _builtin_name,
                    builtin_base_config(_builtin_name, harness.org.children_budget),
                    notices=_overlay_notices.append)
            except Exception:  # noqa: BLE001 - reported at dispatch, where the model sees it
                log.debug("overlay of %s skipped at start", _builtin_name, exc_info=True)

        _start_notices: list = list(dict.fromkeys([
            *loader.notices,
            *_overlay_notices,
            *([instructions_handle.too_large_notice]
              if instructions_handle is not None and instructions_handle.too_large_notice is not None
              else []),
        ]))
        _dispatch_sink = DispatchNoticeSink(_start_notices, global_dir=server_cfg_path)
        _unseen: list = []
        if _boundary is not None:
            try:
                _unseen = unseen(_boundary, _start_notices, config_dir=config_dir)
            except OSError:
                _unseen = list(_start_notices)  # an unreadable store: nothing is known to be seen

        # --- 9b. Agent delegation tool (ORCH-04 / SUBAGENT-05) ---
        # Bug#1 fix: the runner is built via the module-level make_explore_agent_runner seam (T1)
        # and the AgentTool is registered at GLOBAL scope. The parent loop's agent is the `default`
        # agent (agent_name_str), and get_tools_for_agent resolves agent-scoped tools by that name —
        # an agent_id="orchestrator" registration was NEVER offered to `default`, so delegation was
        # dead. Global scope matches the bench semantics (runner.py registers `agent` globally) and
        # makes the running parent actually see + use the tool. No global-name collision: the builtins
        # are read/glob/grep/write/bash_exec. parent_session_id is read at call time via the getter
        # (agent_loop is constructed just below, before any turn runs / the model can delegate).
        from localharness.tools.builtin.agent_tool import AgentTool
        from localharness.agent.subagent import make_explore_agent_runner

        perm_eval = PermissionEvaluator()

        # --- the human-approval gate (PRD §3.1, §3.5) ------------------------------------
        # The boundary is DERIVED from where you stand, never read from config: the workspace
        # layer v0.13 discovery applied, else the nearest checkout, else the cwd. Config's
        # `permissions.workspace_root` may only NARROW it; a value outside it is ignored with a
        # warning, because a repo that could move the boundary could approve its own writes.
        from localharness.agent.gate import (
            PermissionGate,
            deny_fn_from,
            settings_from,
        )
        from localharness.config.grants import GrantStore

        if _boundary_warning:
            console.print(f"[yellow]{_boundary_warning}[/yellow]")
        if _boundary is None:
            console.print(f"[dim]{NO_BOUNDARY_NOTICE}[/dim]")
        gate = PermissionGate(
            boundary=_boundary,
            workspace=Path.cwd(),
            grants=GrantStore(),
            mode=agent_config.permissions.mode,
            asker=None,  # attached to the channel below, once it exists
            channel_name="none",
            has_review_surface=False,
            # AgentLoop hands its OWN agent's deny tier in on every check (a subagent may have
            # tightened it); this is the fallback for anything else that consults the gate —
            # the ACP adapter, the trust dialog — so a caller can never skip the DENY tier.
            deny=deny_fn_from(perm_eval, agent_config.permissions),
            settings=settings_from(agent_config.permissions),
            bus=bus,
            remote_unattended=harness.channels.remote_unattended,
            trusted_for_run=trust_flag,
            config_dir=cfg_path,  # the session's own tools folder and settings files are guarded
        )
        if resume is not None and resume.gate_mode != gate.mode:
            from localharness.agent.gate_types import MODE_STRICTNESS

            if resume.slept and MODE_STRICTNESS.get(resume.gate_mode, -1) < MODE_STRICTNESS[gate.mode]:
                # A file may carry a mode but may not loosen one: the folder it sits in is one a
                # cloned repo can ship and the agent's own write tool can reach, and the gate's
                # rule for a layer that is not a human is "stricter only" (gate.set_mode).
                warnings.append(f"resume: the sleeping chat's mode {resume.gate_mode} is looser than "
                                f"{gate.mode}; kept {gate.mode}")
            else:
                try:
                    gate.set_mode(resume.gate_mode)  # /mode as the person left it (and a declined trust's mode)
                except Exception as exc:  # noqa: BLE001 — only a file can carry a mode the gate refuses
                    warnings.append(f"resume: kept mode {gate.mode} ({exc})")

        # Built-in subagents wired in the runner (subagent.make_explore_agent_runner) — advertise them
        # alongside any configured agent cards so the model knows it can delegate to them. search-verifier
        # is a standalone capability (route a user's "re-check X" straight to it); the web-researcher
        # nests it only when its task asks for verification (rigor=on-request default) or under
        # RESEARCH_RIGOR=high (auto-verify material claims).
        # v0.5.3: every default builtin is quarantined-or-read-only; bash-holding specialist roles
        # (data-analyst, frontend-designer) ship as opt-in examples/agents/ configs instead.
        available_agent_names = ["explore", "web-researcher", "cruncher",
                                 "search-verifier"] + [c.name for c in card_registry.all_cards()]

        async def _override_builtin(n: str, base: Any) -> Any:
            """The overlay of a built-in subagent at dispatch. A rejection the start set did not
            hold reaches the channel's status line once per sitting through the bus; it is awaited
            here, inside the dispatch, never scheduled and forgotten (D-57)."""
            cfg = loader.overlay_builtin_config(n, base, notices=_dispatch_sink.record)
            try:
                await _dispatch_sink.flush(bus, agent_id=agent_config.name, session_id=sitting_id)
            except Exception:  # noqa: BLE001 - a lost status line never fails a dispatch
                log.debug("narrowing note not published", exc_info=True)
            return cfg

        # D-21: ONE supplier for the root loop and every child; the REPL fills the cell below.
        repl_cell: list = []
        server_recovery_fn = _server_recovery_supplier(repl_cell)
        _run_agent = make_explore_agent_runner(
            llm=llm,
            # D-51: children get the instructions too unless the machine opted out
            # (`instructions.children`, machine-only); the loop keeps the handle either way. An
            # opted-out child still holds a mark-only view (W-1): no text, no notice, but its
            # write to the file is marked so the root never adopts it as the person's edit.
            project_instructions=_children_instructions(instructions_handle, harness.instructions.children),
            bus=bus,
            base_registry=tool_registry,
            permission_evaluator=perm_eval,
            # The SAME gate object the orchestrator holds (PRD §3.4): a /mode switch and a fresh
            # grant reach a running child, and a child's boundary crossing surfaces in the same
            # prompt as the parent's.
            gate=gate,
            get_parent_session_id=lambda: agent_loop.current_session_id,
            # bypass_cache: a yaml the model just WROTE must be dispatchable in the same turn
            load_agent=lambda n: loader.load_agent(n, bypass_cache=True),
            # Built-in subagents (explore/web-researcher/search-verifier) are TUNABLE: an optional
            # agents/<name>.yaml overlays the code-defined base (e.g. a bigger web-researcher
            # budget) — absent file = built-in defaults, malformed = explicit error, never silent.
            load_builtin_override=_override_builtin,
            # Children inherit the parent's exact /tokenize counter + resolved window so the
            # sub-agent fleet accounts for context correctly instead of bare 131,072 + tiktoken.
            token_counter=token_counter,
            max_context_tokens=agent_config.context.max_context_tokens,
            # Real, config-capped recursion depth: the orchestrator is depth 0; a non-leaf child
            # (e.g. web-researcher) may nest a grandchild (search-verifier) up to this cap.
            depth=0,
            max_subagent_depth=agent_config.max_subagent_depth,
            available_agents=available_agent_names,
            memory_handle=memory_slot.bind_subagent,  # bound per delegation (PAPI-04); the slot answers None when empty
            # Grant keystone: the root's ContentStore is the parent store children read through when the
            # model delegates with grant_handles (it IS `eviction_store` — start_cmd:315/406 — so an
            # evicted/handle id the model sees in a stub is exactly what it can grant to a cruncher).
            parent_store=eviction_store,
            # Cruncher exec policy (agent.cruncher.*): offered to a clean-origin cruncher iff exec_enabled.
            cruncher_config=agent_config.cruncher,
            # criterion 2: children resolve compact.md + kill-file under THIS session's dir, not
            # ~/.localharness or CWD. Two dirs since 41-02: the child's KILL file stays on
            # `config_dir` (a machine-global control artifact — one file must stop every agent on
            # the box), and only compact.md follows `state_dir` into the workspace.
            config_dir=cfg_path,
            state_dir=state_dir,
            # D-11: the callable, read at each dispatch — a /effort between two delegations reaches
            # the second child (and, through the nested runner, a grandchild).
            effort_fn=lambda: (agent_loop.effective_effort, agent_loop.effort_rung),
            effort_ratios=_effort_ratios,
            # D-34: the value the session loop below is built with (the loop does not exist yet).
            template_margin_ratio=harness.autosense.template_margin_ratio,
            server_recovery=server_recovery_fn,
            # D-09: the operator's per-child ceilings, a value (the block does not change mid-session).
            children_budget=harness.org.children_budget,
            # D-57: the machine knob reaches every child loop, so a session with no local operator
            # (any mode, WR-70) has a ceiling on each of them, not only on the root.
            unattended_max_actions=harness.channels.unattended_max_actions,
            # D-32: the root loop's live remainder, read at each dispatch (late-bound like effort_fn:
            # the loop is built after this runner).
            parent_remaining_fn=lambda: agent_loop.remaining_budget(),
        )

        agent_tool = AgentTool(
            agent_runner=_run_agent,
            available_agents=available_agent_names,
        )
        await tool_registry.register(agent_tool, scope="global")
        # The working task record (0.16.5): root only, loaded from this agent's folder; children
        # and bench never hold it. No record = no packet and no file.
        from localharness.agent.task_record import TaskState
        from localharness.tools.builtin.task_tool import TaskTool
        task_state, task_notice = TaskState.load(agent_dir / "task.json", workspace=os.getcwd())
        if task_notice:
            console.print(f"[yellow]{task_notice}[/yellow]")
        await tool_registry.register(TaskTool(task_state), scope="global")

        # --- 10. Agent loop ---
        # #35: resolve the kill-file value against THIS config dir (a bare default 'KILL' lands at
        # <cfg_path>/KILL — i.e. ~/.localharness/KILL in a default setup). None = kill switch off,
        # left for AgentLoop's own fallback. Passing it here keeps agent/loop.py config-dir-agnostic.
        _kill_value = getattr(getattr(agent_config.permissions, "budget", None), "kill_file", None)
        # `cfg_path` ON PURPOSE, two lines from a `state_dir` user: the kill switch is a
        # machine-global CONTROL artifact — one file stops every agent on the box — so it must not
        # move into a project folder where a per-workspace copy would be needed to halt each one.
        kill_file_path = resolve_runtime_path(_kill_value, cfg_path) if _kill_value else None
        agent_loop = AgentLoop(
            config=agent_config,
            llm=llm,
            bus=bus,
            context_manager=ctx_mgr,
            tool_registry=tool_registry,
            permission_evaluator=perm_eval,
            gate=gate,
            kill_file_path=kill_file_path,
            compact_md_path=compact_md_path,
            session_id=sitting_id,  # SESS-01: the whole sitting shares this id
            # SAFE-04: core reads the org guardrails every turn, memory on or off. `cfg_path` is the
            # always-global config dir — the same global-only primitive MemoryStore's
            # global_base_dir receives and the plugin context's paths.global_config_dir will carry;
            # "default" is the org id the store is built with. A workspace can neither rewrite nor
            # blank the org's safety voice.
            guardrails_path=cfg_path / "orgs" / "default" / "GUARDRAILS.md",
            # D-21: the same supplier the runner hands every child; late-bound to the REPL below (ACP
            # never builds one, so its loop shares the poll-only pair).
            server_recovery=server_recovery_fn,
            # BUDGET-02: the machine's cap for a session with no local operator, in any mode (WR-70);
            # the loop reads it live against the gate, so an interactive terminal root stays unbounded.
            unattended_max_actions=harness.channels.unattended_max_actions,
            project_instructions=instructions_handle,
            # PAPI-04: the memory slot — the lifecycle's; its occupant (the memory plugin) feeds the
            # prompt's memory section.
            memory_slot=memory_slot,
            task_context=task_state,
            template_margin_ratio=harness.autosense.template_margin_ratio,
            effort=(_effort_level, _effort_rung),
            effort_ratios=_effort_ratios,
        )
        if resume is not None:  # the next turn carries the earlier conversation, byte for byte
            agent_loop.resume(list(resume.conversation), resume.prior_context)
        if acp_channel is not None:
            # Built at the ACP handshake (it had to answer `initialize` before any of this
            # existed); the gate attaches to it below exactly like any other channel.
            channel = acp_channel
        elif mobile_channel is not None:
            # Built by `localharness mobile` before this call, for the same reason: the HTTP server
            # answers, serves history and holds the pending queue with the model server cold.
            # Unlike ACP it does NOT take a `serve()` branch below — the REPL drives it.
            channel = mobile_channel
        elif channel_mode in plugin_channel_names() - OWN_COMMAND:
            # Any bundled channel plugin, built without core naming it: the running plugin's own
            # make_channel (it hands the channel its validated settings), else its channels() class.
            owner_name = _channel_plugin(channel_mode)
            running = plugin_result.running if plugin_result is not None else []
            owner = next((r for r in running if r.name == owner_name), None)
            if owner is None:  # tier 3: it resolved on, it is not running; the finally stops the rest
                unset = plugin_result.unconfigured.get(owner_name) if plugin_result is not None else None
                reason = ((plugin_result.failed.get(owner_name) if plugin_result is not None else None)
                          or (f"unconfigured — set {unset}" if unset else None)
                          or plugin_error or "see the warnings above")
                raise typer.BadParameter(CHANNEL_PLUGIN_NOT_RUNNING.format(
                    name=channel_mode, plugin=owner_name, reason=reason), param_hint="--channel")
            make = getattr(owner.plugin, "make_channel", None)
            channel = make(channel_mode, bus) if make else owner.plugin.channels()[channel_mode](bus=bus, config={})
            if channel.start_banner:
                console.print(f"[dim]{channel.start_banner}[/dim]")
        else:
            # #35: REPL history resolves under this session's state dir (default
            # ~/.localharness/.repl_history). The transcript records what you typed while working in
            # THIS project, so it follows the work (ruled); the kill file above deliberately does not.
            # CR-52: `--no-input`, a piped stdin and a model-spawned start are no local operator,
            # so every mode caps them at channels.unattended_max_actions (WR-70).
            channel = TerminalChannel(
                bus=bus, config={}, history_file=str(resolve_runtime_path(".repl_history", state_dir)),
                no_input=no_input,
            )
        # PRD §3.5: the channel decides how an ASK is rendered and whether it can be rendered at
        # all. A channel that cannot ask leaves the gate with no asker, which is the fail-closed
        # path — every ASK denies with a reason the model can re-plan against, and the gate logs
        # the fix (`permissions.mode: unattended`) once.
        gate.attach_channel(channel)

        # --- WEBCH-29: is anybody else driving this agent right now? ---
        # One hook for every channel, because this function is the funnel all four come through.
        # It WARNS and never refuses (owner ruling): `history.jsonl` and `compact.md` take
        # unlocked appends, so two live sessions on one agent can interleave and a long append
        # can tear — a hazard that is latent today and reachable by habit, since running the
        # terminal while a phone is attached is a normal thing to do, not an exotic one.
        await _announce_presence(
            channel, config_dir=config_dir, agent=agent_config.name,
            channel_mode=channel_mode, session_id=sitting_id, workspace=state_dir,
        )

        # --- Determine returning user ---
        is_returning = events_path.exists() and events_path.stat().st_size > 0

        # --- Startup banner ---
        # #49: in an interactive TTY the first-run hint is fragile scrollback the prompt_toolkit
        # box repaints over, so relocate it INTO the first input bubble (show_hint=False here,
        # channel.first_prompt_hint below). Piped/non-interactive sessions keep the banner hint.
        interactive = console.is_terminal
        elapsed = _time.monotonic() - start_time
        if resume is None:
            from localharness.cli.ui import startup_banner
            console.print(startup_banner(
                model=resolved_model, is_returning=is_returning, show_hint=not interactive,
            ))
            if interactive and isinstance(channel, TerminalChannel):
                channel.first_prompt_hint = _first_prompt_hint(is_returning)
        elif resume.slept:
            console.print(Text(RESUMED_LINE.format(previous=resume.previous_sitting_id)), soft_wrap=True)
        else:  # R3: no wordmark mid-conversation — one line, then the plugin lines and the status
            console.print(Text(RESTARTED_LINE.format(
                name=resume.action[1], state="on" if resume.action[0] == "enable" else "off")),
                soft_wrap=True)
        if instructions_handle is not None and hasattr(channel, "instructions_on_delivered"):
            # D-47: delivery is the receipt. The channel calls this once its own checked send / hint
            # / attached client has happened; a channel that never calls it writes no record.
            channel.instructions_on_delivered = instructions_handle.on_delivered
            if (can_ask and isinstance(channel, TerminalChannel)
                    and instructions_handle.start_line):
                # D-06 / D-64: only a person at a tty (no --no-input) can read the line; piped stdin
                # sets no hint, so no receipt reaches the handle and no record is written.
                channel.instructions_hint = instructions_handle.start_line
                channel.first_prompt_hint = _hint_with_instructions(
                    channel.first_prompt_hint, instructions_handle.start_line)
        if hasattr(channel, "tps_source"):
            # The instrument cluster. Colored tok/s readout (status row / thinking label): poll
            # the client's live decode-speed snapshot. The client object survives /model rebinds,
            # so this stays valid across swaps.
            #
            # NONE of this is on the bus — `stream_snapshot` and friends are direct callables —
            # so a channel that wants a status row gets it by being wired here, exactly as the
            # terminal is. Keyed on the attribute rather than on `isinstance(TerminalChannel)`
            # because the mobile channel needs the identical three sources to build its `StatusTick`
            # frame, and a widening isinstance chain here is a second list to keep in step.
            channel.tps_source = llm.gen_speed_snapshot
            channel.progress_source = llm.stream_snapshot  # phase + live token tallies
            channel.model_source = lambda: llm.config.model  # footer model chip, swap-safe

        # --- Startup summary line ---
        # Entity-typed per the localharness.dev architecture plates (cli/theme.py): the
        # agent count wears the runtime hue, MCP servers and plugins the tool hue. Counts
        # used to be one undifferentiated line, so "what did the harness actually bring
        # up?" needed reading rather than glancing.
        parts = [f"[dim]({elapsed:.1f}s startup)[/dim]"]
        counts: list[str] = [entity("agent", "1 agent")]
        if mcp_connected > 0 or mcp_failed > 0:
            mcp_str = f"{mcp_connected} MCP server{'s' if mcp_connected != 1 else ''}"
            if mcp_failed > 0:
                mcp_str += f" ({mcp_failed} failed)"
            counts.append(entity("tool", mcp_str))
        if plugins_loaded > 0:
            counts.append(entity("tool", f"{plugins_loaded} plugin{'s' if plugins_loaded != 1 else ''}"))
        if (instructions_handle is not None and instructions_handle.summary_entity
                and instructions_handle.too_large_notice not in _unseen):
            # D-54: the file in force (or why not) is named in the counts the person already reads;
            # no new line. "config" is the closest existing entity kind. The start that prints the
            # oversized line through the notice set shows the line instead (D-53).
            counts.append(entity("config", instructions_handle.summary_entity))
        if _boundary is not None:
            # D-16, D-83: a notice's text leaves the slot where the loader put it, and comes back at
            # the end only when it is unseen, so nothing prints twice and a seen one is silent.
            _notice_texts = {n.text for n in _start_notices}
            warnings[:] = [w for w in warnings if w not in _notice_texts]
            warnings.extend(dict.fromkeys(n.text for n in _unseen if n.text not in warnings))
            if can_ask:
                try:
                    replace_notices(_boundary, _start_notices, config_dir=config_dir)
                except OSError:
                    pass  # an unwritable store: the notices simply print again next start
        summary_line = " -- ".join(parts + [", ".join(counts)])
        if warnings:
            # escape() via entity(): a warning is an exception string the user's environment
            # wrote (paths, `[Errno 2]`), and it was printed raw INSIDE literal brackets.
            # Rich read the whole `[memory: ... (in-memory mode)]` group as one tag and
            # DELETED it, so every degraded startup — memory fallen back to in-memory, a
            # hook that failed to load, a crashed session-start — printed a clean-looking
            # summary and reported nothing. Amber is the site's warning tone.
            summary_line += " " + entity("warning", f"[{'; '.join(warnings)}]")
        if resume is None:
            console.print(summary_line, soft_wrap=True)
        elif warnings:  # a rebuild prints its warnings alone, one amber line
            console.print("  " + entity("warning", f"[{'; '.join(warnings)}]"), soft_wrap=True)
        if plugin_result is not None and plugin_result.running:
            console.print("  " + entity("tool", "Plugins: " + ", ".join(plugin_result.loaded_names)),
                          soft_wrap=True)
        if plugin_resolution is not None:
            available = [e.name for e in plugin_resolution.plan.entries if e.state == "available"]
            if available:  # Text: a plugin name is outside text, never markup (39-04's lesson)
                console.print(Text(_available_hint(available)), soft_wrap=True)
        if resume is not None and not resume.slept:
            console.print(Text(_resume_status(resume, plugin_result)), soft_wrap=True)

        # --- Verbose output ---
        if verbose:
            # entity() escapes each value as it colors it, so a server or plugin NAMED with
            # brackets survives here too — these lines interpolated config-supplied names
            # into markup unescaped before.
            if mcp_manager and mcp_manager.connected_servers:
                for srv in mcp_manager.connected_servers:
                    console.print("  " + entity("tool", f"MCP: {srv}"), soft_wrap=True)
            tool_count = len(tool_registry._tools["global"]) + len(tool_registry._tools["mcp"])
            console.print("  " + entity("tool", f"Tools: {tool_count} total"))
            if memory_slot.occupied:
                # escape(): the agent's state dir is derived from the config dir, which is a path
                # the user chose. Unescaped it crashed the banner AFTER the session was wired.
                console.print("  " + entity("memory", f"Memory: {agent_dir / 'memory.db'} (WAL)"), soft_wrap=True)
            else:
                console.print("  " + entity("memory", "Memory: in-memory (no persistence)"))

        # --- Run REPL ---
        from localharness.cli.repl import OrchestratorREPL

        def _register_deployed_agent(name: str) -> None:
            """#58: make a just-deployed agent reachable in THIS session — no restart. Load its
            freshly-written yaml (bypass_cache, exactly like the dispatch path at :815), register
            its card so /agents lists it, and append it to `available_agent_names` — which IS the
            list `agent_tool` advertises (AgentTool stores the same list object), so the model sees
            it as a delegation target. Delegation itself already resolved fresh yamls by name via
            the bypass_cache loader; this closes the visibility/advertisement gap so same-session
            delegation genuinely works."""
            try:
                a_cfg = loader.load_agent(name, bypass_cache=True)
            except Exception as exc:  # noqa: BLE001 — never turn a successful deploy into a failure
                log.warning("post-deploy live registration failed for %s: %s", name, exc)
                return
            card_registry.register_from_config(a_cfg)
            if name not in available_agent_names:
                available_agent_names.append(name)

        async def _plugin_switch(verb: str, name: str) -> str | None:
            """/plugins for this session: a fresh plan over the session's own layers. A plan that
            cannot be read answers one line, and the session goes on."""
            from localharness.cli.plugins_cmd import plugins_overview, switch_decision
            running = frozenset(plugin_result.loaded_names) if plugin_result is not None else frozenset()
            try:
                if verb == "list":
                    return plugins_overview(running, cfg_path, workspace)
                return await switch_decision(name, verb == "enable", running, cfg_path, workspace)
            except Exception as exc:  # noqa: BLE001 — a config broken mid-session never ends it
                log.warning("/plugins could not read the plugin settings", exc_info=True)
                return (f"/plugins could not read the plugin settings ({type(exc).__name__}) — "
                        "`localharness plugins list` in a shell says why.")

        if getattr(channel, "has_display_toggles", False):
            # Reasoning stream: the sink is always wired (it no-ops while the flag is off) so
            # /reasoning can toggle it mid-session; the flag comes from --show-reasoning or
            # terminal.show_reasoning in config.yaml.
            #
            # `llm.on_reasoning` was gated to `TerminalChannel` here, and widening THIS ONE LINE
            # is the whole of what live reasoning on the phone required — the callback already
            # carried the raw text, and no provider change was needed. Keyed on the capability
            # flag so the next channel that can render reasoning declares itself rather than
            # being added to a list in `cli`.
            term_cfg = getattr(harness, "terminal", None)
            channel.show_reasoning = show_reasoning or bool(
                getattr(term_cfg, "show_reasoning", False)
            )
            # --verbose is the whole-session detailed view: startup detail above, and in the
            # turn output every tool call itemized with its arguments plus the reasoning
            # stream (the default view groups read/memory/web calls); /verbose toggles it live.
            channel.verbose = verbose
            if llm is not None:
                llm.on_reasoning = channel.on_reasoning

        if mobile_channel is not None:
            # Everything the HTTP surface has to answer a STATE question about — `/api/tools`,
            # `/api/permissions`, `/api/health`, and the session log `?from={seq}` replays from.
            # One explicit handoff rather than the channel reconstructing state from the event
            # stream it also forwards, which would give it two sources of truth for the same fact.
            mobile_channel.bind_runtime(
                session_id=sitting_id,
                agent_id=agent_config.name,
                gate=gate,
                tool_registry=tool_registry,
                llm=llm,
                agent_loop=agent_loop,
                session_dir=events_path.parent / "sessions",
                # THE bus this session publishes on. The channel was constructed on a
                # placeholder bus before any session existed; without this hand-over its
                # event subscriptions point at a bus nothing writes to, and live events
                # never reach a phone (2026-09-15 — the foundational webchat bug).
                bus=bus,
                # The memory page reaches memory only through the slot's browse API (WEBP-03) —
                # the SESSION's store behind it, never a second connection.
                memory_slot=memory_slot,
                # PAPI-10: the only roots /api/artifacts/{plugin}/{id} serves — the ones core
                # computed for this session's running plugins and accepted back from them.
                artifact_roots=plugin_result.artifact_roots if plugin_result is not None else {},
            )
            bound = True  # from here a thread taken off disk has become this session


        if acp_channel is not None:
            # The workspace-trust question, on the path that has no REPL to host it (owner
            # ruling 2026-09-11). The ACP session is already established — the client answered
            # `initialize` and `session/new` at the handshake — so the question renders as a
            # permission dialog, which is where every other one renders in Zed.
            from localharness.cli.session_trust import establish_session_trust

            try:
                await establish_session_trust(
                    gate, lambda text: console.print(f"[dim]{text}[/dim]"),
                    instructions=instructions_handle,
                )
            except Exception:  # noqa: BLE001 — a broken trust store costs a notice, not the session
                log.warning("could not settle workspace trust", exc_info=True)
            # Co-author consent (spec 14): settle once per project + install the hook.
            from localharness.cli.coauthor import settle_coauthor_startup
            try:
                await settle_coauthor_startup(
                    gate, workspace, lambda text: console.print(f"[dim]{text}[/dim]")
                )
            except Exception:  # noqa: BLE001
                log.warning("could not settle co-author consent", exc_info=True)
            # ACP's turn loop is the protocol itself: `session/prompt` requests run turns
            # directly on this loop, so there is no line to pull and no REPL. `serve()` returns
            # when the connection closes, and the ordered teardown in `finally` runs unchanged.
            await acp_channel.serve(
                bus=bus, agent_loop=agent_loop, gate=gate, tool_registry=tool_registry,
                instructions=instructions_handle,
            )
            return

        if launch_notes.printed:
            pending = []  # start's own launch said it all; the REPL has nothing to add
        elif (conversion := _conversion_launch_line(harness, launch_key, sensed.converted)):
            pending.insert(0, conversion)  # D-17: the conversion's line leads the first launch
        repl = OrchestratorREPL(
            orchestrator=orchestrator,
            agent_loop=agent_loop,
            channel=channel,
            bus=bus,
            config_dir=cfg_path,
            # 41-04: the raw Optional[Path], NOT state_dir. The REPL's _audit_base_dir does its own
            # `or self._config_dir` fallback, so a None here IS the signal "no workspace applies".
            workspace=workspace,
            harness_config=harness,
            on_agent_deployed=_register_deployed_agent,
            gate=gate,  # what /mode switches; the same object the loop and subagents hold
            instructions=instructions_handle,  # /instructions and the session-trust pass-through
            # /mcp: the live MCPClientManager so list-status / hot-add / hot-remove reach it.
            # None when no MCP server is configured (the command degrades to config-only).
            mcp_manager=mcp_manager,
            # /plugins enable|disable restarts a terminal session (the channel's own
            # can_switch_plugins decides); the web and ACP paths never get the hook.
            on_plugin_switch=_plugin_switch if mobile_channel is None and acp_channel is None else None,
            queued=resume.queued if resume is not None else (),
            launch_lines=tuple(pending),
        )

        repl_cell.append(repl)  # late binding: the supplier now answers repl.server_recovery()
        await repl.run()
        request = repl.restart_request or (
            SLEEP_ACTION if mobile_channel is not None and mobile_channel.sleeping else None)
        if request is not None:
            # The REPL ended for /plugins (hand start_app the plain-data handle) or for a sleep
            # (the same handle goes to disk, before a teardown that could hang); either way the
            # teardown below runs in its usual order and this sitting's memory row closes
            # normally ("complete").
            conversation, prior = agent_loop.resume_state()
            restart_request = Restart(request, Resume(
                action=request, agent_name=agent_name_str, conversation=tuple(conversation),
                prior_context=prior, eviction_store=eviction_store, queued=repl.queued,
                gate_mode=gate.mode, previous_sitting_id=sitting_id))
            if request == SLEEP_ACTION:
                write_asleep(asleep_file, restart_request.resume, workspace=os.getcwd())
                log.info("session %s asleep at %s", sitting_id, asleep_file)
    except KeyboardInterrupt:
        _exit_reason = "interrupt"
        console.print("\nGoodbye.")
    except Exception:
        _exit_reason = "error"
        raise  # finally still records the session; behavior for callers unchanged
    finally:
        if disk_resume is not None and not bound:
            # The wake failed (or was abandoned) before a session existed: the thread is still
            # asleep, and the next message gets to try again.
            try:
                write_asleep(asleep_file, disk_resume, workspace=os.getcwd())
            except Exception:  # noqa: BLE001 — never let the put-back replace the real failure
                log.warning("could not put the sleeping thread back at %s", asleep_file, exc_info=True)
        # --- Ordered shutdown: MCP -> plugins (reverse start order; memory's stop() runs dreaming ->
        # accumulator -> end_session -> router -> store) -> LLMClient LAST (#154) --- (EventBus
        # handles its own file closing on GC/process exit)
        if mcp_manager:
            try:
                await mcp_manager.shutdown()
            except Exception as exc:
                # Swallow — a teardown failure must not replace the real exit reason — but never
                # silently: an MCP server that refuses to die is a diagnosable thing.
                log.debug("MCP shutdown failed: %s", exc)
        if session_info is not None:
            session_info.exit_reason = _exit_reason  # memory's stop() records it on the session row
        if plugin_result is not None:
            try:
                await stop_plugins(plugin_result)  # reverse start order, each stop contained
            except Exception as exc:  # noqa: BLE001
                log.debug("plugin shutdown failed: %s", exc)
            set_plugin_rows(())  # the rows are process-wide: they leave with the session
        # LAST (#154): the LLM client's httpx pool outlives every consumer that could still issue
        # a request through it — the REPL and its agents (repl.run() has returned), consolidation,
        # the gates — and the close-out summary above is pure. Closing it earlier would risk
        # pulling the pool out from under an in-flight shutdown write; closing it never leaked a
        # pool per session (and one more per /model swap).
        await llm.aclose()
    return restart_request


_CHANNEL_HELP = ("Input channel: " + ", ".join(sorted(channel_names())) + " (default terminal). An "
                 "unknown name is refused, not silently treated as terminal. mobile and acp are served "
                 "by their own commands: `localharness mobile`, `localharness acp`.")


def start_app(
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Start specific agent")] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Detailed view: per-component startup detail, the model's reasoning stream, and "
             "one line per tool call with its arguments (the default groups read/memory/web "
             "calls into one counter line each); /verbose toggles it live",
    )] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    config_dir: Annotated[
        str | None,
        typer.Option(
            "--config-dir",
            envvar="LOCALHARNESS_DIR",
            show_default=False,
            help="Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
        ),
    ] = None,
    channel: Annotated[str, typer.Option(
        "--channel", "-c",
        help=_CHANNEL_HELP,
    )] = "terminal",
    subagents: Annotated[bool, typer.Option("--subagents", help="Show the agent picker on startup when multiple agents are configured")] = False,
    model: Annotated[str | None, typer.Option("--model", "-m", help="Use this model for THIS session only (never persisted). Must already be served — a harness-managed single-model server (llama.cpp/vLLM) cannot be hot-switched this way; use `localharness model <name>` or the REPL `/model` command instead.")] = None,
    list_models: Annotated[bool, typer.Option("--list-models", help="List models available at the configured provider, then exit")] = False,
    no_input: Annotated[
        bool,
        # The same flag doctor/validate/agent create carry, described by the same string (F6): a
        # run with nobody watching must not be asked to trust an outside workspace, and must not
        # be offered a new one either. Both questions this command can ask write something
        # permanent — a trust record, or a directory.
        typer.Option("--no-input", help=NO_INPUT_HELP),
    ] = False,
    show_reasoning: Annotated[bool, typer.Option(
        "--show-reasoning",
        help="Stream the model's reasoning (thinking) into the terminal as it generates "
             "(same as terminal.show_reasoning: true; /reasoning toggles it live)",
    )] = False,
    trust_project: Annotated[bool, typer.Option(
        "--trust-project",
        help="Trust the project you are in for this run only, recording nothing: its agent files' "
             "MCP servers start without the trust question and the session runs as trusted. For CI "
             "and scripts; LOCALHARNESS_TRUST_PROJECT=1 does the same for any command.",
    )] = False,
) -> None:
    """Launch the agent REPL. Zero to chatting in one command."""
    from localharness.config.paths import resolve_config_dir
    if not list_models and not (resolve_config_dir(config_dir) / "config.yaml").exists():
        # First start: no config. On a terminal, run init's core setup here and go on into the
        # session — no "run localharness init" bounce (owner ruling 2026-10-03). Without a person
        # to ask, _start_async's own welcome hint and exit 0 stand (the web/ACP callers use it too).
        from localharness.cli.init_cmd import core_setup, is_interactive  # only a first start pays
        if is_interactive(no_input):
            _refuse_unbuildable_channel(channel)  # a typo is refused before the setup, not after it
            console.print(FIRST_START_LEAD)
            if not core_setup(config_dir, endpoint=None, model=None, force=False,
                              interactive=True, starting=True).server_ready:
                return  # skipped or nothing saved: the setup printed what to do next
            # The setup probed the server in event loops of its own, now closed. A client it left
            # open is finalized by scheduling its close on whatever loop is running when the GC
            # reaches it — inside the session's, that raises "Event loop is closed". Collect now.
            import gc
            gc.collect()
    resume: Resume | None = None
    try:
        while True:
            restart = asyncio.run(_start_async(
                resume.agent_name if resume is not None else agent, verbose, debug, config_dir,
                channel, subagents and resume is None, model, list_models, no_input,
                show_reasoning=show_reasoning, resume=resume, trust_project=trust_project))
            if restart is None or channel != "terminal":
                return
            # The REPL ended for /plugins and the session is fully torn down: run the plugin's
            # step here, on the plain terminal (nothing else draws on it; a typed secret never
            # reaches the input box's history), then rebuild with the conversation carried over.
            import gc

            from localharness.cli.plugins_cmd import session_step
            from localharness.provider.client import reset_inference_gate
            # An object the session left in a reference cycle is finalized when the GC reaches it,
            # and a finalizer that closes on "the running loop" must find none: collect before the
            # step's own event loops run, and again before the rebuilt session's.
            gc.collect()
            outcome = session_step(restart.action, config_dir)
            reset_inference_gate()  # the next asyncio.run is a new event loop
            gc.collect()
            resume = replace(restart.resume, failed_check=outcome.failed_check,
                             skipped_check=outcome.skipped_check, step_stopped=outcome.stopped)
    except KeyboardInterrupt:
        console.print("\nGoodbye.")
