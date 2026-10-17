"""`localharness serve --config-dir <dir>` — the body of the autostart unit (spec 20 §9.3, D-14..D-17).

Hidden: the unit's `ExecStart` runs it (`python -I -m localharness serve --config-dir <dir>`), a person
never does. It decides what to serve exactly as `start` would, WITHOUT asking anyone, refuses anything the
machine record or the unit file does not cover, and otherwise plans the same launch `activate` would,
records it and replaces itself with the server.

The steps, in order (each refusal prints one line on stderr, which the journal keeps):

 1. `LOCALHARNESS_DIR` is set to the absolute pinned dir before any resolver runs.
 2. No machine record -> 78 (a first start adopts the whole disk; the unit never does).
 3. `decide_machine_trust(ask=False)`; the harness loads with `machine_withheld` and no workspace layer.
 4. The ladder (`resolve_serving`): a refusal -> 78; a landed record the machine record withholds -> 78.
 5. A record the unit cannot serve (docker, Ollama, LM Studio, attach-only) -> its line and 0.
 6. The unit's own `autostart.env` entry: withheld, or the unit file present with no usable entry -> 78.
 7. The service manager is asked (it cannot be -> 1); a verified live launch of this record -> 0.
 8. The own-unit identity test fails -> 78; the captured environment differs from the record -> 78.
 9. Launch decision (refused -> 78), plan, then under the launch lock: the live-pidfile guard (a live
    process holds the pidfile -> 75), the config.ini home, the pidfile with this process's own pid and the
    boot marker; the lock is released and `server.exec_server` replaces this process.

Exit codes are the contract the unit's restart policy reads (`RestartPreventExitStatus=78`):

  78  refused by trust, the record, the anchor or the identity test: restarting cannot help, so it stops;
   0  nothing to do (a record the unit does not serve, a server already up);
  75  a live process holds the pidfile: retried within the start limit;
   1  a transient failure (the manager could not be asked, the lock wait timed out, the config failed to
      load, a file could not be written, the binary is not on the captured PATH): retried within the limit.

`serve` never adopts, never launches through a strategy and never reads a stored selection itself: what it
serves is whatever the one `resolve_serving` call returns. It writes the pidfile, the boot marker and the
llama.cpp config.ini home, nothing else. The server's environment is built from an allowlist (see
`EXEC_INHERITED`), never from the service manager's own.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import typer
from rich.console import Console

from localharness.cli import model_ops
from localharness.cli.workspace import decide_machine_trust, visible_text
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle, procfs, server, units
from localharness.provider.runtimes import RUNTIMES

err_console = Console(stderr=True)

EXIT_REFUSED = 78
EXIT_NOTHING = 0
EXIT_LIVE_PIDFILE = 75
EXIT_TRANSIENT = 1

EXEC_INHERITED: tuple[str, ...] = (
    "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "LC_MESSAGES", "XDG_RUNTIME_DIR",
    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
)
"""The eleven variables the exec'd server takes from the service manager's environment, named one by one
(no wildcard): everything else it gets is the pinned home, the values captured in the verified unit file,
the record's own `env:`, the llama.cpp config home and the key (D-15)."""

NO_RECORD_LINE = ("serve: this machine has no confirmed record yet — run localharness start in a terminal "
                  "first")
CONFIG_FAILED_LINE = "serve: cannot load the config: {why}"
NOT_OWN_UNIT_LINE = "serve: the autostart unit is not the harness's own — {why}; run localharness model --autostart on"
ENV_WITHHELD_LINE = ("serve: the machine record withholds the autostart unit's environment ({path}) — run "
                     "localharness start in a terminal")
ENV_MISSING_LINE = ("serve: the machine record holds no usable autostart environment for {path} — run "
                    "localharness model --autostart on")
ENV_CHANGED_LINE = ("serve: the autostart unit's environment ({path}) differs from the machine record — run "
                    "localharness start in a terminal")
ALREADY_SERVING_LINE = "already serving (pid {pid}) — not started twice"
NO_MANAGER_LINE = "serve: the service manager did not answer — will try again"
NOT_ON_PATH_LINE = "serve: {binary} is not on the captured PATH ({path}) — set it in the shell and run localharness model --autostart on"
NOT_EXECUTABLE_LINE = "serve: {binary} is not an executable file"
HOME_FAILED_LINE = "serve: could not prepare the llama.cpp config home: {why}"
PIDFILE_FAILED_LINE = "serve: could not record the launch: {why}"
MARKER_FAILED_LINE = "serve: the boot marker was not written ({why}) — launching anyway"
EXEC_FAILED_LINE = "serve: could not start {binary}: {why}"
LAUNCHING_LINE = "serve: launching {name} ({shape})"


def _say(line: str) -> None:
    err_console.print(line, markup=False, soft_wrap=True)


def _stop(code: int, line: str | None = None) -> NoReturn:
    if line:
        _say(line)
    raise typer.Exit(code)


def _planned_argv(spec: Any, config_dir: Path, autosense: Any, decision: Any, *, keyed: bool,
                  unit: Any) -> "tuple[list[str], lifecycle.LaunchPlan | None, lifecycle.Sensed | None]":
    """The argv `activate` would start for ``spec``, its `LaunchPlan` and the `Sensed` it came from: sense,
    then plan, when the machine has autosense knobs (both servable runtimes plan, as their `activate` does),
    else the 0.18 command. ``unit`` is the autostart unit's runtime state (None outside the unit): it is what
    lets `sense_launch` read the boot marker. Nothing is written."""
    if autosense is not None:
        sensed = lifecycle.sense_launch(spec, config_dir, autosense, unit=unit)
        plan = lifecycle.plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed)
        return list(plan.argv), plan, sensed
    argv = server.serve_command(
        spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok,
        context=None, cache_ram=None)
    return argv, None, None


def _exec_environment(captured: Mapping[str, str], home: str, inherited: Mapping[str, str], spec: Any,
                      api_key: str | None, ini_home: str | None) -> dict[str, str]:
    """The environment the server is exec'd with (D-15): the pinned ``home``, then the values ``captured``
    from the verified unit file (`PATH`, `HF_HOME`, `LD_LIBRARY_PATH`), then exactly the `EXEC_INHERITED`
    names present in ``inherited``; `child_environment` adds the record's `env:`, the llama.cpp config home
    and the key and scrubs the runtime's own namespace. Returns `child_environment`'s dict, or a copy of the
    base when it had nothing to add — never ``inherited`` itself, so nothing the manager set outside the
    eleven names reaches the server."""
    base = {"HOME": home, **captured, **{name: inherited[name] for name in EXEC_INHERITED if name in inherited}}
    env, _dropped = server.child_environment(
        spec.runtime, base, record_env=getattr(spec, "env", None), api_key=api_key, xdg_config_home=ini_home)
    return env if env is not None else dict(base)


def _resolve_binary(argv0: str, captured_path: str) -> "tuple[str | None, str]":
    """``(path, line)``: the executable ``argv0`` names, or None with the line saying why not. An absolute
    path must be an executable regular file; any other name is looked up on the CAPTURED `PATH` only (never
    the manager's, never the current directory); a name with a separator that is not absolute is not a name."""
    if os.path.isabs(argv0):
        if os.path.isfile(argv0) and os.access(argv0, os.X_OK):
            return argv0, ""
        return None, NOT_EXECUTABLE_LINE.format(binary=visible_text(argv0))
    found = None if os.sep in argv0 else shutil.which(argv0, path=captured_path)
    if found is None:
        return None, NOT_ON_PATH_LINE.format(binary=visible_text(argv0), path=visible_text(captured_path))
    return found, ""


def _marker(spec: Any, plan: "lifecycle.LaunchPlan | None") -> "units.Marker":
    """The boot marker of the launch ``plan`` describes (D-17): which record and binary, the invocation
    the manager gave this run, why the retry shape was chosen (`death` for the boot retry, `remembered`
    for the ledger's, else `none`), and the window and memory the plan was made from."""
    report = plan.report if plan is not None else None
    cause = {"boot": "death", "remembered": "remembered"}.get(plan.cause if plan is not None else None, "none")
    return units.Marker(
        record=lifecycle.record_key(spec), binary=str(spec.binary),
        invocation=os.environ.get("INVOCATION_ID") or None, cause=cause,
        shape=plan.shape if plan is not None else "full",
        planned_context=report.asked if report is not None else None,
        full_fit=plan.full_fit if plan is not None else None,
        free_bytes=report.free_bytes if report is not None else None,
        dropped=tuple(plan.dropped) if plan is not None else (),
        at=datetime.now(timezone.utc).isoformat())


def serve(
    config_dir: str = typer.Option(
        ..., "--config-dir", help="The global config dir the autostart unit serves (pinned by the unit)."),
) -> None:
    """Serve the preset the autostart unit was installed for: the unit's own start command."""
    pinned = units.unit_config_dir(config_dir)
    os.environ["LOCALHARNESS_DIR"] = pinned
    cfg = Path(pinned)

    # 2-3. The record, the trust decision (never a question) and the harness it leaves.
    if trust.recorded_machine(cfg) is None:
        _stop(EXIT_REFUSED, NO_RECORD_LINE)
    machine_trust = decide_machine_trust(cfg, ask=False)
    if machine_trust.line:
        _say(machine_trust.line)
    loader = ConfigLoader(config_dir=cfg, local_config_dir=None, machine_withheld=machine_trust.withheld)
    try:
        harness = loader.load_harness()
    except Exception as exc:  # noqa: BLE001 — any failure to load is the same transient: say it, retry
        _stop(EXIT_TRANSIENT, CONFIG_FAILED_LINE.format(why=visible_text(f"{type(exc).__name__}: {exc}")))

    # 4. What to serve: the ladder's answer, refused when the machine record withholds it.
    serving = model_ops.resolve_serving(harness, machine_trust, None, config_dir=cfg)
    if isinstance(serving, model_ops.ServingRefusal):
        if serving.line != machine_trust.line:
            _say(serving.line)
        _stop(EXIT_REFUSED)
    record = serving.record
    if model_ops.record_withheld(machine_trust, record):
        refusal = model_ops._withheld_refusal(machine_trust, record, cfg)
        if refusal.line != machine_trust.line:
            _say(refusal.line)
        _stop(EXIT_REFUSED)

    # 5. A record the unit cannot serve is nothing to do.
    spec = record.spec
    if not lifecycle.unit_servable(spec):
        _stop(EXIT_NOTHING, lifecycle.NOT_SERVED_BY_UNIT_LINE.format(
            name=visible_text(record.name), kind=lifecycle.unit_kind(spec)))

    # 6. The unit's own environment entry: never "nothing to check".
    unit_path = str(units.harness_unit_path())
    body = units.read_harness_unit()
    if body is None:
        _stop(EXIT_REFUSED, NOT_OWN_UNIT_LINE.format(why=f"there is no readable unit file at {unit_path}"))
    if (trust.LAUNCH_KIND, trust.AUTOSTART_ENTRY_NAME) in machine_trust.withheld.get(
            trust.AUTOSTART_FILE, frozenset()):
        _stop(EXIT_REFUSED, ENV_WITHHELD_LINE.format(path=unit_path))
    captured = units.captured_env(body)
    recorded = trust.recorded_autostart_entry(cfg)
    if captured is None or recorded is None or not trust.autostart_entries(cfg):
        _stop(EXIT_REFUSED, ENV_MISSING_LINE.format(path=unit_path))

    # 7. The manager, and whether this record is already being served.
    state = units.unit_runtime_state()
    if state is None:
        _stop(EXIT_TRANSIENT, NO_MANAGER_LINE)
    identity = lifecycle.record_identity(spec)
    live = procfs.read_launch(cfg, expect=identity, main_pid=state.main_pid) if identity is not None else None
    if live is not None and live.verified:
        _stop(EXIT_NOTHING, ALREADY_SERVING_LINE.format(pid=live.pid))

    # 8. The own-unit identity test, then the environment the person confirmed.
    why = units.harness_unit_failure(state, cfg, body=body)
    if why is not None:
        _stop(EXIT_REFUSED, NOT_OWN_UNIT_LINE.format(why=why))
    try:
        confirmed = json.loads(recorded.get("shown", ""))
    except (TypeError, ValueError):
        confirmed = None
    if confirmed != captured:
        _stop(EXIT_REFUSED, ENV_CHANGED_LINE.format(path=unit_path))

    # 9. The key (read once, held in memory), the decision and the plan.
    from localharness.cli import start_cmd  # heavy: only a launch needs the key reader

    if record.peer is None:
        key_line = start_cmd._read_session_key(harness)
    else:
        peer_key, key_line = model_ops._peer_session_key(record.peer)
        model_ops._hold_peer_session_key(record.peer, peer_key)
    if key_line:
        _say(key_line)
    api_key = model_ops._launch_key(harness, record.base_url, spec).get("api_key")
    keyed = bool(api_key)
    decision = server.launch_decision(spec, keyed, peer=record.peer.name if record.peer is not None else None)
    if decision.refusal is not None:
        _stop(EXIT_REFUSED, decision.refusal)
    if decision.note:
        _say(decision.note)
    argv, plan, _sensed = _planned_argv(spec, cfg, harness.autosense, decision, keyed=keyed, unit=state)
    binary, why_not = _resolve_binary(argv[0], captured.get("PATH", ""))
    if binary is None:
        _stop(EXIT_TRANSIENT, why_not)

    # The one launch lock (the REPL's and `start`'s own), taken synchronously: the pidfile guard, the
    # pidfile with this process's pid (an exec keeps it) and the boot marker.
    ini_home: str | None = None
    try:
        with lifecycle.launch_lock_sync(cfg, timeout_seconds=units.TIMEOUT_START_SECONDS):
            try:
                stale = server.refuse_live_launch(cfg, argv)
            except RuntimeError as refused:
                _stop(EXIT_LIVE_PIDFILE, str(refused))
            if stale:
                _say(stale.strip())
            if "config_ini" in RUNTIMES[spec.runtime].record_fields:
                try:
                    ini_home = str(server.prepare_config_ini_home(
                        cfg, spec.port, getattr(spec, "config_ini", None)))
                except (OSError, UnicodeError) as exc:
                    _stop(EXIT_TRANSIENT, HOME_FAILED_LINE.format(why=getattr(exc, "strerror", None) or exc))
            try:
                server.write_pidfile(cfg, os.getpid(), port=spec.port)
            except OSError as exc:
                _stop(EXIT_TRANSIENT, PIDFILE_FAILED_LINE.format(why=exc.strerror or exc))
            try:
                units.write_marker(cfg, _marker(spec, plan))
            except (OSError, ValueError) as exc:
                _say(MARKER_FAILED_LINE.format(why=getattr(exc, "strerror", None) or exc))
    except TimeoutError as timed_out:
        _stop(EXIT_TRANSIENT, str(timed_out))

    env = _exec_environment(captured, units.harness_home(), os.environ, spec, api_key, ini_home)
    _say(LAUNCHING_LINE.format(name=visible_text(record.name), shape=plan.shape if plan is not None else "full"))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    try:
        server.exec_server(binary, argv, env)
    except OSError as exc:
        _stop(EXIT_TRANSIENT, EXEC_FAILED_LINE.format(binary=visible_text(binary), why=exc.strerror or exc))
