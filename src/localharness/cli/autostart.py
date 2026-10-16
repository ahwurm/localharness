"""`localharness model --autostart on|off|status` — the body of the switch (spec 20 §9.1, §9.2, §12; D-01..D-13).

`cli/model_cmd.py` owns the command surface: the option, the local-operator gate and the one machine-trust
question. Everything after the question lives here, so `status` and the doctor row share one module with `on`:

- the lines (the gate's, the scope refusal's, the receipt's, the failures', the status's);
- `turn_on`: capture the shell, render the unit, and under the launch lock write it, adopt its captured
  environment (the entry of the text written, not of a re-read) into the machine record, `daemon-reload`,
  re-read the file and require it byte-equal to the text written, re-check that the manager loaded the
  harness's own unit, and `enable` it (`--now` only when nothing of the harness's is serving; a running server is kept);
- `turn_off`: the undo. Only the harness's own unit (all three identity tests, before and again under the
  lock) is disabled, removed and pruned from the machine record; anything else is left as it is and named;
- `status_lines` / `doctor_row`: what autostart is doing, read once and never written (D-11). Anyone may ask
  — no terminal, no trust question, no verdict — because it decides nothing and removes nothing.

Exit codes: 2 when the person can fix the call (a value the unit cannot carry); 1 when the system could not
do it (no service manager, a failing verb, a file that could not be written, a unit the manager loaded from
elsewhere, a launch lock that stayed busy); 0 on success. Nothing here runs the login manager's program: linger
is read from the filesystem and the command that would turn it on is only named in the receipt.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from localharness.cli import model_ops
from localharness.cli.workspace import MachineTrust, shown_path, uncovered, visible_text
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.paths import global_config_dir
from localharness.provider import lifecycle, procfs, units

AUTOSTART_TERMINAL_LINE = "autostart is set from a terminal: localharness model --autostart {mode}"
AUTOSTART_SCOPE_LINE = ("autostart serves llama.cpp and a vLLM binary; {name} is {kind} — "
                        "localharness model <a llama.cpp preset> first")
AUTOSTART_MODE_LINE = "--autostart takes on, off or status"
AUTOSTART_ALONE_LINE = "--autostart takes no model name, --download or --file"

RECEIPT_BOOT = "autostart on: {path} serves {name} at login and boot (linger on; {captured})"
RECEIPT_LOGIN = ("autostart on: {path} serves {name} at login "
                 "(at boot too after: loginctl enable-linger {user}; {captured})")
TAKES_OVER_CLAUSE = " — takes over at the next login or /model"
CAPTURED_TEXT = "PATH captured from this shell"

NO_MANAGER_LINE = "autostart needs the user service manager (systemctl in /usr/bin or /bin) — not found"
REFUSED_LINE = "autostart {mode}: {why}"
VERB_FAILED_LINE = "autostart {mode}: {verb} failed (exit {code}): {err}"
VERB_NO_ANSWER_LINE = "autostart {mode}: {verb} failed: the service manager did not answer"
WRITE_FAILED_LINE = "autostart {mode}: could not write {path}: {why}"
REWRITE_LINE = "rewriting {path}; it now reads:"
CHANGED_SINCE_WRITE_LINE = ("autostart on: {path} changed after localharness wrote it — "
                            "nothing was enabled; look at the file, then run it again")
LOADED_ELSEWHERE_LINE = "autostart on: {unit} {why} — nothing was enabled"
OFF_RECEIPT = "autostart off: removed {path}; nothing starts at login"
OFF_UNLOADED_CLEARED = " (the service manager had not loaded it; its boot link was cleared)"
OFF_UNLOADED_LINK = (" (the service manager had not loaded it and would not clear its boot link: a dangling link "
                     "may remain, which systemd ignores)")
OFF_NOTHING = "autostart is off — {path} does not exist"
OFF_FOREIGN = "autostart off: {path} is not the unit localharness wrote ({why}) — left as it is"
SECOND_LAUNCHER_LINE = "{unit} ({state}) also launches a server on port {port} — two launchers race at login"

STATUS_ON_LINE = "Autostart: on — {path} ({file_state}, {active_state}, NRestarts {restarts}, MainPID {pid})"
STATUS_FOREIGN_LINE = "Autostart: foreign — {path} {why}"
STATUS_OFF_LINE = "Autostart: off — localharness model --autostart on serves the last choice at login"
SERVES_LINE = "serves: {name}"
SERVES_UNKNOWN_LINE = "serves: (the config could not be loaded: {why})"
SERVES_REFUSED_LINE = "serves: ({why})"
LINGER_ON_LINE = "linger: on"
LINGER_OFF_LINE = "linger: off — at boot too after: loginctl enable-linger {user}"
CAPTURED_PATH_LINE = "PATH captured: {path}"
MARKER_LINE = "serving {name} at the {shape} shape since boot{parts}"
IN_FLIGHT_LINE = "in flight: {name} (requested by pid {pid})"
IN_FLIGHT_OUTRANKS_LINE = ("in flight: {name} (requested by pid {pid}) — "
                           "outranks pending_preset {pending} while that process runs")
STATUS_ENV_RECORD = "autostart.env ({path})"
STATUS_UNKNOWN_RECORD = "the serving record"

LOCK_WAIT_SECONDS = 60
ERROR_TEXT_CAP = 400

Say = Callable[[str], None]


def display_path(path: Any) -> str:
    """``path`` as a printed line shows it: the home directory as `~`, control characters escaped."""
    return shown_path(path)


def captured_text(captured: "units.CapturedShell") -> str:
    """What the receipt says was taken from the shell: `PATH captured from this shell`, then each dropped
    `PATH` / `LD_LIBRARY_PATH` element (an empty one as `(empty)`) and each `HF_HOME` made absolute."""
    text = CAPTURED_TEXT
    if captured.dropped:
        shown = ", ".join(f"{name} {visible_text(element) or '(empty)'}" for name, element in captured.dropped)
        text += f", dropped relative: {shown}"
    for name, written, absolute in captured.absolutized:
        text += f", {name} made absolute: {visible_text(written)} → {visible_text(absolute)}"
    return text


def harness_serves(harness: Any, config_dir: Any, state: "units.UnitRuntimeState | None") -> bool:
    """Is something of the harness's serving right now? The unit is active, or the pidfile names a live
    process that `/proc` verifies as the launch of one of the records the config holds. `on` keeps such a
    server (`enable` without `--now`)."""
    if state is not None and state.active_state == "active":
        return True
    for record in model_ops.serving_records(harness):
        if record.spec is None:
            continue
        identity = lifecycle.record_identity(record.spec)
        if identity is None:
            continue
        live = procfs.read_launch(config_dir, expect=identity)
        if live is not None and live.verified:
            return True
    return False


def _verb_error(mode: str, verb: str, result: "units.ManagerResult | None", harness: Any) -> str:
    if result is None:
        return VERB_NO_ANSWER_LINE.format(mode=mode, verb=verb)
    text = model_ops.redact_for_channel(result.err, harness)
    text = visible_text(" ".join(text.split()))[:ERROR_TEXT_CAP]
    return VERB_FAILED_LINE.format(mode=mode, verb=verb, code=result.code, err=text)


def turn_on(harness: Any, record: Any, config_dir: Path, *, environ: Mapping[str, str], cwd: str,
            out: Say, err: Say) -> int:
    """Install and enable the harness's own autostart unit for ``config_dir`` (spec 20 §9.2). ``record`` is
    what the unit will serve (its name is in the receipt); ``environ`` and ``cwd`` are the shell's, from
    which `PATH` (always) and `HF_HOME` / `LD_LIBRARY_PATH` (when set) are captured. ``out`` and ``err``
    each take one line. Returns the exit code (see the module doc)."""
    if units.systemctl_path() is None:
        err(NO_MANAGER_LINE)
        return 1
    try:
        captured = units.capture_shell_env(environ, cwd)
        body = units.render_harness_unit(sys.executable, config_dir, captured.env, home=units.harness_home())
    except ValueError as exc:
        err(REFUSED_LINE.format(mode="on", why=visible_text(exc)))
        return 2

    try:
        with lifecycle.launch_lock_sync(config_dir, timeout_seconds=LOCK_WAIT_SECONDS):
            previous = units.read_harness_unit()
            try:
                path = units.write_harness_unit(body)
            except OSError as exc:
                err(WRITE_FAILED_LINE.format(mode="on", path=display_path(units.harness_unit_path()),
                                             why=visible_text(exc)))
                return 1
            if previous is not None and previous != body:
                out(REWRITE_LINE.format(path=display_path(path)))
                out(body)
            # The unit file now holds an environment nobody confirmed. Until this adoption lands, the
            # next terminal `start` asks about it and `serve` exits 78: nothing launches unconfirmed. The
            # entry is the one the text just written holds, never a re-read of the disk: the record
            # changes by exactly what this act wrote, even when something else replaced the file.
            trust.adopt_entries(config_dir, trust.autostart_entries_for_body(config_dir, body))

            reloaded = units.systemctl_user("daemon-reload")
            if reloaded is None or reloaded.code != 0:
                err(_verb_error("on", "daemon-reload", reloaded, harness))
                return 1
            state = units.unit_runtime_state()
            # The manager loaded whatever was on disk at the reload. Read the file again and require it to
            # be the text written above before anything is enabled (a residual window remains between this
            # read and the manager's own, which `serve` closes at every start by comparing the record).
            on_disk = units.read_harness_unit()
            if on_disk != body:
                err(CHANGED_SINCE_WRITE_LINE.format(path=display_path(path)))
                return 1
            failure = units.harness_unit_failure(state, config_dir, body=on_disk)
            if failure is not None:
                err(LOADED_ELSEWHERE_LINE.format(unit=units.HARNESS_UNIT, why=visible_text(failure)))
                return 1

            keep = harness_serves(harness, config_dir, state)
            args = ("enable", units.HARNESS_UNIT) if keep else ("enable", "--now", units.HARNESS_UNIT)
            enabled = units.systemctl_user(*args)
            if enabled is None or enabled.code != 0:
                err(_verb_error("on", "enable", enabled, harness))
                return 1
    except TimeoutError as exc:
        err(REFUSED_LINE.format(mode="on", why=visible_text(exc)))
        return 1
    except OSError as exc:
        err(REFUSED_LINE.format(mode="on", why=visible_text(exc)))
        return 1

    shown = dict(path=display_path(path), name=visible_text(record.name), captured=captured_text(captured))
    if units.linger_on():
        receipt = RECEIPT_BOOT.format(**shown)
    else:
        receipt = RECEIPT_LOGIN.format(user=visible_text(units.current_user()), **shown)
    out(receipt + (TAKES_OVER_CLAUSE if keep else ""))

    ports = {r.spec.port for r in model_ops.serving_records(harness)
             if r.spec is not None and isinstance(getattr(r.spec, "port", None), int)}
    for other in units.other_launchers(ports):
        err(SECOND_LAUNCHER_LINE.format(unit=visible_text(other.unit), state=other.state, port=other.port))
    return 0


def _foreign(path: Path, why: str, err: Say) -> int:
    err(OFF_FOREIGN.format(path=display_path(path), why=visible_text(why)))
    return 1


def _off_verdict(state: Any, config_dir: Path) -> tuple[bool, str | None]:
    """``(unloaded, why)`` for `off`: ``why`` is None when the unit may be undone. ``unloaded`` is True for the
    file `on` wrote and the manager never loaded (`FragmentPath` empty, the file byte-equal to the render):
    nothing runs and nothing is enabled, so `off` removes the file without `disable`. Every other unit that
    fails the identity tests is foreign and named in ``why``."""
    why = units.harness_unit_failure(state, config_dir)
    if why is None:
        return False, None
    if units.unloaded_own_unit(state, config_dir):
        return True, None
    return False, why


def turn_off(config_dir: Path, *, out: Say, err: Say, harness: Any = None) -> int:
    """Disable and remove the harness's own autostart unit for ``config_dir`` and prune its `autostart.env`
    entry (spec 20 §9.2 `off`, D-11, D-13). It acts only on a unit that passes all three identity tests —
    loaded from `harness_unit_path()`, no drop-ins, the file byte-equal to what localharness renders — read
    once up front and again under the launch lock, before anything is stopped; any other unit is left as it
    is and named (exit 1). The one exception is the file `on` wrote and the manager never loaded (a failing
    reload): empty `FragmentPath`, no drop-ins, the file byte-equal to the render. Nothing runs, so it is
    removed and pruned with no `disable --now`; a plain `disable`, best effort, clears a boot link an earlier
    `on` left (IN-06), and the receipt says whether it did. No file and nothing loaded is already off (exit 0). A failing
    `disable --now` removes nothing and prunes nothing. ``harness`` only scrubs a failing verb's stderr of the config's
    secrets; it is never asked about trust and `unit-target` is left as it is (R-14-26). Returns the exit
    code (see the module doc)."""
    if units.systemctl_path() is None:
        err(NO_MANAGER_LINE)
        return 1
    path = units.harness_unit_path()
    state = units.unit_runtime_state()
    if not os.path.lexists(path) and not getattr(state, "fragment_path", ""):
        out(OFF_NOTHING.format(path=display_path(path)))
        return 0
    unloaded, why = _off_verdict(state, config_dir)
    if why is not None:
        return _foreign(path, why, err)

    try:
        with lifecycle.launch_lock_sync(config_dir, timeout_seconds=LOCK_WAIT_SECONDS):
            # The first read may be stale by the time the lock is ours (another `on` or `off`, an edit):
            # nothing is stopped on a unit the fresh state does not prove to be the harness's own.
            unloaded, why = _off_verdict(units.unit_runtime_state(), config_dir)
            if why is not None:
                return _foreign(path, why, err)

            tail = ""
            if not unloaded:
                units.systemctl_user("reset-failed", units.HARNESS_UNIT)  # best effort: a clean slate, never a gate
                disabled = units.systemctl_user("disable", "--now", units.HARNESS_UNIT)
                if disabled is None or disabled.code != 0:
                    err(_verb_error("off", "disable", disabled, harness))
                    return 1
            else:
                # Nothing runs, but an earlier `on` may have left the boot link the manager no longer maps to a
                # loaded unit. `disable` reads the file on disk, so it clears that link while the file exists;
                # a manager that will not is no reason to keep the file (nothing starts from a dangling link).
                cleared = units.systemctl_user("disable", units.HARNESS_UNIT)
                tail = OFF_UNLOADED_CLEARED if cleared is not None and cleared.code == 0 else OFF_UNLOADED_LINK
            units.remove_harness_unit()
            reloaded = units.systemctl_user("daemon-reload")
            trust.forget_entries(config_dir, [trust.AUTOSTART_KEY])
            if reloaded is None or reloaded.code != 0:
                err(_verb_error("off", "daemon-reload", reloaded, harness))
                return 1
    except TimeoutError as exc:
        err(REFUSED_LINE.format(mode="off", why=visible_text(exc)))
        return 1
    except OSError as exc:
        err(REFUSED_LINE.format(mode="off", why=visible_text(exc)))
        return 1

    out(OFF_RECEIPT.format(path=display_path(path)) + tail)
    return 0


# ------------------------------------------------------------------------------------------- status


def _first_status_line(state: Any, path: Path, config_dir: Path, body: str | None) -> tuple[str, bool]:
    """The `Autostart:` line and whether the unit is the harness's own: `on` with the manager's own words, or
    `foreign` with the phrase of the FIRST identity test it failed (D-11, D-13)."""
    why = units.harness_unit_failure(state, config_dir, body=body)
    if why is not None:
        return STATUS_FOREIGN_LINE.format(path=display_path(path), why=visible_text(why)), False
    return STATUS_ON_LINE.format(
        path=display_path(path), file_state=visible_text(state.unit_file_state or "-"),
        active_state=visible_text(state.active_state or "-"), restarts=state.n_restarts,
        pid=state.main_pid if state.main_pid is not None else "-"), True


def _marker_line(marker: "units.Marker", records: Any) -> str:
    """The boot marker as a line (R-14-31): the record's name (the raw key when no record has it), the shape
    it served at, and what was dropped or halved to get there."""
    names = {lifecycle.record_key(r.spec): r.name for r in records if r.spec is not None}
    parts = []
    if "draft-mtp" in marker.dropped:
        parts.append("mtp off")
    if "parsers" in marker.dropped:
        parts.append("parsers off")
    if (marker.planned_context is not None and marker.full_fit is not None
            and marker.planned_context < marker.full_fit):
        parts.append("-c halved")
    return MARKER_LINE.format(
        name=visible_text(names.get(marker.record, marker.record)), shape=visible_text(marker.shape),
        parts=" — " + ", ".join(parts) if parts else "")


def _withheld_record_name(cfg: Path, serving: Any) -> str:
    """What a 78 withholds (D-11): `autostart.env (<unit path>)` when the machine record does not cover the
    unit's captured environment as it is now (a read-only compare of the record against the snapshot), else
    the name of the record the unit would serve."""
    recorded = trust.recorded_machine(cfg)
    gap = uncovered(recorded["entries"] if recorded is not None else [], trust.machine_snapshot(cfg))
    if trust.AUTOSTART_KEY in {(e.get("file"), e.get("kind"), e.get("name")) for e in gap}:
        return STATUS_ENV_RECORD.format(path=display_path(units.harness_unit_path()))
    if isinstance(serving, model_ops.Serving):
        return visible_text(serving.record.name)
    return STATUS_UNKNOWN_RECORD


def _gather(config_dir: Any) -> tuple[list[str], str | None]:
    """Every line `status` prints, and the in-flight line that outranks a `pending_preset` (or None). Reads
    the manager, the unit file, the global config, linger and the stores once each and writes nothing: no
    machine-trust decision, no adoption, no removal, no clearing of a stale target (R-14-30, R-14-46). The
    journal program is never run: the start-limit line only names the command for the person."""
    cfg = Path(config_dir)
    path = units.harness_unit_path()
    state = units.unit_runtime_state()
    if not os.path.lexists(path) and not getattr(state, "fragment_path", ""):
        return [STATUS_OFF_LINE], None
    body = units.read_harness_unit()
    first, _own = _first_status_line(state, path, cfg, body)
    lines = [first]
    outranks: str | None = None

    # The target is named from the global config alone with an empty `MachineTrust()`: display only, the
    # same answer `serve` reaches unless the machine record withholds it (which `stopped:` then says).
    harness: Any = None
    serving: Any = None
    try:
        harness = ConfigLoader(config_dir=cfg, local_config_dir=None).load_harness()
    except Exception as exc:  # noqa: BLE001 — status must answer even when the config cannot be read
        lines.append(SERVES_UNKNOWN_LINE.format(why=visible_text(f"{type(exc).__name__}: {exc}")))
    else:
        serving = model_ops.resolve_serving(harness, MachineTrust(), None, config_dir=cfg)
        if isinstance(serving, model_ops.ServingRefusal):
            lines.append(SERVES_REFUSED_LINE.format(why=visible_text(serving.line)))
        else:
            lines.append(SERVES_LINE.format(name=visible_text(serving.record.name)))

    # A live in-flight `unit-target` outranks the person's `pending_preset` (R-14-45); never silent here.
    target_dir = global_config_dir(cfg)
    request = units.in_flight_request(target_dir)
    if request is not None:
        pending = getattr(harness, "pending_preset", None)
        if pending and isinstance(serving, model_ops.Serving) and serving.rung == model_ops.RUNG_UNIT:
            outranks = IN_FLIGHT_OUTRANKS_LINE.format(
                name=visible_text(serving.record.name), pid=request.pid, pending=visible_text(pending))
            lines.append(outranks)
        else:
            lines.append(IN_FLIGHT_LINE.format(name=visible_text(request.name), pid=request.pid))

    lines.append(LINGER_ON_LINE if units.linger_on() else LINGER_OFF_LINE.format(
        user=visible_text(units.current_user())))
    captured = units.captured_env(body) if body is not None else None
    if captured is not None and captured.get("PATH"):
        lines.append(CAPTURED_PATH_LINE.format(path=visible_text(captured["PATH"])))

    records = model_ops.serving_records(harness) if harness is not None else ()
    ports = {r.spec.port for r in records
             if r.spec is not None and isinstance(getattr(r.spec, "port", None), int)}
    for other in units.other_launchers(ports):
        lines.append(SECOND_LAUNCHER_LINE.format(unit=visible_text(other.unit), state=other.state, port=other.port))

    # Only the run serving now wrote a marker worth showing as "since boot" (R-14-43); another run's is left.
    live_run = getattr(state, "invocation_id", None)
    marker = units.read_marker(target_dir) if live_run else None
    if marker is not None and marker.invocation == live_run:
        lines.append(_marker_line(marker, records))

    not_served = None
    if isinstance(serving, model_ops.Serving) and not lifecycle.unit_servable(serving.record.spec):
        not_served = lifecycle.NOT_SERVED_BY_UNIT_LINE.format(
            name=visible_text(serving.record.name), kind=lifecycle.unit_kind(serving.record.spec))
    record_name = _withheld_record_name(cfg, serving) if getattr(state, "exec_main_status", None) == 78 else ""
    stopped = units.translate_exit(state, record_name=record_name, not_served=not_served)
    if stopped is not None:
        lines.append(stopped)
    return lines, outranks


def status_lines(config_dir: Any) -> list[str]:
    """What `localharness model --autostart status` prints, one string per line (spec 20 §9.2, D-11): the
    `Autostart:` line first, then the target, a live in-flight request, linger, the captured `PATH`, each
    second launcher, the boot marker of the run serving now, and the unit's exit in words when it stopped.
    Read-only; anyone may ask."""
    return _gather(config_dir)[0]


def doctor_row(config_dir: Any) -> str:
    """The one `Autostart:` row `doctor` shows: every line `status` prints, `; `-joined in its order."""
    return "; ".join(status_lines(config_dir))
