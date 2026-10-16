"""`localharness model` — list served/downloaded models, or switch what is served.

CLI parity for the REPL /model, for scripts, pre-`start` config, and CI. Reuses cli/model_ops
for the EXACT same atomic, audited overlay persistence and pin-trap warning. With the harness's own
autostart unit active, a recorded machine and a command that is not the agent's, `localharness model
<name>` swaps NOW through the unit (`_swap_now`, the same `lifecycle.swap` `start` and the REPL run);
otherwise it only records the choice (`pending_preset`) for the next start, or boot. The live-session
bits — hot-swap, TokenCounter rebind — are REPL-only and deliberately absent: with no running session
there is nothing to hot-swap.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import NoReturn, Optional

import typer
from rich.console import Console
from rich.markup import escape

from localharness import resolved_version
from localharness.cli import autostart, library, model_ops, start_serving
from localharness.cli.workspace import MachineTrust, decide_machine_trust, local_operator, shown_path, visible_text
from localharness.config import session_presence, trust
from localharness.config.loader import ConfigLoader
from localharness.config.paths import global_config_dir, resolve_config_dir
from localharness.provider import lifecycle, procfs, server, tuning
from localharness.provider.units import read_target

console = Console()
err_console = Console(stderr=True)


SCRIPT_ALONE_LINE = "--script takes a preset name or --all, nothing else"
SCRIPT_NOTHING_LINE = "nothing to render — no record launches a server (llama.cpp or vLLM)"
SCRIPT_KIND_LINE = "{name} is {kind} — --script renders llama.cpp and vLLM launches"
SCRIPT_SKIP_LINE = "# {name}: {why}"
SETUP_TERMINAL_LINE = "--setup asks which models to set up — run it from a terminal: localharness model --setup"
SETUP_NO_CONFIG_LINE = "no config yet — localharness start sets one up"
SETUP_ALONE_LINE = "--setup takes nothing else"


def _refuse_autostart(line: str) -> None:
    """One plain line on stderr, exit 2: the call is wrong and the line says how to fix it."""
    err_console.print(line, markup=False, soft_wrap=True)
    raise typer.Exit(2)


def _setup(config_dir: Optional[str], *, person: bool) -> None:
    """`localharness model --setup`: the multi-select over every model the box holds that has no preset."""
    cfg_path = resolve_config_dir(config_dir)
    if not person:
        _refuse_autostart(SETUP_TERMINAL_LINE)
    if not (cfg_path / "config.yaml").exists():
        _refuse_autostart(SETUP_NO_CONFIG_LINE)
    library.sync(cfg_path, scope="all", person=person, choose=True)


def _autostart(mode: str, config_dir: Optional[str], *, with_other_arguments: bool) -> None:
    """`localharness model --autostart <mode>` (spec 20 §9.1, D-01). `on`, `off` and `status` exist in this
    release of the command; any other value is refused with the line that names all three. Changing what runs
    at every login needs a person at a terminal: the check comes BEFORE anything is decided, asked, loaded or
    written, and `_autostart_on` below is the one place under `cli/` that puts the machine-trust question
    (D-03); `off` removes and never asks. `status` only reads, so it runs anywhere — a script, a pipe, an
    agent's command — ahead of the terminal check (D-01, D-11)."""
    if with_other_arguments:
        _refuse_autostart(autostart.AUTOSTART_ALONE_LINE)
    if mode not in ("on", "off", "status"):
        _refuse_autostart(autostart.AUTOSTART_MODE_LINE)
    if mode == "status":
        _autostart_status(resolve_config_dir(config_dir))
    if not local_operator(False):
        _refuse_autostart(autostart.AUTOSTART_TERMINAL_LINE.format(mode=mode))
    if mode == "off":
        _autostart_off(resolve_config_dir(config_dir))
    _autostart_on(resolve_config_dir(config_dir))


def _autostart_status(cfg_path: Path) -> None:
    """`--autostart status`: print what autostart is doing and exit 0. It asks no trust question, decides and
    writes nothing, and needs no terminal."""
    for line in autostart.status_lines(cfg_path):
        console.print(line, markup=False, soft_wrap=True)
    raise typer.Exit(0)


def _autostart_off(cfg_path: Path) -> None:
    """`--autostart off` after the gate: remove the harness's own unit and its machine-record entry. It asks
    no trust question (it removes, never adopts). The config is read only so a failing verb's stderr is
    scrubbed of its secrets; one that cannot be read scrubs by pattern alone."""
    try:
        harness = ConfigLoader(config_dir=cfg_path, local_config_dir=None).load_harness()
    except Exception:  # noqa: BLE001 — undoing autostart must not depend on a readable config
        harness = None

    def out(line: str) -> None:
        console.print(line, markup=False, soft_wrap=True)

    def err(line: str) -> None:
        err_console.print(line, markup=False, soft_wrap=True)

    raise typer.Exit(autostart.turn_off(cfg_path, out=out, err=err, harness=harness))


def _autostart_on(cfg_path: Path) -> None:
    """`--autostart on` after the gate: decide machine trust (the one question a switch may ask), resolve
    what the unit would serve from the global config only (no workspace layer, no preset: the unit has
    neither), refuse a target the unit cannot serve, and hand the rest to `autostart.turn_on`."""
    machine_trust = decide_machine_trust(cfg_path, ask=True)
    if machine_trust.line:
        err_console.print(machine_trust.line, markup=False, soft_wrap=True)
    loader = ConfigLoader(config_dir=cfg_path, local_config_dir=None, machine_withheld=machine_trust.withheld)
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"failed to load config: {exc}"), soft_wrap=True)
        raise typer.Exit(2)
    serving = model_ops.resolve_serving(harness, machine_trust, None, config_dir=cfg_path)
    if isinstance(serving, model_ops.ServingRefusal):
        if serving.line != machine_trust.line:
            err_console.print(serving.line, markup=False, soft_wrap=True)
        raise typer.Exit(serving.exit_code)
    record = serving.record
    if model_ops.record_withheld(machine_trust, record):
        # The person said No to something this record starts: the unit's `serve` would refuse it at every
        # boot (exit 78), so there is nothing to turn on.
        refusal = model_ops._withheld_refusal(machine_trust, record, cfg_path)
        if refusal.line != machine_trust.line:
            err_console.print(refusal.line, markup=False, soft_wrap=True)
        raise typer.Exit(refusal.exit_code)
    if not lifecycle.unit_servable(record.spec):
        _refuse_autostart(autostart.AUTOSTART_SCOPE_LINE.format(
            name=record.name, kind=lifecycle.unit_kind(record.spec)))

    def out(line: str) -> None:
        console.print(line, markup=False, soft_wrap=True)

    def err(line: str) -> None:
        err_console.print(line, markup=False, soft_wrap=True)

    raise typer.Exit(autostart.turn_on(
        harness, record, cfg_path, environ=os.environ, cwd=os.getcwd(), out=out, err=err))


def _download(repo_id: str, filename: Optional[str], config_dir: Optional[str]) -> None:
    """`localharness model --download <repo_id> [--file <name>]` — standalone HF fetch, no
    running server / config required (unlike `init`'s guided-setup download, which is tied to
    the vLLM install flow). Whole-repo snapshot by default; --file fetches exactly one sibling
    file (the GGUF case, where a repo ships several quantizations side by side). One `.gguf` the
    hub vouches for, with a config and a person at the terminal, is recorded as a preset."""
    from localharness.provider import server as managed_server

    gguf = bool(filename) and filename.lower().endswith(".gguf")
    facts = managed_server.hub_file(repo_id, filename) if gguf else None
    try:
        if gguf:
            console.print(f"Downloading {filename!r} from {repo_id}...")
            path = managed_server.download_file(repo_id, filename, revision=facts.commit if facts else None)
        elif filename:
            console.print(f"Downloading {filename!r} from {repo_id}...")
            path = managed_server.download_file(repo_id, filename)
        else:
            console.print(f"Downloading {repo_id} (full snapshot)...")
            path = managed_server.download_model(repo_id)
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"download failed: {exc}"), soft_wrap=True
        )
        raise typer.Exit(2)

    # escape() around every path and repo id, markup outside it: a checkpoint under a folder
    # named `[old] models` is legal, and rich would silently delete `[old]` — printing a path
    # that does not exist right where the user is told what to copy (39-05). soft_wrap keeps a
    # deep path on one line for the copy.
    console.print("[green]✓[/green] " + escape(f"Downloaded to {path}"), soft_wrap=True)
    if gguf and _record_download(path, facts, filename, config_dir):
        return
    console.print(
        "To use it: point a llama.cpp profile's `server.model` at this path in config.yaml "
        "(single-model managed server), or run `localharness model "
        + escape(f"{path}`") + ("" if filename else escape(f" or `localharness model {repo_id}`"))
        + " once it's being served, to set it as the default.",
        soft_wrap=True,
    )


def _record_download(path: str, facts: server.HubFile | None, filename: str, config_dir: Optional[str]) -> bool:
    """Record a downloaded `.gguf` as a preset when the hub vouched for its bytes; else say why not. True when recorded."""
    cfg = resolve_config_dir(config_dir)
    checked = ("no config yet" if not (cfg / "config.yaml").exists()
               else "no person at this terminal" if not local_operator(False)
               else "the hub gave no file metadata" if facts is None
               else library.verify_download(path, size=facts.size, sha256=facts.sha256))
    if isinstance(checked, str):
        console.print(library.START_FINDS_IT_LINE.format(why=visible_text(checked)), markup=False, soft_wrap=True)
        return False
    console.print(library.VERIFIED_LINE.format(file=visible_text(filename), commit=facts.commit, sha256=facts.sha256),
                  markup=False, soft_wrap=True)
    if library.record_path(cfg, checked, first_run=False, adopt=True) is None:
        console.print(library.START_FINDS_IT_LINE.format(why="no preset was added"), markup=False, soft_wrap=True)
        return False
    return True


def _another_record_serves(harness, machine_trust, record) -> bool:
    """True when a launchable loopback record other than ``record`` answers at its own address now (the
    sensor `start` uses; loopback only, off the network). A failed look says False: it only decides
    whether "already the default" may stay silent."""
    others = model_ops.launchable_loopback_others(harness, machine_trust, record)
    if not others:
        return False
    try:
        sense = asyncio.run(model_ops.sense_serving(harness, (*others, record)))
    except Exception:  # noqa: BLE001 — an unreadable look is "nothing else answers"
        return False
    return any(r.prefix != record.prefix for r in sense.answering)


_BOOT_ENABLED = frozenset({"enabled", "enabled-runtime"})


def _select_preset(harness, record, current, loader, audit_dir, ws, machine_trust=None, unit=None) -> None:
    """Record ``record`` as what the next start serves (D-10, R-13-18): ``select_pending`` writes the
    peer's ``active_endpoint`` (the primary clears one) and ``pending_preset``; the primary's served
    name also becomes the default when it differs, without writing ``server.model`` (an alias must
    not become the model the next start loads, D-70, WR-54). This path never launches (D-10). ``unit`` is the
    harness's own unit state or None: the receipt says "the next start or boot serves it" only when that
    unit is enabled (it serves the last choice at every login), else "the next start serves it" (D-23)."""
    if (
        record.peer is None and record.served == current
        and harness.active_endpoint is None and not getattr(harness, "pending_preset", None)
        and not read_target(global_config_dir(loader._config_dir))
        and not _another_record_serves(harness, machine_trust, record)
    ):
        # The primary is already the default, nothing else is selected and no other record is serving
        # (one that is, after an attach or a failed write, still has to be swapped back from): nothing
        # to switch, and nothing is written (an overlay's bytes stay as they were).
        console.print(escape(f"{record.served} is already the default."), soft_wrap=True)
        return
    try:
        asyncio.run(model_ops.select_pending(harness, record, config_dir=loader._config_dir))
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"failed to select {record.name!r}: {exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(2)
    audit_warning = None
    if record.peer is None and record.served != current:
        try:
            audit_warning = asyncio.run(
                model_ops.persist_default_model(
                    harness, record.served, config_dir=loader._config_dir, audit_base_dir=audit_dir,
                    local_operator=local_operator(False), write_server_model=False,
                )
            )
        except Exception as exc:
            err_console.print(
                "[bold red]Error:[/bold red] "
                + escape(f"failed to persist {record.served!r}: {exc}"),
                soft_wrap=True,
            )
            raise typer.Exit(2)
    enabled = unit is not None and getattr(unit, "unit_file_state", "") in _BOOT_ENABLED
    receipt = model_ops.SELECTED_NEXT_START_OR_BOOT_LINE if enabled else model_ops.SELECTED_NEXT_START_LINE
    console.print(receipt.format(name=record.name), markup=False, soft_wrap=True)
    if audit_warning:
        console.print("[yellow]Note:[/yellow] " + escape(str(audit_warning)), soft_wrap=True)
    _pin_notes(loader, ws)


def _own_unit(cfg_path: Path):
    """The harness's own autostart unit as the user manager reports it right now, or None: sensed ONCE per
    command (D-19), identity-checked (``start_serving.harness_unit``). A manager that cannot be asked, a unit
    that is not the harness's own and any failure of the look all say None: the command then selects."""
    try:
        return start_serving.harness_unit(start_serving.unit_runtime_state(), cfg_path)
    except Exception:  # noqa: BLE001 — a failed look is "no unit": the deferred path stands
        return None


def _will_launch(unit, cfg_path: Path) -> bool:
    """Whether this command swaps NOW (D-23, START-06): the harness's own unit is active, the machine has a
    record (read BEFORE the trust decision, so its first-start adoption can never enable a swap) and this is
    not an agent's command. Decided before the trust question, which it feeds as ``launching``; the target
    being withheld is judged after it."""
    return (
        unit is not None and getattr(unit, "active_state", "") == "active"
        and trust.recorded_machine(cfg_path) is not None
        and "LOCALHARNESS_COMMIT" not in os.environ)


def _incumbent(harness, config_dir: Path, unit, target) -> Optional[model_ops.ServingRecord]:
    """The record whose server runs now, other than ``target`` (R-14-33): the one whose launch identity
    verifies the pidfile's live process (``read_launch(expect=record_identity(spec), main_pid=...)``). A
    pidfile is a claim, so a planted one names nothing. None: nothing the harness launched runs (the unit's
    own restart replaces its process anyway)."""
    main_pid = getattr(unit, "main_pid", None)
    for record in model_ops.serving_records(harness):
        if record.prefix == target.prefix or record.spec is None:
            continue
        identity = lifecycle.record_identity(record.spec)
        if identity is None:
            continue
        launch = procfs.read_launch(config_dir, expect=identity, main_pid=main_pid)
        if launch is not None and launch.verified:
            return record
    return None


def _answers_now(harness, machine_trust: MachineTrust, record) -> bool:
    """True when ``record`` answers at its own address (the sensor `start` uses, loopback only). Records
    sharing an address are sensed together so the sensor can say whose it is. A failed look says False."""
    records = model_ops.serving_records(harness)
    sensed = (record,)
    if model_ops.shares_base_url(record, records):
        sensed = (record, *model_ops.launchable_loopback_others(harness, machine_trust, record))
    try:
        sense = asyncio.run(model_ops.sense_serving(harness, sensed))
    except Exception:  # noqa: BLE001 — an unreadable look is "not serving": the swap's own checks stand
        return False
    return any(r.prefix == record.prefix for r in sense.answering)


def _swap_now(
    harness, machine_trust: MachineTrust, serving: model_ops.Serving, *, config_dir: Path, unit
) -> None:
    """Switch what autostart serves AT ONCE (D-23, START-06): the swap runs through ``lifecycle.swap`` via
    ``start_serving.run_swap`` under the launch lock, with the harness's own unit state. It takes a
    ``MachineTrust`` and a ``Serving`` with no default: the target is a record the ladder resolved under a
    trust decision, and nothing else can reach a launch from here (structural, test_trust_structural_cli).

    A target already answering is not restarted (an attach). Otherwise the launch line is said BEFORE the
    wait, a failed swap exits 1 after the reporter's redacted line, and a success says what it did and
    lands as a swap (``persist_landed`` is the one writer, which also removes a stale ``pending_preset``)."""
    record = serving.record

    def say(line: str) -> None:
        console.print(line, markup=False, soft_wrap=True)

    if _answers_now(harness, machine_trust, record):
        # An earlier selection still stored (an agent's, naming a third record) is read before the landing
        # removes it, so the receipt can say that it is gone: the attach records no selection of its own.
        older = getattr(harness, "pending_preset", None)
        if older and str(older) == record.name:
            older = None  # the selection of this very record: "was cleared" would read as undoing the choice
        if not older:
            say(start_serving.ALREADY_SERVING_TARGET_LINE.format(name=visible_text(record.name)))
        # The name typed here is the shell's own intent, so the attach lands on the ``pending_preset`` rung:
        # an earlier selection must not outlive an explicit choice.
        with _landing_write(harness, record):
            asyncio.run(model_ops.persist_landed(
                harness, record, outcome="attach", rung=model_ops.RUNG_PENDING, config_dir=config_dir))
        if older:
            say(start_serving.ALREADY_SERVING_CLEARED_LINE.format(
                name=visible_text(record.name), older=visible_text(str(older))))
        return
    incumbent = _incumbent(harness, config_dir, unit, record)
    peer_name = record.peer.name if record.peer is not None else None
    undo_field = f"extra_endpoints.{peer_name}.lifecycle.speculative" if peer_name else "server.speculative"
    outside = lifecycle.outside_autostart(record.spec, unit)  # the unit exists, this record runs beside it

    def say_line(clause: str = "") -> None:
        line = start_serving.swap_launch_line(
            incumbent=visible_text(incumbent.name) if incumbent is not None else None,
            target=visible_text(record.name), inside=f"{clause}; " if clause else "")
        say(f"{line} {lifecycle.RUNS_OUTSIDE_AUTOSTART}" if outside else line)

    def on_plan(plan) -> None:
        say_line(start_serving.plan_clause(plan, undo_field=undo_field))

    async def say_lock_wait() -> None:
        err_console.print(lifecycle.LAUNCH_LOCK_WAIT_LINE, markup=False, soft_wrap=True)

    if not getattr(lifecycle.strategy_for(record.spec, unit=unit), "plans_launch", False):
        say_line()  # a strategy that plans nothing never calls `on_plan`: the line is said up front
    reporter = start_serving.ConsoleSwapReporter(err_console.print, harness, harness.provider.api_key)
    try:
        outcome = asyncio.run(start_serving.run_swap(
            harness, incumbent, serving, config_dir=config_dir, autosense=harness.autosense,
            reporter=reporter, redact=reporter.redact, on_plan=on_plan, on_wait=say_lock_wait, unit=unit))
    except Exception as exc:  # noqa: BLE001 — a launch that could not even begin (a lock, a bad record)
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(reporter.redact(f"failed to switch to {record.name!r}: {exc}")),
            soft_wrap=True)
        raise typer.Exit(1)
    if outcome.error is not None:
        raise typer.Exit(1)  # the reporter already said why, redacted
    receipt = start_serving.SERVING_LINE if outside else start_serving.SERVING_RESTARTED_LINE
    clause = start_serving.REPL_TALKS_CLAUSE if _repl_sessions(config_dir) else ""
    say(receipt.format(name=visible_text(record.name)) + clause)
    with _landing_write(harness, record):
        asyncio.run(model_ops.persist_landed(
            harness, record, outcome="swap", rung=serving.rung, config_dir=config_dir))


def _script_kind(record) -> bool:
    return record.spec is not None and record.spec.runtime in ("llamacpp", "vllm")


def _script_decision(record, harness) -> "tuple[bool, server.LaunchDecision]":
    """Whether `start` would hold a key for the record (judged from the record, the key file never opened) and the launch decision."""
    spec, peer = record.spec, record.peer
    key = harness.provider.api_key if peer is None else peer.api_key
    keyed = bool(server.required_key_kwargs(spec, key)) or (bool(spec.api_key_file) and spec.require_api_key is True)
    return keyed, server.launch_decision(spec, keyed, peer=None if peer is None else peer.name)


def _script_text(record, harness, cfg: Path, keyed: bool, decision, *, function: str | None = None) -> str:
    spec, peer = record.spec, record.peer
    plan = lifecycle.plan_launch(
        spec, lifecycle.sense_launch(spec, cfg, harness.autosense, unit=None), decision,
        autosense=harness.autosense, api_key_held=keyed)
    return tuning.render_script(
        plan, spec, name=record.name, version=resolved_version(), visible=visible_text, function=function,
        key_placeholder="provider.api_key" if peer is None else f"extra_endpoints.{peer.name}.api_key",
        home_shown="" if plan.config_ini_home is None else shown_path(plan.config_ini_home))


def _script_skip(record, view) -> str | None:
    if model_ops.record_withheld(view, record):
        return "not rendered — its record has changes you have not confirmed (run localharness start on a terminal)"
    if record.spec is None:
        return "not rendered — it launches nothing"
    if not _script_kind(record):
        return f"{record.spec.runtime} — not rendered"
    return None


def _script(name: str | None, config_dir: str | None, *, every: bool) -> None:
    """`localharness model --script [NAME]` and `--script --all`: print the launch as sh, decide no trust, prepare no home, write nothing."""
    cfg = resolve_config_dir(config_dir)
    view = model_ops.trust_as_recorded(cfg)
    try:
        harness = ConfigLoader(config_dir=cfg, local_config_dir=None, machine_withheld=view.withheld).load_harness()
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[bold red]Error:[/bold red] failed to load config: {escape(str(exc))}", soft_wrap=True)
        raise typer.Exit(2)
    if every:
        taken: list[str] = []
        blocks = []
        for record in model_ops.serving_records(harness):
            why = _script_skip(record, view)
            if why is None:
                keyed, decision = _script_decision(record, harness)
                why = decision.refusal
            if why is None:
                taken.append(tuning.script_function_name(record.name, taken))
                blocks.append(_script_text(record, harness, cfg, keyed, decision, function=taken[-1]))
            else:
                blocks.append(visible_text(SCRIPT_SKIP_LINE.format(name=record.name, why=why)))
        typer.echo("\n\n".join(blocks))
        raise typer.Exit(0)
    serving = model_ops.resolve_serving(harness, view, name, config_dir=cfg)
    if isinstance(serving, model_ops.ServingRefusal):
        _say_script_refusal(serving.line, serving.exit_code)
    record = serving.record
    if not _script_kind(record):
        nothing = name is None and not any(map(_script_kind, model_ops.serving_records(harness)))
        _say_script_refusal(SCRIPT_NOTHING_LINE if nothing else SCRIPT_KIND_LINE.format(
            name=record.name, kind=lifecycle.unit_kind(record.spec)))
    keyed, decision = _script_decision(record, harness)
    if decision.refusal is not None:
        _say_script_refusal(decision.refusal)
    typer.echo(_script_text(record, harness, cfg, keyed, decision))
    raise typer.Exit(0)


def _say_script_refusal(line: str, code: int = 2) -> NoReturn:
    err_console.print(visible_text(line), markup=False, soft_wrap=True)
    raise typer.Exit(code)


def _repl_sessions(config_dir: Path) -> bool:
    """Is a REPL session registered on this machine (its process alive)? The line then says it now talks
    to the new server. A failed look says False."""
    try:
        return bool(session_presence.live_sessions(config_dir))
    except Exception:  # noqa: BLE001 — presence is a courtesy
        return False


@contextlib.contextmanager
def _landing_write(harness, record):
    """Around the one ``persist_landed`` call: a failed write is said (redacted) and exits 2 (the server is
    up, but the choice is not recorded)."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        err_console.print(
            "[bold red]Error:[/bold red] "
            + escape(model_ops.redact_for_channel(
                f"failed to record {record.name!r} as serving: {exc}", harness, harness.provider.api_key)),
            soft_wrap=True)
        raise typer.Exit(2)


def _pin_notes(loader, ws) -> None:
    # Pin trap: a persisted default won't reach an agent whose yaml pins a concrete model.
    for aname, pin in model_ops.pinned_agents(loader._config_dir, local_config_dir=ws):
        console.print(
            "[yellow]Note:[/yellow] "
            + escape(f"agent {aname!r} pins model={pin!r} in its yaml")
            + " — this won't reach it on next start until that pin changes.",
            soft_wrap=True,
        )


def model(
    name: Optional[str] = typer.Argument(
        None,
        help="Model name/number to switch to (or a checkpoint path for a managed server). "
        "Omit to list available models.",
    ),
    config_dir: Optional[str] = typer.Option(
        None,
        "--config-dir",
        envvar="LOCALHARNESS_DIR",
        help="Config directory (default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, "
        "else ~/.localharness). Parity with `start`/`doctor`/`validate`; the persisted "
        "overlay is written HERE.",
    ),
    download: Optional[str] = typer.Option(
        None,
        "--download",
        help="Download a Hugging Face repo id into the local HF cache, then exit "
        "(does not change the configured default — follow up with `localharness model <path>`).",
    ),
    file: Optional[str] = typer.Option(
        None,
        "--file",
        help="With --download: fetch only this ONE filename from the repo (a specific GGUF "
        "quant, e.g. `Qwen3-32B-Q4_K_M.gguf`) instead of the whole repo snapshot. Use this for "
        "llama.cpp models — GGUF repos commonly ship several quantizations as sibling files.",
    ),
    autostart: Optional[str] = typer.Option(
        None,
        "--autostart",
        help="on | off | status — the autostart unit localharness-llm.service serves the last choice at login.",
    ),
    script: bool = typer.Option(
        False, "--script", help="Print the launch `start` runs as sh, for NAME or the preset the ladder resolves now."),
    every: bool = typer.Option(
        False, "--all", help="With --script: one shell function per preset."),
    setup: bool = typer.Option(
        False, "--setup", help="Choose which of the models found on this machine to set up as presets."),
) -> None:
    """List available models, or switch the persisted default with `localharness model <name>`."""
    if setup:
        if name is not None or script or every or download is not None or file is not None or autostart is not None:
            _refuse_autostart(SETUP_ALONE_LINE)
        _setup(config_dir, person=local_operator(False))
        return
    if script or every:
        if not script or autostart is not None or download is not None or file is not None or (every and name is not None):
            _refuse_autostart(SCRIPT_ALONE_LINE)
        _script(name, config_dir, every=every)
    if autostart is not None:
        _autostart(
            autostart, config_dir,
            with_other_arguments=name is not None or download is not None or file is not None)
        return
    if download is not None:
        _download(download, file, config_dir)
        return
    if file is not None:
        err_console.print("[bold red]Error:[/bold red] --file requires --download <repo_id>.")
        raise typer.Exit(2)

    # config_dir=None routes through the resolver's env/default chain (#35); an explicit flag or
    # LOCALHARNESS_DIR isolates the overlay to that dir.
    # local_config_dir carries the workspace layer, or None.
    from localharness.cli.workspace import resolve_workspace_layer
    # D-09 / D-10: a switch decides machine trust exactly as `start` does, before anything loads, and
    # loads with what that decision withheld. A person at a terminal is asked; a pipe, a script or an
    # agent's command (LOCALHARNESS_COMMIT) is not, so a piped `y` adopts nothing and an EOF records
    # no No. `will_launch` says whether this switch swaps now (the unit is active, a machine record
    # exists, no agent marker; D-23). The list and `--download` decide nothing (R-13-16).
    machine_trust = MachineTrust()
    unit = None
    will_launch = False
    if name is not None and name.strip():
        cfg_path = resolve_config_dir(config_dir)
        # D-23: the unit is sensed once, before the decision, because `launching` is the decision's input.
        # The question is gated on the terminal (`ask`), the action is not (`will_launch`).
        unit = _own_unit(cfg_path)
        will_launch = _will_launch(unit, cfg_path)
        machine_trust = decide_machine_trust(cfg_path, ask=local_operator(False), launching=will_launch)
        if machine_trust.line:
            err_console.print(machine_trust.line, markup=False, soft_wrap=True)
    loader = ConfigLoader(
        config_dir=config_dir, local_config_dir=resolve_workspace_layer(config_dir),
        machine_withheld=machine_trust.withheld)
    # v0.13: two values, two jobs. `loader._config_dir` is machine-wide truth (the overlay target,
    # one GPU daemon); `_ws` is the workspace layer, or None. The audit record follows the WORK
    # (MEMS-04) and falls back to the session's config dir — NOT to None, which would re-resolve
    # through the env chain and let `--config-dir D` leak its audit log out of D.
    # Bound to NAMED LOCALS on purpose, not written inline at the call sites: the structural guard
    # in tests/unit/test_provider_carveout_workspace.py scans this file for a literal binding of
    # the workspace attribute to a keyword ending in `config_dir`, and an inline workspace
    # expression at either call site below would tail-match it. (This very comment cannot spell
    # the forbidden literal either — that is how strict a plain substring scan is.)
    _ws = loader._local_dir
    _audit_dir = _ws or loader._config_dir
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] failed to load config: {exc}")
        raise typer.Exit(2)

    provider = harness.provider
    from localharness.provider.server import required_key_kwargs
    try:
        # the provider's key only when the launched server is set to require it (default off)
        live, reachable = model_ops.list_live_models(
            provider.base_url, **required_key_kwargs(harness.server, provider.api_key))
    except model_ops.MalformedModelListError as exc:
        # #38: reached but the reply isn't a model list — its OWN message, not "Is it running?".
        err_console.print(
            f"[bold red]Error:[/bold red] the server at {provider.base_url} responded, but the "
            f"response wasn't understood — is base_url pointing at an OpenAI-compatible API? ({exc})"
        )
        raise typer.Exit(2)
    downloaded: list[str] = []
    if harness.server is not None:
        from localharness.provider import server as managed_server
        downloaded = [m for m in managed_server.list_cached_models() if m not in live]
    choices = live + downloaded
    current = provider.default_model

    # --- List --- #
    if name is None:
        if not reachable and not choices:
            err_console.print(
                f"[bold red]Error:[/bold red] could not reach the model server at "
                f"{provider.base_url}, and no downloaded models were found. "
                f"Is it running? Try `localharness doctor`."
            )
            raise typer.Exit(2)
        if not choices:
            console.print(
                f"No models served at {provider.base_url} or in the local download cache."
            )
            return
        console.print("Models:")
        for i, m in enumerate(live, start=1):
            mark = "  [active]" if m == current else ""
            console.print(f"  {i}. {m}  (serving){mark}", markup=False)
        for i, m in enumerate(downloaded, start=len(live) + 1):
            console.print(
                f"  {i}. {m}  (downloaded — `localharness start` will launch it)", markup=False
            )
        console.print("Switch with `localharness model <name|number>`.")
        # #50: a reachable server whose served/downloaded set omits the configured default
        # otherwise shows NO [active] marker and no other signal — the moment of highest risk
        # (checking in on an unverified degrade-persist) gets none. State it plainly.
        if reachable and current not in choices:
            console.print(
                "[yellow]⚠[/yellow]  " + escape(f"configured default {current!r}")
                + " is not among the served models — switch with "
                "`localharness model <name|number>` or check the server.",
                soft_wrap=True,
            )
        return

    # --- Switch: resolve the target --- #
    # #39: reject an empty/whitespace name FIRST — before any resolution. Otherwise "" falls
    # through isdigit/exact/checkpoint (note Path("").expanduser().exists() == cwd) into the
    # unreachable-degrade branch and persists "" as the default.
    if not name.strip():
        err_console.print("[bold red]Error:[/bold red] model name cannot be empty.")
        raise typer.Exit(2)

    # D-03 / R-13-17: a number keeps its list meaning; any other name goes through the ladder's
    # preset resolution first. A `Serving` is a preset selection; a withheld or ambiguous name is
    # refused with exit 2; an unknown name falls through to today's served / downloaded / checkpoint
    # resolution and its messages, unchanged.
    if not (name.isdigit() and 1 <= int(name) <= len(choices)):
        serving = model_ops.resolve_serving(
            harness, machine_trust, name, config_dir=loader._config_dir)
        if isinstance(serving, model_ops.ServingRefusal) and serving.kind != "unknown":
            # The withheld line was already printed once as `machine_trust.line`; say a refusal
            # that is not that line (an ambiguous name, or a withheld record start composed no line for).
            if serving.line != machine_trust.line:
                err_console.print(
                    "[bold red]Error:[/bold red] " + escape(serving.line), soft_wrap=True)
            raise typer.Exit(serving.exit_code)
        if isinstance(serving, model_ops.Serving):
            if (will_launch and serving.record.spec is not None
                    and not model_ops.record_withheld(machine_trust, serving.record)):
                _swap_now(harness, machine_trust, serving, config_dir=loader._config_dir, unit=unit)
                return
            _select_preset(
                harness, serving.record, current, loader, _audit_dir, _ws, machine_trust, unit=unit)
            return

    target: Optional[str] = None
    if name.isdigit() and 1 <= int(name) <= len(choices):
        target = choices[int(name) - 1]
    elif name in choices:
        target = name
    elif harness.server is not None and Path(name).expanduser().exists():
        target = name

    if target is None:
        if reachable:
            # Reached the runtime and the target isn't served/downloaded → fail loud, name options.
            avail = ", ".join(choices) if choices else "(none served or downloaded)"
            if name.isdigit():
                # A number that didn't resolve is out of range — say how many are listed.
                err_console.print(
                    "[bold red]Error:[/bold red] " + escape(f"{name} is out of range")
                    + f" — {len(choices)} model(s) listed. Run `localharness model` to see them.",
                    soft_wrap=True,
                )
                raise typer.Exit(2)
            # Fat-finger hint: a case-insensitive exact match is almost certainly the intent.
            ci = next((m for m in choices if m.lower() == name.lower()), None)
            if ci is not None:
                err_console.print(
                    "[bold red]Error:[/bold red] "
                    + escape(f"unknown model {name!r} — did you mean {ci!r}? Available: {avail}."),
                    soft_wrap=True,
                )
            else:
                err_console.print(
                    "[bold red]Error:[/bold red] "
                    + escape(f"unknown model {name!r}. Available: {avail}."),
                    soft_wrap=True,
                )
            raise typer.Exit(2)
        # Runtime unreachable → can't verify. Degrade with an explicit disclosure (mirrors the
        # TokenCounter `.approximate` convention: proceed, but label it clearly).
        target = name
        console.print(
            "[yellow]⚠[/yellow]  "
            + escape(f"Could not reach {provider.base_url} to verify {name!r} is served")
            + " — persisting it as the default UNVERIFIED. Run `localharness doctor` once the "
            "server is up.",
            soft_wrap=True,
        )

    if target == current:
        console.print(escape(f"{target} is already the default."), soft_wrap=True)
        return

    try:
        audit_warning = asyncio.run(
            model_ops.persist_default_model(
                harness, target, config_dir=loader._config_dir, audit_base_dir=_audit_dir,
                local_operator=local_operator(False),
            )
        )
    except Exception as exc:
        err_console.print(
            "[bold red]Error:[/bold red] " + escape(f"failed to persist {target!r}: {exc}"),
            soft_wrap=True,
        )
        raise typer.Exit(2)

    # A model name is data — served ids and checkpoint PATHS both arrive here, and `repr` does
    # not neutralize markup: `'[old]/gguf'` still parses. escape() it, markup outside.
    console.print(
        "[green]Default model set to[/green] " + escape(str(target))
        + ". `localharness start` will use it.",
        soft_wrap=True,
    )
    # #37: the switch is durably persisted; a post-write audit-emit failure is a secondary note.
    if audit_warning:
        console.print("[yellow]Note:[/yellow] " + escape(str(audit_warning)), soft_wrap=True)

    _pin_notes(loader, _ws)
