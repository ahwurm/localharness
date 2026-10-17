"""Provider lifecycle layer — a launch-STRATEGY abstraction, orthogonal to `provider_type`.

`provider_type` (vllm/ollama/llamacpp/lmstudio) is WHICH wire protocol / token-counting a
backend uses. Lifecycle is the orthogonal axis: HOW the harness brings a backend up, checks
it, and takes it down. A `LifecycleStrategy` unifies that so the heavy-to-heavy cross-framework
`/model` swap (stop the incumbent → confirm the GPU is free → launch the target) becomes real
instead of the manual box-dance.

Phases A–B (0.11) are ADDITIVE scaffolding: `ManagedVllmStrategy` (Phase A) and
`SpawnedProcessStrategy` (Phase B, llama.cpp) exist, each a thin, ZERO-BEHAVIOR-CHANGE wrapper
over `provider/server.py` (all launch paths already flow through those functions). Existing
callers — repl `/model` swap, init guided setup, `start`
autostart, `model` list — keep calling `provider/server.py` DIRECTLY; rewiring them through a
strategy is Phase C (a strategy with a bound import would silently break the module-attribute
monkeypatch the harness's tests rely on). The one behavior change shipped now is the
liveness-by-name fix, and it lives in `server.py` (`docker_container_running`) so both this
strategy AND the `start` autostart pre-check use it without going through the strategy.

`DaemonStrategy` (Ollama, Phase D) and `LmsStrategy` (LM Studio, Phase D5) implement the same Protocol.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import errno
import functools
import json
import logging
import math
import os
import time
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

try:
    import fcntl
except ImportError:  # non-POSIX: the launch lock degrades to an in-process asyncio.Lock
    fcntl = None  # type: ignore[assignment]

from localharness.config import redact as _redact
from localharness.config.models import ManagedServerConfig
from localharness.core.private_files import atomic_write_text, open_regular
from localharness.provider import autosense as _auto
from localharness.provider import procfs, server, speed_stats, tuning, units
from localharness.provider.procfs import LaunchIdentity
from localharness.provider.runtimes import RUNTIMES
from localharness.provider.tuning import Tuned

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # annotations only: autosense imports nothing from here, but the dataclass stays import-light
    from localharness.config.models import AutosenseConfig
    from localharness.provider.autosense import WindowReport
    from localharness.provider.procfs import ProcRecord


async def _kill_and_reap(proc: asyncio.subprocess.Process) -> None:
    """Kill a still-running child and reap it — on EVERY exit path (#153).

    The timeout paths already killed; CancelledError (a BaseException — a /model swap abandoned
    by a Ctrl+C) did not, leaving an `ollama pull` / `lms` subprocess running. kill() is a
    synchronous signal; the reap needs an await, and an await inside a `finally` unwinding a
    cancellation can itself be interrupted — hence the shield, with its CancelledError
    suppressed so the ORIGINAL cancellation still propagates."""
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:  # raced us to exit — nothing left to signal
        return
    with contextlib.suppress(asyncio.CancelledError, ProcessLookupError):
        await asyncio.shield(proc.wait())


@dataclass
class LiveEndpoint:
    """A serving endpoint a strategy brought up: its `base_url`, the model ids it actually
    serves (`served_models`), and an opaque stop `handle` the SAME strategy understands
    (docker: the container name; binary/spawned: the pid). The handle carries a strategy's
    teardown target forward to later phases; Phase A populates it but stops via the config.

    `note` is the launch decision's one-line note when it withheld a widening (a key file it could
    not hand to the server), else None; the caller shows it instead of reporting nothing.

    `window` is the launch's window report (served / trained / fits / asked, D-20) for the caller to
    print, None when no fit was asked for or the runtime has none (vLLM, a daemon); `dropped_env` is
    the environment names scrubbed from the launched server's environment (D-25); `tuned` is the
    `Tuned` of the plan that served (what the launch derived), None for an untuned launch (D-21)."""
    base_url: str
    served_models: list[str]
    handle: Any
    note: str | None = None
    window: "WindowReport | None" = None
    dropped_env: tuple[str, ...] = ()
    tuned: Tuned | None = None


@dataclass
class Liveness:
    """Is the endpoint serving RIGHT NOW? `detail` carries a human-readable reason (container
    name, pid, or why it is considered down) for diagnostics/logging. Endpoint/name-based —
    never the client pid (the #99/#100 orphan-client-pid bug class)."""
    alive: bool
    detail: str | None = None


@runtime_checkable
class LifecycleStrategy(Protocol):
    """HOW a backend is launched / stopped / checked — orthogonal to `provider_type`.

    activate: ensure the target is serving; return its `LiveEndpoint` (base_url + served ids
      + a stop handle). stop: verified + idempotent teardown. liveness: is it serving NOW,
      endpoint/name-based (never the client pid).

    `api_key` is the key the launched server requires, or None. A strategy never decides that:
    callers pass one only through `server.required_key_kwargs` (machine-level
    `require_api_key`, default off), so a keyless activation calls everything as it always did.

    `peer` is the `EndpointRef.name` when `spec` is a peer's lifecycle, None for the primary; it is
    used only for the wording of a launch refusal. A strategy whose runtime has no binary to refuse
    (a PATH CLI) accepts it and ignores it.

    `autosense` is the machine's window-autosense ratios, or None. The fit and `--cache-ram` are
    computed inside `activate` because every launch path goes through it (five paths, one place);
    with None nothing is sized and `activate` behaves exactly as it did before the autosense.

    `on_plan` is the caller's printer for the launch line: a planning strategy (`plans_launch = True`)
    calls it once with each `LaunchPlan` right before that plan's launch (the first plan, then the
    retry plan), and with None before an untuned launch; it must not raise into the launch. A strategy
    that plans nothing accepts it and never calls it.
    """

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint: ...

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None: ...

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness: ...

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        """What a launch of `spec` by THIS strategy looks like in `/proc`, as plain data
        (`procfs.LaunchIdentity`), or None when the strategy owns no process to recognise (LM Studio's
        daemon). The strategy, not a caller, knows its own argv, executable and port source (D-25)."""
        ...

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir a reconnect poll for a server THIS strategy launched should hand to
        `wait_ready`: with it a dead process is known at once from its pidfile and its log tail is
        shown; None means a pure poll (nothing records a pid to read a death from). The strategy, not
        the caller, knows its own paths (D-20)."""
        ...


class ManagedVllmStrategy:
    """Lifecycle for the harness-managed vLLM server (init guided setup / reboot autostart /
    REPL `/model` swap). A thin, ZERO-BEHAVIOR-CHANGE wrapper over `provider/server.py`, which
    both launch modes (`docker`, `binary`) already flow through:

    - activate = `launch_decision` → `serve_command` → `start_server` → `wait_ready` (returns served
                 ids). The decision is asked once, here — activate is the one enforcement point:
                 its refusal is raised as `LaunchRefused` before anything starts, and its
                 loopback / key-file judgment and note ride the command and the `LiveEndpoint`. A
                 required `api_key` reaches the launch and the probe (the launch environment, never
                 argv), keyword arguments passed only when there is one.
    - stop     = `stop_server(config_dir, launch=spec.launch)` — the #100 VERIFIED stop
                 (docker: stop → poll name-free → `rm -f` fallback → raise if still stuck).
    - liveness = NAME-based for docker (`docker_container_running` via `docker inspect`), the
                 fix; pid-based for binary (`server_pid`, correct there).

    With `autosense` set the launch is a `LaunchPlan` (`sense_launch` → `plan_launch`) and the argv
    started is `plan.argv`; `plans_launch` tells `start` the `on_plan` callback prints the launch line.
    A planned launch that derived its parsers and died before readiness is stopped (verified) and
    relaunched once as `plan.retry()`, without the derived parsers (D-37); a second death is raised
    with its log tail and the record fields that pin the parsers off.
    """

    plans_launch = True

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint:
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        plan: LaunchPlan | None = None
        if autosense is not None:
            # OFF the event loop: sensing opens the checkpoint's config.json.
            sensed = await asyncio.to_thread(sense_launch, spec, config_dir, autosense)
            plan = plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed, floor=floor)
            cmd = list(plan.argv)
        else:
            cmd = server.serve_command(
                spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok,
                context=None, cache_ram=None)
        _say_plan(on_plan, plan)

        offline = {"hf_offline": True} if plan is not None and plan.offline else {}

        def launch(argv: list[str]) -> tuple[int, list[str]]:
            names: list[str] = []
            started = server.start_server(
                config_dir, argv, runtime=spec.runtime, record_env=getattr(spec, "env", None),
                dropped_out=names, port=spec.port, **offline, **server.key_kwargs(api_key))
            return started, names

        async def ready() -> list[str]:
            return await server.wait_ready(
                base_url,
                config_dir=config_dir,
                timeout_seconds=timeout_seconds,
                on_poll=on_poll,
                **server.key_kwargs(api_key),
            )

        pid, dropped = launch(cmd)
        try:
            served = await ready()
        except RuntimeError as died:
            # D-37: a launch that dies before readiness with a DERIVED parser pair is retried once
            # without them (`LaunchPlan.retry`); no plan, no derived parser or a TimeoutError re-raise.
            retry = plan.retry() if plan is not None else None
            if retry is None:
                raise
            try:
                await asyncio.to_thread(
                    server.stop_server, config_dir, launch=spec.launch, expect=self.launch_identity(spec))
            except RuntimeError:
                raise died from None  # a server that survived its stop: never relaunch beside it
            _say_plan(on_plan, retry)
            pid, dropped = launch(list(retry.argv))
            try:
                served = await ready()
            except RuntimeError as again:
                with contextlib.suppress(RuntimeError):
                    await asyncio.to_thread(
                        server.stop_server, config_dir, launch=spec.launch,
                        expect=self.launch_identity(spec))
                where = f"extra_endpoints.{peer}.lifecycle" if peer else "server"
                raise RuntimeError(f"{again}\n{parser_pin_hint(retry, record_prefix=where)}") from None
            plan = retry
        handle = server.DOCKER_CONTAINER_NAME if spec.launch == "docker" else pid
        report = plan.report if plan is not None else None
        if report is not None:
            from localharness.agent.context import probe_window

            probe = await asyncio.to_thread(
                probe_window, base_url, served[0] if served else None, spec.runtime, api_key=api_key)
            report = dataclasses.replace(
                report, served=probe.served, trained=report.trained or probe.trained, verified=probe.served is not None)
        live = LiveEndpoint(base_url=base_url, served_models=served, handle=handle, note=decision.note,
                            window=report, dropped_env=tuple(dropped), tuned=plan.tuned if plan else None)
        confirm_launch(config_dir, record_key(spec), plan=plan, live=live, spec=spec)
        return live

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir `activate` polls with, so a reconnect poll reads the same pidfile and log."""
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # OFF the event loop: the #100 verified docker stop can block ~90s worst-case (drain +
        # name-free poll + rm -f), and a synchronous call inside this `async def` would freeze
        # the whole loop — heartbeats, the Discord adapter, idle consolidation. `server.stop_server`
        # is looked up at CALL time (module-attribute), so the harness's monkeypatches still fire.
        await asyncio.to_thread(
            server.stop_server, config_dir, launch=spec.launch, expect=self.launch_identity(spec))

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        if spec.launch == "docker":
            name = server.DOCKER_CONTAINER_NAME
            return Liveness(alive=server.docker_container_running(name), detail=f"container {name}")
        pid = server.server_pid(config_dir)
        return Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        """docker: the `docker run` client under the container's fixed name; binary: the `vllm`
        entry point (or the interpreter running it) with `serve` on its command line and the record's
        port (D-25)."""
        if spec.launch == "docker":
            return LaunchIdentity(
                procfs.DOCKER_EXES, None, ("--name", server.DOCKER_CONTAINER_NAME), None)
        return LaunchIdentity(procfs.VLLM_EXES, spec.port, ("serve",), None)


def parser_pin_hint(plan: "LaunchPlan", *, record_prefix: str = "server") -> str:
    """The give-up line after a parser retry also died (D-37): it was retried once without the derived
    parsers, and the record fields that pin them off. `record_prefix` is where the record lives
    (`server`, or a peer's `extra_endpoints.<name>.lifecycle`)."""
    fields = plan.dropped_parser_fields or ("reasoning_parser", "tool_call_parser")
    pins = " / ".join(f"{record_prefix}.{f}: off" for f in fields)
    return f"Retried once without the derived parsers; {pins} keeps them pinned off."


def record_identity(spec: ManagedServerConfig) -> LaunchIdentity | None:
    """`strategy_for(spec).launch_identity(spec)` with the RECORD's own tokens added to
    `argv_contains`, so two records on one binary and port can be told apart (R-13-07): llama.cpp
    `-a <served_name>` when the record has one, else `-m <model>` as the launch spells it (`~`
    expanded); vLLM binary the model argument (`serve <model>`); vLLM docker the model it serves and the port it
    publishes; Ollama and LM Studio add nothing."""
    # an identity does not depend on who runs the launch: the unit wraps the strategy and states its identity
    identity = strategy_for(spec, unit=None).launch_identity(spec)
    if identity is None:
        return None
    tokens: tuple[str, ...] = ()
    row = RUNTIMES[spec.runtime]
    if row.checkpoint and spec.launch == "docker":
        # Every docker record runs under the one container name, so the record is told apart by what
        # the `docker run` line says of it: the model it mounts and serves, and the port it publishes
        # (127.0.0.1:<port>:8000 when bound to loopback, <port>:8000 otherwise).
        entry = spec.entry_for(spec.model)
        tokens = ("/models/serving", entry.name) if entry is not None else (str(spec.model),)
        return dataclasses.replace(
            identity, argv_contains=identity.argv_contains + tokens,
            argv_one_of=((f"127.0.0.1:{spec.port}:8000", f"{spec.port}:8000"),))
    if row.gguf:
        if spec.served_name is not None:
            tokens = ("-a", spec.served_name)
        else:
            tokens = ("-m", os.path.expanduser(str(spec.model)))
    elif row.checkpoint:
        entry = spec.entry_for(spec.model)
        tokens = (str(Path(entry.path).expanduser()) if entry is not None else str(spec.model),)
    if not tokens:
        return identity
    return dataclasses.replace(identity, argv_contains=identity.argv_contains + tokens)


def record_key(spec: ManagedServerConfig) -> str:
    """The ledger entry a record's retry lives on: `speed_key(runtime, served id)` (R-09a), the same
    key its speed samples use. A boot marker's `record` must equal it."""
    return speed_stats.speed_key(spec.runtime, spec.served_name or str(spec.model))


def first_build(spec: ManagedServerConfig, config_dir: Path | None) -> bool:
    """A checkpoint record no launch has answered for on this box (no config directory, no ledger: not one)."""
    return (config_dir is not None and RUNTIMES[spec.runtime].checkpoint
            and not speed_stats.served_here(speed_stats.default_speed_stats_path(config_dir), record_key(spec)))


FLOOR_FILE = "server-floor.json"
_FLOOR_READ_CAP = 64 * 1024
_FLOOR_MIN = 1_024


def record_floor(config_dir: Path, root: int, roles: Mapping[str, int], *, now: datetime | None = None) -> None:
    """Cache the full window floor for the next start's first launch; a start never fails on it."""
    at = (now or datetime.now(timezone.utc)).isoformat()
    try:
        atomic_write_text(Path(config_dir) / FLOOR_FILE, json.dumps({"root": root, "roles": dict(roles), "at": at}))
    except OSError:
        log.debug("could not write %s", FLOOR_FILE, exc_info=True)


def cached_floor(config_dir: Path) -> int | None:
    """The cached root floor, at least 1,024; None when absent or not a plain positive int."""
    try:
        with os.fdopen(open_regular(Path(config_dir) / FLOOR_FILE), "rb") as fh:
            raw = fh.read(_FLOOR_READ_CAP + 1)
        if len(raw) > _FLOOR_READ_CAP:
            return None
        root = json.loads(raw).get("root")
    except (OSError, ValueError, AttributeError):
        return None
    if not _plain_int(root) or root <= 0:
        return None
    return max(_FLOOR_MIN, root)


def raised_floor(live: int | None, cached: int | None) -> int | None:
    """The larger known figure: the cache only raises the live floor."""
    known = [v for v in (live, cached) if v is not None]
    return max(known) if known else None


@dataclass(frozen=True)
class Sensed:
    """Everything one launch reads from the machine, gathered once by `sense_launch` and handed to the
    pure `plan_launch` (spec 20 §6.0, D-14): the model header (a typed, bounded dict), the GPU
    residency of the weights (`offload`), the free memory, the weights' size, `/proc/meminfo`, a
    vLLM checkpoint's `config.json` and the ledger's `remembered` retry record. `marker` is the boot
    marker `sense_launch` read when it was given the unit's state, and `prior_death` the verdict
    `boot_prior_death` drew from it; without a unit both stay at their defaults. Every field is None /
    empty when it could not be known."""

    runtime: str
    config_dir: Path
    header: Mapping[str, Any] = field(default_factory=dict)
    offload: str = "unknown"
    memory: "_auto.FreeMemory | None" = None
    weights: int | None = None
    meminfo: Mapping[str, int] | None = None
    model_config: Mapping[str, Any] | None = None
    remembered: "speed_stats.RetryRecord | None" = None
    marker: "units.Marker | None" = None
    prior_death: bool = False
    split: "_auto.TensorSplit | None" = None
    mmproj: int = 0
    snapshot: str | None = None
    snapshot_refusal: str | None = None
    first_build: bool = False


def sense_launch(
    spec: ManagedServerConfig, config_dir: Path, autosense: "AutosenseConfig", *,
    unit: "units.UnitRuntimeState | None" = None, now: datetime | None = None,
    memory: "_auto.FreeMemory | None" = _auto._UNSET,
) -> Sensed:
    """Every read a launch needs, once: the GGUF header, free memory, the weights, `/proc/meminfo` and
    the ledger's remembered retry for llama.cpp; a checkpoint's `config.json` for vLLM; nothing for any
    other runtime. Writes nothing and never raises on bad input (the readers return None / `{}`; the
    ledger read is bounded by `speed_stats._load`). Run it off the event loop: it shells out to
    nvidia-smi.

    `unit` is the autostart unit's runtime state (spec 20 §9.3, D-17, AUTO-08). Given one, a llama.cpp or
    vLLM launch also reads the boot marker (`units.read_marker`, read only: a marker for another record,
    another binary or past the window is ignored here and replaced by the next one `serve` writes, never
    removed by sensing) and sets `marker` and `prior_death` from `boot_prior_death`, with the unit's own
    `TimeoutStartSec` as the window. `now` is the clock for that age test (UTC now when None). `memory`
    (a `FreeMemory` or None) replaces the probe of free memory."""
    row = RUNTIMES[spec.runtime]
    if row.gguf:
        fi = _auto.compute_fit(spec, autosense, meminfo=True, memory=memory)  # its `fits` is priced without the derived types: unused
        remembered = speed_stats.last_retry(speed_stats.default_speed_stats_path(config_dir), record_key(spec))
        sensed = Sensed(spec.runtime, config_dir, fi.header, fi.offload, fi.memory, fi.weights, fi.meminfo,
                        remembered=remembered, split=fi.split, mmproj=fi.mmproj)
    elif row.checkpoint:
        snapshot, refusal = _verified_snapshot(spec)
        located = spec.model_copy(update={"model": snapshot}) if snapshot is not None else spec
        config = tuning.read_model_config(located)
        free = weights = None
        path = tuning._checkpoint_config_path(located) if config is not None else None
        if path is not None:
            free = _auto.free_memory(autosense) if memory is _auto._UNSET else memory
            weights = _auto.checkpoint_bytes(os.path.dirname(path))
        sensed = Sensed(spec.runtime, config_dir, model_config=config, memory=free, weights=weights,
                        snapshot=snapshot, snapshot_refusal=refusal, first_build=first_build(spec, config_dir))
    else:
        return Sensed(spec.runtime, config_dir)
    if unit is None:
        return sensed
    marker = units.read_marker(config_dir)
    prior = boot_prior_death(
        marker, unit, record_key=record_key(spec), binary=str(spec.binary),
        now=now or datetime.now(timezone.utc),
        max_age_seconds=getattr(unit, "timeout_start_seconds", units.TIMEOUT_START_SECONDS))
    return dataclasses.replace(sensed, marker=marker, prior_death=prior)


def _verified_snapshot(spec: ManagedServerConfig) -> tuple[str | None, str | None]:
    """``(snapshot, refusal)`` for a vLLM binary launch of a cached repo id, re-walked now; (None, None) when it is not one."""
    from localharness.provider import model_library

    if spec.launch != "binary" or spec.entry_for(spec.model) is not None:
        return None, None
    model = os.path.expanduser(spec.model)
    if os.path.isabs(model):
        return None, None
    found = model_library.snapshot_of(spec.model)
    if found is None:
        return None, None
    if found.refusal is not None or found.kind != "vllm":
        why = found.refusal or "its snapshot is not a vLLM checkpoint"
        return None, f"{_auto._label(spec)} cannot run here: {why} — the harness launches only a verified checkpoint."
    return str(found.snapshot), None


def boot_prior_death(
    marker: "units.Marker | None", unit: "units.UnitRuntimeState | None", *, record_key: str,
    binary: str, now: datetime, max_age_seconds: float,
) -> bool:
    """Does the boot marker say the last full shape died (spec 20 §9.3, D-17)? True only when a marker
    exists for this record and binary, its cause is `death` or it was written with `shape: full`, it is
    younger than `max_age_seconds` (the unit's load window), and the unit's `NRestarts` is at least 1.
    The marker's `invocation` plays no part: a `Restart=` is a new invocation. An unparsable or naive
    `at`, a future one, a missing marker or a missing unit is False. Pure."""
    if marker is None or unit is None:
        return False
    if marker.record != record_key or marker.binary != binary:
        return False
    if marker.cause != "death" and marker.shape != "full":
        return False
    if unit.n_restarts < 1:
        return False
    try:
        at = datetime.fromisoformat(marker.at)
        if at.tzinfo is None or now.tzinfo is None:  # IN-02: never a naive comparison, whatever `now` is
            return False
        age = (now - at).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return False
    return 0 <= age < max_age_seconds


@dataclass(frozen=True)
class LaunchPlan:
    """The launch a strategy runs, as a value: the exact `argv`, the keywords that built it
    (`serve_kwargs`), the NAMES of the environment variables the child gets that the harness sets
    (`env_names` — never a value, never the key), the config.ini home a llama.cpp launch prepares
    (`config_ini_home`, a path only), the window `report`, what the launch `tuned`, and how this plan
    came about: `shape` is "full" or "retry", `cause` why a retry was planned ("death", "remembered" or
    "boot"), `dropped` the derived names the retry gave up, `full_fit` the fit `-c` a retry halves from,
    `remembered_at` the `at` of the ledger record a "remembered" retry follows.

    `inputs` is what `retry()` rebuilds from; it is neither compared nor shown, so two plans of the
    same `Sensed` are equal and a repr carries nothing but the plan."""

    argv: tuple[str, ...]
    serve_kwargs: Mapping[str, Any]
    env_names: tuple[str, ...]
    config_ini_home: Path | None
    report: "_auto.WindowReport | None"
    tuned: Tuned
    shape: str = "full"
    cause: str | None = None
    dropped: tuple[str, ...] = ()
    full_fit: int | None = None
    remembered_at: str | None = None
    inputs: tuple | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "serve_kwargs", MappingProxyType(dict(self.serve_kwargs)))

    @property
    def offline(self) -> bool:
        return self.serve_kwargs.get("snapshot") is not None

    def boot_retry(self) -> "LaunchPlan | None":
        """The retry shape of a boot that died (spec 20 §9.3, D-17): the derived spec type dropped and the
        fit `-c` halved, each only when this full plan has it; the cause is "boot". None for a plan already
        in the retry shape and for one with nothing to drop or halve (a KV-only launch, a vLLM launch:
        nothing is dropped at boot there). `plan_launch` returns this for a prior death, and a caller asks
        it whether the unit's own retry would change anything. Pure."""
        return self._shaped("boot", drop_spec=True)

    def retry(self) -> "LaunchPlan | None":
        """The one relaunch after a death (D-16, D-35, D-37): the derived spec type dropped and the fit
        `-c` halved (`max(floor or 1, full_fit // 2)`, not at all when that is below the floor), each only when the
        plan has it; for vLLM, both derived parsers dropped (and with them `--enable-auto-tool-choice`, unless the record's own tool-call
        parser keeps it), `dropped` then naming "parsers". None for a plan already in the retry shape
        and for one with none of these (a KV-only launch, a vLLM launch that derived no parser). A
        record or `extra_args` value is never touched. Pure: the same plan gives an equal retry."""
        return self._shaped("death", drop_spec=True, drop_parsers=True)

    @property
    def dropped_parser_fields(self) -> tuple[str, ...]:
        """The record fields (`reasoning_parser`, `tool_call_parser`) whose derived value a parser retry
        dropped, re-derived from the plan's own inputs (pure); empty when no parser was dropped."""
        if "parsers" not in self.dropped or self.inputs is None:
            return ()
        spec, sensed, _decision, autosense, _held, _floor = self.inputs
        full = tuning.derive(spec, sensed.header, autosense, model_config=sensed.model_config)
        return tuple(f for f in ("reasoning_parser", "tool_call_parser") if getattr(full, f) is not None)

    def _shaped(self, cause: str, *, drop_spec: bool, drop_parsers: bool = False,
                remembered_at: str | None = None) -> "LaunchPlan | None":
        """The retry shape of this full plan, the one builder behind `retry()` and the remembered and
        boot branches of `plan_launch` (D-17): the derived spec type dropped when `drop_spec` and the
        plan has one, the derived vLLM parsers dropped when `drop_parsers` and the plan has any (D-37),
        the fit `-c` halved from this plan's own fit when its mode is "fit". None when none applies or
        the plan is not a full one."""
        if self.shape != "full" or self.inputs is None:
            return None
        spec, sensed, decision, autosense, held, floor = self.inputs
        kw = dict(self.serve_kwargs)
        report, tuned, dropped = self.report, self.tuned, ()
        drops_spec = drop_spec and tuned.spec_type is not None
        drops_parsers = drop_parsers and (tuned.reasoning_parser is not None or tuned.tool_call_parser is not None)
        halves = (report is not None and report.mode == "fit" and self.full_fit is not None
                  and (floor is None or self.full_fit // 2 >= floor))
        if not drops_spec and not halves and not drops_parsers:
            return None
        if drops_parsers:
            kw.pop("reasoning_parser", None)
            kw.pop("tool_call_parser", None)
            reasons = {k: v for k, v in tuned.reasons.items() if k not in ("reasoning_parser", "tool_call_parser")}
            tuned = dataclasses.replace(tuned, reasoning_parser=None, tool_call_parser=None, reasons=reasons)
            dropped = ("parsers",)
        if drops_spec:
            kw.pop("spec_type", None)
            kw.pop("draft_p_min", None)
            reasons = {k: v for k, v in tuned.reasons.items() if k != "spec_type"}
            tuned = dataclasses.replace(tuned, spec_type=None, draft_p_min=None, mtp_layers=None, reasons=reasons)
            dropped = (*dropped, "draft-mtp")
        if halves:
            halved = max(floor or 1, self.full_fit // 2)
            kw["context"] = halved
            report = dataclasses.replace(report, asked=halved, retried=cause == "death")  # IN-01: only a death retried
        return dataclasses.replace(
            self, argv=tuple(server.serve_command(spec, **kw)), serve_kwargs=kw, report=report, tuned=tuned,
            shape="retry", cause=cause, dropped=dropped, remembered_at=remembered_at)



def plan_launch(
    spec: ManagedServerConfig, sensed: Sensed, decision: "server.LaunchDecision", *,
    autosense: "AutosenseConfig", api_key_held: bool, floor: int | None = None,
) -> LaunchPlan:
    """The launch of `spec` as a value (D-15): derive, fit, build the `serve_command` keywords and the
    argv. Pure — every read happened in `sense_launch`, nothing is written, and `api_key_held` is a
    bool so the key itself never enters a plan. The derived keywords reach `serve_command` only when
    not None, so a launch that derived nothing calls it with exactly the 0.18 keywords (R-08c). The
    fit `-c` matches 0.18 only for a record with no `env:` / `config_ini` cache type and no hand-set
    MTP drafting: those are now priced at the types the launch runs with, as intended (IN-03)."""
    if sensed.snapshot_refusal is not None:
        raise server.LaunchRefused(sensed.snapshot_refusal)
    tuned = tuning.derive(spec, sensed.header, autosense, model_config=sensed.model_config)
    kw: dict[str, Any] = {
        "keyed": api_key_held, "loopback": decision.loopback, "key_file_ok": decision.key_file_ok,
        "context": None, "cache_ram": None}
    report = None
    full_fit = None
    home = None
    names = sorted(getattr(spec, "env", None) or {})
    row = RUNTIMES[spec.runtime]
    if row.price is not None:
        priced = row.price(spec, sensed, tuned, autosense, floor)
        if priced.refusal is not None:
            raise server.LaunchRefused(priced.refusal)
        report, kw["context"], kw["cache_ram"], full_fit = priced.report, priced.context, priced.cache_ram, priced.full_fit
        if priced.fit_ctx is not None:
            kw["fit_ctx"] = priced.fit_ctx
    if "config_ini" in row.record_fields:
        home = server.config_ini_home(sensed.config_dir, spec.port)
        names.append("XDG_CONFIG_HOME")
    for name in ("cache_type", "spec_type", "draft_p_min", "reasoning_parser", "tool_call_parser"):
        value = getattr(tuned, name)
        if value is not None:
            kw[name] = value
    if sensed.snapshot is not None:
        kw["snapshot"] = sensed.snapshot
        names.append("HF_HUB_OFFLINE")
    if api_key_held and row.key_env is not None:
        names.append(row.key_env)
    full = LaunchPlan(
        argv=tuple(server.serve_command(spec, **kw)), serve_kwargs=kw,
        env_names=tuple(dict.fromkeys(names)), config_ini_home=home, report=report, tuned=tuned,
        full_fit=full_fit, inputs=(spec, sensed, decision, autosense, api_key_held, floor))
    # The retry shape is chosen here and nowhere else (D-17): first a remembered record that still
    # applies, then the boot marker's verdict. Either gives the full plan back when it has nothing
    # to drop or halve.
    rem = sensed.remembered
    if rem is not None and _remembered_applies(spec, rem, autosense, full_fit):
        shaped = full._shaped("remembered", drop_spec="draft-mtp" in rem.dropped, remembered_at=rem.at)
        if shaped is not None:
            return shaped
    if sensed.prior_death:
        shaped = full.boot_retry()
        if shaped is not None:
            return shaped
    return full


def _remembered_applies(
    spec: ManagedServerConfig, rem: "speed_stats.RetryRecord", autosense: "AutosenseConfig",
    fresh_fit: int | None,
) -> bool:
    """Is the ledger's retry record still the shape to plan (spec 20 §6.4)? Not when the binary, the KV
    cache type knob or `draft_p_min` changed since, or the record now sets `context` / `speculative`
    (the person chose); not when the fresh fit is above the recorded one (the full shape is tried
    again). An unknown fresh fit counts as not larger (R-09c). Pure."""
    if rem.binary != str(spec.binary):
        return False
    if rem.kv_cache_type != autosense.kv_cache_type or rem.draft_p_min != autosense.draft_p_min:
        return False
    if spec.context is not None or spec.speculative is not None:
        return False
    return fresh_fit is None or fresh_fit <= rem.full_fit


def _say_plan(on_plan: "Callable[[LaunchPlan | None], None] | None", plan: "LaunchPlan | None") -> None:
    """Hand a plan to the caller's printer right before its launch. The printer is the caller's, and a
    printer that raises must not stop a launch."""
    if on_plan is None:
        return
    try:
        on_plan(plan)
    except Exception:  # noqa: BLE001 - the callback is display only
        log.debug("on_plan callback raised", exc_info=True)


def _plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def confirm_launch(
    config_dir: Path, record_key: str, *,
    plan: "LaunchPlan | None" = None, marker: "units.Marker | None" = None,
    live: "LiveEndpoint | None" = None, process: "ProcRecord | None" = None,
    unit: "units.UnitRuntimeState | None" = None, context_length: int | None = None,
    spec: ManagedServerConfig | None = None, autosense: "AutosenseConfig | None" = None,
    header: Mapping[str, Any] | None = None, total_bytes: int | None = None,
) -> None:
    """Confirm a launch that answered, and remember what worked (spec 20 §6.0, §6.4, D-18). This is the
    only writer of the ledger's `retry` record and the only remover of the boot marker, and it never
    senses: after a load the free memory is the loaded machine's and would say nothing true about the
    fit. It returns None and never raises, since the ledger is a convenience and a launch that served
    must not fail on it.

    From a `plan` (`activate`, which still holds the `Sensed` taken before the launch): a served
    death-retry whose fit is known writes the record from the plan's pre-launch values; a served full
    shape clears the record when `Sensed.remembered` was set; a remembered- or boot-shaped plan leaves
    it as it is.

    From a `marker` (`start`'s attach and the unit's wait, later phases): the ledger is written only
    when the marker is bound to the run that is serving, `marker.invocation` equal to the unit's live
    `InvocationID`, the live argv's `-c` equal to the marker's `planned_context` (llama.cpp rounds the
    served window up), the unit's `NRestarts` at least 1, `cause` `death`, and the marker's record and
    binary this record's, with the verified live `process` given. `dropped` is `draft-mtp` only when the record's own `derive` (over the
    caller's `header` and `autosense`, never sensed here) would emit a spec type and the live argv has
    none; the marker's own `dropped` is never read. `full_fit` is clamped to `context_length` (default
    the header's), and with no positive context length known nothing is recorded; `free_bytes` is
    clamped to `total_bytes`. A marker whose invocation matches is removed
    whatever the outcome; one that does not is left for the run it belongs to."""
    ledger = speed_stats.default_speed_stats_path(config_dir)
    stamped = spec if spec is not None else plan.inputs[0] if plan is not None and plan.inputs else None
    if live is not None and stamped is not None and RUNTIMES[stamped.runtime].checkpoint:
        try:
            speed_stats.record_served_here(ledger, record_key)
        except OSError:
            pass
    if plan is not None:
        if plan.inputs is None:
            return
        plan_spec, sensed, _, plan_autosense, *_ = plan.inputs
        try:
            if plan.shape == "retry" and plan.cause == "death":
                if plan.full_fit is not None:
                    speed_stats.record_retry(
                        ledger, record_key, full_fit=plan.full_fit,
                        free_bytes=sensed.memory.free_bytes if sensed.memory else None,
                        dropped=plan.dropped, binary=str(plan_spec.binary),
                        kv_cache_type=plan_autosense.kv_cache_type, draft_p_min=plan_autosense.draft_p_min)
            elif plan.shape == "full" and sensed.remembered is not None:
                speed_stats.clear_retry(ledger, record_key)
        except (OSError, ValueError):
            pass
        return
    if marker is None:
        return
    bound = (unit is not None and marker.invocation is not None
             and marker.invocation == unit.invocation_id)
    try:
        known = context_length if context_length is not None else (header or {}).get("context_length")
        # A planted full_fit is only ever recorded clamped (D-18): with no positive context length to
        # clamp it to, nothing is recorded (WR-02).
        if _plain_int(known) and known > 0 and _plain_int(marker.full_fit):
            full_fit = min(marker.full_fit, known)
        else:
            full_fit = None
        if (bound and unit.n_restarts >= 1 and marker.cause == "death" and process is not None
                and spec is not None and autosense is not None
                and marker.record == record_key and marker.binary == str(spec.binary)
                and _plain_int(marker.planned_context)
                and _auto._flag_value(process.argv, ("-c", "--ctx-size")) == str(marker.planned_context)
                and _plain_int(full_fit) and full_fit > 0):
            derived = tuning.derive(spec, header or {}, autosense)
            hand_set = any(t == "--spec-type" or t.startswith("--spec-type=") for t in process.argv)
            dropped = ("draft-mtp",) if derived.spec_type is not None and not hand_set else ()
            free = marker.free_bytes
            if _plain_int(free) and free >= 0 and _plain_int(total_bytes) and total_bytes >= 0:
                free = min(free, total_bytes)
            else:
                free = None
            speed_stats.record_retry(
                ledger, record_key, full_fit=full_fit, free_bytes=free, dropped=dropped,
                binary=str(spec.binary), kv_cache_type=autosense.kv_cache_type,
                draft_p_min=autosense.draft_p_min)
    except (OSError, ValueError):
        pass
    if bound:
        try:
            units.remove_marker(config_dir)
        except OSError:
            pass


class SpawnedProcessStrategy:
    """Lifecycle for a harness-SPAWNED OpenAI-compatible server process — llama.cpp's
    `llama-server` (0.11 Phase B) and, later, any single-process binary that serves the OpenAI
    wire. The harness launches it ITSELF (the 0.11 goal) through the SAME `provider/server.py`
    primitives `ManagedVllmStrategy` uses, so the proven detached-launch / pidfile / readiness /
    verified-stop recipe is REUSED, not reinvented:

    - activate = `launch_decision` → `serve_command` → `start_server` → `wait_ready` (returns
                 served ids); the stop handle is the launched PID. Like the vLLM strategy it is the
                 one enforcement point of the decision: a declared-only record (no binary) is
                 refused as `LaunchRefused` before anything starts.
    - stop     = `stop_server(config_dir, launch="binary")` — SIGTERM the process group, SIGKILL
                 on timeout. Always the pid-group teardown: a spawned server is a single tracked
                 process, never a `docker run` client, so `launch` is pinned to "binary" here.
    - liveness = `server_pid(config_dir) is not None` — CORRECT here (unlike docker): the pidfile
                 pid IS the real `llama-server` process, so a live pid means it is serving. This is
                 exactly the case the #99/#100 docker orphan-client-pid bug was NOT.

    Phase B ships the mechanism + a gated live spawn test; rewiring `start` / `/model` swap onto
    strategies is Phase C (keeps this new spawn path out of the monkeypatch-sensitive rewire).
    """

    plans_launch = True

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint:
        """Bring the spawned server up and return its endpoint.

        With `autosense` set the launch is planned: `sense_launch` reads everything once (off the event
        loop), the pure `plan_launch` derives the KV type and MTP drafting, fits the window priced for
        them and builds `plan.argv`, which is exactly what is started (the config.ini home is the one
        write, prepared just before). `on_plan` is called with the plan right before its launch. After
        readiness the served window is read back (`probe_window`) into `LiveEndpoint.window`, and a
        window smaller than asked is a mismatch to report, never a failure. With `autosense=None` the
        launch is untuned: no sense, no plan, no probe, no retry, `on_plan(None)`.

        A launch the plan derived a shape for costs exactly one retry. When it carried a derived spec
        type or a fit `-c` and the process died before it answered, the stale process is stopped
        (verified, by `stop_server`) and `plan.retry()` is launched once, from the same `Sensed`: the
        derived spec type dropped, the fit `-c` halved (`max(1, fit // 2)`). A second death is raised
        with the server's log tail after stopping again so no half-started process is left. A dead
        process is the only trigger: a `TimeoutError` is never retried, and nothing is retried when the
        plan has neither (a KV-only launch, a record `context`, an `extra_args` `-c`, an unknown fit).

        "A dead process or a load failure in the log's last lines" (AUTO-05, ROADMAP SC4) is one
        trigger (D-37): llama-server exits on a load failure, so the load failure IS the dead
        process `wait_ready` already detects (the pidfile pid and its start time are gone). No log
        text is parsed; the abandon message quotes the scrubbed tail (`scrubbed_tail`) and a log reference."""
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        # OFF the event loop: sensing reads the header, stats the weights and shells out to
        # nvidia-smi (up to 5 s on a wedged driver), none of which may freeze the REPL's loop.
        plan: LaunchPlan | None = None
        if autosense is not None and RUNTIMES[spec.runtime].gguf:
            sensed = await asyncio.to_thread(sense_launch, spec, config_dir, autosense)
            plan = plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed, floor=floor)
            argv = list(plan.argv)
        else:
            argv = server.serve_command(
                spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok,
                context=None, cache_ram=None)
        _say_plan(on_plan, plan)

        # Once per activate (the retry reuses it): the llama.cpp config the child reads is
        # the record's bytes in a harness-owned directory, never the person's real file (D-30).
        try:
            home = server.prepare_config_ini_home(config_dir, spec.port, getattr(spec, "config_ini", None))
        except (OSError, UnicodeError) as exc:  # a lone surrogate in the record cannot be written as UTF-8
            raise RuntimeError(
                f"could not prepare {server.config_ini_home(config_dir, spec.port)}: "
                f"{getattr(exc, 'strerror', None) or exc}"
            ) from exc

        def launch(cmd: list[str]) -> tuple[int, list[str]]:
            names: list[str] = []
            pid = server.start_server(
                config_dir, cmd, runtime=spec.runtime, record_env=getattr(spec, "env", None),
                dropped_out=names, xdg_config_home=str(home), port=spec.port,
                **server.key_kwargs(api_key))
            return pid, names

        async def ready() -> list[str]:
            return await server.wait_ready(
                base_url,
                config_dir=config_dir,
                timeout_seconds=timeout_seconds,
                on_poll=on_poll,
                runtime=spec.runtime,
                **server.key_kwargs(api_key),
            )

        pid, dropped = launch(argv)
        try:
            served = await ready()
        except RuntimeError as died:
            # The plan alone says whether a death is worth a second try (`LaunchPlan.retry`, D-16):
            # a derived spec type or a fit `-c`. A launch with neither, a record value, an
            # extra_args value, no plan at all and a TimeoutError (not a RuntimeError) re-raise at once.
            retry = plan.retry() if plan is not None else None
            if retry is None:
                raise
            try:
                await asyncio.to_thread(
                    server.stop_server, config_dir, launch="binary", expect=self.launch_identity(spec))
            except RuntimeError:
                # A pid that survived SIGKILL makes a relaunch unsafe; the load failure with its log
                # tail is the message the user needs, not the stop's.
                raise died from None
            _say_plan(on_plan, retry)
            pid, dropped = launch(list(retry.argv))
            try:
                served = await ready()
            except RuntimeError:
                # Best-effort: the second death's log tail stays the message even when the stop fails.
                with contextlib.suppress(RuntimeError):
                    await asyncio.to_thread(
                        server.stop_server, config_dir, launch="binary", expect=self.launch_identity(spec))
                raise
            plan = retry
        report = plan.report if plan is not None else None
        if report is not None:
            from localharness.agent.context import probe_window

            model_id = served[0] if served else spec.served_name
            probe = await asyncio.to_thread(probe_window, base_url, model_id, "llamacpp", api_key=api_key)
            report = dataclasses.replace(
                report, served=probe.served, trained=probe.trained or report.trained,
                verified=probe.served is not None)
            if report.placement is not None and _auto._flag_value(plan.argv, _auto._NGL) is None:
                report = dataclasses.replace(report, held=await asyncio.to_thread(_auto.held_by, pid))
        live = LiveEndpoint(base_url=base_url, served_models=served, handle=pid, note=decision.note,
                            window=report, dropped_env=tuple(dropped), tuned=plan.tuned if plan else None)
        if plan is not None:
            confirm_launch(config_dir, record_key(spec), plan=plan, live=live)
        return live

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir `activate` polls with, so a reconnect poll reads the same pidfile and log."""
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # OFF the event loop (same reason as ManagedVllmStrategy.stop): a SIGTERM→SIGKILL grace
        # window blocks, so run it in a worker thread. `launch="binary"` ALWAYS here — a spawned
        # server is a single tracked process, never a docker-run client. `expect` is what that
        # process must be (D-27): a pidfile is a claim, not proof.
        await asyncio.to_thread(
            server.stop_server, config_dir, launch="binary", expect=self.launch_identity(spec))

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        pid = server.server_pid(config_dir)
        return Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        """The `/proc` shape of a `llama-server` launch of `spec`: exe names `llama-server` (a wrapper execs it), the binary's basename and its realpath's; the port from `--port` or `LLAMA_ARG_PORT`; no extra token."""
        binary = getattr(spec, "binary", None)
        names = {procfs.LLAMA_EXE} | ({os.path.basename(str(binary)), os.path.basename(os.path.realpath(str(binary)))}
                                      if binary is not None else set())
        return LaunchIdentity(frozenset(names), spec.port, (), "LLAMA_ARG_PORT")


class DaemonStrategy:
    """Ollama (0.11 Phase D). The harness spawns the `ollama serve` DAEMON itself and loads the
    target model into it. Two ollama facts shape this:

    - `ollama serve` takes NO model argument — models load lazily per request — so activate must
      LOAD `spec.model` (an HTTP warm-up) after the daemon is up, else `served_models` would just
      mean "daemon up" (Ollama's /v1/models lists PULLED-to-disk models, resident or not).
    - **STOP kills the WHOLE daemon** (owner ruling): killpg SIGTERM→SIGKILL of the daemon's
      process group is a hard free (the kernel reclaims memory on process exit; runner children
      are reaped with the group) — though unlike docker-stop / LM Studio's stop it does not
      re-poll for death after SIGKILL — whereas a per-model `keep_alive:0` unload is a soft
      signal that may not promptly/fully free VRAM.

    Mechanically the daemon is a harness-SPAWNED process, so it reuses SpawnedProcessStrategy's
    proven launch / pidfile / verified-stop / pid-liveness primitives — the ollama-specific parts
    are only the env-built `serve_command`, the model warm-up, and a pull-if-absent. Under the
    GPU-lock (at most one launched server up) the daemon's pid in the shared per-config pidfile is
    the sole live one, so the C2 single-pidfile invariant holds unchanged. (Attaching to a
    PRE-EXISTING daemon the harness didn't spawn is a later concern — this box's ollama.service is
    disabled, so the harness always owns the daemon it starts.)"""

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint:
        # CONTRACT: like every LifecycleStrategy.activate, this raises ONLY RuntimeError/TimeoutError
        # on failure — the /model swap callers catch exactly those to tear down + restore. _load
        # (httpx) and _pull (subprocess) therefore MAP their native errors into that vocabulary;
        # without it an httpx.* / OSError would escape the caller's except and crash the session with
        # the just-spawned daemon orphaned.
        v1 = self._v1(base_url)
        if await self._daemon_already_up(v1):
            # Fail-fast, NOT silent: the harness owns only daemons IT spawns (pidfile-tracked). If one
            # is already listening, a 2nd `ollama serve` can't bind and we'd mis-track it → a later
            # stop misses the real daemon (silent two-heavy). Attach-and-own is deferred (see docstring).
            raise RuntimeError(
                f"an ollama daemon is already listening at {v1} — the harness manages only daemons it "
                "starts; stop the existing one first (attach-and-own is deferred)."
            )
        cmd = server.serve_command(spec)                 # env-prefixed `ollama serve`
        pid = server.start_server(config_dir, cmd, runtime=spec.runtime, port=spec.port)
        available = await server.wait_ready(             # daemon up; returns PULLED model ids
            v1, config_dir=config_dir, timeout_seconds=timeout_seconds, on_poll=on_poll,
            **server.key_kwargs(api_key),
        )
        if spec.model not in available:
            await self._pull(spec)                       # tag not on disk → pull it
        await self._load(v1, spec.model)                 # warm-up: load into memory (no generation)
        return LiveEndpoint(base_url=v1, served_models=[spec.model], handle=pid)

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir `activate` polls with, so a reconnect poll reads the same pidfile and log."""
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # OWNER RULING: stop the WHOLE daemon (verified GPU-free), NOT keep_alive:0. A spawned daemon
        # leads its own process group (start_new_session=True), so stop_server's killpg SIGTERM→SIGKILL
        # reaps the daemon AND its runner children (which hold the weights), off the loop.
        await asyncio.to_thread(
            server.stop_server, config_dir, launch="binary", expect=self.launch_identity(spec))

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        pid = server.server_pid(config_dir)
        return Liveness(
            alive=pid is not None,
            detail=f"ollama daemon pid {pid}" if pid is not None else "no daemon",
        )

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        """The `ollama` daemon (`env` execs it), the port from `OLLAMA_HOST` (D-25)."""
        return LaunchIdentity(procfs.OLLAMA_EXES, spec.port, (), "OLLAMA_HOST")

    @staticmethod
    def _v1(base_url: str) -> str:
        """Ensure the OpenAI-compat /v1 suffix. wait_ready polls {base_url}/models and the client
        speaks OpenAI, but EndpointRef.validate_base_url leniently allows a BARE Ollama root — which
        would make wait_ready poll /models (404) and spin to the full timeout. Normalize here."""
        b = base_url.rstrip("/")
        return b if b.endswith("/v1") else b + "/v1"

    @staticmethod
    def _daemon_root(base_url: str) -> str:
        """Ollama's NATIVE API (/api/*) lives at the daemon root, not under the OpenAI-compat /v1."""
        return base_url.rstrip("/").removesuffix("/v1")

    async def _daemon_already_up(self, v1_base: str) -> bool:
        """A daemon already listening on the target port (one the harness did NOT spawn → can't
        pidfile-track → can't stop). Regular method so tests can monkeypatch it hermetically. Boolean
        probe: catch broadly — a malformed base_url raises httpx.InvalidURL / an OverflowError group
        (SIBLINGS of httpx.HTTPError) that would else escape activate and crash the /model swap."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{v1_base.rstrip('/')}/models")
            return r.status_code == 200
        except Exception:  # noqa: BLE001 — boolean probe: any error = not-up (must never propagate)
            return False

    async def _load(self, base_url: str, model: str) -> None:
        # POST /api/generate with an EMPTY prompt is Ollama's canonical "load into memory without
        # generating" call (returns done_reason=load). No token budget — nothing is generated — so
        # this respects "give thinking room". MAP httpx failures into the activate contract.
        import httpx
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                r = await client.post(
                    f"{self._daemon_root(base_url)}/api/generate",
                    json={"model": model, "prompt": "", "keep_alive": -1, "stream": False},
                )
                r.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError(f"ollama load of {model!r} timed out after 600s: {exc}") from exc
        except httpx.HTTPError as exc:  # incl. HTTPStatusError from raise_for_status
            raise RuntimeError(f"ollama load of {model!r} failed: {exc}") from exc

    async def _pull(self, spec: ManagedServerConfig) -> None:
        # MAP a missing/unrunnable `ollama` binary into the activate contract (else FileNotFoundError
        # escapes the swap caller's except and crashes the session).
        binary = str(spec.binary) if spec.binary else "ollama"
        try:
            proc = await asyncio.create_subprocess_exec(
                binary, "pull", spec.model,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:  # FileNotFoundError (binary not on PATH) is an OSError
            raise RuntimeError(f"could not run `{binary} pull {spec.model}`: {exc}") from exc
        try:
            _, err = await proc.communicate()
        finally:
            await _kill_and_reap(proc)  # #153: a cancelled swap must not leave the pull running
        if proc.returncode != 0:
            raise RuntimeError(
                f"`ollama pull {spec.model}` failed: {(err or b'').decode('utf-8', 'replace')[:200]}"
            )


class LmsStrategy:
    """LM Studio (0.11 Phase D5). The harness drives LM Studio's HEADLESS `lms` CLI to bring the
    backend up — `lms daemon up` (start/attach the persistent `llmster` daemon) → `lms load <model>`
    → `lms server start` (the OpenAI server) — and takes it down with `lms daemon down`. Unlike
    DaemonStrategy (ollama, a harness-SPAWNED foreground `ollama serve`), every `lms` subcommand
    RETURNS immediately (the llmster daemon backgrounds itself), so there is no foreground process to
    pidfile-track. Two consequences:

    - **liveness + the pre-existing-server fail-fast are ENDPOINT-based** (GET {v1}/models), never a
      pidfile — LmsStrategy touches NONE of server.py's pidfile primitives, so it sidesteps the C2
      single-pidfile invariant entirely (an lmstudio launch never writes the shared pidfile a
      vLLM/llama.cpp/ollama launch does).
    - **STOP = `lms daemon down`** — the verified GPU-free teardown (owner's whole-daemon ruling, the
      ollama analog), settled empirically 2026-07-29: `lms server stop` tears down ONLY the HTTP
      listener while the llmster daemon KEEPS the model resident (`lms ps` still IDLE, the
      `liblmstudio` engine still in memory) — a soft signal, the LM Studio analog of ollama's
      keep_alive:0. `lms daemon down` reaps the whole daemon + engine (zero orphans confirmed). It
      returns rc=0 when it stopped something and rc=1 when already down — both ARE the goal state, so
      stop treats {0,1} as success; any other exit / a missing binary / a timeout propagates as the
      activate contract's (RuntimeError|TimeoutError) so a failed teardown never reports GPU-free.

    CONTRACT: every `lms` subprocess AND the endpoint probe MAP their native errors (OSError for a
    missing binary, a non-zero exit, httpx failures) into (RuntimeError|TimeoutError) — the /model
    swap callers catch exactly those to tear down + restore; a leaked httpx.* / OSError would escape
    and crash the session with the just-started daemon orphaned (the CRITICAL trap the ollama critic
    caught). Serving is LOCAL-only (`--bind 127.0.0.1`) on the port from the caller's base_url (where
    it rebinds the client after activate). CPU when `spec.gpu` is False (`--gpu off`, so the lifecycle
    validates with NO GPU window); `--gpu max` when heavy. (Attaching to a PRE-EXISTING daemon the
    harness didn't start is deferred — same as ollama; this box runs no other LM Studio.)"""

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint:
        v1 = self._v1(base_url)
        if await self._server_already_up(v1):
            # Fail-fast, NOT silent (the ollama precedent): a server already answering at v1 is one
            # the harness did NOT start (it pidfile-tracks nothing here), so a `lms daemon down` on a
            # later swap would kill a backend we don't own. Refuse loudly.
            raise RuntimeError(
                f"an LM Studio server is already listening at {v1} — the harness manages only "
                "servers it starts; stop the existing one first (`lms server stop && lms daemon down`)."
            )
        lms = self._lms_bin(spec)
        port = self._port(base_url, spec.port)
        await self._run(lms, "daemon", "up")                     # start/attach llmster (rc0 idempotent)
        gpu = "max" if spec.gpu else "off"                       # spec.gpu → the GPU-lock signal
        await self._run(
            lms, "load", spec.model, "--gpu", gpu, "-y", *spec.extra_args,
            timeout=timeout_seconds,                             # a model load can take minutes
        )
        await self._run(lms, "server", "start", "--port", str(port), "--bind", "127.0.0.1")
        await server.wait_ready(                                 # confirm the OpenAI endpoint answers
            v1, config_dir=None, timeout_seconds=timeout_seconds, on_poll=on_poll,
            **server.key_kwargs(api_key),
        )
        # The served id IS the loaded model key (LM Studio's /v1/models also lists any embedding
        # model; the caller takes served_models[0], so return the intended key — the DaemonStrategy
        # precedent — not wait_ready's raw list). No pidfile → an opaque daemon handle.
        return LiveEndpoint(base_url=v1, served_models=[spec.model], handle="lms-daemon")

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        # `activate` polls with `config_dir=None` and writes no pidfile, so a dead LM Studio server is
        # polled until the bound and is never relaunched (G-5, P-18).
        return None

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # `lms daemon down` (whole daemon) = the verified GPU-free stop (a bare `lms server stop`
        # leaves the model resident). rc0 = stopped, rc1 = already down — both the goal state. Off the
        # event loop via _run's create_subprocess (never blocks the loop).
        await self._run(self._lms_bin(spec), "daemon", "down", ok_codes=(0, 1))
        # VERIFY the llmster DAEMON PROCESS is actually GONE before returning — the true
        # accelerator-free signal on unified memory (the kernel reclaims the model's pages on process
        # EXIT). Measured 2026-07-29: both the HTTP listener AND `lms daemon status` flip to down the
        # INSTANT `daemon down` is issued — ~0.4s (a 0.5B; more for a heavy) BEFORE the process holding
        # the weights exits — so neither is a sound free-signal. This matches the sibling strategies,
        # whose stop_server polls `_alive(pid)` until the process is dead; LmsStrategy has no pidfile
        # (`lms` backgrounds the daemon) so it polls by the stable daemon name. Fail explicit if it
        # never dies (#100 — never report a false accelerator-free that lets the caller launch a 2nd heavy).
        for _ in range(50):  # ~5s, attempt-counted (deterministic with sleep patched)
            if not await self._llmster_running():
                return
            await asyncio.sleep(0.1)
        raise RuntimeError(
            "the LM Studio `llmster` daemon is still running after `lms daemon down` — refusing to "
            "report the accelerator free when it is not"
        )

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        # ENDPOINT-based (LmsStrategy owns no pidfile). No caller wires this yet (Protocol
        # conformance); probe the local port the config declares. Sync (the Protocol is sync).
        v1 = self._v1(f"http://127.0.0.1:{int(spec.port)}")
        import httpx
        try:
            alive = httpx.get(f"{v1}/models", timeout=2.0).status_code == 200
        except Exception:  # noqa: BLE001 — boolean probe (see _server_already_up): any error = down
            alive = False
        return Liveness(alive=alive, detail=f"lms endpoint {v1}" if alive else "lms endpoint down")

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        """None: LM Studio's daemon is not a process the harness launched or can tell apart (it owns
        no pidfile), so there is no identity to verify and nothing to signal (D-25)."""
        return None

    @staticmethod
    def _v1(base_url: str) -> str:
        """Ensure the OpenAI-compat /v1 suffix (EndpointRef.validate_base_url leniently allows a bare
        root); wait_ready / the client poll {base}/models, so a bare root would hit /models → 404."""
        b = base_url.rstrip("/")
        return b if b.endswith("/v1") else b + "/v1"

    @staticmethod
    def _port(base_url: str, default: int) -> int:
        """The port `lms server start` binds — the caller's base_url (repl rebinds the client THERE
        after activate) wins, falling back to spec.port when the URL carries no explicit port."""
        from urllib.parse import urlparse
        try:
            p = urlparse(base_url).port
        except ValueError:
            p = None
        return p if p is not None else int(default)

    @staticmethod
    def _lms_bin(spec: ManagedServerConfig) -> str:
        """The `lms` CLI: `spec.binary` (e.g. ~/.lmstudio/bin/lms, which is NOT on PATH by default)
        or a bare `lms` fallback."""
        return str(spec.binary) if spec.binary else "lms"

    async def _server_already_up(self, v1_base: str) -> bool:
        """A server already answering at v1 (one the harness did NOT start). Regular method so tests
        monkeypatch it hermetically — mirrors DaemonStrategy._daemon_already_up. A BOOLEAN probe: any
        failure to get a clean 200 means 'not confirmably up' → proceed. Catches broadly on purpose —
        a malformed base_url is a DELIBERATE probe-time skip (EndpointRef.validate_base_url defers
        port validity to here), and a bad port raises httpx.InvalidURL / an OverflowError group,
        SIBLINGS of httpx.HTTPError that would else escape and crash the /model swap."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{v1_base.rstrip('/')}/models")
            return r.status_code == 200
        except Exception:  # noqa: BLE001 — boolean probe: any error = not-up (must never propagate)
            return False

    async def _llmster_running(self) -> bool:
        """Is an `llmster` daemon process alive? A process-NAME check (LmsStrategy owns no pidfile,
        and `lms daemon up` backgrounds the daemon so its pid is not ours to track). The harness owns
        the ONLY llmster it starts (activate fail-fasts on a pre-existing server), so any live
        `llmster` is ours. POSIX `pgrep -x` — the lifecycle layer is already POSIX-only. Instance
        method (ignores self) so tests monkeypatch it hermetically, like _server_already_up."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-x", "llmster",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
            return await proc.wait() == 0  # rc 0 = at least one match; 1 = none
        except OSError:
            return False  # no `pgrep` (missing / non-POSIX) → cannot verify → best-effort "gone"

    async def _run(self, *args: str, timeout: float = 600.0, ok_codes: tuple[int, ...] = (0,)) -> None:
        """Run an `lms` subcommand, MAPPING native failures into the activate contract: a missing
        binary (OSError) → RuntimeError; a timeout → TimeoutError; an exit code not in `ok_codes` →
        RuntimeError with the stderr tail. Without this an OSError would escape the /model swap
        caller's (RuntimeError|TimeoutError) except and crash the session. `ok_codes` lets the
        idempotent `lms daemon down` (rc=1 when already down) count as success."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:  # FileNotFoundError (lms not on PATH) is an OSError
            raise RuntimeError(f"could not run `{' '.join(args)}`: {exc}") from exc
        try:
            _, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TimeoutError(f"`{' '.join(args)}` timed out after {timeout:.0f}s") from exc
        finally:
            # Timeout, cancellation, or any other escape (#153). ProcessLookupError from a proc
            # that raced us to exit stays swallowed in there, so nothing but the activate
            # contract's (RuntimeError|TimeoutError) ever leaves this method.
            await _kill_and_reap(proc)
        if proc.returncode not in ok_codes:
            raise RuntimeError(
                f"`{' '.join(args)}` failed (exit {proc.returncode}): "
                f"{(err or b'').decode('utf-8', 'replace')[:200]}"
            )


def strategy_for(
    spec: ManagedServerConfig, *, unit: "units.UnitRuntimeState | None" = None,
) -> LifecycleStrategy:
    """The runtime row's launch strategy, wrapped in a `UnitStrategy` only for the harness's own unit and a unit-servable record."""
    base = RUNTIMES[spec.runtime].strategy(spec)
    if isinstance(unit, units.UnitRuntimeState) and unit.own and unit_servable(spec):
        return UnitStrategy(base, unit, spec)
    return base


NOT_SERVED_BY_UNIT_LINE = "{name} is {kind} — not served by the unit"
"""What `serve`, `--autostart status` and the doctor row say for a record the autostart unit cannot
serve (spec 20 §9.1, D-02); `kind` comes from `unit_kind`."""


def unit_servable(spec: ManagedServerConfig | None) -> bool:
    """Can the autostart unit serve this record: llama.cpp and a vLLM `launch: binary` record only."""
    return spec is not None and RUNTIMES[spec.runtime].unit_servable(spec)


def unit_kind(spec: ManagedServerConfig | None) -> str:
    """What a record is, for `NOT_SERVED_BY_UNIT_LINE` and the scope refusal; `attach-only` for no record."""
    return "attach-only" if spec is None else RUNTIMES[spec.runtime].label(spec)


RUNS_OUTSIDE_AUTOSTART = "(runs outside autostart — the next boot serves nothing)"
"""The clause a launch line carries when the harness's own unit exists but the target is a record the
unit cannot serve (a docker, Ollama or LM Studio record): it launches through its own strategy, outside
the unit, and the next boot's `serve` exits 0 with nothing to serve (spec 20 §8.2, D-21)."""


def outside_autostart(spec: ManagedServerConfig | None, unit: Any) -> bool:
    """Does a launch of `spec` run outside the autostart unit that exists (D-21)? Only for the harness's
    OWN unit (a `UnitRuntimeState` with `own=True`) and a record `unit_servable` says the unit cannot
    serve. No unit, a raw or foreign state, and a record the unit serves are all False."""
    return isinstance(unit, units.UnitRuntimeState) and unit.own and not unit_servable(spec)


def leave_unit(spec: ManagedServerConfig | None, unit: Any, config_dir: Path) -> bool:
    """Take `unit-target` away when a launch of `spec` runs outside the harness's own unit (R-14-28,
    T-14-30): the ladder then falls to `active_endpoint`, and the next boot's `serve` exits 0 with its
    not-served line instead of bringing up the older choice beside the server this launch starts. True
    when the launch is outside the unit. The caller holds the launch lock and calls this right before
    the launch. A file that cannot be removed is a RuntimeError, before anything is launched."""
    if not outside_autostart(spec, unit):
        return False
    try:
        units.remove_target(config_dir)
    except OSError as exc:
        raise RuntimeError(
            f"could not leave the unit target: {getattr(exc, 'strerror', None) or type(exc).__name__}"
        ) from None
    return True


# ---------------------------------------------------------------------------
# Launching through the harness's own autostart unit (spec 20 §9.4)
# ---------------------------------------------------------------------------
UNIT_TYPED_LINE = ("under autostart /model takes preset names — put {path} under autosense.model_roots "
                   "and run localharness init to add it as a peer")
"""The refusal for a typed `.gguf` restart (or a model swapped in memory) under the unit: the unit serves
the records the machine record fingerprinted, so the model must be a record first (D-20)."""

UNIT_NAME_LINE = ("the autostart unit takes a record name of letters, digits and . _ @ + : - (and not "
                  "server:); {name!r} is not one")
"""The refusal for a peer name `unit-target` cannot hold (R-14-23)."""

UNIT_ENV_LINE = ("the autostart unit would exit 78 at its start: {why}. localharness model --autostart on "
                 "re-adopts it; localharness start in a terminal asks about it")
"""The refusal when the machine record does not cover the unit's captured environment (a withheld, changed
or missing `autostart.env` entry): `serve` would exit 78 on the restart, so nothing is stopped for it
(`unit_launch_refusal`)."""

UNIT_RESTORE_LINE = "{label} could not be put back through the autostart unit if this swap failed: {why}"
"""The refusal when the INCUMBENT of a swap would be restored through the unit and the pre-flight meets the
refusal for it that the restore would meet (`unit_launch_refusal`): a target outside the unit that failed to
launch would leave nothing serving, so nothing is stopped for it. `why` is the refusal's own line."""

UNIT_FOREIGN_LINE = "{unit} {why} — it is not the unit localharness wrote, so it was not restarted"
"""The refusal when the unit failed its identity re-check before a restart (D-13, T-14-22); `why` is
`units.harness_unit_failure`'s phrase."""

UNIT_FOREIGN_STOP_LINE = ("pidfile names pid {pid}, run by localharness-llm.service, which is no longer the unit "
                          "localharness wrote ({why}) — not stopped")
"""The refusal when the launch's owner is the unit but the unit failed its identity re-check before a stop
(D-21, T-14-23); `why` is `units.harness_unit_failure`'s phrase. The process is named and never signalled."""

UNIT_UNASKED_STOP_LINE = ("pid {pid} runs under {unit}, and the service manager could not be asked, or does not "
                          "name it as the unit's main process — not stopped (the unit would restart it); stop "
                          "the unit from a terminal once the manager answers")
"""The refusal when the pidfile's process sits in the harness unit's cgroup but the manager cannot be
asked (MainPID unknown) or names another pid as the unit's main process: the process is the unit's whatever
the owner verdict says, and a signal sent behind the manager's back is counted as a failure by
`Restart=on-failure` (WR-06, WR-09). Named, never signalled."""

UNIT_FAILED_LINE = "{label} did not come up under the autostart unit"
UNIT_EXITED_LINE = "{label}: the autostart unit exited without starting a server"
UNIT_TIMEOUT_LINE = "{label}: the autostart unit's server was not ready after {seconds}s"

LEND_POLL_SECONDS = 0.25
"""How often the lent launch lock's holder looks at the unit. A poll interval, never a bound: the loan
ends on an event or on the activate's own deadline (R-14-21)."""


def unit_name_refusal(peer: str | None) -> str | None:
    """`UNIT_NAME_LINE` for a peer name `unit-target` cannot hold (the primary's own reserved name, a name
    outside `units.TARGET_NAME_PATTERN`: a space, a non-ASCII letter, over 128 characters), else None. A
    primary (``peer`` None) always has a name."""
    if peer is not None and (peer == units.PRIMARY_TARGET or not units.TARGET_NAME_PATTERN.fullmatch(peer)):
        return UNIT_NAME_LINE.format(name=peer)
    return None


def unit_launch_refusal(
    spec: ManagedServerConfig | None, peer: str | None, unit: Any, config_dir: Path,
) -> str | None:
    """Why a launch of ``spec`` (the peer named ``peer``, None for the primary) through the harness's own
    unit is known to fail before anything is touched, else None. A swap stops the incumbent first, so every
    refusal that is knowable beforehand is checked here, before that stop (the repo's rule: a refused
    launch leaves the incumbent serving):

    - a peer name `unit-target` cannot hold (`unit_name_refusal`);
    - the machine record does not cover the unit's captured environment (`trust.autostart_env_failure`):
      `serve` would exit 78, the restore of the incumbent would be refused the same way, and the box would
      end with nothing serving.

    Only a launch that goes through the harness's OWN unit (`unit` a `UnitRuntimeState` with ``own=True``
    and a record the unit serves) is judged; anything else is None. Blocking: it reads the unit file and
    the trust store, so async callers run it off the loop. Nothing is written."""
    if not (isinstance(unit, units.UnitRuntimeState) and unit.own and unit_servable(spec)):
        return None
    named = unit_name_refusal(peer)
    if named is not None:
        return named
    from localharness.config import trust as _trust

    why = _trust.autostart_env_failure(config_dir)
    return UNIT_ENV_LINE.format(why=why) if why is not None else None


def _unit_failure_text(head: str, invocation: str | None, api_key: Any) -> str:
    """`head`, the unit's journal tail for `invocation` and the reference line, every character scrubbed of
    the held key, `Bearer` tokens, `api_keys:` values and URL credentials (T-14-24). Blocking: it runs the
    journal reader, so callers run it off the event loop. The raw lines stay in the journal."""
    held = [raw] if (raw := server.key_kwargs(api_key).get("api_key")) else []
    try:
        tail = units.journal_tail(invocation=invocation, secrets=held)
    except ValueError:  # an id the manager reported that is not one: ask for the unit's latest lines
        invocation = None
        tail = units.journal_tail(secrets=held)
    head = server.strip_url_credentials(_redact.scrub(head, held, patterns=_redact.LOGGED_KEY_PATTERNS))
    return f"{head}\n{tail}\n{units.JOURNAL_REF_LINE.format(invocation=invocation or 'latest')}"


def _manager_words(done: "units.ManagerResult | None", api_key: Any) -> str:
    """What the service manager said when a verb failed, scrubbed and one line long."""
    if done is None:
        return "the service manager did not answer"
    held = [raw] if (raw := server.key_kwargs(api_key).get("api_key")) else []
    said = _redact.scrub(done.err.strip() or f"exit {done.code}", held, patterns=_redact.LOGGED_KEY_PATTERNS)
    return server.strip_url_credentials(" ".join(said.split())[:300])


async def _lend_until_recorded(config_dir: Path, deadline: float) -> None:
    """Lend the launch lock to the unit's `serve` until it has recorded its pidfile (R-14-21, D-14).

    `UnitStrategy.activate` runs inside its caller's launch lock while `serve` takes that same lock to write
    the pidfile and the boot marker, so the lock is lent for the block. The loan ends on an EVENT, read from
    a fresh state every `LEND_POLL_SECONDS`: the pidfile names a live process whose pid is the unit's
    current MainPID (`serve` wrote it under the lock), or the unit is `failed` or `inactive` (no `serve`
    will take the lock), or `deadline` (the activate's own, on `time.monotonic`) has passed, which is the
    only backstop. A MainPID of none (the unit between its own restarts) keeps the loan open. Nothing here
    counts seconds to decide the loan is long enough, because a clock-ended loan strands a slow `serve`
    outside the lock while the lender waits on its server."""
    async with lend_launch_lock(config_dir):
        while True:
            state = await asyncio.to_thread(units.unit_runtime_state)
            if state is not None:
                if state.active_state in ("failed", "inactive"):
                    return
                if state.main_pid is not None:
                    launch = await asyncio.to_thread(procfs.read_launch, config_dir, main_pid=state.main_pid)
                    if launch is not None and launch.pid == state.main_pid:
                        return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(LEND_POLL_SECONDS)


class UnitStrategy:
    """Launch a llama.cpp or vLLM binary record THROUGH the harness's own autostart unit (spec 20 §9.4,
    D-19, D-20, AUTO-09), wrapping the strategy a launch outside the unit would use.

    `strategy_for(spec, unit=state)` makes one when `state` is the harness's own unit and the record is one
    the unit can serve. `activate` does not start a process: it names the record in `unit-target`, re-checks
    that the unit is still the harness's, resets and restarts it, lends the launch lock to the unit's
    `serve` until that `serve` has recorded its pidfile, and waits with the unit as the judge of death.
    `liveness`, `launch_identity` and `recovery_config_dir` are the wrapped strategy's, so a record is
    recognised in `/proc` exactly as before. Only this class writes `unit-target` (RULING W3)."""

    plans_launch = True

    def __init__(self, inner: LifecycleStrategy, unit: "units.UnitRuntimeState", spec: ManagedServerConfig) -> None:
        self._inner = inner
        self._unit = unit
        self._spec = spec
        self._launch = (spec.model, spec.served_name)

    @property
    def inner(self) -> LifecycleStrategy:
        return self._inner

    @property
    def unit(self) -> "units.UnitRuntimeState":
        return self._unit

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: "AutosenseConfig | None" = None,
        on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
        floor: int | None = None,
    ) -> LiveEndpoint:
        """Restart the unit for `spec` and wait until its `serve` has the server answering.

        In order, with nothing written before the refusals: the record check (`spec` must be the record
        this strategy was chosen for, with the model and served name it had, else `UNIT_TYPED_LINE`; a peer
        name `unit-target` cannot hold is `UNIT_NAME_LINE`), the launch decision (`LaunchRefused`), the
        local plan for the launch line (`on_plan`), then `units.write_target(in_flight=True)`, the own-unit re-check,
        `reset-failed`, `restart` (each verb off the event loop), the loan of the launch lock until the
        unit's `serve` has recorded, and `server.wait_ready(unit_poll=)`. The wait tolerates exactly one
        automatic restart of the unit when the plan's boot retry would change the launch (the unit's own
        retry) and none otherwise; a second rise of `NRestarts` or `ActiveState=failed` raises a
        RuntimeError carrying the SCRUBBED journal tail of that invocation and a reference line, never a raw
        line. `unit-target` carries `<name> <pid> <start>` (this process) from the write on, so the ladder serves
        the unit's restart the record asked for even beside an older `pending_preset` (R-14-38); ONE `finally`
        in this body settles it on every exit (R-14-41, RULING W3): a cancellation, or a refusal or failing
        verb before the restart, puts the prior value back (plain), and every other exit, success included,
        leaves the plain name; an error out of the wait propagates as raised. This never starts a process and
        never stops one outside the unit."""
        if spec is not self._spec or (spec.model, spec.served_name) != self._launch:
            raise server.LaunchRefused(UNIT_TYPED_LINE.format(path=spec.model))
        name = units.PRIMARY_TARGET if peer is None else peer
        named = unit_name_refusal(peer)
        if named is not None:
            raise server.LaunchRefused(named)
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        deadline = time.monotonic() + timeout_seconds
        plan: LaunchPlan | None = None
        sensed: Sensed | None = None
        if autosense is not None:
            # Sensed as the restarted unit will: a manual restart resets `NRestarts`, so the unit's own
            # `serve` plans the full shape (or the remembered one) and the harness predicts the same.
            sensed = await asyncio.to_thread(sense_launch, spec, config_dir, autosense)
            plan = plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed, floor=floor)
        _say_plan(on_plan, plan)
        tolerated = 1 if (plan is not None and plan.shape == "full" and plan.boot_retry() is not None) else 0
        label = server.RUNTIME_LABELS.get(spec.runtime, "vLLM")
        record_name = peer or spec.served_name or label

        prior = units.read_target(config_dir)
        try:
            units.write_target(config_dir, name, in_flight=True)
        except OSError as exc:
            # Nothing was written (or the previous file stands) and nothing was restarted.
            raise RuntimeError(
                f"could not record the unit target: {getattr(exc, 'strerror', None) or type(exc).__name__}"
            ) from None
        # From here on the file names this process as the requester (the in-flight form, R-14-38). The one
        # `finally` below settles it on EVERY exit (R-14-41): the prior value, plain, while no restart was
        # asked (or on a cancellation); the target's plain name otherwise.
        restore_prior = True
        try:
            if await asyncio.to_thread(units.harness_unit_state, config_dir) is None:
                raw = await asyncio.to_thread(units.unit_runtime_state)
                why = units.harness_unit_failure(raw, config_dir) or "could not be confirmed as the harness's own"
                raise server.LaunchRefused(UNIT_FOREIGN_LINE.format(unit=procfs.HARNESS_UNIT, why=why))
            for verb in ("reset-failed", "restart"):
                if verb == "restart":
                    restore_prior = False  # from the moment the verb is issued the unit may be serving `name`
                done = await asyncio.to_thread(units.systemctl_user, verb, procfs.HARNESS_UNIT)
                if done is None or done.code != 0:
                    restore_prior = True  # the verb failed: nothing was restarted
                    raise RuntimeError(f"{verb} of {procfs.HARNESS_UNIT} failed: {_manager_words(done, api_key)}")
            after = await asyncio.to_thread(units.unit_runtime_state)
            base = after.n_restarts if after is not None else 0
            identity = record_identity(spec)

            async def failure(state: "units.UnitRuntimeState", head: str) -> str:
                said = units.translate_exit(state, record_name=record_name)
                return await asyncio.to_thread(
                    _unit_failure_text, f"{head}: {said}" if said else head, state.invocation_id, api_key)

            async def poll() -> str | None:
                """The unit decides death (R-14-20): `failed`, or more automatic restarts than the plan's
                boot retry accounts for, ends the wait with the scrubbed tail; a launch that is neither this
                record's nor the unit's own `serve` ends it too. A unit whose current process has not
                recorded its pidfile (a tolerated restart) gets the lock lent again first."""
                state = await asyncio.to_thread(units.unit_runtime_state)
                if state is None:
                    return None
                if state.active_state == "failed" or state.n_restarts - base > tolerated:
                    return await failure(state, UNIT_FAILED_LINE.format(label=record_name))
                launch = await asyncio.to_thread(
                    procfs.read_launch, config_dir, expect=identity, main_pid=state.main_pid)
                if state.main_pid is None:
                    if state.active_state == "inactive" and (launch is None or not launch.verified):
                        return await failure(state, UNIT_EXITED_LINE.format(label=record_name))
                    return None
                if launch is None or launch.pid != state.main_pid:
                    await _lend_until_recorded(config_dir, deadline)
                    return None
                if not (launch.verified or launch.starting):
                    return server.NOT_THIS_LAUNCH_LINE.format(label=label, pid=launch.pid)
                return None

            async def timed_out() -> TimeoutError:
                state = await asyncio.to_thread(units.unit_runtime_state)
                head = UNIT_TIMEOUT_LINE.format(label=record_name, seconds=int(timeout_seconds))
                return TimeoutError(await asyncio.to_thread(
                    _unit_failure_text, head, state.invocation_id if state else None, api_key))

            await _lend_until_recorded(config_dir, deadline)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise (await timed_out()) from None
            try:
                served = await server.wait_ready(
                    base_url, timeout_seconds=remaining, on_poll=on_poll, runtime=spec.runtime,
                    unit_poll=poll, **server.key_kwargs(api_key))
            except RuntimeError as died:
                raise died from None  # the poll's text is already scrubbed; nothing chains to it
            except TimeoutError:
                raise (await timed_out()) from None

            final = await asyncio.to_thread(units.unit_runtime_state)
            main_pid = final.main_pid if final is not None else None
            launch = await asyncio.to_thread(procfs.read_launch, config_dir, expect=identity, main_pid=main_pid)
            process = None
            if launch is not None and launch.verified:
                found = await asyncio.to_thread(procfs.read_process, launch.pid)
                process = found if isinstance(found, procfs.ProcRecord) else None
            if tolerated and final is not None and final.n_restarts - base >= 1 and plan is not None:
                plan = plan.boot_retry() or plan  # the unit's own retry is what answered
            report = plan.report if plan is not None else None
            if report is not None:
                from localharness.agent.context import probe_window

                model_id = served[0] if served else spec.served_name
                probe = await asyncio.to_thread(probe_window, base_url, model_id, "llamacpp", api_key=api_key)
                report = dataclasses.replace(
                    report, served=probe.served, trained=probe.trained or report.trained,
                    verified=probe.served is not None)
            live = LiveEndpoint(
                base_url=base_url, served_models=served,
                handle=launch.pid if launch is not None else main_pid, note=decision.note,
                window=report, tuned=plan.tuned if plan is not None else None)
            confirm_launch(
                config_dir, record_key(spec), marker=units.read_marker(config_dir), live=live, process=process,
                unit=final, spec=spec, autosense=autosense, header=sensed.header if sensed is not None else None,
                total_bytes=sensed.memory.total_bytes if sensed is not None and sensed.memory is not None else None)
            return live
        except asyncio.CancelledError:
            restore_prior = True  # the person walked away: whatever the unit now runs, the prior stands
            raise
        finally:
            try:
                if restore_prior:
                    if prior is not None:
                        units.write_target(config_dir, prior)
                    else:
                        units.remove_target(config_dir)
                else:
                    units.write_target(config_dir, name)
            except OSError:
                log.debug("could not settle unit-target", exc_info=True)
                try:  # an unlink needs no space: the in-flight form must not outlive this call
                    units.remove_target(config_dir)
                except OSError:
                    log.debug("could not remove unit-target", exc_info=True)

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        """Stop the launch `config_dir`'s pidfile names, by the stopper its OWNER calls for (D-21, START-10).

        The unit is asked first (a fresh state, off the event loop), then `procfs.read_launch(expect=,
        main_pid=)` gives the owner: `unit` only for a process in `localharness-llm.service`'s cgroup that
        is the fresh MainPID with the pidfile's start time (no MainPID makes every launch `spawned`). No
        live launch: nothing. Owner `spawned`, outside the unit's cgroup: the wrapped strategy's own verified
        stop, untouched. Owner `unit`: while the unit is still the harness's own (the identity tests re-run now, D-13), `reset-failed`
        then `stop` through the manager, and the pid must be gone afterwards, else RuntimeError naming it; a
        unit that fails the re-check, or a unit process that is not this record's launch, is named in a
        RuntimeError and nothing is signalled. The cgroup line decides before the owner verdict does: a
        launch whose process sits in the harness unit's cgroup is never handed to the wrapped stopper, whatever
        the manager says (WR-06, WR-09). With the manager unable to answer, or naming another pid as the unit's
        main process, it is named in a RuntimeError (`UNIT_UNASKED_STOP_LINE`) and not signalled; with the
        manager answering that the unit has no main process (`MainPID=0`: failed, between restarts), it is
        stopped through the unit like an owned launch, the same checks and the same proof the pid is gone. A
        launch outside the unit's cgroup still goes to the wrapped stopper. `server.stop_server` is never
        called for a launch in the unit's cgroup (the unit's stop removes it) and `unit-target` is not
        written (RULING W3)."""
        raw = await asyncio.to_thread(units.unit_runtime_state)
        failure = await asyncio.to_thread(units.harness_unit_failure, raw, config_dir)
        launch = await asyncio.to_thread(
            procfs.read_launch, config_dir, expect=self.launch_identity(spec),
            main_pid=getattr(raw, "main_pid", None))
        if launch is None:
            return
        if launch.owner != "unit":
            found = await asyncio.to_thread(procfs.read_process, launch.pid)
            if getattr(found, "unit", None) != procfs.HARNESS_UNIT:
                await self._inner.stop(spec, config_dir)
                return
            # In the unit's cgroup but not its MainPID (or no MainPID known): the unit's process all the same.
            # Only a manager that answered "no main process" lets it be stopped, and then through the manager.
            if raw is None or getattr(raw, "main_pid", None) is not None:
                raise RuntimeError(UNIT_UNASKED_STOP_LINE.format(pid=launch.pid, unit=procfs.HARNESS_UNIT))
        if failure is not None:
            raise RuntimeError(UNIT_FOREIGN_STOP_LINE.format(pid=launch.pid, why=failure))
        if not (launch.verified or launch.starting):
            found = await asyncio.to_thread(procfs.read_process, launch.pid)
            exe = os.path.basename(found.exe) if isinstance(found, procfs.ProcRecord) and found.exe else "?"
            raise RuntimeError(server.NOT_A_LAUNCH_LINE.format(
                pid=launch.pid, exe=exe, pidfile=server.pid_path(config_dir)))
        start = launch.start
        if start is None:
            before = await asyncio.to_thread(procfs.read_process, launch.pid)
            start = before.start_time if isinstance(before, procfs.ProcRecord) else None
        for verb in ("reset-failed", "stop"):
            done = await asyncio.to_thread(units.systemctl_user, verb, procfs.HARNESS_UNIT)
            if done is None or done.code != 0:
                raise RuntimeError(f"{verb} of {procfs.HARNESS_UNIT} failed: {_manager_words(done, None)}")
        after = await asyncio.to_thread(procfs.read_process, launch.pid)
        if isinstance(after, procfs.ProcRecord) and (start is None or after.start_time == start):
            raise RuntimeError(
                f"pid {launch.pid} still alive after stop of {procfs.HARNESS_UNIT} — refusing to report "
                "the accelerator free while the process still holds it")

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        return self._inner.recovery_config_dir(config_dir)

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        return self._inner.liveness(spec, config_dir)

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity | None:
        return self._inner.launch_identity(spec)


GPU_FREE_SETTLE_SECONDS = 3.0
"""Brief pause after a verified-stop before launching the next heavy server. On the DGX Spark's
UNIFIED memory (119 GiB shared CPU+GPU; no discrete VRAM — `nvidia-smi` reports memory N/A here),
the verified-stop (process/container GONE) IS the accelerator-free guarantee: the kernel reclaims
the pages on process exit. This settle is cheap insurance against a reclaim-vs-mmap race under
memory pressure, NOT the gate itself. Tune from live heavy-swap data."""


async def free_accelerator(
    incumbent: tuple[ManagedServerConfig, str] | None,
    target: ManagedServerConfig,
    config_dir: Path,
    *,
    settle_seconds: float | None = None,
    unit: "units.UnitRuntimeState | None" = None,
) -> tuple[ManagedServerConfig, str] | None:
    """GPU-lock: make the single accelerator free for `target`, honoring the box rule "at most one
    GPU-heavy server up at a time".

    If a DIFFERENT heavy server holds the accelerator (`incumbent` is not None and is not `target`
    itself), verified-STOP it through its own strategy — the stop contract (process/container gone)
    IS the accelerator-free guarantee on unified memory — then settle briefly, and RETURN the
    stopped ``(spec, base_url)`` so a failed launch can restore it. Returns ``None`` (no action)
    when nothing heavy is up, or when the incumbent IS the target (a same-server model-reload
    restart, whose stop the caller performs itself). This function only ever FREES — it never starts
    or probes anything.

    ``unit`` is the autostart unit's own state (D-19): the incumbent is stopped by the stopper its OWNER
    calls for (`UnitStrategy.stop`), so a server the unit runs is stopped through the unit and a server
    the harness spawned is stopped by the spawned strategy's verified stop."""
    if incumbent is None:
        return None
    inc_spec, _inc_url = incumbent
    if inc_spec is target:
        return None
    await strategy_for(inc_spec, unit=unit).stop(inc_spec, config_dir)
    # Read the settle from the module global at CALL time (not a def-time default) so it stays
    # tunable from live heavy-swap data and silenceable in tests.
    await asyncio.sleep(GPU_FREE_SETTLE_SECONDS if settle_seconds is None else settle_seconds)
    return incumbent


# ---------------------------------------------------------------------------
# The swap: free the accelerator, launch the target, restore on failure (spec 20 §8.2)
# ---------------------------------------------------------------------------
SWAP_FREE_FAILED = ("Could not free the GPU for {name}: {exc}. No change made — the current model "
                    "should still be serving.")
SWAP_LAUNCH_FAILED = "Launch of {name} failed: {exc}{tail}"
SWAP_RESTORED_TAIL = " Restored {label}."
SWAP_NOT_RESTORED_TAIL = " Could not restore it: {why}"
SWAP_NOTHING_SERVING_TAIL = " The box now has NO model serving — manual attention needed."
SWAP_BELOW_FLOOR = "{name} served {window}, below the {floor} the harness needs"
SWAP_REFUSED = ("Did not start {name}: {why}. No change made — the current model should still be "
                "serving.")


@dataclass(frozen=True)
class Heavy:
    """One GPU-heavy server a swap stops or starts: its record `spec`, the `base_url` it answers at,
    the configured `peer` name (None for the primary) and the `launch_kw` its launch carries (the
    key, `{"api_key": …}`, only when the record requires one). The CALLER builds it (the provider
    layer imports no cli): `model_ops.heavy_for` from a trust-resolved `Serving`,
    `model_ops.incumbent_heavy` for what runs now. `launch_kw` is a read-only copy, left out of the
    hash and of the repr (it holds the raw key)."""
    spec: ManagedServerConfig
    base_url: str
    peer: str | None = None
    launch_kw: Mapping[str, str] = field(default_factory=dict, hash=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "launch_kw", MappingProxyType(dict(self.launch_kw)))


def heavy_label(heavy: Heavy) -> str:
    """What a swap line calls `heavy`: its peer name, else the name it serves, else its model."""
    return heavy.peer or getattr(heavy.spec, "served_name", None) or heavy.spec.model


class SwapReporter(Protocol):
    """What a surface says while a swap runs. `swap` awaits `freed` after the accelerator is free
    (every time the free succeeds, and again after a cancellation stopped the target), `launched`
    after the target answered, `restored` after the incumbent was relaunched, and `line` for the two
    failure sentences it composes (already redacted)."""

    async def freed(self) -> None: ...

    async def launched(self, live: LiveEndpoint) -> None: ...

    async def restored(self, spec: ManagedServerConfig) -> None: ...

    async def line(self, text: str) -> None: ...


@dataclass(frozen=True)
class SwapOutcome:
    """How a swap ended, for a failure that was handled: `live` the target's endpoint (None unless it
    came up), `stopped` the incumbent the swap freed, `restored` that incumbent when it came back,
    `error` the failure — a NEW exception whose text is the redacted one, with no cause and no
    context — and `stage` where it happened ("refused" before anything was stopped, "free" or "launch";
    None on success)."""
    live: LiveEndpoint | None
    stopped: Heavy | None
    restored: Heavy | None
    error: BaseException | None
    stage: str | None


def _redacted_error(exc: BaseException, redact: Callable[[str], str]) -> BaseException:
    """A new exception of `exc`'s type (a subclass whose constructor will not take one message falls
    back to its RuntimeError / TimeoutError base) whose text is `redact(str(exc))`. It is never
    raised, so it has no traceback; `__cause__` is None and `__suppress_context__` is True with no
    `__context__`, so nothing reachable from it holds the original (the raw tail) (D-22, SEC-04)."""
    text = redact(str(exc))
    try:
        new = type(exc)(text)
    except Exception:  # noqa: BLE001 — a subclass with its own constructor
        new = (TimeoutError if isinstance(exc, TimeoutError) else RuntimeError)(text)
    new.__cause__ = None
    new.__context__ = None
    new.__suppress_context__ = True
    return new


async def safe_stop(
    spec: ManagedServerConfig, config_dir: Path, *, unit: "units.UnitRuntimeState | None" = None,
) -> None:
    """Best-effort verified-stop of a server, swallowing errors. Used to tear a HALF-STARTED target
    down before restoring an incumbent: the pidfile is one-per-config_dir (shared across primary and
    peer), so a still-alive orphan would be clobbered-and-lost by the restore's own launch
    (server.py start_server overwrites the pidfile) — leaving a second, untracked heavy.

    Only the TARGET's own launch is torn down: the strategy's stop proves a binary and a port, which a
    second record on that binary and port shares. So when the pidfile names a live process that is
    verifiably not this RECORD's (`record_identity`: its served name or model), nothing half-started
    here is that process, and it is left running."""
    try:
        identity = record_identity(spec)
        if identity is not None:
            launch = await asyncio.to_thread(procfs.read_launch, config_dir, expect=identity)
            if launch is not None and not launch.verified:
                return
    except Exception:  # noqa: BLE001 — a look that fails changes nothing: the strategy's own stop decides
        pass
    try:
        await strategy_for(spec, unit=unit).stop(spec, config_dir)
    except Exception:  # noqa: BLE001 — cleanup is best-effort
        pass


async def restore_incumbent(
    stopped: Heavy | None, *, config_dir: Path, autosense: "AutosenseConfig | None",
    unit: "units.UnitRuntimeState | None" = None,
) -> tuple[str | None, str | None]:
    """Best-effort re-launch of a heavy incumbent stopped for a swap that then FAILED, so the box is
    never left with nothing serving. Returns `(label, note)` on success — `note` is the launch
    decision's one line when it withheld a widening — and `(None, why)` when the decision refuses the
    relaunch or the launch itself fails, so the caller reports the cause instead of "nothing serving"
    (D-08). `(None, None)` is nothing to restore: the caller then reports the box needs manual
    attention. Takes no launch lock and flushes no line: the caller holds the lock around the whole
    swap (D-21). Under the harness's own unit (``unit``) the incumbent comes back through the unit, so
    ``unit-target`` is rewritten to it."""
    if stopped is None:
        return None, None
    try:
        live = await strategy_for(stopped.spec, unit=unit).activate(
            stopped.spec, config_dir, stopped.base_url, peer=stopped.peer, autosense=autosense,
            **stopped.launch_kw)
        return stopped.spec.model, getattr(live, "note", None)
    except server.LaunchRefused as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — restore is best-effort; the caller surfaces the failure
        # The same (None, why) shape as a refusal, so the caller's "Could not restore it: <why>"
        # names the cause (a bad binary, a timeout, a port clash) instead of hiding it. The activate
        # contract's own failures (RuntimeError / TimeoutError) are already one readable line;
        # anything else keeps its type name so it is not an empty or cryptic string.
        text = str(exc)
        if isinstance(exc, (RuntimeError, TimeoutError)) and text:
            return None, text
        return None, f"{type(exc).__name__}: {text}" if text else type(exc).__name__


async def _roll_back(
    target: Heavy, stopped: Heavy | None, reporter: SwapReporter, config_dir: Path,
    autosense: "AutosenseConfig | None", unit: Any,
) -> tuple[str | None, str | None]:
    """Stop the target, then put the incumbent back; `restore_incumbent`'s `(label, why)`."""
    await safe_stop(target.spec, config_dir, unit=unit)
    label, why = await restore_incumbent(stopped, config_dir=config_dir, autosense=autosense, unit=unit)
    if label is not None and stopped is not None:
        await reporter.restored(stopped.spec)
    return label, why


def own_launch(config_dir: Path, specs: "Iterable[ManagedServerConfig | None]") -> procfs.Launch | None:
    """The harness's verified live launch of one of `specs` under `config_dir`, if any."""
    launches = (procfs.read_launch(config_dir, expect=record_identity(spec)) for spec in specs if spec is not None)
    return next((launch for launch in launches if launch is not None and launch.verified), None)


def memory_after_launch(
    autosense: "AutosenseConfig", launch: procfs.Launch | None, *, stopping: bool = False,
) -> "_auto.FreeMemory | None":
    """Free memory once `launch` is stopped; with none, the probe itself, or the card less 1 GiB when `stopping` an incumbent anyway."""
    free = _auto.free_memory(autosense)
    if launch is None and not stopping:
        return free
    return _auto.memory_after_stop(free, None if launch is None else _auto.held_by(launch.pid))


def swap_refusal(
    target: Heavy, incumbent: Heavy | None, *, config_dir: Path, autosense: "AutosenseConfig | None",
    floor: int | None, unit: Any = None,
) -> str | None:
    """The target's plan refusal against the memory the swap will have, naming the incumbent that keeps serving."""
    spec = target.spec
    if autosense is None or not getattr(strategy_for(spec, unit=unit), "plans_launch", False):
        return None
    keyed = bool(target.launch_kw.get("api_key"))
    decision = server.launch_decision(spec, keyed, peer=target.peer)
    if decision.refusal is not None:
        return None
    seen = {}
    if incumbent is not None:
        seen["memory"] = memory_after_launch(autosense, own_launch(config_dir, [incumbent.spec]), stopping=True)
    try:
        plan_launch(spec, sense_launch(spec, config_dir, autosense, **seen), decision,
                    autosense=autosense, api_key_held=keyed, floor=floor)
    except server.LaunchRefused as exc:
        return f"{exc} {heavy_label(incumbent)} keeps serving." if incumbent is not None else str(exc)
    return None


async def swap(
    incumbent: Heavy | None,
    target: Heavy,
    *,
    config_dir: Path,
    autosense: "AutosenseConfig | None",
    unit: Any = None,
    reporter: SwapReporter,
    redact: Callable[[str], str],
    on_poll: Callable[[float], None] | None = None,
    on_plan: "Callable[[LaunchPlan | None], None] | None" = None,
    timeout_seconds: float = 1800.0,
    floor: int | None = None,
) -> SwapOutcome:
    """Free the accelerator of `incumbent`, launch `target`, tell `reporter`; a failed launch stops the
    half-started target and restores the incumbent (the REPL's cold branch, moved here so every
    surface runs the same sequence — spec 20 §8.2).

    The CALLER holds the launch lock around the whole call (this module is where the strategies
    define `activate`; the lock is the caller's, D-21) and builds both `Heavy` values.

    A failure comes back as a `SwapOutcome` (`stage` "free" or "launch"), never raised. `redact` is
    required: `outcome.error` is rebuilt from `redact(str(exc))` and every line handed to
    `reporter.line` goes through it, so nothing `swap` returns or says carries the raw tail (D-22).
    A cancellation after the incumbent was freed stops the target (verified), awaits `freed()`, runs
    no restore and re-raises (D-20).

    A target the plan refuses is reported with the incumbent still serving before anything
    stops (`stage` "refused"); with `floor`, a launch that serves a window below it is a failed launch.

    ``unit`` is the autostart unit's state when it is the harness's OWN unit (`harness_unit`; anything
    else behaves as None). With it, every strategy `swap` builds is the unit's for a record the unit
    serves: the incumbent is freed by its owner's stopper, the target is launched by restarting the
    unit, a half-started target is stopped the same way, and the restore re-activates the incumbent
    through the unit, which rewrites `unit-target` to it (D-19, D-21). `swap` itself never writes
    `unit-target` (RULING W3: only `UnitStrategy.activate` does); it only takes it away, through
    `leave_unit`, when the target is a record the unit cannot serve, right before that target's own
    launch (R-14-28)."""
    # Before the incumbent is stopped: a launch through the unit that is known to fail (a peer name the
    # unit target cannot hold, an `autostart.env` the record does not cover) is refused with the incumbent
    # still serving, so the person never pays two model loads for a swap that could not succeed. The way
    # back is judged the same way: a target outside the unit that fails to launch restores the incumbent
    # through it.
    refusal = await asyncio.to_thread(unit_launch_refusal, target.spec, target.peer, unit, config_dir)
    if refusal is None and incumbent is not None and incumbent.spec is not target.spec:
        # The failed launch of a target outside the unit restores the incumbent through it, and `serve` meets
        # the same exit 78 there: judge the way back before the way out (WR-10).
        way_back = await asyncio.to_thread(unit_launch_refusal, incumbent.spec, incumbent.peer, unit, config_dir)
        if way_back is not None:
            refusal = UNIT_RESTORE_LINE.format(label=heavy_label(incumbent), why=way_back)
    if refusal is not None:
        await reporter.line(redact(SWAP_REFUSED.format(name=heavy_label(target), why=refusal)))
        return SwapOutcome(None, None, None, _redacted_error(RuntimeError(refusal), redact), "refused")
    unplanned = await asyncio.to_thread(
        swap_refusal, target, incumbent, config_dir=config_dir, autosense=autosense, floor=floor, unit=unit)
    if unplanned is not None:
        await reporter.line(redact(unplanned))
        return SwapOutcome(None, None, None, _redacted_error(RuntimeError(unplanned), redact), "refused")
    try:
        freed = await free_accelerator(
            (incumbent.spec, incumbent.base_url) if incumbent is not None else None,
            target.spec, config_dir, unit=unit)
    except (RuntimeError, TimeoutError) as exc:
        # The incumbent would not stop → we launched NOTHING and it is still up (the verified-stop
        # only raises when the server survives).
        await reporter.line(redact(SWAP_FREE_FAILED.format(name=heavy_label(target), exc=exc)))
        return SwapOutcome(None, None, None, _redacted_error(exc, redact), "free")
    stopped = incumbent if freed is not None else None
    await reporter.freed()
    try:
        leave_unit(target.spec, unit, config_dir)
        live = await strategy_for(target.spec, unit=unit).activate(
            target.spec, config_dir, target.base_url, timeout_seconds=timeout_seconds,
            on_poll=on_poll, peer=target.peer, autosense=autosense, on_plan=on_plan, floor=floor,
            **target.launch_kw)
    except asyncio.CancelledError:
        # Nothing half-started stays behind, and nothing is restored: the person walked away.
        await safe_stop(target.spec, config_dir, unit=unit)
        await reporter.freed()
        raise
    except (RuntimeError, TimeoutError) as exc:
        # The target half-started (a TimeoutError means its process is still ALIVE, per server.py
        # wait_ready). Tear it down FIRST — the shared pidfile means the restore's own launch would
        # otherwise clobber its handle and orphan it as a 2nd heavy — THEN restore the incumbent.
        label, why = await _roll_back(target, stopped, reporter, config_dir, autosense, unit)
        tail = (SWAP_RESTORED_TAIL.format(label=label) + (f" {why}" if why else "") if label
                else SWAP_NOT_RESTORED_TAIL.format(why=why) if why
                else SWAP_NOTHING_SERVING_TAIL if stopped is not None
                else "")
        await reporter.line(redact(SWAP_LAUNCH_FAILED.format(name=heavy_label(target), exc=exc, tail=tail)))
        return SwapOutcome(None, stopped, stopped if label is not None else None,
                           _redacted_error(exc, redact), "launch")
    served = live.window.served if live.window is not None else None
    if floor is not None and served is not None and served < floor:
        head = SWAP_BELOW_FLOOR.format(name=heavy_label(target), window=_auto.window_phrase(served), floor=f"{floor:,}")
        label, why = await _roll_back(target, stopped, reporter, config_dir, autosense, unit)
        tail = (f" — {heavy_label(stopped)} serves again." if label is not None and stopped is not None
                else f". Could not restore it: {why}" if why else " — it was stopped.")
        await reporter.line(redact(head + tail))
        return SwapOutcome(None, stopped, stopped if label is not None else None,
                           _redacted_error(RuntimeError(head), redact), "launch")
    await reporter.launched(live)
    return SwapOutcome(live, stopped, None, None, None)


# ---------------------------------------------------------------------------
# The cross-process launch lock
# ---------------------------------------------------------------------------
LAUNCH_LOCK_WAIT_LINE = "another LocalHarness is launching the model server — waiting"
"""What the person is told, once, when the launch lock is still held by someone else after the
queue-visibility threshold. The REPL and `start` print it through `launch_lock(on_wait=...)`."""

LAUNCH_LOCK_LOCAL_WAIT_LINE = "a model server launch in this session is still finishing — waiting"
"""The same moment when the holder is a task of THIS process (a relaunch still unwinding after Esc, or
running for another caller): "another LocalHarness" would be false."""


def launch_lock_wait_line(config_dir: Path) -> str:
    """The wait line for `config_dir`'s launch lock: the local line when a task of this process holds
    it, the cross-process line otherwise."""
    held = _HELD.get(server.lock_path(config_dir).resolve())
    return LAUNCH_LOCK_LOCAL_WAIT_LINE if held is not None else LAUNCH_LOCK_WAIT_LINE

_HELD: dict[Path, list] = {}
"""Resolved lock path -> `[owning asyncio task, depth, fd | None, lent]`. Re-entrancy is per TASK identity
(`asyncio.current_task()`), never a ContextVar: `create_task` copies the context, so a task spawned
inside the lock would otherwise believe it holds it. `fd` is the holder's flock descriptor (None on the
in-process fallback) and `lent` is True only while `lend_launch_lock` has the flock released for the
autostart unit's `serve` (R-14-09). The entry names the OWNING task for the whole life of a lock, loan
included: another task of this process waits while any entry names a different task (R-14-36), so the
flock being released during a loan never lets it overwrite or pop the lender's entry. Entries exist only
while a lock is held and are removed on every exit path."""

_CONTENDED = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES})
"""The `flock(LOCK_NB)` errnos that mean "another holder has it". Any other errno is a filesystem that
cannot lock at all, and the lock degrades to the in-process one instead of polling forever."""

_FALLBACK_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)
"""The in-process stand-in when the flock is unavailable, keyed by running loop because an
`asyncio.Lock` binds to the loop it is first contended in (one loop per test, one per process)."""

LAUNCH_LOCK_TIMEOUT_LINE = "another LocalHarness is launching the model server — gave up after {seconds}s"
"""What `launch_lock_sync` raises (as a `TimeoutError`) when the lock stays held past its timeout."""

LAUNCH_LOCK_LENT_LINE = "the launch lock is lent to the autostart unit — a launch inside the loan would race it"
"""What a nested `launch_lock` by the lender raises while its loan is open (R-14-36)."""


def _in_process_lock(key: Path, cause: str) -> asyncio.Lock:
    """The in-process lock for `key`; the first use of it on this loop logs the one warning that names
    why the cross-process lock is off."""
    locks = _FALLBACK_LOCKS.setdefault(asyncio.get_running_loop(), {})
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
        log.warning("launch lock unavailable (%s) — cross-process launch exclusion disabled", cause)
    return lock


@contextlib.asynccontextmanager
async def launch_lock(config_dir: Path, *, on_wait: Callable[[], Awaitable[None]] | None = None):
    """Hold the cross-process launch lock for `config_dir` (D-26).

    One advisory `flock(LOCK_EX | LOCK_NB)` on `server.lock_path(config_dir)`, an empty 0600 file
    beside `server.pid`, polled every 50 ms with `asyncio.sleep` (never a thread parked on the lock:
    see `_inference_gate` for how that wedged) and released by closing the fd. The kernel also releases
    it when a holder process dies, so there is never a stale lock to clear (D-28). It has no timeout of
    its own: a holder's `activate` is bounded by its own `timeout_seconds`, and `on_wait` tells the
    person once, after `provider.client._QUEUE_VISIBILITY_SECONDS`, that someone else is launching.

    Who takes it: every launch through `activate` from the harness's own code (a structural test pins
    that). It is re-entrant per owning asyncio task (D-47): a nested acquire by the holder is free,
    any other task waits, for as long as `_HELD` names a different task, whether or not the flock is
    lent at that moment (R-14-36). A nested acquire by the holder WHILE it has lent the flock raises
    `RuntimeError(LAUNCH_LOCK_LENT_LINE)`: a loan's block must not launch. The fd is not inheritable,
    so the long-lived server spawned inside the lock never holds it. With no `fcntl`, a lock path that
    cannot be opened (a planted symlink is refused by `O_NOFOLLOW`), or a filesystem whose `flock`
    fails with anything but contention, launches serialise through an in-process lock and one warning
    names the cause. The file is not a protected config-dir entry (D-29): a replaced file costs only the
    exclusion, which the pidfile re-check still narrows.

    `launch_lock_sync` is the same lock for a synchronous caller (`serve`), and `lend_launch_lock`
    lets this lock's holder hand the flock to that caller for a while (spec 20 §9.3).
    """
    from localharness.provider import client as _client

    path = server.lock_path(config_dir)
    key = path.resolve()
    me = asyncio.current_task()
    held = _HELD.get(key)
    if held is not None and held[0] is me:  # nested acquire by the owner: nothing to take
        if held[3]:
            raise RuntimeError(LAUNCH_LOCK_LENT_LINE)
        held[1] += 1
        try:
            yield
        finally:
            held[1] -= 1
        return

    fd: int | None = None
    cause = "no fcntl on this platform"
    if fcntl is not None:
        try:
            server.server_dir(config_dir).mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            cause = str(exc)
    fallback: asyncio.Lock | None = None
    got_fallback = False
    entry: list | None = None
    try:
        started = time.monotonic()
        told = False

        async def pause() -> None:
            """One 50 ms wait, and the once-only "someone else is launching" announcement."""
            nonlocal told
            await asyncio.sleep(0.05)
            if (on_wait is not None and not told
                    and time.monotonic() - started >= _client._QUEUE_VISIBILITY_SECONDS):
                told = True
                try:
                    await on_wait()
                except Exception:  # noqa: BLE001 - telling the person never aborts the launch
                    log.debug("launch lock: the wait announcement failed", exc_info=True)

        while fd is not None:
            if key in _HELD:  # another task of this process owns it (lent or not): wait for it first
                await pause()
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in _CONTENDED:
                    # Not "someone else holds it": this filesystem cannot flock (NFS mounted
                    # nolock, some FUSE or overlay mounts). Every retry would fail the same way,
                    # so degrade exactly as an unopenable lock file does.
                    os.close(fd)
                    fd = None
                    cause = f"flock failed: {exc}"
                    break
                await pause()
        if fd is None:
            fallback = _in_process_lock(key, cause)
            await fallback.acquire()
            got_fallback = True
        # No await between the successful flock (or the fallback acquire) and this assignment, and it is
        # written only when no entry exists: it can neither overwrite nor later pop another task's.
        if key not in _HELD:
            entry = _HELD[key] = [me, 1, fd, False]
        try:
            yield
        finally:
            if entry is not None and _HELD.get(key) is entry:
                _HELD.pop(key, None)
    finally:
        if got_fallback and fallback is not None:
            fallback.release()
        if fd is not None:
            os.close(fd)  # releases the flock; the kernel also does when a holder dies


@contextlib.contextmanager
def launch_lock_sync(config_dir: Path, *, timeout_seconds: float):
    """The launch lock for a synchronous caller: `serve` takes it with no event loop (spec 20 §9.3, D-14,
    "one lock, two entry points" with `launch_lock`).

    The SAME advisory `flock(LOCK_EX | LOCK_NB)` on `server.lock_path(config_dir)` with the same
    `_CONTENDED` errnos, polled every 50 ms with `time.sleep`. It raises `TimeoutError(LAUNCH_LOCK_TIMEOUT_LINE)`
    when the lock is still held after `timeout_seconds`. A lock file that cannot be opened (a planted
    symlink is refused by `O_NOFOLLOW`), no `fcntl`, or a filesystem whose `flock` fails with anything
    but contention gives no cross-process lock, with one warning naming the cause: the pidfile guard
    still refuses to replace a live launch. The fd is not inheritable and is closed on every exit path,
    so the server `serve` execs never holds the lock."""
    path = server.lock_path(config_dir)
    fd: int | None = None
    cause = "no fcntl on this platform"
    if fcntl is not None:
        try:
            server.server_dir(config_dir).mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        except OSError as exc:
            cause = str(exc)
    try:
        if fd is not None:
            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in _CONTENDED:
                        os.close(fd)
                        fd = None
                        cause = f"flock failed: {exc}"
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(LAUNCH_LOCK_TIMEOUT_LINE.format(seconds=int(timeout_seconds))) from None
                    time.sleep(0.05)
        if fd is None:
            log.warning("launch lock unavailable (%s) — cross-process launch exclusion disabled", cause)
        yield
    finally:
        if fd is not None:
            os.close(fd)


@contextlib.asynccontextmanager
async def lend_launch_lock(config_dir: Path):
    """Let the task that holds `launch_lock` release the flock for the length of a block and take it back
    afterwards (spec 20 §9.3, R-14-09, R-14-36).

    `UnitStrategy.activate` runs inside its caller's launch lock (the v0.18 structural rule) while the
    unit's `serve` must take that same lock to write the pidfile: without a loan the two would wait on
    each other until the 30-minute timeout. Inside the block the flock is free, so `serve` (a
    `launch_lock_sync` in another process) can take it; afterwards the lender polls the same fd back
    every 50 ms, so a second holder taking the lock during the loan makes the reclaim wait, never break.
    Correctness against that second launcher is the live-pidfile guard: whoever takes the flock second
    and finds a live pidfile refuses, so no pidfile is replaced while its process lives.

    The loan has no clock of its own: the block decides when it ends (the caller ends it on an event).
    `_HELD[key]` keeps naming the lending task and is marked lent, so another task of this process still
    waits and the lender's nested `launch_lock` raises `LAUNCH_LOCK_LENT_LINE`. A task that does not hold
    the lock, a lock on the in-process fallback and a nested loan are no-ops."""
    key = server.lock_path(config_dir).resolve()
    entry = _HELD.get(key)
    if (fcntl is None or entry is None or entry[0] is not asyncio.current_task()
            or entry[2] is None or entry[3]):
        yield
        return
    fd = entry[2]
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError as exc:
        log.warning("launch lock: could not lend it (%s) — the autostart unit may wait", exc)
        yield
        return
    entry[3] = True
    try:
        yield
    finally:
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in _CONTENDED:
                        log.warning("launch lock: could not take it back (%s) — continuing without the flock", exc)
                        break
                    await asyncio.sleep(0.05)
        finally:
            entry[3] = False


# ---------------------------------------------------------------------------
# Waiting for a dropped model server, and relaunching it once
# ---------------------------------------------------------------------------
_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("lh_relaunch_deadline", default=None)
"""The monotonic time the current recovery flight's bound ends. Set INSIDE the flight task, so it is
visible to the `relaunch` the flight awaits and to nothing the caller of `recover` runs."""

_RELAUNCH_DEFAULT_SECONDS = 1800.0  # `activate`'s own default timeout, used outside any flight

_UNWINDING: set[asyncio.Task] = set()
"""Detached flights still unwinding after their last awaiter left: asyncio keeps only weak references
to tasks, and the `finally` that releases a launch lock must get to run."""

BOUND_TEXT = "the model server did not come back within {seconds}s"
EXITED_TEXT = "the model server exited. Log tail:\n"
EXITED_AGAIN_TEXT = "the model server exited; it was restarted once and exited again. Log tail:\n"
RESTART_FAILED_TEXT = "the model server exited; it was restarted once and the restart failed: "
"""The four sentences a recovery ends with when it fails; the loop appends them to the turn's ending."""


def relaunch_budget() -> float:
    """Seconds left in the current recovery flight's bound, never below 1.0; the zero-argument
    `relaunch` passes this as `activate`'s `timeout_seconds` (D-22). Outside any flight it is
    `activate`'s own default, 1800.0."""
    deadline = _DEADLINE.get()
    if deadline is None:
        return _RELAUNCH_DEFAULT_SECONDS
    return max(1.0, deadline - time.monotonic())


class _Run:
    """One flight: its task, how many callers are awaiting it, when it started and the deadline it runs
    to (the outage's start plus the configured bound)."""

    __slots__ = ("task", "awaiters", "outage", "started", "deadline")

    def __init__(self, task: asyncio.Task, outage: "_Outage | None" = None,
                 deadline: float = float("inf")) -> None:
        self.task = task
        self.awaiters = 0
        self.outage = outage
        self.started = time.monotonic()
        self.deadline = deadline


class _Outage:
    """A drop that no flight has ended yet: what it has already spent. It is opened by the first flight
    for a drop and kept on the pair after a flight that failed (or whose last awaiter left), so a loop
    that runs after it in the same root turn (the root after its child, a web retry, the next reduce
    level) continues the same outage instead of starting a whole new bound and another relaunch. It is
    dropped when a flight ends well, when a call to the server goes through, or when the person starts a
    new root turn (`forget_outages`)."""

    __slots__ = ("started", "relaunched", "announced_until", "failed")

    def __init__(self, started: float) -> None:
        self.started = started
        self.relaunched = False
        # when the outage's one announcement said the wait ends (monotonic): its configured bound's end
        self.announced_until: float | None = None
        # how the last flight of this outage failed: its exception TYPE and its text with URL credentials
        # removed, never the exception itself (its traceback holds the flight's frame and the raw key)
        self.failed: tuple[type[BaseException], str] | None = None


class _Flight:
    """The mutable single-flight slot inside a frozen `ServerRecovery`: the flight now under way, if any,
    and the outage it belongs to."""

    __slots__ = ("current", "outage", "__weakref__")

    def __init__(self) -> None:
        self.current: _Run | None = None
        self.outage: _Outage | None = None


_OPEN_OUTAGES: "weakref.WeakKeyDictionary[_Flight, str | None]" = weakref.WeakKeyDictionary()
"""Every pair slot with an outage on record, mapped to its pair's client address (None for a pure poll
pair). Weak: a pair the REPL's memo dropped takes its record with it."""


def forget_outages(base_url: str | None = None, *, every: bool = False) -> None:
    """End the outage record of every pair for `base_url` (and of the address-less pairs): a call to that
    server has just gone through, so the next drop is a new one with its own whole bound, relaunch and
    announcement. `every=True` ends them all: the person started a new root turn, and a flight that
    nobody awaits any more (every caller left at its own budget) is cancelled with it. A flight that
    still has awaiters keeps running; only the record is dropped."""
    for slot, pair_url in list(_OPEN_OUTAGES.items()):
        if every or pair_url is None or (
                base_url is not None and pair_url.rstrip("/") == base_url.rstrip("/")):
            slot.outage = None
            _OPEN_OUTAGES.pop(slot, None)
            if every:
                _cancel_if_idle(slot)


def abandon_idle_flights() -> None:
    """Cancel every flight nobody awaits any more (each caller left at its own budget), keeping the outage
    records. Called when a root turn ends, however it ends (Esc included), and when a KILL file stops a
    loop: no relaunch may start, and no launch lock stay held, with no turn waiting on it (D-17, IN-58).
    A flight that still has an awaiter is left alone."""
    for slot in list(_OPEN_OUTAGES):
        _cancel_if_idle(slot)


def _cancel_if_idle(slot: "_Flight") -> None:
    run = slot.current
    if run is not None and run.awaiters == 0 and not run.task.done():
        slot.current = None  # detach before cancel, as recover does for its last awaiter
        _UNWINDING.add(run.task)
        run.task.add_done_callback(_UNWINDING.discard)
        run.task.cancel()


def _settle(slot: _Flight, outage: _Outage, task: asyncio.Task) -> None:
    """A flight ended: a success ends its outage; a failure is kept on it for the loops still to come."""
    if task.cancelled():
        return
    failed = task.exception()
    if failed is None:
        if slot.outage is outage:
            slot.outage = None
            _OPEN_OUTAGES.pop(slot, None)
    else:
        outage.failed = (type(failed), server.strip_url_credentials(str(failed)))


def _again(failed: tuple[type[BaseException], str]) -> BaseException:
    """A fresh exception of an outage's recorded failure type and text, raised to a later loop."""
    kind, text = failed
    try:
        return kind(text)
    except Exception:  # noqa: BLE001 - a type that cannot be rebuilt still reads as its text
        return RuntimeError(text)


@dataclass(frozen=True)
class ServerRecovery:
    """What to do when the model server a session talks to drops: wait for it, and relaunch it once if
    this harness launched it and it died. Once per server death per session (D-16).

    `base_url` is the client address the pair was built for (the asking loop compares it with its own
    client's address before using the pair). `config_dir` is the strategy's own config dir
    (`recovery_config_dir`), or None for a pure poll (LM Studio, a hand-run server): with it a dead
    process is known at once and its log tail is shown. `relaunch` is the zero-argument coroutine the
    REPL builds over the active spec, or None when nothing can be relaunched; it reads its time bound
    from `relaunch_budget()`. `runtime` names the server in `wait_ready`'s messages.

    `recover` takes the key per call and stores nothing of it: an outage keeps only its failure's type
    and redacted text, never the exception (whose traceback would hold the flight's frame). Callers of one pair share one flight, and
    loops that call it one after another share one outage: a drop no flight could end is continued by
    the next caller within the same bound (measured from the drop), never with a second relaunch or a
    second announcement (the one line carries the configured bound, true for the whole wait); once that
    bound is spent a later caller gets one look at the address and the same failure. A call that goes through, or a new root turn, ends the outage (`forget_outages`).
    """

    base_url: str | None
    config_dir: Path | None
    relaunch: Callable[[], Awaitable["LiveEndpoint | None"]] | None
    runtime: str | None = None
    _flight: _Flight = field(default_factory=_Flight, init=False, compare=False, repr=False)

    async def recover(
        self,
        *,
        base_url: str,
        api_key: Any,
        wait_seconds: float,
        on_lost: Callable[[], Awaitable[None]],
        leave_after: float | None = None,
    ) -> None:
        """Wait until the server at `base_url` answers `/models` (relaunching a dead launched server
        once), within `wait_seconds`. `on_lost` is awaited once if it is still down after the
        queue-visibility threshold. Raises `TimeoutError` / `RuntimeError` / `LaunchRefused` with one
        readable sentence; a cancelled caller leaves, and the last one to leave cancels the flight.

        `wait_seconds` is the configured bound: the flight, and its one relaunch, run to it (measured
        from the drop) whoever started them. `leave_after`, when given and shorter, is how long THIS
        caller waits (a child's remaining duration budget): at that point it leaves with the bound
        sentence for its own seconds, and the flight is NOT cancelled, even when it was the last
        awaiter, so the next loop of the turn (the root) joins the same flight and its one relaunch
        instead of finding the outage spent. A caller's budget never cuts the shared relaunch short."""
        slot = self._flight
        run = slot.current
        if run is None or run.task.done():
            outage = slot.outage
            if outage is None:  # a new drop; otherwise this continues one an earlier loop could not end
                outage = slot.outage = _Outage(time.monotonic())
                _OPEN_OUTAGES[slot] = self.base_url
            task = asyncio.get_running_loop().create_task(
                self._fly(base_url, api_key, wait_seconds, outage)
            )
            task.add_done_callback(_retrieve)
            task.add_done_callback(functools.partial(_settle, slot, outage))
            run = slot.current = _Run(task, outage, outage.started + wait_seconds)
        own = None
        if leave_after is not None and leave_after < wait_seconds:
            own = max(0.0, leave_after)
        # when this caller's wait ends: its own bound, or the flight's, whichever comes first
        mine = run.deadline if own is None else min(time.monotonic() + own, run.deadline)
        timer = asyncio.get_running_loop().create_task(_announce(run, mine, on_lost))
        run.awaiters += 1
        left = False  # this caller left at its own (shorter) bound: the flight goes on for the next one
        try:
            if own is None:
                return await asyncio.shield(run.task)
            window = asyncio.timeout(own)
            try:
                async with window:
                    return await asyncio.shield(run.task)
            except TimeoutError:
                if not window.expired():
                    raise  # the flight's own failure, not this caller's bound
                left = True
                raise TimeoutError(self._bound_text(own, api_key)) from None
        finally:
            timer.cancel()
            run.awaiters -= 1
            if run.task.done():
                if slot.current is run:
                    slot.current = None  # a resolved flight is cleared: the next death is a new flight
            elif run.awaiters == 0 and not left:
                if slot.current is run:
                    slot.current = None  # detach BEFORE cancel, or a recover this tick joins a dying task
                if slot.outage is run.outage:
                    # the person stopped the wait (Esc): nothing it spent, a relaunch cut short included,
                    # carries over to whoever asks next
                    slot.outage = None
                    _OPEN_OUTAGES.pop(slot, None)
                _UNWINDING.add(run.task)
                run.task.add_done_callback(_UNWINDING.discard)
                run.task.cancel()

    async def _fly(
        self,
        base_url: str,
        api_key: Any,
        wait_seconds: float,
        outage: _Outage,
    ) -> None:
        from localharness.provider import client as _client

        deadline = outage.started + wait_seconds  # the bound runs from the drop, not from this caller
        _DEADLINE.set(deadline)
        poll = _client._QUEUE_VISIBILITY_SECONDS
        if deadline - time.monotonic() <= 0 and outage.failed is not None:
            # An earlier loop of this root turn already spent this outage's bound and failed: one look
            # at the address (it may be back), never another whole wait or another relaunch.
            if await self._answers(base_url, api_key, time.monotonic() + poll):
                return
            if issubclass(outage.failed[0], TimeoutError):
                # the bound passed: say THIS loop's bound, not the seconds an earlier loop was told
                raise TimeoutError(self._bound_text(wait_seconds, api_key)) from None
            raise _again(outage.failed) from None
        # what the outage's one relaunch already came to, when an earlier loop spent it and failed
        earlier = outage.failed if outage.relaunched else None
        relaunched = outage.relaunched
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(self._bound_text(wait_seconds, api_key))
            try:
                await server.wait_ready(
                    base_url,
                    config_dir=self.config_dir,
                    timeout_seconds=remaining,
                    poll_seconds=poll,
                    runtime=self.runtime,
                    liveness=True,  # any non-5xx answer is a server that is back (D-14 refined)
                    **server.key_kwargs(api_key),
                )
                return
            except TimeoutError as late:
                raise TimeoutError(self._bound_text(wait_seconds, api_key)) from late
            except RuntimeError as died:  # the launched process is gone (wait_ready's own signal)
                if self.relaunch is None:
                    raise RuntimeError(EXITED_TEXT + self._tail("the model server exited", api_key)) from died
                if relaunched:
                    # The one relaunch is spent (here, or by an earlier loop of the turn), but the person
                    # may have restarted the server by hand: look at the address before giving up.
                    if await self._answers(base_url, api_key, deadline):
                        log.info("model server process exited again, but the address answers")
                        return
                    if earlier is not None and (issubclass(earlier[0], server.LaunchRefused)
                                                or earlier[1].startswith(RESTART_FAILED_TEXT)):
                        raise _again(earlier) from died  # the one relaunch was refused or failed
                    raise RuntimeError(
                        EXITED_AGAIN_TEXT + self._tail("the model server exited again", api_key)) from died
                relaunched = True
            # The launched process is gone, but something may already serve at the address (the
            # person restarted the server by hand): the pidfile check runs before any request, so
            # only a plain probe of the address sees it. Launching onto that port would fail to
            # bind, or worse, pass activate's wait on the other server's answer.
            if await self._answers(base_url, api_key, deadline):
                log.info("model server process exited, but the address answers; not relaunching")
                return
            # Outside the try above: a LaunchRefused (itself a RuntimeError) must never be read as
            # a poll failure.
            log.info("model server process exited; relaunching it once")
            # The relaunch is held to the flight's bound as a whole: its launch-lock wait (the lock
            # itself has no timeout, D-26), a liveness check that hangs, and activate's own readiness
            # wait (whose clock starts later and never goes below 1 s) all end at the deadline.
            # Cancelling it there is the same path Esc takes (D-17).
            window = asyncio.timeout(max(0.0, deadline - time.monotonic()))
            # The outage counts as relaunched only once a launch really happened (or was refused):
            # a window that closed while the relaunch still waited for the lock or sized the launch
            # started nothing, and a later loop must still get its one relaunch.
            try:
                async with window:
                    live = await self.relaunch()
            except server.LaunchRefused:
                outage.relaunched = True
                raise
            except TimeoutError as failed:
                if window.expired():
                    if self.config_dir is not None and server.server_pid(self.config_dir) is not None:
                        outage.relaunched = True  # a process was started before the bound passed
                    raise TimeoutError(self._bound_text(wait_seconds, api_key)) from failed
                outage.relaunched = True
                raise RuntimeError(RESTART_FAILED_TEXT + str(failed)) from failed
            except (RuntimeError, OSError) as failed:  # OSError: e.g. the server binary is gone
                outage.relaunched = True
                raise RuntimeError(RESTART_FAILED_TEXT + str(failed)) from failed
            outage.relaunched = True
            if live is not None and deadline - time.monotonic() <= 0:
                # activate's own readiness wait has just seen /models answer at this address: the
                # server is serving, so a relaunch that finished on the bound's edge is a success,
                # never "did not come back". With time left the loop polls once more, which is
                # what catches a relaunched server that dies again at once.
                return

    async def _answers(self, base_url: str, api_key: Any, deadline: float) -> bool:
        """One look at the address with no pidfile check: True when anything there answers (the
        flight's liveness rule), False when nothing does within one poll or the bound is spent."""
        from localharness.provider import client as _client

        look = min(_client._QUEUE_VISIBILITY_SECONDS, deadline - time.monotonic())
        if look <= 0:
            return False
        try:
            await server.wait_ready(
                base_url,
                config_dir=None,
                timeout_seconds=look,
                poll_seconds=look,
                runtime=self.runtime,
                liveness=True,
                **server.key_kwargs(api_key),
            )
            return True
        except (TimeoutError, RuntimeError):
            return False

    def _tail(self, what: str, api_key: Any = None) -> str:
        """The scrubbed tail of the server log and where its raw lines are (D-22), or "" without a log
        directory. `api_key` is the key this recovery holds, masked wherever the log echoes it."""
        if self.config_dir is None:
            return ""
        tail = server.scrubbed_tail(self.config_dir, secrets=(api_key,))
        ref = server.log_reference(self.config_dir, what)
        return tail + "\n" + server.LOG_REF_LINE.format(path=server.log_path(self.config_dir), ref=ref)

    def _bound_text(self, wait_seconds: float, api_key: Any = None) -> str:
        # rounded UP, as the ProviderLost line is (channels.base.provider_lost_seconds): a child with
        # 12.4 s left is told "up to 13s" and must not end "within 12s", nor a 0.3 s clamp "within 0s"
        text = BOUND_TEXT.format(seconds=math.ceil(wait_seconds))
        if self.config_dir is not None:
            text += ". Log tail:\n" + self._tail(text, api_key)
        return text


def _retrieve(task: asyncio.Task) -> None:
    """Retrieve a flight's outcome so a failure nobody awaits any more is never logged as unretrieved."""
    if not task.cancelled():
        task.exception()


async def _announce(run: _Run, mine: float, on_lost: Callable[[], Awaitable[None]]) -> None:
    """The outage's one announcement (D-15/D-16, RECON-03: one ProviderLost per death): once the flight
    has been unresolved for the queue-visibility threshold (a blip under it says nothing), the first
    caller still waiting tells the person, and nobody after it does, whatever loop of the turn waits
    next. The line carries the CONFIGURED bound (the loop publishes it): the flight runs to that bound
    from the drop whoever waits on it, so one line stays true for the whole wait, including a root that
    continues a budget-clamped child's. `mine` is when this caller's wait ends: a caller whose wait is
    already over (the one look past a spent bound) says nothing. A publish that fails costs nothing."""
    from localharness.provider import client as _client

    await asyncio.sleep(max(0.0, _client._QUEUE_VISIBILITY_SECONDS - (time.monotonic() - run.started)))
    outage = run.outage
    if run.task.done() or outage is None or mine <= time.monotonic():
        return
    if outage.announced_until is not None:
        return  # this outage has been announced: one line per death
    outage.announced_until = run.deadline  # the configured bound's end, which the line states
    try:
        await on_lost()
    except Exception:  # noqa: BLE001 - telling the person must never fail the recovery
        log.debug("could not announce the dropped model server", exc_info=True)
