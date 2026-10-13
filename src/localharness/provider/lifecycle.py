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
import logging
import math
import os
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol, runtime_checkable

try:
    import fcntl
except ImportError:  # non-POSIX: the launch lock degrades to an in-process asyncio.Lock
    fcntl = None  # type: ignore[assignment]

from localharness.config.models import ManagedServerConfig
from localharness.provider import autosense as _auto
from localharness.provider import server, tuning
from localharness.provider.tuning import Tuned

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # annotations only: autosense imports nothing from here, but the dataclass stays import-light
    from localharness.config.models import AutosenseConfig
    from localharness.provider.autosense import WindowReport


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
    ) -> LiveEndpoint: ...

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None: ...

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness: ...

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
    ) -> LiveEndpoint:
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        plan: LaunchPlan | None = None
        if autosense is not None:
            # OFF the event loop: sensing opens the checkpoint's config.json.
            sensed = await asyncio.to_thread(sense_launch, spec, config_dir, autosense)
            plan = plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed)
            cmd = list(plan.argv)
        else:
            cmd = server.serve_command(
                spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok,
                context=None, cache_ram=None)
        _say_plan(on_plan, plan)
        dropped: list[str] = []
        pid = server.start_server(
            config_dir, cmd, runtime=spec.runtime, record_env=getattr(spec, "env", None),
            dropped_out=dropped, **server.key_kwargs(api_key))
        served = await server.wait_ready(
            base_url,
            config_dir=config_dir,
            timeout_seconds=timeout_seconds,
            on_poll=on_poll,
            **server.key_kwargs(api_key),
        )
        handle = server.DOCKER_CONTAINER_NAME if spec.launch == "docker" else pid
        # No fit for vLLM (D-10): the plan derives the parsers only, `window` stays None, no retry.
        return LiveEndpoint(base_url=base_url, served_models=served, handle=handle, note=decision.note,
                            dropped_env=tuple(dropped), tuned=plan.tuned if plan else None)

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir `activate` polls with, so a reconnect poll reads the same pidfile and log."""
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # OFF the event loop: the #100 verified docker stop can block ~90s worst-case (drain +
        # name-free poll + rm -f), and a synchronous call inside this `async def` would freeze
        # the whole loop — heartbeats, the Discord adapter, idle consolidation. `server.stop_server`
        # is looked up at CALL time (module-attribute), so the harness's monkeypatches still fire.
        await asyncio.to_thread(server.stop_server, config_dir, launch=spec.launch)

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        if spec.launch == "docker":
            name = server.DOCKER_CONTAINER_NAME
            return Liveness(alive=server.docker_container_running(name), detail=f"container {name}")
        pid = server.server_pid(config_dir)
        return Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")


@dataclass(frozen=True)
class Sensed:
    """Everything one launch reads from the machine, gathered once by `sense_launch` and handed to the
    pure `plan_launch` (spec 20 §6.0, D-14): the model header (a typed, bounded dict), the GPU
    residency of the weights (`offload`), the free memory, the weights' size, `/proc/meminfo` and a
    vLLM checkpoint's `config.json`. Every field is None / empty when it could not be known."""

    runtime: str
    config_dir: Path
    header: Mapping[str, Any] = field(default_factory=dict)
    offload: str = "unknown"
    memory: "_auto.FreeMemory | None" = None
    weights: int | None = None
    meminfo: Mapping[str, int] | None = None
    model_config: Mapping[str, Any] | None = None


def sense_launch(spec: ManagedServerConfig, config_dir: Path, autosense: "AutosenseConfig") -> Sensed:
    """Every read a launch needs, once: the GGUF header, free memory, the weights and `/proc/meminfo`
    for llama.cpp; a checkpoint's `config.json` for vLLM; nothing for any other runtime. Writes nothing
    and never raises on bad input (the readers return None / `{}`). Run it off the event loop: it shells
    out to nvidia-smi."""
    if spec.runtime == "llamacpp":
        fi = _auto.compute_fit(spec, autosense, meminfo=True)  # its `fits` is priced without the derived types: unused
        return Sensed(spec.runtime, config_dir, fi.header, fi.offload, fi.memory, fi.weights, fi.meminfo)
    if spec.runtime == "vllm":
        return Sensed(spec.runtime, config_dir, model_config=tuning.read_model_config(spec))
    return Sensed(spec.runtime, config_dir)


@dataclass(frozen=True)
class LaunchPlan:
    """The launch a strategy runs, as a value: the exact `argv`, the keywords that built it
    (`serve_kwargs`), the NAMES of the environment variables the child gets that the harness sets
    (`env_names` — never a value, never the key), the config.ini home a llama.cpp launch prepares
    (`config_ini_home`, a path only), the window `report`, what the launch `tuned`, and how this plan
    came about: `shape` is "full" or "retry", `cause` why a retry was planned ("death"), `dropped` the
    derived names the retry gave up, `full_fit` the fit `-c` a retry halves from.

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
    inputs: tuple | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "serve_kwargs", MappingProxyType(dict(self.serve_kwargs)))

    def retry(self) -> "LaunchPlan | None":
        """The one relaunch after a death (D-16, D-35), chosen here and nowhere else: the derived
        spec type dropped and the fit `-c` halved (`max(1, full_fit // 2)`), each only when the plan
        has it. None for a plan already in the retry shape and for one with neither (a KV-only
        launch). A record or `extra_args` value is never touched. Pure: the same plan gives an equal
        retry."""
        if self.shape != "full" or self.inputs is None:
            return None
        spec, sensed, decision, autosense, held = self.inputs
        kw = dict(self.serve_kwargs)
        report, tuned, dropped = self.report, self.tuned, ()
        drops_spec = tuned.spec_type is not None
        halves = report is not None and report.mode == "fit" and self.full_fit is not None
        if not drops_spec and not halves:
            return None
        if drops_spec:
            kw.pop("spec_type", None)
            kw.pop("draft_p_min", None)
            reasons = {k: v for k, v in tuned.reasons.items() if k != "spec_type"}
            tuned = dataclasses.replace(tuned, spec_type=None, draft_p_min=None, mtp_layers=None, reasons=reasons)
            dropped = ("draft-mtp",)
        if halves:
            halved = max(1, self.full_fit // 2)
            kw["context"] = halved
            report = dataclasses.replace(report, asked=halved, retried=True)
        return dataclasses.replace(
            self, argv=tuple(server.serve_command(spec, **kw)), serve_kwargs=kw, report=report, tuned=tuned,
            shape="retry", cause="death", dropped=dropped)


def _effective_cache_types(spec: ManagedServerConfig, tuned: Tuned) -> tuple[str | None, str | None]:
    """The KV cache types the planned argv carries: the record's, else `extra_args` `-ctk` / `-ctv`,
    else env `LLAMA_ARG_CACHE_TYPE_K` / `_V` (the record's `env:`), else the derived one (R-08e)."""
    extra_k, extra_v = _auto.extra_cache_types(list(spec.extra_args or []))
    env = dict(getattr(spec, "env", None) or {})
    return (spec.cache_type_k or extra_k or env.get("LLAMA_ARG_CACHE_TYPE_K") or tuned.cache_type,
            spec.cache_type_v or extra_v or env.get("LLAMA_ARG_CACHE_TYPE_V") or tuned.cache_type)


def _size_llamacpp(
    spec: ManagedServerConfig, sensed: Sensed, tuned: Tuned, autosense: "AutosenseConfig",
) -> "tuple[_auto.WindowReport, int | None, int | None, int | None]":
    """What a llama.cpp launch asks for, from what was sensed: `(report, context, cache_ram, full_fit)`.

    `context` and `cache_ram` are the fit keywords for `serve_command`: the fit's `-c` only when
    the record sets no `context` and `extra_args` carry no `-c` / `--ctx-size`, and the formula's
    `--cache-ram` only when neither the record nor `extra_args` set one. The fit is priced at the
    KV types the argv carries and, when the argv drafts with MTP, at the draft context's share. An
    unknown or non-positive fit gives no `-c` (llama-server's own `--fit` sizes the window, mode
    "fit-by-server"); a header that cannot be read, a missing key or an unknown memory is "not
    computed", never a guess. Pure."""
    extra = list(spec.extra_args or [])
    env = dict(getattr(spec, "env", None) or {})
    header, mem, weights, mi = sensed.header, sensed.memory, sensed.weights, sensed.meminfo
    k, v = _effective_cache_types(spec, tuned)
    drafting = (tuned.spec_type == "draft-mtp" or spec.speculative == "draft-mtp"
                or _auto.spec_type_flag(extra, env) == "draft-mtp")
    fits = _auto.fit(
        header, mem.free_bytes if mem else None, autosense, cache_type_k=k, cache_type_v=v,
        weights=weights, offload=sensed.offload, runtime="llamacpp",
        draft_share=tuning.draft_share(header) if drafting else 1.0)
    has_c = _auto.has_ctx_arg(extra)
    cache_set = spec.cache_ram is not None or _auto.has_cache_ram_arg(extra)
    cache_kw = None if cache_set else _auto.cache_ram_mib(
        mi["MemAvailable"] if mi else None, mi["MemTotal"] if mi else None,
        weights, sensed.offload, extra, autosense)
    if spec.context is not None:
        mode, asked, ctx_kw = "record", spec.context, None
    elif has_c:
        mode, asked, ctx_kw = "extra_args", None, None
    elif fits:
        mode, asked, ctx_kw = "fit", fits, fits
    else:
        mode, asked, ctx_kw = "fit-by-server", None, None
    report = _auto.WindowReport(
        served=None, trained=header.get("context_length"), fits=fits,
        free_bytes=mem.free_bytes if mem else None, total_bytes=mem.total_bytes if mem else None,
        weights_bytes=weights, asked=asked, mode=mode, attached=False, retried=False,
        verified=False, runtime="llamacpp")
    return report, ctx_kw, cache_kw, (fits if mode == "fit" else None)


def plan_launch(
    spec: ManagedServerConfig, sensed: Sensed, decision: "server.LaunchDecision", *,
    autosense: "AutosenseConfig", api_key_held: bool,
) -> LaunchPlan:
    """The launch of `spec` as a value (D-15): derive, fit, build the `serve_command` keywords and the
    argv. Pure — every read happened in `sense_launch`, nothing is written, and `api_key_held` is a
    bool so the key itself never enters a plan. The derived keywords reach `serve_command` only when
    not None, so a launch that derived nothing calls it with exactly the 0.18 keywords (R-08c)."""
    tuned = tuning.derive(spec, sensed.header, autosense, model_config=sensed.model_config)
    kw: dict[str, Any] = {
        "keyed": api_key_held, "loopback": decision.loopback, "key_file_ok": decision.key_file_ok,
        "context": None, "cache_ram": None}
    report = None
    full_fit = None
    home = None
    names = sorted(getattr(spec, "env", None) or {})
    if spec.runtime == "llamacpp":
        report, kw["context"], kw["cache_ram"], full_fit = _size_llamacpp(spec, sensed, tuned, autosense)
        home = server.config_ini_home(sensed.config_dir, spec.port)
        names.append("XDG_CONFIG_HOME")
    for name in ("cache_type", "spec_type", "draft_p_min", "reasoning_parser", "tool_call_parser"):
        value = getattr(tuned, name)
        if value is not None:
            kw[name] = value
    if api_key_held and spec.runtime in server.KEY_ENV:
        names.append(server.KEY_ENV[spec.runtime])
    return LaunchPlan(
        argv=tuple(server.serve_command(spec, **kw)), serve_kwargs=kw,
        env_names=tuple(dict.fromkeys(names)), config_ini_home=home, report=report, tuned=tuned,
        full_fit=full_fit, inputs=(spec, sensed, decision, autosense, api_key_held))


def _say_plan(on_plan: "Callable[[LaunchPlan | None], None] | None", plan: "LaunchPlan | None") -> None:
    """Hand a plan to the caller's printer right before its launch. The printer is the caller's, and a
    printer that raises must not stop a launch."""
    if on_plan is None:
        return
    try:
        on_plan(plan)
    except Exception:  # noqa: BLE001 - the callback is display only
        log.debug("on_plan callback raised", exc_info=True)


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
        text is parsed; the abandon message quotes `log_tail`."""
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        # OFF the event loop: sensing reads the header, stats the weights and shells out to
        # nvidia-smi (up to 5 s on a wedged driver), none of which may freeze the REPL's loop.
        plan: LaunchPlan | None = None
        if autosense is not None and spec.runtime == "llamacpp":
            sensed = await asyncio.to_thread(sense_launch, spec, config_dir, autosense)
            plan = plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed)
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
                dropped_out=names, xdg_config_home=str(home), **server.key_kwargs(api_key))
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
                await asyncio.to_thread(server.stop_server, config_dir, launch="binary")
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
                    await asyncio.to_thread(server.stop_server, config_dir, launch="binary")
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
        return LiveEndpoint(base_url=base_url, served_models=served, handle=pid, note=decision.note,
                            window=report, dropped_env=tuple(dropped), tuned=plan.tuned if plan else None)

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        """The config dir `activate` polls with, so a reconnect poll reads the same pidfile and log."""
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        # OFF the event loop (same reason as ManagedVllmStrategy.stop): a SIGTERM→SIGKILL grace
        # window blocks, so run it in a worker thread. `launch="binary"` ALWAYS here — a spawned
        # server is a single tracked process, never a docker-run client.
        await asyncio.to_thread(server.stop_server, config_dir, launch="binary")

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        pid = server.server_pid(config_dir)
        return Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")


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
        pid = server.start_server(config_dir, cmd, runtime=spec.runtime)
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
        await asyncio.to_thread(server.stop_server, config_dir, launch="binary")

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        pid = server.server_pid(config_dir)
        return Liveness(
            alive=pid is not None,
            detail=f"ollama daemon pid {pid}" if pid is not None else "no daemon",
        )

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


def strategy_for(spec: ManagedServerConfig) -> LifecycleStrategy:
    """Map a managed-server spec to its launch strategy by `runtime` (the launch-STRATEGY axis,
    orthogonal to provider_type): 'llamacpp' → SpawnedProcessStrategy (the harness spawns
    llama-server itself), 'ollama' → DaemonStrategy (the harness spawns + owns the `ollama serve`
    daemon), 'lmstudio' → LmsStrategy (the harness drives LM Studio's headless `lms` CLI), anything
    else ('vllm', the default) → ManagedVllmStrategy (docker/binary vLLM). The single construction
    point the callers — the REPL /model swap and `start` autostart — route through, so the strategy,
    not the caller, owns HOW a backend is launched/stopped."""
    if spec.runtime == "llamacpp":
        return SpawnedProcessStrategy()
    if spec.runtime == "ollama":
        return DaemonStrategy()
    if spec.runtime == "lmstudio":
        return LmsStrategy()
    return ManagedVllmStrategy()


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
) -> tuple[ManagedServerConfig, str] | None:
    """GPU-lock: make the single accelerator free for `target`, honoring the box rule "at most one
    GPU-heavy server up at a time".

    If a DIFFERENT heavy server holds the accelerator (`incumbent` is not None and is not `target`
    itself), verified-STOP it through its own strategy — the stop contract (process/container gone)
    IS the accelerator-free guarantee on unified memory — then settle briefly, and RETURN the
    stopped ``(spec, base_url)`` so a failed launch can restore it. Returns ``None`` (no action)
    when nothing heavy is up, or when the incumbent IS the target (a same-server model-reload
    restart, whose stop the caller performs itself). This function only ever FREES — it never starts
    or probes anything."""
    if incumbent is None:
        return None
    inc_spec, _inc_url = incumbent
    if inc_spec is target:
        return None
    await strategy_for(inc_spec).stop(inc_spec, config_dir)
    # Read the settle from the module global at CALL time (not a def-time default) so it stays
    # tunable from live heavy-swap data and silenceable in tests.
    await asyncio.sleep(GPU_FREE_SETTLE_SECONDS if settle_seconds is None else settle_seconds)
    return incumbent


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
"""Resolved lock path -> [owning asyncio task, depth]. Re-entrancy is per TASK identity
(`asyncio.current_task()`), never a ContextVar: `create_task` copies the context, so a task spawned
inside the lock would otherwise believe it holds it. Entries exist only while a lock is held and are
removed on every exit path."""

_CONTENDED = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES})
"""The `flock(LOCK_NB)` errnos that mean "another holder has it". Any other errno is a filesystem that
cannot lock at all, and the lock degrades to the in-process one instead of polling forever."""

_FALLBACK_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[Path, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)
"""The in-process stand-in when the flock is unavailable, keyed by running loop because an
`asyncio.Lock` binds to the loop it is first contended in (one loop per test, one per process)."""


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
    any other task waits. The fd is not inheritable, so the long-lived server spawned inside the lock
    never holds it. With no `fcntl`, a lock path that cannot be opened (a planted symlink is
    refused by `O_NOFOLLOW`), or a filesystem whose `flock` fails with anything but contention,
    launches serialise through an in-process lock and one warning names the cause. The file is not a protected config-dir entry (D-29): a replaced file costs only the
    exclusion, which the pidfile re-check still narrows.
    """
    from localharness.provider import client as _client

    path = server.lock_path(config_dir)
    key = path.resolve()
    me = asyncio.current_task()
    held = _HELD.get(key)
    if held is not None and held[0] is me:  # nested acquire by the owner: nothing to take
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
    try:
        if fd is not None:
            started = time.monotonic()
            told = False
            while True:
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
                    await asyncio.sleep(0.05)
                    if (on_wait is not None and not told
                            and time.monotonic() - started >= _client._QUEUE_VISIBILITY_SECONDS):
                        told = True
                        try:
                            await on_wait()
                        except Exception:  # noqa: BLE001 - telling the person never aborts the launch
                            log.debug("launch lock: the wait announcement failed", exc_info=True)
        if fd is None:
            fallback = _in_process_lock(key, cause)
            await fallback.acquire()
            got_fallback = True
        _HELD[key] = [me, 1]
        try:
            yield
        finally:
            if _HELD.get(key, [None])[0] is me:
                _HELD.pop(key, None)
    finally:
        if got_fallback and fallback is not None:
            fallback.release()
        if fd is not None:
            os.close(fd)  # releases the flock; the kernel also does when a holder dies


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
                raise TimeoutError(self._bound_text(own)) from None
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
                raise TimeoutError(self._bound_text(wait_seconds)) from None
            raise _again(outage.failed) from None
        # what the outage's one relaunch already came to, when an earlier loop spent it and failed
        earlier = outage.failed if outage.relaunched else None
        relaunched = outage.relaunched
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(self._bound_text(wait_seconds))
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
                raise TimeoutError(self._bound_text(wait_seconds)) from late
            except RuntimeError as died:  # the launched process is gone (wait_ready's own signal)
                if self.relaunch is None:
                    raise RuntimeError(EXITED_TEXT + self._tail()) from died
                if relaunched:
                    # The one relaunch is spent (here, or by an earlier loop of the turn), but the person
                    # may have restarted the server by hand: look at the address before giving up.
                    if await self._answers(base_url, api_key, deadline):
                        log.info("model server process exited again, but the address answers")
                        return
                    if earlier is not None and (issubclass(earlier[0], server.LaunchRefused)
                                                or earlier[1].startswith(RESTART_FAILED_TEXT)):
                        raise _again(earlier) from died  # the one relaunch was refused or failed
                    raise RuntimeError(EXITED_AGAIN_TEXT + self._tail()) from died
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
                    raise TimeoutError(self._bound_text(wait_seconds)) from failed
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

    def _tail(self) -> str:
        return server.log_tail(self.config_dir) if self.config_dir is not None else ""

    def _bound_text(self, wait_seconds: float) -> str:
        # rounded UP, as the ProviderLost line is (channels.base.provider_lost_seconds): a child with
        # 12.4 s left is told "up to 13s" and must not end "within 12s", nor a 0.3 s clamp "within 0s"
        text = BOUND_TEXT.format(seconds=math.ceil(wait_seconds))
        if self.config_dir is not None:
            text += ". Log tail:\n" + server.log_tail(self.config_dir)
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
