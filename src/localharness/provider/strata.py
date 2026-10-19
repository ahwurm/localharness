"""Strata, the MoE server, as one runtime."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from localharness.config import trust
from localharness.config.models import ManagedServerConfig
from localharness.core import private_files
from localharness.provider import autosense as _auto
from localharness.provider import gguf_header
from localharness.provider import lifecycle, model_library, server, tuning
from localharness.provider.procfs import LaunchIdentity
from localharness.provider.runtimes import RUNTIMES
from localharness.provider.tuning import MODEL_CONFIG_CAP

READY_SECONDS = 900 + 60
ALLOWED_KEYS = ("model_name", "sampling", "reasoning_effort", "parallel", "gpu", "args")
HARNESS_KEYS = ("host", "port", "cwd", "log")
ALLOWED_FLAGS = (
    "--pack", "--native", "--ple-gguf", "--expert-profile", "--expert-cache", "--prefill", "--spec",
    "--spec-min-p", "--mtp", "--max-context", "--kv", "--kv-resident", "--vram-reserve-mib", "--mmap-experts",
    "--resident-experts", "--resident-budget-gib")
NATIVE_MISMATCH = ("{name}'s Strata config serves another file than its `model:` path — "
                   "re-run Strata's setup with --gguf-dir, or fix the record.")
CONFIG_REFUSAL = ("{name}'s Strata config {what} — fix {engine_config} (or re-run Strata's setup); "
                  "the harness launches Strata only with the keys, flags and paths it allows.")
UNCONFIRMED = "{name} cannot run here: its record has changes you have not confirmed — run localharness start on a terminal."
NOT_AN_INSTALL = "{name} cannot run here: its Strata install at {engine_dir} no longer passes the owner and mode checks."
STRATA_HEADROOM = 10 * 10**9
STRATA_TWICE = "Strata could not load {name} twice; its low-RAM setup (setup.sh --low-ram) or the llama.cpp preset may."
NO_WINDOW = ("{name}'s Strata config sets no --max-context — re-run Strata's setup with --context, "
             "or set `context:` on {name}'s record.")
COPY_DELIMITER = "STRATA_COPY"
VENV_PYTHON = ".venv/bin/python"
SERVE_PY = "serve/server.py"
SINGLE_FLAGS = ("--max-context", "--native", "--kv")
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
EXPANDED = re.compile(r"[$`%]")
DATA_DIR_NAME = "Strata-data"
EFFORT_VALUES = ("none", "low", "medium", "high")
SAMPLING_KEYS = (
    "sequence", "top_k", "top_p", "min_p", "xtc_probability", "xtc_threshold", "temp", "penalty_last_n",
    "penalty_repeat", "mirostat", "mirostat_tau", "mirostat_eta")


def _root(spec: ManagedServerConfig) -> Path:
    return Path(os.path.expanduser(str(spec.engine_dir)))


def interpreter(spec: ManagedServerConfig) -> str:
    return str(_root(spec) / VENV_PYTHON)


def serve_py(spec: ManagedServerConfig) -> str:
    return str(_root(spec) / SERVE_PY)


def copy_path(config_dir: Path, port: int) -> Path:
    return Path(os.path.abspath(config_dir)) / "strata" / f"{port}.json"


def log_path(config_dir: Path, port: int) -> Path:
    return copy_path(config_dir, port).with_suffix(".log")


def argv(srv: ManagedServerConfig, *, config_copy: str | None = None, **_: Any) -> list[str]:
    return [interpreter(srv), serve_py(srv), "--engine", "strata", *(["--config", config_copy] if config_copy else []),
            "--port", str(srv.port), "--host", "127.0.0.1"]


@dataclass(frozen=True)
class EngineConfig:
    raw: bytes
    data: Mapping[str, Any]
    sha256: str
    sampling: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def is_flag(token: str) -> bool:
    if not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _shown(text: str, cap: int = 60) -> str:
    return "".join(c for c in text if c.isprintable())[:cap]


def _value(args: list[str], flag: str) -> str | None:
    for i, token in enumerate(args):
        if token == flag:
            return args[i + 1] if i + 1 < len(args) else None
        if token.startswith(f"{flag}="):
            return token.partition("=")[2]
    return None


def roots(spec: ManagedServerConfig) -> tuple[str, ...]:
    """The realpaths a path argument may sit under: the install, Strata's data beside it, the model's folder."""
    engine = os.path.realpath(_root(spec))
    return (engine, os.path.join(os.path.dirname(engine), DATA_DIR_NAME),
            os.path.dirname(os.path.realpath(os.path.expanduser(str(spec.model)))))


def shared_folder(spec: ManagedServerConfig) -> str | None:
    """The model's folder when it is `/`, the home directory or writable by everyone, else None."""
    folder = roots(spec)[2]
    try:
        open_to_all = bool(os.stat(folder).st_mode & 0o002)
    except OSError:
        open_to_all = False
    return folder if open_to_all or folder in (os.sep, os.path.realpath(os.path.expanduser("~"))) else None


def inside(path: str, allowed: tuple[str, ...]) -> bool:
    try:
        return any(os.path.commonpath([path, root]) == root for root in allowed)
    except ValueError:
        return False


def _number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _paths(args: list[str], spec: ManagedServerConfig) -> list[str]:
    allowed = roots(spec)

    def outside(part: str) -> bool:
        like = ("/" in part or part[:1] in ("~", ".")) and not _number(part)
        return like and bool(SCHEME.match(part) or not inside(os.path.realpath(_root(spec) / os.path.expanduser(part)), allowed))

    values = [t.partition("=")[2] if is_flag(t) else t for t in args]
    return [v for v in values if EXPANDED.search(v) or any(outside(p) for p in (re.split(r"[=,:]", v) if "=" in v else [v]))]


def _flag_values(args: list[str], flag: str) -> list[str | None]:
    return [t.partition("=")[2] if "=" in t else (args[i + 1] if i + 1 < len(args) else None)
            for i, t in enumerate(args) if t.partition("=")[0] == flag]


def _count(value: Any, least: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= least


def _bad_keys(data: Mapping[str, Any], args: list[str]) -> list[str]:
    checks = {
        "gpu": lambda v: _count(v, 0), "parallel": lambda v: _count(v, 1),
        "model_name": lambda v: isinstance(v, str) and "\n" not in v and "\r" not in v,
        "reasoning_effort": lambda v: v in EFFORT_VALUES and isinstance(v, str),
        "sampling": lambda v: isinstance(v, dict) and all(
            isinstance(x, (int, float, str)) and not isinstance(x, bool) for x in v.values())}
    bad = [k for k in sorted(checks) if k in data and not checks[k](data[k])]
    if any(v is None or not re.fullmatch(r"[0-9]{1,9}", v) or int(v) < 1 for v in _flag_values(args, "--max-context")):
        bad.append("--max-context")
    bad += [f"--kv {_shown(v or '')}".strip() for v in _flag_values(args, "--kv") if v is None or v.lower() not in _auto.KV_BYTES]
    return bad


def _header_sampling(spec: ManagedServerConfig) -> dict[str, Any]:
    try:
        found = gguf_header.read_header(
            os.path.expanduser(str(spec.model)), {f"general.sampling.{k}" for k in SAMPLING_KEYS})
    except (OSError, gguf_header.GgufHeaderError):
        return {}
    return {k.rpartition(".")[2]: float(f"{v:.7g}") if isinstance(v, float) else v for k, v in found.items()}


def sense(spec: ManagedServerConfig) -> tuple[EngineConfig | None, str | None]:
    """The engine config read once and checked: `(config, None)`, or `(None, the refusal sentence)`."""
    name = _auto._label(spec)

    def refuse(what: str) -> tuple[None, str]:
        return None, CONFIG_REFUSAL.format(name=name, what=what, engine_config=_shown(str(spec.engine_config), 120))

    try:
        with os.fdopen(private_files.open_regular(_root(spec) / str(spec.engine_config)), "rb") as fh:
            raw = fh.read(MODEL_CONFIG_CAP + 1)
    except OSError as exc:
        return refuse(f"cannot be read ({exc.strerror or type(exc).__name__})")
    if len(raw) > MODEL_CONFIG_CAP:
        return refuse(f"cannot be read (over {MODEL_CONFIG_CAP} bytes)")
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        data = None
    args = data.get("args", []) if isinstance(data, dict) else None
    if not isinstance(args, list) or not all(isinstance(t, str) for t in args):
        return refuse("is not a JSON object with a list of string arguments")
    if any("\x00" in t for t in args):
        return refuse("has a NUL byte in an argument")
    repeated = [f for f in SINGLE_FLAGS if sum(t.partition("=")[0] == f for t in args) > 1]
    if repeated:
        return refuse(f"repeats {', '.join(repeated)}")
    native = _value(args, "--native")
    served = os.path.realpath(_root(spec) / native) if native else None
    if served != os.path.realpath(os.path.expanduser(str(spec.model))):
        return None, NATIVE_MISMATCH.format(name=name)
    offences = []
    keys = sorted(k for k in data if k not in ALLOWED_KEYS + HARNESS_KEYS)
    if keys:
        offences.append(f"sets {', '.join(_shown(k) for k in keys)}")
    flags = dict.fromkeys(t.partition("=")[0] for t in args if is_flag(t))
    stray = [f for f in flags if f not in ALLOWED_FLAGS]
    if stray:
        offences.append(f"passes {', '.join(_shown(f) for f in stray)}")
    outside = _paths(args, spec)
    if outside:
        offences.append(f"names {', '.join(_shown(p) for p in outside)} outside the Strata install, its data and the model folder")
    shared = shared_folder(spec)
    if shared:
        offences.append(f"has its model in {_shown(shared, 200)}, which is the root, your home or open to every user")
    bad = _bad_keys(data, args)
    if bad:
        offences.append(f"has a bad value for {', '.join(bad)}")
    if offences:
        return refuse("; ".join(offences))
    return EngineConfig(raw, data, hashlib.sha256(raw).hexdigest(), _header_sampling(spec)), None


def verify_recorded(config_dir: Path, spec: ManagedServerConfig, peer: str | None, engine: EngineConfig) -> str | None:
    """The refusal sentence unless the install passes its checks and its code and config bytes are the recorded ones."""
    name = _auto._label(spec)
    known = {os.path.realpath(os.path.expanduser(str(spec.model)))}
    install = model_library.strata_install((_root(spec),), known)
    if install is None or str(spec.engine_config) not in dict(install.configs):
        return NOT_AN_INSTALL.format(name=name, engine_dir=_shown(str(spec.engine_dir), 200))
    recorded = trust.recorded_machine(config_dir)
    if recorded is None:
        return UNCONFIRMED.format(name=name)
    prefix = "server." if peer is None else f"extra_endpoints.{peer}.lifecycle."
    held = {(e.get("kind"), e.get("name"), e.get("sha256")) for e in recorded["entries"]}
    code = [(e["kind"], e["name"], e.get("sha256")) for e in trust.strata_entries(
        config_dir, "", prefix, {"engine_dir": spec.engine_dir, "engine_config": spec.engine_config})
        if e["name"] == prefix + "strata_code"]
    if code and all(entry[2] and entry in held for entry in code) and (
            trust.LAUNCH_KIND, prefix + "engine_config", engine.sha256) in held:
        return None
    return UNCONFIRMED.format(name=name)


def price(spec: ManagedServerConfig, sensed: "lifecycle.Sensed", tuned: Any, autosense: Any,
          floor: int | None = None) -> "_auto.Price":
    """The window and the three refusals a Strata launch prices itself by: RAM for the experts, GPU for the rest, the floor."""
    name, header, split, mem, meminfo = _auto._label(spec), sensed.header, sensed.split, sensed.memory, sensed.meminfo
    args = list(sensed.engine.data.get("args", []))
    configured = _value(args, "--max-context")
    if spec.context is not None:
        window, where, mode = spec.context, "context:", "record"
    elif configured is not None:
        window, where, mode = int(configured), f"--max-context in {spec.engine_config}", "fit"
    else:
        return _auto.Price(None, None, None, None, NO_WINDOW.format(name=name))
    experts = split.experts if split else sensed.weights or 0
    if meminfo is None:
        return _auto.Price(None, None, None, None, _auto.RAM_UNKNOWN_REFUSAL.format(name=name, need=_auto.gib(experts)))
    headroom = max(STRATA_HEADROOM, autosense.unified_reserve_ratio * meminfo["MemTotal"])
    if experts + headroom > meminfo["MemAvailable"]:
        return _auto.Price(None, None, None, None, _auto.RAM_REFUSAL.format(
            name=name, need=_auto.gib(experts), free=_auto.gib(max(0, meminfo["MemAvailable"] - headroom))))
    gpu = (split.gpu if split else 0) + sensed.mmproj
    kv = _auto.kv_bytes_per_token(header, *[_value(args, "--kv")] * 2, split)
    free = mem.free_bytes if mem else None
    if free is not None and kv is not None and gpu + kv * window > free:
        return _auto.Price(None, None, None, None, _auto.VRAM_REFUSAL.format(
            name=name, need=_auto.gib(gpu + kv * window), window=f" for {_auto.window_phrase(window)}", free=_auto.gib(free),
            remedy=_auto.VRAM_REMEDY))
    if floor and window < floor:
        return _auto.Price(None, None, None, None, _auto.CONTEXT_FLOOR_REFUSAL.format(
            name=name, window=_auto.window_phrase(window), where=where, floor=floor))
    placement = _auto.Placement(
        window, "strata", gpu, sensed.mmproj, experts, kv, free or 0, free or 0, None, None,
        parallel=sensed.engine.data.get("parallel", 1))
    report = _auto.WindowReport(
        served=None, trained=header.get("context_length"), fits=window, free_bytes=free,
        total_bytes=mem.total_bytes if mem else None, weights_bytes=sensed.weights, asked=window, mode=mode,
        verified=False, runtime="strata", placement=placement, floor=floor)
    return _auto.Price(report, window, None, window) if mode == "fit" else _auto.Price(report, None, None, None)


def _escaped(spec: ManagedServerConfig) -> str:
    name = spec.served_name or Path(os.path.expanduser(str(spec.model))).name
    return "".join(c if c.isprintable() else f"\\x{ord(c):02x}" if ord(c) < 0x100 else f"\\u{ord(c):04x}" for c in name)


def launch_context(plan: "lifecycle.LaunchPlan | None", spec: ManagedServerConfig) -> int | None:
    planned = plan.serve_kwargs.get("context") if plan is not None else None
    return planned if planned is not None else spec.context


def copy_of(engine: EngineConfig, spec: ManagedServerConfig, *, context: int | None, port: int,
            config_dir: Path) -> dict[str, Any]:
    args = list(engine.data.get("args", []))
    if context is not None:
        at = next((i for i, t in enumerate(args) if t == "--max-context" or t.startswith("--max-context=")), None)
        if at is None:
            args += ["--max-context", str(context)]
        elif args[at] == "--max-context" and at + 1 < len(args):
            args[at + 1] = str(context)
        else:
            args[at] = f"--max-context={context}"
    copy = {k: engine.data[k] for k in ALLOWED_KEYS if k in engine.data}
    if not copy.get("sampling") and engine.sampling:
        copy["sampling"] = dict(engine.sampling)
    return {**copy, "model_name": spec.served_name, "args": args, "host": "127.0.0.1", "port": port,
            "cwd": str(_root(spec)), "log": str(log_path(config_dir, port))}


def copy_text(copy: Mapping[str, Any]) -> str:
    return json.dumps(copy, sort_keys=True, indent=2) + "\n"


def _own_log(fd: int, directory: Path, name: str) -> None:
    try:
        found = os.lstat(name, dir_fd=fd)
        mode, links = found.st_mode, found.st_nlink
    except FileNotFoundError:
        mode, links = stat.S_IFREG, 1
    if stat.S_ISDIR(mode):
        shutil.rmtree(directory / name)
    elif not stat.S_ISREG(mode) or links > 1:
        os.unlink(name, dir_fd=fd)
    with open(name, "wb", opener=private_files.nofollow_opener(fd)) as fh:
        os.fchmod(fh.fileno(), private_files.PRIVATE_FILE_MODE)


def write_copy(config_dir: Path, port: int, copy: Mapping[str, Any]) -> Path:
    """The copy and an empty log, both 0600 in a 0700 directory; a planted link is replaced, never followed."""
    path = copy_path(config_dir, port)
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    server._own_directory(path.parent)
    fd = private_files.open_private_dir(path.parent, create=False)
    try:
        for name in os.listdir(fd):
            if name.startswith(f".{path.name}."):
                os.unlink(name, dir_fd=fd)
        private_files.atomic_write_text(path.name, copy_text(copy), dir_fd=fd)
        _own_log(fd, path.parent, log_path(config_dir, port).name)
    finally:
        os.close(fd)
    return path


def render_script(
    plan: Any, spec: ManagedServerConfig, copy: Mapping[str, Any], copy_file: Path, *, name: str, version: str,
    key_placeholder: str, visible: Callable[[str], str], function: str | None = None,
) -> str:
    """The launch as POSIX sh: write the copy through a quoted here-document, then exec the venv python; no key value."""
    window = _value(copy.get("args", []), "--max-context")
    clauses = ([f"window: {int(window):,} (--max-context)"] if window else []) + [
        f"environment: {tuning.SCRUBBED_SHOWN[spec.runtime]} dropped"]
    head = [f"# {name} — the launch `localharness start` / `/model` runs ({version})", "# " + "; ".join(clauses),
            f"# writes {copy_file}, the file localharness start uses for this port — stop the managed server first"]
    key_env = RUNTIMES[spec.runtime].key_env
    if key_env in plan.env_names:
        head.append(f"# {key_env}=<{key_placeholder}>")
    quote = lambda tok: visible(shlex.quote(tok))  # noqa: E731
    head = [visible(line) for line in head]
    head += [f"mkdir -p -m 700 {quote(str(copy_file.parent))}", f"cat > {quote(str(copy_file))} <<'{COPY_DELIMITER}'"]
    run = " ".join(map(quote, [VENV_PYTHON, SERVE_PY, *plan.argv[2:]]))
    tail = [f"chmod 600 {quote(str(copy_file))}", f"cd {quote(str(_root(spec)))} && exec {run}"]
    document = [*copy_text(copy).removesuffix("\n").split("\n"), COPY_DELIMITER]
    if function is None:
        return "\n".join([*head, *document, *tail])
    return "\n".join([f"{function}() {{", "  (", *(f"    {line}" for line in head), *document,
                      *(f"    {line}" for line in tail), "  )", "}"])


async def health_ready(client: httpx.AsyncClient, base_url: str, headers: dict[str, str]) -> list[str] | None:
    """The served ids once `/health` says the engine is loaded, else None (still loading)."""
    root = base_url.rstrip("/")
    health = await client.get(f"{root.removesuffix('/v1')}/health", timeout=3.0)
    try:
        body = health.json() if health.status_code == 200 else None
    except ValueError:
        body = None
    if not (isinstance(body, dict) and body.get("loaded") is True and body.get("service") == "strata"):
        return None
    models = await client.get(f"{root}/models", timeout=3.0, **({"headers": headers} if headers else {}))
    try:
        data = models.json().get("data", []) if models.status_code == 200 else None
    except (ValueError, AttributeError):
        data = None
    return [m["id"] for m in data if isinstance(m, dict) and "id" in m] if isinstance(data, list) else None


def identity(spec: ManagedServerConfig, config_dir: Path) -> LaunchIdentity:
    """The launch identity plus this port's copy path: the engine child and another port's launch fail it."""
    base = StrataStrategy().launch_identity(spec)
    return dataclasses.replace(base, argv_contains=(*base.argv_contains, str(copy_path(config_dir, spec.port))))


class StrataStrategy:
    plans_launch = True

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Any = None,
        api_key: str | None = None,
        peer: str | None = None,
        autosense: Any = None,
        on_plan: Any = None,
        floor: int | None = None,
    ) -> lifecycle.LiveEndpoint:
        keyed = bool(api_key)
        decision = server.launch_decision(spec, keyed, peer=peer)
        if decision.refusal is not None:
            raise server.LaunchRefused(decision.refusal)
        copy_file = copy_path(config_dir, spec.port)
        if autosense is not None:
            sensed = await asyncio.to_thread(lifecycle.sense_launch, spec, config_dir, autosense)
            plan = lifecycle.plan_launch(spec, sensed, decision, autosense=autosense, api_key_held=keyed, floor=floor)
            engine, cmd = sensed.engine, list(plan.argv)
        else:
            plan = None
            engine, refusal = await asyncio.to_thread(sense, spec)
            if refusal is not None:
                raise server.LaunchRefused(refusal)
            cmd = server.serve_command(
                spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok,
                context=None, cache_ram=None, config_copy=str(copy_file))

        async def confirmed() -> None:
            unconfirmed = await asyncio.to_thread(verify_recorded, config_dir, spec, peer, engine)
            if unconfirmed is not None:
                raise server.LaunchRefused(unconfirmed)

        await confirmed()

        async def prepare(chosen: "lifecycle.LaunchPlan | None") -> None:
            copy = copy_of(engine, spec, context=launch_context(chosen, spec), port=spec.port, config_dir=config_dir)
            try:
                await asyncio.to_thread(write_copy, config_dir, spec.port, copy)
            except OSError as exc:
                raise RuntimeError(f"could not prepare {copy_file}: {exc.strerror or exc}") from exc

        def launch(argv_: list[str]) -> tuple[int, list[str]]:
            names: list[str] = []
            pid_ = server.start_server(
                config_dir, argv_, runtime=spec.runtime, dropped_out=names, cwd=str(_root(spec)), port=spec.port,
                **server.key_kwargs(api_key))
            return pid_, names

        async def ready() -> list[str]:
            return await server.wait_ready(
                base_url, config_dir=config_dir, timeout_seconds=min(timeout_seconds, READY_SECONDS), on_poll=on_poll,
                runtime=spec.runtime, ready=health_ready, engine_log=log_path(config_dir, spec.port),
                **server.key_kwargs(api_key))

        lifecycle._say_plan(on_plan, plan)
        await prepare(plan)
        pid, dropped = launch(cmd)
        try:
            served = await ready()
        except RuntimeError as died:
            retry = plan.retry() if plan is not None else None
            if retry is None and plan is not None and plan.report is not None and plan.report.mode == "record":
                retry = plan
            if retry is None:
                raise
            try:
                await self.stop(spec, config_dir)
            except RuntimeError:
                raise died from None
            await confirmed()
            lifecycle._say_plan(on_plan, retry)
            await prepare(retry)
            pid, dropped = launch(list(retry.argv))
            try:
                served = await ready()
            except RuntimeError as second:
                with contextlib.suppress(RuntimeError):
                    await self.stop(spec, config_dir)
                raise RuntimeError(STRATA_TWICE.format(name=_escaped(spec)) + "\n" + str(second)) from None
            plan = retry
        report = plan.report if plan is not None else None
        if report is not None:
            from localharness.agent.context import probe_window

            probe = await asyncio.to_thread(
                probe_window, base_url, served[0] if served else spec.served_name, "strata", api_key=api_key)
            report = dataclasses.replace(
                report, served=probe.served, trained=probe.trained or report.trained,
                verified=probe.served is not None)
        live = lifecycle.LiveEndpoint(
            base_url=base_url, served_models=served, handle=pid, note=decision.note, window=report,
            dropped_env=tuple(dropped), tuned=plan.tuned if plan else None)
        if plan is not None:
            lifecycle.confirm_launch(config_dir, lifecycle.record_key(spec), plan=plan, live=live)
        return live

    def recovery_config_dir(self, config_dir: Path | None) -> Path | None:
        return config_dir

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        await asyncio.to_thread(
            server.stop_server, config_dir, launch="binary", expect=identity(spec, config_dir), parent_first=True)

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> lifecycle.Liveness:
        pid = server.server_pid(config_dir)
        return lifecycle.Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity:
        return LaunchIdentity(
            frozenset({"python*"}), spec.port, (serve_py(spec), "--engine", "strata", "--config"), None)
