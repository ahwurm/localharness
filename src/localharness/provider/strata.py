"""Strata, the MoE server, as one runtime."""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from localharness.config.models import ManagedServerConfig
from localharness.core import private_files
from localharness.provider import autosense as _auto
from localharness.provider import lifecycle, server, tuning
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
NATIVE_MISMATCH = ("{name}'s Strata config serves another file than its model: path — "
                   "re-run Strata's setup with --gguf-dir, or fix the record.")
CONFIG_REFUSAL = "{name}'s Strata config {what} — the harness launches Strata only with the keys, flags and paths it checks."
COPY_DELIMITER = "STRATA_COPY"


def _root(spec: ManagedServerConfig) -> Path:
    return Path(os.path.expanduser(str(spec.engine_dir)))


def interpreter(spec: ManagedServerConfig) -> str:
    return str(_root(spec) / ".venv" / "bin" / "python")


def serve_py(spec: ManagedServerConfig) -> str:
    return str(_root(spec) / "serve" / "server.py")


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


def is_flag(token: str) -> bool:
    if not token.startswith("-"):
        return False
    try:
        float(token)
    except ValueError:
        return True
    return False


def _shown(text: str) -> str:
    return "".join(c for c in text if c.isprintable())[:60]


def _value(args: list[str], flag: str) -> str | None:
    for i, token in enumerate(args):
        if token == flag:
            return args[i + 1] if i + 1 < len(args) else None
        if token.startswith(f"{flag}="):
            return token.partition("=")[2]
    return None


def sense(spec: ManagedServerConfig) -> tuple[EngineConfig | None, str | None]:
    """The engine config read once and checked: `(config, None)`, or `(None, the refusal sentence)`."""
    name = _auto._label(spec)

    def refuse(what: str) -> tuple[None, str]:
        return None, CONFIG_REFUSAL.format(name=name, what=what)

    try:
        with os.fdopen(private_files.open_regular(_root(spec) / str(spec.engine_config)), "rb") as fh:
            raw = fh.read(MODEL_CONFIG_CAP + 1)
    except OSError as exc:
        return refuse(f"cannot be read ({exc.strerror or type(exc).__name__})")
    try:
        data = json.loads(raw) if len(raw) <= MODEL_CONFIG_CAP else None
    except (ValueError, RecursionError):
        data = None
    args = data.get("args", []) if isinstance(data, dict) else None
    if not isinstance(args, list) or not all(isinstance(t, str) for t in args):
        return refuse("is not a JSON object with a list of string arguments")
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
    if offences:
        return refuse("; ".join(offences))
    return EngineConfig(raw, data, hashlib.sha256(raw).hexdigest()), None


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
    return {**copy, "model_name": spec.served_name, "args": args, "host": "127.0.0.1", "port": port,
            "cwd": str(_root(spec)), "log": str(log_path(config_dir, port))}


def copy_text(copy: Mapping[str, Any]) -> str:
    return json.dumps(copy, sort_keys=True, indent=2) + "\n"


def write_copy(config_dir: Path, port: int, copy: Mapping[str, Any]) -> Path:
    path = copy_path(config_dir, port)
    server._own_directory(path.parent)
    private_files.atomic_write_text(path, copy_text(copy))
    return path


def render_script(
    plan: Any, spec: ManagedServerConfig, copy_text: str, copy_file: Path, *, name: str, version: str,
    key_placeholder: str, visible: Callable[[str], str], function: str | None = None,
) -> str:
    """The launch as POSIX sh: write the copy through a quoted here-document, then exec the venv python; no key value."""
    window = _value(json.loads(copy_text).get("args", []), "--max-context")
    clauses = ([f"window: {window} (--max-context)"] if window else []) + [
        f"environment: {tuning.SCRUBBED_SHOWN[spec.runtime]} dropped"]
    head = [f"# {name} — the launch `localharness start` / `/model` runs ({version})", "# " + "; ".join(clauses)]
    key_env = RUNTIMES[spec.runtime].key_env
    if key_env in plan.env_names:
        head.append(f"# {key_env}=<{key_placeholder}>")
    quote = lambda tok: visible(shlex.quote(tok))  # noqa: E731
    head = [visible(line) for line in head]
    head += [f"mkdir -p {quote(str(copy_file.parent))}", f"cat > {quote(str(copy_file))} <<'{COPY_DELIMITER}'"]
    run = " ".join(map(quote, [".venv/bin/python", "serve/server.py", *plan.argv[2:]]))
    tail = f"cd {quote(str(_root(spec)))} && exec {run}"
    document = [*copy_text.removesuffix("\n").split("\n"), COPY_DELIMITER]
    if function is None:
        return "\n".join([*head, *document, tail])
    return "\n".join([f"{function}() {{", "  (", *(f"    {line}" for line in head), *document, f"    {tail}", "  )", "}"])


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
        lifecycle._say_plan(on_plan, plan)
        copy = copy_of(engine, spec, context=launch_context(plan, spec), port=spec.port, config_dir=config_dir)
        try:
            await asyncio.to_thread(write_copy, config_dir, spec.port, copy)
        except OSError as exc:
            raise RuntimeError(f"could not prepare {copy_file}: {exc.strerror or exc}") from exc
        dropped: list[str] = []
        pid = server.start_server(
            config_dir, cmd, runtime=spec.runtime, dropped_out=dropped, cwd=str(_root(spec)), port=spec.port,
            **server.key_kwargs(api_key))
        served = await server.wait_ready(
            base_url, config_dir=config_dir, timeout_seconds=min(timeout_seconds, READY_SECONDS), on_poll=on_poll,
            runtime=spec.runtime, ready=health_ready, **server.key_kwargs(api_key))
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
            server.stop_server, config_dir, launch="binary", expect=self.launch_identity(spec))

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> lifecycle.Liveness:
        pid = server.server_pid(config_dir)
        return lifecycle.Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")

    def launch_identity(self, spec: ManagedServerConfig) -> LaunchIdentity:
        return LaunchIdentity(
            frozenset({"python*"}), spec.port, (serve_py(spec), "--engine", "strata", "--config"), None)
