"""Provider lifecycle layer — a launch-STRATEGY abstraction, orthogonal to `provider_type`.

`provider_type` (vllm/ollama/llamacpp/lmstudio) is WHICH wire protocol / token-counting a
backend uses. Lifecycle is the orthogonal axis: HOW the harness brings a backend up, checks
it, and takes it down. A `LifecycleStrategy` unifies that so the heavy-to-heavy cross-framework
`/model` swap (stop the incumbent → confirm the GPU is free → launch the target) becomes real
instead of the manual box-dance.

Phase A (0.11) is ADDITIVE scaffolding: only `ManagedVllmStrategy` exists, and it is a
thin, ZERO-BEHAVIOR-CHANGE wrapper over `provider/server.py` (both launch modes already flow
through those functions). Existing callers — repl `/model` swap, init guided setup, `start`
autostart, `model` list — keep calling `provider/server.py` DIRECTLY; rewiring them through a
strategy is Phase C (a strategy with a bound import would silently break the module-attribute
monkeypatch the harness's tests rely on). The one behavior change shipped now is the
liveness-by-name fix, and it lives in `server.py` (`docker_container_running`) so both this
strategy AND the `start` autostart pre-check use it without going through the strategy.

`SpawnedProcessStrategy` (llama.cpp, Phase B), `DaemonStrategy` (Ollama) and `LmsStrategy`
(LM Studio, Phase D) implement the same Protocol later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from localharness.config.models import ManagedServerConfig
from localharness.provider import server


@dataclass
class LiveEndpoint:
    """A serving endpoint a strategy brought up: its `base_url`, the model ids it actually
    serves (`served_models`), and an opaque stop `handle` the SAME strategy understands
    (docker: the container name; binary/spawned: the pid). The handle carries a strategy's
    teardown target forward to later phases; Phase A populates it but stops via the config."""
    base_url: str
    served_models: list[str]
    handle: Any


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
    """

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
    ) -> LiveEndpoint: ...

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None: ...

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness: ...


class ManagedVllmStrategy:
    """Lifecycle for the harness-managed vLLM server (init guided setup / reboot autostart /
    REPL `/model` swap). A thin, ZERO-BEHAVIOR-CHANGE wrapper over `provider/server.py`, which
    both launch modes (`docker`, `binary`) already flow through:

    - activate = `serve_command` → `start_server` → `wait_ready` (returns served ids).
    - stop     = `stop_server(config_dir, launch=spec.launch)` — the #100 VERIFIED stop
                 (docker: stop → poll name-free → `rm -f` fallback → raise if still stuck).
    - liveness = NAME-based for docker (`docker_container_running` via `docker inspect`), the
                 fix; pid-based for binary (`server_pid`, correct there).
    """

    async def activate(
        self,
        spec: ManagedServerConfig,
        config_dir: Path,
        base_url: str,
        *,
        timeout_seconds: float = 1800.0,
        on_poll: Callable[[float], None] | None = None,
    ) -> LiveEndpoint:
        cmd = server.serve_command(spec)
        pid = server.start_server(config_dir, cmd)
        served = await server.wait_ready(
            base_url,
            config_dir=config_dir,
            timeout_seconds=timeout_seconds,
            on_poll=on_poll,
        )
        handle = server.DOCKER_CONTAINER_NAME if spec.launch == "docker" else pid
        return LiveEndpoint(base_url=base_url, served_models=served, handle=handle)

    async def stop(self, spec: ManagedServerConfig, config_dir: Path) -> None:
        server.stop_server(config_dir, launch=spec.launch)

    def liveness(self, spec: ManagedServerConfig, config_dir: Path) -> Liveness:
        if spec.launch == "docker":
            name = server.DOCKER_CONTAINER_NAME
            return Liveness(alive=server.docker_container_running(name), detail=f"container {name}")
        pid = server.server_pid(config_dir)
        return Liveness(alive=pid is not None, detail=f"pid {pid}" if pid is not None else "no live pidfile")
