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


class SpawnedProcessStrategy:
    """Lifecycle for a harness-SPAWNED OpenAI-compatible server process — llama.cpp's
    `llama-server` (0.11 Phase B) and, later, any single-process binary that serves the OpenAI
    wire. The harness launches it ITSELF (the 0.11 goal) through the SAME `provider/server.py`
    primitives `ManagedVllmStrategy` uses, so the proven detached-launch / pidfile / readiness /
    verified-stop recipe is REUSED, not reinvented:

    - activate = `serve_command` → `start_server` → `wait_ready` (returns served ids); the stop
                 handle is the launched PID.
    - stop     = `stop_server(config_dir, launch="binary")` — SIGTERM the process group, SIGKILL
                 on timeout. Always the pid-group teardown: a spawned server is a single tracked
                 process, never a `docker run` client, so `launch` is pinned to "binary" here.
    - liveness = `server_pid(config_dir) is not None` — CORRECT here (unlike docker): the pidfile
                 pid IS the real `llama-server` process, so a live pid means it is serving. This is
                 exactly the case the #99/#100 docker orphan-client-pid bug was NOT.

    Phase B ships the mechanism + a gated live spawn test; rewiring `start` / `/model` swap onto
    strategies is Phase C (keeps this new spawn path out of the monkeypatch-sensitive rewire).
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
        return LiveEndpoint(base_url=base_url, served_models=served, handle=pid)

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
    - **STOP kills the WHOLE daemon** (owner ruling): stopping the daemon process is a hard,
      verified GPU-free guarantee (same class as docker-stop / process-kill), whereas a per-model
      `keep_alive:0` unload is a soft signal that may not promptly/fully free VRAM.

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
        pid = server.start_server(config_dir, cmd)
        available = await server.wait_ready(             # daemon up; returns PULLED model ids
            v1, config_dir=config_dir, timeout_seconds=timeout_seconds, on_poll=on_poll,
        )
        if spec.model not in available:
            await self._pull(spec)                       # tag not on disk → pull it
        await self._load(v1, spec.model)                 # warm-up: load into memory (no generation)
        return LiveEndpoint(base_url=v1, served_models=[spec.model], handle=pid)

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
        _, err = await proc.communicate()
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
        )
        # The served id IS the loaded model key (LM Studio's /v1/models also lists any embedding
        # model; the caller takes served_models[0], so return the intended key — the DaemonStrategy
        # precedent — not wait_ready's raw list). No pidfile → an opaque daemon handle.
        return LiveEndpoint(base_url=v1, served_models=[spec.model], handle="lms-daemon")

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
            try:
                proc.kill()          # guard BOTH kill and wait: if the proc already exited in the
                await proc.wait()    # timeout race, either raises ProcessLookupError (an OSError)
            except ProcessLookupError:  # that would ELSE escape the (RuntimeError|TimeoutError)
                pass                    # activate contract and crash the session. Reap on the live loop.
            raise TimeoutError(f"`{' '.join(args)}` timed out after {timeout:.0f}s") from exc
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
