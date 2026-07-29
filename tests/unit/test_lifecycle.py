"""Provider lifecycle layer (0.11 Phase A): LifecycleStrategy interface + ManagedVllmStrategy.

ADDITIVE scaffolding — the strategy DELEGATES to provider/server.py (zero behavior change) and
no caller routes through it yet (Phase C). These tests monkeypatch server.* on the module
object (the harness's module-attribute pattern) to prove the delegation chain and the NEW
name-based liveness, without launching anything real.
"""
from __future__ import annotations

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.provider import server
from localharness.provider.lifecycle import (
    LifecycleStrategy,
    LiveEndpoint,
    Liveness,
    ManagedVllmStrategy,
    SpawnedProcessStrategy,
)


def _docker_srv(**over) -> ManagedServerConfig:
    kw = dict(launch="docker", docker_image="img:tag", model="m", port=8000)
    kw.update(over)
    return ManagedServerConfig(**kw)


def _binary_srv(**over) -> ManagedServerConfig:
    kw = dict(launch="binary", binary="/usr/bin/vllm", model="m", port=8001)
    kw.update(over)
    return ManagedServerConfig(**kw)


def _llamacpp_srv(**over) -> ManagedServerConfig:
    kw = dict(
        runtime="llamacpp", launch="binary", binary="/x/llama-server",
        model="/x/model.gguf", port=8080,
        extra_args=["-c", "32768", "-a", "qwen3.6-35b-a3b"],
    )
    kw.update(over)
    return ManagedServerConfig(**kw)


# ---------------------------------------------------------------------------
# free_accelerator — the GPU-lock (Phase C2): at most one heavy server up
# ---------------------------------------------------------------------------


class _RecordingStrategy:
    """A fake strategy that records stop() and forbids activate()/liveness() — free_accelerator
    must only ever STOP, never launch or probe."""
    def __init__(self, log):
        self._log = log

    async def stop(self, spec, config_dir):
        self._log.append(("stop", spec, config_dir))

    async def activate(self, *a, **k):
        raise AssertionError("free_accelerator must never activate")

    def liveness(self, *a, **k):
        raise AssertionError("free_accelerator must never check liveness")


async def test_free_accelerator_stops_different_heavy_incumbent(tmp_path, monkeypatch):
    """A cold heavy target with a DIFFERENT incumbent up → verified-stop the incumbent (through its
    OWN strategy) + settle, and RETURN it so a failed launch can restore it. Encodes the box rule."""
    from localharness.provider import lifecycle
    log: list = []
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: _RecordingStrategy(log))
    incumbent_spec = _docker_srv(gpu=True)
    target = _llamacpp_srv(gpu=True)
    returned = await lifecycle.free_accelerator(
        (incumbent_spec, "http://127.0.0.1:8000/v1"), target, tmp_path, settle_seconds=0.0,
    )
    assert log == [("stop", incumbent_spec, tmp_path)]
    assert returned == (incumbent_spec, "http://127.0.0.1:8000/v1")


async def test_free_accelerator_noop_when_incumbent_is_target(tmp_path, monkeypatch):
    """Same-server restart (incumbent IS the target — e.g. the primary vLLM reloading a new model):
    free_accelerator does NOT stop; the caller's own stop→activate performs the reload."""
    from localharness.provider import lifecycle
    log: list = []
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: _RecordingStrategy(log))
    spec = _docker_srv(gpu=True)
    returned = await lifecycle.free_accelerator(
        (spec, "http://127.0.0.1:8000/v1"), spec, tmp_path, settle_seconds=0.0,
    )
    assert returned is None
    assert log == []


async def test_free_accelerator_noop_when_nothing_heavy_up(tmp_path, monkeypatch):
    """No heavy incumbent (bound to a CPU-light peer, or an unmanaged primary) → nothing to free."""
    from localharness.provider import lifecycle
    log: list = []
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: _RecordingStrategy(log))
    returned = await lifecycle.free_accelerator(
        None, _llamacpp_srv(gpu=True), tmp_path, settle_seconds=0.0,
    )
    assert returned is None
    assert log == []


# ---------------------------------------------------------------------------
# Interface shape
# ---------------------------------------------------------------------------


def test_managed_vllm_strategy_satisfies_protocol():
    """The concrete strategy structurally implements the LifecycleStrategy Protocol."""
    assert isinstance(ManagedVllmStrategy(), LifecycleStrategy)


# ---------------------------------------------------------------------------
# activate — serve_command -> start_server -> wait_ready (both launch modes)
# ---------------------------------------------------------------------------


async def test_activate_docker_delegates_and_returns_live_endpoint(tmp_path, monkeypatch):
    """activate runs the real server.py chain and returns a LiveEndpoint whose served_models
    come from wait_ready and whose handle is the CONTAINER NAME in docker mode (its stop
    target). on_poll + timeout are threaded straight through to wait_ready — no behavior added."""
    calls: dict = {}

    def fake_serve_command(spec):
        calls["serve_command"] = spec
        return ["CMD", "--model", "m"]

    def fake_start_server(config_dir, cmd):
        calls["start_server"] = (config_dir, cmd)
        return 4321

    async def fake_wait_ready(base_url, config_dir=None, timeout_seconds=1800.0, poll_seconds=3.0, on_poll=None):
        calls["wait_ready"] = (base_url, config_dir, timeout_seconds)
        if on_poll is not None:
            on_poll(1.0)
        return ["served-m"]

    monkeypatch.setattr(server, "serve_command", fake_serve_command)
    monkeypatch.setattr(server, "start_server", fake_start_server)
    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    srv = _docker_srv()
    seen_poll: list[float] = []
    ep = await ManagedVllmStrategy().activate(
        srv, tmp_path, "http://localhost:8000/v1", timeout_seconds=42.0, on_poll=seen_poll.append,
    )

    assert isinstance(ep, LiveEndpoint)
    assert ep.base_url == "http://localhost:8000/v1"
    assert ep.served_models == ["served-m"]
    assert ep.handle == server.DOCKER_CONTAINER_NAME              # docker stop-handle = container name
    assert calls["serve_command"] is srv                          # spec passed through unchanged
    assert calls["start_server"] == (tmp_path, ["CMD", "--model", "m"])
    assert calls["wait_ready"] == ("http://localhost:8000/v1", tmp_path, 42.0)
    assert seen_poll == [1.0]                                     # on_poll delegated to wait_ready


async def test_activate_binary_handle_is_the_launched_pid(tmp_path, monkeypatch):
    """binary mode: the handle is the pid start_server returned (there the pid IS the server)."""
    monkeypatch.setattr(server, "serve_command", lambda spec: ["/usr/bin/vllm", "serve", "m"])
    monkeypatch.setattr(server, "start_server", lambda config_dir, cmd: 9999)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        return ["served-m"]

    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    ep = await ManagedVllmStrategy().activate(_binary_srv(), tmp_path, "http://localhost:8001/v1")
    assert ep.handle == 9999
    assert ep.served_models == ["served-m"]


# ---------------------------------------------------------------------------
# stop — the #100 verified stop, with launch threaded from the spec
# ---------------------------------------------------------------------------


async def test_stop_delegates_to_verified_stop_server(tmp_path, monkeypatch):
    """stop -> server.stop_server(config_dir, launch=spec.launch) — the existing verified stop."""
    seen: dict = {}

    def fake_stop(config_dir, launch="binary", **kw):
        seen["args"] = (config_dir, launch)
        return True

    monkeypatch.setattr(server, "stop_server", fake_stop)
    await ManagedVllmStrategy().stop(_docker_srv(), tmp_path)
    assert seen["args"] == (tmp_path, "docker")


# ---------------------------------------------------------------------------
# liveness — NAME-based for docker (the fix), pid-based for binary
# ---------------------------------------------------------------------------


def test_liveness_docker_is_name_based_never_pidfile(tmp_path, monkeypatch):
    """docker liveness reads the CONTAINER by name (docker_container_running); server_pid — the
    docker-run CLIENT pid — must NEVER be consulted in docker mode (the bug this fixes)."""
    def _boom(*a, **k):
        raise AssertionError("server_pid must not be called for docker liveness")

    monkeypatch.setattr(server, "server_pid", _boom)

    monkeypatch.setattr(server, "docker_container_running", lambda name=server.DOCKER_CONTAINER_NAME: True)
    live = ManagedVllmStrategy().liveness(_docker_srv(), tmp_path)
    assert isinstance(live, Liveness)
    assert live.alive is True
    assert server.DOCKER_CONTAINER_NAME in (live.detail or "")

    monkeypatch.setattr(server, "docker_container_running", lambda name=server.DOCKER_CONTAINER_NAME: False)
    assert ManagedVllmStrategy().liveness(_docker_srv(), tmp_path).alive is False


def test_liveness_binary_uses_server_pid(tmp_path, monkeypatch):
    """binary liveness = server_pid is not None (correct there — the pidfile pid IS vLLM)."""
    monkeypatch.setattr(server, "server_pid", lambda config_dir: 1234)
    assert ManagedVllmStrategy().liveness(_binary_srv(), tmp_path).alive is True
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    assert ManagedVllmStrategy().liveness(_binary_srv(), tmp_path).alive is False


# ---------------------------------------------------------------------------
# SpawnedProcessStrategy (0.11 Phase B): the harness spawns llama.cpp itself.
# Same server.py delegation chain as ManagedVllmStrategy, but the stop-handle is
# always the launched PID, stop is always the pid-group teardown (launch="binary"),
# and liveness is always pid-based (the pidfile pid IS the real llama-server here —
# never the docker orphan-client-pid case #99/#100). Mocked, launches nothing real.
# ---------------------------------------------------------------------------


def test_spawned_process_strategy_satisfies_protocol():
    assert isinstance(SpawnedProcessStrategy(), LifecycleStrategy)


async def test_spawned_activate_delegates_and_handle_is_pid(tmp_path, monkeypatch):
    """activate runs serve_command -> start_server -> wait_ready and returns a LiveEndpoint whose
    handle is the launched PID (a spawned process, never a container name). on_poll + timeout are
    threaded straight through to wait_ready."""
    calls: dict = {}

    def fake_serve_command(spec):
        calls["serve_command"] = spec
        return ["llama-server", "-m", "x"]

    def fake_start_server(config_dir, cmd):
        calls["start_server"] = (config_dir, cmd)
        return 5555

    async def fake_wait_ready(base_url, config_dir=None, timeout_seconds=1800.0, poll_seconds=3.0, on_poll=None):
        calls["wait_ready"] = (base_url, config_dir, timeout_seconds)
        if on_poll is not None:
            on_poll(2.0)
        return ["qwen3.6-35b-a3b"]

    monkeypatch.setattr(server, "serve_command", fake_serve_command)
    monkeypatch.setattr(server, "start_server", fake_start_server)
    monkeypatch.setattr(server, "wait_ready", fake_wait_ready)

    srv = _llamacpp_srv()
    seen_poll: list[float] = []
    ep = await SpawnedProcessStrategy().activate(
        srv, tmp_path, "http://127.0.0.1:8080/v1", timeout_seconds=99.0, on_poll=seen_poll.append,
    )

    assert isinstance(ep, LiveEndpoint)
    assert ep.base_url == "http://127.0.0.1:8080/v1"
    assert ep.served_models == ["qwen3.6-35b-a3b"]
    assert ep.handle == 5555                                     # spawned stop-handle = the real pid
    assert calls["serve_command"] is srv
    assert calls["start_server"] == (tmp_path, ["llama-server", "-m", "x"])
    assert calls["wait_ready"] == ("http://127.0.0.1:8080/v1", tmp_path, 99.0)
    assert seen_poll == [2.0]


async def test_spawned_stop_uses_binary_verified_stop(tmp_path, monkeypatch):
    """stop -> server.stop_server(config_dir, launch="binary"): the pid-group teardown, ALWAYS —
    a spawned server is a single tracked process, never a docker-run client."""
    seen: dict = {}

    def fake_stop(config_dir, launch="binary", **kw):
        seen["args"] = (config_dir, launch)
        return True

    monkeypatch.setattr(server, "stop_server", fake_stop)
    await SpawnedProcessStrategy().stop(_llamacpp_srv(), tmp_path)
    assert seen["args"] == (tmp_path, "binary")


def test_spawned_liveness_is_pid_based_never_docker(tmp_path, monkeypatch):
    """spawned liveness = server_pid is not None (the pidfile pid IS the llama-server process);
    docker_container_running — the docker-only path — must NEVER be consulted here."""
    def _boom(*a, **k):
        raise AssertionError("docker_container_running must not be called for a spawned process")

    monkeypatch.setattr(server, "docker_container_running", _boom)

    monkeypatch.setattr(server, "server_pid", lambda config_dir: 5555)
    live = SpawnedProcessStrategy().liveness(_llamacpp_srv(), tmp_path)
    assert isinstance(live, Liveness)
    assert live.alive is True
    assert "5555" in (live.detail or "")

    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    assert SpawnedProcessStrategy().liveness(_llamacpp_srv(), tmp_path).alive is False


# ---------------------------------------------------------------------------
# DaemonStrategy — Ollama: harness spawns `ollama serve`, loads the model, OWNS the stop (Phase D)
# ---------------------------------------------------------------------------


def _ollama_srv(**over) -> ManagedServerConfig:
    kw = dict(runtime="ollama", model="qwen2.5:7b", port=11434, gpu=False)
    kw.update(over)
    return ManagedServerConfig(**kw)


def test_daemon_strategy_satisfies_protocol():
    from localharness.provider import lifecycle
    assert isinstance(lifecycle.DaemonStrategy(), LifecycleStrategy)


async def test_daemon_activate_spawns_daemon_and_loads_target_model(tmp_path, monkeypatch):
    """activate spawns `ollama serve` (serve_command→start_server), waits for the daemon, and
    warm-LOADS the target tag → served_models is [the tag] (NOT the full pulled list), handle=pid."""
    from localharness.provider import lifecycle
    calls: dict = {}
    monkeypatch.setattr(server, "serve_command",
                        lambda spec: ["env", "OLLAMA_HOST=127.0.0.1:11434", "ollama", "serve"])

    def fake_start(cd, cmd):
        calls["start"] = (cd, cmd)
        return 5555

    monkeypatch.setattr(server, "start_server", fake_start)

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        calls["wait"] = base_url
        return ["qwen2.5:7b", "qwen3"]  # PULLED-to-disk models (resident or not)

    monkeypatch.setattr(server, "wait_ready", fake_wait)
    loaded: dict = {}

    async def fake_load(self, base_url, model):
        loaded["load"] = (base_url, model)

    monkeypatch.setattr(lifecycle.DaemonStrategy, "_load", fake_load)

    async def _not_up(self, v1):  # no pre-existing daemon on the port
        return False
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_daemon_already_up", _not_up)

    ep = await lifecycle.DaemonStrategy().activate(_ollama_srv(), tmp_path, "http://localhost:11434/v1")
    assert ep.served_models == ["qwen2.5:7b"]
    assert ep.handle == 5555
    assert loaded["load"] == ("http://localhost:11434/v1", "qwen2.5:7b")
    assert calls["wait"] == "http://localhost:11434/v1"


async def test_daemon_activate_pulls_when_tag_not_on_disk(tmp_path, monkeypatch):
    """If the tag isn't in wait_ready's pulled list, activate pulls it BEFORE loading (pull→load)."""
    from localharness.provider import lifecycle
    monkeypatch.setattr(server, "serve_command", lambda spec: ["ollama", "serve"])
    monkeypatch.setattr(server, "start_server", lambda cd, cmd: 1)

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        return ["other:1b"]  # target NOT present on disk

    monkeypatch.setattr(server, "wait_ready", fake_wait)
    order: list = []

    async def fake_pull(self, spec):
        order.append(("pull", spec.model))

    async def fake_load(self, base_url, model):
        order.append(("load", model))

    monkeypatch.setattr(lifecycle.DaemonStrategy, "_pull", fake_pull)
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_load", fake_load)

    async def _not_up(self, v1):
        return False
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_daemon_already_up", _not_up)

    await lifecycle.DaemonStrategy().activate(_ollama_srv(model="target:7b"), tmp_path, "http://x/v1")
    assert order == [("pull", "target:7b"), ("load", "target:7b")]


# --- D3 critic fixes: activate honors the (RuntimeError|TimeoutError) contract + /v1 + fail-fast --- #


async def test_daemon_load_and_pull_map_native_errors(tmp_path, monkeypatch):
    """_load maps httpx timeout→TimeoutError & http-status→RuntimeError; _pull maps a missing binary
    (FileNotFoundError/OSError)→RuntimeError. All within the activate failure contract."""
    import httpx
    from localharness.provider import lifecycle
    strat = lifecycle.DaemonStrategy()

    class _TimeoutClient:
        def __init__(self, *a, **k): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(httpx, "AsyncClient", _TimeoutClient)
    with pytest.raises(TimeoutError):
        await strat._load("http://x/v1", "m")

    class _StatusClient(_TimeoutClient):
        async def post(self, *a, **k):
            return httpx.Response(404, request=httpx.Request("POST", "http://x"))

    monkeypatch.setattr(httpx, "AsyncClient", _StatusClient)
    with pytest.raises(RuntimeError):
        await strat._load("http://x/v1", "m")

    # _pull with a binary that does not exist → RuntimeError (not a bare OSError)
    with pytest.raises(RuntimeError):
        await strat._pull(_ollama_srv(binary="/no/such/ollama-binary-xyz", model="m"))


async def test_daemon_activate_fails_fast_on_preexisting_daemon(tmp_path, monkeypatch):
    """A daemon already listening on the port → activate raises (loud) BEFORE spawning a second one,
    rather than silently mis-tracking it (which a later stop would miss = silent two-heavy)."""
    from localharness.provider import lifecycle
    spawned = {"n": 0}
    monkeypatch.setattr(server, "start_server", lambda cd, cmd: spawned.__setitem__("n", spawned["n"] + 1) or 1)

    async def _already_up(self, v1):
        return True
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_daemon_already_up", _already_up)
    with pytest.raises(RuntimeError, match="already listening"):
        await lifecycle.DaemonStrategy().activate(_ollama_srv(), tmp_path, "http://localhost:11434/v1")
    assert spawned["n"] == 0  # never spawned a second daemon


async def test_daemon_activate_normalizes_bare_base_url_to_v1(tmp_path, monkeypatch):
    """A BARE base_url (no /v1 — EndpointRef allows it for Ollama) is normalized to /v1 for wait_ready,
    so readiness polls /v1/models (not /models → 404 → 1800s hang)."""
    from localharness.provider import lifecycle
    seen = {}

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        seen["wait_url"] = base_url
        return ["qwen2.5:7b"]

    monkeypatch.setattr(server, "serve_command", lambda spec: ["ollama", "serve"])
    monkeypatch.setattr(server, "start_server", lambda cd, cmd: 1)
    monkeypatch.setattr(server, "wait_ready", fake_wait)

    async def _not_up(self, v1):
        return False

    async def _noop_load(self, base_url, model):
        return None
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_daemon_already_up", _not_up)
    monkeypatch.setattr(lifecycle.DaemonStrategy, "_load", _noop_load)

    ep = await lifecycle.DaemonStrategy().activate(_ollama_srv(), tmp_path, "http://localhost:11434")
    assert seen["wait_url"] == "http://localhost:11434/v1"   # normalized
    assert ep.base_url == "http://localhost:11434/v1"


async def test_daemon_stop_kills_whole_daemon_not_keepalive(tmp_path, monkeypatch):
    """Owner ruling: stop = kill the WHOLE daemon (stop_server launch='binary'), off the loop —
    NOT a per-model keep_alive:0 unload."""
    from localharness.provider import lifecycle
    seen: dict = {}
    monkeypatch.setattr(server, "stop_server",
                        lambda cd, launch="binary": seen.setdefault("args", (cd, launch)) or True)
    await lifecycle.DaemonStrategy().stop(_ollama_srv(), tmp_path)
    assert seen["args"] == (tmp_path, "binary")


def test_daemon_liveness_is_daemon_pid(tmp_path, monkeypatch):
    from localharness.provider import lifecycle
    monkeypatch.setattr(server, "server_pid", lambda cd: 4242)
    assert lifecycle.DaemonStrategy().liveness(_ollama_srv(), tmp_path).alive is True
    monkeypatch.setattr(server, "server_pid", lambda cd: None)
    assert lifecycle.DaemonStrategy().liveness(_ollama_srv(), tmp_path).alive is False


def test_daemon_root_strips_v1():
    from localharness.provider import lifecycle
    assert lifecycle.DaemonStrategy._daemon_root("http://localhost:11434/v1") == "http://localhost:11434"
    assert lifecycle.DaemonStrategy._daemon_root("http://localhost:11434/v1/") == "http://localhost:11434"


def test_strategy_for_ollama_dispatches_daemon():
    from localharness.provider import lifecycle
    assert isinstance(lifecycle.strategy_for(_ollama_srv()), lifecycle.DaemonStrategy)
    assert isinstance(lifecycle.strategy_for(_llamacpp_srv()), lifecycle.SpawnedProcessStrategy)
    assert isinstance(lifecycle.strategy_for(_docker_srv()), lifecycle.ManagedVllmStrategy)


# ---------------------------------------------------------------------------
# LmsStrategy — LM Studio: harness drives the headless `lms` CLI; STOP = `lms daemon down` (Phase D5)
# ---------------------------------------------------------------------------


def _lms_srv(**over) -> ManagedServerConfig:
    kw = dict(runtime="lmstudio", model="qwen2.5-0.5b-instruct",
              binary="/home/u/.lmstudio/bin/lms", port=1234, gpu=False)
    kw.update(over)
    return ManagedServerConfig(**kw)


def test_lms_strategy_satisfies_protocol():
    from localharness.provider import lifecycle
    assert isinstance(lifecycle.LmsStrategy(), LifecycleStrategy)


async def test_lms_activate_brings_up_daemon_loads_and_serves(tmp_path, monkeypatch):
    """activate drives, IN ORDER: `lms daemon up` → `lms load <model> --gpu off -y` (CPU, gpu=False)
    → `lms server start --port <base_url port> --bind 127.0.0.1`, then wait_ready(v1). served_models
    is [the model key] (NOT wait_ready's raw list, which also carries the embedding model); handle is
    the opaque 'lms-daemon' (LmsStrategy owns no pid)."""
    from localharness.provider import lifecycle
    calls: list = []

    async def fake_run(self, *args, **kw):
        calls.append(args)

    async def _not_up(self, v1):
        return False

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        calls.append(("wait", base_url))
        return ["qwen2.5-0.5b-instruct", "text-embedding-nomic-embed-text-v1.5"]

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _not_up)
    monkeypatch.setattr(server, "wait_ready", fake_wait)

    ep = await lifecycle.LmsStrategy().activate(_lms_srv(), tmp_path, "http://127.0.0.1:1234/v1")
    L = "/home/u/.lmstudio/bin/lms"
    assert calls[0] == (L, "daemon", "up")
    assert calls[1] == (L, "load", "qwen2.5-0.5b-instruct", "--gpu", "off", "-y")
    assert calls[2] == (L, "server", "start", "--port", "1234", "--bind", "127.0.0.1")
    assert ("wait", "http://127.0.0.1:1234/v1") in calls
    assert ep.served_models == ["qwen2.5-0.5b-instruct"]  # the key, not the embedding model too
    assert ep.handle == "lms-daemon"


async def test_lms_activate_uses_gpu_max_when_heavy(tmp_path, monkeypatch):
    """spec.gpu True (a heavy peer under the GPU-lock) → `lms load --gpu max` (full offload)."""
    from localharness.provider import lifecycle
    seen: dict = {}

    async def fake_run(self, *args, **kw):
        if len(args) > 1 and args[1] == "load":
            seen["load"] = args

    async def _not_up(self, v1):
        return False

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        return ["m"]

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _not_up)
    monkeypatch.setattr(server, "wait_ready", fake_wait)
    await lifecycle.LmsStrategy().activate(_lms_srv(gpu=True), tmp_path, "http://127.0.0.1:1234/v1")
    a = seen["load"]
    assert a[a.index("--gpu") + 1] == "max"


async def test_lms_activate_serves_on_base_url_port(tmp_path, monkeypatch):
    """`lms server start` binds the port from the CALLER's base_url (repl rebinds the client THERE
    after activate) — not spec.port when they differ."""
    from localharness.provider import lifecycle
    seen: dict = {}

    async def fake_run(self, *args, **kw):
        if len(args) > 1 and args[1] == "server":
            seen["start"] = args

    async def _not_up(self, v1):
        return False

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        return ["m"]

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _not_up)
    monkeypatch.setattr(server, "wait_ready", fake_wait)
    await lifecycle.LmsStrategy().activate(_lms_srv(port=1234), tmp_path, "http://127.0.0.1:9099/v1")
    a = seen["start"]
    assert a[a.index("--port") + 1] == "9099"


async def test_lms_run_maps_native_errors():
    """_run MAPS native failures into the activate contract: a missing binary (OSError)→RuntimeError,
    a non-zero exit not in ok_codes→RuntimeError, a timeout→TimeoutError — and ok_codes lets the
    idempotent `lms daemon down` (rc=1 already-down) pass. Real subprocesses (/bin/sh), no network."""
    from localharness.provider import lifecycle
    strat = lifecycle.LmsStrategy()
    with pytest.raises(RuntimeError):
        await strat._run("/no/such/lms-binary-xyz", "daemon", "up")
    with pytest.raises(RuntimeError):
        await strat._run("/bin/sh", "-c", "exit 3")
    with pytest.raises(TimeoutError):
        await strat._run("/bin/sh", "-c", "exec sleep 5", timeout=0.1)  # exec: no orphan grandchild
    await strat._run("/bin/sh", "-c", "exit 1", ok_codes=(0, 1))  # already-down tolerated
    await strat._run("/bin/sh", "-c", "exit 0")                    # success


async def test_lms_activate_fails_fast_on_preexisting_server(tmp_path, monkeypatch):
    """A server already answering at v1 (one the harness didn't start) → activate raises loudly
    BEFORE running any `lms` command, rather than mis-tracking it (a later `lms daemon down` would
    then kill a backend we don't own = silent two-heavy)."""
    from localharness.provider import lifecycle
    ran = {"n": 0}

    async def fake_run(self, *args, **kw):
        ran["n"] += 1

    async def _already_up(self, v1):
        return True

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _already_up)
    with pytest.raises(RuntimeError, match="already listening"):
        await lifecycle.LmsStrategy().activate(_lms_srv(), tmp_path, "http://127.0.0.1:1234/v1")
    assert ran["n"] == 0


async def test_lms_stop_is_whole_daemon_down_not_server_stop(tmp_path, monkeypatch):
    """Owner ruling + empirically settled: stop = `lms daemon down` (whole daemon = verified GPU-free),
    NOT `lms server stop` (which leaves the model resident). ok_codes (0,1) so an already-down daemon
    (rc=1) is success, not a spurious failure."""
    from localharness.provider import lifecycle
    seen: dict = {}

    async def fake_run(self, *args, ok_codes=(0,), **kw):
        seen["args"] = args
        seen["ok_codes"] = ok_codes

    async def _gone(self, v1):  # endpoint gone after daemon down → the verify passes
        return False

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _gone)
    await lifecycle.LmsStrategy().stop(_lms_srv(), tmp_path)
    assert seen["args"] == ("/home/u/.lmstudio/bin/lms", "daemon", "down")
    assert seen["ok_codes"] == (0, 1)


async def test_lms_stop_fails_explicit_if_still_serving(tmp_path, monkeypatch):
    """stop VERIFIES the endpoint is gone after `lms daemon down` (the #100 fail-explicit doctrine):
    if LM Studio is somehow STILL answering, stop raises rather than reporting a false GPU-free — the
    caller then aborts the swap instead of launching a 2nd heavy on an occupied accelerator."""
    from localharness.provider import lifecycle

    async def fake_run(self, *args, **kw):
        pass

    async def _still_up(self, v1):
        return True  # endpoint NEVER goes away

    async def _no_sleep(*a, **k):
        pass

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _still_up)
    monkeypatch.setattr(lifecycle.asyncio, "sleep", _no_sleep)  # skip the ~3s poll
    with pytest.raises(RuntimeError, match="still answering"):
        await lifecycle.LmsStrategy().stop(_lms_srv(), tmp_path)


async def test_lms_activate_normalizes_bare_base_url_to_v1(tmp_path, monkeypatch):
    """A bare base_url (no /v1) is normalized so wait_ready polls /v1/models and the returned
    LiveEndpoint carries the /v1 form."""
    from localharness.provider import lifecycle
    seen = {}

    async def fake_run(self, *args, **kw):
        pass

    async def _not_up(self, v1):
        return False

    async def fake_wait(base_url, config_dir=None, timeout_seconds=1800.0, on_poll=None):
        seen["wait"] = base_url
        return ["m"]

    monkeypatch.setattr(lifecycle.LmsStrategy, "_run", fake_run)
    monkeypatch.setattr(lifecycle.LmsStrategy, "_server_already_up", _not_up)
    monkeypatch.setattr(server, "wait_ready", fake_wait)
    ep = await lifecycle.LmsStrategy().activate(_lms_srv(), tmp_path, "http://127.0.0.1:1234")
    assert seen["wait"] == "http://127.0.0.1:1234/v1"
    assert ep.base_url == "http://127.0.0.1:1234/v1"


def test_lms_liveness_is_endpoint_based(monkeypatch, tmp_path):
    """liveness is ENDPOINT-based (LmsStrategy owns no pidfile): GET {v1}/models 200 → alive; a
    connection error → not alive."""
    import httpx
    from localharness.provider import lifecycle

    class _Resp:
        status_code = 200

    monkeypatch.setattr(httpx, "get", lambda *a, **k: _Resp())
    assert lifecycle.LmsStrategy().liveness(_lms_srv(), tmp_path).alive is True

    def _boom(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", _boom)
    assert lifecycle.LmsStrategy().liveness(_lms_srv(), tmp_path).alive is False


def test_strategy_for_lmstudio_dispatches_lms():
    from localharness.provider import lifecycle
    assert isinstance(lifecycle.strategy_for(_lms_srv()), lifecycle.LmsStrategy)


# --- Critic-round hardening: probes never propagate on a malformed URL; _run's kill-race is safe --- #


async def test_lms_probe_swallows_malformed_base_url():
    """_server_already_up is a BOOLEAN probe: a malformed base_url (bad port — a DELIBERATE
    probe-time skip per validate_base_url) must return False, NOT escape as httpx.InvalidURL / an
    OverflowError group and crash the /model swap. Exercises the REAL (unmocked) probe."""
    from localharness.provider import lifecycle
    strat = lifecycle.LmsStrategy()
    assert await strat._server_already_up("http://127.0.0.1:abc/v1") is False     # InvalidURL
    assert await strat._server_already_up("http://127.0.0.1:99999/v1") is False   # OverflowError group


async def test_daemon_probe_swallows_malformed_base_url():
    """DaemonStrategy._daemon_already_up hardened the same way (shipped-twin consistency)."""
    from localharness.provider import lifecycle
    strat = lifecycle.DaemonStrategy()
    assert await strat._daemon_already_up("http://127.0.0.1:abc/v1") is False
    assert await strat._daemon_already_up("http://127.0.0.1:99999/v1") is False


def test_lms_liveness_swallows_bad_port(tmp_path):
    """liveness likewise never raises on an out-of-range spec.port (99999 → OverflowError group)."""
    from localharness.provider import lifecycle
    assert lifecycle.LmsStrategy().liveness(_lms_srv(port=99999), tmp_path).alive is False


async def test_lms_run_timeout_tolerates_proc_dead_at_kill(monkeypatch):
    """_run's timeout path guards proc.kill()/wait() against ProcessLookupError (the proc exiting in
    the timeout race) → it still raises TimeoutError, never a leaked OSError that would crash the
    session. A fake proc whose communicate() hangs then whose kill()/wait() raise ProcessLookupError."""
    from localharness.provider import lifecycle

    class _DeadProc:
        returncode = None

        async def communicate(self):
            await lifecycle.asyncio.sleep(10)  # never returns before the tiny timeout

        def kill(self):
            raise ProcessLookupError  # already exited in the race

        async def wait(self):
            raise ProcessLookupError

    async def fake_exec(*a, **k):
        return _DeadProc()

    monkeypatch.setattr(lifecycle.asyncio, "create_subprocess_exec", fake_exec)
    with pytest.raises(TimeoutError):
        await lifecycle.LmsStrategy()._run("/x/lms", "daemon", "up", timeout=0.05)
