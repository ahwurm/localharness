"""A model server the harness launches that is set to require its key still answers every harness
probe; one that is not is launched and probed exactly as in 0.16 — through the real `_start_async`,
the REPL's /model and `localharness doctor`.

REAL: `_start_async` (loader, the launch branch, the window probe, the summary),
ManagedVllmStrategy (`serve_command`, `wait_ready`), `list_live_models`, `probe_served_window`, the
REPL's /model swap, peer launch and restore, doctor's endpoint probe.
FAKED: the capability probe (`_probe_llm` fails once so start launches the server, then answers),
the token counter (the start boundary stub), `start_server` (records the argv and the key; nothing
starts) and `server_pid` after it, and the model server itself — `server._TRANSPORT` for wait_ready
and `httpx.get` for every other probe. When the key is required the fake answers 401 to any request
without `Authorization: Bearer <key>`, as vLLM's key check does for `/v1/*`. No socket is opened.
NOT PROVEN here: a real vLLM reading VLLM_API_KEY (vLLM is not installed on this machine).
"""
from __future__ import annotations

import asyncio
import socket
from types import SimpleNamespace

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.models import EndpointRef, HarnessConfig, ManagedServerConfig, ProviderConfig
from localharness.provider import server
from tests.unit.test_repl_model_cmd import _heavy_primary, _xrepl
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

KEY = "sk-SERVER-KEY"
BASE = "http://127.0.0.1:8099/v1"
PRIMARY = "http://localhost:8081/v1"  # the REPL doubles' primary endpoint
PEER = "http://127.0.0.1:8090/v1"
runner = CliRunner()


# ------------------------------------------------------------------------------- the fakes


class _Server:
    """The launched model server. `key` set: refuses (401) any request without its bearer."""

    def __init__(self, key: str | None):
        self.key = key
        self.seen: list[tuple[str, str, str | None]] = []  # (which probe, path, Authorization)
        self.refused = 0

    def answer(self, request: httpx.Request, via: str) -> httpx.Response:
        auth = request.headers.get("Authorization")
        self.seen.append((via, request.url.path, auth))
        if self.key and auth != f"Bearer {self.key}":
            self.refused += 1
            return httpx.Response(401, json={"error": "Unauthorized"})
        return httpx.Response(200, json={"data": [{"id": "m", "max_model_len": 262_144},
                                                  {"id": "test-model", "max_model_len": 262_144}]})


def _serve(monkeypatch, key: str | None) -> _Server:
    fake = _Server(key)

    def readiness(request: httpx.Request) -> httpx.Response:
        response = fake.answer(request, "wait_ready")
        if response.status_code == 401:  # fail fast: wait_ready would poll a refusing server for 30 min
            raise RuntimeError("the readiness probe was refused: it sent no key")
        return response

    monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(readiness))

    def get(url, **kw):
        return fake.answer(httpx.Request("GET", url, headers=kw.get("headers")), "get")

    monkeypatch.setattr(httpx, "get", get)
    return fake


def _record_launch(monkeypatch) -> list[dict]:
    launched: list[dict] = []

    def start_server(config_dir, cmd, *, api_key=None):
        launched.append({"cmd": list(cmd), "api_key": api_key})
        monkeypatch.setattr(server, "server_pid", lambda _d: 4242)  # the launched process lives
        return 4242

    monkeypatch.setattr(server, "start_server", start_server)
    return launched


def _probe_fails_once(monkeypatch) -> list[int]:
    calls: list[int] = []

    async def probe(llm, max_retries=3, delay=2.0):
        calls.append(1)
        return (False, None, None, "Connection error.") if len(calls) == 1 else (True, "native", 262_144, None)

    monkeypatch.setattr("localharness.cli.start_cmd._probe_llm", probe)
    return calls


def _record_dials(monkeypatch) -> list:
    real, dialed = socket.socket.connect, []

    def connect(self, address):
        dialed.append(address)
        return real(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    return dialed


def _config(path, *, base_url: str = BASE, key: str = KEY, require: bool | None = None,
            launched: bool = True) -> None:
    data: dict = {"version": "1",
                  "provider": {"provider_type": "vllm", "base_url": base_url, "api_key": key,
                               "default_model": "m" if launched else "test-model"},
                  "org": {"permissions": {"defaults_revision": CURRENT_DEFAULTS_REVISION}}}
    if launched:
        data["server"] = {"runtime": "vllm", "launch": "binary", "binary": "/usr/bin/true", "model": "m",
                          "port": 8099, "bind_all": False,
                          **({} if require is None else {"require_api_key": require})}
    (path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _summary(printed: list[str]) -> str:
    return next(line for line in printed if "startup)" in line)


async def _start(tmp_path, monkeypatch, *, key: str = KEY, require: bool | None, served_key: str | None,
                 **start_kw) -> SimpleNamespace:
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    _config(tmp_path, key=key, require=require)
    seen = SimpleNamespace(probes=_probe_fails_once(monkeypatch), launched=_record_launch(monkeypatch),
                           server=_serve(monkeypatch, served_key), dialed=_record_dials(monkeypatch),
                           printed=_capture_start_console(monkeypatch))
    await asyncio.wait_for(_start_async(None, False, False, str(tmp_path), **start_kw), timeout=60)
    assert not [a for a in seen.dialed if isinstance(a, tuple)], f"a socket was opened: {seen.dialed}"
    return seen


# ------------------------------------------------------------------------------- start


async def test_a_start_that_launches_a_server_requiring_its_key_reaches_the_banner(tmp_path, monkeypatch):
    seen = await _start(tmp_path, monkeypatch, require=True, served_key=KEY)

    assert "startup)" in _summary(seen.printed)
    (launch,) = seen.launched
    cmd = launch["cmd"]
    assert cmd[cmd.index("--host"):cmd.index("--host") + 2] == ["--host", "127.0.0.1"]
    assert not any(KEY in part for part in cmd)
    assert launch["api_key"] == KEY
    assert seen.server.refused == 0
    # readiness and the window probe both reached the server, each with the key
    assert {via for via, _path, _auth in seen.server.seen} == {"wait_ready", "get"}
    assert all(auth == f"Bearer {KEY}" for _via, _path, auth in seen.server.seen)
    assert KEY not in "\n".join(seen.printed)


async def test_by_default_the_launch_and_every_probe_stay_keyless(tmp_path, monkeypatch):
    seen = await _start(tmp_path, monkeypatch, require=None, served_key=None)

    assert "startup)" in _summary(seen.printed)
    (launch,) = seen.launched
    assert launch["api_key"] is None
    assert "--host" in launch["cmd"]  # this config is on loopback (bind_all: false)
    assert {via for via, _path, _auth in seen.server.seen} == {"wait_ready", "get"}
    assert [auth for _via, _path, auth in seen.server.seen if auth is not None] == []


async def test_a_required_key_that_is_none_launches_and_probes_keyless(tmp_path, monkeypatch):
    seen = await _start(tmp_path, monkeypatch, key="none", require=True, served_key=None)

    assert "startup)" in _summary(seen.printed)
    assert [launch["api_key"] for launch in seen.launched] == [None]
    assert [auth for _via, _path, auth in seen.server.seen if auth is not None] == []


async def test_list_models_on_a_server_requiring_its_key_lists_its_models(tmp_path, monkeypatch):
    seen = await _start(tmp_path, monkeypatch, require=True, served_key=KEY, list_models=True)

    assert any("m  (serving)" in line for line in seen.printed), seen.printed
    assert seen.server.refused == 0 and seen.launched == []


async def test_start_names_a_key_sent_over_plain_http(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    _serve(monkeypatch, None)
    dialed = _record_dials(monkeypatch)
    printed = _capture_start_console(monkeypatch)
    warning = "provider.api_key travels unencrypted to http://10.0.0.5:8000 — use https"

    _config(tmp_path, base_url="http://10.0.0.5:8000/v1", key="sk-x", launched=False)
    await _start_async(None, False, False, str(tmp_path))
    assert warning in _summary(printed)

    printed.clear()
    _config(tmp_path, base_url="http://127.0.0.1:8000/v1", key="sk-x", launched=False)
    await _start_async(None, False, False, str(tmp_path))
    assert "unencrypted" not in _summary(printed)
    assert not [a for a in dialed if isinstance(a, tuple)], f"a socket was opened: {dialed}"


# ------------------------------------------------------------------------------- doctor


@pytest.mark.parametrize("require", [True, False])
def test_doctors_endpoint_probe_sends_the_key_only_when_the_server_requires_it(tmp_path, monkeypatch,
                                                                                 require):
    _config(tmp_path, require=require)
    fake = _serve(monkeypatch, None)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: httpx.Response(404))  # /tokenize: not served
    result = runner.invoke(app, ["doctor", "--config-dir", str(tmp_path)])
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output

    models = [auth for via, path, auth in fake.seen if path == "/v1/models"]
    assert models and models[0] == (f"Bearer {KEY}" if require else None)
    assert KEY not in result.output


# ------------------------------------------------------------------------------- /model


def _record_strategies(monkeypatch, *, fail: tuple[str, ...] = ()) -> list[tuple]:
    """Every strategy is a recorder: activate keeps the base_url and the api_key keyword it got."""
    from localharness.provider import lifecycle

    log: list[tuple] = []

    class _Recorder:
        async def activate(self, spec, config_dir, base_url, **kw):
            log.append(("activate", base_url, {k: v for k, v in kw.items() if k == "api_key"}))
            if base_url in fail:
                raise RuntimeError("never came up")
            return lifecycle.LiveEndpoint(base_url=base_url, served_models=[spec.model], handle=1)

        async def stop(self, spec, config_dir):
            log.append(("stop", spec.model))

        def liveness(self, spec, config_dir):
            return lifecycle.Liveness(alive=False)

    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: _Recorder())
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(server, "list_cached_models", lambda: [])
    return log


def _harness(server_cfg=None, peers=()) -> HarnessConfig:
    return HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url=PRIMARY, default_model="model-a",
                                available_models=["model-a"], api_key="sk-P"),
        server=server_cfg, extra_endpoints=list(peers))


def _peer(require: bool) -> EndpointRef:
    return EndpointRef(name="peer-vllm", base_url=PEER, provider_type="vllm", api_key="sk-PEER", gpu=True,
                       lifecycle=ManagedServerConfig(launch="binary", binary="/x/vllm", model="peer-m", port=8090,
                                                     extra_args=["--served-model-name", "peer-m"],
                                                     require_api_key=require))


def _quiet(repl) -> None:
    async def nothing(*_a, **_kw):
        return ""

    repl._refresh_token_counter = nothing
    repl._persist_default_model = nothing
    repl._persist_active_endpoint = nothing


def _activations(log) -> list[tuple]:
    return [entry[1:] for entry in log if entry[0] == "activate"]


@pytest.mark.parametrize("require", [True, False])
async def test_the_model_swap_sends_the_providers_key_only_to_a_server_that_requires_it(tmp_path, monkeypatch,
                                                                                         require):
    log = _record_strategies(monkeypatch)
    managed = ManagedServerConfig(launch="docker", docker_image="vllm:latest", model="model-a",
                                  require_api_key=require, local_models=[{"name": "model-b", "path": "/x/B"}])
    repl, channel, _agent, _ = _xrepl(tmp_path, _harness(managed), {PRIMARY: ["model-a"]})
    _quiet(repl)

    await repl._handle_slash("/model model-b")

    assert _activations(log) == [(PRIMARY, {"api_key": "sk-P"} if require else {})], channel.messages


@pytest.mark.parametrize("require", [True, False])
async def test_a_peer_launch_sends_the_peers_own_key_only_when_it_requires_one(tmp_path, monkeypatch, require):
    log = _record_strategies(monkeypatch)
    harness = _harness(_heavy_primary(), [_peer(require)])
    repl, channel, _agent, _ = _xrepl(tmp_path, harness, {PRIMARY: ["model-a"], PEER: "unreachable"})
    _quiet(repl)

    await repl._handle_slash("/model peer-m")

    assert _activations(log) == [(PEER, {"api_key": "sk-PEER"} if require else {})], channel.messages


async def test_a_restored_incumbent_gets_its_own_key(tmp_path, monkeypatch):
    log = _record_strategies(monkeypatch, fail=(PEER,))
    primary = ManagedServerConfig(launch="docker", docker_image="vllm:latest", model="model-a",
                                  require_api_key=True)
    repl, channel, _agent, _ = _xrepl(tmp_path, _harness(primary, [_peer(False)]),
                                      {PRIMARY: ["model-a"], PEER: "unreachable"})
    _quiet(repl)

    await repl._handle_slash("/model peer-m")

    assert _activations(log) == [(PEER, {}), (PRIMARY, {"api_key": "sk-P"})], channel.messages
    assert "Restored" in channel.messages[-1]


@pytest.mark.parametrize("require", [True, False])
async def test_the_repl_model_list_and_window_probes_carry_a_required_key(tmp_path, monkeypatch, require):
    from localharness.agent import context as context_mod
    from localharness.cli import model_ops
    from localharness.cli.repl import OrchestratorREPL
    from tests.unit.test_repl_model_cmd import FakeChannel, FakeLLM

    listed: list[tuple] = []
    windows: list[tuple] = []
    monkeypatch.setattr(model_ops, "list_live_models",
                        lambda base_url, **kw: listed.append((base_url, kw)) or (["model-a"], True))
    monkeypatch.setattr(context_mod, "probe_served_window",
                        lambda base_url, model, ptype=None, **kw: windows.append((base_url, kw)) or 2_000)
    primary = ManagedServerConfig(launch="docker", docker_image="vllm:latest", model="model-a",
                                  require_api_key=require)
    attach = EndpointRef(name="attach", base_url="http://127.0.0.1:11434/v1", provider_type="ollama",
                         api_key="sk-ATTACH")
    agent = SimpleNamespace(_llm=FakeLLM(), _ctx=SimpleNamespace(max_context_tokens=1_000),
                            _config=SimpleNamespace(context=SimpleNamespace(resolve_budget=lambda m: (None, False)),
                                                    max_tokens=None))
    repl = OrchestratorREPL(orchestrator=SimpleNamespace(), agent_loop=agent, channel=FakeChannel(),
                            bus=SimpleNamespace(), config_dir=tmp_path,
                            harness_config=_harness(primary, [attach]))

    await repl._live_models(PRIMARY)
    await repl._live_models(attach.base_url)
    await repl._refresh_token_counter("model-a", base_url=PRIMARY, provider_type="vllm")

    want = {"api_key": "sk-P"} if require else {}
    assert listed == [(PRIMARY, want), (attach.base_url, {})]  # an attach-only peer: as before
    assert windows == [(PRIMARY, want)]
