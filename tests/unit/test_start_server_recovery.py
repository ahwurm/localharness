"""`start` wires the reconnect into a real session (07-06: D-21, D-27 item 1, D-41).

- one supplier object (`_server_recovery_supplier(repl_cell)`) goes to the delegation runner AND the
  root loop, and is late-bound to the REPL once it exists; ACP never builds a REPL, so its loop and
  runner share one poll-only pair;
- `provider.reconnect_wait_seconds` reaches the client's `LLMConfig` (without it the bound is silently
  off in production);
- `start`'s own down-server launch is taken under the launch lock with a liveness re-check inside it.

Hermetic (D-34): the boundaries of the other `start` suites are stubbed, the "other process" is a raw
fd in this process, nothing under `~/.localharness`, no server, no port.
"""
from __future__ import annotations

import asyncio
import os

import pytest
import yaml

from localharness.config.models import ManagedServerConfig
from localharness.cli.start_cmd import _server_recovery_supplier, _start_async, _start_managed_server
from localharness.provider import client as client_mod
from localharness.provider import lifecycle, server
from localharness.provider.lifecycle import LAUNCH_LOCK_WAIT_LINE, ServerRecovery
from tests.unit.launch_support import stub_liveness
from tests.unit.test_launch_paths_start import URL as LLAMA_URL
from tests.unit.test_launch_paths_start import _Launch, _llama
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries, _write_agent

fcntl = pytest.importorskip("fcntl")

DEAD_URL = "http://127.0.0.1:9/v1"  # the discard port: a live :8080 would rename the model


@pytest.fixture(autouse=True)
def _no_leaked_holds():
    yield
    assert lifecycle._HELD == {}


# --------------------------------------------------------------------------- the supplier


def test_an_empty_cell_gives_one_shared_poll_only_pair():
    supplier = _server_recovery_supplier([])
    first, second = supplier(), supplier()
    assert first is second
    assert isinstance(first, ServerRecovery)
    assert (first.base_url, first.config_dir, first.relaunch) == (None, None, None)


def test_two_suppliers_do_not_share_a_pair():
    assert _server_recovery_supplier([])() is not _server_recovery_supplier([])()


def test_a_filled_cell_answers_the_repls_pair_and_binding_is_late():
    class _Repl:
        def __init__(self):
            self.pair = ServerRecovery("http://x", None, None)

        def server_recovery(self):
            return self.pair

    cell: list = []
    supplier = _server_recovery_supplier(cell)
    before = supplier()
    repl = _Repl()
    cell.append(repl)
    assert supplier() is repl.pair
    assert supplier() is not before


# --------------------------------------------------------------------------- the wiring in a real start


class _World:
    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.repls: list = []
        self.runner_kwargs: list[dict] = []

        async def keep_repl(repl):
            self.repls.append(repl)

        _stub_start_boundaries(tmp_path, monkeypatch, repl_run=keep_repl)
        stub_liveness(monkeypatch, False)  # not alive: an absent pidfile
        _capture_start_console(monkeypatch)
        import localharness.agent.subagent as subagent
        real = subagent.make_explore_agent_runner

        def spy(**kw):
            self.runner_kwargs.append(kw)
            return real(**kw)

        monkeypatch.setattr(subagent, "make_explore_agent_runner", spy)

    def config(self, *, provider_extra: dict | None = None) -> None:
        data = {"version": "1",
                "provider": {"provider_type": "ollama", "base_url": DEAD_URL, "api_key": "none",
                             "default_model": "qwen", **(provider_extra or {})}}
        (self.tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    async def start(self, **kw):
        return await _start_async(None, False, False, str(self.tmp_path), **kw)

    @property
    def repl(self):
        return self.repls[-1]

    @property
    def loop(self):
        return self.repl._agent


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = _World(tmp_path, monkeypatch)
    w.config()
    return w


async def test_the_runner_and_the_root_loop_get_one_supplier(world):
    await world.start()
    (kw,) = world.runner_kwargs
    assert callable(kw["server_recovery"])
    assert kw["server_recovery"] is world.loop._server_recovery


async def test_the_supplier_is_late_bound_to_the_repl(world):
    await world.start()
    assert world.loop._server_recovery() is world.repl.server_recovery()


async def test_the_supplier_answers_before_the_repl_exists(world, monkeypatch):
    seen: list = []
    from localharness.cli import repl as repl_mod
    original = repl_mod.OrchestratorREPL.__init__

    def init(self, *a, **kw):
        # the loop is built; the REPL is not yet in the cell
        loop = kw["agent_loop"]
        seen.append(loop._server_recovery())
        original(self, *a, **kw)

    monkeypatch.setattr(repl_mod.OrchestratorREPL, "__init__", init)
    await world.start()
    (early,) = seen
    assert isinstance(early, ServerRecovery)
    assert (early.base_url, early.config_dir, early.relaunch) == (None, None, None)
    assert world.loop._server_recovery() is world.repl.server_recovery()


async def test_acp_shares_one_poll_only_pair_between_loop_and_runner(world):
    got: dict = {}

    class _Acp:
        async def serve(self, **kw):
            got.update(kw)

        def __getattr__(self, name):
            raise AttributeError(name)

    _write_agent(world.tmp_path / "agents", "solo")
    await world.start(channel_mode="acp", no_input=True, acp_channel=_Acp())
    (kw,) = world.runner_kwargs
    supplier = got["agent_loop"]._server_recovery
    assert supplier is kw["server_recovery"]
    pair = supplier()
    assert supplier() is pair
    assert (pair.base_url, pair.config_dir, pair.relaunch) == (None, None, None)
    assert world.repls == [], "ACP never builds a REPL"


async def test_the_configured_reconnect_bound_reaches_the_client(world):
    world.config(provider_extra={"reconnect_wait_seconds": 45})
    await world.start()
    assert world.loop._llm.config.reconnect_wait_seconds == 45.0


async def test_the_default_reconnect_bound_is_1800_seconds(world):
    await world.start()
    assert world.loop._llm.config.reconnect_wait_seconds == 1800.0


async def test_a_zero_bound_turns_the_reconnect_off(world):
    world.config(provider_extra={"reconnect_wait_seconds": 0})
    await world.start()
    assert world.loop._llm.config.reconnect_wait_seconds == 0.0


# --------------------------------------------------------------------------- start's launch under the lock


async def test_the_launch_runs_while_the_lock_is_held_and_releases_it(tmp_path, monkeypatch):
    _Launch(monkeypatch)
    stub_liveness(monkeypatch, False)
    held_at_launch: list = []

    def start(config_dir, cmd, **kw):
        held_at_launch.append(set(lifecycle._HELD))
        return 4321

    monkeypatch.setattr(server, "start_server", start)
    out: list = []
    assert await _start_managed_server(_llama(), tmp_path, LLAMA_URL, {}, endpoint_out=out) is None
    assert held_at_launch == [{server.lock_path(tmp_path).resolve()}]
    assert lifecycle._HELD == {}
    assert len(out) == 1


async def test_a_server_another_process_brought_up_meanwhile_is_waited_on_not_launched(tmp_path, monkeypatch):
    monkeypatch.setattr(client_mod, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    seen = _Launch(monkeypatch)
    up = {"now": False}
    stub_liveness(monkeypatch, lambda: up["now"])
    waited: list[tuple] = []

    async def wait_ready(base_url, **kw):
        waited.append((base_url, kw))
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    path = server.lock_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    holder = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(holder, fcntl.LOCK_EX)
    out: list = []
    task = asyncio.create_task(_start_managed_server(_llama(), tmp_path, LLAMA_URL, {}, endpoint_out=out))
    await asyncio.sleep(0.4)  # past the visibility threshold: the wait line has been printed
    assert not task.done()
    up["now"] = True  # the other process's server is up by the time its launch finishes
    os.close(holder)
    assert await asyncio.wait_for(task, 2) is None
    assert seen.cmds == [], "start launched over a server another process brought up"
    # the loading wait carries the proof it waits on THIS record's launch (D-18): `expect` is its identity
    assert waited == [(LLAMA_URL, {"config_dir": tmp_path,
                                   "expect": lifecycle.record_identity(_llama())})]
    assert out == []
    assert [line for line in seen.err_lines if line == LAUNCH_LOCK_WAIT_LINE] == [LAUNCH_LOCK_WAIT_LINE]
    assert "Managed llama.cpp is still loading — waiting..." in seen.lines
    assert not [line for line in seen.lines if "starting it" in line]


async def test_an_lm_studio_server_already_up_is_waited_on_with_no_pidfile_check(tmp_path, monkeypatch):
    """IN-55: LM Studio writes no pidfile, so waiting on it with the raw config dir read "exited during
    startup" at once; the wait uses the strategy's own dir (None for LM Studio), as the reconnect does."""
    _Launch(monkeypatch)
    stub_liveness(monkeypatch, True)
    waited: list[tuple] = []

    async def wait_ready(base_url, **kw):
        waited.append((base_url, kw))
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", wait_ready)
    lms = ManagedServerConfig(runtime="lmstudio", model="qwen", port=1234)
    assert await _start_managed_server(lms, tmp_path, "http://localhost:1234/v1", {}) is None
    assert waited == [("http://localhost:1234/v1", {"config_dir": None, "expect": None})]  # LM Studio has no identity


async def test_a_free_lock_prints_no_wait_line(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    stub_liveness(monkeypatch, False)
    assert await _start_managed_server(_llama(), tmp_path, LLAMA_URL, {}) is None
    assert LAUNCH_LOCK_WAIT_LINE not in seen.err_lines


async def test_a_refused_launch_never_creates_the_lock_file(tmp_path, monkeypatch):
    _Launch(monkeypatch)
    stub_liveness(monkeypatch, False)
    spec = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080)  # declared only: no binary
    refusal = await _start_managed_server(spec, tmp_path, LLAMA_URL, {})
    assert refusal and "llama-server" in refusal
    assert not server.lock_path(tmp_path).exists()
