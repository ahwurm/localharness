# ruff: noqa: F401, F811
"""Start holds the served window to the harness floor before any turn."""
from __future__ import annotations

import pytest
import typer

from localharness.agent.context import TokenCounter
from localharness.cli import start_serving
from localharness.cli.start_cmd import _start_async
from localharness.config.models import AgentConfig
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.procfs_support import fake_proc
from tests.unit.test_sense_serving import home
from tests.unit.test_start_cmd import _capture_err_console, _stub_nonvllm_start
from tests.unit.test_start_serving_e2e import World, world

UNIT_LINE = ("test-model on llm.service serves a 4,096-token window and the harness needs at least 15,360 — "
             "raise its -c, or stop the unit and let the harness launch it.")


def _stub_floor(monkeypatch, floor, unit="llm.service"):
    async def fake_floor(agent_config, skills=(), **kw):
        return floor
    monkeypatch.setattr(start_serving, "launch_floor", fake_floor)
    monkeypatch.setattr(start_serving, "served_unit", lambda base_url: unit)


def _spy_repl(monkeypatch):
    ran: list[int] = []

    async def run(self):
        ran.append(1)
    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", run)
    return ran


async def test_an_attach_to_a_4096_server_exits_1_with_the_unit_sentence_before_any_turn(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=4_096, probe_window=None)
    _stub_floor(monkeypatch, 15_360)
    ran = _spy_repl(monkeypatch)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit) as exit_:
        await _start_async(None, False, False, str(tmp_path))

    assert exit_.value.exit_code == 1
    assert errs == [UNIT_LINE]
    assert ran == []


async def test_a_window_with_no_reply_reserve_exits_1_with_the_floor_sentence(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=1_000, probe_window=None)
    _stub_floor(monkeypatch, 256)
    ran = _spy_repl(monkeypatch)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit) as exit_:
        await _start_async(None, False, False, str(tmp_path))

    assert exit_.value.exit_code == 1
    assert errs == [UNIT_LINE.replace("4,096", "1,000").replace("15,360", "256")]
    assert "too small" not in errs[0]
    assert ran == []


async def test_a_server_no_unit_holds_is_told_to_raise_its_window(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=4_096, probe_window=None)
    _stub_floor(monkeypatch, 15_360, unit=None)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit):
        await _start_async(None, False, False, str(tmp_path))

    assert errs == ["test-model at http://localhost:8080/v1 serves a 4,096-token window and the harness "
                    "needs at least 15,360 — raise the window at the server."]


async def test_a_window_at_the_floor_starts(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=16_384, probe_window=None)
    _stub_floor(monkeypatch, 15_360)
    ran = _spy_repl(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    assert ran == [1]


async def test_an_unknown_served_window_is_not_compared(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=None, probe_window=None)
    _stub_floor(monkeypatch, 15_360)
    ran = _spy_repl(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    assert ran == [1]


def test_the_offline_counter_estimates_bytes_and_never_builds_the_vocabulary(monkeypatch):
    import tiktoken
    built: list[str] = []

    class _Encoder:
        def encode(self, text, **kw):
            return [0] * len(text)

    def get_encoding(name):
        built.append(name)
        return _Encoder()
    monkeypatch.setattr(tiktoken, "get_encoding", get_encoding)

    assert TokenCounter(offline=True).count("abcdef") == 2
    assert built == []
    assert TokenCounter().count("abcdef") == 6 and built == ["cl100k_base"]


def _spy_activate(monkeypatch, refusal=None):
    seen: list[dict] = []
    real = lifecycle.strategy_for

    def strategy_for(spec, **kw):
        strategy = real(spec, **kw)
        activate = strategy.activate

        async def spy(*a, **k):
            seen.append(k)
            if refusal is not None:
                raise managed_server.LaunchRefused(refusal)
            return await activate(*a, **k)
        strategy.activate = spy
        return strategy
    monkeypatch.setattr(lifecycle, "strategy_for", strategy_for)
    return seen


async def test_the_managed_launch_hands_the_pre_launch_floor_to_activate(world, monkeypatch):
    _stub_floor(monkeypatch, 12_288)
    seen = _spy_activate(monkeypatch)
    world.config()

    await world.start()

    assert [k["floor"] for k in seen] == [12_288]
    assert world.kinds() == ["start", "wait"]


async def test_a_priced_window_under_the_floor_is_refused_before_anything_starts(world, monkeypatch):
    line = ("qwen-27b-swift-1.5 fits only a 9,216-token window here and the harness needs at least 15,360 — "
            "free GPU memory, or pick a smaller model.")
    _stub_floor(monkeypatch, 15_360)
    seen = _spy_activate(monkeypatch, refusal=line)
    world.config()

    assert await world.start_exit() == 1

    assert [k["floor"] for k in seen] == [15_360]
    assert not world.starts
    assert world.err_lines == [line]


async def test_the_launch_floor_is_the_live_figure_and_ignores_the_cache(tmp_path):
    cfg = AgentConfig(name="orchestrator", role="r", model="inherit")
    live = await start_serving.launch_floor(cfg)

    lifecycle.record_floor(tmp_path, live + 5_000, {})

    assert await start_serving.launch_floor(cfg) == live
    assert lifecycle.raised_floor(live, lifecycle.cached_floor(tmp_path)) == live + 5_000


def _capture_agent(monkeypatch):
    agents: list = []

    async def run(self):
        agents.append(self._agent)
    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", run)
    return agents


def _spy_record_floor(monkeypatch):
    calls: list[tuple] = []
    real = lifecycle.record_floor

    def record(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(lifecycle, "record_floor", record)
    return calls


async def test_the_full_floor_is_written_once_after_the_registry(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=65_536, probe_window=None)
    _stub_floor(monkeypatch, 256)
    agents = _capture_agent(monkeypatch)
    calls = _spy_record_floor(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    ((config_dir, root, roles),) = calls
    assert root == agents[0].floor() and root > 0
    assert set(roles) == {"explore", "web-researcher", "search-verifier"} and all(roles.values())
    assert lifecycle.cached_floor(config_dir) == max(1_024, root)


async def test_a_stale_cached_floor_does_not_refuse_an_attach_and_is_rewritten(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=65_536, probe_window=None)
    _stub_floor(monkeypatch, 256)
    lifecycle.record_floor(tmp_path, 100_000, {})
    ran = _spy_repl(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    assert ran == [1]
    assert lifecycle.cached_floor(tmp_path) < 65_536


async def test_a_window_between_the_pre_launch_and_the_full_floor_exits_1_before_any_turn(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=16_384, probe_window=None)
    _stub_floor(monkeypatch, 256)
    monkeypatch.setattr("localharness.agent.loop.AgentLoop.floor", lambda self: 20_000)
    ran = _spy_repl(monkeypatch)
    errs = _capture_err_console(monkeypatch)

    with pytest.raises(typer.Exit) as exit_:
        await _start_async(None, False, False, str(tmp_path))

    assert exit_.value.exit_code == 1
    assert errs == [UNIT_LINE.replace("4,096", "16,384").replace("15,360", "20,000")]
    assert ran == []
    assert lifecycle.cached_floor(tmp_path) == 20_000


async def test_a_window_at_the_full_floor_starts_even_when_the_cache_cannot_be_written(tmp_path, monkeypatch):
    _stub_nonvllm_start(tmp_path, monkeypatch, served=20_000, probe_window=None)
    _stub_floor(monkeypatch, 256)
    monkeypatch.setattr("localharness.agent.loop.AgentLoop.floor", lambda self: 20_000)

    def refuse(path, text):
        raise OSError("read-only")
    monkeypatch.setattr(lifecycle, "atomic_write_text", refuse)
    ran = _spy_repl(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    assert ran == [1]
