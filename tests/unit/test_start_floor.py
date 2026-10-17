"""Start holds the served window to the harness floor before any turn."""
from __future__ import annotations

import pytest
import typer

from localharness.agent.context import TokenCounter
from localharness.cli import start_serving
from localharness.cli.start_cmd import _start_async
from localharness.config.models import AgentConfig
from tests.unit.test_start_cmd import _capture_err_console, _stub_nonvllm_start

UNIT_LINE = ("test-model on llm.service serves a 4,096-token window and the harness needs at least 15,360 — "
             "raise its -c, or stop the unit and let the harness launch it.")


def _stub_floor(monkeypatch, floor, unit="llm.service"):
    async def fake_floor(agent_config, *, config_dir):
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
