"""A first `localharness start` with no config goes straight into init's core setup and on into the
session in the same command — no "run localharness init" bounce (52-RULINGS R9, owner ruling
2026-10-03). Without a person to ask (`--no-input`, no terminal) today's welcome hint stands.

Driven through the real CLI on a clean config dir: `CliRunner` -> `start_app` -> `core_setup` ->
`asyncio.run(_start_async)`. REAL: start_app's pre-step, init's core setup (the gate, the quick-path
question, the config write, the receipt, the plugin list) and the session bring-up that reads the
config init just wrote. STUBBED: detection and init's capability probe (fakes answering for the
loopback discard port, so the written config is offline by construction), the session's LLM probe,
tokenizer and REPL read loop (`_stub_start_boundaries`; the REPL is a spy counting the sessions that
opened), and the terminal: init's own `sys` is swapped for a fake tty, because CliRunner replaces the
real stdin while the command runs. A question that must not be asked is booby-trapped.

Plain `def` tests: CliRunner -> start_app -> asyncio.run, and asyncio_mode is "auto", so an
`async def` test would already be running inside a loop.

NOT proven: a real model server, a real terminal's rendering.
"""
from __future__ import annotations

import asyncio
import gc
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

import localharness.cli.init_cmd as init_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider.detector import DetectorResult
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_init_cmd import _make_capability_result
from tests.unit.test_start_cmd import _stub_start_boundaries

LEAD = "LocalHarness is not set up yet. Setting up the model server first:"
HINT = "To configure, run: localharness init"
DISCARD = "http://127.0.0.1:9/v1"  # the loopback discard port: nothing sent there is answered
MODEL = "first-start-model"


class _Asked:
    """A question that must not be asked: asking it fails the run."""

    @staticmethod
    def ask(*_a, **_k):
        raise AssertionError("asked")


def _boom(*_a, **_k):
    raise AssertionError("init code ran on a configured start")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No real HTTP from init: the served-window, runtime and Ollama hot-model probes are stubbed."""
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")
    monkeypatch.setattr(init_cmd, "_get_ollama_hot_model", lambda *_: None)


def _machine(g: Path, monkeypatch, *, configured: bool = False) -> list:
    """The session's boundaries stubbed, the REPL a spy that collects garbage inside the session's loop
    (a real session's GC gets there sooner or later). `_stub_start_boundaries` writes a config: a
    first start removes it, a configured one keeps it, pointed at the discard port. Returns the list
    every session that opened appends to."""
    sessions: list = []

    async def repl_run(self):
        gc.collect()
        sessions.append(self)

    _stub_start_boundaries(g, monkeypatch, repl_run=repl_run)
    if configured:
        _offline_provider(g)
    else:
        (g / "config.yaml").unlink()
    monkeypatch.chdir(g)
    return sessions


def _server(monkeypatch, *, found: bool = True) -> tuple[AsyncMock, MagicMock]:
    """Detection finds one model at the discard port, or nothing (the detector's own not-found
    shape); init's capability probe answers native. Returns (detect_provider, LLMClient)."""
    detect = AsyncMock(return_value=DetectorResult(
        found=found, provider_type="vllm" if found else "unknown", base_url=DISCARD if found else "",
        models=[MODEL] if found else [], suggested_model=MODEL if found else "", probe_duration_ms=1.0))
    monkeypatch.setattr(init_cmd, "detect_provider", detect)
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_make_capability_result())
    client_cls = MagicMock(return_value=client)
    monkeypatch.setattr(init_cmd, "LLMClient", client_cls)
    return detect, client_cls


def _terminal(monkeypatch, on: bool = True) -> None:
    fake_sys = MagicMock()
    fake_sys.stdin.isatty.return_value = on
    monkeypatch.setattr(init_cmd, "sys", fake_sys)


def _questions(monkeypatch, *, confirms=None, prompts=None) -> MagicMock | None:
    """init's Confirm answers `confirms` in order and Prompt answers `prompts` (one question more
    than scripted fails the run); None = must not be asked. IntPrompt is never asked. Returns the
    Confirm mock (None when booby-trapped)."""
    def scripted(answers):
        if answers is None:
            return _Asked
        mock = MagicMock()
        mock.ask.side_effect = list(answers)
        return mock
    confirm = scripted(confirms)
    monkeypatch.setattr(init_cmd, "Confirm", confirm)
    monkeypatch.setattr(init_cmd, "Prompt", scripted(prompts))
    monkeypatch.setattr(init_cmd, "IntPrompt", _Asked)
    return None if confirms is None else confirm


def _start(g: Path, *flags: str):
    return CliRunner().invoke(app, ["start", "--config-dir", str(g), *flags])


def _flat(result) -> str:
    """Rich wraps at 80 columns — normalize whitespace so a phrase survives the wrap."""
    return " ".join((result.output or "").split())


def _exited(result, code: int = 0) -> None:
    """A clean exit with `code` — not a booby-trapped question or a traceback."""
    assert result.exit_code == code, result.output
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(result.exception)


def test_a_first_start_on_a_terminal_sets_up_the_core_and_goes_on_into_the_session(tmp_path, monkeypatch):
    sessions = _machine(tmp_path, monkeypatch)
    detect, _ = _server(monkeypatch)
    _terminal(monkeypatch)
    confirm = _questions(monkeypatch, confirms=[True])  # "Keep the usual settings?" — the one question

    result = _start(tmp_path)

    _exited(result)
    flat = _flat(result)
    marks = [flat.find(s) for s in (LEAD, "LocalHarness configured at", init_cmd.PLUGINS_HEADER, "startup)")]
    assert -1 not in marks and marks == sorted(marks), flat
    assert "Run 'localharness start' to begin." not in flat, flat
    assert confirm.ask.call_count == 1
    detect.assert_awaited_once()
    assert len(sessions) == 1
    assert ConfigLoader(config_dir=tmp_path).load_harness().provider.base_url == DISCARD


def _hint_only(result, g: Path, detect: AsyncMock, sessions: list) -> None:
    _exited(result)
    flat = _flat(result)
    assert HINT in flat and LEAD not in flat, flat
    detect.assert_not_called()
    assert not (g / "config.yaml").exists()
    assert sessions == []


@pytest.mark.parametrize("flag", ["--no-input", "--list-models"])
def test_a_first_start_with_no_input_keeps_the_welcome_hint_and_writes_nothing(tmp_path, monkeypatch, flag):
    """A terminal IS attached: --no-input (nobody to ask) alone keeps the setup out, and so does
    --list-models (a listing of the configured server, not a session)."""
    sessions = _machine(tmp_path, monkeypatch)
    detect, _ = _server(monkeypatch)
    _terminal(monkeypatch)
    _questions(monkeypatch)

    _hint_only(_start(tmp_path, flag), tmp_path, detect, sessions)


def test_a_first_start_without_a_terminal_keeps_the_welcome_hint_and_writes_nothing(tmp_path, monkeypatch):
    sessions = _machine(tmp_path, monkeypatch)
    detect, _ = _server(monkeypatch)
    _terminal(monkeypatch, on=False)
    _questions(monkeypatch)

    _hint_only(_start(tmp_path), tmp_path, detect, sessions)


@pytest.mark.parametrize("model", ["", "my-model"], ids=["nothing-saved", "saved-unchecked"])
def test_a_first_start_that_finds_no_server_opens_no_session(tmp_path, monkeypatch, model):
    """No server answered, the guided vLLM setup declined, skip for now: the setup's own next step
    is the last word. Saved unchecked leaves a config, but no server answered, so no session."""
    sessions = _machine(tmp_path, monkeypatch)
    _, client_cls = _server(monkeypatch, found=False)
    _terminal(monkeypatch)
    _questions(monkeypatch, confirms=[False], prompts=["http://localhost:8081/v1", model])

    result = _start(tmp_path)

    _exited(result)
    flat = _flat(result)
    assert LEAD in flat and "startup)" not in flat and HINT not in flat, flat  # no bounce after it
    if model:
        assert "not checked yet" in flat and (tmp_path / "config.yaml").exists(), flat
    else:
        assert "Nothing saved." in flat and not (tmp_path / "config.yaml").exists(), flat
    client_cls.assert_not_called()
    assert sessions == []


def test_a_configured_start_runs_no_setup(tmp_path, monkeypatch):
    sessions = _machine(tmp_path, monkeypatch, configured=True)
    detect = AsyncMock(side_effect=AssertionError("detection ran on a configured start"))
    monkeypatch.setattr(init_cmd, "detect_provider", detect)
    monkeypatch.setattr(init_cmd, "core_setup", _boom)
    monkeypatch.setattr(init_cmd, "is_interactive", _boom)
    _terminal(monkeypatch)  # a terminal: with the config missing, the setup would run here
    _questions(monkeypatch)

    result = _start(tmp_path)

    _exited(result)
    flat = _flat(result)
    assert LEAD not in flat and "startup)" in flat, flat
    detect.assert_not_called()
    assert len(sessions) == 1


@pytest.mark.parametrize(("channel", "refusal"), [
    ("discrod", "unknown channel 'discrod'"), ("web", "the web channel is served by its own command")])
def test_a_first_start_refuses_a_channel_it_cannot_build_before_any_setup(tmp_path, monkeypatch, channel, refusal):
    """A typo in --channel is a typo whether or not the box is set up: the tier-one refusal comes
    before the lead line, detection or any question, not after a setup walk that wrote a config."""
    sessions = _machine(tmp_path, monkeypatch)
    detect, _ = _server(monkeypatch)
    _terminal(monkeypatch)
    _questions(monkeypatch)

    result = _start(tmp_path, "--channel", channel)

    assert result.exit_code == 2, result.output
    flat = _flat(result)
    assert refusal in flat and LEAD not in flat, flat
    detect.assert_not_called()
    assert not (tmp_path / "config.yaml").exists()
    assert sessions == []


def test_a_first_start_collects_the_setup_s_open_clients_before_the_session_s_loop(tmp_path, monkeypatch):
    """init probes the server in event loops of its own and leaves its client open. The openai SDK
    finalizes such a client by scheduling its close on whatever loop is running when the GC reaches
    it; inside the session's loop, closing a transport of the setup's closed loop raised "Event loop
    is closed" on a real terminal. The setup's garbage is collected before the session's loop runs."""
    finalized_in: list = []

    class _OpenProbeClient:
        def __init__(self, *_a, **_k):
            self._cycle = self  # like the SDK's client graph: only the cyclic GC frees it

        async def __aenter__(self):  # the probe closes its client in its own loop (deferred item 3);
            return self              # this one stands for anything else the setup leaves behind,

        async def __aexit__(self, *_exc):  # still in a reference cycle until the GC reaches it
            return None

        async def detect_capabilities(self):
            gc.collect()  # it lived through the probe's collections, so it sits in the oldest generation
            return _make_capability_result()

        def __del__(self):
            try:
                finalized_in.append(asyncio.get_running_loop())
            except RuntimeError:
                finalized_in.append(None)

    sessions = _machine(tmp_path, monkeypatch)
    _server(monkeypatch)
    monkeypatch.setattr(init_cmd, "LLMClient", _OpenProbeClient)
    _terminal(monkeypatch)
    _questions(monkeypatch, confirms=[True])

    _exited(_start(tmp_path))
    assert len(sessions) == 1
    assert finalized_in == [None]  # finalized while no loop ran, not inside the session's
