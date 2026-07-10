"""REPL /model — list, hot-swap, managed restart, persistence."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import HarnessConfig, ManagedServerConfig, ProviderConfig


class FakeChannel:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, text, metadata=None):
        self.messages.append(text)


class FakeLLM:
    def __init__(self, model="model-a"):
        self.config = SimpleNamespace(base_url="http://localhost:8081/v1", model=model)

    async def detect_capabilities(self):
        return SimpleNamespace(tool_call_mode="native")


def _repl(tmp_path, harness, live):
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM())
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(),
        agent_loop=agent,
        channel=channel,
        bus=SimpleNamespace(),
        config_dir=tmp_path,
        harness_config=harness,
    )

    async def _live_models(base_url):
        return list(live)

    repl._live_models = _live_models
    return repl, channel, agent


def _harness(server=None):
    return HarnessConfig(
        provider=ProviderConfig(
            provider_type="vllm",
            base_url="http://localhost:8081/v1",
            default_model="model-a",
            available_models=["model-a"],
        ),
        server=server,
    )


@pytest.mark.asyncio
async def test_model_list_shows_serving_and_downloaded(tmp_path, monkeypatch):
    from localharness.provider import server as managed_server

    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["model-a", "cached-b"])
    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    repl, channel, _ = _repl(tmp_path, _harness(srv), live=["model-a"])

    handled = await repl._handle_slash("/model")
    assert handled is True
    out = channel.messages[-1]
    assert "model-a" in out and "(serving)" in out and "[active]" in out
    assert "cached-b" in out and "downloaded" in out


@pytest.mark.asyncio
async def test_model_hotswap_updates_client_and_persists(tmp_path):
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])

    await repl._handle_slash("/model model-b")
    assert agent._llm.config.model == "model-b"
    written = (tmp_path / "config.yaml").read_text()
    assert "model-b" in written
    assert "Switched to model-b" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_swap_by_number(tmp_path):
    repl, _, agent = _repl(tmp_path, _harness(), live=["model-a", "model-b"])
    await repl._handle_slash("/model 2")
    assert agent._llm.config.model == "model-b"


@pytest.mark.asyncio
async def test_model_unknown_rejected(tmp_path):
    repl, channel, agent = _repl(tmp_path, _harness(), live=["model-a"])
    await repl._handle_slash("/model nope")
    assert agent._llm.config.model == "model-a"
    assert "Unknown model" in channel.messages[-1]


@pytest.mark.asyncio
async def test_model_managed_restart_path(tmp_path, monkeypatch):
    from localharness.provider import server as managed_server

    calls: list[str] = []
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    monkeypatch.setattr(managed_server, "stop_server", lambda cfg, launch="binary": calls.append("stop") or True)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd: calls.append("start") or 1234)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        calls.append("wait")
        return ["cached-b"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)

    srv = ManagedServerConfig(binary="/x/vllm", model="model-a")
    harness = _harness(srv)
    repl, channel, agent = _repl(tmp_path, harness, live=["model-a"])

    await repl._handle_slash("/model cached-b")
    assert calls == ["stop", "start", "wait"]
    assert agent._llm.config.model == "cached-b"
    assert harness.server.model == "cached-b"
    written = (tmp_path / "config.yaml").read_text()
    assert "cached-b" in written


@pytest.mark.asyncio
async def test_model_unavailable_without_harness_config(tmp_path):
    repl, channel, _ = _repl(tmp_path, None, live=[])
    await repl._handle_slash("/model")
    assert "unavailable" in channel.messages[-1]
