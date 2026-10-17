"""In-REPL full model swap: /model <registry-name> restarts the managed server with the
named local checkpoint — stop → start(registry-resolved command) → readiness poll with a
live progress line → rebind + persist. The registry names appear in the listing and the
picker cache; a failed swap reports and leaves the session model untouched.

Offline: managed_server lifecycle functions are monkeypatched at the module the REPL
imports; serve_command stays REAL so the command the swap builds is asserted end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import HarnessConfig, ManagedServerConfig, OrgConfig, ProviderConfig
from tests.unit.launch_support import offer_primary_as, stub_liveness


class _Chan:
    def __init__(self):
        self.msgs: list[str] = []
        self.activity: list[str | None] = []
        self.menu_opened = 0

    async def send_message(self, content, **kw):
        self.msgs.append(content)

    def box_activity(self, text):
        self.activity.append(text)

    def box_open_model_menu(self):
        self.menu_opened += 1


def _managed() -> ManagedServerConfig:
    return ManagedServerConfig(
        launch="docker", docker_image="img:tag", model="qwen3.6-35b-a3b", port=8000,
        extra_args=["--kv-cache-dtype", "fp8"],
        local_models=[
            {"name": "qwen3.6-35b-a3b", "path": "/x/Qwen35",
             "extra_args": ["--moe-backend", "marlin"],
             "quant": "nvfp4 (modelopt)", "tps": 30.0},
            {"name": "qwen3.6-27b", "path": "/x/Qwen27",
             "quant": "nvfp4 (compressed-tensors)", "tps": 9.5},
        ],
    )


def _mk(tmp_path, managed, monkeypatch):
    stub_liveness(monkeypatch, False)
    harness = HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url="http://x/v1",
                                default_model="qwen3.6-35b-a3b", available_models=["qwen3.6-35b-a3b"]),
        org=OrgConfig(default_model="qwen3.6-35b-a3b"), server=managed)
    llm = NS(config=NS(base_url="http://x/v1", model="qwen3.6-35b-a3b"))

    async def caps():
        return NS(tool_call_mode="native")

    llm.detect_capabilities = caps
    chan = _Chan()
    r = OrchestratorREPL(
        orchestrator=None, agent_loop=NS(_llm=llm), channel=chan, bus=None,
        config_dir=tmp_path, harness_config=harness,
    )

    async def fake_live(base_url):
        return (["qwen3.6-35b-a3b"], True)

    r._live_models = fake_live

    async def fake_refresh(model):
        return ""

    r._refresh_token_counter = fake_refresh
    persisted: list[str] = []

    async def fake_persist(model):
        persisted.append(model)

    r._persist_default_model = fake_persist
    return r, chan, llm, persisted


async def test_listing_names_the_serving_model_and_opens_the_picker(tmp_path, monkeypatch):
    r, chan, _llm, _p = _mk(tmp_path, _managed(), monkeypatch)
    await r._handle_model_cmd("")
    out = "\n".join(chan.msgs)
    assert "1. qwen3.6-35b-a3b  (serving)  ● [active]" in out
    assert "downloaded" not in out and "restarts the managed server" not in out
    assert chan.menu_opened == 1                       # bare /model auto-opens the picker menu


async def test_swap_to_registry_model_full_choreography(tmp_path, monkeypatch):
    from localharness.provider import server as ms

    r, chan, llm, persisted = _mk(tmp_path, _managed(), monkeypatch)
    offer_primary_as(monkeypatch, "qwen3.6-27b")
    calls: dict = {}

    def fake_stop(cfg, launch="binary", **kw):
        calls["stop"] = (cfg, launch)

    def fake_start(cfg, cmd, **kw):
        calls["cmd"] = cmd
        return 1

    async def fake_ready(base_url, config_dir=None, on_poll=None, **kw):
        if on_poll is not None:
            on_poll(1.0)
            on_poll(2.0)
        return ["qwen3.6-27b"]

    monkeypatch.setattr(ms, "stop_server", fake_stop)
    monkeypatch.setattr(ms, "start_server", fake_start)
    monkeypatch.setattr(ms, "wait_ready", fake_ready)

    await r._handle_model_cmd("qwen3.6-27b")

    assert calls["stop"][1] == "docker"
    joined = " ".join(calls["cmd"])                       # REAL serve_command output
    assert "-v /x/Qwen27:/models/serving:ro" in joined
    assert "--model /models/serving" in joined
    assert "--served-model-name qwen3.6-27b" in joined
    assert "--moe-backend" not in joined
    assert llm.config.model == "qwen3.6-27b"
    assert persisted == ["qwen3.6-27b"]
    notes = [a for a in chan.activity if a]
    assert notes and all("qwen3.6-27b" in n for n in notes)   # progress line while loading
    assert chan.activity[-1] is None                          # cleared when done


async def test_swap_failure_reports_and_keeps_current_model(tmp_path, monkeypatch):
    from localharness.provider import server as ms

    r, chan, llm, persisted = _mk(tmp_path, _managed(), monkeypatch)
    offer_primary_as(monkeypatch, "qwen3.6-27b")

    monkeypatch.setattr(ms, "stop_server", lambda cfg, launch="binary", **kw: None)
    monkeypatch.setattr(ms, "start_server", lambda cfg, cmd, **kw: 1)

    async def fake_ready(base_url, config_dir=None, on_poll=None, **kw):
        raise TimeoutError("vLLM not ready after 1800s.")

    monkeypatch.setattr(ms, "wait_ready", fake_ready)

    await r._handle_model_cmd("qwen3.6-27b")

    assert any("Model swap failed" in m for m in chan.msgs)
    assert llm.config.model == "qwen3.6-35b-a3b"          # session model untouched
    assert persisted == []
    assert chan.activity and chan.activity[-1] is None    # progress line cleared on failure too
