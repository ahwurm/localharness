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
from localharness.config.models import EndpointRef, HarnessConfig, ManagedServerConfig, OrgConfig, ProviderConfig
from localharness.core.bus import EventBus
from localharness.core.events import ModelSwitched
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


def _mk(tmp_path, managed, monkeypatch, *, bus=None, session_id=None, peers=(), live=None):
    stub_liveness(monkeypatch, False)
    harness = HarnessConfig(
        provider=ProviderConfig(provider_type="vllm", base_url="http://x/v1",
                                default_model="qwen3.6-35b-a3b", available_models=["qwen3.6-35b-a3b"]),
        org=OrgConfig(default_model="qwen3.6-35b-a3b"), server=managed, extra_endpoints=list(peers))
    llm = NS(config=NS(base_url="http://x/v1", model="qwen3.6-35b-a3b"))

    def rebind(base_url, *, api_key=None, extra_headers=None, provider_type=None):
        llm.config.base_url = base_url

    async def caps():
        return NS(tool_call_mode="native")

    llm.detect_capabilities = caps
    llm.rebind_endpoint = rebind
    chan = _Chan()
    agent = NS(_llm=llm, _config=NS(name="root"), current_session_id=session_id) if session_id else NS(_llm=llm)
    r = OrchestratorREPL(
        orchestrator=None, agent_loop=agent, channel=chan, bus=bus,
        config_dir=tmp_path, harness_config=harness,
    )
    served = live or {}

    async def fake_live(base_url):
        return (served.get(base_url, ["qwen3.6-35b-a3b"]), True)

    r._live_models = fake_live

    async def fake_refresh(model, **kw):
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
    assert r._model_cache == [("qwen3.6-35b-a3b", "serving now")]   # the menu offers what the list numbers
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


async def test_a_hot_model_through_the_switch_publishes_one_hot_event(tmp_path, monkeypatch):
    bus = EventBus()
    r, _chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1",
                            live={"http://x/v1": ["qwen3.6-35b-a3b", "qwen3.6-27b"]})

    await r._handle_slash("/model qwen3.6-27b")
    [event] = bus.history(event_types=[ModelSwitched])

    assert (event.kind, event.endpoint, event.model, event.session_id) == ("hot", "primary", "qwen3.6-27b", "s1")
    assert llm.config.model == "qwen3.6-27b"


async def test_an_already_active_pick_through_the_switch_publishes_nothing(tmp_path, monkeypatch):
    bus = EventBus()
    r, _chan, _llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1",
                             live={"http://x/v1": ["qwen3.6-35b-a3b", "qwen3.6-27b"]})

    await r._handle_slash("/model qwen3.6-35b-a3b")

    assert bus.history(event_types=[ModelSwitched]) == []


async def test_a_cross_model_through_the_switch_publishes_one_cross_event(tmp_path, monkeypatch):
    bus = EventBus()
    peer = EndpointRef(name="side", base_url="http://side/v1", provider_type="ollama", api_key="none")
    r, _chan, llm, _p = _mk(tmp_path, _managed(), monkeypatch, bus=bus, session_id="s1", peers=[peer],
                            live={"http://x/v1": ["qwen3.6-35b-a3b"], "http://side/v1": ["gpt-oss:20b"]})

    async def persist_active(endpoint, model):
        return None

    r._persist_active_endpoint = persist_active

    await r._handle_slash("/model gpt-oss:20b")
    [event] = bus.history(event_types=[ModelSwitched])

    assert (event.kind, event.endpoint, event.model) == ("cross", "side", "gpt-oss:20b")
    assert llm.config.base_url == "http://side/v1"
