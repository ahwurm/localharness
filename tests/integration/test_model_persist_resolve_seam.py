"""Cross-feature seam 1 — create an agent, switch its model, persist, and resolve on restart.

The seam no unit test crosses: the #19 REPL agent-creation flow writes a real agent yaml INTO
the same config dir that the #22 /model overlay persistence writes to, and that a FRESH START
re-resolves through the real precedence chain (defaults → config.yaml → overlay → per-agent
yaml). This module drives the REAL creation flow to deploy an agent, does a REAL /model switch
with persistence, and then proves — through a fresh ConfigLoader — that the persisted default
actually reaches the freshly-created (inheriting) agent, while a model-PINNED agent keeps its
pin and trips the documented warning.

Resolution mirrors start_cmd.py:356-358 exactly (the one place the effective model is chosen):
    effective = agent.model if agent.model != "inherit" else harness.provider.default_model
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import yaml

from localharness.cli import model_ops
from localharness.cli.repl import OrchestratorREPL
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import atomic_write_overlay, load_overlay
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow

ROLE = "summarize quarterly finance reports for the team"


# --- Transport-boundary fakes (truthful shapes; see tests/unit/test_repl_*.py) ------------- #


class ScriptedChannel:
    """Terminal stand-in: feeds queued inputs, records sent text (mirrors test_repl_creation_flow)."""

    channel_id = "terminal"

    def __init__(self, inputs):
        self._inputs = list(inputs)
        self.sent: list[str] = []

    async def start(self):
        pass

    async def stop(self):
        pass

    async def read_input(self):
        if not self._inputs:
            raise EOFError()
        return self._inputs.pop(0)

    async def send_message(self, text, metadata=None):
        self.sent.append(text)


class FakeChannel:
    def __init__(self):
        self.messages: list[str] = []

    async def send_message(self, text, metadata=None):
        self.messages.append(text)


class FakeLLM:
    """agent_loop._llm for the /model path: plain-attr config + native capabilities probe."""

    def __init__(self, model="model-a"):
        self.config = SimpleNamespace(base_url="http://localhost:8081/v1", model=model)

    async def detect_capabilities(self):
        return SimpleNamespace(tool_call_mode="native")


def _write_config(home, *, default_model="model-a", available=("model-a", "model-b")):
    data = {
        "version": "1",
        "provider": {
            "provider_type": "vllm",
            "base_url": "http://localhost:8081/v1",
            "default_model": default_model,
            "available_models": list(available),
        },
        "org": {"default_model": default_model, "audit_log_path": str(home / "audit.jsonl")},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _effective_model(agent_cfg, harness):
    """The exact expression start_cmd.py:356-358 uses to pick the model at launch."""
    return agent_cfg.model if agent_cfg.model != "inherit" else harness.provider.default_model


def _drive_creation(home, mock_llm_client):
    """Drive the REAL #19 creation flow end-to-end; returns the deployed agent yaml path."""
    gen_yaml = f"```yaml\nname: finance-helper\nrole: {ROLE}\nmodel: inherit\n```"
    llm = mock_llm_client([mock_llm_client.Response(content=gen_yaml)])
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}", "yes"])
    orch = AgentCreationFlow(AgentCardRegistry())
    agent = MagicMock()
    agent._config.name = "orchestrator"
    agent.current_session_id = "sess-1"
    agent._llm = llm
    agent.run_turn = AsyncMock()
    repl = OrchestratorREPL(
        orchestrator=orch, agent_loop=agent, channel=channel, bus=AsyncMock(), config_dir=home
    )
    asyncio.run(repl.run())
    assert orch.active_workflow is None
    return home / "agents" / "finance-helper.yaml"


def _model_switch(home, harness, target, *, live=("model-a", "model-b")):
    """Drive a REAL /model switch-with-persistence through the REPL; returns the channel."""
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM(model=harness.provider.default_model))
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel,
        bus=SimpleNamespace(), config_dir=home, harness_config=harness,
    )

    async def _live_models(base_url):
        return list(live), True  # (#38) shared (ids, reachable) contract

    repl._live_models = _live_models
    asyncio.run(repl._handle_slash(f"/model {target}"))
    return channel, agent


def test_create_switch_persist_resolve(components_home, mock_llm_client):
    home = components_home
    _write_config(home)
    # Pre-seed unrelated overlay slices: the #22 agent kill-lever section + an org key. The
    # persist path must touch ONLY provider.*/org.* and leave these untouched.
    atomic_write_overlay(
        home / "overrides.yaml",
        {"agent": {"stuck_detector": {"window_size": 9}}, "org": {"log_level": "debug"}},
    )

    # 1. REAL creation flow deploys finance-helper.yaml (model: inherit) into home/agents/.
    created = _drive_creation(home, mock_llm_client)
    assert created.exists()
    assert yaml.safe_load(created.read_text())["name"] == "finance-helper"

    # 2. REAL /model switch WITH persistence, off a harness loaded from the seeded config.yaml.
    harness = ConfigLoader(config_dir=home).load_harness()
    assert harness.provider.default_model == "model-a"
    channel, magent = _model_switch(home, harness, "model-b")
    assert magent._llm.config.model == "model-b"  # live hot-swap happened in-session
    assert any("Switched to model-b" in m for m in channel.messages)

    # Overlay written atomically; provider+org set; available_models UNION-merged, not clobbered.
    overlay = load_overlay(home / "overrides.yaml")
    assert overlay["provider"]["default_model"] == "model-b"
    assert overlay["org"]["default_model"] == "model-b"
    assert set(overlay["provider"]["available_models"]) == {"model-a", "model-b"}
    # Unrelated slices survive the write.
    assert overlay["agent"]["stuck_detector"]["window_size"] == 9
    assert overlay["org"]["log_level"] == "debug"

    # ComponentMutated audit: one per path written, actor=cli, layer=user, after=model-b.
    muts = [
        json.loads(line)
        for line in (home / "audit.jsonl").read_text().splitlines()
        if line.strip()
    ]
    muts = [e for e in muts if e.get("event_type") == "ComponentMutated"]
    assert {e["path"] for e in muts} == {"provider.default_model", "org.default_model"}
    assert all(e["after_value"] == "model-b" for e in muts)
    assert all(e["actor"] == "cli" and e["layer"] == "user" for e in muts)

    # 3. FRESH START: re-load through the real precedence chain; the new default resolves.
    fresh = ConfigLoader(config_dir=home)
    fharness = fresh.load_harness()
    assert fharness.provider.default_model == "model-b"  # defaults → config.yaml → overlay
    agent_cfg = fresh.load_agent("finance-helper")
    assert agent_cfg.model == "inherit"  # the created agent carries no pin
    assert _effective_model(agent_cfg, fharness) == "model-b"  # inherit → the switched default


def test_model_switch_pin_survives_and_warns(components_home):
    """Variant: a per-agent yaml pinning a concrete model keeps that pin across a persisted
    switch (precedence: per-agent yaml wins), the harness default still moves for inheritors,
    and the switch emits the documented pin-trap warning."""
    home = components_home
    _write_config(home)
    agents = home / "agents"
    agents.mkdir(exist_ok=True)
    (agents / "pinned.yaml").write_text(
        "name: pinned-agent\nrole: pinned role\nmodel: some-pinned-model\n", encoding="utf-8"
    )
    (agents / "inheritor.yaml").write_text(
        "name: inheritor\nrole: inheriting role\nmodel: inherit\n", encoding="utf-8"
    )

    harness = ConfigLoader(config_dir=home).load_harness()
    channel, _ = _model_switch(home, harness, "model-b")

    # The switch warns that the pinned agent won't be reached.
    joined = "\n".join(channel.messages)
    assert "won't reach" in joined and "pinned-agent" in joined
    assert model_ops.pinned_agents(home) == [("pinned-agent", "some-pinned-model")]

    fresh = ConfigLoader(config_dir=home)
    fharness = fresh.load_harness()
    pinned_cfg = fresh.load_agent("pinned")
    inheritor_cfg = fresh.load_agent("inheritor")
    # Pin survives; inheritor moves to the persisted default.
    assert pinned_cfg.model == "some-pinned-model"
    assert _effective_model(pinned_cfg, fharness) == "some-pinned-model"
    assert _effective_model(inheritor_cfg, fharness) == "model-b"
