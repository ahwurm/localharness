"""Cross-feature seam 5 — the #19 creation flow deploys an agent that actually loads.

The seam no unit test crosses: test_repl_creation_flow asserts the written YAML's fields on
disk, but never re-loads it through the real ConfigLoader.load_agent() inheritance/validation
path. This closes the "deploy produced a green checkmark, but does the harness accept it?" gap
end-to-end — driving the REAL creation flow, then loading the deployed agent the way `start`
would (name rules + model-inherit resolution). The negative half proves deploy never writes an
UNLOADABLE agent: an invalid generated name fails loudly with no file on disk.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from localharness.cli.repl import OrchestratorREPL
from localharness.config.loader import ConfigError, ConfigLoader
from localharness.orchestrator.cards import AgentCardRegistry
from localharness.orchestrator.router import AgentCreationFlow

ROLE = "monitor hacker news and summarize the top stories each morning"


class ScriptedChannel:
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


def _drive_creation(home, mock_llm_client, generated_yaml):
    channel = ScriptedChannel(["create an agent", f"it should {ROLE}", "yes"])
    llm = mock_llm_client([mock_llm_client.Response(content=generated_yaml)])
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
    return channel


def test_created_agent_loads_through_config_loader(components_home, mock_llm_client):
    """Happy path: the deployed yaml loads through load_agent() and validates — valid name,
    model 'inherit' resolved without error — the 'deploy produces a loadable agent' contract."""
    home = components_home
    gen = f"```yaml\nname: hn-monitor\nrole: {ROLE}\nmodel: inherit\n```"
    channel = _drive_creation(home, mock_llm_client, gen)

    deployed = home / "agents" / "hn-monitor.yaml"
    assert deployed.exists()
    assert any("Agent deployed to" in m for m in channel.sent)

    # Load it exactly as `start` would — real inheritance + AgentConfig validation, no cache.
    agent_cfg = ConfigLoader(config_dir=home).load_agent("hn-monitor")
    assert agent_cfg.name == "hn-monitor"
    assert agent_cfg.role == ROLE
    assert agent_cfg.model == "inherit"  # honored + validated; resolves to the provider default at launch
    assert "hn-monitor" in ConfigLoader(config_dir=home).list_agents()


def test_deploy_rejects_unloadable_agent_leaves_no_file(components_home, mock_llm_client):
    """Negative half of the contract: a generated name that violates AgentConfig's name rule
    fails LOUDLY and writes NO file — the harness never ends up with an agent yaml it cannot
    load. (#57 catches this even earlier now: an invalid config is rejected at the pre-confirm
    generation gate — regenerated once, then abandoned — never reaching deploy or the channel
    as a raw Pydantic wall. The contract holds; the loud message is the generation abort.)"""
    home = components_home
    # 'Finance Helper' — uppercase + space — is rejected by AgentConfig.validate_name_format.
    # _drive_creation scripts a single response; the one regeneration retry (#57) re-serves the
    # same invalid name, so creation is abandoned with no file written.
    gen = f"```yaml\nname: Finance Helper\nrole: {ROLE}\n```"
    channel = _drive_creation(home, mock_llm_client, gen)

    assert any("NOT created" in m for m in channel.sent)  # loud, truthful failure
    assert not any("deployed" in m.lower() for m in channel.sent)  # never claimed success
    assert not any("pydantic" in m.lower() for m in channel.sent)  # no raw ValidationError wall
    assert not (home / "agents" / "Finance Helper.yaml").exists()
    # Nothing loadable was written under any name derived from that config.
    assert ConfigLoader(config_dir=home).list_agents() == []
    with pytest.raises(ConfigError):
        ConfigLoader(config_dir=home).load_agent("finance-helper")
