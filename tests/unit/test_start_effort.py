"""Where `start` hands the level on (03-04: D-10, D-11, D-19, D-22, D-34).

`start` resolves the session agent's level once (`resolve_effort(None, agent file, harness, base_url)`),
builds the loop with it and the machine's budget ratios, gives the subagent runner a CALLABLE for the
live level plus the same ratios and the same template margin, and names a key the server refused in an
earlier session once, in the startup summary.

Nothing here touches a real server: the probe and the REPL are faked through the module seams the
other `start` suites use, and the ledger lives in `tmp_path`.
"""
from __future__ import annotations

import pytest
import yaml

from localharness.cli import start_cmd
from localharness.cli.agent_cmd import _build_agent_yaml
from localharness.cli.start_cmd import EFFORT_CUT_OFF_LINE, EFFORT_UNENFORCED_LINE, _start_async
from localharness.provider.speed_stats import default_speed_stats_path, record_rejected_effort_keys
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

URL = "http://127.0.0.1:9/v1"  # the discard port: a live :8080 would rename the model


class World:
    """One start's surroundings: the config, the loop the REPL was handed, the runner's keywords."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.agents: list = []
        self.runner_kwargs: list[dict] = []
        self.ledger = default_speed_stats_path(tmp_path)

        async def keep_agent(repl):
            self.agents.append(repl._agent)

        _stub_start_boundaries(tmp_path, monkeypatch, repl_run=keep_agent)
        self.out = _capture_start_console(monkeypatch)

        import localharness.agent.subagent as subagent
        real = subagent.make_explore_agent_runner

        def spy(**kw):
            self.runner_kwargs.append(kw)
            return real(**kw)

        monkeypatch.setattr(subagent, "make_explore_agent_runner", spy)

    def config(self, *, provider_type="ollama", model="qwen", server: dict | None = None,
               extra: dict | None = None) -> None:
        data: dict = {"version": "1",
                      "provider": {"provider_type": provider_type, "base_url": URL, "api_key": "none",
                                   "default_model": model}}
        if server is not None:
            data["server"] = {"runtime": "llamacpp", "binary": "/x/llama-server", "port": 8080, **server}
        data.update(extra or {})
        (self.tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    def agent_file(self, **over) -> None:
        agents_dir = self.tmp_path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        data = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
        data.update(over)
        (agents_dir / "orchestrator.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    async def start(self) -> list[str]:
        before = len(self.out)
        await _start_async(None, False, False, str(self.tmp_path))
        return self.out[before:]

    @property
    def loop(self):
        return self.agents[-1]


@pytest.fixture
def world(tmp_path, monkeypatch):
    # a start with no managed server would try to reconcile the model name against the (dead) URL
    w = World(tmp_path, monkeypatch)
    w.config()
    return w


def _summary(lines: list[str]) -> str:
    (line,) = [x for x in lines if " -- " in x]
    return line


# ------------------------------------------------------------------- D-10: the level into the loop


async def test_no_server_and_no_org_default_starts_at_high_default(world):
    await world.start()
    assert (world.loop.effective_effort, world.loop.effort_rung) == ("high", "default")


async def test_an_org_default_is_the_org_rung(world):
    world.config(extra={"org": {"default_effort": "low"}})
    await world.start()
    assert (world.loop.effective_effort, world.loop.effort_rung) == ("low", "org")


async def test_the_servers_effort_with_an_inheriting_agent_file_is_the_endpoint_rung(world):
    world.config(server={"model": "/x/m.gguf", "effort": "max"})
    world.agent_file(effort="inherit")
    await world.start()
    assert (world.loop.effective_effort, world.loop.effort_rung) == ("max", "endpoint")


async def test_the_agent_files_own_level_is_the_agent_rung(world):
    world.config(server={"model": "/x/m.gguf", "effort": "max"})
    world.agent_file(effort="low")
    await world.start()
    assert (world.loop.effective_effort, world.loop.effort_rung) == ("low", "agent")
    assert world.loop._config.effort == "low", "the file's rung is never rewritten"


async def test_the_loop_gets_the_machines_budget_ratios(world):
    world.config(extra={"effort_budget": {"low_ratio": 0.2, "high_ratio": 0.6}})
    await world.start()
    assert world.loop._effort_ratios == (0.2, 0.6)


# ------------------------------------------------------------------- D-11: the runner


async def test_the_runner_reads_the_loops_live_level_at_call_time(world):
    world.config(extra={"effort_budget": {"low_ratio": 0.2, "high_ratio": 0.6}})
    await world.start()
    (kw,) = world.runner_kwargs
    assert kw["effort_fn"]() == ("high", "default")
    world.loop.set_effort("max", "session")
    assert kw["effort_fn"]() == ("max", "session"), "effort_fn captured the start-time level"
    assert kw["effort_ratios"] == world.loop._effort_ratios == (0.2, 0.6)


# ------------------------------------------------------------------- D-34: the template margin


async def test_the_runner_gets_the_template_margin_the_session_loop_is_built_with(world):
    world.config(extra={"autosense": {"template_margin_ratio": 0.03}})
    await world.start()
    (kw,) = world.runner_kwargs
    assert kw["template_margin_ratio"] == 0.03 == world.loop._template_margin_ratio


async def test_the_default_template_margin_matches_too(world):
    await world.start()
    (kw,) = world.runner_kwargs
    assert kw["template_margin_ratio"] == world.loop._template_margin_ratio


# ------------------------------------------------------------------- D-19 / D-22: the summary lines


async def test_a_recorded_rejected_key_is_named_once_per_key(world):
    record_rejected_effort_keys(world.ledger, "ollama", "qwen", ["reasoning_budget_tokens"])
    summary = _summary(await world.start())
    assert EFFORT_UNENFORCED_LINE.format(key="reasoning_budget_tokens") in summary
    assert "effort unenforced on this server: reasoning_budget_tokens rejected" in summary


async def test_two_recorded_keys_are_two_lines(world):
    record_rejected_effort_keys(
        world.ledger, "ollama", "qwen", ["reasoning_budget_tokens", "chat_template_kwargs"])
    summary = _summary(await world.start())
    assert "effort unenforced on this server: reasoning_budget_tokens rejected" in summary
    assert "effort unenforced on this server: chat_template_kwargs rejected" in summary
    assert summary.count("effort unenforced on this server") == 2


async def test_no_recorded_key_adds_no_line(world):
    summary = _summary(await world.start())
    assert "effort unenforced" not in summary
    assert EFFORT_CUT_OFF_LINE not in summary


async def test_a_recorded_empty_list_adds_no_line(world):
    record_rejected_effort_keys(world.ledger, "ollama", "qwen", [])
    summary = _summary(await world.start())
    assert "effort unenforced" not in summary
    assert EFFORT_CUT_OFF_LINE not in summary


async def test_the_other_models_rejections_do_not_leak(world):
    record_rejected_effort_keys(world.ledger, "ollama", "some-other-model", ["reasoning_budget_tokens"])
    assert "effort unenforced" not in _summary(await world.start())


async def test_thinking_off_rejected_on_a_runtime_without_a_budget_key_says_the_cut_is_off(world):
    record_rejected_effort_keys(world.ledger, "vllm", "qwen", ["chat_template_kwargs"])
    world.config(provider_type="vllm")
    summary = _summary(await world.start())
    assert EFFORT_CUT_OFF_LINE in summary
    assert "effort unenforced on this server: chat_template_kwargs rejected" in summary


async def test_thinking_off_rejected_on_llamacpp_does_not_say_the_cut_is_off(world):
    """llama.cpp enforces the budget server-side, so the loop-owned cut is not what ended."""
    record_rejected_effort_keys(world.ledger, "llamacpp", "qwen", ["chat_template_kwargs"])
    world.config(provider_type="llamacpp")
    summary = _summary(await world.start())
    assert EFFORT_CUT_OFF_LINE not in summary
    assert "effort unenforced on this server: chat_template_kwargs rejected" in summary


async def test_an_unreadable_ledger_adds_no_line_and_raises_nothing(world):
    world.ledger.parent.mkdir(parents=True, exist_ok=True)
    world.ledger.write_text("{not json at all", encoding="utf-8")
    summary = _summary(await world.start())
    assert "effort unenforced" not in summary


async def test_a_ledger_that_cannot_be_read_does_not_stop_start(world, monkeypatch):
    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr("localharness.provider.speed_stats.last_rejected_effort_keys", boom)
    summary = _summary(await world.start())
    assert "effort unenforced" not in summary
    assert start_cmd.EFFORT_CUT_OFF_LINE not in summary
