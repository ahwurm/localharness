"""The three memory-seeded bench scenarios: per-call system prompt + offered tool names, pinned.

Captured from 7757e09 (the private `_seed_memory_store` path). After the bench builds memory through
the plugin the SAME goldens must hold unedited; a diff is a finding to explain, never a golden to
regenerate. 48-07 runs this test at the Arm-A commit and at HEAD as the bench-flat pre-check.

Drive: the REAL `accumulate_runs -> execute_one_run -> _build_agent_loop -> AgentLoop` with only the
MODEL faked (`llm_client_factory`). The fake records each call's system message and sorted tool
names, then answers from the system prompt alone: the seeded tokens it finds there, or `NOT FOUND`
— so the run's `success` proves recall reached the prompt through whichever memory path the tree
has. Train-slice scenario files only; never touch bench/scenarios/holdout/.

Normalisation (`_norm`): `str(tmp_path)` -> `<TMP>` only. The clock is frozen at its source
(`localharness.agent.loop.datetime`) and CWD is `tmp_path`. Fact text, headings and section order
are never normalised.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.bench.runner import accumulate_runs
from localharness.bench.schema import load_scenario
from tests.conftest import FakeCompletionUsage, _NativeMsg
from tests.integration.test_memory_compat_baseline_e2e import REGEN, _FrozenNow, _golden

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "memory_surfaces"
SCENARIOS = {
    "memory_recall": ("10_memory_recall.yaml", ["STARFRUIT_42"]),
    "stateful_behavior_two_facts": ("23_stateful_behavior_two_facts.yaml",
                                    ["STARFRUIT_42", "MOONFRUIT_88"]),
    "stateful_behavior_overwrite_recall": ("24_stateful_behavior_overwrite_recall.yaml", ["amber"]),
}


class _RecallFake:
    """Native-mode fake shaped like FaithfulFakeLLM (tool_plan []) that records every call."""

    def __init__(self, tokens: list[str]):
        self.tokens, self.calls = tokens, []

        class _Cfg:
            tool_call_mode = "native"
            context_window = 128_000
        self.config = _Cfg()

    async def complete(self, messages=None, tools=None, stream=False):
        system = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
        names = sorted(t["function"]["name"] if isinstance(t, dict) else t.name for t in tools or ())
        self.calls.append({"system": system, "tools": names})
        found = [t for t in self.tokens if t in system]
        answer = ", ".join(found) if len(found) == len(self.tokens) else "NOT FOUND"
        return _NativeMsg(content=answer, tool_calls=[]), \
            FakeCompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    async def stream_complete(self, messages=None, tools=None, on_token=None):
        return await self.complete(messages, tools)


def _norm(text: str, tmp_path: Path) -> str:
    return text.replace(str(tmp_path), "<TMP>")


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_bench_memory_prompt_and_tools_match_the_golden(name, tmp_path, monkeypatch):
    monkeypatch.setattr("localharness.agent.loop.datetime", _FrozenNow)
    monkeypatch.setenv("LOCALHARNESS_CATEGORIES_PATH", str(REPO / "bench" / "categories.yaml"))
    monkeypatch.chdir(tmp_path)
    file, tokens = SCENARIOS[name]
    scen = load_scenario(REPO / "bench" / "scenarios" / "train" / file)
    assert scen.name == name
    fake = _RecallFake(tokens)

    samples, _stop = await accumulate_runs(scen, "fake", tmp_path / "results",
                                           llm_client_factory=lambda _s: fake,
                                           min_runs_override=1, max_runs_override=1)

    assert fake.calls, "no request reached the model boundary"
    text = _norm("\n".join(f"=== call {i} tools={c['tools']}\n{c['system']}"
                           for i, c in enumerate(fake.calls)), tmp_path)
    assert "## Agent Memory" in text and all(t in text for t in tokens), text
    assert samples[0].success is True, "the seeded memory did not reach the prompt"
    assert text == _golden(FIXTURES / f"bench_{name}.txt", text), f"{name}: prompt/tools drifted"
