"""The critics: loading, the one parser, the data fence and the reply panel's sensing."""
from __future__ import annotations

from localharness.agent.subagent import DelegationResult
from localharness.config.models import ChildBudget
from localharness.orchestrator.task_flow import panel
from localharness.orchestrator.task_flow.panel import FENCE_CLOSE, FENCE_LABEL, fence, run_critic


def scripted(answers: dict[str, DelegationResult]):
    calls: list[tuple] = []

    async def run_child(config, task, **kw):
        calls.append((config, task, kw))
        return answers[config.name]

    run_child.calls = calls
    return run_child


def answered(raw: str, reason: str | None = None) -> DelegationResult:
    return DelegationResult(text="SUBAGENT RUN COMPLETE\n" + raw, raw=raw, terminated_reason=reason)


BLOCKER = "[BLOCKER] the reply cites a real line — README.md:41 does not say that — example fix: quote line 41"


async def test_a_critic_runs_from_its_yaml_through_the_child_to_a_fenced_finding():
    child = scripted({"answer": answered(BLOCKER)})
    parsed = await run_critic("answer", "brief", child, ChildBudget(actions=3, minutes=1))
    (finding,) = parsed.findings
    assert (finding.tag, finding.rule) == ("BLOCKER", "the reply cites a real line")
    assert finding.evidence == "README.md:41 does not say that"
    assert "example fix" not in finding.evidence and "quote line 41" not in finding.evidence
    config, task, kw = child.calls[0]
    assert config.name == "answer" and config.tools.add == list(panel.CRITIC_TOOLS)
    assert task == "brief"
    assert kw["budget"] == ("task_flow.critic", ChildBudget(actions=3, minutes=1))
    assert kw["grant_handles"] is None
    block = fence(parsed.findings)
    assert FENCE_LABEL in block and "README.md:41 does not say that" in block
    assert "quote line 41" not in block
    assert block.endswith(FENCE_CLOSE)


async def test_a_critic_run_reads_the_raw_answer_not_the_wrapped_text():
    child = scripted({"answer": DelegationResult(text="SUBAGENT RUN COMPLETE\n" + BLOCKER, raw="NONE")})
    parsed = await run_critic("answer", "brief", child, ChildBudget(actions=3, minutes=1), handles=("h1",))
    assert parsed.clean and parsed.readable and parsed.findings == ()
    assert child.calls[0][2]["grant_handles"] == ["h1"]
