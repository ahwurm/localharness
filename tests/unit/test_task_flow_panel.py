"""The critics: loading, the one parser, the data fence and the reply panel's sensing."""
from __future__ import annotations

import re
from importlib import resources

import pytest
import yaml

from localharness.agent.subagent import DelegationResult
from localharness.config.loader import ConfigLoader
from localharness.config.models import ChildBudget
from localharness.core.caps import Caps
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


NINE = ("jtbd", "answer", "factual", "recency", "big-picture", "parsimony", "security", "spec-compliance", "quality")


def yaml_files():
    return {i.name: i for i in resources.files("localharness").joinpath("assets", "critics").iterdir()}


def test_the_nine_critics_ship_as_package_data():
    assert panel.critic_names() == frozenset(NINE)
    assert sorted(yaml_files()) == sorted(f"{n}.yaml" for n in NINE)


@pytest.mark.parametrize("name", NINE)
def test_yaml_holds_exactly_name_purpose_and_role(name):
    spec = yaml.safe_load(yaml_files()[f"{name}.yaml"].read_text(encoding="utf-8"))
    assert set(spec) == {"name", "purpose", "role"} and spec["name"] == name


@pytest.mark.parametrize("name", NINE)
def test_load_critic_builds_the_config_with_the_fixed_toolset(name):
    config = panel.load_critic(name)
    assert config.name == name
    assert config.tools.add == list(panel.CRITIC_TOOLS)
    assert "\n" not in config.purpose.strip() and 0 < len(config.purpose) <= Caps.LINE
    assert len(config.role.strip().splitlines()) <= 12


@pytest.mark.parametrize("name", NINE)
def test_yaml_role_names_the_finding_format_and_the_exact_none(name):
    role = panel.load_critic(name).role
    assert "[BLOCKER|WARNING|NOTE]" in role and "example fix" in role
    assert "exactly: NONE" in role
    assert "instructions" in role


def test_the_jtbd_role_adds_the_new_job_document_and_the_cited_turn():
    role = panel.load_critic("jtbd").role
    assert "---" in role and "jtbd.md" in role and "UNCHANGED" in role
    assert "[JOB CHANGED]" in role and "turn number" in role


def test_yaml_carries_no_planning_ids():
    for item in yaml_files().values():
        assert not re.search(r"D-\d+|§|spec 31", item.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["../answer", "answer.yaml", "Answer", "ans wer", "", "nonexistent", "critics/answer"])
def test_load_critic_refuses_a_name_outside_the_listing(name):
    with pytest.raises(ValueError):
        panel.load_critic(name)


def test_load_critic_ignores_a_tools_key_in_the_file(monkeypatch):
    class Item:
        name = "answer.yaml"

        def is_file(self):
            return True

        def read_text(self, encoding="utf-8"):
            return "name: answer\npurpose: p\nrole: r\ntools: [bash_exec]\n"

    monkeypatch.setattr(panel, "_listing", lambda: [Item()])
    assert panel.load_critic("answer").tools.add == list(panel.CRITIC_TOOLS)


def test_no_critic_is_a_root_agent(tmp_path):
    loader = ConfigLoader(config_dir=tmp_path)
    assert not set(NINE) & set(loader.list_agents())
    assert not any("critics" in str(p) for p in loader.agent_yaml_paths())
