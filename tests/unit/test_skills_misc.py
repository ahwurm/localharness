"""Skills seams outside the skill modules: the removed microagents key, protected paths and the origins they guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate_types import PROTECTED_CONFIG_DIR_ENTRIES, Verdict
from localharness.agent.verdict import evaluate
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.models import ContextConfig
from tests.unit.agent.test_auto_mode import SETTINGS, SHELL, WRITE, _ctx, _shell, _verdict
from tests.unit.test_leftover_plugin_section import _CONFIG, _agent_file


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    return g


def test_a_yaml_that_still_sets_context_microagents_fails_naming_the_key(g) -> None:
    agent = _agent_file(g, "context:\n  microagents: [style]\n")

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    assert "context.microagents (line 4): Extra inputs are not permitted" in str(exc.value), str(exc.value)


def test_the_microagents_field_and_loader_methods_are_gone() -> None:
    assert "microagents" not in ContextConfig.model_fields
    assert not hasattr(ConfigLoader, "microagent_paths")
    assert not hasattr(ConfigLoader, "discover_microagents")


_REFUSED = "Extra inputs are not permitted"


@pytest.mark.parametrize(
    "tail,expected",
    [
        pytest.param("  microagents: []\n", ["context.microagents (line 4): " + _REFUSED], id="empty-list"),
        pytest.param("  microagents:\n", ["context.microagents (line 4): " + _REFUSED], id="null"),
        pytest.param(
            "  microagents: [style]\n  bogus: 1\n",
            ["context.microagents (line 4): " + _REFUSED, "context.bogus (line 5): " + _REFUSED],
            id="beside-another-unknown-key",
        ),
        pytest.param("  microagent: [style]\n", ["context.microagent (line 4): " + _REFUSED], id="singular"),
    ],
)
def test_the_refusal_holds_for_every_form_of_the_key(g, tail, expected) -> None:
    agent = _agent_file(g, "context:\n" + tail)

    with pytest.raises(ConfigValidationError) as exc:
        ConfigLoader(config_dir=g).load_agent_file(agent)

    assert all(text in str(exc.value) for text in expected), str(exc.value)


def test_a_valid_context_block_still_loads(g) -> None:
    agent = _agent_file(g, "context:\n  max_context_tokens: 4096\n")

    assert ConfigLoader(config_dir=g).load_agent_file(agent).context.max_context_tokens == 4096


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def global_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "global"
    folder.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(folder))
    return folder


def test_skills_is_the_thirteenth_protected_config_entry() -> None:
    assert len(PROTECTED_CONFIG_DIR_ENTRIES) == 13
    assert PROTECTED_CONFIG_DIR_ENTRIES[-1] == "skills"


def test_a_write_under_localharness_skills_is_an_ungrantable_protected_ask(project, global_dir) -> None:
    target = project / ".localharness/skills/release/SKILL.md"

    decision = evaluate("write", {"path": str(target)}, WRITE, _ctx(project), SETTINGS)

    assert decision.verdict is Verdict.ASK
    assert decision.request.klass == "protected-path"
    assert decision.request.grantable is False
    assert decision.request.reason.startswith("writes a protected path ("), decision.request.reason


@pytest.mark.parametrize(
    "tail,verdict",
    [
        pytest.param(".agents/skills/release/SKILL.md", Verdict.ASK, id="below-agents-skills"),
        pytest.param(".agents/skills", Verdict.ASK, id="agents-skills-itself"),
        pytest.param(".agents/skills/.localharness/x", Verdict.ASK, id="nested-localharness-cannot-reopen"),
        pytest.param(".agents", Verdict.ALLOW, id="agents-itself"),
        pytest.param(".agents/skillsx/x", Verdict.ALLOW, id="longer-name"),
        pytest.param(".agents/skill/x", Verdict.ALLOW, id="shorter-name"),
        pytest.param(".agents/notes.md", Verdict.ALLOW, id="sibling-file"),
        pytest.param("sub/.agents/skills/x", Verdict.ALLOW, id="not-at-a-container-root"),
        pytest.param(".agents/./skills/x/SKILL.md", Verdict.ASK, id="dot-part"),
        pytest.param(".agents//skills/x/SKILL.md", Verdict.ASK, id="doubled-slash"),
        pytest.param(".agents/skills/../notes.md", Verdict.ALLOW, id="dotdot-out"),
        pytest.param(".Agents/Skills/x", Verdict.ALLOW, id="case-is-exact"),
    ],
)
def test_agents_skills_is_protected_exactly_at_a_container_root(project, global_dir, tail, verdict) -> None:
    assert _verdict("write", {"path": f"{project}/{tail}"}, _ctx(project), WRITE) is verdict


def test_agents_skills_is_protected_from_the_workspace_and_from_the_boundary(project, global_dir) -> None:
    target = {"path": str(project / ".agents/skills/release/SKILL.md")}

    assert _verdict("write", target, _ctx(project, workspace=project / "sub"), WRITE) is Verdict.ASK
    assert _verdict("write", target, _ctx(project, boundary=None), WRITE) is Verdict.ASK


@pytest.mark.parametrize(
    "command",
    [
        "mv .agents away",
        "rm -rf .agents",
        "rm -rf .agents/skills",
        "mv .agents/skills elsewhere",
        "rm .agents/skills/release/SKILL.md",
        "rm .agents/plugins",
    ],
)
def test_moving_or_deleting_agents_skills_or_its_parent_asks(project, global_dir, command) -> None:
    decision = evaluate("bash_exec", {"command": command}, SHELL, _ctx(project), SETTINGS)

    assert decision.verdict is Verdict.ASK, command
    assert "deletes or moves a protected config entry" in decision.request.reason, decision.request.reason


@pytest.mark.parametrize(
    "command",
    ["rm .agents/notes.md", "mv .agents/notes.md .agents/n.md", "rm -rf skills", "rm -rf docs/skills"],
)
def test_other_children_of_agents_and_plain_skills_folders_do_not_ask(project, global_dir, command) -> None:
    assert _shell(command, _ctx(project)) is Verdict.ALLOW, command
