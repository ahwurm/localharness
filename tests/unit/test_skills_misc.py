"""Skills seams outside the skill modules: the removed microagents key, protected paths and the origins they guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate_types import PROTECTED_CONFIG_DIR_ENTRIES, Verdict
from localharness.agent.verdict import evaluate
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.models import ContextConfig
from tests.unit.agent.test_auto_mode import SETTINGS, WRITE, _ctx
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
