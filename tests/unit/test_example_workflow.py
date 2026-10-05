"""The shipped research-note example: its lint, instructions, setup line, and specialist agents."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from localharness.agent.subagent import _config_child_allowed
from localharness.config.models import AgentConfig

EX = Path(__file__).resolve().parents[2] / "examples" / "workflows" / "research-note"
SETUP = "mkdir -p .localharness/agents && cp agents/*.yaml .localharness/agents/"


def lint(draft):
    return subprocess.run([sys.executable, str(EX / "checks" / "lint.py"), str(draft)],
                          capture_output=True, text=True)


def test_lint_passes_a_clean_draft_and_fails_a_banned_phrase(tmp_path):
    draft = tmp_path / "draft.md"
    clean = "# Weekly numbers\n\n" + "The team reviews its weekly numbers on Monday. " * 50 + "\n"
    draft.write_text(clean)
    ok = lint(draft)
    assert ok.returncode == 0 and ok.stdout.startswith("PASS"), ok.stdout
    draft.write_text(clean + "The new report was a game-changer for the team.\n")
    bad = lint(draft)
    assert bad.returncode == 1 and "banned phrase" in bad.stdout, bad.stdout


def test_instructions_name_the_lint_command_and_working_record():
    text = (EX / "INSTRUCTIONS.md").read_text()
    assert "python3 checks/lint.py draft.md" in text
    assert "## Working record" in text and "exit_code" in text


def test_readme_and_docs_carry_the_agent_setup_line():
    """LocalHarness loads project agents from .localharness/agents/, not the example's agents/."""
    assert SETUP in (EX / "README.md").read_text()
    assert SETUP in (EX.parents[2] / "docs" / "task-context.md").read_text()
    assert SETUP in (EX.parents[1] / "README.md").read_text()


@pytest.mark.parametrize("name", ["reviewer", "writer"])
def test_example_agents_validate_without_agent_or_task(name):
    cfg = AgentConfig.model_validate(yaml.safe_load((EX / "agents" / f"{name}.yaml").read_text()))
    assert cfg.name == name
    assert "agent" not in cfg.tools.add and "task" not in cfg.tools.add
    allowed = _config_child_allowed(cfg)
    assert "agent" not in allowed and "task" not in allowed
