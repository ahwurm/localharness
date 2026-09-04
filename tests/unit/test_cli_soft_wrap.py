"""A path folded at the console width is a path the user cannot copy.

Rich hard-wraps `console.print` output at the terminal width, so on a normal 80-column terminal a
deep project path arrives with a newline folded into the middle of it. The line still LOOKS
wrapped either way — the difference is whether the newline is in the DATA. `soft_wrap=True` hands
the line to the terminal whole and lets the terminal do the folding, which is what makes a
double-click select the whole path.

Found by driving the real binary, not by a test (43-04); these are the tests that were missing.
Every case here runs at a deliberately cruel width so an unwrapped assertion cannot pass by
accident, and asserts the path appears in the output as ONE unbroken substring.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()

# Long enough that any width-based fold lands inside the path, not before it.
_DEEP = "a-long-project-directory-name/and-a-nested-subdirectory-inside-it"

_MINIMAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "test-model",
    },
}


@pytest.fixture
def narrow_project(tmp_path, monkeypatch) -> Path:
    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT",
                "LOCALHARNESS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    (home / WORKSPACE_DIR_NAME / "config.yaml").write_text(
        yaml.dump(_MINIMAL_CONFIG), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "40")  # cruel on purpose
    proj = tmp_path / _DEEP
    proj.mkdir(parents=True)
    monkeypatch.chdir(proj)
    return proj


def test_init_workspace_paths_arrive_unbroken(narrow_project):
    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 0, result.output
    target = narrow_project / WORKSPACE_DIR_NAME
    assert str(target / "config.yaml") in result.output
    assert str(target / "agents") in result.output


def test_init_workspace_refusal_path_arrives_unbroken(narrow_project):
    (narrow_project / WORKSPACE_DIR_NAME).mkdir()

    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 1
    assert str(narrow_project / WORKSPACE_DIR_NAME) in result.output


def test_agent_create_receipt_path_arrives_unbroken(narrow_project):
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["agent", "create", "deployer", "--project"])

    assert result.exit_code == 0, result.output
    assert str(narrow_project / WORKSPACE_DIR_NAME / "agents" / "deployer.yaml") in result.output


def test_doctor_workspace_line_arrives_unbroken(narrow_project):
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["doctor"])

    assert str(narrow_project / WORKSPACE_DIR_NAME) in result.output


def test_trust_notice_arrives_unbroken(narrow_project, capsys):
    """F12: the notices and the trust PROMPT share one console, and that console — not each call
    site — is where soft_wrap lives, because `Console.input` takes no soft_wrap argument. Grading
    the notice is what pins the console-level setting."""
    from localharness.cli.workspace import _notice

    long_path = narrow_project / WORKSPACE_DIR_NAME
    _notice(f"Workspace {long_path} is not trusted — its config layer is ignored.")

    assert str(long_path) in capsys.readouterr().err
