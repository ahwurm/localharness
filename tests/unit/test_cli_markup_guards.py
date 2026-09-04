"""Every CLI line that prints a PATH or a VALUE must print the one the user actually has.

Rich parses `[...]` in whatever you hand `console.print`. Directory names, model names, config
values and exception texts are all data the user controls, and two things happen when data reaches
the markup parser:

* `[old] proj` — a balanced-looking tag — is SILENTLY DELETED. The command then reports a path
  that does not exist, which is worst precisely in the commands people run to find out where their
  config comes from (39-05's measured lesson).
* `[/red]proj` — a closing tag with nothing open — raises `MarkupError` and the command DIES. The
  bad-mood review's E cluster traced most of its 21 crashes to this one root cause.

So every fixture directory here is named with both shapes, and every assertion is `in
result.output` against the verbatim path. That makes each test a markup guard as a side effect of
being an ordinary "does it say the right thing" test — no test here exists to grade `escape()`
itself, and none would keep passing if the escaping were removed.

`COLUMNS` is set wide on purpose: rich hard-wraps at the console width, so a narrow terminal would
fold a newline into the path and fail these assertions for a reason that has nothing to do with
markup. The soft_wrap half of the same fix is graded in test_cli_soft_wrap.py's narrow console.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()

# Both failure shapes in one name: the silent-deletion tag and the outright-crash tag.
HOSTILE = "[old] [/red]proj"

_MINIMAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "test-model",
    },
}


@pytest.fixture
def hostile_project(tmp_path, monkeypatch) -> Path:
    """CWD inside a project whose directory name is rich markup, with an empty global layer.

    Every env var the commands read is cleared or repointed: a developer's real `~/.localharness`
    must not answer these, and `LOCALHARNESS_DIR` being set would make `--project` take the
    explicit-config-dir refusal path instead of the one under test.
    """
    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT",
                "LOCALHARNESS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    (home / WORKSPACE_DIR_NAME / "config.yaml").write_text(
        yaml.dump(_MINIMAL_CONFIG), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    proj = tmp_path / HOSTILE
    proj.mkdir(parents=True)
    monkeypatch.chdir(proj)
    return proj


def test_init_workspace_names_the_directory_it_made(hostile_project):
    """The three lines that say WHERE the new workspace is must survive the parser."""
    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 0, result.output
    target = hostile_project / WORKSPACE_DIR_NAME
    assert str(target) in result.output
    assert str(target / "config.yaml") in result.output
    assert str(target / "agents") in result.output


def test_init_workspace_refusal_names_the_existing_directory(hostile_project):
    """The refusal path prints the same path from a different branch — and used to crash here."""
    (hostile_project / WORKSPACE_DIR_NAME).mkdir()

    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 1
    assert str(hostile_project / WORKSPACE_DIR_NAME) in result.output


def test_agent_create_receipt_names_the_file_it_wrote(hostile_project):
    """Scaffold first, so `--project` resolves the DISCOVERED workspace and the receipt carries an
    absolute path — the shape that actually goes through the markup parser. (With nothing to
    discover the command writes the relative `.localharness/...` literal, which has no brackets in
    it and would grade nothing here.)"""
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["agent", "create", "deployer", "--project"])

    written = hostile_project / WORKSPACE_DIR_NAME / "agents" / "deployer.yaml"
    assert result.exit_code == 0, result.output
    assert written.exists()
    assert str(written) in result.output


def test_agent_create_refusal_names_the_file_it_kept(hostile_project):
    runner.invoke(app, ["agent", "create", "deployer", "--project"])

    result = runner.invoke(app, ["agent", "create", "deployer", "--project"])

    written = hostile_project / WORKSPACE_DIR_NAME / "agents" / "deployer.yaml"
    assert result.exit_code == 1
    assert str(written) in result.output


def test_agent_list_skip_warning_names_the_unreadable_file(hostile_project):
    """The one message whose entire job is naming the file that went wrong."""
    agents = hostile_project / WORKSPACE_DIR_NAME / "agents"
    agents.mkdir(parents=True)
    broken = agents / "broken.yaml"
    broken.write_text("name: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["agent", "list"])

    assert str(broken) in result.output


def test_validate_missing_file_echoes_the_name_you_typed(hostile_project):
    missing = hostile_project / "[/red]nope.yaml"

    result = runner.invoke(app, ["validate", str(missing)])

    assert result.exit_code == 1
    assert str(missing) in result.output


def test_config_show_layer_rows_name_the_real_files(hostile_project):
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, result.output
    workspace = hostile_project / WORKSPACE_DIR_NAME
    assert str(workspace / "config.yaml") in result.output


def test_components_set_receipt_shows_the_value_you_set(hostile_project):
    """A VALUE is data too: `[dim]` as a value renders as nothing at all, so the receipt would
    confirm setting the key to an empty string."""
    result = runner.invoke(
        app, ["components", "set", "org.log_level", "[/red]debug"]
    )

    # The value is rejected or accepted depending on the field's type; either way the receipt or
    # the error must quote back what was typed, not a parsed-away fragment of it.
    assert "[/red]debug" in result.output, result.output


def test_doctor_names_the_workspace_it_found(hostile_project):
    runner.invoke(app, ["init", "--workspace"])

    result = runner.invoke(app, ["doctor"])

    assert str(hostile_project / WORKSPACE_DIR_NAME) in result.output
