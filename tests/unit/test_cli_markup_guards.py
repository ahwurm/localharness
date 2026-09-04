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


def test_init_receipt_names_the_config_it_wrote(hostile_project, tmp_path, monkeypatch):
    """The receipt prints AFTER config.yaml is on disk, so this crash exited 1 on a SUCCESS.

    A user with a markup-named config dir was told init had failed by the very line that proves
    it worked — and `doctor` then found the config the receipt said was never written.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    import localharness.cli.init_cmd as init_cmd
    from localharness.provider.client import CapabilityResult
    from localharness.provider.detector import DetectorResult

    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")
    cfg_dir = tmp_path / HOSTILE / "cfg"
    cfg_dir.mkdir(parents=True)

    with patch.object(init_cmd, "detect_provider") as detect, \
            patch.object(init_cmd, "LLMClient") as client_cls:
        detect.return_value = DetectorResult(
            found=True, provider_type="vllm", base_url="http://localhost:8000/v1",
            models=["m"], suggested_model="m", probe_duration_ms=1.0,
        )
        client = MagicMock()
        client.detect_capabilities = AsyncMock(return_value=CapabilityResult(
            tool_call_mode="native", context_window=128_000, supports_streaming=True,
            probe_duration_ms=10.0, probe_error=None, server_reached=True,
        ))
        client_cls.return_value = client
        result = runner.invoke(app, ["init", "--config-dir", str(cfg_dir), "--force"])

    assert (cfg_dir / "config.yaml").exists(), "the write itself never happened"
    assert result.exit_code == 0, result.output
    assert str(cfg_dir / "config.yaml") in result.output


def test_init_guided_setup_names_the_launch_command_and_log(hostile_project, tmp_path, monkeypatch):
    """Guided setup prints the launch command and the server log path — both derived from the
    config dir, so both died on a markup-named one, after the model had already been downloaded."""
    import io
    import sys

    from rich.console import Console

    import localharness.cli.init_cmd as init_cmd
    from localharness.provider import server as managed_server

    buf = io.StringIO()
    monkeypatch.setattr(init_cmd, "console", Console(file=buf, width=400))
    cfg_dir = tmp_path / HOSTILE / "cfg"
    cfg_dir.mkdir(parents=True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(init_cmd.Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(init_cmd.IntPrompt, "ask", staticmethod(lambda *a, **k: 1))
    # An existing local path as the model: skips the HF-cache/download branch entirely.
    monkeypatch.setattr(init_cmd.Prompt, "ask", staticmethod(lambda *a, **k: str(tmp_path)))
    # The one real path that matters is log_path(); find_vllm is stubbed only to skip the install.
    monkeypatch.setattr(managed_server, "find_vllm", lambda d: str(d / "server" / "venv" / "vllm"))
    monkeypatch.setattr(managed_server, "start_server", lambda d, cmd: None)

    async def _ready(base_url, config_dir=None):
        return ["m"]
    monkeypatch.setattr(managed_server, "wait_ready", _ready)

    init_cmd._guided_setup(cfg_dir)

    printed = buf.getvalue()
    assert str(managed_server.log_path(cfg_dir)) in printed, printed
    assert str(cfg_dir / "server" / "venv" / "vllm") in printed, printed


def test_agent_list_names_an_unparseable_workspace_config(hostile_project, monkeypatch):
    """`agent list` reads the workspace layer, so a file in that layer it could not parse is
    news — and it was silent about it: a broken workspace config.yaml produced the same
    "No agents configured" as an empty project, exit 0, nothing on stderr.

    Same line `config show` prints for the same condition, and the path is escaped for the same
    reason every other path in cli/ is.
    """
    ws = hostile_project / WORKSPACE_DIR_NAME
    ws.mkdir(parents=True)
    (hostile_project / ".git").mkdir()  # in-project workspace = loads silently (LAYR-05)
    broken = ws / "config.yaml"
    broken.write_text("org:\n  name: [unclosed\n", encoding="utf-8")

    result = runner.invoke(app, ["agent", "list"])

    assert result.exit_code == 0, result.output
    assert "unreadable, skipped" in result.output, result.output
    assert str(broken) in result.output, result.output
