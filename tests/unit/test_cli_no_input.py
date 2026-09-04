"""F6 — `--no-input`: a run with nobody watching never spends the one-time trust answer.

The workspace trust question is asked ONCE and the answer is permanent. `--json` already forces
the non-interactive path for the commands that have it, but `doctor`, `validate` and `agent create`
have no machine-output mode at all — so a git hook, a CI job or a `watch` loop running any of them
from a directory with an untrusted workspace hit the prompt. On a tty that is a permanent decision
made by whatever was on stdin; with no tty it was already inert, but there was no way to ASK for
that behavior when a terminal happened to be attached.

Every test forces a terminal to be present. That is the whole point: with no tty the
non-interactive path is reached anyway and the flag would grade nothing.

The trust store is checked after each run, not just the output — "did not prompt" and "did not
record" are two different promises, and the second is the one that outlives the process.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config import trust
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()

_MINIMAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "default_model": "test-model",
    },
}


class _FakeSys:
    """A stdin that claims to be a terminal, for the whole CLI invocation.

    Patching `sys.stdin.isatty` does not survive CliRunner — click reassigns `sys.stdin` while the
    command runs. The resolver reads stdin off its own module-level `sys`, so replacing that name
    is the patch that holds (39-04's technique).
    """

    class stdin:  # noqa: N801 - mimicking the module attribute
        @staticmethod
        def isatty() -> bool:
            return True


@pytest.fixture
def undecided_workspace(tmp_path, monkeypatch) -> Path:
    """An untrusted, undecided workspace ABOVE the cwd and outside any repository — the only
    shape the trust gate fires for — plus a terminal that is definitely attached."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    (home / WORKSPACE_DIR_NAME / "config.yaml").write_text(
        yaml.dump(_MINIMAL_CONFIG), encoding="utf-8"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    ws = tmp_path / "proj" / WORKSPACE_DIR_NAME
    (ws / "agents").mkdir(parents=True)
    deep = tmp_path / "proj" / "src"
    deep.mkdir()
    monkeypatch.chdir(deep)

    from localharness.config.paths import workspace_is_within_repo

    assert not workspace_is_within_repo(ws, deep), "TMPDIR is inside a repo; the gate is inert"
    monkeypatch.setattr("localharness.cli.workspace.sys", _FakeSys())
    return ws.resolve()


@pytest.fixture
def asked(monkeypatch) -> list:
    """A spy, not a raising stub: `CliRunner.invoke` SWALLOWS exceptions into `result.exception`,
    so an `assert` inside the prompt would be caught by the runner and the test would pass while
    the prompt fired. The list is checked by the test itself, where nothing can swallow it."""
    seen: list = []

    def _ask(*args, **kwargs):
        seen.append(args[0] if args else kwargs.get("prompt"))
        return False

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return seen


@pytest.fixture
def prompt_answers_yes(monkeypatch) -> list:
    asked: list = []

    def _ask(*args, **kwargs):
        asked.append(args[0] if args else kwargs.get("prompt"))
        return True

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


@pytest.mark.parametrize("argv", [
    ["doctor", "--no-input"],
    ["validate", "--no-input"],
    ["agent", "create", "hooked", "--project", "--no-input", "--role", "R"],
])
def test_no_input_never_asks(undecided_workspace, asked, argv):
    runner.invoke(app, argv)

    assert asked == []


@pytest.mark.parametrize("argv", [
    ["doctor", "--no-input"],
    ["validate", "--no-input"],
    ["agent", "create", "hooked", "--project", "--no-input", "--role", "R"],
])
def test_no_input_records_nothing(undecided_workspace, asked, argv):
    """The promise that outlives the process: a later interactive run in this directory still
    gets its one question."""
    runner.invoke(app, argv)

    assert trust.is_trusted(undecided_workspace) is None


@pytest.mark.parametrize("argv", [["doctor"], ["validate"]])
def test_without_the_flag_the_same_setup_does_ask(
    undecided_workspace, prompt_answers_yes, argv
):
    """The control. Without it, "did not prompt" could mean the fake terminal never reached the
    resolver at all — this proves the same setup DOES ask, so the difference is the flag."""
    runner.invoke(app, argv)

    assert len(prompt_answers_yes) == 1


def test_no_input_says_why_the_layer_was_skipped(undecided_workspace, asked):
    """Skipping silently would be a session quietly running on different config from the one the
    project's files describe."""
    result = runner.invoke(app, ["doctor", "--no-input"])

    assert "non-interactive" in result.output
    assert str(undecided_workspace) in result.output


def test_agent_create_no_input_without_a_scope_refuses(tmp_path, monkeypatch):
    """`agent create` has a second prompt — which layer to write to — and no safe default: writing
    to the wrong one is the mistake A-B3 is about."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["agent", "create", "scopeless", "--no-input", "--role", "R"])

    assert result.exit_code == 2, result.output
    assert "--global" in result.output and "--project" in result.output


@pytest.mark.parametrize("argv", [
    ["doctor", "--help"],
    ["validate", "--help"],
    ["agent", "create", "--help"],
])
def test_the_help_says_what_the_flag_is_for(argv):
    result = runner.invoke(app, argv)

    assert "--no-input" in result.output
    assert "CI" in result.output
