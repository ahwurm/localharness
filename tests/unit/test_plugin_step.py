"""52-03: the plugin setup step — the one `plugins enable NAME` runs on a terminal, and (in-session
mode) the one `/plugins enable NAME` runs between the two halves of a session's restart (52-06).

After the questions and the one write: the plugin's setup action (asked first when its manifest
sets a question), the doctor check whenever it declares questions or an action, then setup_help
and the coding-agent prompt — filled with the real values — only when that check is not clean,
and next_steps on every outcome. Without a terminal nothing is asked and nothing runs; the next
step names what to run on one.

Stubs are bundled-shaped plugins swapped into BUILTIN_PLUGINS; every run passes an explicit
--config-dir; gpu_name is patched in every test, so no test runs nvidia-smi. Plain `def` tests:
the step calls asyncio.run, and the suite's asyncio_mode would put an `async def` test inside a
running loop."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery, setup
from localharness.plugins.api import GLOBAL_ONLY, Check, Plugin, PluginManifest, SetupField
from localharness.plugins.setup import AGENT_PROMPT_LEAD
from tests.unit.test_plugin_sections import _INFO_KEYS
from tests.unit.test_plugins_enable_setup import _CONFIG

runner = CliRunner()
CALLS: list[str] = []          # every setup action that ran, by plugin name
ACT_PASSES = [True]            # act's doctor: pass or fail
ACT_QUESTION = "Fetch the thing now (about 1 MB)?"


class UrlConfig(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)


class Act(Plugin):
    """fetches the thing"""

    manifest = PluginManifest(name="act", version="0.1.0", kind="tools", enabled_by_default=False,
                              setup_action=ACT_QUESTION, next_steps="Then: run the thing.",
                              agent_prompt="Set up the thing under {config_dir}.")

    def setup_action(self, ctx):
        CALLS.append("act")
        return [Check(name="act", status="pass", detail="fetched")]

    def doctor(self, ctx):
        return [Check(name="act", status="pass", detail="ready") if ACT_PASSES[0]
                else Check(name="act", status="fail", detail="not ready")]


class Quiet(Plugin):
    """sets itself up without asking"""

    manifest = PluginManifest(name="quiet", version="0.1.0", kind="tools", enabled_by_default=False)

    async def setup_action(self, ctx):
        CALLS.append("quiet")
        return [Check(name="quiet", status="pass", detail="done quietly")]


class Boom(Plugin):
    """breaks while it sets up"""

    manifest = PluginManifest(name="boom", version="0.1.0", kind="tools", enabled_by_default=False)

    def setup_action(self, ctx):
        raise RuntimeError("boom")


class Pro(Plugin):
    """answers at an address"""

    manifest = PluginManifest(
        name="pro", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),),
        setup_help="PRO-HELP", agent_prompt="Run it at {url}. {machine} Done when it answers.")
    ConfigModel = UrlConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "pro.url")

    def doctor(self, ctx):
        url = ctx.config.url
        if url == "http://ok":
            return [Check(name="pro", status="pass", detail=f"answers at {url}")]
        return [Check(name="pro", status="fail", detail=f"no answer at {url}")]


class Needx(Plugin):
    """needs an install extra nobody has"""

    manifest = PluginManifest(
        name="needx", version="0.1.0", kind="tools", enabled_by_default=False,
        requires_extra="nosuchextra", next_steps="NEEDX-NEXT",
        setup=(SetupField(key="url", prompt="Needx address", default="http://127.0.0.1:2"),))
    ConfigModel = UrlConfig


class Secty(Plugin):
    """sets two core settings under the section it claims"""

    manifest = PluginManifest(
        name="secty", version="0.1.0", kind="dev", sections=("proposer",), setup_help="SECTY-HELP",
        setup=(SetupField(key="proposer.base_url", prompt="Proposer address"),
               SetupField(key="proposer.model", prompt="Proposer model")))

    def doctor(self, ctx):
        return [Check(name="secty", status="skip", detail="not checked")]


class Plain(Plugin):
    """has nothing to set up"""

    manifest = PluginManifest(name="plain", version="0.1.0", kind="tools", enabled_by_default=False)


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Act, Quiet, Boom, Pro, Needx, Secty, Plain))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")
    CALLS.clear()
    ACT_PASSES[0] = True


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


@pytest.fixture
def term(monkeypatch):
    """A terminal: `answers` are typed and `yes` given in order; every question is recorded. A
    question nobody scripted fails the test (pop from an empty list)."""
    t = SimpleNamespace(asked=[], confirmed=[], answers=[], yes=[])

    def prompt(text, default=None, **kw):
        t.asked.append((text, default))
        return t.answers.pop(0)

    def confirm(text, default=None, **kw):
        t.confirmed.append((text, default))
        return t.yes.pop(0)

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    monkeypatch.setattr(plugins_cmd.typer, "confirm", confirm)
    return t


@pytest.fixture
def no_term(monkeypatch):
    def asked(*a, **kw):
        raise AssertionError("asked")
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: False)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", asked)
    monkeypatch.setattr(plugins_cmd.typer, "confirm", asked)


def _enable(g: Path, *args: str):
    return runner.invoke(app, ["plugins", "enable", *args, "--config-dir", str(g)])


def _info(g: Path, *args: str):
    return runner.invoke(app, ["plugins", "info", *args, "--config-dir", str(g)])


def _in_order(output: str, *texts: str) -> None:
    at = [output.find(t) for t in texts]
    assert -1 not in at, [t for t, i in zip(texts, at) if i == -1]
    assert at == sorted(at), list(zip(texts, at))


# --- the setup action ------------------------------------------------------------------------------


def test_the_action_is_asked_first_then_runs_then_the_check(g, term) -> None:
    term.yes[:] = [True]
    result = _enable(g, "act")

    assert result.exit_code == 0, result.output
    assert term.confirmed == [(ACT_QUESTION, True)] and CALLS == ["act"]
    _in_order(result.output, "✓ act: fetched", "Checking it now:", "✓ act: ready", "  Then: run the thing.")


def test_no_to_its_question_skips_the_action_and_still_checks(g, term) -> None:
    term.yes[:] = [False]
    result = _enable(g, "act")

    assert result.exit_code == 0, result.output
    assert CALLS == [] and "fetched" not in result.output
    _in_order(result.output, "Checking it now:", "✓ act: ready")


def test_an_action_without_a_question_runs_without_asking(g, term) -> None:
    result = _enable(g, "quiet")

    assert result.exit_code == 0, result.output
    assert term.confirmed == [] and CALLS == ["quiet"]
    _in_order(result.output, "✓ quiet: done quietly", "Checking it now:")  # rule e: an action, no fields


def test_an_action_that_raises_is_one_failing_row(g, term) -> None:
    result = _enable(g, "boom")

    assert result.exit_code == 0, result.output
    assert "✗ boom: its setup action raised RuntimeError: boom" in result.output


def test_without_a_terminal_no_action_runs_and_the_next_step_names_its_question(g, no_term) -> None:
    result = _enable(g, "act")

    assert result.exit_code == 0, result.output
    assert CALLS == [] and "Checking it now" not in result.output
    assert ("next step — run `localharness plugins enable act` on a terminal to answer: "
            f"{ACT_QUESTION}") in result.output
    assert "  Then: run the thing." in result.output


def test_a_plugin_with_nothing_to_set_up_gets_no_check_and_no_next_step(g, term) -> None:
    result = _enable(g, "plain")

    assert result.exit_code == 0, result.output
    assert "next step" not in result.output and "Checking it now" not in result.output


# --- the check, setup_help and the coding-agent prompt ----------------------------------------------


PROMPT_BAD = "Run it at http://bad. This machine reports NVIDIA GB10. Done when it answers."


def test_a_failing_check_prints_the_row_the_help_then_the_filled_prompt(g, term) -> None:
    term.answers[:] = ["http://bad"]
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    _in_order(result.output, "✗ pro: no answer at http://bad", "PRO-HELP", AGENT_PROMPT_LEAD, PROMPT_BAD)


def test_a_passing_check_prints_no_help_and_no_prompt(g, term) -> None:
    term.answers[:] = ["http://ok"]
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    assert "✓ pro: answers at http://ok" in result.output
    assert "PRO-HELP" not in result.output and AGENT_PROMPT_LEAD not in result.output


def test_with_no_gpu_reported_the_prompt_has_no_machine_sentence(g, term, monkeypatch) -> None:
    monkeypatch.setattr(setup, "gpu_name", lambda: None)
    term.answers[:] = ["http://bad"]
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    assert "Run it at http://bad. Done when it answers." in result.output
    assert "This machine reports" not in result.output


def test_setup_help_prints_once_on_a_failing_check(g, term) -> None:
    term.answers[:] = ["http://bad"]
    result = _enable(g, "pro")

    assert result.output.count("PRO-HELP") == 1, result.output


def test_the_check_reads_what_was_just_written(g, term) -> None:
    """Its questions count as answered only when read through a NEW loader after the write:
    ConfigLoader caches its harness and raw sources, so the pre-write one still sees no proposer."""
    term.answers[:] = ["http://p/v1", "p-model"]
    result = _enable(g, "secty")

    assert result.exit_code == 0, result.output
    assert "Checking it now:" in result.output and "not checked" in result.output
    assert "SECTY-HELP" not in result.output


def test_a_prompt_without_machine_never_runs_gpu_name(g, term, monkeypatch) -> None:
    def ran():
        raise AssertionError("ran")
    monkeypatch.setattr(setup, "gpu_name", ran)
    wanted = f"Set up the thing under {g}."

    info = _info(g, "act")
    assert info.exit_code == 0, info.output
    assert wanted in info.output

    ACT_PASSES[0] = False
    term.yes[:] = [False]
    result = _enable(g, "act")
    assert result.exit_code == 0, result.output
    _in_order(result.output, "✗ act: not ready", AGENT_PROMPT_LEAD, wanted)


# --- next steps, on every outcome --------------------------------------------------------------------


def test_next_steps_print_after_a_passing_check(g, term) -> None:
    term.yes[:] = [True]
    assert "  Then: run the thing." in _enable(g, "act").output


def test_next_steps_print_after_a_failing_check(g, term) -> None:
    ACT_PASSES[0] = False
    term.yes[:] = [True]
    result = _enable(g, "act")
    _in_order(result.output, "✗ act: not ready", "  Then: run the thing.")


def test_next_steps_print_when_the_install_extra_is_missing(g, term) -> None:
    result = _enable(g, "needx")

    assert result.exit_code == 0, result.output
    assert term.asked == [] and "Checking it now" not in result.output
    _in_order(result.output, "needx is missing its install extra", "  NEEDX-NEXT")


def test_next_steps_print_without_a_terminal(g, no_term) -> None:
    assert "  Then: run the thing." in _enable(g, "act").output


# --- plugins info ----------------------------------------------------------------------------------


def test_info_ends_with_the_prompt_filled_with_the_stored_value_else_the_default(g, no_term) -> None:
    lines = _info(g, "pro").output.rstrip().splitlines()
    assert lines[-3:] == [AGENT_PROMPT_LEAD, "",
                          "  Run it at http://127.0.0.1:1. This machine reports NVIDIA GB10. Done when it answers."]

    assert _enable(g, "pro", "--set", "url=http://saved").exit_code == 0
    assert "  Run it at http://saved. This machine reports" in _info(g, "pro").output
    data = json.loads(_info(g, "pro", "--json").stdout)
    assert set(data) == _INFO_KEYS | {"sections"}
