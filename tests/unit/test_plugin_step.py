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

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import click
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
ACT_READY = [False]            # act's thing is already there, as if fetched before this run
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

    def doctor(self, ctx):  # passes once the thing is there: fetched now, or before this run
        return [Check(name="act", status="pass", detail="ready") if ACT_READY[0] or "act" in CALLS
                else Check(name="act", status="fail", detail="not fetched")]


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


class Kit(Plugin):
    """asks an address, then warms up"""

    manifest = PluginManifest(
        name="kit", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Kit address", default="http://127.0.0.1:3"),),
        setup_action="Warm it up now?")
    ConfigModel = UrlConfig

    def setup_action(self, ctx):
        CALLS.append("kit")
        return [Check(name="kit", status="pass", detail="warm")]

    def doctor(self, ctx):
        return [Check(name="kit", status="pass" if ctx.config.url == "http://ok" else "fail",
                      detail=f"at {ctx.config.url}")]


class Plain(Plugin):
    """has nothing to set up"""

    manifest = PluginManifest(name="plain", version="0.1.0", kind="tools", enabled_by_default=False)


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Act, Quiet, Boom, Pro, Needx, Secty, Kit, Plain))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")
    CALLS.clear()
    ACT_READY[0] = False


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
    assert CALLS == [] and "✓ act: fetched" not in result.output
    _in_order(result.output, "Checking it now:", "✗ act: not fetched")


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

    term.yes[:] = [False]
    result = _enable(g, "act")
    assert result.exit_code == 0, result.output
    _in_order(result.output, "✗ act: not fetched", AGENT_PROMPT_LEAD, wanted)


# --- next steps, on every outcome --------------------------------------------------------------------


def test_next_steps_print_after_a_passing_check(g, term) -> None:
    term.yes[:] = [True]
    assert "  Then: run the thing." in _enable(g, "act").output


def test_next_steps_print_after_a_failing_check(g, term) -> None:
    term.yes[:] = [False]
    result = _enable(g, "act")
    _in_order(result.output, "✗ act: not fetched", "  Then: run the thing.")


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


def test_a_filled_value_keeps_its_own_spaces(tmp_path, no_term) -> None:
    """The renderer squeezes the spaces a placeholder that renders empty leaves, never a value's
    own: a config folder whose name holds two spaces is printed as it is."""
    g = tmp_path / "My  Home"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    assert f"Set up the thing under {g}." in _info(g, "act").output


# --- step_pending: does the step still have work? ---------------------------------------------------


def _write(g: Path, overrides: dict) -> None:
    (g / "overrides.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")


def _pending(g: Path, name: str) -> bool:
    from localharness.config.loader import ConfigLoader
    from localharness.plugins.resolve import resolve

    loader = ConfigLoader(config_dir=g)
    resolution = resolve(loader)
    return asyncio.run(plugins_cmd.step_pending(resolution, loader, resolution.plan.entry(name),
                                                plugins_cmd._paths(loader, None)))


@pytest.mark.parametrize("name, overrides, ready, pending", [
    ("plain", {}, False, False),                                   # declares no step
    ("pro", {}, False, True),                                      # configure(): unconfigured
    ("pro", {"pro": {"url": "http://bad"}}, False, True),          # a failing check
    ("pro", {"pro": {"url": "http://ok"}}, False, False),          # a passing check
    ("secty", {}, False, True),                                    # skipped, a question unanswered
    ("secty", {"proposer": {"base_url": "http://p/v1", "model": "p-model"}}, False, False),  # all answered
    ("act", {}, False, True),                                      # an action, its check failing
    ("act", {}, True, False),                                      # an action, its check passing
], ids=["no-step", "unconfigured", "failing", "passing", "skip-unanswered", "skip-answered",
        "action-failing", "action-passing"])
def test_step_pending(g, name, overrides, ready, pending) -> None:
    _write(g, overrides)
    ACT_READY[0] = ready
    assert _pending(g, name) is pending


def test_step_pending_for_settings_that_do_not_validate_is_true(g) -> None:
    """Its settings are refused at load, so it has no settings to build a check from: not set up."""
    _write(g, {"pro": {"url": 3}})
    assert _pending(g, "pro") is True


# --- in-session mode (the /plugins restart, 52-06) ---------------------------------------------------


def _in_session(g: Path, name: str, on: bool = True):
    return plugins_cmd._switch(name, on, [], False, str(g), ask=True, in_session=True)


def test_in_session_a_configured_plugin_is_not_asked_again(g, term, capsys) -> None:
    _write(g, {"pro": {"url": "http://ok"}})
    outcome = _in_session(g, "pro")
    out = capsys.readouterr().out

    assert term.asked == [] and term.confirmed == []
    assert yaml.safe_load((g / "overrides.yaml").read_text()) == {"pro": {"url": "http://ok", "enabled": True}}
    assert f"✓ pro enabled in {g / 'overrides.yaml'}" in out.splitlines()
    assert "takes effect on the next" not in out
    _in_order(out, "Checking it now:", "✓ pro: answers at http://ok")
    assert outcome == plugins_cmd.StepOutcome(name="pro", on=True, failed_check="", stopped=False)


def test_in_session_an_unconfigured_plugin_is_asked_as_the_shell_asks(g, term, capsys) -> None:
    term.answers[:] = ["http://ok"]
    outcome = _in_session(g, "pro")

    assert term.asked == [("Server address", "http://127.0.0.1:1")]
    assert yaml.safe_load((g / "overrides.yaml").read_text()) == {"pro": {"enabled": True, "url": "http://ok"}}
    assert outcome.failed_check == "" and not outcome.stopped


def test_in_session_a_failing_check_is_returned(g, term, capsys) -> None:
    term.answers[:] = ["http://bad"]
    outcome = _in_session(g, "pro")

    assert outcome == plugins_cmd.StepOutcome(name="pro", on=True, failed_check="no answer at http://bad")
    assert AGENT_PROMPT_LEAD in capsys.readouterr().out


def test_in_session_a_configured_action_is_not_asked_or_run(g, term, capsys) -> None:
    ACT_READY[0] = True
    outcome = _in_session(g, "act")

    assert term.confirmed == [] and CALLS == []
    assert outcome.failed_check == ""


def test_the_shell_still_asks_every_question_with_the_stored_value(g, term) -> None:
    _write(g, {"pro": {"url": "http://ok"}})
    term.answers[:] = ["http://ok"]
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    assert term.asked == [("Server address", "http://ok")]
    assert "— takes effect on the next `localharness start`" in result.output


def test_the_shell_runs_no_action_for_a_configured_plugin_with_nothing_typed(g, term) -> None:
    ACT_READY[0] = True
    result = _enable(g, "act")

    assert result.exit_code == 0, result.output
    assert term.confirmed == [] and CALLS == []
    _in_order(result.output, "Checking it now:", "✓ act: ready")


def test_answers_written_this_run_run_the_action_of_a_configured_plugin(g, term) -> None:
    _write(g, {"kit": {"url": "http://ok"}})
    term.answers[:] = ["http://ok"]
    term.yes[:] = [True]
    result = _enable(g, "kit")

    assert result.exit_code == 0, result.output
    assert term.confirmed == [("Warm it up now?", True)] and CALLS == ["kit"]


def test_in_session_a_configured_plugin_with_questions_and_an_action_does_neither(g, term, capsys) -> None:
    _write(g, {"kit": {"url": "http://ok"}})
    outcome = _in_session(g, "kit")

    assert term.asked == [] and term.confirmed == [] and CALLS == []
    assert outcome.failed_check == ""


# --- session_step: never raises out of the restart ---------------------------------------------------


def test_session_step_ctrl_c_at_a_question_stops_the_step(g, monkeypatch, capsys) -> None:
    def prompt(*a, **kw):
        raise click.exceptions.Abort()
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    outcome = plugins_cmd.session_step(("enable", "pro"), str(g))

    assert "Stopped. Your conversation continues." in capsys.readouterr().out
    assert outcome == plugins_cmd.StepOutcome(name="pro", on=True, stopped=True)
    assert not (g / "overrides.yaml").exists()


def test_session_step_a_refusal_stops_the_step(g, term, capsys) -> None:
    term.answers[:] = ["http://p/v1", ""]  # half a proposer: refused, exit 2
    outcome = plugins_cmd.session_step(("enable", "secty"), str(g))

    captured = capsys.readouterr()
    assert "proposer.model: Field required" in " ".join(captured.err.split())
    assert "Stopped. Your conversation continues." in captured.out
    assert outcome == plugins_cmd.StepOutcome(name="secty", on=True, stopped=True)


def test_session_step_an_error_after_the_teardown_is_one_line(g, term, monkeypatch, capsys) -> None:
    _write(g, {"pro": {"url": "http://ok"}})

    def disk_full(path, data):
        raise OSError("disk full")
    monkeypatch.setattr(plugins_cmd, "atomic_write_overlay", disk_full)
    outcome = plugins_cmd.session_step(("enable", "pro"), str(g))

    assert "pro's setup hit an error: OSError: disk full. Your conversation continues." in capsys.readouterr().out
    assert outcome == plugins_cmd.StepOutcome(name="pro", on=True, stopped=True)


def test_session_step_disable_writes_the_switch_off(g, term, capsys) -> None:
    _write(g, {"pro": {"url": "http://ok", "enabled": True}})
    outcome = plugins_cmd.session_step(("disable", "pro"), str(g))

    assert yaml.safe_load((g / "overrides.yaml").read_text())["pro"]["enabled"] is False
    assert outcome == plugins_cmd.StepOutcome(name="pro", on=False)
    assert term.asked == [] and term.confirmed == []
