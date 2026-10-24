"""`plugins enable NAME` asks a plugin's declared setup questions — on a terminal, and only there.

A plugin declares what to ask as DATA (`PluginManifest.setup`, a tuple of `SetupField`) plus a few
lines of `setup_help`; the harness asks, writes the answers through the same checked, atomic overlay
path `--set` uses, then runs the plugin's own doctor check once and prints it with doctor's own row
printer. Off a terminal (or with --no-input, or --workspace) enable writes the switch only and names
the `--set` spelling as the next step.

`stub` is a bundled plugin swapped into BUILTIN_PLUGINS; every run passes an explicit --config-dir
(no workspace discovery, never the ambient home) except the --workspace case, which uses the fake
home + project layout from test_doctor_layer_report.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd, update_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery, lifecycle, resolve
from localharness.plugins.api import GLOBAL_ONLY, Check, Plugin, PluginManifest, SetupField
from tests.plugin_prompts import fake_prompts
from tests.unit.test_doctor_layer_report import _layout
from tests.unit.test_extra_install import dist  # noqa: F401
from tests.unit.test_plugins_pass import Mob

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
_CONFIG = {  # port 9 (discard): doctor's endpoint probe fails fast and never reaches a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}


class StubConfig(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)


class Stub(Plugin):
    """answers stub questions"""

    manifest = PluginManifest(
        name="stub", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),),
        setup_help="HELP-LINE-1\nHELP-LINE-2")
    ConfigModel = StubConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "stub.url")

    def doctor(self, ctx):
        url = ctx.config.url
        if url == "http://ok":
            return [Check(name="stub", status="pass", detail=f"answers at {url}")]
        return [Check(name="stub", status="fail", detail=f"no answer at {url}", hint="start it")]


class ProConfig(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)

    @field_validator("url")
    @classmethod
    def _not_bad(cls, value: str) -> str:
        if "bad" in value:
            raise ValueError(f"cannot reach {value!r}")
        return value


class Pro(Plugin):
    """answers at an address, and says so in help"""

    manifest = PluginManifest(
        name="pro", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),),
        setup_help="PRO-HELP", agent_prompt="Run it at {url}. Done when it answers.",
        next_steps="Then: start it.")
    ConfigModel = ProConfig

    def doctor(self, ctx):
        return [Check(name="pro", status="pass", detail=f"at {ctx.config.url}")]


class Plain(Plugin):
    """has nothing to ask"""

    manifest = PluginManifest(name="plain", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = StubConfig


class TwoConfig(BaseModel):
    url: str = ""
    label: str = ""


class Two(Plugin):
    """asks two questions, one with nothing to offer"""

    manifest = PluginManifest(
        name="two", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),
               SetupField(key="label", prompt="Label")))
    ConfigModel = TwoConfig


class Needy(Plugin):
    """needs an install extra nobody has"""

    manifest = PluginManifest(
        name="needy", version="0.1.0", kind="tools", enabled_by_default=True,
        requires_extra="nosuchextra",
        setup=(SetupField(key="url", prompt="Needy address", default="http://127.0.0.1:2"),))
    ConfigModel = StubConfig


class Secty(Plugin):
    """sets two core settings under the section it claims"""

    manifest = PluginManifest(
        name="secty", version="0.1.0", kind="dev", sections=("proposer",),
        setup=(SetupField(key="proposer.base_url", prompt="Proposer address"),
               SetupField(key="proposer.model", prompt="Proposer model")))


@pytest.fixture(autouse=True)
def bundled(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Plain, Two, Needy, Secty, Pro))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


@pytest.fixture
def prompts(monkeypatch):
    """A terminal whose person answers `answer[0]`; every prompt is recorded."""
    calls: list[tuple] = []
    answer = ["http://ok"]

    def prompt(text, default=None, **kw):
        calls.append((text, default))
        return answer[0]

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    fake_prompts(monkeypatch, prompt=prompt)
    return calls, answer


@pytest.fixture
def no_prompt(monkeypatch):
    def prompt(*a, **kw):
        raise AssertionError("prompted")
    fake_prompts(monkeypatch, prompt=prompt)


@pytest.fixture
def scripted(monkeypatch):
    """A terminal whose person types `answers` in order; every question is recorded as
    (text, default, the other keyword arguments)."""
    calls: list[tuple] = []
    answers: list[str] = []

    def prompt(text, default=None, **kw):
        calls.append((text, default, kw))
        return answers.pop(0)

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    fake_prompts(monkeypatch, prompt=prompt)
    return calls, answers


def _enable(g: Path, *args: str):
    return runner.invoke(app, ["plugins", "enable", *args, "--config-dir", str(g)])


def _overrides(g: Path):
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def _stub_line(output: str) -> str:
    return next(line for line in output.splitlines() if "stub:" in line)


NEXT = ("next step — give it the Server address: "
        "localharness plugins enable stub --set url=http://127.0.0.1:1")


def test_on_a_terminal_enable_asks_writes_and_checks(g, prompts) -> None:
    calls, _ = prompts
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert calls == [("Server address", "http://127.0.0.1:1")]
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://ok"}}
    assert "Checking it now:" in result.output
    assert "✓ stub: answers at http://ok" in result.output
    assert "HELP-LINE-1" not in result.output


def test_a_failing_check_prints_its_hint_and_the_setup_help_and_still_enables(g, prompts) -> None:
    prompts[1][0] = "http://bad"
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert "✗ stub: no answer at http://bad" in result.output
    assert "start it" in result.output
    assert "HELP-LINE-1" in result.output and "HELP-LINE-2" in result.output
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://bad"}}


def test_the_probe_line_is_the_line_doctor_prints(g, prompts) -> None:
    prompts[1][0] = "http://bad"
    enabled = _enable(g, "stub")
    doctor = runner.invoke(app, ["doctor", "--config-dir", str(g)])

    assert _stub_line(enabled.output) == _stub_line(doctor.output)


def test_without_a_terminal_enable_writes_the_switch_and_names_the_set_spelling(g, no_prompt) -> None:
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True}}
    assert NEXT in result.output


def test_no_input_on_a_terminal_is_the_same_as_no_terminal(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "stub", "--no-input")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True}}
    assert NEXT in result.output


def test_explicit_set_values_win_over_asking(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "stub", "--set", "url=http://ok")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://ok"}}
    assert "next step" not in result.output


def test_workspace_never_asks(tmp_path, monkeypatch, fake_home, no_prompt) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)

    result = runner.invoke(app, ["plugins", "enable", "stub", "--workspace"])

    assert result.exit_code == 0, result.output
    assert yaml.safe_load((layout.ws_dir / "overrides.yaml").read_text()) == {"stub": {"enabled": True}}


def test_a_plugin_without_setup_fields_is_unchanged(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "plain")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"plain": {"enabled": True}}
    assert "next step" not in result.output and "Checking it now" not in result.output


def test_info_shows_the_set_spelling(g) -> None:
    text = runner.invoke(app, ["plugins", "info", "stub", "--config-dir", str(g)]).output
    assert "set up:" in text
    assert "localharness plugins enable stub --set url=http://127.0.0.1:1" in text

    data = json.loads(runner.invoke(app, ["plugins", "info", "stub", "--json",
                                          "--config-dir", str(g)]).stdout)
    assert data["setup_command"] == "localharness plugins enable stub --set url=http://127.0.0.1:1"
    plain = json.loads(runner.invoke(app, ["plugins", "info", "plain", "--json",
                                           "--config-dir", str(g)]).stdout)
    assert plain["setup_command"] is None


def test_setup_is_optional_and_frozen() -> None:
    m = PluginManifest(name="x", version="1", kind="tools")
    assert m.setup == () and m.setup_help == ""
    field = SetupField(key="url", prompt="Server address")
    assert field.default == ""
    with pytest.raises(ValidationError):
        field.key = "other"


# --- the step's asking rules (52-03) ---------------------------------------------------------------


def test_enter_on_a_question_with_nothing_to_offer_writes_nothing(g, scripted) -> None:
    calls, answers = scripted
    answers[:] = ["http://ok", ""]
    result = _enable(g, "two")

    assert result.exit_code == 0, result.output
    assert [(text, default, kw["show_default"]) for text, default, kw in calls] == [
        ("Server address", "http://127.0.0.1:1", True), ("Label", "", False)]
    assert _overrides(g) == {"two": {"enabled": True, "url": "http://ok"}}


def test_a_question_offers_the_value_stored_now(g, scripted) -> None:
    calls, answers = scripted
    assert _enable(g, "stub", "--set", "url=http://saved").exit_code == 0
    answers[:] = ["http://ok"]
    result = _enable(g, "stub")

    assert result.exit_code == 0, result.output
    assert [(text, default) for text, default, _ in calls] == [("Server address", "http://saved")]


NEEDY_NOTE = "needy is missing its install extra — install `localharness[nosuchextra]` to use it"


def test_needy_on_a_terminal_asks_nothing_and_names_its_extra(g, monkeypatch, no_prompt) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "needy")

    assert result.exit_code == 0, result.output
    assert NEEDY_NOTE in result.output
    assert _overrides(g) == {"needy": {"enabled": True}}
    assert "Checking it now" not in result.output and "next step" not in result.output


def test_needy_turned_off_still_skips_its_questions(g, monkeypatch, no_prompt) -> None:
    """Turned off, its plan state is `off`, not `needs-extra`: the extra is tested directly."""
    (g / "overrides.yaml").write_text(yaml.safe_dump({"needy": {"enabled": False}}), encoding="utf-8")
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "needy")

    assert result.exit_code == 0, result.output
    assert NEEDY_NOTE in result.output
    assert _overrides(g) == {"needy": {"enabled": True}}


@pytest.fixture
def installable(monkeypatch, dist):  # noqa: F811
    """mob needs an extra this distribution provides and the venv lacks; any spawn or question fails the test."""
    def forbidden(*a, **kw):
        raise AssertionError(f"spawned or asked: {a}")
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Mob,))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda extra: False)
    fake_prompts(monkeypatch, prompt=forbidden, confirm=forbidden)
    assert update_cmd.install_route("mobile").argv is not None


def test_the_shell_never_installs_a_missing_extra_it_could_install(g, monkeypatch, installable) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "mob")

    assert result.exit_code == 0, result.output
    assert "mob is missing its install extra — install `localharness[mobile]` to use it" in result.output
    assert _overrides(g) == {"mob": {"enabled": True}}
    assert "Checking it now" not in result.output and "installing" not in result.output


def test_the_shell_without_a_terminal_spawns_nothing_for_an_installable_extra(g, monkeypatch, installable) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: False)
    result = _enable(g, "mob")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"mob": {"enabled": True}}


def test_sections_answers_land_in_one_core_write(g, scripted) -> None:
    calls, answers = scripted
    answers[:] = ["http://p/v1", "p-model"]
    result = _enable(g, "secty")

    assert result.exit_code == 0, result.output
    assert [text for text, _, _ in calls] == ["Proposer address", "Proposer model"]
    assert _overrides(g) == {"secty": {"enabled": True},
                             "proposer": {"base_url": "http://p/v1", "model": "p-model"}}
    assert "set proposer.base_url = 'http://p/v1'" in result.output
    assert "set proposer.model = 'p-model'" in result.output


def test_the_main_model_id_at_another_address_is_accepted_and_written(g, scripted) -> None:
    """The proposer may be the main model again, served at the proposer's own address."""
    _, answers = scripted
    answers[:] = ["http://p/v1", "test-model"]  # = provider.default_model
    result = _enable(g, "secty")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"secty": {"enabled": True},
                             "proposer": {"base_url": "http://p/v1", "model": "test-model"}}
    assert "set proposer.model = 'test-model'" in result.output


def test_half_a_sections_answer_writes_nothing(g, scripted) -> None:
    _, answers = scripted
    answers[:] = ["http://p/v1", ""]
    result = _enable(g, "secty")

    assert result.exit_code == 2, result.output
    assert "proposer.model: Field required" in " ".join(result.output.split())
    assert not (g / "overrides.yaml").exists()


def test_sections_set_values_write_the_same_one_overlay(g, no_prompt) -> None:
    result = _enable(g, "secty", "--set", "proposer.base_url=http://p/v1", "--set", "proposer.model=p-model")

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"secty": {"enabled": True},
                             "proposer": {"base_url": "http://p/v1", "model": "p-model"}}


def test_sections_set_values_are_machine_level(tmp_path, monkeypatch, fake_home, no_prompt) -> None:
    layout = _layout(tmp_path, monkeypatch, fake_home)
    result = runner.invoke(app, ["plugins", "enable", "secty", "--workspace", "--set",
                                 "proposer.base_url=http://p/v1", "--set", "proposer.model=p-model"])

    assert result.exit_code == 2, result.output
    assert "machine-level settings for secty" in " ".join(result.output.split())
    assert not (layout.ws_dir / "overrides.yaml").exists()


def test_a_refused_sections_write_never_echoes_a_typed_secret(g, no_prompt) -> None:
    """pydantic's str(ValidationError) carries input_value — for a missing field, the section
    around it, the typed proposer.api_key included — so a refusal must never print it."""
    secret = "SENTINEL-KEY-52"
    result = _enable(g, "secty", "--set", "proposer.base_url=http://p/v1",
                     "--set", f"proposer.api_key={secret}")

    assert result.exit_code == 2, result.output
    assert "proposer.model: Field required" in " ".join(result.output.split())
    for where in (result.stdout, result.stderr, repr(result.exception)):
        assert secret not in where, where
    assert not (g / "overrides.yaml").exists()


def test_a_sections_write_over_a_config_that_cannot_load_is_reported(g, no_prompt) -> None:
    """As `components set` reports it (exit 2), never a traceback."""
    (g / "config.yaml").write_text(yaml.safe_dump(dict(_CONFIG, org={"no_such_key": 1})),
                                   encoding="utf-8")
    result = _enable(g, "secty", "--set", "proposer.base_url=http://p/v1", "--set", "proposer.model=p-model")

    assert result.exit_code == 2, result.output
    assert "Failed to load config" in result.output
    assert not (g / "overrides.yaml").exists()


# --- a check that could not run is a failed check; a credential is stripped where it shows ----------


def _doctor_raises(monkeypatch, g: Path, message: str = "boom") -> None:
    real = lifecycle._doctor_row

    async def doctor_row(*a, **kw):
        if (g / "overrides.yaml").exists():
            raise RuntimeError(message)
        return await real(*a, **kw)
    monkeypatch.setattr(lifecycle, "_doctor_row", doctor_row)


def _second_resolve_raises(monkeypatch) -> None:
    real, calls = resolve.resolve, []

    def resolve_twice(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a, **kw)
    monkeypatch.setattr(resolve, "resolve", resolve_twice)


def test_a_check_that_raises_is_a_failed_row_with_the_help_and_the_prompt(g, prompts, monkeypatch) -> None:
    _doctor_raises(monkeypatch, g)
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    row = "✗ pro: could not check it now: RuntimeError: boom"
    assert (result.output.index("Checking it now:") < result.output.index(row)
            < result.output.index("PRO-HELP") < result.output.index("Run it at http://ok."))
    assert _overrides(g) == {"pro": {"enabled": True, "url": "http://ok"}}


def test_a_resolve_that_raises_after_the_write_is_the_same_failed_row(g, prompts, monkeypatch) -> None:
    _second_resolve_raises(monkeypatch)
    result = _enable(g, "pro")

    assert result.exit_code == 0, result.output
    assert result.output.count("could not check it now") == 1
    assert (result.output.index("Checking it now:")
            < result.output.index("✗ pro: could not check it now: RuntimeError: boom")
            < result.output.index("PRO-HELP") < result.output.index("Then: start it."))


def test_could_not_check_is_a_failed_row_with_the_credential_stripped() -> None:
    row = plugins_cmd.could_not_check("x", RuntimeError("cannot reach https://u:pw@h.test/x"))

    assert row.state == "failed"
    assert row.detail.startswith("could not check it now: RuntimeError: ")
    assert "https://h.test/x" in row.detail and "u:pw" not in row.detail


def test_a_refusal_quoting_a_credentialed_url_prints_it_stripped(g, no_prompt) -> None:
    result = _enable(g, "pro", "--set", "url=https://user:pw@bad.test")

    assert result.exit_code == 2, result.output
    assert "https://bad.test" in result.stderr and "user:pw" not in result.stderr + result.stdout
    assert not (g / "overrides.yaml").exists()


def test_the_set_receipt_strips_a_credential_and_the_overlay_keeps_it(g, no_prompt) -> None:
    result = _enable(g, "stub", "--set", "url=https://user:pw@ok.test")

    assert result.exit_code == 0, result.output
    assert "  set stub.url = 'https://ok.test'" in result.output and "user:pw" not in result.output
    assert _overrides(g)["stub"]["url"] == "https://user:pw@ok.test"
    assert "  set stub.url = 'http://ok'" in _enable(g, "stub", "--set", "url=http://ok").output


# --- the phases and the one printer ------------------------------------------------------------------


def test_shell_report_sanitizes_plugin_text_and_prints_it_as_text(capsys) -> None:
    plugins_cmd.shell_report("help", ("Use [browser] and [/x]", "see \x1b]8;;http://x\x1b\\link\x1b]8;;\x1b\\ here"))
    plugins_cmd.shell_report("note", "a [b] \x1b[2Jc")
    plugins_cmd.shell_report("next_steps", (True, ("run [x]", "then \x1b]0;title\x07done")))
    plugins_cmd.shell_report("wrote", ("stub enabled in /t", (("stub.url", "'http://ok'"),)))

    assert capsys.readouterr().out == (
        "\nUse [browser] and [/x]\n\nsee link here\n  note: a [b] c\n\n  run [x]\n  then done\n"
        "✓ stub enabled in /t\n  set stub.url = 'http://ok'\n")


def _state(g: Path, name: str, **kw):
    resolution, loader, workspace = plugins_cmd._resolve(str(g), json_output=False)
    entry = plugins_cmd._entry(resolution, loader, name, False)
    args = dict(pairs=(), to_workspace=False, ask=True, in_session=True) | kw
    return plugins_cmd.PassState(name, True, args["pairs"], args["to_workspace"], args["ask"],
                                 args["in_session"], resolution, loader, workspace, entry)


async def test_the_phases_run_in_a_running_loop_and_report_in_print_order(g) -> None:
    reports, asked = [], []

    async def prompt(text, default="", secret=False):
        asked.append(text)
        return "http://ok"

    async def confirm(text, default):
        raise AssertionError("confirmed")

    done = await plugins_cmd._run_pass(_state(g, "stub"), prompt=prompt, confirm=confirm,
                                       report=lambda kind, payload: reports.append(kind))

    assert asked == ["Server address: "]
    assert reports == ["wrote", "checking", "row"]
    assert done.row.state == "on" and done.failed == "" and not done.declined
    assert _overrides(g) == {"stub": {"enabled": True, "url": "http://ok"}}


async def test_a_refusal_leaves_the_phases_unreported(g) -> None:
    reports = []

    async def never(*a, **kw):
        raise AssertionError("asked")

    with pytest.raises(plugins_cmd.Refusal, match="no setting 'nokey'"):
        await plugins_cmd._run_pass(_state(g, "stub", pairs=("nokey=1",), ask=False), prompt=never,
                                    confirm=never, report=lambda kind, payload: reports.append(kind))

    assert reports == [] and not (g / "overrides.yaml").exists()


def test_a_plugin_with_nothing_to_set_up_prints_only_the_enabled_line(g, no_prompt, monkeypatch) -> None:
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    result = _enable(g, "plain")

    assert [line[0] for line in result.output.splitlines()] == ["✓"]


def test_a_check_detail_is_sanitized_on_its_way_to_failed_check(g, prompts, monkeypatch) -> None:
    _doctor_raises(monkeypatch, g, "x\x1b[2Jy\x1b]8;;http://e\x1b\\z")
    outcome = plugins_cmd.session_step(("enable", "pro"), str(g))

    assert outcome.failed_check == "could not check it now: RuntimeError: xyz"
