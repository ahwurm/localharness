"""The project override (its one predicate and its one sentence) and the enable and disable producers."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel, SecretStr

from localharness.cli import plugins_cmd
from localharness.cli.plugins_cmd import (
    CTRLC, CTRLC_DOES, OVERRIDE_LINE, OVERRIDE_SAVED, THIRD_PARTY_CONFIRM, PassProgress, PassResult, PluginSession,
    Refusal, StepOutcome, Touch, disable_pass, enable_pass, override_line, override_of,
)
from localharness.config.loader import ConfigLoader
from localharness.core.bus import EventBus
from localharness.plugins import builtin, discovery, lifecycle
from localharness.plugins import resolve as resolve_module
from localharness.plugins.api import Check, Plugin, PluginManifest, SetupField
from localharness.plugins.resolve import resolve
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_plugins_bookkeeping import _CONFIG, Stub

_REAL_DISCOVER = discovery.discover


class _AskConfig(BaseModel):
    url: str = ""
    token: SecretStr = SecretStr("")


class Ask(Plugin):
    """asks an address and a token"""

    manifest = PluginManifest(
        name="ask", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Address", default="http://127.0.0.1:1"), SetupField(key="token", prompt="Token", secret=True)),
        setup_help="ASK-HELP", agent_prompt="Run it at {url}.", next_steps="ASK-NEXT")
    ConfigModel = _AskConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "ask.url")

    def doctor(self, ctx):
        url = ctx.config.url
        status = "pass" if url.endswith("-ok") else "skip" if url.endswith("-skip") else "fail"
        return [Check(name="ask", status=status, detail={"pass": "answers", "skip": "not paired"}.get(status, "no answer"))]


@pytest.fixture
def layers(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Ask))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g, ws = tmp_path / "g", tmp_path / "proj" / ".localharness"
    g.mkdir()
    ws.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return SimpleNamespace(g=g, ws=ws)


def _write(path: Path, enabled) -> None:
    path.write_text(yaml.safe_dump({"stub": {"enabled": enabled}}), encoding="utf-8")


def _of(layers, on: bool, *, to_workspace: bool = False, workspace: bool = True):
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws if workspace else None)
    return override_of(loader, resolve(loader).plan.entry("stub"), on, to_workspace)


def test_override_of_names_the_overrides_file_that_turns_it_off(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True) == ("off", layers.ws / "overrides.yaml")
    assert _of(layers, False) is None


def test_override_of_names_the_config_file_when_only_it_holds_the_switch(layers) -> None:
    _write(layers.ws / "config.yaml", True)

    assert _of(layers, False) == ("on", layers.ws / "config.yaml")
    assert _of(layers, True) is None


def test_override_of_the_overrides_file_outranks_the_config_file(layers) -> None:
    _write(layers.ws / "config.yaml", True)
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True) == ("off", layers.ws / "overrides.yaml")
    assert _of(layers, False) is None


def test_override_of_a_later_layer_that_agrees_leaves_nothing_to_say(layers) -> None:
    _write(layers.ws / "config.yaml", False)
    _write(layers.ws / "overrides.yaml", True)

    assert _of(layers, True) is None
    assert _of(layers, False) == ("on", layers.ws / "overrides.yaml")


def test_override_of_ignores_a_value_that_is_not_a_bool(layers) -> None:
    _write(layers.ws / "overrides.yaml", "no")

    assert _of(layers, True) is None


def test_override_of_ignores_the_machine_layers(layers) -> None:
    _write(layers.g / "overrides.yaml", False)

    assert _of(layers, True) is None


def test_override_of_is_none_for_a_workspace_write(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)

    assert _of(layers, True, to_workspace=True) is None


def test_override_of_is_none_with_no_workspace(layers) -> None:
    assert _of(layers, True, workspace=False) is None


def test_override_of_is_none_for_a_plugin_you_installed(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws)

    assert override_of(loader, SimpleNamespace(name="stub", bundled=False), True, False) is None


def test_override_line_fills_the_one_template(layers) -> None:
    file = layers.ws / "overrides.yaml"

    assert override_line("stub", ("off", file)) == (
        f"this project turns it off ({file}) and that still wins — "
        "localharness plugins enable stub --workspace changes it")
    assert override_line("stub", ("on", file), OVERRIDE_SAVED) == (
        f"saved on this machine, but this project turns it on ({file}) and that still wins — "
        "localharness plugins disable stub --workspace changes it")


def test_the_template_and_the_saved_prefix() -> None:
    assert OVERRIDE_LINE.startswith("{saved}this project turns it {state} ({file})")
    assert OVERRIDE_SAVED == "saved on this machine, but "


def _session(layers, **kw) -> PluginSession:
    return PluginSession(layers.g, layers.ws, kw.pop("running", frozenset()), kw.pop("unconfigured", {}),
                         kw.pop("failed", {}), **kw)


async def _disable(layers, name: str = "stub", **kw):
    progress, reports = PassProgress(), []
    result = await disable_pass(_session(layers, **kw), name, progress=progress,
                                report=lambda kind, payload: reports.append((kind, payload)))
    return result, progress, reports


def _machine(layers) -> dict:
    return yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8"))


async def test_disable_pass_writes_the_machine_switch_off(layers) -> None:
    _write(layers.g / "overrides.yaml", True)

    result, progress, reports = await _disable(layers)

    assert _machine(layers) == {"stub": {"enabled": False}}
    assert result == PassResult("disabled", StepOutcome("stub", False))
    assert (progress.phase, progress.wrote) == ("disable", True)
    assert reports == [("wrote", (f"stub disabled in {layers.g / 'overrides.yaml'}", ()))]


async def test_disable_pass_under_a_project_that_turns_it_on_is_an_override_that_still_loads(layers) -> None:
    _write(layers.ws / "config.yaml", True)

    result, progress, reports = await _disable(layers)

    assert _machine(layers) == {"stub": {"enabled": False}}
    assert (result.kind, result.override, result.step, result.loads) == (
        "override", ("on", layers.ws / "config.yaml"), StepOutcome("stub", False), True)
    assert progress.wrote and [kind for kind, _ in reports] == ["wrote"]
    assert not (layers.ws / "overrides.yaml").exists()


@pytest.mark.parametrize("kw", [
    {"touched": {"stub": Touch(False, True, ("failed", "x"))}},
    {"failed": {"stub": "import broke"}},
    {"unconfigured": {"stub": "stub.url"}}])
async def test_disable_pass_keeps_a_plugin_that_would_not_load_out_of_loads(layers, kw) -> None:
    _write(layers.ws / "config.yaml", True)

    result, _, _ = await _disable(layers, **kw)

    assert (result.kind, result.loads) == ("override", False)


async def test_disable_pass_publishes_nothing_and_resolves_once(layers, monkeypatch) -> None:
    async def publish(self, event):
        raise AssertionError("published")

    calls = []

    def counting(loader, *a, **kw):
        calls.append(loader)
        return resolve(loader, *a, **kw)

    monkeypatch.setattr(EventBus, "publish", publish)
    monkeypatch.setattr(resolve_module, "resolve", counting)
    _write(layers.ws / "config.yaml", True)

    await _disable(layers)

    assert len(calls) == 1


@pytest.mark.parametrize("name", ["nope", "org"])
async def test_disable_pass_refuses_a_name_the_plan_refused_or_does_not_hold(layers, monkeypatch, name) -> None:
    monkeypatch.setattr(discovery, "discover", _REAL_DISCOVER)
    write_folder_plugin(layers.g, "org")

    with pytest.raises(Refusal):
        await _disable(layers, name)

    assert not (layers.g / "overrides.yaml").exists()


def test_disable_pass_is_a_coroutine_with_no_await_in_it() -> None:
    assert inspect.iscoroutinefunction(disable_pass)
    assert "await" not in inspect.getsource(disable_pass)


class Talk:
    """Answers the questions in order and records what was asked, in which phase, and what was reported."""

    def __init__(self, answers=(), yes=()):
        self.answers, self.yes = list(answers), list(yes)
        self.progress, self.asked, self.confirmed, self.reports, self.seen = PassProgress(), [], [], [], []

    async def prompt(self, text, *, default="", secret=False):
        self.asked.append(text)
        self.seen.append((self.progress.phase, self.progress.written))
        return self.answers.pop(0)

    async def confirm(self, text, *, default, tail=""):
        self.confirmed.append((text, default, tail))
        self.seen.append((self.progress.phase, self.progress.written))
        return self.yes.pop(0)

    def report(self, kind, payload) -> None:
        self.reports.append((kind, payload))

    @property
    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.reports]


async def _enable(layers, name, talk=None, *, session=None, **kw):
    talk = talk or Talk()
    kw.setdefault("mode", "enable")
    result = await enable_pass(session or _session(layers), name, progress=talk.progress, prompt=talk.prompt,
                               confirm=talk.confirm, report=talk.report, **kw)
    return result, talk


def test_enable_pass_is_a_coroutine_with_keyword_only_options() -> None:
    params = inspect.signature(enable_pass).parameters

    assert inspect.iscoroutinefunction(enable_pass)
    assert [p for p in params if params[p].kind is inspect.Parameter.KEYWORD_ONLY] == [
        "mode", "progress", "prompt", "confirm", "report", "install", "in_batch", "console"]
    assert (params["install"].default, params["in_batch"].default) == ("own", False)


async def test_enable_pass_asks_writes_checks_and_says_what_loads(layers) -> None:
    result, talk = await _enable(layers, "ask", Talk(["box-ok", "s3cr3t-typed"]))

    assert (result.kind, result.answered, result.questions, result.loads) == ("enabled", ("url", "token"), "answered", True)
    assert (result.row.state, result.next_steps, result.install, result.help) == ("on", "ASK-NEXT", "none", ())
    assert result.step == StepOutcome("ask", True)
    assert yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8")) == {
        "ask": {"enabled": True, "url": "box-ok", "token": "s3cr3t-typed"}}
    assert talk.seen == [("question", False), ("question", False)]
    assert (talk.progress.written, talk.progress.wrote, talk.progress.answered, talk.progress.rerun) == (
        True, True, True, False)
    assert talk.kinds == ["wrote", "checking", "row", "next_steps"]


async def test_enable_pass_results_hold_key_names_never_a_typed_answer(layers) -> None:
    result, _ = await _enable(layers, "ask", Talk(["box-ok", "s3cr3t-typed"]))
    bad, _ = await _enable(layers, "ask", Talk(["box-bad", "s3cr3t-typed"]))

    for got in (result, bad):
        assert "s3cr3t-typed" not in repr(got)
    assert "box-ok" not in repr(result)


async def test_enable_pass_a_failing_check_carries_the_help_and_the_failing_row_text(layers) -> None:
    result, _ = await _enable(layers, "ask", Talk(["box-bad", "tok"]))

    assert (result.kind, result.loads, result.row.state, result.step.failed_check) == (
        "check_failed", True, "on", "no answer")
    assert result.help[0] == "ASK-HELP" and "box-bad" in result.help[1]


async def test_enable_pass_a_skipped_check_is_check_skipped(layers) -> None:
    result, _ = await _enable(layers, "ask", Talk(["box-skip", "tok"]))

    assert (result.kind, result.loads, result.step.skipped_check, result.step.failed_check) == (
        "check_skipped", True, "not paired", "")


async def test_enable_pass_an_unconfigured_probe_is_enabled_unconfigured_with_the_rows_detail(layers) -> None:
    result, _ = await _enable(layers, "ask", Talk(["", ""]))

    assert (result.kind, result.questions, result.answered, result.loads) == (
        "enabled_unconfigured", "skipped", (), False)
    assert result.detail == result.row.detail == "unconfigured — set ask.url"


async def test_enable_pass_skipped_questions_with_no_probe_row_are_enabled_unconfigured(layers, monkeypatch) -> None:
    async def no_check(s, *, report):
        return s

    monkeypatch.setattr(plugins_cmd, "_check", no_check)

    result, _ = await _enable(layers, "ask", Talk(["", ""]))

    assert (result.kind, result.row, result.detail, result.loads) == (
        "enabled_unconfigured", None, "setup questions skipped", False)


async def test_enable_pass_a_probe_that_raises_is_check_failed_noload(layers, monkeypatch) -> None:
    real = lifecycle._doctor_row

    async def boom(*a, **kw):
        if (layers.g / "overrides.yaml").exists():
            raise RuntimeError("boom")
        return await real(*a, **kw)

    monkeypatch.setattr(lifecycle, "_doctor_row", boom)

    result, _ = await _enable(layers, "ask", Talk(["box-ok", "tok"]))

    assert result.kind == "check_failed_noload" and result.loads is False
    assert result.row.detail == "could not check it now: RuntimeError: boom"
    assert result.detail == result.step.failed_check == result.row.detail


async def test_enable_pass_a_resolve_that_raises_after_the_write_is_the_same_failed_check(layers, monkeypatch) -> None:
    calls = []

    def second_raises(loader, *a, **kw):
        calls.append(loader)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return resolve(loader, *a, **kw)

    monkeypatch.setattr(resolve_module, "resolve", second_raises)

    result, _ = await _enable(layers, "ask", Talk(["box-ok", "tok"]))

    assert (result.kind, result.loads) == ("check_failed_noload", False)
    assert result.row.detail == "could not check it now: RuntimeError: boom" == result.detail
    assert result.help[0] == "ASK-HELP"


async def test_enable_pass_a_plugin_with_nothing_to_set_up_runs_no_probe(layers) -> None:
    result, talk = await _enable(layers, "stub")

    assert (result.kind, result.row, result.loads, result.questions, result.detail) == ("enabled", None, True, "none", "")
    assert talk.kinds == ["wrote"] and (talk.progress.written, talk.progress.wrote, talk.progress.answered) == (
        True, True, False)


async def test_enable_pass_a_plugin_the_build_failed_is_not_loading_with_the_builds_text(layers) -> None:
    result, _ = await _enable(layers, "stub", session=_session(layers, failed={"stub": "import broke"}))

    assert (result.kind, result.detail, result.loads) == ("enabled_not_loading", "import broke", False)
    assert (layers.g / "overrides.yaml").exists()


async def test_enable_pass_a_stored_failed_probe_is_the_text_when_the_build_has_none(layers) -> None:
    touched = {"stub": Touch(False, False, ("failed", "its last check failed"))}

    result, _ = await _enable(layers, "stub", session=_session(layers, touched=touched))

    assert (result.kind, result.detail, result.loads) == ("enabled_not_loading", "its last check failed", False)


@pytest.fixture
def folder(layers, tmp_path, monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda g: [f for f in _REAL_DISCOVER(g) if f.source == "folder"])
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield layers
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


async def test_enable_pass_a_plugin_you_installed_asks_the_trust_question_first(folder) -> None:
    write_folder_plugin(folder.g, "swatch")

    result, talk = await _enable(folder, "swatch", Talk(yes=[True]))

    assert talk.confirmed == [(THIRD_PARTY_CONFIRM.format(name="swatch"), False,
                               CTRLC.format(does=CTRLC_DOES["prompt"]))]
    assert (result.kind, result.loads) == ("enabled", True)
    assert yaml.safe_load((folder.g / "overrides.yaml").read_text(encoding="utf-8")) == {"swatch": {"enabled": True}}


async def test_enable_pass_a_no_to_the_trust_question_writes_nothing(folder) -> None:
    write_folder_plugin(folder.g, "swatch")

    result, talk = await _enable(folder, "swatch", Talk(yes=[False]))

    assert (result.kind, result.loads, result.step.stopped) == ("declined", False, True)
    assert not (folder.g / "overrides.yaml").exists() and talk.progress.wrote is False


async def test_enable_pass_a_plugin_you_installed_that_cannot_be_imported_is_not_loading(folder) -> None:
    write_folder_plugin(folder.g, "swatch", prelude="raise RuntimeError('boom')")

    result, _ = await _enable(folder, "swatch", Talk(yes=[True]))

    assert (result.kind, result.loads) == ("enabled_not_loading", False)
    assert "could not be imported" in result.detail and "boom" in result.detail


async def test_enable_pass_a_plugin_you_installed_that_needs_an_extra_nobody_has_is_not_loading(folder) -> None:
    write_folder_plugin(folder.g, "swatch", manifest=', requires_extra="nosuchextra"')

    result, talk = await _enable(folder, "swatch", Talk(yes=[True]))

    assert (result.kind, result.loads, talk.asked) == ("enabled_not_loading", False, [])
    assert "nosuchextra" in result.detail


async def test_enable_pass_an_overridden_plugin_prints_its_line_and_does_nothing_else(layers) -> None:
    _write(layers.ws / "overrides.yaml", False)

    result, talk = await _enable(layers, "stub")

    assert (result.kind, result.install, result.loads, result.override) == (
        "override", "none", False, ("off", layers.ws / "overrides.yaml"))
    assert talk.kinds == ["note"] and talk.asked == [] and talk.confirmed == []
    assert not (layers.g / "overrides.yaml").exists()
    assert (talk.progress.wrote, talk.progress.phase) == (False, "consent")


async def test_enable_pass_refuses_a_name_the_plan_does_not_hold(layers) -> None:
    with pytest.raises(Refusal):
        await _enable(layers, "nope")

    assert not (layers.g / "overrides.yaml").exists()


async def test_enable_pass_written_starts_from_the_flag_not_from_the_name(layers) -> None:
    (layers.g / "overrides.yaml").write_text(yaml.safe_dump({"ask": {"enabled": True}}), encoding="utf-8")

    _, talk = await _enable(layers, "ask", Talk(["box-ok", "tok"]))

    assert talk.seen[0] == ("question", True)
