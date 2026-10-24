"""The project override (its one predicate and its one sentence) and the enable and disable producers."""
from __future__ import annotations

import asyncio
import dataclasses
import inspect
import io
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import BaseModel, SecretStr, field_validator
from rich.console import Console

from localharness.cli import plugins_cmd
from localharness.cli.plugins_cmd import (
    CONSENT, CTRLC, CTRLC_DOES, CTRLC_IGNORED, OVERRIDE_LINE, OVERRIDE_SAVED, THIRD_PARTY_CONFIRM, PassProgress, PassResult, PluginSession,
    Refusal, StepOutcome, Touch, configure_one, disable_pass, enable_pass, info_renderable, override_line, override_of, settings_rows,
)
from localharness.config.loader import ConfigLoader
from localharness.core.bus import EventBus
from localharness.plugins import builtin, discovery, lifecycle
from localharness.plugins import resolve as resolve_module
from localharness.plugins.api import Check, Plugin, PluginManifest, SetupField
from localharness.plugins.resolve import resolve
from tests.plugin_prompts import fake_prompts
from tests.unit.test_extra_install import SPEC, _Proc, dist  # noqa: F401
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_plugins_bookkeeping import _CONFIG, Stub

_REAL_DISCOVER = discovery.discover
CALLS: list[str] = []
DOWNLOAD = "Download the model (1.2 GB)?"


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
        if ctx.config.url.endswith("-dead"):
            raise RuntimeError("dead")
        return "ready" if ctx.config.url else ("unconfigured", "ask.url")

    def doctor(self, ctx):
        url = ctx.config.url
        status = "pass" if url.endswith("-ok") else "skip" if url.endswith("-skip") else "fail"
        return [Check(name="ask", status=status, detail={"pass": "answers", "skip": "not paired"}.get(status, "no answer"))]


class _MobConfig(BaseModel):
    url: str = ""

    @field_validator("url")
    @classmethod
    def _not_refused(cls, value: str) -> str:
        if "refuse" in value:
            raise ValueError(f"cannot reach {value!r}")
        return value


class Mob(Plugin):
    """needs the mobile extra, asks an address and downloads a model"""

    manifest = PluginManifest(
        name="mob", version="0.1.0", kind="tools", enabled_by_default=False, requires_extra="mobile",
        setup=(SetupField(key="url", prompt="Address", default="http://127.0.0.1:1"),),
        setup_action=DOWNLOAD, next_steps="MOB-NEXT")
    ConfigModel = _MobConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "mob.url")

    def setup_action(self, ctx):
        CALLS.append("download")
        return [Check(name="mob", status="pass", detail="fetched")]

    def doctor(self, ctx):
        return [Check(name="mob", status="pass" if "download" in CALLS else "fail", detail="ready")]


@pytest.fixture
def layers(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Ask, Mob))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    CALLS.clear()
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


class _Tee(io.StringIO):
    def write(self, text: str) -> int:
        CALLS.append(text.strip())
        return super().write(text)


@pytest.fixture
def mobile(monkeypatch):
    """The mobile extra, missing until a faked installer exits 0."""
    state = SimpleNamespace(installed=False, spawns=[], proc=_Proc([b"installed\n"]))

    async def spawn(*argv, **kw):
        state.spawns.append((argv, kw))
        state.installed = True
        return state.proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setitem(resolve_module.resolve.__kwdefaults__, "extra_installed",
                        lambda extra: state.installed or extra != "mobile")
    return state


@pytest.fixture
def route(monkeypatch):
    """A route for mob without the real distribution, and an install_pass that answers as scripted."""
    state = SimpleNamespace(extra="mobile", argv=["pip", "install", "x"], done=True, calls=[])

    async def fake_install(session, extras, names, **kw):
        state.calls.append((extras, names, kw))
        return frozenset(names) if state.done else frozenset()

    monkeypatch.setattr(plugins_cmd, "extra_for", lambda entry: state.extra)
    monkeypatch.setattr(plugins_cmd, "install_route_for", lambda entry: SimpleNamespace(argv=state.argv))
    monkeypatch.setattr(plugins_cmd, "install_pass", fake_install)
    return state


def _written(layers) -> dict:
    return yaml.safe_load((layers.g / "overrides.yaml").read_text(encoding="utf-8"))


async def test_enable_pass_installs_the_missing_extra_after_one_consent_and_asks_as_question_installed(
        layers, mobile, dist) -> None:  # noqa: F811
    out = io.StringIO()

    result, talk = await _enable(layers, "mob", Talk(["http://box"], yes=[True]), console=Console(file=out, width=400))

    assert (result.kind, result.install, result.downloaded, result.next_steps, result.loads) == (
        "enabled", "done", True, "MOB-NEXT", True)
    assert CALLS == ["download"]
    assert len(talk.confirmed) == 1 and talk.confirmed[0][0] == "" and talk.confirmed[0][1] is True
    assert DOWNLOAD not in [text for text, *_ in talk.confirmed]
    assert talk.seen == [("consent", False), ("question_installed", False)]
    assert talk.progress.phase == "check" and talk.progress.written is True
    assert len(mobile.spawns) == 1 and mobile.spawns[0][0][-1] == SPEC
    assert CONSENT[0].format(names="mob") in out.getvalue() and DOWNLOAD in out.getvalue()
    assert _written(layers) == {"mob": {"enabled": True, "url": "http://box"}}


async def test_enable_pass_a_declined_install_writes_the_enable_and_stops(layers, mobile, route) -> None:
    route.done = False

    result, talk = await _enable(layers, "mob")

    assert (result.kind, result.install, result.loads, result.downloaded, result.questions) == (
        "install_skipped", "skipped", False, False, "none")
    assert len(route.calls) == 1 and route.calls[0][:2] == ({"mobile"}, ["mob"])
    assert talk.asked == [] and talk.confirmed == [] and CALLS == []
    assert _written(layers) == {"mob": {"enabled": True}} and talk.progress.wrote is True


async def test_enable_pass_install_skipped_by_a_batch_does_not_run_the_install(layers, mobile, route) -> None:
    result, talk = await _enable(layers, "mob", install="skipped")

    assert (result.kind, result.install, result.loads) == ("install_skipped", "skipped", False)
    assert route.calls == [] and talk.asked == [] and talk.confirmed == []
    assert _written(layers) == {"mob": {"enabled": True}}


async def test_enable_pass_a_manual_install_route_writes_the_enable_and_nothing_else(layers, mobile, route) -> None:
    route.argv = None

    result, talk = await _enable(layers, "mob")

    assert (result.kind, result.install, result.loads, result.questions) == ("enabled_manual", "none", False, "none")
    assert route.calls == [] and talk.asked == [] and talk.confirmed == [] and CALLS == []
    assert _written(layers) == {"mob": {"enabled": True}}


async def test_enable_pass_a_missing_extra_with_no_route_is_not_loading(layers, mobile, route) -> None:
    route.extra = None

    result, talk = await _enable(layers, "mob")

    assert (result.kind, result.loads, result.install, result.detail) == (
        "enabled_not_loading", False, "none", "install `localharness[mobile]` to use it")
    assert route.calls == [] and talk.asked == [] and talk.confirmed == []
    assert _written(layers) == {"mob": {"enabled": True}}


async def test_enable_pass_a_resolve_that_raises_after_the_install_is_refused_with_nothing_written(
        layers, mobile, route, monkeypatch) -> None:
    calls = []

    def second_raises(loader, *a, **kw):
        calls.append(loader)
        if len(calls) == 2:
            raise RuntimeError("boom http://user:pw@host/x")
        return resolve(loader, *a, **kw)

    monkeypatch.setattr(resolve_module, "resolve", second_raises)

    result, talk = await _enable(layers, "mob")

    assert (result.kind, result.install, result.loads) == ("refused", "done", False)
    assert result.detail.startswith("RuntimeError: boom") and "pw" not in result.detail
    assert not (layers.g / "overrides.yaml").exists() and talk.progress.wrote is False
    assert talk.progress.phase == "question_installed"


async def test_enable_pass_install_done_runs_every_question_as_question_installed_and_skips_the_download_confirm(
        layers, mobile) -> None:
    mobile.installed = True

    result, talk = await _enable(layers, "mob", Talk(["http://box"]), install="done")

    assert (result.kind, result.install, result.downloaded) == ("enabled", "done", True)
    assert talk.seen == [("question_installed", False)] and talk.confirmed == [] and CALLS == ["download"]
    assert talk.progress.phase == "check"


async def test_enable_pass_install_done_a_refused_answer_leaves_the_phase_at_question_installed(
        layers, mobile) -> None:
    mobile.installed = True
    talk = Talk(["http://refuse"])

    with pytest.raises(Refusal):
        await _enable(layers, "mob", talk, install="done")

    assert talk.progress.phase == "question_installed" and not (layers.g / "overrides.yaml").exists()


async def test_enable_pass_install_none_asks_the_plugins_own_download_confirm_in_the_download_phase(
        layers, mobile) -> None:
    mobile.installed = True

    result, talk = await _enable(layers, "mob", Talk(["http://box"], yes=[True]), install="none")

    assert (result.install, result.downloaded, CALLS) == ("none", True, ["download"])
    assert talk.confirmed == [(DOWNLOAD, True, "")]
    assert talk.seen == [("question", False), ("download", True)]


async def test_enable_pass_a_no_to_the_download_confirm_downloads_nothing(layers, mobile) -> None:
    mobile.installed = True

    result, talk = await _enable(layers, "mob", Talk(["http://box"], yes=[False]), install="none")

    assert (result.downloaded, CALLS, talk.progress.phase) == (False, [], "check")


async def test_enable_pass_in_batch_puts_the_batch_tail_on_the_download_confirm(layers, mobile) -> None:
    mobile.installed = True

    _, talk = await _enable(layers, "mob", Talk(["http://box"], yes=[True]), install="none", in_batch=True)

    assert talk.confirmed == [(DOWNLOAD, True, "Ctrl-C stops the batch")]


async def test_enable_pass_in_batch_puts_the_batch_tail_on_the_trust_question(folder) -> None:
    write_folder_plugin(folder.g, "swatch")

    _, talk = await _enable(folder, "swatch", Talk(yes=[True]), in_batch=True)

    assert talk.confirmed == [(THIRD_PARTY_CONFIRM.format(name="swatch"), False, "Ctrl-C stops the batch")]


async def test_enable_pass_in_an_ignore_launch_says_the_download_cannot_be_stopped_before_it_starts(
        layers, mobile, monkeypatch) -> None:
    mobile.installed = True
    monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
    line = CTRLC_IGNORED.format(step="download")

    await _enable(layers, "mob", Talk(["http://box"], yes=[True]), install="none", console=Console(file=_Tee(), width=400))

    assert CALLS.count(line) == 1 and CALLS.index(line) < CALLS.index("download")


async def test_enable_pass_prints_no_ignored_line_when_sigint_is_not_ignored_or_the_download_is_declined(
        layers, mobile, monkeypatch) -> None:
    mobile.installed = True
    out = io.StringIO()

    await _enable(layers, "mob", Talk(["http://box"], yes=[True]), install="none", console=Console(file=out, width=400))
    monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
    await _enable(layers, "mob", Talk(["http://box"], yes=[False]), install="none", console=Console(file=out, width=400))

    assert out.getvalue() == ""


def test_the_shell_step_never_prints_the_ignored_line(layers, mobile, monkeypatch) -> None:
    mobile.installed = True
    out = io.StringIO()
    monkeypatch.setattr(plugins_cmd, "console", Console(file=out, width=400))
    monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_IGN)
    fake_prompts(monkeypatch, prompt=lambda text, **kw: "http://box", confirm=lambda text, **kw: True)

    plugins_cmd._switch("mob", True, [], False, str(layers.g), ask=True, in_session=True)

    assert CTRLC_IGNORED.format(step="download") not in out.getvalue() and CALLS == ["download"]


def _on(layers, name: str = "ask", **values) -> bytes:
    path = layers.g / "overrides.yaml"
    path.write_text(yaml.safe_dump({name: {"enabled": True, **values}}), encoding="utf-8")
    return path.read_bytes()


async def _rerun(layers, name, talk=None, **kw):
    return await _enable(layers, name, talk, mode="rerun", **kw)


async def test_rerun_a_healthy_running_plugin_asks_its_own_download_confirm_and_probes(layers, mobile) -> None:
    mobile.installed = True
    CALLS.append("download")
    before = _on(layers, "mob", url="http://box")

    result, talk = await _rerun(layers, "mob", Talk(yes=[True]), session=_session(layers, running=frozenset({"mob"})))

    assert (result.kind, result.rebind, result.loads, result.questions, result.downloaded) == (
        "rerun_ok", False, True, "none", True)
    assert talk.confirmed == [(DOWNLOAD, True, "")] and talk.asked == []
    assert talk.seen == [("download", True)] and talk.progress.phase == "check" and talk.progress.rerun is True
    assert (layers.g / "overrides.yaml").read_bytes() == before and talk.progress.wrote is False
    assert CALLS == ["download", "download"] and result.row.state == "on"


async def test_rerun_asks_what_is_pending_and_writes_the_answers_never_the_switch(layers) -> None:
    (layers.ws / "config.yaml").write_text(yaml.safe_dump({"ask": {"enabled": True}}), encoding="utf-8")

    result, talk = await _rerun(layers, "ask", Talk(["box-ok", "tok"]), session=_session(layers, running=frozenset({"ask"})))

    assert (result.kind, result.answered, result.questions, result.rebind, result.loads) == (
        "rerun_ok", ("url", "token"), "answered", True, True)
    assert _written(layers) == {"ask": {"url": "box-ok", "token": "tok"}}
    assert talk.seen == [("question", True), ("question", True)]
    assert (talk.progress.wrote, talk.progress.answered, talk.progress.written) == (True, True, True)
    assert talk.kinds == ["wrote", "checking", "row", "next_steps"]


async def test_rerun_asks_no_trust_question_and_no_consent_for_a_plugin_you_installed(folder) -> None:
    write_folder_plugin(folder.g, "swatch")
    _on(folder, "swatch")

    result, talk = await _rerun(folder, "swatch", session=_session(folder, running=frozenset({"swatch"})))

    assert (result.kind, talk.confirmed, talk.asked) == ("rerun_ok", [], [])


async def test_rerun_failed_names_the_failing_check_and_is_no_rebind_when_nothing_was_written(layers) -> None:
    _on(layers, url="box-bad")

    result, _ = await _rerun(layers, "ask", Talk(["", ""]), session=_session(layers, running=frozenset({"ask"})))

    assert (result.kind, result.questions, result.step.failed_check, result.rebind, result.loads) == (
        "rerun_failed", "skipped", "no answer", False, True)
    assert result.help[0] == "ASK-HELP"


async def test_rerun_failed_after_writing_answers_on_a_running_plugin_is_a_rebind(layers) -> None:
    _on(layers, url="box-bad")

    result, _ = await _rerun(layers, "ask", Talk(["box-worse", "tok"]), session=_session(layers, running=frozenset({"ask"})))

    assert (result.kind, result.rebind, result.answered) == ("rerun_failed", True, ("url", "token"))


async def test_rebind_is_false_for_a_plugin_that_is_not_running(layers) -> None:
    _on(layers, url="box-bad")

    result, _ = await _rerun(layers, "ask", Talk(["box-ok", "tok"]))

    assert (result.kind, result.answered, result.rebind) == ("rerun_ok", ("url", "token"), False)


async def test_rebind_when_the_probe_state_differs_from_the_stored_one(layers) -> None:
    _on(layers, url="box-ok")
    running = _session(layers, running=frozenset({"ask"}), touched={"ask": Touch(False, False, ("failed", "x"))})

    result, _ = await _rerun(layers, "ask", session=running)

    assert (result.kind, result.rebind, result.loads, result.answered) == ("rerun_ok", True, True, ())


async def test_rerun_probe_raises_is_check_failed_noload_never_rerun_ok(layers, monkeypatch) -> None:
    _on(layers, "stub")

    async def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(lifecycle, "_doctor_row", boom)

    result, _ = await _rerun(layers, "stub")

    assert result.kind == "check_failed_noload" and result.loads is False
    assert result.row.detail == "could not check it now: RuntimeError: boom"
    assert result.detail == result.step.failed_check == result.row.detail


async def test_rerun_probe_raises_on_a_running_plugin_whose_stored_probe_was_on_is_a_rebind(layers, monkeypatch) -> None:
    _on(layers, url="box-ok")
    real, calls = lifecycle._doctor_row, []

    async def second_raises(*a, **kw):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("boom")
        return await real(*a, **kw)

    monkeypatch.setattr(lifecycle, "_doctor_row", second_raises)
    running = _session(layers, running=frozenset({"ask"}), touched={"ask": Touch(False, True, ("on", ""))})

    result, _ = await _rerun(layers, "ask", session=running)

    assert (result.kind, result.rebind, result.loads, result.answered) == ("check_failed_noload", True, False, ())


async def test_rerun_resolve_raises_before_the_check_is_the_same_failed_check(layers, monkeypatch) -> None:
    _on(layers, "stub")
    calls = []

    def second_raises(loader, *a, **kw):
        calls.append(loader)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return resolve(loader, *a, **kw)

    monkeypatch.setattr(resolve_module, "resolve", second_raises)

    result, _ = await _rerun(layers, "stub")

    assert (result.kind, result.loads) == ("check_failed_noload", False)
    assert result.row.detail == "could not check it now: RuntimeError: boom" == result.detail


async def test_rerun_on_a_plugin_the_build_failed_whose_probe_still_fails_does_not_load(layers) -> None:
    _on(layers, url="box-dead")

    result, _ = await _rerun(layers, "ask", Talk(["", ""]), session=_session(layers, failed={"ask": "import broke"}))

    assert (result.kind, result.loads, result.row.state) == ("check_failed_noload", False, "failed")
    assert result.detail == result.step.failed_check == result.row.detail and "RuntimeError: dead" in result.detail


@pytest.mark.parametrize("kw", [
    {"failed": {"ask": "import broke"}}, {"unconfigured": {"ask": "ask.url"}},
    {"touched": {"ask": Touch(False, False, ("failed", "x"))}}])
async def test_rerun_whose_probe_passes_loads_whatever_the_build_or_the_stored_probe_said(layers, kw) -> None:
    _on(layers, url="box-ok")

    result, _ = await _rerun(layers, "ask", session=_session(layers, **kw))

    assert (result.kind, result.loads) == ("rerun_ok", True)


async def test_rerun_with_skipped_questions_and_an_unconfigured_probe_names_what_to_set(layers) -> None:
    _on(layers)

    result, _ = await _rerun(layers, "ask", Talk(["", ""]))

    assert (result.kind, result.questions, result.loads) == ("enabled_unconfigured", "skipped", False)
    assert result.detail == result.row.detail == "unconfigured — set ask.url"


async def test_rerun_a_plugin_that_is_on_and_not_running_with_no_setup_step_only_probes(layers) -> None:
    _on(layers, "stub")

    result, talk = await _rerun(layers, "stub")

    assert (result.kind, result.loads, result.rebind, result.questions) == ("rerun_ok", True, False, "none")
    assert result.row is not None and talk.asked == [] and talk.confirmed == []


async def test_rerun_while_a_download_still_runs_is_blocked_before_anything_else(layers, monkeypatch) -> None:
    monkeypatch.setattr(lifecycle, "_downloading", {"mob"})

    def never(*a, **kw):
        raise AssertionError("read the plan")

    monkeypatch.setattr(resolve_module, "resolve", never)
    session = _session(layers, running=frozenset({"mob"}), touched={"mob": Touch(False, False, None)})

    result, talk = await _rerun(layers, "mob", session=session)

    assert (result.kind, result.detail, result.loads) == ("rerun_blocked", lifecycle.DOWNLOAD_RUNNING, False)
    assert talk.asked == [] and talk.confirmed == [] and talk.reports == []


class _CfgConfig(BaseModel):
    port: int = 1
    label: str = ""
    token: SecretStr = SecretStr("")

    @field_validator("label")
    @classmethod
    def _not_banned(cls, value: str) -> str:
        if "banned" in value:
            raise ValueError(f"label {value!r} is banned")
        return value


class Cfg(Plugin):
    """has a port, a label and a token"""

    manifest = PluginManifest(name="cfg", version="0.1.0", kind="tools", enabled_by_default=False)
    ConfigModel = _CfgConfig


@pytest.fixture
def cfg(layers, monkeypatch) -> SimpleNamespace:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Cfg))
    audit = layers.g / "audit.jsonl"
    (layers.g / "config.yaml").write_text(yaml.safe_dump(_CONFIG | {"org": {"audit_log_path": str(audit)}}), encoding="utf-8")
    return SimpleNamespace(g=layers.g, ws=layers.ws, audit=audit)


def _rows(layers, name: str = "cfg"):
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws)
    resolution = resolve(loader)
    return settings_rows(resolution, loader, resolution.plan.entry(name))


def _row(layers, path: str):
    return next(r for r in _rows(layers) if r.path == path)


async def _configure(layers, row, raw, **kw):
    progress, reports = PassProgress(), []
    result = await configure_one(_session(layers, **kw), row, raw, progress=progress,
                                 report=lambda kind, payload: reports.append((kind, payload)))
    return result, progress, reports


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def test_settings_rows_are_the_plugins_own_sorted_without_its_switch_or_agent_rows(cfg) -> None:
    assert [r.path for r in _rows(cfg)] == ["cfg.label", "cfg.port", "cfg.token"]
    assert {r.plugin for r in _rows(cfg)} == {"cfg"}


def test_settings_rows_without_a_class_is_empty_and_imports_nothing(folder, monkeypatch) -> None:
    write_folder_plugin(folder.g, "swatch")

    def never(*a, **kw):
        raise AssertionError("imported")

    monkeypatch.setattr(discovery, "load_plugin_class", never)

    loader = ConfigLoader(config_dir=folder.g, local_config_dir=folder.ws)
    resolution = resolve(loader)

    assert resolution.plan.entry("swatch") is not None
    assert settings_rows(resolution, loader, resolution.plan.entry("swatch")) == ()
    assert not list((folder.g.parent / "sentinels").iterdir())


async def test_configure_one_writes_the_machine_overlay_and_audits_once(cfg) -> None:
    result, progress, reports = await _configure(cfg, _row(cfg, "cfg.port"), "7", running=frozenset({"cfg"}))

    assert yaml.safe_load((cfg.g / "overrides.yaml").read_text(encoding="utf-8")) == {"cfg": {"port": 7}}
    assert len(_lines(cfg.audit)) == 1
    assert (result.kind, result.rebind, result.loads, result.shown) == ("saved", True, True, "7")
    assert result.step == StepOutcome("cfg", True)
    assert (progress.phase, progress.wrote, progress.overridden) == ("edit", True, False)
    assert reports == [("wrote", (f"cfg.port saved in {cfg.g / 'overrides.yaml'}", (("cfg.port", "7"),)))]
    assert "cfg.port" in _lines(cfg.audit)[0]


async def test_configure_one_a_plugin_that_is_not_running_is_no_rebind_and_keeps_loads(cfg) -> None:
    result, _, _ = await _configure(cfg, _row(cfg, "cfg.label"), "hello", touched={"cfg": Touch(False, False, ("failed", "x"))})

    assert (result.kind, result.rebind, result.loads, result.shown) == ("saved", False, False, "'hello'")


async def test_configure_one_a_secret_is_shown_masked_and_never_written_to_a_project_file(cfg) -> None:
    result, _, reports = await _configure(cfg, _row(cfg, "cfg.token"), "s3cret-value")

    assert "s3cret-value" not in result.shown and "s3cret-value" not in repr(reports)
    assert "s3cret-value" not in "".join(_lines(cfg.audit))
    assert not (cfg.ws / "overrides.yaml").exists()


async def test_configure_one_a_value_the_plugins_model_refuses_is_refused_and_unaudited(cfg, monkeypatch) -> None:
    (cfg.g / "overrides.yaml").write_text(yaml.safe_dump({"cfg": {"port": 3}}), encoding="utf-8")
    before = (cfg.g / "overrides.yaml").read_bytes()

    async def never(written):
        raise AssertionError("audited")

    monkeypatch.setattr(plugins_cmd, "audit_value", never)

    result, progress, reports = await _configure(cfg, _row(cfg, "cfg.label"), "banned!")

    assert (result.kind, result.install, result.step) == ("refused", "none", StepOutcome("cfg", False))
    assert "banned" in result.detail and reports == []
    assert (cfg.g / "overrides.yaml").read_bytes() == before
    assert (progress.phase, progress.wrote) == ("edit", False)


async def test_configure_one_a_row_the_project_wins_is_saved_overridden_and_leaves_the_project_file(cfg) -> None:
    project = cfg.ws / "overrides.yaml"
    project.write_text(yaml.safe_dump({"cfg": {"port": 9}}), encoding="utf-8")
    before = project.read_bytes()
    row = _row(cfg, "cfg.port")
    assert row.winning_layer == "workspace-overrides"

    result, progress, _ = await _configure(cfg, row, "7", running=frozenset({"cfg"}))

    assert (result.kind, result.override, result.rebind, result.shown) == ("saved_overridden", (None, project), False, "7")
    assert progress.overridden and progress.wrote
    assert project.read_bytes() == before
    assert yaml.safe_load((cfg.g / "overrides.yaml").read_text(encoding="utf-8")) == {"cfg": {"port": 7}}


async def test_configure_one_refuses_a_core_row_and_the_switch_and_writes_nothing(cfg) -> None:
    loader = ConfigLoader(config_dir=cfg.g, local_config_dir=cfg.ws)
    resolution = resolve(loader)
    catalogue, _ = plugins_cmd.layered_catalogue(cfg.g, cfg.ws, loader=loader, plugins=plugins_cmd.plugin_catalogue_rows(resolution))
    core = next(e for e in catalogue.values() if e.plugin is None)
    switch = catalogue["cfg.enabled"]

    for row in (core, switch):
        result, progress, _ = await _configure(cfg, row, "1")
        assert result.kind == "refused" and "not one of" in result.detail
        assert progress.phase == "edit" and not progress.wrote

    assert not (cfg.g / "overrides.yaml").exists()
    assert _lines(cfg.audit) == []


async def test_configure_one_sets_wrote_and_overridden_before_the_audit_awaits(cfg, monkeypatch) -> None:
    (cfg.ws / "overrides.yaml").write_text(yaml.safe_dump({"cfg": {"port": 9}}), encoding="utf-8")
    seen = []

    async def cancelled(written):
        seen.append((progress.wrote, progress.overridden))
        raise asyncio.CancelledError

    monkeypatch.setattr(plugins_cmd, "audit_value", cancelled)
    progress = PassProgress()

    with pytest.raises(asyncio.CancelledError):
        await configure_one(_session(cfg), _row(cfg, "cfg.port"), "7", progress=progress, report=lambda k, p: None)

    assert seen == [(True, True)]
    assert yaml.safe_load((cfg.g / "overrides.yaml").read_text(encoding="utf-8")) == {"cfg": {"port": 7}}


def _info(layers, name: str, entry=None, **kw) -> str:
    loader = ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws)
    out = io.StringIO()
    Console(file=out, width=400, color_system=None).print(
        info_renderable(_session(layers, **kw), entry or resolve(loader).plan.entry(name)), soft_wrap=True)
    return out.getvalue()


def test_info_renderable_is_the_text_body_of_plugins_info(layers) -> None:
    text = _info(layers, "ask")

    assert text.splitlines()[:5] == ["ask", "  state:        off — turn on: localharness plugins enable ask",
                                     "  what it does: asks an address and a token", "  from:         built in",
                                     "  version:      0.1.0"]
    assert "Settings:" in text and "ask.url" in text and "Or paste this into your coding agent" in text


def test_info_renderable_says_why_a_plugin_did_not_load_or_is_not_set_up(layers) -> None:
    assert "  did not load: import broke\n" in _info(layers, "ask", failed={"ask": "import broke"})
    assert "  not set up:   ask.url\n" in _info(layers, "ask", unconfigured={"ask": "ask.url"})
    plain = _info(layers, "ask")
    assert "did not load" not in plain and "not set up" not in plain and "last check" not in plain


def test_info_renderable_says_what_the_last_check_found_unless_it_was_on(layers) -> None:
    failed = _info(layers, "ask", touched={"ask": Touch(False, True, ("failed", "token rejected"))})
    on = _info(layers, "ask", touched={"ask": Touch(False, True, ("on", "answers"))})

    assert "  last check:   failed — token rejected\n" in failed
    assert "last check" not in on


def test_info_renderable_renders_every_name_and_summary_as_text_after_sanitizing(layers) -> None:
    entry = dataclasses.replace(resolve(ConfigLoader(config_dir=layers.g, local_config_dir=layers.ws)).plan.entry("ask"),
                                name="bad\x1b[2Jname", summary="[browser] [red]tools[/red]", source="from\x1b]0;x\x07 here")

    text = _info(layers, "ask", entry)

    assert "\x1b" not in text and "\x07" not in text
    assert text.splitlines()[0] == "badname"
    assert "[browser] [red]tools[/red]" in text and "from here" in text


def test_info_renderable_strips_a_url_credential_and_never_shows_a_stored_secret(layers) -> None:
    (layers.g / "overrides.yaml").write_text(yaml.safe_dump({"ask": {"enabled": True, "url": "https://user:pw@example.test/v1", "token": "s3cret-value"}}),
                                             encoding="utf-8")

    text = _info(layers, "ask")

    assert "pw@" not in text and "s3cret-value" not in text
    assert "'https://example.test/v1'" in text and "Run it at https://example.test/v1." in text
