"""`plugins enable|disable|info` and `session_step` pinned from the unchanged shell: stdout, stderr
and exit code (or the StepOutcome) per scenario, one JSON golden.

A diff is a finding to explain, never a golden to regenerate; LOCALHARNESS_REGEN_GOLDEN=1 rewrites it.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery, lifecycle, resolve, setup
from localharness.plugins.api import GLOBAL_ONLY, Check, Plugin, PluginManifest, SetupField
from tests.plugin_prompts import fake_prompts
from tests.unit.test_doctor_layer_report import _layout
from tests.unit.test_plugin_resolve import write_folder_plugin
from tests.unit.test_plugins_enable_setup import _CONFIG

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "plugins_surfaces" / "shell_golden.json"
REGEN = os.environ.get("LOCALHARNESS_REGEN_GOLDEN") == "1"
runner = CliRunner()
_REAL_DISCOVER = discovery.discover
CHECK = ["pass"]
DONE: set[str] = set()
OK, BAD_CREDENTIAL, CREDENTIAL = "http://ok.test", "https://user:pw@bad.test", "https://user:pw@example.test"


class ProConfig(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)
    token: SecretStr = Field(SecretStr(""), json_schema_extra=GLOBAL_ONLY)

    @field_validator("url")
    @classmethod
    def _not_bad(cls, value: str) -> str:
        if "bad" in value:
            raise ValueError(f"cannot reach {value!r}")
        return value


class Pro(Plugin):
    """answers at an address"""

    manifest = PluginManifest(
        name="pro", version="0.1.0", kind="tools", enabled_by_default=False,
        setup=(SetupField(key="url", prompt="Server address", default="http://127.0.0.1:1"),
               SetupField(key="token", prompt="Access token", secret=True)),
        setup_help="PRO-HELP", agent_prompt="Run it at {url}. Done when it answers.",
        next_steps="Then: start it.\nThen: use it.")
    ConfigModel = ProConfig

    async def configure(self, ctx):
        return "ready" if ctx.config.url else ("unconfigured", "pro.url")

    def doctor(self, ctx):
        return [Check(name="pro", status=CHECK[0], detail=f"at {ctx.config.url}")]


class Act(Plugin):
    """fetches the thing"""

    manifest = PluginManifest(name="act", version="0.1.0", kind="tools", enabled_by_default=False,
                              setup_action="Fetch the thing now?", next_steps="Then: run the thing.")

    def setup_action(self, ctx):
        DONE.add("act")
        return [Check(name="act", status="pass", detail="fetched")]

    def doctor(self, ctx):
        return [Check(name="act", status="pass", detail="ready") if DONE
                else Check(name="act", status="fail", detail="not fetched")]


class Plain(Plugin):
    """has nothing to set up"""

    manifest = PluginManifest(name="plain", version="0.1.0", kind="tools", enabled_by_default=False)


class Needx(Plugin):
    """needs an install extra nobody has"""

    manifest = PluginManifest(
        name="needx", version="0.1.0", kind="tools", enabled_by_default=False,
        requires_extra="nosuchextra", next_steps="NEEDX-NEXT",
        setup=(SetupField(key="url", prompt="Needx address", default="http://127.0.0.1:2"),))
    ConfigModel = ProConfig


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Pro, Act, Plain, Needx))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")
    CHECK[0] = "pass"
    DONE.clear()
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _config_dir(env) -> Path:
    g = env.tmp / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _terminal(env, answers=(), yes=()) -> None:
    typed, confirmed = list(answers), list(yes)
    env.mp.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    fake_prompts(env.mp, prompt=lambda text, default=None, **kw: typed.pop(0),
                 confirm=lambda text, default=None, **kw: confirmed.pop(0))


def _asks_nothing(env, *, terminal: bool) -> None:
    def asked(*a, **kw):
        raise AssertionError("asked")
    env.mp.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: terminal)
    fake_prompts(env.mp, prompt=asked, confirm=asked)


def _installed(env, g: Path) -> None:
    marks = env.tmp / "sentinels"
    marks.mkdir()
    env.mp.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    write_folder_plugin(g, "swatch")
    env.mp.setattr(discovery, "discover", lambda d: [f for f in _REAL_DISCOVER(d) if f.source == "folder"])


def _overrides(g: Path, data: dict) -> None:
    (g / "overrides.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _probe_raises(env, g: Path) -> None:
    real = lifecycle._doctor_row

    async def doctor_row(*a, **kw):
        if (g / "overrides.yaml").exists():
            raise RuntimeError("boom")
        return await real(*a, **kw)
    env.mp.setattr(lifecycle, "_doctor_row", doctor_row)


def _run(*argv: str) -> dict:
    r = runner.invoke(app, list(argv))
    return {"stdout": r.stdout, "stderr": r.stderr, "exit": r.exit_code}


def _cli(g: Path, *args: str) -> dict:
    return _run("plugins", *args, "--config-dir", str(g))


def _step(env, g: Path, name: str = "pro", verb: str = "enable") -> dict:
    out = plugins_cmd.session_step((verb, name), str(g))
    cap = env.capsys.readouterr()
    return {"stdout": cap.out, "stderr": cap.err, "on": out.on, "failed_check": out.failed_check,
            "skipped_check": out.skipped_check, "stopped": out.stopped}


CASES = {}


def case(fn):
    CASES[fn.__name__] = fn
    return fn


@case
def enable_questions_check_passes(env):
    g = _config_dir(env)
    _terminal(env, [OK, "s3cret"])
    return _cli(g, "enable", "pro")


@case
def enable_check_fails_help_and_prompt(env):
    g = _config_dir(env)
    CHECK[0] = "fail"
    _terminal(env, ["http://nope.test", "s3cret"])
    return _cli(g, "enable", "pro")


@case
def enable_check_skipped(env):
    g = _config_dir(env)
    CHECK[0] = "skip"
    _terminal(env, [OK, ""])
    return _cli(g, "enable", "pro")


@case
def enable_setup_action_confirmed(env):
    g = _config_dir(env)
    _terminal(env, yes=[True])
    return _cli(g, "enable", "act")


@case
def enable_missing_extra_note(env):
    g = _config_dir(env)
    env.mp.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda extra: False)
    _asks_nothing(env, terminal=True)
    return _cli(g, "enable", "needx")


@case
def enable_project_override_note(env):
    layout = _layout(env.tmp, env.mp, env.fake_home)
    (layout.ws_dir / "config.yaml").write_text("plain: {enabled: false}\n", encoding="utf-8")
    return _run("plugins", "enable", "plain")


@case
def enable_set_values(env):
    return _cli(_config_dir(env), "enable", "pro", "--set", f"url={OK}", "--set", "token=s3cret")


@case
def enable_no_terminal(env):
    g = _config_dir(env)
    _asks_nothing(env, terminal=False)
    return _cli(g, "enable", "pro")


@case
def enable_refused_credential_url(env):
    return _cli(_config_dir(env), "enable", "pro", "--set", f"url={BAD_CREDENTIAL}")


@case
def enable_probe_raises(env):
    g = _config_dir(env)
    _probe_raises(env, g)
    _terminal(env, [OK, "s3cret"])
    return _cli(g, "enable", "pro")


@case
def enable_resolve_raises(env):
    g = _config_dir(env)
    real, calls = resolve.resolve, []

    def second_raises(*a, **kw):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("boom")
        return real(*a, **kw)
    env.mp.setattr(resolve, "resolve", second_raises)
    _terminal(env, [OK, "s3cret"])
    return _cli(g, "enable", "pro")


@case
def enable_help_prompt_credential_url(env):
    g = _config_dir(env)
    CHECK[0] = "fail"
    _terminal(env, [CREDENTIAL, "s3cret"])
    return _cli(g, "enable", "pro")


@case
def disable_plain(env):
    return _cli(_config_dir(env), "disable", "pro")


@case
def step_questions_check_passes(env):
    g = _config_dir(env)
    _terminal(env, [OK, "s3cret"])
    return _step(env, g)


@case
def step_check_fails(env):
    g = _config_dir(env)
    CHECK[0] = "fail"
    _terminal(env, ["http://nope.test", "s3cret"])
    return _step(env, g)


@case
def step_third_party_declined(env):
    g = _config_dir(env)
    _installed(env, g)
    _terminal(env, yes=[False])
    return _step(env, g, "swatch")


@case
def step_stopped_at_question(env):
    g = _config_dir(env)

    def abort(*a, **kw):
        raise click.exceptions.Abort()
    env.mp.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    fake_prompts(env.mp, prompt=abort)
    return _step(env, g)


@case
def step_setup_action_confirmed(env):
    g = _config_dir(env)
    _terminal(env, yes=[True])
    return _step(env, g, "act")


@case
def step_refused_credential_url(env):
    g = _config_dir(env)
    _terminal(env, [BAD_CREDENTIAL, ""])
    return _step(env, g)


@case
def step_probe_raises(env):
    g = _config_dir(env)
    _probe_raises(env, g)
    _terminal(env, [OK, "s3cret"])
    return _step(env, g)


@case
def step_configured_not_asked(env):
    g = _config_dir(env)
    _overrides(g, {"pro": {"url": OK}})
    _asks_nothing(env, terminal=True)
    return _step(env, g)


@case
def info_settings_and_prompt(env):
    g = _config_dir(env)
    _overrides(g, {"pro": {"enabled": True, "url": OK}})
    return _cli(g, "info", "pro")


@case
def info_switch_only_note(env):
    g = _config_dir(env)
    _installed(env, g)
    return _cli(g, "info", "swatch")


@case
def info_credential_url(env):
    g = _config_dir(env)
    _overrides(g, {"pro": {"enabled": True, "url": f"{CREDENTIAL}/v1"}})
    return _cli(g, "info", "pro")


def _norm(got: dict, tmp_path: Path) -> dict:
    return {k: v.replace(str(tmp_path.resolve()), "<TMP>").replace(str(tmp_path), "<TMP>")
            if isinstance(v, str) else v for k, v in got.items()}


@pytest.mark.parametrize("key", list(CASES))
def test_matches_the_golden(key, tmp_path, monkeypatch, capsys, fake_home):
    got = _norm(CASES[key](SimpleNamespace(tmp=tmp_path, mp=monkeypatch, capsys=capsys,
                                           fake_home=fake_home)), tmp_path)
    if REGEN:
        data = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
        data[key] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert got == json.loads(GOLDEN.read_text(encoding="utf-8"))[key], f"{key} drifted"


def test_every_scenario_has_a_golden_entry():
    assert set(CASES) == set(json.loads(GOLDEN.read_text(encoding="utf-8")))
