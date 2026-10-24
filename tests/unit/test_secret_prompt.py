"""A setup question for a secret that is already stored says what Enter does and how to clear it
(#28): `[stored key kept — type none to clear, or paste a new one]`, naming the old host when an
address asked in the same step just changed. Enter still writes nothing (R5); `none` writes the
setting's own default; anything else is the new key. The stored value is never shown or offered.

Plain `def` tests: `plugins enable` calls asyncio.run itself (asyncio_mode is "auto"). Every question is
asked by the real `prompt_line` on a pipe input; a recording PromptSession notes each one."""
from __future__ import annotations

import asyncio
import io
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import click
import httpx
import pytest
import yaml
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from typer.testing import CliRunner

from localharness.autoresearch.plugin import AutoresearchPlugin
from localharness.cli import interview, plugins_cmd
from localharness.cli.app import app
from localharness.cli.interview import prompt_line
from localharness.plugins import discovery, resolve, setup
from localharness.plugins.api import SetupField

runner = CliRunner()
KEPT = "stored key kept — type none to clear, or paste a new one"
KEY_Q = "Proposer API key (leave empty for a local server)"
HIDDEN = True
REPLACE = "\x15"  # Ctrl-U: the offered default is pre-filled, so a new value first clears it
KEY = "sk-SENTINEL-53-04-prompt-0123456789abcdef"
TOKEN = "SENTINEL-53-04-discord-token-0123456789"


@pytest.fixture(autouse=True)
def wide(monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")


@pytest.fixture
def typed(monkeypatch):
    """A terminal on a pipe input: `answers` are typed in order by `feed()` (Enter at a question is
    ""); every question is recorded as (text, default, is_password)."""
    t = SimpleNamespace(calls=[], answers=[], sessions=[])

    class Recording(PromptSession):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            t.sessions.append(self)

        async def prompt_async(self, *a, **kw):
            t.calls.append(("".join(x for _, x in self.message), kw["default"], self.is_password))
            return await super().prompt_async(*a, **kw)

    def feed():
        inp.send_text("".join(a + "\r" for a in t.answers))
        t.answers.clear()

    t.feed = feed
    monkeypatch.setattr(interview, "PromptSession", Recording)
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        yield t


# --- ask_fields: the question and what each answer writes --------------------------------------

CLOUD = "https://api.openai.com/v1"
LOCAL = "http://127.0.0.1:8001/v1"
SETUP = AutoresearchPlugin.manifest.setup


def _ask_over(typed, fields, answers: list[str], *, key_stored: bool = True):
    typed.answers.extend(answers)
    typed.feed()
    stored = {"proposer.base_url": CLOUD, "proposer.model": "gpt-x"}
    return asyncio.run(plugins_cmd.ask_fields(
        fields, lambda key: stored.get(key, ""),
        lambda key: key_stored and key == "proposer.api_key",
        lambda key: "none" if key == "proposer.api_key" else "",
        prompt=partial(prompt_line, console=Console(file=io.StringIO()))))


def _ask(typed, answers: list[str], *, key_stored: bool = True):
    return _ask_over(typed, SETUP, answers, key_stored=key_stored), typed.calls[-1]


def test_a_stored_key_question_says_enter_keeps_it_and_enter_writes_nothing(typed) -> None:
    pairs, question = _ask(typed, ["", "", ""])  # the address and model as stored (Enter)
    assert question == (f"{KEY_Q} [{KEPT}]: ", "", HIDDEN)
    assert pairs == [f"proposer.base_url={CLOUD}", "proposer.model=gpt-x"]


@pytest.mark.parametrize("answer", ["none", "NONE", " None "])
def test_none_at_a_stored_key_writes_the_settings_own_default(typed, answer) -> None:
    pairs, _ = _ask(typed, ["", "", answer])
    assert pairs[-1] == "proposer.api_key=none"


def test_a_new_key_at_a_stored_key_is_written(typed) -> None:
    pairs, _ = _ask(typed, ["", "", "sk-new"])
    assert pairs[-1] == "proposer.api_key=sk-new"


def test_a_changed_address_names_the_host_the_stored_key_was_for(typed) -> None:
    pairs, question = _ask(typed, [REPLACE + LOCAL, "", ""])
    assert question == (
        f"{KEY_Q} [stored key for api.openai.com kept — type none to clear, or paste a new one]: ", "", HIDDEN)
    assert pairs == [f"proposer.base_url={LOCAL}", "proposer.model=gpt-x"]


def test_an_unchanged_address_names_no_host(typed) -> None:
    _, question = _ask(typed, ["", "", ""])  # Enter everywhere: nothing changed
    assert question == (f"{KEY_Q} [{KEPT}]: ", "", HIDDEN)


def test_with_nothing_stored_the_secret_question_is_the_plain_prompt(typed) -> None:
    pairs, question = _ask(typed, [REPLACE + LOCAL, "", ""], key_stored=False)
    assert question == (f"{KEY_Q}: ", "", HIDDEN)
    assert pairs == [f"proposer.base_url={LOCAL}", "proposer.model=gpt-x"]


def test_the_questions_come_in_the_manifests_order_offering_what_is_stored(typed) -> None:
    _ask(typed, ["", "", ""])
    assert typed.calls == [(f"{SETUP[0].prompt}: ", CLOUD, False), ("Proposer model id: ", "gpt-x", False),
                           (f"{KEY_Q} [{KEPT}]: ", "", True)]


def test_no_fields_ask_nothing_and_write_nothing(typed) -> None:
    assert _ask_over(typed, (), []) == []
    assert typed.calls == [] and typed.sessions == []


@pytest.mark.parametrize("keys", [["\x03"], ["", "\x03"]])
def test_ctrl_c_at_any_question_aborts(typed, keys) -> None:
    with pytest.raises(click.exceptions.Abort):
        _ask(typed, keys)


def test_two_questions_build_two_sessions_and_the_first_answer_is_not_in_the_second_history(typed) -> None:
    _ask(typed, ["", REPLACE + "gpt-y", ""])
    assert len(typed.sessions) == 3 and len({id(s) for s in typed.sessions}) == 3
    assert typed.sessions[0].history.get_strings() == [CLOUD]
    assert typed.sessions[1].history.get_strings() == ["gpt-y"]
    assert "gpt-y" not in typed.sessions[2].history.get_strings()


def test_a_questions_control_characters_never_reach_the_prompt(typed) -> None:
    field = SetupField(key="proposer.model", prompt="Model\x1b[2J id\r")
    _ask_over(typed, (field,), [""])
    assert typed.calls[-1][0] == "Model id: "


# --- composed: the real commands ----------------------------------------------------------------

@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                                     "default_model": "test-model"}}), encoding="utf-8")
    return g


def _enable(g: Path, *args: str, typed=None):
    if typed is not None:
        typed.feed()
    result = runner.invoke(app, ["plugins", "enable", *args, "--config-dir", str(g)])
    for where in (result.stdout, result.stderr, repr(result.exception)):
        for secret in (KEY, KEY[-12:], TOKEN, TOKEN[-12:]):
            assert secret not in where, where
    return result


def _overrides(g: Path) -> dict:
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def test_autoresearch_moving_from_a_cloud_api_names_the_cloud_host_at_the_key(g, typed, monkeypatch) -> None:
    from localharness.autoresearch import plugin as autoresearch_plugin

    seen: list = []

    def answer(request):
        seen.append(request)
        return httpx.Response(200, json={"object": "list", "data": [{"id": "p-model"}]})

    monkeypatch.setattr(autoresearch_plugin, "_TRANSPORT", httpx.MockTransport(answer))
    (g / "overrides.yaml").write_text(yaml.safe_dump({"proposer": {
        "base_url": "https://api.cloud.test/v1", "model": "cloud-model", "api_key": KEY}}), encoding="utf-8")
    typed.answers.extend([REPLACE + "https://p.test/v1", REPLACE + "p-model", ""])
    result = _enable(g, "autoresearch", typed=typed)

    assert result.exit_code == 0, result.output
    assert [(text, default) for text, default, _ in typed.calls] == [
        (f"{SETUP[0].prompt}: ", "https://api.cloud.test/v1"),
        ("Proposer model id: ", "cloud-model"),
        (f"{KEY_Q} [stored key for api.cloud.test kept — type none to clear, or paste a new one]: ", "")]
    assert _overrides(g)["proposer"] == {"base_url": "https://p.test/v1", "model": "p-model", "api_key": KEY}
    assert [r.headers.get("Authorization") for r in seen] == [f"Bearer {KEY}"]


@pytest.fixture
def dispatch_ready(monkeypatch, tmp_path):
    """The real dispatch plugin with its extra present; Discord's env sources isolated."""
    from tests.dispatch_support import isolate_discord_env

    isolate_discord_env(monkeypatch, tmp_path)
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)


def test_dispatch_none_at_a_stored_token_clears_it_and_the_plugin_reads_unconfigured(
        g, typed, dispatch_ready) -> None:
    stored = _enable(g, "dispatch", "--set", f"discord.token={TOKEN}", "--set", "discord.allow=123")
    assert stored.exit_code == 0, stored.output

    typed.answers.extend(["none", ""])  # clear the token; Enter keeps the allow list
    cleared = _enable(g, "dispatch", typed=typed)
    assert cleared.exit_code == 0, cleared.output
    assert typed.calls[0] == (f"Discord bot token [{KEPT}]: ", "", HIDDEN)
    assert _overrides(g)["dispatch"]["discord"]["token"] == ""
    assert "dispatch: Discord not configured" in cleared.output, cleared.output

    typed.answers.extend(["", ""])  # asked again: nothing is stored now, so the plain question
    again = _enable(g, "dispatch", typed=typed)
    assert again.exit_code == 0, again.output
    assert typed.calls[2] == ("Discord bot token: ", "", HIDDEN)
