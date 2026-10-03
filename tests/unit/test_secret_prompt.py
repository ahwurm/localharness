"""A setup question for a secret that is already stored says what Enter does and how to clear it
(#28): `[stored key kept — type none to clear, or paste a new one]`, naming the old host when an
address asked in the same step just changed. Enter still writes nothing (R5); `none` writes the
setting's own default; anything else is the new key. The stored value is never shown or offered.

Plain `def` tests: `plugins enable` calls asyncio.run itself (asyncio_mode is "auto")."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from localharness.autoresearch.plugin import AutoresearchPlugin
from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import discovery, resolve, setup

runner = CliRunner()
KEPT = "stored key kept — type none to clear, or paste a new one"
KEY_Q = "Proposer API key (leave empty for a local server)"
HIDDEN = {"hide_input": True, "show_default": False}
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
    """A terminal: `answers` are typed in order (Enter at a question is ""); every question is
    recorded as (text, default, keywords)."""
    calls: list[tuple] = []
    answers: list[str] = []

    def prompt(text, default=None, **kw):
        calls.append((text, default, kw))
        return answers.pop(0)

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    return calls, answers


# --- _ask_fields: the question and what each answer writes ------------------------------------

CLOUD = "https://api.openai.com/v1"
LOCAL = "http://127.0.0.1:8001/v1"


def _ask(typed, answers: list[str], *, key_stored: bool = True):
    calls, scripted = typed
    scripted.extend(answers)
    stored = {"proposer.base_url": CLOUD, "proposer.model": "gpt-x"}
    pairs = plugins_cmd._ask_fields(
        AutoresearchPlugin.manifest.setup, lambda key: stored.get(key, ""),
        lambda key: key_stored and key == "proposer.api_key",
        lambda key: "none" if key == "proposer.api_key" else "")
    return pairs, calls[-1]


def test_a_stored_key_question_says_enter_keeps_it_and_enter_writes_nothing(typed) -> None:
    pairs, question = _ask(typed, [CLOUD, "gpt-x", ""])  # the address and model as stored (Enter)
    assert question == (f"{KEY_Q} [{KEPT}]", "", HIDDEN)
    assert pairs == [f"proposer.base_url={CLOUD}", "proposer.model=gpt-x"]


@pytest.mark.parametrize("answer", ["none", "NONE", " None "])
def test_none_at_a_stored_key_writes_the_settings_own_default(typed, answer) -> None:
    pairs, _ = _ask(typed, [CLOUD, "gpt-x", answer])
    assert pairs[-1] == "proposer.api_key=none"


def test_a_new_key_at_a_stored_key_is_written(typed) -> None:
    pairs, _ = _ask(typed, [CLOUD, "gpt-x", "sk-new"])
    assert pairs[-1] == "proposer.api_key=sk-new"


def test_a_changed_address_names_the_host_the_stored_key_was_for(typed) -> None:
    pairs, question = _ask(typed, [LOCAL, "gpt-x", ""])
    assert question == (
        f"{KEY_Q} [stored key for api.openai.com kept — type none to clear, or paste a new one]", "", HIDDEN)
    assert pairs == [f"proposer.base_url={LOCAL}", "proposer.model=gpt-x"]


def test_an_unchanged_address_names_no_host(typed) -> None:
    _, question = _ask(typed, ["", "", ""])  # Enter everywhere: nothing changed
    assert question == (f"{KEY_Q} [{KEPT}]", "", HIDDEN)


def test_with_nothing_stored_the_secret_question_is_the_plain_prompt(typed) -> None:
    pairs, question = _ask(typed, [LOCAL, "gpt-x", ""], key_stored=False)
    assert question == (KEY_Q, "", HIDDEN)
    assert pairs == [f"proposer.base_url={LOCAL}", "proposer.model=gpt-x"]


# --- composed: the real commands ----------------------------------------------------------------

@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                                     "default_model": "test-model"}}), encoding="utf-8")
    return g


def _enable(g: Path, *args: str):
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
    calls, answers = typed
    answers.extend(["https://p.test/v1", "p-model", ""])
    result = _enable(g, "autoresearch")

    assert result.exit_code == 0, result.output
    assert [(text, default) for text, default, _ in calls] == [
        (AutoresearchPlugin.manifest.setup[0].prompt, "https://api.cloud.test/v1"),
        ("Proposer model id", "cloud-model"),
        (f"{KEY_Q} [stored key for api.cloud.test kept — type none to clear, or paste a new one]", "")]
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
    calls, answers = typed
    stored = _enable(g, "dispatch", "--set", f"discord.token={TOKEN}", "--set", "discord.allow=123")
    assert stored.exit_code == 0, stored.output

    answers.extend(["none", ""])  # clear the token; Enter keeps the allow list
    cleared = _enable(g, "dispatch")
    assert cleared.exit_code == 0, cleared.output
    assert calls[0] == (f"Discord bot token [{KEPT}]", "", HIDDEN)
    assert _overrides(g)["dispatch"]["discord"]["token"] == ""
    assert "dispatch: Discord not configured" in cleared.output, cleared.output

    answers.extend(["", ""])  # asked again: nothing is stored now, so the plain question
    again = _enable(g, "dispatch")
    assert again.exit_code == 0, again.output
    assert calls[2] == ("Discord bot token", "", HIDDEN)
