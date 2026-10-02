"""dispatch/config.py: the `dispatch.discord.*` settings shape (DISP-02) and its env fallback."""
from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from localharness.config.plugin_sections import global_only_paths
from localharness.dispatch.config import ENV_FILE_RELATIVE, DiscordSettings, DispatchConfig, _ids, env_fallback
from tests.dispatch_support import isolate_discord_env

SENTINEL = "SENTINEL-TOKEN"


@pytest.mark.parametrize("raw, ids", [
    ("1, 2,,3", ["1", "2", "3"]),
    (42, ["42"]),
    ([42, "7 "], ["42", "7"]),
    ("", []),
    (None, []),
])
def test_ids_coerce_to_digit_strings(raw, ids):
    assert _ids(raw) == ids


def test_ids_refuse_non_digit_naming_it():
    with pytest.raises(ValueError, match="'abc'"):
        _ids("1,abc")


def test_set_style_raw_string_validates_to_list():
    """`--set discord.allow=1,2` reaches the model as the raw string "1,2" (registry/coerce.py
    passes list[str] through), and YAML gives int snowflakes."""
    cfg = DispatchConfig.model_validate({"discord": {"allow": "1,2", "channels": [123456789012345678]}})
    assert cfg.discord.allow == ["1", "2"]
    assert cfg.discord.channels == ["123456789012345678"]
    with pytest.raises(ValidationError, match="'x'"):
        DispatchConfig.model_validate({"discord": {"channels": "x"}})


def test_typo_key_refused():
    with pytest.raises(ValidationError):
        DispatchConfig.model_validate({"discord": {"alow": []}})
    with pytest.raises(ValidationError):
        DispatchConfig.model_validate({"discrod": {}})


def test_defaults():
    d = DispatchConfig().discord
    assert (d.token.get_secret_value(), d.allow, d.channels, d.ack) == ("", [], [], "✅")


def test_token_never_rendered():
    cfg = DispatchConfig.model_validate({"discord": {"token": SENTINEL, "allow": "1"}})
    assert cfg.discord.token.get_secret_value() == SENTINEL
    for text in (repr(cfg), str(cfg), repr(cfg.discord), str(cfg.discord),
                 str(cfg.model_dump(mode="json")), cfg.model_dump_json(), str(cfg.model_dump())):
        assert SENTINEL not in text


def test_global_only_paths_exactly_token_allow_channels():
    assert global_only_paths(DispatchConfig) == {"discord.token", "discord.allow", "discord.channels"}


def test_importing_config_pulls_no_channel_dependencies():
    code = ("import sys, localharness.dispatch.config; "
            "assert 'discord' not in sys.modules and 'prompt_toolkit' not in sys.modules, "
            "[m for m in ('discord', 'prompt_toolkit') if m in sys.modules]")
    subprocess.run([sys.executable, "-c", code], check=True)


# --- env_fallback: per field, the setting wins; env fills only a field at its default ----------

@pytest.fixture
def home(monkeypatch, tmp_path):
    """Isolated from the live ~/.claude/channels/discord/.env; env_fallback gets this home only."""
    return isolate_discord_env(monkeypatch, tmp_path)


def _line(source, field):
    return (f"dispatch: {source} is deprecated and stops working in 0.17.0 — set dispatch.discord.{field} "
            f"(localharness components set dispatch.discord.{field} …)")


ALL_ENV = {"LOCALHARNESS_DISCORD_TOKEN": SENTINEL, "LOCALHARNESS_DISCORD_ALLOW": "1, 2,,3",
           "LOCALHARNESS_DISCORD_CHANNELS": "7", "LOCALHARNESS_DISCORD_ACK": "👀"}
SET = {"token": "SET-TOKEN", "allow": ["9"], "channels": ["8"], "ack": "🔔"}


def _view(s):
    return {"token": s.token.get_secret_value(), "allow": s.allow, "channels": s.channels, "ack": s.ack}


@pytest.mark.parametrize("settings, environ, want, sources", [
    pytest.param({}, {}, {"token": "", "allow": [], "channels": [], "ack": "✅"}, [], id="nothing"),
    pytest.param(SET, {}, SET, [], id="setting-only"),
    pytest.param({}, ALL_ENV, {"token": SENTINEL, "allow": ["1", "2", "3"], "channels": ["7"], "ack": "👀"},
                 [("LOCALHARNESS_DISCORD_TOKEN", "token"), ("LOCALHARNESS_DISCORD_ALLOW", "allow"),
                  ("LOCALHARNESS_DISCORD_CHANNELS", "channels"), ("LOCALHARNESS_DISCORD_ACK", "ack")],
                 id="env-only"),
    pytest.param(SET, ALL_ENV, SET, [], id="both-setting-wins"),
    pytest.param({"token": "SET-TOKEN"}, {"LOCALHARNESS_DISCORD_TOKEN": SENTINEL, "LOCALHARNESS_DISCORD_ALLOW": "5"},
                 {"token": "SET-TOKEN", "allow": ["5"], "channels": [], "ack": "✅"},
                 [("LOCALHARNESS_DISCORD_ALLOW", "allow")], id="mixed"),
    pytest.param({}, {"LOCALHARNESS_DISCORD_TOKEN": "", "DISCORD_BOT_TOKEN": SENTINEL},
                 {"token": SENTINEL, "allow": [], "channels": [], "ack": "✅"},
                 [("DISCORD_BOT_TOKEN", "token")], id="bot-token-when-first-empty"),
    pytest.param({}, {"LOCALHARNESS_DISCORD_TOKEN": "A", "DISCORD_BOT_TOKEN": "B"},
                 {"token": "A", "allow": [], "channels": [], "ack": "✅"},
                 [("LOCALHARNESS_DISCORD_TOKEN", "token")], id="first-token-beats-bot-token"),
    pytest.param({}, {"LOCALHARNESS_DISCORD_ACK": ""}, {"token": "", "allow": [], "channels": [], "ack": ""},
                 [("LOCALHARNESS_DISCORD_ACK", "ack")], id="ack-set-empty-disables"),
    pytest.param({}, {"LOCALHARNESS_DISCORD_ALLOW": " , ,"}, {"token": "", "allow": [], "channels": [], "ack": "✅"},
                 [], id="allow-only-commas-decides-nothing"),
    pytest.param({"allow": [], "ack": "✅"}, {"LOCALHARNESS_DISCORD_ALLOW": "4", "LOCALHARNESS_DISCORD_ACK": "👀"},
                 {"token": "", "allow": ["4"], "channels": [], "ack": "👀"},
                 [("LOCALHARNESS_DISCORD_ALLOW", "allow"), ("LOCALHARNESS_DISCORD_ACK", "ack")],
                 id="caveat-explicit-default-is-unset"),
])
def test_env_fallback_table(home, settings, environ, want, sources):
    """Caveat row: a setting explicitly equal to its default (`allow: []`, `ack: "✅"`) cannot be
    told from unset, so the env fills it — accepted for a one-release fallback."""
    out, lines = env_fallback(DiscordSettings.model_validate(settings), environ, home)
    assert _view(out) == want
    assert lines == [_line(s, f) for s, f in sources]
    assert not any(SENTINEL in line or "SET-TOKEN" in line for line in lines)


def _write_env_file(home, text):
    path = home / ENV_FILE_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_text(text)


@pytest.mark.parametrize("value", [SENTINEL, f'"{SENTINEL}"', f"'{SENTINEL}' "])
def test_env_file_token_only_when_both_vars_empty(home, value):
    _write_env_file(home, f"OTHER=1\nDISCORD_BOT_TOKEN={value}\n")
    out, lines = env_fallback(DiscordSettings(), {"LOCALHARNESS_DISCORD_TOKEN": "", "DISCORD_BOT_TOKEN": ""}, home)
    assert out.token.get_secret_value() == SENTINEL
    assert lines == [_line("~/.claude/channels/discord/.env", "token")]
    out, lines = env_fallback(DiscordSettings(), {"DISCORD_BOT_TOKEN": "B"}, home)
    assert out.token.get_secret_value() == "B" and lines == [_line("DISCORD_BOT_TOKEN", "token")]
    out, lines = env_fallback(DiscordSettings(token="S"), {}, home)
    assert out.token.get_secret_value() == "S" and lines == []


def test_env_file_missing_or_unreadable_is_no_token(home):
    assert env_fallback(DiscordSettings(), {}, home) == (DiscordSettings(), [])
    (home / ENV_FILE_RELATIVE).mkdir(parents=True)  # a directory: read_text raises
    assert env_fallback(DiscordSettings(), {}, home)[1] == []


def test_env_bad_id_refused_naming_it(home):
    with pytest.raises(ValidationError, match="'abc'"):
        env_fallback(DiscordSettings(), {"LOCALHARNESS_DISCORD_ALLOW": "1,abc"}, home)


def test_env_fallback_is_pure(home, monkeypatch):
    """It never reads os.environ or Path.home(): both are made to raise, and it still answers."""
    import os
    from pathlib import Path

    class Boom(dict):
        def __getitem__(self, k):
            raise AssertionError("os.environ read")
        get = __contains__ = __iter__ = __getitem__

    with monkeypatch.context() as m:  # undone before pytest's own teardown touches os.environ
        m.setattr(os, "environ", Boom())
        m.setattr(Path, "home", classmethod(lambda cls: (_ for _ in ()).throw(AssertionError("Path.home"))))
        out, lines = env_fallback(DiscordSettings(), {"LOCALHARNESS_DISCORD_ALLOW": "3"}, home)
    assert out.allow == ["3"] and lines == [_line("LOCALHARNESS_DISCORD_ALLOW", "allow")]
