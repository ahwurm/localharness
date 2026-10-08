"""dispatch/config.py: the `dispatch.discord.*` settings shape (DISP-02)."""
from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from localharness.config.plugin_sections import global_only_paths
from localharness.dispatch.config import DispatchConfig, _ids

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
