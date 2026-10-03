"""52-03 follow-up: a refused settings write never prints a stored secret.

`components set` (and `plugins enable --set` for a bundled plugin's core keys) checks the write
against the global config.yaml merged with the overrides.yaml about to be written. Its refusal used
to print pydantic's `str(ValidationError)`, whose `input_value` is that whole merged dict, so a
`proposer.api_key` stored in config.yaml was printed. pydantic shortens that repr to its head and
tail, so the leak shows when the key sits in the tail (written last, as a hand-written config has
it) and a long key leaks its last characters even when it does not fit, which a whole-string scrub
cannot catch. These tests store the key last, and check both the key and its tail.

A refusal now prints each error's location and message only, and scrubs every secret value of the
merged settings, plugin sections included (a plugin validator's own message can quote its input).
Every run passes an explicit --config-dir; nothing reads the real home."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd, plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest
from tests.unit.test_plugins_enable_setup import _CONFIG

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
KEY = "sk-SENTINEL-components-52-0123456789abcdef"  # long, as a real key is
TAIL = KEY[-12:]


class SecqConfig(BaseModel):
    tok: SecretStr = Field(SecretStr(""), json_schema_extra=GLOBAL_ONLY)
    n: int = 0

    @model_validator(mode="before")
    @classmethod
    def _quotes_its_section(cls, data):  # a plugin validator whose message quotes its whole input
        if isinstance(data, dict) and data.get("n") == 4:
            raise ValueError(f"bad section {data!r}")
        return data


class Secq(Plugin):
    """holds a secret its own validator quotes"""

    manifest = PluginManifest(name="secq", version="0.1.0", kind="tools")
    ConfigModel = SecqConfig


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(components_cmd, "console", Console(width=400))
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _home(tmp_path: Path, **sections) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    cfg = dict(_CONFIG, org={"audit_log_path": str(g / "audit.jsonl")}, **sections)
    # sort_keys=False: the key stays LAST, as in a hand-written config.yaml (see the docstring)
    (g / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return g


def _run(g: Path, *args: str):
    result = runner.invoke(app, [*args, "--config-dir", str(g)])
    for where in (result.stdout, result.stderr, repr(result.exception)):
        for piece in (KEY, TAIL):
            assert piece not in where, f"{piece!r} reached the output of {args}:\n{where}"
    audit = g / "audit.jsonl"
    assert not audit.exists() or TAIL not in audit.read_text(encoding="utf-8")
    return result


def _flat(result) -> str:
    return " ".join((result.stdout + result.stderr).split())


PROPOSER = {"base_url": "http://p/v1", "model": "p-old", "api_key": KEY}


@pytest.mark.parametrize("flags", [(), ("--json",)], ids=["text", "json"])
def test_components_set_refusal_never_echoes_the_stored_proposer_key(tmp_path, flags) -> None:
    g = _home(tmp_path, proposer=PROPOSER)
    result = _run(g, "components", "set", "proposer.model", "test-model", *flags)

    assert result.exit_code == 2, result.output
    assert "proposer.model" in _flat(result) and "must differ" in _flat(result)
    assert not (g / "overrides.yaml").exists()


def test_components_set_refusal_scrubs_a_plugin_secret_stored_in_config_yaml(tmp_path, monkeypatch) -> None:
    """The plugin's own validator quotes its section, token included: the scrub covers the
    config.yaml value too, not only what the new overlay holds."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Secq,))
    g = _home(tmp_path, secq={"n": 1, "tok": KEY})
    result = _run(g, "components", "set", "secq.n", "4")

    assert result.exit_code == 2, result.output
    assert "Validation failed for secq.n" in _flat(result) and "bad section" in _flat(result)
    assert components_cmd.SECRET_MASK in _flat(result)


def test_plugins_enable_core_keys_refusal_never_echoes_the_key_tail(tmp_path) -> None:
    """The real autoresearch plugin's claimed `proposer` section through plugins enable --set: a
    long key is shown only by its tail, which only the location-and-message text keeps out."""
    g = _home(tmp_path, proposer=PROPOSER)
    result = _run(g, "plugins", "enable", "autoresearch", "--set", "proposer.base_url=http://p/v1",
                  "--set", "proposer.model=test-model")

    assert result.exit_code == 2, result.output
    assert "proposer.model must differ" in _flat(result)


def test_plugins_enable_own_key_refusal_scrubs_a_plugin_secret_stored_in_config_yaml(tmp_path, monkeypatch) -> None:
    """The plugin's own keys through plugins enable --set: its validator quotes the section it was
    given, the config.yaml layer included, so every layer's secret is scrubbed."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Secq,))
    g = _home(tmp_path, secq={"n": 1, "tok": KEY})
    result = _run(g, "plugins", "enable", "secq", "--set", "n=4")

    assert result.exit_code == 2, result.output
    assert "Validation failed for secq" in _flat(result) and "bad section" in _flat(result)
    assert components_cmd.SECRET_MASK in _flat(result)
