"""R16: a config that fails validation never echoes a secret — not in any command's text or --json.

Every command that loads the config reports a refusal with the loader's field errors. Each used to
carry the rejected input as `(got: …)`, and validate printed the same value. For an error on a
whole section or model (the proposer-must-differ rule; a missing field, whose input is the section
around it), that input held `proposer.api_key`, and the bounded repr sorts dict keys, so the key was
printed whatever the file's order. A key typed under a misspelled name, or a removed plugin's
leftover section, has no schema that says what is secret, so its value is never shown at all.

Each case stores a long key LAST in config.yaml and checks the whole key and its last 8 characters
in stdout, stderr and repr(exception), for every command that reports a load refusal. Every run
passes an explicit --config-dir; nothing reads the real home, and start stops before any model
server (a listing raises)."""
from __future__ import annotations

import sys

import pytest
import yaml
from pydantic import BaseModel, SecretStr, model_validator
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.plugins import builtin, discovery
from localharness.plugins.api import Plugin, PluginManifest

runner = CliRunner()
_REAL_DISCOVER = discovery.discover
KEY = "sk-SENTINEL-load-52-0123456789abcdefXYZ"
TAIL = KEY[-8:]
INT_KEY = 9876543210987654321
FLOAT_KEY = 98765.43219876  # a key typed as a number YAML reads as a float: only "on a secret field" masks it
_PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
             "default_model": "test-model", "available_models": ["test-model"]}

CASES = {  # what config.yaml holds after `provider:`; each fails validation with the key inside
    "must-differ": {"proposer": {"base_url": "http://p/v1", "model": "test-model", "api_key": KEY}},
    "field-missing": {"proposer": {"base_url": "http://p/v1", "api_key": KEY}},
    "not-a-string": {"proposer": {"base_url": "http://p/v1", "model": "p2", "api_key": INT_KEY}},
    "a-float": {"proposer": {"base_url": "http://p/v1", "model": "p2", "api_key": FLOAT_KEY}},
    "a-float-beside-a-missing-field": {"proposer": {"base_url": "http://p/v1", "api_key": FLOAT_KEY}},
    "misspelled-key": {"proposer": {"base_url": "http://p/v1", "model": "p2", "api_kye": KEY}},
    "removed-plugin": {"gone_plugin": {"enabled": True, "api_key": KEY}},
}
COMMANDS = {
    "doctor": ("doctor",),
    "validate": ("validate",),
    "start": ("start", "--no-input"),
    "components-list": ("components", "list"),
    "components-list-json": ("components", "list", "--json"),
    "components-get": ("components", "get", "proposer.model"),
    "model": ("model",),
    "config-show": ("config", "show"),
    "config-show-json": ("config", "show", "--json"),
}


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(components_cmd, "console", Console(width=400))

    def _never(*args, **kwargs):
        raise AssertionError("start reached a model listing before refusing the config")
    monkeypatch.setattr("localharness.cli.model_ops.list_live_models", _never)
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _home(tmp_path, case: str):
    g = tmp_path / "g"
    g.mkdir()
    cfg = {"version": "1", "provider": _PROVIDER, "org": {"audit_log_path": str(g / "audit.jsonl")},
           **CASES[case]}
    (g / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return g


@pytest.mark.parametrize("command", list(COMMANDS))
@pytest.mark.parametrize("case", list(CASES))
def test_a_refused_config_never_echoes_the_key(tmp_path, case, command) -> None:
    g = _home(tmp_path, case)
    result = runner.invoke(app, [*COMMANDS[command], "--config-dir", str(g)])

    texts = (result.stdout, result.stderr, repr(result.exception))
    assert result.exit_code != 0, result.output
    assert any("proposer" in t or "gone_plugin" in t for t in texts), result.output  # it is the refusal
    secret = {"not-a-string": str(INT_KEY), "a-float": str(FLOAT_KEY),
              "a-float-beside-a-missing-field": str(FLOAT_KEY)}.get(case, KEY)
    for where in texts:
        for piece in (secret, secret[-8:]):
            assert piece not in where, f"{piece!r} reached {command} ({case}):\n{where}"
    audit = g / "audit.jsonl"
    assert not audit.exists() or TAIL not in audit.read_text(encoding="utf-8")


def test_the_refusal_still_says_what_is_wrong(tmp_path) -> None:
    """Masked, not silenced: the field, the rule and the values that are not secret stay."""
    g = _home(tmp_path, "must-differ")
    out = " ".join(runner.invoke(app, ["validate", "--config-dir", str(g)]).output.split())
    assert "proposer.model must differ from provider.default_model" in out, out
    assert components_cmd.SECRET_MASK in out and "http://p/v1" in out, out

    g2 = tmp_path / "two"
    g2.mkdir()
    (g2 / "config.yaml").write_text(yaml.safe_dump(
        {"version": "1", "provider": {**_PROVIDER, "base_url": 123}}, sort_keys=False), encoding="utf-8")
    out = " ".join(runner.invoke(app, ["doctor", "--config-dir", str(g2)]).output.split())
    assert "provider.base_url" in out and "(got: 123)" in out, out  # a value that is not secret is shown


class SecagAgent(BaseModel):
    tok: SecretStr = SecretStr("")
    n: int = 0

    @model_validator(mode="before")
    @classmethod
    def _quotes_its_section(cls, data):  # a plugin validator whose message quotes its input
        if isinstance(data, dict) and data.get("n") == 4:
            raise ValueError(f"bad section {data!r}")
        return data


class Secag(Plugin):
    """holds an agent-level secret its own validator quotes"""

    manifest = PluginManifest(name="secag", version="0.1.0", kind="tools")
    AgentConfigModel = SecagAgent


def test_a_plugin_validator_quoting_its_agent_section_is_scrubbed(tmp_path, monkeypatch) -> None:
    """The loader validates a bundled plugin's agent-level section: the message it raises is
    scrubbed of that section's secrets, not only the value shown."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Secag,))
    g = tmp_path / "g"
    (g / "agents").mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump({"version": "1", "provider": _PROVIDER}), encoding="utf-8")
    (g / "agents" / "orchestrator.yaml").write_text(
        yaml.safe_dump({"name": "orchestrator", "role": "orchestrates"}), encoding="utf-8")
    (g / "overrides.yaml").write_text(
        yaml.safe_dump({"agent": {"secag": {"n": 4, "tok": KEY}}}, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigValidationError) as info:
        ConfigLoader(config_dir=g).load_agent("orchestrator")
    text = str(info.value)
    assert "secag" in text and "bad section" in text, text
    assert KEY not in text and TAIL not in text, text


def test_a_load_failure_carries_no_key_into_a_traceback(tmp_path) -> None:
    """A command that does not catch the error prints its traceback: the pydantic error it was
    raised from (input_value, the whole config) is not chained to it."""
    import traceback

    with pytest.raises(ConfigValidationError) as info:
        ConfigLoader(config_dir=_home(tmp_path, "must-differ")).load_harness()
    text = "".join(traceback.format_exception(info.value))
    assert "proposer.model must differ" in text
    for piece in (KEY, TAIL, "input_value"):
        assert piece not in text, text
