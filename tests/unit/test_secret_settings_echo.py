"""49-04: a setting whose annotation is SecretStr is never echoed by the core settings writers —
`plugins enable --set`, `components set` (success, coerce/validation failure, JSON, the audit
event), `components list/get` — and a `secret=True` setup question is asked with hidden input. The
value written stays the raw string in the owner-only (0600) overlay, readable by the loader.

`secp` is a bundled-shaped plugin swapped into BUILTIN_PLUGINS; every run passes --config-dir."""
from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, SecretStr
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import components_cmd, plugins_cmd
from localharness.cli.app import app
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest, SetupField
from tests.unit.test_plugins_enable_setup import _CONFIG

SENTINEL = "SENTINEL-XYZ"
MASK = "**********"
runner = CliRunner()
_REAL_DISCOVER = discovery.discover


class SecpConfig(BaseModel):
    tok: SecretStr = Field(SecretStr(""), json_schema_extra=GLOBAL_ONLY)
    n: int = 0


class Secp(Plugin):
    """holds a secret"""

    manifest = PluginManifest(name="secp", version="0.1.0", kind="tools", enabled_by_default=False,
                              setup=(SetupField(key="tok", prompt="Token", secret=True),))
    ConfigModel = SecpConfig


@pytest.fixture(autouse=True)
def bundled(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Secp,))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(components_cmd, "console", Console(width=400))
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    cfg = dict(_CONFIG, org={"audit_log_path": str(g / "audit.jsonl")})
    (g / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return g


def _run(g: Path, *args: str):
    result = runner.invoke(app, [*args, "--config-dir", str(g)])
    assert SENTINEL not in result.output, result.output
    audit = g / "audit.jsonl"
    assert not audit.exists() or SENTINEL not in audit.read_text(encoding="utf-8")
    return result


def _overrides(g: Path):
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def test_enable_set_masks_and_stores_the_raw_string_owner_only(g) -> None:
    result = _run(g, "plugins", "enable", "secp", "--set", f"tok={SENTINEL}")
    assert result.exit_code == 0, result.output
    assert f"set secp.tok = '{MASK}'" in result.output
    assert _overrides(g)["secp"]["tok"] == SENTINEL  # a plain string the loader reads back
    assert stat.S_IMODE((g / "overrides.yaml").stat().st_mode) == 0o600


def test_enable_set_with_a_bad_sibling_fails_without_the_secret(g) -> None:
    result = _run(g, "plugins", "enable", "secp", "--set", f"tok={SENTINEL}", "--set", "n=notanint")
    assert result.exit_code == 2, result.output


def test_enable_set_validation_failure_is_scrubbed(g, monkeypatch) -> None:
    """A model-level validator's error text can carry the whole section, token included."""
    from pydantic import model_validator

    class Loud(SecpConfig):
        @model_validator(mode="before")
        @classmethod
        def _boom(cls, data):
            raise ValueError(f"bad section {data!r}")
    monkeypatch.setattr(Secp, "ConfigModel", Loud)
    result = _run(g, "plugins", "enable", "secp", "--set", f"tok={SENTINEL}")
    assert result.exit_code == 2 and "Validation failed for secp" in result.output, result.output
    assert MASK in result.output


def test_a_non_secret_setting_still_echoes_as_today(g) -> None:
    result = _run(g, "plugins", "enable", "secp", "--set", "n=3")
    assert result.exit_code == 0, result.output
    assert "set secp.n = 3" in result.output


def test_components_set_get_list_never_show_it(g) -> None:
    _run(g, "plugins", "enable", "secp")
    result = _run(g, "components", "set", "secp.tok", SENTINEL)
    assert result.exit_code == 0, result.output
    assert f"secp.tok = '{MASK}' (was: '{MASK}')" in result.output
    assert _overrides(g)["secp"]["tok"] == SENTINEL
    result = _run(g, "components", "set", "secp.tok", SENTINEL, "--json")
    assert result.exit_code == 0 and f'"after": "{MASK}"' in result.output, result.output
    result = _run(g, "components", "set", "secp.n", "notanint")
    assert result.exit_code == 2, result.output
    for args in (("components", "list"), ("components", "list", "--json"),
                 ("components", "get", "secp.tok"), ("components", "get", "secp.tok", "--json"),
                 ("plugins", "info", "secp"), ("plugins", "info", "secp", "--json")):
        result = _run(g, *args)
        assert result.exit_code == 0, (args, result.output)


def test_components_set_validation_failure_is_scrubbed(g, monkeypatch) -> None:
    from pydantic import model_validator

    _run(g, "plugins", "enable", "secp", "--set", f"tok={SENTINEL}")

    class Loud(SecpConfig):
        @model_validator(mode="before")
        @classmethod
        def _boom(cls, data):  # only the value being set trips it: the catalogue still builds
            if isinstance(data, dict) and data.get("n") == 4:
                raise ValueError(f"bad section {data!r}")
            return data
    monkeypatch.setattr(Secp, "ConfigModel", Loud)
    result = _run(g, "components", "set", "secp.n", "4")
    assert result.exit_code == 2 and "Validation failed" in result.output, result.output


def test_the_secret_setup_question_hides_its_input(g, monkeypatch) -> None:
    calls: list[dict] = []

    def prompt(text, default=None, **kw):
        calls.append({"text": text, **kw})
        return SENTINEL
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    result = _run(g, "plugins", "enable", "secp")
    assert result.exit_code == 0, result.output
    assert calls == [{"text": "Token", "hide_input": True}]
    assert _overrides(g)["secp"]["tok"] == SENTINEL
