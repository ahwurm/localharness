"""49-02: the additive plugin-API seams the dispatch plugin needs, landed before it exists.

Manifest-declared channel names, the enablement-aware accepted set, a doctor `warn` level that is
shown but never counted as a failure, `SetupField.secret`, and PLUGIN_API_VERSION still "1".
"""
from __future__ import annotations

import io

import pytest
import yaml
from rich.console import Console

from localharness.cli import doctor_cmd
from localharness.config.loader import ConfigLoader
from localharness.plugins import builtin, discovery
from localharness.plugins.api import PLUGIN_API_VERSION, Check, Plugin, PluginManifest, SetupField
from localharness.plugins.channels import accepted_channels, plugin_channel_names
from localharness.plugins.lifecycle import DoctorRow
from localharness.plugins.resolve import resolve
from localharness.cli.web_plugin import WebPlugin


class _MultiChannel(Plugin):
    manifest = PluginManifest(name="multichat", version="0.1.0", kind="channel",
                              channels=("alpha", "beta"))


def _global(tmp_path, extra: dict | None = None):
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump({"version": "1", "provider": {
        "provider_type": "vllm", "base_url": "http://localhost:8000/v1",
        "default_model": "m"}, **(extra or {})}), encoding="utf-8")
    return g


@pytest.fixture
def multichat(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (WebPlugin, _MultiChannel))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])


def test_plugin_api_version_stays_1():
    assert PLUGIN_API_VERSION == "1"


def test_every_bundled_manifest_declares_no_channels():
    for cls in builtin.BUILTIN_PLUGINS:
        assert cls.manifest.channels == (), cls.manifest.name


def test_manifest_channels_replace_the_plugin_name(multichat):
    names = plugin_channel_names()
    assert {"alpha", "beta", "web"} == names
    assert "multichat" not in names


def test_web_still_contributes_exactly_web(monkeypatch):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (WebPlugin,))
    assert plugin_channel_names() == {"web"}


def test_accepted_channels_follows_the_plan(tmp_path, multichat):
    on = accepted_channels(resolve(ConfigLoader(config_dir=_global(tmp_path))))
    assert {"alpha", "beta", "terminal", "acp"} <= on


def test_accepted_channels_drops_a_channel_plugin_turned_off(tmp_path, multichat):
    g = _global(tmp_path, {"multichat": {"enabled": False}})
    res = resolve(ConfigLoader(config_dir=g))
    assert {e.name: e.state for e in res.plan.entries}["multichat"] == "off"
    acc = accepted_channels(res)
    assert not {"alpha", "beta"} & acc
    assert {"terminal", "acp"} <= acc


def test_a_warn_check_is_shown_and_never_a_failure(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(doctor_cmd, "console", Console(file=buf, width=200, color_system=None))
    failures: list[str] = []
    row = DoctorRow(name="p", state="on", detail="",
                    checks=(Check(name="x", status="warn", detail="d", hint="h"),))
    doctor_cmd.print_plugin_row(row, failures)
    out = buf.getvalue().splitlines()
    assert out[0] == "⚠ x: d"
    assert out[1].strip() == "h"
    assert failures == []


def test_setup_field_secret_is_optional():
    assert SetupField(key="k", prompt="p").secret is False
    assert SetupField(key="k", prompt="p", secret=True).secret is True
