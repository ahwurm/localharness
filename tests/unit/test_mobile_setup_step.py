"""mobile's (the `mobile` plugin's) setup step: one setting, mobile.public_url, the address the pairing QR
sends a phone to. It is machine-level only (a project folder must never choose where a phone is
pointed) and `localharness mobile` uses it when --public-url is omitted. The bind address and
--allow-unsafe-bind stay flags of `localharness mobile`, never settings."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from localharness.cli import mobile_cmd
from localharness.cli.mobile_plugin import MobileConfig, MobilePlugin
from localharness.config.loader import ConfigLoader
from localharness.plugins import discovery
from localharness.plugins.api import AGENT_PROMPT_PLACEHOLDER
from localharness.plugins.resolve import resolve

_CONFIG = {  # port 9 (discard): nothing ever reaches a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}


def _g(tmp_path: Path, overrides: dict | None = None) -> Path:
    g = tmp_path / "g"
    g.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    if overrides is not None:
        (g / "overrides.yaml").write_text(yaml.safe_dump(overrides), encoding="utf-8")
    return g


def test_the_one_setting_is_the_phone_address() -> None:
    assert MobilePlugin.ConfigModel is MobileConfig
    assert set(MobileConfig.model_fields) == {"public_url"}  # never host, port or allow_unsafe_bind


def test_the_address_is_cleaned_and_must_be_http() -> None:
    assert MobileConfig(public_url=" https://spark.example.ts.net/ ").public_url == "https://spark.example.ts.net"
    assert MobileConfig(public_url="").public_url == "" and MobileConfig().public_url == ""
    with pytest.raises(ValidationError, match="mobile.public_url must start with http:// or https://"):
        MobileConfig(public_url="spark.local")


def test_the_manifest_asks_the_phone_address_and_names_the_next_step() -> None:
    m = MobilePlugin.manifest
    [field] = m.setup
    assert (field.key, field.secret, field.default) == ("public_url", False, "")
    assert m.next_steps == "Run `localharness mobile`, then scan its pairing QR with your phone."
    assert "tailscale serve --bg 8765" in m.agent_prompt
    assert "do not pass --allow-unsafe-bind" in m.agent_prompt
    assert not AGENT_PROMPT_PLACEHOLDER.search(m.agent_prompt)


def test_a_project_cannot_choose_where_a_phone_is_pointed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [])
    g, ws = tmp_path / "g", tmp_path / "proj" / ".localharness"
    g.mkdir()
    ws.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(
        {**_CONFIG, "mobile": {"public_url": "https://mine.example.ts.net"}}), encoding="utf-8")
    (ws / "config.yaml").write_text(yaml.safe_dump(
        {"mobile": {"public_url": "https://elsewhere.example"}}), encoding="utf-8")
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)
    loader.load_harness()

    res = resolve(loader, extra_installed=lambda e: True)

    assert res.settings["mobile"].config.public_url == "https://mine.example.ts.net"
    dropped = [w for w in res.warnings if w.startswith("ignoring mobile.public_url")]
    assert len(dropped) == 1 and str(ws / "config.yaml") in dropped[0], res.warnings


def test_the_saved_address_is_read_back(tmp_path) -> None:
    g = _g(tmp_path, {"mobile": {"enabled": True, "public_url": "https://spark.example.ts.net"}})
    assert mobile_cmd._saved_public_url(str(g)) == "https://spark.example.ts.net"


def test_nothing_saved_reads_as_none(tmp_path) -> None:
    assert mobile_cmd._saved_public_url(str(_g(tmp_path))) is None
    assert mobile_cmd._saved_public_url(str(_g(tmp_path / "x", {"mobile": {"public_url": ""}}))) is None


@pytest.mark.parametrize("broken", ["config", "value"])
def test_a_config_that_cannot_be_read_reads_as_none(tmp_path, broken) -> None:
    """The guess stands: no config yet, an unreadable one, or a saved value that does not validate."""
    g = _g(tmp_path, {"mobile": {"public_url": "spark.local"}} if broken == "value" else None)
    if broken == "config":
        (g / "config.yaml").write_text("provider: [unclosed\n", encoding="utf-8")
    assert mobile_cmd._saved_public_url(str(g)) is None
    assert mobile_cmd._saved_public_url(str(tmp_path / "no-such-dir")) is None
