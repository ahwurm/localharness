"""`pending_preset` has one writer and one remover, both on the machine's GLOBAL overlay (D-31).

Offline: a tmp config dir, no server, no service manager.
"""
from __future__ import annotations

import pytest
import yaml

from localharness.cli import model_ops
from localharness.config.loader import ConfigLoader
from localharness.config.overlay import load_overlay


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".localharness"
    h.mkdir()
    (h / "config.yaml").write_text(yaml.safe_dump({
        "version": "1",
        "provider": {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1",
                     "default_model": "m", "available_models": ["m"]},
        "org": {"default_model": "m", "audit_log_path": str(h / "audit.jsonl")},
    }), encoding="utf-8")
    return h


def _overlay(home):
    return load_overlay(home / "overrides.yaml")


def test_the_writer_puts_the_name_in_the_global_overlay_and_the_harness(home):
    harness = ConfigLoader(config_dir=home).load_harness()
    model_ops.write_pending_preset(harness, "gemma", config_dir=home)
    assert _overlay(home) == {"pending_preset": "gemma"}
    assert harness.pending_preset == "gemma"
    assert ConfigLoader(config_dir=home).load_harness().pending_preset == "gemma"


def test_the_writer_keeps_every_other_key_and_the_agent_slice(home):
    (home / "overrides.yaml").write_text(yaml.safe_dump({
        "org": {"default_effort": "high"},
        "agent": {"temperature": 0.3},
        "active_endpoint": {"name": "x", "base_url": "http://127.0.0.1:9/v1", "model": "y"},
    }), encoding="utf-8")
    harness = ConfigLoader(config_dir=home).load_harness()
    model_ops.write_pending_preset(harness, "gemma", config_dir=home)
    overlay = _overlay(home)
    assert overlay["pending_preset"] == "gemma"
    assert overlay["agent"] == {"temperature": 0.3}
    assert overlay["org"] == {"default_effort": "high"}
    assert overlay["active_endpoint"]["model"] == "y"


@pytest.mark.parametrize("bad", ["a\nb", "", "   ", "a\x00b", "a\rb"])
def test_an_invalid_name_raises_and_writes_nothing(home, bad):
    harness = ConfigLoader(config_dir=home).load_harness()
    with pytest.raises(ValueError):
        model_ops.write_pending_preset(harness, bad, config_dir=home)
    assert not (home / "overrides.yaml").exists()
    assert harness.pending_preset is None


def test_the_writer_targets_the_global_layer_never_a_workspace(home, tmp_path, monkeypatch):
    work = tmp_path / "project" / ".localharness"
    work.mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "project")
    harness = ConfigLoader(config_dir=home).load_harness()
    model_ops.write_pending_preset(harness, "gemma", config_dir=home)
    assert _overlay(home)["pending_preset"] == "gemma"
    assert not (work / "overrides.yaml").exists()


def test_the_remover_drops_only_that_key(home):
    (home / "overrides.yaml").write_text(yaml.safe_dump({
        "pending_preset": "gemma", "org": {"default_effort": "high"}, "agent": {"temperature": 0.3},
    }), encoding="utf-8")
    harness = ConfigLoader(config_dir=home).load_harness()
    assert harness.pending_preset == "gemma"
    model_ops.clear_pending_preset(harness, config_dir=home)
    assert _overlay(home) == {"org": {"default_effort": "high"}, "agent": {"temperature": 0.3}}
    assert harness.pending_preset is None


def test_the_remover_writes_nothing_when_the_key_is_absent(home):
    harness = ConfigLoader(config_dir=home).load_harness()
    model_ops.clear_pending_preset(harness, config_dir=home)
    assert not (home / "overrides.yaml").exists()
    assert harness.pending_preset is None

    (home / "overrides.yaml").write_text("org:\n  default_effort: high\n", encoding="utf-8")
    before = (home / "overrides.yaml").read_bytes()
    model_ops.clear_pending_preset(harness, config_dir=home)
    assert (home / "overrides.yaml").read_bytes() == before
