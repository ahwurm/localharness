"""#151: the two init posture questions — host tools and memory — and per-workspace persistence.

Global `init` asks each question once (TTY only; a scripted init is never prompted and keeps
today's defaults). A "no" writes ordinary org keys — `org.permissions.mode: read-only` /
`org.memory_enabled: false` — so a project can flip either one in its own
`.localharness/config.yaml` and the existing deep-merge layering persists the posture per
workspace: no new machinery, workspace wins per key, deny union untouched (MERG-02).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.provider.client import CapabilityResult
from localharness.provider.detector import DetectorResult

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_window_probe(monkeypatch):
    """Hermetic init: the live-window and runtime-identification probes make real HTTP calls."""
    import localharness.cli.init_cmd as init_cmd
    monkeypatch.setattr(init_cmd, "_detect_max_model_len", lambda *_: None)
    monkeypatch.setattr(init_cmd, "_identify_endpoint_provider", lambda *_: "unknown")


def _detector() -> DetectorResult:
    return DetectorResult(
        found=True, provider_type="ollama", base_url="http://localhost:11434",
        models=["test-model:7b"], suggested_model="test-model:7b", probe_duration_ms=42.0,
    )


def _capabilities() -> CapabilityResult:
    return CapabilityResult(
        tool_call_mode="native", context_window=128_000, supports_streaming=True,
        probe_duration_ms=10.0, probe_error=None, server_reached=True,
    )


def _run_init(tmp_path: Path, mock_client_cls, mock_detect) -> dict:
    mock_detect.return_value = _detector()
    client = MagicMock()
    client.detect_capabilities = AsyncMock(return_value=_capabilities())
    mock_client_cls.return_value = client
    result = runner.invoke(app, ["init", "--config-dir", str(tmp_path), "--force"])
    assert result.exit_code == 0, result.output
    return yaml.safe_load((tmp_path / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# The init questions
# ---------------------------------------------------------------------------


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_scripted_init_asks_nothing_and_keeps_defaults(mock_client_cls, mock_detect, tmp_path):
    """No TTY (CliRunner's stdin): zero posture prompts, today's defaults written."""
    cfg = _run_init(tmp_path, mock_client_cls, mock_detect)
    org = cfg["org"]
    assert org.get("memory_enabled", True) is True
    assert org["permissions"].get("mode") != "read-only"


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_no_answers_write_read_only_and_memory_off(mock_client_cls, mock_detect, tmp_path, monkeypatch):
    """'n' to both questions -> org.permissions.mode: read-only + org.memory_enabled: false,
    with the shipped deny defaults still stamped (the mode key never touches MERG-02)."""
    import localharness.cli.init_cmd as init_cmd
    fake_sys = MagicMock()
    fake_sys.stdin.isatty.return_value = True
    monkeypatch.setattr(init_cmd, "sys", fake_sys)
    fake_confirm = MagicMock()
    fake_confirm.ask.side_effect = [False, False]  # host tools: no; memory: no
    monkeypatch.setattr(init_cmd, "Confirm", fake_confirm)

    cfg = _run_init(tmp_path, mock_client_cls, mock_detect)
    org = cfg["org"]
    assert org["permissions"]["mode"] == "read-only"
    assert org["memory_enabled"] is False
    assert fake_confirm.ask.call_count == 2
    assert any("sudo" in p for p in org["permissions"]["deny_patterns"])


@patch("localharness.cli.init_cmd.detect_provider")
@patch("localharness.cli.init_cmd.LLMClient")
def test_yes_answers_keep_defaults(mock_client_cls, mock_detect, tmp_path, monkeypatch):
    import localharness.cli.init_cmd as init_cmd
    fake_sys = MagicMock()
    fake_sys.stdin.isatty.return_value = True
    monkeypatch.setattr(init_cmd, "sys", fake_sys)
    fake_confirm = MagicMock()
    fake_confirm.ask.side_effect = [True, True]
    monkeypatch.setattr(init_cmd, "Confirm", fake_confirm)

    cfg = _run_init(tmp_path, mock_client_cls, mock_detect)
    org = cfg["org"]
    assert org.get("memory_enabled", True) is True
    assert org["permissions"].get("mode") != "read-only"


# ---------------------------------------------------------------------------
# Per-workspace persistence (Alex 2026-09-21: "both simple questions by workspace persisted")
# ---------------------------------------------------------------------------


_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    global_dir = tmp_path / "global"
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    return global_dir, ws


def test_workspace_persists_memory_off(layers) -> None:
    """A project's .localharness/config.yaml turns memory off for THAT project only."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"memory_enabled": False}})
    harness = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    assert harness.org.memory_enabled is False


def test_workspace_wins_both_directions_on_memory(layers) -> None:
    """Workspace wins per key: a memory-off machine can host a memory-on project."""
    global_dir, ws = layers
    _write_yaml(global_dir / "config.yaml", {**_MINIMAL, "org": {"memory_enabled": False}})
    _write_yaml(ws / "config.yaml", {"org": {"memory_enabled": True}})
    harness = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    assert harness.org.memory_enabled is True


def test_workspace_persists_read_only_mode_and_keeps_deny_union(layers) -> None:
    """A workspace can pin read-only sessions for its project; the mode key rides the
    normal per-key merge and the shipped deny defaults survive untouched."""
    global_dir, ws = layers
    _write_yaml(ws / "config.yaml", {"org": {"permissions": {"mode": "read-only"}}})
    harness = ConfigLoader(config_dir=global_dir, local_config_dir=ws).load_harness()
    assert harness.org.permissions.mode == "read-only"
    assert any("sudo" in p for p in harness.org.permissions.deny_patterns)


def test_no_workspace_reads_global_posture(layers) -> None:
    global_dir, _ws = layers
    _write_yaml(
        global_dir / "config.yaml",
        {**_MINIMAL, "org": {"memory_enabled": False, "permissions": {"mode": "read-only"}}},
    )
    harness = ConfigLoader(config_dir=global_dir).load_harness()
    assert harness.org.memory_enabled is False
    assert harness.org.permissions.mode == "read-only"
