"""`plugins enable image` is the guided setup: on a terminal it asks the ComfyUI address, writes it,
checks it once (the same probe doctor runs) and, if the check does not pass, prints the short setup
text and a prompt to paste into a coding agent, filled with the address typed and the GPU the
machine reports. Off a terminal it names the one `--set` to run.

The fake ComfyUI answers through generate_image_tool._TRANSPORT only; no test reaches a real server.
gpu_name is patched in every test (no test runs nvidia-smi). Every run passes an explicit
--config-dir."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from rich.console import Console
from typer.testing import CliRunner

from localharness.cli import plugins_cmd
from localharness.cli.app import app
from localharness.plugins import setup
from localharness.plugins.setup import AGENT_PROMPT_LEAD
from tests.unit.test_image_plugin import UNET, _full, fake_comfy
from tests.unit.test_plugin_sections import _INFO_KEYS

runner = CliRunner()
URL = "http://comfy.test"
_CONFIG = {  # port 9 (discard): nothing ever reaches a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model", "available_models": ["test-model"]},
}
NEXT = ("next step — give it the ComfyUI address: "
        "localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188")


@pytest.fixture(autouse=True)
def wide(monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    monkeypatch.setattr(plugins_cmd, "console", Console(width=400))
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")


@pytest.fixture
def g(tmp_path: Path) -> Path:
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


@pytest.fixture
def prompts(monkeypatch):
    calls: list[tuple] = []

    def prompt(text, default=None, **kw):
        calls.append((text, default))
        return URL

    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    return calls


def _enable(g: Path):
    return runner.invoke(app, ["plugins", "enable", "image", "--config-dir", str(g)])


def _overrides(g: Path):
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def test_on_a_terminal_it_asks_writes_and_checks(g, prompts, monkeypatch):
    fake_comfy(monkeypatch)
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert prompts == [("ComfyUI address", "http://127.0.0.1:8188")]
    assert _overrides(g) == {"image": {"enabled": True, "comfyui_url": URL}}
    assert f"✓ image: ComfyUI reachable at {URL} (template: qwen-image-2.1)" in result.output
    assert "coding agent" not in result.output


def test_comfyui_down_prints_the_setup_text_and_the_prompt(g, prompts, monkeypatch):
    fake_comfy(monkeypatch, stats="down")
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"image": {"enabled": True, "comfyui_url": URL}}
    for text in (f"✗ image: ComfyUI unreachable at {URL}", "Image needs ComfyUI running on this machine",
                 UNET, "paste this into your coding agent", "C_INCLUDE_PATH", "weight_dtype", "8188",
                 f"answers at {URL}"):
        assert text in result.output, text
    assert result.output.count("Image needs ComfyUI running on this machine") == 1, result.output


def test_a_missing_model_file_is_named_and_the_setup_text_follows(g, prompts, monkeypatch):
    info = _full()
    info["UNETLoader"]["input"]["required"]["unet_name"] = [[], {}]
    fake_comfy(monkeypatch, info)
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert f"✗ image: missing in ComfyUI: {UNET}" in result.output
    assert "Image needs ComfyUI running on this machine" in result.output


def test_off_a_terminal_it_names_the_next_step(g, monkeypatch):
    def prompt(*a, **kw):
        raise AssertionError("prompted")
    monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: False)
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert _overrides(g) == {"image": {"enabled": True}}
    assert NEXT in result.output


def test_info_shows_the_set_up_line_and_the_four_settings(g):
    text = runner.invoke(app, ["plugins", "info", "image", "--config-dir", str(g)]).output
    assert "set up:" in text
    assert "localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188" in text
    rows = {k: [ln for ln in text.splitlines() if k in ln and "--set" not in ln]
            for k in ("image.comfyui_url", "image.enabled", "image.timeout_s", "image.workflow")}
    for key, lines in rows.items():
        assert lines, key
        assert any("(machine-level only)" in ln for ln in lines) == (
            key in ("image.comfyui_url", "image.workflow")), key


def test_the_short_setup_text_is_six_lines_and_the_prompt_is_its_own_field():
    from localharness.tools.builtin.image_plugin import IMAGE_SETUP_HELP, ImagePlugin

    assert "\n\n" not in IMAGE_SETUP_HELP and len(IMAGE_SETUP_HELP.splitlines()) <= 6
    assert "coding agent" not in IMAGE_SETUP_HELP
    m = ImagePlugin.manifest
    for text in ("{comfyui_url}", "{machine}", "C_INCLUDE_PATH", "weight_dtype"):
        assert text in m.agent_prompt, text
    assert m.next_steps == "Once ComfyUI answers, the agent can make pictures with generate_image."


def test_the_prompt_carries_the_typed_address_and_the_gpu(g, prompts, monkeypatch):
    fake_comfy(monkeypatch, stats="down")
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert f"it answers at {URL}; keep it off the open internet." in result.output
    assert f"{URL}/system_stats answers" in result.output
    assert "This machine reports NVIDIA GB10." in result.output
    assert "  Once ComfyUI answers, the agent can make pictures with generate_image." in result.output


def test_with_no_gpu_reported_the_prompt_has_no_machine_sentence(g, prompts, monkeypatch):
    monkeypatch.setattr(setup, "gpu_name", lambda: None)
    fake_comfy(monkeypatch, stats="down")
    result = _enable(g)

    assert result.exit_code == 0, result.output
    assert AGENT_PROMPT_LEAD in result.output and "This machine reports" not in result.output


def test_info_prints_the_prompt_with_the_default_address(g):
    text = runner.invoke(app, ["plugins", "info", "image", "--config-dir", str(g)]).output
    assert AGENT_PROMPT_LEAD in text
    assert "http://127.0.0.1:8188/system_stats answers" in text  # the default: nothing is stored
    data = runner.invoke(app, ["plugins", "info", "image", "--json", "--config-dir", str(g)]).stdout
    assert set(json.loads(data)) == _INFO_KEYS | {"sections"}
