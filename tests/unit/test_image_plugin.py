"""The image plugin: manifest, settings (machine-only URL and template, no env), availability, and
the one tool it contributes — its module imported only when a session asks for tools."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from localharness.config.plugin_sections import global_only_paths
from localharness.plugins.api import CliDescriptor, PluginContext, PluginPaths
from localharness.tools.builtin.image_plugin import ImageConfig, ImagePlugin


def _ctx(tmp_path: Path, **cfg) -> PluginContext:
    return PluginContext(
        bus=None, tools=None, hooks=None, config=ImageConfig(**cfg), agent_config=None,
        paths=PluginPaths(global_config_dir=tmp_path / "cfg", workspace=None,
                          state_dir=tmp_path / "state",
                          artifact_dir=tmp_path / "state" / "artifacts" / "image"),
        llm=None)


def test_the_manifest_is_off_by_default_and_mounts_generate_image():
    m = ImagePlugin.manifest
    assert (m.name, m.version, m.kind, m.enabled_by_default) == ("image", "0.1.0", "tools", False)
    assert m.cli == (CliDescriptor(name="generate-image",
                                   help="Make a picture with your local ComfyUI server.",
                                   target="localharness.cli.generate_image_cmd:app"),)
    assert ImagePlugin.wants_artifacts is True
    assert ImagePlugin.ConfigModel is ImageConfig
    assert ImagePlugin.__doc__.splitlines()[0] == "makes pictures with ComfyUI"


def test_the_url_and_the_template_are_machine_only():
    assert global_only_paths(ImageConfig) == {"comfyui_url", "workflow"}


def test_settings_defaults_and_validation():
    assert ImageConfig().timeout_s == 570.0
    assert ImageConfig().comfyui_url == "" and ImageConfig().workflow == ""
    assert ImageConfig(comfyui_url=" http://h:8188/ ").comfyui_url == "http://h:8188"
    for bad in ({"timeout_s": 0}, {"comfyui_url": "ftp://x"}, {"foo": 1}):
        with pytest.raises(ValidationError):
            ImageConfig(**bad)


async def test_unconfigured_until_the_url_is_set(tmp_path):
    assert await ImagePlugin().configure(_ctx(tmp_path)) == ("unconfigured", "image.comfyui_url")
    assert await ImagePlugin().configure(_ctx(tmp_path, comfyui_url="http://127.0.0.1:8188")) == "ready"


def test_importing_the_plugin_does_not_import_the_tool():
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, localharness.tools.builtin.image_plugin; "
         "print('httpx' in sys.modules, 'localharness.tools.builtin.generate_image_tool' in sys.modules)"],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


async def test_tools_contributes_generate_image(tmp_path):
    tools = await ImagePlugin().tools(_ctx(tmp_path, comfyui_url="http://comfy.test"))
    assert [t.info().name for t in tools] == ["generate_image"]


# --- doctor: probe() asks a fake ComfyUI through the one _TRANSPORT seam ------------------------

import httpx  # noqa: E402

from localharness.plugins.api import Check  # noqa: E402
from localharness.tools.builtin import generate_image_tool  # noqa: E402

URL = "http://comfy.test"
UNET, CLIP, VAE = ("qwen_image_2.1_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors",
                   "qwen_image_2.1_vae_bf16.safetensors")


def _full() -> dict:
    return {
        "UNETLoader": {"input": {"required": {"unet_name": [[UNET], {}],
                                              "weight_dtype": [["default", "fp8_e4m3fn"], {}]}}},
        "CLIPLoader": {"input": {"required": {"clip_name": [[CLIP], {}], "type": [["qwen_image"], {}],
                                              "device": [["default", "cpu"], {}]}}},
        "VAELoader": {"input": {"required": {"vae_name": ["COMBO", {"options": [VAE, "pixel_space"]}]}}},
    }


def fake_comfy(monkeypatch, info: dict | None = None, stats=200) -> list[str]:
    """Route _TRANSPORT to a fake ComfyUI; returns the list of request paths it saw."""
    info = _full() if info is None else info
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if req.url.path == "/system_stats":
            if stats == "down":
                raise httpx.ConnectError("refused", request=req)
            return httpx.Response(stats, json={})
        cls = req.url.path.removeprefix("/object_info/")
        if cls in info:
            body = {} if info[cls] is None else {cls: info[cls]}
        else:
            body = {cls: {"input": {"required": {}}}}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(generate_image_tool, "_TRANSPORT", httpx.MockTransport(handler))
    return seen


def test_probe_all_present_passes(monkeypatch):
    seen = fake_comfy(monkeypatch)
    assert generate_image_tool.probe(URL, "") == [Check(
        name="image", status="pass", detail=f"ComfyUI reachable at {URL} (template: qwen-image-2.1)")]
    assert seen[0] == "/system_stats"
    assert all(p == "/system_stats" or (p.startswith("/object_info/") and p != "/object_info/")
               for p in seen)
    assert len(seen) == len(set(seen))  # each class asked at most once


def test_probe_unreachable_asks_nothing_else(monkeypatch):
    for stats in ("down", 500):
        seen = fake_comfy(monkeypatch, stats=stats)
        checks = generate_image_tool.probe(URL, "")
        assert len(checks) == 1
        c = checks[0]
        assert (c.status, c.detail) == ("fail", f"ComfyUI unreachable at {URL}")
        assert c.hint and "\n" not in c.hint
        assert not any(p.startswith("/object_info") for p in seen)


def test_probe_unusable_template_makes_no_request(monkeypatch):
    seen = fake_comfy(monkeypatch)
    checks = generate_image_tool.probe(URL, "/nonexistent.json")
    assert len(checks) == 1 and checks[0].status == "fail"
    assert checks[0].detail.startswith("workflow template unusable:")
    assert seen == []


def test_probe_names_a_missing_unet_file_and_its_folder(monkeypatch):
    info = _full()
    info["UNETLoader"]["input"]["required"]["unet_name"] = [["other.safetensors"], {}]
    fake_comfy(monkeypatch, info)
    checks = generate_image_tool.probe(URL, "")
    assert [c.status for c in checks] == ["pass", "fail"]
    assert checks[1].detail == f"missing in ComfyUI: {UNET}"
    assert "models/diffusion_models/" in checks[1].hint and "\n" not in checks[1].hint


def test_probe_reads_the_v3_combo_form(monkeypatch):
    info = _full()
    info["VAELoader"]["input"]["required"]["vae_name"] = ["COMBO", {"options": ["pixel_space"]}]
    fake_comfy(monkeypatch, info)
    fail = [c for c in generate_image_tool.probe(URL, "") if c.status == "fail"]
    assert len(fail) == 1 and VAE in fail[0].detail and "models/vae/" in fail[0].hint


def test_probe_names_a_node_comfyui_lacks(monkeypatch):
    fake_comfy(monkeypatch, {**_full(), "TextEncodeQwenImage21": None})
    fail = [c for c in generate_image_tool.probe(URL, "") if c.status == "fail"]
    assert any("ComfyUI has no TextEncodeQwenImage21 node" in c.detail for c in fail)


def test_probe_names_a_rejected_literal(monkeypatch):
    info = _full()
    info["UNETLoader"]["input"]["required"]["weight_dtype"] = [["fp8_e4m3fn"], {}]
    fake_comfy(monkeypatch, info)
    fail = [c for c in generate_image_tool.probe(URL, "") if c.status == "fail"]
    assert any("UNETLoader.weight_dtype=default" in c.detail for c in fail)


def test_probe_never_asks_for_the_whole_object_info(monkeypatch):
    info = _full()
    info["UNETLoader"]["input"]["required"]["unet_name"] = [[], {}]
    seen = fake_comfy(monkeypatch, info)
    generate_image_tool.probe(URL, "")
    assert "/object_info" not in seen
    classes = [p for p in seen if p.startswith("/object_info/")]
    assert len(classes) == len(set(classes)) == 8


def test_doctor_returns_what_probe_returns(monkeypatch, tmp_path):
    info = _full()
    info["CLIPLoader"]["input"]["required"]["clip_name"] = [[], {}]
    fake_comfy(monkeypatch, info)
    got = ImagePlugin().doctor(_ctx(tmp_path, comfyui_url=URL))
    assert got == generate_image_tool.probe(URL, "")
    assert got[1].detail == f"missing in ComfyUI: {CLIP}"
