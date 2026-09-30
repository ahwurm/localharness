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
