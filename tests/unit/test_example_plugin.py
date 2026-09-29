"""The example plugin (examples/plugin-template/) is the fixture every plugin proof uses, so it cannot
drift from the plugin API unnoticed (DOCS-05). Its entry point is looked up through the UNMOCKED
importlib.metadata: the dev environment must install it as a real distribution (`uv sync --extra
dev`), and a missing install FAILS here, never skips — ENAB-06's proof that listing a plugin imports
nothing needs a real distribution, not a mocked entry point."""
from __future__ import annotations

import importlib
import inspect
import json
import os
import struct
import subprocess
import sys
import zlib
from importlib.metadata import entry_points, version
from pathlib import Path

import pytest
import typer
from packaging.specifiers import SpecifierSet
from pydantic import ValidationError
from typer.testing import CliRunner

import localharness
from localharness.core.events import ArtifactRef
from localharness.plugins.api import Plugin, PluginContext, PluginPaths

PKG = "localharness_plugin_example"
EXPECTED_ENTRY_POINT = ("example", f"{PKG}:ExamplePlugin", "localharness-plugin-example", "0.1.0")
SENTINEL_ENV = "LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL"


def _entry_point():
    found = [ep for ep in entry_points(group="localharness.plugins") if ep.name == "example"]
    if not found:
        pytest.fail("the example plugin is not installed — run `uv sync --extra dev`")
    return found[0]


def _target(spec: str):
    """Resolve a descriptor target "package.module:attr[.sub]"."""
    module, _, attr = spec.partition(":")
    obj = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.fixture(scope="module")
def plugin_cls():
    """The plugin class, loaded through its real entry point. The modules it imported leave
    sys.modules afterwards, so tests that measure import state start from where they were."""
    before = set(sys.modules)
    yield _entry_point().load()
    for name in set(sys.modules) - before:
        if name == PKG or name.startswith(PKG + "."):
            del sys.modules[name]


def _ctx(plugin_cls, tmp_path: Path, artifact_dir: Path | None, **agent) -> PluginContext:
    return PluginContext(
        bus=None, tools=None, hooks=None, config=plugin_cls.ConfigModel(),
        agent_config=plugin_cls.AgentConfigModel(**agent),
        paths=PluginPaths(global_config_dir=tmp_path / "cfg", workspace=None,
                          state_dir=tmp_path / "state", artifact_dir=artifact_dir),
        llm=None)


def _root(tmp_path: Path) -> Path:
    return tmp_path / "state" / "artifacts" / "example"


def _png(data: bytes) -> tuple[tuple[int, ...], bytes]:
    """(IHDR fields, inflated image data), read by the PNG spec rather than by the code under test,
    every chunk's CRC checked."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF, tag
        chunks.append((tag, body))
        pos += 12 + length
    assert [tag for tag, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    return struct.unpack(">IIBBBBB", chunks[0][1]), zlib.decompress(chunks[1][1])


def test_the_unmocked_metadata_resolves_the_installed_distribution():
    ep = _entry_point()
    assert (ep.name, ep.value, ep.dist.name, ep.dist.version) == EXPECTED_ENTRY_POINT


def test_the_entry_point_loads_a_plugin_that_matches_its_distribution(plugin_cls):
    assert issubclass(plugin_cls, Plugin)
    assert plugin_cls.manifest.name == _entry_point().name  # the copy checklist's rule
    assert plugin_cls.manifest.version == version("localharness-plugin-example")


def test_the_manifest_declares_one_of_each_contribution(plugin_cls):
    m = plugin_cls.manifest
    assert (m.name, m.version, m.kind, m.requires_localharness) == ("example", "0.1.0", "tools", ">=0.15,<1")
    assert [(d.name, d.help, d.target) for d in m.cli] == [
        ("example", "Show what the example plugin does.", f"{PKG}.cli:app")]
    assert [(d.name, d.target) for d in m.slash] == [("/example", f"{PKG}.slash:run")]
    assert plugin_cls.wants_artifacts is True
    assert plugin_cls.ConfigModel().color == "#4a90d9"
    assert plugin_cls.AgentConfigModel().size == 8


def test_requires_localharness_admits_this_release_and_the_next(plugin_cls):
    spec = SpecifierSet(plugin_cls.manifest.requires_localharness)
    assert "0.15.0" in spec and "0.16.3" in spec
    assert localharness.__version__ in spec


def test_the_cli_target_is_a_typer_app_that_runs_as_one_command(plugin_cls):
    app = _target(plugin_cls.manifest.cli[0].target)
    assert isinstance(app, typer.Typer)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 0, result.output
    assert "example plugin 0.1.0" in result.output


async def test_the_slash_target_is_a_coroutine_taking_ctx_and_args(plugin_cls, tmp_path):
    fn = _target(plugin_cls.manifest.slash[0].target)
    assert inspect.iscoroutinefunction(fn)
    assert list(inspect.signature(fn).parameters) == ["ctx", "args"]
    reply = await fn(_ctx(plugin_cls, tmp_path, _root(tmp_path)), "")
    assert reply == "example plugin: swatches render in #4a90d9 at 8px"


async def test_the_tool_declares_all_four_axes_honestly(plugin_cls, tmp_path):
    tools = await plugin_cls().tools(_ctx(plugin_cls, tmp_path, _root(tmp_path)))
    assert len(tools) == 1
    schema = tools[0].info()
    assert schema.name == "example_swatch"
    assert {"ingest", "host", "result_origin", "gate_family"} <= schema.model_fields_set
    assert (schema.ingest, schema.host, schema.result_origin, schema.gate_family) == (
        "none", "safe", "trusted", None)
    assert schema.source_plugin is None  # core stamps provenance at registration, not the plugin


async def test_the_tool_writes_one_png_into_the_core_computed_root(plugin_cls, tmp_path):
    root = _root(tmp_path)
    (tool,) = await plugin_cls().tools(_ctx(plugin_cls, tmp_path, root, size=4))
    result = await tool.run()
    assert result.success, result.error
    ref = ArtifactRef.model_validate(result.metadata["artifact"])
    assert (ref.plugin, ref.kind, ref.mime) == ("example", "image", "image/png")
    assert result.output == f"Rendered a 4x4 #4a90d9 swatch: artifact {ref.id}"
    assert [p.name for p in root.iterdir()] == [f"{ref.id}.png"]
    ihdr, pixels = _png((root / f"{ref.id}.png").read_bytes())
    assert ihdr == (4, 4, 8, 2, 0, 0, 0)  # 4x4, 8-bit truecolor, no interlace
    assert pixels == (b"\x00" + bytes.fromhex("4a90d9") * 4) * 4


async def test_the_tool_refuses_without_an_artifact_root(plugin_cls, tmp_path):
    (tool,) = await plugin_cls().tools(_ctx(plugin_cls, tmp_path, None))
    result = await tool.run()
    assert not result.success
    assert "no artifact directory" in result.error
    assert not (tmp_path / "state").exists()


def test_doctor_passes_on_a_writable_root_without_creating_it(plugin_cls, tmp_path):
    root = _root(tmp_path)
    checks = plugin_cls().doctor(_ctx(plugin_cls, tmp_path, root))
    assert [(c.name, c.status, c.detail) for c in checks] == [
        ("example", "pass", f"swatches render in #4a90d9; artifacts go to {root}")]
    assert not (tmp_path / "state").exists()


def test_doctor_fails_without_an_artifact_root(plugin_cls, tmp_path):
    (check,) = plugin_cls().doctor(_ctx(plugin_cls, tmp_path, None))
    assert (check.name, check.status) == ("example", "fail")
    assert "wants_artifacts" in check.hint


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only directory")
def test_doctor_fails_when_the_nearest_existing_parent_is_read_only(plugin_cls, tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        (check,) = plugin_cls().doctor(_ctx(plugin_cls, tmp_path, locked / "artifacts" / "example"))
    finally:
        locked.chmod(0o700)
    assert (check.name, check.status) == ("example", "fail")
    assert str(locked) in check.hint


def test_settings_are_validated(plugin_cls):
    for bad in ({"color": "red"}, {"color": "#4a90d9\n"}, {"colour": "#4a90d9"}):
        with pytest.raises(ValidationError):
            plugin_cls.ConfigModel(**bad)
    for bad in ({"size": 0}, {"size": 257}, {"sise": 8}):
        with pytest.raises(ValidationError):
            plugin_cls.AgentConfigModel(**bad)


_PROBE = f"""
import json, os, sys
from importlib.metadata import entry_points
from pathlib import Path
sentinel = Path(os.environ["{SENTINEL_ENV}"])
mods = lambda: sorted(m for m in sys.modules if m.startswith("{PKG}"))
listed = [(e.name, e.value, e.dist.name, e.dist.version) for e in entry_points(group="localharness.plugins")]
after_listing = [sentinel.exists(), mods()]
import {PKG}
print(json.dumps({{"listed": listed, "after_listing": after_listing, "after_import": [sentinel.exists(), mods()]}}))
"""


def test_listing_imports_nothing_and_importing_writes_the_sentinel(tmp_path):
    _entry_point()
    sentinel = tmp_path / "imported"
    run = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=120,
                         env={**os.environ, SENTINEL_ENV: str(sentinel)})
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout)
    assert list(EXPECTED_ENTRY_POINT) in seen["listed"]
    assert seen["after_listing"] == [False, []]  # metadata read, nothing imported, no sentinel
    # importing the package loads the plugin class only — never the tool, the CLI or the slash module
    assert seen["after_import"] == [True, [PKG, f"{PKG}.plugin"]]
    assert sentinel.read_text(encoding="utf-8") == "imported\n"
