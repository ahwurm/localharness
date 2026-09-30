"""QA-03: plugins built on 0.15.0's documented plugin API vanished silently after the upgrade.

0.15.0's `start` loaded two kinds of plugin that nothing reads since 0.16: installed packages with a
`localharness.tools` or `localharness.hooks` entry point, and `plugins/<dir>/manifest.yaml` folders
in the machine's config dir. After the upgrade `plugins list` printed `[]` and `doctor` said "All
checks passed." — the user was never told a plugin they rely on had stopped loading.

This file proves that such a plugin is NAMED (which one, that it was written for the 0.15 plugin API,
that it is no longer loaded, where the migration note is) and that nothing of it is ever loaded.

The distributions here are REAL metadata: a `*.dist-info` on sys.path, read by the unmocked
importlib.metadata, exactly as pip leaves an installed package. Every legacy module writes a
sentinel file when it is imported, so "never loaded" is observed, not inferred.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader
from localharness.plugins import discovery
from localharness.plugins.resolve import resolve
from tests.unit.test_plugin_resolve import write_folder_plugin

_CONFIG = {  # the provider is the discard port: nothing here can reach a model
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1",
                 "default_model": "test-model"},
}
_EVERY_PART = ("0.15 plugin API", "no longer loaded")


@pytest.fixture(autouse=True)
def _forget_folder_plugins():
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _global(tmp_path: Path, name: str = "g") -> Path:
    """A machine config dir holding a minimal valid config.yaml."""
    g = tmp_path / name
    g.mkdir(parents=True)
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _marking_module(path: Path, sentinel: Path, *classes: str) -> None:
    """A module that writes `sentinel` the moment anything imports it."""
    path.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('imported')\n"
                    + "".join(f"class {c}:\n    pass\n" for c in classes), encoding="utf-8")


def _dist(site_dir: Path, name: str, version: str, groups: dict[str, dict[str, str]]) -> None:
    """`<site_dir>/<name>-<version>.dist-info/` as an installer leaves it: METADATA and
    entry_points.txt."""
    info = site_dir / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                   encoding="utf-8")
    (info / "entry_points.txt").write_text("".join(
        f"[{group}]\n" + "".join(f"{ep} = {target}\n" for ep, target in eps.items()) + "\n"
        for group, eps in groups.items()), encoding="utf-8")


def install_legacy_dist(site_dir: Path, sentinel: Path, monkeypatch) -> None:
    """lh-oldtool 0.1.0, written for 0.15.0: one `localharness.tools` and one `localharness.hooks`
    entry point, and its module, which marks `sentinel` if it is ever imported."""
    _dist(site_dir, "lh-oldtool", "0.1.0", {"localharness.tools": {"oldtool": "lh_oldtool:OldTool"},
                                            "localharness.hooks": {"oldhooks": "lh_oldtool:OldHooks"}})
    _marking_module(site_dir / "lh_oldtool.py", sentinel, "OldTool", "OldHooks")
    monkeypatch.syspath_prepend(str(site_dir))


@pytest.fixture
def legacy_dist(tmp_path: Path, monkeypatch):
    """The installed 0.15 tool pack; yields the sentinel its module writes when imported."""
    sentinel = tmp_path / "lh_oldtool-imported"
    install_legacy_dist(tmp_path / "site", sentinel, monkeypatch)
    yield sentinel
    sys.modules.pop("lh_oldtool", None)


def legacy_folder(config_dir: Path, name: str, sentinel: Path) -> Path:
    """`<config_dir>/plugins/<name>/` as 0.15.0 read it: a manifest.yaml naming `tools:HelloTool`
    and that tools.py (which marks `sentinel` if imported) — and no __init__.py. Returns the
    manifest's path."""
    folder = config_dir / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "manifest.yaml").write_text(yaml.safe_dump({
        "name": name, "version": "0.1.0", "description": "a 0.15 manifest plugin",
        "tools": [{"name": f"{name}_hello", "description": "say hello",
                   "entrypoint": "tools:HelloTool"}]}), encoding="utf-8")
    _marking_module(folder / "tools.py", sentinel, "HelloTool")
    return folder / "manifest.yaml"


def notices(warnings) -> list[str]:
    """The warnings that name a plugin written for the 0.15 API."""
    return [w for w in warnings if "0.15 plugin API" in w]


def assert_names_the_015_api(line: str) -> None:
    for part in (*_EVERY_PART, discovery.LEGACY_API_DOC):
        assert part in line, f"{part!r} is missing from the notice: {line}"


# --------------------------------------------------------------------------- what resolve() says


def test_a_distribution_built_for_the_015_api_is_named_and_never_loaded(tmp_path, legacy_dist) -> None:
    resolution = resolve(ConfigLoader(config_dir=_global(tmp_path)))

    lines = [w for w in resolution.warnings if "lh-oldtool" in w]
    assert len(lines) == 1, f"expected one line naming lh-oldtool: {resolution.warnings}"
    line = lines[0]
    for part in ("lh-oldtool 0.1.0", "`localharness.tools`", "`oldtool`", "`localharness.hooks`",
                 "`oldhooks`"):
        assert part in line, f"{part!r} is missing from the notice: {line}"
    assert_names_the_015_api(line)
    assert resolution.warnings == (line,), resolution.warnings
    # Never loaded: its module never ran, is not imported, and the plan has no row for it.
    assert not legacy_dist.exists(), "a 0.15 plugin's module was imported"
    assert "lh_oldtool" not in sys.modules
    assert not {e.name for e in resolution.plan.entries} & {"oldtool", "oldhooks", "lh-oldtool"}


def test_a_015_manifest_folder_is_named_and_never_loaded(tmp_path) -> None:
    g = _global(tmp_path)
    sentinel = tmp_path / "oldstyle-imported"
    manifest = legacy_folder(g, "oldstyle", sentinel)

    resolution = resolve(ConfigLoader(config_dir=g))

    assert len(resolution.warnings) == 1, resolution.warnings
    line = resolution.warnings[0]
    assert str(manifest) in line, line
    assert_names_the_015_api(line)
    assert not sentinel.exists(), "a 0.15 manifest plugin's tools.py was imported"
    assert "oldstyle" not in {e.name for e in resolution.plan.entries}


def test_a_folder_with_an_init_py_is_todays_api_and_gets_no_notice(tmp_path, monkeypatch) -> None:
    g = _global(tmp_path)
    marks = tmp_path / "marks"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    write_folder_plugin(g, "ported")  # binds `plugin`: a plugin of today's API ...
    (g / "plugins" / "ported" / "manifest.yaml").write_text("name: ported\nversion: 0.1.0\n")
    old = legacy_folder(g, "oldstyle", tmp_path / "oldstyle-imported")  # ... beside a 0.15 one

    resolution = resolve(ConfigLoader(config_dir=g))

    named = notices(resolution.warnings)
    assert len(named) == 1 and str(old) in named[0], named  # the scan ran and found the old one
    assert not any(str(g / "plugins" / "ported") in w for w in resolution.warnings)
    assert resolution.plan.entry("ported").state == "available"
    assert not (marks / "ported").exists()


def test_a_distribution_that_also_declares_todays_api_gets_no_notice(tmp_path, monkeypatch,
                                                                     legacy_dist) -> None:
    site_dir, sentinel = tmp_path / "site2", tmp_path / "bothapis-imported"
    _dist(site_dir, "lh-bothapis", "0.2.0", {"localharness.tools": {"bt": "lh_bothapis:T"},
                                             "localharness.plugins": {"bothapis": "lh_bothapis:P"}})
    _marking_module(site_dir / "lh_bothapis.py", sentinel, "T", "P")
    monkeypatch.syspath_prepend(str(site_dir))
    try:
        resolution = resolve(ConfigLoader(config_dir=_global(tmp_path)))
    finally:
        sys.modules.pop("lh_bothapis", None)

    named = notices(resolution.warnings)
    assert len(named) == 1 and "lh-oldtool 0.1.0" in named[0], named  # the control is named
    assert not any("lh-bothapis" in w for w in resolution.warnings), resolution.warnings
    assert resolution.plan.entry("bothapis").state == "available"  # listed under today's API
    assert not sentinel.exists()


def test_a_workspace_is_never_a_source(tmp_path) -> None:
    g = _global(tmp_path)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    old = legacy_folder(g, "oldstyle", tmp_path / "oldstyle-imported")
    in_ws = legacy_folder(ws, "wsold", tmp_path / "wsold-imported")

    resolution = resolve(ConfigLoader(config_dir=g, local_config_dir=ws))

    named = notices(resolution.warnings)
    assert len(named) == 1 and str(old) in named[0], named
    assert not any(str(in_ws.parent) in w for w in resolution.warnings), resolution.warnings


def test_the_check_is_one_metadata_scan(tmp_path, monkeypatch, legacy_dist) -> None:
    calls: list[tuple] = []
    real = importlib.metadata.entry_points

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", spy)
    found = discovery.legacy_notices(_global(tmp_path))

    assert calls == [((), {})], f"expected ONE unfiltered entry-point scan, got {calls}"
    assert len(found) == 1 and "lh-oldtool 0.1.0" in found[0], found
