"""Plugin discovery reads METADATA only (ENAB-06): listing a plugin never imports it.

Entry points are mocked here, the idiom CONTEXT allows for loader unit tests only; the proof on a
real installed distribution is the example plugin's. Folder plugins are real files under a tmp
GLOBAL config dir, and each one appends a line to a sentinel file when it is imported, so "not
imported" is something the test observes rather than infers.
"""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import pytest

from localharness.plugins.api import Plugin
from localharness.plugins.discovery import (
    PLUGIN_ENTRY_POINT_GROUP,
    DiscoveredPlugin,
    discover,
    import_target,
    load_plugin_class,
)

_FOO = """
from localharness.plugins.api import Plugin, PluginManifest

class FooPlugin(Plugin):
    manifest = PluginManifest(name="foo", version="0.1.0", kind="tools")

plugin = FooPlugin
"""


class _Dist:
    def __init__(self, name: str, version: str) -> None:
        self.name, self.version = name, version


class _EntryPoint:
    """load() is a tripwire: it records the call AND raises, so a discovery that swallowed the
    AssertionError is still caught by the record."""

    def __init__(self, name: str, value: str, dist: _Dist | None, loads: list[str]) -> None:
        self.name, self.value, self.dist, self._loads = name, value, dist, loads

    def load(self):
        self._loads.append(self.name)
        raise AssertionError("discovery must not load")


@pytest.fixture
def entry_points(monkeypatch):
    """Patch importlib.metadata.entry_points. Yields (set_eps, groups_asked, loads)."""
    state: dict[str, list] = {"eps": []}
    groups: list[str | None] = []
    loads: list[str] = []

    def fake(*, group: str | None = None, **_kw):
        groups.append(group)
        return list(state["eps"]) if group == PLUGIN_ENTRY_POINT_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", fake)

    def set_eps(*specs):
        state["eps"] = [_EntryPoint(n, v, d, loads) for n, v, d in specs]

    return set_eps, groups, loads


@pytest.fixture(autouse=True)
def _forget_folder_plugins():
    yield
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def _folder_plugin(global_dir: Path, name: str, body: str, sentinel: Path) -> Path:
    folder = global_dir / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(
        f"with open({str(sentinel)!r}, 'a') as _f:\n    _f.write('imported\\n')\n" + body
    )
    return folder


# --------------------------------------------------------------------------- listing


def test_the_group_is_localharness_plugins() -> None:
    assert PLUGIN_ENTRY_POINT_GROUP == "localharness.plugins"


def test_entry_points_are_listed_from_metadata_and_never_loaded(entry_points, tmp_path) -> None:
    set_eps, groups, loads = entry_points
    set_eps(
        ("alpha", "alpha_pkg:AlphaPlugin", _Dist("lh-alpha", "1.2.0")),
        ("beta", "beta_pkg.sub:Outer.Beta", _Dist("lh-beta", "0.3.1")),
    )

    found = discover(tmp_path)

    assert found == [
        DiscoveredPlugin("alpha", "entry_point", "alpha_pkg:AlphaPlugin", "lh-alpha", "1.2.0"),
        DiscoveredPlugin("beta", "entry_point", "beta_pkg.sub:Outer.Beta", "lh-beta", "0.3.1"),
    ]
    assert loads == [], "discovery called .load() on an entry point"
    assert groups == [PLUGIN_ENTRY_POINT_GROUP]


def test_an_entry_point_without_a_distribution_is_still_listed(entry_points, tmp_path) -> None:
    set_eps, _, loads = entry_points
    set_eps(("gamma", "gamma_pkg:G", None))

    assert discover(tmp_path) == [DiscoveredPlugin("gamma", "entry_point", "gamma_pkg:G")]
    assert loads == []


def test_a_folder_plugin_is_listed_without_being_imported(entry_points, tmp_path) -> None:
    sentinel = tmp_path / "imported.txt"
    folder = _folder_plugin(tmp_path, "foo", _FOO, sentinel)
    modules_before = set(sys.modules)

    found = discover(tmp_path)

    assert found == [DiscoveredPlugin("foo", "folder", str(folder))]
    assert not sentinel.exists(), "listing a folder plugin ran its __init__.py"
    assert not [m for m in set(sys.modules) - modules_before if "folder_plugins" in m]


def test_entry_points_come_first_then_folders_by_name(entry_points, tmp_path) -> None:
    set_eps, _, _ = entry_points
    set_eps(("zz-ep", "z:Z", _Dist("lh-z", "1.0")), ("aa-ep", "a:A", _Dist("lh-a", "1.0")))
    sentinel = tmp_path / "s.txt"
    for name in ("zeta", "alpha"):
        _folder_plugin(tmp_path, name, _FOO, sentinel)

    assert [d.name for d in discover(tmp_path)] == ["zz-ep", "aa-ep", "alpha", "zeta"]


def test_only_folders_with_an_init_file_are_listed(entry_points, tmp_path) -> None:
    plugins = tmp_path / "plugins"
    (plugins / "no_init").mkdir(parents=True)
    (plugins / "no_init" / "code.py").write_text("raise SystemExit('never')\n")
    (plugins / "__pycache__").mkdir()
    (plugins / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"\x00")
    (plugins / "README.md").write_text("# plugins\n")
    (plugins / "init_is_a_dir" / "__init__.py").mkdir(parents=True)
    kept = _folder_plugin(tmp_path, "kept", _FOO, tmp_path / "s.txt")

    assert discover(tmp_path) == [DiscoveredPlugin("kept", "folder", str(kept))]


def test_no_plugins_folder_lists_entry_points_only(entry_points, tmp_path) -> None:
    set_eps, _, _ = entry_points
    set_eps(("alpha", "alpha_pkg:A", _Dist("lh-alpha", "1.0")))

    assert [d.source for d in discover(tmp_path / "does-not-exist")] == ["entry_point"]


def test_from_label_names_where_a_plugin_came_from(tmp_path) -> None:
    ep = DiscoveredPlugin("example", "entry_point", "pkg:Ex", "localharness-plugin-example", "0.1.0")
    folder = DiscoveredPlugin("foo", "folder", str(tmp_path / "plugins" / "foo"))

    assert ep.from_label == "pip: localharness-plugin-example 0.1.0"
    assert folder.from_label == f"folder: {tmp_path / 'plugins' / 'foo'}"


# --------------------------------------------------------------------------- importing


def test_load_plugin_class_imports_a_folder_plugin_without_touching_sys_path(
    entry_points, tmp_path
) -> None:
    sentinel = tmp_path / "imported.txt"
    _folder_plugin(tmp_path, "foo", _FOO, sentinel)
    [found] = discover(tmp_path)
    path_before = list(sys.path)

    cls = load_plugin_class(found)

    assert sentinel.read_text() == "imported\n"
    assert isinstance(cls, type) and issubclass(cls, Plugin)
    assert (cls.__name__, cls.manifest.name) == ("FooPlugin", "foo")
    assert sys.path == path_before


def test_a_folder_plugin_can_use_relative_imports(entry_points, tmp_path) -> None:
    folder = _folder_plugin(
        tmp_path, "rel-plug", "from .impl import RelPlugin as plugin\n", tmp_path / "s.txt"
    )
    (folder / "impl.py").write_text(_FOO.replace("FooPlugin", "RelPlugin"))
    [found] = discover(tmp_path)

    assert load_plugin_class(found).__name__ == "RelPlugin"


def test_a_folder_plugin_is_imported_once_per_process(entry_points, tmp_path) -> None:
    """The same as an installed package: a second load returns the same class and does not run
    the module again (module-level state survives; class identity holds across sessions)."""
    sentinel = tmp_path / "imported.txt"
    _folder_plugin(tmp_path, "foo", _FOO, sentinel)
    [found] = discover(tmp_path)

    first, second = load_plugin_class(found), load_plugin_class(found)

    assert first is second
    assert sentinel.read_text() == "imported\n"


def test_same_named_folders_in_two_config_dirs_are_different_modules(entry_points, tmp_path) -> None:
    one = _folder_plugin(tmp_path / "one", "foo", _FOO, tmp_path / "s1.txt")
    two = _folder_plugin(tmp_path / "two", "foo", _FOO, tmp_path / "s2.txt")

    first = load_plugin_class(DiscoveredPlugin("foo", "folder", str(one)))
    second = load_plugin_class(DiscoveredPlugin("foo", "folder", str(two)))

    assert first is not second
    assert (tmp_path / "s2.txt").exists()


def test_a_failed_folder_import_propagates_unchanged_and_leaves_no_module(
    entry_points, tmp_path
) -> None:
    sentinel = tmp_path / "imported.txt"
    _folder_plugin(tmp_path, "boom", "raise RuntimeError('boom at import')\n", sentinel)
    [found] = discover(tmp_path)

    for _ in range(2):  # a half-initialized module must not be handed back the second time
        with pytest.raises(RuntimeError, match="boom at import"):
            load_plugin_class(found)
    assert sentinel.read_text() == "imported\nimported\n"
    assert not [m for m in sys.modules if m.startswith("localharness_folder_plugins")]


def test_an_entry_point_import_error_propagates_unchanged() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_plugin_class(DiscoveredPlugin("x", "entry_point", "no_such_module_lh_xyz:Thing"))


@pytest.mark.parametrize(
    "binding", ["plugin = 42\n", "plugin = object\n", "not_the_name = 1\n"], ids=["int", "type", "unbound"]
)
def test_a_folder_that_does_not_bind_a_plugin_class_is_refused(entry_points, tmp_path, binding) -> None:
    _folder_plugin(tmp_path, "notplug", binding, tmp_path / "s.txt")
    [found] = discover(tmp_path)

    with pytest.raises(TypeError, match="does not name a localharness Plugin subclass"):
        load_plugin_class(found)


def test_an_entry_point_that_is_not_a_plugin_class_is_refused() -> None:
    found = DiscoveredPlugin("od", "entry_point", "collections:OrderedDict", "x", "1")

    with pytest.raises(TypeError, match="collections:OrderedDict does not name a localharness Plugin"):
        load_plugin_class(found)


def test_an_entry_point_plugin_class_is_imported_by_its_target() -> None:
    found = DiscoveredPlugin("p", "entry_point", "localharness.plugins.api:MemorySlotPlugin", "x", "1")

    from localharness.plugins.api import MemorySlotPlugin

    assert load_plugin_class(found) is MemorySlotPlugin


def test_import_target_resolves_a_nested_attribute_and_a_bare_module() -> None:
    import localharness.plugins.api as api

    assert import_target("localharness.plugins.api:Plugin.configure") is Plugin.configure
    assert import_target("localharness.plugins.api:PLUGIN_NAME_RE") is api.PLUGIN_NAME_RE
    assert import_target("localharness.plugins.api") is api
