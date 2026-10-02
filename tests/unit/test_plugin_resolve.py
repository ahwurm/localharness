"""The resolver: config + discovery + ENABLED-ONLY import + per-plugin settings validation.

ENAB-06: a plugin that is not enabled is never imported; each folder plugin here writes a sentinel
file when its module runs, so "never imported" is observed, not assumed. ENAB-02/SAFE-06: turning
on a plugin you installed is a machine-level (global) setting, while a bundled plugin's `enabled`
is layered like any key. ENAB-01/PAPI-11: invalid settings disable that plugin, never the harness.

Entry-point discovery is stubbed to none around the REAL folder scan, so the venv's installed
example plugin stays out of these tests — except the one test at the end that uses it on purpose.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, Field

from localharness.config.loader import ConfigLoader
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest
from localharness.plugins.resolve import PluginSettings, Resolution, resolve

_REAL_DISCOVER = discovery.discover
_MINIMAL = {
    "version": "1",
    "provider": {"provider_type": "vllm", "base_url": "http://localhost:8000/v1",
                 "default_model": "global-model"},
}


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


def write_folder_plugin(g: Path, name: str, *, manifest: str = "", attrs: str = "",
                        prelude: str = "", manifest_name: str | None = None) -> None:
    """`<g>/plugins/<name>/__init__.py`: writes `<sentinel dir>/<name>` when imported, then runs
    `prelude`; its ConfigModel has a hex `color` and a machine-level `url`."""
    folder = g / "plugins" / name
    folder.mkdir(parents=True)
    (folder / "__init__.py").write_text(textwrap.dedent(f'''\
        import os
        from pathlib import Path
        from pydantic import BaseModel, Field
        from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest

        (Path(os.environ["LH_TEST_SENTINEL_DIR"]) / "{name}").write_text("imported")
        {prelude}

        class Config(BaseModel):
            color: str = Field("#4a90d9", pattern=r"^#[0-9a-f]{{6}}$")
            url: str = Field("http://default.invalid", json_schema_extra=GLOBAL_ONLY)

        class AgentConfig(BaseModel):
            size: int = 8

        class MarkedAgentConfig(BaseModel):
            token: str = Field("", json_schema_extra=GLOBAL_ONLY)

        class ThePlugin(Plugin):
            """draws {name} swatches"""
            manifest = PluginManifest(name="{manifest_name or name}", version="0.1.0", kind="tools"{manifest})
            ConfigModel = Config
            AgentConfigModel = AgentConfig
            {attrs}

        plugin = ThePlugin
        '''), encoding="utf-8")


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """A global dir holding a minimal valid config.yaml, and an empty workspace `.localharness/`."""
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    _write_yaml(g / "config.yaml", _MINIMAL)
    return g, ws


@pytest.fixture(autouse=True)
def folder_scan_only(monkeypatch) -> None:
    monkeypatch.setattr("localharness.plugins.discovery.discover",
                        lambda global_config_dir: [f for f in _REAL_DISCOVER(global_config_dir)
                                                   if f.source == "folder"])


@pytest.fixture(autouse=True)
def sentinels(tmp_path: Path, monkeypatch):
    """The directory the folder plugins mark on import; their modules leave sys.modules after."""
    marks = tmp_path / "sentinels"
    marks.mkdir()
    monkeypatch.setenv("LH_TEST_SENTINEL_DIR", str(marks))
    yield marks
    for name in [m for m in sys.modules if m.startswith("localharness_folder_plugins")]:
        del sys.modules[name]


def resolved(g: Path, ws: Path, **kwargs) -> Resolution:
    return resolve(ConfigLoader(config_dir=g, local_config_dir=ws), **kwargs)


def bundle(monkeypatch, *classes: type[Plugin]) -> None:
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", classes)


# --------------------------------------------------------------------------- enabled-only import


def test_a_folder_plugin_that_is_not_enabled_is_available_and_never_imported(layers, sentinels, monkeypatch) -> None:
    bundle(monkeypatch)  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "config.yaml", {**_MINIMAL, "foo": {"color": "#000000"}})  # settings, no enable

    r = resolved(g, ws)

    entry = r.plan.entry("foo")
    assert (entry.state, entry.source, entry.summary) == (
        "available", f"folder: {g / 'plugins' / 'foo'}", "(not loaded)")
    assert not (sentinels / "foo").exists()
    assert (r.enabled, r.classes, r.settings, r.warnings) == ({"foo": False}, {}, {}, ())


def test_enabled_in_the_global_layer_it_is_imported_and_its_settings_validated(layers, sentinels, monkeypatch) -> None:
    bundle(monkeypatch)  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True}})

    r = resolved(g, ws)

    assert r.plan.entry("foo").state == "on" and r.plan.order == ("foo",)
    assert (sentinels / "foo").exists()
    assert r.classes["foo"].__name__ == "ThePlugin" and r.enabled == {"foo": True}
    config = r.settings["foo"].config
    assert type(config).__name__ == "Config"
    assert (config.color, config.url) == ("#4a90d9", "http://default.invalid")
    assert r.settings["foo"].agent_config.size == 8  # no agent: its model's defaults
    assert r.warnings == () and r.problems() == []


def test_a_workspace_cannot_turn_on_a_plugin_you_installed(layers, sentinels) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(ws / "config.yaml", {"foo": {"enabled": True}})

    r = resolved(g, ws)

    assert r.plan.entry("foo").state == "available" and r.enabled["foo"] is False
    assert not (sentinels / "foo").exists()
    assert [w for w in r.warnings if "foo.enabled" in w and str(ws / "config.yaml") in w
            and "only the global config may set it" in w]


def test_a_workspace_cannot_turn_off_a_plugin_you_enabled_either(layers, sentinels) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True}})
    _write_yaml(ws / "overrides.yaml", {"foo": {"enabled": False}})

    r = resolved(g, ws)

    assert r.plan.entry("foo").state == "on"
    dropped = [w for w in r.warnings if "foo.enabled" in w]
    assert len(dropped) == 1 and str(ws / "overrides.yaml") in dropped[0]  # said once


def test_a_project_can_switch_a_bundled_plugin_either_way(layers, monkeypatch) -> None:
    class Shipped(Plugin):
        """ships with it"""
        manifest = PluginManifest(name="shipped", version="1.0", kind="tools")

    class Image(Plugin):
        """makes pictures"""
        manifest = PluginManifest(name="image", version="1.0", kind="tools", enabled_by_default=False)

    bundle(monkeypatch, Shipped, Image)
    g, ws = layers
    before = resolved(g, ws)
    assert [(e.name, e.state) for e in before.plan.entries] == [("shipped", "on"), ("image", "off")]

    _write_yaml(ws / "config.yaml", {"shipped": {"enabled": False}, "image": {"enabled": True}})
    r = resolved(g, ws)

    assert [(e.name, e.state) for e in r.plan.entries] == [("shipped", "off"), ("image", "on")]
    assert r.enabled == {"shipped": False, "image": True} and r.warnings == ()
    assert r.classes == {"shipped": Shipped, "image": Image}
    assert r.settings["shipped"] == PluginSettings(None, None)  # off, and still validated


def test_a_plugin_you_installed_never_imports_under_a_name_it_cannot_have(layers, monkeypatch,
                                                                          sentinels) -> None:
    class Shipped(Plugin):
        """ships with it"""
        manifest = PluginManifest(name="shipped", version="1.0", kind="tools")

    bundle(monkeypatch, Shipped)
    g, ws = layers
    for name in ("shipped", "org", "Bad_Name"):
        write_folder_plugin(g, name)
    _write_yaml(g / "overrides.yaml", {"shipped": {"enabled": True}, "Bad_Name": {"enabled": True}})

    r = resolved(g, ws)

    states = [(e.name, e.bundled, e.state) for e in r.plan.entries]
    assert states == [("shipped", True, "on"), ("Bad_Name", False, "refused"),
                      ("org", False, "refused"), ("shipped", False, "refused")]
    assert list(sentinels.iterdir()) == []  # none of the three was imported
    assert r.classes == {"shipped": Shipped}


def test_a_class_that_names_another_plugin_is_refused_and_shows_nothing(layers, sentinels) -> None:
    """Its manifest says it is `other`: refused, and neither its class nor its settings appear
    under the name it was found as (components and `plugins info` read those)."""
    g, ws = layers
    write_folder_plugin(g, "exa", manifest_name="other")
    _write_yaml(g / "overrides.yaml", {"exa": {"enabled": True, "color": "#000000"}})

    r = resolved(g, ws)

    entry = r.plan.entry("exa")
    assert (entry.state, entry.reason) == (
        "refused", "it was found as 'exa' but its class names 'other' in its manifest")
    assert (sentinels / "exa").exists()  # enabled, so imported — then refused
    assert "exa" not in r.classes and "exa" not in r.settings


# --------------------------------------------------------------------------- settings


def test_invalid_settings_fail_the_plugin_not_the_harness(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "config.yaml", {**_MINIMAL, "foo": {"enabled": True, "color": "red"}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)

    r = resolve(loader)

    entry = r.plan.entry("foo")
    assert entry.state == "failed" and entry.reason.startswith("invalid settings — foo.color: ")
    assert "foo" not in r.settings and r.plan.order == ("web", "memory")  # web (46-02) and memory (47) are bundled and on by default
    assert r.problems() == [f"plugin foo: {entry.reason}"]
    assert not [w for w in r.warnings if "foo.color" in w]  # reported once, by problems()
    assert loader.load_harness().provider.default_model == "global-model"


def test_an_off_bundled_plugin_with_invalid_settings_stays_off_and_says_so(layers, monkeypatch) -> None:
    class Settings(BaseModel):
        size: int = 8

    class Image(Plugin):
        """makes pictures"""
        manifest = PluginManifest(name="image", version="1.0", kind="tools", enabled_by_default=False)
        ConfigModel = Settings

    bundle(monkeypatch, Image)
    g, ws = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "image": {"size": "huge"}})

    r = resolved(g, ws)

    assert (r.plan.entry("image").state, r.problems()) == ("off", [])
    assert [w for w in r.warnings if w.startswith("plugin image: invalid settings — image.size: ")]
    assert "image" not in r.settings


def test_a_machine_level_field_set_in_a_project_is_dropped_and_the_global_value_stands(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True, "url": "http://global.example"}})
    _write_yaml(ws / "config.yaml", {"foo": {"url": "http://project.example", "color": "#123456"}})

    r = resolved(g, ws)

    config = r.settings["foo"].config
    assert (config.url, config.color) == ("http://global.example", "#123456")
    assert [w for w in r.warnings if "foo.url" in w and str(ws / "config.yaml") in w]


def test_a_bundled_plugin_meets_the_same_machine_level_rule(layers, monkeypatch) -> None:
    class Settings(BaseModel):
        url: str = Field("", json_schema_extra=GLOBAL_ONLY)

    class Shipped(Plugin):
        """ships with it"""
        manifest = PluginManifest(name="shipped", version="1.0", kind="tools")
        ConfigModel = Settings

    bundle(monkeypatch, Shipped)
    g, ws = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "shipped": {"url": "http://global.example"}})
    _write_yaml(ws / "config.yaml", {"shipped": {"url": "http://project.example"}})

    r = resolved(g, ws)

    assert r.settings["shipped"].config.url == "http://global.example"
    assert [w for w in r.warnings if "shipped.url" in w and str(ws / "config.yaml") in w]


def test_validated_settings_never_alias_the_loaders_cached_config(layers, monkeypatch) -> None:
    """A plugin that edits its own settings object must not edit what `components` and the next
    resolve read (the merged section shares nested values with the loader's cache)."""
    class Settings(BaseModel):
        data: dict[str, Any] = {}

    class Shipped(Plugin):
        """ships with it"""
        manifest = PluginManifest(name="shipped", version="1.0", kind="tools")
        ConfigModel = Settings

    bundle(monkeypatch, Shipped)
    g, ws = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "shipped": {"data": {"nested": {"x": 1}}}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)

    resolve(loader).settings["shipped"].config.data["nested"]["x"] = 2

    assert loader.plugin_layers()["shipped"][0] == {"data": {"nested": {"x": 1}}}
    assert resolve(loader).settings["shipped"].config.data == {"nested": {"x": 1}}


def test_the_agent_level_settings_come_from_the_loaded_agent(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True}})
    _write_yaml(g / "agents" / "helper.yaml", {"name": "helper", "role": "helps", "foo": {"size": 4}})
    _write_yaml(g / "agents" / "broken.yaml", {"name": "broken", "role": "helps", "foo": {"size": "x"}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)
    loader.load_agent("helper")
    loader.load_agent("broken")

    assert resolve(loader, agent_name="helper").settings["foo"].agent_config.size == 4
    broken = resolve(loader, agent_name="broken").plan.entry("foo")
    assert broken.state == "failed" and broken.reason.startswith("invalid settings — agent.foo.size: ")


def test_a_machine_level_marker_on_agent_settings_is_refused(layers) -> None:
    """Agent-level sections carry no layer of origin (a project's agent file could set the field),
    so the marker cannot be honoured there: the plugin fails rather than trust it silently."""
    g, ws = layers
    write_folder_plugin(g, "marked", attrs="AgentConfigModel = MarkedAgentConfig")
    _write_yaml(g / "overrides.yaml", {"marked": {"enabled": True}})

    r = resolved(g, ws)

    entry = r.plan.entry("marked")
    assert entry.state == "failed" and "agent.marked.token" in entry.reason
    assert "machine-level" in entry.reason and "marked" not in r.settings


def test_a_plugin_that_takes_no_settings_refuses_a_section(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "plain", attrs="ConfigModel = None")
    _write_yaml(g / "overrides.yaml", {"plain": {"enabled": True, "color": "#000000"}})

    r = resolved(g, ws)

    assert (r.plan.entry("plain").state, r.plan.entry("plain").reason) == (
        "failed", "invalid settings — it takes no settings, but `plain:` sets color")


@pytest.mark.parametrize("config, warning", [
    ({"foo": {"enabled": "yes please"}}, "`foo.enabled` must be true or false"),
    ({"foo": True}, "must be a mapping of settings"),
])
def test_an_unreadable_enable_warns_and_keeps_a_plugin_you_installed_off(layers, sentinels,
                                                                         config, warning) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", config)

    r = resolved(g, ws)

    assert r.plan.entry("foo").state == "available" and not (sentinels / "foo").exists()
    assert [w for w in r.warnings if w.startswith("plugin foo: ") and warning in w]


def test_a_projects_broken_section_cannot_decide_what_the_machine_turned_on(layers, sentinels) -> None:
    """Not a mapping in the project: reported, the global layers decide `enabled`, and the plugin
    — imported, because the machine turned it on — fails for its invalid settings, said once."""
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True}})
    _write_yaml(ws / "config.yaml", {"foo": 1})

    r = resolved(g, ws)

    entry = r.plan.entry("foo")
    assert r.enabled["foo"] is True and (sentinels / "foo").exists()
    assert entry.state == "failed" and entry.reason == (
        f"invalid settings — `foo:` in {ws / 'config.yaml'} must be a mapping of settings, not int")
    assert r.problems() == [f"plugin foo: {entry.reason}"] and r.warnings == ()


def test_a_non_mapping_agent_section_is_invalid_settings(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "foo")
    _write_yaml(g / "overrides.yaml", {"foo": {"enabled": True}})
    _write_yaml(g / "agents" / "helper.yaml", {"name": "helper", "role": "helps", "foo": 5})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)
    loader.load_agent("helper")

    entry = resolve(loader, agent_name="helper").plan.entry("foo")

    assert (entry.state, entry.reason) == (
        "failed", "invalid settings — `agent.foo:` must be a mapping of settings, not int")


# --------------------------------------------------------------------------- containment, version


@pytest.mark.parametrize("prelude, reason", [
    ('raise RuntimeError("kaput")', "could not be imported: RuntimeError: kaput"),
    ("import sys; sys.exit(3)", "could not be imported: SystemExit: 3"),
])
def test_a_plugin_that_breaks_on_import_is_failed_and_the_resolver_returns(layers, prelude,
                                                                           reason) -> None:
    g, ws = layers
    write_folder_plugin(g, "boom", prelude=prelude)
    _write_yaml(g / "overrides.yaml", {"boom": {"enabled": True}})

    r = resolved(g, ws)

    assert (r.plan.entry("boom").state, r.plan.entry("boom").reason) == ("failed", reason)
    assert "boom" not in r.classes and "boom" not in r.settings


def test_requires_localharness_is_checked_against_the_version_given(layers) -> None:
    g, ws = layers
    write_folder_plugin(g, "future", manifest=', requires_localharness=">=9"')
    _write_yaml(g / "overrides.yaml", {"future": {"enabled": True}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)

    entry = resolve(loader, version="0.15.0").plan.entry("future")
    assert (entry.state, entry.reason) == ("skipped", "requires localharness >=9, this is 0.15.0")
    assert resolve(loader, version="9.1").plan.entry("future").state == "on"


def test_problems_is_one_line_per_plugin_that_will_not_load(layers, monkeypatch, sentinels) -> None:
    class Web(Plugin):
        """the phone app"""
        manifest = PluginManifest(name="web", version="1.0", kind="tools", requires_extra="web")

    class Image(Plugin):
        """makes pictures"""
        manifest = PluginManifest(name="image", version="1.0", kind="tools", enabled_by_default=False)

    bundle(monkeypatch, Web, Image)
    g, ws = layers
    write_folder_plugin(g, "boom", prelude='raise RuntimeError("kaput")')
    write_folder_plugin(g, "future", manifest=', requires_localharness=">=9"')
    write_folder_plugin(g, "idle")
    write_folder_plugin(g, "Bad_Name")
    _write_yaml(g / "overrides.yaml", {"boom": {"enabled": True}, "future": {"enabled": True}})

    r = resolved(g, ws, version="0.15.0", extra_installed=lambda extra: False)

    assert r.problems() == [
        "plugin web: install `localharness[web]` to use it",
        "plugin Bad_Name: 'Bad_Name' is not a valid plugin name (a lower-case letter, then up to 63 "
        "lower-case letters, digits, '_' or '-')",
        "plugin boom: could not be imported: RuntimeError: kaput",
        "plugin future: requires localharness >=9, this is 0.15.0",
    ]
    assert [(e.name, e.display) for e in r.plan.entries if e.name in ("image", "idle")] == [
        ("image", "off — turn on: localharness plugins enable image"),
        ("idle", "available — turn on: localharness plugins enable idle")]


# --------------------------------------------------------------------------- the real distribution


@pytest.fixture
def example_unimported(monkeypatch, tmp_path: Path):
    """The installed example plugin's modules out of sys.modules before and after, so its import
    sentinel fires if — and only if — this test's resolver imports it."""
    def purge() -> None:
        for name in [m for m in sys.modules if m.startswith("localharness_plugin_example")]:
            del sys.modules[name]

    purge()
    monkeypatch.setattr("localharness.plugins.discovery.discover", _REAL_DISCOVER)
    sentinel = tmp_path / "example-imported"
    monkeypatch.setenv("LOCALHARNESS_EXAMPLE_PLUGIN_SENTINEL", str(sentinel))
    yield sentinel
    purge()


def test_the_installed_example_plugin_is_available_until_enabled_then_on(layers,
                                                                         example_unimported) -> None:
    """Unmocked importlib.metadata. The version is NOT injected: the check reads this package's own
    version (0.15.0), not the venv's stale distribution metadata, which would skip the plugin."""
    g, ws = layers

    r = resolved(g, ws)
    entry = r.plan.entry("example")
    assert (entry.state, entry.source) == ("available", "pip: localharness-plugin-example 0.1.0")
    assert not example_unimported.exists()

    _write_yaml(g / "overrides.yaml", {"example": {"enabled": True}})
    r = resolved(g, ws)
    assert r.plan.entry("example").state == "on", r.plan.entry("example").display
    assert example_unimported.exists()
    assert r.settings["example"].config.color == "#4a90d9"
