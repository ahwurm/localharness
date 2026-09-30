"""Plugin settings leave the merged config BEFORE core validation; global-only fields stay global.

ENAB-01: every core model is extra="forbid", so a plugin's `<name>:` and `agent.<name>` sections are
split off first — and a key that is neither core nor a known plugin is still refused (the typo
guard). ENAB-02: a field a plugin marks GLOBAL_ONLY, set in a workspace layer, is dropped with a
warning naming the key and the file, and the global value stands.

Discovery is stubbed where a test needs a known plugin name, so the venv's installed plugins never
decide a result; the workspace-folder test runs the REAL folder scan.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import pytest
import yaml
from pydantic import BaseModel, Field

from localharness.config.loader import ConfigLoader, ConfigValidationError
from localharness.config.models import AgentConfig, HarnessConfig
from localharness.config.plugin_sections import (
    CORE_AGENT_KEYS,
    CORE_HARNESS_KEYS,
    PluginSectionError,
    global_only_paths,
    merge_plugin_layers,
    split_plugin_keys,
)
from localharness.plugins import builtin, discovery
from localharness.plugins.api import GLOBAL_ONLY, Plugin, PluginManifest
from localharness.plugins.discovery import DiscoveredPlugin

_MINIMAL = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://localhost:8000/v1",
        "default_model": "global-model",
    },
}
_FILES = ("g/config.yaml", "g/overrides.yaml", "ws/config.yaml", "ws/overrides.yaml")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data), encoding="utf-8")


@pytest.fixture
def layers(tmp_path: Path) -> tuple[Path, Path]:
    """A global dir holding a minimal valid config.yaml, and an empty workspace `.localharness/`."""
    global_dir = tmp_path / "global"
    workspace_dir = tmp_path / "proj" / ".localharness"
    workspace_dir.mkdir(parents=True)
    _write_yaml(global_dir / "config.yaml", _MINIMAL)
    return global_dir, workspace_dir


@pytest.fixture
def known(monkeypatch):
    """Stub discovery to list exactly these plugin names (default: "example")."""

    def set_names(*names: str) -> None:
        monkeypatch.setattr(
            discovery, "discover",
            lambda global_config_dir: [DiscoveredPlugin(n, "entry_point", f"{n}_pkg:P", n, "1")
                                       for n in names],
        )

    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())  # discovery mechanics, not the bundled list: swap it out (as test_doctor_plugins does)
    set_names("example")
    return set_names


# --------------------------------------------------------------------------- harness level


def test_a_plugin_section_loads_and_keeps_its_four_layers(layers, known) -> None:
    g, ws = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "example": {"enabled": True, "color": "#ff0000"}})
    _write_yaml(g / "overrides.yaml", {"example": {"color": "#00ff00"}})
    _write_yaml(ws / "config.yaml", {"example": {"size": 2}})
    _write_yaml(ws / "overrides.yaml", {"example": {"shade": "dark"}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)

    cfg = loader.load_harness()

    assert isinstance(cfg, HarnessConfig)
    assert loader.plugin_layers()["example"] == (
        {"enabled": True, "color": "#ff0000"}, {"color": "#00ff00"}, {"size": 2}, {"shade": "dark"},
    )
    assert loader.plugin_layer_files() == (
        str(g / "config.yaml"), str(g / "overrides.yaml"),
        str(ws / "config.yaml"), str(ws / "overrides.yaml"),
    )


def test_a_plugin_with_no_section_has_four_empty_layers(layers, known) -> None:
    g, _ = layers
    loader = ConfigLoader(config_dir=g)

    assert loader.plugin_layers() == {"example": (None, None, None, None)}
    assert loader.plugin_layer_files()[2:] == ("", "")


@pytest.mark.parametrize("key", ["provder", "not_a_plugin"])
def test_the_typo_guard_survives(layers, known, key) -> None:
    g, _ = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, key: {"x": 1}, "example": {"color": "red"}})

    with pytest.raises(ConfigValidationError, match=key):
        ConfigLoader(config_dir=g).load_harness()


def test_a_plugin_named_after_a_core_key_is_never_routed_out(layers, known) -> None:
    g, _ = layers
    known("org", "example")
    _write_yaml(g / "config.yaml", {**_MINIMAL, "org": {"default_temperature": 0.5}})
    loader = ConfigLoader(config_dir=g)

    assert loader.load_harness().org.default_temperature == 0.5
    assert "org" not in loader.plugin_layers()

    _write_yaml(g / "config.yaml", {**_MINIMAL, "org": {"not_an_org_field": 1}})
    with pytest.raises(ConfigValidationError, match="not_an_org_field"):
        ConfigLoader(config_dir=g).load_harness()


def test_a_bundled_plugin_owns_its_section_like_an_installed_one(layers, known, monkeypatch) -> None:
    class Shipped(Plugin):
        manifest = PluginManifest(name="shipped", version="1.0", kind="tools")

    known()  # nothing discovered: the bundled list alone must be enough
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Shipped,))
    g, _ = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "shipped": {"level": 3}})
    loader = ConfigLoader(config_dir=g)

    loader.load_harness()

    assert loader.plugin_names() == frozenset({"shipped"})
    assert loader.plugin_layers()["shipped"][0] == {"level": 3}


def test_discovery_reads_the_global_dir_and_never_the_workspace(layers, monkeypatch) -> None:
    """REAL folder scan: a plugin folder in the workspace is not a plugin (ENAB-06)."""
    g, ws = layers
    for base, name in ((g, "good"), (ws, "evil")):
        (base / "plugins" / name).mkdir(parents=True)
        (base / "plugins" / name / "__init__.py").write_text("raise SystemExit('never imported')\n")
    seen: list[Path] = []
    real = discovery.discover
    monkeypatch.setattr(discovery, "discover", lambda d: (seen.append(Path(d)), real(d))[1])
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)

    names = loader.plugin_names()

    assert "good" in names
    assert "evil" not in names
    assert seen == [g]
    assert loader.global_config_dir == g


def test_invalidate_cache_rediscovers_plugin_names(layers, known) -> None:
    g, _ = layers
    loader = ConfigLoader(config_dir=g)
    assert loader.plugin_names() == frozenset({"example"})

    known("other")
    assert loader.plugin_names() == frozenset({"example"}), "names are cached per loader"
    loader.invalidate_cache()
    assert loader.plugin_names() == frozenset({"other"})


# --------------------------------------------------------------------------- agent level


def _agent(g: Path, **extra) -> None:
    _write_yaml(g / "agents" / "helper.yaml", {"name": "helper", "role": "helps", **extra})


def test_an_agent_plugin_section_loads_and_is_kept(layers, known) -> None:
    """Kept under the agent's `name:` — what start hands the resolver (agent_config.name) — not
    under its file stem."""
    g, _ = layers
    _write_yaml(g / "agents" / "helper.yaml",
                {"name": "helper-agent", "role": "helps", "example": {"size": 4}})
    loader = ConfigLoader(config_dir=g)

    agent = loader.load_agent("helper")

    assert isinstance(agent, AgentConfig) and agent.name == "helper-agent"
    assert loader.agent_plugin_sections("helper-agent") == {"example": {"size": 4}}


def test_the_overlay_agent_section_merges_under_the_agent_file(layers, known) -> None:
    """Same precedence as every core agent key (issue #22): the overlay's `agent:` is the default
    layer — keys it alone sets arrive, a key the agent file also sets is the agent file's."""
    g, _ = layers
    _write_yaml(g / "overrides.yaml", {"agent": {"example": {"size": 5, "shade": "red"}}})
    _agent(g, example={"size": 4})
    loader = ConfigLoader(config_dir=g)
    loader.load_agent("helper")

    assert loader.agent_plugin_sections("helper") == {"example": {"size": 4, "shade": "red"}}

    _agent(g)
    fresh = ConfigLoader(config_dir=g)
    fresh.load_agent("helper")
    assert fresh.agent_plugin_sections("helper") == {"example": {"size": 5, "shade": "red"}}


def test_an_agent_never_loaded_has_no_plugin_sections(layers, known) -> None:
    assert ConfigLoader(config_dir=layers[0]).agent_plugin_sections("nobody") == {}


def test_the_agent_level_typo_guard_survives(layers, known) -> None:
    g, _ = layers
    _agent(g, rol="typo", example={"size": 4})

    with pytest.raises(ConfigValidationError, match="rol"):
        ConfigLoader(config_dir=g).load_agent("helper")


def test_reload_drops_agent_plugin_sections_with_the_agent_cache(layers, known) -> None:
    g, _ = layers
    _agent(g, example={"size": 4})
    loader = ConfigLoader(config_dir=g)
    loader.load_agent("helper")
    loader.invalidate_cache()  # does not drop the agent cache, so the sections must stay with it
    assert loader.agent_plugin_sections("helper") == {"example": {"size": 4}}

    loader.reload()
    assert loader.agent_plugin_sections("helper") == {}


# --------------------------------------------------------------------------- split


def test_split_plugin_keys_routes_only_known_non_core_names_and_never_mutates() -> None:
    data = {"version": "1", "org": {"a": 1}, "example": {"b": 2}, "typo": 3}
    before = copy.deepcopy(data)

    core, owned = split_plugin_keys(data, {"example", "org", "absent"}, CORE_HARNESS_KEYS)

    assert core == {"version": "1", "org": {"a": 1}, "typo": 3}
    assert owned == {"example": {"b": 2}}
    assert data == before


def test_the_core_key_sets_are_the_models_fields() -> None:
    assert CORE_HARNESS_KEYS == frozenset(HarnessConfig.model_fields) | {"agent"}
    assert CORE_AGENT_KEYS == frozenset(AgentConfig.model_fields)


# --------------------------------------------------------------------------- global-only narrowing


def test_a_workspace_value_for_a_global_only_field_is_dropped_with_a_warning() -> None:
    layers = ({"url": "http://global", "color": "red"}, None, {"url": "http://evil", "color": "blue"}, None)
    before = copy.deepcopy(layers)

    merged, warnings = merge_plugin_layers("example", layers, global_only={"url"}, layer_files=_FILES)

    assert merged == {"url": "http://global", "color": "blue"}
    assert len(warnings) == 1
    assert "example.url" in warnings[0] and "ws/config.yaml" in warnings[0]
    assert layers == before, "narrowing must not edit the loader's cached raw sources"


def test_global_only_narrowing_end_to_end_through_the_loader(layers, known) -> None:
    g, ws = layers
    _write_yaml(g / "config.yaml", {**_MINIMAL, "example": {"url": "http://global", "color": "red"}})
    _write_yaml(ws / "config.yaml", {"example": {"url": "http://evil", "color": "blue"}})
    loader = ConfigLoader(config_dir=g, local_config_dir=ws)
    loader.load_harness()

    merged, warnings = merge_plugin_layers(
        "example", loader.plugin_layers()["example"], global_only={"url"},
        layer_files=loader.plugin_layer_files(),
    )

    assert merged == {"url": "http://global", "color": "blue"}
    assert len(warnings) == 1
    assert "example.url" in warnings[0] and str(ws / "config.yaml") in warnings[0]


def test_both_workspace_files_are_narrowed_and_global_overrides_may_set_it() -> None:
    layers = ({"url": "a"}, {"url": "b"}, {"url": "c"}, {"url": "d"})

    merged, warnings = merge_plugin_layers("example", layers, global_only={"url"}, layer_files=_FILES)

    assert merged == {"url": "b"}
    assert [("ws/config.yaml" in w, "ws/overrides.yaml" in w) for w in warnings] == [
        (True, False), (False, True)]


def test_a_nested_global_only_field_is_narrowed_and_its_siblings_are_not() -> None:
    layers = ({"discord": {"token": "g", "room": "lobby"}}, None,
              {"discord": {"token": "evil", "room": "dev"}}, None)
    before = copy.deepcopy(layers)

    merged, warnings = merge_plugin_layers(
        "dispatch", layers, global_only={"discord.token"}, layer_files=_FILES)

    assert merged == {"discord": {"token": "g", "room": "dev"}}
    assert len(warnings) == 1 and "dispatch.discord.token" in warnings[0]
    assert layers == before, "a nested drop must copy along the path, not edit the cached source"


def test_a_workspace_cannot_replace_the_subtree_that_holds_a_global_only_field() -> None:
    layers = ({"discord": {"token": "g"}}, None, {"discord": None}, None)

    merged, warnings = merge_plugin_layers(
        "dispatch", layers, global_only={"discord.token"}, layer_files=_FILES)

    assert merged == {"discord": {"token": "g"}}
    assert len(warnings) == 1 and "dispatch.discord.token" in warnings[0]


def test_a_global_only_field_the_global_layer_leaves_unset_stays_unset() -> None:
    merged, warnings = merge_plugin_layers(
        "example", (None, None, {"enabled": True}, None), global_only={"enabled"}, layer_files=_FILES)

    assert merged == {}
    assert len(warnings) == 1 and "example.enabled" in warnings[0]


def test_restating_the_global_value_in_the_workspace_is_not_a_warning() -> None:
    merged, warnings = merge_plugin_layers(
        "example", ({"url": "a"}, None, {"url": "a"}, None), global_only={"url"}, layer_files=_FILES)

    assert (merged, warnings) == ({"url": "a"}, [])


def test_a_section_that_is_not_a_mapping_is_an_error_naming_the_file() -> None:
    with pytest.raises(PluginSectionError, match="ws/overrides.yaml"):
        merge_plugin_layers("example", ({}, None, None, "on"), global_only=set(), layer_files=_FILES)


# --------------------------------------------------------------------------- the marker


class _Discord(BaseModel):
    token: str = Field("", json_schema_extra=GLOBAL_ONLY)
    room: str = ""


class _Server(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)


class _Node(BaseModel):
    child: Optional["_Node"] = None
    secret: str = Field("", json_schema_extra=GLOBAL_ONLY)


class _Settings(BaseModel):
    url: str = Field("", json_schema_extra=GLOBAL_ONLY)
    color: str = "red"
    discord: Optional[_Discord] = None
    endpoint: str = Field("", alias="endpoint-url", json_schema_extra=GLOBAL_ONLY)
    servers: list[_Server] = []
    plain: list[str] = []


def test_global_only_paths_recurse_into_nested_models() -> None:
    assert global_only_paths(_Settings) == {
        "url", "discord.token", "endpoint", "endpoint-url", "servers"}


def test_global_only_paths_of_no_model_is_empty() -> None:
    assert global_only_paths(None) == frozenset()


def test_a_field_no_dot_path_can_fully_address_is_global_only_as_a_whole() -> None:
    """Fail closed: a marked field inside a list element (`servers` above) or behind a recursive
    reference cannot be narrowed key by key, so the field that holds it is global-only entirely —
    and the recursion terminates."""
    assert global_only_paths(_Node) == {"secret", "child"}
