"""The plugin API's types: every shape the loader, the lifecycle, the slot and the example plugin
build against, each defined once (PAPI-01/02/03/04, PAPI-12's version constant, CORE-03's list).

Nothing here loads a plugin. What it pins is the CONTRACT: the manifest's defaults and validation,
the no-op contribution methods, the context's exact field set (no idle-LLM field — PAPI-03), and
that the API module itself imports no plugin.
"""
from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

import localharness
from localharness.plugins import api, builtin
from localharness.plugins.api import (
    AVAILABILITY_DOC,
    DEFAULT_REQUIRES_LOCALHARNESS,
    GLOBAL_ONLY,
    GLOBAL_ONLY_KEY,
    PLUGIN_API_VERSION,
    Availability,
    CliDescriptor,
    ContextBudget,
    ContextContribution,
    MemorySlotPlugin,
    Plugin,
    PluginContext,
    PluginManifest,
    PluginPaths,
    SlashDescriptor,
    plugin_summary,
)
from localharness.plugins.builtin import bundled_plugins

PUBLIC_TYPES = (
    "PluginManifest", "CliDescriptor", "SlashDescriptor", "Check", "ContextContribution",
    "MemoryBrowse", "MemoryWriteHandle", "Plugin", "MemorySlotPlugin", "PluginContext",
    "PluginPaths", "BrowseQuery", "ContextBudget",
)


def _manifest(**overrides) -> PluginManifest:
    return PluginManifest(**{"name": "example", "version": "0.1.0", "kind": "tools", **overrides})


class _Bare(Plugin):
    manifest = _manifest()


class _Slot(MemorySlotPlugin):
    manifest = _manifest(name="mem", kind="memory")


def _ctx(tmp_path: Path) -> PluginContext:
    paths = PluginPaths(global_config_dir=tmp_path, workspace=None, state_dir=tmp_path,
                        artifact_dir=tmp_path / "artifacts" / "example")
    return PluginContext(bus=None, tools=None, hooks=None, config=None, agent_config=None,
                         paths=paths, llm=None)


# --- PluginManifest (PAPI-01) ---------------------------------------------------------------


def test_manifest_defaults_and_frozen():
    m = _manifest()
    assert m.requires_localharness == DEFAULT_REQUIRES_LOCALHARNESS == ">=0.15,<1"
    assert m.enabled_by_default is True
    assert m.requires_extra is None
    assert (m.requires, m.uses, m.cli, m.slash) == ((), (), (), ())
    with pytest.raises(ValidationError):
        m.name = "other"


def test_default_version_range_admits_this_line_and_stops_before_1_0():
    """The package reports 0.15.0 until the 0.16.0 cut, so a default that rejected it would turn
    every plugin (the example included) off. Versions are injected, never read live."""
    spec = SpecifierSet(DEFAULT_REQUIRES_LOCALHARNESS)
    assert {v: v in spec for v in ("0.14.9", "0.15.0", "0.16.0", "0.99.1", "1.0.0")} == {
        "0.14.9": False, "0.15.0": True, "0.16.0": True, "0.99.1": True, "1.0.0": False,
    }


@pytest.mark.parametrize("name", ["example", "lh-exa", "image", "a", "x_1", "a" * 64])
def test_manifest_accepts_plugin_names(name):
    assert _manifest(name=name).name == name


@pytest.mark.parametrize("name", ["Bad Name", "1x", "", "a.b", "Example", "-x", "a" * 65])
def test_manifest_rejects_bad_names(name):
    with pytest.raises(ValueError, match="plugin name"):
        _manifest(name=name)


def test_manifest_rejects_an_unknown_kind():
    with pytest.raises(ValidationError):
        _manifest(kind="widget")


# --- CliDescriptor / SlashDescriptor ----------------------------------------------------------


@pytest.mark.parametrize("name", ["/example", "/ex-ample_2"])
def test_slash_names_accepted(name):
    assert SlashDescriptor(name=name, help="h", target="pkg.mod:fn").name == name


@pytest.mark.parametrize("name", ["example", "/Example", "/", "/1x", "/a b"])
def test_slash_names_rejected(name):
    """No leading slash, or upper case (the REPL matches LOWER-CASED input, so an upper-case row
    could never be reached)."""
    with pytest.raises(ValueError, match="slash command"):
        SlashDescriptor(name=name, help="h", target="pkg.mod:fn")


@pytest.mark.parametrize("name", ["example", "generate-image"])
def test_cli_names_accepted(name):
    assert CliDescriptor(name=name, help="h", target="pkg.mod:app").name == name


@pytest.mark.parametrize("name", ["Example", "two words", "", "-x", "under_score"])
def test_cli_names_rejected(name):
    with pytest.raises(ValueError, match="command word"):
        CliDescriptor(name=name, help="h", target="pkg.mod:app")


@pytest.mark.parametrize("descriptor", [CliDescriptor, SlashDescriptor])
@pytest.mark.parametrize("target", ["pkg.mod:app", "mod:app", "pkg.mod:Outer.inner"])
def test_descriptor_targets_accepted(descriptor, target):
    name = "/x" if descriptor is SlashDescriptor else "x"
    assert descriptor(name=name, help="h", target=target).target == target


@pytest.mark.parametrize("descriptor", [CliDescriptor, SlashDescriptor])
@pytest.mark.parametrize("target", ["pkg.mod", "pkg.mod:", ":app", "pkg/mod:app", "pkg.mod:app()"])
def test_descriptor_targets_rejected(descriptor, target):
    """A target is imported only when its command runs, so a malformed one is refused HERE, when
    the manifest is built, not when a user first types the command."""
    name = "/x" if descriptor is SlashDescriptor else "x"
    with pytest.raises(ValueError, match="import target"):
        descriptor(name=name, help="h", target=target)


# --- Plugin / MemorySlotPlugin: every contribution optional (PAPI-01, PAPI-04) ---------------


async def test_plugin_contributions_default_to_no_ops(tmp_path):
    p, ctx = _Bare(), _ctx(tmp_path)
    assert await p.configure(ctx) == "ready"
    assert await p.tools(ctx) == []
    assert await p.start(ctx) is None
    assert await p.stop(ctx) is None
    assert p.doctor(ctx) == []
    assert p.channels() == {}
    assert p.artifact_root(ctx) == ctx.paths.artifact_dir == tmp_path / "artifacts" / "example"
    assert (_Bare.ConfigModel, _Bare.AgentConfigModel, _Bare.wants_artifacts) == (None, None, False)


async def test_memory_slot_contributions_default_to_empty(tmp_path):
    p, ctx = _Slot(), _ctx(tmp_path)
    got = await p.context(ctx, "t", ContextBudget(max_chars=1, max_session_history=1))
    assert got == ContextContribution() and got.sections == ()
    assert p.browse() is None
    assert p.bind_subagent(ctx) is None


# --- PluginContext / PluginPaths (PAPI-03) ----------------------------------------------------


def test_context_fields_are_exactly_the_papi03_set_in_order(tmp_path):
    """Order is pinned, not just the set: config and agent_config share a type, so a positional
    swap would type-check and hand a plugin the wrong settings."""
    assert [f.name for f in dataclasses.fields(PluginContext)] == [
        "bus", "tools", "hooks", "config", "agent_config", "paths", "llm", "idle_llm", "session",
    ]
    assert [f.name for f in dataclasses.fields(PluginPaths)] == [
        "global_config_dir", "workspace", "state_dir", "artifact_dir",
    ]
    ctx = _ctx(tmp_path)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.llm = object()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.paths.artifact_dir = tmp_path


# --- Small contract pieces --------------------------------------------------------------------


def test_availability_is_a_usable_type():
    ta = TypeAdapter(Availability)
    assert ta.validate_python("ready") == "ready"
    assert ta.validate_python(("unconfigured", "example.url")) == ("unconfigured", "example.url")
    for bad in ("nope", ("ready", "x"), ("unconfigured",)):
        with pytest.raises(ValidationError):
            ta.validate_python(bad)
    assert AVAILABILITY_DOC.strip()


def test_global_only_marker_reads_back_from_a_config_model():
    class Cfg(BaseModel):
        url: str = Field("", json_schema_extra=GLOBAL_ONLY)
        color: str = "#4a90d9"

    assert Cfg.model_fields["url"].json_schema_extra == {GLOBAL_ONLY_KEY: True}
    assert Cfg.model_fields["color"].json_schema_extra is None


def test_plugin_summary_is_the_first_docstring_line():
    class Documented(Plugin):
        """
        Generates images through a local ComfyUI.

        More detail that `plugins list` does not show."""

    class Undocumented(Plugin):
        pass

    assert plugin_summary(Documented) == "Generates images through a local ComfyUI."
    assert plugin_summary(Undocumented) == ""


def test_api_version_constant():
    assert PLUGIN_API_VERSION == "1"


# --- CORE-03: the one list --------------------------------------------------------------------


def test_the_one_list_holds_the_image_plugin_and_is_read_at_call_time(monkeypatch):
    from localharness.cli.web_plugin import WebPlugin
    from localharness.tools.builtin.image_plugin import ImagePlugin

    assert bundled_plugins() == (ImagePlugin, WebPlugin) and builtin.BUILTIN_PLUGINS == (ImagePlugin, WebPlugin)  # web is bundled and on by default (46-02)
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", ())
    assert bundled_plugins() == ()
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (_Bare,))
    assert bundled_plugins() == (_Bare,)


# --- PAPI-02: each type defined once, with a docstring ----------------------------------------


@pytest.mark.parametrize("name", PUBLIC_TYPES)
def test_every_public_type_has_a_written_docstring(name):
    doc = (getattr(api, name).__doc__ or "").strip()
    # @dataclass invents "Name(field: type, ...)" when there is none — that is not a docstring.
    assert doc and not doc.startswith(f"{name}("), f"{name} has no written docstring"


def _top_level_bindings() -> dict[str, list[str]]:
    """name -> every module (relative to the package) that binds it at top level."""
    root = Path(localharness.__file__).parent
    files = sorted(root.rglob("*.py"))
    assert len(files) > 100, f"scanned only {len(files)} files under {root}"
    found: dict[str, list[str]] = {}
    for path in files:
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [node.name]
            elif isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names = [node.target.id]
            else:
                continue
            for n in names:
                found.setdefault(n, []).append(path.relative_to(root).as_posix())
    return found


def test_each_api_name_is_defined_exactly_once():
    """Every public API name is bound in exactly one module — PluginManifest included, now that the
    dormant legacy loader, which defined its own, is deleted (44-14)."""
    found = _top_level_bindings()
    once_in_api = [*PUBLIC_TYPES] + [
        "Availability", "PLUGIN_API_VERSION", "GLOBAL_ONLY", "plugin_summary",
    ]
    expected = {n: ["plugins/api.py"] for n in once_in_api}
    expected |= {"BUILTIN_PLUGINS": ["plugins/builtin.py"], "bundled_plugins": ["plugins/builtin.py"]}
    # PAPI-02/PAPI-10: ArtifactRef lives in core/events.py, beside the one id shape and allowlist.
    expected |= {n: ["core/events.py"] for n in ("ArtifactRef", "ARTIFACT_ID_RE", "ARTIFACT_MIMES")}
    expected |= {n: ["core/artifacts.py"] for n in ("artifact_root", "mint_artifact_id", "write_artifact")}
    assert {n: found.get(n) for n in expected} == expected


def test_importing_the_api_imports_no_plugin():
    code = (
        "import sys, localharness.plugins.api; print(sorted(m for m in sys.modules if m.startswith(("
        "'localharness.memory', 'localharness.channels', 'localharness.autoresearch', "
        "'localharness.tools.builtin', 'localharness.cli'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_session_info_fields_in_order_and_exit_reason_is_assignable():
    """SessionInfo is the session identity a per-session plugin (memory) needs. It is NOT frozen:
    core sets exit_reason just before stop_plugins and the plugin reads it in stop()."""
    from localharness.plugins.api import SessionInfo
    assert [f.name for f in dataclasses.fields(SessionInfo)] == [
        "agent_id", "division_id", "sitting_id", "model", "context_tokens", "budget", "exit_reason",
    ]
    info = SessionInfo("a", "d", "s", "m", 1000, {})
    assert info.exit_reason == "complete"
    info.exit_reason = "interrupted"
    assert info.exit_reason == "interrupted"


def test_plugin_api_version_is_unchanged_by_the_additive_fields():
    """idle_llm and session are defaulted, optional additions — no version bump."""
    from localharness.plugins.api import PLUGIN_API_VERSION
    assert PLUGIN_API_VERSION == "1"
