"""The load plan — what loads, in what order, and why not — is a PURE function (PRD §6).

Every behavior is proven with small Plugin classes and DiscoveredPlugin values built inline. The
localharness version is always INJECTED: several cases use a version other than the running one, so a
plan that read the live version instead of its argument cannot pass them.
"""
from __future__ import annotations

import builtins
import importlib
import importlib.metadata
from types import SimpleNamespace

import pytest

from localharness.plugins.api import MemorySlotPlugin, Plugin, PluginManifest
from localharness.plugins.discovery import DiscoveredPlugin
from localharness.plugins.plan import LoadPlan, PlanEntry, build_load_plan, extra_installed

CORE = frozenset({"org", "provider", "memory"})
VERSION = "0.15.0"


def make(name: str, *, kind: str = "tools", doc: str | None = None, base: type[Plugin] = Plugin,
         **manifest) -> type[Plugin]:
    """A plugin class called `name`; the first docstring line is its WHAT IT DOES summary."""
    return type(f"P_{name.replace('-', '_')}", (base,), {
        "manifest": PluginManifest(name=name, version="1.0", kind=kind, **manifest),
        "__doc__": doc or f"does {name} things"})


def found(name: str, *, dist: str | None = None, version: str = "0.3.1") -> DiscoveredPlugin:
    return DiscoveredPlugin(name, "entry_point", f"{name.replace('-', '_')}:P", dist or name, version)


def build(*, bundled=(), discovered=(), enabled=None, imported=None, version=VERSION,
          extras=lambda extra: True, invalid=None) -> LoadPlan:
    return build_load_plan(bundled=bundled, discovered=discovered, enabled=enabled or {},
                           imported=imported or {}, version=version, core_keys=CORE,
                           extra_installed=extras, invalid=invalid or {})


def state(plan: LoadPlan, name: str) -> tuple[str, str]:
    entry = plan.entry(name)
    assert entry is not None, name
    return entry.state, entry.reason


# --------------------------------------------------------------------------- bundled


def test_a_bundled_plugin_is_on_when_enabled_else_off_with_its_enable_command() -> None:
    alpha, beta = make("alpha"), make("beta")
    plan = build(bundled=(alpha, beta), enabled={"alpha": True, "beta": False})

    on, off = plan.entries
    assert (on.name, on.bundled, on.state, on.source, on.summary, on.reason, on.manifest) == (
        "alpha", True, "on", "built in", "does alpha things", "", alpha.manifest)
    assert on.enable_command is None and on.display == "on"
    assert (off.state, off.reason, off.manifest) == ("off", "", beta.manifest)
    assert off.enable_command == "localharness plugins enable beta"
    assert off.display == "off — turn on: localharness plugins enable beta"
    assert plan.order == ("alpha",)
    assert plan.entry("beta") is off and plan.entry("nobody") is None


def test_a_plugin_missing_from_the_enabled_map_is_off() -> None:
    plan = build(bundled=(make("alpha", enabled_by_default=True),), enabled={})

    assert state(plan, "alpha") == ("off", "")


# --------------------------------------------------------------------------- discovered


def test_a_discovered_plugin_that_is_not_enabled_is_available_and_its_class_is_never_read() -> None:
    example = DiscoveredPlugin("example", "entry_point", "localharness_plugin_example:ExamplePlugin",
                               "localharness-plugin-example", "0.1.0")
    # Even a class sitting in `imported` is not consulted for a plugin that is not enabled.
    plan = build(discovered=(example,), imported={"example": make("example")})

    (entry,) = plan.entries
    assert (entry.name, entry.bundled, entry.state, entry.summary, entry.manifest, entry.source) == (
        "example", False, "available", "(not loaded)", None, "pip: localharness-plugin-example 0.1.0")
    assert entry.enable_command == "localharness plugins enable example"
    assert entry.display == "available — turn on: localharness plugins enable example"
    assert plan.order == ()


def test_an_enabled_discovered_plugin_is_on_with_its_class_read() -> None:
    exa = make("lh-exa", doc="searches the web\n\nlong description")
    plan = build(discovered=(found("lh-exa"),), enabled={"lh-exa": True}, imported={"lh-exa": exa})

    entry = plan.entry("lh-exa")
    assert (entry.state, entry.summary, entry.manifest, entry.source) == (
        "on", "searches the web", exa.manifest, "pip: lh-exa 0.3.1")
    assert plan.order == ("lh-exa",)


@pytest.mark.parametrize("exc, reason", [
    (RuntimeError("kaput"), "could not be imported: RuntimeError: kaput"),
    (SystemExit(3), "could not be imported: SystemExit: 3"),
])
def test_an_import_that_raised_is_failed_naming_the_exception(exc, reason) -> None:
    plan = build(discovered=(found("boom"),), enabled={"boom": True}, imported={"boom": exc})

    entry = plan.entry("boom")
    assert (entry.state, entry.reason, entry.summary, entry.manifest) == (
        "failed", reason, "(not loaded)", None)
    assert entry.display == f"failed — {reason}"
    assert entry.enable_command is None


def test_an_enabled_discovered_plugin_with_no_import_result_is_failed() -> None:
    plan = build(discovered=(found("ghost"),), enabled={"ghost": True})

    assert state(plan, "ghost") == ("failed", "it was not imported")


# --------------------------------------------------------------------------- the version


@pytest.mark.parametrize("spec, version, expected", [
    (">=9", "0.15.0", ("skipped", "requires localharness >=9, this is 0.15.0")),
    (">=0.15,<1", "0.15.0", ("on", "")),
    # Versions other than the running one: only the INJECTED version can decide these.
    (">=9", "9.2.0", ("on", "")),
    (">=0.15,<1", "1.0.0", ("skipped", "requires localharness >=0.15,<1, this is 1.0.0")),
    (">=0.15,<1", "0.16.0.dev1", ("on", "")),  # a development build is in range
    (">=0.15,<1", "unknown", ("skipped", "requires localharness >=0.15,<1, this is unknown")),
])
def test_requires_localharness_is_checked_against_the_injected_version(spec, version, expected) -> None:
    plan = build(discovered=(found("exa"),), enabled={"exa": True},
                 imported={"exa": make("exa", requires_localharness=spec)}, version=version)

    assert state(plan, "exa") == expected


def test_an_invalid_version_range_is_skipped_naming_it() -> None:
    plan = build(discovered=(found("exa"),), enabled={"exa": True},
                 imported={"exa": make("exa", requires_localharness="banana")})

    status, reason = state(plan, "exa")
    assert status == "skipped" and "'banana'" in reason and "not a version range" in reason


def test_a_bundled_plugin_meets_the_same_version_rule() -> None:
    plan = build(bundled=(make("alpha", requires_localharness=">=9"),), enabled={"alpha": True})

    assert state(plan, "alpha") == ("skipped", "requires localharness >=9, this is 0.15.0")


# --------------------------------------------------------------------------- names


def test_name_checks_come_before_anything_else() -> None:
    bad = DiscoveredPlugin("Bad.Name", "folder", "/g/plugins/Bad.Name")
    first = found("dup", dist="lh-dup")
    second = DiscoveredPlugin("dup", "folder", "/g/plugins/dup")
    plan = build(discovered=(bad, found("org"), first, second),
                 enabled={"Bad.Name": True, "org": True, "dup": True},
                 imported={"dup": make("dup"), "org": make("org"), "Bad.Name": make("bad")})

    bad_e, core_e, first_e, second_e = plan.entries
    assert bad_e.state == "refused" and "'Bad.Name' is not a valid plugin name" in bad_e.reason
    assert (core_e.state, core_e.reason) == (
        "refused", "its name collides with the core settings key 'org'")
    assert first_e.state == "on"
    assert (second_e.state, second_e.reason) == (
        "refused", "the name 'dup' is already taken (pip: lh-dup 0.3.1)")
    assert plan.order == ("dup",)


def test_a_name_check_refuses_even_a_plugin_that_is_not_enabled() -> None:
    plan = build(bundled=(make("memory"), make("alpha")), discovered=(found("alpha"),))

    memory, alpha, taken = plan.entries
    assert (memory.state, memory.reason) == (
        "refused", "its name collides with the core settings key 'memory'")
    assert alpha.state == "off"
    assert (taken.bundled, taken.state, taken.reason) == (
        False, "refused", "the name 'alpha' is already taken (built in)")


def test_a_class_whose_manifest_names_another_plugin_is_refused() -> None:
    nameless = type("Nameless", (Plugin,), {"__doc__": "has no manifest"})
    plan = build(discovered=(found("exa"), found("anon")), enabled={"exa": True, "anon": True},
                 imported={"exa": make("other"), "anon": nameless})

    exa, anon = plan.entries
    assert (exa.state, exa.manifest) == ("refused", None)
    assert exa.reason == "it was found as 'exa' but its class names 'other' in its manifest"
    assert (anon.state, anon.reason) == ("refused", "it was found as 'anon' but its class declares no manifest")
    assert plan.order == ()


# --------------------------------------------------------------------------- the memory slot


def test_two_on_memory_plugins_are_both_refused_and_the_slot_is_empty() -> None:
    mem_a = make("mem-a", kind="memory", base=MemorySlotPlugin)
    mem_b = make("mem-b", kind="memory", base=MemorySlotPlugin)
    plan = build(bundled=(mem_a,), discovered=(found("mem-b"),),
                 enabled={"mem-a": True, "mem-b": True}, imported={"mem-b": mem_b})

    for name in ("mem-a", "mem-b"):
        status, reason = state(plan, name)
        assert status == "refused" and "claim the memory slot" in reason
        assert "mem-a" in reason and "mem-b" in reason
    assert plan.memory_occupant is None
    assert plan.order == ()


def test_one_on_memory_plugin_occupies_the_slot_and_an_off_one_does_not_count() -> None:
    mem_a = make("mem-a", kind="memory", base=MemorySlotPlugin)
    mem_b = make("mem-b", kind="memory", base=MemorySlotPlugin)
    plan = build(bundled=(mem_a, mem_b), enabled={"mem-a": False, "mem-b": True})

    assert plan.memory_occupant == "mem-b"
    assert state(plan, "mem-a") == ("off", "")
    assert build(bundled=(make("alpha"),), enabled={"alpha": True}).memory_occupant is None


def test_kind_memory_must_be_a_memory_slot_plugin() -> None:
    plan = build(bundled=(make("mem", kind="memory"),), enabled={"mem": True})

    assert state(plan, "mem") == ("refused", 'its kind is "memory" but it is not a MemorySlotPlugin')
    assert plan.memory_occupant is None


# --------------------------------------------------------------------------- dependencies


def test_a_hard_requirement_that_is_not_on_fails_the_dependent_transitively() -> None:
    top, mid, base = make("top", requires=("mid",)), make("mid", requires=("base",)), make("base")
    plan = build(bundled=(top, mid, base), discovered=(found("needs-exa"),),
                 enabled={"top": True, "mid": True, "base": False, "needs-exa": True},
                 imported={"needs-exa": make("needs-exa", requires=("lh-exa",))})

    assert state(plan, "mid") == ("failed", "requires base, which is off")
    assert state(plan, "top") == ("failed", "requires mid, which is failed")
    assert state(plan, "needs-exa") == ("failed", "requires lh-exa, which is not installed")
    assert plan.order == ()


def test_dependencies_start_first_and_independent_plugins_keep_display_order() -> None:
    plan = build(bundled=(make("a", requires=("c",)), make("b"), make("c"), make("d")),
                 enabled=dict.fromkeys("abcd", True))

    assert plan.order == ("b", "c", "a", "d")
    assert build(bundled=(make("x"), make("y"), make("z")),
                 enabled=dict.fromkeys("xyz", True)).order == ("x", "y", "z")


def test_uses_orders_a_soft_dependency_first_only_when_it_is_on() -> None:
    a, b = make("a", uses=("b",)), make("b")

    assert build(bundled=(a, b), enabled={"a": True, "b": True}).order == ("b", "a")
    plan = build(bundled=(a, b), enabled={"a": True, "b": False})
    assert plan.order == ("a",) and state(plan, "a") == ("on", "")


def test_a_dependency_cycle_refuses_its_members_and_fails_their_hard_dependents() -> None:
    plan = build(bundled=(make("a", requires=("b",)), make("b", requires=("a",)),
                          make("c", requires=("a",)), make("d", uses=("a",)), make("e")),
                 enabled=dict.fromkeys("abcde", True))

    assert state(plan, "a") == ("refused", "dependency cycle: a → b → a")
    assert state(plan, "b") == ("refused", "dependency cycle: a → b → a")
    assert state(plan, "c") == ("failed", "requires a, which is refused")
    assert state(plan, "d") == ("on", "")  # a soft user of a refused plugin still loads
    assert plan.order == ("d", "e")


def test_a_cycle_is_named_from_where_it_closes() -> None:
    plan = build(bundled=(make("top", requires=("x",)), make("x", uses=("y",)),
                          make("y", requires=("x",)), make("self", requires=("self",))),
                 enabled={"top": True, "x": True, "y": True, "self": True})

    assert state(plan, "x") == ("refused", "dependency cycle: x → y → x")
    assert state(plan, "y") == ("refused", "dependency cycle: x → y → x")
    assert state(plan, "self") == ("refused", "dependency cycle: self → self")
    assert state(plan, "top") == ("failed", "requires x, which is refused")


# --------------------------------------------------------------------------- extras, invalid settings


def test_a_missing_install_extra_is_a_displayed_state_not_a_load() -> None:
    web = make("web", requires_extra="web")
    plan = build(bundled=(web,), enabled={"web": True}, extras=lambda extra: extra != "web")

    entry = plan.entry("web")
    assert (entry.state, entry.reason) == ("needs-extra", "install `localharness[web]` to use it")
    assert entry.display == "on (install `localharness[web]` to use it)"
    assert entry.enable_command is None and plan.order == ()
    assert state(build(bundled=(web,), enabled={"web": True}), "web") == ("on", "")


def test_invalid_settings_fail_an_on_plugin_and_its_dependents_but_an_off_one_stays_off() -> None:
    plan = build(bundled=(make("x"), make("y", requires=("x",)), make("z")),
                 enabled={"x": True, "y": True, "z": False},
                 invalid={"x": "invalid settings — x.color: bad", "z": "invalid settings — z.size: bad"})

    assert state(plan, "x") == ("failed", "invalid settings — x.color: bad")
    assert plan.entry("x").display == "failed — invalid settings — x.color: bad"
    assert state(plan, "y") == ("failed", "requires x, which is failed")
    assert state(plan, "z") == ("off", "")
    assert plan.order == ()


# --------------------------------------------------------------------------- display, purity


def test_display_is_the_one_state_text_every_reader_prints() -> None:
    def show(state_: str, reason: str = "") -> str:
        return PlanEntry("x", False, state_, "pip: x 1", "does x", reason).display

    assert show("on") == "on"
    assert show("off") == "off — turn on: localharness plugins enable x"
    assert show("available") == "available — turn on: localharness plugins enable x"
    assert show("needs-extra", "install `localharness[web]` to use it") == (
        "on (install `localharness[web]` to use it)")
    assert show("skipped", "requires localharness >=9, this is 0.15.0") == (
        "skipped — requires localharness >=9, this is 0.15.0")
    assert show("refused", "dependency cycle: a → b → a") == "refused — dependency cycle: a → b → a"


def _every_state() -> dict:
    return dict(
        bundled=(make("on"), make("off"), make("web", requires_extra="web"),
                 make("mem", kind="memory", base=MemorySlotPlugin), make("a", requires=("b",)),
                 make("b", requires=("a",))),
        discovered=(found("avail"), found("boom"), found("old"), found("exa", dist="lh-exa"),
                    found("org")),
        enabled={"on": True, "off": False, "web": True, "mem": True, "a": True, "b": True,
                 "boom": True, "old": True, "exa": True, "org": True},
        imported={"boom": RuntimeError("x"), "old": make("old", requires_localharness=">=9"),
                  "exa": make("exa", uses=("mem",))},
        version=VERSION, core_keys=CORE, extra_installed=lambda extra: False,
        invalid={"off": "invalid settings — off.x: bad"})


def test_build_load_plan_reads_nothing_and_imports_nothing(monkeypatch) -> None:
    before = build_load_plan(**_every_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("build_load_plan did I/O or imported something")

    for target, name in [(builtins, "open"), (importlib, "import_module"),
                         (importlib.metadata, "entry_points"), (importlib.metadata, "distribution"),
                         (importlib.metadata, "requires"), (importlib.metadata, "version"),
                         (builtins, "__import__")]:
        monkeypatch.setattr(target, name, forbidden)
    try:
        after = build_load_plan(**_every_state())
    finally:
        monkeypatch.undo()

    assert after == before
    assert [e.state for e in after.entries] == [
        "on", "off", "needs-extra", "on", "refused", "refused",
        "available", "failed", "skipped", "on", "refused"]
    assert after.order == ("on", "mem", "exa") and after.memory_occupant == "mem"


# --------------------------------------------------------------------------- extra_installed


class _Metadata:
    def __init__(self, extras: list[str]) -> None:
        self._extras = extras

    def get_all(self, key: str):
        return list(self._extras) if key == "Provides-Extra" else None


def _fake_dists(monkeypatch, *, requires: list[str], extras: list[str], installed: dict[str, str]):
    def distribution(name: str):
        if name == "localharness":
            return SimpleNamespace(requires=requires, metadata=_Metadata(extras))
        if name in installed:
            return SimpleNamespace(version=installed[name])
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", distribution)


_REQUIRES = ["pydantic>=2", "colorama>=0.4; python_version >= '3'",  # base requirements: not web's
             "starlette<1,>=0.40; extra == 'web'", "uvicorn>=0.30; extra == 'web'",
             "discord.py>=2.3; extra == 'dispatch'",
             "pywin32>=300; sys_platform == 'nonesuch' and extra == 'web'"]


@pytest.mark.parametrize("installed, extra, expected", [
    ({"starlette": "0.46.0", "uvicorn": "0.34.0"}, "web", True),
    ({"starlette": "0.46.0", "uvicorn": "0.34.0"}, "Web", True),       # extras compare normalized
    ({"starlette": "0.46.0"}, "web", False),                           # one requirement missing
    ({"starlette": "1.2.0", "uvicorn": "0.34.0"}, "web", False),       # one out of its range
    ({"starlette": "0.46.0", "uvicorn": "0.34.0"}, "dispatch", False),
    ({"starlette": "0.46.0", "uvicorn": "0.34.0"}, "image", False),    # an extra this release lacks
    ({"starlette": "0.46.0", "uvicorn": "2004d"}, "web", False),       # a version that is not PEP 440
])
def test_extra_installed_reads_the_distribution_metadata(monkeypatch, installed, extra, expected) -> None:
    _fake_dists(monkeypatch, requires=_REQUIRES, extras=["web", "dispatch"], installed=installed)

    assert extra_installed(extra) is expected


def test_extra_installed_is_false_without_an_installed_localharness(monkeypatch) -> None:
    _fake_dists(monkeypatch, requires=[], extras=[], installed={})
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError(name)))

    assert extra_installed("web") is False


def test_extra_installed_on_this_install() -> None:
    """Unmocked: this suite runs under the dev extra (`uv sync --extra dev`)."""
    assert extra_installed("dev") is True
    assert extra_installed("no-such-extra") is False
