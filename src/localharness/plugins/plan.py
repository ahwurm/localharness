"""The load plan: which plugins load, in what order, and why each of the others does not (PRD §6).

PURE. `build_load_plan` reads only its arguments — the one bundled list, discovered METADATA, the
resolved `enabled` flags, the classes the resolver imported (for enabled plugins only), the version
and the core settings keys — and returns the same plan every time: it reads no file, imports
nothing and runs no plugin code. So every reader (the session, the banner, `plugins list`, doctor,
components, the CLI mount) gets one answer from one function. The impure half — reading config,
discovering, importing what is on, validating settings — is plugins/resolve.py.

Bundled plugins and plugins you install pass the same checks, in the same order. They differ only
where the API says they do: a bundled class is already in hand, an installed one is imported only
once it is enabled.
"""
from __future__ import annotations

import importlib.metadata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from localharness.plugins.api import (
    PLUGIN_NAME_RE, MemorySlotPlugin, Plugin, PluginManifest, plugin_summary,
)
from localharness.plugins.discovery import DiscoveredPlugin

PlanState = Literal["on", "off", "available", "failed", "skipped", "refused", "needs-extra"]
"""What the plan decided for one plugin.

- on: it loads this session, at its place in LoadPlan.order.
- off: a bundled plugin that is not enabled; none of its code runs.
- available: an installed plugin that is not enabled. Only its metadata was read — it was never
  imported — so what it does is unknown until you enable it.
- failed: enabled, but it cannot run — importing it raised, its settings are invalid, or a plugin
  it `requires` is not on.
- skipped: enabled, but its `requires_localharness` excludes this version (or is not a range).
- refused: it breaks a rule of the plan — its name is invalid, a core settings key or taken; its
  class names another plugin; it is kind "memory" without being a MemorySlotPlugin; it claims the
  memory slot along with another plugin; it is a plugin you installed that declares `sections`; or
  it is in a dependency cycle.
- needs-extra: enabled, but the `localharness[<extra>]` install extra it needs is missing. Shown
  as "on (install … to use it)"; it does not load.
"""

_NOT_ON = {"available": "available but not enabled", "needs-extra": "missing its install extra"}


def _enable_command_text(name: str) -> str:
    """The one place `localharness plugins enable <name>` is spelled."""
    return f"localharness plugins enable {name}"


@dataclass(frozen=True)
class PlanEntry:
    """One plugin's row in the plan — the words `plugins list`, doctor and the banner print."""

    name: str
    bundled: bool
    state: PlanState
    source: str                          # "built in" | DiscoveredPlugin.from_label
    summary: str                         # WHAT IT DOES, or "(not loaded)"
    reason: str = ""
    manifest: PluginManifest | None = None

    @property
    def enable_command(self) -> str | None:
        """The exact command that turns an off or available plugin on; None otherwise."""
        return _enable_command_text(self.name) if self.state in ("off", "available") else None

    @property
    def step_command(self) -> str | None:
        """The command that turns this plugin on and runs its setup step: for a plugin that is off
        or available (it equals enable_command) and for one that is on or holds for its install
        extra, where the same command runs the step again; None for a plugin the plan failed,
        refused or skipped (`display` says why). `init` prints it for every bundled plugin."""
        return _enable_command_text(self.name) if self.state in ("on", "off", "available", "needs-extra") else None

    @property
    def display(self) -> str:
        """The STATE text `plugins list`, doctor and the banner all print (PRD §4): "on";
        "off — turn on: localharness plugins enable x"; "available — turn on: …";
        "on (install `localharness[mobile]` to use it)" for needs-extra; else "<state> — <reason>".
        Reasons never repeat the state word."""
        if self.state == "on":
            return "on"
        if self.state == "needs-extra":
            return f"on ({self.reason})"
        if self.enable_command:
            return f"{self.state} — turn on: {self.enable_command}"
        return f"{self.state} — {self.reason}"


@dataclass(frozen=True)
class LoadPlan:
    """Every plugin in display order, the start order of the ones that are on, and the memory
    slot's occupant (None: the slot is empty and the harness runs without memory)."""

    entries: tuple[PlanEntry, ...]       # display order: bundled (tuple order), then discovered
    order: tuple[str, ...]               # start order of the ON entries (dependencies first)
    memory_occupant: str | None

    def entry(self, name: str) -> PlanEntry | None:
        """The first entry named `name` (a later one of the same name is refused), or None."""
        return next((e for e in self.entries if e.name == name), None)


def manifest_of(cls: type) -> PluginManifest | None:
    """A plugin class's manifest, or None when it declares none."""
    manifest = getattr(cls, "manifest", None)
    return manifest if isinstance(manifest, PluginManifest) else None


def name_refusals(candidates: Sequence[tuple[str, str]], core_keys: frozenset[str]) -> list[str | None]:
    """For each (name, source) in display order: why that name cannot be a plugin here, else None.
    The first check of all; the resolver uses it to decide what it may import."""
    taken: dict[str, str] = {}
    out: list[str | None] = []
    for name, source in candidates:
        if not PLUGIN_NAME_RE.fullmatch(name):
            out.append(f"'{name}' is not a valid plugin name (a lower-case letter, then up to 63 "
                       "lower-case letters, digits, '_' or '-')")
        elif name in core_keys:
            out.append(f"its name collides with the core settings key '{name}'")
        elif name in taken:
            out.append(f"the name '{name}' is already taken ({taken[name]})")
        else:
            taken[name] = source
            out.append(None)
    return out


def build_load_plan(*, bundled: Sequence[type[Plugin]], discovered: Sequence[DiscoveredPlugin],
                    enabled: Mapping[str, bool], imported: Mapping[str, type[Plugin] | BaseException],
                    version: str, core_keys: frozenset[str],
                    extra_installed: Callable[[str], bool],
                    invalid: Mapping[str, str] = MappingProxyType({})) -> LoadPlan:
    """Decide every plugin's state, and the start order of the ones that are on.

    Each candidate — bundled (tuple order), then discovered — meets these checks in order, the first
    that fails deciding its state: its name (valid, not a core key, not taken) → enabled (else off
    or available) → for an installed plugin, its import (the class in `imported`, or the exception
    importing it raised) and its manifest naming it → kind "memory" is a MemorySlotPlugin → the
    version → the install extra → `invalid` settings. Then, over the plugins still on: two or more
    of kind "memory" are all refused (at most one occupant; none is fine); a plugin whose `requires`
    names one that is not on fails, to a fixed point; and the rest are ordered dependencies first
    (`requires`, and `uses` naming an on plugin), ties in display order, with the members of a
    dependency cycle refused.

    `enabled`: each plugin's resolved flag — a name missing from it is off. `imported`: what
    importing an enabled installed plugin produced. `version`: the localharness version the
    `requires_localharness` ranges are checked against — injected, never read here. `invalid`:
    plugin name → the reason its settings are invalid; it fails a plugin that is on, and a plugin
    that is off stays off.
    """
    candidates = [(c.manifest.name, "built in", c) for c in bundled] + [
        (d.name, d.from_label, None) for d in discovered]
    refusals = name_refusals([(name, source) for name, source, _ in candidates], core_keys)
    entries = [_check(name, source, cls, refusal, enabled=enabled, imported=imported,
                      version=version, extra_installed=extra_installed, invalid=invalid)
               for (name, source, cls), refusal in zip(candidates, refusals, strict=True)]

    claims = [e.name for e in entries if e.state == "on" and _kind(e) == "memory"]
    if len(claims) > 1:
        entries = _refuse(entries, claims, f"{', '.join(claims)} each claim the memory slot, which "
                                           "holds one plugin — turn all but one off")
    while True:
        entries = _cascade(entries)
        order, cycle = _order(entries)
        if not cycle:
            break
        entries = _refuse(entries, cycle, "dependency cycle: " + " → ".join([*cycle, cycle[0]]))
    occupant = next((e.name for e in entries if e.state == "on" and _kind(e) == "memory"), None)
    return LoadPlan(tuple(entries), tuple(order), occupant)


def _check(name: str, source: str, bundled_cls: type[Plugin] | None, refusal: str | None, *,
           enabled: Mapping[str, bool], imported: Mapping[str, type[Plugin] | BaseException],
           version: str, extra_installed: Callable[[str], bool],
           invalid: Mapping[str, str]) -> PlanEntry:
    """One candidate through the per-plugin checks; the first that fails decides."""
    bundled = bundled_cls is not None

    def entry(state: PlanState, reason: str = "", cls: type[Plugin] | None = bundled_cls) -> PlanEntry:
        return PlanEntry(name, bundled, state, source,
                         plugin_summary(cls) if cls is not None else "(not loaded)", reason,
                         manifest_of(cls) if cls is not None else None)

    if refusal is not None:
        return entry("refused", refusal)
    if not enabled.get(name, False):
        return entry("off" if bundled else "available")
    cls = bundled_cls
    if cls is None:
        got = imported.get(name)
        if isinstance(got, BaseException):
            return entry("failed", f"could not be imported: {type(got).__name__}: {got}")
        if got is None:
            return entry("failed", "it was not imported")
        found = manifest_of(got)
        if found is None or found.name != name:
            what = "declares no manifest" if found is None else f"names '{found.name}' in its manifest"
            return PlanEntry(name, False, "refused", source, plugin_summary(got),
                             f"it was found as '{name}' but its class {what}")
        cls = got
    manifest = cls.manifest
    if manifest.kind == "memory" and not issubclass(cls, MemorySlotPlugin):
        return entry("refused", 'its kind is "memory" but it is not a MemorySlotPlugin', cls)
    if manifest.sections and not bundled:
        return entry("refused", "declares sections (claiming core settings), which only bundled "
                                "plugins may do", cls)
    spec = manifest.requires_localharness
    try:
        specifier = SpecifierSet(spec)
    except InvalidSpecifier:
        return entry("skipped", f"its requires_localharness '{spec}' is not a version range", cls)
    if not _admits(specifier, version):
        return entry("skipped", f"requires localharness {spec}, this is {version}", cls)
    if manifest.requires_extra and not extra_installed(manifest.requires_extra):
        return entry("needs-extra", f"install `localharness[{manifest.requires_extra}]` to use it", cls)
    if name in invalid:
        return entry("failed", invalid[name], cls)
    return entry("on", cls=cls)


def _admits(specifier: SpecifierSet, version: str) -> bool:
    """`version` is in range, pre-releases and development builds included (before packaging 24
    the default excluded them); a version that is not PEP 440 is in no range (packaging 23 raised)."""
    try:
        return specifier.contains(Version(version), prereleases=True)
    except InvalidVersion:
        return False


def _kind(entry: PlanEntry) -> str | None:
    return entry.manifest.kind if entry.manifest is not None else None


def _deps(entry: PlanEntry, *, soft: bool) -> tuple[str, ...]:
    """What a plugin waits for: its `requires`, and its `uses` too when `soft`."""
    m = entry.manifest
    if m is None:
        return ()
    return (*m.requires, *m.uses) if soft else m.requires


def _refuse(entries: list[PlanEntry], names: Sequence[str], reason: str) -> list[PlanEntry]:
    return [replace(e, state="refused", reason=reason) if e.state == "on" and e.name in names else e
            for e in entries]


def _first(entries: list[PlanEntry]) -> dict[str, PlanEntry]:
    out: dict[str, PlanEntry] = {}
    for e in entries:
        out.setdefault(e.name, e)
    return out


def _cascade(entries: list[PlanEntry]) -> list[PlanEntry]:
    """Fail every on plugin whose `requires` names one that is not on, until nothing changes."""
    while True:
        first = _first(entries)
        unmet: dict[str, str] = {}
        for e in entries:
            miss = next((r for r in _deps(e, soft=False)
                         if r not in first or first[r].state != "on"), None)
            if e.state == "on" and miss is not None:
                unmet[e.name] = _unmet(miss, first.get(miss))
        if not unmet:
            return entries
        entries = [replace(e, state="failed", reason=unmet[e.name])
                   if e.state == "on" and e.name in unmet else e for e in entries]


def _unmet(name: str, dep: PlanEntry | None) -> str:
    return f"requires {name}, which is " + (
        "not installed" if dep is None else _NOT_ON.get(dep.state, dep.state))


def _order(entries: list[PlanEntry]) -> tuple[list[str], list[str]]:
    """Kahn's algorithm over the on plugins — `requires` plus `uses` naming an on plugin — ties in
    display order: (start order, one dependency cycle among what is left, or [] when none is)."""
    on = [e for e in entries if e.state == "on"]
    names = {e.name for e in on}
    deps = {e.name: [d for d in dict.fromkeys(_deps(e, soft=True)) if d in names] for e in on}
    order: list[str] = []
    done: set[str] = set()
    while ready := [n for n in deps if n not in done and all(d in done for d in deps[n])]:
        order.append(ready[0])
        done.add(ready[0])
    left = [n for n in deps if n not in done]
    if not left:
        return order, []
    path = [left[0]]  # every plugin left waits on another one left: walk until a name repeats
    while (step := next(d for d in deps[path[-1]] if d not in done)) not in path:
        path.append(step)
    return order, path[path.index(step):]


def extra_installed(extra: str) -> bool:
    """Are `localharness[<extra>]`'s requirements installed? Reads the installed distribution's
    metadata — its Provides-Extra and Requires-Dist, evaluated with packaging markers — and imports
    nothing. An extra this installation does not declare is not installed."""
    try:
        dist = importlib.metadata.distribution("localharness")
        declared = {canonicalize_name(e) for e in dist.metadata.get_all("Provides-Extra") or ()}
        if canonicalize_name(extra) not in declared:
            return False
        for line in dist.requires or ():
            req = Requirement(line)
            if req.marker is None or not req.marker.evaluate({"extra": extra}) \
                    or req.marker.evaluate({"extra": ""}):
                continue  # not this extra's requirement (or not needed on this platform)
            if not _admits(req.specifier, importlib.metadata.distribution(req.name).version):
                return False
        return True
    except importlib.metadata.PackageNotFoundError:
        return False
