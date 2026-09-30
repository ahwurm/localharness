"""Plugin discovery — METADATA ONLY (ENAB-06, PRD decision 12).

Two sources, both machine-level: installed packages that declare a `localharness.plugins` entry point,
and folders `plugins/<name>/` in the GLOBAL config dir. A workspace is never a source. Discovering a
plugin reads names and versions and nothing else — no import, no entry point is loaded, no
module-level code runs. `load_plugin_class()` is the import step; core calls it only for a plugin
already enabled. The incident record for auto-loaded plugins (the ClawHub campaign, Open WebUI
CVE-2025-64496) is why.

0.15.0's plugin API (`localharness.tools` / `localharness.hooks` entry points, `plugins/<dir>/
manifest.yaml` folders) is not a source. `legacy_notices()` names what still uses it, from metadata
and file names only, so an upgrade never drops a plugin in silence; nothing loads it.
"""
from __future__ import annotations

import functools
import importlib
import importlib.metadata
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from localharness.plugins.api import Plugin

PLUGIN_ENTRY_POINT_GROUP = "localharness.plugins"
PLUGINS_DIR_NAME = "plugins"
FOLDER_PLUGIN_ATTR = "plugin"   # a folder plugin's __init__.py binds `plugin = <its Plugin class>`
_FOLDER_PACKAGE = "localharness_folder_plugins"  # the synthetic parent a folder plugin imports under
LEGACY_ENTRY_POINT_GROUPS = ("localharness.tools", "localharness.hooks")
"""0.15.0's documented plugin API. Nothing loads either group since 0.16; legacy_notices() names them."""
LEGACY_API_DOC = ("https://github.com/ahwurm/localharness/blob/main/docs/specs/"
                  "09-hooks-plugins.md#plugins-written-for-015")
_LEGACY_MANIFEST = "manifest.yaml"  # a 0.15.0 folder plugin's declaration, beside no __init__.py
_NOT_LOADED = f"was built for the 0.15 plugin API and is no longer loaded — see {LEGACY_API_DOC}"


@dataclass(frozen=True)
class DiscoveredPlugin:
    """One plugin as its metadata describes it — nothing it would say about itself if it ran."""

    name: str
    source: Literal["entry_point", "folder"]
    target: str                      # "module:attr" for an entry point; the folder's absolute path
    dist_name: str | None = None
    dist_version: str | None = None

    @property
    def from_label(self) -> str:
        """Where it came from, for `plugins list`: "pip: <dist> <version>" or "folder: <path>"."""
        if self.source == "folder":
            return f"folder: {self.target}"
        return " ".join(p for p in ("pip:", self.dist_name or self.target, self.dist_version) if p)


def discover(global_config_dir: Path) -> list[DiscoveredPlugin]:
    """Every plugin the machine offers: entry points first (in importlib.metadata's order), then the
    global `plugins/` folder's subdirectories that hold an `__init__.py`, by name. Duplicate or
    invalid names are listed as found — refusing them is the load plan's job, with a reason."""
    found = [
        DiscoveredPlugin(ep.name, "entry_point", ep.value,
                         ep.dist.name if ep.dist else None, ep.dist.version if ep.dist else None)
        for ep in importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    ]
    folder = Path(global_config_dir) / PLUGINS_DIR_NAME
    if folder.is_dir():
        found += [DiscoveredPlugin(d.name, "folder", str(d.absolute()))
                  for d in sorted(folder.iterdir()) if (d / "__init__.py").is_file()]
    return found


def legacy_notices(global_config_dir: Path) -> list[str]:
    """One line per plugin written for 0.15.0's plugin API: an installed distribution with a
    `localharness.tools` / `localharness.hooks` entry point (unless it also declares a
    `localharness.plugins` one), and a GLOBAL `plugins/<dir>/` holding `manifest.yaml` and no
    `__init__.py`. Names only: one metadata scan, nothing imported, no plugin file opened — and
    none of them is ever loaded."""
    legacy: dict[str, list[str]] = {}
    ported: set[str] = set()
    for ep in importlib.metadata.entry_points():  # ONE scan of every group, filtered here
        if ep.group not in (*LEGACY_ENTRY_POINT_GROUPS, PLUGIN_ENTRY_POINT_GROUP):
            continue
        dist = f"{ep.dist.name} {ep.dist.version}" if ep.dist else ep.value
        if ep.group == PLUGIN_ENTRY_POINT_GROUP:
            ported.add(dist)  # it supports today's API too: that entry point is what loads
        else:
            legacy.setdefault(dist, []).append(f"`{ep.group}` entry point `{ep.name}`")
    lines = [f"{dist} ({', '.join(eps)}) {_NOT_LOADED}"
             for dist, eps in legacy.items() if dist not in ported]
    folder = Path(global_config_dir) / PLUGINS_DIR_NAME
    if folder.is_dir():
        lines += [f"{d.absolute() / _LEGACY_MANIFEST} {_NOT_LOADED}" for d in sorted(folder.iterdir())
                  if (d / _LEGACY_MANIFEST).is_file() and not (d / "__init__.py").is_file()]
    return lines


def load_plugin_class(found: DiscoveredPlugin) -> type[Plugin]:
    """Import one discovered plugin and return its class — THE import step, for an enabled plugin
    only. An import error propagates unchanged (the caller contains it and names the plugin); a
    target that is not a Plugin subclass is a TypeError."""
    obj = _import_folder(found) if found.source == "folder" else import_target(found.target)
    if not (isinstance(obj, type) and issubclass(obj, Plugin)):
        hint = (f" (its __init__.py must bind `{FOLDER_PLUGIN_ATTR} = <the class>`)"
                if found.source == "folder" else "")
        raise TypeError(f"{found.target} does not name a localharness Plugin subclass{hint}")
    return obj


def import_target(target: str) -> Any:
    """`"package.module:attr.sub"` → that object; a target with no `:` is the module itself."""
    module_name, _, attrs = target.partition(":")
    module = importlib.import_module(module_name)
    return functools.reduce(getattr, attrs.split("."), module) if attrs else module


def _import_folder(found: DiscoveredPlugin) -> Any:
    """Import a folder plugin by FILE LOCATION under a synthetic package name, once per process,
    like any import."""
    # By location, never by path insertion: sys.path is not touched (the dormant loader's was).
    folder = Path(found.target)
    init = folder / "__init__.py"
    name = f"{_FOLDER_PACKAGE}.{found.name.replace('-', '_')}"
    module = sys.modules.get(name)
    if module is None or getattr(module, "__file__", None) != str(init):
        # A same-named folder from another config dir: its submodules must not answer for this one.
        for stale in [m for m in sys.modules if m.startswith(name + ".")]:
            del sys.modules[stale]
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(folder)])
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import the folder plugin at {folder}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module  # BEFORE exec: the package's relative imports resolve through it
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)  # as importlib does: no half-initialized module stays behind
            raise
    return getattr(module, FOLDER_PLUGIN_ATTR, None)
