"""Resolve this config's plugins: discover (metadata), decide what is on, import ONLY what is on,
build the plan, validate each loaded plugin's own settings (ENAB-01/02/06, SAFE-06, PAPI-11).

The one answer to "what plugins are there, what is on, and why" — the session, the banner, `plugins
list`, doctor, components and the CLI mount all call resolve(). Nothing a user has not enabled is
ever imported: a plugin you installed is off until the machine-level config turns it on, and until
then only its metadata is read.
"""
from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from localharness import resolved_version
from localharness.config.plugin_sections import (
    CORE_AGENT_KEYS, CORE_HARNESS_KEYS, PluginSectionError, global_only_paths, merge_plugin_layers,
)
from localharness.plugins import discovery, plan
from localharness.plugins.api import Plugin
from localharness.plugins.builtin import bundled_plugins

if TYPE_CHECKING:
    from localharness.config.loader import ConfigLoader


@dataclass(frozen=True)
class PluginSettings:
    """One plugin's validated settings: its ConfigModel instance (the harness-level `<name>:`) and
    its AgentConfigModel instance (`agent.<name>`), each None when it declares no such model."""

    config: BaseModel | None
    agent_config: BaseModel | None


@dataclass(frozen=True)
class Resolution:
    """What resolve() found: the plan, the plugin classes in hand, the settings that validated,
    every plugin's resolved `enabled` flag, and the warnings to show.

    `classes` and `settings` cover every bundled plugin (on or off, so its settings can be listed
    before it is switched on) and every imported one; never a plugin that is only available.
    `warnings` never repeats what problems() says."""

    plan: plan.LoadPlan
    classes: Mapping[str, type[Plugin]]
    settings: Mapping[str, PluginSettings]
    enabled: Mapping[str, bool]          # the resolved `<name>.enabled` flag of every known plugin
    warnings: tuple[str, ...]

    def problems(self) -> list[str]:
        """One line per plugin the plan failed, refused, skipped or holds for an install extra:
        "plugin <name>: <reason>"."""
        return _problems(self.plan)


def _problems(load_plan: plan.LoadPlan) -> list[str]:
    return [f"plugin {e.name}: {e.reason}" for e in load_plan.entries
            if e.state in ("failed", "refused", "skipped", "needs-extra")]


class _Unusable(Exception):
    """A plugin's settings (or its declaration of them) cannot be used; the message is the reason."""


def resolve(loader: ConfigLoader, *, agent_name: str | None = None, version: str | None = None,
            extra_installed: Callable[[str], bool] = plan.extra_installed) -> Resolution:
    """Resolve the plugins of the config `loader` reads.

    1. Inputs: the bundled list (read now), discovered METADATA from the GLOBAL config dir (entry
       points, plugins/ folders — never the workspace), each plugin's four raw settings layers.
    2. `enabled`: a bundled plugin's is layered like any key and defaults to its manifest's
       enabled_by_default; a plugin you installed has its read from the GLOBAL layers only (a
       workspace value is dropped with a warning) and defaults to False — its manifest is not even
       read before it is enabled. A value that is not true or false warns and keeps the default.
    3. Import ONLY an installed plugin that is enabled and whose name the plan admits (valid, not a
       core key, not taken). Whatever its import raises, SystemExit included, goes to the plan: a
       refused plugin's code never runs and the harness never exits because of a plugin.
    4. Settings, for every plugin in hand — every bundled one, on or off, and every imported one:
       its layers merged and narrowed at its ConfigModel's GLOBAL_ONLY paths (plus `enabled` for a
       plugin you installed), `enabled` removed, validated with its ConfigModel; the agent level
       (the loaded agent's `agent.<name>`) with its AgentConfigModel. A failure is that plugin's
       reason in the plan, never an error for the harness.
    5. The plan, pure, against `version` (default: this package's own version).
    6. Warnings in the order met, each once, and never a line problems() already says — so an off
       plugin's invalid settings are a warning, an on plugin's are its reason in the plan. Plugins
       built for the 0.15 API come first (discovery.legacy_notices), named and never loaded.

    `agent_name` is the `name:` of an agent this loader has already loaded; None: no agent, and
    each AgentConfigModel validates empty (its defaults).
    """
    bundled = bundled_plugins()
    found = discovery.discover(loader.global_config_dir)
    layers, files = loader.plugin_layers(), loader.plugin_layer_files()
    core_keys = CORE_HARNESS_KEYS | CORE_AGENT_KEYS
    refusals = plan.name_refusals(
        [(c.manifest.name, "built in") for c in bundled] + [(d.name, d.from_label) for d in found],
        core_keys)
    admitted_bundled = [c for c, why in zip(bundled, refusals) if why is None]
    admitted_found = [d for d, why in zip(found, refusals[len(bundled):]) if why is None]
    warnings: list[str] = discovery.legacy_notices(loader.global_config_dir)

    def merged(name: str, global_only: frozenset[str]) -> dict[str, Any]:
        section, dropped = merge_plugin_layers(name, layers.get(name, (None,) * 4),
                                               global_only=global_only, layer_files=files)
        warnings.extend(dropped)
        return section

    def flag(name: str, default: bool, *, bundled: bool) -> bool:
        try:
            value = merged(name, _machine_only(None, bundled)).get("enabled", default)
        except PluginSectionError as exc:  # a section that is not a mapping: reported, and then a
            warnings.append(f"plugin {name}: invalid settings — {exc}")  # project's cannot decide
            machine = (*layers.get(name, (None,) * 4)[:2], None, None)
            try:
                value = merge_plugin_layers(name, machine, global_only=frozenset(),
                                            layer_files=files)[0].get("enabled", default)
            except PluginSectionError:
                return default
        if isinstance(value, bool):
            return value
        warnings.append(f"plugin {name}: `{name}.enabled` must be true or false, not {value!r} — "
                        f"it stays {'on' if default else 'off'}")
        return default

    enabled = {c.manifest.name: flag(c.manifest.name, c.manifest.enabled_by_default, bundled=True)
               for c in admitted_bundled}
    enabled |= {d.name: flag(d.name, False, bundled=False) for d in admitted_found}

    imported: dict[str, type[Plugin] | BaseException] = {}
    for d in admitted_found:
        if enabled[d.name]:
            try:
                imported[d.name] = discovery.load_plugin_class(d)
            except (Exception, SystemExit) as exc:
                imported[d.name] = exc

    in_hand = [(c.manifest.name, c, True) for c in admitted_bundled] + [
        (n, c, False) for n, c in imported.items()
        if isinstance(c, type) and (m := plan.manifest_of(c)) is not None and m.name == n]
    agent_sections = loader.agent_plugin_sections(agent_name) if agent_name is not None else {}
    settings: dict[str, PluginSettings] = {}
    invalid: dict[str, str] = {}
    for name, cls, is_bundled in in_hand:
        try:
            marked = global_only_paths(cls.AgentConfigModel)
            if marked:
                raise _Unusable(f"it marks agent.{name}.{min(marked)} machine-level only, which "
                                "agent-level settings cannot enforce")
            section = merged(name, _machine_only(cls.ConfigModel, is_bundled))
            settings[name] = PluginSettings(
                _validate(cls.ConfigModel, {k: v for k, v in section.items() if k != "enabled"}, name),
                _validate(cls.AgentConfigModel, agent_sections.get(name, {}), f"agent.{name}"))
        except _Unusable as exc:
            invalid[name] = str(exc)
        except PluginSectionError as exc:
            invalid[name] = f"invalid settings — {exc}"
        except (Exception, SystemExit) as exc:  # a plugin's own validator raised
            invalid[name] = f"invalid settings — checking them raised {type(exc).__name__}: {exc}"

    load_plan = plan.build_load_plan(
        bundled=bundled, discovered=found, enabled=enabled, imported=imported,
        version=version or resolved_version(), core_keys=core_keys,
        extra_installed=extra_installed, invalid=invalid)
    warnings += [f"plugin {n}: {why}" for n, why in invalid.items()]
    said = set(_problems(load_plan))
    return Resolution(load_plan, {n: c for n, c, _ in in_hand}, settings, enabled,
                      tuple(w for w in dict.fromkeys(warnings) if w not in said))


def _machine_only(model: type[BaseModel] | None, bundled: bool) -> frozenset[str]:
    """The paths only the global layers may set: the model's GLOBAL_ONLY fields, and `enabled` too
    for a plugin you installed — turning it on is the operator's trust grant (SAFE-06)."""
    return global_only_paths(model) | (frozenset() if bundled else frozenset({"enabled"}))


def _validate(model: type[BaseModel] | None, section: Any, where: str) -> BaseModel | None:
    """`section` validated with `model` (a deep copy: the loader's cached sources stay untouched),
    or None for a plugin that declares no model and has no section."""
    section = {} if section is None else section
    if not isinstance(section, Mapping):
        raise _Unusable(f"invalid settings — `{where}:` must be a mapping of settings, not "
                        f"{type(section).__name__}")
    if model is None:
        if section:
            raise _Unusable(f"invalid settings — it takes no settings, but `{where}:` sets "
                            f"{', '.join(sorted(map(str, section)))}")
        return None
    try:
        return model.model_validate(copy.deepcopy(dict(section)))
    except ValidationError as exc:
        first = exc.errors()[0]
        more = f" (and {exc.error_count() - 1} more)" if exc.error_count() > 1 else ""
        raise _Unusable(f"invalid settings — {'.'.join([where, *map(str, first['loc'])])}: "
                        f"{' '.join(first['msg'].split())}{more}") from exc
