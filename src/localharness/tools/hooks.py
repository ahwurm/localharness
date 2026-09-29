"""HookSystem: pluggy dispatch of the two tool hooks (pre_tool / post_tool), wired to ToolRegistry."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pluggy

from localharness.tools.base import ToolVetoed

if TYPE_CHECKING:
    from localharness.tools.registry import ToolRegistry

HARNESS_HOOKSPEC = pluggy.HookspecMarker("localharness")
HARNESS_HOOKIMPL = pluggy.HookimplMarker("localharness")

_log = logging.getLogger(__name__)


class HarnesHookSpec:
    """The two tool hooks — the only pluggy hooks LocalHarness keeps (PRD decision 6). A plugin's
    lifecycle (configure / tools / start / stop) comes from plugins/api.Plugin, not from pluggy.

    Implementations are plain functions, each called on its own and synchronously: an exception is
    caught and logged under the implementing plugin's name (PAPI-11). pluggy wrapper
    implementations are not supported and are skipped with a warning.
    """

    @HARNESS_HOOKSPEC
    def pre_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_id: str,
        division_id: str,
    ) -> None:
        """Called before a tool's run() method is invoked.

        Implementations MAY raise ToolVetoed to prevent execution. Any other exception is caught,
        logged with the plugin's name, and the call goes on.
        """

    @HARNESS_HOOKSPEC
    def post_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        agent_id: str,
        division_id: str,
    ) -> None:
        """Called after a tool's run() method returns. Observability only.

        An exception here (ToolVetoed included) is caught and logged with the plugin's name; the
        tool's result is unchanged.
        """


class HookSystem:
    """Registers pre_tool / post_tool implementations and calls them from ToolRegistry.dispatch.

    pluggy is kept ONLY for these two tool hooks (PRD decision 6); plugins get their lifecycle
    through plugins/api.Plugin. Instantiated once at harness startup; wire_to_registry() connects
    it to a ToolRegistry, and implementations registered afterwards are called too.
    """

    def __init__(self) -> None:
        self.pm = pluggy.PluginManager("localharness")
        self.pm.add_hookspecs(HarnesHookSpec)
        self._loaded_plugins: list[str] = []

    def register_plugin(self, plugin: object, name: str | None = None) -> None:
        """Register a hook implementation object (dedup-safe). Pass `name` — a plugin passes its own
        plugin name — so a hook that raises is reported under it rather than an object id."""
        if not self.pm.is_registered(plugin):
            self.pm.register(plugin, name=name)

    # register_impl and loaded_plugin_names are removed with their last readers (start_cmd verbose
    # line, catalogue hooks source).
    def register_impl(self, instance: object, name: str) -> None:
        """Register a hook implementation by name (used by PluginLoader)."""
        if self.pm.is_registered(instance):
            return
        self.pm.register(instance, name=name)
        self._loaded_plugins.append(name)

    def _call_each(self, hook_name: str, *, veto: bool, **kwargs: Any) -> None:
        """Call every implementation of `hook_name` separately (PAPI-11): an exception is caught and
        attributed — by pluggy plugin name and the implementation's module — instead of being
        swallowed whole, as the single pm.hook.<name>(...) call did. ToolVetoed propagates from
        pre_tool only. Order matches pluggy's own call order (last registered first)."""
        for impl in reversed(getattr(self.pm.hook, hook_name).get_hookimpls()):
            where = getattr(impl.function, "__module__", "?")
            if impl.wrapper or impl.hookwrapper:
                _log.warning("%s hook from plugin %r (%s) is a pluggy wrapper — not supported, skipped",
                             hook_name, impl.plugin_name, where)
                continue
            try:
                impl.function(**{a: kwargs[a] for a in impl.argnames if a in kwargs})
            except ToolVetoed:
                if veto:
                    raise
                _log.warning("%s hook from plugin %r (%s) raised ToolVetoed — only pre_tool can veto",
                             hook_name, impl.plugin_name, where)
            except Exception:
                _log.warning("%s hook from plugin %r (%s) raised — ignored, the tool call goes on",
                             hook_name, impl.plugin_name, where, exc_info=True)

    def wire_to_registry(self, registry: "ToolRegistry") -> None:
        """Connect pluggy hook dispatch to ToolRegistry pre/post hook lists."""

        async def pre_hook_caller(
            name: str, arguments: dict, agent_id: str, **kwargs: Any
        ) -> None:
            self._call_each(
                "pre_tool", veto=True, name=name, arguments=arguments, agent_id=agent_id,
                division_id=kwargs.get("division_id", "default"),
            )

        async def post_hook_caller(
            name: str, arguments: dict, result: Any, agent_id: str, **kwargs: Any
        ) -> None:
            self._call_each(
                "post_tool", veto=False, name=name, arguments=arguments, result=result,
                agent_id=agent_id, division_id=kwargs.get("division_id", "default"),
            )

        registry.register_pre_hook(pre_hook_caller)
        registry.register_post_hook(post_hook_caller)

    @property
    def loaded_plugin_names(self) -> list[str]:
        return list(self._loaded_plugins)
