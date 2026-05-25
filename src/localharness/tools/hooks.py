"""HookSystem: pluggy-based hook dispatch wired to ToolRegistry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pluggy

from localharness.tools.base import ToolVetoed

if TYPE_CHECKING:
    from localharness.tools.registry import ToolRegistry

HARNESS_HOOKSPEC = pluggy.HookspecMarker("localharness")
HARNESS_HOOKIMPL = pluggy.HookimplMarker("localharness")


class HarnesHookSpec:
    """Pluggy hook specifications for LocalHarness.

    All hooks are optional. Plugin implementations may implement any subset.
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

        Implementations MAY raise ToolVetoed to prevent execution.
        Any other exception is swallowed (plugins must not crash the harness).
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

        Exceptions raised here are caught and ignored.
        """

    @HARNESS_HOOKSPEC
    def on_agent_start(
        self,
        agent_id: str,
        division_id: str,
        task: str,
        iteration_budget: int,
    ) -> None:
        """Called once when an agent loop starts a new turn."""

    @HARNESS_HOOKSPEC
    def on_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        """Called once when an agent loop ends (success or failure)."""

    @HARNESS_HOOKSPEC
    def on_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        agent_id: str | None,
    ) -> None:
        """Called for every event emitted on the event bus."""


class HookSystem:
    """Manages plugin discovery, registration, and hook dispatch.

    Instantiated once at harness startup. Wire to a ToolRegistry via
    wire_to_registry() after all plugins are registered.
    """

    def __init__(self) -> None:
        self.pm = pluggy.PluginManager("localharness")
        self.pm.add_hookspecs(HarnesHookSpec)
        self._loaded_plugins: list[str] = []

    def register_plugin(self, plugin: object) -> None:
        """Register a hook implementation instance (dedup-safe)."""
        if not self.pm.is_registered(plugin):
            self.pm.register(plugin)

    def register_impl(self, instance: object, name: str) -> None:
        """Register a hook implementation by name (used by PluginLoader)."""
        if self.pm.is_registered(instance):
            return
        self.pm.register(instance, name=name)
        self._loaded_plugins.append(name)

    def wire_to_registry(self, registry: "ToolRegistry") -> None:
        """Connect pluggy hook dispatch to ToolRegistry pre/post hook lists."""

        async def pre_hook_caller(
            name: str, arguments: dict, agent_id: str, **kwargs: Any
        ) -> None:
            division_id = kwargs.get("division_id", "default")
            try:
                self.pm.hook.pre_tool(
                    name=name,
                    arguments=arguments,
                    agent_id=agent_id,
                    division_id=division_id,
                )
            except ToolVetoed:
                raise
            except Exception:
                pass

        async def post_hook_caller(
            name: str, arguments: dict, result: Any, agent_id: str, **kwargs: Any
        ) -> None:
            division_id = kwargs.get("division_id", "default")
            try:
                self.pm.hook.post_tool(
                    name=name,
                    arguments=arguments,
                    result=result,
                    agent_id=agent_id,
                    division_id=division_id,
                )
            except Exception:
                pass

        registry.register_pre_hook(pre_hook_caller)
        registry.register_post_hook(post_hook_caller)

    def call_agent_start(
        self, agent_id: str, division_id: str, task: str, iteration_budget: int
    ) -> None:
        try:
            self.pm.hook.on_agent_start(
                agent_id=agent_id,
                division_id=division_id,
                task=task,
                iteration_budget=iteration_budget,
            )
        except Exception:
            pass

    def call_agent_end(
        self,
        agent_id: str,
        division_id: str,
        summary: str,
        iterations_used: int,
        success: bool,
        error: str | None,
    ) -> None:
        try:
            self.pm.hook.on_agent_end(
                agent_id=agent_id,
                division_id=division_id,
                summary=summary,
                iterations_used=iterations_used,
                success=success,
                error=error,
            )
        except Exception:
            pass

    def call_on_event(
        self, event_type: str, event_data: dict, agent_id: str | None
    ) -> None:
        try:
            self.pm.hook.on_event(
                event_type=event_type,
                event_data=event_data,
                agent_id=agent_id,
            )
        except Exception:
            pass

    @property
    def loaded_plugin_names(self) -> list[str]:
        return list(self._loaded_plugins)
