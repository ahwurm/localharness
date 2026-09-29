"""pluggy keeps only pre_tool/post_tool, and every hook call is contained and attributed (PAPI-11,
PRD decision 6).

Driven through a real ToolRegistry + HookSystem.wire_to_registry + registry.dispatch. The property:
one plugin's exception cannot swallow another plugin's hook or the tool, and the warning NAMES the
plugin that raised — a single pm.hook.<name>(...) call inside one try/except did neither.
"""
from __future__ import annotations

import logging
from typing import Any

from localharness.config.models import ToolConfig
from localharness.tools.base import Tool, ToolResult, ToolSchema, ToolVetoed
from localharness.tools.hooks import HARNESS_HOOKIMPL, HarnesHookSpec, HookSystem
from localharness.tools.registry import ToolRegistry

LOGGER = "localharness.tools.hooks"


class _Echo(Tool):
    """A trivial tool that counts its runs."""

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="echo", description="Echo.",
            parameters={"type": "object", "properties": {}, "required": []},
            ingest="none", host="safe", result_origin="trusted", gate_family=None,  # SAFE-01, explicit
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.runs += 1
        return self.ok("echoed")


async def _dispatch(*plugins: tuple[str, object]) -> tuple[ToolResult, _Echo]:
    registry, hooks, tool = ToolRegistry(), HookSystem(), _Echo()
    await registry.register(tool, scope="global")
    for name, obj in plugins:
        hooks.register_plugin(obj, name=name)
    hooks.wire_to_registry(registry)
    result = await registry.dispatch(
        name="echo", arguments={}, agent_id="a", division_id="d", tool_config=ToolConfig())
    return result, tool


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def test_only_the_two_tool_hooks_remain():
    assert {n for n in vars(HarnesHookSpec) if not n.startswith("_")} == {"pre_tool", "post_tool"}
    assert sorted(n for n in vars(HookSystem().pm.hook) if not n.startswith("_")) == ["post_tool", "pre_tool"]
    # No per-event dispatchers: tool hooks reach pluggy only through wire_to_registry.
    assert not [n for n in vars(HookSystem) if n.startswith("call_")]


async def test_a_raising_pre_tool_is_named_and_contained(caplog):
    calls: list[str] = []

    class Good:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name, arguments, agent_id, division_id):
            calls.append(name)

    class Boom:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name, arguments, agent_id, division_id):
            raise RuntimeError("boom")

    # pluggy calls the LAST registered first, so boomplug runs before goodplug: one
    # pm.hook.pre_tool(...) call would stop at boomplug and goodplug would never run.
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result, tool = await _dispatch(("goodplug", Good()), ("boomplug", Boom()))

    assert (result.success, result.output, tool.runs) == (True, "echoed", 1)
    assert calls == ["echo"]
    [record] = _warnings(caplog)
    message = record.getMessage()
    assert "boomplug" in message and "pre_tool" in message and __name__ in message
    assert "goodplug" not in message
    assert record.exc_info and record.exc_info[0] is RuntimeError


async def test_a_pre_tool_veto_still_vetoes():
    class Veto:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name, arguments, agent_id, division_id):
            raise ToolVetoed("blocked by policy")

    result, tool = await _dispatch(("vetoplug", Veto()))
    assert (result.success, result.error_type, tool.runs) == (False, "permission_denied", 0)
    assert "blocked by policy" in result.error


async def test_a_raising_post_tool_is_named_and_the_result_is_unchanged(caplog):
    seen: list[str] = []

    class Watch:
        @HARNESS_HOOKIMPL
        def post_tool(self, name, arguments, result, agent_id, division_id):
            seen.append(result.output)

    class Boom:
        @HARNESS_HOOKIMPL
        def post_tool(self, name, arguments, result, agent_id, division_id):
            raise RuntimeError("post boom")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result, tool = await _dispatch(("watchplug", Watch()), ("postboom", Boom()))

    assert (result.success, result.output, tool.runs) == (True, "echoed", 1)
    assert seen == ["echoed"]
    [record] = _warnings(caplog)
    assert "postboom" in record.getMessage() and "post_tool" in record.getMessage()


async def test_a_veto_from_post_tool_is_reported_not_obeyed(caplog):
    class LateVeto:
        @HARNESS_HOOKIMPL
        def post_tool(self, name, arguments, result, agent_id, division_id):
            raise ToolVetoed("too late")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result, tool = await _dispatch(("lateveto", LateVeto()))

    assert (result.success, tool.runs) == (True, 1)
    [record] = _warnings(caplog)
    assert "lateveto" in record.getMessage() and "only pre_tool can veto" in record.getMessage()


async def test_a_wrapper_implementation_is_skipped_loudly(caplog):
    """A wrapper cannot run as a plain call; it is skipped with a named warning rather than turned
    into a generator nobody iterates (which would silently drop, e.g., a veto)."""

    class Wrap:
        @HARNESS_HOOKIMPL(wrapper=True)
        def pre_tool(self, name, arguments, agent_id, division_id):
            raise ToolVetoed("a wrapper's veto")
            yield  # pragma: no cover — makes this a generator, as pluggy wrappers are

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result, tool = await _dispatch(("wrapplug", Wrap()))

    assert (result.success, tool.runs) == (True, 1)
    [record] = _warnings(caplog)
    assert "wrapplug" in record.getMessage() and "wrapper" in record.getMessage()


def test_register_plugin_by_name_is_dedup_safe():
    class Noop:
        @HARNESS_HOOKIMPL
        def pre_tool(self, name, arguments, agent_id, division_id):
            pass

    hooks, obj = HookSystem(), Noop()
    hooks.register_plugin(obj, name="x")
    hooks.register_plugin(obj, name="x")
    hooks.register_plugin(obj)
    assert hooks.pm.get_plugin("x") is obj and hooks.pm.get_name(obj) == "x"
    assert len(hooks.pm.hook.pre_tool.get_hookimpls()) == 1
