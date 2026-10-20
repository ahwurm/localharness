"""The subject model reaches for browser_fetch when web_fetch comes back thin: one real web-researcher turn."""
from __future__ import annotations

import importlib.util

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import WEB_TOOLS, builtin_toolset, dispatch_web_subagent
from localharness.bench.config import MatrixEntry
from localharness.bench.orchestrator import _build_bench_client
from localharness.core.bus import EventBus
from localharness.core.events import Action, Observation
from localharness.plugins.api import PluginContext, PluginPaths
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.builtin.browser_plugin import BrowserConfig, BrowserPlugin
from localharness.tools.builtin.web_tool import WebFetchTool
from localharness.tools.registry import ToolRegistry
from tests.conftest import live_target
from tests.integration.test_browser_reach_live import BROWSERS, CLIENT_RENDERED, SANDBOX

pytestmark = pytest.mark.live_vllm


def same(a: str | None, b: str | None) -> bool:
    return (a or "").strip().rstrip("/") == (b or "").strip().rstrip("/")


async def test_after_web_fetch_comes_back_thin_the_researcher_calls_browser_fetch_on_the_same_url(
        live_endpoint, monkeypatch, tmp_path):
    if importlib.util.find_spec("playwright") is None:
        pytest.skip("the browser is not installed: uv sync --inexact --extra dev --extra browser")
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", BROWSERS)
    model, base_url = live_target()
    client = _build_bench_client(MatrixEntry(name=model, provider="vllm", model_id=model, base_url=base_url))
    await client.detect_capabilities()

    thin = await WebFetchTool().run(url=CLIENT_RENDERED)
    assert "low extractable text" in thin.output, \
        f"web_fetch reads {CLIENT_RENDERED} without a browser; set LOCALHARNESS_LIVE_BROWSER_URL to a client-rendered https page: {thin.output[:200]!r}"

    base = ToolRegistry()
    await register_builtin_tools(base)
    ctx = PluginContext(
        bus=None, tools=None, hooks=None, config=BrowserConfig(sandbox=SANDBOX), agent_config=None,
        paths=PluginPaths(global_config_dir=tmp_path / "cfg", workspace=None, state_dir=tmp_path / "state"), llm=None)
    plugin = BrowserPlugin()
    tools = await plugin.tools(ctx)
    if not tools:
        pytest.skip(f"Playwright's headless shell is not in {BROWSERS}: uv run localharness plugins enable browser downloads it")
    try:
        for tool in tools:
            await base.register(tool, scope="global", source_plugin="browser")
        assert builtin_toolset("web-researcher", base) == [*WEB_TOOLS, "browser_fetch"]
        bus = EventBus()
        await dispatch_web_subagent(
            f"Read {CLIENT_RENDERED} and say in one sentence what the page offers.",
            llm=client, bus=bus, base_registry=base, parent_session_id="live",
            permission_evaluator=PermissionEvaluator(), context_manager=ContextManager())
    finally:
        await plugin.stop(ctx)

    events = bus.history(event_types=[Action, Observation])
    is_call = lambda e: isinstance(e, Action) and e.action_type == "tool_call"  # noqa: E731
    url = lambda e: (e.tool_params or {}).get("url")  # noqa: E731
    calls = [(e.tool_name, url(e)) for e in events if is_call(e)]
    first = next((i for i, e in enumerate(events) if is_call(e) and e.tool_name == "web_fetch" and same(url(e), CLIENT_RENDERED)), None)
    if first is None:
        pytest.fail(f"the model never called web_fetch on the page; its calls were {calls}")
    seen = next((i for i, e in enumerate(events) if i > first and isinstance(e, Observation)
                 and e.observation_type == "tool_result" and e.tool_name == "web_fetch"), None)
    if seen is None:
        pytest.fail(f"web_fetch on the page returned no result; the calls were {calls}")
    nxt = next((e for e in events[seen + 1:] if is_call(e)), None)
    assert nxt is not None and nxt.tool_name == "browser_fetch" and same(url(nxt), CLIENT_RENDERED), \
        f"after web_fetch came back thin the next call was not browser_fetch on {CLIENT_RENDERED}; the calls were {calls}"
