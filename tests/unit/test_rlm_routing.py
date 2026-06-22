"""RLM auto-router (`_rlm_should_activate`) and the in-REPL recursion bridge
(`_make_rlm_recursion_fn`) — both in localharness.agent.loop."""
import asyncio
from types import SimpleNamespace

import pytest

from localharness.agent.loop import _rlm_should_activate, _make_rlm_recursion_fn
from localharness.config.models import RLMConfig
from localharness.tools.builtin.python_tool import PythonExecTool


# --- router: _rlm_should_activate -------------------------------------------------

def test_router_enabled_always_active():
    # enabled wins regardless of input size / auto
    assert _rlm_should_activate(RLMConfig(enabled=True), 0, 100_000) is True


def test_router_off_when_neither_flag():
    # default config: never routes, even for an absurdly large input
    assert _rlm_should_activate(RLMConfig(), 10_000_000, 1000) is False


def test_router_auto_routes_when_over_threshold():
    cfg = RLMConfig(auto=True, auto_threshold=0.80)  # threshold = int(1000*0.8) = 800
    assert _rlm_should_activate(cfg, 801, 1000) is True


def test_router_auto_stays_direct_within_window():
    cfg = RLMConfig(auto=True, auto_threshold=0.80)
    assert _rlm_should_activate(cfg, 800, 1000) is False  # strict > : equal does NOT route
    assert _rlm_should_activate(cfg, 200, 1000) is False


def test_router_zero_window_does_not_route():
    assert _rlm_should_activate(RLMConfig(auto=True), 999, 0) is False


def test_router_threshold_is_configurable():
    cfg = RLMConfig(auto=True, auto_threshold=0.5)
    assert _rlm_should_activate(cfg, 501, 1000) is True
    assert _rlm_should_activate(cfg, 499, 1000) is False


# --- recursion bridge: _make_rlm_recursion_fn ------------------------------------

class _FakeLLM:
    """Stand-in for LLMClient: stream_complete(messages) -> (message, usage)."""
    def __init__(self, reply):
        self._reply = reply
        self.seen = []

    async def stream_complete(self, messages, tools=None, on_token=None):
        self.seen.append(messages)
        await asyncio.sleep(0)  # really yield to the loop, like a real network call
        return SimpleNamespace(content=self._reply), None


@pytest.mark.asyncio
async def test_recursion_fn_bridges_thread_to_loop():
    """python_exec runs the model's code in a WORKER thread; the in-REPL llm() must bridge
    back to the running loop and return the sub-model reply."""
    fake = _FakeLLM("chunk summary: 3 incidents")
    fn = _make_rlm_recursion_fn(fake, asyncio.get_running_loop())
    result = await asyncio.to_thread(fn, "Summarize: <chunk>")  # call from a non-loop thread
    assert result == "chunk summary: 3 incidents"
    assert fake.seen[0][0]["content"] == "Summarize: <chunk>"


@pytest.mark.asyncio
async def test_recursion_fn_none_content_returns_empty_string():
    fn = _make_rlm_recursion_fn(_FakeLLM(None), asyncio.get_running_loop())
    assert await asyncio.to_thread(fn, "x") == ""


@pytest.mark.asyncio
async def test_recursion_fn_usable_inside_python_exec():
    """End-to-end: seed the recursion callable as `llm` in the REPL and have code call it on a
    slice of `ctx`, exactly as the RLM root model delegates a chunk to a fresh model."""
    fake = _FakeLLM("DELEGATED-OK")
    fn = _make_rlm_recursion_fn(fake, asyncio.get_running_loop())
    tool = PythonExecTool(namespace={"ctx": "abc " * 100, "llm": fn})
    r = await tool.run(code="print(llm('Summarize:\\n' + ctx[:12]))")
    assert r.success is True
    assert "DELEGATED-OK" in r.output
    assert "abc" in fake.seen[0][0]["content"]  # a real ctx slice was passed by reference
