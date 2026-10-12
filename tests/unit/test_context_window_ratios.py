"""Tool-output caps follow the live context window.

The tool output cap, the eviction threshold and the registry's result cap are ratios of the
window the model has NOW (a server overflow shrinks it, a /model switch replaces it), with named
floors; an explicit integer is used as written. The tool registry's cap is a late-bound source its
children inherit, the web query budget reads it, and `start` and the bench feed all of it from
the context manager.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from localharness.agent import context as context_mod
from localharness.agent.context import (
    EVICT_RATIO_DEFAULT,
    RESULT_CAP_RATIO_DEFAULT,
    TOOL_EVICT_THRESHOLD_CHARS,
    TOOL_OUTPUT_RATIO_DEFAULT,
    CompactionPipeline,
    ContextManager,
    TokenCounter,
    ToolResultCapStage,
    context_size_kwargs,
)
from localharness.config.models import ContextConfig


# --------------------------------------------------------------------------- schema


def test_ratio_defaults_are_the_one_sixteenth_one_sixty_fourth_three_thirty_seconds():
    cfg = ContextConfig()
    assert cfg.tool_output_ratio == 0.0625
    assert cfg.evict_ratio == 0.015625
    assert cfg.result_cap_ratio == 0.09375


@pytest.mark.parametrize(
    "kwargs",
    [{"tool_output_ratio": 0}, {"evict_ratio": -0.01}, {"result_cap_ratio": 1.5}],
)
def test_a_zero_negative_or_over_one_ratio_is_refused(kwargs):
    with pytest.raises(ValidationError):
        ContextConfig(**kwargs)


def test_a_ratio_of_exactly_one_validates():
    assert ContextConfig(result_cap_ratio=1.0).result_cap_ratio == 1.0


def test_the_context_module_defaults_equal_the_config_defaults():
    fields = ContextConfig.model_fields
    assert TOOL_OUTPUT_RATIO_DEFAULT == fields["tool_output_ratio"].default
    assert EVICT_RATIO_DEFAULT == fields["evict_ratio"].default
    assert RESULT_CAP_RATIO_DEFAULT == fields["result_cap_ratio"].default


def test_context_size_kwargs_maps_config_to_the_five_manager_keywords():
    kw = context_size_kwargs(ContextConfig(max_tool_output_chars=20_000, evict_ratio=0.01))
    assert kw == {
        "tool_output_chars": 20_000,
        "tool_output_ratio": 0.0625,
        "tool_evict_threshold_chars": None,
        "evict_ratio": 0.01,
        "result_cap_ratio": 0.09375,
    }


# --------------------------------------------------------------------------- resolvers


def _unset(window: int, **kw) -> ContextManager:
    return ContextManager(max_context_tokens=window, tool_evict_threshold_chars=None, **kw)


def _caps(ctx: ContextManager) -> tuple[int, int, int]:
    return ctx.tool_output_cap_chars(), ctx.evict_threshold_chars(), ctx.result_cap_chars()


@pytest.mark.parametrize(
    ("window", "expected"),
    [
        (131_072, (32_768, 8_192, 49_152)),
        (65_536, (16_384, 4_096, 24_576)),
        (8_192, (2_048, 512, 3_072)),
        (1_000, (248, 500, 1_000)),  # the last two sit on their floors
    ],
)
def test_unset_sizes_are_the_ratio_of_the_window_with_floors(window, expected):
    assert _caps(_unset(window)) == expected


@pytest.mark.parametrize("window", [1_000, 8_192, 65_536, 131_072])
def test_an_explicit_output_size_is_used_at_every_window(window):
    ctx = ContextManager(max_context_tokens=window, tool_output_chars=20_000)
    assert ctx.tool_output_cap_chars() == 20_000


def test_an_explicit_integer_is_never_floored_or_scaled():
    assert ContextManager(max_context_tokens=131_072, tool_output_chars=100).tool_output_cap_chars() == 100
    small = ContextManager(max_context_tokens=1_000, tool_evict_threshold_chars=100)
    assert small.evict_threshold_chars() == 100


def test_the_caps_follow_an_overflow_shrink_and_a_refit():
    ctx = _unset(65_536)
    assert _caps(ctx) == (16_384, 4_096, 24_576)
    ctx.window_overflow(8_192, 9_000)
    assert _caps(ctx) == (2_048, 512, 3_072)
    ctx.max_context_tokens = 131_072  # what the /model refit assigns on the live manager
    assert _caps(ctx) == (32_768, 8_192, 49_152)


def test_a_manager_with_no_size_keywords_keeps_the_historical_eviction_threshold():
    ctx = ContextManager()
    assert ctx.evict_threshold_chars() == TOOL_EVICT_THRESHOLD_CHARS == 8_000
    assert ctx.tool_output_cap_chars() == 32_768
    assert ctx.result_cap_chars() == 49_152


# --------------------------------------------------------------------------- pipeline stage


def _tool_message(n: int) -> dict:
    return {"role": "tool", "tool_call_id": "t1", "content": "x" * n}


def _apply(stage: ToolResultCapStage, msgs: list) -> list:
    out, _ = stage.apply(msgs, None, TokenCounter())
    return out


def test_the_stage_resolves_a_callable_cap_once_per_apply():
    caps = iter([200, 100])
    stage = ToolResultCapStage(max_chars=lambda: next(caps))
    first = _apply(stage, [_tool_message(1_000)])
    second = _apply(stage, [_tool_message(1_000)])
    assert len(first[0]["content"]) <= 200 + 100  # head+tail keep plus the elision marker
    assert len(second[0]["content"]) < len(first[0]["content"])
    assert len(second[0]["content"]) <= 100 + 100


def test_an_integer_stage_cap_behaves_as_before():
    out = _apply(ToolResultCapStage(max_chars=100), [_tool_message(1_000)])
    assert "elided" in out[0]["content"]
    untouched = _apply(ToolResultCapStage(max_chars=2_000), [_tool_message(1_000)])
    assert untouched[0]["content"] == "x" * 1_000


def test_the_pipeline_hands_its_callable_cap_to_the_stage_unchanged():
    def fn() -> int:
        return 123

    pipe = CompactionPipeline(token_counter=TokenCounter(), tool_result_cap=fn)
    stage = pipe._deterministic_stages[0]
    assert isinstance(stage, ToolResultCapStage)
    assert stage.max_chars is fn


# --------------------------------------------------------------------------- build_messages


async def test_build_messages_evicts_at_the_live_threshold(monkeypatch):
    seen: list[int] = []
    real = context_mod._evict_large_tool_results

    def spy(messages, store, *, threshold_chars, **kw):
        seen.append(threshold_chars)
        return real(messages, store, threshold_chars=threshold_chars, **kw)

    monkeypatch.setattr(context_mod, "_evict_large_tool_results", spy)
    monkeypatch.setattr(context_mod, "TOOL_EVICT_USAGE_FRACTION", 0.0)
    ctx = ContextManager(
        max_context_tokens=65_536,
        eviction_store=context_mod.ContentStore(),
        tool_evict_threshold_chars=None,
    )
    msgs = [{"role": "user", "content": "hi"}]
    await ctx.build_messages(msgs)
    assert seen[-1] == ctx.evict_threshold_chars() == 4_096
    ctx.max_context_tokens = 8_192
    await ctx.build_messages(msgs)
    assert seen[-1] == ctx.evict_threshold_chars() == 512


# --------------------------------------------------------------------------- registry cap source

from typing import Any  # noqa: E402

from localharness.config.models import ToolConfig  # noqa: E402
from localharness.tools import ToolRegistry  # noqa: E402
from localharness.tools.base import Tool, ToolResult, ToolSchema  # noqa: E402
from localharness.tools.builtin import bind_agent_store_tools, register_builtin_tools  # noqa: E402
from localharness.tools.builtin.web_tool import WebPageQueryTool  # noqa: E402


class _Big(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="big",
            description="Returns 2,000 characters, or fails with 2,000 characters of error text.",
            ingest="none",
            host="safe",
            parameters={
                "type": "object",
                "properties": {"fail": {"type": "boolean"}},
                "required": [],
            },
        )

    async def _execute(self, fail: bool = False, **kw: Any) -> ToolResult:
        if fail:
            return ToolResult(output="", success=False, error="e" * 2_000, error_type="execution_error")
        return self.ok("x" * 2_000)


async def _dispatch(reg: ToolRegistry, **args: Any) -> ToolResult:
    return await reg.dispatch("big", args, "a1", "d1", ToolConfig(inherit=["global"]))


def test_a_registry_nobody_wired_caps_at_fifty_thousand():
    assert ToolRegistry().result_size_cap_chars == 50_000
    assert ToolRegistry(result_size_cap=500).result_size_cap_chars == 500


async def test_the_cap_is_read_per_dispatch_for_output_and_error():
    caps = iter([500, 1_000, 300])
    reg = ToolRegistry(result_size_cap=lambda: next(caps))
    await reg.register(_Big(), scope="global")
    assert len((await _dispatch(reg)).output) == 500
    assert len((await _dispatch(reg)).output) == 1_000
    failed = await _dispatch(reg, fail=True)
    assert failed.error.startswith("e" * 300) and "e" * 301 not in failed.error


async def test_set_result_size_cap_late_binds():
    reg = ToolRegistry()
    await reg.register(_Big(), scope="global")
    assert len((await _dispatch(reg)).output) == 2_000
    reg.set_result_size_cap(lambda: 700)
    assert len((await _dispatch(reg)).output) == 700


async def test_a_child_registry_follows_the_base_cap_source_not_a_snapshot():
    base = ToolRegistry()
    await base.register(_Big(), scope="global")
    live = [900]
    base.set_result_size_cap(lambda: live[0])
    child = ToolRegistry.from_allowed(["big"], base_registry=base)
    assert child.result_size_cap_chars == 900
    live[0] = 400
    assert child.result_size_cap_chars == 400
    assert len((await _dispatch(child)).output) == 400


async def test_a_child_of_an_unwired_base_keeps_fifty_thousand():
    base = ToolRegistry()
    await base.register(_Big(), scope="global")
    assert ToolRegistry.from_allowed(["big"], base_registry=base).result_size_cap_chars == 50_000


# --------------------------------------------------------------------------- web query budget


def _web_query(reg: ToolRegistry) -> WebPageQueryTool:
    return reg._tools["global"]["web_page_query"]


@pytest.mark.parametrize(("window", "cap", "budget"), [(131_072, 49_152, 44_236), (65_536, 24_576, 22_118)])
async def test_the_web_query_budget_is_nine_tenths_of_the_registry_cap(window, cap, budget):
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    ctx = ContextManager(max_context_tokens=window)
    reg.set_result_size_cap(ctx.result_cap_chars)
    assert reg.result_size_cap_chars == cap
    assert _web_query(reg)._output_budget() == budget
    ctx.max_context_tokens = 8_192  # the next call sees the shrunken window
    assert _web_query(reg)._output_budget() == int(3_072 * 0.9)


async def test_a_rebound_child_web_query_reports_the_same_budget_as_the_root():
    root = ToolRegistry()
    await register_builtin_tools(root)
    root.set_result_size_cap(ContextManager(max_context_tokens=65_536).result_cap_chars)
    child = ToolRegistry.from_allowed(["web_page_query"], base_registry=root)
    bind_agent_store_tools(child, context_mod.ContentStore())
    assert _web_query(child)._output_budget() == _web_query(root)._output_budget() == 22_118


def test_an_unwired_web_query_tool_keeps_forty_five_thousand():
    assert WebPageQueryTool()._output_budget() == 45_000
