"""Thinking-scope: INTERNAL harness LLM calls (mining/consolidation via LLMTextAdapter,
compaction summarizer) must send per-request chat_template_kwargs.enable_thinking=false,
while subject/user-facing turns keep thinking ON (#11 ruling — explicitly NOT an
is_local blanket; that blanket is what #11 removed).

C0 sweep control vs run-17 baseline: under --reasoning-parser qwen3 the internal calls'
bounded completion budgets were spent on hidden chain-of-thought (A1 recall 0.909->0.727,
semantic atoms 51->22, residue rescues 19->0, ARI 0.759->0.221) — pure yield starvation
on internal calls; turn-level crashes were already fixed separately."""
from __future__ import annotations

from types import SimpleNamespace

from localharness.agent.context import make_compaction_summarize_fn
from localharness.memory.idle_llm import LLMTextAdapter
from localharness.provider.client import LLMClient, LLMConfig

_DISABLE = {"chat_template_kwargs": {"enable_thinking": False}}


def _cfg(**kw) -> LLMConfig:
    kw.setdefault("base_url", "http://127.0.0.1:9")
    kw.setdefault("model", "m")
    kw.setdefault("timeout_seconds", 300.0)
    kw.setdefault("tool_call_mode", "native")
    kw.setdefault("is_local", False)  # skip the inference gate; kwarg plumbing is the subject
    return LLMConfig(**kw)


class _Recorder:
    """Fake completions endpoint recording the exact outgoing request kwargs."""

    def __init__(self):
        self.kwargs: list[dict] = []

    async def create(self, **kwargs):
        self.kwargs.append(kwargs)
        msg = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _wire(client: LLMClient) -> _Recorder:
    rec = _Recorder()
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=rec.create))
    )
    return rec


async def test_complete_disable_thinking_sets_chat_template_kwargs():
    """The per-call opt-in on LLMClient.complete lands in the outgoing request body."""
    c = LLMClient(_cfg())
    rec = _wire(c)
    await c.complete([{"role": "user", "content": "hi"}], disable_thinking=True)
    assert rec.kwargs[0].get("extra_body") == _DISABLE


async def test_subject_turn_keeps_thinking():
    """#11 guard: a normal call WITHOUT the opt-in carries no thinking suppression."""
    c = LLMClient(_cfg())
    rec = _wire(c)
    await c.complete([{"role": "user", "content": "hi"}])
    assert "extra_body" not in rec.kwargs[0]


async def test_llm_text_adapter_disables_thinking():
    """(i) The idle bridge — EVERY Phase-36 model-look (mining/chapter-writer/
    reconciliation) routes through LLMTextAdapter — opts in: bounded idle completion
    budgets must not fund hidden chain-of-thought."""
    c = LLMClient(_cfg())
    rec = _wire(c)
    out = await LLMTextAdapter(c).complete("mine this transcript")
    assert out == "ok"
    assert rec.kwargs[0].get("extra_body") == _DISABLE


async def test_compaction_summarizer_disables_thinking():
    """(iii) The compaction summarizer builds its own request — it opts in too."""
    c = LLMClient(_cfg())
    rec = _wire(c)
    out = await make_compaction_summarize_fn(c)([{"role": "user", "content": "hi"}])
    assert out == "ok"
    assert rec.kwargs[0].get("extra_body") == _DISABLE
