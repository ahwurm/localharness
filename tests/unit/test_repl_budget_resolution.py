"""The REPL's /model refit resolves the budget through ContextConfig.resolve_budget.

`resolve_budget` is documented (#137) as the ONE pin-vs-scalar resolution `start` and `doctor`
both read; the refit path hand-read `model_context_overrides` instead. A second copy of that
lookup is exactly how doctor once blessed a 61,440 scalar for a session running on a 40,000 pin,
so this binds the refit to the shared call.

The UNPINNED half is deliberately not bound: with no pin the refit still takes the served window,
which can differ from the scalar in config.yaml that start_cmd calls the single source of truth.
That divergence is real, is the only defensible answer when the model just changed under the
session, and is disclosed in the note the user gets — the second test is its record, not a wish.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from localharness.agent import context as context_mod
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import ContextConfig

CONFIGURED = 40_000
SERVED = 131_072


def _stub(ctxcfg: ContextConfig, live_budget: int):
    """The minimum surface `_refresh_token_counter` touches, so the real method runs unmodified."""
    ctx = SimpleNamespace(max_context_tokens=live_budget, _token_counter=None)
    agent = SimpleNamespace(
        _ctx=ctx,
        _llm=SimpleNamespace(config=SimpleNamespace(base_url="http://localhost:8000/v1",
                                                    max_tokens=4096)),
        _config=SimpleNamespace(context=ctxcfg),
    )
    return SimpleNamespace(
        _agent=agent,
        _channel=SimpleNamespace(),
        _provider_type_for_base_url=lambda _url: "vllm",
    ), ctx


def test_pinned_refit_matches_resolve_budget(monkeypatch):
    """An override for the target model wins over the probe, with resolve_budget's exact answer."""
    ctxcfg = ContextConfig(max_context_tokens=CONFIGURED, model_context_overrides={"new-model": 32_768})
    stub, ctx = _stub(ctxcfg, CONFIGURED)
    monkeypatch.setattr(context_mod, "probe_served_window", lambda *a, **k: SERVED)

    asyncio.run(OrchestratorREPL._refresh_token_counter(stub, "new-model"))

    expected, pinned = ctxcfg.resolve_budget("new-model")
    assert pinned is True
    assert ctx.max_context_tokens == expected


def test_the_refit_goes_THROUGH_resolve_budget(monkeypatch):
    """The binding itself: whatever resolve_budget answers is what the session runs on.

    Reddens for any re-hand-rolled pin lookup — the shared call is stubbed here, so a second copy
    reading `model_context_overrides` directly would take the probe's answer instead.
    """
    monkeypatch.setattr(ContextConfig, "resolve_budget", lambda self, model: (12_345, True))
    stub, ctx = _stub(ContextConfig(max_context_tokens=CONFIGURED), CONFIGURED)
    monkeypatch.setattr(context_mod, "probe_served_window", lambda *a, **k: SERVED)

    asyncio.run(OrchestratorREPL._refresh_token_counter(stub, "new-model"))

    assert ctx.max_context_tokens == 12_345


def test_unpinned_refit_takes_the_served_window_and_says_so(monkeypatch):
    """No pin: the served window wins over the configured scalar, and the note discloses it."""
    ctxcfg = ContextConfig(max_context_tokens=CONFIGURED)
    stub, ctx = _stub(ctxcfg, CONFIGURED)
    monkeypatch.setattr(context_mod, "probe_served_window", lambda *a, **k: SERVED)

    note = asyncio.run(OrchestratorREPL._refresh_token_counter(stub, "new-model"))

    assert ctxcfg.resolve_budget("new-model") == (CONFIGURED, False)
    assert ctx.max_context_tokens == SERVED
    assert "served window" in note and f"{SERVED:,}" in note


def test_no_context_config_falls_back_to_the_probe(monkeypatch):
    """A session whose agent exposes no ContextConfig must still refit, never raise."""
    stub, ctx = _stub(None, CONFIGURED)
    stub._agent._config = SimpleNamespace()  # no `context` attribute at all
    monkeypatch.setattr(context_mod, "probe_served_window", lambda *a, **k: SERVED)

    asyncio.run(OrchestratorREPL._refresh_token_counter(stub, "new-model"))

    assert ctx.max_context_tokens == SERVED


@pytest.mark.parametrize("probe", [None, 0])
def test_unknowable_window_keeps_the_budget(monkeypatch, probe):
    """An undiscoverable window leaves the budget alone rather than guessing."""
    ctxcfg = ContextConfig(max_context_tokens=CONFIGURED)
    stub, ctx = _stub(ctxcfg, CONFIGURED)
    monkeypatch.setattr(context_mod, "probe_served_window", lambda *a, **k: probe)

    note = asyncio.run(OrchestratorREPL._refresh_token_counter(stub, "new-model"))

    assert ctx.max_context_tokens == CONFIGURED
    assert "couldn't read this model's served window" in note
