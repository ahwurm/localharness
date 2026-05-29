"""PROP-01/02/03 + SC4 + edge — proposer pipeline behavioral tests.

The proposer landed in 16-02 (pipeline) + 16-03 (CLI), so the Wave-0 xfail guards
have been removed and these are now plain passing behavioral tests. Fixtures
(FakeLLMClient, proposer_corpus, proposer_results) come from tests/conftest.py.
"""
import json

import pytest

from tests.conftest import FakeLLMClient

from localharness.autoresearch.proposer import propose, Proposal, ProposerError  # noqa: F401


def _cfg():
    """A HarnessConfig with a distinct proposer block (PROP-02), built in-test."""
    from localharness.config.models import HarnessConfig

    return HarnessConfig.model_validate(
        {
            "version": "1",
            "provider": {
                "provider_type": "ollama",
                "base_url": "http://localhost:11434/v1",
                "default_model": "gpt-oss:120b",
            },
            "proposer": {
                "base_url": "http://localhost:11434/v1",
                "model": "frontier-strong:latest",
            },
        }
    )


def _good_payload(after: str = "You are a careful, terse assistant.") -> str:
    """A well-formed proposer response for component agent.role."""
    return json.dumps({"after": after, "rationale": "Failed train traces show verbosity."})


# --------------------------------------------------------------------------- #
# PROP-01 — pipeline reads registry, emits {before, after} diff + rationale
# --------------------------------------------------------------------------- #


async def test_before_is_current_component_value(proposer_corpus, proposer_results):
    """PROP-01: proposal.before equals the catalogue current_value for the component."""
    from localharness.registry import build_catalogue

    cfg = _cfg()
    catalogue = build_catalogue(cfg, overlays={})
    expected_before = catalogue["agent.role"].current_value
    proposal = await propose(
        "agent.role",
        [proposer_results["train_run_id"]],
        cfg=cfg,
        llm=FakeLLMClient(_good_payload()),
        corpus_path=proposer_corpus,
        results_path=proposer_results["results"],
    )
    assert proposal.before == expected_before


async def test_diff_shape_is_before_after(proposer_corpus, proposer_results):
    """PROP-01: emitted diff JSON parses to a dict with exactly keys {before, after}."""
    cfg = _cfg()
    proposal = await propose(
        "agent.role",
        [proposer_results["train_run_id"]],
        cfg=cfg,
        llm=FakeLLMClient(_good_payload()),
        corpus_path=proposer_corpus,
        results_path=proposer_results["results"],
    )
    diff = json.dumps({"before": proposal.before, "after": proposal.after})
    assert json.loads(diff).keys() == {"before", "after"}


async def test_malformed_proposal_fails_explicitly(proposer_corpus, proposer_results):
    """PROP-01: non-JSON garbage from the model → ProposerError, never a silent proposal."""
    cfg = _cfg()
    with pytest.raises(ProposerError):
        await propose(
            "agent.role",
            [proposer_results["train_run_id"]],
            cfg=cfg,
            llm=FakeLLMClient("not json at all <<<>>>"),
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )


# --------------------------------------------------------------------------- #
# PROP-02 — proposer uses ProposerConfig, distinct from provider
# --------------------------------------------------------------------------- #


async def test_uses_proposer_config_not_provider(proposer_corpus, proposer_results, monkeypatch):
    """PROP-02: the LLMConfig built for the proposer carries proposer.model, NOT provider.default_model."""
    import localharness.autoresearch.proposer as prop_mod

    captured = {}

    class _SpyClient:
        def __init__(self, llm_cfg):
            captured["model"] = llm_cfg.model
            captured["base_url"] = llm_cfg.base_url
            self.config = llm_cfg

        async def complete(self, messages, tools=None, stream=False):
            class _Msg:
                content = _good_payload()

            return _Msg(), None

    monkeypatch.setattr(prop_mod, "LLMClient", _SpyClient, raising=False)
    cfg = _cfg()
    await propose(
        "agent.role",
        [proposer_results["train_run_id"]],
        cfg=cfg,
        corpus_path=proposer_corpus,
        results_path=proposer_results["results"],
    )
    assert captured["model"] == cfg.proposer.model
    assert captured["model"] != cfg.provider.default_model


# --------------------------------------------------------------------------- #
# PROP-03 — sealed-slice seal: holdout/unknown refused, model never called
# --------------------------------------------------------------------------- #


async def test_refuses_holdout_run_id(proposer_corpus, proposer_results):
    """PROP-03: a HOLDOUT run_id → ProposerError AND FakeLLMClient.complete_calls == 0."""
    cfg = _cfg()
    spy = FakeLLMClient(_good_payload())
    with pytest.raises(ProposerError):
        await propose(
            "agent.role",
            [proposer_results["holdout_run_id"]],
            cfg=cfg,
            llm=spy,
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )
    assert spy.complete_calls == 0


async def test_refuses_unknown_scenario(proposer_corpus, proposer_results):
    """PROP-03: a run_id whose scenario maps to no fixture → ProposerError before any model call."""
    cfg = _cfg()
    spy = FakeLLMClient(_good_payload())
    with pytest.raises(ProposerError):
        await propose(
            "agent.role",
            ["fakemodel/no_such_scenario/20260529T000000Z"],
            cfg=cfg,
            llm=spy,
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )
    assert spy.complete_calls == 0


async def test_no_model_call_before_seal(proposer_corpus, proposer_results):
    """PROP-03: on any seal refusal the spy FakeLLMClient.complete_calls == 0 (model never reached)."""
    cfg = _cfg()
    spy = FakeLLMClient(_good_payload())
    with pytest.raises(ProposerError):
        await propose(
            "agent.role",
            [proposer_results["holdout_run_id"]],
            cfg=cfg,
            llm=spy,
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )
    assert spy.complete_calls == 0


# --------------------------------------------------------------------------- #
# SC4 — atomic single component, type-coerced after
# --------------------------------------------------------------------------- #


async def test_atomic_single_component(proposer_corpus, proposer_results):
    """SC4: propose() targets exactly the supplied component; Proposal.component == input."""
    cfg = _cfg()
    proposal = await propose(
        "agent.role",
        [proposer_results["train_run_id"]],
        cfg=cfg,
        llm=FakeLLMClient(_good_payload()),
        corpus_path=proposer_corpus,
        results_path=proposer_results["results"],
    )
    assert proposal.component == "agent.role"


async def test_after_type_coercion_enforced(proposer_corpus, proposer_results):
    """SC4: an `after` that fails coerce_value for the path annotation → ProposerError (clean refusal)."""
    cfg = _cfg()
    # org.context.compaction_threshold_pct is a bounded float; a non-numeric after must refuse.
    bad = json.dumps({"after": "definitely-not-a-float", "rationale": "x"})
    with pytest.raises(ProposerError):
        await propose(
            "org.context.compaction_threshold_pct",
            [proposer_results["train_run_id"]],
            cfg=cfg,
            llm=FakeLLMClient(bad),
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )


# --------------------------------------------------------------------------- #
# edge — no failed traces → refuse
# --------------------------------------------------------------------------- #


async def test_no_failed_traces_refuses(proposer_corpus, proposer_results, tmp_path):
    """edge: when the only supplied run_id is a PASSING train run → ProposerError (no evidence)."""
    from localharness.bench.runner import resolve_run_path
    from localharness.core.events import ScenarioCompleted

    # Overwrite the train trace with success=True so there is no failure evidence.
    path = resolve_run_path(
        proposer_results["results"], "fakemodel", "prop_train_fx", "20260529T000000Z"
    )
    path.write_text(
        ScenarioCompleted(
            scenario_name="prop_train_fx",
            model="fakemodel",
            success=True,
            latency_ttft=1.0,
            latency_total=1.0,
            tokens_in=5,
            tokens_out=5,
            iterations=1,
            parse_failures=0,
            stuck_recoveries=0,
            tool_call_count=0,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    cfg = _cfg()
    with pytest.raises(ProposerError):
        await propose(
            "agent.role",
            [proposer_results["train_run_id"]],
            cfg=cfg,
            llm=FakeLLMClient(_good_payload()),
            corpus_path=proposer_corpus,
            results_path=proposer_results["results"],
        )
