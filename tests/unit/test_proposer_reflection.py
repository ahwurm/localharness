"""MODP-03 — reflective proposer: the rendered reflection prompt cites Pareto-front /
per-fixture evidence ALONGSIDE the failed-trace summaries.

These are DETERMINISTIC / OFFLINE pure-builder tests (Pitfall 6): they assert on the
output of ``_build_reflection_messages`` (a pure function) and ``_render_pareto_evidence``
directly — NO live model, NO network, NO ``proposer.base_url``, NO FakeLLMClient. The
end-to-end threading (propose(store=...)) is covered in tests/unit/test_proposer.py.
"""
import types

from localharness.autoresearch.proposer import (
    _build_reflection_messages,
    _render_pareto_evidence,
)


def _user(msgs) -> str:
    return next(m["content"] for m in msgs if m["role"] == "user")


def _front():
    """Two minimal fake front entries — the renderer reads only .component +
    .train_scores_per_fixture (no real ArchiveEntry needed)."""
    return [
        types.SimpleNamespace(
            component="agent.role_sections.tool_use",
            train_scores_per_fixture={"write_secret": 1.0},
        ),
        types.SimpleNamespace(
            component="agent.role",
            train_scores_per_fixture={"mcp_list_tools": 0.4},
        ),
    ]


# --------------------------------------------------------------------------- #
# Test A — the rendered prompt cites Pareto / per-fixture evidence
# --------------------------------------------------------------------------- #


def test_reflection_prompt_cites_pareto_per_fixture():
    """MODP-03: with pareto_evidence the user message names the Pareto front + each
    fixture's best rate + holder — the per-fixture evidence the model will see."""
    pareto = _render_pareto_evidence(_front())
    msgs = _build_reflection_messages(
        "agent.role_sections.tool_use", before="", type_name="str",
        traces=[], pareto_evidence=pareto,
    )
    user = _user(msgs)
    assert "Pareto front" in user
    assert "write_secret: best rate 1.00" in user   # per-fixture evidence present
    assert "agent.role_sections.tool_use" in user   # the holder component
    assert "mcp_list_tools: best rate 0.40" in user
    # AUGMENTS, does not replace — the failed-trace block is still there.
    assert "Failed TRAIN traces" in user


# --------------------------------------------------------------------------- #
# Test B — rationale contract references BOTH the failures AND the front
# --------------------------------------------------------------------------- #


def test_rationale_contract_cites_both_failures_and_front():
    """MODP-03: when evidence is present the contract sentence requires tying the change
    to BOTH the failures AND the per-fixture Pareto evidence (not instead of)."""
    pareto = _render_pareto_evidence(_front())
    user = _user(
        _build_reflection_messages("agent.role", before="x", type_name="str",
                                   traces=[], pareto_evidence=pareto)
    )
    assert "failures" in user            # the failed-trace contract is retained
    assert "Pareto" in user              # AND the per-fixture front is required
    assert "per-fixture" in user
    # the augmented contract names both in one breath
    assert "failures AND the per-fixture" in user


# --------------------------------------------------------------------------- #
# Test C — back-compat: empty evidence ⇒ NO Pareto block (byte-identical to today)
# --------------------------------------------------------------------------- #


def test_empty_evidence_is_unchanged_back_compat():
    """Back-compat: the default pareto_evidence="" reproduces today's user message — no
    Pareto block, the original 'tying the change to the failures' contract preserved."""
    user = _user(
        _build_reflection_messages("agent.role", before="x", type_name="str", traces=[])
    )
    assert "Pareto front" not in user
    assert "per-fixture" not in user
    # the original (pre-MODP-03) contract sentence, verbatim
    assert '"rationale": <one sentence tying the change to the failures>}.' in user
    assert "Failed TRAIN traces" in user


def test_empty_evidence_byte_identical_to_pre_modp03():
    """Byte-exact back-compat: the full user string with no evidence equals the exact
    pre-MODP-03 assembly for the same inputs (no reformat, no drift)."""
    import json
    component, before, type_name = "agent.role", "old role", "str"
    expected = (
        f"Component: {component} (type {type_name})\n"
        f"Current value:\n{json.dumps(before)}\n\n"
        f"Failed TRAIN traces (the change must address these failures):\n"
        f"- (failed train run)\n\n"
        'Respond with JSON only: {"after": <new value for this component>, '
        '"rationale": <one sentence tying the change to the failures>}.'
    )
    user = _user(
        _build_reflection_messages(component, before, type_name, traces=[])
    )
    assert user == expected


# --------------------------------------------------------------------------- #
# Test D — _render_pareto_evidence: empty ⇒ "", non-empty ⇒ one line/fixture, no holdout
# --------------------------------------------------------------------------- #


def test_render_pareto_evidence_empty_front():
    """Empty/absent front ⇒ "" (the back-compat sentinel)."""
    assert _render_pareto_evidence([]) == ""


def test_render_pareto_evidence_lines_and_no_holdout():
    """Non-empty front ⇒ one line per fixture (best rate + holder), sealed-slice-safe
    (reads only train_scores_per_fixture, NEVER holdout — Pitfall 5)."""
    ev = _render_pareto_evidence(_front())
    assert ev != ""
    lines = ev.splitlines()
    assert len(lines) == 2                                  # one line per fixture
    assert "- mcp_list_tools: best rate 0.40 (agent.role)" in lines
    assert "- write_secret: best rate 1.00 (agent.role_sections.tool_use)" in lines
    assert "holdout" not in ev                              # zero holdout reference


def test_render_pareto_evidence_keeps_best_rate_holder():
    """When two mutations score the same fixture, the higher rate's holder wins the line."""
    front = [
        types.SimpleNamespace(component="agent.role", train_scores_per_fixture={"fx": 0.3}),
        types.SimpleNamespace(
            component="agent.role_sections.tool_use", train_scores_per_fixture={"fx": 0.9}
        ),
    ]
    ev = _render_pareto_evidence(front)
    assert ev == "- fx: best rate 0.90 (agent.role_sections.tool_use)"
