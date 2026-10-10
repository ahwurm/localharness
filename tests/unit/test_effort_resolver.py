"""EFFORT-03: the ladder session > agent > endpoint > org > default, spelled once."""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import localharness.config.effort as effort_mod
from localharness.config.effort import LEVELS, endpoint_effort, resolve_effort
from localharness.config.models import ProviderConfig
from tests.unit.effort_support import PRIMARY_URL, RecordingLLM, harness_with

PEER_URL = "http://127.0.0.1:9000/v1"


# --- resolve_effort: one rung at a time --------------------------------------------------------

def test_a_session_choice_wins_over_everything() -> None:
    h = harness_with(server_effort="max", org_default="low")
    assert resolve_effort("off", "low", h, PRIMARY_URL) == ("off", "session")


def test_an_agent_file_level_beats_the_endpoint_and_the_org() -> None:
    h = harness_with(server_effort="max", org_default="off")
    assert resolve_effort(None, "low", h, PRIMARY_URL) == ("low", "agent")


def test_inherit_and_none_both_fall_through_the_agent_rung() -> None:
    h = harness_with(server_effort="max")
    assert resolve_effort(None, "inherit", h, PRIMARY_URL) == ("max", "endpoint")
    assert resolve_effort(None, None, h, PRIMARY_URL) == ("max", "endpoint")


def test_the_org_default_is_the_rung_below_the_endpoint() -> None:
    h = harness_with(org_default="low")
    assert resolve_effort(None, None, h, PRIMARY_URL) == ("low", "org")


def test_the_endpoint_beats_the_org_default() -> None:
    h = harness_with(server_effort="off", org_default="low")
    assert resolve_effort(None, "inherit", h, PRIMARY_URL) == ("off", "endpoint")


def test_nothing_set_resolves_to_high_by_default() -> None:
    assert resolve_effort(None, "inherit", harness_with(), PRIMARY_URL) == ("high", "default")


def test_empty_a_harness_with_no_server_no_active_no_peers_no_org_default() -> None:
    h = harness_with(hand_run=True)
    assert h.server is None and h.active_endpoint is None and h.extra_endpoints == []
    assert resolve_effort(None, "inherit", h, PRIMARY_URL) == ("high", "default")


def test_the_ladder_end_to_end_agent_file_then_a_switch_to_the_peer() -> None:
    """An agent file saying `low` under a harness whose peer says `max` is ('low', 'agent'); a
    /model to that peer with no session choice and an inheriting agent is ('max', 'endpoint')."""
    h = harness_with(peer=PEER_URL, peer_effort="max", active=PEER_URL)
    assert resolve_effort(None, "low", h, PEER_URL) == ("low", "agent")
    assert resolve_effort(None, "inherit", h, PEER_URL) == ("max", "endpoint")


# --- endpoint_effort ---------------------------------------------------------------------------

def test_the_primarys_effort_is_not_the_answer_for_another_base_url() -> None:
    assert endpoint_effort(harness_with(server_effort="max"), PEER_URL) is None


def test_a_peer_answers_with_its_lifecycle_effort() -> None:
    assert endpoint_effort(harness_with(peer=PEER_URL, peer_effort="max"), PEER_URL) == "max"


def test_the_active_selection_effort_wins_over_the_peer_record() -> None:
    h = harness_with(peer=PEER_URL, peer_effort="max", active=PEER_URL, active_effort="low")
    assert endpoint_effort(h, PEER_URL) == "low"


def test_an_active_selection_without_effort_falls_back_to_the_peer_record() -> None:
    h = harness_with(peer=PEER_URL, peer_effort="max", active=PEER_URL, active_effort=None)
    assert endpoint_effort(h, PEER_URL) == "max"


def test_a_hand_run_primary_has_no_endpoint_effort() -> None:
    assert endpoint_effort(harness_with(hand_run=True, org_default="low"), PRIMARY_URL) is None


def test_a_peer_effort_never_reaches_the_primary() -> None:
    h = harness_with(peer=PEER_URL, peer_effort="max", active=PEER_URL, active_effort="max")
    assert endpoint_effort(h, PRIMARY_URL) is None
    assert resolve_effort(None, "inherit", h, PRIMARY_URL) == ("high", "default")


def test_the_active_selection_is_never_read_for_the_primarys_url() -> None:
    """Even when the persisted selection carries the primary's own URL (hand-edited, or a stale
    switch), a hand-run primary asks no one."""
    h = harness_with(hand_run=True, active=PRIMARY_URL, active_effort="max")
    assert endpoint_effort(h, PRIMARY_URL) is None


def test_a_trailing_slash_matches_the_record_without_it_and_vice_versa() -> None:
    h = harness_with(server_effort="high")
    assert endpoint_effort(h, PRIMARY_URL + "/") == "high"
    assert endpoint_effort(h, PRIMARY_URL.rstrip("/")) == "high"
    p = harness_with(peer=PEER_URL + "/", peer_effort="low")
    assert endpoint_effort(p, PEER_URL) == "low"
    assert endpoint_effort(p, PEER_URL + "/") == "low"


def test_a_peer_with_no_lifecycle_has_no_effort() -> None:
    h = harness_with(peer=PEER_URL)
    h.extra_endpoints[0].lifecycle = None
    assert endpoint_effort(h, PEER_URL) is None


# --- purity ------------------------------------------------------------------------------------

def test_resolve_effort_answers_for_plain_objects() -> None:
    h = SimpleNamespace(
        provider=SimpleNamespace(base_url=PRIMARY_URL),
        server=SimpleNamespace(effort="low"),
        active_endpoint=None, extra_endpoints=[],
        org=SimpleNamespace(default_effort=None),
    )
    assert resolve_effort(None, None, h, PRIMARY_URL) == ("low", "endpoint")
    assert resolve_effort(None, None, SimpleNamespace(), PRIMARY_URL) == ("high", "default")


def test_effort_module_imports_only_typing_and_config_models() -> None:
    tree = ast.parse(Path(effort_mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported <= {"__future__", "typing", "localharness.config.models"}, imported


def test_levels_are_reexported() -> None:
    assert LEVELS == ("off", "low", "high", "max")


# --- the Wave 0 helpers ------------------------------------------------------------------------

def test_harness_with_defaults_validates_on_the_primary_url() -> None:
    h = harness_with()
    assert h.provider.base_url == PRIMARY_URL
    assert isinstance(h.provider, ProviderConfig)


def test_recording_llm_records_the_stream_complete_keywords() -> None:
    llm = RecordingLLM()
    asyncio.run(llm.stream_complete(messages=[{"role": "user", "content": "x"}], tools=None,
                                    on_token=None, reasoning_budget=3))
    assert llm.calls[-1] == {"reasoning_budget": 3}


def test_recording_llm_budget_key_follows_the_runtime() -> None:
    assert RecordingLLM(provider_type="llamacpp").effort_budget_key == "reasoning_budget_tokens"
    assert RecordingLLM(provider_type="vllm").effort_budget_key is None


def test_recording_llm_key_sets_are_settable_frozensets() -> None:
    llm = RecordingLLM()
    assert llm.rejected_effort_keys == frozenset() and llm.accepted_effort_keys == frozenset()
    assert RecordingLLM(accepted=("reasoning_budget_tokens",)).accepted_effort_keys == frozenset(
        {"reasoning_budget_tokens"})
    llm.rejected_effort_keys = frozenset({"chat_template_kwargs"})
    assert llm.rejected_effort_keys == frozenset({"chat_template_kwargs"})
