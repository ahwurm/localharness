"""EFFORT-09 / EFFORT-11 data shapes: the ledger's rejected-effort-key pair (03-CONTEXT D-20) and
`reasoning_cut` among the phone's known finish reasons (D-22)."""
from __future__ import annotations

import json
import os

import pytest

from localharness.channels.mobile.protocol import FINISH_REASONS_KNOWN
from localharness.provider import speed_stats as ss

KEYS = ["reasoning_budget_tokens", "chat_template_kwargs", "reasoning_budget_tokens"]


def test_record_then_last_round_trips_sorted_and_deduplicated(tmp_path):
    p = tmp_path / "s.json"
    ss.record_rejected_effort_keys(p, "llamacpp", "m", KEYS)
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") == [
        "chat_template_kwargs", "reasoning_budget_tokens"]


def test_a_fresh_ledger_has_no_recorded_list(tmp_path):
    p = tmp_path / "s.json"
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") is None
    ss.record_tps(p, "llamacpp", "other", 50.0)
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") is None


def test_an_empty_list_is_stored_and_read_back_as_empty(tmp_path):
    p = tmp_path / "s.json"
    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["reasoning_budget_tokens"])
    ss.record_rejected_effort_keys(p, "llamacpp", "m", [])
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") == []


def test_a_hand_edited_non_list_reads_as_none(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"llamacpp:m": {"rejected_effort_keys": "chat_template_kwargs"}}))
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") is None


def test_recording_keys_leaves_the_other_fields_of_the_entry_alone(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "m", 55.0)
    ss.record_tokens_per_chunk(p, "llamacpp", "m", 2.5)
    ss.record_window_report(p, "llamacpp", "m", 65536, 262144, 60000)
    ss.record_dropped_env(p, "llamacpp", ["LLAMA_ARG_X"])
    before = json.loads(p.read_text())

    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["chat_template_kwargs"])
    after = json.loads(p.read_text())

    assert list(after) == list(before)  # one entry for (ptype, model), the reserved map kept
    entry = dict(after["llamacpp:m"])
    assert entry.pop("rejected_effort_keys") == ["chat_template_kwargs"]
    entry.pop("updated_at")
    old = dict(before["llamacpp:m"])
    old.pop("updated_at")
    assert entry == old  # samples, window_report, tokens_per_chunk byte-identical
    assert after[ss.DROPPED_ENV_KEY] == before[ss.DROPPED_ENV_KEY]
    assert ss.all_median_tps(p) == {"llamacpp:m": 55.0}  # DROPPED_ENV_KEY still ignored


def test_a_later_sample_does_not_disturb_the_recorded_keys(tmp_path):
    p = tmp_path / "s.json"
    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["reasoning_budget_tokens"])
    ss.record_tps(p, "llamacpp", "m", 40.0)
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") == ["reasoning_budget_tokens"]


def test_idempotency_the_same_keys_twice_leaves_one_key_and_one_write_per_call(tmp_path, monkeypatch):
    p = tmp_path / "s.json"
    writes: list[int] = []
    real_write = ss._write

    def counting_write(path, data):
        writes.append(1)
        real_write(path, data)

    monkeypatch.setattr(ss, "_write", counting_write)

    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["reasoning_budget_tokens"])
    first = json.loads(p.read_text())["llamacpp:m"]
    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["reasoning_budget_tokens"])
    second = json.loads(p.read_text())["llamacpp:m"]

    assert list(json.loads(p.read_text())) == [ss.speed_key("llamacpp", "m")]
    assert first["rejected_effort_keys"] == second["rejected_effort_keys"] == ["reasoning_budget_tokens"]
    assert second["updated_at"] >= first["updated_at"]
    assert len(writes) == 2


def test_a_missing_parent_directory_is_created(tmp_path):
    p = tmp_path / "state" / "deeper" / "s.json"
    ss.record_rejected_effort_keys(p, "llamacpp", "m", ["chat_template_kwargs"])
    assert ss.last_rejected_effort_keys(p, "llamacpp", "m") == ["chat_template_kwargs"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unwritable_parent_raises_oserror_for_the_caller_to_wrap(tmp_path):
    d = tmp_path / "ro"
    d.mkdir()
    d.chmod(0o500)
    try:
        with pytest.raises(OSError):
            ss.record_rejected_effort_keys(d / "s.json", "llamacpp", "m", ["chat_template_kwargs"])
    finally:
        d.chmod(0o700)


def test_reasoning_cut_is_a_known_finish_reason():
    assert "reasoning_cut" in FINISH_REASONS_KNOWN
    assert "degenerate" in FINISH_REASONS_KNOWN
    assert FINISH_REASONS_KNOWN == (
        "stop", "length", "tool_calls", "content_filter", "function_call", "degenerate",
        "reasoning_cut",
    )
