"""The sleeping thread on disk: written owner-only and whole, taken (read and removed) at wake, and
never guessed at — a file that is not a thread is moved aside, not resumed."""
from __future__ import annotations

import stat

import pytest

from localharness.cli import start_cmd
from localharness.cli.session_resume import (
    ASLEEP_FILE, SLEEP_ACTION, Resume, asleep_path, take_asleep, write_asleep,
)

CONVERSATION = (
    {"role": "user", "content": "find the blue key"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "web_search", "arguments": "{\"q\": \"blue key\"}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "[evicted: tool_result_get ev-1]"},
    {"role": "assistant", "content": "it is under the mat"},
)


def _resume(**kw) -> Resume:
    fields = dict(action=SLEEP_ACTION, agent_name="orchestrator", conversation=CONVERSATION,
                  prior_context="PRIOR", eviction_store=object(), queued=("typed ahead",),
                  gate_mode="unattended", previous_sitting_id="sit-1")
    return Resume(**(fields | kw))


def test_the_handle_round_trips_through_the_file_and_the_file_is_consumed(tmp_path):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume())
    assert path.name == ASLEEP_FILE and path.is_file()
    assert not list(path.parent.glob("*.jsonl")), "nothing a session-log scan would list"

    back = take_asleep(path)
    assert back is not None and back.slept and back.action == SLEEP_ACTION
    assert back.conversation == CONVERSATION, "the model-side messages, exactly"
    assert (back.prior_context, back.queued, back.gate_mode, back.previous_sitting_id,
            back.agent_name) == ("PRIOR", ("typed ahead",), "unattended", "sit-1", "orchestrator")
    assert back.eviction_store is None, "the ContentStore is not in the file: a disk resume starts empty"
    assert not path.exists(), "taken: the file exists exactly while the thread is asleep"
    assert take_asleep(path) is None


def test_the_file_is_owner_only_and_written_whole(tmp_path):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob("*.tmp")), "renamed into place, nothing half-written left"
    write_asleep(path, _resume(gate_mode="auto"))  # the later sleep replaces the earlier one
    assert take_asleep(path).gate_mode == "auto"


_WELL_FORMED = ('"agent_name": "a", "previous_sitting_id": "s", "gate_mode": "auto", '
                '"prior_context": ""')


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[]",
    b'{"format": 99, "conversation": [], ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": "no", ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": [{"content": "no role"}], ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": [], "queued": [1], ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": [], "agent_name": 1, "previous_sitting_id": "s", '
    b'"gate_mode": "auto", "prior_context": ""}',
    b'{"format": 1, "conversation": []}',
])
def test_a_file_that_is_not_a_thread_is_moved_aside_not_resumed(tmp_path, raw, caplog):
    path = asleep_path(tmp_path / "sessions")
    path.parent.mkdir()
    path.write_bytes(raw)
    with caplog.at_level("WARNING"):
        assert take_asleep(path) is None
    assert not path.exists(), "left in place it would fail every wake the same way"
    assert path.with_name(path.name + ".corrupt").read_bytes() == raw
    assert "starting fresh" in caplog.text


def test_start_cmd_still_exports_the_handle():
    """The terminal's /plugins restart (and its tests) reach `Resume` through start_cmd."""
    assert start_cmd.Resume is Resume and start_cmd.Restart.__name__ == "Restart"
    assert not _resume(action=("enable", "memory")).slept
