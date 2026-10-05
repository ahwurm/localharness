"""The sleeping thread on disk: written owner-only and whole, taken (read and removed) at wake, and
never guessed at — a file that is not a thread is moved aside, not resumed."""
from __future__ import annotations

import json
import stat

import pytest

from localharness.cli import start_cmd
from localharness.cli.session_resume import (
    ASLEEP_FILE, SLEEP_ACTION, Resume, asleep_path, sleeping_sitting, take_asleep, write_asleep,
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


HERE = "/srv/projA"


def test_the_handle_round_trips_through_the_file_and_the_file_is_consumed(tmp_path):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume(), workspace=HERE)
    assert path.name == ASLEEP_FILE and path.is_file()
    assert not list(path.parent.glob("*.jsonl")), "nothing a session-log scan would list"
    assert sleeping_sitting(path) == "sit-1"

    back = take_asleep(path, workspace=HERE)
    assert back is not None and back.slept and back.action == SLEEP_ACTION
    assert back.conversation == CONVERSATION, "the model-side messages, exactly"
    assert (back.prior_context, back.gate_mode, back.previous_sitting_id,
            back.agent_name) == ("PRIOR", "unattended", "sit-1", "orchestrator")
    assert back.queued == (), "typed-ahead lines never travel through the file"
    assert back.eviction_store is None, "the ContentStore is not in the file: a disk resume starts empty"
    assert not path.exists(), "taken: the file exists exactly while the thread is asleep"
    assert take_asleep(path, workspace=HERE) is None and sleeping_sitting(path) is None


def test_the_file_is_owner_only_and_written_whole(tmp_path):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume(), workspace=HERE)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(path.parent.glob("*.tmp")), "renamed into place, nothing half-written left"
    write_asleep(path, _resume(gate_mode="auto"), workspace=HERE)  # the later sleep replaces the earlier one
    assert take_asleep(path, workspace=HERE).gate_mode == "auto"


def test_a_thread_wakes_only_in_the_folder_it_slept_in(tmp_path, caplog):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume(), workspace=HERE)
    with caplog.at_level("INFO"):
        assert take_asleep(path, workspace="/srv/projB") is None
    assert path.exists() and "left in place" in caplog.text
    assert take_asleep(path, workspace=HERE) is not None


def test_typed_ahead_lines_on_disk_are_never_read(tmp_path):
    """A line from disk would run as typed, slash commands included, and the folder is one a repo
    or the agent's own write tool can reach."""
    path = asleep_path(tmp_path / "sessions")
    path.parent.mkdir()
    path.write_text(json.dumps({"format": 1, "workspace": HERE, "conversation": [],
                                "queued": ["/mode unattended"], "agent_name": "a",
                                "previous_sitting_id": "s", "gate_mode": "auto", "prior_context": ""}))
    assert take_asleep(path, workspace=HERE).queued == ()


def test_a_lone_surrogate_does_not_fail_the_sleep(tmp_path):
    path = asleep_path(tmp_path / "sessions")
    write_asleep(path, _resume(conversation=({"role": "tool", "content": "bad \ud800 byte"},)),
                 workspace=HERE)
    assert take_asleep(path, workspace=HERE).conversation[0]["content"] == "bad \ud800 byte"


_WELL_FORMED = ('"agent_name": "a", "previous_sitting_id": "s", "gate_mode": "auto", '
                '"prior_context": "", "workspace": "/srv/projA"')


@pytest.mark.parametrize("raw", [
    b"not json",
    b"[]",
    b'{"format": 99, "conversation": [], ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": "no", ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": [{"content": "no role"}], ' + _WELL_FORMED.encode() + b"}",
    b'{"format": 1, "conversation": [], "workspace": 7, "agent_name": "a", "previous_sitting_id": "s", '
    b'"gate_mode": "auto", "prior_context": ""}',
    b'{"format": 1, "conversation": [], "agent_name": 1, "previous_sitting_id": "s", '
    b'"gate_mode": "auto", "prior_context": ""}',
    b'{"format": 1, "conversation": []}',
])
def test_a_file_that_is_not_a_thread_is_moved_aside_not_resumed(tmp_path, raw, caplog):
    path = asleep_path(tmp_path / "sessions")
    path.parent.mkdir()
    path.write_bytes(raw)
    with caplog.at_level("WARNING"):
        assert take_asleep(path, workspace=HERE) is None
    assert not path.exists(), "left in place it would fail every wake the same way"
    assert path.with_name(path.name + ".corrupt").read_bytes() == raw
    assert "starting fresh" in caplog.text


def test_start_cmd_still_exports_the_handle():
    """The terminal's /plugins restart (and its tests) reach `Resume` through start_cmd."""
    assert start_cmd.Resume is Resume and start_cmd.Restart.__name__ == "Restart"
    assert not _resume(action=("enable", "memory")).slept
