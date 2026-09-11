"""The ask-rate report over a hand-written fixture corpus (PRD §3.6).

The fixture is built here, event by event, on purpose: the report must be provable without the
owner's real session files, and a test that depends on `~/.localharness/agents/*/sessions` is a
test that passes or fails for reasons that have nothing to do with this code.
"""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from localharness.bench.askrate import (
    FIRST_N_SESSIONS_DEFAULT,
    build_report,
    bucket_of,
    has_permission_events,
    load_sessions,
    render,
    tool_meta_for,
)
from localharness.cli.app import app


def _action(seq: int, stamp: str, tool: str, params: dict) -> dict:
    return {
        "seq": seq,
        "timestamp": stamp,
        "event_type": "Action",
        "action_type": "tool_call",
        "tool_name": tool,
        "tool_params": params,
    }


def _write_session(root: Path, name: str, events: list[dict]) -> Path:
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return path


def _replay_corpus(root: Path, workspace: Path) -> None:
    """Three sessions, chronological by first timestamp, written out of order on disk."""
    _write_session(root, "b-second", [
        # `ls` is read-only -> no prompt; `pytest` was already asked about in a-first.
        _action(0, "2026-09-02T10:00:00Z", "bash_exec", {"command": "ls -la"}),
        _action(1, "2026-09-02T10:00:01Z", "bash_exec", {"command": "pytest tests/"}),
        _action(2, "2026-09-02T10:00:02Z", "read", {"path": str(workspace / "README.md")}),
    ])
    _write_session(root, "a-first", [
        _action(0, "2026-09-01T09:00:00Z", "bash_exec", {"command": "pytest tests/"}),
        _action(1, "2026-09-01T09:00:01Z", "bash_exec", {"command": "rm -rf build"}),
        _action(2, "2026-09-01T09:00:02Z", "write", {"path": str(workspace / "src" / "x.py")}),
    ])
    _write_session(root, "c-third", [
        _action(0, "2026-09-03T11:00:00Z", "web_fetch", {"url": "https://example.com/a"}),
    ])


def test_replay_counts_prompts_and_remembers_each_key(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / ".git").mkdir(parents=True)
    traces = tmp_path / "traces"
    traces.mkdir()
    _replay_corpus(traces, workspace)

    report = build_report(traces, workspace=workspace, first_n=2)

    assert report.source == "replay"
    assert [s.session_id for s in report.sessions] == ["a-first", "b-second", "c-third"]
    assert report.total_calls == 7
    # a-first: `pytest` (shell-unfamiliar, first time) + `rm -rf` (ungrantable) = 2 prompts.
    # In-workspace write is silent (review surface), `ls` is read-only, web_fetch is silent.
    assert [s.prompts for s in report.sessions] == [2, 0, 0]
    assert report.first_ask_keys == ("pytest",)
    assert report.destructive == (("shell-destructive: rm -rf", 1),)


def test_replay_asks_once_per_key_across_sessions(tmp_path: Path) -> None:
    """The same unfamiliar command in a later session costs nothing (PRD §3.3 ask-once-per-key)."""
    workspace = tmp_path / "project"
    (workspace / ".git").mkdir(parents=True)
    traces = tmp_path / "traces"
    traces.mkdir()
    _replay_corpus(traces, workspace)

    report = build_report(traces, workspace=workspace, first_n=2)
    second = next(s for s in report.sessions if s.session_id == "b-second")
    assert second.prompts == 0


def test_replay_flags_edits_outside_the_boundary(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / ".git").mkdir(parents=True)
    outside = tmp_path / "elsewhere" / "notes.md"
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_session(traces, "s", [_action(0, "2026-09-01T09:00:00Z", "write", {"path": str(outside)})])

    report = build_report(traces, workspace=workspace)
    assert report.total_prompts == 1
    assert report.first_ask_keys == (str(outside.parent),)


def test_event_source_counts_prompts_and_decisions(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    _write_session(traces, "s1", [
        _action(0, "2026-09-05T09:00:00Z", "bash_exec", {"command": "cargo build"}),
        {
            "seq": 1, "timestamp": "2026-09-05T09:00:01Z", "event_type": "PermissionAsked",
            "agent_id": "a", "session_id": "s1", "tool_name": "bash_exec",
            "klass": "shell-unfamiliar", "key": "cargo build", "channel": "terminal",
        },
        {
            "seq": 2, "timestamp": "2026-09-05T09:00:09Z", "event_type": "PermissionResolved",
            "agent_id": "a", "session_id": "s1", "tool_name": "bash_exec",
            "klass": "shell-unfamiliar", "key": "cargo build", "decision": "allow_always",
            "latency_ms": 8000, "wrote_grant": True,
        },
        {
            "seq": 3, "timestamp": "2026-09-05T09:01:00Z", "event_type": "PermissionAsked",
            "agent_id": "a", "session_id": "s1", "tool_name": "bash_exec",
            "klass": "shell-destructive", "key": "rm -rf", "channel": "terminal",
        },
        {
            "seq": 4, "timestamp": "2026-09-05T09:01:05Z", "event_type": "PermissionResolved",
            "agent_id": "a", "session_id": "s1", "tool_name": "bash_exec",
            "klass": "shell-destructive", "key": "rm -rf", "decision": "reject_always",
            "latency_ms": 5000, "wrote_grant": False,
        },
    ])
    _write_session(traces, "s2", [_action(0, "2026-09-06T09:00:00Z", "read", {"path": "/tmp/x"})])

    report = build_report(traces)

    assert report.source == "events"
    assert report.total_prompts == 2
    assert report.decisions == {"allow_always": 1, "reject_always": 1}
    assert report.first_ask_keys == ("cargo build",)
    assert report.destructive == (("shell-destructive: rm -rf", 1),)
    assert [s.prompts for s in report.sessions] == [2, 0]


def test_sessions_load_chronologically_and_empty_files_count(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    (traces / "nested").mkdir(parents=True)
    _write_session(traces, "late", [_action(0, "2026-09-09T00:00:00Z", "read", {"path": "/tmp/a"})])
    _write_session(traces / "nested", "early", [_action(0, "2026-09-01T00:00:00Z", "read", {"path": "/tmp/b"})])
    (traces / "blank.jsonl").write_text("", encoding="utf-8")

    sessions = load_sessions(traces)
    assert [s.session_id for s in sessions] == ["early", "late", "blank"]
    assert not has_permission_events(sessions)


def test_buckets_and_meta_resolution() -> None:
    assert [bucket_of(n) for n in (0, 1, 2, 5, 6, 99)] == ["0", "1", "2-5", "2-5", ">5", ">5"]
    assert tool_meta_for("bash_exec").group == "shell"
    mcp = tool_meta_for("discord__reply")
    assert mcp.is_mcp and mcp.mcp_server == "discord"
    assert tool_meta_for("some_plugin_tool").group == "other"


def test_render_reports_slo_lines_and_caveats(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / ".git").mkdir(parents=True)
    traces = tmp_path / "traces"
    traces.mkdir()
    _replay_corpus(traces, workspace)

    text = render(build_report(traces, workspace=workspace, first_n=FIRST_N_SESSIONS_DEFAULT))
    assert "zero-prompt sessions: 2/3 (66.7%; SLO ≥90%)" in text
    assert "median prompts: 0 (SLO 0)" in text
    assert "ungrantable prompts — asked every time (1)" in text
    assert "caveats" in text
    assert "The DENY tier is not replayed" in text


def test_cli_command_is_hidden_and_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    (workspace / ".git").mkdir(parents=True)
    traces = tmp_path / "traces"
    traces.mkdir()
    _replay_corpus(traces, workspace)
    runner = CliRunner()

    result = runner.invoke(app, ["ask-rate", "--traces", str(traces), "--workspace", str(workspace)])
    assert result.exit_code == 0, result.output
    assert "ask-rate report" in result.output

    assert "ask-rate" not in runner.invoke(app, ["--help"]).output
    assert runner.invoke(app, ["ask-rate", "--traces", str(tmp_path / "missing")]).exit_code == 2
