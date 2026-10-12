"""The instructions clause rides the existing trust questions (phase 11, plan 06; D-02, D-14, D-15,
D-49, D-61).

There is no question of its own anywhere. Where the project holds a `CLAUDE.md`, `AGENTS.md` or
`.localharness/INSTRUCTIONS.md`, the one trust question names it (one more noun), and a yes writes
two records: the trust record it writes today and the instructions `yes` record. A no writes no
instructions record. Where no file is found, every question is asked byte for byte as before.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.cli import session_trust
from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import (
    OFFER_PROMPT,
    OFFER_PROMPT_WITH_INSTRUCTIONS,
    TRUST_ONLY_PROMPT,
    TRUST_ONLY_PROMPT_WITH_INSTRUCTIONS,
    settle_startup_trust,
)
from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision, PermissionRequest
from localharness.config import trust
from localharness.config.grants import GrantStore
from localharness.config.project_records import read_instructions, store_path

OFFER_CLAUSE = (
    "Trust this workspace? LocalHarness will keep its state in ./.localharness here, load that "
    "config and its CLAUDE.md, and run tools without asking except for dangerous actions."
)
TRUST_ONLY_CLAUSE = (
    "Trust this workspace? LocalHarness will load its .localharness config and its AGENTS.md, "
    "and run tools here without asking, except for dangerous actions."
)
SESSION_CLAUSE = (
    "Trust this workspace? LocalHarness will load its .localharness config and its CLAUDE.md, "
    "and run tools here without asking, except for dangerous actions."
)
# Copied before the edit: the outside-workspace question is not touched by this plan (D-14).
OUTSIDE_TRUST_QUESTION = (
    "Trust the workspace at {parent}? It is outside the project you are in. LocalHarness will "
    "load its .localharness config — agents, models and tool permissions, which you should treat "
    "like code you are about to run — and run tools there without asking, except for dangerous "
    "actions."
)


@pytest.fixture
def project(tmp_path, monkeypatch, fake_home) -> Path:
    """A hermetic `$HOME` with the global layer and a project directory with NO workspace."""
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    fake_home(home)
    proj = home / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _tty(monkeypatch, present: bool = True) -> None:
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: present)


def _answer(monkeypatch, answer: bool | None) -> list:
    """Patch the confirmation, recording the QUESTION it was asked with."""
    asked: list[str] = []

    def _ask(*args, **kwargs):
        asked.append(str(args[0]) if args else str(kwargs.get("prompt")))
        if answer is None:
            raise EOFError()
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _store_roots() -> dict:
    """The store's roots as plain data, read where `fake_home` put the global dir."""
    path = store_path()
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("roots") or {}


# ------------------------------------------------------------------ the offer prompt (Task 1)


def test_the_constants_are_the_fixed_texts():
    assert OFFER_PROMPT_WITH_INSTRUCTIONS.format(name="CLAUDE.md") == OFFER_CLAUSE
    assert TRUST_ONLY_PROMPT_WITH_INSTRUCTIONS.format(name="AGENTS.md") == TRUST_ONLY_CLAUSE
    # the originals are still their own strings
    assert session_trust.TRUST_QUESTION_WITH_INSTRUCTIONS.format(name="CLAUDE.md") == SESSION_CLAUSE
    assert OFFER_PROMPT.endswith("load that config, and run tools without asking except for "
                                 "dangerous actions.")


def test_a_fresh_clone_with_claude_md_gets_one_offer_with_the_clause_and_a_yes_writes_both_records(
    project, monkeypatch
):
    body = b"# Project rules\nuse tabs\n"
    (project / "CLAUDE.md").write_bytes(body)
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    result = settle_startup_trust(None)

    assert result == project / ".localharness"
    assert asked == [OFFER_CLAUSE], "exactly one ask, naming the file"
    assert trust.is_trusted_tree(project) is True
    record = read_instructions(project)
    assert record is not None
    assert record.state == "yes"
    assert record.name == "CLAUDE.md"
    assert record.digest == hashlib.sha256(body).hexdigest()
    assert Path(record.path) == (project / "CLAUDE.md").resolve()


def test_the_same_project_without_a_candidate_is_asked_the_original_byte_for_byte(
    project, monkeypatch
):
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    settle_startup_trust(None)

    assert asked == [OFFER_PROMPT]
    assert read_instructions(project) is None
    assert _store_roots() == {}


# ------------------------------------------------------------------ trust-only and the No paths


def test_an_existing_workspace_with_agents_md_is_asked_the_trust_only_clause(project, monkeypatch):
    (project / ".localharness").mkdir()
    (project / "AGENTS.md").write_text("agents rules\n")
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    assert settle_startup_trust(None) is None

    assert asked == [TRUST_ONLY_CLAUSE]
    assert trust.is_trusted_tree(project) is True
    record = read_instructions(project)
    assert record is not None and record.name == "AGENTS.md" and record.state == "yes"


def test_an_existing_workspace_without_a_candidate_is_asked_the_original(project, monkeypatch):
    (project / ".localharness").mkdir()
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    settle_startup_trust(None)

    assert asked == [TRUST_ONLY_PROMPT]
    assert read_instructions(project) is None


def test_a_no_on_the_offer_records_the_decline_and_no_instructions_record(project, monkeypatch):
    (project / "CLAUDE.md").write_text("rules\n")
    _tty(monkeypatch)
    asked = _answer(monkeypatch, False)

    assert settle_startup_trust(None) is None

    assert asked == [OFFER_CLAUSE]
    assert trust.is_trusted_tree(project) is False
    assert trust.offer_was_declined(project / ".localharness") is True
    assert read_instructions(project) is None
    assert not (project / ".localharness").exists()


def test_a_no_on_the_trust_only_question_records_no_instructions_record(project, monkeypatch):
    (project / ".localharness").mkdir()
    (project / "AGENTS.md").write_text("rules\n")
    _tty(monkeypatch)
    _answer(monkeypatch, False)

    settle_startup_trust(None)

    assert trust.is_trusted_tree(project) is False
    assert read_instructions(project) is None


def test_eof_records_nothing_at_all(project, monkeypatch):
    (project / "CLAUDE.md").write_text("rules\n")
    _tty(monkeypatch)
    asked = _answer(monkeypatch, None)

    assert settle_startup_trust(None) is None

    assert asked == [OFFER_CLAUSE]
    assert trust.is_trusted_tree(project) is None
    assert read_instructions(project) is None
    assert _store_roots() == {}


def test_a_symlink_leading_outside_the_project_is_skipped_and_the_original_is_asked(
    project, tmp_path, monkeypatch
):
    outside = tmp_path / "elsewhere.md"
    outside.write_text("not yours\n")
    (project / "CLAUDE.md").symlink_to(outside)
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    settle_startup_trust(None)

    assert asked == [OFFER_PROMPT]
    assert read_instructions(project) is None


def test_a_store_failure_while_recording_costs_the_record_never_the_answer(
    project, monkeypatch, caplog
):
    (project / "CLAUDE.md").write_text("rules\n")
    _tty(monkeypatch)
    _answer(monkeypatch, True)

    def _boom(*_a, **_kw):
        raise OSError("read-only store")

    monkeypatch.setattr("localharness.config.project_records.accept_instructions", _boom)

    with caplog.at_level(logging.WARNING, logger=ws_mod.log.name):
        result = settle_startup_trust(None)

    assert result == project / ".localharness"
    assert trust.is_trusted_tree(project) is True
    assert any("instructions answer" in r.getMessage() for r in caplog.records)


def test_the_outside_workspace_question_is_unchanged_byte_for_byte():
    assert ws_mod.TRUST_QUESTION == OUTSIDE_TRUST_QUESTION
    assert session_trust.TRUST_QUESTION != ws_mod.TRUST_QUESTION


# ------------------------------------------------------------------ the session-trust question


class _Handle:
    """The `ProjectInstructions` surface `establish_session_trust` uses: the display name, and the
    two calls a yes and a no make."""

    def __init__(self, name: str = "CLAUDE.md", *, raises: bool = False) -> None:
        self.state = SimpleNamespace(name=name)
        self.calls: list[tuple] = []
        self._raises = raises

    def set_state(self, value: str) -> None:
        self.calls.append(("set_state", value))
        if self._raises:
            raise OSError("store unwritable")

    def decline(self) -> None:
        self.calls.append(("decline",))
        if self._raises:
            raise OSError("store unwritable")


def _gate(tmp_path: Path, boundary: Path, asker=None, mode: str = "auto") -> PermissionGate:
    return PermissionGate(
        boundary=boundary,
        workspace=boundary,
        grants=GrantStore(tmp_path / "grants.yaml"),
        mode=mode,  # type: ignore[arg-type]
        asker=asker,
        channel_name="test",
    )


def _asker(kind: str, seen: list):
    async def _ask(request: PermissionRequest) -> Decision:
        seen.append(request)
        return Decision(kind=kind)

    return _ask


@pytest.fixture
def fresh(tmp_path) -> Path:
    root = tmp_path / "fresh"
    root.mkdir()
    return root


async def test_the_session_question_carries_the_clause_and_a_yes_sets_the_state(tmp_path, fresh):
    seen: list[PermissionRequest] = []
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))

    mode = await session_trust.establish_session_trust(gate, instructions=handle)

    assert mode == "auto"
    assert len(seen) == 1, "one question, never a second"
    assert seen[0].reason == SESSION_CLAUSE
    assert seen[0].display.startswith(SESSION_CLAUSE)
    assert trust.is_trusted_tree(fresh) is True
    assert handle.calls == [("set_state", "yes")]


async def test_a_session_no_declines_the_file_and_records_the_no(tmp_path, fresh):
    seen: list[PermissionRequest] = []
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=_asker("reject_once", seen))

    mode = await session_trust.establish_session_trust(gate, instructions=handle)

    assert mode == session_trust.UNTRUSTED_MODE
    assert len(seen) == 1
    assert trust.is_trusted_tree(fresh) is False
    assert handle.calls == [("decline",)], "set_state is never called on a no"


async def test_without_a_handle_the_request_is_the_original_question(tmp_path, fresh):
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))

    assert await session_trust.establish_session_trust(gate) == "auto"

    assert seen[0].reason == session_trust.TRUST_QUESTION
    assert seen[0].display.startswith(session_trust.TRUST_QUESTION)


async def test_a_handle_with_no_name_asks_the_original_question(tmp_path, fresh):
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))

    await session_trust.establish_session_trust(gate, instructions=_Handle(name=""))

    assert seen[0].reason == session_trust.TRUST_QUESTION


async def test_a_recorded_no_declines_the_file_and_does_not_ask(tmp_path, fresh):
    trust.record_trust(fresh, False)
    seen: list[PermissionRequest] = []
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))

    mode = await session_trust.establish_session_trust(gate, instructions=handle)

    assert mode == session_trust.UNTRUSTED_MODE
    assert seen == []
    assert handle.calls == [("decline",)]


async def test_a_recorded_yes_asks_nothing_and_never_touches_the_handle(tmp_path, fresh):
    trust.record_trust(fresh, True)
    seen: list[PermissionRequest] = []
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))

    assert await session_trust.establish_session_trust(gate, instructions=handle) == "auto"

    assert seen == []
    assert handle.calls == []


async def test_a_channel_that_cannot_ask_never_touches_the_handle(tmp_path, fresh):
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=None)

    assert await session_trust.establish_session_trust(gate, instructions=handle) == (
        session_trust.UNTRUSTED_MODE
    )

    assert handle.calls == []
    assert trust.is_trusted_tree(fresh) is None


async def test_a_handle_that_raises_fails_neither_answer(tmp_path, fresh, caplog):
    seen: list[PermissionRequest] = []
    yes = _gate(tmp_path, fresh, asker=_asker("allow_once", seen))
    with caplog.at_level(logging.WARNING, logger=session_trust.log.name):
        assert await session_trust.establish_session_trust(
            yes, instructions=_Handle(raises=True)
        ) == "auto"
    assert trust.is_trusted_tree(fresh) is True
    assert any("project instructions" in r.getMessage() for r in caplog.records)

    other = tmp_path / "other"
    other.mkdir()
    no = _gate(tmp_path, other, asker=_asker("reject_once", []))
    assert await session_trust.establish_session_trust(
        no, instructions=_Handle(raises=True)
    ) == session_trust.UNTRUSTED_MODE
    assert trust.is_trusted_tree(other) is False


async def test_a_mode_nobody_gates_never_touches_the_handle(tmp_path, fresh):
    handle = _Handle()
    gate = _gate(tmp_path, fresh, asker=_asker("allow_once", []), mode="guarded")

    assert await session_trust.establish_session_trust(gate, instructions=handle) == "guarded"

    assert handle.calls == []
