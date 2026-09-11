"""One question per new workspace, and none for a workspace you have already worked in.

Owner ruling 2026-09-11: "default to auto mode so that it asks to trust the workspace, and
allows anything except a dangerous blacklist", and then "it should recognize I've been in this
environment before, used X tools etc."

These drive `cli/session_trust.establish_session_trust` against a real `PermissionGate` and the
real `config/trust` store (pointed at a tmp dir by the autouse fixture in conftest), so the
record on disk is the record the next session would read.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate import PermissionGate
from localharness.agent.gate_types import Decision, PermissionRequest
from localharness.cli.session_trust import (
    RECOGNIZED_NOTICE,
    TRUST_OPTIONS_LEGEND,
    TRUST_QUESTION,
    UNTRUSTED_MODE,
    establish_session_trust,
    state_store_for,
    trust_root,
)
from localharness.config import trust
from localharness.config.grants import GrantStore
from localharness.config.paths import WORKSPACE_DIR_NAME


def _gate(tmp_path: Path, boundary: Path | None, asker=None, mode: str = "auto") -> PermissionGate:
    return PermissionGate(
        boundary=boundary,
        workspace=boundary if boundary is not None else Path.home(),
        grants=GrantStore(tmp_path / "grants.yaml"),
        mode=mode,  # type: ignore[arg-type]
        asker=asker,
        channel_name="test",
    )


def _answers(kind: str, seen: list | None = None):
    async def _asker(request: PermissionRequest) -> Decision:
        if seen is not None:
            seen.append(request)
        return Decision(kind=kind)

    return _asker


def _worked_here(root: Path, sessions: int = 3) -> None:
    """Give a workspace root the state store a few real sessions would have left behind."""
    store = root / WORKSPACE_DIR_NAME / "agents" / "orchestrator" / "sessions"
    store.mkdir(parents=True)
    for index in range(sessions):
        (store / f"{index}.jsonl").write_text('{"event_type": "TurnCompleted"}\n', encoding="utf-8")


# ------------------------------------------------------------------- the question

@pytest.mark.asyncio
async def test_a_brand_new_workspace_asks_once_and_a_yes_is_permanent(tmp_path):
    project = tmp_path / "fresh"
    project.mkdir()
    seen: list[PermissionRequest] = []
    gate = _gate(tmp_path, project, asker=_answers("allow_once", seen))

    assert await establish_session_trust(gate) == "auto"
    assert len(seen) == 1
    assert TRUST_QUESTION in seen[0].display
    assert seen[0].options_legend == TRUST_OPTIONS_LEGEND
    assert trust.is_trusted_tree(project) is True

    # the next session in the same directory asks nothing
    again: list[PermissionRequest] = []
    second = _gate(tmp_path, project, asker=_answers("allow_once", again))
    assert await establish_session_trust(second) == "auto"
    assert again == []


@pytest.mark.asyncio
async def test_a_no_runs_the_session_guarded_and_is_remembered(tmp_path):
    """"No" is an answer, not an absence: it is recorded, and it does not stop the session —
    it puts it in the mode that asks before it crosses a line."""
    project = tmp_path / "declined"
    project.mkdir()
    notices: list[str] = []
    gate = _gate(tmp_path, project, asker=_answers("reject_once"))

    assert await establish_session_trust(gate, notices.append) == UNTRUSTED_MODE
    assert gate.mode == UNTRUSTED_MODE
    assert trust.is_trusted_tree(project) is False
    assert UNTRUSTED_MODE in notices[0]

    asked: list[PermissionRequest] = []
    second = _gate(tmp_path, project, asker=_answers("allow_once", asked))
    assert await establish_session_trust(second) == UNTRUSTED_MODE
    assert asked == [], "a recorded no is not re-litigated every session"


@pytest.mark.asyncio
async def test_a_channel_that_cannot_ask_runs_guarded_and_records_nothing(tmp_path):
    """Fail closed, but never permanently: nobody was asked, so nobody answered, and the next
    interactive session in this directory still gets its one question."""
    project = tmp_path / "piped"
    project.mkdir()
    notices: list[str] = []
    gate = _gate(tmp_path, project, asker=None)

    assert await establish_session_trust(gate, notices.append) == UNTRUSTED_MODE
    assert trust.is_trusted_tree(project) is None, "an unasked question must record nothing"
    assert UNTRUSTED_MODE in notices[0]


# ------------------------------------------------------------------ recognition

@pytest.mark.asyncio
async def test_a_workspace_with_earlier_sessions_is_recognized_not_asked(tmp_path):
    """Owner: "it should recognize I've been in this environment before, used X tools etc."

    A place you have already worked in is not a place to be asked about. The record is written
    anyway, so from here on the answer is explicit rather than re-derived from what happens to
    be on disk.
    """
    project = tmp_path / "familiar"
    project.mkdir()
    _worked_here(project, sessions=3)
    notices: list[str] = []
    asked: list[PermissionRequest] = []
    gate = _gate(tmp_path, project, asker=_answers("reject_once", asked))

    assert await establish_session_trust(gate, notices.append) == "auto"
    assert asked == []
    assert notices == [RECOGNIZED_NOTICE.format(count=3, plural="s")]
    assert trust.is_trusted_tree(project) is True


@pytest.mark.asyncio
async def test_one_earlier_session_reads_as_one_not_ones(tmp_path):
    project = tmp_path / "once"
    project.mkdir()
    _worked_here(project, sessions=1)
    notices: list[str] = []
    await establish_session_trust(_gate(tmp_path, project, asker=_answers("reject_once")),
                                  notices.append)
    assert notices == ["recognized this workspace (1 earlier session)"]


@pytest.mark.asyncio
async def test_an_empty_new_folder_is_not_recognized(tmp_path):
    """A directory that merely EXISTS is not evidence. A bare `localharness init` leaves a
    state store with no sessions in it, and that must still ask."""
    project = tmp_path / "initialised"
    (project / WORKSPACE_DIR_NAME).mkdir(parents=True)
    (project / WORKSPACE_DIR_NAME / "agents").mkdir()
    asked: list[PermissionRequest] = []

    await establish_session_trust(_gate(tmp_path, project, asker=_answers("allow_once", asked)))
    assert len(asked) == 1


@pytest.mark.asyncio
async def test_the_owners_own_global_store_shape_is_recognized(tmp_path, monkeypatch):
    """A home-rooted session has no boundary, so its evidence is the GLOBAL store — the shape
    the owner's own `~/.localharness` has: agents/<name>/sessions/*.jsonl.

    Built from a copy of real session filenames, never from the real directory.
    """
    home = tmp_path / "home"
    config_dir = home / WORKSPACE_DIR_NAME
    global_store = config_dir / "agents" / "orchestrator" / "sessions"
    global_store.mkdir(parents=True)
    for name in ("00548a4c-e82e-43ce-9a91-dde48f5aa20b.jsonl",
                 "00e8d7a2-a5bb-4246-b23c-9367958e8234.jsonl"):
        (global_store / name).write_text('{"event_type": "Action"}\n', encoding="utf-8")
    monkeypatch.setenv("LOCALHARNESS_DIR", str(config_dir))
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))

    notices: list[str] = []
    asked: list[PermissionRequest] = []
    gate = _gate(tmp_path, None, asker=_answers("reject_once", asked))
    assert await establish_session_trust(gate, notices.append) == "auto"
    assert asked == []
    assert notices == [RECOGNIZED_NOTICE.format(count=2, plural="s")]


# ------------------------------------------------------------------ inheritance

@pytest.mark.asyncio
async def test_a_nested_folder_inherits_the_root_it_is_inside(tmp_path):
    project = tmp_path / "proj"
    nested = project / "src" / "thing"
    nested.mkdir(parents=True)
    trust.record_trust(project, True)

    asked: list[PermissionRequest] = []
    gate = _gate(tmp_path, nested, asker=_answers("reject_once", asked))
    assert await establish_session_trust(gate) == "auto"
    assert asked == []


@pytest.mark.asyncio
async def test_trusting_a_project_does_not_trust_the_directory_above_it(tmp_path):
    """The walk goes upward only. A yes on `~/work/thing` must never answer for `~/work`."""
    parent = tmp_path / "work"
    child = parent / "thing"
    child.mkdir(parents=True)
    trust.record_trust(child, True)

    asked: list[PermissionRequest] = []
    await establish_session_trust(_gate(tmp_path, parent, asker=_answers("allow_once", asked)))
    assert len(asked) == 1


def test_one_record_answers_the_config_layer_question_too(tmp_path):
    """Unification (owner: "so there is ONE question and ONE record"): trusting a project root
    also trusts the `.localharness` directory inside it, which is what the workspace layer asks
    about."""
    project = tmp_path / "proj"
    (project / WORKSPACE_DIR_NAME).mkdir(parents=True)
    trust.record_trust(project, True)
    assert trust.is_trusted_tree(project / WORKSPACE_DIR_NAME) is True


# ----------------------------------------------------------------- other modes

@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["guarded", "trusted", "read-only", "unattended"])
async def test_a_mode_somebody_chose_explicitly_is_never_second_guessed(tmp_path, mode):
    """`auto` is the mode whose safety rests on the answer. Every other mode was typed into a
    config on purpose — `unattended` included, which still bypasses everything as it did."""
    project = tmp_path / "pinned"
    project.mkdir()
    asked: list[PermissionRequest] = []
    gate = _gate(tmp_path, project, asker=_answers("reject_once", asked), mode=mode)

    assert await establish_session_trust(gate) == mode
    assert asked == []
    assert trust.is_trusted_tree(project) is None


# ------------------------------------------------------------------- the shapes

def test_a_session_with_no_boundary_is_about_home_and_the_global_store(tmp_path):
    assert trust_root(None) == Path.home().resolve()
    assert trust_root(tmp_path) == tmp_path.resolve()
    assert state_store_for(tmp_path) == tmp_path / WORKSPACE_DIR_NAME
    assert state_store_for(None).name == WORKSPACE_DIR_NAME


# --------------------------------------------------------------- the ask path

@pytest.mark.asyncio
async def test_the_question_goes_through_the_channels_own_ask_path(tmp_path):
    """The ACP requirement, and the terminal's and Discord's: the question is a
    `PermissionRequest` handed to `gate.asker`, which is `channel.ask_permission`.

    So it renders wherever permission questions render — inline in the terminal, as a
    `request_permission` dialog in Zed, as a message in Discord — and no channel needs a second
    code path for it. `grantable=False` puts the two-option pair in front of the person, which
    is the [y/N] the ruling asked for; ACP reads `grantable` and `display` as plain attributes,
    and both are present.
    """
    project = tmp_path / "acp"
    project.mkdir()
    seen: list[PermissionRequest] = []

    class _Channel:
        can_ask = True

        async def ask_permission(self, request):
            seen.append(request)
            return Decision(kind="allow_once")

    gate = _gate(tmp_path, project)
    gate.attach_channel(_Channel())
    assert await establish_session_trust(gate) == "auto"

    (request,) = seen
    assert request.grantable is False
    assert isinstance(request.display, str) and "\n" in request.display, (
        "ACP splits display into a dialog title and a body; the detail needs its own line"
    )
    assert str(project) in request.display
    assert request.call_id is None, "there is no tool call to pair this with"
