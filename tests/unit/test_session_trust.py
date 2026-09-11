"""One question per new workspace, and none for a workspace you have already worked in.

Owner ruling 2026-09-11: "default to auto mode so that it asks to trust the workspace, and
allows anything except a dangerous blacklist", and then "it should recognize I've been in this
environment before, used X tools etc."

These drive `cli/session_trust.establish_session_trust` against a real `PermissionGate` and the
real `config/trust` store (pointed at a tmp dir by the autouse fixture in conftest), so the
record on disk is the record the next session would read.
"""
from __future__ import annotations

import os
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


def _backdate(path: Path) -> Path:
    """Make a file look like it was written by an EARLIER run.

    `prior_session_count` only counts session files older than `trust.PROCESS_STARTED_AT`,
    because the bug it closes was a brand-new project recognizing the file the RUNNING session had
    just written. A test that wants the recognition path has to write evidence that genuinely
    predates this process, and backdating the mtime is how.
    """
    earlier = trust.PROCESS_STARTED_AT - EVIDENCE_AGE_SECONDS
    os.utime(path, (earlier, earlier))
    return path


EVIDENCE_AGE_SECONDS = 60
"""How far back `_backdate` puts a file. Any positive number works; a minute is far enough to be
unambiguous and near enough to read as "the session before this one"."""


def _worked_here(root: Path, sessions: int = 3) -> None:
    """Give a workspace root the state store a few EARLIER sessions would have left behind."""
    store = root / WORKSPACE_DIR_NAME / "agents" / "orchestrator" / "sessions"
    store.mkdir(parents=True)
    for index in range(sessions):
        path = store / f"{index}.jsonl"
        path.write_text('{"event_type": "TurnCompleted"}\n', encoding="utf-8")
        _backdate(path)


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
        path = global_store / name
        path.write_text('{"event_type": "Action"}\n', encoding="utf-8")
        _backdate(path)

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


# ------------------------------------------------- exactly one question, per path

def _outside_workspace(tmp_path: Path, monkeypatch, fake_home) -> tuple[Path, Path]:
    """A project whose `.localharness/` reaches in from OUTSIDE any repository — the one case
    `resolve_workspace_layer` trust-gates — with a never-seen root, and the cwd deep inside it.

    No `.git` anywhere on purpose: its absence is what makes this "config from outside the tree
    you opened" rather than your own project's. `tmp_path/proj` is not under the fake home, so
    the `$HOME` walk stop does not fire before the workspace is found — the same shape
    `tests/unit/test_resolve_workspace_layer.py` uses, for the same reasons.
    """
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    fake_home(home)
    root = tmp_path / "proj"
    (root / WORKSPACE_DIR_NAME / "agents").mkdir(parents=True)
    deep = root / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    return root, root / WORKSPACE_DIR_NAME


@pytest.mark.asyncio
async def test_the_repl_path_asks_exactly_once_and_then_loads_the_config(tmp_path, monkeypatch, fake_home):
    """The double-question that v0.14.1 could have shipped: the config-layer question fires on
    the synchronous startup path, and the session-trust question fires once the channel is live.

    Owner ruling 2026-09-11: "unify… so there is ONE question and ONE record". The layer resolver
    records the workspace ROOT, and `session_trust` reads the root with `is_trusted_tree`, so the
    second question finds the first one's answer and never asks. After the yes, the outside
    config layer IS loaded and the session is still in `auto`.
    """
    from localharness.cli.workspace import resolve_workspace_layer

    root, workspace = _outside_workspace(tmp_path, monkeypatch, fake_home)
    asked: list[str] = []

    def _asker(question: str) -> bool:
        asked.append(question)
        return True

    assert resolve_workspace_layer(asker=_asker) == workspace, "the config layer loads"

    gate = _gate(tmp_path, root, asker=_answers("reject_once"))
    assert await establish_session_trust(gate) == "auto"
    assert len(asked) == 1, "the session was asked to trust the same workspace twice"


@pytest.mark.asyncio
async def test_the_acp_path_asks_exactly_once_through_request_permission(tmp_path, monkeypatch, fake_home):
    """Same guarantee on the path with no REPL. ACP injects its own asker into the layer
    resolver (`acp._trust_asker`) and reaches `establish_session_trust` before `serve()`; both
    render through `request_permission`, and between them they must draw ONE dialog."""
    from localharness.cli.workspace import resolve_workspace_layer

    root, workspace = _outside_workspace(tmp_path, monkeypatch, fake_home)
    dialogs: list[str] = []

    class _AcpLike:
        """Stands in for the ACP channel at both call sites: a sync asker for the layer
        resolver, and `ask_permission` for the gate."""

        can_ask = True

        def trust_asker(self):
            def _ask(question: str) -> bool:
                dialogs.append(question)
                return True
            return _ask

        async def ask_permission(self, request):
            dialogs.append(request.display)
            return Decision(kind="allow_once")

    channel = _AcpLike()
    assert resolve_workspace_layer(asker=channel.trust_asker()) == workspace

    gate = _gate(tmp_path, root)
    gate.attach_channel(channel)
    assert await establish_session_trust(gate) == "auto"
    assert len(dialogs) == 1


@pytest.mark.asyncio
async def test_a_declined_config_layer_also_settles_the_session_mode(tmp_path, monkeypatch, fake_home):
    """The other direction of the same record: saying no to the outside config also means no to
    running tools there silently, and it is not asked a second time."""
    from localharness.cli.workspace import resolve_workspace_layer

    root, _workspace = _outside_workspace(tmp_path, monkeypatch, fake_home)
    asked: list[str] = []

    assert resolve_workspace_layer(asker=lambda q: asked.append(q) or False) is None

    gate = _gate(tmp_path, root, asker=_answers("allow_once"))
    assert await establish_session_trust(gate) == UNTRUSTED_MODE
    assert len(asked) == 1


# --------------------------------------------- the order the real start path uses

def _session_file(root: Path, name: str = "s1") -> Path:
    """The file a session writes as it runs — the thing that must not vouch for its own run."""
    sessions = root / WORKSPACE_DIR_NAME / "agents" / "orchestrator" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    path = sessions / f"{name}.jsonl"
    path.write_text('{"event_type": "TurnCompleted"}\n', encoding="utf-8")
    return path


def test_a_brand_new_project_asks_and_the_second_start_does_not(tmp_path, monkeypatch, fake_home):
    """The blocker a live end-to-end run found: a brand-new project printed "recognized this
    workspace (1 earlier session)", trusted itself and wrote the record, with nobody asked.

    The evidence it recognized was the file the RUNNING session had just written. Two things fix
    it and both are asserted here — the question runs BEFORE any session store is opened (which
    is why `settle_startup_trust` is called from the same place `_start_async` calls it, ahead of
    the state dir), and a session file is only evidence if it predates this process.
    """
    from localharness.cli import workspace as workspace_mod
    from localharness.cli.workspace import settle_startup_trust

    fake_home(tmp_path / "home")
    project = tmp_path / "fresh"
    project.mkdir()
    monkeypatch.chdir(project)

    asked: list[str] = []
    monkeypatch.setattr(workspace_mod, "_ask_create", lambda prompt=None: asked.append(prompt) or True)
    monkeypatch.setattr(workspace_mod, "_stdin_is_a_terminal", lambda: True)

    assert settle_startup_trust() is not None, "a yes creates the workspace"
    assert len(asked) == 1, "a brand-new project must be asked, not recognized"
    assert trust.is_trusted_tree(project) is True

    # the session then runs and writes its own file; a second start must not re-ask, and must not
    # have needed that file to decide. It creates nothing either — the workspace is already
    # there, and `resolve_workspace_layer` is what loads it.
    _session_file(project)
    assert settle_startup_trust() is None
    assert len(asked) == 1


def test_this_runs_own_session_file_is_not_evidence(tmp_path, fake_home):
    """The belt to that pair of braces, for the channels — ACP, Discord — whose question cannot
    be drawn until after the session store is already open."""
    fake_home(tmp_path / "home")
    project = tmp_path / "fresh"
    project.mkdir()
    _session_file(project)  # written NOW, i.e. after trust.PROCESS_STARTED_AT

    assert trust.prior_session_count(project / WORKSPACE_DIR_NAME) == 0


@pytest.mark.asyncio
async def test_a_session_file_from_a_previous_run_is_evidence(tmp_path, fake_home):
    """The other direction, so the fix cannot be read as "recognition stopped working"."""
    fake_home(tmp_path / "home")
    project = tmp_path / "familiar"
    project.mkdir()
    _backdate(_session_file(project))

    assert trust.prior_session_count(project / WORKSPACE_DIR_NAME) == 1

    asked: list[PermissionRequest] = []
    gate = _gate(tmp_path, project, asker=_answers("reject_once", asked))
    assert await establish_session_trust(gate) == "auto"
    assert asked == []


def test_a_declined_outside_workspace_is_not_loaded_by_the_back_door(tmp_path, monkeypatch, fake_home):
    """`settle_startup_trust` returns only what it CREATED.

    `_start_async` falls back to its return value when `resolve_workspace_layer` returned None —
    and None is exactly what the layer resolver returns for a workspace the trust gate declined.
    Returning the existing workspace from here would have handed that caller the very config
    layer the human had just refused.
    """
    from localharness.cli.workspace import resolve_workspace_layer, settle_startup_trust

    root, workspace = _outside_workspace(tmp_path, monkeypatch, fake_home)
    assert resolve_workspace_layer(asker=lambda _q: False) is None, "declined, so no layer"
    assert workspace.is_dir(), "the workspace is still there; it is simply not loaded"
    assert settle_startup_trust() is None, "a declined workspace must not come back this way"
