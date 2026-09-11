"""`start` asks ONE question in a project it has never seen (owner bar, 2026-09-11).

It used to ask two, one straight after the other: "No workspace here — create ./.localharness for
this project?" and then "Trust this workspace?". A live end-to-end run met both and the bar is
one, so `offer_workspace_creation` became `settle_startup_trust` — same signature, one sentence,
and a yes does all of it: create `./.localharness` when there is none, record the workspace ROOT
as trusted (the key `cli/session_trust` reads later, which is what keeps the second question from
ever firing), and let the session stay in `auto`. A no creates nothing, records the root as
untrusted, and — when there was a directory to offer — keeps the create-offer's own "asked once,
ever" memory.

Everything here is about the guards and the record, because those are the whole design: the
question WRITES (a directory, a permanent trust decision, or both), so it must be silent in every
run where nobody is watching or where the answer would be wrong, and what it writes has to be
what the session later reads. What it RETURNS is narrow — the workspace this call created, or None,
since whether an already-existing one loads is `resolve_workspace_layer`'s answer — so most of
what a question settles is read back out of the trust store rather than off the return value.
`settle_startup_trust` is tested directly (the prompt is patched; no terminal is involved) and the same-session activation is proven end to end in
`tests/integration/test_start_workspace_offer_e2e.py`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer

from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import (
    DECLINED_NOTICE,
    OFFER_PROMPT,
    RECOGNIZED_NOTICE,
    TRUST_ONLY_PROMPT,
    settle_startup_trust,
)


@pytest.fixture
def project(tmp_path, monkeypatch, fake_home) -> Path:
    """A hermetic `$HOME` with the global layer, and a project directory with NO workspace.

    Both env overrides are cleared: `resolve_workspace_layer` counts either one as an explicit
    selection, and conftest sets `LOCALHARNESS_HOME` for the whole suite — left set, every
    assertion below would pass for the wrong reason.
    """
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    fake_home(home)
    proj = home / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _tty(monkeypatch, present: bool = True) -> None:
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: present)


def _answer(monkeypatch, answer: bool) -> list:
    """Patch the confirmation, recording the QUESTION it was asked with."""
    asked = []

    def _ask(*args, **kwargs):
        asked.append(str(args[0]) if args else str(kwargs.get("prompt")))
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _never_asked(monkeypatch) -> None:
    def _boom(*_a, **_kw):
        raise AssertionError("asked about this workspace where the rule forbids asking")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _decided(root: Path):
    """What `cli/session_trust` will find later — the whole point of recording the ROOT."""
    from localharness.config.trust import is_trusted_tree

    return is_trusted_tree(root)


def _said(captured: str) -> str:
    """stderr with its line breaks flattened, so an assertion is about the words and not about
    the terminal width rich happened to render at."""
    return " ".join(captured.split())


# ------------------------------------------------------------------ yes, no, and nobody there


def test_yes_creates_the_workspace_and_returns_it_as_the_layer(project, monkeypatch, capsys):
    """The returned path is what the caller layers with — created AND active, one session."""
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    result = settle_startup_trust(None)

    assert result == project / ".localharness"
    assert (project / ".localharness" / "config.yaml").is_file()
    assert (project / ".localharness" / "agents").is_dir()
    assert asked == [OFFER_PROMPT], (
        "with nothing here yet, the one question is the one that names what a yes CREATES"
    )
    # `init --workspace`'s closing line tells you to run `start`; this caller IS start.
    assert "run `localharness start`" not in capsys.readouterr().out


def test_no_creates_nothing_and_returns_none(project, monkeypatch):
    """A refusal costs nothing — the session carries on globally (and, now, guarded)."""
    _tty(monkeypatch)
    _answer(monkeypatch, False)

    assert settle_startup_trust(None) is None
    assert not (project / ".localharness").exists()


def test_eof_creates_nothing_and_decides_nothing(project, monkeypatch):
    """A closed stdin is not consent to write to the filesystem — or to the trust store."""
    _tty(monkeypatch)

    def _eof(*_a, **_kw):
        raise EOFError()

    monkeypatch.setattr("rich.prompt.Confirm.ask", _eof)

    assert settle_startup_trust(None) is None
    assert not (project / ".localharness").exists()
    assert _decided(project) is None


def test_no_terminal_never_prompts(project, monkeypatch):
    """Scripts, hooks and CI: the question must be invisible, not a hang — and must not spend
    the one decision this directory gets."""
    _tty(monkeypatch, present=False)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None
    assert not (project / ".localharness").exists()
    assert _decided(project) is None


def test_no_input_never_prompts_even_with_a_terminal(project, monkeypatch):
    """`--no-input` is a caller saying "not me" — a tty it happens to have inherited is not an
    invitation (F6, the same reason doctor/validate/agent create carry the flag)."""
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None, interactive=False) is None
    assert not (project / ".localharness").exists()
    assert _decided(project) is None


def test_the_question_and_its_answer_are_stderr_shaped(project, monkeypatch):
    """Machine output is stdout's job. `agent list --json` renders JSON there, and a trust banner
    or a prompt in the middle of it is a corrupted document — so the console this question uses is
    the stderr one, and its default is NO (return on a prompt nobody read must not trust a tree).
    """
    _tty(monkeypatch)
    seen = {}

    def _ask(*args, **kwargs):
        seen.update(kwargs)
        return False

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)

    settle_startup_trust(None)

    assert seen["console"].stderr is True
    assert seen["default"] is False


# ------------------------------------------------------------------ the "no" is remembered
#
# Asked once per directory, ever — the same contract the trust question has always had (owner
# ruling 2026-09-04). A prompt that comes back every time you start is one people dismiss without
# reading, and this one writes to disk.


def _decline_store(project: Path) -> Path:
    from localharness.config.trust import declined_offers_path

    return declined_offers_path()


def test_a_recorded_no_is_never_asked_again(project, monkeypatch):
    """Two runs, one question: the second must not reach the prompt at all."""
    _tty(monkeypatch)
    _answer(monkeypatch, False)
    assert settle_startup_trust(None) is None

    _never_asked(monkeypatch)
    assert settle_startup_trust(None) is None
    assert not (project / ".localharness").exists()


def test_the_decline_lands_in_the_global_dir_not_the_project(project, monkeypatch):
    """A directory cannot hold the record of its own answer — the same rule the trust store has,
    and here also the only rule that could work: there is no workspace to write it into."""
    _tty(monkeypatch)
    _answer(monkeypatch, False)

    settle_startup_trust(None)

    store = _decline_store(project)
    assert store == project.parent / ".localharness" / "declined_workspace_offers.yaml"
    assert str(project / ".localharness") in store.read_text(encoding="utf-8")
    assert not (project / ".localharness").exists()


def test_eof_records_nothing_and_the_next_session_is_still_asked(project, monkeypatch):
    """A terminal that went away has not decided anything. Only an ANSWERED prompt spends the
    one question this directory gets — the same principle as an unanswered trust prompt."""
    _tty(monkeypatch)

    def _eof(*_a, **_kw):
        raise EOFError()

    monkeypatch.setattr("rich.prompt.Confirm.ask", _eof)
    assert settle_startup_trust(None) is None
    assert not _decline_store(project).exists()

    asked = _answer(monkeypatch, False)
    assert settle_startup_trust(None) is None
    assert len(asked) == 1, "the EOF run had already spent this directory's question"


def test_a_yes_records_no_decline(project, monkeypatch):
    """Nothing to remember in THAT store: the workspace now exists, and the trust record is what
    silences the question from then on."""
    _tty(monkeypatch)
    _answer(monkeypatch, True)

    settle_startup_trust(None)

    assert not _decline_store(project).exists()


def test_init_workspace_still_works_after_a_decline(project, monkeypatch):
    """The store is the offer's memory, not a lock. A user who goes and asks for a workspace gets
    one — `init --workspace` never reads this file, and neither does `mkdir`."""
    from typer.testing import CliRunner

    from localharness.cli.app import app

    _tty(monkeypatch)
    _answer(monkeypatch, False)
    settle_startup_trust(None)

    result = CliRunner().invoke(app, ["init", "--workspace"])

    assert result.exit_code == 0, result.output
    assert (project / ".localharness" / "config.yaml").is_file()


def test_a_corrupt_decline_store_costs_a_question_not_a_session(project, monkeypatch):
    """Fail OPEN here, deliberately — the opposite of the trust store. Nothing but this prompt
    reads the file, so the worst an unreadable one can do is ask again."""
    _tty(monkeypatch)
    store = _decline_store(project)
    store.write_text("{[not: yaml", encoding="utf-8")
    asked = _answer(monkeypatch, False)

    assert settle_startup_trust(None) is None
    assert len(asked) == 1


# ------------------------------------------------------------------ the trust half of the answer
#
# The merged question's other half, and the reason it could be merged at all: the decision is
# recorded on the workspace ROOT, which is the key `cli/session_trust` looks up with
# `is_trusted_tree`. One answer here, and the session question finds it and never fires.


def test_a_yes_records_the_root_as_trusted(project, monkeypatch):
    """The yes that created the workspace is also the yes that trusts it — same sentence, same
    record. Without this the user answers "create ./.localharness" and is then asked to trust the
    directory they just made, which is the two-prompt startup this release deleted."""
    _tty(monkeypatch)
    _answer(monkeypatch, True)

    settle_startup_trust(None)

    assert _decided(project) is True
    sub = project / "src" / "deep"
    sub.mkdir(parents=True)
    assert _decided(sub) is True, "nested dirs inherit the root's answer — `cd src` asks nothing"


def test_a_no_records_the_root_as_untrusted_and_the_offer_as_declined(project, monkeypatch, capsys):
    """A no is an ANSWER, not an absence. It is recorded so the session runs `guarded` instead of
    `auto` — and it is recorded on the same key a yes uses, so neither answer is asked twice."""
    from localharness.config.trust import offer_was_declined

    _tty(monkeypatch)
    _answer(monkeypatch, False)

    settle_startup_trust(None)

    assert _decided(project) is False
    assert offer_was_declined(project / ".localharness") is True
    assert DECLINED_NOTICE.format(root=project) in _said(capsys.readouterr().err)


@pytest.mark.parametrize("answer", [True, False])
def test_a_root_already_decided_is_never_asked_again(project, monkeypatch, answer):
    """Either answer is permanent (v0.13's rule, unchanged): the store is read BEFORE the prompt,
    so a second start in the same project is silent whichever way the first one went."""
    from localharness.config.trust import record_trust

    record_trust(project, answer)
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None
    assert not (project / ".localharness").exists()


def test_a_decision_recorded_on_a_parent_answers_for_the_project(project, monkeypatch):
    """`is_trusted_tree` walks upward, and this is where that matters: a user who trusted a
    checkout is not re-asked in every subdirectory of it they start the harness in."""
    from localharness.config.trust import record_trust

    record_trust(project, True)
    inner = project / "services" / "api"
    inner.mkdir(parents=True)
    monkeypatch.chdir(inner)
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None
    assert not (inner / ".localharness").exists()


def _worked_here_before_this_run(root: Path, sessions: int = 2) -> Path:
    """The state store a few EARLIER sessions leave behind, back-dated so it is evidence.

    `trust.PROCESS_STARTED_AT` is captured when the module is imported — before any test runs —
    so a session file created here and now is newer than the cutoff and is (correctly) not
    counted. Real prior sessions are older than the process asking the question; the `os.utime`
    is what makes these files honest stand-ins for them.
    """
    from localharness.config import trust

    store = root / ".localharness" / "agents" / "orchestrator" / "sessions"
    store.mkdir(parents=True, exist_ok=True)
    old = trust.PROCESS_STARTED_AT - 60
    for index in range(sessions):
        path = store / f"{index}.jsonl"
        path.write_text('{"event_type": "TurnCompleted"}\n', encoding="utf-8")
        os.utime(path, (old, old))
    return root / ".localharness"


def test_a_workspace_with_earlier_sessions_is_recognized_not_asked(project, monkeypatch, capsys):
    """Work has happened here before this run started, so there is nothing to ask about (owner:
    "it should recognize I've been in this environment before"). The answer is recorded anyway,
    so it is explicit from then on rather than re-derived from the filesystem every startup.

    Nothing is returned because nothing was created — `resolve_workspace_layer` is what decides
    whether the workspace that was already here loads.
    """
    _worked_here_before_this_run(project)
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None
    assert _decided(project) is True
    assert RECOGNIZED_NOTICE.format(root=project) in _said(capsys.readouterr().err)


# ------------------------------------------------------------------ where asking would be wrong


@pytest.fixture
def discovery_spy(monkeypatch) -> list:
    """An empty list is the only proof the question short-circuited BEFORE touching the
    filesystem."""
    calls = []

    def _spy(start=None):
        calls.append(start)
        return None

    monkeypatch.setattr("localharness.config.paths.discover_workspace_dir", _spy)
    return calls


def test_an_explicit_config_dir_is_a_full_replacement_not_a_project(
    project, monkeypatch, discovery_spy, tmp_path
):
    """`--config-dir` asked for one directory; layering a new one under it is not that (LAYR-02)."""
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(str(tmp_path / "elsewhere")) is None
    assert discovery_spy == []
    assert not (project / ".localharness").exists()
    assert _decided(project) is None


@pytest.mark.parametrize("var", ["LOCALHARNESS_DIR", "LOCALHARNESS_HOME"])
def test_an_env_override_is_a_full_replacement_too(
    project, monkeypatch, discovery_spy, tmp_path, var
):
    """Both env names count, exactly as they do for discovery itself."""
    monkeypatch.setenv(var, str(tmp_path / "elsewhere"))
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None
    assert discovery_spy == []


def test_an_existing_workspace_is_asked_about_trust_and_never_duplicated(project, monkeypatch):
    """A workspace up-tree means there is nothing to CREATE — not that there is nothing to ask.

    Until v0.14.1 this returned None and asked nothing, and the trust question fired a moment
    later out of `session_trust` instead: one directory, two prompts. Now the same sentence runs
    with the create clause dropped (`TRUST_ONLY_PROMPT`), the existing workspace comes back as the
    layer, and no second `.localharness` is scaffolded beside the one that is already there.
    """
    workspace = project / ".localharness"
    workspace.mkdir()
    sub = project / "src"
    sub.mkdir()
    monkeypatch.chdir(sub)
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    assert settle_startup_trust(None) is None, (
        "this call returns what it CREATED, and an existing workspace is not that — whether that "
        "one's config loads is `resolve_workspace_layer`'s answer and only its"
    )
    assert asked == [TRUST_ONLY_PROMPT], (
        "offering to create a directory that is already there reads as a bug; the question keeps "
        "only the trust clause"
    )
    assert not (sub / ".localharness").exists()
    assert list(workspace.iterdir()) == [], "an existing workspace is never scaffolded into"
    assert _decided(project) is True, "the answer is recorded on the ROOT, not on the dotdir"


def test_a_no_to_an_existing_workspace_records_the_root_only(project, monkeypatch):
    """Nothing was offered, so there is no offer to remember — but the tree is still decided, and
    `session_trust` reads that decision rather than asking its own copy of the question.

    The None matters here beyond bookkeeping: handing the declined workspace back as a layer
    would load the config of the directory the user had just refused to trust.
    """
    from localharness.config.trust import offer_was_declined

    workspace = project / ".localharness"
    workspace.mkdir()
    _tty(monkeypatch)
    _answer(monkeypatch, False)

    assert settle_startup_trust(None) is None
    assert _decided(project) is False
    assert offer_was_declined(workspace) is False


def test_home_is_not_a_project(project, monkeypatch):
    """`./.localharness` standing in `$HOME` IS the machine's global layer — "create a workspace"
    there would mean writing over the config for every project on the machine."""
    home = project.parent
    monkeypatch.chdir(home)
    _tty(monkeypatch)
    _never_asked(monkeypatch)
    before = sorted((home / ".localharness").iterdir())

    assert settle_startup_trust(None) is None
    assert sorted((home / ".localharness").iterdir()) == before
    assert _decided(home) is None


def test_the_global_config_dir_is_not_a_project_even_outside_home(tmp_path, monkeypatch, fake_home):
    """The realpath-keyed check, not the home rule: a global dir somewhere else is still not a
    workspace to be created — or trusted on the strength of being started in."""
    fake_home(tmp_path / "home")
    (tmp_path / "home" / ".localharness").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "home")
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert settle_startup_trust(None) is None


def test_a_failed_scaffold_leaves_startup_alive(project, monkeypatch):
    """The scaffolder exits the process on a filesystem error it has already reported. Inside a
    starting session that would kill the REPL the user actually asked for, so the question
    swallows the exit and the session continues on the global layer."""
    _tty(monkeypatch)
    _answer(monkeypatch, True)

    def _explode(**_kw):
        raise typer.Exit(1)

    monkeypatch.setattr("localharness.cli.init_cmd._scaffold_workspace", _explode)

    assert settle_startup_trust(None) is None
