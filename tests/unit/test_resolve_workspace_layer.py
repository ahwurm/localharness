"""`cli/workspace.resolve_workspace_layer()` — the eight-row decision table, row by row.

One function answers "does a workspace layer apply to THIS invocation, and may we load it?".
Every row below is a row of the table in that function's docstring, and the rows are the
contract 39-05/39-06/39-07 and phase 40 reason against:

- an explicit `--config-dir` or either env var means discovery never runs (LAYR-02) — proven
  with a spy on the walk, because "returns None" alone cannot tell a skipped walk from a
  walked-and-found-nothing one;
- a workspace INSIDE the project you are standing in loads silently and records nothing
  (owner ruling 2026-09-03, "nested inherits") — the two in-project tests make the prompt
  RAISE, so a prompt there fails loudly instead of passing quietly;
- a workspace from OUTSIDE that project is trust-gated (LAYR-05), asked once, and the answer
  is permanent; undecided without a terminal is inert AND unrecorded, so a scripted run cannot
  spend the user's one-time answer for them.

Env note: `Path.home()` reads `$HOME` on POSIX and both walks stop there, so every fixture
plants a fake home. The `.git` markers are created by hand — the boundary check reads them off
the filesystem and never shells out, so no repository has to be created for real.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _squash(text: str) -> str:
    """Whitespace-stripped comparison. Rich wraps at the console width, so a long tmp path
    arrives in the capture with newlines folded into it; squashing both sides is how a path
    assertion survives that wrap."""
    return "".join(text.split())


# --------------------------------------------------------------------------- fixtures


def _fake_home(tmp_path, monkeypatch) -> Path:
    """A hermetic `$HOME` holding the GLOBAL layer (and therefore the trust store), with both
    env overrides cleared so discovery is actually allowed to run."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A discoverable workspace OUTSIDE any repository, two levels above the cwd.

    Deliberately has NO `.git` anywhere: that absence is exactly what makes this workspace
    "config reaching in from outside the tree you opened", which is the only trust-gated case.
    Do not add one. `tmp_path/proj` is also not under `tmp_path/home`, so the `$HOME` walk stop
    does not fire before the workspace is found.
    """
    _fake_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj" / ".localharness"
    (ws / "agents").mkdir(parents=True)
    deep = tmp_path / "proj" / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    # Guard: if TMPDIR itself lived inside a repository, these rows would silently become the
    # in-project case and the gate would never be exercised.
    from localharness.config.paths import workspace_is_within_repo

    assert not workspace_is_within_repo(ws, deep)
    return ws.resolve()


@pytest.fixture
def in_repo_project(project):
    """The same shape plus the repository marker at the project root — the workspace is now
    inside the project you are standing in, so it must load with no prompt at all."""
    (project.parent / ".git").mkdir()
    return project


@pytest.fixture
def cwd_workspace(tmp_path, monkeypatch):
    """The literal `./.localharness` case: the workspace belongs to the current directory and
    there is no repository anywhere. This has always loaded ungated and must keep doing so."""
    _fake_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "proj")
    return ws.resolve()


@pytest.fixture
def workspace_above_repo(tmp_path, monkeypatch):
    """The nearest `.localharness/` sits ABOVE the repository root: you are inside a project,
    but this config comes from a tree outside it, so the gate still applies."""
    _fake_home(tmp_path, monkeypatch)
    ws = tmp_path / "outer" / ".localharness"
    ws.mkdir(parents=True)
    (tmp_path / "outer" / "inner" / ".git").mkdir(parents=True)
    sub = tmp_path / "outer" / "inner" / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    return ws.resolve()


@pytest.fixture
def discovery_spy(monkeypatch):
    """Records every call to the up-tree walk. An empty list is the only proof that the gate
    short-circuited BEFORE touching the filesystem."""
    calls = []

    def _spy(start=None):
        calls.append(start)
        return None

    monkeypatch.setattr("localharness.config.paths.discover_workspace_dir", _spy)
    return calls


def _prompt_must_not_fire(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("prompted on a path whose rule forbids prompting")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _prompt_answers(monkeypatch, answer: bool) -> list:
    """Patch the confirmation to answer `answer`, recording how it was called."""
    asked = []

    def _ask(*args, **kwargs):
        asked.append((args[0] if args else kwargs.get("prompt"), kwargs))
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


# ----------------------------------------------------------------- rows 1-3: the gate


def test_explicit_config_dir_skips_discovery_entirely(project, discovery_spy, tmp_path):
    """LAYR-02: naming a config dir is a full replacement — there is no walk to overrule."""
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(str(tmp_path / "explicit")) is None
    assert discovery_spy == []


def test_localharness_dir_env_skips_discovery(project, discovery_spy, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "elsewhere"))
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer() is None
    assert discovery_spy == []


def test_localharness_home_env_skips_discovery(project, discovery_spy, tmp_path, monkeypatch):
    """The state EVERY existing test runs in (conftest's autouse fixture). This row is what
    keeps the whole suite discovery-inert with zero test edits — if it fails, the gate is
    checking the wrong thing."""
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "elsewhere"))
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer() is None
    assert discovery_spy == []


# ------------------------------------------------------------- row 4: nothing to find


def test_no_workspace_returns_none_silently(tmp_path, monkeypatch, capsys):
    """LAYR-03: a workspace-less session must be byte-identical to today — including silence."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # the walk stops here, so it stays bounded
    empty = tmp_path / "empty" / "sub"
    empty.mkdir(parents=True)
    monkeypatch.chdir(empty)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer() is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# --------------------------------------------------------- row 5: inside your project


def test_workspace_at_the_current_directory_loads_without_prompting(
    cwd_workspace, monkeypatch, capsys
):
    """Today's ungated `./.localharness` read, preserved exactly — and just as quiet."""
    _prompt_must_not_fire(monkeypatch)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=True) == cwd_workspace
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_workspace_inside_the_same_repo_loads_without_prompting(
    in_repo_project, monkeypatch, capsys
):
    """Nested inherits: from deep inside a repository, the workspace at its root is yours."""
    _prompt_must_not_fire(monkeypatch)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=True) == in_repo_project
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_in_project_workspace_records_nothing_in_the_trust_store(in_repo_project, monkeypatch):
    """`interactive=True` on purpose: the prompt is skipped by the RULE, not by the tty check,
    and the in-project path returns before the trust store is even consulted."""
    _prompt_must_not_fire(monkeypatch)
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    ws = in_repo_project
    assert resolve_workspace_layer(interactive=True) == ws
    assert trust.is_trusted(ws) is None


# ------------------------------------------------------- rows 6-8: outside your project


def test_workspace_above_the_repo_root_is_trust_gated(workspace_above_repo, capsys):
    """Being in a repository is not enough — the workspace must be at or below its root."""
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    assert resolve_workspace_layer(interactive=False) is None
    assert _squash(str(workspace_above_repo)) in _squash(capsys.readouterr().err)
    assert trust.is_trusted(workspace_above_repo) is None


def test_outside_project_stored_trust_true_returns_path_without_prompting(project, monkeypatch):
    from localharness.config import trust

    trust.record_trust(project, True)
    _prompt_must_not_fire(monkeypatch)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=True) == project


def test_outside_project_stored_trust_false_is_inert_with_notice(project, monkeypatch, capsys):
    from localharness.config import trust

    trust.record_trust(project, False)
    _prompt_must_not_fire(monkeypatch)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=True) is None
    assert _squash(str(project)) in _squash(capsys.readouterr().err)


def test_undecided_without_tty_is_inert_and_records_nothing(project, capsys):
    """Fail closed, but do NOT record: a later interactive session here still gets asked once."""
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    ws = project
    assert resolve_workspace_layer(interactive=False) is None
    assert _squash(str(ws)) in _squash(capsys.readouterr().err)
    assert trust.is_trusted(ws) is None
    store = trust.trust_store_path()
    stored = store.read_text(encoding="utf-8") if store.exists() else ""
    assert str(ws) not in stored


def test_undecided_prompt_yes_trusts_and_returns_path(project, monkeypatch):
    asked = _prompt_answers(monkeypatch, True)
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    assert resolve_workspace_layer(interactive=True) == project
    assert len(asked) == 1
    _question, kwargs = asked[0]
    assert kwargs["console"].stderr is True  # the prompt itself is on stderr
    assert kwargs["default"] is False  # answering blind declines
    assert trust.is_trusted(project) is True


def test_undecided_prompt_no_records_false_and_returns_none(project, monkeypatch, capsys):
    asked = _prompt_answers(monkeypatch, False)
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    assert resolve_workspace_layer(interactive=True) is None
    assert len(asked) == 1
    assert trust.is_trusted(project) is False
    assert _squash(str(project)) in _squash(capsys.readouterr().err)


def test_trust_is_remembered_on_the_next_invocation(project, monkeypatch):
    """"Trust forever after" — one yes, and no later invocation asks again."""
    _prompt_answers(monkeypatch, True)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=True) == project

    _prompt_must_not_fire(monkeypatch)
    assert resolve_workspace_layer(interactive=True) == project


# ------------------------------------------------------------------ output discipline


def test_all_notices_go_to_stderr_not_stdout(project, monkeypatch, capsys):
    """`agent list --json` writes machine-readable JSON on stdout; a trust banner there would
    corrupt it. Every gated branch must leave stdout empty."""
    from localharness.cli.workspace import resolve_workspace_layer
    from localharness.config import trust

    assert resolve_workspace_layer(interactive=False) is None  # undecided, no terminal
    no_tty = capsys.readouterr()
    assert no_tty.out == ""
    assert no_tty.err != ""

    _prompt_answers(monkeypatch, False)
    assert resolve_workspace_layer(interactive=True) is None  # answered no
    declined = capsys.readouterr()
    assert declined.out == ""
    assert declined.err != ""

    assert trust.is_trusted(project) is False
    assert resolve_workspace_layer(interactive=True) is None  # stored no
    stored = capsys.readouterr()
    assert stored.out == ""
    assert stored.err != ""


def test_a_workspace_path_with_markup_brackets_does_not_crash(tmp_path, monkeypatch, capsys):
    """A folder named `[old] proj` is legal on every OS, and rich reads `[...]` as markup. The
    notice must name it, not raise a MarkupError that takes the whole command down."""
    _fake_home(tmp_path, monkeypatch)
    ws = tmp_path / "[old] proj" / ".localharness"
    ws.mkdir(parents=True)
    sub = tmp_path / "[old] proj" / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    from localharness.cli.workspace import resolve_workspace_layer

    assert resolve_workspace_layer(interactive=False) is None
    assert _squash(str(ws.resolve())) in _squash(capsys.readouterr().err)
