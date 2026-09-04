"""`start` offers to create a workspace when a project has none (owner ruling 2026-09-04).

The gap this closes is not "how do I make a workspace" — `init --workspace` has always done that —
it is that nobody knows the layer exists until they read the docs, and the moment they would want
one is the moment they start the harness inside a project. So `start` asks, once, at that moment.

Everything here is about the guards, because the guards are the whole design: the question WRITES
a directory, so it must be silent in every run where nobody is watching or where the answer would
be wrong. `offer_workspace_creation` is tested directly (the prompt is patched; no terminal is
involved) and the same-session activation is proven end to end in
`tests/integration/test_start_workspace_offer_e2e.py`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import typer

from localharness.cli import workspace as ws_mod
from localharness.cli.workspace import offer_workspace_creation


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A hermetic `$HOME` with the global layer, and a project directory with NO workspace.

    Both env overrides are cleared: `resolve_workspace_layer` counts either one as an explicit
    selection, and conftest sets `LOCALHARNESS_HOME` for the whole suite — left set, every
    assertion below would pass for the wrong reason.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    proj = home / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


def _tty(monkeypatch, present: bool = True) -> None:
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: present)


def _answer(monkeypatch, answer: bool) -> list:
    """Patch the confirmation, recording that it was asked."""
    asked = []

    def _ask(*args, **kwargs):
        asked.append(args[0] if args else kwargs.get("prompt"))
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _never_asked(monkeypatch) -> None:
    def _boom(*_a, **_kw):
        raise AssertionError("offered to create a workspace where the rule forbids asking")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


# ------------------------------------------------------------------ yes, no, and nobody there


def test_yes_creates_the_workspace_and_returns_it_as_the_layer(project, monkeypatch, capsys):
    """The returned path is what the caller layers with — created AND active, one session."""
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)

    result = offer_workspace_creation(None)

    assert result == project / ".localharness"
    assert (project / ".localharness" / "config.yaml").is_file()
    assert (project / ".localharness" / "agents").is_dir()
    assert len(asked) == 1 and "create ./.localharness" in str(asked[0])
    # `init --workspace`'s closing line tells you to run `start`; this caller IS start.
    assert "run `localharness start`" not in capsys.readouterr().out


def test_no_creates_nothing_and_returns_none(project, monkeypatch):
    """A refusal costs nothing and says nothing — the session carries on globally."""
    _tty(monkeypatch)
    _answer(monkeypatch, False)

    assert offer_workspace_creation(None) is None
    assert not (project / ".localharness").exists()


def test_eof_is_a_no(project, monkeypatch):
    """A closed stdin is not consent to write to the filesystem."""
    _tty(monkeypatch)

    def _eof(*_a, **_kw):
        raise EOFError()

    monkeypatch.setattr("rich.prompt.Confirm.ask", _eof)

    assert offer_workspace_creation(None) is None
    assert not (project / ".localharness").exists()


def test_no_terminal_never_prompts(project, monkeypatch):
    """Scripts, hooks and CI: the offer must be invisible, not a hang."""
    _tty(monkeypatch, present=False)
    _never_asked(monkeypatch)

    assert offer_workspace_creation(None) is None
    assert not (project / ".localharness").exists()


def test_no_input_never_prompts_even_with_a_terminal(project, monkeypatch):
    """`--no-input` is a caller saying "not me" — a tty it happens to have inherited is not an
    invitation (F6, the same reason doctor/validate/agent create carry the flag)."""
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert offer_workspace_creation(None, interactive=False) is None
    assert not (project / ".localharness").exists()


# ------------------------------------------------------------------ where asking would be wrong


@pytest.fixture
def discovery_spy(monkeypatch) -> list:
    """An empty list is the only proof the offer short-circuited BEFORE touching the filesystem."""
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

    assert offer_workspace_creation(str(tmp_path / "elsewhere")) is None
    assert discovery_spy == []
    assert not (project / ".localharness").exists()


@pytest.mark.parametrize("var", ["LOCALHARNESS_DIR", "LOCALHARNESS_HOME"])
def test_an_env_override_is_a_full_replacement_too(
    project, monkeypatch, discovery_spy, tmp_path, var
):
    """Both env names count, exactly as they do for discovery itself."""
    monkeypatch.setenv(var, str(tmp_path / "elsewhere"))
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert offer_workspace_creation(None) is None
    assert discovery_spy == []


def test_an_existing_workspace_up_tree_is_not_offered_a_second_one(project, monkeypatch):
    """"Is there a workspace here" is answered by the walk, not by whether the gate loaded it —
    otherwise a declined workspace gets a duplicate created beside it."""
    (project / ".localharness").mkdir()
    sub = project / "src"
    sub.mkdir()
    monkeypatch.chdir(sub)
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert offer_workspace_creation(None) is None
    assert not (sub / ".localharness").exists()


def test_home_is_not_a_project(project, monkeypatch):
    """`./.localharness` standing in `$HOME` IS the machine's global layer — "create a workspace"
    there would mean writing over the config for every project on the machine."""
    home = project.parent
    monkeypatch.chdir(home)
    _tty(monkeypatch)
    _never_asked(monkeypatch)
    before = sorted((home / ".localharness").iterdir())

    assert offer_workspace_creation(None) is None
    assert sorted((home / ".localharness").iterdir()) == before


def test_the_global_config_dir_is_not_a_project_even_outside_home(tmp_path, monkeypatch):
    """The realpath-keyed check, not the home rule: a global dir somewhere else is still not a
    workspace to be created."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".localharness").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "home")
    _tty(monkeypatch)
    _never_asked(monkeypatch)

    assert offer_workspace_creation(None) is None


def test_a_failed_scaffold_leaves_startup_alive(project, monkeypatch):
    """The scaffolder exits the process on a filesystem error it has already reported. Inside a
    starting session that would kill the REPL the user actually asked for, so the offer swallows
    the exit and the session continues on the global layer."""
    _tty(monkeypatch)
    _answer(monkeypatch, True)

    def _explode(**_kw):
        raise typer.Exit(1)

    monkeypatch.setattr("localharness.cli.init_cmd._scaffold_workspace", _explode)

    assert offer_workspace_creation(None) is None
