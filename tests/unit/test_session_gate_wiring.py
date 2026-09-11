"""What `localharness start` hands the gate, and who else holds it (PRD §3.1, §3.4, §3.5).

Two things a unit test of `PermissionGate` alone cannot prove: that the boundary a real session
derives is the project it is standing in, and that the ONE gate object reaches the loop, the
subagent runner and the REPL. Plus the trust dialog's new seam — the injectable asker the ACP
adapter will pass in phase B.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from localharness.agent.gate import PermissionGate, derive_session_boundary
from localharness.agent.verdict import narrow_boundary
from localharness.cli import start_cmd
from localharness.cli.repl import OrchestratorREPL
from localharness.cli.workspace import TRUST_QUESTION, resolve_workspace_layer
from localharness.config.grants import GrantStore


# ------------------------------------------------------------------- boundary

def test_the_boundary_is_the_project_you_are_standing_in(tmp_path, monkeypatch):
    project = tmp_path / "proj"
    (project / "sub").mkdir(parents=True)
    (project / ".git").mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    (tmp_path / "home").mkdir()

    assert derive_session_boundary(cwd=project / "sub") == project.resolve()


def test_a_workspace_layer_wins_over_the_checkout(tmp_path, monkeypatch):
    """v0.13 discovery found a project; that folder is the boundary (PRD §3.1)."""
    repo = tmp_path / "repo"
    inner = repo / "inner"
    (inner / ".localharness").mkdir(parents=True)
    (repo / ".git").mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "home"))
    (tmp_path / "home").mkdir()

    assert derive_session_boundary(cwd=inner, local_dir=inner / ".localharness") == inner.resolve()


def test_home_collapses_to_no_boundary(tmp_path, monkeypatch):
    """Critic finding 1: a boundary containing your whole home directory is not a boundary."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    assert derive_session_boundary(cwd=home) is None


def test_config_may_narrow_the_boundary_but_never_move_it(tmp_path):
    project = tmp_path / "proj"
    (project / "src").mkdir(parents=True)
    inside, warning = narrow_boundary(project, str(project / "src"))
    assert inside == (project / "src").resolve() and warning is None

    outside, warning = narrow_boundary(project, str(tmp_path))
    assert outside == project, "config moved the boundary outward"
    assert warning and "ignored" in warning


def test_the_no_boundary_notice_names_the_fix():
    assert "ask" in start_cmd.NO_BOUNDARY_NOTICE
    assert ".git" in start_cmd.NO_BOUNDARY_NOTICE


# ------------------------------------------------------------ one shared object

def _gate(tmp_path: Path) -> PermissionGate:
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    return PermissionGate(
        boundary=workspace, workspace=workspace, grants=GrantStore(tmp_path / "g.yaml"),
        channel_name="none",
    )


class _Channel:
    channel_id = "fake"
    can_ask = True
    has_review_surface = True

    async def ask_permission(self, request):  # pragma: no cover - identity only
        raise AssertionError("not called")


class _MuteChannel:
    channel_id = "mute"
    can_ask = False
    has_review_surface = False


def test_attach_channel_takes_the_asker_and_both_flags(tmp_path):
    gate, channel = _gate(tmp_path), _Channel()
    gate.attach_channel(channel)
    assert gate.channel_name == "fake"
    assert gate.has_review_surface is True
    assert gate.asker == channel.ask_permission


def test_a_channel_that_cannot_ask_leaves_the_gate_fail_closed(tmp_path):
    gate = _gate(tmp_path)
    gate.attach_channel(_MuteChannel())
    assert gate.asker is None
    assert gate.channel_name == "mute"


def test_start_cmd_passes_one_gate_to_loop_runner_and_repl():
    """The structural guard: three call sites, one object. A gate the REPL does not hold makes
    `/mode` a no-op; a gate the subagent runner does not hold makes a child ungated."""
    source = Path(start_cmd.__file__).read_text(encoding="utf-8")
    assert source.count("gate=gate,") >= 3, (
        "start_cmd no longer hands the same gate to the AgentLoop, the subagent runner and the "
        "REPL — one of them is running with a different (or no) gate"
    )
    assert "gate.attach_channel(channel)" in source, (
        "start_cmd no longer points the gate at the channel — every ASK would fail closed"
    )


def test_the_repl_takes_a_gate_and_falls_back_to_the_loops():
    assert "gate" in inspect.signature(OrchestratorREPL.__init__).parameters
    repl = OrchestratorREPL(
        orchestrator=None, agent_loop=None, channel=None, bus=None, gate="explicit"
    )
    assert repl._session_gate() == "explicit"

    class _Loop:
        gate = "the loop's"

    repl = OrchestratorREPL(orchestrator=None, agent_loop=_Loop(), channel=None, bus=None)
    assert repl._session_gate() == "the loop's"


# ------------------------------------------------------------- the trust dialog

@pytest.fixture
def outside_workspace(tmp_path, monkeypatch, fake_home):
    """A discoverable workspace OUTSIDE any repository — the one trust-gated shape. Same setup
    as tests/unit/test_resolve_workspace_layer.py's `project` fixture: a hermetic $HOME (which
    is also where the trust store lives) and both config-dir env overrides cleared, or discovery
    never runs at all."""
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    fake_home(home)
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    # Stand two levels down, with no repository marker anywhere: the workspace is then "config
    # reaching in from outside the tree you opened", which is the only trust-gated case.
    deep = tmp_path / "proj" / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    from localharness.config.paths import workspace_is_within_repo

    assert not workspace_is_within_repo(ws, deep), "TMPDIR lives inside a repository"
    return ws.resolve()


def test_an_injected_asker_answers_the_trust_question(outside_workspace, monkeypatch):
    """PRD §3.5: the trust dialog stops being terminal-only — Zed is not a TTY."""
    monkeypatch.setattr(
        "localharness.cli.workspace._stdin_is_a_terminal", lambda: False, raising=False
    )

    seen: list[str] = []

    def _asker(question: str) -> bool:
        seen.append(question)
        return True

    got = resolve_workspace_layer(asker=_asker)
    assert got is not None, "an injected asker did not satisfy interactivity"
    assert got.resolve() == outside_workspace
    assert seen and "outside the project you are in" in seen[0]


def test_without_an_asker_a_non_tty_run_stays_inert(outside_workspace):
    """Phase-39 semantics, unchanged: nothing loaded AND nothing recorded."""
    from localharness.config import trust

    assert resolve_workspace_layer(interactive=False) is None
    assert trust.is_trusted(outside_workspace) is None, "a scripted run spent the answer"


def test_the_trust_question_is_one_named_string():
    """One string, and as of v0.14.1 it names BOTH halves of what a yes does — the config layer
    loads AND tools run without asking (owner ruling 2026-09-11, "trusted = load its config AND
    auto"). A question that named only the config half would be describing half the answer."""
    assert "{parent}" in TRUST_QUESTION
    assert "treat like code you are about to run" in TRUST_QUESTION
    assert "without asking" in TRUST_QUESTION
