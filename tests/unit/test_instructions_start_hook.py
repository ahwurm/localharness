"""Phase 11-09 — the workspace-open hook in `start`.

One handle per session, built from the session's effective boundary with the basis the trust state
calls for, handed to the root loop, the children, the REPL and ACP, its receipt bound onto the
channel and its dim line riding the terminal's first hint. The drive tests run the REAL
`_start_async` offline with the recorders from test_workspace_state_landing.py; the unit tests at the
top call the module helpers directly.
"""
from __future__ import annotations

from pathlib import Path

from localharness.cli import start_cmd
from localharness.config import trust
from localharness.config.project_records import read_instructions
from tests.unit.test_workspace_state_landing import (
    _drive,
    _install_recorders,
    _one,
    _workspace_start,
)

CLAUDE = "# Rules\nUse tabs.\n"


def _project(ws: Path) -> Path:
    return ws.parent


def _put_instructions(proj: Path, text: str = CLAUDE) -> Path:
    path = proj / "CLAUDE.md"
    path.write_text(text, encoding="utf-8")
    return path


def _handle_of(rec) -> object:
    return _one(rec, "loop")["project_instructions"]


# ------------------------------------------------------------------ _open_project_instructions


def _open(tmp_path, monkeypatch, fake_home, *, boundary="proj", **over):
    """Call the helper against a real tmp project; returns (handle_or_None, seen_kwargs)."""
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    _put_instructions(proj)
    seen: dict = {}

    def _fake(root, **kw):
        seen["root"] = root
        seen.update(kw)
        return "HANDLE"

    monkeypatch.setattr("localharness.config.project_records.open_project_instructions", _fake)
    args = dict(
        workspace=proj / ".localharness", decision=True, trust_flag=False, sitting_id="s1",
        max_ratio=0.25, no_input=False, channel_mode="terminal", window_tokens=65536,
        config_dir=None,
    )
    args.update(over)
    return start_cmd._open_project_instructions(proj if boundary == "proj" else boundary, **args), seen


def test_no_boundary_builds_nothing(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, boundary=None)
    assert got is None and not seen


def test_a_declined_root_builds_nothing(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, decision=False)
    assert got is None and not seen


def test_basis_layer_when_trusted_with_a_layer(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, decision=True)
    assert got == "HANDLE" and seen["basis"] == "layer"
    assert seen["sitting_id"] == "s1" and seen["max_ratio"] == 0.25
    assert seen["window_tokens"] == 65536


def test_basis_trusted_when_trusted_without_a_layer(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=True, workspace=None)
    assert seen["basis"] == "trusted"


def test_basis_trusted_under_the_trust_flag_with_no_record(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=None, trust_flag=True)
    assert seen["basis"] == "trusted"


def test_basis_awaiting_trust_with_no_record_and_no_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=None, trust_flag=False)
    assert seen["basis"] == "awaiting_trust"


def test_acp_internal_no_input_is_not_the_cli_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, channel_mode="acp", no_input=True)
    assert seen["no_input"] is False


def test_terminal_no_input_is_the_cli_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, channel_mode="terminal", no_input=True)
    assert seen["no_input"] is True


def test_a_broken_store_returns_none(tmp_path, monkeypatch, fake_home):
    proj = tmp_path / "proj"
    proj.mkdir()

    def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr("localharness.config.project_records.open_project_instructions", _boom)
    got = start_cmd._open_project_instructions(
        proj, workspace=None, decision=True, trust_flag=False, sitting_id="s", max_ratio=0.25,
        no_input=False, channel_mode="terminal", window_tokens=1000, config_dir=None,
    )
    assert got is None


def test_hint_with_instructions_helper():
    line = "Using CLAUDE.md (1 KB); /instructions off"
    assert start_cmd._hint_with_instructions("", line) == line
    assert start_cmd._hint_with_instructions("/help for commands.", line) == line
    assert start_cmd._hint_with_instructions("Describe a task, or /help for commands.", line) == (
        "Describe a task, or /help for commands. · " + line
    )


# ------------------------------------------------------------------ the drive: scope and basis


async def test_a_trusted_project_with_claude_md_hands_one_handle_to_the_loop(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj)
    trust.record_trust(proj, True)
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None
    assert handle.candidates[0] == proj.resolve() / "CLAUDE.md"
    assert handle._basis == "layer"


async def test_no_instructions_file_means_no_handle(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(_project(ws), True)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert _handle_of(rec) is None


async def test_a_declined_root_gets_no_handle_anywhere(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj)
    trust.record_trust(proj, False)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert _handle_of(rec) is None


async def test_no_record_and_no_flag_is_awaiting_trust(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _put_instructions(_project(ws))
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None and handle._basis == "awaiting_trust"


async def test_the_trust_flag_with_no_record_is_trusted(tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _put_instructions(_project(ws))
    rec = _install_recorders(monkeypatch)

    await _start_async(None, False, False, None, trust_project=True)

    handle = _handle_of(rec)
    assert handle is not None and handle._basis == "trusted"


def test_the_store_is_untouched_by_building_the_handle(tmp_path, monkeypatch, fake_home):
    """read_instructions is importable from here and a fresh root has no record."""
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    assert read_instructions(_project(ws)) is None
