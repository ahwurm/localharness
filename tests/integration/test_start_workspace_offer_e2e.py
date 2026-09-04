"""`start`'s workspace offer, driven through the REAL `_start_async` (owner ruling 2026-09-04).

The unit tests in `tests/unit/test_start_offers_workspace.py` grade the offer's guards. This file
grades the only claim they cannot: that a "yes" produces a layer THIS session is already using —
created and active in one command, not created with an instruction to start again.

Offline by construction, like every other start drive here: only the external boundaries are
stubbed (`_stub_start_boundaries`: the LLM probe, the tokenizer, the REPL loop, plugin discovery)
and the provider is pointed at the discard port. No model is started and no socket is opened.
Helpers are imported, never copied (41-06) — a copied drive grades its own copy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.cli import workspace as ws_mod
from localharness.config.paths import WORKSPACE_DIR_NAME

from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _stub_start_boundaries
from tests.unit.test_workspace_state_landing import _drive, _hermetic

pytestmark = pytest.mark.asyncio


def _project_without_a_workspace(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """A configured machine and a project that has no `.localharness/` yet. Returns
    `(global_dir, project_root)` with the process standing in the project."""
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, home)
    _stub_start_boundaries(global_dir, monkeypatch)  # writes the GLOBAL config.yaml
    _offline_provider(global_dir)

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    return global_dir, proj


async def test_yes_creates_the_workspace_and_the_session_is_already_using_it(
    tmp_path, monkeypatch, capsys
):
    """One command: the directory appears and the session that made it runs layered on it."""
    _global_dir, proj = _project_without_a_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: True)

    await _drive()

    ws = proj / WORKSPACE_DIR_NAME
    assert ws.is_dir() and (ws / "config.yaml").is_file()
    out = capsys.readouterr().out
    assert f"Workspace layer: {ws}" in out, (
        "start printed no layer line for the workspace it had just created — it was not active "
        f"for this session. Output was:\n{out}"
    )


async def test_no_leaves_the_session_on_the_global_layer(tmp_path, monkeypatch, capsys):
    """Declining is not a failure: startup continues, with nothing created."""
    _global_dir, proj = _project_without_a_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **k: False)

    await _drive()

    assert not (proj / WORKSPACE_DIR_NAME).exists()
    assert "Workspace layer:" not in capsys.readouterr().out


async def test_a_scripted_start_is_never_offered_anything(tmp_path, monkeypatch):
    """No terminal, no question — `start` runs from hooks and wrappers all day."""
    _global_dir, proj = _project_without_a_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: False)

    def _boom(*_a, **_kw):
        raise AssertionError("prompted a run with no terminal")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    await _drive()

    assert not (proj / WORKSPACE_DIR_NAME).exists()


async def test_no_input_declines_for_the_whole_command(tmp_path, monkeypatch):
    """`start --no-input` with a terminal attached: the flag, not the tty, decides.

    Driven with the flag rather than through `_drive`, because the flag IS the claim — the
    keyword is what start_app passes and what must reach the offer.
    """
    from localharness.cli.start_cmd import _start_async

    _global_dir, proj = _project_without_a_workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)

    def _boom(*_a, **_kw):
        raise AssertionError("prompted a --no-input run")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    await _start_async(None, False, False, None, no_input=True)

    assert not (proj / WORKSPACE_DIR_NAME).exists()
