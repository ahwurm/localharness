"""`doctor` in a workspace made before workspaces shipped a .gitignore: one `i` line, nothing written.

The hint is advice, not a failure: doctor's exit code is the same with and without the file (the
fixture's provider is unreachable, so both runs exit on that same core failure).
"""
from __future__ import annotations

from typer.testing import CliRunner

from localharness.cli.app import app
from tests.unit.test_doctor_layer_report import _layout, _write_workspace

runner = CliRunner()
HINT = "No .gitignore in this workspace"


def _doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    return result


def test_a_workspace_without_gitignore_gets_one_line_and_no_file(tmp_path, monkeypatch, fake_home):
    layout = _layout(tmp_path, monkeypatch, fake_home)
    _write_workspace(layout, {"org": {"name": "WS"}})

    result = _doctor()

    assert result.stdout.count(HINT) == 1, result.stdout
    line = next(ln for ln in result.stdout.splitlines() if HINT in ln)
    assert line.startswith("i"), line
    assert not (layout.ws_dir / ".gitignore").exists(), "doctor wrote a .gitignore"


def test_a_workspace_with_gitignore_gets_no_line_and_the_same_exit(tmp_path, monkeypatch, fake_home):
    layout = _layout(tmp_path, monkeypatch, fake_home)
    _write_workspace(layout, {"org": {"name": "WS"}})
    without = _doctor().exit_code
    (layout.ws_dir / ".gitignore").write_text("*.db\n", encoding="utf-8")

    result = _doctor()

    assert HINT not in result.stdout, result.stdout
    assert result.exit_code == without
