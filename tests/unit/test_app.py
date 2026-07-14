"""Tests for the top-level `localharness` Typer app (version callback)."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from typer.testing import CliRunner

from localharness.cli.app import app

runner = CliRunner()


def test_version_flag_prints_installed_version():
    """A user's reflexive first command `localharness --version` must work, not error."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "localharness" in result.output.lower()
    assert version("localharness") in result.output


def test_version_flag_unknown_when_metadata_missing(monkeypatch):
    """From a raw checkout with no installed metadata, report 'unknown' rather than crash."""
    import localharness.cli.app as app_mod

    def _raise(_name):
        raise PackageNotFoundError("localharness")

    monkeypatch.setattr(app_mod, "_pkg_version", _raise)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "unknown" in result.output.lower()
