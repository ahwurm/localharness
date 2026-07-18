"""Tests for the top-level `localharness` Typer app (version callback)."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError

from typer.testing import CliRunner

import localharness
from localharness.cli.app import app

runner = CliRunner()


def test_version_flag_prints_source_version():
    """A user's reflexive first command `localharness --version` must work, and report the
    in-source __version__ (the source of truth, #97) — not stale installed dist metadata."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "localharness" in result.output.lower()
    assert localharness.__version__ in result.output


def test_version_flag_uses_source_even_when_metadata_missing(monkeypatch):
    """A raw checkout / editable install with no (or stale) dist metadata still reports the real
    version from __version__ — never crashes, never degrades to 'unknown' (#97)."""

    def _raise(_name):
        raise PackageNotFoundError("localharness")

    monkeypatch.setattr("importlib.metadata.version", _raise)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert localharness.__version__ in result.output
