"""Tests for `localharness update` — PyPI check, install-method detection, and the
source-install guard that keeps a published wheel from shadowing a developer's checkout."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli import update_cmd
from localharness.cli.app import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# _latest_pypi_version — an offline box gets a clear message, never a traceback
# ---------------------------------------------------------------------------

def test_latest_version_returns_none_when_pypi_unreachable(monkeypatch):
    def _boom(*_a, **_kw):
        raise OSError("no route to host")

    monkeypatch.setattr(update_cmd.urllib.request, "urlopen", _boom)
    assert update_cmd._latest_pypi_version() is None


def test_update_exits_nonzero_when_pypi_unreachable(monkeypatch):
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: None)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    assert "could not reach PyPI" in result.output


# ---------------------------------------------------------------------------
# Install-method detection
# ---------------------------------------------------------------------------

def test_upgrade_command_uses_uv_for_a_uv_tool_install(monkeypatch, tmp_path):
    uv_prefix = tmp_path / "uv" / "tools" / "localharness"
    uv_prefix.mkdir(parents=True)
    monkeypatch.setattr(update_cmd.sys, "prefix", str(uv_prefix))
    monkeypatch.setattr(update_cmd.shutil, "which", lambda _n: "/usr/bin/uv")
    cmd = update_cmd._upgrade_command()
    assert cmd == ["/usr/bin/uv", "tool", "upgrade", "localharness"]


def test_upgrade_command_returns_none_when_uv_install_but_uv_missing(monkeypatch, tmp_path):
    """Better to say 'uv is not on PATH' than to let pip corrupt a uv-managed env."""
    uv_prefix = tmp_path / "uv" / "tools" / "localharness"
    uv_prefix.mkdir(parents=True)
    monkeypatch.setattr(update_cmd.sys, "prefix", str(uv_prefix))
    monkeypatch.setattr(update_cmd.shutil, "which", lambda _n: None)
    assert update_cmd._upgrade_command() is None


def test_upgrade_command_falls_back_to_pip(monkeypatch, tmp_path):
    venv = tmp_path / "some" / "venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(update_cmd.sys, "prefix", str(venv))
    cmd = update_cmd._upgrade_command()
    assert cmd[1:] == ["-m", "pip", "install", "--upgrade", "localharness"]


# ---------------------------------------------------------------------------
# Source-install guard — the load-bearing safety property
# ---------------------------------------------------------------------------

def test_this_checkout_is_detected_as_a_source_install():
    """The repo under test is a checkout, not a wheel — the guard must see that."""
    assert update_cmd._is_source_install() is True


def test_a_site_packages_install_is_not_a_source_install(monkeypatch, tmp_path):
    installed = tmp_path / "lib" / "python3.13" / "site-packages" / "localharness"
    installed.mkdir(parents=True)
    monkeypatch.setattr(update_cmd.localharness, "__file__", str(installed / "__init__.py"))
    assert update_cmd._is_source_install() is False


def test_update_refuses_to_pip_over_a_source_checkout(monkeypatch):
    """A checkout is ahead of PyPI as often as behind it. Upgrading it with pip would
    shadow the working tree with a published wheel and discard uncommitted work."""
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "99.0.0")
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: True)
    called = []
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **k: called.append(a))

    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "git pull" in result.output
    assert called == [], "must not shell out to an installer for a source install"


# ---------------------------------------------------------------------------
# Up-to-date / newer paths
# ---------------------------------------------------------------------------

def test_update_reports_up_to_date_without_running_anything(monkeypatch):
    monkeypatch.setattr(update_cmd, "resolved_version", lambda: "0.12.8")
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "0.12.8")
    called = []
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **k: called.append(a))

    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "up to date" in result.output
    assert called == []


def test_update_does_not_downgrade_when_local_is_ahead_of_pypi(monkeypatch):
    """A version ahead of PyPI (a pre-release build) is not an 'update available'."""
    monkeypatch.setattr(update_cmd, "resolved_version", lambda: "0.13.0")
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "0.12.8")
    called = []
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **k: called.append(a))

    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0
    assert "up to date" in result.output
    assert called == []


def test_check_flag_reports_but_does_not_upgrade(monkeypatch):
    monkeypatch.setattr(update_cmd, "resolved_version", lambda: "0.12.7")
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "0.12.8")
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: False)
    called = []
    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **k: called.append(a))

    result = runner.invoke(app, ["update", "--check"])
    assert result.exit_code == 0
    assert "0.12.8" in result.output
    assert called == [], "--check must never mutate the install"


def test_update_runs_the_installer_and_reports_failure(monkeypatch):
    monkeypatch.setattr(update_cmd, "resolved_version", lambda: "0.12.7")
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "0.12.8")
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: False)
    monkeypatch.setattr(update_cmd, "_upgrade_command", lambda: ["true"])

    class _Fail:
        returncode = 2

    monkeypatch.setattr(update_cmd.subprocess, "run", lambda *a, **k: _Fail())
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 2
    assert "upgrade command failed" in result.output
