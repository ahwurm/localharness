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
    monkeypatch.setattr(update_cmd.sys, "platform", "linux")  # the detached path is Windows-only
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


# ---------------------------------------------------------------------------
# Windows: the running localharness.exe is the file uv has to replace (#156)
#
# On Windows a running executable is locked, so `uv tool upgrade` installs the new package fine
# and then fails to copy the entrypoint shim over itself — os error 32. `update` reported that as
# a bare failure on a machine that HAD just upgraded. The fix is to hand the upgrade to a process
# that outlives this one; the captured fallback exists for when that spawn is not available, and
# is the only path that can tell "only the shim was locked" from a real failure.
#
# Every test here fakes the platform and mocks the process API: no upgrade is ever run.
# ---------------------------------------------------------------------------

def _pending_upgrade(monkeypatch, platform: str) -> None:
    monkeypatch.setattr(update_cmd.sys, "platform", platform)
    monkeypatch.setattr(update_cmd, "resolved_version", lambda: "0.13.0")
    monkeypatch.setattr(update_cmd, "_latest_pypi_version", lambda *a, **k: "0.13.1")
    monkeypatch.setattr(update_cmd, "_is_source_install", lambda: False)
    monkeypatch.setattr(update_cmd, "_upgrade_command", lambda: ["uv", "tool", "upgrade", "localharness"])


def _spy_popen(monkeypatch, *, fails: bool = False) -> list:
    calls = []

    def _popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if fails:
            raise OSError("detached spawn unavailable")
        return object()

    monkeypatch.setattr(update_cmd.subprocess, "Popen", _popen)
    return calls


def _spy_run(monkeypatch, *, returncode: int = 0, stdout: str = "", stderr: str = "") -> list:
    calls = []

    class _Result:
        pass

    def _run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        result = _Result()
        result.returncode = returncode
        result.stdout = stdout
        result.stderr = stderr
        return result

    monkeypatch.setattr(update_cmd.subprocess, "run", _run)
    return calls


UV_SHIM_LOCKED = (
    "Resolved 1 package in 12ms\n"
    "Installed 1 package in 30ms\n"
    " + localharness==0.13.1\n"
    "error: Failed to install entrypoints\n"
    "  Caused by: failed to copy file from "
    "C:\\Users\\a\\AppData\\Roaming\\uv\\tools\\localharness\\Scripts\\localharness.exe to "
    "C:\\Users\\a\\.local\\bin\\localharness.exe\n"
    "  Caused by: The process cannot access the file because it is being used by another "
    "process. (os error 32)\n"
)


def test_windows_hands_the_upgrade_to_a_detached_process(monkeypatch):
    """The whole point: the copy must happen after this exe stops running."""
    _pending_upgrade(monkeypatch, "win32")
    spawned = _spy_popen(monkeypatch)
    ran = _spy_run(monkeypatch)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert len(spawned) == 1, "the upgrade must be spawned, not run in this process"
    assert spawned[0][1]["creationflags"] == (
        update_cmd._WIN_DETACHED_PROCESS | update_cmd._WIN_CREATE_NEW_PROCESS_GROUP
    )
    assert ran == [], "a detached spawn must not also run the upgrade inline"
    assert "background" in result.output
    assert "localharness --version" in result.output


def test_non_windows_still_runs_the_upgrade_in_this_process(monkeypatch):
    """POSIX never locks a running file — that path must not change."""
    _pending_upgrade(monkeypatch, "linux")
    spawned = _spy_popen(monkeypatch)
    ran = _spy_run(monkeypatch, returncode=0)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert spawned == [], "detaching is a Windows workaround, not the default"
    assert len(ran) == 1
    assert "capture_output" not in ran[0][1], "uv's progress must keep streaming to the terminal"
    assert "Upgraded to 0.13.1" in result.output


def test_windows_falls_back_to_a_captured_run_and_forgives_a_locked_shim(monkeypatch):
    """Package in, shim not replaced: that is a success with a note, not exit 1 (#156)."""
    _pending_upgrade(monkeypatch, "win32")
    _spy_popen(monkeypatch, fails=True)
    ran = _spy_run(monkeypatch, returncode=1, stderr=UV_SHIM_LOCKED)

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 0
    assert len(ran) == 1 and ran[0][1].get("capture_output") is True
    assert "0.13.1" in result.output
    assert "os error 32" in result.output, "the real uv output must still be shown"


def test_windows_reports_a_genuine_upgrade_failure(monkeypatch):
    """Forgiveness is narrow: no install line means the upgrade really did fail."""
    _pending_upgrade(monkeypatch, "win32")
    _spy_popen(monkeypatch, fails=True)
    _spy_run(monkeypatch, returncode=2, stderr="error: network unreachable\n")

    result = runner.invoke(app, ["update"])

    assert result.exit_code == 2
    assert "upgrade command failed" in result.output


@pytest.mark.parametrize(
    "output, forgiven",
    [
        (UV_SHIM_LOCKED, True),
        (" + localharness==0.13.1\nerror: something else entirely\n", False),
        ("error: ... it is being used by another process. (os error 32)\n", False),
        ("", False),
    ],
    ids=["both-signals", "installed-but-other-error", "locked-but-never-installed", "no-output"],
)
def test_the_locked_shim_verdict_needs_both_signals(output, forgiven):
    """Two signals, both required: the package went in AND the only thing that failed was the
    copy of a file that was in use. One of them alone is an ordinary failure."""
    assert update_cmd._only_the_shim_was_locked(output) is forgiven
