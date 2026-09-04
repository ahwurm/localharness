"""localharness update command — upgrade an installed LocalHarness to the latest PyPI release.

The friction this removes: a released fix is useless until the machine that hits the bug is
actually running it, and the honest upgrade line (`uv tool install --force git+https://...`)
is long enough that people don't run it. `update` is the short, memorable form.

Deliberately NOT a self-mutating in-process upgrade: it detects how this copy was installed
and shells out to that installer. Anything cleverer (patching a running interpreter's own
site-packages) is how you get a half-upgraded install.

Windows is the one platform where that shell-out cannot finish while this process lives: the
file the installer replaces at the end is the running `localharness.exe` itself, and Windows
locks a running executable (#156). There the upgrade is handed to a detached process instead
and this one returns immediately.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

import localharness
from localharness import resolved_version

console = Console()
err_console = Console(stderr=True)

PYPI_JSON_URL = "https://pypi.org/pypi/localharness/json"
_TIMEOUT_SECONDS = 10.0

# Windows process-creation flags (winbase.h, via CPython's `subprocess` docs). Named here with
# their documented values as the fallback because `subprocess` only defines these attributes on
# Windows — this module has to import, and be testable, on every platform.
_WIN_DETACHED_PROCESS = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
_WIN_CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)

# The two halves of "only the shim was locked" (#156). uv prints the install line when the package
# itself went in, and the copy of the entrypoint fails with the Win32 sharing violation, whose
# message and error number are both stable enough to match on. BOTH are required: either alone is
# an ordinary failure, and forgiving an ordinary failure would be the worse bug.
_UPGRADED_MARKERS = ("+ localharness==", "updated localharness", "installed localharness")
_FILE_IN_USE_MARKERS = ("os error 32", "being used by another process")


def _latest_pypi_version(timeout: float = _TIMEOUT_SECONDS) -> str | None:
    """Latest published version, or None if PyPI is unreachable/unparseable.

    Returns None rather than raising: an offline box should get a clear "couldn't check",
    never a traceback.
    """
    try:
        with urllib.request.urlopen(PYPI_JSON_URL, timeout=timeout) as resp:
            return json.load(resp)["info"]["version"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
        return None


def _is_source_install() -> bool:
    """True when running from a git checkout / editable install rather than a wheel.

    Load-bearing guard: `pip install --upgrade` over a developer's editable checkout would
    shadow their working tree with a published wheel and silently discard uncommitted work.
    A source install upgrades with `git pull`, not with this command.
    """
    pkg_dir = Path(localharness.__file__).resolve().parent
    return not any(part in {"site-packages", "dist-packages"} for part in pkg_dir.parts)


def _upgrade_command() -> list[str] | None:
    """The installer command matching how THIS copy was installed, or None if undetectable.

    A `uv tool` install lives in a uv-managed environment that pip must not touch, so it is
    detected first and upgraded with uv. Everything else is a normal pip-managed environment.
    """
    prefix = str(Path(sys.prefix).resolve()).replace("\\", "/")
    if "/uv/tools/" in f"{prefix}/":
        uv = shutil.which("uv")
        return [uv, "tool", "upgrade", "localharness"] if uv else None
    return [sys.executable, "-m", "pip", "install", "--upgrade", "localharness"]


def _is_windows() -> bool:
    """Read at call time, never at import: the tests fake the platform, and a module-level
    constant would freeze whichever machine built the wheel."""
    return sys.platform.startswith("win")


def _spawn_detached(cmd: list[str]) -> bool:
    """Start the upgrade in a process that outlives this one. True when it started.

    The fix for #156. Windows locks a running executable, and the file uv has to replace at the
    end of the upgrade IS the `localharness.exe` performing it — so the copy can only succeed
    after this process is gone. DETACHED_PROCESS gives the child no console to be killed with,
    CREATE_NEW_PROCESS_GROUP keeps a Ctrl-C in this shell from reaching it, and `update` returns
    within milliseconds while uv is still resolving — long before it reaches the shim.

    False, rather than a traceback, when the spawn is refused: the caller falls back to running
    the upgrade here and reporting what happened.
    """
    try:
        subprocess.Popen(  # noqa: S603 - cmd is built by _upgrade_command, never user input
            cmd,
            close_fds=True,
            creationflags=_WIN_DETACHED_PROCESS | _WIN_CREATE_NEW_PROCESS_GROUP,
        )
    except OSError:
        return False
    return True


def _only_the_shim_was_locked(output: str) -> bool:
    """Did the package upgrade and only the entrypoint copy fail on a file in use?

    That is a success with a note, not the bare exit 1 this used to be: the new version is
    installed in the tool environment, and the `localharness` command catches up the moment
    something replaces the shim. Both signals required — see the marker constants.
    """
    lowered = output.lower()
    return any(m in lowered for m in _UPGRADED_MARKERS) and any(
        m in lowered for m in _FILE_IN_USE_MARKERS
    )


def _run_captured(cmd: list[str]) -> tuple[int, str]:
    """Run the upgrade here, keeping its output so the verdict above can be reached.

    Only the Windows fallback path uses this. Capturing costs the live progress display, which is
    why the ordinary path does not: on POSIX there is no locked shim to diagnose.
    """
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return result.returncode, f"{result.stdout or ''}{result.stderr or ''}"


def update(
    check: Annotated[
        bool,
        typer.Option("--check", help="Report whether an update is available; change nothing."),
    ] = False,
) -> None:
    """Upgrade LocalHarness to the latest release on PyPI."""
    current = resolved_version()
    latest = _latest_pypi_version()

    if latest is None:
        err_console.print(
            "[bold red]Error:[/bold red] could not reach PyPI to check for updates "
            f"({PYPI_JSON_URL}). Check your connection, or upgrade manually."
        )
        raise typer.Exit(1)

    try:
        from packaging.version import InvalidVersion, Version

        try:
            newer = Version(latest) > Version(current)
        except InvalidVersion:
            newer = latest != current
    except ImportError:  # pragma: no cover - packaging ships with pip-installed envs
        newer = latest != current

    if not newer:
        console.print(f"LocalHarness [bold]{current}[/bold] is up to date (PyPI: {latest}).")
        return

    console.print(f"Update available: [bold]{current}[/bold] → [bold green]{latest}[/bold green]")

    if _is_source_install():
        # A checkout is ahead of PyPI as often as behind it; never pip over it.
        console.print(
            "This is a source/editable install — upgrade it with [bold]git pull[/bold] "
            "in the repo, not with `localharness update`."
        )
        return

    if check:
        return

    cmd = _upgrade_command()
    if cmd is None:
        err_console.print(
            "[bold red]Error:[/bold red] this looks like a `uv tool` install but `uv` is not on "
            "PATH. Install uv, or upgrade manually with `uv tool upgrade localharness`."
        )
        raise typer.Exit(1)

    console.print(f"Running: [dim]{' '.join(cmd)}[/dim]")

    if _is_windows() and _spawn_detached(cmd):
        console.print(
            f"Upgrading to [bold green]{latest}[/bold green] in the background — the "
            "`localharness` command will be updated when this process exits (Windows cannot "
            "replace a running program). Run `localharness --version` to confirm."
        )
        return

    if _is_windows():
        # The spawn was refused, so the upgrade runs here after all — and hits the locked shim
        # this command was trying to get out of the way of. Captured, so the two outcomes can be
        # told apart, and echoed, so nothing uv said is swallowed.
        returncode, output = _run_captured(cmd)
        if output.strip():
            console.print(output.rstrip(), markup=False, soft_wrap=True)
        if returncode != 0 and _only_the_shim_was_locked(output):
            console.print(
                f"[green]✓[/green] LocalHarness {latest} is installed, but the `localharness` "
                "command could not be replaced while it is running. Close this shell and run "
                "`uv tool upgrade localharness` once to finish, then `localharness --version`."
            )
            return
    else:
        returncode = subprocess.run(cmd, check=False).returncode

    if returncode != 0:
        err_console.print(
            f"[bold red]Error:[/bold red] upgrade command failed (exit {returncode}). "
            "Run it by hand to see the full output."
        )
        raise typer.Exit(returncode)
    console.print(f"[green]✓[/green] Upgraded to {latest}. Run `localharness doctor` to verify.")
