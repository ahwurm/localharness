"""localharness update command — upgrade an installed LocalHarness to the latest PyPI release.

The friction this removes: a released fix is useless until the machine that hits the bug is
actually running it, and the honest upgrade line (`uv tool install --force git+https://...`)
is long enough that people don't run it. `update` is the short, memorable form.

Deliberately NOT a self-mutating in-process upgrade: it detects how this copy was installed
and shells out to that installer. Anything cleverer (patching a running interpreter's own
site-packages) is how you get a half-upgraded install.
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
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        err_console.print(
            f"[bold red]Error:[/bold red] upgrade command failed (exit {result.returncode}). "
            "Run it by hand to see the full output."
        )
        raise typer.Exit(result.returncode)
    console.print(f"[green]✓[/green] Upgraded to {latest}. Run `localharness doctor` to verify.")
