"""One polite handler for the filesystem going wrong underneath a read-only command.

`doctor` has always had this — a broad `except` around the config load, so the command people run
when things are already broken prints a message instead of a traceback. `config show`, `validate`,
`init --workspace` and `agent create` did not, and the bad-mood review's persona E found the whole
family through them: a `.localharness` that is a FILE, one that is a dangling symlink, one that is
a symlink pointing at itself, a config.yaml that is a directory, one holding bytes that are not
UTF-8, an unreadable one, a YAML alias bomb, and a working directory deleted out from under the
process. Every one of them ended in a rich traceback.

None of these are exotic. They are what a half-finished `mv`, a stale symlink from a moved repo, a
`git checkout` of a binary file, a root-owned config and a deleted worktree look like from inside
the process — and the user's next move is always the same: find out WHICH path is broken. So that
is what this prints, along with the reason, and nothing else.

Deliberately NOT a decorator: typer reads a command's signature to build its options, and a
wrapper that has to be transparent to `inspect.signature` is a failure mode of its own. A `with`
block at the top of each command body says exactly as much and hides nothing.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Sequence

import typer
from rich.console import Console
from rich.markup import escape

# Everything the E cluster actually produced. All but two are OSError subclasses — IsADirectory,
# NotADirectory, FileNotFound, FileExists, Permission, and the bare OSError that ELOOP arrives as
# — so listing them individually would be a list that goes stale the first time the kernel returns
# a code nobody thought of. UnicodeDecodeError (bytes that are not text) is a ValueError, and
# RecursionError (an alias bomb, or a symlink chain resolved in Python) is a RuntimeError.
_HANDLED = (OSError, UnicodeDecodeError, RecursionError)


def _describe(exc: BaseException) -> str:
    """The reason, in the words the user needs, without the traceback."""
    if isinstance(exc, UnicodeDecodeError):
        return f"the file is not valid UTF-8 text ({exc.reason}, at byte {exc.start})"
    if isinstance(exc, RecursionError):
        return "the file nests too deeply to load (a YAML alias chain does this)"
    if isinstance(exc, OSError):
        detail = exc.strerror or str(exc)
        return f"{detail}: {exc.filename}" if exc.filename else detail
    return str(exc)


def report_filesystem_error(
    exc: BaseException,
    action: str,
    *,
    console: Console,
    paths: Optional[Sequence[Optional[Path]]] = None,
    exit_code: int = 1,
) -> None:
    """One message naming the reason and the files in play, then exit. Never returns.

    `action` completes "could not ..." — e.g. "read your configuration". `paths` are the files this
    command was working with, printed because the exception often cannot say which one it choked
    on: a UnicodeDecodeError carries no filename at all, and neither does a RecursionError.
    """
    console.print(
        "[bold red]Error:[/bold red] " + escape(f"could not {action} — {_describe(exc)}"),
        soft_wrap=True,
    )
    for path in paths or ():
        if path is not None:
            console.print(escape(f"       in play: {path}"), soft_wrap=True)
    console.print("       Run 'localharness doctor' for a full check of these files.")
    raise typer.Exit(code=exit_code)


@contextmanager
def polite_filesystem_errors(
    action: str,
    *,
    console: Console,
    paths: Optional[Sequence[Optional[Path]]] = None,
    exit_code: int = 1,
) -> Iterator[None]:
    """`report_filesystem_error` as a `with` block, for commands whose whole body is at risk."""
    try:
        yield
    except _HANDLED as exc:
        report_filesystem_error(
            exc, action, console=console, paths=paths, exit_code=exit_code
        )
