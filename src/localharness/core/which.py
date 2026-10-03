"""Finding a program on PATH without ever taking it from the current directory by accident."""
from __future__ import annotations

import os
import shutil


def which_outside_cwd(cmd: str) -> str | None:
    """shutil.which, never answered from the current directory by accident. On POSIX an empty or
    "." PATH entry means the current directory, so those entries are skipped; every other entry —
    an absolute one that happens to be the directory you are in included — is searched as PATH
    says. Each entry is searched by joining it to the name: CPython 3.12's shutil.which puts the
    current directory first on Windows whenever the name has no directory part (even with an
    explicit `path=`), and a name WITH a directory part is looked up in that directory alone
    (PATHEXT still applied) — so a repository holding bash.exe or git.exe is never picked, and the
    real program is still found. A command that already has a directory part is resolved as given."""
    if os.path.dirname(cmd):
        return shutil.which(cmd)
    for entry in os.environ.get("PATH", os.defpath).split(os.pathsep):
        if entry in ("", os.curdir):
            continue
        found = shutil.which(os.path.join(entry, cmd))
        if found:
            return found
    return None
