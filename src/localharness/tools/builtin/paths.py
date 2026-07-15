"""Shared user-path resolution for builtin file tools."""

import os
import tempfile
from pathlib import Path


def resolve_user_path(path: str) -> Path:
    """expanduser + resolve, with one Windows-only remap: POSIX-absolute `/tmp/...` goes to
    the OS temp dir instead of `<drive>\\tmp`.

    Models and scenario prompts speak POSIX `/tmp`. On Windows, pathlib anchors that at the
    current drive's root while git-bash (the bash_exec interpreter) mounts `/tmp` at %TEMP% —
    so the write tool and bash silently operated on two different trees (observed live: a
    file written to C:\\tmp\\... that `python3 /tmp/...` under bash could never find, three
    identical retry cycles, scenario green only because event counts don't check outcomes).
    Mapping `/tmp` to tempfile.gettempdir() puts every native tool in the one place bash
    already looks. POSIX behavior is unchanged.
    """
    if os.name == "nt" and (path == "/tmp" or path.startswith("/tmp/")):
        rest = path[len("/tmp"):].lstrip("/")
        base = Path(tempfile.gettempdir())
        path = str(base / rest) if rest else str(base)
    return Path(path).expanduser().resolve()
