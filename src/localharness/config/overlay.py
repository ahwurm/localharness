"""User overlay file: load, deep-merge, atomic write.

Sits above project YAML in the config cascade (defaults → project → user → experiment).
Phase 14 ships the user layer; Phase 17 adds the experiment layer (per git-isolated workspace).

Atomic write protocol: tempfile in SAME DIRECTORY as target + os.replace. Per Pitfall 2
in 14-RESEARCH.md, cross-filesystem tempfile defeats atomicity on container/NFS mounts.
"""
from __future__ import annotations

import errno
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional  # noqa: F401  (Any kept for downstream typing imports)

import yaml

from localharness.config.paths import resolve_overlay_path

log = logging.getLogger(__name__)


CONFIG_FILE_MODE = 0o600
"""Permission bits every config file this package writes must end up with: owner only.

`config.yaml` holds `provider.api_key`, and it and its agent/division siblings hold the policy a
session is gated by — the deny list, the ask rule sets, the kill file. `Path.write_text` creates a
file at 0666 masked by the process umask, which on the near-universal 022 gives 0664: every other
account on the machine could read the key, and read exactly which commands this operator's agents
are allowed to run. The overlay's atomic write was already 0600, but only as a side effect of
`NamedTemporaryFile`'s own hardening — an accident, not a stated guarantee, and one that
disappears the moment that write is spelled differently.

Owner-only rather than 0640: there is no group that should be reading a personal agent's API key,
and no writer here ever needs to share the file. `localharness doctor` reads it as the same user.
"""


def restrict_config_file(path: Path) -> None:
    """Set `path` to :data:`CONFIG_FILE_MODE`. Best effort — never raises.

    POSIX only. Windows has no POSIX mode bits (its `os.chmod` understands one flag, read-only),
    so applying this there would be theatre at best and a read-only config file at worst; access
    control on that platform is the ACL the directory already carries.

    Swallowing the error is deliberate and is the honest trade. This runs AFTER the bytes are on
    disk, so by the time a chmod can fail the write has already succeeded — turning a
    mode-tightening failure into a failed `localharness init` would mean a user with an exotic
    filesystem (some FUSE and network mounts reject chmod outright) cannot configure the tool at
    all, to protect a file that is already written correctly.
    """
    if os.name != "posix":
        return
    try:
        os.chmod(path, CONFIG_FILE_MODE)
    except OSError:
        log.debug("could not restrict %s to %o", path, CONFIG_FILE_MODE, exc_info=True)


def _resolve_user_overlay_path(config_dir: Optional[Path] = None) -> Path:
    """The user overlay lives at ``<resolved config_dir>/overrides.yaml`` (#35).

    ``config_dir`` precedence (via config/paths): explicit arg > LOCALHARNESS_DIR >
    LOCALHARNESS_HOME (legacy, set by tests/conftest.py `components_home`) > ~/.localharness.
    Callers that know their config dir (the loader, model_ops persist) MUST pass it so the
    overlay actually tracks ``--config-dir``; a None arg falls back to the env/default chain.
    """
    return resolve_overlay_path(config_dir)


# NOTE: module-level constant captured AT IMPORT TIME. Tests using monkeypatch.setenv
# AFTER import must call `_resolve_user_overlay_path()` directly instead of importing
# USER_OVERLAY_PATH. CLI code paths import this constant; tests resolve at call time.
USER_OVERLAY_PATH = _resolve_user_overlay_path()


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base. Overlay wins for scalars; dicts merge.

    Returns a NEW dict — does not mutate base. Strategy: REPLACE-scalars, RECURSE-dicts.
    On type mismatch (base scalar, overlay dict or vice versa), overlay wins outright.
    """
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_overlay(path: Path) -> dict:
    """Load YAML overlay file. Missing file → empty dict.

    A malformed overlay raises ConfigParseError — the SAME error type `_load_yaml_file` raises for
    config.yaml, so `validate_all`'s `except ConfigError` catches it and the user gets a clean
    report instead of a raw yaml.ParserError traceback. Phase 40 measured this crash on BOTH
    overlay layers and deferred it (deferred-items.md); phase 43 owns it because it is the same
    config-error-honesty spine F5 rebuilds.
    """
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        # Function-local import: loader.py imports THIS module at module level, so a
        # module-level back-import would be circular. Deferred to the raise path only.
        from localharness.config.loader import ConfigParseError

        from localharness.config.redact import yaml_problem, yaml_where

        # file, line and column, never the parser's snippet of the line (R16); not chained either
        line, column = yaml_where(e)
        raise ConfigParseError(str(path), line, column, yaml_problem(e)) from None
    return data or {}


def refuse_unwritable(path: Path) -> None:
    """Raise PermissionError when `path` (a symlink's target) exists and may not be written, or
    its directory may not be entered and written, as `write_text` would refuse it. Replacing a
    read-only file through its directory would quietly override a read-only bit the owner set, and
    a caller that keeps an undo (migrate's backup) asks this BEFORE making one, so a file that will
    be refused never leaves a copy behind."""
    path = Path(path).expanduser()
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    if target.exists() and not os.access(target, os.W_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(target))
    parent = target.parent
    if parent.exists() and not os.access(parent, os.W_OK | os.X_OK):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(parent))


def atomic_write_text(path: Path, text: str, *, refuse_read_only: bool = False) -> None:
    """Replace the file `path` names with `text`, atomically. The one writer every config file
    goes through (IN-21): a same-directory temp file (a cross-filesystem temp file would defeat
    `os.replace`'s atomicity on container/NFS mounts), fsynced, restricted to
    :data:`CONFIG_FILE_MODE` BEFORE the rename, then `os.replace`d over the target, and the
    directory fsynced after it (best effort). A crash, Ctrl-C or a full disk leaves the old file or
    the new one, never a truncated one, and never a stranded temp file.

    A symlinked file keeps its link: the file it points at is the one replaced. The parent
    directory is created when missing. `refuse_read_only` refuses a file (or directory) the caller
    may not write (:func:`refuse_unwritable`) rather than replacing it through the directory."""
    path = Path(path).expanduser()
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    if refuse_read_only:
        refuse_unwritable(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
        # Tightened BEFORE the rename, so the file at `path` is never briefly group-readable — and
        # stated rather than inherited: `NamedTemporaryFile` happens to create at 0600 today, which
        # is not a guarantee this module should be resting an api_key on.
        restrict_config_file(tmp_path)
        os.replace(str(tmp_path), str(target))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    _fsync_dir(target.parent)


def _fsync_dir(directory: Path) -> None:
    """fsync `directory` so a rename into it survives a power cut. POSIX only, best effort: the
    file itself is already whole, and some filesystems refuse a directory fsync."""
    if os.name != "posix":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        log.debug("could not fsync directory %s", directory, exc_info=True)
    finally:
        os.close(fd)


def atomic_write_overlay(path: Path, data: dict) -> None:
    """Write `data` as YAML through :func:`atomic_write_text`. POSIX + Windows compatible."""
    atomic_write_text(path, yaml.safe_dump(data, default_flow_style=False, sort_keys=False))
