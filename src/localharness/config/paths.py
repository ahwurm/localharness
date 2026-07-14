"""ONE place that answers "where is the config dir" and resolves config-dir-relative paths.

Before this module (#35) the answer was scattered and inconsistent: the overlay keyed only on
``LOCALHARNESS_HOME``/``~/.localharness`` while every CLI ``--config-dir`` flag binds the
``LOCALHARNESS_DIR`` env var — so ``--config-dir`` silently failed to move the overlay, and the
kill-file/audit-log/repl-history defaults were hardcoded under ``~/.localharness`` regardless.

Precedence (one chain, everywhere): explicit arg > ``LOCALHARNESS_DIR`` (canonical) >
``LOCALHARNESS_HOME`` (legacy alias, still honored for hermetic tests + the
components/autoresearch archive helpers) > ``~/.localharness``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

_DEFAULT_CONFIG_DIR = "~/.localharness"

# Accept str or Path so callers can pass a raw flag value or an already-resolved Path.
PathLike = Union[str, Path]


def config_dir_env_override() -> Optional[str]:
    """The config-dir env override: ``LOCALHARNESS_DIR`` (canonical), else ``LOCALHARNESS_HOME``
    (legacy alias). Returns None when neither is set. Kept as its own helper so the archive-db
    resolvers (which have a different *default*) can honor the same env precedence."""
    return os.environ.get("LOCALHARNESS_DIR") or os.environ.get("LOCALHARNESS_HOME")


def resolve_config_dir(config_dir: Optional[PathLike] = None) -> Path:
    """Resolve the active config directory. Precedence: explicit arg > LOCALHARNESS_DIR >
    LOCALHARNESS_HOME (legacy) > ~/.localharness. Always ``expanduser``'d."""
    chosen = config_dir or config_dir_env_override() or _DEFAULT_CONFIG_DIR
    return Path(chosen).expanduser()


def resolve_overlay_path(config_dir: Optional[PathLike] = None) -> Path:
    """The user overlay lives at ``<resolved config_dir>/overrides.yaml``."""
    return resolve_config_dir(config_dir) / "overrides.yaml"


def resolve_runtime_path(value: str, config_dir: Optional[PathLike] = None) -> Path:
    """Resolve a config-*value* path (kill_file, audit_log_path, repl history).

    Absolute or ``~``-prefixed values are honored as-is (never re-rooted); a bare relative name
    resolves UNDER the config dir. Back-compat invariant: the default value ``KILL`` under the
    default config dir ``~/.localharness`` lands at ``~/.localharness/KILL`` — exactly where the
    old hardcoded default pointed.
    """
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return resolve_config_dir(config_dir) / value
