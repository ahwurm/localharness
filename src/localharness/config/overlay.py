"""User overlay file: load, deep-merge, atomic write.

Sits above project YAML in the config cascade (defaults → project → user → experiment).
Phase 14 ships the user layer; Phase 17 adds the experiment layer (per git-isolated workspace).

Atomic write protocol: tempfile in SAME DIRECTORY as target + os.replace. Per Pitfall 2
in 14-RESEARCH.md, cross-filesystem tempfile defeats atomicity on container/NFS mounts.
"""
from __future__ import annotations

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

        mark = getattr(e, "problem_mark", None)
        line = (mark.line + 1) if mark else 0
        column = (mark.column + 1) if mark else 0
        raise ConfigParseError(str(path), line, column, str(e)) from e
    return data or {}


def atomic_write_overlay(path: Path, data: dict) -> None:
    """Write YAML atomically. Tempfile in same dir → fsync → os.replace.

    POSIX + Windows compatible. NamedTemporaryFile(dir=path.parent) is required
    to keep os.replace atomic across all filesystems.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)

    # Same-dir tempfile keeps os.replace atomic across filesystems
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(yaml_text)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    # Tightened BEFORE the rename, so the file at `path` is never briefly group-readable — and
    # stated rather than inherited: `NamedTemporaryFile` happens to create at 0600 today, which
    # is not a guarantee this module should be resting an api_key on.
    restrict_config_file(Path(tmp_path))

    try:
        os.replace(tmp_path, str(path))
    except Exception:
        # Best-effort cleanup of stranded tempfile on replace failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
