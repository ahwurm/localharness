"""User overlay file: load, deep-merge, atomic write.

Sits above project YAML in the config cascade (defaults → project → user → experiment).
Phase 14 ships the user layer; Phase 17 adds the experiment layer (per git-isolated workspace).

Atomic write protocol: tempfile in SAME DIRECTORY as target + os.replace. Per Pitfall 2
in 14-RESEARCH.md, cross-filesystem tempfile defeats atomicity on container/NFS mounts.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional  # noqa: F401  (Any kept for downstream typing imports)

import yaml

from localharness.config.paths import resolve_overlay_path


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
    """Load YAML overlay file. Missing file → empty dict. Invalid YAML raises."""
    path = Path(path).expanduser()
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
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

    try:
        os.replace(tmp_path, str(path))
    except Exception:
        # Best-effort cleanup of stranded tempfile on replace failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
