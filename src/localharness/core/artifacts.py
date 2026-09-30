"""Where a plugin's artifacts live and what they are called — both decided by core (PAPI-10).

Decision 13: "plugins choose whether they have artifacts, never where they live". A plugin opts in
with `Plugin.wants_artifacts`; core computes its root, mints every id in the one shape
(core/events.ARTIFACT_ID_RE), and writes only allowlisted media types (core/events.ARTIFACT_MIMES).
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from pathlib import Path

from localharness.core.events import ARTIFACT_MIMES, ArtifactRef
from localharness.plugins.api import PLUGIN_NAME_RE

ARTIFACTS_DIR_NAME = "artifacts"


def artifact_root(state_dir: Path, plugin: str) -> Path:
    """`<state_dir>/artifacts/<plugin>/` — the ONLY place a plugin's artifacts may live. A plugin
    opts in with `Plugin.wants_artifacts` and never chooses this path; `plugin` must be a plugin
    name, so the root cannot leave the artifacts directory."""
    if not PLUGIN_NAME_RE.fullmatch(plugin):
        raise ValueError(f"{plugin!r} is not a plugin name")
    return Path(state_dir) / ARTIFACTS_DIR_NAME / plugin


def mint_artifact_id(now: datetime | None = None) -> str:
    """A fresh id in the one core-owned shape: `art-YYYYMMDD-HHMMSS-<6 hex>`, UTC by default."""
    return f"art-{now or datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"


def write_artifact(root: Path, plugin: str, data: bytes, mime: str) -> ArtifactRef:
    """Write `data` as one new file `<id><suffix>` under `root` (created if missing) and return its
    reference. A mime off the allowlist, or a bad plugin name, raises ValueError before anything is
    written; an existing file is never overwritten (FileExistsError)."""
    suffix = ARTIFACT_MIMES.get(mime)
    if suffix is None:
        raise ValueError(f"artifact mime {mime!r} is not allowed — only {', '.join(ARTIFACT_MIMES)}")
    ref = ArtifactRef(plugin=plugin, kind="image", id=mint_artifact_id(), mime=mime)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{ref.id}{suffix}").open("xb") as fh:
        fh.write(data)
    return ref
