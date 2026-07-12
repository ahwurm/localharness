"""Additive, revision-stamped sync of shipped default deny patterns into an existing config.

The shared engine behind two surfaces:
  * `localharness config migrate` — explicit, with a `--dry-run` preview (cli/config_cmd.py).
  * `localharness start` — auto-applies on the first start after a package upgrade
    (cli/start_cmd.py._auto_migrate_deny_defaults).

Why a revision stamp instead of "add every missing default"? Because auto-apply on startup
must be SAFE. The stamp (`org.permissions.defaults_revision`) records which revision of the
shipped list the config was last synced to. The sync is gated on `stamped < current`, NOT on
"is any default missing" — so once a config reaches the current revision, a default the user
DELIBERATELY deleted is never re-added. Removal-respect is the whole reason it can run
unattended. `init` stamps fresh configs at the current revision, so a new install is never
touched and any later removal is respected from day one.

Additive only: existing entries are never removed or reordered, and no key other than
`org.permissions.{deny_patterns,defaults_revision}` is touched. The updated config is validated
through the real HarnessConfig model before anything is written — a migrate that writes an
invalid config is worse than none.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.models import HarnessConfig, PermissionConfig


class MigrationError(Exception):
    """Config could not be read, parsed, or (post-merge) validated."""


@dataclass(frozen=True)
class MigrationPlan:
    """A pending additive sync. `added` may be empty (config already has every default but is
    below the current revision — the stamp still advances). `updated` is the full config dict
    with the deny list + stamp applied, ready to validate and write."""

    added: list[str]
    from_revision: int
    to_revision: int
    updated: dict


def plan(data: dict) -> Optional[MigrationPlan]:
    """Return a MigrationPlan if `data` is below the current defaults revision, else None.

    None = already at/above the current revision → nothing to do. This is also the
    removal-respect path: a stamped-current config is never inspected for missing defaults,
    so anything the user deleted stays deleted.
    """
    org = data.get("org") if isinstance(data.get("org"), dict) else {}
    perms = org.get("permissions") if isinstance(org.get("permissions"), dict) else {}
    stamped = perms.get("defaults_revision")
    stamped = stamped if isinstance(stamped, int) else 0
    if stamped >= CURRENT_DEFAULTS_REVISION:
        return None

    user_deny = perms.get("deny_patterns")
    user_deny = list(user_deny) if isinstance(user_deny, list) else []
    added = [p for p in PermissionConfig().deny_patterns if p not in user_deny]

    updated = dict(data)
    updated_org = dict(org)
    updated_perms = dict(perms)
    updated_perms["deny_patterns"] = [*user_deny, *added]
    updated_perms["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    updated_org["permissions"] = updated_perms
    updated["org"] = updated_org
    return MigrationPlan(added, stamped, CURRENT_DEFAULTS_REVISION, updated)


def load_plan(config_file: Path) -> tuple[bytes, Optional[MigrationPlan]]:
    """Read + parse config_file and compute its migration plan.

    Returns (original_bytes, plan). Raises MigrationError on a missing/unparseable/non-mapping
    config so callers can present a clear failure.
    """
    if not config_file.exists():
        raise MigrationError(
            f"No config found at {config_file} — run 'localharness init' first."
        )
    original = config_file.read_bytes()
    try:
        data = yaml.safe_load(original.decode("utf-8"))
    except yaml.YAMLError as exc:
        raise MigrationError(f"Could not parse {config_file}: {exc}") from exc
    if not isinstance(data, dict):
        raise MigrationError(f"{config_file} is not a valid config mapping.")
    return original, plan(data)


def apply(config_file: Path, original: bytes, migration: MigrationPlan) -> Path:
    """Validate the updated config, write a timestamped backup of `original`, then the config.

    Returns the backup path. Raises MigrationError if the updated config fails validation
    (nothing is written in that case).
    """
    try:
        HarnessConfig.model_validate(migration.updated)
    except Exception as exc:
        raise MigrationError(f"migrated config fails validation: {exc}") from exc

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = config_file.with_name(f"config.yaml.bak-{stamp}")
    backup.write_bytes(original)
    config_file.write_text(
        yaml.safe_dump(migration.updated, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )
    return backup
