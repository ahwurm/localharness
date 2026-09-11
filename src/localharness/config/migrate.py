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

Additive only, with one deliberate exception: existing entries are never removed or reordered,
and no key other than `org.permissions.{deny_patterns,defaults_revision}` is touched — except
`org.permissions.allow_patterns`, which v0.14 REMOVED and which every pre-v0.14 `init` wrote as
`allow_patterns: []`. A config carrying it cannot load at all, so migrate deletes the key; this
is the documented repair path, and it has to be able to repair the thing that breaks. That
removal is also why the plan is not purely revision-gated: a config already stamped at the
current revision still gets a plan when the dead key is present.

The two halves stay separate, and that separation is load-bearing (review finding R10): when the
stamp is already current, the plan is the REMOVAL ALONE. Letting the dead key drag the "add every
missing shipped default" pass along with it would re-appear a default the user deliberately
deleted — the exact thing the revision gate exists to prevent — just because an unrelated key
happened to be in the same file.

The dead key is not only in config.yaml. Pre-v0.14 `write_agent` serialized `AgentConfig`
wholesale, so every `<config dir>/agents/<name>.yaml` (and every `divisions/<name>.yaml`) it ever
wrote carries `allow_patterns: []` too — files on the same permission cascade the merged config
reads (`loader._global_layer_permission`). Left alone, an empty one warns on EVERY load and a
populated one fails validation with no repair path at all, so migrate walks those files and
strips the same key, each with its own timestamped backup (:data:`SIDECAR_DIRS`).

The updated config is validated through the real HarnessConfig model before anything is
written — a migrate that writes an invalid config is worse than none.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.models import HarnessConfig, PermissionConfig
from localharness.config.overlay import restrict_config_file

log = logging.getLogger(__name__)


# The backup file IS the durable record of a migration — no separate state is written, and its own
# filename carries the timestamp. Both halves are module-level constants because `doctor` reads
# them back (cli/doctor_cmd._print_migration_state): a name typed in two places is a name that
# drifts, and the drift would be SILENT — doctor's glob would simply stop matching and quietly
# print no backup line at all. Lexicographic order on this format is also chronological, which is
# what lets a reader sort the glob instead of stat-ing every file for its mtime.
BACKUP_STAMP_FORMAT = "%Y%m%d-%H%M%S"
BACKUP_INFIX = ".bak-"
"""What sits between a migrated file's name and its timestamp. One spelling for every file
migrate writes — config.yaml and the agent/division sidecars alike — so `BACKUP_PREFIX` below
and a sidecar's backup name cannot drift apart."""
BACKUP_PREFIX = "config.yaml" + BACKUP_INFIX

BACKUP_RETENTION = 5
"""How many timestamped backups of ONE file migrate keeps, newest first.

A backup is written on every migration and nothing ever removed one, so the count grew with the
number of releases a user had installed — eleven beside a single config.yaml on the machine this
was found on, each a full copy of a file that may carry `provider.api_key`. Unbounded copies of a
secret are their own exposure, and they bury the one backup a person actually wants to read.

Five, because a backup's whole job is the undo for a migration you have just noticed went wrong,
and the horizon for noticing is a session or two — not eleven releases. Deliberately generous
against that: it keeps every intermediate state of a user who upgrades several versions at once,
which is the case where the oldest copy is the one worth having.
"""


class MigrationError(Exception):
    """Config could not be read, parsed, or (post-merge) validated."""


DEAD_KEY = "allow_patterns"
"""`org.permissions.allow_patterns`, removed in v0.14 (PRD §3.3). Named here because migrate is
the only writer allowed to delete a user's key, and the name it deletes should be readable at
the top of the file rather than quoted inline three times."""


SIDECAR_DIRS: tuple[str, ...] = ("agents", "divisions")
"""The global config dir's per-agent and per-division files, which sit on the SAME permission
cascade as `org.permissions` (`loader._global_layer_permission`) and were written by the same
model serializer — so they carry the same dead key and need the same repair (review finding
R10). Named as a tuple rather than globbed from the dir because migrate must only ever rewrite
files whose shape it knows; a future sibling directory is opted in here, deliberately."""


@dataclass(frozen=True)
class SidecarPlan:
    """One agent/division file still carrying `permissions.allow_patterns`.

    `removed` is the entries deleted (`[]` for the empty value pre-v0.14 `write_agent` wrote).
    `updated` is the file's whole mapping with the key gone, ready to write."""

    path: Path
    removed: list
    updated: dict


@dataclass(frozen=True)
class MigrationPlan:
    """A pending sync. `added` may be empty (config already has every default but is below the
    current revision — the stamp still advances). `removed_allow_patterns` is None when the dead
    `allow_patterns` key is absent, and otherwise the entries deleted (`[]` for the empty value
    every pre-v0.14 `init` wrote). `updated` is the full config dict with the deny list, the
    stamp and the removal applied, ready to validate and write. `sidecars` are the agent/division
    files needing the same key removed — they can be the ONLY reason a plan exists, which is why
    `config_unchanged` is asked before config.yaml is rewritten."""

    added: list[str]
    from_revision: int
    to_revision: int
    updated: dict
    removed_allow_patterns: Optional[list] = None
    sidecars: tuple[SidecarPlan, ...] = ()

    @property
    def config_unchanged(self) -> bool:
        """True when config.yaml itself needs nothing — the plan exists only for the sidecars.

        Rewriting an untouched file would still cost the user a backup and a YAML round-trip
        (comments and key order are not preserved), so the writer skips it."""
        return (
            not self.added
            and self.removed_allow_patterns is None
            and self.from_revision >= self.to_revision
        )


def scan_sidecars(config_dir: Path) -> list[SidecarPlan]:
    """Every `agents/*.yaml` and `divisions/*.yaml` under `config_dir` still carrying the dead key.

    Read-only and forgiving: a file that will not parse, or that is not a mapping, is left alone
    rather than rewritten — migrate deletes one key it understands, and a file it cannot read is
    not a file it can safely edit. Sorted so the CLI's listing (and its backups) are in a stable
    order. Nothing outside the global config dir is ever considered (review finding R10)."""
    found: list[SidecarPlan] = []
    for name in SIDECAR_DIRS:
        for path in sorted((config_dir / name).glob("*.yaml")):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if not isinstance(data, dict):
                continue
            perms = data.get("permissions")
            if not isinstance(perms, dict) or DEAD_KEY not in perms:
                continue
            updated_perms = dict(perms)
            dead = updated_perms.pop(DEAD_KEY)
            updated = dict(data)
            updated["permissions"] = updated_perms
            found.append(
                SidecarPlan(path, list(dead) if isinstance(dead, list) else [], updated)
            )
    return found


def plan(data: dict, sidecars: tuple[SidecarPlan, ...] = ()) -> Optional[MigrationPlan]:
    """Return a MigrationPlan if `data` (or any sidecar file) needs work, else None.

    Work means below the current defaults revision, still carrying the removed
    `org.permissions.allow_patterns` key, or an agent/division file carrying that key. The
    second and third clauses are not decoration: a file with that key fails validation outright
    (or warns on every load when empty), so the repair has to reach it even when the revision
    stamp is already current.

    When the stamp IS current the plan is the removal alone — `added` stays empty. That is the
    removal-respect path: a stamped-current config is never inspected for missing defaults, so
    anything the user deleted stays deleted, and repairing the dead key never smuggles it back
    (review finding R10).

    None = nothing to do.
    """
    org = data.get("org") if isinstance(data.get("org"), dict) else {}
    perms = org.get("permissions") if isinstance(org.get("permissions"), dict) else {}
    stamped = perms.get("defaults_revision")
    stamped = stamped if isinstance(stamped, int) else 0
    has_dead_key = DEAD_KEY in perms
    at_current = stamped >= CURRENT_DEFAULTS_REVISION
    if at_current and not has_dead_key and not sidecars:
        return None

    user_deny = perms.get("deny_patterns")
    user_deny = list(user_deny) if isinstance(user_deny, list) else []
    added = [] if at_current else [
        p for p in PermissionConfig().deny_patterns if p not in user_deny
    ]

    updated = dict(data)
    updated_org = dict(org)
    updated_perms = dict(perms)
    removed = None
    if has_dead_key:
        dead = updated_perms.pop(DEAD_KEY)
        removed = list(dead) if isinstance(dead, list) else []
    updated_perms["deny_patterns"] = [*user_deny, *added]
    updated_perms["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    updated_org["permissions"] = updated_perms
    updated["org"] = updated_org
    return MigrationPlan(
        added, stamped, CURRENT_DEFAULTS_REVISION, updated, removed, tuple(sidecars)
    )


def load_plan(config_file: Path) -> tuple[bytes, Optional[MigrationPlan]]:
    """Read + parse config_file and compute its migration plan, sidecar files included.

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
    return original, plan(data, tuple(scan_sidecars(config_file.parent)))


def apply(config_file: Path, original: bytes, migration: MigrationPlan) -> list[Path]:
    """Write the plan: config.yaml (when it changed) and every sidecar file, each backed up first.

    Returns the backup paths in write order. Raises MigrationError if the updated config fails
    validation (nothing is written in that case).

    The sidecars are NOT model-validated: migrate is the only repair path for a file that cannot
    load, so refusing to delete the fatal key because something ELSE in the file is invalid would
    leave the user with no way out at all. The one key removed is fatal by definition, and the
    original bytes are kept in the backup beside it.
    """
    backups: list[Path] = []
    if not migration.config_unchanged:
        try:
            HarnessConfig.model_validate(migration.updated)
        except Exception as exc:
            raise MigrationError(f"migrated config fails validation: {exc}") from exc
        backups.append(_write_with_backup(config_file, original, migration.updated))
    for sidecar in migration.sidecars:
        backups.append(
            _write_with_backup(sidecar.path, sidecar.path.read_bytes(), sidecar.updated)
        )
    return backups


def _write_with_backup(path: Path, original: bytes, updated: dict) -> Path:
    """Timestamped backup of `original` beside `path`, then `updated` dumped over `path`.

    Both files get `CONFIG_FILE_MODE`: the backup is a byte-for-byte copy of a config.yaml that
    may carry `provider.api_key`, so leaving IT at the umask default would hand out the secret the
    rewritten file no longer exposes. Older backups of this same file are pruned afterwards
    (:data:`BACKUP_RETENTION`).
    """
    stamp = datetime.now().strftime(BACKUP_STAMP_FORMAT)
    backup = path.with_name(f"{path.name}{BACKUP_INFIX}{stamp}")
    backup.write_bytes(original)
    restrict_config_file(backup)
    path.write_text(
        yaml.safe_dump(updated, default_flow_style=False, sort_keys=False), encoding="utf-8"
    )
    restrict_config_file(path)
    _prune_backups(path, keep=backup)
    return backup


def _prune_backups(path: Path, *, keep: Path) -> None:
    """Delete all but the newest :data:`BACKUP_RETENTION` backups of `path`. Best effort.

    Scoped as tightly as deleting files in a user's config directory deserves. A candidate must
    be a FILE, sit in the same directory, and be named exactly `<path.name>.bak-<stamp>` where
    `<stamp>` parses under :data:`BACKUP_STAMP_FORMAT` — so a hand-kept `config.yaml.bak-before-
    the-upgrade` is not one of ours and is never touched, and neither is a backup of a DIFFERENT
    file in the same folder. The backup just written is excluded explicitly as well as by age:
    the undo for the migration that is happening right now is the one copy that must survive
    whatever the retention number says.

    Lexicographic order on `BACKUP_STAMP_FORMAT` is chronological (that is why the format is
    zero-padded and big-endian), so the name is the sort key and no file needs stat-ing.

    A failed unlink is logged and shrugged off: migrate's job is the config rewrite, which has
    already succeeded by the time this runs, and a read-only leftover must not turn a successful
    migration into a failed one.
    """
    prefix = f"{path.name}{BACKUP_INFIX}"
    ours: list[Path] = []
    for sibling in path.parent.glob(f"{prefix}*"):
        if not sibling.is_file():
            continue
        try:
            datetime.strptime(sibling.name[len(prefix):], BACKUP_STAMP_FORMAT)
        except ValueError:
            continue  # not a backup this module wrote — leave it alone
        ours.append(sibling)

    for stale in sorted(ours, key=lambda p: p.name)[:-BACKUP_RETENTION]:
        if stale == keep:
            continue
        try:
            stale.unlink()
        except OSError:
            log.debug("could not prune stale config backup %s", stale, exc_info=True)
