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

# The ONE directory name both layers use. The global default and the workspace walk-up derive
# from this anchor so they can never drift apart.
WORKSPACE_DIR_NAME = ".localharness"
_DEFAULT_CONFIG_DIR = f"~/{WORKSPACE_DIR_NAME}"

# The repository marker. `.git` is a DIRECTORY in an ordinary clone and a FILE in a git worktree
# or a submodule, so both shapes count as "there is a repo here".
GIT_DIR_NAME = ".git"

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


def global_config_dir(config_dir: Optional[PathLike] = None) -> Path:
    """The GLOBAL (machine-wide, never-workspace) config dir.

    Identical to ``resolve_config_dir`` — and stays identical. v0.13's workspace layer is NOT
    reached through this chain: discovery lives only in ``ConfigLoader._local_dir`` (fed by
    ``discover_workspace_dir`` via ``cli/workspace.resolve_workspace_layer``), so both functions
    always answer with the GLOBAL dir. Call THIS one wherever the target is machine-wide truth,
    so the site says so in its own text:

    - the model / active-endpoint overlay writes (``cli/model_ops.py``) — there is ONE physical
      GPU daemon, so a workspace must never fork ``server.model`` against it;
    - anything that must not be reached by workspace discovery (bench's own resolution).

    Explicit ``--config-dir`` still wins: a full replacement replaces the global layer itself.
    """
    return resolve_config_dir(config_dir)


def discover_workspace_dir(start: Optional[PathLike] = None) -> Optional[Path]:
    """The nearest ``.localharness/`` at or above ``start`` (default: CWD), or None.

    Returns on the FIRST hit — "exactly one workspace layer, nearest wins" (LAYR-04) is a
    property of stopping here, never of a later merge step.

    Never returns the GLOBAL dir: the walk stops at ``$HOME`` without inspecting it, because
    ``~/.localharness/`` is the global layer and every ordinary CWD is a descendant of ``$HOME``
    (without this stop, a user with no project workspace would "discover" their own global dir
    from anywhere under home, and LAYR-03's byte-identical guarantee would not hold).

    Says nothing about whether the result may be LOADED — that is
    ``workspace_is_within_repo`` plus ``cli/workspace.resolve_workspace_layer``.

    Pure: ``Path.is_dir()`` only. No env reading, no trust check, no prompting — those belong to
    ``cli/workspace.resolve_workspace_layer()``. Must be called inside a function body at the
    moment a command runs, NEVER bound to a module-level name (the workspace root varies per
    CWD; ``config/overlay.py:32-35``'s frozen ``USER_OVERLAY_PATH`` is the trap to avoid).
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except RuntimeError:  # no home resolvable (rare, e.g. some container users)
        home = None
    for ancestor in (here, *here.parents):
        if home is not None and ancestor == home:
            return None
        candidate = ancestor / WORKSPACE_DIR_NAME
        if candidate.is_dir():
            return candidate
    return None


ARCHIVE_DB_NAME = "archive.db"


def resolve_archive_db_path() -> Path:
    """Where ``propose`` / ``experiment`` / ``autoresearch`` keep their proposal history.

    ONE algorithm for "find my ``.localharness``", shared by all three commands (it used to be
    the same six lines copied three times). Precedence: ``LOCALHARNESS_DIR`` >
    ``LOCALHARNESS_HOME`` > the nearest ``.localharness/`` up-tree > ``./.localharness`` — note
    the default is project-local, which differs deliberately from the config dir's
    ``~/.localharness`` default.

    NOT trust-gated: this is a SQLite file the harness WRITES, not agent or config YAML the
    harness executes, so LAYR-05's prompt does not apply — gating a storage-location choice
    behind a security question would be unmotivated and surprising. It calls
    ``discover_workspace_dir()`` directly rather than ``cli/workspace.resolve_workspace_layer()``.
    """
    override = config_dir_env_override()
    if override:
        return Path(override).expanduser() / ARCHIVE_DB_NAME
    return (discover_workspace_dir() or Path.cwd() / WORKSPACE_DIR_NAME) / ARCHIVE_DB_NAME


def _nearest_repo_root(here: Path, home: Optional[Path]) -> Optional[Path]:
    """The nearest directory at or above ``here`` holding a ``.git`` entry, or None.

    Stops at ``$HOME`` for the same reason the workspace walk does: a home-directory dotfiles
    repository must not turn every folder under home into one giant "project".
    """
    for ancestor in (here, *here.parents):
        if home is not None and ancestor == home:
            return None
        marker = ancestor / GIT_DIR_NAME
        if marker.is_dir() or marker.is_file():
            return ancestor
    return None


def workspace_is_within_repo(
    workspace_dir: PathLike, start: Optional[PathLike] = None
) -> bool:
    """Does that discovered workspace belong to the project ``start`` (default CWD) is inside?

    True when a repository contains ``start`` AND the workspace's project folder (the parent of
    ``.localharness/``) sits at or below that repository's root. This is the "nested inherits"
    case: you are already inside the project, so its config is not config from elsewhere.

    False when no repository contains ``start``, or when the workspace's folder sits ABOVE the
    repository root — config reaching in from outside the tree you opened. Only that case is
    trust-gated (owner ruling 2026-09-03); see ``cli/workspace.resolve_workspace_layer``.

    Pure and process-spawn-free like the rest of this module: it reads ``.git`` markers off the
    filesystem and never shells out, so it works with no git binary installed and adds no child
    process to every command's startup.
    """
    here = Path(start).resolve() if start is not None else Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except RuntimeError:
        home = None
    repo_root = _nearest_repo_root(here, home)
    if repo_root is None:
        return False
    project = Path(workspace_dir).resolve().parent
    return project == repo_root or repo_root in project.parents


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
