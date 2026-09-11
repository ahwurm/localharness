"""Which workspaces from OUTSIDE the current project has the user agreed to load config from
(LAYR-05).

The record lives in the GLOBAL config dir, never inside the workspace being judged — a
workspace that could vouch for itself is not a trust boundary. Keyed by the resolved
(realpath) `.localharness/` path, so a symlinked checkout and its real path are one entry
and a git worktree is a separate workspace by design (v0.13 ruling).

Only consulted for workspaces the caller has already judged to be outside the project the
user is standing in — a workspace inside your own repository loads without ever reaching this
module (owner ruling 2026-09-03). See cli/workspace.resolve_workspace_layer.

Undecided is None, not False: a declined-in-a-script session must not become a permanent
"no" — only an answered prompt records anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from localharness.config.overlay import atomic_write_overlay
from localharness.config.paths import global_config_dir

WORKSPACE_TRUST_FILE = "trusted_workspaces.yaml"

# `start`'s offer to CREATE a workspace remembers a "no" the same way the trust question
# remembers one — asked once per directory, ever — and keeps that memory in its own file. Two
# reasons for the sibling rather than a second key inside the trust store: this is a preference,
# not a security boundary, and a file whose whole subject is "config I agreed to load" must not
# grow entries that mean something else. Same directory, same realpath keying, same atomic write.
DECLINED_OFFERS_FILE = "declined_workspace_offers.yaml"


def trust_store_path() -> Path:
    """Always the GLOBAL dir — resolved at call time so tests' env changes take effect."""
    return global_config_dir() / WORKSPACE_TRUST_FILE


def declined_offers_path() -> Path:
    """The sibling store, in the same GLOBAL dir and resolved at call time for the same reason."""
    return global_config_dir() / DECLINED_OFFERS_FILE


def _key(workspace_dir: Path) -> str:
    return str(Path(workspace_dir).resolve())


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a corrupt store means "undecided", never a crashed session
        return {}
    return data if isinstance(data, dict) else {}


def is_trusted(workspace_dir: Path) -> Optional[bool]:
    """True / False / None (never asked). None is the fail-closed case for the caller."""
    entry = _load(trust_store_path()).get(_key(workspace_dir))
    if isinstance(entry, dict) and isinstance(entry.get("trusted"), bool):
        return entry["trusted"]
    return None


def is_trusted_tree(path: Path) -> Optional[bool]:
    """The nearest recorded decision at or above ``path`` — True / False / None (never asked).

    Nested folders inherit: a decision recorded for a project root answers for every directory
    under it, so ``cd src/thing`` inside a workspace you already trusted asks nothing. The walk
    stops at the FIRST record it meets, so a "no" recorded on a subdirectory still stands inside
    a trusted parent.

    This is also what unifies the two questions v0.14.1 could have had. The workspace-layer
    question (LAYR-05, "load this outside-the-project config?") records the ``.localharness``
    directory; the session question (owner ruling 2026-09-11, "it asks to trust the workspace")
    records the workspace ROOT, which is that directory's parent. Walking upward means one yes
    answers both: trusted = its config layer loads AND its tool calls run in ``auto``. Walking
    upward and not downward is what keeps that from running backwards — trusting a project does
    not trust the directory above it.
    """
    here = Path(path).resolve()
    for candidate in (here, *here.parents):
        decision = is_trusted(candidate)
        if decision is not None:
            return decision
    return None


SESSION_EVIDENCE_GLOB = "agents/*/sessions/*.jsonl"
"""Where a state store records that a session actually ran: one JSONL per session, under the
agent that ran it (``config/paths.resolve_runtime_path``, and the layout the ask-rate report
reads).

Counting FILES is the whole recognition test (owner, 2026-09-11: "it should recognize I've been
in this environment before, used X tools etc."). Nothing is opened and nothing is parsed: the
question is "has work happened here", the file's existence is the answer, and a startup that
reads 384 session transcripts to decide whether to print one line is a startup nobody wants."""

LEGACY_SESSION_EVIDENCE_GLOB = "agents/*/history.jsonl"
"""The same evidence in an older store, which kept one rolling history per agent instead of a
directory of sessions. A user who has been running the harness since before the sessions
directory existed has been in this environment before, and the recognition must say so."""


def prior_session_count(state_dir: Path) -> int:
    """How many sessions this state store has already recorded.

    Zero for a store that does not exist, and zero for one that exists but has never run
    anything — which is the case a bare ``localharness init`` leaves behind, and the reason the
    ``memory.db`` file is deliberately NOT counted: init creates it empty, so its presence would
    say "you have been here before" about a directory nobody has worked in.
    """
    store = Path(state_dir)
    if not store.is_dir():
        return 0
    sessions = list(store.glob(SESSION_EVIDENCE_GLOB))
    if sessions:
        return len(sessions)
    return sum(
        1 for history in store.glob(LEGACY_SESSION_EVIDENCE_GLOB)
        if history.is_file() and history.stat().st_size > 0
    )


def record_trust(workspace_dir: Path, trusted: bool) -> None:
    """Persist a decision. Permanent by design — v0.13 ships no expiry and no `workspace trust`
    CLI verb (owner: "trust forever after"); changing an answer means hand-editing
    `~/.localharness/trusted_workspaces.yaml`, which the config spec documents."""
    data = _load(trust_store_path())
    data[_key(workspace_dir)] = {"trusted": trusted}
    atomic_write_overlay(trust_store_path(), data)


def offer_was_declined(workspace_dir: Path) -> bool:
    """Has the user already said no to creating THIS workspace?

    False when unreadable, missing or corrupt — the fail-open direction here, deliberately, and
    the opposite of `is_trusted`'s: the worst case is one repeated question, never config loaded
    without consent. Nothing else reads this file, so it can only ever cost a prompt.
    """
    entry = _load(declined_offers_path()).get(_key(workspace_dir))
    return isinstance(entry, dict) and entry.get("declined") is True


def record_offer_decline(workspace_dir: Path) -> None:
    """Remember a "no" so the offer is asked once per directory, ever.

    Only a person answering the prompt gets here — EOF and every non-interactive path record
    nothing, exactly as an unanswered trust prompt does. Undo it by deleting the entry (or the
    file); `init --workspace` and creating the directory by hand ignore this store entirely, so a
    recorded "no" can never stand between a user and a workspace they went and asked for.
    """
    data = _load(declined_offers_path())
    data[_key(workspace_dir)] = {"declined": True}
    atomic_write_overlay(declined_offers_path(), data)
