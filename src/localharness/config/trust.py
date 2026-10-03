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

Trust is for what you saw. Beside the yes/no, a project's entry keeps the MCP servers its agent
files start AS THEY WERE SHOWN at the Yes (`executables`), and the machine keeps one record of what
its own agent files start, load or loosen (`machine_key`). Start compares both with the files on
disk and asks about a difference once, on a terminal (cli/workspace.decide_project_trust,
decide_machine_trust). Same file on purpose: it is a protected path in every mode, so the agent can
ask to write it but never be granted it.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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

PROCESS_STARTED_AT = time.time()
"""When this process first imported the trust store — the cutoff for "EARLIER session".

The bug this closes, found by a live end-to-end run against the real model: a brand-new project
printed "recognized this workspace (1 earlier session)", trusted itself and wrote the record,
with nobody asked. The evidence it recognized was the file the RUNNING session had just written.

A rolling per-agent ``history.jsonl`` used to be counted too, and that is why it no longer is:
one file, appended to by every session including this one, with no way to ask "was any of this
here before I started". It could not be made safe, so it is gone — a pre-sessions-dir install is
simply asked its one question. Per-session files can be judged, and are: a file whose mtime is
at or after this moment is this run's own and is not evidence of anything.

Captured at import rather than passed in, because the earliest thing that touches this module is
the startup path that asks the question, and a cutoff that arrives later than the writes it has
to exclude is not a cutoff. The real defence is ORDER — ``cli/workspace.settle_startup_trust``
runs before any session store is opened — and this is the belt to that pair of braces, for the
channels (ACP, Discord) whose question cannot be drawn until later."""


def prior_session_count(state_dir: Path, before: Optional[float] = None) -> int:
    """How many sessions this state store recorded BEFORE this one.

    Zero for a store that does not exist, and zero for one that exists but has never run
    anything — the case a bare ``localharness init`` leaves behind, and the reason ``memory.db``
    is deliberately not counted: init creates it empty, so its presence would say "you have been
    here before" about a directory nobody has worked in.

    ``before`` defaults to :data:`PROCESS_STARTED_AT`. A file the current run wrote is not
    evidence that the current run should be trusted.
    """
    store = Path(state_dir)
    if not store.is_dir():
        return 0
    cutoff = PROCESS_STARTED_AT if before is None else before
    count = 0
    for session in store.glob(SESSION_EVIDENCE_GLOB):
        try:
            if session.is_file() and session.stat().st_mtime < cutoff:
                count += 1
        except OSError:  # a file that vanished between the glob and the stat is not evidence
            continue
    return count


def record_trust(workspace_dir: Path, trusted: bool, *, unseen_executables: bool = False) -> None:
    """Persist a decision. Permanent by design — v0.13 ships no expiry and no `workspace trust`
    CLI verb (owner: "trust forever after"); changing an answer means hand-editing
    `~/.localharness/trusted_workspaces.yaml`, which the config spec documents.

    Only `trusted` changes: the servers a Yes approved stay. `unseen_executables`: this Yes was
    given without the server list (the in-session question), so an entry with no servers recorded
    gets "nothing approved yet" — the next start asks about them rather than adopting them."""
    data = _load(trust_store_path())
    key = _key(workspace_dir)
    entry = dict(data[key]) if isinstance(data.get(key), dict) else {}
    entry["trusted"] = trusted
    if trusted and unseen_executables and EXECUTABLES_KEY not in entry:
        entry[EXECUTABLES_KEY] = {**NOTHING_APPROVED, "recorded": _now()}
    data[key] = entry
    atomic_write_overlay(trust_store_path(), data)


EXECUTABLES_KEY = "executables"

NOTHING_APPROVED: dict = {"fingerprint": "", "servers": []}
"""What a Yes given WITHOUT the server list records: nothing approved yet, so the next start asks."""

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(entry: Any) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


def _read_yaml(path: Path) -> Any:
    """A file's YAML, or None when it cannot be read or parsed — the loader names that file and
    loads nothing from it, so there is nothing in it to start."""
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None


def _servers(raw: Any) -> list[dict]:
    tools = raw.get("tools") if isinstance(raw, dict) else None
    servers = tools.get("mcp_servers") if isinstance(tools, dict) else None
    return [s for s in servers if isinstance(s, dict)] if isinstance(servers, list) else []


def _names(mapping: Any) -> list[str]:
    """The keys of an `env` or `headers` mapping, never a value (values are often secrets)."""
    return sorted(map(str, mapping)) if isinstance(mapping, dict) else []


def _args(server: dict) -> list[str]:
    args = server.get("args")
    args = args if isinstance(args, list) else ([] if args is None else [args])
    return [str(a) for a in args if a is not None]


def executables_snapshot(workspace_dir: Path) -> list[dict]:
    """What a project's own agent files would start or connect to: every `tools.mcp_servers` entry
    in `<workspace>/agents/*.yaml`, as {file, name, transport, command, args, env, url, headers}
    with env and header NAMES only (values are often secrets: never stored, never shown), sorted by
    file, name, then the entry's canonical JSON (two servers may share a name: their order in the
    file never changes the fingerprint). A file that does not parse, or is not a mapping, is
    skipped — the loader warns about it and loads nothing from it. Scripts a command runs are not
    read (named in SECURITY.md)."""
    out: list[dict] = []
    for path in sorted((Path(workspace_dir) / "agents").glob("*.yaml")):
        for s in _servers(_read_yaml(path)):
            out.append({"file": path.name, "name": str(s.get("name") or ""),
                        "transport": str(s.get("transport") or ""),
                        "command": str(s.get("command") or ""), "args": _args(s),
                        "env": _names(s.get("env")), "url": str(s.get("url") or ""),
                        "headers": _names(s.get("headers"))})
    return sorted(out, key=lambda e: (e["file"], e["name"], _canonical(e)))


def fingerprint(snapshot: list[dict]) -> str:
    """One digest of a snapshot: "sha256:" + hex of its canonical JSON."""
    return "sha256:" + hashlib.sha256(_canonical(snapshot).encode("utf-8")).hexdigest()


def _entry(data: dict, key: str) -> dict:
    return dict(data[key]) if isinstance(data.get(key), dict) else {}


def recorded_executables(workspace_root: Path) -> Optional[dict]:
    """The executables record kept for this exact root, or None (never recorded — a trust record
    written before this release, or one written by hand). A malformed record reads as "nothing
    approved yet", the direction that asks."""
    rec = _entry(_load(trust_store_path()), _key(workspace_root)).get(EXECUTABLES_KEY)
    if not isinstance(rec, dict):
        return None
    servers = rec.get("servers")
    return {"fingerprint": rec["fingerprint"] if isinstance(rec.get("fingerprint"), str) else "",
            "servers": [e for e in servers if isinstance(e, dict)] if isinstance(servers, list) else []}


def record_executables(workspace_root: Path, snapshot: list[dict]) -> None:
    """Store {fingerprint, servers, recorded (UTC ISO seconds)} under the root's own entry, keeping
    its "trusted" key. On the exact root's key, not a parent's: sibling projects under one trusted
    folder each keep their own record."""
    data = _load(trust_store_path())
    key = _key(workspace_root)
    entry = _entry(data, key)
    entry[EXECUTABLES_KEY] = {"fingerprint": fingerprint(snapshot), "servers": snapshot,
                              "recorded": _now()}
    data[key] = entry
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
