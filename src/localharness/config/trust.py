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

import functools
import hashlib
import json
import os
import shlex
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

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

TOOL_SCRIPT_KIND = "tool_script"

ORG_PERMISSION_KIND = "org_permission"
LAUNCH_KIND = "launch"
LAUNCH_KEYS: tuple[str, ...] = (
    "binary", "docker_image", "extra_args", "runtime", "launch", "bind_all", "require_api_key",
    "api_key_file", "env", "local_models", "config_ini")
"""The one key table: the `server:` keys (and every peer `lifecycle:` key) that decide what the
harness executes, how it is exposed or what it is given when it launches a model server.
`machine_snapshot`, `init`'s adoption and `SECURITY.md` all read this tuple, so no list is kept by
hand. `effort` is never a fingerprinted key: it is a request parameter, not a launch decision."""

_POSTURE_KEYS: tuple[str, ...] = ("bind_all", "require_api_key")
"""The two LAUNCH_KEYS emitted once per record from the merged view of both files, not per file."""

LAUNCH_MODEL_KIND = "launch_model"
LAUNCH_MODEL_KEYS: tuple[str, ...] = (
    "model", "mmproj", "served_name", "context", "cache_type_k", "cache_type_v", "cache_ram",
    "speculative", "reasoning_parser", "tool_call_parser")
"""The model keys of `server:` and of every peer `lifecycle:`, including the three record fields
(`speculative`, `reasoning_parser`, `tool_call_parser`) that pin what the harness would otherwise
derive at launch; an unset one yields no entry. They are their own kind because a
changed model is not a changed command: the first start that knows this kind adopts the current
values once with no question (a record made before a kind existed adopts that kind silently), and
a later change is a change like any other entry."""

MACHINE_KINDS: frozenset[str] = frozenset(
    {"mcp_server", "embedding_model", "permission", TOOL_SCRIPT_KIND, ORG_PERMISSION_KIND,
     LAUNCH_KIND, LAUNCH_MODEL_KIND})
"""Kinds of entry machine_snapshot() produces. A record made before a kind existed adopts that
kind's current entries silently once — they predate the rule. That is how the first start that knows
tool scripts adopts every script already in the tools folder."""

AUTOSTART_FILE = "systemd/localharness-llm.service"
AUTOSTART_ENTRY_NAME = "autostart.env"
AUTOSTART_KEY = (AUTOSTART_FILE, LAUNCH_KIND, AUTOSTART_ENTRY_NAME)
"""The pseudo-file, name and `(file, kind, name)` of the one entry that anchors the autostart unit's
captured `# env` block in the machine record. It is a `launch`-kind entry on purpose: a kind of its
own would be adopted silently once (a record predates the kind), and so would accept a unit planted
before the first start that knows it. The pseudo-file is a pairing key only; no loader consumer
matches it, so withholding the entry drops nothing the loader reads."""

_DEPENDENCY_DIRS: frozenset[str] = frozenset(
    {"node_modules", ".venv", "venv", "__pycache__", ".git", "site-packages"})
"""Folders under <global>/tools/ that hold installed dependencies or caches, not scripts: the
packaged helper asks for `cd ~/.localharness/tools && npm install playwright` (~1,500 files), and a
venv there holds thousands more. Never listed at start, never gated (a named gap in SECURITY.md).
A folder holding a `pyvenv.cfg` is skipped too, whatever its name (_is_venv)."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(entry: Any) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), default=str)


def _entry_key(entry: dict) -> tuple[str, str, str, str]:
    """The order a machine record keeps its entries in: (file, kind, name, canonical JSON). One
    key for `machine_snapshot` and `adopt_entries`, so an adopted record sorts as a snapshot does
    and the fingerprints agree."""
    return (entry["file"], entry["kind"], entry["name"], _canonical(entry))


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


def _value_digests(mapping: Any) -> dict[str, str]:
    """SHA-256 hex of each env/header value, keyed by name. A value-only change (e.g. swapping a
    TOKEN) now trips the fingerprint; the raw value is never stored (no-plaintext preserved).
    No salt: the value is already non-public and the digest is one-way, so a salt only adds
    management for no security gain, and an unsalted digest is stable across runs.
    Handles both plain strings (from YAML) and SecretStr (from config models)."""
    if not isinstance(mapping, dict):
        return {}
    def _resolve(v: Any) -> str:
        return v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
    return {str(k): hashlib.sha256(_resolve(v).encode("utf-8")).hexdigest() for k, v in mapping.items()}


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
    skipped — the loader warns about it and loads nothing from it; so is a file that leads outside
    the project's `.localharness/` (a symlink, or a symlinked `agents/` folder), which the loader
    never loads. Scripts a command runs are not read (named in SECURITY.md)."""
    from localharness.config.loader import layer_files  # the roster's own enumeration

    out: list[dict] = []
    for path in layer_files(workspace_dir, "agents", contained=True):
        for s in _servers(_read_yaml(path)):
            out.append({"file": path.name, "name": str(s.get("name") or ""),
                        "transport": str(s.get("transport") or ""),
                        "command": str(s.get("command") or ""), "args": _args(s),
                        "env": _value_digests(s.get("env")), "url": str(s.get("url") or ""),
                        "headers": _value_digests(s.get("headers"))})
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


def machine_key(global_dir: Path) -> str:
    """The trust-store key of this machine's own record: never a resolved path, so no workspace
    lookup reads it; per config dir, so an explicit --config-dir keeps its own record."""
    return "<machine>" + str(Path(global_dir).resolve())


def _shown_server(server: dict) -> str:
    """What a server runs or reaches: its url, or its command line quoted so that two different
    argument lists can never read the same."""
    if str(server.get("transport") or "") == "streamable_http":
        return str(server.get("url") or "")
    return shlex.join([str(server.get("command") or ""), *_args(server)])


@functools.lru_cache(maxsize=1)
def _packaged_hashes() -> frozenset[str]:
    """sha256 of every file LocalHarness ships in localharness/assets/ (cached): start installs some
    of them into the tools folder itself, and they are never the user's or the agent's scripts. Each
    also as start's text-mode copy writes it on Windows, with CRLF line ends."""
    from importlib import resources

    out: set[str] = set()
    try:
        for item in resources.files("localharness").joinpath("assets").iterdir():
            if item.is_file():
                data = item.read_bytes().replace(b"\r\n", b"\n")
                out |= {hashlib.sha256(data).hexdigest(),
                        hashlib.sha256(data.replace(b"\n", b"\r\n")).hexdigest()}
    except (OSError, ValueError):  # nothing shipped: nothing is exempt
        pass
    return frozenset(out)


def _sha256_file(path: Path) -> Optional[str]:
    """Hex sha256 of a regular file's bytes, or None when it is not one or cannot be read. Opened
    non-blocking and checked once open, so a FIFO swapped in for a script cannot stall the gate."""
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        with open(os.open(path, flags), "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return None
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError:
        return None


def tool_script_entries(global_dir: Path) -> list[dict]:
    """Every file under <global>/tools/ (recursively through real folders — a symlinked folder is
    not walked — never descending into a _DEPENDENCY_DIRS folder), as {kind: "tool_script", file:
    "tools/<rel>", name: "<rel>", shown: "sha256:<first 12 hex>", sha256: "<hex>"}, sorted by
    file. A file is judged by its entry, not by where it links: a symlink there is listed under its
    own name with the content it points to (skipping one that leads outside let it run past the
    tool-script rule as an ordinary command). A file whose bytes equal a packaged asset is left
    out; a file that cannot be read, or is not a regular file, is skipped."""
    tools = Path(global_dir) / "tools"
    out: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(tools):
        dirnames[:] = [d for d in dirnames
                       if d not in _DEPENDENCY_DIRS and not _is_venv(Path(dirpath) / d)]
        for name in filenames:
            path = Path(dirpath) / name
            digest = _sha256_file(path)
            if digest is None or digest in _packaged_hashes():
                continue
            rel = path.relative_to(tools).as_posix()
            out.append({"kind": TOOL_SCRIPT_KIND, "file": f"tools/{rel}", "name": rel,
                        "shown": f"sha256:{digest[:12]}", "sha256": digest})
    return sorted(out, key=lambda e: e["file"])


_PENDING: dict[tuple, Optional[str]] = {}
"""tool_script_pending's answers, keyed on the file's and the store's stat (ctime included: it
cannot be set back the way mtime can)."""


def tool_script_pending(path: Path, global_dir: Optional[Path] = None) -> Optional[str]:
    """None when this file may run as an ordinary command: anything outside <global>/tools/, a file
    below a _DEPENDENCY_DIRS folder there, an exact copy of a packaged asset, or a tool script whose
    current sha256 the machine record holds. Otherwise the script's current short digest
    ("sha256:<first 12 hex>"), which keys the guarded-mode ask so an "always" there covers this
    exact content only. Reads the store and hashes the file, cached on both files' stat (inode,
    size, mtime_ns and ctime_ns); a tools file that cannot be read is pending, with the digest
    "unreadable".

    <global> is `global_dir` — the session's config folder, which the gate passes (a session
    started with --config-dir gates its own tools folder) — else global_config_dir(). A file is a
    tool script when it resolves into the tools folder, or when its own entry sits there through
    real folders (a symlink in the tools folder to a file elsewhere is that entry, hashed by what it
    points to). A record from before scripts were a kind holds none: start adopts every script of a
    kind the record predates, so they all read as confirmed until it records them."""
    global_dir = Path(global_dir) if global_dir is not None else global_config_dir()
    try:
        root = (global_dir / "tools").resolve()
        real = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    rel = _tools_rel(Path(path), global_dir, root, real)
    if rel is None or _DEPENDENCY_DIRS.intersection(rel.parts[:-1]) or any(
            _is_venv(Path(global_dir) / "tools" / Path(*rel.parts[:i]))
            for i in range(1, len(rel.parts))):
        return None
    try:
        st = real.stat()
        store = trust_store_path()
        sst = store.stat() if store.exists() else None
    except OSError:
        return "unreadable"
    key = (str(real), rel.as_posix(), str(root), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns,
           (sst.st_ino, sst.st_size, sst.st_mtime_ns, sst.st_ctime_ns) if sst else None)
    if key not in _PENDING:
        if len(_PENDING) > 256:
            _PENDING.clear()
        _PENDING[key] = _script_pending(global_dir, f"tools/{rel.as_posix()}", real)
    return _PENDING[key]


def _is_venv(folder: Path) -> bool:
    """A Python virtual environment, whatever it is called: it holds a `pyvenv.cfg`. Its interpreter
    symlinks and site-packages are installed dependencies, not scripts (a named gap in SECURITY.md:
    a script placed in such a folder is not tracked either)."""
    try:
        return (folder / "pyvenv.cfg").is_file()
    except OSError:
        return False


def _tools_rel(path: Path, global_dir: Path, root: Path, real: Path) -> Optional[Path]:
    """`path` relative to the tools folder, or None when it is not a tool script: by where it
    resolves (inside the folder), else by its own entry — the path as written, normalised, through
    real folders only, since start never walks a symlinked folder."""
    if real != root and real.is_relative_to(root):
        return real.relative_to(root)
    lex_root = Path(os.path.abspath(Path(global_dir) / "tools"))
    lex = Path(os.path.abspath(path))
    if lex == lex_root or not lex.is_relative_to(lex_root):
        return None
    rel = lex.relative_to(lex_root)
    step = lex_root
    for part in rel.parts[:-1]:
        step = step / part
        if step.is_symlink():
            return None
    return rel


def _script_pending(global_dir: Path, file: str, real: Path) -> Optional[str]:
    digest = _sha256_file(real)
    if digest is None:
        return "unreadable"
    if digest in _packaged_hashes():
        return None
    rec = recorded_machine(global_dir)
    if rec is not None and (TOOL_SCRIPT_KIND not in rec["kinds"] or any(
            e.get("kind") == TOOL_SCRIPT_KIND and e.get("file") == file and e.get("sha256") == digest
            for e in rec["entries"])):
        return None
    return f"sha256:{digest[:12]}"


@functools.lru_cache(maxsize=1)
def _bool_adapter() -> Any:
    from pydantic import TypeAdapter
    return TypeAdapter(bool)


def coerce_bool(value: Any) -> Optional[bool]:
    """``value`` read the way the config model reads a `bool` field (pydantic's lax coercion: `True`,
    `1`, `"1"`, `"true"`, `"True"`, `"yes"`, `"on"` and the false spellings), or None when it is not
    one. The posture entries are built from the raw files, and the model, not the YAML spelling, decides
    what a launch does with `bind_all: 1`; a spelling the model reads as true must be recorded, shown
    and withheld as `true`, never as a string that only looks different."""
    try:
        return _bool_adapter().validate_python(value)
    except ValueError:
        return None


_NOT_A_PEER: Any = object()
"""`_key_state`'s marker for the primary: a peer with no `api_key` is a peer, not the primary."""


def _key_state(mapping: Any, raws: dict, runtime: str, *, peer_key: Any = _NOT_A_PEER) -> str:
    """`keyed` or `open`, from DECLARED key sources only: `require_api_key` is true and one of
    `api_key_file` is set (any string — whether it is a regular file is `doctor`'s and
    `launch_decision`'s business), the runtime's key variable (`KEY_ENV`) is in the environment,
    for the primary a `provider.api_key` other than `none` in either settings file, or for a peer
    its own `api_key` other than `none`. No file is ever opened here: a FIFO named as the key file
    must not be able to stall a start."""
    from localharness.provider.server import KEY_ENV  # lazy: config must not import provider at import

    def real(value: Any) -> bool:
        return isinstance(value, str) and value not in ("", "none")

    if not isinstance(mapping, dict) or coerce_bool(mapping.get("require_api_key")) is not True:
        return "open"
    if mapping.get("api_key_file"):
        return "keyed"
    if os.environ.get(KEY_ENV.get(runtime, "")):
        return "keyed"
    if peer_key is _NOT_A_PEER:
        for raw in raws.values():
            provider = raw.get("provider") if isinstance(raw, dict) else None
            if isinstance(provider, dict) and real(provider.get("api_key")):
                return "keyed"
        return "open"
    return "keyed" if real(peer_key) else "open"


def _posture_entries(raws: dict) -> list[dict]:
    """The `bind_all` / `require_api_key` entries, once per record, from the MERGED view of the two
    settings files (the one exception to per-file emission: a posture is the product of both files,
    and a planted `runtime: vllm` or `bind_all: false` in one file changes what the other means).
    `file` is the file whose value wins (overrides.yaml beats config.yaml), `shown` is
    `<runtime>:<true|false>:<keyed|open>`. Only for a vllm or llamacpp runtime — Ollama and LM
    Studio's `bind_all` is inert. Peers: the `extra_endpoints` list of overrides.yaml when it has one
    (lists replace wholesale), else config.yaml's; each peer is judged by its own `lifecycle`."""
    from localharness.config.overlay import deep_merge

    def section(raw: Any, key: str) -> dict:
        value = raw.get(key) if isinstance(raw, dict) else None
        return value if isinstance(value, dict) else {}

    def entry(file: str, prefix: str, key: str, view: dict, keyed: Any) -> list[dict]:
        runtime = view.get("runtime", "vllm")
        if runtime not in ("vllm", "llamacpp") or view.get(key) is None:
            return []
        value = coerce_bool(view[key])  # the model's reading: `1`, "yes", "on" are `true` (CR-02)
        spelled = json.dumps(value if value is not None else view[key])
        return [{"file": file, "kind": LAUNCH_KIND, "name": prefix + key,
                 "shown": f"{runtime}:{spelled}:{keyed(runtime)}"}]

    cfg, over = raws.get("config.yaml"), raws.get("overrides.yaml")
    over_server = section(over, "server")
    merged = deep_merge(section(cfg, "server"), over_server)
    out: list[dict] = []
    for key in _POSTURE_KEYS:
        file = "overrides.yaml" if key in over_server else "config.yaml"
        out += entry(file, "server.", key, merged, lambda rt: _key_state(merged, raws, rt))
    over_peers = over.get("extra_endpoints") if isinstance(over, dict) else None
    if isinstance(over_peers, list):
        peers, pfile = over_peers, "overrides.yaml"
    else:
        cfg_peers = cfg.get("extra_endpoints") if isinstance(cfg, dict) else None
        peers, pfile = (cfg_peers if isinstance(cfg_peers, list) else []), "config.yaml"
    seen: set[str] = set()
    for peer in peers:
        if not (isinstance(peer, dict) and peer.get("name") is not None
                and isinstance(peer.get("lifecycle"), dict)) or str(peer["name"]) in seen:
            continue
        seen.add(str(peer["name"]))
        life = peer["lifecycle"]
        for key in _POSTURE_KEYS:
            out += entry(pfile, f"extra_endpoints.{peer['name']}.lifecycle.", key, life,
                         lambda rt, life=life, peer=peer: _key_state(
                             life, raws, rt, peer_key=peer.get("api_key")))
    return out


def _launch_entries(file: str, prefix: str, mapping: Any) -> list[dict]:
    """One entry per launch key present (and not None) in ``mapping``: `launch`-kind for
    LAUNCH_KEYS, `launch_model`-kind for LAUNCH_MODEL_KEYS, named ``prefix + key``, ``shown`` the
    value when it is a string, else compact JSON (so `false` and `0` are entries; an absent key is
    not). An empty `local_models` list is no entry. `bind_all` and `require_api_key` are emitted
    from the merged view by `machine_snapshot`, never here."""
    if not isinstance(mapping, dict):
        return []
    out: list[dict] = []
    for kind, keys in ((LAUNCH_KIND, LAUNCH_KEYS), (LAUNCH_MODEL_KIND, LAUNCH_MODEL_KEYS)):
        for key in keys:
            value = mapping.get(key)
            if value is None or key in _POSTURE_KEYS or (key == "local_models" and value == []):
                continue
            out.append({"file": file, "kind": kind, "name": prefix + key,
                        "shown": value if isinstance(value, str) else json.dumps(
                            value, separators=(",", ":"), default=str)})
    return out


def autostart_entries(global_dir: Path) -> list[dict]:
    """The autostart unit's captured environment as zero or one `launch` entry
    ({file: AUTOSTART_FILE, kind: "launch", name: "autostart.env", shown: compact JSON of the
    PATH / HF_HOME / LD_LIBRARY_PATH triple in that fixed order}): present only when the harness unit
    file exists, its `ExecStart` names ``global_dir`` as `--config-dir` and its `# env` block parses;
    otherwise none. The values become the exec environment of `serve` at boot, so a change to them is
    a change to what the machine's files start.

    `localharness.provider.units` is imported inside the function: the config layer never imports the
    provider layer at import time."""
    from localharness.provider import units

    return autostart_entries_for_body(global_dir, units.read_harness_unit())


def autostart_entries_for_body(global_dir: Path, body: str | None) -> list[dict]:
    """`autostart_entries` for a unit text the caller already holds instead of the file on disk: the entry
    ``body`` itself would produce, or none when ``body`` is None, does not name ``global_dir`` as
    `--config-dir` or has an unparsable `# env` block. `--autostart on` adopts through this with the text it
    just wrote, so the record changes by exactly what the act wrote and never by what a later reader finds
    on disk."""
    from localharness.provider import units

    if body is None or not units.unit_serves(global_dir, body):
        return []
    captured = units.captured_env(body)
    if captured is None:
        return []
    return _launch_entries(AUTOSTART_FILE, "autostart.", {"env": captured})


AUTOSTART_ENV_MISSING = "no environment is recorded for it"
AUTOSTART_ENV_UNREADABLE = "its # env block cannot be read"
AUTOSTART_ENV_CHANGED = "its # env block is not the one recorded for this machine"


def autostart_env_failure(global_dir: Path) -> str | None:
    """None when the machine record covers the autostart unit's captured environment exactly as `serve` will
    ask for it at its start: the unit file reads, its `# env` block parses, the recorded `autostart.env`
    entry exists and holds the very values the file holds now. Else the phrase naming what is missing. This
    is the check `serve` makes before it execs (exit 78 on a miss), read-only and without a trust decision,
    so a surface that is about to restart the unit can refuse BEFORE it stops anything. A recorded entry the
    person declined is the old one and fails the comparison, as a withheld one does at `serve`."""
    from localharness.provider import units

    body = units.read_harness_unit()
    captured = units.captured_env(body) if body is not None else None
    if captured is None:
        return AUTOSTART_ENV_UNREADABLE
    recorded = recorded_autostart_entry(global_dir)
    if recorded is None or not autostart_entries_for_body(global_dir, body):
        return AUTOSTART_ENV_MISSING
    try:
        confirmed = json.loads(recorded.get("shown", ""))
    except (TypeError, ValueError):
        confirmed = None
    return None if confirmed == captured else AUTOSTART_ENV_CHANGED


def machine_snapshot(global_dir: Path) -> list[dict]:
    """What your machine's own files would start, load or loosen: for every `<global>/agents/*.yaml`
    each `tools.mcp_servers` entry ({kind: "mcp_server", file: "agents/<f>", name, shown: command +
    args or url, env: [names], headers: [names]}) and its `memory.embedding_model` when set
    ({kind: "embedding_model", name: "memory.embedding_model", shown: value}); for those files,
    every `<global>/divisions/*.yaml` and `<global>/org.yaml`, each permission loosening
    ({kind: "permission", name: <dotted key>, shown: <value>}, loader.permission_loosenings — the
    legacy org.yaml as the base rung, whose shorter `deny_patterns` does drop shipped patterns); and
    every script in `<global>/tools/` (tool_script_entries — a shell command runs one, so the gate
    treats an unconfirmed one as a shell command it has not seen). Also the `agent:` section of
    `<global>/overrides.yaml` — every agent's default layer — read like an agent file (file
    "overrides.yaml"); and in `<global>/config.yaml` and `overrides.yaml` each loosening of the
    `org:` section's permissions ({kind: "org_permission", name: "org.permissions.<key>"}) and each
    launch key of the `server:` section ({kind: "launch", name: "server.binary" …} for LAUNCH_KEYS,
    {kind: "launch_model", name: "server.model" …} for LAUNCH_MODEL_KEYS) and of every
    `extra_endpoints` peer's `lifecycle:` ({name: "extra_endpoints.<peer name>.lifecycle.<key>"} —
    by name, never by index; the peer's `api_key` and `extra_headers` are siblings of `lifecycle` and
    contribute no entry by construction). `bind_all` and `require_api_key` are the exception to
    per-file emission: one entry each per record from the merged view (`_posture_entries`). The
    autostart unit's captured `# env` block is one more `launch` entry (`autostart_entries`) for the
    config dir the unit serves, not a kind of its own: a new kind is adopted silently once.
    Sorted by (file, kind, name, canonical JSON); unparsable files skipped; never an env or header
    value."""
    from localharness.config.loader import layer_files, org_deny_loosenings, permission_loosenings

    root = Path(global_dir)

    def loosenings(file: str, raw: Any) -> list[dict]:
        perms = raw.get("permissions") if isinstance(raw, dict) else None
        base = org_deny_loosenings(perms) if file == "org.yaml" else []
        return [{"file": file, "kind": "permission", "name": key, "shown": shown}
                for key, shown in sorted(permission_loosenings(perms) + base)]

    def agent_entries(file: str, raw: Any) -> list[dict]:
        found = [{"file": file, "kind": "mcp_server", "name": str(s.get("name") or ""),
                  "shown": _shown_server(s), "env": _value_digests(s.get("env")),
                  "headers": _value_digests(s.get("headers"))} for s in _servers(raw)]
        memory = raw.get("memory") if isinstance(raw, dict) else None
        if isinstance(memory, dict) and memory.get("embedding_model") is not None:
            found.append({"file": file, "kind": "embedding_model", "name": "memory.embedding_model",
                          "shown": str(memory["embedding_model"])})
        return found + loosenings(file, raw)

    out: list[dict] = []
    for path in layer_files(root, "agents"):
        out += agent_entries(f"agents/{path.name}", _read_yaml(path))
    # The `agent:` section of overrides.yaml is every agent's default layer (the loader merges it
    # under each agent file), so what it starts, loads or loosens is fingerprinted the same way:
    # a server put there — by `python_exec`, or a shell write the settings-file rule does not read —
    # waits for the same one Yes as a server put in an agent file. (config.yaml cannot carry one.)
    overrides = _read_yaml(root / "overrides.yaml") if (root / "overrides.yaml").is_file() else None
    if isinstance(overrides, dict) and isinstance(overrides.get("agent"), dict):
        out += agent_entries("overrides.yaml", overrides["agent"])
    # The two settings files themselves: the `org:` section's permissions reach every agent (its
    # mode is the one an agent with none of its own runs in; its ask lists feed every gate), and
    # the `server:` section names what start executes when the model server is down. Code the
    # agent runs can write either file, so a loosening or a new launch command there waits for the
    # same one Yes. (A shorter org deny list removes nothing — the shipped list is unioned in.)
    raws = {file: _read_yaml(root / file) if (root / file).is_file() else None
            for file in ("config.yaml", "overrides.yaml")}
    for file, raw in raws.items():
        org = raw.get("org") if isinstance(raw, dict) else None
        out += [{"file": file, "kind": ORG_PERMISSION_KIND, "name": f"org.{key}", "shown": shown}
                for key, shown in permission_loosenings(
                    org.get("permissions") if isinstance(org, dict) else None)]
        out += _launch_entries(file, "server.", raw.get("server") if isinstance(raw, dict) else None)
        # A peer endpoint's `lifecycle:` is the same launch decision as the primary's, named by the
        # peer's `name` (never its list position). Its `api_key` and `extra_headers` are siblings
        # of `lifecycle`, so no secret can reach an entry: they are never read here.
        for peer in (raw.get("extra_endpoints") if isinstance(raw, dict) else None) or []:
            if (isinstance(peer, dict) and peer.get("name") is not None
                    and isinstance(peer.get("lifecycle"), dict)):
                out += _launch_entries(
                    file, f"extra_endpoints.{peer['name']}.lifecycle.", peer["lifecycle"])
    out += _posture_entries(raws)
    for path in layer_files(root, "divisions"):
        out += loosenings(f"divisions/{path.name}", _read_yaml(path))
    if (root / "org.yaml").is_file():
        out += loosenings("org.yaml", _read_yaml(root / "org.yaml"))
    out += tool_script_entries(global_dir)
    out += autostart_entries(global_dir)
    return sorted(out, key=_entry_key)


def confirmed_at_upgrade(snap: list[dict]) -> list[dict]:
    """The entries of ``snap`` that governed what 0.17 launched and exposed, and so are confirmed
    ONCE at the first start that knows them: the `launch`-kind entries whose key (the part after
    the last `.`) is `runtime`, `launch`, `bind_all`, `require_api_key` or `local_models` — the
    primary's and every peer's — except a llama.cpp `bind_all` posture entry that is not exactly
    `false` (`shown` starts `llamacpp:true:`, or any spelling the posture entry could not normalise):
    0.17 ignored it entirely, so it cannot have been relied on and is never adopted unasked. The test
    fails closed: only `llamacpp:false:` is confirmed.

    A pure filter over the snapshot the caller already holds: no second snapshot function, no disk
    read. Entries are compared whole afterwards (`uncovered`)."""
    governing = ("runtime", "launch", "bind_all", "require_api_key", "local_models")
    return [e for e in snap
            if e.get("kind") == LAUNCH_KIND and str(e.get("name", "")).rsplit(".", 1)[-1] in governing
            and not (str(e.get("name", "")).rsplit(".", 1)[-1] == "bind_all"
                     and str(e.get("shown", "")).startswith("llamacpp:")
                     and not str(e.get("shown", "")).startswith("llamacpp:false:"))]


def recorded_machine(global_dir: Path) -> Optional[dict]:
    """This machine's record {fingerprint, entries, kinds}, or None (the first start after the
    upgrade). Malformed parts read as empty."""
    rec = _load(trust_store_path()).get(machine_key(global_dir))
    if not isinstance(rec, dict):
        return None
    entries, kinds = rec.get("entries"), rec.get("kinds")
    return {"fingerprint": rec["fingerprint"] if isinstance(rec.get("fingerprint"), str) else "",
            "entries": [e for e in entries if isinstance(e, dict)] if isinstance(entries, list) else [],
            "kinds": [str(k) for k in kinds] if isinstance(kinds, list) else []}


def record_machine(global_dir: Path, entries: list[dict],
                   kinds: frozenset[str] = MACHINE_KINDS) -> None:
    """Store what was confirmed (or adopted) for this machine, through the same 0600 atomic write."""
    data = _load(trust_store_path())
    data[machine_key(global_dir)] = {"fingerprint": fingerprint(entries), "entries": entries,
                                     "kinds": sorted(kinds), "recorded": _now()}
    atomic_write_overlay(trust_store_path(), data)


def recorded_autostart_entry(global_dir: Path) -> Optional[dict]:
    """The recorded `autostart.env` entry (the one whose `(file, kind, name)` is AUTOSTART_KEY) of this
    machine's record, or None when there is no record or it holds none. What `serve` compares the
    unit's own captured environment against."""
    rec = recorded_machine(global_dir)
    if rec is None:
        return None
    return next((e for e in rec["entries"]
                 if (e.get("file"), e.get("kind"), e.get("name")) == AUTOSTART_KEY), None)


def adopt_entries(global_dir: Path, entries: list[dict]) -> None:
    """Re-record an act's own write from the values it wrote: each of ``entries`` replaces the
    recorded entry with the same (file, kind, name) or is added; `kinds` is kept; the list is
    re-sorted as `machine_snapshot` sorts and re-fingerprinted; the write is `record_machine`'s
    0600 atomic one. A no-op when this machine has no record yet (the first start adopts the whole
    disk anyway).

    It never calls `machine_snapshot`. A re-snapshot would adopt whatever else is on disk at that
    moment — an agent's write made in the meantime — and the record must change by exactly what
    the act wrote."""
    rec = recorded_machine(global_dir)
    if rec is None:
        return
    written = {(n["file"], n["kind"], n["name"]) for n in entries}
    keep = [e for e in rec["entries"] if (e.get("file"), e.get("kind"), e.get("name")) not in written]
    record_machine(global_dir, sorted([*keep, *entries], key=_entry_key),
                   kinds=frozenset(rec["kinds"]))


def forget_entries(global_dir: Path, keys: Iterable[tuple[str, str, str]]) -> None:
    """Remove exactly the recorded entries whose `(file, kind, name)` is in ``keys`` and re-record
    the rest: `kinds` kept, the list re-sorted as `machine_snapshot` sorts, re-fingerprinted, written
    by `record_machine`'s 0600 atomic write. The mirror of `adopt_entries` — the record changes by
    exactly what the act removed — used by `off` to prune the entry it made obsolete. A no-op when
    this machine has no record. It never calls `machine_snapshot`."""
    rec = recorded_machine(global_dir)
    if rec is None:
        return
    gone = set(keys)
    keep = [e for e in rec["entries"] if (e.get("file"), e.get("kind"), e.get("name")) not in gone]
    if len(keep) == len(rec["entries"]):
        return
    record_machine(global_dir, sorted(keep, key=_entry_key), kinds=frozenset(rec["kinds"]))


def decline_key(kind: str, *parts: str) -> str:
    """The key of a remembered No that is not about a workspace directory: ``kind`` followed by
    ``parts`` verbatim. The three variants: `decline_key("<binary>", <resolved path>)` for a No on
    a `PATH`-found binary, `decline_key("<peer>", <entry name>, <shown>)` for a peer's posture No
    (a changed value is a new question), `decline_key("<config.ini>", <digest>)` for a sensed
    `config.ini` No."""
    return kind + "".join(parts)


def offer_was_declined(target: Path | str) -> bool:
    """Has the user already said no to creating THIS workspace (a ``Path``), or to the question
    named by a `decline_key` (a ``str``, used verbatim)?

    False when unreadable, missing or corrupt — the fail-open direction here, deliberately, and
    the opposite of `is_trusted`'s: the worst case is one repeated question, never config loaded
    without consent. Nothing else reads this file, so it can only ever cost a prompt. A remembered
    No can suppress a question; it never causes a launch.
    """
    entry = _load(declined_offers_path()).get(target if isinstance(target, str) else _key(target))
    return isinstance(entry, dict) and entry.get("declined") is True


def record_offer_decline(target: Path | str) -> None:
    """Remember a "no" so the offer is asked once per directory (a ``Path``) or once per
    `decline_key` (a ``str``), ever.

    Only a person answering the prompt gets here — EOF and every non-interactive path record
    nothing, exactly as an unanswered trust prompt does. Undo it by deleting the entry (or the
    file); `init --workspace` and creating the directory by hand ignore this store entirely, so a
    recorded "no" can never stand between a user and a workspace they went and asked for.
    """
    data = _load(declined_offers_path())
    data[target if isinstance(target, str) else _key(target)] = {"declined": True}
    atomic_write_overlay(declined_offers_path(), data)
