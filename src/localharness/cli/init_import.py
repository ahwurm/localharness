"""The one translation from a running (or journalled, or unit-file) llama-server to a llama.cpp record.

Pure: `import_record` takes a `procfs.ProcRecord` and returns data, a validated `ManagedServerConfig`
mapping with its window input and the lines `init` shows before it adopts anything, or the one reason the
process can only be attached to. It reads the tables in `provider/server.py` (`excluded_option`,
`classify_hosts`, `llama_annotation`, `RUNTIME_LABELS`); there is no second list here. It performs no I/O of
its own beyond the injected readers: the binary resolver and, for a download identity, the GGUF header
reader behind `gguf_header.served_model`. A key file a process names is recorded, never opened (D-18).
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pydantic import SecretStr, ValidationError

from localharness.agent.gguf_header import preset_name, served_model
from localharness.cli.workspace import shown_path, visible_text
from localharness.config.models import ManagedServerConfig
from localharness.core.private_files import open_regular
from localharness.provider.procfs import ProcRecord
from localharness.provider.server import (
    LLAMA_EXCLUDED_OPTIONS,
    RUNTIME_LABELS,
    classify_hosts,
    effective_posture,
    excluded_option,
    launch_decision,
    llama_annotation,
)

ATTACH_ONLY_SUFFIX = " — attached only, no server record written"
"""The tail every attach-only sentence carries after its `<who> <reason>` (D-25)."""

HF_CACHE_SHOWN = "~/.cache/huggingface/hub"
"""The download cache as the attach-only reason names it (D-19(1))."""

VERBOSE_DROPPED = ("--verbose dropped: llama-server logs the key's last four characters; "
                   "add it under server.extra_args to keep the log")
"""The shown line for `-v`/`--verbose`/`--log-verbose` removed from a keyed record (D-19(6))."""

UNIT_LOOPBACK_LINE = ("{unit} binds {host} with no key file — imported on 127.0.0.1 · "
                      "server.bind_all: true beside server.api_key_file restores it (asks once)")
"""The shown line for a journal or unit-file record whose non-loopback bind is not imported (D-23)."""

KEY_FILE_LINE = "key file: {path} (read at start, never by init)"
"""The shown line for a key file the process names: recorded, never opened (D-19(2))."""

BIND_ALL_LINE = "server.bind_all: true (--host {host})"
"""The shown line for a widened bind the exposure phrase does not cover (a keyed or key-filed record)."""

CONFIG_INI_CAP = 65536
"""The most bytes of `config.ini` the import reads (D-28); one byte more is read to tell the cap from over it."""

DEFAULT_PORT = 8080
"""llama-server's own default: a process that names no port serves there."""

# argv flag -> the field it fills; every spelling llama.cpp accepts, `--x_y` already read as `--x-y`.
_ARGV_FIELDS: dict[str, str] = {
    "-m": "model", "--model": "model",
    "-a": "alias", "--alias": "alias",
    "-c": "ctx", "--ctx-size": "ctx",
    "-ctk": "ctk", "--cache-type-k": "ctk",
    "-ctv": "ctv", "--cache-type-v": "ctv",
    "-cram": "cram", "--cache-ram": "cram",
    "-mm": "mmproj", "--mmproj": "mmproj",
    "--host": "host", "--port": "port",
    "--api-key": "api_key", "--api-key-file": "key_file",
    "-hft": "hf_token", "--hf-token": "hf_token",
}
_VERBOSE = frozenset({"-v", "--verbose", "--log-verbose"})
_HF_FILE_FLAGS = frozenset({"-hff", "--hf-file"})
_HF_FILE_ENV = "LLAMA_ARG_HF_FILE"
_KEY_ENV = "LLAMA_API_KEY"
_KEY_FILE_ENV = "LLAMA_ARG_API_KEY_FILE"

# the LLAMA_ARG_* names that fill the same fields when argv did not
_ENV_FIELDS: dict[str, str] = {
    "LLAMA_ARG_MODEL": "model",
    "LLAMA_ARG_ALIAS": "alias",
    "LLAMA_ARG_CTX_SIZE": "ctx",
    "LLAMA_ARG_CACHE_TYPE_K": "ctk",
    "LLAMA_ARG_CACHE_TYPE_V": "ctv",
    "LLAMA_ARG_CACHE_RAM": "cram",
    "LLAMA_ARG_MMPROJ": "mmproj",
    "LLAMA_ARG_HOST": "host",
    "LLAMA_ARG_PORT": "port",
}

_ENV_PREFIX = "LLAMA_ARG_"

_SECRETISH = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|CRED", re.IGNORECASE)
"""A flag, `LLAMA_ARG_*` name or `config.ini` key that reads as a credential. The table in
`provider/server.py` names the credentials llama-server really reads; this is the net under it, so a
name the table does not list is still never carried, shown or audited (D-19(2): anything key-shaped left
makes the process attach-only)."""

_DRAFT_DOWNLOAD_ENV = "LLAMA_ARG_SPEC_DRAFT_HF_REPO"
_DRAFT_DOWNLOAD_FLAGS = frozenset(flag for _, flags, env in LLAMA_EXCLUDED_OPTIONS
                                  if env == _DRAFT_DOWNLOAD_ENV for flag in flags)
_DRAFT_DOWNLOAD_NAMES = frozenset({flag.lstrip("-").lower() for flag in _DRAFT_DOWNLOAD_FLAGS}
                                  | {_DRAFT_DOWNLOAD_ENV.replace("_", "-").lower()})
"""The server's own download rows for the draft model (`-hfd`, `--hf-repo-draft`, `--spec-draft-hf`...):
read from the one table, not respelled; the second name is each spelling dash-stripped and lower-cased,
the form a flag, an env name and a config.ini key are compared in."""

_DRAFT_MODEL_FLAGS = frozenset({"-md", "--model-draft"})
_DRAFT_MODEL_ENV = "LLAMA_ARG_MODEL_DRAFT"

_CREDENTIAL_VALUED = frozenset({"mcp-servers-json"})
"""Options whose VALUE is a document that can embed credentials (`headers`, `env` of an MCP server
definition). D-27 annotates them for display only and refuses nothing for the annotation; a value that can
hold a bearer token is refused under D-19(2) all the same, so it never reaches a shown line, the record
or an audit entry."""


def _credential_name(name: str) -> bool:
    """True when `name` (a flag with its dashes, an env name or a config.ini key) reads as a credential or
    takes a credential-bearing document as its value."""
    if _SECRETISH.search(name):
        return True
    bare = name.lstrip("-").lower().replace("_", "-").removeprefix("llama-arg-")
    return bare in _CREDENTIAL_VALUED

_RUNTIME_BY_NAME = {
    "llama-server": "llamacpp",
    "ollama": "ollama",
    "lms": "lmstudio",
    "llmster": "lmstudio",
}


@dataclass(frozen=True)
class Imported:
    """A process that translates to a record. `spec` is validated; `api_key` is the one inline key the
    process held (never in `spec`, never in `shown`); `shown` are the lines `init` prints before it
    adopts; `ctx` is the `-c` the process served (the window input, never `spec.context`, D-29);
    `binary_source` is `"process"` when the binary came from the process's own exe or a path argv[0],
    `"path"` when the resolver found it, None when it stays unset."""

    spec: ManagedServerConfig
    api_key: SecretStr | None
    shown: tuple[str, ...]
    ctx: int | None
    binary_source: str | None


@dataclass(frozen=True)
class AttachOnly:
    """Why the process is attached to and no record is written; `reason` is the text the sentence
    `<who> <reason>` + `ATTACH_ONLY_SUFFIX` carries (D-25)."""

    reason: str


@dataclass(frozen=True)
class IniImport:
    """`config.ini` after masking (D-28). `kept` is the text a record may carry (no key, key file,
    Hugging Face token, host or port line); `keys` holds the values that are routed to the import's own
    key and bind handling, under the key as written (`api-key`, `LLAMA_API_KEY`, `api-key-file`,
    `LLAMA_ARG_API_KEY_FILE`, `host`, `port`, ...); `refused` is the first download-identity or TLS key as
    written, which makes the import attach-only; `keyshaped` is the first other key that reads as a
    credential (`_credential_name`), left out of `kept` and also attach-only at import."""

    kept: str
    keys: dict[str, str]
    refused: str | None
    keyshaped: str | None = None


@dataclass(frozen=True)
class IniRead:
    """The outcome of reading `config.ini`: `dir` is its directory as shown, `ini` the masked content
    (None when absent, unreadable, over the cap or not text) and `line` the one line to show when it was
    present but not carried (D-56)."""

    dir: str
    ini: IniImport | None
    line: str | None


def _basename(path: str | None) -> str:
    return os.path.basename(path) if path else ""


def runtime_of(rec: ProcRecord) -> str | None:
    """`"llamacpp"`, `"ollama"` or `"lmstudio"` from the basename of the process's exe, else of its
    argv[0] (a ` (deleted)` exe still names what it was); None for anything else."""
    candidates = (_basename(rec.exe).removesuffix(" (deleted)"), _basename(rec.argv[0] if rec.argv else None))
    for name in candidates:
        runtime = _RUNTIME_BY_NAME.get(name)
        if runtime is not None:
            return runtime
    return None


@dataclass
class _Walk:
    """What one pass over argv found: the field values (last wins), what stripped out of the argument
    list on the way (keys, key files, download identity, TLS flags) and the tokens nothing claimed."""

    fields: dict[str, str]
    extra: list[str]
    api_keys: list[str]
    key_files: list[tuple[str, str]]
    downloads: list[str]
    tls: list[str]
    hf_file: str | None = None
    verbose: list[int] | None = None
    keyshaped: str | None = None
    dangling: str | None = None


def _walk(tokens: tuple[str, ...]) -> _Walk:
    """Walk argv after argv[0]. A `--` token is looked up with `_` read as `-` and an `=value` split
    off (llama.cpp's own rule); a token nothing claims stays in `extra` exactly as written. The table
    of excluded options is the server's: download identity and TLS flags are found through it."""
    walk = _Walk(fields={}, extra=[], api_keys=[], key_files=[], downloads=[], tls=[], verbose=[])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok.startswith("-"):
            walk.extra.append(tok)
            continue
        written, eq, rest = tok.partition("=")  # a single-dash token splits on `=` too: `-hft=<token>`
        name = written.replace("_", "-") if tok.startswith("--") else written
        inline = rest if eq else None
        field = _ARGV_FIELDS.get(name)
        group = None
        if field is None:
            category = excluded_option(name.lstrip("-"))
            if category in ("download", "tls"):
                group = category
            elif name in _VERBOSE:
                walk.extra.append(tok)
                walk.verbose.append(len(walk.extra) - 1)
                continue
            else:
                if walk.keyshaped is None and _credential_name(name):
                    walk.keyshaped = written
                walk.extra.append(tok)
                continue
        if inline is not None:
            value = inline
        elif i < len(tokens):
            value = tokens[i]
            i += 1
        else:
            walk.dangling = written
            continue
        if group == "download":
            walk.downloads.append(written)
            if name in _HF_FILE_FLAGS:
                walk.hf_file = value
        elif group == "tls":
            walk.tls.append(written)
        elif field == "api_key":
            walk.api_keys.append(value)
        elif field == "key_file":
            walk.key_files.append((written, value))
        elif field == "hf_token":
            pass  # a download credential: never carried, never shown
        elif field == "cram" and not _plain_ram(value):
            # `-cram -1` (no limit) is not a value the record's field can carry: it stays an argument
            walk.extra.append(tok)
            if inline is None:
                walk.extra.append(value)
        else:
            walk.fields[field] = value
    return walk


def _join_cwd(path: str, cwd: str | None) -> str | None:
    """`path` as the process saw it: absolute stays, a relative one is joined to the process cwd, or
    None when the cwd is unknown (a journal record)."""
    if os.path.isabs(path):
        return path
    if cwd is None:
        return None
    while path.startswith("./"):
        path = path[2:]
    return os.path.join(cwd, path)


def _binary(rec: ProcRecord, resolver: Callable[[str], str | None]) -> tuple[str | None, str | None]:
    """(binary, source) per D-19(5): the exe when its basename is argv[0]'s; else a bare argv[0]
    through `resolver` (never the process cwd); else argv[0] with a directory part joined to the
    process cwd; unresolvable is unset."""
    argv0 = rec.argv[0] if rec.argv else ""
    if rec.exe and argv0 and _basename(rec.exe) == _basename(argv0):
        return rec.exe, "process"
    if not argv0:
        return None, None
    if os.path.dirname(argv0):
        joined = _join_cwd(argv0, rec.cwd)
        return (joined, "process") if joined is not None else (None, None)
    found = resolver(argv0)
    return (found, "path") if found else (None, None)


def _plain_ram(text: str) -> bool:
    """A `--cache-ram` value `ManagedServerConfig.cache_ram` can hold: plain ASCII digits, at most nine
    (a longer string is not a MiB figure, and `int()` of thousands of digits raises)."""
    return text.isascii() and text.isdigit() and len(text) <= 9


def _is_draft_download(name: str) -> bool:
    """`-hfd`, `--hf_repo_draft`, `LLAMA_ARG_SPEC_DRAFT_HF_REPO`, or the config.ini key of one."""
    return name.lstrip("-").replace("_", "-").lower() in _DRAFT_DOWNLOAD_NAMES


def _draft_paths(extra: list[str], environ: "dict[str, str] | object", cwd: str | None) -> set[str]:
    """The local draft model a process names (`-md`, `--model-draft`, `LLAMA_ARG_MODEL_DRAFT`) as the paths
    `/proc/<pid>/maps` would list it under, so the served-model search can pass it over (WR-07). Maps is
    address ordered, a later mapping sits lower, and the draft is mapped after the main model."""
    values: list[str] = []
    for i, tok in enumerate(extra):
        flag, eq, rest = tok.partition("=")
        flag = flag.replace("_", "-") if flag.startswith("--") else flag
        if flag in _DRAFT_MODEL_FLAGS:
            if eq:
                values.append(rest)
            elif i + 1 < len(extra):
                values.append(extra[i + 1])
    named = environ.get(_DRAFT_MODEL_ENV) if hasattr(environ, "get") else None
    if named:
        values.append(named)
    paths = set()
    for value in values:
        joined = _join_cwd(value, cwd)
        if joined is not None:
            paths.add(os.path.normpath(joined))
    return paths


def _int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _cannot_carry(field: str) -> AttachOnly:
    return AttachOnly(f"holds a value a record cannot carry ({visible_text(field)})")


# ------------------------------------------------------------------------------- config.ini
#
# llama.cpp's own grammar (common/preset.cpp): `[section]` headers, `key = value`, whole-line and trailing
# `;`/`#` comments (a comment starts at whitespace then `;`/`#`), CRLF / CR / LF all newlines; a key is an
# env name or a flag with its dashes removed. Only the `*` and `default` sections (and the lines before any
# header) apply to a plain llama-server.

_INI_NEWLINE = re.compile(r"\r\n|\r|\n")
_INI_HEADER = re.compile(r"[ \t]*\[([^\]]*)\]")
_INI_KV = re.compile(r"[ \t]*([A-Za-z_][A-Za-z0-9_.-]*)[ \t]*=[ \t]*((?:(?![ \t]*[;#]).)*)")
_INI_ACTIVE_SECTIONS = frozenset({"", "*", "default"})
_INI_INLINE_KEYS = frozenset({"api-key", "llama_api_key"})
_INI_KEY_FILE_KEYS = frozenset({"api-key-file", "llama_arg_api_key_file"})
_INI_TOKEN_KEYS = frozenset({"hf-token", "hft"})


def mask_config_ini(data: bytes) -> IniImport:
    """Mask `config.ini` text (D-28). The `api-key`, `api-key-file` and Hugging Face token lines (env and
    flag spellings, any letter case) leave `kept` and every shown line: the first two route their value
    through `keys`, the token is dropped. `host` and `port` lines leave `kept` too (the record owns the
    bind, and the `config_ini` field refuses them) and ride `keys` to the host and port resolution. A
    download-identity or TLS key is named in `refused`. Everything else is kept verbatim. Text that is not
    UTF-8 masks to nothing."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return IniImport("", {}, None)
    kept: list[str] = []
    keys: dict[str, str] = {}
    refused: str | None = None
    keyshaped: str | None = None
    section = ""
    for raw in _INI_NEWLINE.split(text):
        header = _INI_HEADER.match(raw)
        if header is not None:
            section = header.group(1).strip()
            kept.append(raw)
            continue
        kv = _INI_KV.match(raw)
        if kv is not None:
            key, value = kv.group(1), kv.group(2).rstrip(" \t")  # llama.cpp's value excludes trailing blanks
            lowered = key.lower()
            category = excluded_option(key)
            dashed = lowered.replace("_", "-")
            if category == "key":
                if section in _INI_ACTIVE_SECTIONS and (lowered in _INI_INLINE_KEYS
                                                         or lowered in _INI_KEY_FILE_KEYS):
                    keys[key] = value
                continue  # LLAMA_ARG_API_KEY is read by nothing: dropped
            if category == "host/port":
                if section in _INI_ACTIVE_SECTIONS:
                    keys[key] = value
                continue
            if category in ("download", "tls"):
                refused = refused or key
                continue
            if dashed in _INI_TOKEN_KEYS or "api-key" in dashed:
                continue  # a credential llama-server would not read from this spelling: still never kept
            if _credential_name(key):
                keyshaped = keyshaped or key  # the import cannot place it: attach-only, never kept or shown
                continue
        kept.append(raw)
    while kept and not kept[-1].strip():
        kept.pop()
    return IniImport("\n".join(kept) + "\n" if any(line.strip() for line in kept) else "", keys, refused,
                     keyshaped)


def user_config_ini() -> Path:
    """`${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini`, the file llama-server applies before its
    environment and arguments."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path(os.path.expanduser("~")) / ".config"
    return root / "llama.cpp" / "config.ini"


def read_config_ini(path: str | os.PathLike[str]) -> IniRead:
    """Read `path` through `open_regular` (a FIFO or device is treated as absent without blocking), under
    `CONFIG_INI_CAP`. Absent or unreadable is `IniRead(dir, None, None)`: nothing shown, nothing recorded.
    Over the cap or not UTF-8 text is not carried, with the one line that says so (D-56)."""
    target = Path(path)
    shown_dir = shown_path(target.parent)
    try:
        fd = open_regular(target)
    except OSError:
        return IniRead(shown_dir, None, None)
    try:
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(CONFIG_INI_CAP + 1)
    except OSError:
        return IniRead(shown_dir, None, None)
    if len(data) > CONFIG_INI_CAP:
        return IniRead(shown_dir, None, f"config.ini ({shown_dir}) is over {CONFIG_INI_CAP} bytes — not carried")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return IniRead(shown_dir, None, f"config.ini ({shown_dir}) is not UTF-8 text — not carried")
    return IniRead(shown_dir, mask_config_ini(data), None)


def _ini_pairs(kept: str) -> list[tuple[str, str]]:
    """The `key=value` pairs of masked `config.ini` text, for the one shown line."""
    pairs = []
    for line in _INI_NEWLINE.split(kept):
        kv = _INI_KV.match(line)
        if kv is not None and _INI_HEADER.match(line) is None:
            pairs.append((kv.group(1), kv.group(2).rstrip(" \t")))
    return pairs


def _split_keys(value: str) -> list[str]:
    """llama-server splits an `--api-key` / `LLAMA_API_KEY` value on commas; empty entries are no key."""
    return [part for part in (piece.strip() for piece in value.split(",")) if part]


def _annotated(name: str, suffix: str | None) -> str:
    return name if suffix is None else f"{name} {suffix}"


def import_record(
    rec: ProcRecord,
    *,
    source: str,
    binary_resolver: Callable[[str], str | None],
    port: int | None = None,
    ini: IniRead | None = None,
) -> Imported | AttachOnly:
    """`_import_record`, with a value no step could place becoming the attach-only line
    `holds a value a record cannot carry` and never a traceback: `init` calls this after it has detected
    a process and before it writes a file (WR-01)."""
    try:
        return _import_record(rec, source=source, binary_resolver=binary_resolver, port=port, ini=ini)
    except (ValueError, OverflowError):
        return AttachOnly("holds a value a record cannot carry (record)")


def _import_record(
    rec: ProcRecord,
    *,
    source: str,
    binary_resolver: Callable[[str], str | None],
    port: int | None = None,
    ini: IniRead | None = None,
) -> Imported | AttachOnly:
    """Translate `rec` (a live process, a journal entry or a unit file, named by `source`) into a
    record, or into the one reason it is attached to only. `port` is the listener port the caller found
    the process on, used when the process names none; `ini` is `config.ini` read and masked by
    `read_config_ini`, whose keys join the key, host and port resolution at the lowest precedence.

    Order (D-19), all before validation: TLS; download identity or no local `-m`; keys; hosts; the
    field table; the `--verbose` family. The key a process holds goes into `Imported.api_key` and
    nowhere else; a key file is recorded as a path and never opened."""
    runtime = runtime_of(rec)
    if runtime in ("ollama", "lmstudio"):
        return AttachOnly(f"runs {RUNTIME_LABELS[runtime]}, which cannot carry a key")
    if runtime != "llamacpp":
        return AttachOnly(f"runs {visible_text(_basename(rec.argv[0] if rec.argv else rec.exe))}, "
                          "which the import does not translate")
    walk = _walk(rec.argv[1:])
    if walk.dangling is not None:
        return _cannot_carry(walk.dangling)

    environ = rec.environ or {}
    fields = dict(walk.fields)
    env: dict[str, str] = {}
    env_tls: list[str] = []
    env_download = False
    env_draft_download = False
    env_keys: list[str] = []
    env_key_files: list[tuple[str, str]] = []
    env_keyshaped: str | None = None
    hf_file_env: str | None = None
    for name in sorted(environ):
        value = environ[name]
        if name == _KEY_ENV:
            env_keys.extend(_split_keys(value))
            continue
        if not name.startswith(_ENV_PREFIX):
            continue  # nothing else from environ is read: not its value, not its presence
        field = _ENV_FIELDS.get(name)
        if field is not None:
            if field == "cram" and value and not _plain_ram(value):
                # `LLAMA_ARG_CACHE_RAM=-1` (no limit), like `-cram -1`, is not a value the field can carry:
                # it stays what it was, a variable, unless argv already named the option
                if "cram" not in fields:
                    env[name] = value
            elif value:
                fields.setdefault(field, value)
            continue
        category = excluded_option(name)
        if category == "tls":
            env_tls.append(name)
        elif category == "download":
            if name == _DRAFT_DOWNLOAD_ENV:
                env_draft_download = True
            else:
                env_download = True
            if name == _HF_FILE_ENV and value:
                hf_file_env = value
        elif category == "key":
            if name == _KEY_FILE_ENV and value:
                env_key_files.append((name, value))
            # LLAMA_ARG_API_KEY is read by nothing: stripped and dropped
        elif _credential_name(name):
            env_keyshaped = env_keyshaped or name
        elif category is None:
            env[name] = value

    # config.ini applies before the environment and argv: its keys fill what neither named
    ini_masked = ini.ini if ini is not None else None
    ini_tls = ini_download = False
    ini_keys: list[str] = []
    ini_key_files: list[tuple[str, str]] = []
    if ini_masked is not None:
        category = excluded_option(ini_masked.refused) if ini_masked.refused else None
        ini_tls, ini_download = category == "tls", category == "download"
        for key, value in ini_masked.keys.items():
            lowered = key.lower()
            if lowered in _INI_INLINE_KEYS:
                ini_keys.extend(_split_keys(value))
            elif lowered in _INI_KEY_FILE_KEYS and value:
                ini_key_files.append((key, value))
            elif value:  # the only other routed keys are host and port, in either spelling
                fields.setdefault("host" if lowered.endswith("host") else "port", value)

    # (1) TLS, then the download identity or a missing local -m
    tls = walk.tls[0] if walk.tls else (env_tls[0] if env_tls else None)
    if tls is None and ini_tls:
        tls = ini_masked.refused
    if tls is not None:
        return AttachOnly(f"serves TLS ({visible_text(tls)})")
    hf_file = walk.hf_file or hf_file_env
    ini_draft = ini_download and _is_draft_download(ini_masked.refused)
    draft_download = env_draft_download or ini_draft or any(_is_draft_download(f) for f in walk.downloads)
    main_download = (env_download or (ini_download and not ini_draft)
                     or any(not _is_draft_download(f) for f in walk.downloads))
    if main_download or "model" not in fields:
        if draft_download:
            # the draft is mapped after the main model, so maps lists it first: which file is the main one
            # cannot be told (D-52's "first qualifying path" would pick the draft)
            return AttachOnly("downloads a draft model beside the main one, so the mapped file cannot be "
                              "told apart (a record needs the one local path)")
        drafts = _draft_paths(walk.extra, environ, rec.cwd)
        model = served_model([p for p in rec.mapped_ggufs if os.path.normpath(p) not in drafts])
        if model is None:
            return AttachOnly("a record needs a local path; none mapped under " + HF_CACHE_SHOWN
                              + (" (--no-mmap maps nothing)" if "--no-mmap" in rec.argv[1:] else ""))
        default_name = preset_name(hf_file) if hf_file else preset_name(model)
    else:
        model = _join_cwd(fields["model"], rec.cwd)
        if model is None:
            return _cannot_carry("model")
        default_name = preset_name(model)

    # (2) keys
    inline = list(dict.fromkeys(_split_keys(",".join(walk.api_keys)) + env_keys + ini_keys))
    key_files = [*walk.key_files, *env_key_files, *ini_key_files]
    if len(inline) > 1:
        return AttachOnly("holds more than one --api-key")
    resolved_files: list[tuple[str, str]] = []
    for name, raw in key_files:
        if not raw:
            continue
        path = _join_cwd(raw, rec.cwd)
        if path is None:
            return AttachOnly(f"holds a key the import cannot place ({visible_text(name)})")
        resolved_files.append((name, path))
    if len({path for _, path in resolved_files}) > 1:
        return AttachOnly("holds more than one --api-key")
    if inline and resolved_files:
        return AttachOnly("holds a key the import cannot place (--api-key and --api-key-file)")
    keyshaped = walk.keyshaped or env_keyshaped or (ini_masked.keyshaped if ini_masked is not None else None)
    if keyshaped is not None:
        return AttachOnly(f"holds a key the import cannot place ({visible_text(keyshaped)})")
    key_file = resolved_files[0][1] if resolved_files else None
    keyed = bool(inline) or key_file is not None

    # (3) the bind: every entry judged by its literal text
    host_value = fields.get("host")
    addresses: list[str] = []
    if host_value is not None:
        entries = [e for e in (piece.strip() for piece in host_value.split(",")) if e]
        if not entries:
            return _cannot_carry("host")
        for entry in entries:
            kind = classify_hosts(entry)
            if kind == "unsupported":
                what = "(a UNIX socket)" if entry.endswith(".sock") else "(a host name)"
                return AttachOnly(f"binds {visible_text(entry)} {what}")
            if kind == "address":
                addresses.append(entry)
    bind_all = False
    unit_line: str | None = None
    if addresses:
        evidence = key_file is not None or (source == "unit" and bool(inline))
        if source in ("journal", "unit") and not evidence:
            unit_line = UNIT_LOOPBACK_LINE.format(
                unit=visible_text(rec.unit or "the unit"), host=visible_text(addresses[0]))
        else:
            bind_all = True

    # (4) the field table
    mapping: dict[str, object] = {"runtime": "llamacpp", "model": model}
    binary, binary_source = _binary(rec, binary_resolver)
    if binary is not None:
        mapping["binary"] = binary
    mapping["served_name"] = fields.get("alias") or default_name
    mmproj = fields.get("mmproj")
    if mmproj is not None:
        mmproj = _join_cwd(mmproj, rec.cwd)
        if mmproj is None:
            return _cannot_carry("mmproj")
        mapping["mmproj"] = mmproj
    if "ctk" in fields:
        mapping["cache_type_k"] = fields["ctk"]
    if "ctv" in fields:
        mapping["cache_type_v"] = fields["ctv"]
    if "cram" in fields:
        mapping["cache_ram"] = int(fields["cram"])
    resolved_port = _int(fields["port"]) if "port" in fields else (port or DEFAULT_PORT)
    if resolved_port is None or not 1 <= resolved_port <= 65535:
        return _cannot_carry("port")
    mapping["port"] = resolved_port
    ctx: int | None = None
    if "ctx" in fields:
        ctx = _int(fields["ctx"])
        if ctx is None:
            return _cannot_carry("ctx-size")
        ctx = ctx if ctx > 0 else None
    if bind_all:
        mapping["bind_all"] = True
    if keyed:
        mapping["require_api_key"] = True
    if key_file is not None:
        mapping["api_key_file"] = key_file

    # (6) the --verbose family: dropped from a keyed record, kept on an unkeyed one
    extra = list(walk.extra)
    verbose_dropped = False
    if keyed and walk.verbose:
        dropped = set(walk.verbose)
        extra = [tok for idx, tok in enumerate(extra) if idx not in dropped]
        verbose_dropped = True
    if extra:
        mapping["extra_args"] = extra
    if env:
        mapping["env"] = env
    if ini_masked is not None and ini_masked.kept:
        mapping["config_ini"] = ini_masked.kept
    try:
        spec = ManagedServerConfig.model_validate(mapping)
    except ValidationError as exc:
        errors = exc.errors()
        loc = errors[0]["loc"] if errors else ()
        return _cannot_carry(str(loc[0]) if loc else "record")

    # shown lines (D-14), every value through visible_text (D-45)
    shown: list[str] = []
    if spec.extra_args:
        flags = "flags: " + visible_text(shlex.join(spec.extra_args))
        marks: list[str] = []
        for tok in spec.extra_args:
            if tok.startswith("-"):
                flag = tok.partition("=")[0] if tok.startswith("--") else tok
                suffix = llama_annotation(flag, keyed=keyed)
                if suffix is not None and (mark := f"{visible_text(flag)} {suffix}") not in marks:
                    marks.append(mark)
        shown.append(flags + "".join(f" · {mark}" for mark in marks))
    if spec.env:
        items = []
        for name, value in spec.env.items():
            suffix = llama_annotation(name, keyed=keyed)
            items.append(_annotated(f"{visible_text(name)}={visible_text(value)}", suffix))
        shown.append("env: " + ", ".join(items))
    if ini is not None and ini.line is not None:
        shown.append(visible_text(ini.line))
    if spec.config_ini:
        pairs = []
        for key, value in _ini_pairs(spec.config_ini):
            pairs.append(_annotated(f"{visible_text(key)}={visible_text(value)}", llama_annotation(key, keyed=keyed)))
        if pairs:
            shown.append(f"config.ini ({visible_text(ini.dir)}): " + ", ".join(pairs))
    if key_file is not None:
        shown.append(KEY_FILE_LINE.format(path=shown_path(key_file)))
    if verbose_dropped:
        shown.append(VERBOSE_DROPPED)
    if unit_line is not None:
        shown.append(unit_line)
    # D-20: the exposure phrase is computed over the validated record, never read from the flag. A
    # declared-only record (no binary) is judged as the launch it becomes once a binary is set.
    judged = spec if spec.binary else spec.model_copy(update={"binary": "llama-server"})
    decision = launch_decision(judged, bool(inline))
    _, phrase = effective_posture(judged, keyed=bool(inline), decision=decision)
    if phrase is not None:
        shown.append(phrase)
    elif spec.bind_all:
        shown.append(BIND_ALL_LINE.format(host=visible_text(addresses[0])))
    return Imported(spec=spec, api_key=SecretStr(inline[0]) if inline else None, shown=tuple(shown),
                    ctx=ctx, binary_source=binary_source)
