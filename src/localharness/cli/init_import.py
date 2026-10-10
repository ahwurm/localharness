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
import shlex
from dataclasses import dataclass
from typing import Callable

from pydantic import SecretStr, ValidationError

from localharness.agent.gguf_header import preset_name, served_model
from localharness.cli.workspace import shown_path, visible_text
from localharness.config.models import ManagedServerConfig
from localharness.provider.procfs import ProcRecord
from localharness.provider.server import (
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
        name, written, inline = tok, tok, None
        if tok.startswith("--"):
            written, eq, rest = tok.partition("=")
            name = written.replace("_", "-")
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
                if walk.keyshaped is None and "api-key" in name.lstrip("-").lower():
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
        elif field == "cram" and not (value.isascii() and value.isdigit()):
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


def _int(text: str) -> int | None:
    try:
        return int(text.strip())
    except ValueError:
        return None


def _cannot_carry(field: str) -> AttachOnly:
    return AttachOnly(f"holds a value a record cannot carry ({visible_text(field)})")


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
    ini: object | None = None,
) -> Imported | AttachOnly:
    """Translate `rec` (a live process, a journal entry or a unit file, named by `source`) into a
    record, or into the one reason it is attached to only. `port` is the listener port the caller found
    the process on, used when the process names none.

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
            if value:
                fields.setdefault(field, value)
            continue
        category = excluded_option(name)
        if category == "tls":
            env_tls.append(name)
        elif category == "download":
            env_download = True
            if name == _HF_FILE_ENV and value:
                hf_file_env = value
        elif category == "key":
            if name == _KEY_FILE_ENV and value:
                env_key_files.append((name, value))
            # LLAMA_ARG_API_KEY is read by nothing: stripped and dropped
        elif "API_KEY" in name:
            env_keyshaped = env_keyshaped or name
        elif category is None:
            env[name] = value

    # (1) TLS, then the download identity or a missing local -m
    tls = walk.tls[0] if walk.tls else (env_tls[0] if env_tls else None)
    if tls is not None:
        return AttachOnly(f"serves TLS ({visible_text(tls)})")
    hf_file = walk.hf_file or hf_file_env
    if walk.downloads or env_download or "model" not in fields:
        model = served_model(rec.mapped_ggufs)
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
    inline = list(dict.fromkeys(_split_keys(",".join(walk.api_keys)) + env_keys))
    key_files = [*walk.key_files, *env_key_files]
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
    keyshaped = walk.keyshaped or env_keyshaped
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
