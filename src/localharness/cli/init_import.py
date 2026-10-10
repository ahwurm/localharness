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

from pydantic import ValidationError

from localharness.agent.gguf_header import preset_name
from localharness.cli.workspace import visible_text
from localharness.config.models import ManagedServerConfig
from localharness.provider.procfs import ProcRecord
from localharness.provider.server import excluded_option

ATTACH_ONLY_SUFFIX = " — attached only, no server record written"
"""The tail every attach-only sentence carries after its `<who> <reason>` (D-25)."""

HF_CACHE_SHOWN = "~/.cache/huggingface/hub"
"""The download cache as the attach-only reason names it (D-19(1))."""

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
}

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
    api_key: object | None
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
    """What one pass over argv found: the field values (last wins), and the tokens nothing claimed."""

    fields: dict[str, str]
    extra: list[str]
    dangling: str | None = None


def _walk(tokens: tuple[str, ...]) -> _Walk:
    """Walk argv after argv[0]. A `--` token is looked up with `_` read as `-` and an `=value` split
    off (llama.cpp's own rule); a token nothing claims stays in `extra` exactly as written."""
    walk = _Walk(fields={}, extra=[])
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if not tok.startswith("-"):
            walk.extra.append(tok)
            continue
        name, inline = tok, None
        if tok.startswith("--"):
            name, eq, rest = tok.partition("=")
            name = name.replace("_", "-")
            inline = rest if eq else None
        field = _ARGV_FIELDS.get(name)
        if field is None:
            walk.extra.append(tok)
            continue
        if inline is not None:
            value = inline
        elif i < len(tokens):
            value = tokens[i]
            i += 1
        else:
            walk.dangling = name
            continue
        if field == "cram" and not value.isdigit():
            # `-cram -1` (no limit) is not a value the record's field can carry: it stays an argument
            walk.extra.append(tok)
            if inline is None:
                walk.extra.append(value)
            continue
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
    the process on, used when the process names none."""
    runtime = runtime_of(rec)
    if runtime != "llamacpp":
        return AttachOnly(f"runs {visible_text(_basename(rec.argv[0] if rec.argv else rec.exe))}, "
                          "which the import does not translate")
    walk = _walk(rec.argv[1:])
    if walk.dangling is not None:
        return _cannot_carry(walk.dangling)
    environ = rec.environ or {}
    fields = dict(walk.fields)
    env: dict[str, str] = {}
    for name in sorted(environ):
        if not name.startswith(_ENV_PREFIX):
            continue
        field = _ENV_FIELDS.get(name)
        if field is not None:
            if environ[name]:
                fields.setdefault(field, environ[name])
        elif excluded_option(name) is None:
            env[name] = environ[name]

    model = fields.get("model")
    if model is not None:
        model = _join_cwd(model, rec.cwd)
        if model is None:
            return _cannot_carry("model")
    if model is None:
        return AttachOnly("a record needs a local path; none mapped under " + HF_CACHE_SHOWN)
    mapping: dict[str, object] = {"runtime": "llamacpp", "model": model}
    binary, binary_source = _binary(rec, binary_resolver)
    if binary is not None:
        mapping["binary"] = binary
    mapping["served_name"] = fields.get("alias") or preset_name(model)
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
    if walk.extra:
        mapping["extra_args"] = list(walk.extra)
    if env:
        mapping["env"] = env
    try:
        spec = ManagedServerConfig.model_validate(mapping)
    except ValidationError as exc:
        errors = exc.errors()
        loc = errors[0]["loc"] if errors else ()
        return _cannot_carry(str(loc[0]) if loc else "record")

    shown: list[str] = []
    if spec.extra_args:
        shown.append("flags: " + visible_text(shlex.join(spec.extra_args)))
    if spec.env:
        shown.append("env: " + ", ".join(
            f"{visible_text(name)}={visible_text(value)}" for name, value in spec.env.items()))
    return Imported(spec=spec, api_key=None, shown=tuple(shown), ctx=ctx, binary_source=binary_source)
