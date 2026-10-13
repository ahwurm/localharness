"""Managed vLLM server lifecycle — install, model download, launch, readiness, stop.

Used by `localharness init` (guided setup), `start` (autostart after reboot),
and the REPL /model command (swap = restart with a different downloaded model).

Both launch modes run as a direct child in a new session with a pidfile and a
log at <config-dir>/vllm/serve.log: `binary` execs a vllm executable; `docker`
runs a *foreground* `docker run --rm --name localharness-vllm` client whose
SIGTERM propagates to the container (--sig-proxy default), so one lifecycle
covers both.
"""
from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog
from pydantic import SecretStr

from types import MappingProxyType
from typing import Any, Callable, Mapping

from localharness.config.models import ManagedServerConfig
from localharness.core import private_files
from localharness.core.private_files import NotRegularFile, open_regular, private_opener
from localharness.core.which import which_outside_cwd
from localharness.provider import procfs

logger = structlog.get_logger(__name__)
"""The module logger. Named `logger`, not `log`: `start_server` holds its serve.log handle as `log`."""

DOCKER_CONTAINER_NAME = "localharness-vllm"

_TRANSPORT: Any = None
"""Tests' httpx.MockTransport for wait_ready; None in production (the real network)."""


def _raw_key(api_key: Any) -> str | None:
    raw = api_key.get_secret_value() if isinstance(api_key, SecretStr) else api_key
    return raw if raw and raw != "none" else None


def auth_headers(api_key: Any) -> dict[str, str]:
    """The Authorization header a keyed server needs, or {} for no key ("none" or empty)."""
    raw = _raw_key(api_key)
    return {"Authorization": f"Bearer {raw}"} if raw else {}


def key_kwargs(api_key: Any) -> dict[str, str]:
    """{"api_key": <raw>} when a key is set, else {} — call sites splat it, so a probe of a keyless
    server is called exactly as before."""
    raw = _raw_key(api_key)
    return {"api_key": raw} if raw else {}


_URL_USERINFO = re.compile(r"(?<=://)[^/\s?#]*@")


def strip_url_credentials(text: str) -> str:
    """`text` with URL userinfo removed (`http://user:pass@host` -> `http://host`), up to the LAST "@"
    of the authority, as httpx splits it: `http://u:p@ss@host` is a working URL whose password is
    "p@ss". An "@" after the path, query, fragment or a space is not userinfo and stays. The one helper
    every surface that publishes or keeps server error text uses (the loop's ProviderLost detail and
    failure note, the recovery's outage record), so a fix here reaches all of them."""
    return _URL_USERINFO.sub("", text)


def required_key_kwargs(spec: Any, api_key: Any) -> dict[str, str]:
    """The key to launch and probe a server with: {"api_key": <raw>} only when this launched server
    is set to require one (`require_api_key`, machine-level, default false) AND a key is set;
    otherwise {} — today's keyless launch and probes, byte for byte. The one place that decision
    is made (orchestrator ruling R11: requiring a key is opt-in, never a stricter default). Only a
    real `True` counts, so a stand-in object can never switch it on."""
    return key_kwargs(api_key) if getattr(spec, "require_api_key", False) is True else {}


LLAMA_EXCLUDED_OPTIONS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("key", ("--api-key",), "LLAMA_API_KEY"),
    ("key", ("--api-key-file",), "LLAMA_ARG_API_KEY_FILE"),
    ("key", (), "LLAMA_ARG_API_KEY"),
    ("host/port", ("--host",), "LLAMA_ARG_HOST"),
    ("host/port", ("--port",), "LLAMA_ARG_PORT"),
    ("download", ("-mu", "--model-url"), "LLAMA_ARG_MODEL_URL"),
    ("download", ("-mmu", "--mmproj-url"), "LLAMA_ARG_MMPROJ_URL"),
    ("download", ("-dr", "--docker-repo"), "LLAMA_ARG_DOCKER_REPO"),
    ("download", ("-hf", "-hfr", "--hf-repo"), "LLAMA_ARG_HF_REPO"),
    ("download", ("-hff", "--hf-file"), "LLAMA_ARG_HF_FILE"),
    ("download", ("--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"),
     "LLAMA_ARG_SPEC_DRAFT_HF_REPO"),
    ("tls", ("--ssl-key-file",), "LLAMA_ARG_SSL_KEY_FILE"),
    ("tls", ("--ssl-cert-file",), "LLAMA_ARG_SSL_CERT_FILE"),
)
"""The llama-server options a record may not carry through `env` or `config_ini`: rows of
(category, flag spellings, env name). The key, host/port, download-identity and TLS options are
owned by other record fields and by the launch decision, so a record never smuggles one in by a
second door. The rows come from the owner's llama.cpp checkout (/home/awurm/src/llama.cpp HEAD
43fe9c642, common/arg.cpp `set_env` bindings). The `env` validator, the `config_ini` validator and
the later process import all read this one table, so no second list can drift.
`LLAMA_ARG_NO_HOST` (`--no-host`) is a host-buffer compute switch, not a bind option, and is
deliberately absent. `LLAMA_ARG_API_KEY` is read by nothing; it is listed so it is stripped with
the key names."""

LLAMA_EXCLUDED_ENV: frozenset[str] = frozenset(env for _, _, env in LLAMA_EXCLUDED_OPTIONS)
"""The env names of `LLAMA_EXCLUDED_OPTIONS`."""

def _excluded_by_key() -> dict[str, str]:
    """Lower-cased env name or dash-stripped flag name -> category, built once from the table. The
    first row wins a key two rows spell."""
    by_key: dict[str, str] = {}
    for category, flags, env in LLAMA_EXCLUDED_OPTIONS:
        for key in (env.lower(), *(flag.lstrip("-").lower() for flag in flags)):
            by_key.setdefault(key, category)
    return by_key


_EXCLUDED_BY_KEY: dict[str, str] = _excluded_by_key()

LLAMA_EXCLUDED_INI_KEYS: frozenset[str] = frozenset(_EXCLUDED_BY_KEY)
"""The config.ini keys of `LLAMA_EXCLUDED_OPTIONS`: llama-server resolves a key as an env name
verbatim or as a flag name with its leading dashes removed, in any section, in any letter case."""


def excluded_option(name: str) -> str | None:
    """The category ("key", "host/port", "download" or "tls") of the excluded option `name` spells,
    as an env name or a dash-stripped flag name in any letter case, or None. Pure, no I/O. Reads
    the same index `LLAMA_EXCLUDED_INI_KEYS` is built from, so the set and the function cannot drift."""
    return _EXCLUDED_BY_KEY.get(name.strip().lower())


SHELL_EXEC_NOTE = "(serves a shell-exec tool on the port)"
EXEC_DIR_NOTE = "(a directory the server runs executables from)"
SERVED_DIR_NOTE = "(served to anyone who reaches the port)"

LLAMA_ANNOTATIONS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    (SHELL_EXEC_NOTE,
     ("--tools", "LLAMA_ARG_TOOLS", "tools",
      "-ag", "--agent", "LLAMA_ARG_AGENT", "agent",
      "--tools-runtime", "LLAMA_ARG_TOOLS_RUNTIME", "tools-runtime",
      "--mcp-servers-config", "LLAMA_ARG_MCP_SERVERS_CONFIG", "mcp-servers-config",
      "--mcp-servers-json", "LLAMA_ARG_MCP_SERVERS_JSON", "mcp-servers-json",
      "--ui-mcp-proxy", "--webui-mcp-proxy", "LLAMA_ARG_UI_MCP_PROXY", "ui-mcp-proxy"),
     False),
    (EXEC_DIR_NOTE,
     ("--video-ffmpeg-dir", "LLAMA_ARG_VIDEO_FFMPEG_DIR", "video-ffmpeg-dir"),
     False),
    (SERVED_DIR_NOTE,
     ("--path", "LLAMA_ARG_STATIC_PATH", "path"),
     True),
)
"""Display only, nothing is refused for it: rows of (suffix shown after the option, the spellings of
the options it marks, whether it is shown on an unkeyed record only). A spelling is a flag
(leading dash), a `LLAMA_ARG_*` env name, or a `config.ini` key (the flag name without its dashes).
The options are the ones that make a server hand out a shell, run executables from a directory, or
serve a directory to whoever reaches the port. A separate table from `LLAMA_EXCLUDED_OPTIONS`,
which stays 13 rows: nothing here is stripped or refused. The rows come from the owner's llama.cpp
checkout (HEAD 43fe9c642, common/arg.cpp), including the `--webui-mcp-proxy` spelling and `--path`."""


def _annotation_key(name: str) -> str:
    """The lookup key of an option spelling: a `LLAMA_ARG_*` name compares upper-cased, anything
    else has its leading dashes removed, `_` read as `-` (llama.cpp's own rule) and is lower-cased."""
    name = name.strip()
    if name.upper().startswith("LLAMA_ARG_"):
        return name.upper()
    return name.lstrip("-").replace("_", "-").lower()


def _annotations_by_key() -> dict[str, tuple[str, bool]]:
    """Option key -> (suffix, unkeyed_only), built once from the table. The first row wins."""
    by_key: dict[str, tuple[str, bool]] = {}
    for suffix, names, unkeyed_only in LLAMA_ANNOTATIONS:
        for name in names:
            by_key.setdefault(_annotation_key(name), (suffix, unkeyed_only))
    return by_key


_ANNOTATION_BY_KEY: dict[str, tuple[str, bool]] = _annotations_by_key()


def llama_annotation(name: str, *, keyed: bool) -> str | None:
    """The suffix to show after the option `name` (a flag, a `LLAMA_ARG_*` name or a config.ini key,
    any letter case, `_` or `-`), or None. `keyed` is whether the record carries an API key: the
    served-directory suffix is for an unkeyed record only. Pure, no I/O, display only."""
    hit = _ANNOTATION_BY_KEY.get(_annotation_key(name))
    if hit is None:
        return None
    suffix, unkeyed_only = hit
    return None if (unkeyed_only and keyed) else suffix


@dataclass(frozen=True)
class LaunchDecision:
    """The one launch decision every path meets: `refusal` (why nothing may start, or None),
    `loopback` (the launch must bind this machine only), `key_file_ok` (`--api-key-file` may ride
    argv) and `note` (the one line to show when the widening was withheld, or None).

    It is never a narrowed copy of the spec: `_active_heavy` holds the spec object and
    `free_accelerator` compares it by identity, so a copy would read as a second heavy."""

    refusal: str | None
    loopback: bool
    key_file_ok: bool
    note: str | None


class LaunchRefused(RuntimeError):
    """The decision's refusal, raised from `activate`. A RuntimeError so every caller that prints a
    failed launch as one line (the Ollama / LM Studio foreign-daemon shape) keeps doing so, while
    `_restore_incumbent` can tell a refusal from a failed launch."""


def launch_decision(spec: Any, keyed: bool, *, peer: str | None = None) -> LaunchDecision:
    """The one place a launch is decided, beside `required_key_kwargs`. `keyed` means a key is
    actually held on the launch path (the caller already gated through `required_key_kwargs`);
    `peer` is the `EndpointRef.name` when `spec` is a peer's lifecycle, None for the primary.

    - Ollama and LM Studio always bind loopback (their argv hard-codes it).
    - A llama.cpp record with no `binary` is declared, not launchable: refused with one line.
    - With no `api_key_file` the launch binds by `bind_all`.
    - With one, `key_file_ok` is `keyed and the path is a regular file`, judged by `os.stat` alone —
      the file is NEVER opened here, so a FIFO or a swapped file can neither block nor leak through
      the decision. Anything else withholds the widening, not the launch: loopback for this run,
      the file left off argv, and a note saying why. A file the session's own read refused
      (`spec.key_file_refused`, set by `mark_key_file`) is one of those: even when another key keeps
      the launch keyed, the file stays off argv and the launch binds loopback.

    Pure with respect to the spec (it is read, never copied or changed)."""
    runtime = getattr(spec, "runtime", "vllm")
    if runtime in ("ollama", "lmstudio"):
        return LaunchDecision(None, True, False, None)
    if runtime == "llamacpp" and not getattr(spec, "binary", None):
        refusal = ("no llama-server — set server.binary" if peer is None
                   else f"no llama-server — set extra_endpoints[{peer}].lifecycle.binary")
        return LaunchDecision(refusal, True, False, None)
    loopback = getattr(spec, "bind_all", False) is not True
    key_file = getattr(spec, "api_key_file", None)
    if not key_file:
        return LaunchDecision(None, loopback, False, None)
    path = Path(key_file).expanduser()
    try:
        regular = stat.S_ISREG(os.stat(path).st_mode)
        reason = None if regular else "not a regular file"
    except FileNotFoundError:
        reason = "no such file"
    except (OSError, ValueError) as exc:
        reason = getattr(exc, "strerror", None) or type(exc).__name__
    if reason is None and not keyed:
        reason = "the harness holds no key for it"
    if reason is None and getattr(spec, "key_file_refused", False) is True:
        # The session read this file and the byte rule refused it (`start` / a cold peer say so in one
        # line). The harness holds a key from another source, but the file must not reach argv: a
        # server started with it would take a key the harness never validated.
        reason = "refused by the key-file rule"
    if reason is not None:
        return LaunchDecision(
            None, True, False,
            f"--api-key-file {path} unreadable: {reason} — launched on 127.0.0.1 for this run")
    return LaunchDecision(None, loopback, True, None)


def mark_key_file(spec: Any, refused: bool) -> None:
    """Record on ``spec`` whether this session's read of its `api_key_file` was refused, so the one
    launch decision (and so the argv and the exposure phrase built from it) agrees with the line `start`
    printed. A stand-in without the setter is left alone."""
    setter = getattr(spec, "set_key_file_refused", None)
    if callable(setter):
        setter(refused)


KEY_FILE_CAP = 4096
"""The most bytes a key file may hold. One byte more is read to tell "exactly the cap" from "over it"."""


class KeyFileError(RuntimeError):
    """A key file the byte rule refuses. `str(exc)` is the reason, worded for the one line `start`
    prints (`server.api_key_file <path>: <reason> — this session sends no key`)."""


def read_key_file(path: str | os.PathLike[str]) -> str:
    r"""The key a `server.api_key_file` holds, or `KeyFileError(reason)`.

    The one place the key file is OPENED (`launch_decision` and `doctor` only stat it), and only
    after the trust decision covers the entry. The open is `open_regular`: a FIFO or a device is
    refused without blocking. The rule is stricter than llama-server's own `getline` read, which
    skips `#` lines and keeps a trailing `\r`, so a CRLF file would be sent to the server as `K\r`
    while the harness sent `K`; and llama-server splits `LLAMA_API_KEY` on commas, so a comma in the
    key would make the server accept two keys. Hence: at most `KEY_FILE_CAP` bytes; split on `\n`
    only; empty lines dropped; a `#` line refuses; exactly one line must remain; every byte in
    0x21-0x7E and none of `,` or `"`. The key is that line exactly, never trimmed or normalised.
    Reading is idempotent and writes nothing."""
    target = Path(path).expanduser()
    try:
        fd = open_regular(target)
    except NotRegularFile:
        raise KeyFileError("not a regular file") from None
    except FileNotFoundError:
        raise KeyFileError("no such file") from None
    except (OSError, ValueError) as exc:
        raise KeyFileError(getattr(exc, "strerror", None) or type(exc).__name__) from None
    try:
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(KEY_FILE_CAP + 1)
    except OSError as exc:
        raise KeyFileError(exc.strerror or type(exc).__name__) from None
    if len(data) > KEY_FILE_CAP:
        raise KeyFileError(f"over {KEY_FILE_CAP} bytes")
    lines = [line for line in data.split(b"\n") if line]
    if any(line.startswith(b"#") for line in lines):
        raise KeyFileError("comment line")
    if not lines:
        raise KeyFileError("no key")
    if len(lines) > 1:
        raise KeyFileError("more than one line")
    line = lines[0]
    if any(not 0x21 <= byte <= 0x7E for byte in line):
        raise KeyFileError("byte outside printable ASCII")
    if b"," in line or b'"' in line:
        raise KeyFileError("comma or quote")
    return line.decode("ascii")


KEY_ENV: Mapping[str, str] = MappingProxyType({"vllm": "VLLM_API_KEY", "llamacpp": "LLAMA_API_KEY"})
"""The one key-variable table: the environment variable each launched runtime reads its key from.
`start_server`'s child environment and the docker `-e` both read it, so the mapping lives in one
place. (llama-server reads `LLAMA_API_KEY`; `LLAMA_ARG_API_KEY` is read by nothing.)"""

RUNTIME_LABELS: Mapping[str, str] = MappingProxyType(
    {"vllm": "vLLM", "llamacpp": "llama.cpp", "ollama": "Ollama", "lmstudio": "LM Studio"})
"""The runtime names `start`'s and `/model`'s launch lines print."""

EVERY_INTERFACE_NO_KEY = "every interface, no key"
"""The exposure phrase: a launch that binds beyond loopback with no key accompanying it."""


def classify_hosts(value: str) -> str:
    """Classify a `--host` value (or a comma-separated list of them) as "loopback", "address" or
    "unsupported", by its literal text alone — a name is never resolved, so no DNS answer can turn
    an exposure into a loopback.

    Entries split on commas, are stripped, and empty ones dropped; nothing left is unsupported.
    The literal `localhost` and any address `ipaddress` calls loopback (an IPv4-mapped IPv6 address
    is judged by the IPv4 address it wraps) are loopback; any other address is "address"; anything
    that is not an address (a hostname, a `.sock` path) is unsupported. One unsupported entry makes
    the whole value unsupported, else one address entry makes it "address". The import and the
    upgrade conversion read this same function (spec §8, §9)."""
    entries = [e for e in (part.strip() for part in value.split(",")) if e]
    if not entries:
        return "unsupported"
    kinds = set()
    for entry in entries:
        if entry == "localhost":
            kinds.add("loopback")
            continue
        try:
            addr = ipaddress.ip_address(entry)
        except ValueError:
            kinds.add("unsupported")
            continue
        addr = getattr(addr, "ipv4_mapped", None) or addr
        kinds.add("loopback" if addr.is_loopback else "address")
    if "unsupported" in kinds:
        return "unsupported"
    return "address" if "address" in kinds else "loopback"


def key_destination(base_url: Any) -> tuple[bool, str]:
    """`(this_machine, shown)` for the endpoint a session's bearer is sent to: ``this_machine`` is true
    only when the host of ``base_url`` is literally loopback (`classify_hosts`, no DNS), ``shown`` is
    `host[:port]` for a line (userinfo, path and query are left out, so a URL's secret never prints).
    An unparseable URL is not this machine and is shown as written. A key read from a key file is
    sent to `provider.base_url` (or a peer's `base_url`), which nothing ties to the managed server's
    own port, so the read is allowed only for a destination that is this machine and a question about
    the file names the destination when it is not."""
    from urllib.parse import urlparse

    text = str(base_url)
    try:
        parsed = urlparse(text)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return False, text
    if not host:
        return False, text
    shown = f"{host}:{port}" if port else host
    return classify_hosts(host) == "loopback", shown


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def server_dir(config_dir: Path) -> Path:
    return Path(config_dir) / procfs.SERVER_DIR_NAME


def config_ini_home(config_dir: Path, port: int) -> Path:
    """The `XDG_CONFIG_HOME` a llama.cpp launch on `port` is given (D-30, D-59): one harness-owned
    directory per port under the GLOBAL config dir `activate` receives, never under `vllm/`. Always an
    absolute path (the XDG base directory spec requires one): a relative `--config-dir` or
    `LOCALHARNESS_DIR` is joined to the process cwd here, once (IN-53)."""
    return Path(os.path.abspath(config_dir)) / "xdg" / str(port)


def _own_directory(path: Path) -> None:
    """Leave `path` a real directory, 0o700, that nothing planted can redirect.

    A symlink or any non-directory in its place is unlinked (never followed) and the directory
    created; an existing real directory is opened `O_NOFOLLOW | O_DIRECTORY` and `fchmod`-ed through
    that descriptor, so the mode change cannot land on a link swapped in after the check."""
    try:
        planted = not stat.S_ISDIR(os.lstat(path).st_mode)
        missing = False
    except FileNotFoundError:
        planted, missing = False, True
    if planted:
        os.unlink(path)
        missing = True
    if missing:
        os.mkdir(path, 0o700)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_CLOEXEC", 0))
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


def prepare_config_ini_home(config_dir: Path, port: int, config_ini: str | None) -> Path:
    """Make `config_ini_home(config_dir, port)` hold the record's `config_ini` bytes as
    `llama.cpp/config.ini` (or no `config.ini` when `config_ini` is None) and return the home.
    Temporary files a killed launch left (`.config.ini.*`) are removed; anything else in the directory
    is left alone, since llama.cpp reads only `config.ini` there. The checks are path-based, a
    confused-deputy guard and not a sandbox (spec 16 §9): code running as this user can still swap
    a component between a check and its use.

    llama-server reads `$XDG_CONFIG_HOME/llama.cpp/config.ini` before argv and env can say anything,
    so a relaunch must read only the bytes its record carries (the ones the trust snapshot
    fingerprinted), never the person's own file and never one somebody planted. Every component
    under `<config dir>/xdg` is `lstat`-ed: a symlink or non-directory is removed and recreated, a
    real directory is tightened to 0o700, and nothing is opened through a path that was a link. The
    file goes through a temporary file in the same directory (0600) and `os.replace`, which replaces
    a planted link rather than writing through it; a planted directory named `config.ini` is removed
    with `shutil.rmtree`, which does not follow links."""
    home = config_ini_home(config_dir, port)
    target_dir = home / "llama.cpp"
    for path in (home.parent, home, target_dir):
        _own_directory(path)
    with contextlib.suppress(OSError):
        for stray in os.scandir(target_dir):
            if stray.name.startswith(".config.ini."):
                with contextlib.suppress(OSError):  # unlinks a link itself, never what it points at
                    os.unlink(stray.path)
    target = target_dir / "config.ini"
    try:
        is_dir = stat.S_ISDIR(os.lstat(target).st_mode)
        present = True
    except FileNotFoundError:
        is_dir = present = False
    if is_dir:
        shutil.rmtree(target)
    if config_ini is None:
        if present and not is_dir:
            os.unlink(target)
        return home
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".config.ini.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(config_ini.encode("utf-8"))
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    return home


def log_path(config_dir: Path) -> Path:
    return server_dir(config_dir) / "serve.log"


def pid_path(config_dir: Path) -> Path:
    return server_dir(config_dir) / procfs.PIDFILE_NAME


def lock_path(config_dir: Path) -> Path:
    """The cross-process launch lock, beside `server.pid`: an empty advisory-flock file, never read or
    written for content and deliberately not one of the protected config-dir entries."""
    return server_dir(config_dir) / "launch.lock"


def venv_vllm_bin(config_dir: Path) -> Path:
    return server_dir(config_dir) / "venv" / "bin" / "vllm"


# ---------------------------------------------------------------------------
# Runtime install
# ---------------------------------------------------------------------------


def find_vllm(config_dir: Path) -> str | None:
    """The harness-managed venv wins over a system vllm (its version is known)."""
    venv_bin = venv_vllm_bin(config_dir)
    if venv_bin.exists():
        return str(venv_bin)
    return which_outside_cwd("vllm")


def install_vllm_venv(config_dir: Path, package: str) -> str:
    """Install `package` into a dedicated venv at <config-dir>/vllm/venv.

    Streams installer output to the terminal; raises RuntimeError on failure.
    Prefers uv (fast, `--torch-backend=auto` resolves the right CUDA torch);
    falls back to stdlib venv + pip.
    """
    venv = server_dir(config_dir) / "venv"
    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    uv = which_outside_cwd("uv")
    if uv:
        _run_streaming([uv, "venv", "--python", "3.12", str(venv)])
        install = [uv, "pip", "install", "--python", str(venv / "bin" / "python"), package]
        if package == "vllm":
            install.append("--torch-backend=auto")
        _run_streaming(install)
    else:
        _run_streaming([sys.executable, "-m", "venv", str(venv)])
        _run_streaming([str(venv / "bin" / "pip"), "install", package])
    bin_path = venv_vllm_bin(config_dir)
    if not bin_path.exists():
        raise RuntimeError(f"install finished but {bin_path} does not exist")
    return str(bin_path)


def _run_streaming(cmd: list[str]) -> None:
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise RuntimeError(f"`{' '.join(cmd)}` failed (exit {rc})")


# ---------------------------------------------------------------------------
# Model download (Hugging Face cache — shared with the docker mount)
# ---------------------------------------------------------------------------


def list_cached_models() -> list[str]:
    """HF-cache model repos with actual bytes on disk. Empty list on any error."""
    try:
        from huggingface_hub import scan_cache_dir

        return sorted(
            r.repo_id
            for r in scan_cache_dir().repos
            if r.repo_type == "model" and r.size_on_disk > 0
        )
    except Exception:
        return []


def is_model_cached(repo_id: str) -> bool:
    return repo_id in list_cached_models()


def download_model(repo_id: str) -> str:
    """Download a checkpoint into the HF cache (tqdm progress). Raises on failure.

    Pulls the WHOLE repo snapshot — right for a vLLM/transformers-format checkpoint (many
    shard files, all needed). Wrong for a GGUF repo, which typically ships several quant
    variants as siblings; use `download_file` for those instead."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id)


def download_file(repo_id: str, filename: str) -> str:
    """Download ONE file from a HF repo into the HF cache (tqdm progress). Raises on failure.

    The llama.cpp/GGUF case: a GGUF repo commonly ships several quantizations as sibling files
    (e.g. `Q4_K_M.gguf`, `Q8_0.gguf`) in the SAME repo — `download_model`'s whole-repo
    `snapshot_download` would pull every one of them. `hf_hub_download` fetches exactly the
    named file, returning its local cache path (ready to hand to a `server.model`/`-m` flag)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename)


# ---------------------------------------------------------------------------
# Launch / readiness / stop
# ---------------------------------------------------------------------------


def serve_command(
    srv: ManagedServerConfig,
    *,
    keyed: bool = False,
    loopback: bool = False,
    key_file_ok: bool = False,
    context: int | None = None,
    cache_ram: int | None = None,
    cache_type: str | None = None,
    spec_type: str | None = None,
    draft_p_min: float | None = None,
    reasoning_parser: str | None = None,
    tool_call_parser: str | None = None,
) -> list[str]:
    """Build the launch command for srv.model. Pure: list argv, no shell, no I/O.

    A server listens on this machine only unless `srv.bind_all` (and `loopback`, the launch
    decision's narrowing, is not set): the loopback `--host 127.0.0.1` comes LAST, after
    `extra_args`, because the server keeps the last `--host` it sees — so `bind_all` is the only
    exposure switch and an `extra_args` `--host 0.0.0.0` can never outlive the narrowing. Docker
    changes only the PUBLISHED address (`-p 127.0.0.1:<port>:8000`): inside the container vLLM
    must keep listening on the container's own interfaces, because the container's loopback is
    not what `-p` forwards to. `key_file_ok` and the two fit keywords `context` / `cache_ram` are
    what `activate` passes (the decision's judgment, then the fit); this function judges nothing.
    `keyed` (docker only) passes `-e VLLM_API_KEY` — the variable's NAME, its value coming from the
    launch environment (`start_server(api_key=)`); a binary reads that environment directly.

    When srv.model names a `local_models` registry entry, the entry's checkpoint dir is
    served (docker: bind-mounted read-only at /models/serving) under --served-model-name
    <entry.name>, with the entry's per-model extra_args appended after the shared ones —
    so a swap changes checkpoint, served id, AND model-specific flags in one place.
    Otherwise srv.model passes through untouched (HF repo id resolved via the cache).

    runtime='llamacpp' (0.11 Phase B) is a separate, simpler shape the harness spawns directly:
    `llama-server -m <gguf> [-a <alias>] [--mmproj <p>] [--cache-type-k/-v <t>]
    [--spec-type <t> [--spec-draft-p-min <p>]] [-c <n>] [--cache-ram <n>] [--api-key-file <p>]
    --port <p> [--host 0.0.0.0] <extra_args> [--host 127.0.0.1]` — every item from the record, in that order, each only when set; the
    record's own `context` / `cache_ram` / `cache_type_*` / `speculative` win over the keywords
    (`context`, `cache_ram`, `cache_type`, `spec_type`, `draft_p_min`), which fill an unset one
    (`speculative: draft-mtp` emits `--spec-type draft-mtp` alone, `off` emits nothing); `~` in
    the model, mmproj and key-file paths is expanded as a string (no stat); the key file rides argv
    only when `key_file_ok`. A GGUF file IS the model, so there is no docker mount /
    --served-model-name registry indirection.

    vLLM (binary and docker) emits `--reasoning-parser X --tool-call-parser Y
    --enable-auto-tool-choice` just before the shared `extra_args`: each side is the record's own
    `reasoning_parser` / `tool_call_parser` when it is a name, nothing when it is `off`, else the
    keyword (`reasoning_parser`, `tool_call_parser`); `--enable-auto-tool-choice` rides with a
    tool-call parser unless `extra_args` already carry it, and never with a reasoning parser alone."""
    loop = loopback or not srv.bind_all
    if srv.runtime == "llamacpp":
        ctx = srv.context if srv.context is not None else context
        ram = srv.cache_ram if srv.cache_ram is not None else cache_ram
        key_file = os.path.expanduser(srv.api_key_file) if srv.api_key_file else None
        k = srv.cache_type_k if srv.cache_type_k is not None else cache_type
        v = srv.cache_type_v if srv.cache_type_v is not None else cache_type
        if srv.speculative == "draft-mtp":
            spec = ["--spec-type", "draft-mtp"]
        elif srv.speculative is None and spec_type is not None:
            spec = ["--spec-type", spec_type,
                    *(["--spec-draft-p-min", str(float(draft_p_min))] if draft_p_min is not None else [])]
        else:
            spec = []
        return [
            str(srv.binary), "-m", os.path.expanduser(srv.model),
            *(["-a", srv.served_name] if srv.served_name is not None else []),
            *(["--mmproj", os.path.expanduser(srv.mmproj)] if srv.mmproj is not None else []),
            *(["--cache-type-k", k] if k is not None else []),
            *(["--cache-type-v", v] if v is not None else []),
            *spec,
            *(["-c", str(ctx)] if ctx is not None else []),
            *(["--cache-ram", str(ram)] if ram is not None else []),
            *(["--api-key-file", key_file] if key_file_ok and key_file else []),
            "--port", str(srv.port),
            *([] if loop else ["--host", "0.0.0.0"]),
            *srv.extra_args,
            *(["--host", "127.0.0.1"] if loop else []),
        ]
    if srv.runtime == "ollama":
        # Ollama (0.11 Phase D): the harness spawns the DAEMON — `ollama serve` takes NO model arg
        # (models load lazily per request; DaemonStrategy warm-loads srv.model after). Env carries
        # the bind host/port + keep-loaded-while-the-daemon-is-up; CPU-only when gpu=False (lets the
        # daemon lifecycle be validated with NO GPU window). The `env` prefix sets vars through
        # start_server's plain Popen without adding a param. `binary` defaults to PATH `ollama`.
        env = ["env", f"OLLAMA_HOST=127.0.0.1:{srv.port}", "OLLAMA_KEEP_ALIVE=-1"]
        if not srv.gpu:
            env.append("CUDA_VISIBLE_DEVICES=")
        return [*env, str(srv.binary) if srv.binary else "ollama", "serve"]
    entry = srv.entry_for(srv.model)
    extra = [*srv.extra_args, *(entry.extra_args if entry is not None else [])]
    rp = srv.reasoning_parser if srv.reasoning_parser is not None else reasoning_parser
    tp = srv.tool_call_parser if srv.tool_call_parser is not None else tool_call_parser
    rp = None if rp == "off" else rp
    tp = None if tp == "off" else tp
    parsers = [
        *(["--reasoning-parser", rp] if rp is not None else []),
        *(["--tool-call-parser", tp] if tp is not None else []),
        *(["--enable-auto-tool-choice"]
          if tp is not None and "--enable-auto-tool-choice" not in extra else []),
    ]
    if srv.launch == "docker":
        hf_cache = str(Path("~/.cache/huggingface").expanduser())
        model_arg = srv.model
        mounts = ["-v", f"{hf_cache}:/root/.cache/huggingface"]
        served: list[str] = []
        if entry is not None:
            mounts += ["-v", f"{str(Path(entry.path).expanduser())}:/models/serving:ro"]
            model_arg = "/models/serving"
            served = ["--served-model-name", entry.name]
        return [
            "docker", "run", "--rm", "--name", DOCKER_CONTAINER_NAME,
            "--gpus", "all", "--ipc=host",
            *mounts,
            *(["-e", KEY_ENV["vllm"]] if keyed else []),
            "-p", f"127.0.0.1:{srv.port}:8000" if loop else f"{srv.port}:8000",
            str(srv.docker_image),
            "--model", model_arg,
            *served,
            *parsers,
            *extra,
        ]
    model_arg = str(Path(entry.path).expanduser()) if entry is not None else srv.model
    served = ["--served-model-name", entry.name] if entry is not None else []
    return [str(srv.binary), "serve", model_arg, "--port", str(srv.port), *served, *parsers, *extra,
            *(["--host", "127.0.0.1"] if loop else [])]


def effective_posture(spec: Any, *, keyed: bool, decision: LaunchDecision) -> tuple[bool, str | None]:
    """(exposed, line): does the launch bind beyond this machine, and the exposure phrase when it
    does so with no key. Read from the argv `serve_command` would build for `decision` plus the key
    actually held — never from the `bind_all` / `require_api_key` flags, so `require_api_key: true`
    with no key held still reads `every interface, no key`. Ollama, LM Studio and a refused launch
    expose nothing and build no command. Pure; the key file is never opened."""
    runtime = getattr(spec, "runtime", "vllm")
    if runtime in ("ollama", "lmstudio") or decision.refusal is not None:
        return False, None
    cmd = serve_command(spec, keyed=keyed, loopback=decision.loopback, key_file_ok=decision.key_file_ok)
    if runtime != "llamacpp" and getattr(spec, "launch", "binary") == "docker":
        exposed = not cmd[cmd.index("-p") + 1].startswith("127.0.0.1:")
    else:
        host: str | None = None
        for i, token in enumerate(cmd):
            if token == "--host" and i + 1 < len(cmd):
                host = cmd[i + 1]
            elif token.startswith("--host="):
                host = token[len("--host="):]
        # llama-server's own default is loopback, vLLM's is every interface
        exposed = runtime != "llamacpp" if host is None else classify_hosts(host) != "loopback"
    argv_keyed = keyed or any(
        token in ("--api-key", "--api-key-file") or token.startswith(("--api-key=", "--api-key-file="))
        for token in cmd)
    return exposed, EVERY_INTERFACE_NO_KEY if exposed and not argv_keyed else None


def _scrubbed(runtime: str, name: str) -> bool:
    """True when `name` is in the launched runtime's own argv-equivalent namespace (D-15).

    A blocklist: it names only what that runtime reads as a setting or a key. HF_TOKEN,
    CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, PATH and HF_HOME are never in it."""
    if runtime == "llamacpp":
        return name.startswith("LLAMA_ARG_") or name in {"LLAMA_API_KEY", "MTMD_BACKEND_DEVICE"}
    if runtime == "vllm":
        return name == "VLLM_API_KEY"
    return False


def child_environment(
    runtime: str,
    base: Mapping[str, str],
    *,
    record_env: Mapping[str, str] | None = None,
    api_key: str | None = None,
    xdg_config_home: str | None = None,
) -> tuple[dict[str, str] | None, list[str]]:
    """The environment a launched server inherits, and the sorted names dropped from `base`.

    Built in this order: copy `base`, scrub the launched runtime's own namespace, apply the
    record's `env:`, then set `KEY_ENV[runtime]` last when keyed, so nothing inherited or recorded
    can replace the key the harness holds. The scrub is a blocklist, never an allowlist:
    CUDA_VISIBLE_DEVICES, LD_LIBRARY_PATH, PATH and HF_HOME must survive. Pure over `base`.

    `xdg_config_home` (a llama.cpp launch's harness-owned config directory, D-30) is set after the
    scrub and the record env, before the key, so nothing inherited or recorded can point the server
    at another `config.ini`. It is not in the scrub list: it is added, never dropped.

    With nothing dropped, no record env, no key and no `xdg_config_home` it returns `(None, [])`:
    Popen inherits the environment unchanged, byte for byte as 0.17 did."""
    dropped = sorted(name for name in base if _scrubbed(runtime, name))
    if not dropped and not record_env and not api_key and xdg_config_home is None:
        return None, []
    env = {name: value for name, value in base.items() if name not in dropped}
    env.update(record_env or {})
    if xdg_config_home is not None:
        env["XDG_CONFIG_HOME"] = xdg_config_home
    if api_key:
        if runtime not in KEY_ENV:
            raise ValueError(f"runtime {runtime!r} launches with no key variable")
        env[KEY_ENV[runtime]] = api_key
    return env, dropped


def start_server(
    config_dir: Path,
    cmd: list[str],
    *,
    runtime: str = "vllm",
    api_key: str | None = None,
    record_env: Mapping[str, str] | None = None,
    dropped_out: list[str] | None = None,
    xdg_config_home: str | None = None,
    port: int | None = None,
) -> int:
    """Launch detached (new session), append stdout+stderr to serve.log, write pidfile.

    The pidfile is `<pid> <start> <port>` (the start time and `port`, when known; two tokens without
    a port, one without a start time), written through `atomic_write_text` by descriptor into the
    server directory, which is opened no-follow BEFORE anything starts: a planted link at the pidfile
    is replaced, never followed, and a symlinked server directory stops the launch.

    `api_key` (a server set to require one) goes into the child's copied environment as
    `KEY_ENV[runtime]` — never into `cmd`, so it is in neither the process list nor the launch line
    logged here. `child_environment` scrubs the launched runtime's own namespace first and adds
    `record_env` (the record's `env:`); the dropped names are logged, names only. A caller that
    wants them for the ledger and the start summary passes `dropped_out`, which they extend.
    `xdg_config_home` is the llama.cpp config directory `prepare_config_ini_home` made for this
    launch; it rides the child environment only, so argv and the logged launch line are unchanged."""
    env, dropped = child_environment(
        runtime, os.environ, record_env=record_env, api_key=api_key, xdg_config_home=xdg_config_home)
    if dropped_out is not None:
        dropped_out.extend(dropped)
    if dropped:
        logger.info("launch environment scrubbed", runtime=runtime, dropped=dropped)
    if runtime == "vllm":
        inherited = sorted(n for n in os.environ if n.startswith("VLLM_") and n != "VLLM_API_KEY")
        if inherited:
            logger.info("launch environment inherits VLLM_ settings", runtime=runtime, names=inherited)
    try:
        dir_fd = private_files.open_private_dir(server_dir(config_dir))
    except OSError as exc:
        # A launch with no pidfile would be an untracked heavy: nothing starts without one.
        raise RuntimeError(
            f"could not open the server directory {server_dir(config_dir)}: {exc.strerror or exc}"
        ) from exc
    try:
        log = open(log_path(config_dir), "ab", opener=private_opener)
        log.write(f"\n=== launch: {' '.join(cmd)} ===\n".encode())
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
        except OSError as exc:
            # A binary that is set but missing or not executable passes `launch_decision` (it never
            # checks the disk) and fails only here. Raise the failure every `activate` caller already
            # prints and restores the incumbent from (RuntimeError), not a bare OSError that escapes
            # them after the incumbent was stopped; and do not leak the serve.log handle.
            log.close()
            raise RuntimeError(f"could not launch {cmd[0]}: {exc.strerror or exc}") from exc
        start = _proc_start_time(proc.pid)
        tokens = [str(proc.pid)]
        if start is not None:
            tokens.append(str(start))
            if port is not None and 1 <= port <= 65535:
                tokens.append(str(port))
        private_files.atomic_write_text(procfs.PIDFILE_NAME, " ".join(tokens), dir_fd=dir_fd)
        return proc.pid
    finally:
        os.close(dir_fd)


def _read_proc_stat(pid: int) -> bytes | None:
    """The raw `/proc/<pid>/stat`, or None when it cannot be read. The seam tests patch."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            return fh.read()
    except OSError:
        return None


def _is_zombie(pid: int) -> bool:
    """Linux: a zombie is DEAD for lifecycle purposes. os.kill(pid, 0) succeeds on one —
    a crashed `docker run --rm` client sits unreaped under the REPL (nothing wait()s it),
    which defeated wait_ready's fail-fast until the full timeout (#99). Non-Linux: False
    (managed-server lifecycle is POSIX-only; /proc is the cheap authoritative check)."""
    raw = _read_proc_stat(pid)
    if raw is None:
        return False
    try:
        # field 3 (after the parenthesised comm, which may itself contain parens)
        return raw.rsplit(b")", 1)[1].split()[0] == b"Z"
    except IndexError:
        return False


# The one parser the pidfile code and `provider.procfs` share lives in procfs (server imports it,
# never the reverse); the name stays importable from here.
start_time_from_stat = procfs.start_time_from_stat


def _proc_start_time(pid: int) -> int | None:
    """Field 22 of `/proc/<pid>/stat`: the process's start time, or None when /proc has no such
    entry or the line is not shaped like a stat line (see `start_time_from_stat`)."""
    raw = _read_proc_stat(pid)
    if raw is None:
        return None
    return start_time_from_stat(raw)


def _parse_pidfile(text: str) -> tuple[int, int | None, int | None]:
    """`<pid>`, `<pid> <start>` or `<pid> <start> <port>` -> (pid, start, port); the missing parts read
    as None, and a bare pid means "match by pid only" (D-26). Anything else, and any pid <= 1, raises
    ValueError. One parser: `procfs.parse_pidfile`."""
    return procfs.parse_pidfile(text)


def _read_pidfile(config_dir: Path) -> tuple[int, int | None, int | None] | None:
    """The pidfile's (pid, start, port), or None when it is missing or malformed. Read by descriptor
    in a no-follow server directory (see `procfs.read_pidfile`)."""
    return procfs.read_pidfile(config_dir)


def _drop_pidfile(config_dir: Path) -> None:
    """Unlink the pidfile by descriptor in a no-follow server directory; a missing file or directory,
    or a symlinked directory, is nothing to remove."""
    try:
        dir_fd = private_files.open_private_dir(server_dir(config_dir), create=False)
    except OSError:
        return
    try:
        os.unlink(procfs.PIDFILE_NAME, dir_fd=dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


def _same_process(pid: int, start: int | None) -> bool:
    """True when `pid` is the process the harness launched (D-17): its /proc start time equals the
    recorded one. `start is None` (a 0.17 bare-pid pidfile) matches by pid alone and reads no /proc.
    A missing /proc entry is False, never an exception."""
    if start is None:
        return True
    return _proc_start_time(pid) == start


def _start_kw(start: int | None) -> dict[str, int]:
    """`{"start": start}` when a start time is recorded, else `{}`: splatting it keeps the suite's
    one-argument `_alive` fakes valid for bare-pid pidfiles."""
    return {"start": start} if start is not None else {}


def server_pid(config_dir: Path) -> int | None:
    """Pid from the pidfile if that process is alive, else None (stale file removed).

    A zombie counts as dead (#99): reap it opportunistically if it's ours, drop the
    pidfile, report None — so wait_ready fails fast with the crash log tail instead of
    polling a corpse for half an hour. A pid whose start time differs from the recorded one
    was reused by an unrelated process: the harness's own is gone, so the pidfile is dropped.
    Built on `procfs.read_launch`; the start-time test it applies reads `/proc` through
    `_read_proc_stat`."""
    launch = procfs.read_launch(config_dir, stat_reader=lambda pid: _read_proc_stat(pid))
    if launch is None:  # missing, malformed, or a reused pid
        _drop_pidfile(config_dir)
        return None
    pid = launch.pid
    try:
        os.kill(pid, 0)
        if _is_zombie(pid):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            _drop_pidfile(config_dir)
            return None
        return pid
    except ProcessLookupError:
        _drop_pidfile(config_dir)
        return None
    except PermissionError:  # alive, owned by another user
        return None


NOT_A_LAUNCH_LINE = "pidfile names pid {pid} ({exe}), which is not a harness launch — not signalled; stop it, or remove {pidfile}"
"""The one line a refused stop raises (D-27): the pidfile is a claim, not proof, and `exe` is the
basename of the process's executable (`?` when `/proc` refused the read)."""

NOT_THIS_LAUNCH_LINE = "{label}: the live launch (pid {pid}) is not this record's — not waited on"
"""What `wait_ready(expect=)` raises when the live launch is not the record's (D-18)."""


def _launch_refusal(config_dir: Path, pid: int, expect: procfs.LaunchIdentity | None) -> str | None:
    """None when `pid` is proven to be a launch of `expect`; otherwise the refusal line. Nothing is
    proven without `expect` (`expect=None` never authorises a signal), and a process `/proc` will not
    show is not proven either (D-27)."""
    record = procfs.read_process(pid)
    exe = "?"
    if isinstance(record, procfs.ProcRecord):
        if record.exe is not None:
            exe = os.path.basename(record.exe)
        if expect is not None and procfs.identity_mismatch(record, expect) is None:
            return None
    return NOT_A_LAUNCH_LINE.format(pid=pid, exe=exe, pidfile=pid_path(config_dir))


def stop_server(config_dir: Path, launch: str = "binary", timeout_seconds: float = 30.0, *, expect: procfs.LaunchIdentity | None) -> bool:
    """SIGTERM the process group, SIGKILL on timeout. Docker: also stop by name.

    `expect` is required (D-27): nothing is signalled unless the pidfile's live process matches it
    (`procfs.identity_mismatch`); `expect=None` never authorises a signal. A mismatch raises
    `NOT_A_LAUNCH_LINE` and leaves the pidfile; the docker name-based stop runs only after that verdict
    passed or when no pid is live.

    Returns True if something was stopped — and ONLY once death is confirmed: raises
    RuntimeError (naming the surviving pid / container) rather than report a false free."""
    pid = server_pid(config_dir)
    recorded = _read_pidfile(config_dir)
    start = recorded[1] if recorded is not None and recorded[0] == pid else None
    stopped = False
    if pid is not None and _same_process(pid, start):
        refusal = _launch_refusal(config_dir, pid, expect)
        if refusal is not None:
            raise RuntimeError(refusal)
        try:
            os.killpg(pid, signal.SIGTERM)  # start_new_session=True → pid == pgid
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and _alive(pid, **_start_kw(start)):
            time.sleep(0.2)
        if _alive(pid, **_start_kw(start)):
            if _same_process(pid, start):  # the pid may have been reused during the grace
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            # VERIFIED death, same contract as the docker branch below: SIGKILL delivery is
            # ASYNCHRONOUS — a process unmapping tens of GiB of weights, or blocked in
            # uninterruptible I/O, stays resident for seconds after the signal. Returning True
            # here told free_accelerator the accelerator was free while the old model still held
            # its pages (two heavies on unified memory), and unlinking the pidfile made the
            # strategies' liveness report "no daemon" for a live process. Fail explicit instead.
            if not _wait_dead(pid, attempts=20, **_start_kw(start)):  # 10s at the 0.5s poll
                raise RuntimeError(
                    f"pid {pid} still alive after SIGTERM+SIGKILL — refusing to report the "
                    "accelerator free while the process still holds it"
                )
        stopped = True
    if launch == "docker":
        # VERIFIED stop (#100): a vLLM drain can outlast the grace, and the --rm removal
        # races the next `docker run` into a name conflict — the old container must be
        # GONE (name free), not merely signalled, before this returns. Longer grace,
        # poll the daemon, force-remove as fallback, and fail explicit if the name never
        # frees: returning with the name taken guarantees the swap crash this prevents.
        subprocess.run(
            ["docker", "stop", "-t", "60", DOCKER_CONTAINER_NAME],
            capture_output=True,
        )
        if not _wait_name_free(DOCKER_CONTAINER_NAME, attempts=60):
            subprocess.run(
                ["docker", "rm", "-f", DOCKER_CONTAINER_NAME],
                capture_output=True,
            )
            if not _wait_name_free(DOCKER_CONTAINER_NAME, attempts=30):
                raise RuntimeError(
                    f"container {DOCKER_CONTAINER_NAME} still exists after stop + rm -f — "
                    "refusing to race a relaunch into the name conflict"
                )
        stopped = True
    _drop_pidfile(config_dir)
    return stopped


def _wait_dead(pid: int, attempts: int, poll_seconds: float = 0.5, *, start: int | None = None) -> bool:
    """Poll until the pid is gone (post-SIGKILL confirmation). Attempt-counted so tests stay
    deterministic with sleep patched out — mirrors _wait_name_free's contract. A pid reused by
    another process counts as gone: `start` is the recorded start time."""
    for _ in range(max(1, attempts)):
        if not _alive(pid, **_start_kw(start)):
            return True
        time.sleep(poll_seconds)
    return False


def _wait_name_free(name: str, attempts: int, poll_seconds: float = 0.5) -> bool:
    """Poll `docker inspect` until the container name no longer resolves (removal
    complete). Attempt-counted so tests stay deterministic with sleep patched out."""
    for _ in range(max(1, attempts)):
        rc = subprocess.run(["docker", "inspect", name], capture_output=True).returncode
        if rc != 0:
            return True
        time.sleep(poll_seconds)
    return False


def docker_container_running(name: str = DOCKER_CONTAINER_NAME) -> bool:
    """True iff a container named `name` exists AND is currently running (NAME-based liveness).

    The fix for the docker liveness bug: docker mode runs a *foreground* `docker run --rm`
    client whose pid the pidfile holds, and that pid is the sig-proxy CLIENT — never the
    container — so `server_pid()` there answers "is the docker-run client alive", not "is the
    server up" (the orphan-client-pid bug class, #99/#100). `docker inspect -f
    '{{.State.Running}}'` prints `true`/`false` for an existing container (rc 0) and errors
    (rc != 0) when the name does not resolve, so a MISSING container and a STOPPED one both
    read as not-running — exactly what a liveness check wants. Correct for `binary` mode is
    still `server_pid()` (there the pidfile pid IS the vLLM process)."""
    proc = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == b"true"


def _alive(pid: int, start: int | None = None) -> bool:
    """True while `pid` is a live, non-zombie process — and, when a start time was recorded, the
    SAME process (a reused pid is not the harness's server)."""
    if _is_zombie(pid):  # #99: a zombie can't be signalled down — don't stall stop on it
        return False
    if not _same_process(pid, start):
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


async def wait_ready(
    base_url: str,
    config_dir: Path | None = None,
    timeout_seconds: float = 1800.0,
    poll_seconds: float = 3.0,
    on_poll: "Callable[[float], None] | None" = None,
    *,
    api_key: str | None = None,
    runtime: str | None = None,
    liveness: bool = False,
    expect: procfs.LaunchIdentity | None = None,
) -> list[str]:
    """Poll {base_url}/models until the server answers; return served model ids.

    Model load can take minutes (weights → GPU), hence the generous default.
    When config_dir is given, a dead managed process fails fast with the log tail.
    `on_poll(elapsed_seconds)` fires after each unready poll — the REPL renders it as a
    live loading line so a minutes-long swap never looks frozen. Must never block.
    `api_key` (a server that requires one) is sent as a bearer on every poll; without it the
    poll is exactly the keyless request it always was. `runtime` names the server in the failure
    messages ("llama.cpp exited during startup"); unset, they read "vLLM" as they always did.

    `liveness=True` is for a reconnect wait only (`ServerRecovery`), never a launch: any HTTP answer
    below 500 means the server is back (a 401 from a key that changed, a 404 or 403 from a proxy that
    hides `/models`), and a 200 whose body is not the OpenAI shape still counts; the ids are then
    best-effort (possibly empty). The re-sent request reports the server's real status. A 5xx (503
    while a model loads) is still not ready. Launch-time callers keep requiring a well-formed 200.

    `expect` (with `config_dir`) is the proof the live launch is the record's (D-18): when the pidfile's
    process is live but does not match it the wait raises `NOT_THIS_LAUNCH_LINE` at once, so a waiting
    start never polls one record's port on another launch's pid. Unset, nothing changes.
    """
    label = RUNTIME_LABELS.get(runtime, "vLLM") if runtime else "vLLM"
    start = time.monotonic()
    deadline = start + timeout_seconds
    headers = auth_headers(api_key)
    async with httpx.AsyncClient(**({} if _TRANSPORT is None else {"transport": _TRANSPORT})) as client:
        while time.monotonic() < deadline:
            if config_dir is not None and server_pid(config_dir) is None:
                raise RuntimeError(
                    f"{label} exited during startup. Log tail:\n" + log_tail(config_dir)
                )
            if config_dir is not None and expect is not None:
                live = procfs.read_launch(
                    config_dir, expect=expect, stat_reader=lambda pid: _read_proc_stat(pid))
                if live is not None and not live.verified:
                    raise RuntimeError(NOT_THIS_LAUNCH_LINE.format(label=label, pid=live.pid))
            try:
                resp = await client.get(f"{base_url.rstrip('/')}/models", timeout=3.0,
                                        **({"headers": headers} if headers else {}))
                if resp.status_code == 200 and not liveness:
                    return [m["id"] for m in resp.json().get("data", [])]
                if liveness and resp.status_code < 500:
                    try:
                        return [m["id"] for m in resp.json().get("data", [])]
                    except (ValueError, KeyError, TypeError, AttributeError):
                        return []  # it answered: alive, whatever the body says
            except httpx.HTTPError:
                pass
            if on_poll is not None:
                on_poll(time.monotonic() - start)
            await asyncio.sleep(poll_seconds)
    tail = f" Log tail:\n{log_tail(config_dir)}" if config_dir is not None else ""
    raise TimeoutError(f"{label} not ready after {timeout_seconds:.0f}s.{tail}")


def log_tail(config_dir: Path, lines: int = 15) -> str:
    try:
        content = log_path(config_dir).read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])
    except OSError:
        return "(no log)"
