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
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from typing import Callable

from localharness.config.models import ManagedServerConfig

DOCKER_CONTAINER_NAME = "localharness-vllm"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def server_dir(config_dir: Path) -> Path:
    return Path(config_dir) / "vllm"


def log_path(config_dir: Path) -> Path:
    return server_dir(config_dir) / "serve.log"


def pid_path(config_dir: Path) -> Path:
    return server_dir(config_dir) / "server.pid"


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
    return shutil.which("vllm")


def install_vllm_venv(config_dir: Path, package: str) -> str:
    """Install `package` into a dedicated venv at <config-dir>/vllm/venv.

    Streams installer output to the terminal; raises RuntimeError on failure.
    Prefers uv (fast, `--torch-backend=auto` resolves the right CUDA torch);
    falls back to stdlib venv + pip.
    """
    venv = server_dir(config_dir) / "venv"
    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
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
    """Download a checkpoint into the HF cache (tqdm progress). Raises on failure."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id)


# ---------------------------------------------------------------------------
# Launch / readiness / stop
# ---------------------------------------------------------------------------


def serve_command(srv: ManagedServerConfig) -> list[str]:
    """Build the launch command for srv.model.

    When srv.model names a `local_models` registry entry, the entry's checkpoint dir is
    served (docker: bind-mounted read-only at /models/serving) under --served-model-name
    <entry.name>, with the entry's per-model extra_args appended after the shared ones —
    so a swap changes checkpoint, served id, AND model-specific flags in one place.
    Otherwise srv.model passes through untouched (HF repo id resolved via the cache)."""
    entry = srv.entry_for(srv.model)
    extra = [*srv.extra_args, *(entry.extra_args if entry is not None else [])]
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
            "-p", f"{srv.port}:8000",
            str(srv.docker_image),
            "--model", model_arg,
            *served,
            *extra,
        ]
    model_arg = str(Path(entry.path).expanduser()) if entry is not None else srv.model
    served = ["--served-model-name", entry.name] if entry is not None else []
    return [str(srv.binary), "serve", model_arg, "--port", str(srv.port), *served, *extra]


def start_server(config_dir: Path, cmd: list[str]) -> int:
    """Launch detached (new session), append stdout+stderr to serve.log, write pidfile."""
    server_dir(config_dir).mkdir(parents=True, exist_ok=True)
    log = open(log_path(config_dir), "ab")
    log.write(f"\n=== launch: {' '.join(cmd)} ===\n".encode())
    log.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path(config_dir).write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def _is_zombie(pid: int) -> bool:
    """Linux: a zombie is DEAD for lifecycle purposes. os.kill(pid, 0) succeeds on one —
    a crashed `docker run --rm` client sits unreaped under the REPL (nothing wait()s it),
    which defeated wait_ready's fail-fast until the full timeout (#99). Non-Linux: False
    (managed-server lifecycle is POSIX-only; /proc is the cheap authoritative check)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            # field 3 (after the parenthesised comm, which may itself contain parens)
            return fh.read().rsplit(b")", 1)[1].split()[0] == b"Z"
    except (OSError, IndexError):
        return False


def server_pid(config_dir: Path) -> int | None:
    """Pid from the pidfile if that process is alive, else None (stale file removed).

    A zombie counts as dead (#99): reap it opportunistically if it's ours, drop the
    pidfile, report None — so wait_ready fails fast with the crash log tail instead of
    polling a corpse for half an hour."""
    path = pid_path(config_dir)
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        if _is_zombie(pid):
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            path.unlink(missing_ok=True)
            return None
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError):
        path.unlink(missing_ok=True)
        return None
    except PermissionError:  # alive, owned by another user
        return None


def stop_server(config_dir: Path, launch: str = "binary", timeout_seconds: float = 30.0) -> bool:
    """SIGTERM the process group, SIGKILL on timeout. Docker: also stop by name.

    Returns True if something was stopped."""
    pid = server_pid(config_dir)
    stopped = False
    if pid is not None:
        try:
            os.killpg(pid, signal.SIGTERM)  # start_new_session=True → pid == pgid
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.2)
        if _alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
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
    pid_path(config_dir).unlink(missing_ok=True)
    return stopped


def _wait_name_free(name: str, attempts: int, poll_seconds: float = 0.5) -> bool:
    """Poll `docker inspect` until the container name no longer resolves (removal
    complete). Attempt-counted so tests stay deterministic with sleep patched out."""
    for _ in range(max(1, attempts)):
        rc = subprocess.run(["docker", "inspect", name], capture_output=True).returncode
        if rc != 0:
            return True
        time.sleep(poll_seconds)
    return False


def _alive(pid: int) -> bool:
    if _is_zombie(pid):  # #99: a zombie can't be signalled down — don't stall stop on it
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
) -> list[str]:
    """Poll {base_url}/models until the server answers; return served model ids.

    Model load can take minutes (weights → GPU), hence the generous default.
    When config_dir is given, a dead managed process fails fast with the log tail.
    `on_poll(elapsed_seconds)` fires after each unready poll — the REPL renders it as a
    live loading line so a minutes-long swap never looks frozen. Must never block.
    """
    start = time.monotonic()
    deadline = start + timeout_seconds
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            if config_dir is not None and server_pid(config_dir) is None:
                raise RuntimeError(
                    "vLLM exited during startup. Log tail:\n" + log_tail(config_dir)
                )
            try:
                resp = await client.get(f"{base_url.rstrip('/')}/models", timeout=3.0)
                if resp.status_code == 200:
                    return [m["id"] for m in resp.json().get("data", [])]
            except httpx.HTTPError:
                pass
            if on_poll is not None:
                on_poll(time.monotonic() - start)
            await asyncio.sleep(poll_seconds)
    tail = f" Log tail:\n{log_tail(config_dir)}" if config_dir is not None else ""
    raise TimeoutError(f"vLLM not ready after {timeout_seconds:.0f}s.{tail}")


def log_tail(config_dir: Path, lines: int = 15) -> str:
    try:
        content = log_path(config_dir).read_text(encoding="utf-8", errors="replace")
        return "\n".join(content.splitlines()[-lines:])
    except OSError:
        return "(no log)"
