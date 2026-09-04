"""BashExecTool: Execute bash commands."""
import asyncio
import contextlib
import os
import shutil
import signal
from pathlib import Path

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


def _find_bash() -> str | None:
    """Locate a real bash, never a WSL stub.

    On Windows, PowerShell's PATH order typically resolves `bash` to
    C:\\Windows\\System32\\bash.exe (or the Store alias under WindowsApps) — the WSL
    launcher. Without a WSL distro it prints a UTF-16LE error and exits (observed live:
    NUL-riddled mojibake observations that stuck-looped an agent), and WITH one it would run
    commands in a different filesystem view than the native file tools. Neither is ever what
    bash_exec means by "bash", so the stubs are rejected outright and git-bash is searched
    explicitly.

    Git for Windows ships two bash executables. `Git\\bin\\bash.exe` is the wrapper that
    puts `/usr/bin` (coreutils) on PATH before exec'ing `Git\\usr\\bin\\bash.exe`; the inner
    binary, launched directly, inherits the harness's PATH as-is. From a PowerShell-started
    harness that PATH has no /usr/bin, so every `mkdir`/`ls`/`cp` was "command not found"
    (observed live: three `mkdir -p` calls in a row, all reported ✓). The suite never saw it
    because pytest under git-bash already has /usr/bin on PATH. The wrapper is therefore
    searched first. LOCALHARNESS_BASH overrides everything. Returns None when nothing
    usable exists.
    """
    override = os.environ.get("LOCALHARNESS_BASH")
    if override:
        return override
    which = shutil.which("bash")
    if os.name != "nt":
        return which or "/bin/bash"
    candidates = []
    if which and not any(stub in which.lower() for stub in ("system32", "windowsapps")):
        candidates.append(which)
    for base in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.path.join(os.environ.get("LocalAppData", ""), "Programs"),
    ):
        if base:
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
            candidates.append(os.path.join(base, "Git", "usr", "bin", "bash.exe"))
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _decode_output(raw: bytes) -> str:
    """Decode subprocess output: UTF-8 first, UTF-16 when the bytes say so.

    Windows System32 tools emit UTF-16LE; naive utf-8 decoding renders it as
    \\x00-interleaved mojibake the model cannot read (and will loop on).
    """
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:64]:
        for enc in ("utf-16", "utf-16-le"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    return raw.decode("utf-8", errors="replace")


class _WindowsJob:
    """A Windows job object holding the command's whole process tree (nt only).

    `taskkill /T` walks Windows parent links, and git-bash's forked children are not linked
    to bash that way (verified live: `sleep 30 &` survived `taskkill /F /T` on the bash pid
    and kept the stdout pipe open). A job object catches every process the tree spawns
    after assignment, whatever its parent, and TerminateJobObject kills them all at once —
    the pipe closed within 10ms in the same experiment. No KILL_ON_JOB_CLOSE: a command
    that finishes normally may leave background children running, as it does on POSIX.
    Every failure degrades to `assigned=False`, and _kill_and_reap then falls back to taskkill.
    """

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self.assigned = False
        self._job = None
        try:
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._k32 = k32
            self._job = k32.CreateJobObjectW(None, None) or None
        except (OSError, AttributeError):
            self._k32 = None

    def assign(self, pid: int) -> bool:
        if not self._job:
            return False
        handle = self._k32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
        if not handle:
            return False
        try:
            self.assigned = bool(self._k32.AssignProcessToJobObject(self._job, handle))
        finally:
            self._k32.CloseHandle(handle)
        return self.assigned

    def terminate(self) -> bool:
        return bool(self._job and self.assigned and self._k32.TerminateJobObject(self._job, 1))

    def close(self) -> None:
        if self._job:
            self._k32.CloseHandle(self._job)
            self._job = None


async def _kill_tree(proc: asyncio.subprocess.Process, job: "_WindowsJob | None") -> None:
    """Kill the command's whole process tree, not just the bash parent.

    proc.kill() reaches bash alone. A grandchild still holding the stdout pipe — the live
    case was an interactive cmd.exe; `sleep 30 &` reproduces it anywhere — lived on as an
    orphan. Windows: the job object (taskkill /T as a fallback). POSIX: the command was
    started in its own session, so the process group IS the tree.
    """
    if job is not None and job.terminate():
        return
    if os.name == "nt":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.shield(asyncio.wait_for(killer.wait(), timeout=5.0))
        except (OSError, asyncio.TimeoutError):
            pass
    else:
        with contextlib.suppress(OSError):
            os.killpg(proc.pid, signal.SIGKILL)


async def _kill_and_reap(
    proc: asyncio.subprocess.Process, job: "_WindowsJob | None" = None
) -> None:
    """Kill a still-running child — and its whole tree — and reap it, on EVERY exit path (#153).

    `communicate()` is cancellable, and CancelledError is a BaseException: a mid-turn Ctrl+C
    (the REPL's turn cancel, box-mode interrupt, or the exit drain) sails straight past an
    `except asyncio.TimeoutError`, so before this the child kept running with its pipes open —
    one leaked process per cancelled turn.

    kill() is a synchronous SIGNAL (the child is doomed the moment it returns), but the reap —
    which is what closes the transport's pipes — needs an await, and an await inside a `finally`
    that is unwinding a cancellation can itself be interrupted. So the wait is SHIELDED (it
    survives as a task on the still-running loop) and its CancelledError suppressed, leaving the
    ORIGINAL cancellation to propagate untouched.
    """
    if proc.returncode is not None:
        return
    await _kill_tree(proc, job)
    try:
        proc.kill()
    except ProcessLookupError:  # raced us to exit — nothing left to signal
        return
    with contextlib.suppress(asyncio.CancelledError, ProcessLookupError):
        await asyncio.shield(proc.wait())


class BashExecTool(Tool):
    timeout_s: float = 60.0

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="bash_exec",
            description=(
                "Execute a bash command and return combined stdout+stderr. "
                "Working directory is the harness working directory. "
                "Environment inherits from the harness process."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute.",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "Command timeout in seconds. Max 300.",
                        "default": 60.0,
                        "minimum": 1.0,
                        "maximum": 300.0,
                    },
                    "working_dir": {
                        "type": "string",
                        "description": (
                            "Working directory for the command. Defaults to the workspace root "
                            "when confined (relative paths anchor there), else the harness CWD."
                        ),
                        "default": ".",
                    },
                },
                "required": ["command"],
            },
            destructive=True,
            estimated_tokens=300,
        )

    async def _execute(
        self, command: str, timeout_s: float = 60.0, working_dir: str = "."
    ) -> ToolResult:
        # Confined (workspace_root set): relative working_dir — including the untouched default
        # "." — anchors at the workspace root, so the resting behavior is "your cwd IS the
        # workspace", not "your default call errors". Escapes after resolve() ("../x") are still
        # denied below. Unconfined (None): ambient-CWD resolution, unchanged.
        if self.workspace_root is not None and not Path(working_dir).is_absolute():
            cwd = (Path(self.workspace_root).expanduser().resolve() / working_dir).resolve()
        else:
            cwd = resolve_user_path(working_dir)
        if (denied := self._outside_workspace(cwd)) is not None:
            return denied
        if not cwd.exists():
            return self.err(f"working_dir does not exist: {cwd}")

        timeout = min(timeout_s, 300.0)
        try:
            bash = _find_bash()
            if not bash:
                return self.err(
                    "bash not found — install Git for Windows (git-bash) or point "
                    "LOCALHARNESS_BASH at a bash executable",
                    error_type="execution_error",
                )
            # exec form, not create_subprocess_shell: shell mode defaults to /bin/sh (dash on
            # Ubuntu), which lacks brace expansion, `[[ ]]`, arrays, etc. — a "bash_exec" tool
            # must actually run bash so the model's bashisms behave as written. And on Windows,
            # shell mode re-parses the interpreter path, so a git-bash install under
            # "C:\Program Files\..." splits at the space and never launches; exec + explicit
            # `-c` passes the path as one argv entry on every platform.
            # stdin=DEVNULL: the model's command must never read the harness's terminal.
            # Observed live (Windows): `cmd /c "…"` under git-bash — MSYS path-converts the
            # /c flag, cmd starts INTERACTIVE on the inherited stdin and sits there for the
            # full timeout. start_new_session (POSIX) makes the process group the whole
            # tree so a timeout can kill all of it (see _kill_tree).
            proc = await asyncio.create_subprocess_exec(
                bash,
                "-c",
                command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
                **({} if os.name == "nt" else {"start_new_session": True}),
            )
            job = _WindowsJob() if os.name == "nt" else None
            if job is not None:
                job.assign(proc.pid)  # before bash can fork: it is still loading
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                return ToolResult(
                    output=f"Command timed out after {timeout}s: {command}",
                    success=False,
                    error=f"Timeout after {timeout}s",
                    error_type="timeout_error",
                )
            finally:
                # Timeout, cancellation, or any other escape: the child never outlives this call.
                await _kill_and_reap(proc, job)
                if job is not None:
                    job.close()
        except OSError as exc:
            return self.err(str(exc))

        output = _decode_output(stdout)
        rc = proc.returncode
        if rc != 0:
            # A non-zero exit is a failure, not a success with a number in metadata: the
            # terminal showed ✓ and the model got plain text it read as done. The loop
            # forwards .error (not .output) on failure, so the command's own output rides
            # along in the message — "mkdir: command not found" is what the model must see.
            return ToolResult(
                output=output,
                success=False,
                error=f"exit code {rc}: {output.strip() or '(no output)'}",
                error_type="execution_error",
                metadata={"exit_code": rc, "command": command},
            )
        return self.ok(
            output or "(no output)",
            exit_code=rc,
            command=command,
        )
