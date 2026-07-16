"""BashExecTool: Execute bash commands."""
import asyncio
import os
import shutil
from pathlib import Path

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


def _find_bash() -> str | None:
    """Locate a real bash, never the WSL stub.

    On Windows, PowerShell's PATH order typically resolves `bash` to
    C:\\Windows\\System32\\bash.exe — the WSL launcher. Without a WSL distro it prints a
    UTF-16LE error and exits (observed live: NUL-riddled mojibake observations that
    stuck-looped an agent), and WITH one it would run commands in a different filesystem
    view than the native file tools. Neither is ever what bash_exec means by "bash", so
    the stub is rejected outright and git-bash is searched explicitly.
    LOCALHARNESS_BASH overrides everything. Returns None when nothing usable exists.
    """
    override = os.environ.get("LOCALHARNESS_BASH")
    if override:
        return override
    which = shutil.which("bash")
    if os.name != "nt":
        return which or "/bin/bash"
    candidates = []
    if which and "system32" not in which.lower():
        candidates.append(which)
    for base in (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.path.join(os.environ.get("LocalAppData", ""), "Programs"),
    ):
        if base:
            candidates.append(os.path.join(base, "Git", "usr", "bin", "bash.exe"))
            candidates.append(os.path.join(base, "Git", "bin", "bash.exe"))
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
            proc = await asyncio.create_subprocess_exec(
                bash,
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    output=f"Command timed out after {timeout}s: {command}",
                    success=False,
                    error=f"Timeout after {timeout}s",
                    error_type="timeout_error",
                )
        except OSError as exc:
            return self.err(str(exc))

        output = _decode_output(stdout)
        return self.ok(
            output or "(no output)",
            exit_code=proc.returncode,
            command=command,
        )
