"""BashExecTool: Execute bash commands."""
import asyncio
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema


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
                        "description": "Working directory for the command. Defaults to CWD.",
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
        cwd = Path(working_dir).resolve()
        if not cwd.exists():
            return self.err(f"working_dir does not exist: {cwd}")

        timeout = min(timeout_s, 300.0)
        try:
            proc = await asyncio.create_subprocess_shell(
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

        output = stdout.decode("utf-8", errors="replace")
        return self.ok(
            output or "(no output)",
            exit_code=proc.returncode,
            command=command,
        )
