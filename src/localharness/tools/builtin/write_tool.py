"""WriteTool: Write or overwrite a file."""
import asyncio
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class WriteTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="write",
            description=(
                "Write or overwrite a file. Creates parent directories if needed. "
                "Returns the absolute path written and byte count."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode. Default: overwrite.",
                        "default": "overwrite",
                    },
                },
                "required": ["path", "content"],
            },
            destructive=True,
            estimated_tokens=150,
        )

    async def _execute(self, path: str, content: str, mode: str = "overwrite") -> ToolResult:
        target = Path(path).expanduser().resolve()

        forbidden_suffixes = {".env", ".secret", ".token", ".pem", ".key"}
        if target.suffix in forbidden_suffixes or target.name.startswith(".env"):
            return self.err(
                f"Write to credential/secret file blocked: {target}",
                error_type="permission_denied",
            )

        if (denied := self._outside_workspace(target)) is not None:
            return denied

        target.parent.mkdir(parents=True, exist_ok=True)

        open_mode = "a" if mode == "append" else "w"
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: target.open(open_mode, encoding="utf-8").write(content)
            )
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        return self.ok(
            f"Written {len(content.encode())} bytes to {target}",
            path=str(target),
            bytes_written=len(content.encode()),
        )
