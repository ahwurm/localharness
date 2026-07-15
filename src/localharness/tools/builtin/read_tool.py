"""ReadTool: Read file contents with line numbers."""
import asyncio

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class ReadTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            description=(
                "Read file contents. Returns the file as a string with line numbers "
                "prepended (format: 'N\\t<line>'). Supports optional line range."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "First line to read (1-indexed). Default: 1.",
                        "default": 1,
                        "minimum": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default: 2000.",
                        "default": 2000,
                        "minimum": 1,
                        "maximum": 10000,
                    },
                },
                "required": ["path"],
            },
            destructive=False,
            estimated_tokens=800,
        )

    async def _execute(self, path: str, offset: int = 1, limit: int = 2000) -> ToolResult:
        target = resolve_user_path(path)
        if not target.exists():
            return self.err(f"File not found: {target}", error_type="not_found")
        if target.is_dir():
            return self.err(f"Path is a directory, not a file: {target}")

        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(None, target.read_text, "utf-8", "replace")
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        all_lines = text.splitlines()
        total_lines = len(all_lines)
        start = max(0, offset - 1)
        selected = all_lines[start : start + limit]

        numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        return self.ok(numbered, total_lines=total_lines, lines_returned=len(selected))
