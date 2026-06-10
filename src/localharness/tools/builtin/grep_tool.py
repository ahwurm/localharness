"""GrepTool: Search file contents for a regex pattern."""
import asyncio
import re
from pathlib import Path

from localharness.tools.base import Tool, ToolResult, ToolSchema


class GrepTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description=(
                "Search file contents for a regex pattern. Returns matching lines "
                "with file path and line number. Searches recursively if path is a directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Python regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "File glob filter when path is a directory, e.g. '*.py'.",
                        "default": "*",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "Case-insensitive matching.",
                        "default": False,
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context before and after each match.",
                        "default": 0,
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of matching lines to return.",
                        "default": 200,
                        "minimum": 1,
                        "maximum": 2000,
                    },
                },
                "required": ["pattern", "path"],
            },
            destructive=False,
            estimated_tokens=500,
        )

    async def _execute(
        self,
        pattern: str,
        path: str,
        glob: str = "*",
        ignore_case: bool = False,
        context_lines: int = 0,
        limit: int = 200,
    ) -> ToolResult:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return self.err(f"Invalid regex: {exc}", error_type="validation_error")

        target = Path(path).expanduser().resolve()
        if not target.exists():
            return self.err(f"Path does not exist: {target}")

        files = [target] if target.is_file() else sorted(target.rglob(glob))
        lines_out: list[str] = []
        total = 0

        loop = asyncio.get_running_loop()
        for f in files:
            if not f.is_file():
                continue
            try:
                text = await loop.run_in_executor(None, f.read_text, "utf-8", "replace")
            except OSError:
                continue
            file_lines = text.splitlines()
            for i, line in enumerate(file_lines):
                if compiled.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(file_lines), i + context_lines + 1)
                    for j in range(start, end):
                        lines_out.append(f"{f}:{j+1}: {file_lines[j]}")
                    total += 1
                    if total >= limit:
                        lines_out.append(f"... (limit {limit} reached)")
                        return self.ok("\n".join(lines_out), match_count=total, truncated=True)

        if not lines_out:
            return self.ok("(no matches)")
        return self.ok("\n".join(lines_out), match_count=total)
