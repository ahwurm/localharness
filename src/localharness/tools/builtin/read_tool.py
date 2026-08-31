"""ReadTool: Read file contents with line numbers."""
import asyncio

from localharness.tools.builtin.grep_tool import BINARY_SNIFF_BYTES
from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema

# Bounded-read guard — module-level so it's self-documenting and patchable in tests. Shares
# grep_tool's BINARY_SNIFF_BYTES (imported above) rather than duplicating the constant, so
# `read` and `grep` always agree on what counts as binary.
MAX_RETURNED_CHARS = 100_000   # hard char cap, independent of offset/limit (line count): a
# binary or no-newline file can pack an entire multi-hundred-KB payload into ONE "line",
# which bypasses the line-based limit entirely. Live incident (2026-08-30): `read` on a
# SQLite memory.db returned ~287,000 chars of replacement-character soup in one call,
# instantly overflowing the context budget — this cap is the defense-in-depth backstop
# for ANY oversized single result, binary or not.


def _looks_binary(head: bytes) -> bool:
    """NUL-byte sniff on the first BINARY_SNIFF_BYTES of the file — the same heuristic
    grep_tool._read_text_guarded uses to skip binary files (same constant, imported above)."""
    return b"\x00" in head


class ReadTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            description=(
                "Read file contents. Returns the file as a string with line numbers "
                "prepended (format: 'N\\t<line>'). Supports optional line range. Refuses "
                "binary files (databases, images, archives, etc.) rather than dumping raw "
                "bytes — use bash_exec with a format-specific tool (e.g. sqlite3 for a "
                ".db file) to inspect those instead."
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
            raw = await loop.run_in_executor(None, target.read_bytes)
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

        if _looks_binary(raw[:BINARY_SNIFF_BYTES]):
            return self.err(
                f"{target} looks like a binary file (a NUL byte in the first "
                f"{BINARY_SNIFF_BYTES} bytes) — refusing to read it as text. For a SQLite "
                "database, use bash_exec with sqlite3 (e.g. `sqlite3 <path> '.schema'`) "
                "instead of read.",
                error_type="validation_error",
            )
        text = raw.decode("utf-8", "replace")

        all_lines = text.splitlines()
        total_lines = len(all_lines)
        start = max(0, offset - 1)
        selected = all_lines[start : start + limit]

        numbered = "\n".join(f"{start + i + 1}\t{line}" for i, line in enumerate(selected))
        full_len = len(numbered)
        truncated = full_len > MAX_RETURNED_CHARS
        if truncated:
            numbered = (
                numbered[:MAX_RETURNED_CHARS]
                + f"\n... [truncated at {MAX_RETURNED_CHARS} of {full_len} chars — narrow "
                "offset/limit to read further sections]"
            )
        return self.ok(
            numbered, total_lines=total_lines, lines_returned=len(selected),
            truncated=truncated, original_length=full_len if truncated else None,
        )
