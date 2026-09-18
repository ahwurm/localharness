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


# The memory store's own file names (owner order 2026-09-18). `memory.db` covers the whole
# SQLite family — memory.db, -wal, -shm, .bak — because the check is a substring of the name.
MEMORY_STORE_FILENAMES: tuple[str, ...] = ("memory.db", "memory-archive")
MEMORY_STORE_READ_GUIDANCE = (
    "This is a memory store, and it is not readable from a tool that opens files — not with "
    "read, and not through a shell either. Use your memory tools instead: memory_search to "
    "find facts, memory_get to read one, remember to store one. They read the LIVE store; an "
    "archived or left-over store file is deliberately out of reach, so that stale facts cannot "
    "come back wearing the authority of current ones."
)


def _looks_binary(head: bytes) -> bool:
    """NUL-byte sniff on the first BINARY_SNIFF_BYTES of the file — the same heuristic
    grep_tool._read_text_guarded uses to skip binary files (same constant, imported above)."""
    return b"\x00" in head


def _is_memory_store_artifact(target) -> bool:
    """Is this path one of the memory store's own files? Name-based (case-folded), so it holds
    for any directory — a copy, a backup, another agent's store, a wiped-and-left-behind one."""
    name = target.name.lower()
    return any(stem in name for stem in MEMORY_STORE_FILENAMES)


class ReadTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="read",
            group="fs.read",
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
            # PRD §4: with an editor attached, read the BUFFER, not the disk — otherwise the
            # agent reasons about a version of the file the user is no longer looking at. The
            # binary sniff is a disk-path guard: a client that hands back text has already
            # decided the file is text, and an editor cannot open a SQLite database as a buffer.
            if self.file_read_hook is not None:
                text = await self.file_read_hook(target)
            else:
                raw = await loop.run_in_executor(None, target.read_bytes)
                if _looks_binary(raw[:BINARY_SNIFF_BYTES]):
                    head = (f"{target} looks like a binary file (a NUL byte in the first "
                            f"{BINARY_SNIFF_BYTES} bytes) — refusing to read it as text. ")
                    # The generic hint pointed at `bash_exec` + `sqlite3` — which, for a MEMORY
                    # STORE, is the one route the shipped deny patterns exist to close (owner
                    # order 2026-09-18). Handing the model a recipe for the blocked path is how
                    # a refusal turns into a retry loop, so the store gets its own sentence.
                    return self.err(
                        head + MEMORY_STORE_READ_GUIDANCE
                        if _is_memory_store_artifact(target)
                        else head + "For a SQLite database, use bash_exec with sqlite3 (e.g. "
                             "`sqlite3 <path> '.schema'`) instead of read.",
                        error_type="validation_error",
                    )
                text = raw.decode("utf-8", "replace")
        except PermissionError:
            return self.err(f"Permission denied: {target}", error_type="permission_denied")
        except OSError as exc:
            return self.err(str(exc))

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
