"""GrepTool: Search LOCAL file contents for a regex pattern (bounded, fail-fast walk)."""
import asyncio
import os
import re
import time
from fnmatch import fnmatch

from localharness.tools.builtin.paths import resolve_user_path

from localharness.tools.base import Tool, ToolResult, ToolSchema

# Bounded-walk defaults — module-level so they are self-documenting and patchable in tests.
MAX_FILE_BYTES = 1_000_000       # skip files larger than this (memory guard vs multi-GB blobs)
BINARY_SNIFF_BYTES = 8192        # bytes read to sniff for a NUL byte (binary marker)
SCAN_FILE_CAP = 20_000           # max files visited before returning partial results
SCAN_TIME_BUDGET_S = 20.0        # soft wall-clock budget; return partial results past it

# Directory names pruned at walk time. Every hidden dir (name startswith ".") is also pruned
# unless include_hidden=True — the dotted names below are belt-and-suspenders documentation.
EXCLUDED_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".cache",
    ".tox", ".mypy_cache", ".ruff_cache", "dist", "build", ".eggs",
})


def _iter_candidate_files(root: str, glob: str, include_hidden: bool):
    """Deterministic pre-order os.scandir walk that prunes excluded/hidden dirs before
    descending and yields file paths whose basename matches `glob`. No full-tree sort:
    entries are sorted per directory only, so traversal starts producing immediately."""
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            continue
        subdirs: list[str] = []
        for entry in entries:
            name = entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if name.startswith("."):
                    if not include_hidden:
                        continue
                elif name in EXCLUDED_DIRS:
                    continue
                subdirs.append(entry.path)
            else:
                if name.startswith(".") and not include_hidden:
                    continue
                if fnmatch(name, glob):
                    yield entry.path
        stack.extend(reversed(subdirs))  # pop in sorted order -> deterministic pre-order


def _read_text_guarded(path: str) -> str | None:
    """Decoded text, or None to skip the file: oversized (stat) or binary (NUL in first 8KB).
    Opens once — sniff the head, then read the already size-bounded remainder and decode."""
    try:
        if os.stat(path).st_size > MAX_FILE_BYTES:
            return None
        with open(path, "rb") as fh:
            head = fh.read(BINARY_SNIFF_BYTES)
            if b"\x00" in head:
                return None
            rest = fh.read()
    except OSError:
        return None
    return (head + rest).decode("utf-8", "replace")


class GrepTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="grep",
            description=(
                "Search LOCAL FILE contents on disk for a regex pattern. Returns matching "
                "lines with file path and line number. Searches recursively if path is a "
                "directory. For information that is not in local files, use web_search."
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
                    "include_hidden": {
                        "type": "boolean",
                        "description": "Also search hidden files and vendor/VCS dirs "
                        "(dotfiles, .git, .venv). Off by default.",
                        "default": False,
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
        include_hidden: bool = False,
    ) -> ToolResult:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            compiled = re.compile(pattern, flags)
        except re.error as exc:
            return self.err(f"Invalid regex: {exc}", error_type="validation_error")

        target = resolve_user_path(path)
        if not target.exists():
            return self.err(f"Path does not exist: {target}")

        if target.is_file():
            # Explicitly named file: no walk/exclusions, but the size/binary guards still apply.
            candidates = iter([str(target)])
        else:
            candidates = _iter_candidate_files(str(target), glob, include_hidden)

        loop = asyncio.get_running_loop()
        lines_out: list[str] = []
        total = 0
        scanned = 0
        start_t = time.monotonic()
        deadline = start_t + SCAN_TIME_BUDGET_S
        capped = False

        for fpath in candidates:
            if scanned >= SCAN_FILE_CAP or time.monotonic() >= deadline:
                capped = True
                break
            scanned += 1
            text = await loop.run_in_executor(None, _read_text_guarded, fpath)
            if text is None:
                continue
            file_lines = text.splitlines()
            for i, line in enumerate(file_lines):
                if compiled.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(file_lines), i + context_lines + 1)
                    for j in range(start, end):
                        lines_out.append(f"{fpath}:{j+1}: {file_lines[j]}")
                    total += 1
                    if total >= limit:
                        lines_out.append(f"... (limit {limit} reached)")
                        return self.ok("\n".join(lines_out), match_count=total, truncated=True)

        if capped:
            elapsed = time.monotonic() - start_t
            if lines_out:
                lines_out.append(
                    f"... (scan capped: {scanned} files / {elapsed:.1f}s — narrow path or glob)"
                )
                return self.ok("\n".join(lines_out), match_count=total, truncated=True)
            return self.ok(
                f"(no matches in first {scanned} files — scan capped; narrow path or glob)",
                match_count=0,
                truncated=True,
            )

        if not lines_out:
            return self.ok("(no matches)")
        return self.ok("\n".join(lines_out), match_count=total)
