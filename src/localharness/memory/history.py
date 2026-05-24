"""Append-only JSONL writer for chat history."""
import errno
import json
from pathlib import Path
from typing import Any

import anyio

from localharness.memory.errors import DiskFullError, MemoryCorruptionError, MemoryWriteError

VALID_TYPES = frozenset({
    "user_message",
    "assistant_message",
    "tool_result",
    "system_message",
    "session_event",
})

REQUIRED_FIELDS = frozenset({"v", "type", "id", "session_id", "agent_id", "ts"})


class HistoryWriter:
    """
    Low-level append-only JSONL writer for chat history.

    Used internally by MemoryStore. Uses anyio file I/O with O_APPEND mode.
    On POSIX, O_APPEND writes are atomic up to PIPE_BUF (4096 bytes).
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    async def append(self, record: dict[str, Any]) -> None:
        """Serialize record to JSON and append to file with newline. Creates file if needed."""
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Record missing required fields: {sorted(missing)}")
        if record["type"] not in VALID_TYPES:
            raise ValueError(
                f"Unknown record type: {record['type']!r}. Valid: {sorted(VALID_TYPES)}"
            )
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with await anyio.open_file(str(self._path), "a", encoding="utf-8") as f:
                await f.write(line + "\n")
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise DiskFullError(str(self._path), exc) from exc
            raise MemoryWriteError(str(self._path), exc) from exc

    async def read_all(self) -> list[dict[str, Any]]:
        """
        Read and parse all records from the file.

        Returns empty list if file does not exist.
        Partial last line (crash write) is skipped silently.
        Mid-file corruption raises MemoryCorruptionError.
        """
        if not self._path.exists():
            return []
        try:
            text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            from localharness.memory.errors import MemoryReadError
            raise MemoryReadError(str(self._path), exc) from exc

        lines = text.splitlines()
        records: list[dict[str, Any]] = []
        for lineno, raw in enumerate(lines, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError:
                # Last line: partial crash write — skip silently
                if lineno == len(lines):
                    break
                raise MemoryCorruptionError(
                    str(self._path),
                    f"Line {lineno}: {raw[:100]}",
                )
        return records

    async def read_last_n(self, n: int) -> list[dict[str, Any]]:
        """Read the last n records. Simple approach: read_all() then tail."""
        all_records = await self.read_all()
        return all_records[-n:] if n < len(all_records) else all_records
