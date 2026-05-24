"""Memory error hierarchy for LocalHarness memory subsystem."""


class MemoryError(Exception):
    """Base class for all memory errors."""
    pass


class MemoryWriteError(MemoryError):
    """SQLite write or file I/O write failure."""

    def __init__(self, path: str, underlying: Exception) -> None:
        self.path = path
        self.underlying = underlying
        super().__init__(f"Memory write failed: {path}: {underlying}")


class MemoryReadError(MemoryError):
    """SQLite read or file I/O read failure."""

    def __init__(self, path: str, underlying: Exception) -> None:
        self.path = path
        self.underlying = underlying
        super().__init__(f"Memory read failed: {path}: {underlying}")


class MemoryCorruptionError(MemoryError):
    """Detected corruption: PRAGMA integrity_check failed, or JSONL line fails JSON parsing."""

    def __init__(self, path: str, detail: str) -> None:
        self.path = path
        self.detail = detail
        super().__init__(f"Memory corruption detected: {path}: {detail}")


class SessionNotFoundError(MemoryError):
    """session_id has no records in history.jsonl."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class DiskFullError(MemoryWriteError):
    """Write failed due to ENOSPC. Subclass of MemoryWriteError."""
    pass
