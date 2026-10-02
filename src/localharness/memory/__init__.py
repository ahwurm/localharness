"""Memory persistence primitives for LocalHarness agents."""

_LAZY = {
    "MemoryStore": ".sqlite", "Fact": ".sqlite", "FactQuery": ".sqlite", "MemoryContext": ".sqlite",
    "HistoryWriter": ".history", "MarkdownMemory": ".markdown", "VALID_WRITABLE_SECTIONS": ".markdown",
    "MemoryError": ".errors", "MemoryWriteError": ".errors", "MemoryReadError": ".errors",
    "MemoryCorruptionError": ".errors", "DiskFullError": ".errors",
}


def __getattr__(name: str):
    """Import on first use, so importing a light submodule (memory.plugin, memory.config) does not
    pull aiosqlite into every --help (the plugin-module import rule)."""
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MemoryStore",
    "Fact",
    "FactQuery",
    "MemoryContext",
    "HistoryWriter",
    "MarkdownMemory",
    "VALID_WRITABLE_SECTIONS",
    "MemoryError",
    "MemoryWriteError",
    "MemoryReadError",
    "MemoryCorruptionError",
    "DiskFullError",
]
