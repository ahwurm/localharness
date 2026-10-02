"""Autoresearch domain package: mutation archive (Phase 15), proposer/experiment/orchestrator (later phases)."""

_LAZY = {"ArchiveStore": ".archive", "ArchiveEntry": ".archive", "ArchiveQuery": ".archive"}


def __getattr__(name: str):
    """Import on first use, so importing autoresearch.plugin (every --help, via plugins/builtin.py)
    does not pull aiosqlite in through the archive (the plugin-module import rule)."""
    if name in _LAZY:
        from importlib import import_module
        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ArchiveStore", "ArchiveEntry", "ArchiveQuery"]
