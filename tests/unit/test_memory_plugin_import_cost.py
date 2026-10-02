"""The memory plugin module is light: plugins/builtin.py will import it for every --help, so it must
not pull aiosqlite, numpy or the store. The package's public names still resolve (lazily)."""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_memory_plugin_import_is_light() -> None:
    code = ("import sys, localharness.memory.plugin; "
            "print([m for m in ('aiosqlite', 'numpy', 'localharness.memory.sqlite') if m in sys.modules])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_package_names_still_resolve() -> None:
    import localharness.memory as mem
    from localharness.memory import (  # noqa: F401
        VALID_WRITABLE_SECTIONS, DiskFullError, Fact, FactQuery, HistoryWriter, MarkdownMemory,
        MemoryContext, MemoryError, MemoryStore,
    )
    import localharness.memory.sqlite as sq
    assert mem.MemoryStore is sq.MemoryStore
    with pytest.raises(AttributeError):
        mem.NoSuchName  # noqa: B018


def test_manifest() -> None:
    from localharness.memory.plugin import MemoryPlugin
    from localharness.plugins.api import PluginManifest, SlashDescriptor, plugin_summary
    # 48: /memory is the plugin's own slash row (PAPI-07) — the one manifest field added.
    assert MemoryPlugin.manifest == PluginManifest(
        name="memory", version="0.1.0", kind="memory", enabled_by_default=True,
        slash=(SlashDescriptor(name="/memory",
                               help="Browse the agent's memory by tag; show/forget/search a memory",
                               target="localharness.memory.plugin:MemoryPlugin.slash_memory"),))
    assert plugin_summary(MemoryPlugin) == (
        "persistent memory: facts recalled into each turn, memory tools, background consolidation")
    assert MemoryPlugin.wants_artifacts is False
    assert MemoryPlugin.ConfigModel is None
