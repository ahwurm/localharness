"""CORE-02's line as a ratchet (D5): no core module imports a plugin module.

The milestone is a small core plus plugins, and the line between them is a test, not a folder move.
Every .py under src/localharness is classified by an explicit list, then an AST scan collects every
core -> plugin import edge — lazy imports inside functions, `if TYPE_CHECKING:` blocks (a typing
import is still a dependency) and string-literal `import_module(...)` calls included.

BURN_DOWN names today's legacy edges, keyed by (importer, imported) FILE PAIR — never by line
number, which every later plan moves. A new edge fails; an entry whose import is gone fails too, so
the list only shrinks. The helpers are importable (the phase's criterion-5 test composes them).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"

# Directories that are wholly one kind: the directory entry IS the explicit list.
PLUGIN_DIRS = ("memory/", "dispatch/", "channels/mobile/", "autoresearch/", "tools/builtin/workflows/")
CORE_DIRS = ("core/", "config/", "provider/", "agent/", "orchestrator/", "registry/", "plugins/",
             "bench/")
# cli/, channels/ and tools/ hold both kinds, so every file in them is named. A new file there is
# nobody's until someone decides which side of the line it is on — plugin code moved into a new
# module, or a plugin file turned into a package, never becomes core by default.
PLUGIN_FILES = frozenset({
    "tools/builtin/memory_tools.py", "tools/builtin/generate_image_tool.py", "tools/builtin/image_plugin.py",
    "cli/memory_cmd.py", "cli/memory_cli.py", "cli/mobile_cmd.py", "cli/mobile_plugin.py", "cli/generate_image_cmd.py",
    "cli/autoresearch_cmd.py", "cli/experiment_cmd.py", "cli/propose_cmd.py", "cli/report_cmd.py",
})
_MIXED_DIR_CORE = {
    "cli": ("__init__", "acp_cmd", "agent_cmd", "app", "askrate_cmd", "bench_cmd", "components_cmd",
            "config_cmd", "doctor_cmd", "errors", "init_cmd", "model_cmd", "model_ops",
            "plugin_mount", "plugins_cmd", "repl",
            "session_accumulator", "session_resume", "session_trust", "slash_commands", "start_cmd", "theme", "ui",
            "update_cmd", "validate_cmd", "workspace"),
    "channels": ("__init__", "acp", "base", "errors", "input_router", "terminal"),
    "tools": ("__init__", "base", "capabilities", "hooks", "mcp", "registry"),
    "tools/builtin": ("__init__", "agent_tool", "bash_tool", "chunk_tool", "cruncher_exec",
                      "edit_tool", "glob_tool", "grep_tool", "load_document_tool", "netguard", "paths",
                      "python_tool", "read_tool", "tool_result_get_tool", "web_tool", "write_tool"),
}
CORE_FILES = frozenset({"__init__.py",
                        *(f"{d}/{n}.py" for d, ns in _MIXED_DIR_CORE.items() for n in ns)})
# "Only the bundled-plugin list and config routing may reference plugins" (CORE-02).
ALLOWED_PLUGIN_IMPORTERS = frozenset({"plugins/builtin.py"})

BURN_DOWN: frozenset[tuple[str, str]] = frozenset()

_DYNAMIC_IMPORTS = frozenset({"import_module", "__import__"})


def classify(rel: str) -> str | None:
    """'plugin' | 'core' | None for a posix path relative to SRC. Plugin lists are checked first,
    so a per-plugin file inside a core directory is a plugin file wherever it lives."""
    if rel in PLUGIN_FILES or rel.startswith(PLUGIN_DIRS):
        return "plugin"
    if rel in CORE_FILES or rel.startswith(CORE_DIRS):
        return "core"
    return None


def module_file(module: str) -> str | None:
    """'localharness.a.b' -> 'a/b.py' or 'a/b/__init__.py', whichever exists; else None."""
    head, _, rest = module.partition(".")
    if head != "localharness":
        return None
    base = rest.replace(".", "/")
    candidates = (f"{base}.py", f"{base}/__init__.py") if base else ("__init__.py",)
    return next((rel for rel in candidates if (SRC / rel).is_file()), None)


def _imported_modules(rel: str, node: ast.AST) -> list[str]:
    """Dotted module names one AST node imports, relative imports resolved against `rel`."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        package = ["localharness", *rel.removesuffix(".py").split("/")][:-1]
        base = package[: len(package) + 1 - node.level] if node.level else []
        module = ".".join([*base, *([node.module] if node.module else [])])
        # `from pkg import sub` imports the submodule; `from pkg import Name` imports pkg.
        return [f"{module}.{a.name}" if module_file(f"{module}.{a.name}") else module
                for a in node.names]
    if (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) in _DYNAMIC_IMPORTS):
        return [node.args[0].value]
    return []


def edges_of(rel: str, text: str) -> set[tuple[str, str]]:
    """Every (core importer, plugin file) edge in one file's source — empty unless `rel` is core."""
    if classify(rel) != "core":
        return set()
    files = {module_file(m)
             for node in ast.walk(ast.parse(text, rel)) for m in _imported_modules(rel, node)}
    return {(rel, f) for f in files if f and classify(f) == "plugin"}


def _sources() -> list[tuple[str, Path]]:
    found = [(p.relative_to(SRC).as_posix(), p) for p in sorted(SRC.rglob("*.py"))]
    assert len(found) >= 130, f"scanned {len(found)} files under {SRC} — the root is wrong"
    return found


def scan() -> set[tuple[str, str]]:
    """Every core -> plugin import edge in the source tree."""
    return {e for rel, path in _sources() for e in edges_of(rel, path.read_text(encoding="utf-8"))}


def violations(edges) -> set[tuple[str, str]]:
    """Edges not on BURN_DOWN, except from the one core module allowed to name plugins."""
    return {e for e in set(edges) - BURN_DOWN if e[0] not in ALLOWED_PLUGIN_IMPORTERS}


def _pairs(edges) -> str:
    return "".join(f"\n  {a} -> {b}" for a, b in sorted(edges))


def test_every_source_file_is_classified():
    """An unclassified file fails: name it in PLUGIN_FILES or CORE_FILES (or a one-kind dir)."""
    unclassified = [rel for rel, _ in _sources() if classify(rel) is None]
    assert unclassified == [], (
        "classify these explicitly — CORE-02 classifies by list, not by directory:"
        + "".join(f"\n  {rel}" for rel in unclassified))


def test_files_are_classified_by_explicit_list_not_directory():
    assert classify("tools/builtin/generate_image_tool.py") == "plugin"
    assert classify("tools/builtin/memory_tools.py") == "plugin"
    assert classify("tools/builtin/read_tool.py") == "core"
    assert classify("cli/mobile_cmd.py") == "plugin"
    assert classify("channels/terminal.py") == "core"
    assert classify("plugins/api.py") == "core"
    # In a directory holding both kinds, a new file is nobody's until it is named.
    assert classify("tools/builtin/new_tool.py") is None
    assert classify("channels/discord/__init__.py") is None


def test_no_core_module_imports_a_plugin_module_off_the_burn_down_list():
    """The ratchet. BURN_DOWN is keyed by file pair because line numbers move; each plugin
    conversion deletes its own entries, and the list is empty at the autoresearch conversion —
    which is when CORE-02 is marked complete (D5). Nothing new may join it."""
    new = violations(scan())
    assert not new, (
        "a core module may not import a plugin module (CORE-02) — route it through a core contract "
        "or the plugin's own contribution:" + _pairs(new))


def test_the_burn_down_list_only_shrinks():
    """An entry whose import is gone must leave the list, or a later edge could hide behind it."""
    stale = BURN_DOWN - scan()
    assert not stale, "no longer imported; remove it — the list only shrinks:" + _pairs(stale)


def test_the_burn_down_residue_is_exactly_the_remaining_conversions():
    """Every legacy core->plugin edge is converted; a new edge fails the ratchet test. Memory,
    dispatch and autoresearch all come in through plugins/builtin.py; cli/app.py mounts the
    autoresearch, experiment and propose commands lazily from the plugin's manifest."""
    assert BURN_DOWN == frozenset()
    # The bundled-plugin list is the one core module that names the autoresearch plugin.
    assert {a for a, b in scan() if b == "autoresearch/plugin.py"} == {"plugins/builtin.py"}
    assert not {e for e in scan() if e[0] == "cli/app.py" and e[1] in _AUTORESEARCH_CLI}
    # The bundled-plugin list is the one core module that names the dispatch plugin.
    assert {a for a, b in scan() if b.startswith("dispatch/")} == {"plugins/builtin.py"}


def test_an_injected_plugin_import_in_agent_loop_is_reported():
    """Criterion 5: agent/loop.py contributes no edge today; one `from localharness.memory import`
    line added to it is a violation."""
    text = (SRC / "agent" / "loop.py").read_text(encoding="utf-8")
    assert edges_of("agent/loop.py", text) == set()
    injected = edges_of("agent/loop.py", text + "\nfrom localharness.memory import MemoryStore\n")
    assert ("agent/loop.py", "memory/__init__.py") in injected
    assert violations(injected)


@pytest.mark.parametrize("rel, source, target", [
    ("agent/loop.py", "def f():\n    from localharness.memory.sqlite import MemoryStore\n",
     "memory/sqlite.py"),
    ("agent/loop.py", "if TYPE_CHECKING:\n    from localharness.channels import mobile\n",
     "channels/mobile/__init__.py"),
    ("agent/loop.py", "import localharness.autoresearch.loop as ar\n", "autoresearch/loop.py"),
    ("agent/loop.py", "importlib.import_module('localharness.dispatch.plugin')\n",
     "dispatch/plugin.py"),
    ("cli/app.py", "from . import memory_cmd\n", "cli/memory_cmd.py"),
    ("cli/app.py", "from .. import memory\n", "memory/__init__.py"),
    ("tools/builtin/__init__.py", "from .memory_tools import MemorySearchTool\n",
     "tools/builtin/memory_tools.py"),
])
def test_every_import_shape_is_an_edge(rel, source, target):
    """Lazy, typing-only, aliased, dynamic and relative imports are all dependencies."""
    assert (rel, target) in edges_of(rel, source)


def test_only_core_to_plugin_is_the_rule():
    """An import from a plugin file into core is fine — only core -> plugin is the rule. The
    bundled-plugin list is the one core module allowed to import plugins."""
    assert edges_of("memory/router.py", "from localharness.agent.loop import AgentLoop\n") == set()
    allowed = edges_of("plugins/builtin.py", "from localharness.memory import MemoryStore\n")
    assert allowed == {("plugins/builtin.py", "memory/__init__.py")}
    assert violations(allowed) == set()
    elsewhere_in_core = {("plugins/api.py", "memory/__init__.py")}
    assert violations(elsewhere_in_core) == elsewhere_in_core


_AUTORESEARCH_CLI = frozenset({"cli/autoresearch_cmd.py", "cli/experiment_cmd.py", "cli/propose_cmd.py",
                               "cli/report_cmd.py"})


def autoresearch_edges(rel: str, text: str) -> set[tuple[str, str]]:
    """Every (rel, file) edge from one file's source into autoresearch/ or its four CLI modules."""
    files = {module_file(m)
             for node in ast.walk(ast.parse(text, rel)) for m in _imported_modules(rel, node)}
    return {(rel, f) for f in files
            if f and (f.startswith("autoresearch/") or f in _AUTORESEARCH_CLI)}


def _bench_sources() -> list[tuple[str, Path]]:
    found = [(rel, p) for rel, p in _sources() if rel.startswith("bench/") or rel == "cli/bench_cmd.py"]
    assert len(found) >= 5, f"only {len(found)} bench files found — the root is wrong"
    return found


def test_bench_imports_nothing_from_autoresearch():
    """AUTO-03: the bench is core tooling the plugin calls, never the reverse. No module under
    bench/ (nor `localharness bench`'s CLI) imports autoresearch/ or an autoresearch CLI module —
    so the bench runs with the autoresearch plugin off."""
    edges = {e for rel, p in _bench_sources() for e in autoresearch_edges(rel, p.read_text(encoding="utf-8"))}
    assert not edges, "the bench may not import autoresearch:" + _pairs(edges)


def test_an_injected_autoresearch_import_in_bench_is_reported():
    rel, path = next((r, p) for r, p in _bench_sources() if r == "bench/orchestrator.py")
    text = path.read_text(encoding="utf-8")
    assert autoresearch_edges(rel, text) == set()
    injected = autoresearch_edges(rel, text + "\nfrom localharness.autoresearch import archive\n")
    assert (rel, "autoresearch/archive.py") in injected
    lazy = autoresearch_edges("cli/bench_cmd.py", "def f():\n    from localharness.cli import experiment_cmd\n")
    assert lazy == {("cli/bench_cmd.py", "cli/experiment_cmd.py")}
