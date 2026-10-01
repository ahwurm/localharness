"""MEMP-09 / PAPI-04: subagents reach memory only through the slot's per-delegation handle —
the subagent path and the agent tool never hold, name, or import a raw memory store."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


@pytest.mark.parametrize("rel", ["agent/subagent.py", "tools/builtin/agent_tool.py"])
def test_subagent_and_agent_tool_take_no_store(rel):
    tree = ast.parse((SRC / rel).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            assert node.arg != "memory_store", f"{rel}: parameter memory_store at line {node.lineno}"
        if isinstance(node, ast.Name):
            assert node.id != "MemoryStore", f"{rel}: MemoryStore at line {node.lineno}"
        if isinstance(node, ast.Attribute):
            assert node.attr != "MemoryStore", f"{rel}: .MemoryStore at line {node.lineno}"
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not (mod == "localharness.memory" or mod.startswith("localharness.memory.")), \
                f"{rel}: imports from {mod} at line {node.lineno}"
            assert "MemoryStore" not in {a.name for a in node.names}, f"{rel}: imports MemoryStore"
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("localharness.memory") for a in node.names), rel
