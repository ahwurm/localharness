"""DISP-03: no core module on the channel path names a chat platform (Phase 49-08).

An AST scan of every non-docstring `str` constant in the core files the `start --channel` path runs
through. Neither `discord` (the first adapter) nor `fixturechat` (the fixture second platform,
tests/integration/test_dispatch_fixture_adapter_e2e.py) may appear, case-insensitively: a platform
is known only to dispatch/ (its ADAPTERS dict and manifest `channels` tuple).

Docstrings and comments are prose and may mention Discord as an example; behaviour cannot hide in
them. Sabotage check (run by hand when changing this test): re-adding a string literal such as
`elif channel_mode == "discord":` to cli/start_cmd.py makes this test fail.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
CORE_FILES = ("cli/start_cmd.py", "cli/repl.py", "plugins/channels.py", "plugins/lifecycle.py",
              "channels/base.py", "channels/__init__.py")
PLATFORMS = ("discord", "fixturechat")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
        # attribute docstrings: a bare string expression statement right after an assignment
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(node, field, None)
            for stmt in stmts if isinstance(stmts, list) else ():
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                        and isinstance(stmt.value.value, str):
                    ids.add(id(stmt.value))
    return ids


def platform_literals(source: str) -> list[tuple[int, str]]:
    tree = ast.parse(source)
    docs = _docstring_nodes(tree)
    return [(n.lineno, n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
            and any(p in n.value.lower() for p in PLATFORMS)]


@pytest.mark.parametrize("rel", CORE_FILES)
def test_core_file_names_no_platform(rel):
    hits = platform_literals((SRC / rel).read_text(encoding="utf-8"))
    assert hits == [], f"{rel} names a chat platform in code: {hits}"


def test_the_scan_catches_a_platform_literal():
    """The sabotage case, kept as a test so the scanner cannot silently go blind."""
    planted = '"""Doc mentions Discord."""\nX = 1\n"""attr doc: discord"""\nif m == "Discord":\n    pass\n'
    assert platform_literals(planted) == [(4, "Discord")]
    assert platform_literals('f = f"dispatch.{name}"\ng = "fixturechat-room"\n') == [(2, "fixturechat-room")]
