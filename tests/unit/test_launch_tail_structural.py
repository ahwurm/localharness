"""The raw log tail is read only by `scrubbed_tail` (D-22, SEC-04, T-13-40).

`server.log_tail` returns serve.log's last lines as the server wrote them, keys included. No message may
quote it as it is, so over every `.py` under `src/localharness/`:

- every `Call` of the name or attribute `log_tail`, and every other reference to it (a bound name,
  `getattr(x, "log_tail")`, an `import log_tail`), sits inside `def scrubbed_tail` in provider/server.py;
- the definition itself (`def log_tail`) is a `FunctionDef`, not a reference, and is not reported.

A later message that quotes the raw tail therefore fails here, not in review. The synthetic cases below
fail if the checker stops looking at calls, at the other references, or at where they sit.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
NAME = "log_tail"
OWNER_FILE = "provider/server.py"
OWNER_FUNCTION = "scrubbed_tail"


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_ref(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == NAME) or (
        isinstance(node, ast.Attribute) and node.attr == NAME)


def _inside_owner(node: ast.AST, parents: dict[ast.AST, ast.AST], label: str) -> bool:
    if label != OWNER_FILE:
        return False
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == OWNER_FUNCTION:
            return True
    return False


def tail_reads(source: str, label: str) -> tuple[int, list[int]]:
    """`(calls, violations)`: how many calls of `log_tail` the source holds, and the line number of every
    call or other reference to it outside `scrubbed_tail` in provider/server.py."""
    tree = ast.parse(source)
    parents = _parents(tree)
    calls = 0
    bad: list[int] = []
    for node in ast.walk(tree):
        if _is_ref(node):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and parent.func is node:
                calls += 1
            if not _inside_owner(node, parents, label):
                bad.append(node.lineno)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == NAME and not _inside_owner(node, parents, label)):
            bad.append(node.lineno)
        elif (isinstance(node, ast.ImportFrom) and any(alias.name == NAME for alias in node.names)
              and not _inside_owner(node, parents, label)):
            bad.append(node.lineno)
    return calls, sorted(bad)


def test_log_tail_is_read_only_by_scrubbed_tail():
    calls = 0
    bad: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        label = path.relative_to(SRC).as_posix()
        found, violations = tail_reads(path.read_text(encoding="utf-8"), label)
        calls += found
        bad += [f"{label}:{line}" for line in violations]
    assert bad == [], f"the raw log tail is read outside scrubbed_tail: {bad}"
    assert calls == 1, calls  # not vacuous: scrubbed_tail's own read is the one call there is


@pytest.mark.parametrize(
    ("source", "label", "expected"),
    [
        ("def f(d):\n    return server.log_tail(d)\n", "provider/lifecycle.py", [2]),
        ("def f(d):\n    return log_tail(d)\n", "provider/lifecycle.py", [2]),
        ("def f():\n    f = server.log_tail\n", "provider/lifecycle.py", [2]),
        ("def f():\n    return getattr(server, \"log_tail\")\n", "provider/lifecycle.py", [2]),
        ("from localharness.provider.server import log_tail\n", "provider/lifecycle.py", [1]),
        # inside scrubbed_tail, but in the wrong file
        ("def scrubbed_tail(d):\n    return log_tail(d)\n", "provider/lifecycle.py", [2]),
        # the right file, the wrong function
        ("def other(d):\n    return log_tail(d)\n", "provider/server.py", [2]),
        # clean: scrubbed_tail's own read in provider/server.py
        ("def scrubbed_tail(d):\n    return log_tail(d)\n", "provider/server.py", []),
        ("def scrubbed_tail(d):\n    return [log_tail(x) for x in d]\n", "provider/server.py", []),
        # the definition and unrelated names are not reads
        ("def log_tail(d, lines=15):\n    return ''\n", "provider/server.py", []),
        ("def f(d):\n    return server.log_tails(d) + scrubbed_tail(d)\n", "provider/lifecycle.py", []),
    ],
)
def test_the_checker_reports_exactly_the_reads_outside_scrubbed_tail(source, label, expected):
    assert tail_reads(source, label)[1] == expected


def test_the_checker_counts_the_calls_it_judges():
    source = "def scrubbed_tail(d):\n    log_tail(d)\n    server.log_tail(d)\n"
    assert tail_reads(source, "provider/server.py") == (2, [])
    assert tail_reads(source, "provider/other.py")[0] == 2
