"""No module outside provider/runtimes.py compares a runtime against a string literal: a runtime is one row."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
TABLE = "provider/runtimes.py"
_OPS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn)


def _is_str(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _literal(node: ast.expr) -> bool:
    return _is_str(node) or (isinstance(node, (ast.Tuple, ast.List, ast.Set)) and any(map(_is_str, node.elts)))


def _runtime(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "runtime" or node.id.endswith("_runtime")
    if isinstance(node, ast.Attribute):
        return node.attr == "runtime"
    if isinstance(node, ast.Subscript):
        return _is_str(node.slice) and node.slice.value == "runtime"
    if isinstance(node, ast.Call):
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        args = node.args[1:2] if name == "getattr" else node.args[:1] if name == "get" else []
        return bool(args) and _is_str(args[0]) and args[0].value == "runtime"
    return False


def runtime_comparisons(source: str, name: str) -> list[str]:
    """`name:line` of every comparison or match of a runtime expression against string literals."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            if any(isinstance(op, _OPS) for op in node.ops) and any(map(_runtime, operands)) \
                    and any(map(_literal, operands)):
                hits.append(node.lineno)
        elif isinstance(node, ast.Match) and _runtime(node.subject) and any(
                isinstance(p, ast.MatchValue) and _is_str(p.value)
                for case in node.cases for p in ast.walk(case.pattern)):
            hits.append(node.lineno)
    return [f"{name}:{line}" for line in sorted(hits)]


def test_no_runtime_literal_comparison_outside_the_table():
    found = [hit for path in sorted(SRC.rglob("*.py")) if path.relative_to(SRC).as_posix() != TABLE
             for hit in runtime_comparisons(path.read_text(encoding="utf-8"), path.relative_to(SRC).as_posix())]
    assert found == []


@pytest.mark.parametrize("line", [
    'x.runtime == "a"',
    'runtime != "a"',
    '_runtime in ("a", "b")',
    'getattr(x, "runtime", None) == "a"',
    'x.get("runtime", "vllm") == "a"',
    'x["runtime"] not in ["a"]',
    '"a" == spec.runtime',
])
def test_each_spelling_is_reported(line):
    assert runtime_comparisons(f"if {line}:\n    pass\n", "m.py") == ["m.py:1"]


def test_a_match_on_a_runtime_with_string_cases_is_reported():
    source = textwrap.dedent('''
        match spec.runtime:
            case "a":
                pass
    ''')
    assert runtime_comparisons(source, "m.py") == ["m.py:2"]


@pytest.mark.parametrize("line", ["life.runtime == primary.runtime", "row.gguf", "x.runtime is None", "name == 'a'"])
def test_a_comparison_of_two_runtimes_or_a_non_runtime_is_not_reported(line):
    assert runtime_comparisons(f"if {line}:\n    pass\n", "m.py") == []


def test_the_report_is_sorted_path_and_line():
    source = 'a = 1\nif x.runtime == "b":\n    pass\nif runtime != "c":\n    pass\n'
    assert runtime_comparisons(source, "z.py") == ["z.py:2", "z.py:4"]
