"""Structural pin: every caller of `interview.ask` is a function that took `person` and returned before asking."""
from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
INTERVIEW = SRC / "cli" / "interview.py"


def _ask_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to the interview module, names bound to `ask` itself)."""
    modules, asks = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "interview":
                    modules.add(a.asname or a.name)
                if node.module and node.module.endswith("interview") and a.name == "ask":
                    asks.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.endswith(".interview") and a.asname:
                    modules.add(a.asname)
    return modules, asks


def _is_ask_call(node: ast.AST, modules: set[str], asks: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr == "ask" and isinstance(f.value, ast.Name) and f.value.id in modules
    return isinstance(f, ast.Name) and f.id in asks


def _guards_person_before(fn: ast.FunctionDef | ast.AsyncFunctionDef, line: int) -> bool:
    if "person" not in {a.arg for a in fn.args.args + fn.args.kwonlyargs}:
        return False
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and node.lineno < line
                and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name) and node.test.operand.id == "person"
                and any(isinstance(s, (ast.Return, ast.Raise)) for s in node.body)):
            return True
    return False


def unguarded_callers(source: str, name: str = "<source>") -> list[str]:
    tree = ast.parse(source)
    modules, asks = _ask_names(tree)
    if not (modules or asks):
        return []
    found = []
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(fn):
                if _is_ask_call(node, modules, asks) and not _guards_person_before(fn, node.lineno):
                    found.append(f"{name}:{node.lineno} in {fn.name}")
    return found


def test_every_interview_ask_caller_is_behind_person():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path != INTERVIEW:
            found += unguarded_callers(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    assert found == []


GUARDED = '''
from localharness.cli import interview

def settle(files, person):
    if not person:
        return None
    return interview.ask("q", [], default=0, cancel="skip")
'''


def test_the_rule_passes_a_guarded_caller():
    assert unguarded_callers(GUARDED) == []


def test_the_rule_names_a_caller_without_a_person_guard():
    source = GUARDED.replace("def settle(files, person):\n    if not person:\n        return None\n",
                             "def settle(files):\n")
    assert unguarded_callers(source) == ["<source>:5 in settle"]


def test_the_rule_names_a_guard_that_comes_after_the_ask():
    source = '''
from localharness.cli.interview import ask as question

def settle(person):
    answer = question("q", [], default=0, cancel="skip")
    if not person:
        return None
    return answer
'''
    assert unguarded_callers(source) == ["<source>:5 in settle"]


def test_the_rule_names_a_person_guard_that_does_not_leave():
    source = GUARDED.replace("return None", "print('no')")
    assert len(unguarded_callers(source)) == 1
