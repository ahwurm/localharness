"""One writer of the `unit-target` store, and `private_files` stays below the packages above it.

RULING W3: `units.write_target` is called only from `UnitStrategy.activate` in provider/lifecycle.py
(it arrives with the autostart unit; today there is no caller and the pin holds vacuously). Nothing
else spells the store's path (D-06), and `core/private_files.py` imports nothing from
`localharness.config`, `.provider` or `.cli` (D-28: it sits below them).

Mutation-checked: the synthetic sources at the bottom fail if a checker stops looking at the
enclosing function / class or at the import list.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIFECYCLE = Path("provider") / "lifecycle.py"
UNITS = Path("provider") / "units.py"
PRIVATE_FILES = Path("core") / "private_files.py"
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_FORBIDDEN_UP = ("localharness.config", "localharness.provider", "localharness.cli")


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _name(func: ast.AST) -> str | None:
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> tuple[str | None, str | None]:
    """The nearest enclosing function name and the class that function is defined in (or None)."""
    func: str | None = None
    while node in parents:
        node = parents[node]
        if func is None and isinstance(node, _FUNCTIONS):
            func = node.name
            continue
        if func is not None and isinstance(node, ast.ClassDef):
            return func, node.name
        if func is not None and isinstance(node, _FUNCTIONS):
            continue  # a closure inside a method: keep climbing to the class
    return func, None


def stray_target_writers(source: str, module: Path) -> list[tuple[str, int]]:
    """Calls of `write_target` that are not inside `UnitStrategy.activate` of provider/lifecycle.py."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) == "write_target":
            func, cls = _scope(node, parents)
            if not (module == LIFECYCLE and func == "activate" and cls == "UnitStrategy"):
                out.append((str(module), node.lineno))
    return out


def target_name_literals(source: str) -> list[int]:
    """Line numbers of the string constant "unit-target" (a docstring that merely mentions it is not equal to it)."""
    return [n.lineno for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant) and n.value == "unit-target"]


def upward_imports(source: str) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        out += [n for n in names if any(n == p or n.startswith(p + ".") for p in _FORBIDDEN_UP)]
    return out


# ------------------------------------------------------------------- the live source


def test_write_target_is_called_only_from_unit_strategy_activate():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        found += stray_target_writers(path.read_text(), path.relative_to(SRC))
    assert found == []


def test_the_unit_target_path_is_spelled_only_in_units_py():
    found = [(str(p.relative_to(SRC)), line) for p in sorted(SRC.rglob("*.py"))
             if p.relative_to(SRC) != UNITS for line in target_name_literals(p.read_text())]
    assert found == []
    assert target_name_literals((SRC / UNITS).read_text()), "units.py must still define the name"


def test_private_files_imports_nothing_from_the_packages_above_it():
    assert upward_imports((SRC / PRIVATE_FILES).read_text()) == []


# ------------------------------------------------------------------- the checkers bite


def test_a_planted_caller_outside_activate_is_reported():
    src = 'def f(d):\n    units.write_target(d, "x")\n'
    assert stray_target_writers(src, LIFECYCLE) == [(str(LIFECYCLE), 2)]
    assert stray_target_writers(src, Path("cli") / "app.py") == [(str(Path("cli") / "app.py"), 2)]


def test_a_caller_in_another_method_of_the_class_is_reported():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def other(self, d):
                units.write_target(d, "x")
    ''')
    assert len(stray_target_writers(src, LIFECYCLE)) == 1


def test_the_one_allowed_caller_is_not_reported():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d):
                units.write_target(d, "x")
    ''')
    assert stray_target_writers(src, LIFECYCLE) == []
    assert len(stray_target_writers(src, Path("provider") / "other.py")) == 1


def test_a_planted_second_spelling_and_an_upward_import_are_reported():
    assert target_name_literals('p = d / "unit-target"\n') == [1]
    assert target_name_literals('"""mentions unit-target in prose"""\n') == []
    assert upward_imports("from localharness.config import overlay\n") != []
    assert upward_imports("import localharness.provider.units\n") != []
    assert upward_imports("from localharness.core import which\nimport os\n") == []
