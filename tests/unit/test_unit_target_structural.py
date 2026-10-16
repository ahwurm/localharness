"""One writer of the `unit-target` store, and `private_files` stays below the packages above it.

RULING W3: `units.write_target` is called only from `UnitStrategy.activate` in provider/lifecycle.py, and the
settling of the file lives in `activate`'s OWN body. `activate` holds three `write_target` calls: the in-flight
write (exactly one call passes `in_flight=True`, as a constant) and, inside the `finalbody` of ONE `try`
statement of its own body, the plain conversion and the rollback (R-14-38, R-14-41). Every non-in-flight
`write_target` and every `remove_target` call of `activate` lies in such a `finalbody`, so no exit path can
settle the file by a statement of its own (a call on the success path, in a handler or in a closure nested in
`activate` is reported). Nothing else spells the store's path (D-06), and `core/private_files.py` imports
nothing from `localharness.config`, `.provider` or `.cli` (D-28: it sits below them).

Mutation-checked: the synthetic sources at the bottom fail if a checker stops looking at the
enclosing function / class, at the number of calls in `activate`, at `in_flight=True`, at the `finally`, or at
the import list.
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


def activate_write_count(source: str, module: Path) -> int:
    """How many `write_target` calls sit directly in `UnitStrategy.activate`'s own body of provider/lifecycle.py
    (a call in a closure nested in it is another function and is not counted)."""
    if module != LIFECYCLE:
        return 0
    tree = ast.parse(source)
    parents = _parents(tree)
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.Call) and _name(node.func) == "write_target"
               and _scope(node, parents) == ("activate", "UnitStrategy"))


def _is_in_flight(node: ast.Call) -> bool:
    return any(k.arg == "in_flight" and isinstance(k.value, ast.Constant) and k.value.value is True
               for k in node.keywords)


def in_flight_call_count(source: str, module: Path) -> int:
    """How many `write_target(..., in_flight=True)` calls (a constant True) sit anywhere in provider/lifecycle.py's
    `UnitStrategy.activate` own body; elsewhere `stray_target_writers` reports every call."""
    if module != LIFECYCLE:
        return 0
    tree = ast.parse(source)
    parents = _parents(tree)
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, ast.Call) and _name(node.func) == "write_target" and _is_in_flight(node)
               and _scope(node, parents) == ("activate", "UnitStrategy"))


def settled_outside_finally(source: str, module: Path) -> list[int]:
    """Line numbers of the non-in-flight `write_target` calls and the `remove_target` calls in
    `UnitStrategy.activate`'s own body of provider/lifecycle.py that are NOT inside the `finalbody` of an
    `ast.Try` (a call on the success path, in an `except` clause or `else`, or in a closure is outside)."""
    if module != LIFECYCLE:
        return []
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        if name == "write_target" and _is_in_flight(node) or name not in ("write_target", "remove_target"):
            continue
        if _scope(node, parents) != ("activate", "UnitStrategy"):
            continue
        child, ancestor, settled = node, parents.get(node), False
        while ancestor is not None and not isinstance(ancestor, _FUNCTIONS):
            if isinstance(ancestor, ast.Try) and any(child is stmt for stmt in ancestor.finalbody):
                settled = True
                break
            child, ancestor = ancestor, parents.get(ancestor)
        if not settled:
            out.append(node.lineno)
    return sorted(out)


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


def test_activate_holds_the_write_the_conversion_and_the_rollback_in_its_own_body():
    """Non-vacuous: `UnitStrategy.activate` really does write (RULING W3's one caller): the in-flight write, the
    plain conversion and the rollback are three."""
    count = activate_write_count((SRC / LIFECYCLE).read_text(), LIFECYCLE)
    assert count == 3, (
        f"UnitStrategy.activate holds {count} write_target call(s); the in-flight write, the plain conversion "
        "and the rollback are three")


def test_exactly_one_call_of_activate_passes_in_flight_true():
    assert in_flight_call_count((SRC / LIFECYCLE).read_text(), LIFECYCLE) == 1


def test_every_other_settle_of_activate_sits_in_the_finalbody_of_a_try_of_its_own_body():
    """The finally pin (R-14-41): the plain conversion, the rollback and the `remove_target` fallback all live in
    the `finally`; none is a statement of a success path, a handler or a closure."""
    source = (SRC / LIFECYCLE).read_text()
    assert settled_outside_finally(source, LIFECYCLE) == []
    inside = activate_write_count(source, LIFECYCLE) - in_flight_call_count(source, LIFECYCLE)
    assert inside == 2, "the conversion and the rollback are the finally's two non-in-flight writes"


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


def test_a_call_in_a_closure_nested_inside_activate_is_reported_and_not_counted():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d):
                def rollback():
                    units.write_target(d, "x")
                rollback()
    ''')
    assert len(stray_target_writers(src, LIFECYCLE)) == 1
    assert activate_write_count(src, LIFECYCLE) == 0


def test_a_planted_in_flight_write_in_another_method_or_a_closure_of_activate_is_reported():
    other = textwrap.dedent('''
        class UnitStrategy:
            async def other(self, d):
                units.write_target(d, "x", in_flight=True)
    ''')
    assert len(stray_target_writers(other, LIFECYCLE)) == 1
    closure = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d):
                def go():
                    units.write_target(d, "x", in_flight=True)
                go()
    ''')
    assert len(stray_target_writers(closure, LIFECYCLE)) == 1
    assert in_flight_call_count(closure, LIFECYCLE) == 0


def test_the_in_flight_count_sees_only_a_constant_true_in_activate():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d, flag):
                units.write_target(d, "x", in_flight=True)
                units.write_target(d, "x", in_flight=flag)
                units.write_target(d, "x", in_flight=False)
                units.write_target(d, "x")
    ''')
    assert in_flight_call_count(src, LIFECYCLE) == 1
    assert in_flight_call_count(src, Path("provider") / "other.py") == 0


def test_a_conversion_on_the_success_path_is_reported_by_the_finally_checker():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d, name):
                units.write_target(d, name, in_flight=True)
                try:
                    await wait()
                    units.write_target(d, name)
                finally:
                    pass
    ''')
    assert settled_outside_finally(src, LIFECYCLE) == [7]


def test_a_settle_in_a_handler_or_the_else_is_reported_by_the_finally_checker_and_a_closure_by_the_stray_checker():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d, name):
                try:
                    await wait()
                except Exception:
                    units.remove_target(d)
                else:
                    units.write_target(d, name)
                finally:
                    def late():
                        units.write_target(d, name)
    ''')
    assert settled_outside_finally(src, LIFECYCLE) == [7, 9]          # the closure is not activate's own body
    assert len(stray_target_writers(src, LIFECYCLE)) == 1             # ... and is the stray checker's


def test_calls_in_the_finalbody_including_a_nested_try_are_not_reported_by_the_finally_checker():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d, name):
                units.write_target(d, name, in_flight=True)
                try:
                    await wait()
                finally:
                    try:
                        units.write_target(d, name)
                    except OSError:
                        units.remove_target(d)
    ''')
    assert settled_outside_finally(src, LIFECYCLE) == []
    assert settled_outside_finally(src, Path("provider") / "other.py") == []


def test_the_count_sees_only_activate_of_unit_strategy_in_lifecycle():
    src = textwrap.dedent('''
        class UnitStrategy:
            async def activate(self, d):
                units.write_target(d, "x")
                try:
                    pass
                except BaseException:
                    units.write_target(d, "y")
                    raise
            async def other(self, d):
                units.write_target(d, "z")
        class Other:
            async def activate(self, d):
                units.write_target(d, "w")
    ''')
    assert activate_write_count(src, LIFECYCLE) == 2
    assert activate_write_count(src, Path("provider") / "other.py") == 0
    one = "class UnitStrategy:\n    async def activate(self, d):\n        units.write_target(d, 'x')\n"
    assert activate_write_count(one, LIFECYCLE) == 1  # fewer than two: the live assertion would fail on it
