"""No caller can build a strategy or a swap without deciding about the autostart unit (spec 20 §9.4, D-19).

`lifecycle.strategy_for(spec, unit=)` returns a `UnitStrategy` only when it is handed the harness's own unit state.
A call site that forgets the keyword silently launches beside the unit (two servers, the unit's next boot racing a
spawned one), and a runtime test can only prove the sites that exist today. So this one is structural, over the
source text of every module under `src/localharness/`:

- every call of `strategy_for` carries a `unit` keyword (found by name, so `strategy_for(...)`,
  `lifecycle.strategy_for(...)` and `_lifecycle.strategy_for(...)` are all caught); at least 12 are found, so the
  scan cannot pass by finding nothing;
- a constant `unit=None` appears only inside `record_identity` (provider/lifecycle.py: a record's identity does
  not depend on its owner) and `foreign_incumbent_line` (cli/start_serving.py: a read of who owns a port);
- inside `free_accelerator`, `safe_stop`, `restore_incumbent` and `swap` the keyword's value is the Name `unit`,
  the parameter each of them takes, never a constant and never another expression;
- every `.swap(` call under `src/localharness/cli/` passes a `unit` keyword.

A live failure means a site was missed: thread it, do not widen the checker. The bite cases plant the mistakes in
source text and require the checkers to report them.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIFECYCLE = SRC / "provider" / "lifecycle.py"
CLI = SRC / "cli"

MINIMUM_SITES = 12
NONE_ALLOWED = {("provider/lifecycle.py", "record_identity"), ("cli/start_serving.py", "foreign_incumbent_line")}
THREADED = ("free_accelerator", "safe_stop", "restore_incumbent", "swap")


def _modules() -> list[tuple[str, str]]:
    """Every module under `src/localharness/` as `(path relative to it, source)`."""
    return [(path.relative_to(SRC).as_posix(), path.read_text(encoding="utf-8"))
            for path in sorted(SRC.rglob("*.py"))]


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _call_name(n) == name]


def _unit_keyword(call: ast.Call) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == "unit"), None)


def _enclosing_functions(tree: ast.AST) -> dict[ast.AST, str]:
    """Each node mapped to the name of the function that lexically holds it (the nearest one)."""
    owner: dict[ast.AST, str] = {}

    def visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            owner[child] = name
            visit(child, name)

    owner[tree] = "<module>"
    visit(tree, "<module>")
    return owner


def _is_none(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def strategy_sites(source: str) -> list[tuple[str, ast.Call]]:
    """Every `strategy_for` call in `source` with the name of the function that holds it (one parse)."""
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    return [(owner[call], call) for call in _calls(tree, "strategy_for")]


def calls_without_unit(source: str, module: str) -> list[str]:
    """`strategy_for(` calls in `source` that carry no `unit` keyword, as `module:line`."""
    return [f"{module}:{call.lineno}" for call in _calls(ast.parse(source), "strategy_for")
            if _unit_keyword(call) is None]


def none_unit_outside_allowed(source: str, module: str) -> list[str]:
    """`strategy_for(..., unit=None)` calls outside the two functions allowed to ask for no unit."""
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    found = []
    for call in _calls(tree, "strategy_for"):
        kw = _unit_keyword(call)
        if kw is not None and _is_none(kw.value) and (module, owner[call]) not in NONE_ALLOWED:
            found.append(f"{module}:{call.lineno}")
    return found


def lifecycle_unit_threading(source: str) -> list[str]:
    """`strategy_for` calls inside the four lifecycle functions whose `unit` is not the Name `unit`."""
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    found = []
    for call in _calls(tree, "strategy_for"):
        if owner[call] not in THREADED:
            continue
        kw = _unit_keyword(call)
        if kw is None or not (isinstance(kw.value, ast.Name) and kw.value.id == "unit"):
            found.append(f"{owner[call]}:{call.lineno}")
    return found


def swap_without_unit_in_lifecycle(source: str) -> list[str]:
    """`strategy_for` calls inside `swap` handed anything but the Name `unit` (`unit=None` is the plant)."""
    return [site for site in lifecycle_unit_threading(source) if site.startswith("swap:")]


def cli_swaps_without_unit(source: str) -> list[str]:
    """`.swap(` calls (a method / module attribute named `swap`) with no `unit` keyword."""
    tree = ast.parse(source)
    return [f"line {call.lineno}" for call in _calls(tree, "swap")
            if isinstance(call.func, ast.Attribute) and _unit_keyword(call) is None]


# ------------------------------------------------------------------------------------------------ live


def test_every_strategy_for_call_under_src_carries_a_unit_keyword():
    bare = [site for module, source in _modules() for site in calls_without_unit(source, module)]

    assert bare == [], f"strategy_for( without unit= — thread the sensed unit state through: {bare}"


def test_the_scan_finds_the_strategy_for_calls_it_is_meant_to_police():
    total = sum(len(_calls(ast.parse(source), "strategy_for")) for _module, source in _modules())
    per_module = {module: len(_calls(ast.parse(source), "strategy_for")) for module, source in _modules()}

    assert total >= MINIMUM_SITES, per_module
    assert per_module["cli/repl.py"] >= 7 and per_module["cli/start_cmd.py"] >= 3
    assert per_module["cli/start_serving.py"] >= 1 and per_module["provider/lifecycle.py"] >= 5


def test_a_constant_none_unit_appears_only_where_the_identity_does_not_depend_on_the_owner():
    constant = [site for module, source in _modules() for site in none_unit_outside_allowed(source, module)]

    assert constant == [], f"unit=None outside record_identity / foreign_incumbent_line: {constant}"
    # and the two allowed places really are the only ones that ask for none
    asked = {(module, name) for module, source in _modules() for name, call in strategy_sites(source)
             if (kw := _unit_keyword(call)) is not None and _is_none(kw.value)}
    assert asked <= NONE_ALLOWED and asked, asked


def test_lifecycle_threads_its_own_unit_parameter_into_every_strategy_it_builds_for_a_launch_or_a_stop():
    source = LIFECYCLE.read_text(encoding="utf-8")

    assert lifecycle_unit_threading(source) == []
    names = {name for name, _call in strategy_sites(source)}
    assert set(THREADED) <= names, names                         # all four are really scanned


def test_every_cli_swap_call_passes_a_unit_keyword():
    swaps = {path.relative_to(SRC).as_posix(): cli_swaps_without_unit(path.read_text(encoding="utf-8"))
             for path in sorted(CLI.rglob("*.py"))}
    seen = sum(len([c for c in _calls(ast.parse(path.read_text(encoding="utf-8")), "swap")
                    if isinstance(c.func, ast.Attribute)]) for path in CLI.rglob("*.py"))

    assert {module: sites for module, sites in swaps.items() if sites} == {}
    assert seen >= 2                                             # the REPL's cold swap and start_serving.run_swap


# ------------------------------------------------------------------------------------------- the bites


def test_a_planted_strategy_for_without_the_keyword_is_reported():
    planted = textwrap.dedent('''
        def launch(spec):
            return strategy_for(spec)
    ''')

    assert calls_without_unit(planted, "cli/planted.py") == ["cli/planted.py:3"]
    assert calls_without_unit("x = lifecycle.strategy_for(spec)\n", "m.py") == ["m.py:1"]
    assert calls_without_unit("x = _lifecycle.strategy_for(spec, unit=unit)\n", "m.py") == []


def test_a_planted_unit_none_in_start_cmd_is_reported_and_the_two_allowed_places_are_not():
    planted = "def go(spec):\n    return strategy_for(spec, unit=None)\n"

    assert none_unit_outside_allowed(planted, "cli/start_cmd.py") == ["cli/start_cmd.py:2"]
    assert none_unit_outside_allowed(planted.replace("def go", "def record_identity"),
                                     "provider/lifecycle.py") == []
    assert none_unit_outside_allowed(planted.replace("def go", "def foreign_incumbent_line"),
                                     "cli/start_serving.py") == []
    # the allowance is per function and per module: the same function name elsewhere is not allowed
    assert none_unit_outside_allowed(planted.replace("def go", "def record_identity"), "cli/repl.py") != []


def test_a_planted_cli_swap_without_the_keyword_is_reported():
    planted = textwrap.dedent('''
        async def go(a, b, d):
            return await lifecycle.swap(a, b, config_dir=d)
    ''')

    assert cli_swaps_without_unit(planted) == ["line 3"]
    assert cli_swaps_without_unit(planted.replace("config_dir=d", "config_dir=d, unit=unit")) == []


def test_a_planted_none_unit_inside_the_lifecycle_swap_is_reported():
    planted = textwrap.dedent('''
        async def swap(incumbent, target, *, unit=None):
            return strategy_for(target.spec, unit=None)

        async def safe_stop(spec, config_dir, *, unit=None):
            await strategy_for(spec, unit=unit).stop(spec, config_dir)
    ''')

    assert swap_without_unit_in_lifecycle(planted) == ["swap:3"]
    assert lifecycle_unit_threading(planted) == ["swap:3"]
    assert lifecycle_unit_threading(planted.replace("unit=None)\n\nasync def safe_stop", "unit=unit)\n\nasync def safe_stop")) == []


def test_a_threaded_value_that_is_not_the_parameter_is_reported():
    planted = textwrap.dedent('''
        async def free_accelerator(incumbent, target, config_dir, *, unit=None):
            await strategy_for(incumbent[0], unit=other).stop(incumbent[0], config_dir)

        async def restore_incumbent(stopped, *, config_dir, autosense, unit=None):
            return await strategy_for(stopped.spec).activate(stopped.spec, config_dir, stopped.base_url)
    ''')

    assert lifecycle_unit_threading(planted) == ["free_accelerator:3", "restore_incumbent:6"]
