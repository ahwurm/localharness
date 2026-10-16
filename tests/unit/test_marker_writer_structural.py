"""Structural pins on `serve` (spec 20 §9.3, D-14, D-16, R-14-44; SEC-02 share).

A runtime test proves the paths `serve` takes today; these scan the source under `src/localharness` so the
guarantees hold for the change written tomorrow:

- `units.write_marker(` is called under `src/` only from `serve` in `cli/serve_cmd.py` (the boot marker has
  one author: the process about to exec the server it describes);
- every `decide_machine_trust(` call in `cli/serve_cmd.py` passes `ask` as the constant `False` (a unit has no
  local operator, so nothing it does may ask or adopt);
- `cli/serve_cmd.py` calls no `.activate(` and no `.swap(` (it never launches through a strategy);
- `cli/serve_cmd.py` lands on whatever its one `resolve_serving(` call returns: it calls no `read_target(` and
  no `in_flight_target(`, and reads no `pending_preset` (no attribute, no subscript, no `.get(` / `getattr(`,
  no string constant of that name outside a docstring), so a stored selection can reach it only through the
  ladder (plan 14-13 adds its rungs there).

Each checker has a "bites" test: the same function run over a small source string that plants the forbidden
form must report it, so a checker that stopped looking would fail here.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
SERVE = "cli/serve_cmd.py"

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    while node in parents:
        node = parents[node]
        if isinstance(node, _FUNCTIONS):
            return node.name
    return "<module>"


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n.func) == name]


def _sources() -> list[tuple[str, str]]:
    return [(p.relative_to(SRC).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(SRC.rglob("*.py"))]


# ------------------------------------------------------------------------------ checkers


def marker_writers(source: str, module: str) -> list[tuple[str, int, str]]:
    """`write_marker(` calls outside `serve` of cli/serve_cmd.py. (The definition is a `def`, not a call.)"""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for call in _calls(tree, "write_marker"):
        function = _enclosing_function(call, parents)
        if (module, function) != (SERVE, "serve"):
            out.append((module, call.lineno, function))
    return out


def asking_decisions(source: str) -> tuple[int, list[int]]:
    """``(decide_machine_trust calls, line numbers of those whose `ask` is not the constant False)``."""
    calls = _calls(ast.parse(source), "decide_machine_trust")
    bad = []
    for call in calls:
        asked = [k.value for k in call.keywords if k.arg == "ask"]
        if not (len(asked) == 1 and isinstance(asked[0], ast.Constant) and asked[0].value is False):
            bad.append(call.lineno)
    return len(calls), bad


def strategy_launches(source: str) -> list[tuple[str, int]]:
    """`.activate(` / `.swap(` method calls."""
    return [(n.func.attr, n.lineno) for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("activate", "swap")]


def store_reads(source: str) -> list[tuple[str, int]]:
    """Reads of a stored selection: calls of `read_target` / `in_flight_target`, and any `pending_preset`
    attribute, subscript key, `.get(` / `getattr(` argument or string constant (docstrings excepted)."""
    tree = ast.parse(source)
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, *_FUNCTIONS)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _name(node.func) in ("read_target", "in_flight_target"):
            found.append((_name(node.func) + "(", node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr == "pending_preset":
            found.append(("pending_preset attribute", node.lineno))
        elif (isinstance(node, ast.Constant) and node.value == "pending_preset"
              and id(node) not in docstrings):
            found.append(("pending_preset key", node.lineno))
    return found


# ------------------------------------------------------------------------------ the live tree


def _serve_source() -> str:
    return (SRC / SERVE).read_text(encoding="utf-8")


def test_write_marker_is_called_only_from_serve():
    found, calls = [], 0
    for module, source in _sources():
        found += marker_writers(source, module)
        calls += len(_calls(ast.parse(source), "write_marker"))
    assert found == []
    assert calls == 1  # not vacuous: serve is the one caller


def test_serves_trust_decision_never_asks():
    count, bad = asking_decisions(_serve_source())
    assert count == 1  # not vacuous: serve decides exactly once
    assert bad == []


def test_serve_never_launches_through_a_strategy():
    assert strategy_launches(_serve_source()) == []


def test_serve_reads_no_selection_store_itself():
    assert store_reads(_serve_source()) == []


def test_serve_calls_the_ladder_exactly_once():
    assert len(_calls(ast.parse(_serve_source()), "resolve_serving")) == 1


def test_serves_launch_goes_through_the_shared_guard_and_the_exec_seam():
    tree = ast.parse(_serve_source())
    for name in ("refuse_live_launch", "write_pidfile", "exec_server", "launch_lock_sync"):
        assert len(_calls(tree, name)) == 1, name


# ------------------------------------------------------------------------------ bites


def test_bite_a_marker_write_outside_serve_is_reported():
    planted = textwrap.dedent("""
        def helper(cfg, marker):
            units.write_marker(cfg, marker)
    """)
    assert marker_writers(planted, "provider/lifecycle.py") == [("provider/lifecycle.py", 3, "helper")]
    assert marker_writers(planted, SERVE) == [(SERVE, 3, "helper")]
    inside = "def serve(cfg, marker):\n    units.write_marker(cfg, marker)\n"
    assert marker_writers(inside, SERVE) == []
    assert marker_writers(inside, "cli/other.py") == [("cli/other.py", 2, "serve")]


def test_bite_a_trust_decision_that_may_ask_is_reported():
    for call in ("decide_machine_trust(cfg, ask=True)", "decide_machine_trust(cfg)",
                 "decide_machine_trust(cfg, ask=can_ask)", "decide_machine_trust(cfg, ask=not True)"):
        assert asking_decisions(f"x = {call}\n") == (1, [1]), call
    assert asking_decisions("x = decide_machine_trust(cfg, ask=False)\n") == (1, [])


def test_bite_a_strategy_launch_is_reported():
    assert strategy_launches("async def f(s):\n    await s.activate(spec)\n") == [("activate", 2)]
    assert strategy_launches("def f(s):\n    s.swap(a, b)\n") == [("swap", 2)]
    assert strategy_launches("def f(s):\n    s.activated()\n") == []


def test_bite_a_store_read_is_reported_in_every_form():
    for planted, what in (
        ("x = read_target(cfg)\n", "read_target("),
        ("x = units.read_target(cfg)\n", "read_target("),
        ("x = in_flight_target(cfg)\n", "in_flight_target("),
        ("x = harness.pending_preset\n", "pending_preset attribute"),
        ("x = getattr(harness, 'pending_preset')\n", "pending_preset key"),
        ("x = overlay['pending_preset']\n", "pending_preset key"),
        ("x = overlay.get('pending_preset')\n", "pending_preset key"),
        ("KEY = 'pending_preset'\nx = overlay[KEY]\n", "pending_preset key"),
    ):
        assert [w for w, _ in store_reads(planted)] == [what], planted


def test_bite_a_docstring_may_name_the_store_but_code_may_not():
    named = '"""A module that never reads pending_preset."""\n\ndef f():\n    """Nor read_target( here."""\n'
    assert store_reads(named) == []
    assert store_reads(named + "\nx = pending_preset\ny = {'pending_preset': 1}\n") == [
        ("pending_preset key", 7)]
