"""One writer of a landing's outcome, and bounded writers of the selection stores (spec 20 §14, D-07,
D-10, D-04, R-13-37; START-03, START-04, START-05).

`persist_landed` (cli/model_ops.py) is the ONE writer of what a landing decided: `last_served`,
`active_endpoint` (launch and swap), the removal of `pending_preset`. An attach is an outcome of its
own (`outcome="attach"`); no separate attach writer exists. Beside it exactly two other functions
write a store, each for a reason that is not an outcome:

- `select_pending`: D-10, the CLI's SELECTION (`active_endpoint` as today plus `pending_preset`);
- `clear_gone`: D-04, the ladder's cleanup of a `pending_preset` / `unit-target` that names no record.

The checker is an AST scan over `src/localharness/cli` (definitions are not calls). Each assertion has
a "bites" test: the same checker run over a source string that plants the forbidden call must report it.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
CLI = SRC / "cli"
MODEL_OPS = "cli/model_ops.py"

OUTCOME_WRITER = "persist_landed"
# The two bounded exceptions beside the outcome writer. `select_pending` is a SELECTION, not an outcome
# (D-10); `clear_gone` removes a name that has no record (D-04). Neither records what a landing did.
SELECTION_WRITERS = {"select_pending": "D-10", "clear_gone": "D-04"}

# store writer -> the functions of cli/model_ops.py allowed to call it
ALLOWED_CALLERS: dict[str, frozenset[str]] = {
    "record_last_served": frozenset({OUTCOME_WRITER}),
    "persist_active_endpoint": frozenset({OUTCOME_WRITER, "select_pending"}),
    "clear_active_endpoint": frozenset({OUTCOME_WRITER, "select_pending"}),
    "write_pending_preset": frozenset({"select_pending"}),
    "clear_pending_preset": frozenset({OUTCOME_WRITER, "clear_gone"}),
}
LANDING_OUTCOMES = {"launch", "swap", "attach"}
# A function that forwards its own `outcome` parameter to persist_landed; each of its callers is held
# to the constant rule below.
OUTCOME_FORWARDERS = {("cli/start_cmd.py", "_land")}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[str]:
    names = []
    while node in parents:
        node = parents[node]
        if isinstance(node, _FUNCTIONS):
            names.append(node.name)
    return names


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n.func) == name]


def _sources(root: Path) -> list[tuple[str, str]]:
    return [(p.relative_to(SRC).as_posix(), p.read_text()) for p in sorted(root.rglob("*.py"))]


def stray_writer_calls(source: str, module: str) -> list[tuple[str, int, str]]:
    """Writer calls outside cli/model_ops.py, or inside it but in a function the writer does not allow."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for writer, allowed in ALLOWED_CALLERS.items():
        for call in _calls(tree, writer):
            inside = set(_enclosing(call, parents))
            if module != MODEL_OPS or not inside & allowed:
                out.append((module, call.lineno, writer))
    return out


def calling_functions(source: str, writer: str) -> set[str]:
    tree = ast.parse(source)
    parents = _parents(tree)
    return {(_enclosing(c, parents) or ["<module>"])[0] for c in _calls(tree, writer)}


def _is_outcome(node: ast.expr | None) -> bool:
    """A landing-outcome string constant, or a conditional of two of them (`"launch" if x else "attach"`)."""
    if isinstance(node, ast.IfExp):
        return _is_outcome(node.body) and _is_outcome(node.orelse)
    return isinstance(node, ast.Constant) and node.value in LANDING_OUTCOMES


def bad_outcomes(source: str, module: str) -> list[tuple[str, int, str]]:
    """`persist_landed(` calls whose `outcome` keyword is absent, or not one of the string constants
    (a forwarder's own parameter is allowed only where OUTCOME_FORWARDERS names it)."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for call in _calls(tree, OUTCOME_WRITER):
        outcome = next((k.value for k in call.keywords if k.arg == "outcome"), None)
        function = (_enclosing(call, parents) or ["<module>"])[0]
        if _is_outcome(outcome):
            continue
        if isinstance(outcome, ast.Name) and (module, function) in OUTCOME_FORWARDERS:
            continue
        out.append((module, call.lineno, function))
    return out


def bad_forwarder_calls(source: str, module: str) -> list[tuple[str, int]]:
    """Calls of an outcome forwarder whose second argument is not one of the landing-outcome constants."""
    tree = ast.parse(source)
    out = []
    for fmodule, fname in OUTCOME_FORWARDERS:
        if fmodule != module:
            continue
        for call in _calls(tree, fname):
            arg = call.args[1] if len(call.args) > 1 else next(
                (k.value for k in call.keywords if k.arg == "outcome"), None)
            if not _is_outcome(arg):
                out.append((module, call.lineno))
    return out


# ------------------------------------------------------------------------------ the live tree


def test_every_store_writer_call_sits_in_its_allowed_function_of_model_ops():
    found: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for module, source in _sources(CLI):
        found += stray_writer_calls(source, module)
        seen |= {w for w in ALLOWED_CALLERS if _calls(ast.parse(source), w)}
    assert found == []
    assert seen == set(ALLOWED_CALLERS)  # not vacuous: every writer is really called somewhere


def test_persist_landed_is_the_one_caller_of_record_last_served():
    callers: set[str] = set()
    for _module, source in _sources(SRC):
        callers |= calling_functions(source, "record_last_served")
    assert callers == {OUTCOME_WRITER}


def test_the_selection_writers_are_the_two_named_functions():
    assert set(SELECTION_WRITERS) == {"select_pending", "clear_gone"}
    named = {fn for allowed in ALLOWED_CALLERS.values() for fn in allowed}
    assert named == {OUTCOME_WRITER, *SELECTION_WRITERS}


def test_no_record_attach_function_exists_under_src():
    for module, source in _sources(SRC):
        defined = {n.name for n in ast.walk(ast.parse(source)) if isinstance(n, _FUNCTIONS)}
        assert "record_attach" not in defined, module


def test_every_persist_landed_caller_passes_a_landing_outcome_constant():
    bad: list[tuple[str, int, str]] = []
    forwarded: list[tuple[str, int]] = []
    calls = 0
    for module, source in _sources(CLI):
        bad += bad_outcomes(source, module)
        forwarded += bad_forwarder_calls(source, module)
        calls += len(_calls(ast.parse(source), OUTCOME_WRITER))
    assert bad == []
    assert forwarded == []
    assert calls >= 3  # start's forwarder, the REPL, and settle_attach


# ------------------------------------------------------------------------------ bites


def test_bite_a_repl_like_function_calling_persist_active_endpoint_is_reported():
    planted = textwrap.dedent("""
        async def repl_like(h, ep):
            await model_ops.persist_active_endpoint(h, ep, "m")
    """)
    assert stray_writer_calls(planted, "cli/repl.py") == [("cli/repl.py", 3, "persist_active_endpoint")]


def test_bite_record_last_served_outside_persist_landed_is_reported():
    planted = textwrap.dedent("""
        def other(p):
            speed_stats.record_last_served(p, "x")
    """)
    assert stray_writer_calls(planted, MODEL_OPS) == [(MODEL_OPS, 3, "record_last_served")]
    assert calling_functions(planted, "record_last_served") == {"other"}


def test_bite_clear_pending_preset_from_a_command_is_reported():
    planted = "def cmd(h):\n    model_ops.clear_pending_preset(h)\n"
    assert stray_writer_calls(planted, "cli/model_cmd.py") == [("cli/model_cmd.py", 2, "clear_pending_preset")]


def test_bite_persist_landed_without_an_outcome_is_reported():
    planted = "async def land(h, r):\n    await persist_landed(h, r)\n"
    assert bad_outcomes(planted, "cli/start_cmd.py") == [("cli/start_cmd.py", 2, "land")]
    variable = "async def land(h, r, o):\n    await persist_landed(h, r, outcome=o)\n"
    assert bad_outcomes(variable, "cli/start_cmd.py") == [("cli/start_cmd.py", 2, "land")]
    unknown = "async def land(h, r):\n    await persist_landed(h, r, outcome='boot')\n"
    assert bad_outcomes(unknown, "cli/start_cmd.py") == [("cli/start_cmd.py", 2, "land")]


def test_bite_a_forwarder_call_with_a_variable_outcome_is_reported():
    planted = "async def f(rec, o):\n    await _land(rec, o, None)\n"
    assert bad_forwarder_calls(planted, "cli/start_cmd.py") == [("cli/start_cmd.py", 2)]


def test_a_conditional_of_two_outcome_constants_is_accepted_but_not_with_a_variable_branch():
    ok = "async def f(rec, x):\n    await _land(rec, 'launch' if x else 'attach', None)\n"
    assert bad_forwarder_calls(ok, "cli/start_cmd.py") == []
    bad = "async def f(rec, x, o):\n    await _land(rec, 'launch' if x else o, None)\n"
    assert bad_forwarder_calls(bad, "cli/start_cmd.py") == [("cli/start_cmd.py", 2)]


def test_the_writers_themselves_are_never_reported():
    own = textwrap.dedent("""
        async def persist_landed(h, r, *, outcome):
            await persist_active_endpoint(h, r.peer, "m")
            clear_active_endpoint(h)
            speed_stats.record_last_served(p, "x")
            clear_pending_preset(h)
        async def select_pending(h, r):
            await persist_active_endpoint(h, r.peer, "m")
            clear_active_endpoint(h)
            write_pending_preset(h, r.name)
        def clear_gone(h, gone):
            clear_pending_preset(h)
    """)
    assert stray_writer_calls(own, MODEL_OPS) == []
