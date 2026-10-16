"""Structural pins on the CLI's trust decision and the swap path (spec 20 §8.1, §12, D-11; SEC-02).

A runtime test proves the callers that exist today; these scan the source under `src/localharness/cli`
so the guarantee holds for the one written tomorrow:

- no `decide_machine_trust(` call passes `ask` bound to the constant `True` (an unasked-for question
  to a process with no local operator would be answered by whoever feeds its stdin). The allowlist
  of `(module, function)` pairs that may do so holds one entry, `model_cmd._autostart_on`, which is
  reached only through `_autostart`, after its local-operator gate (a tty and no LOCALHARNESS_COMMIT);
- the swap cannot run without a `MachineTrust` in hand: `resolve_serving` takes `trust` with no
  default, `heavy_for` takes a `Serving`, a `Serving` is built only on the ladder in model_ops, a
  `Heavy` only by `heavy_for` / `incumbent_heavy`, and every function that calls `.swap(` gets its
  target from `heavy_for(`.

Each assertion has a "bites" test: the same checker run over a small source string that plants the
forbidden call must report it, so a checker that stopped looking would fail here.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from localharness.cli import model_ops

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
CLI = SRC / "cli"
MODEL_OPS = "cli/model_ops.py"

# (module relative to src/localharness, enclosing function) pairs allowed to pass the constant True as
# `ask` to decide_machine_trust. One entry: `--autostart on` asks the machine-trust question, and only after
# `_autostart` has checked that a person is at the terminal (see `gate_before_ask`).
ASK_TRUE_ALLOWLIST: frozenset[tuple[str, str]] = frozenset({("cli/model_cmd.py", "_autostart_on")})
MODEL_CMD = "cli/model_cmd.py"
ASKING_FUNCTION = "_autostart_on"
GATE_FUNCTION = "_autostart"
GATE_CALL = "_is_local_operator"

# Where a `Heavy` may be built (cli/model_ops.py only) and where a `Serving` may be built: the ladder
# and the one helper that hands back its last rung.
HEAVY_BUILDERS = frozenset({"heavy_for", "incumbent_heavy"})
SERVING_BUILDERS = frozenset({"resolve_serving", "default_serving"})

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


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _name(n.func) == name]


def _cli_sources() -> list[tuple[str, str]]:
    return [(p.relative_to(SRC).as_posix(), p.read_text()) for p in sorted(CLI.rglob("*.py"))]


def _is_true(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


# ------------------------------------------------------------------------------ checkers


def ask_true_calls(source: str, module: str) -> list[tuple[str, int, str]]:
    """`decide_machine_trust(..., ask=True)` calls whose `(module, function)` is not allowlisted."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for call in _calls_named(tree, "decide_machine_trust"):
        function = _enclosing_function(call, parents)
        asked = [k for k in call.keywords if k.arg == "ask"]
        if any(_is_true(k.value) for k in asked) and (module, function) not in ASK_TRUE_ALLOWLIST:
            out.append((module, call.lineno, function))
    return out


def gate_before_ask(source: str, asking: str = ASKING_FUNCTION, gate_function: str = GATE_FUNCTION,
                    gate_call: str = GATE_CALL) -> list[str]:
    """What is wrong with how `asking` (the function that holds the allowlisted `ask=True`) is reached:
    it must be called only from `gate_function`, and there only after a `gate_call(` that comes earlier in
    the function's own text (a gate lexically later, or in a nested function, is no gate)."""
    tree = ast.parse(source)
    parents = _parents(tree)
    problems = []
    callers = {_enclosing_function(call, parents) for call in _calls_named(tree, asking)}
    if callers != {gate_function}:
        problems.append(f"{asking} is called from {sorted(callers) or 'nowhere'}, not only {gate_function}")
    for node in ast.walk(tree):
        if not (isinstance(node, _FUNCTIONS) and node.name == gate_function):
            continue
        own = [n for stmt in node.body for n in _walk_own(stmt) if isinstance(n, ast.Call)]
        gates = [n.lineno for n in own if _name(n.func) == gate_call]
        asks = [n.lineno for n in own if _name(n.func) == asking]
        if not gates:
            problems.append(f"{gate_function} never calls {gate_call}")
        elif asks and min(asks) < min(gates):
            problems.append(f"{gate_function} calls {asking} (line {min(asks)}) before {gate_call} (line {min(gates)})")
    if not any(isinstance(n, _FUNCTIONS) and n.name == gate_function for n in ast.walk(tree)):
        problems.append(f"no {gate_function} function")
    return problems


def stray_builders(source: str, module: str, constructor: str, allowed: frozenset[str]) -> list[tuple[str, int, str]]:
    """Calls of `constructor` outside cli/model_ops.py, or inside it but outside an `allowed` function."""
    tree = ast.parse(source)
    parents = _parents(tree)
    out = []
    for call in _calls_named(tree, constructor):
        function = _enclosing_function(call, parents)
        if module != MODEL_OPS or function not in allowed:
            out.append((module, call.lineno, function))
    return out


def swaps_without_heavy_for(source: str) -> tuple[list[str], list[str]]:
    """(functions that call `.swap(` and `heavy_for(`, functions that call `.swap(` without `heavy_for(`)."""
    good, bad = [], []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, _FUNCTIONS):
            continue
        # only this function's own body: a nested function is judged by its own body
        own = [n for stmt in node.body for n in _walk_own(stmt)]
        swaps = [n for n in own if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "swap"]
        if not swaps:
            continue
        has_heavy_for = any(isinstance(n, ast.Call) and _name(n.func) == "heavy_for" for n in own)
        (good if has_heavy_for else bad).append(node.name)
    return good, bad


def _walk_own(node: ast.AST):
    """`ast.walk` that does not descend into a nested function or lambda."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (*_FUNCTIONS, ast.Lambda)):
            continue
        yield from _walk_own(child)


# ------------------------------------------------------------------------------ the live tree


def test_no_decide_machine_trust_call_is_bound_to_ask_true():
    found, calls = [], 0
    for module, source in _cli_sources():
        found += ask_true_calls(source, module)
        calls += len(_calls_named(ast.parse(source), "decide_machine_trust"))
    assert found == []
    assert calls >= 2  # not vacuous: start and model_cmd both decide
    assert ASK_TRUE_ALLOWLIST == frozenset({(MODEL_CMD, ASKING_FUNCTION)})


def test_the_one_allowlisted_ask_exists_and_is_the_only_ask_literal_under_cli():
    asking = []
    for module, source in _cli_sources():
        tree = ast.parse(source)
        parents = _parents(tree)
        for call in _calls_named(tree, "decide_machine_trust"):
            if any(k.arg == "ask" and _is_true(k.value) for k in call.keywords):
                asking.append((module, _enclosing_function(call, parents)))
    assert asking == [(MODEL_CMD, ASKING_FUNCTION)]


def test_autostart_on_is_reached_only_through_the_local_operator_gate():
    source = (SRC / MODEL_CMD).read_text()
    assert gate_before_ask(source) == []


def test_resolve_serving_requires_a_machine_trust_with_no_default():
    trust = inspect.signature(model_ops.resolve_serving).parameters["trust"]
    assert trust.default is inspect.Parameter.empty
    assert "MachineTrust" in str(trust.annotation)


def test_heavy_for_takes_a_serving():
    first = next(iter(inspect.signature(model_ops.heavy_for).parameters.values()))
    assert "Serving" in str(first.annotation)


def test_a_heavy_is_built_only_by_heavy_for_and_incumbent_heavy():
    found, built = [], 0
    for module, source in _cli_sources():
        found += stray_builders(source, module, "Heavy", HEAVY_BUILDERS)
        built += len(_calls_named(ast.parse(source), "Heavy"))
    assert found == []
    assert built >= 2  # both builders exist


def test_a_serving_is_built_only_on_the_ladder_in_model_ops():
    found, built = [], 0
    for module, source in _cli_sources():
        found += stray_builders(source, module, "Serving", SERVING_BUILDERS)
        built += len(_calls_named(ast.parse(source), "Serving"))
    assert found == []
    assert built >= 2  # the ladder's rungs


def test_every_function_that_swaps_takes_its_target_from_heavy_for():
    good, bad = [], []
    for _module, source in _cli_sources():
        g, b = swaps_without_heavy_for(source)
        good += g
        bad += b
    assert bad == []
    assert len(good) >= 2  # the REPL's /model handler and start_serving.run_swap
    assert "run_swap" in good


# ------------------------------------------------------------------------------ bites


def test_bite_ask_true_is_reported_and_a_variable_is_not():
    planted = textwrap.dedent("""
        def launches(d):
            return decide_machine_trust(d, ask=True)
        def asks_when_it_can(d, can_ask):
            return decide_machine_trust(d, ask=can_ask)
    """)
    assert ask_true_calls(planted, "cli/x.py") == [("cli/x.py", 3, "launches")]


def test_bite_the_ask_before_its_gate_is_reported():
    ungated = textwrap.dedent("""
        def _autostart(mode, config_dir):
            _autostart_on(config_dir)
            if not _is_local_operator():
                raise SystemExit(2)
        def _autostart_on(cfg):
            return decide_machine_trust(cfg, ask=True)
    """)
    assert gate_before_ask(ungated) == ["_autostart calls _autostart_on (line 3) before _is_local_operator (line 4)"]
    gated = textwrap.dedent("""
        def _autostart(mode, config_dir):
            if not _is_local_operator():
                raise SystemExit(2)
            _autostart_on(config_dir)
        def _autostart_on(cfg):
            return decide_machine_trust(cfg, ask=True)
    """)
    assert gate_before_ask(gated) == []


def test_bite_a_second_caller_of_the_ask_and_a_missing_gate_are_reported():
    sneaky = textwrap.dedent("""
        def _autostart(mode, config_dir):
            if not _is_local_operator():
                raise SystemExit(2)
            _autostart_on(config_dir)
        def elsewhere(config_dir):
            _autostart_on(config_dir)
        def _autostart_on(cfg):
            return decide_machine_trust(cfg, ask=True)
    """)
    assert gate_before_ask(sneaky) == ["_autostart_on is called from ['_autostart', 'elsewhere'], not only _autostart"]
    ungated = "def _autostart(m, c):\n    _autostart_on(c)\n"
    assert gate_before_ask(ungated) == ["_autostart never calls _is_local_operator"]
    assert gate_before_ask("def other():\n    pass\n") == [
        "_autostart_on is called from nowhere, not only _autostart", "no _autostart function"]


def test_bite_an_ask_in_a_function_that_is_not_the_allowlisted_one_is_reported():
    planted = "def _autostart(d):\n    return decide_machine_trust(d, ask=True)\n"
    assert ask_true_calls(planted, MODEL_CMD) == [(MODEL_CMD, 2, "_autostart")]
    assert ask_true_calls(planted.replace("_autostart", "_autostart_on"), "cli/start_cmd.py") == [
        ("cli/start_cmd.py", 2, "_autostart_on")]


def test_bite_a_heavy_outside_the_builders_is_reported():
    planted = textwrap.dedent("""
        def sneaky(spec, url):
            return lifecycle.Heavy(spec, url, None, {})
    """)
    assert stray_builders(planted, "cli/repl.py", "Heavy", HEAVY_BUILDERS) == [("cli/repl.py", 3, "sneaky")]
    assert stray_builders(planted, MODEL_OPS, "Heavy", HEAVY_BUILDERS) == [(MODEL_OPS, 3, "sneaky")]
    ok = "def heavy_for(s, h):\n    return Heavy(1, 2, 3, 4)\n"
    assert stray_builders(ok, MODEL_OPS, "Heavy", HEAVY_BUILDERS) == []


def test_bite_a_serving_outside_the_ladder_is_reported():
    planted = "def pick(r):\n    return model_ops.Serving(r, 'x', False)\n"
    assert stray_builders(planted, "cli/start_cmd.py", "Serving", SERVING_BUILDERS) == [("cli/start_cmd.py", 2, "pick")]


def test_bite_a_swap_without_heavy_for_is_reported():
    planted = textwrap.dedent("""
        async def untrusted(a, b):
            return await lifecycle.swap(a, b)
        async def trusted(h, s):
            t = model_ops.heavy_for(s, h)
            return await lifecycle.swap(None, t)
    """)
    assert swaps_without_heavy_for(planted) == (["trusted"], ["untrusted"])
