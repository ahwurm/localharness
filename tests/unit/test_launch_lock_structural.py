"""Every launch under `src/` takes the launch lock (07-06, D-27; `swap` calls too, 13-08, D-21).

`launch_lock` (provider/lifecycle.py) is the one cross-process exclusion around a server launch: two
LocalHarness processes launching the same server would overwrite its one pidfile. A runtime test can
only prove the sites that exist today, so this one is structural: every call of a method named
`activate` anywhere under `src/localharness/` outside `provider/lifecycle.py` (where the strategies
define it) must sit lexically inside an `async with` whose items call `launch_lock` or `_launch_lock`.
`lifecycle.swap` launches through `activate` itself and takes no lock of its own (its caller holds the
lock around the whole swap), so every call of a method named `swap` outside lifecycle.py is held to the
same rule.

The rule the checker applies: the search for the enclosing block walks up from the call and STOPS at
the nearest function / lambda boundary, so a closure is judged by its own body, never by the block it
was defined in (a nested `def` can be called after the outer block has exited). The REPL's relaunch
closure holds its own `async with self._launch_lock():` for that reason.

Mutation-checked: moving any real site out of its block, or deleting its `async with`, makes the live
test fail; the synthetic cases below fail if the checker stops looking at the block or the boundary.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIFECYCLE = SRC / "provider" / "lifecycle.py"

_LOCK_NAMES = {"launch_lock", "_launch_lock"}
_BOUNDARIES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _is_lock_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    return name in _LOCK_NAMES


def _activate_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "activate"]


def _swap_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "swap"]


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _locked(call: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    node: ast.AST = call
    while node in parents:
        child, node = node, parents[node]
        if isinstance(node, _BOUNDARIES):
            return False  # the function / lambda boundary: the enclosing block is not this body's
        # only the BODY of the `async with` is locked, not the call that builds its items
        if isinstance(node, ast.AsyncWith) and child in node.body \
                and any(_is_lock_call(item.context_expr) for item in node.items):
            return True
    return False


def _activate_references(tree: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    """Every way to reach `activate` that is not a direct `<expr>.activate(` call (IN-59): a bound-method
    reference (`act = strategy.activate`, `partial(strategy.activate, ...)`) and `getattr(x, "activate")`.
    Where such a reference is finally called cannot be judged from here, so each one is reported as a
    launch outside the lock, wherever it sits: a launch must be written as a direct call."""
    found: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "activate":
            parent = parents.get(node)
            if not (isinstance(parent, ast.Call) and parent.func is node):
                found.append(node)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == "activate"):
            found.append(node)
    return found


def unlocked_activate_calls(source: str) -> list[int]:
    """Line numbers of every `.activate(` call in `source` that is not inside a launch-lock block, and of
    every indirect reference to `activate` (see `_activate_references`)."""
    tree = ast.parse(source)
    parents = _parents(tree)
    return sorted([call.lineno for call in _activate_calls(tree) if not _locked(call, parents)]
                  + [ref.lineno for ref in _activate_references(tree, parents)])


def unlocked_swap_calls(source: str) -> list[int]:
    """Line numbers of every `.swap(` call in `source` that is not inside a launch-lock block."""
    tree = ast.parse(source)
    parents = _parents(tree)
    return sorted(call.lineno for call in _swap_calls(tree) if not _locked(call, parents))


def _source_files():
    return [p for p in sorted(SRC.rglob("*.py")) if p != LIFECYCLE]


def test_every_activate_under_src_sits_inside_a_launch_lock_block():
    unlocked: list[str] = []
    found: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        found += [f"{path.relative_to(SRC)}:{c.lineno}" for c in _activate_calls(tree)]
        unlocked += [f"{path.relative_to(SRC)}:{line}" for line in unlocked_activate_calls(text)]
    assert unlocked == [], f"a launch without the launch lock (or through an indirect reference): {unlocked}"
    # The three launch sites (a fourth must take the lock AND change this number on purpose):
    #   cli/start_cmd.py  _start_managed_server        start's down-server branch
    #   cli/repl.py       the managed restart          /model swap-back to the managed primary
    #   cli/repl.py       _build_relaunch's closure    the reconnect's relaunch (its own lock block)
    # The REPL's cold branch and its rollback no longer call `activate` themselves: both run inside
    # `lifecycle.swap` / `lifecycle.restore_incumbent` (exempt, the strategies' home), under the lock
    # the REPL holds around the swap, which the `.swap(` test below pins.
    assert len(found) == 3, found
    assert sorted({f.split(":")[0] for f in found}) == ["cli/repl.py", "cli/start_cmd.py"]
    assert sum(f.startswith("cli/start_cmd.py") for f in found) == 1
    assert sum(f.startswith("cli/repl.py") for f in found) == 2


def test_every_swap_under_src_sits_inside_a_launch_lock_block():
    unlocked: list[str] = []
    found: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        found += [f"{path.relative_to(SRC)}:{c.lineno}" for c in _swap_calls(ast.parse(text))]
        unlocked += [f"{path.relative_to(SRC)}:{line}" for line in unlocked_swap_calls(text)]
    assert unlocked == [], f"a swap without the launch lock: {unlocked}"
    # the REPL's cold branch is the one caller today; start's swap (plans 13-09/13-10) joins it
    assert sum(f.startswith("cli/repl.py") for f in found) == 1, found


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("async def f(s):\n    await _lifecycle.swap(a, b)\n", [2]),
        ("async def f(s):\n    async with self._launch_lock():\n        await _lifecycle.swap(a, b)\n", []),
        ("async def f(s):\n    async with launch_lock(d):\n        await s.swap(a, b)\n", []),
        ("async def f(s):\n    async with launch_lock(d):\n        pass\n    await s.swap(a, b)\n", [4]),
        # a closure is judged by its own body, never by the block it was defined in
        ("async def f(s):\n    async with launch_lock(d):\n        async def later():\n"
         "            await s.swap(a, b)\n        return later\n", [4]),
        ("def f(s):\n    s.swapped()\n    swap(1)\n", []),
    ],
)
def test_the_checker_reports_exactly_the_unlocked_swaps(source, expected):
    assert unlocked_swap_calls(textwrap.dedent(source)) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("async def f(s):\n    await s.activate(spec, d, url)\n", [2]),
        ("async def f(s):\n    async with launch_lock(d):\n        await s.activate(spec, d, url)\n", []),
        ("async def f(s):\n    async with self._launch_lock():\n        await s.activate(spec, d, url)\n", []),
        ("async def f(s):\n    async with _lifecycle.launch_lock(d, on_wait=w):\n"
         "        await s.activate(spec, d, url)\n", []),
        ("async def f(s):\n    async with something():\n        await s.activate(spec, d, url)\n", [3]),
        ("async def f(s):\n    async with launch_lock(d):\n        pass\n    await s.activate(spec, d, url)\n", [4]),
        ("async def f(s):\n    async with a(), launch_lock(d):\n        await s.activate(spec, d, url)\n", []),
        ("async def f(s):\n    async with launch_lock(d):\n        if x:\n            for y in z:\n"
         "                await s.activate(spec, d, url)\n", []),
        # a closure is judged by its own body, never by the block it was defined in
        ("async def f(s):\n    async with launch_lock(d):\n        async def later():\n"
         "            await s.activate(spec, d, url)\n        return later\n", [4]),
        ("async def f(s):\n    async with launch_lock(d):\n        g = lambda: s.activate(spec, d, url)\n", [3]),
        ("async def f(s):\n    async def later():\n        async with launch_lock(d):\n"
         "            await s.activate(spec, d, url)\n    return later\n", []),
        # the lock call that builds the items is not inside its own body
        ("async def f(s):\n    async with launch_lock(s.activate(spec, d, url)):\n        pass\n", [2]),
        ("def f(s):\n    s.activate()\n    s.deactivate()\n    activate(1)\n", [2]),
        # IN-59: a launch reached through a bound reference or getattr is reported, even inside a block
        ("async def f(s):\n    act = s.activate\n    await act(spec, d, url)\n", [2]),
        ("async def f(s):\n    async with launch_lock(d):\n        act = s.activate\n"
         "        await act(spec, d, url)\n", [3]),
        ("async def f(s):\n    await getattr(s, \"activate\")(spec, d, url)\n", [2]),
        ("async def f(s):\n    await functools.partial(s.activate, spec)(d, url)\n", [2]),
        ("async def f(s):\n    await getattr(s, \"deactivate\")()\n    x = s.activated\n", []),
    ],
)
def test_the_checker_reports_exactly_the_unlocked_activates(source, expected):
    assert unlocked_activate_calls(textwrap.dedent(source)) == expected


def test_a_real_site_moved_out_of_its_block_is_reported():
    """Mutation on the real start_cmd.py: take the `async with` off the launch and the checker fails."""
    text = (SRC / "cli" / "start_cmd.py").read_text(encoding="utf-8")
    assert unlocked_activate_calls(text) == []
    mutated = text.replace("async with launch_lock(global_dir, on_wait=_say_launch_lock_wait):",
                           "if True:", 1)
    assert mutated != text
    assert len(unlocked_activate_calls(mutated)) == 1


def test_the_real_repl_swap_moved_out_of_its_block_is_reported():
    """Mutation on the real repl.py: take the `async with` off the cold swap and the checker fails."""
    text = (SRC / "cli" / "repl.py").read_text(encoding="utf-8")
    assert unlocked_swap_calls(text) == []
    mutated = text.replace("            async with self._launch_lock():\n                try:\n                    # free the accelerator",
                           "            if True:\n                try:\n                    # free the accelerator", 1)
    assert mutated != text
    assert len(unlocked_swap_calls(mutated)) == 1
