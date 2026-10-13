"""No `stop_server` under `src/` is called, or handed to a thread, without `expect` (D-27, START-12).

`server.stop_server(config_dir, launch, timeout, *, expect)` signals nothing unless the pidfile's live
process matches `expect`; a runtime test only proves the sites that exist today. This one is structural:
over every `.py` under `src/localharness/`,

- a `Call` whose function is the name or attribute `stop_server` must carry an `expect` keyword;
- a `Call` that passes a `stop_server` reference as an argument (the `asyncio.to_thread(server.stop_server,
  ...)` form) must carry an `expect` keyword in that same call;
- any other reference (a bound name, `getattr(x, "stop_server")`, an `import stop_server`) is reported:
  where it is finally called cannot be judged from here, so a stop is written as a direct call.

The definition (`def stop_server`) is a `FunctionDef`, not a reference, and is not reported.
The synthetic cases below fail if the checker stops looking at the keyword or at the references.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
NAME = "stop_server"


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _is_ref(node: ast.AST) -> bool:
    return (isinstance(node, ast.Name) and node.id == NAME) or (
        isinstance(node, ast.Attribute) and node.attr == NAME)


def _has_expect(call: ast.Call) -> bool:
    return any(kw.arg == "expect" for kw in call.keywords)


def stop_sites(source: str) -> tuple[int, list[int]]:
    """`(sites, violations)`: how many calls and thread hand-offs of `stop_server` the source holds, and
    the line numbers of each one lacking `expect` plus every reference that is neither."""
    tree = ast.parse(source)
    parents = _parents(tree)
    sites = 0
    bad: list[int] = []
    for node in ast.walk(tree):
        if _is_ref(node):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and (parent.func is node or node in parent.args):
                sites += 1
                if not _has_expect(parent):
                    bad.append(node.lineno)
            else:
                bad.append(node.lineno)  # a bound name, a returned or stored reference
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and node.args[1].value == NAME):
            bad.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and any(alias.name == NAME for alias in node.names):
            bad.append(node.lineno)
    return sites, sorted(bad)


def test_every_stop_server_under_src_carries_expect():
    sites = 0
    bad: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        found, violations = stop_sites(path.read_text(encoding="utf-8"))
        sites += found
        bad += [f"{path.relative_to(SRC)}:{line}" for line in violations]
    assert bad == [], f"a stop_server without `expect` (or reached through an indirect reference): {bad}"
    # Not vacuous: the strategies' stops (ManagedVllm x3, Spawned x3, Daemon x1) are all thread hand-offs.
    assert sites >= 7, sites


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("def f(d):\n    server.stop_server(d)\n", [2]),
        ("def f(d):\n    stop_server(d, launch='binary')\n", [2]),
        ("async def f(d):\n    await asyncio.to_thread(server.stop_server, d, launch=\"binary\")\n", [2]),
        ("def f():\n    f = server.stop_server\n", [2]),
        ("def f():\n    return getattr(server, \"stop_server\")\n", [2]),
        ("from localharness.provider.server import stop_server\n", [1]),
        # clean: an `expect` keyword on the call, or on the call that takes the reference
        ("def f(d, x):\n    server.stop_server(d, expect=x)\n", []),
        ("async def f(d, x):\n    await asyncio.to_thread(server.stop_server, d, expect=x)\n", []),
        ("async def f(d, x):\n    await asyncio.to_thread(\n        server.stop_server, d, launch='b', expect=x)\n", []),
        # an unrelated name and the definition itself are not stops
        ("def f(d):\n    server.stop_servers(d)\n    stop(d)\n", []),
        ("def stop_server(d, *, expect):\n    pass\n", []),
    ],
)
def test_the_checker_reports_exactly_the_stops_without_expect(source, expected):
    assert stop_sites(source)[1] == expected


def test_the_checker_counts_the_sites_it_judges():
    assert stop_sites("def f(d, x):\n    server.stop_server(d, expect=x)\n    stop_server(d, expect=x)\n")[0] == 2
    assert stop_sites("def f(d, x):\n    asyncio.to_thread(server.stop_server, d, expect=x)\n")[0] == 1
