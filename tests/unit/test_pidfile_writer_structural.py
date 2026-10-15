"""Every function that writes the server pidfile calls `refuse_live_launch` first (IN-08, T-14-04).

`server.refuse_live_launch(config_dir, cmd)` is the one rule "a pidfile that names a live process nobody
verified is refused, not replaced". `start_server` calls it and so must `serve`, which writes its own pidfile
and does not call `start_server`; a runtime test only proves the writers that exist today. This one is
structural: over every `.py` under `src/localharness/`, a function whose OWN body

- calls `write_pidfile(` (the name or an attribute), or
- calls `atomic_write_text(` with `PIDFILE_NAME` / `procfs.PIDFILE_NAME` / the literal `"server.pid"` as its
  first argument,

must also call `refuse_live_launch(` at an earlier position in that same body. A nested function or lambda
is judged by its own body, not its parent's: a guard in the parent does not cover a closure that writes. The
definition `write_pidfile` in `provider/server.py` is the one exempt function (it is the writer, not a
launch). The synthetic cases below fail if the checker stops looking at the order, the scope or the
direct-write form.
"""
from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterator
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
GUARD = "refuse_live_launch"
WRITER = "write_pidfile"
EXEMPT = ("provider/server.py", "write_pidfile")
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _callee(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _names_pidfile(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "PIDFILE_NAME"
    if isinstance(node, ast.Attribute):
        return node.attr == "PIDFILE_NAME"
    return isinstance(node, ast.Constant) and node.value == "server.pid"


def _is_writer(call: ast.Call) -> bool:
    callee = _callee(call)
    if callee == WRITER:
        return True
    if callee != "atomic_write_text":
        return False
    if call.args:
        return _names_pidfile(call.args[0])
    return any(kw.arg == "path" and _names_pidfile(kw.value) for kw in call.keywords)


def _own_calls(scope: ast.AST) -> Iterator[ast.Call]:
    """The calls in `scope`'s own body: nested functions and lambdas are other scopes and are not entered."""
    body = scope.body if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)) else [scope.body]
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def _scopes(tree: ast.Module) -> Iterator[tuple[str, ast.AST]]:
    yield "<module>", tree
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.Lambda):
            yield "<lambda>", node


def _position(node: ast.AST) -> tuple[int, int]:
    return node.lineno, node.col_offset


def writer_calls(source: str, module: str = "x.py") -> int:
    """How many pidfile-writing calls the source holds (the exempt definition's own call excluded)."""
    tree = ast.parse(source)
    count = 0
    for name, scope in _scopes(tree):
        if module.endswith(EXEMPT[0]) and name == EXEMPT[1]:
            continue
        count += sum(1 for call in _own_calls(scope) if _is_writer(call))
    return count


def unguarded_pidfile_writers(source: str, module: str) -> list[tuple[str, int, str]]:
    """`(function, line of its first pidfile write, why)` for each function that writes the pidfile
    without `refuse_live_launch(` earlier in its own body."""
    tree = ast.parse(source)
    found: list[tuple[str, int, str]] = []
    for name, scope in _scopes(tree):
        if module.endswith(EXEMPT[0]) and name == EXEMPT[1]:
            continue
        calls = list(_own_calls(scope))
        writes = sorted((c for c in calls if _is_writer(c)), key=_position)
        if not writes:
            continue
        first = writes[0]
        guards = [c for c in calls if _callee(c) == GUARD]
        if not guards:
            found.append((name, first.lineno, f"writes the pidfile and never calls {GUARD}"))
        elif min(_position(g) for g in guards) > _position(first):
            found.append((name, first.lineno, f"calls {GUARD} only after its first pidfile write"))
    return found


def _src(text: str) -> str:
    return textwrap.dedent(text)


# ------------------------------------------------------------------------------- the live tree


def test_every_pidfile_writer_under_src_calls_the_live_launch_guard_first():
    problems: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for name, line, why in unguarded_pidfile_writers(path.read_text(encoding="utf-8"), rel):
            problems.append(f"{rel}:{line} {name}: {why}")
    assert problems == [], "\n".join(problems)


def test_the_scan_is_not_vacuous_start_server_is_a_writer_and_the_writer_is_defined():
    total = 0
    for path in sorted(SRC.rglob("*.py")):
        total += writer_calls(path.read_text(encoding="utf-8"), path.relative_to(SRC).as_posix())
    assert total >= 1
    server_py = (SRC / "provider" / "server.py").read_text(encoding="utf-8")
    assert writer_calls(server_py, "provider/server.py") == 1
    tree = ast.parse(server_py)
    assert {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} >= {WRITER, GUARD}


def test_the_pidfile_name_is_written_through_atomic_write_text_in_one_place():
    """The direct form only ever appears inside the exempt `write_pidfile`: a second one would be a writer
    the structural scan can only judge by its own body."""
    direct = 0
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        direct += sum(1 for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and _callee(n) == "atomic_write_text" and _is_writer(n))
    assert direct == 1


# ------------------------------------------------------------------------------- the bites


def test_a_planted_writer_without_the_guard_is_reported():
    planted = _src("""
        def launch(d):
            server.write_pidfile(d, 1, port=None)
        """)
    assert unguarded_pidfile_writers(planted, "x.py") == [
        ("launch", 3, f"writes the pidfile and never calls {GUARD}")]


def test_a_guard_after_the_write_is_reported():
    planted = _src("""
        def launch(d, cmd):
            write_pidfile(d, 1, port=None)
            server.refuse_live_launch(d, cmd)
        """)
    assert unguarded_pidfile_writers(planted, "x.py") == [
        ("launch", 3, f"calls {GUARD} only after its first pidfile write")]


def test_a_direct_atomic_write_of_the_pidfile_without_the_guard_is_reported():
    for name in ("procfs.PIDFILE_NAME", "PIDFILE_NAME", '"server.pid"'):
        planted = _src(f"""
            def launch(f):
                private_files.atomic_write_text({name}, "1", dir_fd=f)
            """)
        assert [w[0] for w in unguarded_pidfile_writers(planted, "x.py")] == ["launch"], name


def test_a_function_that_guards_first_is_not_reported():
    guarded = _src("""
        def launch(d, cmd):
            note = server.refuse_live_launch(d, cmd)
            server.write_pidfile(d, 1, port=None)

        async def alaunch(d, cmd):
            refuse_live_launch(d, cmd)
            private_files.atomic_write_text(procfs.PIDFILE_NAME, "1", dir_fd=3)
        """)
    assert unguarded_pidfile_writers(guarded, "x.py") == []
    assert writer_calls(guarded) == 2


def test_a_guard_in_the_parent_does_not_cover_a_closure_that_writes():
    planted = _src("""
        def launch(d, cmd):
            refuse_live_launch(d, cmd)
            def later():
                write_pidfile(d, 1, port=None)
            return later
        """)
    assert [w[0] for w in unguarded_pidfile_writers(planted, "x.py")] == ["later"]


def test_a_writer_outside_any_function_is_reported():
    assert [w[0] for w in unguarded_pidfile_writers("write_pidfile(d, 1, port=None)\n", "x.py")] == ["<module>"]


def test_other_atomic_writes_are_not_pidfile_writers():
    other = _src("""
        def save(d):
            private_files.atomic_write_text("unit-target", "x", dir_fd=d)
            private_files.atomic_write_text(path, "x")
        """)
    assert unguarded_pidfile_writers(other, "x.py") == []


def test_only_the_definition_of_write_pidfile_in_server_py_is_exempt():
    definition = _src("""
        def write_pidfile(config_dir, pid, *, port, dir_fd=None):
            private_files.atomic_write_text(procfs.PIDFILE_NAME, "1", dir_fd=dir_fd)
        """)
    assert unguarded_pidfile_writers(definition, "provider/server.py") == []
    assert [w[0] for w in unguarded_pidfile_writers(definition, "provider/other.py")] == ["write_pidfile"]
