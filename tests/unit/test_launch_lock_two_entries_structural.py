"""One launch lock, two ways in (spec 20 §9.3, AUTO-07, D-14): the launch lock file is opened for locking by
`launch_lock` (async, every harness launch) and `launch_lock_sync` (`serve`, no event loop) and by nothing
else, and both test the same `_CONTENDED` errnos.

Why structural: a third way in (a second lock file, a second contention rule, a locker in `cli/` that
`flock`s the path itself) would silently split the exclusion that keeps two launchers from overwriting one
pidfile. A runtime test can only prove the entry points that exist today.

The rule the checker applies, over source text, resolving `server` the way the module imports it (so
`config/project_records.py`'s unrelated `lock_path` is never confused with `provider.server.lock_path`):

- every use of `provider.server.lock_path` under `src/localharness/` outside `provider/server.py` itself sits
  in `launch_lock`, `launch_lock_sync`, `launch_lock_wait_line` or `lend_launch_lock` of
  `provider/lifecycle.py`;
- exactly `launch_lock` and `launch_lock_sync` OPEN it (`os.open`); `launch_lock_wait_line` and
  `lend_launch_lock` only read `_HELD` by its key;
- the functions of `provider/lifecycle.py` that call `flock(..., LOCK_EX ...)` are those two plus
  `lend_launch_lock`, which re-takes the lender's OWN fd (it opens nothing, so it is not a third way in),
  and every one of them references `_CONTENDED`.

Mutation-checked by the synthetic cases below: a stray user or a locker without `_CONTENDED` is reported.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIFECYCLE = "provider/lifecycle.py"
SERVER = "provider/server.py"

LOCK_PATH_USERS = {"launch_lock", "launch_lock_sync", "launch_lock_wait_line", "lend_launch_lock"}
OPENERS = {"launch_lock", "launch_lock_sync"}
EXCLUSIVE_FLOCKERS = OPENERS | {"lend_launch_lock"}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _enclosing_function(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    while node in parents:
        node = parents[node]
        if isinstance(node, _FUNCTIONS):
            return node.name
    return "<module>"


def _server_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """How this module can reach `provider.server`: `(module aliases, bare lock_path aliases)`."""
    modules: set[str] = set()
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                bound = alias.asname or alias.name
                if alias.name == "server" and (mod.endswith("provider") or (node.level and not mod)):
                    modules.add(bound)
                elif alias.name == "lock_path" and (mod.endswith("provider.server") or (node.level and mod == "server")):
                    bare.add(bound)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("provider.server") and alias.asname:
                    modules.add(alias.asname)
    return modules, bare


def lock_path_uses(source: str, module: str) -> list[tuple[str, int]]:
    """`(enclosing function, line)` of every use of `provider.server.lock_path` in `source`, a call or a bare
    reference. `provider/server.py` itself defines it and is not scanned."""
    if module == SERVER:
        return []
    tree = ast.parse(source)
    parents = _parents(tree)
    modules, bare = _server_names(tree)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "lock_path":
            base = node.value
            via_alias = isinstance(base, ast.Name) and base.id in modules
            via_chain = isinstance(base, ast.Attribute) and base.attr == "server"
            if via_alias or via_chain:
                out.append((_enclosing_function(node, parents), node.lineno))
        elif isinstance(node, ast.Name) and node.id in bare and isinstance(node.ctx, ast.Load):
            out.append((_enclosing_function(node, parents), node.lineno))
    return sorted(out, key=lambda item: item[1])


def stray_lock_path_uses(source: str, module: str) -> list[tuple[str, str, int]]:
    """Uses of the server lock path outside the four allowed functions of provider/lifecycle.py."""
    return [(module, fn, line) for fn, line in lock_path_uses(source, module)
            if not (module == LIFECYCLE and fn in LOCK_PATH_USERS)]


def lock_openers(source: str, module: str) -> set[str]:
    """Functions that use the server lock path AND call `os.open` (the ones that take a descriptor on it)."""
    uses = {fn for fn, _ in lock_path_uses(source, module)}
    out = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, _FUNCTIONS) and node.name in uses:
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "open"
                        and isinstance(call.func.value, ast.Name) and call.func.value.id == "os"):
                    out.add(node.name)
    return out


def exclusive_flockers(source: str) -> dict[str, bool]:
    """Every function that calls `flock(..., LOCK_EX ...)` (directly in its own body), mapped to whether it
    references `_CONTENDED`."""
    tree = ast.parse(source)
    parents = _parents(tree)
    flockers: dict[str, bool] = {}
    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "flock"):
            continue
        names = {n.attr for a in call.args for n in ast.walk(a) if isinstance(n, ast.Attribute)}
        names |= {n.id for a in call.args for n in ast.walk(a) if isinstance(n, ast.Name)}
        if "LOCK_EX" not in names:
            continue
        flockers.setdefault(_enclosing_function(call, parents), False)
    for node in ast.walk(tree):
        if isinstance(node, _FUNCTIONS) and node.name in flockers:
            if any(isinstance(n, ast.Name) and n.id == "_CONTENDED" for n in ast.walk(node)):
                flockers[node.name] = True
    return flockers


def flock_violations(source: str) -> list[str]:
    """Exclusive flockers that are not the allowed set, and allowed or not, any that skips `_CONTENDED`."""
    found = exclusive_flockers(source)
    return sorted([f"{fn}: not a launch locker" for fn in found if fn not in EXCLUSIVE_FLOCKERS]
                  + [f"{fn}: no _CONTENDED" for fn, tested in found.items() if not tested])


def _files():
    return [(p.relative_to(SRC).as_posix(), p.read_text(encoding="utf-8")) for p in sorted(SRC.rglob("*.py"))]


# ------------------------------------------------------------------- the live source


def test_the_server_lock_path_is_used_only_by_the_launch_lock_functions():
    stray: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for module, text in _files():
        stray += stray_lock_path_uses(text, module)
        if module == LIFECYCLE:
            seen = {fn for fn, _ in lock_path_uses(text, module)}
    assert stray == []
    assert seen == LOCK_PATH_USERS  # the scan is not vacuous: each of the four really uses it


def test_exactly_the_async_and_the_sync_lock_open_the_lock_file():
    text = dict(_files())[LIFECYCLE]
    assert lock_openers(text, LIFECYCLE) == OPENERS
    opened_elsewhere = {module for module, source in _files() if module != LIFECYCLE and lock_openers(source, module)}
    assert opened_elsewhere == set()


def test_every_exclusive_flock_of_the_launch_lock_tests_the_one_contention_set():
    text = dict(_files())[LIFECYCLE]
    assert exclusive_flockers(text) == {name: True for name in EXCLUSIVE_FLOCKERS}
    assert flock_violations(text) == []
    assert text.count("_CONTENDED = frozenset(") == 1  # one contention rule, defined once


def test_no_other_module_defines_a_second_launch_lock_file_name():
    names = [module for module, source in _files() if module != SERVER and "launch.lock" in source]
    assert names == []


# ------------------------------------------------------------------- the scans bite

STRAY_LOCKER = '''
from localharness.provider import server

def other(d):
    fd = os.open(server.lock_path(d), os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
'''

THIRD_LOCKER = '''
from localharness.provider import server

def launch_lock(d):
    path = server.lock_path(d)
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in _CONTENDED:
            raise

def third_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
'''

CLEAN = '''
from localharness.provider import server

def launch_lock(d):
    fd = os.open(server.lock_path(d), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno not in _CONTENDED:
            raise
'''


def test_a_stray_locker_outside_lifecycle_is_reported():
    stray = stray_lock_path_uses(textwrap.dedent(STRAY_LOCKER), "cli/x.py")
    assert [(module, fn) for module, fn, _ in stray] == [("cli/x.py", "other")]
    assert lock_openers(textwrap.dedent(STRAY_LOCKER), "cli/x.py") == {"other"}


def test_a_stray_function_inside_lifecycle_is_reported():
    stray = stray_lock_path_uses(textwrap.dedent(STRAY_LOCKER), LIFECYCLE)
    assert [fn for _, fn, _ in stray] == ["other"]


def test_a_third_locker_without_the_contention_set_is_reported():
    source = textwrap.dedent(THIRD_LOCKER)
    assert flock_violations(source) == ["third_lock: no _CONTENDED", "third_lock: not a launch locker"]
    assert flock_violations(textwrap.dedent(CLEAN)) == []


def test_a_locker_that_tests_its_own_errno_set_is_reported():
    source = textwrap.dedent(CLEAN).replace("_CONTENDED", "{11}")
    assert flock_violations(source) == ["launch_lock: no _CONTENDED"]


def test_the_scan_resolves_the_import_so_another_lock_path_is_not_confused():
    own = textwrap.dedent('''
        from pathlib import Path

        def lock_path(root):
            return Path(root) / "x.lock"

        def other(root):
            return lock_path(root)
    ''')
    assert lock_path_uses(own, "config/project_records.py") == []


def test_every_way_to_reach_the_server_module_is_resolved():
    aliased = "from localharness.provider import server as srv\n\ndef f(d):\n    return srv.lock_path(d)\n"
    bare = "from localharness.provider.server import lock_path\n\ndef f(d):\n    return lock_path(d)\n"
    chain = "import localharness.provider.server\n\ndef f(d):\n    return localharness.provider.server.lock_path(d)\n"
    reference = "from localharness.provider import server\n\ndef f():\n    return server.lock_path\n"
    for source in (aliased, bare, chain, reference):
        assert [fn for fn, _ in lock_path_uses(source, "cli/x.py")] == ["f"], source
    assert lock_path_uses(aliased, SERVER) == []  # the defining module is not scanned
