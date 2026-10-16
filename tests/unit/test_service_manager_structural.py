"""No test runs the service manager, and only `provider/units.py` builds its argv (spec 20 §9.2, D-12, D-27).

A runtime test cannot show that a suite never reached the box's user manager, journal, login manager or
bus: the stubs in conftest hold only while every test goes through them. So this is structural, over the
source:

- no call under `tests/unit` that spawns a process (`subprocess.run` / `Popen` / `call` / `check_call` /
  `check_output`, `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `asyncio.create_subprocess_exec` /
  `_shell`) carries a string constant naming one of the five service-manager programs, anywhere in its
  arguments (a name inside an assertion or a fake's canned answer is not a spawn);
- under `src/localharness`, a list or tuple literal that starts with one of those names (an argv) exists
  only in `provider/units.py`;
- `provider/units.py` spawns a process only inside `_run` (init's sensing seam) and `_run_manager` (the
  autostart unit's).

The names are assembled from fragments so no line of this file is a command. Each rule has a "bites" test:
the same checker run over a planted source must report it, so a checker that stopped looking would fail.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "localharness"
TESTS = ROOT / "tests" / "unit"
UNITS = "provider/units.py"
RUNNERS = {"_run", "_run_manager"}

TOKENS = ("system" + "ctl", "login" + "ctl", "journal" + "ctl", "bus" + "ctl", "system" + "d-run")

SUBPROCESS_FUNCS = {"run", "Popen", "call", "check_call", "check_output"}
BARE_SPAWNS = {"Popen", "check_output", "check_call", "create_subprocess_exec", "create_subprocess_shell"}
ASYNCIO_FUNCS = {"create_subprocess_exec", "create_subprocess_shell"}
OS_FUNCS = {"system", "popen"}
OS_PREFIXES = ("exec", "spawn")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _chain(func: ast.expr) -> list[str]:
    """Every name of a dotted call, innermost first: `u.subprocess.run` is `run`, `subprocess`, `u`."""
    names: list[str] = []
    while isinstance(func, ast.Attribute):
        names.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        names.append(func.id)
    return names


def is_spawn(call: ast.Call) -> bool:
    chain = _chain(call.func)
    if not chain:
        return False
    name, owners = chain[0], chain[1:]
    if name in BARE_SPAWNS:
        return True
    if "subprocess" in owners and name in SUBPROCESS_FUNCS:
        return True
    if "asyncio" in owners and name in ASYNCIO_FUNCS:
        return True
    return "os" in owners and (name in OS_FUNCS or name.startswith(OS_PREFIXES))


def _names_a_program(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str)
               and any(token in n.value for token in TOKENS) for n in ast.walk(node))


def spawned_tokens(source: str) -> list[int]:
    """The line of every spawning call whose arguments hold a string constant naming one of the five
    programs, anywhere inside them (a nested list, an f-string part, a keyword's value)."""
    found: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and is_spawn(node):
            if any(_names_a_program(arg) for arg in (*node.args, *(k.value for k in node.keywords))):
                found.append(node.lineno)
    return sorted(found)


def argv_literal_owners(source: str, module: str) -> list[str]:
    """`<module>:<line>` for every list or tuple literal whose first element is a string constant equal to
    one of the five names, unless `module` is the one module that builds the manager's argv."""
    if module == UNITS:
        return []
    return [f"{module}:{node.lineno}" for node in ast.walk(ast.parse(source))
            if isinstance(node, (ast.List, ast.Tuple)) and node.elts
            and isinstance(node.elts[0], ast.Constant) and node.elts[0].value in TOKENS]


def stray_subprocess_calls(source: str) -> list[str]:
    """`<function>:<line>` for every spawning call that sits inside neither runner (`<module>` at module
    level). A helper nested in a runner is inside it; the name reported is the innermost function."""
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    stray: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and is_spawn(node)):
            continue
        enclosing: list[str] = []
        scope = node
        while scope in parents:
            scope = parents[scope]
            if isinstance(scope, _FUNCTIONS):
                enclosing.append(scope.name)
        if not RUNNERS.intersection(enclosing):
            stray.append((node.lineno, f"{enclosing[0] if enclosing else '<module>'}:{node.lineno}"))
    return [text for _, text in sorted(stray)]


def _sources(root: Path):
    for path in sorted(root.rglob("*.py")):
        yield path.relative_to(root).as_posix(), path.read_text(encoding="utf-8")


# -------------------------------------------------------------------------------------- live scans


def test_no_test_spawns_a_process_naming_a_service_manager_program():
    scanned = 0
    spawns = 0
    offenders: list[str] = []
    for module, source in _sources(TESTS):
        scanned += 1
        spawns += sum(isinstance(n, ast.Call) and is_spawn(n) for n in ast.walk(ast.parse(source)))
        offenders += [f"{module}:{line}" for line in spawned_tokens(source)]
    assert offenders == []
    assert scanned >= 400, f"the scan found {scanned} test files; the root moved"
    assert spawns >= 1, "no test spawns anything: the spawn detector is blind"


def test_only_units_py_builds_a_service_manager_argv():
    owners: list[str] = []
    in_units = 0
    for module, source in _sources(SRC):
        owners += argv_literal_owners(source, module)
        if module == UNITS:
            in_units = sum(
                isinstance(n, (ast.List, ast.Tuple)) and bool(n.elts) and isinstance(n.elts[0], ast.Constant)
                and n.elts[0].value in TOKENS for n in ast.walk(ast.parse(source)))
    assert owners == []
    assert in_units >= 4, "units.py builds no argv: the literal detector is blind"


def test_units_py_spawns_only_inside_its_two_runners():
    source = (SRC / UNITS).read_text(encoding="utf-8")
    assert stray_subprocess_calls(source) == []
    spawning = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Call) and is_spawn(n)]
    assert len(spawning) == 2, "units.py should hold exactly one spawn per runner"


def test_the_five_names_are_the_programs_the_hook_blocks():
    assert len(set(TOKENS)) == 5 and all(t.isascii() and t.islower() for t in TOKENS)
    assert "systemctl" in TOKENS[0] and TOKENS[-1].startswith("systemd-")


# ------------------------------------------------------------------------------------------ bites


def test_a_spawn_of_each_program_in_a_test_is_reported():
    for token in TOKENS:
        assert spawned_tokens(f'import subprocess\nsubprocess.run(["{token}", "--user", "show"])\n') == [2]


def test_other_spawn_forms_are_reported():
    token = TOKENS[0]
    forms = [
        f'import os\nos.system("{token} stop x")',
        f'import os\nos.popen("{token} stop x")',
        f'import os\nos.execv("/bin/{token}", ["{token}"])',
        f'import os\nos.spawnl(0, "/bin/{token}")',
        f'import asyncio\nasyncio.create_subprocess_exec("{token}", "stop")',
        f'import asyncio\nasyncio.create_subprocess_shell("{token} stop")',
        f'from subprocess import Popen\nPopen(["{token}", "stop"])',
        f'import subprocess\nsubprocess.check_output("{token} show", shell=True)',
        f'import subprocess\nsubprocess.run(cmd, env={{"X": "{token}"}})',
        f'import subprocess\nsubprocess.call(["sudo", "{token}", "stop"])',
        f'import subprocess\nsubprocess.run([f"/usr/bin/{token}", "stop"])',
        f'import subprocess\nsubprocess.run(args=("{token}",))',
        f'import localharness.provider.units as u\nu.subprocess.run(["{token}"])',
    ]
    for source in forms:
        assert spawned_tokens(source), source


def test_a_name_that_is_not_in_a_spawn_is_not_reported():
    token = TOKENS[0]
    quiet = [
        f'assert "{token}" in text',
        f'fake.answer(["{token}", "--user", "show"], "ok")',
        f'calls == [("{token}", "--user", "stop")]',
        f'x = ["{token}", "--user"]',
        f'import subprocess\nsubprocess.run(["git", "status"])',
        f'import subprocess\nsubprocess.run(["echo", "systemd"])',
    ]
    for source in quiet:
        assert spawned_tokens(source) == [], source


def test_an_argv_literal_outside_units_is_reported():
    token = TOKENS[0]
    planted = f'argv = ["{token}", "--user", "restart"]\n'
    assert argv_literal_owners(planted, "cli/x.py") == ["cli/x.py:1"]
    assert argv_literal_owners(f'argv = ("{TOKENS[2]}", "-n")\n', "agent/y.py") == ["agent/y.py:1"]
    assert argv_literal_owners(f'run(["{TOKENS[3]}", "--user"])\n', "core/z.py") == ["core/z.py:1"]


def test_the_same_literal_in_units_is_allowed():
    assert argv_literal_owners(f'argv = ["{TOKENS[0]}", "--user", "restart"]\n', UNITS) == []


def test_a_list_that_merely_holds_a_name_later_is_not_an_argv():
    token = TOKENS[0]
    assert argv_literal_owners(f'names = ["x", "{token}"]\nempty = []\nref = ["{token}x"]\n', "cli/x.py") == []


def test_a_spawn_inside_another_units_function_is_reported():
    source = textwrap.dedent("""
        import subprocess

        def _run(argv):
            return subprocess.run(argv)

        def _run_manager(argv):
            return subprocess.run(argv)

        def sneaky(argv):
            return subprocess.run(argv)

        async def lazy(argv):
            return await asyncio.create_subprocess_exec(*argv)

        top = subprocess.check_output(["x"])
    """)
    assert stray_subprocess_calls(source) == ["sneaky:11", "lazy:14", "<module>:16"]


def test_the_two_runners_alone_are_clean():
    source = textwrap.dedent("""
        import subprocess

        def _run(argv):
            return subprocess.run(argv)

        def _run_manager(argv):
            def inner():
                return subprocess.run(argv)
            return inner()
    """)
    assert stray_subprocess_calls(source) == []


def test_a_spawn_nested_in_a_lambda_inside_a_runner_is_seen_as_the_runner():
    source = "import subprocess\ndef _run(a):\n    return (lambda: subprocess.run(a))()\n"
    assert stray_subprocess_calls(source) == []
