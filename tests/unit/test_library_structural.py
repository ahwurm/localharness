"""Structural pins: a person is proven before any sync or setup, one module walks the model roots, one module asks."""
from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.test_interview_structural import _ask_names, _is_ask_call

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
LIBRARY = SRC / "cli" / "library.py"
MODEL_LIBRARY = SRC / "provider" / "model_library.py"
GGUF_HEADER = SRC / "agent" / "gguf_header.py"
PERSON_PARAMS = {"interactive", "person"}
INTERACTIVE_ARG = {"core_setup": ("interactive", None), "_pick_server": ("interactive", 3),
                   "_existing_config": ("interactive", 2)}


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _is_local_operator(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == "local_operator") or (
        isinstance(f, ast.Attribute) and f.attr == "local_operator")


def _enclosing(node: ast.AST, parents: dict) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _inside_a_person_block(node: ast.AST, parents: dict) -> bool:
    child = node
    while child in parents:
        parent = parents[child]
        if isinstance(parent, ast.If) and child in parent.body and _is_local_operator(parent.test):
            return True
        child = parent
    return False


def _proven(value: ast.AST, call: ast.Call, parents: dict) -> bool:
    if _is_local_operator(value):
        return True
    fn = _enclosing(call, parents)
    if isinstance(value, ast.Name) and fn is not None and value.id in PERSON_PARAMS:
        return value.id in {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    return isinstance(value, ast.Constant) and value.value is True and _inside_a_person_block(call, parents)


def _library_sync_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """(names bound to the library module, names bound to its `sync`)."""
    modules, syncs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name == "library":
                    modules.add(a.asname or a.name)
                if node.module and node.module.endswith("cli.library") and a.name == "sync":
                    syncs.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            modules |= {a.asname for a in node.names if a.name.endswith(".library") and a.asname}
    return modules, syncs


def _callee(call: ast.Call) -> str | None:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def unproven_persons(source: str, name: str = "<source>") -> list[str]:
    tree = ast.parse(source)
    parents = _parents(tree)
    modules, syncs = _library_sync_names(tree)
    found = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        callee = _callee(call)
        if callee in INTERACTIVE_ARG:
            keyword, index = INTERACTIVE_ARG[callee]
            value = next((k.value for k in call.keywords if k.arg == keyword), None)
            if value is None and index is not None and len(call.args) > index:
                value = call.args[index]
        elif (isinstance(call.func, ast.Attribute) and call.func.attr == "sync"
              and isinstance(call.func.value, ast.Name) and call.func.value.id in modules) or (
                isinstance(call.func, ast.Name) and call.func.id in syncs):
            value = next((k.value for k in call.keywords if k.arg == "person"), None)
        else:
            continue
        if value is None or not _proven(value, call, parents):
            fn = _enclosing(call, parents)
            found.append(f"{name}:{call.lineno} in {fn.name if fn else '<module>'}")
    return found


def test_every_sync_and_setup_call_passes_a_person_that_local_operator_proved():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        found += unproven_persons(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    assert found == []


PROVEN = '''
from localharness.cli import library
from localharness.cli.workspace import local_operator

def boundary(no_input, config):
    core_setup(config, interactive=local_operator(no_input))
    if local_operator(no_input):
        core_setup(config, interactive=True)

def inner(config, interactive):
    _pick_server(config, None, None, interactive)
    _existing_config(config, config, interactive)
    library.sync(config, scope="all", person=interactive)
'''


def test_the_rule_passes_every_proven_shape():
    assert unproven_persons(PROVEN) == []


def test_the_rule_names_a_sync_handed_a_constant_person():
    source = PROVEN + "\ndef bad(config):\n    library.sync(config, scope='all', person=True)\n"
    assert unproven_persons(source) == ["<source>:16 in bad"]


def test_the_rule_names_a_person_that_is_not_the_enclosing_functions_parameter():
    source = PROVEN + "\ndef bad(config):\n    library.sync(config, scope='all', person=interactive)\n"
    assert unproven_persons(source) == ["<source>:16 in bad"]


def test_the_rule_names_a_setup_handed_its_own_terminal_check():
    source = PROVEN + "\ndef bad(config):\n    core_setup(config, interactive=is_tty())\n    _pick_server(config, 1, 2, True)\n"
    assert unproven_persons(source) == ["<source>:16 in bad", "<source>:17 in bad"]


def test_the_rule_names_a_constant_true_outside_a_person_block():
    source = PROVEN + "\ndef bad(config, no_input):\n    if not local_operator(no_input):\n        core_setup(config, interactive=True)\n"
    assert unproven_persons(source) == ["<source>:17 in bad"]


def _walkers(source: str, name: str = "<source>") -> list[str]:
    tree = ast.parse(source)
    headers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "localharness.agent.gguf_header":
            headers |= {a.asname or a.name for a in node.names if a.name in ("discover", "walk_ggufs")}
    found = []
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        f = call.func
        if (isinstance(f, ast.Name) and f.id in headers) or (
                isinstance(f, ast.Attribute) and f.attr in ("discover", "walk_ggufs")
                and isinstance(f.value, ast.Name) and f.value.id == "gguf_header"):
            found.append(f"{name}:{call.lineno}")
    return found


def test_only_the_model_library_walks_a_root_or_types_a_header():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path not in (MODEL_LIBRARY, GGUF_HEADER):
            found += _walkers(path.read_text(encoding="utf-8"), str(path.relative_to(SRC)))
    assert found == []


def test_a_walker_outside_the_model_library_is_named():
    source = "from localharness.agent.gguf_header import discover, walk_ggufs\n\ndef f(r):\n    walk_ggufs(r)\n    discover(r)\n"
    assert _walkers(source) == ["<source>:4", "<source>:5"]
    assert _walkers("from localharness.agent import gguf_header\n\ndef f(r):\n    gguf_header.discover(r)\n") == ["<source>:4"]


def test_the_plugins_discover_is_not_a_gguf_walker():
    assert _walkers("from localharness.plugins import discovery\n\ndef f(d):\n    discovery.discover(d)\n") == []


def _askers(source: str, name: str = "<source>") -> list[str]:
    tree = ast.parse(source)
    modules, asks = _ask_names(tree)
    return [f"{name}:{n.lineno}" for n in ast.walk(tree) if _is_ask_call(n, modules, asks)]


def test_only_the_library_asks():
    asking = {str(path.relative_to(SRC)) for path in SRC.rglob("*.py")
              if _askers(path.read_text(encoding="utf-8"))}
    assert asking == {str(LIBRARY.relative_to(SRC))}


def test_an_asker_outside_the_library_is_named():
    source = "from localharness.cli import interview\n\ndef f(person):\n    return interview.ask('q', [], default=0, cancel='x')\n"
    assert _askers(source) == ["<source>:4"]
