"""provider/tuning.py runs nothing and executes no file content (spec 20 §6.0; D-07, D-29, SEC-05).

Structural, like tests/unit/test_launch_lock_structural.py: the module's AST must hold no import of
process, exec or dynamic-import machinery and no call of an executing builtin. A synthetic source for
each rule proves the checker still sees it.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

TUNING = Path(__file__).resolve().parents[2] / "src" / "localharness" / "provider" / "tuning.py"

FORBIDDEN_MODULES = {"subprocess", "importlib", "runpy", "ctypes", "pickle", "marshal", "shelve",
                     "multiprocessing", "pty"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__"}
FORBIDDEN_OS_PREFIXES = ("exec", "spawn", "fork")
FORBIDDEN_OS_NAMES = {"system", "popen"}


def violations(source: str) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                found.append(f"from {node.module} import")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                found.append(f"{func.id}()")
            elif isinstance(func, ast.Name) and func.id == "open":
                found.append("open()")  # a read goes through open_regular, never a bare open
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                    and func.value.id == "os" and (func.attr in FORBIDDEN_OS_NAMES
                                                   or func.attr.startswith(FORBIDDEN_OS_PREFIXES)):
                found.append(f"os.{func.attr}()")
    return found


def test_tuning_py_runs_nothing_and_executes_no_file_content():
    assert violations(TUNING.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("source", [
    "import subprocess", "import importlib.util", "from runpy import run_path", "import ctypes",
    "import pickle", "from marshal import loads", "import shelve", "import multiprocessing as mp",
    "import pty", "exec('x')", "eval('1')", "compile('x', 'f', 'exec')", "__import__('os')",
    "open('/x')", "import os\nos.system('x')", "import os\nos.popen('x')", "import os\nos.execv('a', [])",
    "import os\nos.spawnl(0, 'a')", "import os\nos.fork()", "import os\nos.forkpty()",
])
def test_the_checker_sees_each_forbidden_form(source):
    assert violations(textwrap.dedent(source)) != []


def test_the_checker_passes_a_pure_module():
    assert violations("from collections.abc import Mapping\n\ndef f(x):\n    return x\n") == []
