"""Declared dependencies are true in both directions, and there is one JSON-repair spelling (D3/D4).

Every third-party package `src/` imports is declared (main or an extra), and every declared main
dependency is imported somewhere — bubus was declared from the start and never imported.

numpy, anyio and packaging were imported at module scope and never declared — they happened to be
present because scipy, httpx and the build stack pull them in. A transitive dependency is not a
promise: any of those can drop or re-pin it, and the failure lands on a fresh install, not here.

The repair fallback was spelled two ways. `json-repair` exists on PyPI (imported as `json_repair`);
`jsonrepair` does not, so `provider/fn_call.py`'s step 5 could never run whatever a user installed.
"""
from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
_SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


def _project() -> dict:
    return tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"]


@pytest.mark.parametrize("package", ["numpy", "anyio", "packaging"])
def test_directly_imported_packages_are_declared(package):
    declared = {req.split(">=")[0].split("<")[0].split("[")[0].strip().lower()
                for req in _project()["dependencies"]}

    assert package in declared


def test_the_nonexistent_jsonrepair_package_is_not_imported():
    offenders = [
        path
        for path in _SRC.rglob("*.py")
        if "from jsonrepair" in path.read_text(encoding="utf-8")
        or "import jsonrepair" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_json_repair_is_declared_as_an_extra_for_both_sites():
    extras = _project().get("optional-dependencies", {})

    assert any(req.lower().startswith("json-repair") for req in extras.get("json-repair", []))
    sites = [p for p in _SRC.rglob("*.py") if "from json_repair import" in p.read_text(encoding="utf-8")]
    assert {p.name for p in sites} == {"fn_call.py", "proposer.py"}


# Distribution → import name where the two differ. Everything else is `dist.replace("-", "_")`.
_IMPORT_NAME = {
    "pyyaml": "yaml",
    "agent-client-protocol": "acp",
    "discord.py": "discord",
    "llama-cpp-python": "llama_cpp",
}


def _dist_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower()


def _import_name(dist: str) -> str:
    return _IMPORT_NAME.get(dist, dist.replace("-", "_"))


def _imports_in_src() -> dict[str, set[str]]:
    """Top-level module name → the src files that import it (any scope; absolute imports only)."""
    found: dict[str, set[str]] = {}
    for path in _SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                found.setdefault(name.split(".")[0], set()).add(path.relative_to(_SRC).as_posix())
    return found


def test_every_third_party_import_in_src_is_declared():
    """A main dependency or an extra — never something another package happens to pull in."""
    project = _project()
    declared = {_import_name(_dist_name(r)) for r in project["dependencies"]}
    for reqs in project.get("optional-dependencies", {}).values():
        declared |= {_import_name(_dist_name(r)) for r in reqs}
    imported = _imports_in_src()
    third_party = set(imported) - set(sys.stdlib_module_names) - {"localharness"}

    undeclared = {name: sorted(imported[name]) for name in sorted(third_party - declared)}
    assert undeclared == {}


def test_every_declared_main_dependency_is_imported():
    """A dead main dependency cannot come back: bubus sat in pyproject.toml with no import."""
    imported = set(_imports_in_src())
    main = [_dist_name(r) for r in _project()["dependencies"]]

    dead = [dist for dist in main if _import_name(dist) not in imported]
    assert dead == []
