"""Every package `src/` imports directly is declared, and there is one JSON-repair spelling (D3/D4).

numpy, anyio and packaging were imported at module scope and never declared — they happened to be
present because scipy, httpx and the build stack pull them in. A transitive dependency is not a
promise: any of those can drop or re-pin it, and the failure lands on a fresh install, not here.

The repair fallback was spelled two ways. `json-repair` exists on PyPI (imported as `json_repair`);
`jsonrepair` does not, so `provider/fn_call.py`'s step 5 could never run whatever a user installed.
"""
from __future__ import annotations

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
