"""The config layer never imports the provider layer at import time.

`config/trust.py` reads the autostart unit through `localharness.provider.units`, a layer above it.
The edge is allowed only as a function-local import (the file already does the same for
`provider.server`), so importing `localharness.config.trust` pulls no provider module in. Pinned on
the AST, with a planted module-level import to show the pin bites.
"""
from __future__ import annotations

import ast
from pathlib import Path

import localharness.config.trust as trust_mod

SOURCE = Path(trust_mod.__file__)


def _is_provider(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        in_provider = (node.module or "").split(".")[:2] == ["localharness", "provider"]
        from_package = node.module == "localharness" and any(a.name == "provider" for a in node.names)
        return node.level == 0 and (in_provider or from_package)
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[:2] == ["localharness", "provider"] for a in node.names)
    return False


def module_level_provider_imports(source: str) -> list[int]:
    """Line numbers of every provider import that runs when the module is imported: anything outside a
    function body (module level, `if`/`try`/`with` blocks and class bodies included)."""
    found: list[int] = []

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if _is_provider(child):
                found.append(child.lineno)
            walk(child)

    walk(ast.parse(source))
    return sorted(found)


def function_local_provider_imports(source: str) -> dict[str, list[str]]:
    """{function name: the provider modules it imports} for every function that does."""
    out: dict[str, list[str]] = {}
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if _is_provider(node):
                names = (node.module or "") if isinstance(node, ast.ImportFrom) else ",".join(
                    a.name for a in node.names)
                out.setdefault(fn.name, []).append(names)
    return out


def test_trust_has_no_module_level_import_of_the_provider_layer():
    assert module_level_provider_imports(SOURCE.read_text(encoding="utf-8")) == []


def test_the_units_import_sits_inside_autostart_entries():
    local = function_local_provider_imports(SOURCE.read_text(encoding="utf-8"))

    assert "localharness.provider" in local.get("autostart_entries", [])


PLANTED_TOP = "from localharness.provider import units\n\ndef f():\n    return units\n"
PLANTED_IMPORT = "import localharness.provider.units\n"
PLANTED_GUARDED = "try:\n    from localharness.provider.server import KEY_ENV\nexcept ImportError:\n    KEY_ENV = ''\n"
PLANTED_CLASS = "class C:\n    from localharness.provider import units\n"
LOCAL_ONLY = "def f():\n    from localharness.provider import units\n    return units\n"


def test_a_planted_module_level_import_is_reported():
    assert module_level_provider_imports(PLANTED_TOP) == [1]
    assert module_level_provider_imports(PLANTED_IMPORT) == [1]
    assert module_level_provider_imports(PLANTED_GUARDED) == [2]
    assert module_level_provider_imports(PLANTED_CLASS) == [2]


def test_a_function_local_import_is_not_reported():
    assert module_level_provider_imports(LOCAL_ONLY) == []
    assert function_local_provider_imports(LOCAL_ONLY) == {"f": ["localharness.provider"]}


def test_other_layers_are_not_mistaken_for_the_provider_layer():
    assert module_level_provider_imports("from localharness.config import loader\n") == []
    assert module_level_provider_imports("import localharness.providers_like\n") == []
