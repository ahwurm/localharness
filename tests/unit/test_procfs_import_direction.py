"""The `/proc` side imports nothing from the code that uses it (D-24, SC5): `provider/procfs.py`
never imports `provider.server`, `provider.lifecycle`, `provider.units` or anything under
`localharness.cli`, at module level or inside a function, and `provider/server.py` imports procfs.
That one direction is what keeps the procfs <-> server <-> units cycle from coming back.

The scan is an AST walk, so a lazy import inside a function counts; relative imports are resolved
against `localharness.provider`. The checker is a function over source text so the test can show
it bites on a source that breaks the rule."""
from __future__ import annotations

import ast
from pathlib import Path

PROVIDER = Path(__file__).resolve().parents[2] / "src" / "localharness" / "provider"
PACKAGE = "localharness.provider"
FORBIDDEN = ("localharness.provider.server", "localharness.provider.lifecycle",
             "localharness.provider.units", "localharness.cli")


def _names(node: ast.AST) -> list[str]:
    """Every dotted name one import statement could be importing: for `from a import b` both `a`
    and `a.b` (b may be a submodule), relative imports resolved against `localharness.provider`."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        base = PACKAGE.split(".")
        if node.level:
            base = base[: len(base) + 1 - node.level]
            module = ".".join([*base, *([node.module] if node.module else [])])
        else:
            module = node.module or ""
        return [module, *(f"{module}.{alias.name}" for alias in node.names)]
    return []


def forbidden_imports(source: str) -> list[str]:
    """The forbidden modules `source` imports anywhere, sorted and without repeats."""
    found = {
        name
        for node in ast.walk(ast.parse(source))
        for name in _names(node)
        for banned in FORBIDDEN
        if name == banned or name.startswith(banned + ".")
    }
    return sorted(found)


def test_procfs_imports_nothing_from_server_lifecycle_units_or_cli():
    source = (PROVIDER / "procfs.py").read_text(encoding="utf-8")
    assert forbidden_imports(source) == []


def test_server_imports_procfs():
    source = (PROVIDER / "server.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert any("localharness.provider.procfs" in _names(node) for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom))


def test_the_checker_reports_a_module_level_import_of_server():
    assert forbidden_imports("from localharness.provider import server\n") \
        == ["localharness.provider.server"]


def test_the_checker_reports_a_lazy_import_of_cli_inside_a_function():
    assert forbidden_imports("def f():\n    from localharness.cli import repl\n") \
        == ["localharness.cli", "localharness.cli.repl"]


def test_the_checker_reports_relative_and_plain_import_forms():
    assert forbidden_imports("from . import units\n") == ["localharness.provider.units"]
    assert forbidden_imports("from .lifecycle import x\n") \
        == ["localharness.provider.lifecycle", "localharness.provider.lifecycle.x"]
    assert forbidden_imports("import localharness.provider.server\n") \
        == ["localharness.provider.server"]


def test_the_checker_passes_the_imports_procfs_may_make():
    assert forbidden_imports("import os\nfrom localharness.provider import autosense\n"
                             "from localharness.core import private_files\n") == []
