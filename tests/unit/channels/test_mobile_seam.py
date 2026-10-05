"""WEBP-03 / roadmap criterion 4: the phone reaches memory only through the slot and never imports
another plugin. CORE-02's ratchet checks CORE importers only (`edges_of` is empty for a plugin file
like these), so this plugin-to-plugin line has its own scan over the same AST helpers. Lazy
(function-level) imports count — the scanner walks every Import node."""
from __future__ import annotations

import ast

import pytest

from tests.unit.test_import_direction import SRC, _imported_modules, module_file

WEB_FILES = sorted(p.relative_to(SRC).as_posix() for p in (SRC / "channels/mobile").glob("*.py"))
FORBIDDEN = ("memory/", "tools/builtin/image_plugin.py", "tools/builtin/generate_image_tool.py",
             "cli/generate_image_cmd.py", "tools/builtin/workflows/")


def imported_files(rel: str, text: str) -> set[str]:
    """Every localharness source file `text` (as file `rel`) imports, at any depth of the AST."""
    return {f for node in ast.walk(ast.parse(text, rel)) for m in _imported_modules(rel, node)
            if (f := module_file(m))}


@pytest.mark.parametrize("rel", WEB_FILES)
def test_the_web_package_imports_no_memory_and_no_image_code(rel):
    bad = sorted(f for f in imported_files(rel, (SRC / rel).read_text(encoding="utf-8"))
                 if f.startswith(FORBIDDEN))
    assert not bad, f"{rel} imports {bad}"


def test_the_scan_sees_a_top_level_memory_import():
    found = imported_files("channels/mobile/server.py", "from localharness.memory.sqlite import FactQuery\n")
    assert "memory/sqlite.py" in found and any(f.startswith(FORBIDDEN) for f in found)


def test_the_scan_sees_a_function_level_import():
    src = "def f():\n    from localharness.tools.builtin import image_plugin\n"
    assert "tools/builtin/image_plugin.py" in imported_files("channels/mobile/server.py", src)
