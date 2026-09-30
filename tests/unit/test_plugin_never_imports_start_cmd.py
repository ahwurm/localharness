"""PAPI-03's import clause as a ratchet: plugins never import cli/start_cmd.py.

What this covers, and only this: (1) the reference plugin every plugin is copied from — every .py in
examples/plugin-template/src/localharness_plugin_example/, a separate distribution the CORE-02 scan
of src/localharness cannot see — and (2) the plugin API, every module in src/localharness/plugins/.
From those it follows every import the CORE-02 checker counts (module-level, lazy, typing-only,
relative, literal import_module) plus the parent packages each one runs, through src/localharness,
and fails if cli/start_cmd.py is reachable. Not covered: any third-party plugin's own source (it is
not in this repo) and any import whose module name is computed at runtime.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tests.unit.test_import_direction import SRC, _imported_modules, module_file

REPO = SRC.parents[1]
EXAMPLE = REPO / "examples" / "plugin-template" / "src" / "localharness_plugin_example"
START_CMD = "cli/start_cmd.py"


def _runs(rel: str, path: Path) -> set[str]:
    """Files under SRC that importing `path` runs directly: each module it imports and the packages
    above it (importing a.b.c runs a, then a.b, then a.b.c)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), rel)
    return {f for node in ast.walk(tree) for m in _imported_modules(rel, node)
            for i in range(m.count(".") + 1) if (f := module_file(m.rsplit(".", i)[0]))}


def test_the_reference_plugin_and_the_plugin_api_never_import_start_cmd():
    """Any new import on those paths that reaches cli/start_cmd.py fails, naming the chain."""
    sources, api = sorted(EXAMPLE.rglob("*.py")), sorted((SRC / "plugins").rglob("*.py"))
    assert sources and api, f"nothing found under {EXAMPLE} or {SRC / 'plugins'} — a root is wrong"
    # file under SRC -> the file that first imported it (None for a plugin-API module)
    via: dict[str, str | None] = {p.relative_to(SRC).as_posix(): None for p in api}
    for p in sources:  # rel inside its own package, so its relative imports name nothing under SRC
        for f in _runs(p.relative_to(EXAMPLE.parent).as_posix(), p):
            via.setdefault(f, p.relative_to(REPO).as_posix())
    todo = list(via)
    while todo:
        rel = todo.pop()
        for f in _runs(rel, SRC / rel) - via.keys():
            via[f] = rel
            todo.append(f)
    chain = [START_CMD]
    while via.get(chain[-1]):
        chain.append(via[chain[-1]])
    assert START_CMD not in via, (
        "PAPI-03: a plugin gets what it needs from PluginContext, never from cli/start_cmd.py — "
        "this import chain reaches it:\n  " + " -> ".join(reversed(chain)))
