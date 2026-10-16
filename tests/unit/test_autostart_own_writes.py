"""The harness's own writes never pass the agent's permission gate (spec 20 §9.3 last sentence, §12 "the gate
applies to the agent's tools only"; D-26, SEC-03).

The gate (`agent.verdict.evaluate`, `PermissionEvaluator.evaluate`, `PermissionGate.check`) decides what the
AGENT's tools may do. The unit file, `unit-target` and the boot marker are written by the harness on a person's
command (`--autostart on` / `off`) or by `serve` itself, so a verdict must never stand between the harness and
its own file: if one did, an org rule could block what the person ordered, or a verdict could be asked to bless
a write nobody typed. Two pins:

- every gate entry point is patched to raise, and the harness's own writers all succeed;
- provider/units.py, cli/autostart.py and cli/serve_cmd.py import nothing from `localharness.agent`, at module
  level or inside a function (a checker over the source text, mutation-checked below).

Nothing here runs a service manager or a login manager, starts a unit or a server, or dials a port.
"""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from localharness.agent import gate, permissions, verdict
from localharness.cli import autostart, serve_cmd
from localharness.provider import units
from tests.unit.test_autostart_on_off import (  # noqa: F401 — `world` is the fixture
    DISPLAY, World, stdout_lines, world,
)
from tests.unit.units_support import fake_manager  # noqa: F401 — the fixture

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"
OWN_WRITERS = (Path("provider") / "units.py", Path("cli") / "autostart.py", Path("cli") / "serve_cmd.py")


@pytest.fixture
def gate_raises(monkeypatch):
    """Every entry point of the permission gate raises if asked anything."""
    asked: list[str] = []

    def refuse(name: str):
        def boom(*args, **kwargs):
            asked.append(name)
            raise AssertionError(f"the harness's own write went through {name}")
        return boom

    monkeypatch.setattr(verdict, "evaluate", refuse("agent.verdict.evaluate"))
    monkeypatch.setattr(permissions.PermissionEvaluator, "evaluate", refuse("PermissionEvaluator.evaluate"))
    monkeypatch.setattr(gate.PermissionGate, "check", refuse("PermissionGate.check"))
    return asked


# ===================================================================== the harness's writers, with the gate raising


def test_on_and_off_succeed_with_every_gate_entry_point_raising(world, gate_raises):
    world.manager.answer_prefix("systemctl", "--user", "reset-failed", out="")
    world.manager.answer_prefix("systemctl", "--user", "disable", out="")

    turned_on = world.run()

    assert turned_on.exit_code == 0, (turned_on.output, turned_on.exception)
    assert world.unit_path.exists()

    turned_off = world.run(mode="off")

    assert turned_off.exit_code == 0, (turned_off.output, turned_off.exception)
    assert not world.unit_path.exists()
    assert stdout_lines(turned_off) == [f"autostart off: removed {DISPLAY}; nothing starts at login"]
    assert gate_raises == []


def test_the_target_the_marker_and_the_unit_file_are_written_with_the_gate_raising(world, gate_raises):
    body = world.body()
    marker = units.Marker(
        record="llamacpp:qwen", binary="/x/llama-server", invocation="abc", cause="none", shape="full",
        planned_context=None, full_fit=None, free_bytes=None, dropped=(), at="2026-10-15T00:00:00+00:00")

    units.write_target(world.g, "flash-next")
    units.write_marker(world.g, marker)
    written = units.write_harness_unit(body)

    assert units.read_target(world.g) == "flash-next"
    assert units.read_marker(world.g) == marker
    assert written == world.unit_path and world.unit_path.read_text(encoding="utf-8") == body
    assert gate_raises == []


def test_status_asks_the_gate_nothing_either(world, gate_raises):
    assert world.run().exit_code == 0

    assert autostart.status_lines(world.g)[0].startswith("Autostart: ")
    assert gate_raises == []


def test_the_gate_patch_really_bites(gate_raises):
    """The fixture's entry points raise and say so when something does ask: a pin that cannot fail pins nothing."""
    for call in (lambda: verdict.evaluate(), lambda: permissions.PermissionEvaluator().evaluate(None, None),
                 lambda: gate.PermissionGate.check(None)):
        with pytest.raises(AssertionError):
            call()

    assert gate_raises == ["agent.verdict.evaluate", "PermissionEvaluator.evaluate", "PermissionGate.check"]


# ===================================================================== the three modules import nothing from the agent


def agent_imports(source: str) -> list[tuple[int, str]]:
    """Every import in ``source``, at any depth, of `localharness.agent` or one of its submodules: an absolute
    `import localharness.agent...`, a `from localharness.agent... import`, `from localharness import agent`, and a
    relative import that climbs to `agent` (`from ..agent import`, `from .. import agent`)."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "localharness.agent" or alias.name.startswith("localharness.agent."):
                    found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                if module == "agent" or module.startswith("agent.") or (
                        not module and any(alias.name == "agent" for alias in node.names)):
                    found.append((node.lineno, "." * node.level + module))
            elif module == "localharness.agent" or module.startswith("localharness.agent."):
                found.append((node.lineno, module))
            elif module == "localharness" and any(alias.name == "agent" for alias in node.names):
                found.append((node.lineno, "localharness.agent"))
    return found


@pytest.mark.parametrize("module", OWN_WRITERS, ids=lambda p: p.as_posix())
def test_import_the_harness_own_writers_import_nothing_from_the_agent(module):
    source = (SRC / module).read_text(encoding="utf-8")

    assert agent_imports(source) == []


@pytest.mark.parametrize("planted", [
    "from localharness.agent.verdict import evaluate\n",
    "import localharness.agent.gate\n",
    "from localharness import agent\n",
    "from ..agent import permissions\n",
    "def later():\n    from localharness.agent.permissions import PermissionEvaluator\n    return PermissionEvaluator\n",
], ids=["from_verdict", "import_gate", "from_package", "relative", "inside_a_function"])
def test_import_pin_bites_on_a_planted_agent_import(planted):
    source = textwrap.dedent("import os\n") + planted

    assert agent_imports(source) != []


def test_import_pin_leaves_other_packages_alone():
    source = ("from localharness.config import trust\nfrom localharness.provider import units\n"
              "import localharness.agentless\nfrom localharness.cli import autostart\n")

    assert agent_imports(source) == []


def test_import_pin_names_the_modules_it_reads():
    assert [m.as_posix() for m in OWN_WRITERS] == ["provider/units.py", "cli/autostart.py", "cli/serve_cmd.py"]
    assert all((SRC / m).is_file() for m in OWN_WRITERS)
    assert serve_cmd.__file__ == str(SRC / "cli" / "serve_cmd.py")
