"""Phase 26 (MODP-01/02) — modular role sections + the byte-identity safety rail.

Proves the `_assemble_role` helper is byte-identical to the bare `role` monolith when no
section is set (ROADMAP success criterion 4 — the load-bearing assertion: same object, no
reformat) and that overlaying exactly ONE section appends only that section's text (MODP-02).

Offline: Tests A-C are pure-helper assertions (no AgentLoop, no async, no model). Test D drives
a real AgentLoop with FaithfulFakeLLM(tool_plan=[]) + tool_registry=None + native tool_call_mode
+ no memory loader (mirrors test_agent_loop_selfcheck._make_loop) so neither the non-native
suffix nor the Phase-24 memory block fire — proving the seam at loop.py:452 did not perturb the
assembled session.messages[0]['content']. No live model.
"""
from __future__ import annotations

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.loop import AgentLoop, Session, _assemble_role
from localharness.agent.permissions import PermissionEvaluator
from localharness.config.models import AgentConfig


# ---------------------------------------------------------------------------
# Tests A-C: pure-helper assertions (no AgentLoop, no async, no model).
# ---------------------------------------------------------------------------


def test_unmutated_assembly_is_byte_identical():
    """Test A (criterion 4): default sections -> _assemble_role returns cfg.role, same object."""
    role = "You are Octo. Use tools carefully.\n\nStop when the task is done."  # arbitrary multi-line
    cfg = AgentConfig(name="x", role=role)  # all sections default ""
    assert _assemble_role(cfg) == role  # byte-exact, no reformat
    assert _assemble_role(cfg) is cfg.role  # same object — zero mutation


def test_single_section_mutation_changes_only_that_section():
    """Test B (MODP-02): overlaying ONE section appends only that section; role preserved first."""
    role = "BASE ROLE"
    cfg = AgentConfig(
        name="x",
        role=role,
        role_sections={"tool_use": "Prefer the write_secret tool."},
    )
    out = _assemble_role(cfg)
    assert "Prefer the write_secret tool." in out  # the mutated section is present
    assert out.startswith("BASE ROLE")  # role preserved verbatim, first
    # identity/stopping/output still "" -> only one section appended
    assert out == "BASE ROLE\n\nPrefer the write_secret tool."
    # the other three sections stay untouched
    assert cfg.role_sections.identity == ""
    assert cfg.role_sections.stopping == ""
    assert cfg.role_sections.output == ""
    assert cfg.role == "BASE ROLE"  # role itself never rewritten


def test_only_mutated_section_adds_one_join():
    """Test C: exactly one join (base + one section) — empty sections add no stray separators."""
    cfg = AgentConfig(
        name="x",
        role="BASE ROLE",
        role_sections={"tool_use": "Prefer the write_secret tool."},
    )
    out = _assemble_role(cfg)
    assert out.count("\n\n") == 1


# ---------------------------------------------------------------------------
# Test D: loop-level capture (integration of the seam, Pitfall 1 guard).
# ---------------------------------------------------------------------------


def _make_loop(llm, bus) -> AgentLoop:
    """Real AgentLoop, offline deps, native mode, no memory (mirrors test_agent_loop_selfcheck)."""
    cfg = AgentConfig.model_validate({"name": "rolesec-agent", "role": "You are a test agent."})
    loop = AgentLoop(
        config=cfg,
        llm=llm,
        bus=bus,
        context_manager=ContextManager(),
        tool_registry=None,
        permission_evaluator=PermissionEvaluator(),
        memory_loader=None,  # no memory block fires
    )
    return loop


@pytest.mark.asyncio
async def test_loop_system_prompt_unchanged_for_default_sections(faithful_fake_llm, bus):
    """Test D: a default-sections AgentLoop run yields messages[0]['content'] == cfg.role exactly.

    Native tool_call_mode (no non-native suffix) + memory_loader=None (no Phase-24 block) means the
    only contribution to the system message is the assembled role — so the seam at loop.py:452 must
    leave it byte-identical to cfg.role.
    """
    loop = _make_loop(faithful_fake_llm(tool_plan=[]), bus)  # final-answer fake -> natural completion
    session = Session(agent_id="rolesec-agent", session_id="s-rolesec", messages=[])

    await loop._execute_loop(session, "do the task", None)

    assert session.messages[0]["role"] == "system"
    assert session.messages[0]["content"] == loop._config.role
