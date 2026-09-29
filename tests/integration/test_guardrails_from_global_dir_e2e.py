"""Criterion 3 of the plugin substrate (SAFE-04) — the org guardrails come from the GLOBAL dir only,
with memory on AND with memory off.

**What this proves.** `<global>/orgs/default/GUARDRAILS.md` and a project's own
`.localharness/orgs/default/GUARDRAILS.md` hold different text. A real session started inside that
project — the REAL `_start_async` with `config_dir=None`, so workspace discovery is live — hands the
model a system prompt that carries ONLY the global text. It is shown twice, in one test because it is
one claim: once with memory on (the default) and once with `org.memory_enabled: false` in the global
`config.yaml`. Before SAFE-04 the second drive had no guardrails at all: memory was the reader and
the splice sat inside `if self._memory is not None:`, so turning memory off turned the org's safety
rules off with it. Each drive also shows which memory state it really ran in — the division section,
which only memory supplies, is present with memory on and absent with memory off — so a green
memory-off drive cannot be a memory-on drive in disguise.

**Offline by construction, not by trust.** Only the external boundaries are stubbed
(`_workspace_start` -> `tests/unit/test_start_cmd.py::_stub_start_boundaries`: the LLM probe, the
tokenizer, the REPL loop, plugin discovery). The provider's `base_url` is rewritten to the loopback
discard port (`_offline_provider`), so anything that dials the provider is refused instantly
instead of hanging, and `LLMClient.stream_complete` itself is replaced by a recorder that answers
"Done." — no model is reached and no GPU is touched. Measured when this was written (socket.connect
wrapped): each drive makes exactly two connects, both start-up probes aimed at the discard port
(`list_live_models` -> GET /models, `probe_served_window` -> GET /api/ps) and both refused at once;
the turn itself opens none. Every helper is imported, never copied (41-06's rule).

**Standing invariant.** This must stay green at the end of every later phase. The memory phase
re-runs it with the memory plugin as the slot occupant.

**What it does NOT prove.** A turn with tool calls: the scripted turn answers at once, so only the
first request's system prompt is graded. And subagents: no subagent construction passes a guardrails
path, so they never received the org guardrails before this change and still do not.
"""
from __future__ import annotations

from pathlib import Path

from tests.conftest import FakeLLMResponse
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_workspace_state_landing import _drive, _workspace_start

GLOBAL_GUARDRAILS = "# Org guardrails\nGLOBAL-GUARDRAILS-SENTINEL\n"
WORKSPACE_GUARDRAILS = "# Org guardrails\nWORKSPACE-GUARDRAILS-SENTINEL\n"
GLOBAL_DIVISION = "# Division\nGLOBAL-DIVISION-SENTINEL\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _let_the_stub_tokenizer_run_a_turn(monkeypatch) -> None:
    """Extend the harness's offline token counter with the one method a TURN needs.

    `_stub_start_boundaries` installs a counter with `count`/`count_messages` only, which is enough
    for start-up. A turn's `build_messages` also calls `estimate_messages` — measured: without it
    the turn dies on an AttributeError before any request is sent. This subclass adds it with the
    real counter's structural formula (4 tokens of overhead per message plus its content), so the
    stub's own `count` still decides every number.
    """
    import localharness.agent.context as _context

    stub = _context.TokenCounter  # the harness's _StubTokenCounter, installed just above

    class _TurnCapableCounter(stub):
        def estimate_messages(self, messages):
            return sum(4 + self.count(m.get("content") or "") for m in messages)

    monkeypatch.setattr("localharness.agent.context.TokenCounter", _TurnCapableCounter)


async def _first_system_prompt(root: Path, monkeypatch, fake_home, *, memory_enabled: bool) -> str:
    """Drive one real turn of a real workspace session under `root`; return the system prompt of
    the first request that reached the model boundary."""
    _home, global_dir, ws = _workspace_start(root, monkeypatch, fake_home)
    _offline_provider(global_dir)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    _write(global_dir / "orgs" / "default" / "GUARDRAILS.md", GLOBAL_GUARDRAILS)
    _write(ws / "orgs" / "default" / "GUARDRAILS.md", WORKSPACE_GUARDRAILS)
    # The discriminator: DIVISION.md reaches the prompt only through memory.
    _write(global_dir / "divisions" / "default" / "DIVISION.md", GLOBAL_DIVISION)
    if not memory_enabled:
        with (global_dir / "config.yaml").open("a", encoding="utf-8") as f:
            f.write("org:\n  memory_enabled: false\n")

    captured: list[dict] = []

    async def fake_stream_complete(self, messages, tools=None, on_token=None, **kw):
        captured.append(dict(messages[0]))
        return FakeLLMResponse(content="Done."), None

    async def one_turn(self):
        await self._agent.run_turn("say hello")

    monkeypatch.setattr(
        "localharness.provider.client.LLMClient.stream_complete", fake_stream_complete
    )
    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", one_turn)

    await _drive()

    assert captured, "no request reached the model boundary — the turn never ran"
    assert captured[0]["role"] == "system", captured[0]
    return captured[0]["content"]


async def test_guardrails_come_from_the_global_dir_with_memory_on_and_off(
    tmp_path, monkeypatch, fake_home
):
    on = await _first_system_prompt(
        tmp_path / "memory-on", monkeypatch, fake_home, memory_enabled=True
    )
    off = await _first_system_prompt(
        tmp_path / "memory-off", monkeypatch, fake_home, memory_enabled=False
    )

    # The workspace check comes FIRST: under a reader that followed the workspace the global text
    # is also missing, and a presence check on top would fire instead and hide what went wrong.
    for label, prompt in (("memory on", on), ("memory off", off)):
        assert "WORKSPACE-GUARDRAILS-SENTINEL" not in prompt, (
            f"{label}: the project's own GUARDRAILS.md reached the model — a workspace rewrote "
            "the org's safety voice"
        )
        assert "GLOBAL-GUARDRAILS-SENTINEL" in prompt, (
            f"{label}: the org's global guardrails never reached the model"
        )
        assert prompt.count("## Guardrails\n") == 1, f"{label}: guardrails injected twice"

    # Memory really was on: both of its sections are there (a fresh store still renders its
    # index header — measured), after the guardrails, in the order the prompt always had.
    assert "## Division Context\n" + GLOBAL_DIVISION in on
    assert "## Agent Memory\n" in on
    assert on.index("## Guardrails\n") < on.index("## Division Context\n") < on.index(
        "## Agent Memory\n"
    )

    # Memory really was off: nothing memory supplies is in the prompt.
    assert "## Division Context" not in off, "memory-off drive still ran memory"
    assert "## Agent Memory" not in off, "memory-off drive still ran memory"
