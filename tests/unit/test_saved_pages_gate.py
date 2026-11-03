"""An agent that has read a page saved from the web asks before it runs or deletes anything, in every mode."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localharness.agent.context import ContextManager
from localharness.agent.gate import NO_ASKER_REASON, PermissionGate
from localharness.agent.gate_types import Decision, GateSettings, ToolMeta, Verdict
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.shell_classify import classify_shell
from localharness.agent.taint import Taint, content_key
from localharness.agent.verdict import (
    DOWNLOAD_EXEC_DETAIL,
    DOWNLOADS_WRITE_REASON,
    SAVED_PAGES_NOTE,
    WRITTEN_FILE_REASON,
    GateContext,
    evaluate,
)
from localharness.config.grants import GrantStore, new_grant
from localharness.config.models import AgentConfig
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from localharness.tools.builtin.read_tool import ReadTool
from localharness.tools.builtin.write_tool import WriteTool
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient
from tests.unit.test_compact_md_disable_sentinel import _loop as repl_loop
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl

AGENT = "test-agent"
SAVED = ".localharness/downloads"
NOTE = "the agent has read pages saved from the web"


class Stub(Tool):
    """Stands in for bash_exec / python_exec: classified by name like the real one, runs nothing but its recorded `effects`."""

    def __init__(self, name: str, family: str, param: str, effects: dict[str, tuple[Path, str]] | None = None) -> None:
        super().__init__()
        self._name, self._family, self._param = name, family, param
        self.effects = effects or {}
        self.ran: list[str] = []

    def info(self) -> ToolSchema:
        return ToolSchema(
            name=self._name, description="stub", group=self._family, gate_family=self._family,  # type: ignore[arg-type]
            ingest="none", host="dangerous",
            parameters={"type": "object", "properties": {self._param: {"type": "string"}},
                        "required": [self._param]},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        text = kwargs.get(self._param, "")
        self.ran.append(text)
        if text in self.effects:
            path, body = self.effects[text]
            path.write_text(body)
        return self.ok("ran")


class Rig(SimpleNamespace):
    async def run(self, *calls: tuple[str, dict]) -> None:
        plan = [FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id=f"tc-{i}", name=n, arguments=a)])
                for i, (n, a) in enumerate(calls)] + [FakeLLMResponse(content="Done.")]
        loop = AgentLoop(config=AgentConfig(name=AGENT, role="Test agent."), llm=MockLLMClient(plan),
                         bus=self.bus, context_manager=ContextManager(), tool_registry=self.registry,
                         permission_evaluator=PermissionEvaluator(), gate=self.gate)
        await loop.run_turn("t")
        self.loop = loop

    async def read(self, *names: str) -> None:
        await self.run(*(("read", {"path": str(self.ws / SAVED / n)}) for n in names))

    def errors(self) -> list[str]:
        return [e.error or "" for e in self.bus.history() if getattr(e, "observation_type", None) == "tool_result"]

    def marked(self) -> str | None:
        return self.gate.taint.source(AGENT)


@pytest.fixture
def make(bus, tmp_path, monkeypatch):
    async def build(mode: str = "auto", asker=None, **gate_kw) -> Rig:
        ws = (tmp_path / "project").resolve()
        (ws / SAVED).mkdir(parents=True)
        for n in "abc":
            (ws / SAVED / f"{n}.md").write_text(f"page {n}\n")
        (ws / "notes.md").write_text("page a\n")
        (ws / "old.sh").write_text("echo old\n")
        monkeypatch.chdir(ws)
        gate = PermissionGate(boundary=ws, workspace=ws, grants=GrantStore(tmp_path / "grants.yaml"), mode=mode,
                              asker=asker, channel_name="test", bus=bus, has_review_surface=True, **gate_kw)
        shell = Stub("bash_exec", "shell", "command")
        py = Stub("python_exec", "code", "code")
        registry = ToolRegistry()
        for tool in (ReadTool(), WriteTool(), shell, py):
            await registry.register(tool, scope="global")
        return Rig(ws=ws, gate=gate, bus=bus, registry=registry, shell=shell, py=py, tmp=tmp_path)
    return build


class Asker:
    def __init__(self, kind: str = "allow_once") -> None:
        self.kind, self.calls = kind, []

    async def __call__(self, request):
        self.calls.append(request)
        return Decision(kind=self.kind)  # type: ignore[arg-type]


def sh(command: str) -> tuple[str, dict]:
    return "bash_exec", {"command": command}


# ----------------------------------------------------------------------------------- the mark

async def test_reading_a_download_marks_the_agent_and_names_the_files(make):
    r = await make()
    await r.run(("read", {"path": str(r.ws / "notes.md")}))
    assert r.marked() is None, "the same text in a workspace file the person put there does not mark"
    await r.read("a.md")
    assert r.marked() == "a.md"
    await r.read("b.md", "c.md")
    assert r.marked() == "a.md and 2 more"


async def test_the_mark_survives_the_next_message_and_a_cleared_conversation(make):
    r = await make()
    await r.read("a.md")
    await r.run(("read", {"path": str(r.ws / "notes.md")}))
    r.loop.clear()
    assert r.marked() == "a.md"
    r.gate.reset_context_state()
    assert r.marked() is None


async def test_a_new_gate_starts_unmarked(make):
    r = await make()
    await r.read("a.md")
    fresh = PermissionGate(boundary=r.ws, workspace=r.ws, grants=r.gate.grants)
    assert fresh.taint.source(AGENT) is None


async def test_clear_resets_the_gate(make):
    r = await make()
    await r.read("a.md")
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    repl._agent, repl._gate = repl_loop(r.tmp / "repl"), r.gate
    assert await repl._handle_slash("/clear") is True
    assert r.marked() is None


# ------------------------------------------------------------------------ what asks once marked

async def test_rm_rf_build_runs_in_auto_until_the_agent_reads_a_saved_page(make):
    asker = Asker()
    r = await make("auto", asker=asker)
    await r.run(sh("rm -r build"))
    assert r.shell.ran == ["rm -r build"] and not r.gate.pending
    await r.read("a.md")
    await r.run(sh("rm -r out"))
    assert r.shell.ran == ["rm -r build"] and len(r.gate.pending) == 1
    (parked,) = r.gate.pending.values()
    assert parked.rendering.endswith(f" — {NOTE} (a.md).") and not asker.calls


async def test_python_inline_asks_once_then_always_covers_that_exact_code_until_the_state_clears(make):
    asker = Asker("allow_always")
    r = await make("guarded", asker=asker)
    await r.read("a.md")
    await r.run(sh("python -c 'print(1)'"))
    assert len(asker.calls) == 1 and r.shell.ran == ["python -c 'print(1)'"]
    assert asker.calls[0].display.endswith(f" — {NOTE} (a.md).")
    signature = classify_shell("python -c 'print(1)'", GateSettings()).segments[0].signature
    assert r.gate.grants.lookup(r.ws, "interpreter-inline", signature) is None
    assert r.gate.taint.grants(AGENT), "the answer is kept on the gate's mark"

    await r.run(sh("python -c 'print(1)'"))
    assert len(asker.calls) == 1, "the identical call ran without asking"
    await r.run(sh("python -c 'print(2)'"))
    assert len(asker.calls) == 2, "different code asks again"

    r.gate.reset_context_state()
    r.gate.set_mode("auto")
    await r.run(sh("python -c 'print(3)'"))
    assert len(asker.calls) == 2 and r.shell.ran[-1] == "python -c 'print(3)'"


async def test_a_stored_grant_does_not_cover_inline_code_while_marked(make):
    asker = Asker()
    r = await make("guarded", asker=asker)
    signature = classify_shell("python -c 'print(1)'", GateSettings()).segments[0].signature
    r.gate.grants.add(new_grant(key=signature, klass="interpreter-inline", workspace=r.ws,
                                channel="test", session_id="s"))
    await r.run(sh("python -c 'print(1)'"))
    assert not asker.calls
    await r.read("a.md")
    await r.run(sh("python -c 'print(1)'"))
    assert len(asker.calls) == 1


async def test_a_script_written_before_the_mark_runs_silently_and_one_written_after_asks(make):
    r = await make("auto", asker=Asker())
    await r.read("a.md")
    await r.run(sh("bash old.sh"))
    assert r.shell.ran == ["bash old.sh"]

    await r.run(("write", {"path": "run.sh", "content": "echo hi\n"}), ("write", {"path": "out.txt", "content": "x"}),
                sh("ls"))
    assert (r.ws / "run.sh").exists() and (r.ws / "out.txt").exists() and r.shell.ran == ["bash old.sh", "ls"]
    assert not r.gate.pending

    await r.run(sh("bash run.sh"), sh("./run.sh"))
    assert r.shell.ran == ["bash old.sh", "ls"] and len(r.gate.pending) == 2
    assert all(WRITTEN_FILE_REASON.format(path=r.ws / "run.sh") in p.rendering for p in r.gate.pending.values())


async def test_python_exec_asks_in_trusted_when_marked(make):
    asker = Asker()
    r = await make("trusted", asker=asker)
    await r.run(("python_exec", {"code": "print(1)"}))
    assert not asker.calls
    await r.read("a.md")
    await r.run(("python_exec", {"code": "print(2)"}))
    assert len(asker.calls) == 1 and r.py.ran == ["print(1)", "print(2)"]


async def test_cruncher_exec_does_not_gain_an_ask_while_marked(make):
    r = await make("trusted", asker=Asker())
    await r.read("a.md")
    cruncher = ToolMeta(gate_family="code", group="code")
    marked = GateContext(boundary=r.ws, workspace=r.ws, grants=lambda w, k, key: None, mode="trusted",
                         tainted_by="a.md")
    assert evaluate("cruncher_exec", {"code": "1"}, cruncher, marked, GateSettings()).verdict is Verdict.ALLOW
    assert evaluate("python_exec", {"code": "1"}, cruncher, marked, GateSettings()).verdict is Verdict.ASK


async def test_unattended_parks_the_severe_call_and_approve_lets_the_same_call_run(make):
    asker = Asker()
    r = await make("unattended", asker=asker)
    await r.run(sh("ls"), ("python_exec", {"code": "print(0)"}))
    assert r.py.ran == ["print(0)"] and not r.gate.pending

    await r.read("a.md")
    await r.run(("python_exec", {"code": "print(1)"}))
    assert r.py.ran == ["print(0)"] and len(r.gate.pending) == 1 and not asker.calls
    (n, parked), = r.gate.pending.items()
    assert NOTE in parked.rendering

    await r.gate.approve(n)
    await r.run(("python_exec", {"code": "print(1)"}))
    assert r.py.ran == ["print(0)", "print(1)"]


async def test_unattended_with_no_asker_refuses_the_severe_call(make):
    r = await make("unattended")
    await r.read("a.md")
    await r.run(("python_exec", {"code": "print(1)"}))
    assert r.py.ran == [] and any(NO_ASKER_REASON in e for e in r.errors())


async def test_a_write_outside_the_project_asks_even_with_a_stored_grant(make):
    asker = Asker()
    r = await make("guarded", asker=asker)
    outside = r.tmp / "outside"
    outside.mkdir()
    r.gate.grants.add(new_grant(key=str(outside), klass="edit-outside", workspace=r.ws, channel="test",
                                session_id="s"))
    await r.run(("write", {"path": str(outside / "f.txt"), "content": "x"}))
    assert not asker.calls and (outside / "f.txt").exists()
    await r.read("a.md")
    await r.run(("write", {"path": str(outside / "g.txt"), "content": "x"}))
    assert len(asker.calls) == 1


async def test_writing_inside_the_downloads_folder_is_refused_with_the_save_to_reason(make):
    r = await make("auto", asker=Asker())
    await r.run(("write", {"path": f"{SAVED}/z.md", "content": "x"}))
    assert any(DOWNLOADS_WRITE_REASON in e for e in r.errors())


# ------------------------------------------------------------------------------ the subagent

async def test_a_subagent_started_while_the_parent_is_marked_starts_marked_and_its_entry_ends_with_its_run(make):
    r = await make("auto", asker=Asker())
    seen: dict[str, Any] = {}

    class Probe(Tool):
        def info(self) -> ToolSchema:
            return ToolSchema(name="probe", description="p", parameters={"type": "object", "properties": {}},
                              group="other", gate_family="allow", ingest="none", host="safe")

        async def _execute(self, **kwargs: Any) -> ToolResult:
            seen["during"] = r.gate.taint.source("child")
            seen["context"] = r.gate.context(agent_id="child").tainted_by
            return self.ok("p")

    child_registry = ToolRegistry()
    await child_registry.register(Probe(), scope="global")

    class Delegate(Tool):
        def info(self) -> ToolSchema:
            return ToolSchema(name="delegate_stub", description="d", parameters={"type": "object", "properties": {}},
                              group="other", gate_family="allow", ingest="none", host="safe")

        async def _execute(self, **kwargs: Any) -> ToolResult:
            plan = [FakeLLMResponse(content=None, tool_calls=[FakeToolCall(id="c1", name="probe", arguments={})]),
                    FakeLLMResponse(content="child done")]
            child = AgentLoop(config=AgentConfig(name="child", role="Child."), llm=MockLLMClient(plan), bus=r.bus,
                              context_manager=ContextManager(), tool_registry=child_registry,
                              permission_evaluator=PermissionEvaluator(), gate=r.gate)
            await child.run_turn("go")
            seen["after"] = r.gate.taint.source("child")
            return self.ok("delegated")

    await r.registry.register(Delegate(), scope="global")
    await r.read("a.md")
    await r.run(("delegate_stub", {}))
    assert seen == {"during": "a.md", "context": "a.md", "after": None}
    assert r.marked() == "a.md"


async def test_a_subagent_of_an_unmarked_parent_is_not_marked(make):
    r = await make("auto", asker=Asker())
    r.gate.taint.inherit(AGENT, "child")
    assert r.gate.taint.source("child") is None


# ------------------------------------------------------------------------------- the words

def test_the_taint_object_counts_distinct_sources():
    taint = Taint()
    assert taint.note_read("a", "read", [Path("/p/.localharness/downloads/x.md")])
    assert not taint.note_read("a", "read", [Path("/p/notes.md")])
    taint.note_read("a", "read", [Path("/p/.localharness/downloads/x.md")])
    assert taint.source("a") == "x.md"
    taint.grant("a", "k")
    taint.inherit("a", "b")
    assert taint.source("b") == "x.md" and taint.grants("b") == frozenset()
    taint.end("a")
    assert taint.source("a") is None and taint.grants("a") == frozenset()


def test_no_ask_refusal_or_reason_says_taint(tmp_path):
    proj = tmp_path.resolve()
    (proj / SAVED).mkdir(parents=True)
    ctx = GateContext(boundary=proj, workspace=proj, grants=lambda w, k, key: None, mode="guarded",
                      has_review_surface=True, tainted_by="a.md and 2 more")
    shown = [DOWNLOADS_WRITE_REASON, DOWNLOAD_EXEC_DETAIL, SAVED_PAGES_NOTE, WRITTEN_FILE_REASON, NOTE]
    for command in ("rm -rf build", "python -c 'print(1)'", f"bash {SAVED}/x.sh", "ls"):
        result = evaluate("bash_exec", {"command": command}, ToolMeta(group="shell", destructive=True), ctx, GateSettings())
        shown += [result.reason, result.request.display if result.request else ""]
    result = evaluate("python_exec", {"code": "1"}, ToolMeta(gate_family="code"), dataclasses.replace(ctx), GateSettings())
    shown += [result.reason, result.request.display]
    assert not [text for text in shown if "taint" in text.lower()]
