"""Composed: declared references reach the actual request through eviction, change, overflow and
restart; checks report the artifact revisions they ran against.

A scripted model drives the real AgentLoop and the real ContextManager packing; `task` dispatches
to a real TaskTool. Every assertion about references inspects the messages actually handed to
the model (captured at `stream_complete`)."""
import json

from localharness.agent import context as context_mod
from localharness.agent.context import ContentStore, ContextManager
from localharness.agent.task_record import TaskState
from localharness.tools.base import ToolResult
from localharness.tools.builtin.task_tool import TaskTool
from tests.unit.test_task_context import make_loop

INSTRUCTIONS = "Instructions: write a short note on tide pools for a general reader. " * 5
VOICE = "Voice: short declarative sentences, no adverbs, concrete nouns first. " * 5
SOURCE = "Source: barnacles close their plates at low tide to keep moisture in. " * 5
NEW_SOURCE = "Source (revised): anemones retract tentacles when exposed to air. " * 5


class Registry:
    def __init__(self, state, root, codes=()):
        self.tool, self.root, self.codes = TaskTool(state), root, iter(codes)

    def get_tools_for_agent(self, *args):
        return {}

    async def dispatch(self, name, args, *rest):
        if name == "task":
            return await self.tool.run(**args)
        if name == "read":
            return ToolResult(output=(self.root / args["path"]).read_text())
        if name == "write":
            (self.root / args["path"]).write_text(args["content"])
            return ToolResult(output="written", metadata={"path": args["path"]})
        if name == "bash_exec":
            return ToolResult(output="lint ran", metadata={"exit_code": next(self.codes)})
        return ToolResult(output="filler " * 500)


def scripted(mock_llm_client, *steps):
    R, C = mock_llm_client.Response, mock_llm_client.ToolCall
    return mock_llm_client([
        R(content=step) if isinstance(step, str)
        else R(content=None, tool_calls=[C(id=f"c{i}", name=step[0], arguments=step[1])])
        for i, step in enumerate(steps)
    ])


def capture(llm):
    seen = []
    original = llm.stream_complete

    async def wrapped(messages, *a, **kw):
        seen.append(list(messages))
        return await original(messages, *a, **kw)
    llm.stream_complete = wrapped
    return seen


def packet_of(messages):
    found = [m["content"] for m in messages if (m.get("_lh") or {}).get("subtype") == "active_task"]
    assert len(found) <= 1
    return found[0] if found else ""


def restored(messages, handle):
    """Bodies carried as tool data for a tool_result_get call naming `handle`."""
    ids = {c["id"] for m in messages if m.get("role") == "assistant" for c in m.get("tool_calls") or []
           if isinstance(c, dict) and c["function"]["name"] == "tool_result_get"
           and json.loads(c["function"]["arguments"])["id"] == handle}
    return [m["content"] for m in messages if m.get("role") == "tool" and m.get("tool_call_id") in ids]


def setup(tmp_path, **files):
    for name, body in files.items():
        (tmp_path / f"{name}.md").write_text(body)
    return TaskState(tmp_path / "agents" / "t" / "task.json", workspace=str(tmp_path))


START = ("task", {"action": "start", "objective": "Tide pool note", "assignment": "Draft the note"})


def ref(name):
    return ("task", {"action": "reference", "source": name, "path": f"{name}.md"})


async def test_a_references_survive_forced_eviction(tmp_path, bus, mock_llm_client, monkeypatch):
    monkeypatch.setattr(context_mod, "TOOL_EVICT_USAGE_FRACTION", 0.0)
    state = setup(tmp_path, instructions=INSTRUCTIONS, voice=VOICE, source=SOURCE)
    store = ContentStore()
    ctx = ContextManager(max_context_tokens=32768, eviction_store=store, tool_evict_threshold_chars=100)
    llm = scripted(mock_llm_client, START, *(ref(n) for n in ("instructions", "voice", "source")),
                   *(("read", {"path": f"{n}.md"}) for n in ("instructions", "voice", "source")),
                   *(("noop", {"n": i}) for i in range(4)), "Draft outline ready.")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path), ctx=ctx)
    assert await loop.run_turn("Draft the tide pool note from my files.") == "Draft outline ready."
    last = seen[-1]
    assert any("tool result evicted" in (m.get("content") or "") for m in last)
    assert last == ctx.repair_tool_pairing(last)
    for r, body in zip(state.current.references, (INSTRUCTIONS, VOICE, SOURCE)):
        assert restored(last, r.handle) == [body], r.source
    assert "References: instructions: current; voice: current; source: current" in packet_of(last)


async def test_b_overflow_blocks_once_with_the_narrowing_notice(tmp_path, bus, mock_llm_client):
    big = " ".join(f"word{i % 997}" for i in range(150 * 1024 // 8))
    assert 140 * 1024 < len(big) < 200 * 1024
    state = setup(tmp_path, big=big)
    llm = scripted(mock_llm_client, START, ref("big"), "never sent")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))
    out = await loop.run_turn("Use big.md as the reference for the note.")
    assert out.startswith("Active step blocked:") and "split or narrow" in out
    assert len(seen) == 2 and not packet_of(seen[0])  # no model call after the declaration
    assert state.status == "blocked"


async def test_c_changed_source_reaches_the_next_request(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, source=SOURCE)
    llm = scripted(mock_llm_client, START, ref("source"),
                   ("write", {"path": "source.md", "content": NEW_SOURCE}), ("noop", {}), "Done.")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))
    await loop.run_turn("Draft from source.md.")
    before, after, later = seen[2], seen[3], seen[4]
    assert "source: current" in packet_of(before)
    assert any(m.get("content") == SOURCE for m in before if m.get("role") == "tool")
    handle = state.current.references[0].handle
    assert restored(after, handle) == [NEW_SOURCE]
    assert not any(SOURCE in (m.get("content") or "") for m in after)
    assert "source: refreshed (changed)" in packet_of(after)
    assert "source: current" in packet_of(later)


async def test_d_restart_resnapshots_path_references(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, instructions=INSTRUCTIONS, voice=VOICE, draft="Draft v1")
    ctx = ContextManager(max_context_tokens=32768)
    pasted = ctx._content_store.put("Pasted notes from the human about tone.")
    llm = scripted(mock_llm_client, START, ref("instructions"), ref("voice"),
                   ("task", {"action": "reference", "source": "pasted", "handle": pasted}),
                   ("task", {"action": "decide", "text": "Plain tone"}), "Ready.")
    await make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path), ctx=ctx).run_turn("Start the note.")
    assert all(r.handle for r in state.current.references)
    (tmp_path / "voice.md").write_text(VOICE + "Prefer the active voice.")

    reloaded, notice = TaskState.load(state.path, workspace=str(tmp_path))
    assert notice is None and [r.handle for r in reloaded.current.references] == [None] * 3
    llm2 = scripted(mock_llm_client, "Picking up where we left off.")
    seen = capture(llm2)
    loop2 = make_loop(llm2, bus, tmp_path, reloaded, Registry(reloaded, tmp_path))
    assert await loop2.run_turn("Continue.") == "Picking up where we left off."
    instructions, voice, hand = reloaded.current.references
    assert restored(seen[0], instructions.handle) == [INSTRUCTIONS]
    assert restored(seen[0], voice.handle) == [VOICE + "Prefer the active voice."]
    packet = packet_of(seen[0])
    assert "pasted: unavailable; read it again" in packet and hand.handle is None
    assert "voice: refreshed (changed)" in packet and "reference voice changed since last session" in packet
    assert (tmp_path / "draft.md").read_text() == "Draft v1"
    assert [d.text for d in reloaded.current.decisions] == ["Plain tone"]


LINT = {"command": "python checks/lint.py draft.md"}
CHECK = ("task", {"action": "check", "key": "lint", "description": "Lint passes", "tool": "bash_exec",
                  "arguments": LINT, "result_field": "exit_code", "expected": 0, "depends_on": ["draft"]})
DRAFT = ("task", {"action": "artifact", "key": "draft", "path": "draft.md"})
RUN_LINT = ("bash_exec", {**LINT, "timeout": 30})


async def test_e_checks_report_draft_revisions(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, draft="Barnacles are amazing.")
    llm = scripted(mock_llm_client, START, DRAFT, CHECK, RUN_LINT,
                   ("write", {"path": "draft.md", "content": "Barnacles close at low tide."}),
                   RUN_LINT, ("write", {"path": "draft.md", "content": "Barnacles shut at low tide."}),
                   "Done.")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path, codes=(1, 0)))
    out = await loop.run_turn("Draft, then lint with python checks/lint.py draft.md.")
    failed, stale, passed, later = (packet_of(seen[i]) for i in (4, 5, 6, 7))
    assert "lint: failed — Lint passes" in failed and "Artifacts: draft: rev 1" in failed
    assert "lint: failed (ran at draft rev 1, draft now rev 2)" in stale
    assert "lint: passed — Lint passes" in passed and "Artifacts: draft: rev 2" in passed
    assert "lint: passed (ran at draft rev 2, draft now rev 3)" in later
    assert all("Revision budget" not in p for p in (failed, stale, passed, later))
    assert out == "Done.\n\nTask evidence: lint: passed (ran at draft rev 2, draft now rev 3)."


async def test_f_close_complete_stays_live_until_evidence_settles(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, draft="Draft")
    human = "Skip the lint gate for this draft, I accept it as is."
    close = ("task", {"action": "close", "status": "complete"})
    waive = {"action": "waive", "human_turn": 1}
    llm = scripted(mock_llm_client, START, DRAFT, CHECK, RUN_LINT, close,
                   ("task", {"action": "judge", "key": "source-support", "criterion": "Claims cite sources"}),
                   ("task", {**waive, "key": "source-support"}), ("task", {**waive, "key": "lint"}),
                   "Closed.")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path, codes=(1,)))
    out = await loop.run_turn(human)
    closing = [m.get("content") or "" for m in seen[5] if m.get("role") == "tool"][-1]
    assert "Closing task" in closing and "stays live until its declared evidence is settled" in closing
    assert "lint: failed" in packet_of(seen[5])  # closed, not settled: the packet is still sent
    assert state.current.closed and state.current.retired
    assert out == "Closed."  # every check waived by turn 1: settled, retired, words unchanged

    (tmp_path / "x").mkdir()
    other = setup(tmp_path / "x")
    other.observe_human(human)
    tool = TaskTool(other)
    await tool.run(action="start", objective="o", assignment="a")
    assert (await tool.run(action="close", status="partial")).success
    assert not other.current.closed and "Requested stopping status: partial" in other.packet()


async def test_g_open_judgment_stays_in_the_packet_not_the_reply(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, draft="Draft")
    llm = scripted(mock_llm_client, START, DRAFT, CHECK,
                   ("task", {"action": "judge", "key": "source-support", "criterion": "Claims cite sources"}),
                   RUN_LINT, "Draft done.")
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path, codes=(0,)))
    out = await loop.run_turn("Draft and lint it.")
    assert out == "Draft done."  # judgments are opinion, not evidence: no evidence line
    assert "source-support: open — Claims cite sources" in state.packet()
    assert state.current.context.outcomes() == {"lint": "passed"}


async def test_unchanged_reference_is_redeclared_after_the_next_human_turn(tmp_path, bus, mock_llm_client):
    state = setup(tmp_path, voice=VOICE)
    llm = scripted(mock_llm_client, START, ref("voice"), "Noted.", "Second turn answer.")
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))
    await loop.run_turn("Use voice.md for the note.")
    await loop.run_turn("Now continue the note.")  # the turn reset cleared the protection
    handle = state.current.references[0].handle
    assert restored(seen[2], handle) == [VOICE] and restored(seen[3], handle) == [VOICE]
    assert "voice: current" in packet_of(seen[3])
