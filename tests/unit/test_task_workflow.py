"""Composed acceptance: start -> human correction -> aside -> continue -> checkpoint -> restart.

A scripted model drives the real loop; `task` dispatches to a real TaskTool and `write` writes a
real file, so every claim below comes from the actual request packing and the file on disk."""
import os
import stat

from localharness.agent.task_record import TaskState
from localharness.tools.base import ToolResult
from localharness.tools.builtin.task_tool import TaskTool
from tests.unit.test_task_context import make_loop

ASK = ("Help me prepare a short report on tide pools. For now write only the outline in "
       "outline.md and stop after the outline for my review.")
THREE = "# Tide pools\n1. Zones\n2. Species\n3. Threats\n"
FIVE = "# Tide pools\n1. Zones\n2. Species\n3. Adaptations\n4. Threats\n5. Visiting\n"


class Registry:
    def __init__(self, state, root):
        self.tool, self.root = TaskTool(state), root

    def get_tools_for_agent(self, *args):
        return {}

    async def dispatch(self, name, args, *rest):
        if name == "task":
            return await self.tool.run(**args)
        target = self.root / args["path"]
        target.write_text(args["content"])
        return ToolResult(output="written", metadata={"path": str(target)})


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


async def test_correction_aside_checkpoint_and_restart_from_disk(tmp_path, bus, mock_llm_client):
    path = tmp_path / "agents" / "t" / "task.json"
    state = TaskState(path, workspace=str(tmp_path))
    outline = tmp_path / "outline.md"
    llm = scripted(
        mock_llm_client,
        ("task", {"action": "start", "objective": "Short report on tide pools",
                  "assignment": "Outline in outline.md", "stop_boundary": "outline only",
                  "requested_status": "checkpoint", "human_quote": "stop after the outline for my review"}),
        ("write", {"path": "outline.md", "content": THREE}),
        ("task", {"action": "artifact", "key": "outline", "path": "outline.md"}),
        "Outline drafted.",
        # turn 2: the human's correction, plus one decision the human never said
        ("task", {"action": "decide", "text": "Outline has five sections",
                  "human_quote": "the outline must have five sections"}),
        ("task", {"action": "decide", "text": "Use a playful tone"}),
        "Noted.",
        "4",  # turn 3: the aside
        ("write", {"path": "outline.md", "content": FIVE}),
        ("task", {"action": "close", "status": "checkpoint",
                  "human_quote": "stop after the outline for my review"}),
        "Five-section outline ready for your review.",
    )
    seen = capture(llm)
    loop = make_loop(llm, bus, tmp_path, state, Registry(state, tmp_path))

    assert await loop.run_turn(ASK) == "Outline drafted."
    assert not packet_of(seen[0])  # no record yet on the first request
    assert await loop.run_turn("Correction: the outline must have five sections, not three.") == "Noted."
    mark = len(seen)
    assert await loop.run_turn("Quick aside: what is 2 + 2?") == "4"
    aside = packet_of(seen[mark])
    assert "Short report on tide pools" in aside and "[human] Outline has five sections" in aside
    assert "[assumption] Use a playful tone" in aside
    answer = await loop.run_turn("OK, continue.")
    assert answer == "Five-section outline ready for your review."  # a checkpoint is not a failure
    assert state.status == "checkpoint"
    assert outline.read_text() == FIVE
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    decisions = list(state.current.decisions)
    assert [d.origin for d in decisions] == ["human", "model"]

    # Restart: a new loop and model, the record read back from disk.
    reloaded, notice = TaskState.load(path, workspace=str(tmp_path))
    assert notice is None and reloaded.notes == []  # outline untouched since the last save
    llm2 = scripted(mock_llm_client, "Next is the draft, when you say so.")
    seen2 = capture(llm2)
    loop2 = make_loop(llm2, bus, tmp_path, reloaded, Registry(reloaded, tmp_path))
    assert await loop2.run_turn("Looks good. What is next?") == "Next is the draft, when you say so."
    packet = packet_of(seen2[0])
    assert "Outline in outline.md" in packet
    assert "[human] Outline has five sections" in packet
    assert "Requested stopping status: checkpoint" in packet
    assert "changed since last session" not in packet
    assert reloaded.current.decisions == decisions  # no duplicated decisions
    assert outline.read_text() == FIVE  # no duplicate drafting

    outline.write_text(FIVE + "6. Extra\n")
    changed, _ = TaskState.load(path, workspace=str(tmp_path))
    assert "outline changed since last session" in changed.notes
    assert "outline changed since last session" in changed.packet()

    elsewhere, notice = TaskState.load(path, workspace=str(tmp_path / "other"))
    assert elsewhere.current is None and notice is None and path.exists()
