"""The person's turns carry the typed or steering subtype and a turn number; every other caller is request."""
import json
from types import SimpleNamespace

import pytest

from localharness.agent.context import COMPACT_DISABLED, ContextManager
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.cli.session_resume import SLEEP_ACTION, Resume, asleep_path, take_asleep, write_asleep
from localharness.config.models import AgentConfig
from localharness.core.bus import EventBus
from localharness.core.types import human_message, provider_messages
from tests.unit.test_task_context import make_loop


def _humans(loop):
    return [m for m in loop.resume_state()[0] if m.get("_lh", {}).get("origin") == "human"]


@pytest.mark.asyncio
async def test_typed_and_default_turns_are_stamped_and_numbered(tmp_path, bus, mock_llm_client):
    llm = mock_llm_client([mock_llm_client.Response(content="one"), mock_llm_client.Response(content="two")])
    loop = make_loop(llm, bus, tmp_path)
    await loop.run_turn("hello", subtype="typed")
    await loop.run_turn("again")
    first, second = _humans(loop)
    assert first["_lh"] == {"origin": "human", "subtype": "typed", "turn": 1}
    assert second["_lh"] == {"origin": "human", "subtype": "request", "turn": 2}
    assert loop.turn == 2
    wire = provider_messages(loop.resume_state()[0])
    assert not any("_lh" in m for m in wire)
    assert [m["content"] for m in wire if m["role"] == "user"] == ["hello", "again"]


def test_human_message_stamps_turn_only_when_given():
    assert "turn" not in human_message("x")["_lh"]
    assert human_message("x", "typed", 3)["_lh"]["turn"] == 3


class _Agent:
    def __init__(self):
        self.calls = []
        self._config = SimpleNamespace(name="orchestrator", context=SimpleNamespace(max_image_tokens=0))
        self.current_session_id = "s1"

    async def run_turn(self, task, on_token=None, **kw):
        self.calls.append((task, kw))
        return "ok"


def _repl_with_agent():
    from localharness.cli.repl import OrchestratorREPL

    repl = OrchestratorREPL.__new__(OrchestratorREPL)
    repl._agent = _Agent()
    repl._channel = SimpleNamespace(channel_id="terminal")
    repl._bus = EventBus()
    repl._staged_images = []
    repl._skills = None
    repl._slash_followup = None
    repl._orchestrator = SimpleNamespace(active_workflow=None)
    repl._detect_creation_intent = lambda text: False
    return repl


async def _run(repl, coro):
    task = await coro
    await task
    return repl._agent.calls[-1]


@pytest.mark.asyncio
async def test_dispatch_input_runs_the_turn_typed_and_start_user_turn_defaults_to_request():
    repl = _repl_with_agent()
    assert await _run(repl, repl._dispatch_input("hello")) == ("hello", {"subtype": "typed"})
    assert await _run(repl, repl._start_user_turn("x")) == ("x", {"subtype": "request"})


@pytest.mark.asyncio
async def test_nudges_drain_fifo_keeping_their_subtype_and_the_turn_number(tmp_path, bus, mock_llm_client):
    loop = make_loop(mock_llm_client([mock_llm_client.Response(content="done")]), bus, tmp_path)
    loop.push_user_nudge("fix the title", "steering")
    loop.push_user_nudge("approved")
    loop.push_user_nudge("   ")
    assert len(loop._user_nudge_inbox) == 2
    await loop.run_turn("go", subtype="typed")
    stamped = [(m["content"], m["_lh"]["subtype"], m["_lh"]["turn"]) for m in _humans(loop)]
    assert stamped == [("go", "typed", 1), ("fix the title", "steering", 1), ("approved", "request", 1)]


@pytest.mark.asyncio
async def test_a_cancelled_turn_keeps_its_number_and_the_next_takes_the_next(tmp_path, bus, mock_llm_client):
    import asyncio

    started = asyncio.Event()
    llm = mock_llm_client([mock_llm_client.Response(content="after")])
    scripted = llm.stream_complete

    async def hang_once(*args, **kw):
        if not started.is_set():
            started.set()
            await asyncio.Event().wait()
        return await scripted(*args, **kw)

    llm.stream_complete = hang_once
    loop = make_loop(llm, bus, tmp_path)
    turn = asyncio.ensure_future(loop.run_turn("first"))
    await started.wait()
    turn.cancel()
    await asyncio.wait([turn])
    assert loop.turn == 1
    await loop.run_turn("second")
    assert [m["_lh"]["turn"] for m in _humans(loop)] == [2]
    assert loop.turn == 2


def _steering_repl(agent):
    from localharness.cli.repl import OrchestratorREPL

    class _Channel:
        channel_id = "terminal"

        def __init__(self):
            self.sent = []

        async def send_message(self, text, agent_id=None, metadata=None):
            self.sent.append(text)

        async def box_echo_prompt(self, *a, **kw):
            pass

        def box_flash_decision(self, *a):
            pass

    repl = OrchestratorREPL.__new__(OrchestratorREPL)
    repl._agent = agent
    repl._channel = _Channel()
    repl._bus = EventBus()
    repl._staged_images = []
    repl._skills = None
    repl._slash_followup = None
    return repl


class _Nudges(_Agent):
    def __init__(self):
        super().__init__()
        self.nudges = []

    def push_user_nudge(self, text, subtype="request"):
        self.nudges.append((text, subtype))


async def _running():
    import asyncio

    return asyncio.ensure_future(asyncio.sleep(3600))


@pytest.mark.asyncio
async def test_only_the_nudge_button_and_the_router_steer():
    from localharness.channels import input_router

    agent = _Nudges()
    repl = _steering_repl(agent)
    repl._turn_task = await _running()
    try:
        assert await repl._nudge_from_channel(" use ripgrep ") is True
        await repl._deliver_route("narrow it", input_router.Decision(input_router.Route.NUDGE, "force", "force-bang"))
    finally:
        repl._turn_task.cancel()
    assert agent.nudges == [("use ripgrep", "steering"), ("narrow it", "steering")]


@pytest.mark.asyncio
async def test_an_approval_a_denial_and_a_skill_expansion_nudge_as_request(tmp_path):
    from tests.unit.test_channel_permission_asks import _BoxChannel, _gate, _repl, _stage_one

    gate = _gate(tmp_path)
    agent = _Nudges()
    repl = _repl(_BoxChannel(), gate)
    repl._agent = agent
    repl._turn_task = await _running()
    try:
        await _stage_one(gate)
        await repl._handle_pending_answer("", approve=True)
        await _stage_one(gate, "rm -rf ~/older-notes")
        await repl._handle_pending_answer("", approve=False)
        repl._skills = SimpleNamespace(find=lambda name: name, invoke=lambda skill, args: ("expanded", []),
                                       arm=lambda *a: None)
        await repl._slash_skill(SimpleNamespace(name="/notes"), "draft")
    finally:
        repl._turn_task.cancel()
    assert [subtype for _, subtype in agent.nudges] == ["request"] * 3
    assert agent.nudges[-1][0] == "expanded"


@pytest.mark.asyncio
async def test_an_idle_approve_runs_its_followup_turn_as_request(tmp_path):
    from tests.unit.test_channel_permission_asks import _BoxChannel, _gate, _repl, _stage_one

    gate = _gate(tmp_path)
    repl = _repl(_BoxChannel(), gate)
    repl._agent = _Nudges()
    await _stage_one(gate)
    task = await repl._dispatch_input("/approve")
    await task
    (sentence, kw), = repl._agent.calls
    assert "approved pending #1" in sentence and kw == {"subtype": "request"}


@pytest.mark.asyncio
async def test_a_followup_carries_its_own_subtype():
    repl = _repl_with_agent()

    async def handled(line):
        repl._slash_followup = ("do it", "typed")
        return True

    repl._handle_slash = handled
    task = await repl._dispatch_input("/anything")
    await task
    assert repl._agent.calls == [("do it", {"subtype": "typed"})]
    assert repl._slash_followup is None


# modules that only read a subtype, never stamp one
READERS = {"orchestrator/task_flow/panel.py"}


def _count_in_source(literal):
    import re
    from pathlib import Path

    root = Path(__import__("localharness").__file__).parent
    found = {}
    for path in root.rglob("*.py"):
        if path.relative_to(root).as_posix() in READERS:
            continue
        n = len(re.findall(re.escape(literal), path.read_text()))
        if n:
            found[path.relative_to(root).as_posix()] = n
    return found


def test_steering_is_spelled_at_exactly_the_two_nudge_sites():
    assert _count_in_source('"steering"') == {"cli/repl.py": 2}


@pytest.mark.asyncio
async def test_acp_prompt_runs_typed_and_an_idle_approve_follow_up_runs_request(tmp_path, monkeypatch):
    from tests.unit.test_instructions_channels import _block, _handle, _prompt_channel

    channel, _ = _prompt_channel(tmp_path, _handle(tmp_path))

    async def approved(verb, arg):
        return "Human approved pending #1."

    monkeypatch.setattr(channel, "_run_pending_command", approved)
    await channel.prompt("sess", _block("hello"))
    await channel.prompt("sess", _block("/approve 1"))
    assert channel._agent_loop.turns == ["hello", "Human approved pending #1."]
    assert channel._agent_loop.subtypes == ["typed", "request"]


def test_typed_is_spelled_at_the_repl_input_path_and_the_acp_prompt_only():
    assert _count_in_source('"typed"') == {"cli/repl.py": 1, "channels/acp.py": 1}


def _slept(tmp_path, turn=None, conversation=()):
    path = asleep_path(tmp_path / "sessions")
    fields = dict(action=SLEEP_ACTION, agent_name="a", conversation=tuple(conversation), prior_context="",
                  eviction_store=None, queued=(), gate_mode="unattended", previous_sitting_id="s")
    write_asleep(path, Resume(**fields, **({} if turn is None else {"turn": turn})), workspace="/w")
    return path


def test_the_counter_crosses_a_sleep_and_an_old_file_reads_zero(tmp_path):
    assert take_asleep(_slept(tmp_path, turn=7), workspace="/w").turn == 7
    path = _slept(tmp_path)
    record = json.loads(path.read_text())
    del record["turn"]
    path.write_text(json.dumps(record))
    assert take_asleep(path, workspace="/w").turn == 0


@pytest.mark.asyncio
async def test_a_woken_loop_stamps_the_counter_plus_one(tmp_path, bus, mock_llm_client):
    cfg = AgentConfig.model_validate({"name": "w", "role": "r", "permissions": {"mode": "unattended", "deny_patterns": []}})
    loop = AgentLoop(cfg, mock_llm_client([mock_llm_client.Response(content="ok")]), bus,
                     ContextManager(max_context_tokens=32768), None, PermissionEvaluator(),
                     compact_md_path=COMPACT_DISABLED, config_dir=tmp_path, turn=7)
    await loop.run_turn("hi")
    assert _humans(loop)[0]["_lh"]["turn"] == 8 and loop.turn == 8


@pytest.mark.parametrize("planted,kept", [
    (True, 0), ("5", 0), (5.0, 0), (-1, 0), (1_000_001, 0), (None, 0), (1_000_000, 1_000_000), (7, 7)])
def test_the_wake_clamps_a_planted_counter(tmp_path, planted, kept):
    path = _slept(tmp_path, turn=1)
    record = json.loads(path.read_text())
    record["turn"] = planted
    path.write_text(json.dumps(record))
    assert take_asleep(path, workspace="/w").turn == kept


def _answer(text="done"):
    return {"role": "assistant", "content": text, "tool_calls": None}


def test_the_wake_restamps_every_human_origin_message_whatever_its_role(tmp_path):
    def stamped(role, subtype, **extra):
        return {"role": role, "content": "x", "_lh": {"origin": "human", "subtype": subtype, "turn": 3, **extra}}

    harness = {"role": "user", "content": "n", "_lh": {"origin": "harness", "subtype": "act_guard"}}
    plain = {"role": "user", "content": "no stamp"}
    conversation = (stamped("user", "typed"), _answer(), stamped("user", "steering", extra="k"), _answer("b"),
                    harness, plain, stamped("assistant", "typed"))
    woken = take_asleep(_slept(tmp_path, conversation=conversation), workspace="/w").conversation
    request = {"origin": "human", "subtype": "request"}
    assert [m.get("_lh") for m in woken] == [request, None, request, None, harness["_lh"], None, request]
    assert woken[4] == harness and woken[5] == plain and woken[1] == _answer()
    assert [m["content"] for m in woken] == [m["content"] for m in conversation]


def test_the_wake_leaves_the_in_memory_conversation_alone(tmp_path):
    typed = {"role": "user", "content": "x", "_lh": {"origin": "human", "subtype": "typed", "turn": 3}}
    conversation = (typed, _answer())
    take_asleep(_slept(tmp_path, conversation=conversation), workspace="/w")
    assert conversation[0]["_lh"] == {"origin": "human", "subtype": "typed", "turn": 3}


def test_a_sleeping_sitting_is_read_from_a_file_holding_stamped_messages(tmp_path):
    from localharness.cli.session_resume import sleeping_sitting

    typed = {"role": "user", "content": "x", "_lh": {"origin": "human", "subtype": "typed", "turn": 3}}
    assert sleeping_sitting(_slept(tmp_path, conversation=(typed, _answer()))) == "s"
