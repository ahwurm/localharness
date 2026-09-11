"""The ACP adapter, driven over the real protocol (PRD §4).

A green test on the adapter's own methods would prove nothing: the failure mode here is a Zed
panel that shows nothing, and that lives in the WIRE, not in a Python call. So every test below
speaks ACP — the SDK's own client connected to the SDK's own agent connection over a socket pair
— and asserts on what a client actually received: the handshake, the chunks, the tool-call rows,
the permission dialog and its options, the stop reason.

The harness session those turns run on is built by a stand-in for `cli/start_cmd._start_async`
(`_fake_start_async`), because the real one brings up a model server. It builds the SAME objects
that function builds and hands them to the same `serve()` entry point, so everything from the
gate down is real: a real `AgentLoop`, a real `PermissionGate` with a real `GrantStore` on disk,
real builtin tools. What it does NOT prove is that `_start_async` itself reaches `serve()` —
`test_subprocess_agent_answers_initialize_and_new_session` proves the command starts and speaks
the protocol, and the rest of that path is exercised by the live smoke in the phase report.
"""
from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from acp.agent.connection import AgentSideConnection
from acp.core import connect_to_agent
from acp.helpers import text_block
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    DeniedOutcome,
    ReadTextFileResponse,
    RequestPermissionResponse,
    ToolCallProgress,
    ToolCallStart,
)

from localharness.agent.context import ContextManager
from localharness.agent.gate import PermissionGate
from localharness.agent.loop import AgentLoop
from localharness.agent.permissions import PermissionEvaluator
from localharness.channels.acp import AcpChannel
from localharness.config.grants import GrantStore
from localharness.config.models import AgentConfig
from localharness.core.bus import EventBus
from localharness.tools import Tool, ToolRegistry, ToolResult, ToolSchema
from tests.conftest import FakeLLMResponse, FakeToolCall, MockLLMClient

pytestmark = pytest.mark.asyncio


# ------------------------------------------------------------------ fakes


class StreamingLLM(MockLLMClient):
    """The suite's fake client, plus the one behaviour the adapter depends on: it calls
    `on_token`. Production streams token by token (`provider/client.py:1401`); one call with the
    whole content is enough to prove the callback reaches `agent_message_chunk`."""

    def __init__(self, responses: list[FakeLLMResponse]) -> None:
        super().__init__(responses)
        self.seen_messages: list[list[dict]] = []
        self.streams = True
        """Set False to stand in for a runtime that returns a completion with no deltas."""

    async def stream_complete(self, messages=None, tools=None, on_token=None, **kwargs):
        self.seen_messages.append(list(messages or []))
        result = await super().stream_complete(messages, tools, on_token=on_token, **kwargs)
        message = result[0] if isinstance(result, tuple) else result
        content = getattr(message, "content", None)
        if self.streams and on_token is not None and content:
            await on_token(content)
        return result

    def model_saw(self, needle: str) -> bool:
        """Did the MODEL see this text? The denied-observation path is only real if it did."""
        return any(
            needle in str(m.get("content") or "")
            for turn in self.seen_messages
            for m in turn
        )


class FakeClient:
    """The editor side: records every notification, answers every request from a script."""

    def __init__(self, *, answers: list[str] | None = None, files: dict[str, str] | None = None):
        self.updates: list[Any] = []
        self.permission_requests: list[Any] = []
        self.answers = list(answers or [])
        self.files = dict(files or {})
        self.written: dict[str, str] = {}

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(update)

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any
    ) -> RequestPermissionResponse:
        self.permission_requests.append((tool_call, options))
        answer = self.answers.pop(0) if self.answers else "reject_once"
        if answer == "cancelled":
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id=answer))

    async def read_text_file(
        self, session_id: str, path: str, line: int | None = None, limit: int | None = None, **kw: Any
    ) -> ReadTextFileResponse:
        return ReadTextFileResponse(content=self.files[path])

    async def write_text_file(self, session_id: str, path: str, content: str, **kw: Any) -> None:
        self.written[path] = content
        self.files[path] = content
        return None

    def chunks(self) -> list[str]:
        return [
            u.content.text
            for u in self.updates
            if isinstance(u, AgentMessageChunk) and hasattr(u.content, "text")
        ]

    def tool_starts(self) -> list[ToolCallStart]:
        return [u for u in self.updates if isinstance(u, ToolCallStart)]

    def tool_updates(self) -> list[ToolCallProgress]:
        return [u for u in self.updates if isinstance(u, ToolCallProgress)]


class Shell(Tool):
    """Stands in for `bash_exec` — same name and parameter, so the verdict classifies a command
    exactly as it classifies the real one. Records what it was actually asked to run."""

    timeout_s: float = 11.0

    def __init__(self) -> None:
        super().__init__()
        self.ran: list[str] = []

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="bash_exec",
            description="Run a shell command.",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            group="shell",
            destructive=True,
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        self.ran.append(kwargs.get("command", ""))
        return self.ok("ran")


# ------------------------------------------------------------------ harness


def _plan(*calls: tuple[str, dict]) -> list[FakeLLMResponse]:
    """One tool call per step, then a final answer."""
    return [
        FakeLLMResponse(
            content=None,
            tool_calls=[FakeToolCall(id=f"tc-{i}", name=name, arguments=args)],
        )
        for i, (name, args) in enumerate(calls)
    ] + [FakeLLMResponse(content="Done.")]


def _request(**fields: Any):
    """A `PermissionRequest`, tolerant of fields this branch has not grown yet.

    `grant_keys` (every key an "always" would remember) is landing separately. The adapter never
    reads it — the gate owns grant writing — so these tests fill it in only if the dataclass
    declares it, rather than pinning themselves to whichever side merges first.
    """
    import dataclasses

    from localharness.agent.gate_types import PermissionRequest

    declared = {f.name for f in dataclasses.fields(PermissionRequest)}
    if "grant_keys" in declared and "grant_keys" not in fields:
        key = fields.get("key")
        fields["grant_keys"] = ((fields["klass"], key),) if key else ()
    return PermissionRequest(**fields)


class Session:
    """One connected agent+client pair, plus the objects the fake session build made."""

    def __init__(self, agent: AcpChannel, client: FakeClient, conn: Any, llm: StreamingLLM):
        self.agent = agent
        self.client = client
        self.conn = conn
        self.llm = llm
        self.gate: Any = None
        self.tools: dict[str, Tool] = {}


async def _connect(agent: AcpChannel, client: FakeClient) -> tuple[Any, list[asyncio.Task], Any]:
    """Wire the SDK's agent side to the SDK's client side over an in-process socket pair.

    A socket pair rather than OS pipes: `asyncio.open_connection(sock=...)` works on Windows too,
    and the owner dogfoods there.
    """
    left, right = socket.socketpair()
    agent_reader, agent_writer = await asyncio.open_connection(sock=left)
    client_reader, client_writer = await asyncio.open_connection(sock=right)
    agent_conn = AgentSideConnection(agent, agent_writer, agent_reader, listening=False)
    listen = asyncio.create_task(agent_conn.listen())
    client_conn = connect_to_agent(client, client_writer, client_reader)
    return client_conn, [listen], agent_conn


@pytest.fixture
def keep_cwd():
    """`session/new` chdirs into the project folder (the harness derives everything from cwd).
    Put the suite back where it was, or every later test runs somewhere else."""
    before = Path.cwd()
    yield
    os.chdir(before)


async def _start(
    tmp_path: Path,
    monkeypatch,
    *,
    responses: list[FakeLLMResponse],
    tools: list[Tool] | None = None,
    client: FakeClient | None = None,
    fs_read: bool = False,
    fs_write: bool = False,
    cwd: Path | None = None,
    mode: str = "guarded",
    mcp_servers: list[Any] | None = None,
) -> Session:
    """Bring up an ACP session over the real protocol: initialize, then session/new."""
    from acp.schema import ClientCapabilities, FileSystemCapabilities

    workspace = cwd if cwd is not None else (tmp_path / "project")
    workspace.mkdir(exist_ok=True)
    llm = StreamingLLM(responses)
    session_tools = list(tools or [])
    agent = AcpChannel(config_dir=str(tmp_path / "config"))
    session = Session(agent, client or FakeClient(), None, llm)

    async def _fake_start_async(**kwargs: Any) -> None:
        """The objects `_start_async` builds, minus the model server. Same `serve()` handover."""
        channel = kwargs["acp_channel"]
        bus = EventBus()
        registry = ToolRegistry()
        for tool in session_tools:
            await registry.register(tool, scope="global")
            session.tools[tool.info().name] = tool
        gate = PermissionGate(
            boundary=workspace,
            workspace=workspace,
            grants=GrantStore(tmp_path / "grants.yaml"),
            mode=mode,
            bus=bus,
        )
        gate.attach_channel(channel)
        session.gate = gate
        agent_loop = AgentLoop(
            config=AgentConfig(name="test-agent", role="Test agent."),
            llm=llm,
            bus=bus,
            context_manager=ContextManager(),
            tool_registry=registry,
            permission_evaluator=PermissionEvaluator(),
            gate=gate,
        )
        await channel.serve(bus=bus, agent_loop=agent_loop, gate=gate, tool_registry=registry)

    monkeypatch.setattr("localharness.cli.start_cmd._start_async", _fake_start_async)

    conn, tasks, _ = await _connect(agent, session.client)
    session.conn = conn
    session.tasks = tasks  # type: ignore[attr-defined]
    await conn.initialize(
        protocol_version=1,
        client_capabilities=ClientCapabilities(
            fs=FileSystemCapabilities(read_text_file=fs_read, write_text_file=fs_write)
        ),
    )
    response = await conn.new_session(cwd=str(workspace), mcp_servers=mcp_servers)
    session.session_id = response.session_id  # type: ignore[attr-defined]
    session.new_session_response = response  # type: ignore[attr-defined]
    return session


# ------------------------------------------------------------------ handshake


async def test_initialize_advertises_no_auth_and_protocol_v1(tmp_path, monkeypatch, keep_cwd):
    """PRD §4: the registry's CI requires `authMethods`, and a local harness has no account."""
    from acp.schema import ClientCapabilities

    agent = AcpChannel(config_dir=str(tmp_path / "config"))
    conn, _tasks, _ = await _connect(agent, FakeClient())
    response = await conn.initialize(
        protocol_version=1, client_capabilities=ClientCapabilities()
    )
    assert response.protocol_version == 1
    assert response.auth_methods == []
    assert response.agent_capabilities.load_session is False


async def test_new_session_returns_the_mode_picker(tmp_path, monkeypatch, keep_cwd):
    """PRD §3.4: auto is the default (v0.14.1) and `unattended` is never on the picker."""
    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    modes = session.new_session_response.modes
    assert modes.current_mode_id == "auto"
    assert [m.id for m in modes.available_modes] == ["auto", "guarded", "trusted", "read-only"]
    assert all(m.name and m.description for m in modes.available_modes)


# ------------------------------------------------------------------ one thread per process


async def test_a_second_session_is_refused(tmp_path, monkeypatch, keep_cwd):
    """F2: ACP allows many sessions on one connection; this adapter has one loop, one gate, one
    turn task and one working directory, so a second session would be served by the first one's
    state under a different id. Refused, in both folders."""
    from acp.core import RequestError

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    other = tmp_path / "other"
    other.mkdir()
    for folder in (tmp_path / "project", other):
        with pytest.raises(RequestError) as caught:
            await session.conn.new_session(cwd=str(folder))
        assert "one thread" in str(caught.value.data or caught.value)
    assert session.agent._session_id == session.session_id


async def test_a_prompt_for_another_session_is_refused(tmp_path, monkeypatch, keep_cwd):
    """The live id used to be overwritten by whatever id arrived, so a stale thread's prompt
    re-tagged the running turn's file reads, dialogs and updates."""
    from acp.core import RequestError

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    with pytest.raises(RequestError):
        await session.conn.prompt(session_id="not-a-session", prompt=[text_block("hello")])
    assert session.agent._session_id == session.session_id
    assert session.client.chunks() == []


async def test_setting_a_mode_on_another_session_is_refused(tmp_path, monkeypatch, keep_cwd):
    from acp.core import RequestError

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    with pytest.raises(RequestError):
        await session.conn.set_session_mode(session_id="not-a-session", mode_id="trusted")
    assert session.agent._current_mode_id() == "auto"


async def test_cancel_for_another_session_does_nothing(tmp_path, monkeypatch, keep_cwd):
    """`session/cancel` is a notification — it cannot answer — so a foreign id is dropped rather
    than stopping the turn this process is actually running."""
    started = asyncio.Event()

    class SlowShell(Shell):
        async def _execute(self, **kwargs: Any) -> ToolResult:
            started.set()
            await asyncio.sleep(30)
            return self.ok("never")  # pragma: no cover — cancelled first

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "ls"})),
        tools=[SlowShell()],
    )
    turn = asyncio.create_task(
        session.conn.prompt(session_id=session.session_id, prompt=[text_block("wait")])
    )
    await asyncio.wait_for(started.wait(), 10)

    await session.conn.cancel(session_id="not-a-session")
    await asyncio.sleep(0)
    assert not turn.done(), "a foreign cancel stopped this thread's turn"

    await session.conn.cancel(session_id=session.session_id)
    assert (await asyncio.wait_for(turn, 10)).stop_reason == "cancelled"


# ------------------------------------------------------------------ a turn


async def test_prompt_streams_chunks_and_ends_the_turn(tmp_path, monkeypatch, keep_cwd):
    session = await _start(
        tmp_path, monkeypatch, responses=[FakeLLMResponse(content="Hello from the harness.")]
    )
    response = await session.conn.prompt(
        session_id=session.session_id, prompt=[text_block("say hello")]
    )
    assert response.stop_reason == "end_turn"
    assert any("Hello from the harness." in c for c in session.client.chunks())


async def test_a_non_streaming_provider_still_delivers_the_answer(tmp_path, monkeypatch, keep_cwd):
    """`on_token` is the provider's promise, not the loop's.

    A runtime that returns a completion without deltas would otherwise leave the panel showing
    tool rows and no answer, so `TaskComplete` is the fallback — and exactly once.
    """
    session = await _start(
        tmp_path, monkeypatch, responses=[FakeLLMResponse(content="Quiet answer.")]
    )
    session.llm.streams = False
    response = await session.conn.prompt(
        session_id=session.session_id, prompt=[text_block("say something")]
    )
    assert response.stop_reason == "end_turn"
    said = [c for c in session.client.chunks() if "Quiet answer." in c]
    assert len(said) == 1, f"expected the answer exactly once, got {said}"


async def test_a_streamed_answer_is_not_repeated_by_the_completion_summary(
    tmp_path, monkeypatch, keep_cwd
):
    session = await _start(
        tmp_path, monkeypatch, responses=[FakeLLMResponse(content="Streamed answer.")]
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("say something")])
    said = [c for c in session.client.chunks() if "Streamed answer." in c]
    assert len(said) == 1, f"the answer was printed twice: {said}"


async def test_a_tool_call_mirrors_to_tool_call_and_tool_call_update(tmp_path, monkeypatch, keep_cwd):
    """PRD §4: `Action` → `tool_call` (kind from the tool group), `Observation` → update."""
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "ls -la"})),
        tools=[Shell()],
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("list files")])

    starts = session.client.tool_starts()
    assert [s.title for s in starts] == ["bash_exec: ls -la"]
    assert starts[0].kind == "execute"
    assert starts[0].status == "in_progress"
    updates = [u for u in session.client.tool_updates() if u.tool_call_id == starts[0].tool_call_id]
    assert updates and updates[-1].status == "completed"


# ------------------------------------------------------------------ the permission dialog


async def test_destructive_shell_asks_with_two_options_and_a_refusal_reaches_the_model(
    tmp_path, monkeypatch, keep_cwd
):
    """PRD §3.5: an ungrantable class offers only the `_once` pair, and a refusal is an
    observation the model can re-plan against — not a silent no-op.

    A hard reset rather than a recursive delete: the shipped deny patterns stop `rm -rf`
    outright (`config/models.py`), so it never reaches the ask — the next test locks that.
    """
    client = FakeClient(answers=["reject_once"])
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "git reset --hard HEAD~1"})),
        tools=[Shell()],
        client=client,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("undo that")])

    assert len(client.permission_requests) == 1
    _tool_call, options = client.permission_requests[0]
    assert [o.kind for o in options] == ["allow_once", "reject_once"]
    assert session.tools["bash_exec"].ran == []
    assert session.llm.model_saw("Permission denied")


async def test_a_denied_command_never_asks(tmp_path, monkeypatch, keep_cwd):
    """DENY beats ASK (PRD §3.1 order): a shipped deny pattern is not a question."""
    client = FakeClient(answers=["allow_once"])
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "rm -rf build"})),
        tools=[Shell()],
        client=client,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("clean up")])

    assert client.permission_requests == [], "a denied call put a question to the user"
    assert session.tools["bash_exec"].ran == []
    assert session.llm.model_saw("Permission denied")


async def test_allow_always_writes_a_grant_and_the_second_call_does_not_ask(
    tmp_path, monkeypatch, keep_cwd
):
    """PRD §3.3: the prediction-error gate — the first exposure carries the whole cost."""
    client = FakeClient(answers=["allow_always"])
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("bash_exec", {"command": "frobnicate --all"}),
            ("bash_exec", {"command": "frobnicate --all"}),
        ),
        tools=[Shell()],
        client=client,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("frobnicate")])

    assert len(client.permission_requests) == 1, "the second identical call asked again"
    _tool_call, options = client.permission_requests[0]
    assert [o.kind for o in options] == [
        "allow_once", "allow_always", "reject_once", "reject_always",
    ]
    assert session.tools["bash_exec"].ran == ["frobnicate --all", "frobnicate --all"]
    assert (tmp_path / "grants.yaml").exists(), "allow_always wrote no grant"


async def test_zed_holds_the_dialog_so_the_gate_puts_no_deadline_on_it():
    """PRD §3.5, Zed row: "Timeout: none". A deadline here would turn a user who stepped away
    from their editor into a refusal — and one that reads, in the ask-rate report, like a channel
    that could not reach anybody."""
    assert AcpChannel.ask_holds_dialog is True


async def test_a_multi_line_reason_becomes_a_title_and_a_body(tmp_path, monkeypatch, keep_cwd):
    """One command can carry several reasons, and the gate renders them as a short multi-line
    `display`. ACP's title is one row, so the rest has to ride as content or it is lost."""
    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    session.client.answers = ["allow_once"]
    # The channel has a live connection and session id from `_start`; call the asker directly,
    # because the multi-line display comes from the verdict, not from anything a turn does here.
    decision = await session.agent.ask_permission(
        _request(
            tool_name="bash_exec",
            tool_params={"command": "cp a b"},
            klass="shell-unfamiliar",
            key="cp",
            grantable=True,
            reason="unfamiliar",
            display="bash: cp a b\n  · writes outside the project\n  · unfamiliar command",
        )
    )
    assert decision.kind == "allow_once"
    tool_call, options = session.client.permission_requests[-1]
    assert tool_call.title == "bash: cp a b", "a newline leaked into the one-row title"
    assert tool_call.content and "unfamiliar command" in tool_call.content[0].content.text
    assert [o.kind for o in options] == [
        "allow_once", "allow_always", "reject_once", "reject_always",
    ]


async def test_a_single_line_reason_carries_no_body(tmp_path, monkeypatch, keep_cwd):
    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    session.client.answers = ["reject_once"]
    await session.agent.ask_permission(
        _request(
            tool_name="bash_exec",
            tool_params={"command": "cp a b"},
            klass="shell-destructive",
            key=None,
            grantable=False,
            reason="destructive",
            display="bash: cp a b  (destructive, asks every time)",
        )
    )
    tool_call, options = session.client.permission_requests[-1]
    assert tool_call.title == "bash: cp a b  (destructive, asks every time)"
    assert tool_call.content is None
    assert [o.kind for o in options] == ["allow_once", "reject_once"]


async def test_control_characters_never_reach_the_dialog(tmp_path, monkeypatch, keep_cwd):
    """F5: the question quotes the model's own words, and a `\\x1b[2J` or `\\r` in them can
    repaint the dialog the human is answering. Stripped before the title/body split."""
    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    session.client.answers = ["reject_once"]
    await session.agent.ask_permission(
        _request(
            tool_name="bash_exec",
            tool_params={"command": "echo x"},
            klass="shell-unfamiliar",
            key="echo",
            grantable=True,
            reason="unfamiliar",
            display="bash: echo \x1b[2Jharmless\rrm -rf /\n  · \x07unfamiliar command",
        )
    )
    tool_call, _options = session.client.permission_requests[-1]
    body = tool_call.content[0].content.text if tool_call.content else ""
    for rendered in (tool_call.title, body):
        assert "\x1b" not in rendered and "\r" not in rendered and "\x07" not in rendered
        assert "2J" not in rendered
    assert tool_call.title == "bash: echo harmlessrm -rf /"
    assert body == "  · unfamiliar command"


async def test_a_dismissed_dialog_is_a_refusal_of_this_call_only(tmp_path, monkeypatch, keep_cwd):
    """ACP's `DeniedOutcome{cancelled}` is Escape, which is `reject_once` — never a durable
    'never', which the user did not say."""
    client = FakeClient(answers=["cancelled"])
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "frobnicate --all"})),
        tools=[Shell()],
        client=client,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("frobnicate")])
    assert session.tools["bash_exec"].ran == []
    assert not (tmp_path / "grants.yaml").exists()


# ------------------------------------------------------------------ modes


async def test_set_session_mode_read_only_soft_denies_an_edit(tmp_path, monkeypatch, keep_cwd):
    """PRD §3.4: read-only refuses with words the model can re-plan against."""
    from localharness.tools.builtin.edit_tool import EditTool

    target = tmp_path / "project" / "notes.md"
    (tmp_path / "project").mkdir(exist_ok=True)
    target.write_text("alpha\n", encoding="utf-8")

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("edit", {"path": str(target), "old_string": "alpha", "new_string": "beta"})
        ),
        tools=[EditTool()],
    )
    await session.conn.set_session_mode(session_id=session.session_id, mode_id="read-only")
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("rename alpha")])

    assert session.gate.mode == "read-only"
    assert session.llm.model_saw("read-only mode")
    assert target.read_text(encoding="utf-8") == "alpha\n"


async def test_set_session_mode_unattended_is_refused(tmp_path, monkeypatch, keep_cwd):
    """PRD §3.4: `unattended` turns every ASK into an ALLOW, so a picker must never reach it."""
    from acp.core import RequestError

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    with pytest.raises(RequestError):
        await session.conn.set_session_mode(
            session_id=session.session_id, mode_id="unattended"
        )


# ------------------------------------------------------------------ no boundary


async def test_home_directory_gets_the_notice_and_no_turn(tmp_path, monkeypatch, keep_cwd):
    """PRD §3.1 / critic finding 1: standing in $HOME there is no boundary, so say so once
    instead of asking about every write for the rest of the session.

    The home directory is a FAKE one under `tmp_path`, the same way the subprocess test builds
    it (D8). Using the real `Path.home()` made this test chdir the whole suite into the
    developer's home directory and derive a boundary from whatever happened to be there — a
    `.localharness/` or a git checkout above `$HOME` would have made it pass or fail for reasons
    that have nothing to do with the adapter.
    """
    from localharness.cli.start_cmd import NO_BOUNDARY_NOTICE

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    session = await _start(
        tmp_path, monkeypatch, responses=[FakeLLMResponse(content="should not run")], cwd=home
    )
    response = await session.conn.prompt(
        session_id=session.session_id, prompt=[text_block("do something")]
    )
    assert response.stop_reason == "end_turn"
    assert any(NO_BOUNDARY_NOTICE in c for c in session.client.chunks())
    assert session.llm.seen_messages == [], "a turn ran with no boundary"


# ------------------------------------------------------------------ editor-backed file I/O


async def test_an_in_workspace_edit_goes_through_write_text_file_without_asking(
    tmp_path, monkeypatch, keep_cwd
):
    """PRD §4 + §3.1 choice 2: with `fs` advertised the edit lands in the editor's review pane,
    and because that review surface exists it never asks."""
    from localharness.tools.builtin.edit_tool import EditTool

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    target = project / "notes.md"
    target.write_text("alpha\n", encoding="utf-8")
    client = FakeClient(files={str(target): "alpha\n"})

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("edit", {"path": str(target), "old_string": "alpha", "new_string": "beta"})
        ),
        tools=[EditTool()],
        client=client,
        fs_read=True,
        fs_write=True,
    )
    assert session.agent.has_review_surface is True
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("rename alpha")])

    assert client.permission_requests == []
    assert client.written == {str(target): "beta\n"}
    assert target.read_text(encoding="utf-8") == "alpha\n", "the edit went to disk, not the editor"


async def test_a_write_only_client_gets_no_editor_hooks_and_the_edit_asks(
    tmp_path, monkeypatch, keep_cwd
):
    """R5: the editor seam is all-or-nothing (PRD §4).

    A client advertising `fs/write_text_file` but not `fs/read_text_file` used to get the write
    hook alone — an append then truncated the file, and an edit read DISK while writing the
    BUFFER, overwriting unsaved work. Neither hook is wired now, there is no review surface, and
    the in-workspace edit falls back to the `edit-unreviewed` ask (PRD §3.1)."""
    from localharness.tools.builtin.edit_tool import EditTool

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    target = project / "notes.md"
    target.write_text("alpha\n", encoding="utf-8")
    client = FakeClient(answers=["allow_once"], files={str(target): "alpha\n"})

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("edit", {"path": str(target), "old_string": "alpha", "new_string": "beta"})
        ),
        tools=[EditTool()],
        client=client,
        fs_read=False,
        fs_write=True,
    )
    assert session.agent.has_review_surface is False
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("rename alpha")])

    edit = session.tools["edit"]  # the session (and its registry) is built on the first prompt
    assert edit.file_read_hook is None and edit.file_write_hook is None
    assert len(client.permission_requests) == 1, "an unreviewed in-workspace edit must ask once"
    _tool_call, options = client.permission_requests[0]
    assert "allow_always" in [o.kind for o in options], (
        "edit-unreviewed is grantable — it is the ask that can be answered once and for all"
    )
    assert client.written == {}, "nothing may reach fs/write_text_file without the read half"
    assert target.read_text(encoding="utf-8") == "beta\n"


async def test_append_under_the_editor_keeps_the_buffer_content_not_the_disk_copy(
    tmp_path, monkeypatch, keep_cwd
):
    """R5: ACP has no append, so `write(mode='append')` reads the BUFFER and rewrites the whole
    file. The disk copy is stale by definition — appending onto it would silently drop every
    unsaved line (PRD §4)."""
    from localharness.tools.builtin.write_tool import WriteTool

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    target = project / "log.md"
    target.write_text("saved\n", encoding="utf-8")
    client = FakeClient(files={str(target): "saved\nunsaved\n"})

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("write", {"path": str(target), "content": "appended\n", "mode": "append"})
        ),
        tools=[WriteTool()],
        client=client,
        fs_read=True,
        fs_write=True,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("append a line")])

    assert client.written == {str(target): "saved\nunsaved\nappended\n"}
    assert target.read_text(encoding="utf-8") == "saved\n", "the append went to disk, not the editor"


async def test_an_edit_under_the_editor_reads_the_unsaved_buffer(tmp_path, monkeypatch, keep_cwd):
    """R5: the edit's `old_string` is matched against what the user is LOOKING at, not the file on
    disk — the two differ the moment there is an unsaved change (PRD §4)."""
    from localharness.tools.builtin.edit_tool import EditTool

    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    target = project / "notes.md"
    target.write_text("on disk\n", encoding="utf-8")
    client = FakeClient(files={str(target): "in the buffer\n"})

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(
            ("edit", {"path": str(target), "old_string": "in the buffer", "new_string": "edited"})
        ),
        tools=[EditTool()],
        client=client,
        fs_read=True,
        fs_write=True,
    )
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("edit it")])

    assert client.written == {str(target): "edited\n"}
    assert target.read_text(encoding="utf-8") == "on disk\n"


# ------------------------------------------------------------------ cancel


async def test_cancel_during_a_turn_stops_it_with_stop_reason_cancelled(
    tmp_path, monkeypatch, keep_cwd
):
    """PRD §4: `session/cancel` cancels the turn task — the same path SIGINT takes today."""
    started = asyncio.Event()

    class SlowShell(Shell):
        async def _execute(self, **kwargs: Any) -> ToolResult:
            started.set()
            await asyncio.sleep(30)
            return self.ok("never")  # pragma: no cover — cancelled first

    session = await _start(
        tmp_path,
        monkeypatch,
        responses=_plan(("bash_exec", {"command": "ls"})),
        tools=[SlowShell()],
    )
    turn = asyncio.create_task(
        session.conn.prompt(session_id=session.session_id, prompt=[text_block("wait")])
    )
    await asyncio.wait_for(started.wait(), 10)
    await session.conn.cancel(session_id=session.session_id)
    response = await asyncio.wait_for(turn, 10)
    assert response.stop_reason == "cancelled"


# ------------------------------------------------ what the editor sends and we cannot use


async def test_editor_passed_mcp_servers_are_announced_not_dropped(
    tmp_path, monkeypatch, keep_cwd, caplog
):
    """D1: `session/new` carries the editor's own MCP servers and this version connects none.

    The user is waiting for those tools. Saying nothing left them waiting forever with no
    explanation anywhere — the panel, the log, or the model's own view of what it has.
    """
    from acp.schema import McpServerStdio

    from localharness.channels.acp import MCP_SERVERS_NOT_CONNECTED_LOG

    caplog.set_level("WARNING", logger="localharness.channels.acp")
    session = await _start(
        tmp_path,
        monkeypatch,
        responses=[FakeLLMResponse(content="hi")],
        mcp_servers=[
            McpServerStdio(name="github", command="gh-mcp", args=[], env=[]),
            McpServerStdio(name="postgres", command="pg-mcp", args=[], env=[]),
        ],
    )

    logged = caplog.text
    assert MCP_SERVERS_NOT_CONNECTED_LOG.split("%")[0].strip() in logged
    assert "github" in logged and "postgres" in logged, "the log must name each server"

    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("hello")])
    notice = [c for c in session.client.chunks() if "MCP server" in c]
    assert len(notice) == 1, "the panel is told once, on the first prompt"
    assert "github, postgres" in notice[0]
    assert "tools.mcp_servers" in notice[0], "say where MCP servers DO come from"

    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("again")])
    assert len([c for c in session.client.chunks() if "MCP server" in c]) == 1, "repeated notice"


async def test_a_session_with_no_mcp_servers_says_nothing(tmp_path, monkeypatch, keep_cwd):
    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("hello")])
    assert not [c for c in session.client.chunks() if "MCP server" in c]


async def test_an_at_mention_reaches_the_model_as_a_visible_placeholder(
    tmp_path, monkeypatch, keep_cwd
):
    """D2: a Zed @-file-mention is a `ResourceContentBlock` — baseline protocol, not an exotic
    case — and it used to be dropped in silence. The user saw the editor attach a file; the model
    saw a bare sentence with a dangling "this" and answered about nothing."""
    from acp.schema import ImageContentBlock, ResourceContentBlock

    from localharness.channels.acp import ATTACHMENT_PLACEHOLDER

    session = await _start(
        tmp_path, monkeypatch, responses=[FakeLLMResponse(content="I cannot read that.")]
    )
    await session.conn.prompt(
        session_id=session.session_id,
        prompt=[
            text_block("summarise"),
            ResourceContentBlock(type="resource_link", name="notes.md", uri="file:///notes.md"),
            ImageContentBlock(type="image", data="AAAA", mime_type="image/png"),
            text_block("please"),
        ],
    )

    sent = session.llm.seen_messages[0]
    task = "\n".join(str(m.get("content") or "") for m in sent)
    assert ATTACHMENT_PLACEHOLDER.format(label="notes.md") in task
    assert ATTACHMENT_PLACEHOLDER.format(label="image") in task
    assert task.index("summarise") < task.index("notes.md") < task.index("please"), (
        "the placeholder must sit where the block was sent"
    )


async def test_the_prompt_text_keeps_every_block_in_order():
    """The same fact at the unit the wire cannot show: nothing is silently discarded."""
    from acp.schema import EmbeddedResourceContentBlock, TextResourceContents

    from localharness.channels.acp import ATTACHMENT_PLACEHOLDER, _prompt_text

    class Nameless:
        """A block from a client this version has never heard of."""

    embedded = EmbeddedResourceContentBlock(
        type="resource",
        resource=TextResourceContents(uri="file:///x.py", text="print('hi')"),
    )

    assert _prompt_text([]) == ""
    assert _prompt_text([text_block("a"), text_block("b")]) == "a\nb"
    # An embedded resource carries its uri one level down, and its text is NOT read: v1 reads
    # the prompt's own text only, and pretending otherwise is the drop this fix exists to stop.
    assert _prompt_text([embedded]) == ATTACHMENT_PLACEHOLDER.format(label="file:///x.py")
    assert _prompt_text([Nameless()]).startswith("[attachment:")


# ------------------------------------------------------------------ dialog pairing


async def test_two_identical_calls_in_flight_pair_with_the_right_row(
    tmp_path, monkeypatch, keep_cwd
):
    """D4: one shared slot meant the newer `Action` overwrote the older one, so the older call's
    dialog attached to the NEWER call's row — a person approving one command while reading
    another. The map is keyed by call id, and the request's own id wins when it carries one."""
    import types

    from localharness.core.events import Action, Observation

    session = await _start(tmp_path, monkeypatch, responses=[FakeLLMResponse(content="hi")])
    await session.conn.prompt(session_id=session.session_id, prompt=[text_block("warm up")])
    agent = session.agent

    params = {"command": "cargo publish"}
    for call_id in ("tc-first", "tc-second"):
        await agent.on_action(Action(
            agent_id="a", session_id="s", action_type="tool_call", tool_call_id=call_id,
            tool_name="bash_exec", tool_params=params,
        ))

    def _ask(**extra):
        return types.SimpleNamespace(
            tool_name="bash_exec", tool_params=params, grantable=True,
            display="bash_exec: cargo publish", **extra,
        )

    before = len(session.client.permission_requests)
    await agent.ask_permission(_ask(call_id="tc-first"))
    tool_call, _options = session.client.permission_requests[before]
    assert tool_call.tool_call_id == "tc-first", "the request's own call id must win"

    # No call id (the pre-lane-B shape): the heuristic picks the newest matching pending call.
    await agent.ask_permission(_ask())
    tool_call, _options = session.client.permission_requests[before + 1]
    assert tool_call.tool_call_id == "tc-second"

    # A finished call can pair with nothing: its row is closed and the map drops it.
    await agent.on_observation(Observation(
        agent_id="a", session_id="s", observation_type="tool_result",
        tool_call_id="tc-second", tool_name="bash_exec", output="ok",
    ))
    await agent.ask_permission(_ask())
    tool_call, _options = session.client.permission_requests[before + 2]
    assert tool_call.tool_call_id == "tc-first"


# ------------------------------------------------------------------ the real command


async def test_subprocess_agent_answers_initialize_and_new_session(tmp_path, keep_cwd):
    """The honest end-to-end proof: `localharness acp` is a real process that speaks ACP.

    Everything above drives the adapter in-process; this proves the console script starts, that
    stdout carries clean JSON-RPC (a single stray banner line would break the framing here), and
    that `session/new` answers with a session id and the mode picker. It stops short of a prompt
    — that needs a model server, which the live smoke covers.
    """
    from acp.schema import ClientCapabilities
    from acp.stdio import spawn_agent_process

    command = Path(sys.executable).parent / "localharness"
    if not command.exists():  # pragma: no cover — installed console script is the norm here
        pytest.skip("localharness console script not on this interpreter's path")

    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}

    client = FakeClient()
    async with spawn_agent_process(
        client,
        str(command),
        "acp",
        "--config-dir",
        str(tmp_path / "config"),
        env=env,
        cwd=str(project),
    ) as (conn, process):
        init = await asyncio.wait_for(
            conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities()), 60
        )
        assert init.protocol_version == 1
        assert init.auth_methods == []
        new = await asyncio.wait_for(conn.new_session(cwd=str(project)), 60)
        assert new.session_id
        assert new.modes.current_mode_id == "auto"


@pytest.mark.skipif(
    not hasattr(__import__("signal"), "SIGTERM"),
    reason="SIGTERM is POSIX; the Windows path is the signal.signal fallback",
)
async def test_sigterm_tears_the_session_down_and_exits_clean(tmp_path, keep_cwd):
    """D3: closing Zed's agent panel sends SIGTERM, and the process used to die of it.

    Exit -15 meant `serve()` never returned, so `_start_async`'s `finally` never ran: MCP servers
    were left running, memory consolidation and the WAL checkpoint never happened, and the last
    minutes of the session were simply lost. A signal now cancels the protocol task so the same
    teardown runs as on EOF, and the exit code says so.
    """
    import signal

    from acp.schema import ClientCapabilities
    from acp.stdio import spawn_agent_process

    from localharness.cli.acp_cmd import SHUTDOWN_NOTICE

    command = Path(sys.executable).parent / "localharness"
    if not command.exists():  # pragma: no cover — installed console script is the norm here
        pytest.skip("localharness console script not on this interpreter's path")

    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}

    async with spawn_agent_process(
        FakeClient(),
        str(command),
        "acp",
        "--config-dir",
        str(tmp_path / "config"),
        env=env,
        cwd=str(project),
    ) as (conn, process):
        await asyncio.wait_for(
            conn.initialize(protocol_version=1, client_capabilities=ClientCapabilities()), 60
        )
        await asyncio.wait_for(conn.new_session(cwd=str(project)), 60)

        process.send_signal(signal.SIGTERM)
        stderr = await asyncio.wait_for(process.stderr.read(), 60)
        assert await asyncio.wait_for(process.wait(), 60) == 0, "SIGTERM killed the process"

    logged = stderr.decode("utf-8", "replace")
    assert SHUTDOWN_NOTICE.split("%")[0].strip() in logged, (
        f"no orderly-shutdown line on stderr: {logged[-2000:]}"
    )
    assert "SIGTERM" in logged
    assert process.returncode is None or process.returncode == 0
