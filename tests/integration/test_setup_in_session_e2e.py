"""`/plugins enable NAME` in a terminal session: the step runs on the plain terminal between the two
halves of a restart, and the session comes back with the plugin on and the same conversation
(52-RULINGS R1, R2, R3, R14; SETUP-05, SETUP-06).

The gate is composed over `start_app`'s own loop and the real `_start_async`, run twice in one
process — the way the owner's terminal runs it.

REAL: `start_app`'s restart loop; `_start_async` both times (config layers, the resolver, the
plugin lifecycle and its registration, the tool registry, the permission gate, the memory plugin
and its sessions table, the AgentLoop and its hand-over); the REPL's own slash dispatch, the
`/plugins` handler and the session's hook (`switch_decision` over a fresh plan); `session_step` ->
`_switch` (the question, the one checked overlay write, the doctor-row check); the inference gate
and its reset between the two event loops.

STUBBED: the LLM probe and the tokenizer (`_stub_start_boundaries` via `_machine`); the model's
replies (`LLMClient.stream_complete`); the REPL's read loop — `drive` stands in for
`OrchestratorREPL.run` and sends each scripted line through the real `_dispatch_input`, awaiting
each turn; the terminal's answers (`plugins_cmd.typer.prompt`); ComfyUI (`_fake_comfy`, an
`httpx.MockTransport` — no socket); the embedding model (`ResonanceEngine.embed_docs` /
`embed_query` return fixed vectors, as the memory e2e stubs them — memory's consolidation embeds in
a worker thread, and with the hermetic home's empty model cache the real one went to fetch the model
from the Hugging Face CDN, measured). The provider points at the loopback discard port and every
address dialed is recorded.

NOT proven here: the persistent input box closing and reopening on a real terminal, typed-ahead
lines carried through a real box, and how it all looks — the owner's terminal check (52-07).

Every test is a plain `def`: `start_app` and the step call `asyncio.run`, and asyncio_mode is
"auto", so an `async def` test would already be inside a running loop.
"""
from __future__ import annotations

import asyncio
import gc
import io
import json
import stat
from types import SimpleNamespace

import click
import httpx
import numpy as np
import pytest
import yaml
from pydantic import BaseModel, Field, SecretStr
from rich.console import Console

from localharness.channels.terminal import TerminalChannel
from localharness.cli import doctor_cmd, plugins_cmd, start_cmd
from localharness.plugins import builtin
from localharness.plugins.api import GLOBAL_ONLY, Check, Plugin, PluginManifest, SetupField
from localharness.plugins.setup import AGENT_PROMPT_LEAD
from localharness.provider import client as client_mod
from localharness.tools.builtin import generate_image_tool
from tests.conftest import FakeLLMResponse
from tests.integration.test_all_plugins_off_e2e import _record
from tests.integration.test_image_plugin_e2e import _fake_comfy, _machine, _record_dials
from tests.integration.test_setup_steps_e2e import AR_NEXT, PHONE_Q, WEB_NEXT
from tests.unit.test_restart_primitives import _contend
from tests.unit.test_start_cmd import _capture_start_console, _read_sessions
from tests.unit.test_start_plugins import _record_loop

COMFY = "http://comfy.test"
SECRET = "SENTINEL-52-SECRET"
RESTARTED = "Restarted with image on. Your conversation continues."


class SecpConfig(BaseModel):  # the shape of tests/unit/test_secret_settings_echo.py's
    tok: SecretStr = Field(SecretStr(""), json_schema_extra=GLOBAL_ONLY)


class Secp(Plugin):
    """holds a token"""

    manifest = PluginManifest(name="secp", version="0.1.0", kind="tools", enabled_by_default=False,
                              setup=(SetupField(key="tok", prompt="Token", secret=True),))
    ConfigModel = SecpConfig

    async def configure(self, ctx):  # pending (and asked) while the token is empty
        return "ready" if ctx.config.tok.get_secret_value() else ("unconfigured", "secp.tok")

    def doctor(self, ctx):
        return [Check(name="secp", status="pass", detail="a token is set")]


@pytest.fixture
def run(tmp_path, monkeypatch, fake_home, capfd, caplog):
    """`run(scripts, answer=...)`: `localharness start` on a hermetic machine, one scripted REPL per
    sitting. A script line is typed through the REPL's real dispatch (a callable is called with the
    REPL instead); lines carried over from the last sitting play first, as both real loops play them.
    `answer` is what the terminal types at the step's question, or an exception it raises there.
    Returns what was observed."""
    global_dir, cwd = _machine(tmp_path, monkeypatch, fake_home)
    s = SimpleNamespace(global_dir=global_dir, cwd=cwd, runs=0, repl_running=False, prompts=[],
                        said=[], calls=[], conversations=[], contended=0, resumed=[], banners=0,
                        resets=0, lifecycles=[], step=io.StringIO(), doctor=io.StringIO())
    s.comfy = _fake_comfy(monkeypatch, host="comfy.test")
    s.dialed = _record_dials(monkeypatch)
    s.printed = _capture_start_console(monkeypatch)
    s.loops = _record_loop(monkeypatch)
    # the embedding model is an outside endpoint too: never loaded, never fetched
    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine.embed_docs",
                        lambda self, texts: np.ones((len(texts), 8), dtype=np.float32))
    monkeypatch.setattr("localharness.memory.resonance.ResonanceEngine.embed_query",
                        lambda self, text: np.ones(8, dtype=np.float32))
    _record(monkeypatch, "localharness.plugins.lifecycle.start_plugins", s.lifecycles, is_async=True)
    # "Checking it now:" and the step's own lines print through plugins_cmd's console; every check
    # row prints through doctor's row printer (doctor_cmd's console)
    monkeypatch.setattr(plugins_cmd, "console", Console(file=s.step, width=400))
    monkeypatch.setattr(doctor_cmd, "console", Console(file=s.doctor, width=400))
    monkeypatch.setattr(plugins_cmd, "_stdin_is_a_terminal", lambda: True)
    # a fresh real gate with one permit, so each sitting's contention binds it to that loop
    monkeypatch.setattr(client_mod, "_inference_sem", asyncio.Semaphore(1))
    monkeypatch.setattr(client_mod, "_MAX_CONCURRENT_INFERENCE", 1)
    real_reset = client_mod.reset_inference_gate

    def reset():
        s.resets += 1
        real_reset()

    monkeypatch.setattr(client_mod, "reset_inference_gate", reset)

    real_send = TerminalChannel.send_message

    async def send(self, content, *a, **k):  # what the REPL says (the /plugins lines among it)
        s.said.append(content)
        return await real_send(self, content, *a, **k)

    monkeypatch.setattr(TerminalChannel, "send_message", send)

    from localharness.agent.loop import AgentLoop
    real_resume = AgentLoop.resume

    def resume(self, conversation, prior_context):
        s.resumed.append([dict(m) for m in conversation])
        return real_resume(self, conversation, prior_context)

    monkeypatch.setattr(AgentLoop, "resume", resume)

    from localharness.cli import ui
    real_banner = ui.startup_banner

    def banner(*a, **k):
        s.banners += 1
        return real_banner(*a, **k)

    monkeypatch.setattr(ui, "startup_banner", banner)

    async def model(self, messages, tools=None, on_token=None, **_):
        s.calls.append({"run": s.runs, "messages": [dict(m) for m in messages],
                        "tools": [t.name for t in tools or ()]})
        return FakeLLMResponse(content="Hi there." if len(s.calls) == 1 else "Done."), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)

    def _run(scripts, *, answer=COMFY, at_prompt=None) -> SimpleNamespace:
        scripts = [list(script) for script in scripts]

        def prompt(text, default=None, **kw):
            s.prompts.append((text, default, kw, s.repl_running))
            if at_prompt is not None:
                at_prompt()
            if isinstance(answer, BaseException):
                raise answer
            return answer

        monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)

        async def drive(self):  # stands in for OrchestratorREPL.run
            s.runs += 1
            s.repl_running = True
            try:
                for line in [*self._resume_queue, *scripts.pop(0)]:
                    if callable(line):
                        line(self)
                        continue
                    task = await self._dispatch_input(line)
                    if task is not None:
                        await task
            except EOFError:
                pass  # /plugins enable|disable ended the REPL, as /quit ends it
            finally:
                s.conversations.append(self._agent.resume_state()[0])
                await _contend()  # the real inference gate, contended in THIS sitting's loop
                s.contended += 1
                s.repl_running = False

        monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)
        start_cmd.start_app(agent=None, verbose=False, debug=False, config_dir=None, channel="terminal",
                            subagents=False, model=None, list_models=False, no_input=False,
                            show_reasoning=False)
        out, err = capfd.readouterr()
        s.terminal = out + err + caplog.text
        # 52-04's real-terminal finding, now across two event loops: nothing reports a closed loop
        assert "Event loop is closed" not in s.terminal
        assert "Unhandled exception in event loop" not in s.terminal
        addresses = {a[:2] for a in s.dialed if isinstance(a, tuple)}
        assert addresses <= {("127.0.0.1", 9)}, f"the sittings dialed {addresses}"
        return s

    _run.machine = s  # a test may seed the machine's files before it runs
    return _run


def _first_request(s, sitting: int) -> dict:
    return next(c for c in s.calls if c["run"] == sitting)


def _carried(s) -> list[dict]:
    """The first sitting's conversation as its AgentLoop held it, leading system message included.
    Asserts it was handed to AgentLoop.resume, and that the rebuilt sitting's first request carries
    every message after the system message byte for byte (the loop rebuilds the system message each
    turn, for the sitting that sends it). Returns that request's messages."""
    conversation = s.conversations[0]
    assert conversation[0]["role"] == "system"
    assert conversation[1] == {"role": "user", "content": "hello there"}
    assert any(m["role"] == "assistant" and m["content"] == "Hi there." for m in conversation)
    assert s.resumed == [conversation]
    messages = _first_request(s, 2)["messages"]
    assert messages[0]["role"] == "system"
    head = messages[1:len(conversation)]
    assert json.dumps(head, sort_keys=True) == json.dumps(conversation[1:], sort_keys=True)
    return messages


def _overrides(s) -> dict:
    path = s.global_dir / "overrides.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


@pytest.mark.plugin("image")
def test_enabling_a_plugin_in_a_session_keeps_the_conversation_and_offers_its_tool(run):
    s = run([["hello there", "/plugins enable image"], ["draw a lighthouse"]])

    # the REPL said it, and ended
    assert "Restarting with image on — your conversation is kept." in s.said
    # the step: one question, on the plain terminal, while no REPL ran
    assert [(t, d, r) for t, d, _kw, r in s.prompts] == [("ComfyUI address", "http://127.0.0.1:8188", False)]
    assert _overrides(s) == {"image": {"enabled": True, "comfyui_url": COMFY}}
    assert "Checking it now:" in s.step.getvalue()
    assert f"✓ image: ComfyUI reachable at {COMFY}" in s.doctor.getvalue()
    # exactly one rebuild, through the plugin lifecycle (nothing registered by hand)
    assert s.runs == 2 and len(s.lifecycles) == 2 and s.resets == 1
    assert "image" not in s.lifecycles[0][1].loaded_names and "image" in s.lifecycles[1][1].loaded_names
    # the same conversation reaches the model, and the new tool is offered
    messages = _carried(s)
    assert messages[len(s.conversations[0]):] == [{"role": "user", "content": "draw a lighthouse"}]
    assert "generate_image" in _first_request(s, 2)["tools"]
    assert "generate_image" not in _first_request(s, 1)["tools"]
    # the eviction store the carried stubs name comes along
    first, second = (loop["context_manager"]._eviction_store for loop in s.loops)
    assert second is first
    # the compact indicator, never a second wordmark
    assert s.banners == 1
    at = s.printed.index(RESTARTED)
    assert "image: on in this session" in s.printed[at:]
    # two sittings: two memory rows, different ids, both closed
    rows = _read_sessions(s.global_dir)
    assert len(rows) == 2 and rows[0][0] != rows[1][0]
    assert all(row[2] is not None for row in rows)
    # each sitting contended the real gate in its own loop without "bound to a different event loop"
    assert s.contended == 2
    assert "GET /system_stats" in s.comfy


@pytest.mark.plugin("image")
def test_an_already_configured_plugin_is_switched_on_without_a_question(run):
    """Set up before (its address stored, the plugin off): no question — the switch, the check,
    the restart and the indicator (SETUP-06's second half)."""
    (run.machine.global_dir / "overrides.yaml").write_text(
        yaml.safe_dump({"image": {"enabled": False, "comfyui_url": COMFY}}), encoding="utf-8")
    s = run([["hello there", "/plugins enable image"], ["draw a lighthouse"]])

    assert s.prompts == []
    assert _overrides(s) == {"image": {"enabled": True, "comfyui_url": COMFY}}
    assert f"✓ image: ComfyUI reachable at {COMFY}" in s.doctor.getvalue()
    assert s.runs == 2 and len(s.lifecycles) == 2 and s.resets == 1
    _carried(s)
    assert "image: on in this session" in s.printed
    assert "generate_image" in _first_request(s, 2)["tools"]


@pytest.mark.plugin("image")
def test_a_failed_check_never_blocks_the_restart(run, monkeypatch):
    _fake_comfy(monkeypatch, host="comfy.test", down=True)
    s = run([["hello there", "/plugins enable image"], ["draw a lighthouse"]])

    assert f"✗ image: ComfyUI unreachable at {COMFY}" in s.doctor.getvalue()
    assert _overrides(s) == {"image": {"enabled": True, "comfyui_url": COMFY}}  # it wrote
    assert s.runs == 2 and len(s.lifecycles) == 2
    _carried(s)
    assert f"image: on, but its check failed: ComfyUI unreachable at {COMFY}" in s.printed
    assert "generate_image" in _first_request(s, 2)["tools"]


def _not_set_up_yet(s, name: str, detail: str) -> None:
    """A skipped check left unanswered: the step still printed the coding-agent prompt and the next
    step, the session came back with the conversation, and the indicator says not set up yet —
    no line says the check failed."""
    assert s.runs == 2 and len(s.lifecycles) == 2 and s.resets == 1
    assert name in s.lifecycles[1][1].loaded_names
    _carried(s)
    assert f"{name}: on, but not set up yet — {detail}" in s.printed
    assert not any("failed" in line for line in s.printed[s.printed.index(f"Restarted with {name} on. "
                                                                          "Your conversation continues."):])


@pytest.mark.plugin("mobile")
def test_web_left_unset_is_on_but_not_set_up_yet(run):
    """Deferred #23: Enter at mobile's optional phone address, on a machine where `localharness mobile`
    has never run — every new install. Its check is skipped ("not enrolled yet"): a state that is
    not set up yet, not a failure."""
    s = run([["hello there", "/plugins enable mobile"], ["what now"]], answer="")

    assert "Restarting with mobile on — your conversation is kept." in s.said
    assert [(t, d, r) for t, d, _kw, r in s.prompts] == [(PHONE_Q, "", False)]
    assert _overrides(s) == {"mobile": {"enabled": True}}  # Enter wrote no address
    assert "i  mobile: not enrolled yet" in s.doctor.getvalue()
    step = s.step.getvalue()
    assert step.index(AGENT_PROMPT_LEAD) < step.index(WEB_NEXT)
    _not_set_up_yet(s, "mobile", "not enrolled yet")


@pytest.mark.plugin("autoresearch")
def test_autoresearch_with_no_proposer_is_on_but_not_set_up_yet(run, monkeypatch):
    from localharness.autoresearch.plugin import NO_PROPOSER
    from localharness.plugins import setup
    monkeypatch.setattr(setup, "gpu_name", lambda: "NVIDIA GB10")  # its prompt names {machine}
    s = run([["hello there", "/plugins enable autoresearch"], ["what now"]], answer="")

    assert [(t, kw["hide_input"]) for t, _d, kw, _r in s.prompts] == [
        ("Proposer address (an OpenAI-compatible base URL — a local server or a cloud API)", False),
        ("Proposer model id", False), ("Proposer API key (leave empty for a local server)", True)]
    assert _overrides(s) == {"autoresearch": {"enabled": True}}  # Enter on all three: nothing written
    assert f"i  autoresearch: {NO_PROPOSER}" in s.doctor.getvalue()
    step = s.step.getvalue()
    assert step.index(AGENT_PROMPT_LEAD) < step.index("This machine reports NVIDIA GB10.") < step.index(AR_NEXT)
    _not_set_up_yet(s, "autoresearch", NO_PROPOSER)


@pytest.mark.plugin("image")
def test_a_stopped_step_still_resumes_the_conversation(run):
    s = run([["hello there", "/plugins enable image"], ["draw a lighthouse"]],
            answer=click.exceptions.Abort())

    assert "Stopped. Your conversation continues." in s.step.getvalue()
    assert "image" not in _overrides(s)
    assert s.runs == 2 and len(s.lifecycles) == 2 and s.resets == 1
    _carried(s)
    assert "image: not on in this session — its setup stopped before it finished" in s.printed
    assert "generate_image" not in _first_request(s, 2)["tools"]


@pytest.mark.plugin("memory")
def test_an_already_running_plugin_does_not_restart(run, monkeypatch):
    # memory set up: its embedding model in the local cache (the box's own cache must not decide)
    monkeypatch.setattr("localharness.memory.plugin._embedding_check",
                        lambda model: Check(name="memory-embedding", status="pass", detail="cached"))
    s = run([["hello there", "/plugins enable memory", "and more"]])

    assert "memory is already on." in s.said
    assert not any(line.startswith("Restarting") for line in s.said)
    assert s.runs == 1 and len(s.lifecycles) == 1 and s.resets == 0 and s.prompts == []
    assert "memory" in s.lifecycles[0][1].loaded_names
    assert {c["run"] for c in s.calls} == {1}
    typed = [[m["content"] for m in c["messages"] if m["role"] == "user"] for c in s.calls]
    assert any(t[0] == "hello there" and t[-1] == "and more" for t in typed)  # one sitting, both turns


@pytest.mark.plugin("image")
def test_mode_read_only_survives_the_restart(run):
    s = run([["/mode read-only", "/plugins enable image"], []])

    assert s.runs == 2 and len(s.loops) == 2
    first, second = (loop["gate"] for loop in s.loops)
    assert second is not first and second.mode == "read-only"


@pytest.mark.plugin("image")
def test_typed_ahead_lines_are_played_after_the_restart(run):
    """A line typed while a turn ran is queued behind /plugins (as the input box queues it); it
    reaches the rebuilt sitting and is played there first."""
    def typed_ahead(repl):
        repl._fifo.append("and then this")

    s = run([["hello there", typed_ahead, "/plugins enable image"], []])

    assert s.runs == 2
    messages = _carried(s)
    assert messages[len(s.conversations[0]):] == [{"role": "user", "content": "and then this"}]


def test_a_secret_typed_in_the_step_never_reaches_a_session_file(run, monkeypatch, tmp_path):
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (*builtin.BUILTIN_PLUGINS, Secp))
    s = run([["hello there", "/plugins enable secp"], ["hello again"]], answer=SECRET)

    # asked once, hidden, on the plain terminal while no REPL ran
    assert [(t, kw.get("hide_input"), r) for t, _d, kw, r in s.prompts] == [("Token", True, False)]
    assert "✓ secp: a token is set" in s.doctor.getvalue()
    assert "secp: on in this session" in s.printed
    # the one place it lands: the machine's overrides.yaml, owner-only
    overrides = s.global_dir / "overrides.yaml"
    assert _overrides(s)["secp"] == {"enabled": True, "tok": SECRET}
    assert stat.S_IMODE(overrides.stat().st_mode) == 0o600
    # every other file: history, the bus log, the session logs, memory, an audit file, anything
    agent_dir = s.global_dir / "agents" / "orchestrator"
    assert (agent_dir / "bus-events.jsonl").is_file() and (agent_dir / "memory.log").is_file()
    files = [p for p in tmp_path.rglob("*") if p.is_file() and p != overrides]
    assert len(files) > 5
    assert [p for p in files if SECRET.encode() in p.read_bytes()] == []
    # and nothing printed: start's console, the step's, doctor's rows, the REPL's lines, the terminal
    shown = "\n".join([*s.printed, s.step.getvalue(), s.doctor.getvalue(), *s.said, s.terminal])
    assert SECRET not in shown


@pytest.mark.plugin("image")
def test_an_error_in_the_step_still_resumes_the_conversation(run, monkeypatch):
    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(plugins_cmd, "atomic_write_overlay", boom)  # the step's one overlay write
    s = run([["hello there", "/plugins enable image"], ["draw a lighthouse"]])

    assert "image's setup hit an error: OSError: disk full. Your conversation continues." in s.step.getvalue()
    assert s.runs == 2 and len(s.lifecycles) == 2 and s.resets == 1
    _carried(s)
    assert "image: not on in this session — its setup stopped before it finished" in s.printed
    assert "image" not in _overrides(s)


class _Leftover:
    """Garbage only the cyclic GC frees, whose finalizer notes the loop it ran in (the SDK client's
    finalizer schedules its close on "the running loop"; 52-04's finding)."""

    finalized_in: list = []

    def __init__(self):
        self._cycle = self

    def __del__(self):
        try:
            _Leftover.finalized_in.append(asyncio.get_running_loop())
        except RuntimeError:
            _Leftover.finalized_in.append(None)


@pytest.mark.plugin("image")
@pytest.mark.parametrize("left_by", ["the first sitting", "the step"])
def test_what_is_left_in_a_reference_cycle_is_collected_between_the_loops(run, monkeypatch, left_by):
    """With automatic collection off, the only collections are start_app's and the ones planted
    in a loop: the step's own check (inside its asyncio.run) and the rebuilt sitting's first act.
    Whatever the first sitting or the step leaves behind must be finalized while no loop runs."""
    _Leftover.finalized_in = []

    def leave_one(*_repl) -> None:
        _Leftover()

    real = generate_image_tool._TRANSPORT  # _fake_comfy's
    collected_in: list = []

    class _CollectingTransport(httpx.BaseTransport):  # image's check asks through a sync client
        def handle_request(self, request):
            collected_in.append(asyncio.get_running_loop())  # its check runs inside the step's loop
            gc.collect()
            return real.handle_request(request)

    gc.disable()
    try:
        collect = lambda _repl: gc.collect()  # noqa: E731 — the rebuilt sitting's first act
        if left_by == "the first sitting":  # the step's check collects inside its own loop
            monkeypatch.setattr(generate_image_tool, "_TRANSPORT", _CollectingTransport())
            s = run([["hello there", leave_one, "/plugins enable image"], [collect]])
        else:  # left at the step's question (no loop runs there); the rebuilt sitting collects
            s = run([["hello there", "/plugins enable image"], [collect]], at_prompt=leave_one)
    finally:
        gc.enable()
    assert s.runs == 2 and len(s.prompts) == 1
    assert f"✓ image: ComfyUI reachable at {COMFY}" in s.doctor.getvalue()  # the check really ran
    assert bool(collected_in) is (left_by == "the first sitting")  # planted only for that variant
    assert _Leftover.finalized_in == [None]
