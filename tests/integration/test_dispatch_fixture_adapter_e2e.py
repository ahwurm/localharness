"""DISP-03: a second chat platform is one adapter inside the dispatch plugin — `start --channel
fixturechat` through the REAL start, with no core edit.

This proves a second platform is one adapter file + one ADAPTERS line + one manifest name inside
dispatch/, by shape, with a fixture — not a real second platform.

The ONLY registration (asserted below, `REGISTRATION`): `ADAPTERS["fixturechat"]` set to the
fixture's import string, and `BUILTIN_PLUGINS` with DispatchPlugin replaced by a subclass whose
manifest `channels` is `("discord", "fixturechat")` — the two edits a real second platform makes
inside dispatch/. Everything else is real: tier 1/tier 2 channel acceptance, the plugin's own
`make_channel` (NOT overridden), DispatchChannel, the generic construction in start, and the REAL
`OrchestratorREPL.run` loop with the real permission gate.

Named gap: DispatchPlugin has ONE settings section (`dispatch.discord.*`), and `make_channel` hands
every adapter `token=<discord token>` plus the Discord allow/channels/ack. So this fixture's
allow-list comes from `LOCALHARNESS_DISCORD_ALLOW`. A real second platform needs its own settings
section and a per-platform branch in `make_channel` — a plugin-internal edit, not a core one.

Stubbed boundaries as in test_dispatch_start_e2e: LLM probe, tokenizer, plugin discovery,
provider, `LLMClient.stream_complete` (scripted below). HOME is a tmp dir (isolate_discord_env).
"""
from __future__ import annotations

import asyncio

from tests.conftest import FakeLLMResponse, FakeToolCall
from tests.dispatch_support import isolate_discord_env
from tests.fixtures import dispatch_fixture_adapter as fixture
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

IMPORT = "tests.fixtures.dispatch_fixture_adapter:FixtureChatAdapter"
BANNER = "Dispatch mode: Fixturechat — listening for allowlisted messages."
LONG = "x" * 120
CONV = "room-1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
REGISTRATION: list[str] = []


async def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


def _register(monkeypatch) -> None:
    from localharness.dispatch.adapters import ADAPTERS
    from localharness.dispatch.plugin import DispatchPlugin
    from localharness.plugins import builtin

    class _FixturePlugin(DispatchPlugin):
        manifest = DispatchPlugin.manifest.model_copy(update={"channels": ("discord", "fixturechat")})

    monkeypatch.setitem(ADAPTERS, "fixturechat", IMPORT)
    REGISTRATION.append("ADAPTERS")
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", tuple(
        _FixturePlugin if p is DispatchPlugin else p for p in builtin.BUILTIN_PLUGINS))
    REGISTRATION.append("manifest.channels")


def _model():
    """`publish it` -> one bash_exec call the guarded gate asks about; after its result, a reply.
    `long` -> 120 chars; the act-guard nudge -> CONFIRMED; anything else -> `pong`."""
    async def model(self, messages, tools=None, on_token=None, **_):
        last = messages[-1]
        if last.get("role") == "tool":
            return FakeLLMResponse(content="ok, not publishing"), None
        text = str(next(m for m in reversed(messages) if m.get("role") == "user").get("content", ""))
        if "CONFIRMED" in text:  # the loop's act-guard nudge: keep the reply as it was
            return FakeLLMResponse(content="CONFIRMED"), None
        if "publish it" in text:
            return FakeLLMResponse(tool_calls=[FakeToolCall(
                id="c1", name="bash_exec", arguments={"command": "cargo publish"})]), None
        return FakeLLMResponse(content=LONG if "long" in text else "pong"), None
    return model


async def test_fixture_platform_runs_through_the_real_start(tmp_path, monkeypatch):
    isolate_discord_env(monkeypatch, tmp_path)
    monkeypatch.setenv("LOCALHARNESS_DISCORD_ALLOW", "42")  # the named gap: one settings section
    fixture.INSTANCES.clear()
    REGISTRATION.clear()
    _register(monkeypatch)

    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.artifacts import artifact_root, write_artifact
    from localharness.core.events import Observation, UserMessage
    from localharness.plugins import resolve

    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)
    real_run = OrchestratorREPL.run
    seen: list = []
    got: dict = {}

    def sends(pred=lambda t: True):
        return [r for r in got["a"].log if r[0] == "send" and pred(r[2])] if "a" in got else []

    async def say(author, text, until, what):
        got.setdefault("handles", []).append(await got["a"].push(author, CONV, text))
        await _wait_for(until, what)

    async def feeder(repl):
        await _wait_for(lambda: fixture.INSTANCES and fixture.INSTANCES[-1]._on_message, "connect")
        a = got["a"] = fixture.INSTANCES[-1]
        await a.push("43", CONV, "stranger here")                       # not allowlisted
        await say("42", "hello", lambda: sends(lambda t: t == "pong"), "the pong reply")
        await say("42", "long please", lambda: len(sends(lambda t: set(t) == {"x"})) == 3, "3 chunks")
        await say("42", "mode guarded", lambda: sends(lambda t: "guarded" in t), "guarded reply")
        await say("42", "publish it", lambda: sends(lambda t: "cargo publish" in t), "the ask")
        ask = sends(lambda t: "cargo publish" in t)[-1]
        await _wait_for(lambda: ("react", ask[3], "❌") in a.log, "the ask's reactions")
        await a.react_as("43", ask[3], "✅")                             # stranger: ignored
        await a.react_as("42", ask[3], "❌")
        await _wait_for(lambda: sends(lambda t: t == "ok, not publishing"), "the post-ask reply")
        ref = write_artifact(artifact_root(repl._channel._state_dir, "image"), "image", PNG, "image/png")
        got["png"] = artifact_root(repl._channel._state_dir, "image") / f"{ref.id}.png"
        await repl._bus.publish(Observation(agent_id="a", session_id="s", observation_type="tool_result",
                                            tool_name="generate_image", output="ok", artifact=ref))
        await _wait_for(lambda: any(r[0] == "file" for r in a.log), "the file reply")
        await say("42", "mode unattended", lambda: sends(lambda t: "unattended" in t), "unattended")
        got["mode"] = repl._session_gate().mode
        await a.push("42", CONV, "/quit")

    async def run(self):
        async def record(event):
            seen.append(event)
        self._bus.subscribe(UserMessage, record)
        async def guarded_feed():
            try:
                await feeder(self)
            except BaseException as e:  # surface the step, then end the real loop at once
                got["error"] = e
                if "a" in got:
                    await got["a"].push("42", CONV, "/quit")
        feed = asyncio.ensure_future(guarded_feed())
        try:
            await real_run(self)
        finally:
            feed.cancel()
            got["feed"] = feed

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=run)
    _offline_provider(tmp_path)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    printed = _capture_start_console(monkeypatch)
    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", _model())

    from localharness.cli.start_cmd import _start_async
    await asyncio.wait_for(_start_async(None, False, False, str(tmp_path), channel_mode="fixturechat"), 60)

    assert "error" not in got, (got.get("error"), got.get("a") and got["a"].log)
    assert REGISTRATION == ["ADAPTERS", "manifest.channels"]
    log = got["a"].log
    assert any(BANNER in p for p in printed), printed
    # the allow-list: the stranger's message never became a turn and never got an ack
    assert [(e.channel, e.content) for e in seen] == [
        ("fixturechat", "hello"), ("fixturechat", "long please"), ("fixturechat", "publish it")], seen
    acks = [r for r in log if r[0] == "react" and r[2] == "✅"]
    assert [r[1] for r in acks[:2]] == [h.id for h in got["handles"][:2]], log
    hello_ack = ("react", got["handles"][0].id, "✅")
    assert log.index(hello_ack) < log.index(sends(lambda t: t == "pong")[0]), log
    # chunking at the fixture's own limit
    chunks = [r[2] for r in sends(lambda t: set(t) == {"x"})]
    assert chunks == ["x" * 50, "x" * 50, "x" * 20], chunks
    # the ask was posted in the conversation, the stranger's ✅ ignored, the allowlisted ❌ decided
    assert sends(lambda t: "cargo publish" in t)[0][1] == CONV
    assert sends(lambda t: t == "ok, not publishing"), log
    # bare `mode <name>` (no slash) changed the mode through the channel's bare_mode_command
    assert sends(lambda t: "guarded" in t) and sends(lambda t: "unattended" in t), log
    assert got["mode"] == "unattended", got["mode"]
    # the artifact went out as a file to the conversation being answered
    assert [r for r in log if r[0] == "file"] == [("file", CONV, str(got["png"]), "image/png")], log
    assert log[-1] == ("close", None, "")
