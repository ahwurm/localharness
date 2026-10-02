"""`start --channel discord` configured by env only, through the REAL `_start_async`, offline.

Pinned on the pre-move tree (Phase 49 Wave 0); the assertions must survive the move unedited.
This module never imports the old channel module, so it does not care where Discord lives.

Drive: the REAL `OrchestratorREPL.run` (classic loop: `channel.start()` -> `read_input()` ->
`_dispatch_input` -> `UserMessage` publish -> turn -> `channel.stop()`). It is wrapped only to
subscribe a recorder to `UserMessage` on the REPL's own bus and to start a feeder that plays the
fake gateway: one allowlisted "hello" in channel 7, then — once the reply has been posted — an
allowlisted `/quit`, which is how the real loop ends (EOFError out of `_dispatch_input`).

Boundaries stubbed: LLM probe, tokenizer, plugin discovery (`_stub_start_boundaries`), provider at
the discard port, `LLMClient.stream_complete` -> "pong from the model", and the `discord` module
(`tests.dispatch_support`). `extra_installed` is patched True now so that after the cut, when the
dispatch plugin needs its extra, this test passes with the same assertions. HOME is a tmp dir;
the real Discord token file under the owner home is unreachable. No warning-absence is asserted (49-06
adds deprecation lines for the env path).
"""
from __future__ import annotations

import asyncio

from tests.conftest import FakeLLMResponse
from tests.dispatch_support import install_fake_discord, isolate_discord_env
from tests.integration.test_guardrails_from_global_dir_e2e import _let_the_stub_tokenizer_run_a_turn
from tests.integration.test_workspace_cli_surface_e2e import _offline_provider
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

ENV = {
    "LOCALHARNESS_DISCORD_TOKEN": "tkn",
    "LOCALHARNESS_DISCORD_ALLOW": "42",
    "LOCALHARNESS_DISCORD_CHANNELS": "7",
}
REPLY = "pong from the model"
BANNER = "Dispatch mode: Discord — listening for allowlisted messages."


async def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.01)


async def test_env_only_discord_start_turns_one_message_into_one_reply(tmp_path, monkeypatch):
    isolate_discord_env(monkeypatch, tmp_path)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    fake = install_fake_discord(monkeypatch)

    from localharness.cli.repl import OrchestratorREPL
    from localharness.core.events import UserMessage
    from localharness.plugins import resolve

    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: True)
    real_run = OrchestratorREPL.run
    seen: list = []
    hello = fake.message(42, 7, "hello")

    async def feeder():
        await _wait_for(lambda: fake.client is not None and "on_message" in fake.client.events,
                        "the channel to register its gateway handlers")
        await fake.deliver(hello)
        await _wait_for(lambda: ("send", "c7", REPLY) in fake.log, "the reply to be posted")
        await fake.deliver(fake.message(42, 7, "/quit"))

    async def run(self):
        async def record(event):
            seen.append(event)
        self._bus.subscribe(UserMessage, record)
        feed = asyncio.ensure_future(feeder())
        try:
            await real_run(self)
        finally:
            feed.cancel()

    _stub_start_boundaries(tmp_path, monkeypatch, repl_run=run)
    _offline_provider(tmp_path)
    _let_the_stub_tokenizer_run_a_turn(monkeypatch)
    printed = _capture_start_console(monkeypatch)

    async def model(self, messages, tools=None, on_token=None, **_):
        return FakeLLMResponse(content=REPLY), None

    monkeypatch.setattr("localharness.provider.client.LLMClient.stream_complete", model)

    from localharness.cli.start_cmd import _start_async
    await asyncio.wait_for(
        _start_async(None, False, False, str(tmp_path), channel_mode="discord"), 60
    )

    assert [(e.channel, e.content) for e in seen] == [("discord", "hello")], seen
    ack = ("react", f"m{hello.id}", "✅")
    assert ack in fake.log and ("send", "c7", REPLY) in fake.log, fake.log
    assert fake.log.index(ack) < fake.log.index(("send", "c7", REPLY)), fake.log
    assert [r for r in fake.log if r[0] == "send" and r[2] == REPLY] == [("send", "c7", REPLY)]
    assert any(BANNER in p for p in printed), printed
