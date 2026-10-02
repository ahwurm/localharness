"""Discord channel behaviour, pinned at the fake `discord` module boundary (Phase 49 Wave 0).

Captured from the pre-move tree. After the move the SAME golden must hold with only BUILDERS
changed; a diff is a finding, never a regenerate.

The pin sits at the discord.py API (`tests.dispatch_support.FakeDiscord`), not at
`DiscordChannel`'s API, so the class can disappear and the golden still means something. The
script is driven through public channel methods and the handlers the channel registers on the
fake client (`on_message`, `on_raw_reaction_add`). The ONLY private attributes touched — each
must exist on DispatchChannel after the move:
- `_pending_resolver` — installed as a recording async callable (the REPL installs it in prod);
- `_reaction_waiters`, `_pending_notices` — read (len only) after `stop()` to prove the open
  notice's waiter was dropped.

No normaliser: every id is a fake counter and the one measured duration (the timed-out ask's
"No answer within Ns") is formatted `:.0f` of a ~0.01s wait, so it reads `0s`.

`fake.log` records every outbound call; the script appends `("read", None, text)` for each
read_input return, `("decision", None, kind)` for each ask verdict, `("resolve", None, ...)`
for each resolver call and one `("state", None, ...)` after stop.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from localharness.agent.gate_types import PendingCall, PermissionRequest
from localharness.core.bus import EventBus
from tests.dispatch_support import (
    BOT_USER_ID, build_dispatch_discord, install_fake_discord, isolate_discord_env,
)

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "dispatch_plugin" / "discord_script.json"
REGEN = __import__("os").environ.get("LOCALHARNESS_REGEN_GOLDEN") == "1"

ALLOW = ("42", str(BOT_USER_ID))  # the bot's own id is allowlisted so the SELF filter is what drops it
CHANNELS = ("7", "8")


def _build_legacy(bus, *, allow, channels, ack):
    from localharness.channels.discord import DiscordChannel

    return DiscordChannel(bus, {"token": "t", "allow_users": list(allow),
                                "allow_channels": list(channels), "ack_emoji": ack})


BUILDERS = {"legacy": _build_legacy, "dispatch": build_dispatch_discord}


@pytest.fixture(autouse=True)
def fake(monkeypatch, tmp_path):
    isolate_discord_env(monkeypatch, tmp_path)
    return install_fake_discord(monkeypatch)


def _golden(path: Path, text: str) -> str:
    if REGEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path.read_text(encoding="utf-8")


async def _settle() -> None:
    for _ in range(6):
        await asyncio.sleep(0)


def _request(grantable: bool) -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash_exec", tool_params={"command": "cargo publish"},
        klass="shell-unfamiliar" if grantable else "shell-destructive",
        key="cargo publish" if grantable else None, grantable=grantable,
        reason="not seen in this workspace before",
        display="bash_exec: cargo publish  (shell-unfamiliar — not seen before)",
    )


def _pending(pending_id: int) -> PendingCall:
    rendering = f"bash_exec: rm -rf ~/old-notes-{pending_id}"
    return PendingCall(
        id=pending_id,
        request=PermissionRequest(
            tool_name="bash_exec", tool_params={"command": rendering}, klass="shell-destructive",
            key=None, grantable=False, reason="destructive command outside the project",
            display=rendering,
        ),
        rendering=rendering, agent_label="", session_id="sess-1", created_at=0.0,
    )


def _long_reply() -> str:
    """4500 chars: chunk 1 cut at a newline, chunk 2 at a space (the newline in reach sits below
    limit//2), the rest whole. The hard cut is the boundary-free 4500-char reply sent after it."""
    para = "word " * 199 + "end\n"  # 999 chars
    text = para * 3 + "x" * 500 + " " + "x" * 3000
    return text[:4500]


async def _ask(ch, fake, request, *, answers):
    """Ask, then answer with each (user, emoji) through the registered reaction listener."""
    task = asyncio.ensure_future(ch.ask_permission(request))
    await _settle()
    msg_id = fake.last_sent.id
    for user, emoji in answers:
        await fake.react(msg_id, user, emoji)
        await _settle()
    decision = await asyncio.wait_for(task, 1.0)
    fake.log.append(("decision", None, decision.kind))


async def _script(ch, fake) -> list:
    log = fake.log
    await ch.start()

    # --- inbound filters: none of these is ever enqueued
    await fake.deliver(fake.message(99, 7, "stranger"))                        # not allowlisted
    await fake.deliver(fake.message(42, 9, "wrong channel"))                   # channel not allowed
    await fake.deliver(fake.message(42, 7, "a bot", bot=True))                 # bot author
    await fake.deliver(fake.message(BOT_USER_ID, 7, "myself"))                 # self author
    await fake.deliver(fake.message(42, 7, "   \n\t "))                        # whitespace
    await fake.deliver(fake.message(42, 7, "", attachments=["a.png"]))          # attachment only
    first = fake.message(42, 7, "  hello there  \n")
    await fake.deliver(first)
    log.append(("read", None, await ch.read_input()))
    try:
        await asyncio.wait_for(ch.read_input(), 0.05)
        log.append(("read", None, "<unexpected>"))
    except asyncio.TimeoutError:
        log.append(("read", None, "<empty queue>"))

    # --- outbound text
    await ch.send_message(_long_reply())
    await ch.send_message("z" * 4500)  # no newline, no space: hard cuts at 2000
    await ch.send_message("short reply")
    await ch.send_error("boom")
    await ch.send_error("boom with detail", detail="d" * 700)
    await ch.send_tool_call("bash_exec", {"command": "ls"})
    await ch.send_tool_result("bash_exec", "out", False)
    await ch.send_tool_result("bash_exec", "err", True)
    await ch.on_heartbeat(None)

    # --- blocking asks
    await _ask(ch, fake, _request(True), answers=[(99, "✅"), (42, "👍"), (42, "✅")])
    await _ask(ch, fake, _request(True), answers=[(42, "♾️")])
    await _ask(ch, fake, _request(False), answers=[(42, "♾️"), (42, "❌")])
    try:
        await asyncio.wait_for(ch.ask_permission(_request(True)), 0.01)
    except asyncio.TimeoutError:
        log.append(("decision", None, "<timeout>"))
    await _settle()

    # --- parked calls (pending notices)
    async def resolver(action, pending_id):
        log.append(("resolve", None, f"{action} #{pending_id}"))

    ch._pending_resolver = resolver
    await ch.send_pending_notice(_pending(3), 1)
    notice_id = fake.last_sent.id
    await fake.react(notice_id, 99, "✅")  # stranger: ignored
    await _settle()
    await fake.react(notice_id, 42, "✅")
    await _settle()

    second = fake.message(42, 8, "second message")
    await fake.deliver(second)
    log.append(("read", None, await ch.read_input()))
    fake.fail_edits = True
    await ch.send_pending_notice(_pending(4), 1)
    await fake.react(fake.last_sent.id, 42, "❌")
    await _settle()
    fake.fail_edits = False

    # --- stop with a notice still open
    await ch.send_pending_notice(_pending(5), 2)
    await ch.stop()
    log.append(("state", None,
                f"waiters={len(ch._reaction_waiters)} notices={len(ch._pending_notices)}"))
    return [list(r) for r in log]


@pytest.mark.parametrize("builder", list(BUILDERS))
async def test_discord_script_matches_the_golden(builder, fake):
    ch = BUILDERS[builder](EventBus(), allow=ALLOW, channels=CHANNELS, ack="✅")
    records = await _script(ch, fake)
    text = json.dumps(records, indent=1, ensure_ascii=False) + "\n"
    assert json.loads(_golden(GOLDEN, text)) == records, "Discord behaviour drifted from the golden"


async def test_legacy_start_refusal_texts(fake, monkeypatch):
    """Today's three start() refusals, literal. 49-06 deletes this as a recorded deliberate delta."""
    from localharness.channels.discord import DiscordChannel
    from localharness.channels.errors import ChannelStartError

    def build(token="t", allow=("42",)):
        return DiscordChannel(EventBus(), {"token": token, "allow_users": list(allow)})

    for ch, expected in (
        (build(token=""), "Discord bot token missing — set LOCALHARNESS_DISCORD_TOKEN or DISCORD_BOT_TOKEN"),
        (build(allow=()), "Discord allowlist empty — set LOCALHARNESS_DISCORD_ALLOW to your user id(s); "
                          "refusing to listen to everyone"),
    ):
        with pytest.raises(ChannelStartError) as e:
            await ch.start()
        assert str(e.value) == expected

    monkeypatch.setitem(sys.modules, "discord", None)
    with pytest.raises(ChannelStartError) as e:
        await build().start()
    assert str(e.value) == ("discord.py not installed — run: uv pip install 'discord.py>=2.3' "
                            "(or install the 'dispatch' extra)")
    assert fake.log == [], "a refused start reached the client"


async def test_dispatch_start_refusal_texts(fake, monkeypatch):
    """The new refusals name the settings key first (the deliberate wording change); none carries
    the token, and a refused start never reaches the client."""
    from localharness.channels.errors import ChannelStartError
    from localharness.dispatch.adapters import ADAPTERS
    from localharness.dispatch.adapters.discord import DiscordAdapter
    from localharness.dispatch.channel import DispatchChannel

    assert ADAPTERS == {"discord": "localharness.dispatch.adapters.discord:DiscordAdapter"}
    assert (DiscordAdapter.platform, DiscordAdapter.title, DiscordAdapter.message_limit) == (
        "discord", "Discord", 2000)

    def build(token="sekrit-token", allow=("42",)):
        return DispatchChannel(EventBus(), {"adapter": DiscordAdapter(token=token), "allow": set(allow),
                                            "channels": set(), "ack": "✅", "state_dir": None})

    ch = build()
    assert ch.channel_id == "discord"
    assert ch.start_banner == "Dispatch mode: Discord — listening for allowlisted messages."
    for ch, expected in (
        (build(token=""), "Discord bot token missing — set dispatch.discord.token "
                          "(LOCALHARNESS_DISCORD_TOKEN / DISCORD_BOT_TOKEN still work until 0.17.0)"),
        (build(allow=()), "Discord allowlist empty — set dispatch.discord.allow to your user id(s) "
                          "(LOCALHARNESS_DISCORD_ALLOW still works until 0.17.0); "
                          "refusing to listen to everyone"),
    ):
        with pytest.raises(ChannelStartError) as e:
            await ch.start()
        assert str(e.value) == expected

    monkeypatch.setitem(sys.modules, "discord", None)
    with pytest.raises(ChannelStartError) as e:
        await build().start()
    assert str(e.value) == ("discord.py not installed — install the dispatch extra: uv sync --extra "
                            "dispatch (or pip install 'localharness[dispatch]')")
    assert "sekrit" not in str(e.value)
    assert fake.log == [] and fake.client is None, "a refused start reached the client"
