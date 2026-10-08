"""Discord's output and answer paths, hardened (F7, F8, F12, the Discord half of F13).

- ♾️ takes a second tap, as the phone's "always" does: the bot posts a confirm message pre-reacted
  ✅; ✅ on it records always, ✅ on the question records once, ❌ on the question says no.
- The bot's own reactions never answer anything — the gateway echoes every reaction the bot adds
  (the fake does too), and the allowlist may hold the bot's own id by mistake.
- Every message allows no @everyone, @here, role or user ping; a reply still notifies the person
  replied to. A masked link is sent as text with its address shown.
- The token comes from `dispatch.discord.token` or the two documented variables, never from
  Claude Code's `~/.claude/channels/discord/.env`; none at all refuses with one line naming
  `localharness plugins enable dispatch`.

Everything runs on the fake gateway (tests.dispatch_support); nothing logs in to Discord.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from localharness.agent.gate_types import PermissionRequest
from localharness.core.bus import EventBus
from tests.dispatch_support import (
    BOT_USER_ID, build_dispatch_discord, discord_events, install_fake_discord, isolate_discord_env,
)

ALLOW = ("42", str(BOT_USER_ID))  # the bot's own id allowlisted: only the self filter can drop it
EVIL = "see [click here](https://evil.example/x)"
SHOWN = "see click here (<https://evil.example/x>)"
TOKEN_MISSING = ("Discord bot token missing — run `localharness plugins enable dispatch` to set "
                 "dispatch.discord.token (~/.claude/channels/discord/.env is no longer read)")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    isolate_discord_env(monkeypatch, tmp_path)
    return install_fake_discord(monkeypatch)


@pytest.fixture
async def ch(fake):
    """A started DispatchChannel over the real adapter, answering in channel 7."""
    channel = build_dispatch_discord(EventBus(), allow=ALLOW, channels=(), ack="")
    await channel.start()
    await fake.deliver(fake.message(42, 7, "go"))
    await channel.read_input()
    yield channel
    await channel.stop()


async def _settle() -> None:
    for _ in range(8):
        await asyncio.sleep(0)


def _request(grantable: bool = True) -> PermissionRequest:
    return PermissionRequest(
        tool_name="bash_exec", tool_params={"command": "cargo publish"},
        klass="shell-unfamiliar" if grantable else "shell-destructive",
        key="cargo publish" if grantable else None, grantable=grantable,
        reason="not seen in this workspace before", display="bash_exec: cargo publish",
    )


def _reactions(fake, message) -> list[str]:
    return [emoji for op, target, emoji in fake.log if op == "react" and target == f"m{message.id}"]


async def _ask(ch, fake, request=None):
    task = asyncio.ensure_future(ch.ask_permission(request or _request()))
    await _settle()
    return task, fake.last_sent


async def _tap(fake, message, user, emoji) -> None:
    await fake.react(message.id, user, emoji)
    await _settle()


# ------------------------------------------------------------------- ♾️ takes a second tap

async def test_infinity_posts_a_confirm_and_only_its_tick_records_always(ch, fake):
    from localharness.dispatch.channel import PERMISSION_ALWAYS_CONFIRM

    task, question = await _ask(ch, fake)
    assert _reactions(fake, question) == ["✅", "♾️", "❌"]
    await _tap(fake, question, 42, "♾️")
    assert not task.done(), "one tap must never record a permanent grant"

    confirm = fake.last_sent
    assert confirm.id != question.id and confirm.content == PERMISSION_ALWAYS_CONFIRM
    assert _reactions(fake, confirm) == ["✅"]  # its echo came back as the bot's own: ignored
    await _tap(fake, confirm, BOT_USER_ID, "✅")
    assert not task.done(), "the bot's own ✅ on the confirm answered the question"

    await _tap(fake, confirm, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_always"
    assert ch._reaction_waiters == {}, "a waiter outlived the question"


async def test_a_tick_on_the_question_after_infinity_is_once(ch, fake):
    from localharness.dispatch.channel import PERMISSION_ALWAYS_CONFIRM

    task, question = await _ask(ch, fake)
    await _tap(fake, question, 42, "♾️")
    assert not task.done() and fake.last_sent.content == PERMISSION_ALWAYS_CONFIRM
    await _tap(fake, question, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_once"
    assert ch._reaction_waiters == {}


async def test_a_cross_on_the_question_after_infinity_is_no(ch, fake):
    task, question = await _ask(ch, fake)
    await _tap(fake, question, 42, "♾️")
    await _tap(fake, question, 42, "❌")
    assert (await asyncio.wait_for(task, 1.0)).kind == "reject_once"


async def test_a_second_infinity_posts_no_second_confirm(ch, fake):
    from localharness.dispatch.channel import PERMISSION_ALWAYS_CONFIRM

    task, question = await _ask(ch, fake)
    await _tap(fake, question, 42, "♾️")
    await _tap(fake, question, 42, "♾️")
    confirms = [r for r in fake.log if r[0] == "send" and r[2] == PERMISSION_ALWAYS_CONFIRM]
    assert len(confirms) == 1
    assert not task.done()
    await _tap(fake, fake.last_sent, 42, "❌")  # anything but ✅ on the confirm answers nothing
    assert not task.done()
    await _tap(fake, fake.last_sent, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_always"


async def test_an_ungrantable_question_offers_no_infinity_and_posts_no_confirm(ch, fake):
    task, question = await _ask(ch, fake, _request(grantable=False))
    assert _reactions(fake, question) == ["✅", "❌"]
    sends = len([r for r in fake.log if r[0] == "send"])
    await _tap(fake, question, 42, "♾️")
    assert len([r for r in fake.log if r[0] == "send"]) == sends and not task.done()
    await _tap(fake, question, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_once"


async def test_through_the_gate_the_second_tap_writes_the_grant(ch, fake, tmp_path):
    """The default lock: Discord still grants "always" — with the confirm tap — and the grant is
    durable, so the same call does not ask again."""
    from localharness.agent.gate import PermissionGate
    from localharness.agent.gate_types import ToolMeta
    from localharness.config.grants import GrantStore

    workspace = tmp_path / "project"
    workspace.mkdir()
    gate = PermissionGate(boundary=workspace, workspace=workspace, mode="guarded",
                          grants=GrantStore(tmp_path / "grants.yaml"))
    gate.attach_channel(ch)

    def check():
        return asyncio.ensure_future(gate.check("bash_exec", {"command": "cargo build --release"},
                                                ToolMeta(group="shell"), agent_id="a", session_id="s"))
    first = check()
    await _settle()
    await _tap(fake, fake.last_sent, 42, "♾️")
    await _tap(fake, fake.last_sent, 42, "✅")
    assert (await asyncio.wait_for(first, 1.0)).allowed
    sends = len([r for r in fake.log if r[0] == "send"])
    assert (await asyncio.wait_for(check(), 1.0)).allowed
    assert len([r for r in fake.log if r[0] == "send"]) == sends, "a confirmed always asked again"


async def test_the_deadline_still_bounds_a_confirm_in_progress(ch, fake, tmp_path):
    from localharness.agent.gate import PermissionGate
    from localharness.agent.gate_types import GateSettings, ToolMeta
    from localharness.config.grants import GrantStore
    from localharness.dispatch.channel import PERMISSION_TIMEOUT_LINE

    workspace = tmp_path / "project"
    workspace.mkdir()
    gate = PermissionGate(boundary=workspace, workspace=workspace, mode="guarded",
                          grants=GrantStore(tmp_path / "grants.yaml"),
                          settings=GateSettings(ask_timeout_s=0.2))
    gate.attach_channel(ch)
    task = asyncio.ensure_future(gate.check("bash_exec", {"command": "cargo publish"},
                                            ToolMeta(group="shell"), agent_id="a", session_id="s"))
    await _settle()
    question = fake.last_sent
    await _tap(fake, question, 42, "♾️")
    outcome = await asyncio.wait_for(task, 2.0)
    assert not outcome.allowed and "no answer" in outcome.reason
    assert ch._reaction_waiters == {}, "the confirm's waiter leaked past the deadline"
    assert PERMISSION_TIMEOUT_LINE.split("{")[0] in question.content
    assert not (tmp_path / "grants.yaml").exists()


async def test_a_cancel_while_the_confirm_is_being_reacted_leaks_no_waiter(ch, fake, monkeypatch):
    """The gate's deadline can land while the bot is still adding ✅ to the confirm message: both
    waiters must still go, or a dead question keeps a listener for as long as the session runs."""
    from localharness.dispatch.channel import PERMISSION_ALWAYS_CONFIRM

    task, question = await _ask(ch, fake)
    real_react = ch._adapter.react
    stuck = asyncio.Event()

    async def react(message, emoji):
        if message.content == PERMISSION_ALWAYS_CONFIRM:
            stuck.set()
            await asyncio.sleep(3600)  # a gateway that never answers
        await real_react(message, emoji)

    monkeypatch.setattr(ch._adapter, "react", react)
    await _tap(fake, question, 42, "♾️")
    await asyncio.wait_for(stuck.wait(), 1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ch._reaction_waiters == {}, "the confirm's waiter outlived the cancelled question"


async def test_a_confirm_that_cannot_be_posted_leaves_the_question_open(ch, fake, monkeypatch):
    from localharness.dispatch.channel import PERMISSION_ALWAYS_CONFIRM

    task, question = await _ask(ch, fake)
    real_send = ch._adapter.send
    failures = []

    async def send(conversation, text):
        if text == PERMISSION_ALWAYS_CONFIRM and not failures:
            failures.append(text)
            raise RuntimeError("gateway hiccup")
        return await real_send(conversation, text)

    monkeypatch.setattr(ch._adapter, "send", send)
    await _tap(fake, question, 42, "♾️")
    assert failures and not task.done(), "a failed confirm post must not end the question"
    await _tap(fake, question, 42, "♾️")  # tries again
    assert fake.last_sent.content == PERMISSION_ALWAYS_CONFIRM
    await _tap(fake, fake.last_sent, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_always"


# ------------------------------------------------------------------- the bot's own reactions

async def test_the_adapter_drops_the_bots_own_reaction(ch, fake):
    handler = discord_events(ch)["on_raw_reaction_add"]
    waiter: asyncio.Queue = asyncio.Queue()
    ch._reaction_waiters[7] = waiter
    await handler(types.SimpleNamespace(user_id=BOT_USER_ID, message_id=7, emoji="✅"))
    await handler(types.SimpleNamespace(user_id=str(BOT_USER_ID), message_id=7, emoji="✅"))
    assert waiter.empty(), "the bot's own reaction reached the channel"
    await handler(types.SimpleNamespace(user_id=42, message_id=7, emoji="✅"))
    assert waiter.get_nowait() == "✅"


async def test_an_open_question_survives_the_bots_own_tick(ch, fake):
    task, question = await _ask(ch, fake)
    await _tap(fake, question, BOT_USER_ID, "✅")
    assert not task.done(), "the bot answered its own question"
    await _tap(fake, question, 42, "✅")
    assert (await asyncio.wait_for(task, 1.0)).kind == "allow_once"


# ------------------------------------------------------------------- no pings, no masked links

async def test_the_client_allows_no_mass_role_or_user_pings(ch, fake):
    mentions = fake.client.allowed_mentions
    assert mentions is not None, "the client was built with discord.py's default (every ping on)"
    assert mentions.kwargs == {"everyone": False, "users": False, "roles": False, "replied_user": True}


async def test_send_edit_and_reply_show_a_masked_links_address(ch, fake):
    adapter = ch._adapter
    sent = await adapter.send(fake.channel(7), EVIL)
    await adapter.edit(sent, EVIL)
    await adapter.reply(sent, EVIL)
    assert fake.log[-3:] == [("send", "c7", SHOWN), ("edit", f"m{sent.id}", SHOWN),
                             ("reply", f"m{sent.id}", SHOWN)]
    await ch.send_message(f"answer: {EVIL}")
    assert fake.log[-1] == ("send", "c7", f"answer: {SHOWN}")


@pytest.mark.parametrize("text, shown", [
    ("no link here, [brackets] and (parens) https://ok.example", None),
    ("`code[0](x)` stays", None),
    (EVIL, SHOWN),
    ("[a](https://a.example) and [b](http://b.example/p?q=1)",
     "a (<https://a.example>) and b (<http://b.example/p?q=1>)"),
    ("[x](<https://evil.example>)", "x (<https://evil.example>)"),
    ("[x]( https://evil.example )", "x (<https://evil.example>)"),
    ("[w](https://en.wikipedia.org/wiki/Foo_(bar))", "w (<https://en.wikipedia.org/wiki/Foo_(bar)>)"),
    ("[X](HTTPS://EVIL.EXAMPLE)", "X (<HTTPS://EVIL.EXAMPLE>)"),
    ("[a [b] c](https://evil.example)", "[a [b] c] (https://evil.example)"),
    ("[two\nlines](https://evil.example)", "[two\nlines] (https://evil.example)"),
])
def test_plain_links(text, shown):
    from localharness.dispatch.adapters.discord import plain_links

    assert plain_links(text) == (text if shown is None else shown)
    assert "](http" not in plain_links(text).lower() and "](<http" not in plain_links(text).lower()


# ------------------------------------------------------------------- its own token only

def _claude_code_env_file(home: Path) -> None:
    env = home / ".claude" / "channels" / "discord" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("DISCORD_BOT_TOKEN=tok-from-claude-code\n")


async def test_claude_codes_env_file_is_never_read(fake, tmp_path):
    """The plugin's real settings path (os.environ, Path.home()) with only the other program's
    file holding a token: the token stays empty and start refuses, naming the setup command."""
    from localharness.channels.errors import ChannelStartError
    from localharness.dispatch.config import DispatchConfig
    from localharness.dispatch.plugin import DispatchPlugin, _effective
    from localharness.plugins.api import PluginContext, PluginPaths
    from localharness.tools.registry import ToolRegistry

    _claude_code_env_file(Path.home())
    ctx = PluginContext(bus=EventBus(), tools=ToolRegistry(), hooks=None,
                        config=DispatchConfig(discord={"allow": ["42"]}), agent_config=None,
                        paths=PluginPaths(global_config_dir=tmp_path / "g", workspace=None,
                                          state_dir=tmp_path / "s"), llm=None)
    settings = _effective(ctx)
    assert settings.token.get_secret_value() == ""

    plugin = DispatchPlugin()
    await plugin.start(ctx)
    channel = plugin.make_channel("discord", EventBus())
    with pytest.raises(ChannelStartError) as e:
        await channel.start()
    assert str(e.value) == TOKEN_MISSING
    assert fake.client is None and fake.log == [], "a refused start reached the client"


async def test_no_token_refuses_in_one_line_naming_the_setup_command(fake):
    from localharness.channels.errors import ChannelStartError
    from localharness.dispatch.adapters.discord import DiscordAdapter
    from localharness.dispatch.channel import DispatchChannel

    channel = DispatchChannel(EventBus(), {"adapter": DiscordAdapter(token=""), "allow": {"42"},
                                           "channels": set(), "ack": "✅", "state_dir": None})
    with pytest.raises(ChannelStartError) as e:
        await channel.start()
    assert str(e.value) == TOKEN_MISSING and "\n" not in str(e.value)
