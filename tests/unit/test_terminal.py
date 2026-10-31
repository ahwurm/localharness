"""The lever table per channel, a finished notice keeping its mark, and which channels show a status row."""
from __future__ import annotations

from io import StringIO

import pytest

from localharness.channels.acp import AcpChannel
from localharness.channels.mobile.channel import MobileChannel
from localharness.core.bus import EventBus
from tests.unit.test_terminal_channel import make_terminal_channel

BOX = ("/task stop (after this unit)", "Ctrl+C (now)")
NOTICE = "◆ t1 done · /task show review"


def on_the_box(channel, flag=True):
    channel._box_app, channel._box_active = object() if flag else None, flag
    return channel


def test_the_box_names_the_task_stop_and_ctrl_c():
    channel = on_the_box(make_terminal_channel(StringIO()))
    assert channel.levers() == BOX and channel.has_status_row() is True


def test_the_classic_terminal_names_ctrl_c_for_both():
    channel = make_terminal_channel(StringIO())
    assert channel.levers() == ("Ctrl+C (now)", "Ctrl+C (now)") and channel.has_status_row() is False


def test_a_box_that_is_not_active_is_the_classic_path():
    channel = on_the_box(make_terminal_channel(StringIO()), flag=False)
    assert channel.levers() == ("Ctrl+C (now)", "Ctrl+C (now)") and channel.has_status_row() is False


def test_the_phone_says_stop_and_shows_a_status_row():
    channel = MobileChannel(bus=EventBus(), config={})
    assert channel.levers() == ("stop (now)", "stop (now)") and channel.has_status_row() is True


def test_acp_says_cancel_and_shows_no_status_row():
    channel = AcpChannel()
    assert channel.levers() == ("cancel (now)", "cancel (now)") and channel.has_status_row() is False


@pytest.mark.asyncio
async def test_send_notice_prints_the_mark_and_send_message_neutralises_it():
    out = StringIO()
    channel = make_terminal_channel(out, force_terminal=False)
    await channel.send_notice(NOTICE)
    assert out.getvalue().strip() == NOTICE
    out.seek(0), out.truncate()
    await channel.send_message(NOTICE)
    assert "◆" not in out.getvalue() and "t1 done · /task show review" in out.getvalue()


@pytest.mark.asyncio
async def test_send_notice_does_not_open_a_bracket_as_markup():
    out = StringIO()
    channel = make_terminal_channel(out, force_terminal=False)
    await channel.send_notice("◆ t1 [bold]x[/bold] · /task run")
    assert out.getvalue().strip() == "◆ t1 [bold]x[/bold] · /task run"


@pytest.mark.asyncio
async def test_panel_started_prints_off_the_box_and_stays_quiet_on_it():
    out = StringIO()
    channel = make_terminal_channel(out, force_terminal=False)
    await channel.panel_started("panel line")
    assert out.getvalue().strip() == "panel line"
    out.seek(0), out.truncate()
    await on_the_box(channel).panel_started("panel line")
    assert out.getvalue() == ""


def completed(**kw):
    from localharness.core.events import TaskComplete

    return TaskComplete(agent_id="orchestrator", session_id="s", success=True, summary="the answer",
                        duration_seconds=0.1, iterations=1, **kw)


def test_reply_line_of_a_blocked_root_reply_names_the_show_command_and_the_finding():
    from localharness.channels.base import reply_line

    assert reply_line(completed(verdict="blocked", finding="README.md:41 does not say that")) == \
        "◆ reply blocked · /task show reply · README.md:41 does not say that"


def test_reply_line_of_an_unreviewed_reply_needs_a_reason_and_a_cut_finding_fits_the_line():
    from localharness.channels.base import reply_line
    from localharness.core.caps import Caps

    assert reply_line(completed(verdict="unreviewed", finding="the critic ran out of time")) == \
        "◆ reply unreviewed · /task show reply · the critic ran out of time"
    for event in (completed(verdict="unreviewed", finding=""), completed(verdict="pass", finding="x"), completed()):
        assert reply_line(event) is None
    long = reply_line(completed(verdict="blocked", finding="w" * 300))
    assert long.startswith("◆ reply blocked · /task show reply · www") and len(long) == Caps.LINE


def test_reply_line_of_a_childs_completion_is_none():
    from localharness.channels.base import reply_line

    assert reply_line(completed(verdict="blocked", finding="x").model_copy(update={"parent_id": "root-1"})) is None


@pytest.mark.asyncio
async def test_reply_line_follows_the_reply_on_the_terminal_with_its_mark():
    out = StringIO()
    channel = make_terminal_channel(out, force_terminal=False)
    await channel.on_task_complete(completed(verdict="blocked", finding="README.md:41 does not say that"))
    lines = [line.strip() for line in out.getvalue().splitlines() if line.strip()]
    assert lines[-1] == "◆ reply blocked · /task show reply · README.md:41 does not say that"
    assert any("the answer" in line for line in lines[:-1])
