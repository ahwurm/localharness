"""The told-about line: a reply that names a scratch file the turn made says where it lives and when it goes."""
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from localharness.agent.loop import base_system_prompt, render_index
from localharness.channels.base import ChannelAdapter, told_line
from localharness.config.models import AgentConfig
from localharness.core.events import TaskComplete
from localharness.core.run_folder import SCRATCH_SENTENCE, told
from tests.unit.test_agent_loop import _make_agent_loop

pytestmark = pytest.mark.skipif(os.name != "posix", reason="scratch folders")

CFG = AgentConfig(name="a", role="r")


def make(folder, name, *, ago=0.0):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    if ago:
        old = time.time() - ago
        os.utime(path, (old, old))
    return path


def complete(summary="", **kw):
    return TaskComplete(agent_id="a", session_id="s", success=True, summary=summary, duration_seconds=0.1, iterations=1, **kw)


def test_a_file_a_script_made_and_the_reply_names_is_told_and_an_older_one_or_an_unnamed_one_is_not(tmp_path):
    since = time.time() - 0.5
    make(tmp_path, "old.csv", ago=3600)
    subprocess.run([sys.executable, "-c", f"open({str(tmp_path / 'out.csv')!r}, 'w').write('x')"], check=True)
    make(tmp_path, "t3/plot.png")
    make(tmp_path, "unnamed.txt")
    assert told(tmp_path, since, "I made out.csv, plot.png and old.csv; README.md is in the project") == ("out.csv", "t3/plot.png")


def test_a_project_file_the_reply_names_is_never_told(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "README.md").write_text("x")
    assert told(scratch, 0.0, "see README.md") == ()


def test_the_walk_is_bounded_and_follows_no_link(tmp_path):
    outside = tmp_path / "outside"
    make(outside, "secret.txt")
    (tmp_path / "scratch").mkdir()
    (tmp_path / "scratch" / "link").symlink_to(outside)
    assert told(tmp_path / "scratch", 0.0, "secret.txt") == ()


def test_names_are_grouped_by_folder_with_each_folders_lifetime_and_quoted():
    line = told_line(("out.csv", "t3/plot.png", "t3/a.txt"))
    assert line == ('◆ scratch, gone when the session ends: "out.csv"; '
                    'scratch, gone when t3 is cleared or replaced, or the session ends: "plot.png", "a.txt"'
                    " · ask for a copy in your project to keep it")


def test_the_phone_says_chat_and_the_server_stops():
    assert told_line(("out.csv", "t3/a.txt"), phone=True, tail=False) == (
        '◆ scratch, gone when the chat ends or the server stops: "out.csv"; '
        'scratch, gone when t3 is cleared or replaced, the chat ends or the server stops: "a.txt"')


def test_more_than_five_names_end_and_n_more_and_none_is_no_line():
    assert told_line(()) is None
    line = told_line(tuple(f"f{i}.txt" for i in range(8)), tail=False)
    assert line.endswith('"f4.txt" and 3 more') and '"f5.txt"' not in line


def test_a_hostile_name_is_neutralised():
    line = told_line(("\x1b[2Jbad\n[bold]x.txt",), tail=False)
    assert "\x1b" not in line and "\n" not in line and '"bad [bold]x.txt"' in line
    assert line.count("◆") == 1


class Chan(ChannelAdapter):
    def __init__(self):
        self.sent = []

    async def send_message(self, content, agent_id=None, metadata=None):
        self.sent.append(content)

    async def read_input(self): return None
    async def send_streaming(self, *a, **kw): pass
    async def send_error(self, *a, **kw): pass
    async def send_tool_call(self, *a, **kw): pass
    async def send_tool_result(self, *a, **kw): pass
    async def start(self): pass
    async def stop(self): pass


async def test_the_channel_says_it_after_the_reply_and_the_tail_only_the_first_time():
    chan = Chan()
    await chan.on_task_complete(complete("made out.csv", scratch_files=("out.csv",)))
    await chan.on_task_complete(complete("made out.csv", scratch_files=("out.csv",)))
    await chan.on_task_complete(complete("nothing"))
    assert chan.sent[0] == "made out.csv" and chan.sent[1].endswith("keep it") and chan.sent[1].startswith("◆ scratch")
    assert chan.sent[2] == "made out.csv" and chan.sent[3] == '◆ scratch, gone when the session ends: "out.csv"'
    assert chan.sent[4] == "nothing" and len(chan.sent) == 5


async def test_acp_delivers_it_through_send_message(tmp_path):
    from tests.unit.channels.test_acp import _speaking

    ch = _speaking(tmp_path)
    await ch.on_task_complete(complete("made out.csv", scratch_files=("out.csv",)))
    assert ch._conn.texts == ["made out.csv", told_line(("out.csv",))]


async def test_the_phone_pushes_the_phone_line_after_the_reply_and_the_tail_only_once():
    from localharness.core.bus import EventBus
    from tests.unit.channels.test_mobile_channel import _channel, _drain

    bus = EventBus()
    channel = await _channel(bus)
    client = channel.attach_client()
    await bus.publish(complete("made out.csv", scratch_files=("out.csv",)))
    await bus.publish(complete("made out.csv", scratch_files=("out.csv",)))
    notices = [r[2]["text"] for r in _drain(client) if r[0] == "Notice"]
    assert notices == ['◆ scratch, gone when the chat ends or the server stops: "out.csv" · ask for a copy in your project to keep it',
                       '◆ scratch, gone when the chat ends or the server stops: "out.csv"']


def test_the_index_ends_with_the_scratch_line_and_the_old_sentence_goes_with_it(tmp_path):
    index = render_index(None, tmp_path, set(), False, False, scratch=tmp_path / "run" / "a" / "scratch")
    assert index.splitlines()[-1] == f"Scratch: {str(tmp_path / 'run' / 'a' / 'scratch')!r}. {SCRATCH_SENTENCE}"
    with_scratch = base_system_prompt(CFG, tool_call_mode="native", guardrails="", index=index)
    assert "create any files or folders you make" not in with_scratch and SCRATCH_SENTENCE in with_scratch
    plain = base_system_prompt(CFG, tool_call_mode="native", guardrails="", index=render_index(None, tmp_path, set(), False, False))
    assert "create any files or folders you make" in plain and SCRATCH_SENTENCE not in plain


async def test_the_launch_floor_index_carries_a_scratch_line(tmp_path, monkeypatch):
    from localharness.agent import loop
    from localharness.cli.start_serving import launch_floor

    seen = []
    real = loop.render_index
    monkeypatch.setattr(loop, "render_index", lambda *a, **kw: seen.append(kw) or real(*a, **kw))
    await launch_floor(CFG, (), machine_dir=tmp_path)
    assert seen[0]["scratch"] == tmp_path / "run" / ("0" * 12) / "scratch"


async def test_the_loop_publishes_the_scratch_files_the_reply_names(mock_llm_client, bus, tmp_path):
    out = make(tmp_path, "out.csv")
    future = time.time() + 600
    os.utime(out, (future, future))
    loop = _make_agent_loop(mock_llm_client, [mock_llm_client.Response(content="wrote out.csv")], bus)
    loop._gate = SimpleNamespace(scratch=Path(tmp_path))
    await loop.run_turn("task")
    event = bus.history(event_types=[TaskComplete])[-1]
    assert event.scratch_files == ("out.csv",)
