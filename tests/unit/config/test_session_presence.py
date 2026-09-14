"""WEBCH-29: two live sessions on one agent are noticed, named, and NOT refused.

The hazard being made visible is real and latent today: `history.jsonl` and `compact.md` take
unlocked appends, so a phone-driven session and a terminal session on the same agent interleave
their writes. The owner's ruling is warn-don't-refuse, which is why every test here asserts that
the second session keeps its start-up.
"""
from __future__ import annotations

import json
import os

from localharness.config import session_presence


def test_a_lone_session_finds_nobody(tmp_path):
    others = session_presence.register(
        tmp_path, agent="orchestrator", channel="web", session_id="s1", workspace=tmp_path)
    assert others == []


def test_a_second_session_on_the_same_agent_is_detected_and_named(tmp_path):
    """The acceptance criterion: start the terminal while the web channel serves, and the newcomer
    says what else is running — naming it, because "something else is running" sends a person
    hunting through `ps` for the thing this already knew."""
    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=tmp_path, pid=os.getpid())
    others = session_presence.register(
        tmp_path, agent="orchestrator", channel="terminal", session_id="s2",
        workspace=tmp_path, pid=os.getpid() + 100000)

    assert len(others) == 1
    assert others[0].channel == "web"
    text = session_presence.warning(others, agent="orchestrator")
    assert "web" in text and str(os.getpid()) in text
    assert "orchestrator" in text
    # It has to say it is not stopping anybody.
    assert "not a refusal" in text
    assert "history.jsonl" in text


def test_a_different_agent_is_not_a_co_tenant(tmp_path):
    """Two agents in one workspace write different files. Warning about them is noise, and noise
    is how a real warning gets ignored."""
    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=tmp_path, pid=os.getpid())
    others = session_presence.register(tmp_path, agent="cruncher", channel="terminal",
                                       session_id="s2", workspace=tmp_path,
                                       pid=os.getpid() + 100000)
    assert others == []


def test_a_different_workspace_is_not_a_co_tenant(tmp_path):
    one, two = tmp_path / "proj-a", tmp_path / "proj-b"
    one.mkdir()
    two.mkdir()
    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=one, pid=os.getpid())
    others = session_presence.register(tmp_path, agent="orchestrator", channel="terminal",
                                       session_id="s2", workspace=two,
                                       pid=os.getpid() + 100000)
    assert others == []


def test_the_same_workspace_by_a_symlinked_path_is_a_co_tenant(tmp_path):
    """Realpath, like the trust and grant stores. Two sessions reaching one `history.jsonl` by
    different names is exactly the case this exists to catch."""
    real = tmp_path / "project"
    real.mkdir()
    link = tmp_path / "shortcut"
    link.symlink_to(real)

    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=real, pid=os.getpid())
    others = session_presence.register(tmp_path, agent="orchestrator", channel="terminal",
                                       session_id="s2", workspace=link,
                                       pid=os.getpid() + 100000)
    assert len(others) == 1


def test_a_dead_session_is_pruned_not_reported(tmp_path):
    """Nothing is deleted on exit — liveness is asked of the OS, which is also the right answer
    after a SIGKILL. A stale entry that kept warning forever would train people to ignore it."""
    directory = session_presence.presence_dir(tmp_path)
    directory.mkdir(parents=True)
    dead_pid = 2 ** 22   # above any real pid_max on this box
    (directory / f"{dead_pid}.json").write_text(json.dumps({
        "pid": dead_pid, "agent": "orchestrator", "channel": "web",
        "workspace": str(tmp_path), "session_id": "old", "started_at": 0.0,
    }))

    others = session_presence.register(tmp_path, agent="orchestrator", channel="terminal",
                                       session_id="s2", workspace=tmp_path)
    assert others == []
    assert not (directory / f"{dead_pid}.json").exists(), "the stale entry should be pruned"


def test_a_corrupt_entry_is_discarded_rather_than_crashing_startup(tmp_path):
    directory = session_presence.presence_dir(tmp_path)
    directory.mkdir(parents=True)
    (directory / "9999999.json").write_text("{not json at all")

    assert session_presence.register(tmp_path, agent="a", channel="web", session_id="s",
                                     workspace=tmp_path) == []


def test_an_unwritable_registry_never_fails_a_start(tmp_path, monkeypatch):
    """A lost warning is a lost warning. It is not a reason to refuse to start a session."""
    def boom(*args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("pathlib.Path.mkdir", boom)
    assert session_presence.register(tmp_path, agent="a", channel="web", session_id="s",
                                     workspace=tmp_path) == []


def test_release_removes_only_this_session(tmp_path):
    session_presence.register(tmp_path, agent="a", channel="web", session_id="s1",
                              workspace=tmp_path, pid=os.getpid())
    session_presence.release(tmp_path, pid=os.getpid())
    assert session_presence.live_sessions(tmp_path) == []


def test_the_summary_is_serialisable_for_the_health_endpoint(tmp_path):
    """`GET /api/health` carries this, because the warning itself is a one-shot line on the wire
    and a one-shot line is invisible to a phone that connected afterwards."""
    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=tmp_path, pid=os.getpid())
    rows = session_presence.summary(session_presence.live_sessions(tmp_path))
    assert json.loads(json.dumps(rows))[0]["channel"] == "web"


class _FakeChannel:
    """Records what the start-up hook actually did to a channel."""

    def __init__(self) -> None:
        self.co_tenants = None
        self.rescan = None
        self.errors: list[str] = []

    def set_co_tenants(self, others, rescan=None):
        self.co_tenants = others
        self.rescan = rescan

    async def send_error(self, error, detail=None, agent_id=None):
        self.errors.append(error)


async def test_the_startup_hook_warns_and_hands_the_channel_the_co_tenants(tmp_path):
    """Driven through the REAL hook, not by matching strings in its source.

    The string-matching version of this test passed while `set_co_tenants` was deleted — the
    line had no test at all, and `GET /api/health` would have reported no co-tenants forever.
    """
    from localharness.cli.start_cmd import _announce_presence

    # pid 1 is alive and is not us — `register` excludes only the CALLING process, which is
    # what makes "another session" mean another process rather than this one twice.
    session_presence.register(tmp_path, agent="orchestrator", channel="web", session_id="s1",
                              workspace=tmp_path, pid=1)
    channel = _FakeChannel()
    await _announce_presence(channel, config_dir=str(tmp_path), agent="orchestrator",
                             channel_mode="terminal", session_id="s2", workspace=tmp_path)

    assert channel.co_tenants and channel.co_tenants[0].channel == "web"
    assert len(channel.errors) == 1
    assert "ANOTHER SESSION IS LIVE" in channel.errors[0]
    assert "not a refusal" in channel.errors[0]


async def test_the_hook_hands_over_a_rescan_that_sees_a_session_that_joined_later(tmp_path):
    """WEBCH-29. The snapshot is taken before the second session exists — which means the FIRST
    session is precisely the one that can never be told about it, and that is backwards: it is
    the first session's `history.jsonl` the second one interleaves with.
    """
    from localharness.cli.start_cmd import _announce_presence

    channel = _FakeChannel()
    await _announce_presence(channel, config_dir=str(tmp_path), agent="orchestrator",
                             channel_mode="web", session_id="s1", workspace=tmp_path)
    assert channel.co_tenants == []

    # Somebody opens a terminal on the same agent afterwards.
    session_presence.register(tmp_path, agent="orchestrator", channel="terminal",
                              session_id="s2", workspace=tmp_path, pid=1)

    assert channel.rescan is not None, "no rescan was handed over; health stays frozen forever"
    later = channel.rescan()
    assert [s.channel for s in later] == ["terminal"]


async def test_the_startup_hook_is_silent_when_nobody_else_is_there(tmp_path):
    """The commonest case by far. A warning that fires every time is a warning nobody reads."""
    from localharness.cli.start_cmd import _announce_presence

    channel = _FakeChannel()
    await _announce_presence(channel, config_dir=str(tmp_path), agent="orchestrator",
                             channel_mode="web", session_id="s1", workspace=tmp_path)

    assert channel.errors == []
    assert channel.co_tenants == []


async def test_the_startup_hook_survives_a_channel_that_cannot_warn(tmp_path):
    """ACP and a fixture channel are not obliged to implement every optional hook."""
    from localharness.cli.start_cmd import _announce_presence

    class _Minimal:
        pass

    await _announce_presence(_Minimal(), config_dir=str(tmp_path), agent="a",
                             channel_mode="acp", session_id="s", workspace=tmp_path)
    # Registered anyway — the NEXT session's warning depends on this one having written.
    assert len(session_presence.live_sessions(tmp_path)) == 1


def test_start_up_calls_the_presence_hook():
    """The seam is only worth testing if production actually reaches it."""
    import inspect

    from localharness.cli import start_cmd

    assert "_announce_presence(" in inspect.getsource(start_cmd._start_async)
