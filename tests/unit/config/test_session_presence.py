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


def test_start_up_registers_presence_for_every_channel():
    """One hook, in the function all four channels funnel through — not a web-only feature. The
    terminal-vs-web collision is the one the owner will actually hit."""
    import inspect

    from localharness.cli import start_cmd

    source = inspect.getsource(start_cmd._start_async)
    assert "session_presence.register" in source
    assert "send_error" in source[source.index("session_presence.register"):][:900]
