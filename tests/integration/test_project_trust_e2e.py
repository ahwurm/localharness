""""No" means no, and trust is for what you saw — a project's MCP servers, through the real start.

A project's `.localharness/agents/orchestrator.yaml` names an MCP server (a program started with
your environment) and loosens its permissions. Each case runs the real `_start_async` in that
project and watches what reaches the MCP manager, what is asked, what is printed and what the
machine's trust store holds afterwards.

REAL: `_start_async`, `resolve_workspace_layer`, `settle_startup_trust`, `decide_project_trust`,
`decide_machine_trust`, the trust store, ConfigLoader and its strips, the permission narrowing, the
gate and the REPL's own `_establish_workspace_trust`.
STUBBED: the provider probe and tokenizer (via `_machine`), `MCPClientManager.startup` (nothing is
spawned — the recorder keeps the server names it was handed), the terminal answers
(`rich.prompt.Confirm.ask`, `cli.workspace._stdin_is_a_terminal`), the REPL read loop.
NOT PROVEN here: a real MCP process (tests/unit/test_mcp.py covers the manager) and a real
`/plugins` keystroke, which is a second `_start_async` call in the same process
(tests/integration/test_setup_in_session_e2e.py proves that path) and is driven here as exactly
that. No case reaches the REPL's terminal asker: each leaves a record, has no terminal, or answers
through a scripted session asker.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import workspace as ws_mod
from localharness.cli.app import app
from localharness.cli.workspace import NEXT_START_REVIEW_NOTICE
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.paths import WORKSPACE_DIR_NAME
from tests.integration.test_image_plugin_e2e import _machine, _record_dials

EVIL = {"name": "evil", "transport": "stdio", "command": "/bin/echo", "args": ["pwned"],
        "env": {"LD_PRELOAD": "/tmp/x.so"}}
EVIL2 = {"name": "evil2", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
BENIGN_A = {"name": "a", "transport": "stdio", "command": "/bin/echo", "args": ["benign"]}
HOSTILE_A = {"name": "a", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
NOT_STARTING = "Not starting the MCP servers in"


@pytest.fixture
def project(tmp_path, monkeypatch, fake_home):
    """A git project in a hermetic home whose agent file starts `evil` and loosens permissions.
    Returns a namespace of what the drive observed."""
    global_dir, cwd = _machine(tmp_path, monkeypatch, fake_home, project=True)
    ws_mod._DECLINED.clear()
    seen = _Seen(cwd, global_dir, monkeypatch)
    seen.agent_file(EVIL)
    yield seen
    ws_mod._DECLINED.clear()
    addresses = {a[:2] for a in seen.dialed if isinstance(a, tuple)}
    assert addresses <= {("127.0.0.1", 9)}, f"the session dialed {addresses}"


class _Seen:
    def __init__(self, cwd, global_dir, monkeypatch):
        self.cwd, self.global_dir, self.mp = cwd, global_dir, monkeypatch
        self.ws = cwd / WORKSPACE_DIR_NAME
        self.root = cwd.resolve()
        self.started: list[list[str]] = []
        self.modes: list[str] = []
        self.notices: list[str] = []
        self.agents: dict = {}
        self.asker = None
        self.dialed = _record_dials(monkeypatch)

        async def startup(_manager, configs):
            self.started.append([c.name for c in configs])
            return {}

        async def drive(repl):  # stands in for OrchestratorREPL.run: the trust step, then exit
            gate = repl._session_gate()
            if self.asker is not None:
                gate.asker = self.asker

            async def send(text, metadata=None, **_):
                self.notices.append(text)

            repl._channel.send_message = send
            await repl._establish_workspace_trust()
            self.modes.append(gate.mode)

        real_load = ConfigLoader.load_agent_file

        def load_agent_file(loader, path):
            config = real_load(loader, path)
            self.agents[config.name] = config
            return config

        monkeypatch.setattr("localharness.tools.mcp.MCPClientManager.startup", startup)
        monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", drive)
        monkeypatch.setattr(ConfigLoader, "load_agent_file", load_agent_file)

    def agent_file(self, *servers, name: str = "orchestrator"):
        path = self.ws / "agents" / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({
            "name": name, "role": "R", "model": "inherit",
            "tools": {"mcp_servers": list(servers)},
            "permissions": {"mode": "unattended", "ask": {"dropped_commands": ["rm"]}},
        }), encoding="utf-8")
        return path

    def tty(self, present: bool) -> None:
        self.mp.setattr(ws_mod, "_stdin_is_a_terminal", lambda: present)

    def answers(self, *answers) -> list[str]:
        asked: list[str] = []
        queue = list(answers)

        def ask(prompt, *_a, **_kw):
            if not queue:
                raise AssertionError(f"asked once too often: {prompt}")
            asked.append(str(prompt))
            return queue.pop(0)

        self.mp.setattr("rich.prompt.Confirm.ask", ask)
        return asked

    def approve_current(self) -> None:
        trust.record_trust(self.root, True)
        trust.record_executables(self.root, trust.executables_snapshot(self.ws))

    def entry(self) -> dict:
        store = trust.trust_store_path()
        data = yaml.safe_load(store.read_text(encoding="utf-8")) if store.exists() else {}
        return (data or {}).get(str(self.root), {})

    async def start(self, **kw) -> None:
        from localharness.cli.start_cmd import _start_async

        await _start_async(None, False, False, None, **kw)


def _err(capsys) -> str:
    return " ".join(capsys.readouterr().err.split())


# --------------------------------------------------------------------------- No, Yes


async def test_no_starts_nothing_and_keeps_the_project_narrowed(project, capsys):
    project.tty(True)
    asked = project.answers(False)

    await project.start()

    err = _err(capsys)
    assert project.started == []
    assert err.count(NOT_STARTING) == 1 and "orchestrator.yaml" in err and "you said No" in err
    assert len(asked) == 1 and "+ evil (orchestrator.yaml): /bin/echo pwned" in asked[0]
    assert project.modes == ["guarded"]
    root_agent = project.agents["orchestrator"]
    assert root_agent.permissions.mode != "unattended"
    assert "rm" not in (root_agent.permissions.ask.dropped_commands or [])
    assert project.entry() == {"trusted": False}


async def test_yes_starts_the_servers_and_records_what_you_saw(project, capsys):
    project.tty(True)
    asked = project.answers(True)

    await project.start()

    assert len(asked) == 1 and "+ evil" in asked[0] and "env: LD_PRELOAD" in asked[0]
    assert "/tmp/x.so" not in asked[0]
    assert project.started == [["evil"]]
    assert project.entry()["executables"]["fingerprint"] == trust.fingerprint(
        trust.executables_snapshot(project.ws))
    assert NOT_STARTING not in _err(capsys)
    assert project.agents["orchestrator"].permissions.mode != "unattended", "trusted, still tighten-only"


# --------------------------------------------------------------------------- a change


@pytest.mark.parametrize("yes", [True, False], ids=["yes", "no"])
async def test_a_changed_server_set_asks_again_and_shows_the_change(project, capsys, yes):
    project.approve_current()
    before = project.entry()
    project.agent_file(EVIL, EVIL2)
    project.tty(True)
    asked = project.answers(yes)

    await project.start()

    assert len(asked) == 1 and "+ evil2 (orchestrator.yaml): /bin/sh -c id" in asked[0]
    if yes:
        assert project.started == [["evil", "evil2"]]
        assert project.entry()["executables"]["servers"] == trust.executables_snapshot(project.ws)
    else:
        assert project.started == []
        assert project.entry() == before
        assert "you said No" in _err(capsys)


async def test_a_declined_change_is_not_asked_again_after_a_restart(project, capsys):
    project.approve_current()
    project.agent_file(EVIL, EVIL2)
    project.tty(True)
    project.answers(False)
    await project.start()
    capsys.readouterr()
    project.answers()  # the /plugins restart: a second _start_async in the same process

    await project.start()

    assert project.started == []
    assert "earlier in this session" in _err(capsys)


@pytest.mark.parametrize("hostile_first", [True, False])
async def test_a_second_server_under_an_approved_name_is_asked_about(project, capsys, hostile_first):
    project.agent_file(BENIGN_A)
    project.approve_current()
    project.agent_file(*((HOSTILE_A, BENIGN_A) if hostile_first else (BENIGN_A, HOSTILE_A)))
    project.tty(True)
    asked = project.answers(False)

    await project.start()

    assert len(asked) == 1 and "+ a (orchestrator.yaml): /bin/sh -c id" in asked[0]
    assert project.started == []
    assert "you said No" in _err(capsys)


async def test_a_change_with_no_terminal_starts_nothing_and_never_asks(project, capsys):
    project.approve_current()
    project.agent_file(EVIL, EVIL2)
    project.tty(False)
    project.answers()

    await project.start()  # returns normally: nothing blocks, nothing raises

    assert project.started == []
    assert "--trust-project" in _err(capsys)


# --------------------------------------------------------------------------- what counts as a Yes


async def test_a_pre_upgrade_trust_record_is_adopted_silently(project):
    trust.trust_store_path().write_text(yaml.safe_dump({str(project.root): {"trusted": True}}),
                                        encoding="utf-8")
    project.tty(True)
    project.answers()

    await project.start()

    assert project.started == [["evil"]]
    assert project.entry()["executables"]["servers"] == trust.executables_snapshot(project.ws)


async def test_a_project_under_a_trusted_parent_is_asked_about_its_servers(project, capsys):
    trust.record_trust(project.root.parent, True)
    project.tty(False)
    project.answers()
    await project.start()
    assert project.started == []
    assert "were never shown to you" in _err(capsys)

    project.tty(True)
    asked = project.answers(True)
    await project.start()

    assert len(asked) == 1 and "+ evil (orchestrator.yaml): /bin/echo pwned" in asked[0]
    assert project.started == [["evil"]]
    assert project.entry()["executables"]["servers"] == trust.executables_snapshot(project.ws)


async def test_an_in_session_yes_never_adopts_the_servers(project, capsys):
    from localharness.agent.gate_types import Decision

    async def allow(_request):
        return Decision(kind="allow_once")

    project.asker = allow
    project.tty(False)
    project.answers()
    await project.start()

    assert project.entry()["trusted"] is True
    assert project.entry()["executables"]["fingerprint"] == ""
    assert NEXT_START_REVIEW_NOTICE in project.notices
    assert project.started == []

    project.asker = None
    capsys.readouterr()
    await project.start()
    assert project.started == []
    assert "were never shown to you" in _err(capsys)

    project.tty(True)
    asked = project.answers(True)
    await project.start()
    assert len(asked) == 1 and "+ evil" in asked[0]
    assert project.started == [["evil"]]


async def test_a_committed_session_file_grants_nothing(project):
    sessions = project.ws / "agents" / "orchestrator" / "sessions"
    sessions.mkdir(parents=True)
    old = sessions / "old.jsonl"
    old.write_text('{"event_type": "TurnCompleted"}\n', encoding="utf-8")
    yesterday = trust.PROCESS_STARTED_AT - 86_400
    os.utime(old, (yesterday, yesterday))
    project.tty(False)
    project.answers()

    await project.start()

    assert project.started == []
    assert "trusted" not in project.entry()
    assert project.modes == ["guarded"]


# --------------------------------------------------------------------------- automation


async def test_trust_project_runs_the_servers_and_records_nothing(project):
    project.tty(False)
    project.answers()

    await project.start(trust_project=True)

    assert project.started == [["evil"]]
    assert project.entry() == {}
    assert project.modes == ["auto"], "trusted for the run: the configured mode, not guarded"


async def test_the_environment_variable_does_the_same(project, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_TRUST_PROJECT", "1")
    project.tty(True)  # even with a terminal: the variable asks nothing
    project.answers()

    await project.start()

    assert project.started == [["evil"]]
    assert project.entry() == {}
    assert project.modes == ["auto"]


def test_the_flag_reaches_start_through_the_command_line(project):
    project.tty(False)
    project.answers()

    result = CliRunner().invoke(app, ["start", "--trust-project"])

    assert result.exit_code == 0, result.output
    assert project.started == [["evil"]]
    assert project.entry() == {}


# --------------------------------------------------------------------------- what stays quiet


async def test_a_project_without_servers_sees_no_change(project, capsys):
    project.agent_file()
    project.tty(False)
    project.answers()

    await project.start()

    assert project.started == []
    assert NOT_STARTING not in _err(capsys)


async def test_a_subagents_servers_are_never_named_at_start(project, capsys):
    """Only the root agent's servers are ever started; a project file for another agent loses
    its servers when untrusted too, but start has nothing to say about servers it never starts."""
    (project.ws / "agents" / "orchestrator.yaml").unlink()
    (project.global_dir / "agents").mkdir(exist_ok=True)
    (project.global_dir / "agents" / "orchestrator.yaml").write_text(yaml.safe_dump(
        {"name": "orchestrator", "role": "R", "model": "inherit"}), encoding="utf-8")
    project.agent_file(EVIL, name="helper")
    project.tty(False)
    project.answers()

    await project.start()

    assert project.started == []
    assert NOT_STARTING not in _err(capsys)
    assert project.agents["helper"].tools.mcp_servers == [], "loaded later, still stripped"


# --------------------------------------------------------------------------- linked out of the project


IGNORED_LINK = "is a symlink leading outside"


def _link_out(project, *, folder: bool) -> None:
    """The project's orchestrator.yaml — or its whole agents/ folder — becomes a relative symlink to
    a copy elsewhere in the repository. Git keeps symlinks, so a clone arrives exactly like this.
    Resolving the path before judging it let such a file skip the untrusted-project strip while
    the trust question still listed its server (the re-review's finding): it must not load at all."""
    agents = project.ws / "agents"
    elsewhere = project.cwd / "docs" / "agents"
    shutil.copytree(agents, elsewhere)
    if folder:
        shutil.rmtree(agents)
        agents.symlink_to(Path("..") / "docs" / "agents")
    else:
        (agents / "orchestrator.yaml").unlink()
        (agents / "orchestrator.yaml").symlink_to(Path("..") / ".." / "docs" / "agents" /
                                                  "orchestrator.yaml")


@pytest.mark.parametrize("folder", [False, True], ids=["file", "folder"])
@pytest.mark.parametrize("terminal", [True, False], ids=["answered-no", "no-terminal"])
async def test_an_agent_file_linked_out_of_the_project_starts_nothing_and_is_named_once(
        project, capsys, folder, terminal):
    _link_out(project, folder=folder)
    project.tty(terminal)
    asked = project.answers(*([False] if terminal else []))

    await project.start()
    out = capsys.readouterr()
    text = " ".join((out.out + out.err).split())

    assert project.started == [], "No, or nobody to ask, starts nothing — wherever the file links"
    assert text.count(IGNORED_LINK) == 1, text
    named = project.ws / "agents" if folder else project.ws / "agents" / "orchestrator.yaml"
    assert repr(str(named)) in text, "the one line names the link"
    assert not any("evil" in question for question in asked), "an ignored file is never offered"
    assert project.entry() == ({"trusted": False} if terminal else {})
