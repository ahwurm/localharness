"""What your own agent files start or loosen waits for you — through the real start.

The owner kept self-extension (Option 1): the agent may write the machine's agent files, and what
such a file starts (an MCP server) or loosens (a permission looser than the shipped default) is
applied only after one Yes at a start on a terminal. Each case runs the real `_start_async` outside
any project, with the machine's `agents/orchestrator.yaml` naming server g1, and watches what
reaches the MCP manager, what the root agent resolves to, what is asked and what is printed.

REAL: `_start_async`, `decide_machine_trust` and the machine record in the trust store, ConfigLoader
and its withheld strip, the permission resolution.
STUBBED: the provider probe and tokenizer (via `_machine`), `MCPClientManager.startup` (nothing is
spawned), the terminal answers (`rich.prompt.Confirm.ask`, `cli.workspace._stdin_is_a_terminal`),
the REPL read loop (returns at once). The root AgentConfig is read by wrapping
`ConfigLoader.load_agent_file` — the gate's own mode can be lowered by the session's trust step.
NOT PROVEN here: a real MCP process; a real `/plugins` keystroke (a second `_start_async` in the
same process, driven here as exactly that).
"""
from __future__ import annotations

import contextlib

import pytest
import typer
import yaml

from localharness.cli import workspace as ws_mod
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from tests.integration.test_image_plugin_e2e import _machine, _record_dials

G1 = {"name": "g1", "transport": "stdio", "command": "/bin/echo", "args": ["hi"]}
G2 = {"name": "g2", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "id"]}
HOSTILE_G1 = {**G2, "name": "g1"}
ORCH = "agents/orchestrator.yaml"


@pytest.fixture
def machine(tmp_path, monkeypatch, fake_home):
    global_dir, cwd = _machine(tmp_path, monkeypatch, fake_home, project=False)
    trust.record_trust(cwd, True)  # the start's workspace question is not under test here
    ws_mod._DECLINED.clear()
    seen = _Machine(global_dir, monkeypatch)
    seen.orchestrator(G1)
    yield seen
    ws_mod._DECLINED.clear()
    addresses = {a[:2] for a in seen.dialed if isinstance(a, tuple)}
    assert addresses <= {("127.0.0.1", 9)}, f"the session dialed {addresses}"


class _Machine:
    def __init__(self, global_dir, monkeypatch):
        self.g, self.mp = global_dir, monkeypatch
        self.started: list[list[tuple[str, str]]] = []
        self.roots: list = []
        self.dialed = _record_dials(monkeypatch)

        async def startup(_manager, configs):
            self.started.append([(c.name, c.command) for c in configs])
            return {}

        real_load = ConfigLoader.load_agent_file

        def load_agent_file(loader, path):
            config = real_load(loader, path)
            if config.name == "orchestrator":
                self.roots.append(config)
            return config

        monkeypatch.setattr("localharness.tools.mcp.MCPClientManager.startup", startup)
        monkeypatch.setattr(ConfigLoader, "load_agent_file", load_agent_file)

    def orchestrator(self, *servers, **extra) -> None:
        path = self.g / ORCH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"name": "orchestrator", "role": "R", "model": "inherit",
                                        "tools": {"mcp_servers": list(servers)}, **extra}),
                        encoding="utf-8")

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

    def names(self) -> list[str]:
        return [name for name, _ in self.started[-1]] if self.started else []

    async def start(self) -> None:
        from localharness.cli.start_cmd import _start_async

        await _start_async(None, False, False, None)

    async def first_start(self) -> None:
        """The first start after the upgrade: adopts the files as they are, asks nothing."""
        self.tty(True)
        self.answers()
        await self.start()
        self.started.clear()


def _err(capsys) -> str:
    return " ".join(capsys.readouterr().err.split())


async def test_the_first_start_after_the_upgrade_adopts_the_machine_files_silently(machine):
    machine.tty(True)
    machine.answers()

    await machine.start()

    assert machine.names() == ["g1"]
    assert trust.recorded_machine(machine.g)["entries"] == trust.machine_snapshot(machine.g)


async def test_a_server_added_to_a_machine_agent_file_is_asked_once_on_a_terminal(machine):
    await machine.first_start()
    machine.orchestrator(G1, G2)
    asked = machine.answers(True)

    await machine.start()

    assert len(asked) == 1 and f"+ g2 ({ORCH}): /bin/sh -c id" in asked[0]
    assert machine.names() == ["g1", "g2"]
    machine.answers()
    await machine.start()
    assert machine.names() == ["g1", "g2"]


async def test_no_withholds_it_and_a_restart_in_the_same_process_does_not_ask_again(machine, capsys):
    await machine.first_start()
    machine.orchestrator(G1, G2)
    machine.answers(False)
    capsys.readouterr()

    await machine.start()

    err = _err(capsys)
    assert machine.names() == ["g1"]
    assert f"g2 ({ORCH})" in err and "you said No" in err
    machine.answers()  # the /plugins restart
    await machine.start()
    assert machine.names() == ["g1"]
    assert "earlier in this session" in _err(capsys)


@pytest.mark.parametrize("hostile_first", [True, False])
async def test_a_second_server_under_a_confirmed_name_is_withheld(machine, capsys, hostile_first):
    await machine.first_start()
    pair = (HOSTILE_G1, G1) if hostile_first else (G1, HOSTILE_G1)
    machine.orchestrator(*pair)
    asked = machine.answers(False)
    capsys.readouterr()

    await machine.start()

    assert len(asked) == 1 and f"+ g1 ({ORCH}): /bin/sh -c id" in asked[0]
    assert not any(name == "g1" for call in machine.started for name, _ in call)
    assert "you said No" in _err(capsys)

    ws_mod._DECLINED.clear()  # a later process: the same change, answered Yes this time
    machine.answers(True)
    await machine.start()
    assert sorted(machine.started[-1]) == [("g1", "/bin/echo"), ("g1", "/bin/sh")]
    machine.answers()
    await machine.start()
    assert sorted(machine.started[-1]) == [("g1", "/bin/echo"), ("g1", "/bin/sh")]


async def test_with_no_terminal_the_change_is_withheld_and_named(machine, capsys):
    await machine.first_start()
    machine.orchestrator(G1, G2)
    machine.tty(False)
    machine.answers()
    capsys.readouterr()

    await machine.start()

    err = _err(capsys)
    assert machine.names() == ["g1"]
    assert err.count("Not applying 1 change(s)") == 1
    assert "`localharness start`" in err


async def test_a_loosened_mode_waits_for_confirmation_and_a_tightened_one_never_asks(machine, capsys):
    await machine.first_start()
    machine.orchestrator(G1, permissions={"mode": "unattended"})
    machine.tty(False)
    machine.answers()
    capsys.readouterr()

    await machine.start()
    assert machine.roots[-1].permissions.mode == "auto"
    assert "permissions.mode" in _err(capsys)

    machine.tty(True)
    machine.answers(True)
    await machine.start()
    assert machine.roots[-1].permissions.mode == "unattended"

    machine.orchestrator(G1, permissions={"mode": "guarded"})
    machine.answers()
    await machine.start()
    assert machine.roots[-1].permissions.mode == "guarded"


async def test_removing_a_server_is_recorded_silently(machine):
    await machine.first_start()
    machine.orchestrator()
    machine.answers()

    await machine.start()

    assert machine.started == []
    assert trust.recorded_machine(machine.g)["entries"] == []


async def test_listing_models_decides_and_asks_nothing(machine, capsys):
    """`start --list-models` loads no agent and starts nothing, so a pending change is neither
    asked about nor reported there; the next real start still asks."""
    from localharness.cli.start_cmd import _start_async

    await machine.first_start()
    machine.orchestrator(G1, G2)
    machine.answers()
    capsys.readouterr()

    with contextlib.suppress(typer.Exit):
        await _start_async(None, False, False, None, list_models=True)

    assert "Not applying" not in _err(capsys)
    assert machine.started == []
    asked = machine.answers(True)
    await machine.start()
    assert len(asked) == 1 and machine.names() == ["g1", "g2"]


# --------------------------------------------------------------------------- overrides.yaml's agent: section


def _bare_orchestrator(machine) -> None:
    """The machine's root agent file with no `tools.mcp_servers` key at all: an explicit list —
    even an empty one — replaces the overrides' list, since lists are not unioned."""
    (machine.g / ORCH).write_text(yaml.safe_dump({"name": "orchestrator", "role": "R",
                                                  "model": "inherit"}), encoding="utf-8")


def _overrides_agent(machine, section: dict) -> None:
    """The global overrides.yaml's `agent:` section is every agent's default layer: the loader
    merges it under each agent file, so a server there is started for a root agent whose own file
    names none (an agent file's own list replaces it — lists are not unioned)."""
    path = machine.g / "overrides.yaml"
    data = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    data["agent"] = section
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


async def test_a_server_added_to_the_overrides_agent_section_is_asked_once_like_an_agent_file(machine):
    """The re-review's F-A: this section used to be outside the fingerprint, so a server put there
    (python_exec can write the file unasked in auto) started at the next start with no question."""
    _bare_orchestrator(machine)
    await machine.first_start()
    _overrides_agent(machine, {"tools": {"mcp_servers": [G2]}})
    asked = machine.answers(True)

    await machine.start()

    assert len(asked) == 1 and "+ g2 (overrides.yaml): /bin/sh -c id" in asked[0]
    assert machine.names() == ["g2"]


@pytest.mark.parametrize("terminal", [False, True], ids=["no-terminal", "answered-no"])
async def test_without_a_yes_a_server_in_the_overrides_agent_section_never_starts(machine, capsys,
                                                                                   terminal):
    _bare_orchestrator(machine)
    await machine.first_start()
    _overrides_agent(machine, {"tools": {"mcp_servers": [G2]},
                               "memory": {"embedding_model": "./models/evil"},
                               "permissions": {"mode": "unattended"}})
    machine.tty(terminal)
    machine.answers(*([False] if terminal else []))

    await machine.start()

    assert machine.names() == [], "the server in overrides.yaml is held back"
    root = machine.roots[-1]
    assert root.permissions.mode != "unattended", "the loosening is held back with it"
    assert root.memory.embedding_model != "./models/evil", "and so is the embedding model"
    err = _err(capsys)
    assert "Not applying 3 change(s)" in err and "(overrides.yaml)" in err


async def test_an_overrides_agent_section_there_before_the_upgrade_is_adopted_silently(machine):
    _bare_orchestrator(machine)
    _overrides_agent(machine, {"tools": {"mcp_servers": [G2]}})
    machine.tty(True)
    machine.answers()

    await machine.start()

    assert machine.names() == ["g2"]
