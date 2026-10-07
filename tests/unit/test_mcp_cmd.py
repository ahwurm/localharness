"""Spec 15 — `/mcp` slash command unit tests (tests/test_mcp_cmd.py).

Covers list (empty/configured/provenance/shadowing/untrusted), add (stdio
workspace/global, http, untrusted notice), edit (change + shadowing), test
connect (success/failure), remove (write + unregistration, not-found), and
secret masking.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from localharness.cli import mcp_cmd
from localharness.config.paths import WORKSPACE_DIR_NAME


# --------------------------------------------------------------------------- fakes


class FakeChannel:
    """Scripted channel: read_input pops from `answers`; send_message records."""

    def __init__(self, answers=None):
        self.answers = list(answers or [])
        self.sent: list[tuple[str, dict]] = []

    async def read_input(self, prompt=""):
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    async def send_message(self, text, metadata=None):
        self.sent.append((text, metadata or {}))

    def all_text(self) -> str:
        return "\n".join(t for t, _ in self.sent)


class FakeRegistry:
    def __init__(self):
        self.registered: list[tuple] = []
        self.unregistered: list[tuple] = []

    async def register(self, tool, scope=None):
        self.registered.append((tool, scope))

    async def unregister(self, name, scope=None):
        self.unregistered.append((name, scope))


class FakeManager:
    def __init__(self, connected=(), clients=None):
        self._connected = list(connected)
        self._clients = dict(clients or {})

    @property
    def connected_servers(self):
        return self._connected

    def get_client(self, name):
        return self._clients.get(name)

    def add_client(self, name, client):
        self._clients[name] = client
        if name not in self._connected:
            self._connected.append(name)

    def remove_client(self, name):
        client = self._clients.pop(name, None)
        if name in self._connected:
            self._connected.remove(name)
        return client


def make_repl(workspace=None, config_dir=None, channel=None, manager=None,
              registry=None, agent_name="orchestrator"):
    agent = None
    if agent_name is not None:
        agent = SimpleNamespace(_config=SimpleNamespace(name=agent_name), _tools=registry)
    return SimpleNamespace(
        _workspace=workspace, _config_dir=config_dir, _agent=agent,
        _mcp_manager=manager, _channel=channel,
    )


def write_agent(ws_or_dir: Path, name: str, servers: list[dict]) -> Path:
    path = Path(ws_or_dir) / "agents" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"name": name, "role": "R",
                        "tools": {"mcp_servers": servers}}),
        encoding="utf-8",
    )
    return path


def read_servers(path: Path) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    tools = raw.get("tools") or {}
    return tools.get("mcp_servers") or []


def make_fake_client(tools=(), fail=None):
    """Build a FakeMCPClient class with the given tool names / connect failure."""
    tool_objs = [SimpleNamespace(info=lambda n=n: SimpleNamespace(name=n)) for n in tools]

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config
            self.tools = list(tool_objs)
            self.connected = False
            self.disconnected = False

        async def connect(self):
            if fail is not None:
                raise fail
            self.connected = True

        async def disconnect(self):
            self.disconnected = True

    return FakeMCPClient


# --------------------------------------------------------------------------- list


async def test_list_empty(tmp_path):
    ch = FakeChannel()
    repl = make_repl(workspace=tmp_path / WORKSPACE_DIR_NAME, config_dir=tmp_path / "global",
                     channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "")
    assert "No MCP servers configured for 'orchestrator'" in ch.all_text()


async def test_list_configured(tmp_path):
    ws = tmp_path / WORKSPACE_DIR_NAME
    gl = tmp_path / "global"
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node", "args": ["/x.cjs"]},
    ])
    write_agent(gl, "orchestrator", [
        {"name": "github", "transport": "streamable_http", "url": "http://127.0.0.1:3000/mcp"},
    ])
    manager = FakeManager(connected=["gsd"],
                          clients={"gsd": make_fake_client(tools=("a", "b", "c"))(None)})
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=gl, channel=ch, manager=manager)
    await mcp_cmd.run_mcp_slash(repl, "")
    text = ch.all_text()
    assert "MCP Servers — agent: orchestrator" in text
    assert "gsd" in text and "github" in text
    assert "● connected" in text
    assert "○ not connected" in text
    assert "3 tools" in text


async def test_list_provenance(tmp_path):
    ws = tmp_path / WORKSPACE_DIR_NAME
    gl = tmp_path / "global"
    write_agent(ws, "orchestrator", [
        {"name": "wsvc", "transport": "stdio", "command": "node"},
    ])
    write_agent(gl, "orchestrator", [
        {"name": "gsvc", "transport": "stdio", "command": "node"},
    ])
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=gl, channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "")
    text = ch.all_text()
    assert "[workspace]" in text
    assert "[global]" in text
    # wsvc line carries [workspace]; gsvc line carries [global]
    wline = next(l for l in text.splitlines() if "wsvc" in l)
    gline = next(l for l in text.splitlines() if "gsvc" in l)
    assert "[workspace]" in wline
    assert "[global]" in gline


async def test_list_shadowing(tmp_path):
    ws = tmp_path / WORKSPACE_DIR_NAME
    gl = tmp_path / "global"
    write_agent(ws, "orchestrator", [
        {"name": "dup", "transport": "stdio", "command": "node", "args": ["ws.cjs"]},
    ])
    write_agent(gl, "orchestrator", [
        {"name": "dup", "transport": "stdio", "command": "node", "args": ["gl.cjs"]},
    ])
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=gl, channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "")
    text = ch.all_text()
    dup_lines = [l for l in text.splitlines() if "dup" in l and "stdio" in l]
    assert len(dup_lines) == 1, "same-name server must show once (workspace shadows global)"
    assert "[workspace]" in dup_lines[0]


async def test_list_untrusted(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node"},
        {"name": "other", "transport": "stdio", "command": "node"},
    ])
    monkeypatch.setattr(mcp_cmd.trust, "executables_snapshot", lambda d: [{"file": "a.yaml", "name": "gsd"}])
    monkeypatch.setattr(mcp_cmd.trust, "recorded_executables", lambda root: None)  # never recorded
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "")
    text = ch.all_text()
    assert "2 servers withheld (workspace not trusted)" in text


# --------------------------------------------------------------------------- add


async def test_add_stdio_workspace(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    gl = tmp_path / "global"
    (ws / "agents").mkdir(parents=True)
    (gl / "agents").mkdir(parents=True)
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    ch = FakeChannel(answers=["gsd", "1", "node", "/x.cjs", "", "y"])
    repl = make_repl(workspace=ws, config_dir=gl, channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "add")
    text = ch.all_text()
    assert "Saved to" in text
    servers = read_servers(ws / "agents" / "orchestrator.yaml")
    assert len(servers) == 1
    assert servers[0]["name"] == "gsd"
    assert servers[0]["transport"] == "stdio"
    assert servers[0]["command"] == "node"
    assert servers[0]["args"] == ["/x.cjs"]
    # global file untouched
    assert not (gl / "agents" / "orchestrator.yaml").exists()


async def test_add_stdio_global(tmp_path, monkeypatch):
    """Not in a project (no workspace) → write goes to the global agent file."""
    gl = tmp_path / "global"
    (gl / "agents").mkdir(parents=True)
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    ch = FakeChannel(answers=["gsd", "1", "node", "/x.cjs", "", "y"])
    repl = make_repl(workspace=None, config_dir=gl, channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "add")
    servers = read_servers(gl / "agents" / "orchestrator.yaml")
    assert len(servers) == 1
    assert servers[0]["name"] == "gsd"
    assert servers[0]["command"] == "node"


async def test_add_http(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    (ws / "agents").mkdir(parents=True)
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    ch = FakeChannel(answers=["github", "2", "http://127.0.0.1:3000/mcp",
                              "Authorization=Bearer tok", "", "y"])
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "add")
    servers = read_servers(ws / "agents" / "orchestrator.yaml")
    assert len(servers) == 1
    assert servers[0]["name"] == "github"
    assert servers[0]["transport"] == "streamable_http"
    assert servers[0]["url"] == "http://127.0.0.1:3000/mcp"
    assert servers[0]["headers"] == {"Authorization": "Bearer tok"}


async def test_add_untrusted_notice(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    (ws / "agents").mkdir(parents=True)
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    monkeypatch.setattr(mcp_cmd.trust, "executables_snapshot", lambda d: [{"file": "a.yaml", "name": "gsd"}])
    monkeypatch.setattr(mcp_cmd.trust, "recorded_executables", lambda root: None)
    ch = FakeChannel(answers=["gsd", "1", "node", "/x.cjs", "", "y"])
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "add")
    text = ch.all_text()
    assert "Saved to" in text
    assert "not trusted" in text
    assert "will not connect" in text


# --------------------------------------------------------------------------- edit


async def test_edit(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node", "args": ["/old.cjs"],
         "env": {"A": "1"}},
    ])
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    # Command, Args, Env change, Env blank (finish), Save
    ch = FakeChannel(answers=["node2", "/new.cjs", "A=2", "", "y"])
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "edit gsd")
    servers = read_servers(ws / "agents" / "orchestrator.yaml")
    assert len(servers) == 1
    assert servers[0]["command"] == "node2"
    assert servers[0]["args"] == ["/new.cjs"]
    assert servers[0]["env"] == {"A": "2"}


async def test_edit_shadowed(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    gl = tmp_path / "global"
    write_agent(ws, "orchestrator", [
        {"name": "dup", "transport": "stdio", "command": "node", "args": ["ws.cjs"]},
    ])
    write_agent(gl, "orchestrator", [
        {"name": "dup", "transport": "stdio", "command": "node", "args": ["gl.cjs"]},
    ])
    monkeypatch.setattr(mcp_cmd, "MCPServerClient", make_fake_client(tools=("t1",)))
    ch = FakeChannel(answers=["node2", "", "", "y"])  # change command, keep args, no env, save
    repl = make_repl(workspace=ws, config_dir=gl, channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "edit dup")
    # workspace file changed, global untouched
    ws_servers = read_servers(ws / "agents" / "orchestrator.yaml")
    gl_servers = read_servers(gl / "agents" / "orchestrator.yaml")
    assert ws_servers[0]["command"] == "node2"
    assert gl_servers[0]["command"] == "node"  # global unchanged


# --------------------------------------------------------------------------- test


async def test_test_connect_success(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node"},
    ])
    monkeypatch.setattr(mcp_cmd, "MCPServerClient",
                        make_fake_client(tools=("gsd__state_load", "gsd__state_update")))
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "test gsd")
    text = ch.all_text()
    assert "Testing 'gsd'" in text
    assert "✓ Connected" in text
    assert "2 tools" in text
    assert "gsd__state_load" in text


async def test_test_connect_failure(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node"},
    ])
    monkeypatch.setattr(mcp_cmd, "MCPServerClient",
                        make_fake_client(fail=RuntimeError("spawn ENOENT — command 'node' not found")))
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "test gsd")
    text = ch.all_text()
    assert "Testing 'gsd'" in text
    assert "✗ Connection failed" in text
    assert "spawn ENOENT" in text


# --------------------------------------------------------------------------- remove


async def test_remove(tmp_path, monkeypatch):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node"},
        {"name": "keep", "transport": "stdio", "command": "node"},
    ])
    reg = FakeRegistry()
    client = make_fake_client(tools=("gsd__state_load",))(None)
    manager = FakeManager(connected=["gsd"], clients={"gsd": client})
    ch = FakeChannel(answers=["y"])
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch,
                     manager=manager, registry=reg)
    await mcp_cmd.run_mcp_slash(repl, "remove gsd")
    text = ch.all_text()
    assert "Removed 'gsd'" in text
    servers = read_servers(ws / "agents" / "orchestrator.yaml")
    assert [s["name"] for s in servers] == ["keep"]
    assert ("gsd__state_load", "mcp") in reg.unregistered
    assert client.disconnected is True
    assert "gsd" not in manager._clients


async def test_remove_not_found(tmp_path):
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node"},
    ])
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "remove nope")
    assert "Unknown server 'nope'" in ch.all_text()


# --------------------------------------------------------------------------- secrets


async def test_secret_not_exposed(tmp_path, monkeypatch):
    """env values never appear in list output; test-error output is masked."""
    ws = tmp_path / WORKSPACE_DIR_NAME
    write_agent(ws, "orchestrator", [
        {"name": "gsd", "transport": "stdio", "command": "node",
         "env": {"TOKEN": "supersecretvalue123"}},
    ])
    # list: the raw env value must not be printed
    ch = FakeChannel()
    repl = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "")
    assert "supersecretvalue123" not in ch.all_text()

    # test: a connect error embedding the env value must be masked
    monkeypatch.setattr(
        mcp_cmd, "MCPServerClient",
        make_fake_client(fail=RuntimeError("auth failed for supersecretvalue123")))
    ch2 = FakeChannel()
    repl2 = make_repl(workspace=ws, config_dir=tmp_path / "global", channel=ch2)
    await mcp_cmd.run_mcp_slash(repl2, "test gsd")
    assert "supersecretvalue123" not in ch2.all_text()
    assert "●●●●" in ch2.all_text()
    assert "auth failed" in ch2.all_text()


# --------------------------------------------------------------------------- entry point


async def test_help():
    ch = FakeChannel()
    repl = make_repl(channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "help")
    text = ch.all_text()
    assert "/mcp add" in text
    assert "/mcp edit" in text
    assert "/mcp test" in text
    assert "/mcp remove" in text


async def test_unknown_subcommand():
    ch = FakeChannel()
    repl = make_repl(channel=ch)
    await mcp_cmd.run_mcp_slash(repl, "frobnicate")
    assert "Unknown /mcp subcommand: frobnicate" in ch.all_text()
