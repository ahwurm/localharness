"""`/mcp` slash command: in-session MCP server management for the session's agent.

Spec 15. Operates on the session's agent file (workspace layer shadows global).
Reuses MCPServerClient for test/hot-connect, ToolRegistry for tool registration,
and config/trust.py for the trust gate. No new module in tools/; no new merge logic.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, NamedTuple, Optional

import yaml

from localharness.config import trust
from localharness.config.models import MCPServerConfig
from localharness.tools.mcp import MCPServerClient

# ---------------------------------------------------------------------------
# Agent-file helpers
# ---------------------------------------------------------------------------


def _agent_file_path(repl: Any, name: str) -> Optional[Path]:
    """The active agent file for `name`: workspace layer if it exists, else global."""
    workspace = getattr(repl, "_workspace", None)
    config_dir = getattr(repl, "_config_dir", None)
    if workspace is not None:
        ws_path = Path(workspace) / "agents" / f"{name}.yaml"
        if ws_path.is_file():
            return ws_path
    if config_dir is not None:
        gl_path = Path(config_dir) / "agents" / f"{name}.yaml"
        if gl_path.is_file():
            return gl_path
    return None


def _extract_mcp_servers(raw: Any) -> list[dict]:
    """Extract the `tools.mcp_servers` list from a parsed agent-file dict, or []."""
    if not isinstance(raw, dict):
        return []
    tools = raw.get("tools")
    if not isinstance(tools, dict):
        return []
    servers = tools.get("mcp_servers")
    return [s for s in servers if isinstance(s, dict)] if isinstance(servers, list) else []


def _read_agent_servers(repl: Any, name: str) -> list[dict]:
    """The `tools.mcp_servers` list from the active agent file, or []."""
    path = _agent_file_path(repl, name)
    if path is None:
        return []
    try:
        raw = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return []
    return _extract_mcp_servers(raw)


def _atomic_write_yaml(path: Path, data: dict) -> None:
    """Write YAML atomically: tempfile in same dir → fsync → os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(yaml_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_agent_servers(repl: Any, name: str, servers: list[dict]) -> Optional[Path]:
    """Write `servers` to the active agent file's `tools.mcp_servers`.

    Targeted YAML edit: read the file, modify only `tools.mcp_servers`, write back
    atomically. Never str()/json.dumps a SecretStr — the values here are plain
    strings from YAML. Returns the path written, or None if no target could be
    determined.
    """
    workspace = getattr(repl, "_workspace", None)
    config_dir = getattr(repl, "_config_dir", None)
    ws_path = Path(workspace) / "agents" / f"{name}.yaml" if workspace is not None else None
    gl_path = Path(config_dir) / "agents" / f"{name}.yaml" if config_dir is not None else None
    # §7.3: the workspace agent file shadows the global one. Write to the layer
    # that defines the agent (workspace first), else the other layer, else the
    # layer matching the session context (workspace when in a project, else global).
    if ws_path is not None and ws_path.is_file():
        target = ws_path
    elif gl_path is not None and gl_path.is_file():
        target = gl_path
    elif workspace is not None:
        target = ws_path
    elif config_dir is not None:
        target = gl_path
    else:
        return None
    raw: dict = {}
    if target.is_file():
        try:
            loaded = yaml.safe_load(target.read_text())
            if isinstance(loaded, dict):
                raw = loaded
        except (OSError, yaml.YAMLError):
            raw = {}
    tools = raw.get("tools")
    if not isinstance(tools, dict):
        tools = {}
        raw["tools"] = tools
    tools["mcp_servers"] = servers
    _atomic_write_yaml(target, raw)
    return target


class _ProvenancedServer(NamedTuple):
    """A server entry with its source layer."""
    data: dict
    source: str  # "workspace" or "global"


def _all_servers_with_provenance(repl: Any, name: str) -> list[_ProvenancedServer]:
    """All servers for `name` with provenance: workspace shadows same-name global."""
    workspace = getattr(repl, "_workspace", None)
    config_dir = getattr(repl, "_config_dir", None)
    ws_servers: list[dict] = []
    gl_servers: list[dict] = []
    if workspace is not None:
        ws_path = Path(workspace) / "agents" / f"{name}.yaml"
        if ws_path.is_file():
            try:
                ws_servers = _extract_mcp_servers(yaml.safe_load(ws_path.read_text()))
            except (OSError, yaml.YAMLError):
                pass
    if config_dir is not None:
        gl_path = Path(config_dir) / "agents" / f"{name}.yaml"
        if gl_path.is_file():
            try:
                gl_servers = _extract_mcp_servers(yaml.safe_load(gl_path.read_text()))
            except (OSError, yaml.YAMLError):
                pass
    result: list[_ProvenancedServer] = []
    ws_names = {s.get("name") for s in ws_servers}
    for s in ws_servers:
        result.append(_ProvenancedServer(data=s, source="workspace"))
    for s in gl_servers:
        if s.get("name") not in ws_names:
            result.append(_ProvenancedServer(data=s, source="global"))
    return result


# ---------------------------------------------------------------------------
# Trust helpers
# ---------------------------------------------------------------------------


def _trust_root(ws_dir: Path) -> Path:
    """The trust root for a workspace dir (one level up, where trusted_workspaces.yaml lives)."""
    return ws_dir.resolve().parent


def _workspace_trusted(repl: Any) -> bool:
    """Is the session's workspace trusted (fingerprint matches recorded)?

    No workspace = no trust gate (True). Never-recorded = untrusted (fail-closed).
    """
    workspace = getattr(repl, "_workspace", None)
    if workspace is None:
        return True
    ws_dir = Path(workspace)
    ws_root = _trust_root(ws_dir)
    snap = trust.executables_snapshot(ws_dir)
    stored = trust.recorded_executables(ws_root)
    if stored is None:
        return False
    return stored["fingerprint"] == trust.fingerprint(snap)


def _uncovered_servers(repl: Any) -> list[dict]:
    """Servers in the current snapshot not covered by the recorded snapshot (whole-entry)."""
    workspace = getattr(repl, "_workspace", None)
    if workspace is None:
        return []
    ws_dir = Path(workspace)
    ws_root = _trust_root(ws_dir)
    snap = trust.executables_snapshot(ws_dir)
    stored = trust.recorded_executables(ws_root)
    if stored is None:
        return snap
    return [e for e in snap if e not in stored["servers"]]


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------


def _mask_secrets(text: str, config: MCPServerConfig) -> str:
    """Replace any configured url or env/header value in `text` with ●●●●.

    A raw str(exc) from MCPServerClient.connect() can embed the URL or a header,
    so it is never printed unsanitized.
    """
    masked = text
    if config.url:
        masked = masked.replace(config.url, "●●●●")
    for v in config.env.values():
        val = v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
        if val:
            masked = masked.replace(val, "●●●●")
    for v in config.headers.values():
        val = v.get_secret_value() if hasattr(v, "get_secret_value") else str(v)
        if val:
            masked = masked.replace(val, "●●●●")
    return masked


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _build_entry(name: str, transport: str, command: Optional[str], args: list[str],
                 url: Optional[str], env: dict[str, str], headers: dict[str, str]) -> dict:
    """Build a server entry dict for the agent file."""
    entry: dict = {"name": name, "transport": transport}
    if transport == "stdio":
        entry["command"] = command
        if args:
            entry["args"] = args
        if env:
            entry["env"] = env
    else:
        entry["url"] = url
        if headers:
            entry["headers"] = headers
    return entry


def _format_tool_names(names: list[str]) -> str:
    """Show first 5 tool names + ellipsis."""
    return ", ".join(names[:5]) + ("…" if len(names) > 5 else "")


def _config_from_dict(d: dict) -> MCPServerConfig:
    """Build an MCPServerConfig from a raw agent-file server dict."""
    return MCPServerConfig(
        name=d.get("name", ""),
        transport=str(d.get("transport") or "stdio"),
        command=d.get("command"),
        args=d.get("args", []),
        env=d.get("env", {}),
        url=d.get("url"),
        headers=d.get("headers", {}),
    )


def _find_server(servers: list[dict], name: str) -> Optional[dict]:
    """Find a server entry by name, or None."""
    return next((s for s in servers if s.get("name") == name), None)


# ---------------------------------------------------------------------------
# Test connectivity (one-shot)
# ---------------------------------------------------------------------------


async def _test_connect(config: MCPServerConfig) -> tuple[bool, str, list[str]]:
    """One-shot connect test: connect, list tools, disconnect.

    Returns (ok, message, tool_names). On failure, the message is sanitized
    so no configured url or env/header value is exposed.
    """
    client = MCPServerClient(config)
    try:
        await client.connect()
        names = [t.info().name for t in client.tools]
        return True, f"{len(names)} tools", names
    except Exception as exc:
        return False, _mask_secrets(str(exc), config), []
    finally:
        await client.disconnect()


# ---------------------------------------------------------------------------
# Session state accessors
# ---------------------------------------------------------------------------


def _session_agent_name(repl: Any) -> str:
    """The session's agent name (from the agent loop's config)."""
    agent = getattr(repl, "_agent", None)
    if agent is not None:
        config = getattr(agent, "_config", None)
        if config is not None:
            name = getattr(config, "name", None)
            if name:
                return name
    return "orchestrator"


def _tool_registry(repl: Any) -> Any:
    """The live ToolRegistry, or None."""
    agent = getattr(repl, "_agent", None)
    if agent is not None:
        return getattr(agent, "_tools", None)
    return None


def _mcp_manager(repl: Any) -> Any:
    """The live MCPClientManager, or None."""
    return getattr(repl, "_mcp_manager", None)


def _channel(repl: Any) -> Any:
    """The channel (for send_message / read_input), or None."""
    return getattr(repl, "_channel", None)


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


async def _prompt(repl: Any, text: str) -> str:
    """Read a line of input from the user. Returns '' if no channel."""
    ch = _channel(repl)
    if ch is None:
        return ""
    try:
        return (await ch.read_input(prompt=text)).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


async def _confirm(repl: Any, text: str) -> bool:
    """Ask a y/N question. EOF/KeyboardInterrupt answers No."""
    answer = await _prompt(repl, f"{text} [y/N] ")
    return answer.lower() in ("y", "yes")


async def _say(repl: Any, text: str, style: str = "system.info") -> None:
    """Send a message to the channel. No-op if no channel."""
    ch = _channel(repl)
    if ch is not None:
        await ch.send_message(text, metadata={"style": style})


# ---------------------------------------------------------------------------
# Hot-connect / hot-remove
# ---------------------------------------------------------------------------


async def _hot_add(repl: Any, config: MCPServerConfig) -> bool:
    """Hot-add a server: trust re-check, connect, register tools.

    Returns True if connected. Defense-in-depth: re-verifies trust state itself,
    so a future code path that reaches connect() cannot skip the re-check.
    """
    registry = _tool_registry(repl)
    manager = _mcp_manager(repl)
    if registry is None:
        return False
    # Trust re-check (§7.6)
    if not _workspace_trusted(repl):
        return False
    # Per-server confirm for new/changed servers
    uncovered = _uncovered_servers(repl)
    for entry in uncovered:
        if entry.get("name") == config.name:
            if not await _confirm(repl, f"Connect new/changed MCP server '{config.name}' this session?"):
                return False
    # Connect
    client = MCPServerClient(config)
    try:
        await client.connect()
    except Exception:
        return False
    # Register tools
    for tool in client.tools:
        try:
            await registry.register(tool, scope="mcp")
        except ValueError:
            pass  # already registered
    # Track in manager so connected_servers / shutdown see it
    if manager is not None:
        manager.add_client(config.name, client)
    return True


async def _hot_remove(repl: Any, name: str) -> None:
    """Hot-remove a server: unregister tools, disconnect, drop from manager."""
    registry = _tool_registry(repl)
    manager = _mcp_manager(repl)
    if manager is None:
        return
    client = manager.remove_client(name)
    if client is None:
        return
    if registry is not None:
        for tool in client.tools:
            try:
                await registry.unregister(tool.info().name, scope="mcp")
            except Exception:
                pass
    await client.disconnect()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def _list_servers(repl: Any) -> None:
    """List all MCP servers for the session's agent (default action)."""
    name = _session_agent_name(repl)
    servers = _all_servers_with_provenance(repl, name)
    manager = _mcp_manager(repl)
    connected = set(manager.connected_servers) if manager else set()

    if not servers:
        await _say(repl, f"No MCP servers configured for '{name}'. /mcp add to create one.")
        return

    # Trust notice
    workspace = getattr(repl, "_workspace", None)
    untrusted_count = 0
    if workspace is not None and not _workspace_trusted(repl):
        untrusted_count = len(servers)

    lines = [f"MCP Servers — agent: {name}"]
    lines.append(f"{'Name':<16} {'Transport':<16} {'Status':<18} {'Source':<14} {'Tools'}")
    lines.append("─" * 80)
    for ps in servers:
        s = ps.data
        sname = str(s.get("name") or "")
        transport = str(s.get("transport") or "")
        if sname in connected:
            status = "● connected"
            client = manager.get_client(sname) if manager else None
            tool_count = f"{len(client.tools)} tools" if client else "—"
        else:
            status = "○ not connected"
            tool_count = "—"
        lines.append(f"{sname:<16} {transport:<16} {status:<18} [{ps.source}]{'':<7} {tool_count}")

    if untrusted_count:
        lines.append("")
        lines.append(f"⚠ {untrusted_count} servers withheld (workspace not trusted). "
                     f"Run 'localharness start' to grant trust.")

    lines.append("")
    lines.append("  /mcp add        Add a new server")
    lines.append("  /mcp edit <n>   Edit a server")
    lines.append("  /mcp test <n>   Test connectivity")
    lines.append("  /mcp remove <n> Remove a server")

    await _say(repl, "\n".join(lines))


async def _add(repl: Any) -> None:
    """Add a new MCP server interactively."""
    name = _session_agent_name(repl)

    server_name = await _prompt(repl, "Name: ")
    if not server_name:
        await _say(repl, "No name given.")
        return

    transport_choice = await _prompt(repl, "Transport: [1] stdio  [2] streamable_http  → ")
    transport = "streamable_http" if transport_choice.strip() == "2" else "stdio"

    command: Optional[str] = None
    args: list[str] = []
    url: Optional[str] = None
    env: dict[str, str] = {}
    headers: dict[str, str] = {}

    if transport == "stdio":
        command = await _prompt(repl, "Command: ")
        if not command:
            await _say(repl, "No command given.")
            return
        args_str = await _prompt(repl, "Args (space-separated): ")
        args = args_str.split() if args_str else []
        while True:
            kv = await _prompt(repl, "Env (key=value, blank to skip): ")
            if not kv:
                break
            if "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
    else:
        url = await _prompt(repl, "URL: ")
        if not url:
            await _say(repl, "No URL given.")
            return
        while True:
            kv = await _prompt(repl, "Headers (key=value, blank to skip): ")
            if not kv:
                break
            if "=" in kv:
                k, v = kv.split("=", 1)
                headers[k.strip()] = v.strip()

    # Validate with pydantic
    try:
        config = MCPServerConfig(
            name=server_name, transport=transport,
            command=command, args=args, env=env, url=url, headers=headers,
        )
    except Exception as exc:
        await _say(repl, f"Invalid config: {exc}", style="system.error")
        return

    # One-shot connect test
    await _say(repl, f"Testing '{server_name}'…")
    ok, msg, tool_names = await _test_connect(config)
    if ok:
        await _say(repl, f"  ✓ Connected — {msg}")
        if tool_names:
            await _say(repl, f"    {_format_tool_names(tool_names)}")
    else:
        await _say(repl, f"  ✗ Connection failed: {msg}", style="system.error")

    # Ask to save
    if not await _confirm(repl, "Save?"):
        await _say(repl, "Not saved.")
        return

    # Write to agent file
    servers = _read_agent_servers(repl, name)
    servers = [s for s in servers if s.get("name") != server_name]
    servers.append(_build_entry(server_name, transport, command, args, url, env, headers))
    path = _write_agent_servers(repl, name, servers)
    if path is None:
        await _say(repl, "Could not determine agent file path.", style="system.error")
        return
    await _say(repl, f"Saved to {path}")

    # Trust notice / hot-add
    workspace = getattr(repl, "_workspace", None)
    if workspace is not None and not _workspace_trusted(repl):
        await _say(repl, "Note: this workspace is not trusted. The server will not connect "
                         "until you grant trust at 'localharness start'.")
    elif ok:
        connected = await _hot_add(repl, config)
        if connected:
            await _say(repl, f"'{server_name}' is now connected in this session.")
        else:
            await _say(repl, f"'{server_name}' saved but not connected this session.")


async def _edit(repl: Any, name: str) -> None:
    """Edit an existing MCP server. Pre-fills all fields; blank = keep current."""
    agent_name = _session_agent_name(repl)
    servers = _read_agent_servers(repl, agent_name)
    existing = _find_server(servers, name)
    if existing is None:
        await _say(repl, f"Unknown server '{name}'.")
        return

    transport = str(existing.get("transport") or "stdio")
    command = existing.get("command")
    args = list(existing.get("args", []))
    url = existing.get("url")
    env = dict(existing.get("env", {}))
    headers = dict(existing.get("headers", {}))

    if transport == "stdio":
        new_command = await _prompt(repl, f"Command (current: {command or '—'}): ")
        if new_command:
            command = new_command
        new_args = await _prompt(repl, f"Args (current: {' '.join(args) if args else '—'}): ")
        if new_args:
            args = new_args.split()
        while True:
            kv = await _prompt(repl, "Env change (key=value, 'del key', blank to skip): ")
            if not kv:
                break
            if kv.startswith("del "):
                env.pop(kv[4:].strip(), None)
            elif "=" in kv:
                k, v = kv.split("=", 1)
                env[k.strip()] = v.strip()
    else:
        # Mask the URL: show scheme+host only, never the query string (which may carry a token)
        display_url = url or "—"
        if url and "?" in url:
            display_url = url.split("?")[0] + "?●●●●"
        new_url = await _prompt(repl, f"URL (current: {display_url}): ")
        if new_url:
            url = new_url
        while True:
            kv = await _prompt(repl, "Header change (key=value, 'del key', blank to skip): ")
            if not kv:
                break
            if kv.startswith("del "):
                headers.pop(kv[4:].strip(), None)
            elif "=" in kv:
                k, v = kv.split("=", 1)
                headers[k.strip()] = v.strip()

    try:
        config = MCPServerConfig(
            name=name, transport=transport,
            command=command, args=args, env=env, url=url, headers=headers,
        )
    except Exception as exc:
        await _say(repl, f"Invalid config: {exc}", style="system.error")
        return

    await _say(repl, f"Testing '{name}'…")
    ok, msg, tool_names = await _test_connect(config)
    if ok:
        await _say(repl, f"  ✓ Connected — {msg}")
        if tool_names:
            await _say(repl, f"    {_format_tool_names(tool_names)}")
    else:
        await _say(repl, f"  ✗ Connection failed: {msg}", style="system.error")

    if not await _confirm(repl, "Save?"):
        await _say(repl, "Not saved.")
        return

    servers = [s for s in servers if s.get("name") != name]
    servers.append(_build_entry(name, transport, command, args, url, env, headers))
    path = _write_agent_servers(repl, agent_name, servers)
    if path is None:
        await _say(repl, "Could not determine agent file path.", style="system.error")
        return
    await _say(repl, f"Saved to {path}")

    # Hot-edit: disconnect old, connect new
    await _hot_remove(repl, name)
    if ok:
        connected = await _hot_add(repl, config)
        if connected:
            await _say(repl, f"'{name}' reconnected in this session.")


async def _test(repl: Any, name: str) -> None:
    """Test connectivity for a named server (one-shot, no persist)."""
    agent_name = _session_agent_name(repl)
    servers = _read_agent_servers(repl, agent_name)
    existing = _find_server(servers, name)
    if existing is None:
        await _say(repl, f"Unknown server '{name}'.")
        return

    # If already connected in the live session, just re-list its tools
    manager = _mcp_manager(repl)
    if manager and name in manager.connected_servers:
        client = manager.get_client(name)
        if client:
            tool_names = [t.info().name for t in client.tools]
            lines = [f"Testing '{name}'…", f"  ✓ Connected — {len(tool_names)} tools:"]
            if tool_names:
                lines.append(f"    {_format_tool_names(tool_names)}")
            await _say(repl, "\n".join(lines))
            return

    # Trust warning: one-shot test opens a live connection in an untrusted workspace
    workspace = getattr(repl, "_workspace", None)
    if workspace is not None and not _workspace_trusted(repl):
        await _say(repl, "⚠ Workspace not trusted — this test opens a live connection "
                         "to a server defined in an untrusted project.", style="system.error")

    # One-shot test
    try:
        config = _config_from_dict(existing)
    except Exception as exc:
        await _say(repl, f"Invalid config: {exc}", style="system.error")
        return

    await _say(repl, f"Testing '{name}'…")
    ok, msg, tool_names = await _test_connect(config)
    if ok:
        lines = [f"  ✓ Connected — {msg}:"]
        if tool_names:
            lines.append(f"    {_format_tool_names(tool_names)}")
        await _say(repl, "\n".join(lines))
    else:
        await _say(repl, f"  ✗ Connection failed: {msg}", style="system.error")


async def _remove(repl: Any, name: str) -> None:
    """Remove a named server from config and the live session."""
    agent_name = _session_agent_name(repl)
    servers = _read_agent_servers(repl, agent_name)
    existing = _find_server(servers, name)
    if existing is None:
        await _say(repl, f"Unknown server '{name}'.")
        return

    if not await _confirm(repl, f"Remove MCP server '{name}'?"):
        await _say(repl, "Not removed.")
        return

    servers = [s for s in servers if s.get("name") != name]
    path = _write_agent_servers(repl, agent_name, servers)
    if path is None:
        await _say(repl, "Could not determine agent file path.", style="system.error")
        return
    await _say(repl, f"Removed '{name}' from {path}")

    # Hot-remove from live session
    await _hot_remove(repl, name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_mcp_slash(repl: Any, args: str) -> None:
    """The `/mcp` slash command entry point.

    `repl` is the OrchestratorREPL instance; `args` is the text after `/mcp`.
    """
    args = args.strip()
    if not args:
        await _list_servers(repl)
        return
    if args == "help":
        await _say(repl,
            "/mcp — Manage MCP servers for the session's agent\n"
            "  /mcp               List all servers\n"
            "  /mcp add           Add a new server\n"
            "  /mcp edit <name>   Edit a server\n"
            "  /mcp test <name>   Test connectivity\n"
            "  /mcp remove <name> Remove a server")
        return

    words = args.split()
    verb = words[0]
    if verb == "add":
        await _add(repl)
    elif verb == "edit":
        if len(words) < 2:
            await _say(repl, "Usage: /mcp edit <name>")
        else:
            await _edit(repl, words[1])
    elif verb == "test":
        if len(words) < 2:
            await _say(repl, "Usage: /mcp test <name>")
        else:
            await _test(repl, words[1])
    elif verb == "remove":
        if len(words) < 2:
            await _say(repl, "Usage: /mcp remove <name>")
        else:
            await _remove(repl, words[1])
    else:
        await _say(repl, f"Unknown /mcp subcommand: {verb}. Try /mcp help.")
