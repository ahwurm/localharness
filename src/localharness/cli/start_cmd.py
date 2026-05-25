"""localharness start command — smart routing REPL entry point."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.prompt import IntPrompt
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def _discover_agents_for_start(config_dir: Path) -> list[dict]:
    """Return agents from global config dir and local .localharness/agents/, local overrides."""
    global_dir = config_dir / "agents"
    local_dir = Path(".localharness") / "agents"

    agents: dict[str, dict] = {}

    if global_dir.exists():
        for f in sorted(global_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if "name" not in data:
                    data["name"] = f.stem
                agents[f.stem] = data
            except Exception:
                pass

    if local_dir.exists():
        for f in sorted(local_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
                if "name" not in data:
                    data["name"] = f.stem
                agents[f.stem] = data
            except Exception:
                pass

    return list(agents.values())


async def _start_async(agent_name: str | None, debug: bool, config_dir: str) -> None:
    """Async entry point: discover agent, wire dependencies, run REPL."""
    from localharness.agent.context import ContextManager
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.channels.terminal import TerminalChannel
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.init_cmd import init_app
    from localharness.config.loader import ConfigLoader
    from localharness.config.models import AgentConfig
    from localharness.core.bus import EventBus
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.tools.registry import ToolRegistry, register_builtin_tools

    cfg_path = Path(config_dir).expanduser()
    config_file = cfg_path / "config.yaml"

    # Run init if config missing
    if not config_file.exists():
        console.print("No config found. Running init...")
        init_app()

    # Load harness config
    loader = ConfigLoader(config_dir=cfg_path)
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Cannot load config: {exc}")
        raise typer.Exit(1)

    # Discover agents
    agents = _discover_agents_for_start(cfg_path)

    if agent_name:
        # --agent flag: find by name
        match = [a for a in agents if a.get("name") == agent_name]
        if not match:
            err_console.print(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found.")
            raise typer.Exit(1)
        selected_data = match[0]
    elif not agents:
        # No agents: create default
        console.print("[yellow]No agents configured. Creating default agent...[/yellow]")
        agents_dir = cfg_path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        default_data = _build_agent_yaml("default", "General-purpose assistant", None)
        import yaml as _yaml
        (agents_dir / "default.yaml").write_text(
            _yaml.dump(default_data, default_flow_style=False), encoding="utf-8"
        )
        agents = [default_data]
        selected_data = default_data
    elif len(agents) == 1:
        selected_data = agents[0]
    else:
        # Multiple agents: show picker
        table = Table(title="Available Agents")
        table.add_column("No.")
        table.add_column("Name")
        table.add_column("Role")
        for i, a in enumerate(agents, start=1):
            table.add_row(str(i), a.get("name", ""), a.get("role", ""))
        console.print(table)
        choice = IntPrompt.ask("Select agent", default=1)
        idx = max(1, min(choice, len(agents))) - 1
        selected_data = agents[idx]

    # Load full AgentConfig (uses ConfigLoader for inheritance)
    agent_name_str: str = selected_data.get("name", "default")
    try:
        agent_config = loader.load_agent(agent_name_str)
    except Exception:
        # Fall back to building from raw data
        agent_config = AgentConfig(
            name=agent_name_str,
            role=selected_data.get("role", "General-purpose assistant"),
            model=selected_data.get("model", "inherit"),
        )

    # Wire dependencies
    provider = harness.provider
    resolved_model = (
        agent_config.model
        if agent_config.model != "inherit"
        else provider.default_model
    )
    llm_cfg = LLMConfig(
        base_url=provider.base_url,
        model=resolved_model,
        api_key=provider.api_key,
        timeout_seconds=provider.timeout_seconds,
        tool_call_mode="native" if provider.supports_function_calling else "xml",
    )
    llm = LLMClient(llm_cfg)
    bus = EventBus()
    tool_registry = ToolRegistry()
    await register_builtin_tools(tool_registry)
    ctx_mgr = ContextManager(
        max_context_tokens=agent_config.context.max_context_tokens,
        preserve_first_n=agent_config.context.preserve_first_n_messages,
        preserve_last_n=agent_config.context.preserve_last_n_messages,
    )
    perm_eval = PermissionEvaluator()
    agent_loop = AgentLoop(
        config=agent_config,
        llm=llm,
        bus=bus,
        context_manager=ctx_mgr,
        tool_registry=tool_registry,
        permission_evaluator=perm_eval,
    )
    channel = TerminalChannel(bus=bus, config={})

    from localharness.cli.repl import OrchestratorREPL

    repl = OrchestratorREPL(agent_loop=agent_loop, channel=channel, bus=bus)

    try:
        await repl.run()
    except KeyboardInterrupt:
        console.print("\nGoodbye.")


def start_app(
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Start specific agent")] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    config_dir: Annotated[str, typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")] = "~/.localharness",
) -> None:
    """Launch the agent REPL. Zero to chatting in one command."""
    try:
        asyncio.run(_start_async(agent, debug, config_dir))
    except KeyboardInterrupt:
        console.print("\nGoodbye.")
