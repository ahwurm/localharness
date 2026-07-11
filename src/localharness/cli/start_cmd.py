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


async def _probe_llm(
    llm: Any, max_retries: int = 3, delay: float = 2.0
) -> tuple[bool, str | None, int | None]:
    """Probe LLM reachability with retry for cold start.

    Returns (reachable, probed_tool_call_mode, served_context_window). Mode/window are
    None if the probe fails. The served window is the single source of truth for the
    effective context budget — callers must use it rather than the config default.
    Callers must feed the probed mode into LLMConfig rather than using the stored
    provider.supports_function_calling flag (FIDEL-04).
    """
    import asyncio as _asyncio
    for attempt in range(max_retries):
        try:
            result = await llm.detect_capabilities()
            return True, result.tool_call_mode, result.context_window
        except Exception:
            if attempt < max_retries - 1:
                await _asyncio.sleep(delay)
    return False, None, None


def _effective_max_context(
    served_window: int | None, cfg_window: int, reserve: int
) -> int:
    """Single source of truth for the context budget.

    Derive from the SERVED max_model_len minus the output reserve. The config value is
    honored ONLY as an explicit cap when it already fits under served-reserve; otherwise
    the served-derived value wins. If the server didn't report a window, the config value
    is the only signal available.
    """
    if not served_window:
        return cfg_window
    served_effective = max(8_192, served_window - reserve)
    return cfg_window if cfg_window <= served_effective else served_effective


def _resolve_timeout(agent_timeout: float | None, provider_timeout: float) -> float:
    """Per-agent timeout override wins when set; otherwise the provider default.

    AgentConfig.timeout_seconds was previously never read at runtime — the start
    path always passed provider.timeout_seconds — so the per-agent override the
    reference-architecture docs tell slow-decode users to set was dead config."""
    return agent_timeout if agent_timeout is not None else provider_timeout


def _ensure_packaged_tools(config_dir: Path) -> None:
    """Install packaged helper scripts into <config-dir>/tools (idempotent).

    The frontend-designer builtin shells out to design-screenshot.js by path —
    ship it with the package so the builtin works out of the box. A missing or
    uncopyable asset must never block start; the agent reports the gap itself."""
    tools_dir = config_dir / "tools"
    dest = tools_dir / "design-screenshot.js"
    if dest.exists():
        return
    try:
        from importlib import resources
        src = resources.files("localharness").joinpath("assets", "design-screenshot.js")
        tools_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        dest.chmod(0o755)
    except Exception as exc:  # noqa: BLE001
        err_console.print(f"[yellow]⚠ could not install design-screenshot.js: {exc}[/yellow]")


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


def _migrate_legacy_root_agent_yaml(agents_dir: Path) -> None:
    """Phase 33.1 (ORCH-01/03): one-time root-agent rename on the global agents dir —
    agents/default.yaml -> agents/orchestrator.yaml with the name: field REWRITTEN
    (a bare file rename is not enough: discovery reads the name: key, and the legacy
    file carries name: default). The MemoryStore data directory migrates separately
    inside MemoryStore.open() (Phase 33.1 plan 01).

    Idempotent + crash-safe by construction:
    - no default.yaml -> no-op (fresh install, or already migrated);
    - default.yaml whose name: is not 'default' -> no-op (not the minted root);
    - orchestrator.yaml already exists:
        * parsed-equal to the would-be migration -> crash remnant of a previous run
          that died between write and unlink -> finish the job (unlink default.yaml);
        * different content -> GENUINE collision (the user has their own
          'orchestrator' agent): refuse loudly, keep the legacy root under its old
          name — never merge, never clobber;
    - normal path: write orchestrator.yaml FIRST, then unlink default.yaml (a crash
      between the two leaves both files; the remnant branch completes it next start).
    """
    legacy = agents_dir / "default.yaml"
    if not legacy.exists():
        return
    try:
        data = yaml.safe_load(legacy.read_text(encoding="utf-8")) or {}
    except Exception:
        return  # unreadable legacy yaml: leave it to discovery's tolerant path
    if not isinstance(data, dict):
        return  # a list/scalar default.yaml is not an agent config — don't crash startup
    if data.get("name", "default") != "default":
        return
    data["name"] = "orchestrator"
    target = agents_dir / "orchestrator.yaml"
    if target.exists():
        try:
            existing = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        except Exception:
            existing = None
        if existing == data:
            legacy.unlink(missing_ok=True)  # crash remnant — the migrated copy is already in place
        else:
            console.print(
                "[yellow]Warning:[/yellow] cannot rename the root agent 'default' -> "
                "'orchestrator': agents/orchestrator.yaml already exists (an "
                "unrelated agent). Keeping the root under its old name 'default'. "
                "To resolve, rename your 'orchestrator' agent — note it can no "
                "longer be delegated to (the root's name is guarded)."
            )
        return
    target.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    legacy.unlink(missing_ok=True)  # two-process first-start race: the other may have unlinked


async def _start_async(agent_name: str | None, verbose: bool, debug: bool, config_dir: str,
                       channel_mode: str = "terminal", subagents: bool = False) -> None:
    """Async entry point: discover agent, wire dependencies, run REPL."""
    import time as _time
    import uuid

    from localharness.agent.context import CompactionPipeline, ContextManager, TokenCounter
    from localharness.agent.loop import AgentLoop
    from localharness.agent.permissions import PermissionEvaluator
    from localharness.channels.terminal import TerminalChannel
    from localharness.cli.agent_cmd import _build_agent_yaml
    from localharness.cli.init_cmd import init_app
    from localharness.config.loader import ConfigLoader
    from localharness.config.models import AgentConfig
    from localharness.core.bus import EventBus
    from localharness.memory.sqlite import MemoryStore, _migrate_legacy_root_agent_dir
    from localharness.plugins.loader import PluginLoader
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.tools.hooks import HookSystem
    from localharness.tools.mcp import MCPClientManager
    from localharness.tools.registry import ToolRegistry
    from localharness.tools.builtin import register_builtin_tools

    cfg_path = Path(config_dir).expanduser()
    config_file = cfg_path / "config.yaml"

    # No config → welcome message + exit
    if not config_file.exists():
        from localharness.orchestrator.router import AgentCreationFlow
        console.print(AgentCreationFlow.no_config_message())
        raise typer.Exit(0)

    # Load harness config
    loader = ConfigLoader(config_dir=cfg_path)
    try:
        harness = loader.load_harness()
    except Exception as exc:
        err_console.print(f"[bold red]Error:[/bold red] Cannot load config: {exc}")
        raise typer.Exit(1)

    _ensure_packaged_tools(cfg_path)

    # Discover agents (migrate the legacy root-agent YAML first so discovery reads the
    # rewritten name: field, not the stale name: default — Phase 33.1 ORCH-01/03)
    _migrate_legacy_root_agent_yaml(cfg_path / "agents")
    agents = _discover_agents_for_start(cfg_path)

    if agent_name:
        # --agent flag: find by name
        match = [a for a in agents if a.get("name") == agent_name]
        if not match and agent_name == "default":
            # Phase 33.1: the root agent was renamed default -> orchestrator; keep old
            # muscle memory / scripts working instead of hard-erroring (ORCH-03).
            match = [a for a in agents if a.get("name") == "orchestrator"]
            if match:
                console.print(
                    "[yellow]Note:[/yellow] the root agent was renamed 'default' -> "
                    "'orchestrator'; starting 'orchestrator'."
                )
        if not match:
            err_console.print(f"[bold red]Error:[/bold red] Agent '{agent_name}' not found.")
            raise typer.Exit(1)
        selected_data = match[0]
    elif not agents:
        # No agents: mint the root agent as 'orchestrator' (ORCH-01)
        console.print("[yellow]No agents configured. Creating the orchestrator (root agent)...[/yellow]")
        agents_dir = cfg_path / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        root_data = _build_agent_yaml("orchestrator", "General-purpose assistant", None)
        import yaml as _yaml
        (agents_dir / "orchestrator.yaml").write_text(
            _yaml.dump(root_data, default_flow_style=False), encoding="utf-8"
        )
        agents = [root_data]
        selected_data = root_data
    elif len(agents) == 1 or not subagents:
        # Default: open straight to the banner. Pass --subagents to pick from the
        # multi-agent table instead.
        # Prefer 'default' if it still exists: post-migration that only happens when the
        # YAML migration REFUSED (a name collision with a user's own 'orchestrator'
        # agent), and the un-migrated legacy root must keep winning selection or the
        # user lands in a different agent and loses their memory continuity. Normal
        # installs have no default.yaml after migration, so 'orchestrator' wins.
        selected_data = next(
            (a for a in agents if a.get("name") == "default"),
            next((a for a in agents if a.get("name") == "orchestrator"), agents[0]),
        )
    else:
        # --subagents + multiple agents: show picker
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
    agent_name_str: str = selected_data.get("name", "orchestrator")
    try:
        agent_config = loader.load_agent(agent_name_str)
    except Exception:
        # Fall back to building from raw data
        agent_config = AgentConfig(
            name=agent_name_str,
            role=selected_data.get("role", "General-purpose assistant"),
            model=selected_data.get("model", "inherit"),
        )

    # --- Capability floor (P-A): sync the module flag from config, then strip web ingestion from
    # the ROOT agent. Root inherits 'global' scope where web_* are registered, so without this it
    # would co-resident web ingestion with bash/write/edit (prompt-injection->host hole). It delegates
    # ingestion to the web-researcher subagent (no bash). KEEP tool_result_get — not untrusted-ingest.
    from localharness.tools.capabilities import apply_root_capability_floor, set_floor_enabled
    set_floor_enabled(harness.org.enforce_capability_floor)
    apply_root_capability_floor(agent_config.tools)

    # Wire dependencies
    provider = harness.provider
    resolved_model = (
        agent_config.model
        if agent_config.model != "inherit"
        else provider.default_model
    )
    # Build initial client for the probe (tool_call_mode will be overwritten by probe result).
    _initial_cfg = LLMConfig(
        base_url=provider.base_url,
        model=resolved_model,
        api_key=provider.api_key,
        timeout_seconds=_resolve_timeout(agent_config.timeout_seconds, provider.timeout_seconds),
    )
    _probe_client = LLMClient(_initial_cfg)

    # Startup probe — local LLMs may need warm-up; probe returns the real tool_call_mode (FIDEL-04)
    # AND the served context window (single source of truth for the budget).
    probe_ok, probed_mode, served_window = await _probe_llm(_probe_client)
    if not probe_ok and harness.server is not None:
        # Harness-managed server (init guided setup) — start it instead of erroring
        # (covers reboots). If a pid is alive, it's mid-load: just wait.
        from localharness.provider import server as managed_server
        try:
            if managed_server.server_pid(cfg_path) is None:
                console.print("Managed vLLM is not running — starting it (model load can take several minutes)...")
                managed_server.start_server(cfg_path, managed_server.serve_command(harness.server))
            else:
                console.print("Managed vLLM is still loading — waiting...")
            await managed_server.wait_ready(provider.base_url, config_dir=cfg_path)
            probe_ok, probed_mode, served_window = await _probe_llm(_probe_client)
        except (RuntimeError, TimeoutError) as exc:
            err_console.print(f"[bold red]Error:[/bold red] managed vLLM failed to start: {exc}")
    if not probe_ok:
        err_console.print(
            f"[bold red]Error:[/bold red] Cannot reach model '{resolved_model}' "
            f"at {provider.base_url}"
        )
        err_console.print("Check that your LLM backend is running and try again.")
        raise typer.Exit(1)

    # config.yaml is the SINGLE SOURCE OF TRUTH for the window (now inheritance-resolved,
    # which kills the 61,440-in-a-131,072-world bug). We do NOT silently override it — we
    # VALIDATE against the served window and FAIL LOUD if it would 400 mid-session, so the
    # value the user sees in config.yaml is exactly what the agent runs on.
    from localharness.agent.context import RESPONSE_RESERVE_TOKENS
    _cfg_window = agent_config.context.max_context_tokens
    if _effective_max_context(served_window, _cfg_window, RESPONSE_RESERVE_TOKENS) != _cfg_window:
        usable = (served_window or 0) - RESPONSE_RESERVE_TOKENS
        err_console.print(
            f"[bold red]Error:[/bold red] config max_context_tokens={_cfg_window} exceeds the "
            f"served model's usable window ({usable} = {served_window}−{RESPONSE_RESERVE_TOKENS} "
            f"output reserve). It would 400 mid-session. Set max_context_tokens ≤ {usable} in "
            f"your config.yaml, or run `localharness init` to fit it automatically."
        )
        raise typer.Exit(1)

    # Build the real LLMClient with the probe-derived tool_call_mode (FIDEL-04).
    # A model swap re-probes via _probe_llm before constructing the new LLMClient.
    llm_cfg = LLMConfig(
        base_url=provider.base_url,
        model=resolved_model,
        api_key=provider.api_key,
        timeout_seconds=_resolve_timeout(agent_config.timeout_seconds, provider.timeout_seconds),
        tool_call_mode=probed_mode or "native",
    )
    llm = LLMClient(llm_cfg)

    start_time = _time.monotonic()

    # --- Startup state tracker ---
    warnings: list[str] = []
    plugins_loaded = 0
    mcp_connected = 0
    mcp_failed = 0

    # --- 1. HARD requirements (abort on failure) ---
    # Phase 33.1: adopt the legacy root data dir (agents/default -> agents/orchestrator)
    # BEFORE the EventBus below materializes agents/<root>/bus-events.jsonl. The store's
    # own open()-time adoption (plan 01) refuses once agents/orchestrator/ exists, and the
    # bus's persist_path.parent.mkdir would create it first — stranding the old memories in
    # agents/default/. Doing it here preserves the single-adoption contract (open()'s call
    # then no-ops on the existing dir, while its SQL row re-key still runs).
    _migrate_legacy_root_agent_dir(cfg_path, agent_name_str)
    agent_dir = cfg_path / "agents" / agent_name_str
    events_path = agent_dir / "bus-events.jsonl"
    bus = EventBus(persist_path=events_path)
    sitting_id = str(uuid.uuid4())  # SESS-01: one session per SITTING, minted once
    # LLMClient built above with probe-derived tool_call_mode.

    # --- 2. Core infrastructure ---
    tool_registry = ToolRegistry()
    await register_builtin_tools(tool_registry)

    # --- 3. Hook system (soft) ---
    hook_system: HookSystem | None = None
    try:
        hook_system = HookSystem()
        hook_system.wire_to_registry(tool_registry)
    except Exception as exc:
        warnings.append(f"hooks: {exc}")
        hook_system = None

    # --- 4. Memory store (soft -- degrade to None) ---
    memory_store: MemoryStore | None = None
    try:
        memory_store = MemoryStore(
            agent_id=agent_name_str,
            division_id=agent_config.division or "default",
            org_id="default",
            base_dir=str(cfg_path),
            bus=bus,
        )
        await memory_store.open()
    except Exception as exc:
        warnings.append(f"memory: {exc} (in-memory mode)")
        memory_store = None

    # SESS-02: open the sessions row for this sitting (soft — a session-row failure must
    # not cost the whole memory subsystem). All three args are already-resolved locals.
    _session_started = False
    if memory_store is not None:
        try:
            await memory_store.create_session(
                sitting_id,
                budget=agent_config.permissions.budget.model_dump(),
                model=resolved_model,
                context_tokens_available=_cfg_window,
            )
            _session_started = True
        except Exception as exc:
            warnings.append(f"session-start: {exc}")

    # Prediction-error write gate (WRITE-03/06): harness-initiated memory writes from bus
    # signals. Default-on, config-off (agent.memory.write_gate_enabled) — cruncher-style.
    write_gate = None
    if memory_store is not None and getattr(agent_config.memory, "write_gate_enabled", True):
        try:
            from localharness.memory.gate import WriteGate
            write_gate = WriteGate(memory_store, bus, agent_name_str)
            await write_gate.open()
        except Exception as exc:
            warnings.append(f"memory write-gate: {exc}")
            write_gate = None

    # SESS-02/05: sitting-scoped counters feeding the payload-first close-out summary
    # (zero model calls — derived from bus signals the gate already composes payload-first).
    # Same agent_id-filtered bus seam as the write gate; closed before the summary reads.
    session_acc = None
    if memory_store is not None:
        try:
            from localharness.cli.session_accumulator import SessionAccumulator
            session_acc = SessionAccumulator(bus, agent_name_str)
            await session_acc.open()
        except Exception as exc:
            warnings.append(f"session-accumulator: {exc}")

    # Idle-time consolidation (CONS-01..06): session-start staleness check + in-session
    # idle timer, cooperatively cancelled by any user turn. Phase 36: the LLM replay seam is
    # now ON in production — the real LLMClient is bridged through LLMTextAdapter (36-03, the
    # SINGLE cancellable + char-bounded idle path) and passed as llm=, so the pass can write
    # chapters, reconcile the correction queue, and mine transcripts. Each of those is gated
    # per-step by an agent.memory.consolidation.* axis (schema_writer/reconcile/mining_enabled)
    # AND early-returns when llm is None, so the deterministic core stays byte-unchanged. The
    # try/except soft-degrades a wiring fault back to the deterministic pass (warnings.append).
    # on_promotion_sample=None DEFERRED (CONS-06): the SEMA-05 report already surfaces generated
    # chapters to the owner; wiring the Discord sample hook needs channel-construction reordering
    # (§10), out of scope here. Named seam, same contract as llm.
    consolidation_scheduler = None
    _cons_cfg = getattr(agent_config.memory, "consolidation", None)
    if memory_store is not None and _cons_cfg is not None and _cons_cfg.enabled:
        try:
            from localharness.memory.consolidation import ConsolidationScheduler
            from localharness.memory.idle_llm import LLMTextAdapter
            consolidation_scheduler = ConsolidationScheduler(
                memory_store, bus, agent_name_str, _cons_cfg, llm=LLMTextAdapter(llm)
            )
            await consolidation_scheduler.start()
        except Exception as exc:
            warnings.append(f"memory consolidation: {exc}")
            consolidation_scheduler = None

    # Collect-only predictive gate (Phase 34, COLL-01..04): per-tool statistical priors
    # score every outcome; user-signal triggers log labeled prediction errors. Score
    # everything, gate nothing — pure measurement feeding Phase 35's thresholds. Additive
    # bus subscribers only (WriteGate shape); zero loop changes, zero model calls.
    predictive_gate = None
    user_signal_detector = None
    _pg_cfg = getattr(agent_config.memory, "predictive_gate", None)
    if memory_store is not None and _pg_cfg is not None and _pg_cfg.enabled:
        try:
            from localharness.memory.predictive_gate import PredictiveGate
            predictive_gate = PredictiveGate(memory_store, bus, agent_name_str, _pg_cfg)
            await predictive_gate.open()
        except Exception as exc:
            warnings.append(f"predictive-gate: {exc}")
            predictive_gate = None
        try:
            from localharness.memory.user_signals import UserSignalDetector
            user_signal_detector = UserSignalDetector(memory_store, bus, agent_name_str, _pg_cfg)
            await user_signal_detector.open()
        except Exception as exc:
            warnings.append(f"user-signals: {exc}")
            user_signal_detector = None

    # PredictiveWriteGate (Phase 35, PGATE-01/02/03): the LIVE write decision — turns 34's
    # already-published SurpriseScored + correction-worded UserMessage into gated sub-0.7 fact
    # writes. Sibling subscriber (WriteGate shape), reusing the same _pg_cfg; gated on write_live
    # (the pre-committed KILL-revert lever) AND enabled. Its OWN try/except so a wiring fault
    # soft-degrades to motif-only capture and never crashes start.
    predictive_write_gate = None
    if memory_store is not None and _pg_cfg is not None and _pg_cfg.enabled and getattr(_pg_cfg, "write_live", True):
        try:
            from localharness.memory.predictive_write_gate import PredictiveWriteGate
            predictive_write_gate = PredictiveWriteGate(memory_store, bus, agent_name_str, _pg_cfg)
            await predictive_write_gate.open()
        except Exception as exc:
            warnings.append(f"predictive-write-gate: {exc}")
            predictive_write_gate = None

    # Queryable-handle tools: memory_search/memory_get (full fact bodies on demand) and
    # tool_result_get (restore evicted tool-result bodies). The ContentStore is shared with
    # the ContextManager below so eviction-writes and restore-reads hit the same map.
    from localharness.agent.context import ContentStore
    eviction_store = ContentStore()
    try:
        if memory_store is not None:
            # ALL memory tools register whenever a store exists (critic M5): the
            # inject_into_context flag gates INJECTION, not tool availability —
            # otherwise injection-off produces write-only memory (remember succeeds,
            # nothing can ever read it back).
            from localharness.tools.builtin.memory_tools import (
                MemoryGetTool,
                MemoryRememberTool,
                MemorySearchTool,
            )
            await tool_registry.register(MemorySearchTool(memory_store), scope="global")
            await tool_registry.register(MemoryGetTool(memory_store), scope="global")
            await tool_registry.register(MemoryRememberTool(memory_store), scope="global")
        if agent_config.context.tool_result_eviction:
            from localharness.tools.builtin.tool_result_get_tool import ToolResultGetTool
            await tool_registry.register(ToolResultGetTool(eviction_store), scope="global")
    except Exception as exc:
        warnings.append(f"queryable-handle tools: {exc}")

    # Bind the root agent's store-backed verb tools (web_fetch / web_page_query / tool_result_get)
    # to the root ContentStore, so the root has ONE per-agent store (web pages + evicted bodies) and
    # children — which rebind to their OWN store in dispatch — are isolated from it.
    from localharness.tools.builtin import bind_agent_store_tools
    bind_agent_store_tools(tool_registry, eviction_store)

    # --- 5. Plugin loader (soft) ---
    plugin_loader: PluginLoader | None = None
    try:
        if hook_system is not None:
            plugin_loader = PluginLoader(tool_registry, hook_system)
            loaded_names = await plugin_loader.discover_all()
            plugins_loaded = len(loaded_names)
    except Exception as exc:
        warnings.append(f"plugins: {exc}")

    # --- 6. MCP client manager (soft) ---
    mcp_manager: MCPClientManager | None = None
    try:
        mcp_configs = agent_config.tools.mcp_servers
        if mcp_configs:
            mcp_manager = MCPClientManager(tool_registry)
            results = await mcp_manager.startup(mcp_configs)
            mcp_connected = sum(1 for v in results.values() if v > 0)
            mcp_failed = sum(1 for v in results.values() if v == 0)
    except Exception as exc:
        warnings.append(f"mcp: {exc}")

    # --- 6b. Model-aware token counter (one instance, injected everywhere) ---
    # Counts via the served model's exact tokenizer (vLLM /tokenize) when the runtime serves it,
    # so budget gates fire at the real fraction. #8: a runtime WITHOUT that endpoint (Ollama /
    # LM Studio / llama.cpp) no longer hard-fails `start` — TokenCounter self-detects and falls
    # back to the approximate cl100k meter with one clear log line (gates then fire conservatively
    # rather than overflowing). This except only trips if NO tokenizer is available at all
    # (tiktoken missing) — a genuinely unusable environment.
    try:
        token_counter = TokenCounter(base_url=provider.base_url, model=resolved_model)
    except RuntimeError as exc:
        err_console.print(
            f"[bold red]Error:[/bold red] {exc}\n"
            f"No token counter is available (tiktoken not installed). Run 'uv sync' to install it."
        )
        raise typer.Exit(1)

    # --- 7. Compaction pipeline (soft) ---
    pipeline: CompactionPipeline | None = None
    try:
        compact_md_path = agent_dir / "compact.md"

        from localharness.agent.context import make_compaction_summarize_fn
        pipeline = CompactionPipeline(
            token_counter=token_counter,
            tool_result_cap=agent_config.context.max_tool_output_chars,
            preserve_first_n=agent_config.context.preserve_first_n_messages,
            preserve_last_n=agent_config.context.preserve_last_n_messages,
            llm_summarize_fn=make_compaction_summarize_fn(llm),  # shared, tuple-unpack tested
            compact_md_path=compact_md_path,
        )
    except Exception as exc:
        warnings.append(f"compaction: {exc}")
        pipeline = None

    # --- 8. Context manager (with pipeline) ---
    ctx_mgr = ContextManager(
        max_context_tokens=agent_config.context.max_context_tokens,
        preserve_first_n=agent_config.context.preserve_first_n_messages,
        preserve_last_n=agent_config.context.preserve_last_n_messages,
        pipeline=pipeline,
        eviction_store=eviction_store,
        tool_evict_threshold_chars=agent_config.context.tool_result_evict_threshold_chars,
        tool_evict_enabled=agent_config.context.tool_result_eviction,
        token_counter=token_counter,
    )

    # --- 9. Orchestrator layer ---
    from localharness.orchestrator.router import AgentCreationFlow
    from localharness.orchestrator.cards import AgentCardRegistry
    card_registry = AgentCardRegistry()
    for agent_data in agents:
        try:
            a_name = agent_data.get("name", "")
            a_cfg = loader.load_agent(a_name)
            card_registry.register_from_config(a_cfg)
        except Exception:
            pass  # skip agents that fail to load — non-fatal
    orchestrator = AgentCreationFlow(card_registry=card_registry)

    # --- 9b. Agent delegation tool (ORCH-04 / SUBAGENT-05) ---
    # Bug#1 fix: the runner is built via the module-level make_explore_agent_runner seam (T1)
    # and the AgentTool is registered at GLOBAL scope. The parent loop's agent is the `default`
    # agent (agent_name_str), and get_tools_for_agent resolves agent-scoped tools by that name —
    # an agent_id="orchestrator" registration was NEVER offered to `default`, so delegation was
    # dead. Global scope matches the bench semantics (runner.py registers `agent` globally) and
    # makes the running parent actually see + use the tool. No global-name collision: the builtins
    # are read/glob/grep/write/bash_exec. parent_session_id is read at call time via the getter
    # (agent_loop is constructed just below, before any turn runs / the model can delegate).
    from localharness.tools.builtin.agent_tool import AgentTool
    from localharness.agent.subagent import make_explore_agent_runner

    perm_eval = PermissionEvaluator()

    # Built-in subagents wired in the runner (subagent.make_explore_agent_runner) — advertise them
    # alongside any configured agent cards so the model knows it can delegate to them. search-verifier
    # is a standalone capability (route a user's "re-check X" straight to it); the web-researcher
    # nests it only when its task asks for verification (rigor=on-request default) or under
    # RESEARCH_RIGOR=high (auto-verify material claims).
    # v0.5.3: every default builtin is quarantined-or-read-only; bash-holding specialist roles
    # (data-analyst, frontend-designer) ship as opt-in examples/agents/ configs instead.
    available_agent_names = ["explore", "web-researcher", "cruncher",
                             "search-verifier"] + [c.name for c in card_registry.all_cards()]

    _run_agent = make_explore_agent_runner(
        llm=llm,
        bus=bus,
        base_registry=tool_registry,
        permission_evaluator=perm_eval,
        get_parent_session_id=lambda: agent_loop.current_session_id,
        # bypass_cache: a yaml the model just WROTE must be dispatchable in the same turn
        load_agent=lambda n: loader.load_agent(n, bypass_cache=True),
        # Children inherit the parent's exact /tokenize counter + resolved window so the
        # sub-agent fleet accounts for context correctly instead of bare 131,072 + tiktoken.
        token_counter=token_counter,
        max_context_tokens=agent_config.context.max_context_tokens,
        # Real, config-capped recursion depth: the orchestrator is depth 0; a non-leaf child
        # (e.g. web-researcher) may nest a grandchild (search-verifier) up to this cap.
        depth=0,
        max_subagent_depth=agent_config.max_subagent_depth,
        available_agents=available_agent_names,
        # HIER-02: cruncher runs persist their gist tree into the agent's memory graph.
        memory_store=memory_store,
        # Grant keystone: the root's ContentStore is the parent store children read through when the
        # model delegates with grant_handles (it IS `eviction_store` — start_cmd:315/406 — so an
        # evicted/handle id the model sees in a stub is exactly what it can grant to a cruncher).
        parent_store=eviction_store,
        # Cruncher exec policy (agent.cruncher.*): offered to a clean-origin cruncher iff exec_enabled.
        cruncher_config=agent_config.cruncher,
    )

    agent_tool = AgentTool(
        agent_runner=_run_agent,
        available_agents=available_agent_names,
    )
    await tool_registry.register(agent_tool, scope="global")

    # --- 10. Agent loop ---
    agent_loop = AgentLoop(
        config=agent_config,
        llm=llm,
        bus=bus,
        context_manager=ctx_mgr,
        tool_registry=tool_registry,
        permission_evaluator=perm_eval,
        memory_loader=memory_store,
        compact_md_path=compact_md_path,
        session_id=sitting_id,  # SESS-01: the whole sitting shares this id
    )
    if channel_mode == "discord":
        from localharness.channels.discord import DiscordChannel, discord_config_from_env
        channel = DiscordChannel(bus=bus, config=discord_config_from_env())
        console.print("[dim]Dispatch mode: Discord — listening for allowlisted messages.[/dim]")
    else:
        channel = TerminalChannel(bus=bus, config={})

    # --- Determine returning user ---
    is_returning = events_path.exists() and events_path.stat().st_size > 0

    # --- Startup banner ---
    elapsed = _time.monotonic() - start_time
    from localharness.cli.ui import startup_banner
    console.print(startup_banner(model=resolved_model, is_returning=is_returning))

    # --- Startup summary line ---
    parts = [f"({elapsed:.1f}s startup)"]
    counts: list[str] = ["1 agent"]
    if mcp_connected > 0 or mcp_failed > 0:
        mcp_str = f"{mcp_connected} MCP server{'s' if mcp_connected != 1 else ''}"
        if mcp_failed > 0:
            mcp_str += f" ({mcp_failed} failed)"
        counts.append(mcp_str)
    if plugins_loaded > 0:
        counts.append(f"{plugins_loaded} plugin{'s' if plugins_loaded != 1 else ''}")
    summary_line = " -- ".join(parts + [", ".join(counts)])
    if warnings:
        summary_line += f" [{'; '.join(warnings)}]"
    console.print(summary_line)

    # --- Verbose output ---
    if verbose:
        if mcp_manager and mcp_manager.connected_servers:
            for srv in mcp_manager.connected_servers:
                console.print(f"  MCP: {srv}")
        if plugins_loaded > 0 and hook_system:
            for pname in hook_system.loaded_plugin_names:
                console.print(f"  Plugin: {pname}")
        tool_count = len(tool_registry._tools["global"]) + len(tool_registry._tools["mcp"])
        console.print(f"  Tools: {tool_count} total")
        if memory_store:
            console.print(f"  Memory: {agent_dir / 'memory.db'} (WAL)")
        else:
            console.print("  Memory: in-memory (no persistence)")

    # --- Run REPL ---
    from localharness.cli.repl import OrchestratorREPL

    repl = OrchestratorREPL(
        orchestrator=orchestrator,
        agent_loop=agent_loop,
        channel=channel,
        bus=bus,
        config_dir=cfg_path,
        harness_config=harness,
    )

    _exit_reason = "complete"
    try:
        await repl.run()
    except KeyboardInterrupt:
        _exit_reason = "interrupt"
        console.print("\nGoodbye.")
    except Exception:
        _exit_reason = "error"
        raise  # finally still records the session; behavior for callers unchanged
    finally:
        # --- Ordered shutdown: MCP -> Consolidation -> WriteGate -> PredictiveGate/UserSignals/PredictiveWriteGate -> end_session -> MemoryStore ---
        # (EventBus handles its own file closing on GC/process exit)
        if mcp_manager:
            try:
                await mcp_manager.shutdown()
            except Exception:
                pass
        if consolidation_scheduler:
            try:
                await consolidation_scheduler.stop()
            except Exception:
                pass
        if write_gate:
            try:
                await write_gate.close()
            except Exception:
                pass
        # PredictiveGate / UserSignalDetector (Phase 34): additive bus subscribers that call
        # store methods on fire — close them AFTER write_gate, while the store is still open,
        # and BEFORE the close-out summary reads (same discipline as session_acc below).
        if predictive_gate:
            try:
                await predictive_gate.close()
            except Exception:
                pass
        if user_signal_detector:
            try:
                await user_signal_detector.close()
            except Exception:
                pass
        # PredictiveWriteGate (Phase 35): writes facts via store_fact on fire, so close it
        # AFTER user_signal_detector (no racing capture) while the store is still OPEN and
        # BEFORE end_session reads the close-out summary (research Pitfall 4 — same discipline
        # as write_gate/predictive_gate above).
        if predictive_write_gate:
            try:
                await predictive_write_gate.close()
            except Exception:
                pass
        # end_session needs the store OPEN (it writes) but the gate CLOSED (no racing
        # capture mid-summary) and consolidation STOPPED (no in-flight promotion mutating
        # facts mid-read) — hence here, after write_gate.close(), before the store closes
        # (research Pitfall 4).
        if session_acc is not None:
            try:
                await session_acc.close()  # stop counting before the summary reads
            except Exception:
                pass
        if memory_store is not None and _session_started:
            try:
                from localharness.cli.session_accumulator import derive_session_summary
                await memory_store.end_session(
                    sitting_id,
                    exit_reason=_exit_reason,
                    summary=derive_session_summary(session_acc),
                    turn_count=session_acc.turn_count if session_acc else 0,
                    action_count=session_acc.action_count if session_acc else 0,
                    tokens_in=session_acc.tokens_in if session_acc else 0,
                    tokens_out=session_acc.tokens_out if session_acc else 0,
                )
            except Exception as exc:
                # Never silent (2026-07-03 live-test rule): a skipped close-out is the
                # amnesia class. Match the surrounding swallow but leave a one-line trace.
                err_console.print(f"[yellow]⚠ session close-out skipped: {exc}[/yellow]")
        if memory_store:
            try:
                await memory_store.close()
            except Exception:
                pass


def start_app(
    agent: Annotated[str | None, typer.Option("--agent", "-a", help="Start specific agent")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show per-component startup detail")] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug logging")] = False,
    config_dir: Annotated[str, typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")] = "~/.localharness",
    channel: Annotated[str, typer.Option("--channel", "-c", help="Input channel: terminal (default) or discord")] = "terminal",
    subagents: Annotated[bool, typer.Option("--subagents", help="Show the agent picker on startup when multiple agents are configured")] = False,
) -> None:
    """Launch the agent REPL. Zero to chatting in one command."""
    try:
        asyncio.run(_start_async(agent, verbose, debug, config_dir, channel, subagents))
    except KeyboardInterrupt:
        console.print("\nGoodbye.")
