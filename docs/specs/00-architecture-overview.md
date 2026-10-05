# Spec 00: Architecture Overview

**Project:** LocalHarness  
**Version:** v1  
**Status:** The code is authoritative. This document states the architecture rules; its wiring section is checked against the code by `tests/unit/test_docs_convergence.py`.  
**Last updated:** 2026-10-02

---

## 1. Purpose

This document is the architectural map of LocalHarness. It describes the system's dependency layers, component responsibilities, communication rules, data flows, project structure, technology stack, design principles, and the order the first version was built in.

Every other spec document is a deep dive into a specific component. This document is the map; other specs are the territory. Where a map and the source disagree, the source wins and the map is the bug.

---

## 2. System Overview

LocalHarness is a model-agnostic hierarchical agent harness for local LLMs. It provides:

- A typed event bus that connects all components through a single ordered stream
- A ReAct while-loop agent runtime with tool execution and context management
- A thin orchestrator that routes tasks and manages agent creation conversationally
- A YAML configuration system that lets users define agents without writing code
- A CLI entry point with auto-detection of local LLM backends
- A plugin system: the bundled features (`image`, `mobile`, `memory`, `dispatch`, `autoresearch`) are plugins that can be switched on or off, and core never imports them

The harness is the product. The LLM is interchangeable.

---

## 3. Five Dependency Layers, Plus Plugins

Core components are organized into five layers. The intent is that a component in layer N depends only on layers 1 through N-1. That intent is a guide, not a test. The one dependency rule a test enforces is the line between core and plugins (section 5, Enforcement): core never imports a plugin module.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: User Interface                                        │
│                                                                 │
│  cli/app.py       cli/init_cmd.py   cli/start_cmd.py            │
│  cli/agent_cmd.py cli/repl.py       cli/*_cmd.py                │
│  channels/terminal.py               channels/acp.py             │
│  bench/                                                         │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Orchestration                                         │
│                                                                 │
│  orchestrator/router.py  orchestrator/workflow.py               │
│  orchestrator/cards.py                                          │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Agent Runtime                                         │
│                                                                 │
│  agent/loop.py   agent/context.py   agent/permissions.py        │
│  agent/gate.py   agent/verdict.py   agent/subagent.py           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Infrastructure                                        │
│                                                                 │
│  tools/base.py       tools/registry.py   tools/hooks.py         │
│  tools/mcp.py        tools/capabilities.py  tools/builtin/      │
│  provider/client.py  provider/fn_call.py provider/detector.py   │
│  provider/idle_llm.py                                           │
│  config/loader.py    config/defaults.py  config/paths.py        │
│  plugins/            registry/                                  │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Foundation                                            │
│                                                                 │
│  core/events.py    core/bus.py    core/types.py                 │
│  config/models.py                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Plugins (beside the layers; reach core only through ctx)       │
│                                                                 │
│  image         tools/builtin/image_plugin.py (+ generate_image) │
│  mobile        cli/mobile_plugin.py, channels/mobile/                 │
│  memory        memory/                                          │
│  dispatch      dispatch/ (Discord adapter in dispatch/adapters/)│
│  autoresearch  autoresearch/                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Foundation

The event type definitions, the event bus itself, shared primitive types, and the Pydantic config models. Nothing else in the system can exist without these.

**Files:** `core/events.py`, `core/bus.py`, `core/types.py`, `config/models.py`

### Layer 2: Infrastructure

The machinery that the agent runtime depends on: tool interface and registry, the capability floor, LLM provider abstraction, config loading, the plugin system, the component catalogue, MCP client.

**Files:** `tools/base.py`, `tools/registry.py`, `tools/hooks.py`, `tools/mcp.py`, `tools/capabilities.py`, `tools/builtin/`, `provider/detector.py`, `provider/client.py`, `provider/fn_call.py`, `provider/idle_llm.py`, `config/loader.py`, `config/defaults.py`, `config/paths.py`, `plugins/`, `registry/`

### Layer 3: Agent Runtime

The agent execution loop and its direct collaborators: context manager, permission evaluator, the permission gate, and subagent dispatch. The loop receives each collaborator as a constructor parameter (section 5).

**Files:** `agent/loop.py`, `agent/context.py`, `agent/permissions.py`, `agent/gate.py`, `agent/verdict.py`, `agent/subagent.py`

### Layer 4: Orchestration

The thin orchestrator that routes tasks, runs the agent creation workflow, and synthesizes multi-agent results.

**Files:** `orchestrator/router.py`, `orchestrator/workflow.py`, `orchestrator/cards.py`

### Layer 5: User Interface

The CLI entry point and the core channel adapters. These are the components that interact with the user directly. `cli/start_cmd.py` is also where a session is wired: it builds the bus, the registry, the loop and the plugins, and hands each its collaborators.

**Files:** `cli/app.py`, `cli/init_cmd.py`, `cli/start_cmd.py`, `cli/agent_cmd.py`, `cli/repl.py`, the other `cli/*_cmd.py` command modules, `channels/terminal.py`, `channels/acp.py`, `bench/`

### Plugins

A plugin is a feature that ships with LocalHarness but sits outside core: it can be switched on or off, and core does not import it. `BUILTIN_PLUGINS` in `plugins/builtin.py` is the only list of what ships, and that module is the only core module allowed to import a plugin. A plugin reaches the session only through the `PluginContext` it is given (section 5). The plugin API is spec 09.

**Files:** `tools/builtin/image_plugin.py` and `tools/builtin/generate_image_tool.py` (`image`); `cli/mobile_plugin.py` and `channels/mobile/` (`mobile`); `memory/` (`memory`); `dispatch/` (`dispatch`); `autoresearch/` (`autoresearch`)

---

## 4. Component Map

### Component Responsibilities

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `core/events.py` | 1 | All Pydantic event type definitions |
| `core/bus.py` | 1 | EventBus: publish, subscribe, replay, persist (a session's events go to `agents/<name>/bus-events.jsonl`) |
| `core/types.py` | 1 | Shared primitives: AgentID, SessionID, EventSeq |
| `config/models.py` | 1 | Pydantic config models: AgentConfig, DivisionConfig, etc. |
| `config/loader.py` | 2 | YAML parse + validate + inheritance resolution |
| `config/defaults.py` | 2 | Default values for all config fields |
| `provider/detector.py` | 2 | Port probe: Ollama/vLLM/llama.cpp/LM Studio |
| `provider/client.py` | 2 | Thin OpenAI-compat async HTTP client |
| `provider/fn_call.py` | 2 | XML tool call fallback converter |
| `provider/idle_llm.py` | 2 | The one path for background (idle-time) LLM work |
| `tools/base.py` | 2 | Tool protocol, ToolSchema, ToolResult |
| `tools/registry.py` | 2 | Tool registry + scope resolution; `dispatch` runs pre/post hooks around each tool |
| `tools/hooks.py` | 2 | Pre/post hook system (pluggy) |
| `tools/mcp.py` | 2 | MCP discovery: stdio + streamable-HTTP |
| `tools/capabilities.py` | 2 | Capability floor: no agent holds untrusted-ingest and host-dangerous tools together |
| `tools/builtin/` | 2 | Built-in tools: read, write, edit, glob, grep, bash, python, web, load_document, chunk, tool_result_get, agent (subagent), cruncher |
| `plugins/` | 2 | Plugin API (`api.py`), the bundled list (`builtin.py`), discovery, plan, lifecycle, the memory slot |
| `registry/` | 2 | Component catalogue behind `localharness components` |
| `agent/loop.py` | 3 | ReAct while-loop: reason → act → observe |
| `agent/context.py` | 3 | Context window tracking + compaction |
| `agent/permissions.py` | 3 | Deny-pattern evaluator + budget enforcer |
| `agent/gate.py` | 3 | Permission gate: ask a human, remember the answer |
| `agent/subagent.py` | 3 | Bounded child agents (explore, cruncher) on the same bus |
| `orchestrator/router.py` | 4 | Agent Card routing + task dispatch |
| `orchestrator/workflow.py` | 4 | Discuss→configure→deploy agent creation flow |
| `cli/app.py` | 5 | Top-level Typer app with subcommand registration |
| `cli/init_cmd.py` | 5 | `localharness init` implementation |
| `cli/start_cmd.py` | 5 | `localharness start` implementation and session wiring |
| `cli/agent_cmd.py` | 5 | `localharness agent` subcommands |
| `channels/terminal.py` | 5 | stdout channel adapter (Rich streaming) |
| `channels/acp.py` | 5 | Agent Client Protocol adapter (Zed) |
| `memory/` | plugin | Facts store, history, MEMORY.md notes, recall and consolidation (the `memory` plugin) |
| `channels/mobile/` | plugin | The phone app's server and event API (the `mobile` plugin) |
| `dispatch/` | plugin | Chat channels; Discord is the first adapter (the `dispatch` plugin) |
| `autoresearch/` | plugin | Experiment loop (the `autoresearch` plugin) |

### What Talks to What

| Component | Receives From | Sends To |
|-----------|---------------|----------|
| CLI | User stdin | Event bus (UserMessage, TaskRequest) |
| Terminal channel | Event bus (TaskComplete, Action) | User stdout |
| Auto-detector | CLI (init flow) | Config loader (writes provider config) |
| LLM Provider | Agent loop | LLM inference server (HTTP) |
| Event Bus | All components (publish) | All subscribers (async delivery) |
| Config Loader | CLI, Agent loop (at startup) | Agent loop constructor (injected) |
| Agent Loop | Event bus (TaskRequest) | Event bus (Action, Observation, TaskComplete) |
| Tool System | Agent loop (direct call on its injected `tool_registry`) | Agent loop (ToolResult; the loop publishes the Observation) |
| Hook System | Tool registry (pre/post, inside `dispatch`) | Tool registry (gate pass/fail) |
| Permission Evaluator | Agent loop (before tool execute) | Agent loop (allow / deny decision) |
| Memory plugin | Agent loop (through its injected `memory_slot`), Event bus | Agent loop (context contribution each turn) |
| Context Manager | Agent loop | Agent loop (pruned message list) |
| Orchestrator | Event bus (UserMessage, TaskComplete) | Event bus (TaskRequest delegation) |
| Event log | Event bus (all events) | Disk (`agents/<name>/bus-events.jsonl`; component changes also go to `audit.jsonl`, set by `org.audit_log_path`) |

---

## 5. Communication Rule

**Facts travel on the event bus, the event/data plane: no component holds a reference to another in order to tell it what happened.** Every fact a session produces (a user message, a model reply, a tool call and its result, a turn's end) is published as a typed event on the one ordered bus, so the log is the session (P1, P2).

This is the most important structural rule. It enables:
- Replay: reconstruct any session from its event log
- Testing: inject test events without starting real components
- Debugging: inspect the complete event sequence for any failure
- Future parallelism: move components to separate processes/threads without changing interfaces

The rule governs facts, not capabilities. A component that needs to *do* something through another component (call the model, run a tool, ask the permission gate) holds that collaborator as a constructor parameter. That is the second plane, service wiring, and the list below is all of it.

**Config Loader:** a synchronous dependency injected at construction time. It is not a subscriber. It loads config once per agent instantiation and passes it to the agent loop constructor (the `config` parameter below).

### Constructor wiring (the sanctioned exceptions)

A component that needs a capability receives it as a constructor parameter, supplied by the start-up wiring (`cli/start_cmd.py` for a session, `agent/subagent.py` for a child loop, `plugins/lifecycle.py` for a plugin's context). It never discovers one by importing a module or looking it up. These are the exceptions, by name, and a test holds this list equal to the code:

- `AgentLoop.__init__`: `config`, `llm`, `bus`, `context_manager`, `tool_registry`, `permission_evaluator`, `kill_file_path`, `compact_md_path`, `session_id`, `config_dir`, `gate`, `guardrails_path`, `memory_slot`, `task_context`
- `PluginContext`: `bus`, `tools`, `hooks`, `config`, `agent_config`, `paths`, `llm`, `idle_llm`, `session` (the last two are additive fields; the first seven are the original plugin API)

What the loop does with them: it publishes facts on `bus`, calls the model through `llm`, runs tools through `tool_registry.dispatch`, asks `gate` and `permission_evaluator` before a tool runs, and asks `memory_slot` for the recalled context it adds to the turn's system prompt. The results it gets back become events again, so the replay log stays complete.

#### Design notes

Two planes exist because they carry different things. Events carry facts, and a fact must be recorded in order so a session can be replayed. A collaborator is a capability, and a capability must be handed to a component by whoever builds it rather than found by the component itself. Martin Fowler's article gave this second idea its common name, dependency injection, with constructor injection as one of its three forms, and framed it as "separating configuration from use" ([Fowler, 2004](https://martinfowler.com/articles/injection.html)); in LocalHarness, `cli/start_cmd.py` is the configuration and `AgentLoop` is the use. Channels and the memory slot follow the ports-and-adapters idea from Alistair Cockburn ([Cockburn, 2005](https://alistair.cockburn.us/hexagonal-architecture/)): the core defines the port (a channel adapter, the memory slot's interface), and a technology-specific adapter (Discord, the phone app, the memory plugin) plugs into it, so the core stays ignorant of which one is there. The import-direction test is what keeps the wiring plane pointing one way: core never imports a plugin, so a plugin can only receive capabilities, never be reached into.

### Enforcement

When writing a new component:
1. It publishes the facts it produces as events, and reads other components' facts by subscribing
2. It receives every collaborator it calls as a constructor parameter, and nothing else is wired; adding one means adding it to the list above
3. It does not reach into another component's internals (private attributes, module globals)
4. A core module never imports a plugin module; a plugin reaches core only through `ctx` (its `PluginContext`). `plugins/builtin.py` is the one core module allowed to import plugins. `tests/unit/test_import_direction.py` scans every import (including lazy and type-checking imports) and fails on a new edge
5. Its wiring section here stays equal to the code: `tests/unit/test_docs_convergence.py` fails if a constructor parameter or `PluginContext` field is added or removed without this list changing

---

## 6. Data Flow Diagrams

### 6.1 Startup Flow

```
User runs: localharness init
      │
      ▼
CLI (cli/init_cmd.py)
      │
      ├── Calls Auto-detector (provider/detector.py)
      │       │
      │       ├── HTTP GET http://localhost:8000/v1/models  (vLLM)
      │       ├── HTTP GET http://localhost:11434/api/tags  (Ollama)
      │       ├── HTTP GET http://localhost:1234/v1/models  (LM Studio)
      │       └── HTTP GET http://localhost:8080/v1/models  (llama.cpp)
      │           │
      │           └── First 200 response → ProviderConfig(base_url, models)
      │
      ├── Config Loader writes ~/.localharness/config.yaml
      │
      └── CLI prints: "Detected: Ollama at localhost:11434 — model qwen3..."

User runs: localharness start
      │
      ▼
CLI (cli/start_cmd.py)
      │
      ├── Constructs EventBus (core/bus.py), persisting every event to bus-events.jsonl
            ├── Constructs TerminalChannel (channels/terminal.py) → subscribes to bus
      ├── Constructs Orchestrator (orchestrator/router.py) → subscribes to bus
      │
      ├── Orchestrator publishes: SystemReady(timestamp=...)
      │       │
      │       └── TerminalChannel receives SystemReady → prints greeting
      │
      └── CLI enters prompt_toolkit REPL loop
```

### 6.2 Agent Creation Flow

```
User types: "create an agent that monitors Hacker News"
      │
      ▼
Terminal channel reads input
      │
      └── Publishes: UserMessage(content="create an agent...", session_id=...)
            │
            ▼
Orchestrator (router.py) receives UserMessage
      │
      ├── Detects intent: agent creation
      ├── Enters workflow (workflow.py): discuss → configure → deploy
      │
      │   [Discuss phase — Orchestrator calls LLM]
      │   Orchestrator publishes: Action(type="llm_request", ...)
      │   LLM responds with clarifying questions
      │   Orchestrator publishes: Observation(type="llm_response", ...)
      │   TerminalChannel prints questions to user
      │
      │   [User answers questions — repeat until confirmed]
      │
      │   [Configure phase]
      │   Orchestrator generates AgentConfig from gathered requirements
      │   Orchestrator calls Config Loader: writes ~/.localharness/agents/hn-monitor.yaml
      │
      │   [Deploy phase]
      │   Orchestrator publishes: AgentCreated(agent_id="hn-monitor", config_path=...)
      │
      └── TerminalChannel receives AgentCreated → prints confirmation
```

### 6.3 Task Execution Flow

```
User types: "run hn-monitor"
      │
      ▼
Terminal channel publishes: TaskRequest(agent_id="hn-monitor", input="...", budget=...)
      │
      ▼
Orchestrator (router.py) receives TaskRequest
      │
      ├── Looks up Agent Card for "hn-monitor"
      ├── Confirms capability match
      └── Publishes: TaskRequest (routes to agent loop)
            │
            ▼
Agent Loop (agent/loop.py) receives TaskRequest
      │
      ├── 1. Config Loader hydrates AgentConfig from YAML
      ├── 2. Memory slot (when the memory plugin is on) adds recalled context
      ├── 3. Context Manager checks window headroom
      │
      └── WHILE LOOP:
            │
            ├── a. Check iteration / duration budget → raise BudgetExceeded if hit
            ├── b. Context Manager: apply boundary guard, truncate if >80%
            ├── c. LLM Provider streams chat completion
            ├── d. Publish: Action(type="llm_response", content=response)
            ├── e. Extract tool calls (native or XML fallback)
            ├── f. If no tool calls → BREAK
            │
            └── For each tool_call:
                  │
                  ├── Permission Evaluator: check against deny patterns
                  │       └── Deny → publish Observation(type="permission_denied")
                  │
                  ├── Hook System: run pre_tool hooks (pluggy)
                  │
                  ├── Tool registry (injected): dispatch → tool.run(**params)
                  │       └── Publish: Action(type="tool_call", tool=name, params=...)
                  │
                  ├── Hook System: run post_tool hooks
                  │
                  ├── Publish: Observation(type="tool_result", output=..., tool_call_id=...)
                  │
                  └── Event bus persists each event to the session's JSONL log

      │
      ├── Context Manager: stuck detection (action signature hash sliding window)
      │
      └── On BREAK or limit hit:
            │
            ├── Publish: TaskComplete(agent_id=..., summary="...", success=True)
            │
            └── Orchestrator receives TaskComplete
                  └── TerminalChannel prints result
```

---

## 7. Project Structure

The package by directory, with the files the layers above name. It is a map, not a manifest: the source tree is the authority, and small helper modules are left out.

```
localharness/
├── pyproject.toml                    # entry point: localharness = localharness.cli.app:app
├── uv.lock
├── README.md, SECURITY.md, CHANGELOG.md, LICENSE
├── docs/specs/                       # these specs
├── examples/
│   ├── agents/                       # example agent YAML configs
│   └── plugin-template/              # a starting point for a third-party plugin
│
├── src/
│   └── localharness/
│       ├── core/                     # Layer 1: events.py, bus.py, types.py,
│       │                             #   artifacts.py, agent_dir.py, reduce.py
│       ├── config/                   # Layer 1 (models.py) + Layer 2: loader.py, defaults.py,
│       │                             #   paths.py, plugin_sections.py, overlay.py, trust.py, ...
│       ├── provider/                 # Layer 2: client.py, fn_call.py, detector.py, idle_llm.py,
│       │                             #   lifecycle.py, server.py, refarch.py, speed_stats.py
│       ├── tools/                    # Layer 2: base.py, registry.py, hooks.py, mcp.py,
│       │   │                         #   capabilities.py
│       │   └── builtin/              # core built-in tools (read, write, edit, glob, grep, bash,
│       │                             #   python, web, ...) plus the image plugin's files
│       ├── plugins/                  # Layer 2: api.py (the plugin API), builtin.py
│       │                             #   (BUILTIN_PLUGINS), discovery.py, plan.py, resolve.py,
│       │                             #   lifecycle.py, slot.py, channels.py, trust.py
│       ├── registry/                 # Layer 2: component catalogue
│       ├── agent/                    # Layer 3: loop.py, context.py, permissions.py, gate.py,
│       │                             #   verdict.py, subagent.py, ...
│       ├── orchestrator/             # Layer 4: router.py, workflow.py, cards.py
│       ├── channels/                 # Layer 5: base.py (ChannelAdapter), terminal.py, acp.py
│       │   └── mobile/               # plugin: the phone app (mobile)
│       ├── cli/                      # Layer 5: app.py, init_cmd.py, start_cmd.py, agent_cmd.py,
│       │                             #   repl.py and the other *_cmd.py modules; mobile_plugin.py
│       │                             #   and the plugin command modules belong to their plugins
│       ├── bench/                    # Layer 5: the benchmark runner behind `localharness bench`
│       ├── memory/                   # plugin: memory
│       ├── dispatch/                 # plugin: dispatch (adapters/discord.py)
│       └── autoresearch/             # plugin: autoresearch
│
└── tests/
    ├── unit/
    └── integration/
```

---

## 8. Technology Stack

| Library | Version | Purpose | Layer |
|---------|---------|---------|-------|
| Python | 3.12+ | Primary language | — |
| pydantic | 2.13.4 | Event schemas, config models | 1 |
| PyYAML | 6.0.3 | YAML parsing | 2 |
| pydantic-yaml | 1.4.0 | YAML ↔ Pydantic round-trip | 2 |
| openai | 1.x | OpenAI-compat HTTP client | 2 |
| aiosqlite | 0.22.1 | Async SQLite facts store | 2 |
| pluggy | 1.6.0 | Pre/post hook system | 2 |
| structlog | 25.5.0 | Structured JSONL audit logging | 3 |
| Typer | 0.25.1 | CLI subcommand framework | 5 |
| Rich | 15.0.0 | Terminal formatting + streaming | 5 |
| prompt_toolkit | 3.0.52 | Interactive REPL input | 5 |
| uv | 0.11.16 | Package manager + workspace | build |
| maturin | 1.13.3 | PyO3 Rust wheel builder | build (v2) |
| PyO3 | 0.28.3 | Rust ↔ Python FFI | build (v2) |
| pyo3-async-runtimes | 0.28 | asyncio ↔ Tokio bridge | build (v2) |
| tokio | 1.x | Rust async runtime | build (v2) |
| pytest | latest | Test runner | dev |
| pytest-asyncio | latest | Async test support | dev |
| ruff | latest | Linter + formatter | dev |
| mypy | latest | Static type checking | dev |

**Dependency install:**
```bash
uv add pydantic "pydantic[yaml]" pyyaml aiosqlite structlog typer rich prompt-toolkit openai pluggy
uv add --dev pytest pytest-asyncio ruff mypy maturin
```

---

## 9. Design Principles

### P1: Event-Sourced State

Events are the source of truth. An agent's complete state is reconstructible from its event log. The JSONL file is not just audit — it IS the session. The Context Manager reads it to rebuild message history on restart. Crash recovery = replay from JSONL.

Source: OpenHands V1 (Nov 2025) replaced their original pub/sub after it caused "thread/async issues and few guarantees on message order." LocalHarness adopts the V1 model from day one.

### P2: Ordered Append Log, Not Pub/Sub

Events have sequence numbers. Subscribers process events in order. Replay is deterministic. Unordered multi-subscriber pub/sub is explicitly rejected.

### P3: Lean Orchestrator / Fat Subagent

The orchestrator stays at ≤15% context utilization. It passes file paths to agents, never file contents. Each agent starts with a fresh context window. This is the single most important pattern for multi-agent scalability.

### P4: ReAct While-Loop, Not a Graph

The agent loop is a plain Python while-loop. LangGraph-style state machines and directed graphs are explicitly rejected. They add indirection without benefit for single-agent execution.

### P5: Minimal Tool Interface

Every tool implements exactly two methods: `info() → ToolSchema` and `run(**kwargs) → ToolResult`. MCP-discovered tools are wrapped in this interface. No other tool API exists.

### P6: Deny-First Permissions (v2), Auto Mode (v1)

v1: All tool calls are allowed unless the agent's deny_patterns list matches. v2 adds bubblewrap sandboxing and guardian review for flagged operations.

### P7: Config Over Code

Users define agents in YAML. The harness interprets config. No user-written Python required. Agent behavior changes via YAML edits, not code changes.

### P8: Start Pure Python, Port When Measured

Zero Rust in v1. Port to Rust only when a Python profiler identifies a specific bottleneck. Expected v2 Rust candidates: JSONL audit writer, event bus broadcast (>10 concurrent agents), tokenizer.

### P9: Model Agnostic

No model-specific assumptions in harness code. Auto-detection handles endpoint discovery. XML fallback handles models without native function calling. All LLM interaction goes through `LLMClient`, which wraps `openai.AsyncOpenAI(base_url=...)`.

### P10: Zero Cloud Dependencies

All core functionality runs on local hardware. No telemetry, no external API calls, no opt-in required.

---

## 10. Anti-Patterns

### AP1: Reaching Past the Wiring

**Violation:** a core module importing a plugin (for example `from localharness.memory.sqlite import ...` inside `agent/loop.py`), or any component reaching into another's internals (its private attributes or module globals) instead of using what it was handed.

**Why wrong:** The import makes the plugin impossible to switch off, because core now needs it to load. Reaching into internals couples two components to each other's implementation, so neither can change or be tested alone. Either way, the dependency appears nowhere in the wiring list, so nobody reviewing the constructor can see it.

**Correct:** Receive the collaborator as a constructor parameter and call it there. The loop's `tool_registry` parameter is the sanctioned form: `agent/loop.py` runs every tool through `self._tools.dispatch(...)` on the registry it was given, then publishes the result as an `Observation` event. Memory reaches the loop the same way, through the `memory_slot` parameter, never by import.

### AP2: Graph-Based Agent Execution

**Violation:** Using LangGraph, state machines, or directed acyclic graphs to drive agent execution steps.

**Why wrong:** Adds indirection without benefit. The agent loop is a plain loop, not a graph.

**Correct:** Plain Python `while True` loop with explicit `break` conditions.

### AP3: Fat Orchestrator

**Violation:** Orchestrator reads file contents, loads all agent contexts, or does domain-specific reasoning.

**Why wrong:** Context explosion on the first multi-agent task.

**Correct:** Orchestrator reads Agent Cards (compact JSON) and task summaries only. Passes `task_file` path to agents, never task contents.

### AP4: Tight LLM Coupling

**Violation:** Calling `ollama.chat(...)` directly, or assuming `tool_calls` is present in the response.

**Why wrong:** Breaks model-agnostic requirement.

**Correct:** All LLM calls go through `LLMClient`. `fn_call_converter` handles function-calling capability differences transparently.

### AP5: Premature Rust

**Violation:** Writing the event bus or agent loop in Rust from the start.

**Why wrong:** Slows development. PyO3 crossing costs are non-trivial for high-frequency small calls.

**Correct:** Pure Python for v1. Profile under real load. Port measured bottlenecks only.

### AP6: Unordered Pub/Sub

**Violation:** Firing events to multiple subscribers in any order, with no sequence guarantee.

**Why wrong:** OpenHands V1 explicitly replaced this pattern in November 2025.

**Correct:** Ordered append log. Events have sequence numbers. Deterministic replay.

### AP7: Conversation-as-Memory

**Violation:** Treating the LLM's conversation history as the agent's memory.

**Why wrong:** Context window is bounded. Cross-session continuity is impossible. This is the most common harness failure mode.

**Correct:** Memory is always explicit external state: SQLite facts, MEMORY.md, JSONL history. Context injection on each turn is explicit and bounded.

---

## 11. Dependency Wave Build Order (historical)

Historical: the initial build order, the order the first version was built in. It is kept as a record, not as the current module graph. Several files named in the early waves were later renamed, moved into a plugin (the `memory/` files), or never became separate modules (`audit/logger.py`: the bus persists its own events instead; see spec 12). The wave-4 entry for `agent/loop.py` is corrected below because it was false, not merely old.

### Wave 1 — No dependencies

Build these first. They import only from the Python standard library and third-party packages (pydantic; the bus also anyio and structlog).

```
core/types.py          — AgentID, SessionID, EventSeq type aliases
core/events.py         — all Pydantic event models
core/bus.py            — EventBus (the project's own pub/sub + JSONL persistence)
config/models.py       — AgentConfig and all sub-models (Pydantic)
config/defaults.py     — default values dict
```

### Wave 2 — Depends on Wave 1

```
config/loader.py       — imports config/models.py
provider/detector.py   — imports config/models.py (writes ProviderConfig)
provider/client.py     — imports core/types.py
provider/fn_call.py    — imports core/types.py, tools/base.py (Wave 3 dep — do fn_call.py last in wave 3)
memory/sqlite.py       — standalone (aiosqlite only)
memory/history.py      — imports core/events.py (JSONL = event replay log)
memory/markdown.py     — standalone (stdlib pathlib)
audit/logger.py        — imports core/bus.py, core/events.py
tools/base.py          — imports core/types.py
```

### Wave 3 — Depends on Wave 2

```
tools/registry.py      — imports tools/base.py
tools/hooks.py         — imports tools/registry.py (pluggy hookspec)
tools/builtin/         — imports tools/base.py + tools/registry.py
tools/mcp.py           — imports tools/registry.py + provider/client.py
provider/fn_call.py    — imports tools/base.py (ToolSchema)
agent/permissions.py   — imports config/models.py, core/types.py
agent/context.py       — imports provider/client.py (tokenizer), core/types.py
```

### Wave 4 — Depends on Wave 3

```
agent/loop.py          — imports: core/types.py, core/events.py, agent/gate.py,
                          agent/gate_types.py, agent/context.py, provider/client.py,
                          provider/fn_call.py, config/paths.py, plugins/api.py
                          (ContextBudget), tools/capabilities.py (CoResidenceError).
                          Nothing from memory/. The bus, the tool registry, the
                          permission evaluator and the memory slot arrive as
                          constructor parameters (section 5), not imports.
channels/terminal.py   — imports core/bus.py, core/events.py
```

### Wave 5 — Depends on Wave 4

```
orchestrator/router.py    — imports core/bus.py, core/events.py, config/loader.py,
                             agent/loop.py (or invokes via event bus)
orchestrator/workflow.py  — imports orchestrator/router.py, provider/client.py
```

### Wave 6 — Depends on Wave 5

```
cli/app.py             — top-level Typer app
cli/init_cmd.py        — imports provider/detector.py, config/loader.py
cli/start_cmd.py       — imports orchestrator/router.py, channels/terminal.py, core/bus.py
cli/agent_cmd.py       — imports cli/workspace.py, config/loader.py (deferred)
cli/doctor_cmd.py      — imports config/loader.py, cli/workspace.py; reaches provider
                          identification through cli/init_cmd.py, not provider/ directly
cli/validate_cmd.py    — imports config/loader.py
```

---

## 12. Scalability Path

| Concern | v1 | v2 |
|---------|----|----|
| Event bus | In-process asyncio (the project's own, `core/bus.py`) | PyO3 Tokio broadcast (>10 agents) |
| Memory isolation | Per-agent SQLite | Same + FTS5 cross-agent index |
| Context windows | Per-agent `max_context_tokens` | Same + wave-based launch |
| Tool execution | Sequential in loop | Parallel tool execution within one turn |
| Audit logging | structlog JSONL | Rust PyO3 SHA-256 hash chain |
| LLM abstraction | Thin openai client | LiteLLM if multi-provider routing needed |
| Channel adapters | Terminal and ACP in core; the phone (`mobile` plugin) and Discord (`dispatch` plugin, the first adapter of its `DispatchChannel`) as bundled plugins | Slack and other chat platforms as further `dispatch` adapters |
| Permissions | Deny patterns (auto mode) | bubblewrap sandbox + guardian subagent |

---

## 13. Cross-References

| Topic | Spec Document |
|-------|---------------|
| Event type definitions, EventBus API | [01-event-bus.md](01-event-bus.md) |
| Provider detection, LLM client | [02-provider.md](02-provider.md) |
| Agent loop, permission evaluator | [03-agent-loop.md](03-agent-loop.md); the trust model is [SECURITY.md](../../SECURITY.md) |
| Tool interface, registry | [04-tool-system.md](04-tool-system.md) |
| MCP | [04b-mcp-integration.md](04b-mcp-integration.md) |
| Memory | [05-memory.md](05-memory.md) |
| Config YAML schema, config models | [06-config.md](06-config.md) |
| Orchestrator, agent creation workflow | [07-orchestrator.md](07-orchestrator.md) |
| Context manager | [08-context-management.md](08-context-management.md) |
| Hooks and the plugin API | [09-hooks-plugins.md](09-hooks-plugins.md) |
| CLI commands | [10-cli.md](10-cli.md) |
| Channels | [11-channels.md](11-channels.md) |
| Audit and observability | [12-audit.md](12-audit.md) |
| Provider support and lifecycle | [13-provider-support.md](13-provider-support.md) |
