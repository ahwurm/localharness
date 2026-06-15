# Spec 00: Architecture Overview

**Project:** LocalHarness  
**Version:** v1  
**Status:** Authoritative — implement against this document  
**Last updated:** 2026-05-23

---

## 1. Purpose

This document is the definitive architectural reference for LocalHarness. It describes the system's dependency layers, component responsibilities, communication rules, data flows, project structure, technology stack, design principles, and build order.

Every other spec document is a deep dive into a specific component. This document is the map; other specs are the territory.

---

## 2. System Overview

LocalHarness is a model-agnostic hierarchical agent harness for local LLMs. It provides:

- A typed event bus that connects all components through a single ordered stream
- A ReAct while-loop agent runtime with tool execution, context management, and memory
- A thin orchestrator that routes tasks and manages agent creation conversationally
- A YAML configuration system that lets users define agents without writing code
- A CLI entry point with auto-detection of local LLM backends

The harness is the product. The LLM is interchangeable.

---

## 3. Five Dependency Layers

Components are organized into five layers. A component in layer N depends only on components in layers 1 through N-1. No upward dependencies.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: User Interface                                        │
│                                                                 │
│  cli/app.py      cli/init.py     cli/start.py    cli/agent.py  │
│  channels/terminal.py                                           │
├─────────────────────────────────────────────────────────────────┤
│  Layer 4: Orchestration                                         │
│                                                                 │
│  orchestrator/router.py          orchestrator/workflow.py       │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Agent Runtime                                         │
│                                                                 │
│  agent/loop.py   agent/context.py   agent/permissions.py       │
│  audit/logger.py                                                │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Infrastructure                                        │
│                                                                 │
│  tools/base.py       tools/registry.py    tools/hooks.py       │
│  tools/mcp.py        tools/builtin/       provider/client.py   │
│  provider/fn_call.py provider/detector.py                      │
│  memory/sqlite.py    memory/history.py    memory/markdown.py   │
│  config/loader.py    config/defaults.py                        │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Foundation                                            │
│                                                                 │
│  core/events.py    core/bus.py    core/types.py                │
│  config/models.py                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Layer 1: Foundation

The event type definitions, the event bus itself, shared primitive types, and the Pydantic config models. Nothing else in the system can exist without these. They have zero imports from other localharness modules.

**Files:** `core/events.py`, `core/bus.py`, `core/types.py`, `config/models.py`

### Layer 2: Infrastructure

The machinery that the agent runtime depends on: tool interface and registry, LLM provider abstraction, persistence (SQLite + JSONL + markdown), config loading, plugin discovery, MCP client. These modules import from Layer 1 but not from each other unless the dependency is explicitly documented.

**Files:** `tools/base.py`, `tools/registry.py`, `tools/hooks.py`, `tools/mcp.py`, `tools/builtin/`, `provider/detector.py`, `provider/client.py`, `provider/fn_call.py`, `memory/sqlite.py`, `memory/history.py`, `memory/markdown.py`, `config/loader.py`, `config/defaults.py`

### Layer 3: Agent Runtime

The agent execution loop and its direct dependencies: context manager, permission evaluator, and audit logger. These modules orchestrate Layer 2 components through the event bus.

**Files:** `agent/loop.py`, `agent/context.py`, `agent/permissions.py`, `audit/logger.py`

### Layer 4: Orchestration

The thin orchestrator that routes tasks, runs the agent creation workflow, and synthesizes multi-agent results. Depends on the agent runtime (Layer 3) and all infrastructure (Layer 2).

**Files:** `orchestrator/router.py`, `orchestrator/workflow.py`

### Layer 5: User Interface

The CLI entry point and channel adapters. These are the only components that interact with the user directly. They consume the orchestrator (Layer 4) and publish events to the bus.

**Files:** `cli/app.py`, `cli/init.py`, `cli/start.py`, `cli/agent.py`, `channels/terminal.py`

---

## 4. Component Map

### Component Responsibilities

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `core/events.py` | 1 | All Pydantic event type definitions |
| `core/bus.py` | 1 | EventBus: publish, subscribe, replay, persist |
| `core/types.py` | 1 | Shared primitives: AgentID, SessionID, EventSeq |
| `config/models.py` | 1 | Pydantic config models: AgentConfig, DivisionConfig, etc. |
| `config/loader.py` | 2 | YAML parse + validate + inheritance resolution |
| `config/defaults.py` | 2 | Default values for all config fields |
| `provider/detector.py` | 2 | Port probe: Ollama/vLLM/llama.cpp/LM Studio |
| `provider/client.py` | 2 | Thin OpenAI-compat async HTTP client |
| `provider/fn_call.py` | 2 | XML tool call fallback converter |
| `tools/base.py` | 2 | Tool protocol, ToolSchema, ToolResult |
| `tools/registry.py` | 2 | Tool registry + scope resolution |
| `tools/hooks.py` | 2 | Pre/post hook system (pluggy) |
| `tools/mcp.py` | 2 | MCP discovery: stdio + streamable-HTTP |
| `tools/builtin/` | 2 | Built-in tools: glob, grep, read, write, bash |
| `memory/sqlite.py` | 2 | SQLite facts store (async WAL) |
| `memory/history.py` | 2 | JSONL chat history (append-only session log) |
| `memory/markdown.py` | 2 | MEMORY.md persistent notes |
| `agent/loop.py` | 3 | ReAct while-loop: reason → act → observe |
| `agent/context.py` | 3 | Context window tracking + compaction |
| `agent/permissions.py` | 3 | Deny-pattern evaluator + budget enforcer |
| `audit/logger.py` | 3 | Structured JSONL audit writer (structlog) |
| `orchestrator/router.py` | 4 | Agent Card routing + task dispatch |
| `orchestrator/workflow.py` | 4 | Discuss→configure→deploy agent creation flow |
| `cli/app.py` | 5 | Top-level Typer app with subcommand registration |
| `cli/init.py` | 5 | `localharness init` implementation |
| `cli/start.py` | 5 | `localharness start` implementation |
| `cli/agent.py` | 5 | `localharness agent` subcommands |
| `channels/terminal.py` | 5 | stdout channel adapter (Rich streaming) |

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
| Tool System | Agent loop (via event bus) | Event bus (ToolResult observation) |
| Hook System | Tool system (pre/post) | Tool system (gate pass/fail) |
| Permission Evaluator | Agent loop (before tool execute) | Agent loop (allow / deny decision) |
| Memory | Agent loop (read), Event bus (write events) | Agent loop (context injection) |
| Context Manager | Agent loop | Agent loop (pruned message list) |
| Orchestrator | Event bus (UserMessage, TaskComplete) | Event bus (TaskRequest delegation) |
| Audit Logger | Event bus (all events) | Disk (per-agent events.jsonl) |

---

## 5. Communication Rule

**All inter-component communication goes through the event bus. No component holds a direct reference to another.**

This is the single most important structural rule. It enables:
- Replay: reconstruct any session from its event log
- Testing: inject test events without starting real components
- Debugging: inspect the complete event sequence for any failure
- Future parallelism: move components to separate processes/threads without changing interfaces

**The only exception:** Config Loader is a synchronous dependency injected at construction time. It is not a subscriber. It loads config once per agent instantiation and passes it to the agent loop constructor.

### Enforcement

When writing a new component:
1. It receives state by subscribing to event types
2. It changes state by publishing events
3. It does not import other component modules
4. It does not call other components' methods directly
5. It receives the event bus via constructor injection (`__init__(self, bus: EventBus)`)

---

## 6. Data Flow Diagrams

### 6.1 Startup Flow

```
User runs: localharness init
      │
      ▼
CLI (cli/init.py)
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
CLI (cli/start.py)
      │
      ├── Constructs EventBus (core/bus.py)
      ├── Constructs AuditLogger (audit/logger.py) → subscribes to bus
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
      ├── 2. Memory loads MEMORY.md + SQLite facts into system context
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
                  ├── Tool System: execute tool via tool.run(**params)
                  │       └── Publish: Action(type="tool_call", tool=name, params=...)
                  │
                  ├── Hook System: run post_tool hooks
                  │
                  ├── Publish: Observation(type="tool_result", output=..., tool_call_id=...)
                  │
                  └── Audit Logger: writes hash of (prev_hash + event JSON) to events.jsonl

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

Every file in the `src/localharness/` package. This is the complete layout — no file exists outside this structure without explicit rationale.

```
localharness/
├── pyproject.toml                    # uv workspace root; entry point: localharness = localharness.cli.app:app
├── uv.lock                           # single lockfile for entire project
├── maturin.toml                      # PyO3 build (empty config until Rust layer added)
├── Cargo.toml                        # Rust workspace root (empty members until Rust layer added)
├── LICENSE                           # MIT
├── README.md
│
├── src/
│   └── localharness/
│       ├── __init__.py               # package version, __all__
│       │
│       ├── core/                     # Layer 1: Foundation
│       │   ├── __init__.py
│       │   ├── events.py             # All Pydantic event models (UserMessage, Action, etc.)
│       │   ├── bus.py                # EventBus: publish/subscribe/replay, bubus integration
│       │   └── types.py              # AgentID, SessionID, EventSeq, ToolCallID type aliases
│       │
│       ├── config/                   # Layer 1 (models) + Layer 2 (loader)
│       │   ├── __init__.py
│       │   ├── models.py             # AgentConfig, DivisionConfig, OrgConfig, ToolConfig,
│       │   │                         #   PermissionConfig, MemoryConfig, ScheduleConfig,
│       │   │                         #   ContextConfig, ProviderConfig (all Pydantic)
│       │   ├── loader.py             # ConfigLoader: YAML parse, validate, inheritance resolve
│       │   └── defaults.py           # DEFAULT_ORG_CONFIG, DEFAULT_DIVISION_CONFIG, etc.
│       │
│       ├── provider/                 # Layer 2: LLM provider abstraction
│       │   ├── __init__.py
│       │   ├── detector.py           # AutoDetector: port probe → ProviderConfig
│       │   ├── client.py             # LLMClient: thin openai.AsyncOpenAI wrapper
│       │   └── fn_call.py            # FnCallConverter: XML parse + inject tool schema in prompt
│       │
│       ├── tools/                    # Layer 2: Tool system
│       │   ├── __init__.py
│       │   ├── base.py               # Tool protocol, ToolSchema, ToolResult, ToolCall
│       │   ├── registry.py           # ToolRegistry: scope resolution (global→division→agent→MCP)
│       │   ├── hooks.py              # HookSystem: pluggy hookspec + hookimpl for pre/post tool
│       │   ├── mcp.py                # MCPClient: stdio + streamable-HTTP discovery + wrapping
│       │   └── builtin/
│       │       ├── __init__.py       # registers all builtins
│       │       ├── glob_tool.py      # GlobTool
│       │       ├── grep_tool.py      # GrepTool
│       │       ├── read_tool.py      # ReadTool
│       │       ├── write_tool.py     # WriteTool
│       │       └── bash_tool.py      # BashTool
│       │
│       ├── memory/                   # Layer 2: Persistence
│       │   ├── __init__.py
│       │   ├── sqlite.py             # FactsStore: aiosqlite WAL, schema migrations
│       │   ├── history.py            # ChatHistory: JSONL append-only session log
│       │   └── markdown.py           # MarkdownNotes: MEMORY.md read/append
│       │
│       ├── agent/                    # Layer 3: Agent runtime
│       │   ├── __init__.py
│       │   ├── loop.py               # AgentLoop: ReAct while-loop, session management
│       │   ├── context.py            # ContextManager: token tracking, compaction, boundary guard
│       │   └── permissions.py        # PermissionEvaluator: deny-pattern match, budget check
│       │
│       ├── audit/                    # Layer 3: Audit logging
│       │   ├── __init__.py
│       │   └── logger.py             # AuditLogger: structlog JSONL writer, event bus subscriber
│       │
│       ├── orchestrator/             # Layer 4: Orchestration
│       │   ├── __init__.py
│       │   ├── router.py             # Orchestrator: Agent Card routing, task dispatch
│       │   └── workflow.py           # AgentCreationWorkflow: discuss→configure→deploy
│       │
│       ├── channels/                 # Layer 5: Output adapters
│       │   ├── __init__.py
│       │   ├── base.py               # ChannelAdapter protocol (interface for future adapters)
│       │   └── terminal.py           # TerminalChannel: Rich streaming to stdout
│       │
│       └── cli/                      # Layer 5: CLI entry point
│           ├── __init__.py
│           ├── app.py                # Typer app + subcommand registration
│           ├── init.py               # `localharness init` command
│           ├── start.py              # `localharness start` command
│           ├── doctor.py             # `localharness doctor` command
│           ├── validate.py           # `localharness validate` command
│           └── agent.py              # `localharness agent create|list|delete` subcommands
│
├── tests/
│   ├── conftest.py                   # Shared fixtures: in-memory EventBus, mock LLMClient
│   ├── unit/
│   │   ├── test_events.py
│   │   ├── test_bus.py
│   │   ├── test_config_loader.py
│   │   ├── test_fn_call.py
│   │   ├── test_permissions.py
│   │   ├── test_context.py
│   │   └── test_tools.py
│   └── integration/
│       ├── test_agent_loop.py        # Full loop with mock LLM + real tools
│       └── test_orchestrator.py      # Orchestrator routes + agent creation
│
├── agents/                           # Bundled example YAML configs (no personal data)
│   ├── example-researcher.yaml
│   └── example-coder.yaml
│
└── rust/                             # Rust layer (empty until a bottleneck is measured)
    └── src/
        └── lib.rs                    # #[pymodule] stub
```

---

## 8. Technology Stack

| Library | Version | Purpose | Layer |
|---------|---------|---------|-------|
| Python | 3.12+ | Primary language | — |
| pydantic | 2.13.4 | Event schemas, config models | 1 |
| bubus | 1.5.6 | In-process async event bus | 1 |
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
uv add pydantic "pydantic[yaml]" pyyaml bubus aiosqlite structlog typer rich prompt-toolkit openai pluggy
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

### AP1: Direct Component References

**Violation:** `from localharness.tools.registry import ToolRegistry` inside `agent/loop.py`, then calling `registry.execute(call)` directly.

**Why wrong:** Prevents replay, tight coupling, untestable in isolation.

**Correct:** Agent loop publishes `Action` event → Tool System subscribes and executes → publishes `Observation` back → loop receives it.

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

## 11. Dependency Wave Build Order

A component cannot be usefully built before all components in prior waves are complete. This is the implementation order.

### Wave 1 — No dependencies

Build these first. They import only from the Python standard library and third-party packages (pydantic, bubus).

```
core/types.py          — AgentID, SessionID, EventSeq type aliases
core/events.py         — all Pydantic event models
core/bus.py            — EventBus (wraps bubus)
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
agent/loop.py          — imports: core/bus.py, core/events.py, tools/registry.py,
                          tools/hooks.py, agent/permissions.py, agent/context.py,
                          provider/client.py, memory/sqlite.py, memory/history.py,
                          memory/markdown.py, config/models.py
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
cli/init.py            — imports provider/detector.py, config/loader.py
cli/start.py           — imports orchestrator/router.py, channels/terminal.py, core/bus.py
cli/agent.py           — imports config/loader.py, orchestrator/workflow.py
cli/doctor.py          — imports provider/detector.py, config/loader.py
cli/validate.py        — imports config/loader.py
```

**Phase → Wave mapping:**
- Phase 1 (Setup & Config): Waves 1-2
- Phase 2 (Memory & Event Bus): Waves 1-2 (bus spec, memory modules)
- Phase 3 (Agent Loop & Tools): Waves 3-4
- Phase 4 (Orchestrator & CLI): Waves 5-6

---

## 12. Scalability Path

| Concern | v1 | v2 |
|---------|----|----|
| Event bus | In-process asyncio (bubus) | PyO3 Tokio broadcast (>10 agents) |
| Memory isolation | Per-agent SQLite | Same + FTS5 cross-agent index |
| Context windows | Per-agent `max_context_tokens` | Same + wave-based launch |
| Tool execution | Sequential in loop | Parallel tool execution within one turn |
| Audit logging | structlog JSONL | Rust PyO3 SHA-256 hash chain |
| LLM abstraction | Thin openai client | LiteLLM if multi-provider routing needed |
| Channel adapters | Terminal only | Discord + Slack via ChannelAdapter protocol |
| Permissions | Deny patterns (auto mode) | bubblewrap sandbox + guardian subagent |

---

## 13. Cross-References

| Topic | Spec Document |
|-------|---------------|
| Event type definitions, EventBus API | `01-event-bus.md` |
| Config YAML schema, config models | `06-config.md` |
| Tool interface, registry, MCP | *(planned: 02-tool-system.md)* |
| Agent loop, context manager | *(planned: 03-agent-loop.md)* |
| Permission evaluator | *(planned: 04-permissions.md)* |
| Orchestrator, agent creation workflow | *(planned: 05-orchestrator.md)* |
| CLI commands | *(planned: 07-cli.md)* |
| Memory: SQLite, JSONL, markdown | *(planned: 08-memory.md)* |
| Provider detection, LLM client | *(planned: 09-provider.md)* |
| Audit logger | *(planned: 10-audit.md)* |
