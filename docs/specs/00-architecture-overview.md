# Spec 00: Architecture Overview

**Project:** LocalHarness  
**Status:** The code is authoritative. This document states the architecture rules and the shape of the system; where it and the source disagree, the source wins and this document is the bug. Two of its claims are held equal to the code by tests: the constructor-wiring list in section 5 (`tests/unit/test_docs_convergence.py`) and the line between core and plugins (`tests/unit/test_import_direction.py`).  
**Last updated:** 2026-10-16, against release 0.18.1 with the 0.19.0 work on its branch.

---

## 1. Purpose

This document is the map of LocalHarness: its dependency layers, what each component owns, how components talk, how a session starts and how a turn runs, the project layout, the technology it stands on, the principles it follows and the boundaries tests enforce.

Every other spec is a deep dive into one component. This one is kept at the level that changes slowly. The file-by-file map changes with every release, so it is not written here: `python scripts/gen_index.py` generates it from the tree (module, length, purpose from the docstring, package by layer, CLI surface, bundled plugins) whenever it is needed.

---

## 2. System Overview

LocalHarness is a model-agnostic hierarchical agent harness for local LLMs. It provides:

- A typed event bus that carries every fact a session produces on one ordered, persisted stream
- A ReAct while-loop agent runtime with tool execution, a permission gate, budgets, stuck detection and automatic context management
- A provider layer that speaks to vLLM, Ollama, llama.cpp and LM Studio over the OpenAI-compatible wire, and that can also launch, size, swap and stop a local model server itself, including through a systemd user unit at login
- A YAML configuration system with a trust model: the machine-level config, a trusted workspace layer, per-agent files and a written overrides file, with a fixed list of settings only the machine level may set
- A CLI (`localharness init`, `start`, `model`, `doctor`, ...) with an interactive terminal REPL, an Agent Client Protocol adapter for Zed, and bounded helper agents the model can delegate to
- A plugin system: the bundled features (`image`, `mobile`, `memory`, `dispatch`, `autoresearch`, `browser`) are plugins behind one API that a third-party plugin uses too, and core never imports them
- A benchmark runner with a training corpus and a sealed holdout

The harness is the product. The LLM is interchangeable.

---

## 3. Five Dependency Layers, Plus Plugins

Core components are organized into five layers. The intent is that a component in layer N depends only on layers 1 through N-1. That intent is a guide for where new code goes; the dependency rules a test enforces are listed in section 11, and the strongest of them is the line between core and plugins: core never imports a plugin module.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 5: User Interface                                                │
│                                                                         │
│  cli/app.py        cli/start_cmd.py      cli/repl.py     cli/*_cmd.py   │
│  cli/model_ops.py  cli/start_serving.py  cli/autostart.py cli/serve_cmd │
│  cli/workspace.py  cli/session_trust.py  cli/plugin_mount.py            │
│  channels/base.py  channels/terminal.py  channels/acp.py                │
│  channels/input_router.py  channels/keys.py        bench/               │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 4: Orchestration                                                 │
│                                                                         │
│  orchestrator/router.py  orchestrator/workflow.py  orchestrator/cards.py│
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: Agent Runtime                                                 │
│                                                                         │
│  agent/loop.py     agent/context.py   agent/permissions.py              │
│  agent/verdict.py  agent/gate.py      agent/gate_types.py               │
│  agent/subagent.py agent/task_record.py  agent/task_context.py          │
│  agent/shell_classify.py  agent/gguf_header.py  agent/gguf_tokenizer.py │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: Infrastructure                                                │
│                                                                         │
│  tools/base.py  tools/registry.py  tools/hooks.py  tools/mcp.py         │
│  tools/capabilities.py             tools/builtin/                       │
│  provider/client.py  provider/fn_call.py  provider/detector.py          │
│  provider/lifecycle.py  provider/server.py  provider/units.py           │
│  provider/procfs.py  provider/autosense.py  provider/tuning.py          │
│  provider/idle_llm.py  provider/speed_stats.py  provider/refarch.py     │
│  config/loader.py  config/paths.py  config/trust.py  config/grants.py   │
│  config/project_records.py  config/overlay.py  config/migrate.py ...    │
│  plugins/           registry/                                           │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 1: Foundation                                                    │
│                                                                         │
│  core/events.py  core/bus.py  core/types.py  core/content.py            │
│  core/private_files.py  core/which.py  core/reduce.py  core/artifacts.py│
│  core/agent_dir.py                 config/models.py                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  Plugins (beside the layers; reach core only through their ctx)         │
│                                                                         │
│  image         tools/builtin/image_plugin.py, generate_image_tool.py,   │
│                tools/builtin/workflows/, cli/generate_image_cmd.py      │
│  mobile        cli/mobile_plugin.py, cli/mobile_cmd.py, channels/mobile/│
│  memory        memory/, tools/builtin/memory_tools.py,                  │
│                cli/memory_cmd.py, cli/memory_cli.py                     │
│  dispatch      dispatch/ (adapters/discord.py)                          │
│  autoresearch  autoresearch/, cli/autoresearch_cmd.py, experiment_cmd.py│
│                propose_cmd.py, report_cmd.py                            │
│  browser       tools/builtin/browser_plugin.py, browser_tool.py,        │
│                browser_proxy.py                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Foundation

The event types, the bus, shared primitives, message content parts, owner-only file creation, and the Pydantic config models. Nothing else can exist without these; nothing here imports a higher layer.

**Anchors:** `core/events.py` (the event models and the type map the bus deserializes by), `core/bus.py`, `core/types.py`, `core/content.py` (text and image parts), `core/private_files.py` (files and directories owner-only from the moment they exist), `config/models.py`.

### Layer 2: Infrastructure

The machinery the agent runtime stands on: the tool interface and registry with its hooks, the capability floor, the MCP client, the LLM client and function-call fallback, the model-server lifecycle (detection, launch strategies, the systemd side, `/proc` reads, window autosense and launch tuning), config loading, paths and trust records, the plugin API and lifecycle, and the component catalogue.

**Anchors:** `tools/registry.py`, `tools/capabilities.py`, `tools/mcp.py`, `provider/client.py`, `provider/lifecycle.py`, `provider/server.py`, `provider/units.py`, `provider/procfs.py`, `provider/autosense.py`, `config/loader.py`, `config/paths.py`, `config/trust.py`, `plugins/api.py`, `plugins/lifecycle.py`, `registry/catalogue.py`.

### Layer 3: Agent Runtime

The loop and its direct collaborators: the context manager and its compaction pipeline, the pure permission verdict and the effectful gate, bounded subagents, the persisted task record, the shell-command classifier the verdict uses, and the harness's own GGUF header reader and tokenizer. The loop receives each collaborator as a constructor parameter (section 5).

**Anchors:** `agent/loop.py`, `agent/context.py`, `agent/verdict.py`, `agent/gate.py`, `agent/subagent.py`, `agent/task_record.py`.

### Layer 4: Orchestration

The thin layer above a single agent: the agent-creation flow and its state machine, and the Agent Cards that let a coordinator hold many agents' capabilities in a small context. The REPL drives it.

**Anchors:** `orchestrator/router.py` (`AgentCreationFlow`), `orchestrator/workflow.py` (`AgentCreationWorkflow`), `orchestrator/cards.py` (`AgentCard`, `AgentCardRegistry`).

### Layer 5: User Interface

The CLI, the REPL, and the core channel adapters. `cli/start_cmd.py` is also where a session is wired: it decides trust, decides what model serves, builds the bus, the registry, the hooks, the plugins, the loop and the channel, and hands each its collaborators. `cli/model_ops.py` and `cli/start_serving.py` hold the one selection ladder and the attach / swap / launch decision; `cli/autostart.py` and `cli/serve_cmd.py` are the autostart switch and the unit's body; `cli/plugin_mount.py` mounts plugin commands lazily.

**Anchors:** `cli/app.py`, `cli/start_cmd.py`, `cli/repl.py`, `cli/model_cmd.py`, `cli/doctor_cmd.py`, `cli/init_cmd.py`, `channels/base.py` (`ChannelAdapter`), `channels/terminal.py`, `channels/acp.py`, `bench/runner.py`.

### Plugins

A plugin is a feature that ships with LocalHarness but sits outside core: it can be switched on or off, and core does not import it. `BUILTIN_PLUGINS` in `plugins/builtin.py` is the only list of what ships, and that module is the only core module allowed to import a plugin. A plugin reaches the session only through the `PluginContext` it is given (section 5). A plugin's files may live inside a core directory (the image plugin's tool, the memory plugin's commands); what makes them plugin files is the classification list in the import-direction test, not the folder. The plugin API is spec 09.

| Plugin | Kind | Files |
|---|---|---|
| `image` | tools, off by default | `tools/builtin/image_plugin.py`, `tools/builtin/generate_image_tool.py`, `tools/builtin/workflows/`, `cli/generate_image_cmd.py` |
| `mobile` | channel, needs the `mobile` extra | `cli/mobile_plugin.py`, `cli/mobile_cmd.py`, `channels/mobile/` |
| `memory` | memory (the one occupant of core's memory slot) | `memory/`, `tools/builtin/memory_tools.py`, `cli/memory_cmd.py`, `cli/memory_cli.py` |
| `dispatch` | channel | `dispatch/`, with `dispatch/adapters/discord.py` the first adapter |
| `autoresearch` | dev | `autoresearch/`, `cli/autoresearch_cmd.py`, `cli/experiment_cmd.py`, `cli/propose_cmd.py`, `cli/report_cmd.py` |
| `browser` | tools, off by default, needs the `browser` extra | `tools/builtin/browser_plugin.py`, `tools/builtin/browser_tool.py`, `tools/builtin/browser_proxy.py` |

---

## 4. Component Map

### Component Responsibilities

The anchor modules by layer. The generated index lists every module.

| Component | Layer | Responsibility |
|-----------|-------|----------------|
| `core/events.py` | 1 | The Pydantic event models, the budget spec, the event union and the type map `deserialize_event` reads |
| `core/bus.py` | 1 | `EventBus`: publish, subscribe, replay, wait_for, history; persists each event as one JSONL line (`agents/<name>/bus-events.jsonl`) |
| `core/types.py` | 1 | Shared primitives: AgentID, SessionID, EventSeq |
| `core/content.py` | 1 | Message content parts: text and images |
| `core/private_files.py` | 1 | Owner-only files and directories from the moment they exist; the no-follow writers the stores use |
| `config/models.py` | 1 | All Pydantic config models |
| `config/loader.py` | 2 | YAML parse, validate, inheritance, the layered merge, writes |
| `config/paths.py` | 2 | The one answer to "where is the config dir" and config-relative paths |
| `config/trust.py` | 2 | Which outside workspaces the user agreed to load config from; the machine's record of launch entries it has seen |
| `config/grants.py` | 2 | Durable "allow always" and "never here" gate answers, keyed by workspace |
| `config/project_records.py` | 2 | The yes/off record for a project's instructions file |
| `config/migrate.py` | 2 | Revision-stamped sync of shipped default deny patterns into an existing config |
| `provider/detector.py` | 2 | Parallel port probe for a local backend (harness-managed vLLM, stock vLLM, Ollama, LM Studio, llama.cpp) |
| `provider/client.py` | 2 | OpenAI-compatible async client with the XML fallback and local timeout handling |
| `provider/fn_call.py` | 2 | XML function-call converter for models without native tool calls |
| `provider/lifecycle.py` | 2 | Launch strategies (managed vLLM, spawned process, daemon, LM Studio, the harness's own unit), `strategy_for`, the swap with its reporter, server recovery |
| `provider/server.py` | 2 | Server lifecycle primitives: install, download, launch, readiness, stop, the live-pidfile guard |
| `provider/units.py` | 2 | The systemd side: what a user unit says, the service-manager runner, the harness unit's render and identity |
| `provider/procfs.py` | 2 | Every `/proc` read: who listens on a port, which process holds a pidfile; imports nothing above it |
| `provider/autosense.py`, `provider/tuning.py` | 2 | What the context window fit is and what a launch derives from the model's own header and the machine's knobs |
| `provider/idle_llm.py` | 2 | The one path for background (idle-time) model calls |
| `tools/base.py` | 2 | `ToolProtocol` (`info()`, `run()`), `Tool`, `ToolSchema`, `ToolResult`, `ToolVetoed` |
| `tools/registry.py` | 2 | `ToolRegistry`: scope resolution, Pydantic dispatch; `dispatch` runs the pre and post hooks around each tool |
| `tools/hooks.py` | 2 | `HookSystem`: pluggy dispatch of the two tool hooks |
| `tools/mcp.py` | 2 | MCP client: stdio and streamable-HTTP servers wrapped as tools |
| `tools/capabilities.py` | 2 | Capability floor: no agent holds untrusted-ingest and host-dangerous tools together |
| `tools/builtin/` | 2 | read, write, edit, glob, grep, bash, python, web search and fetch and page query, load_document, chunk, tool_result_get, git_read, `agent` (delegation), `task` (the working record), the cruncher's exec |
| `plugins/` | 2 | `api.py` (every type a plugin touches), `builtin.py` (the one list), discovery (metadata only), plan, resolve, lifecycle, the memory slot, third-party gate trust |
| `registry/` | 2 | The component catalogue behind `localharness components`: enumerate, resolve, mutate by dot path, provenance |
| `agent/loop.py` | 3 | `AgentLoop`, `Session`, `StuckDetector`, `BudgetTracker`, `KillWatcher`: the ReAct while-loop |
| `agent/context.py` | 3 | `ContextManager`, `TokenCounter`, `ContentStore`, the compaction pipeline (tool-result cap, boundary guard, summary compaction, full auto-compact) |
| `agent/permissions.py`, `agent/verdict.py` | 3 | Deny-pattern matching, and the pure verdict: DENY, ungrantable ASK, grant, grantable ASK, ALLOW |
| `agent/gate.py`, `agent/gate_types.py` | 3 | The effectful gate: ask a human, remember the answer; the shared vocabulary |
| `agent/subagent.py` | 3 | Bounded child agents (explore, web-researcher, search-verifier, cruncher, and YAML specialists) on the same bus |
| `agent/task_record.py`, `agent/task_context.py` | 3 | One persisted working record per agent over an evidence core that is never model-authored proof |
| `orchestrator/router.py` | 4 | `AgentCreationFlow`: holds the card registry, begins and tracks the creation workflow |
| `orchestrator/workflow.py` | 4 | The agent-creation state machine |
| `orchestrator/cards.py` | 4 | `AgentCard`: compact capability declarations, one per configured agent |
| `cli/app.py` | 5 | The Typer app; core commands registered here, plugin commands mounted lazily |
| `cli/init_cmd.py`, `cli/init_import.py` | 5 | `localharness init`: detect or guide, write config; translate a running llama-server into a record |
| `cli/start_cmd.py` | 5 | `localharness start`: trust, what serves, session wiring, the REPL |
| `cli/model_ops.py`, `cli/start_serving.py` | 5 | The one selection ladder and the sensor; the attach / swap / launch decision and the swap under the launch lock |
| `cli/model_cmd.py`, `cli/autostart.py`, `cli/serve_cmd.py` | 5 | `localharness model`: list, switch, `--autostart on\|off\|status`; the hidden `serve` the unit runs |
| `cli/repl.py` | 5 | `OrchestratorREPL`: slash commands, the creation workflow, the turn; `cli/slash_commands.py` is the one slash table |
| `cli/doctor_cmd.py` | 5 | Prerequisite checks, each plugin's rows included |
| `cli/workspace.py`, `cli/session_trust.py` | 5 | Does a workspace layer apply to this invocation; one trust question per new workspace |
| `channels/base.py` | 5 | `ChannelAdapter`: the port every channel implements (send, stream, ask permission, the event handlers) |
| `channels/terminal.py`, `channels/acp.py` | 5 | The terminal (Rich output, prompt_toolkit input) and the Agent Client Protocol adapter (Zed) |
| `bench/` | 5 | Scenario schema, runner, aggregator, compare, ask-rate report behind `localharness bench` |
| `memory/` | plugin | Resonance recall, salience, the facts store, history, MEMORY.md notes, idle consolidation |
| `channels/mobile/` | plugin | The phone app: SSE event API, auth, Web Push, the static shell |
| `dispatch/` | plugin | Chat platforms as a channel; Discord is the first adapter |
| `autoresearch/` | plugin | The experiment loop: proposer, experiment gate, archive, adoption, sentinel |

### What Talks to What

| Component | Receives From | Sends To |
|-----------|---------------|----------|
| REPL (`cli/repl.py`) | The channel's input | Event bus (UserMessage), then `AgentLoop.run_turn`; slash commands and the creation workflow are handled before a turn starts |
| Terminal channel | Event bus (Action, Observation, TaskComplete, TurnFailed, ...) | User stdout |
| Detector | CLI (init flow) | Config loader (writes the provider config) |
| LLM client | Agent loop | The model server (HTTP) |
| Event bus | All components (publish) | All subscribers, in order; the JSONL log |
| Config loader | CLI at startup | Agent loop constructor (injected) |
| Agent loop | `run_turn` (the REPL, a channel, a parent loop) | Event bus (TurnStarted, Action, Observation, TaskComplete, TurnCompleted or TurnFailed) |
| Tool registry | Agent loop (direct call on its injected `tool_registry`) | Agent loop (ToolResult; the loop publishes the Observation) |
| Hook system | Tool registry (pre and post, inside `dispatch`) | Tool registry (veto or pass) |
| Permission verdict and gate | Agent loop (before a tool runs) | Agent loop (allow, deny, or a human's answer) |
| Memory plugin | Agent loop (through its injected `memory_slot`), event bus | Agent loop (the recalled section each turn) |
| Context manager | Agent loop | Agent loop (the message list that fits the window) |
| Launch strategy | `start`, `model`, the REPL's `/model` | The model server process or its unit; the pidfile and the launch records |
| Event log | Event bus (all events) | Disk (`agents/<name>/bus-events.jsonl`; component changes also go to `audit.jsonl`, set by `org.audit_log_path`) |

---

## 5. Communication Rule

**Facts travel on the event bus, the event/data plane: no component holds a reference to another in order to tell it what happened.** Every fact a session produces (a user message, a model reply, a tool call and its result, a turn's end) is published as a typed event on the one ordered bus, so the log is the session (P1, P2).

This is the most important structural rule. It enables:
- Replay: reconstruct any session from its event log
- Testing: inject test events without starting real components
- Debugging: inspect the complete event sequence for any failure
- Future parallelism: move components to separate processes or threads without changing interfaces

The rule governs facts, not capabilities. A component that needs to *do* something through another component (call the model, run a tool, ask the permission gate) holds that collaborator as a constructor parameter. That is the second plane, service wiring, and the list below is all of it.

**Config Loader:** a synchronous dependency injected at construction time. It is not a subscriber. It loads config once per agent instantiation and passes it to the agent loop constructor (the `config` parameter below).

### Constructor wiring (the sanctioned exceptions)

A component that needs a capability receives it as a constructor parameter, supplied by the start-up wiring (`cli/start_cmd.py` for a session, `agent/subagent.py` for a child loop, `plugins/lifecycle.py` for a plugin's context). It never discovers one by importing a module or looking it up. These are the exceptions, by name, and a test holds this list equal to the code:

- `AgentLoop.__init__`: `config`, `llm`, `bus`, `context_manager`, `tool_registry`, `permission_evaluator`, `kill_file_path`, `compact_md_path`, `session_id`, `config_dir`, `gate`, `guardrails_path`, `server_recovery`, `memory_slot`, `task_context`, `template_margin_ratio`, `effort`, `effort_ratios`, `budget_fallback`, `budget_labels`, `unattended_max_actions`, `project_instructions`
- `PluginContext`: `bus`, `tools`, `hooks`, `config`, `agent_config`, `paths`, `llm`, `idle_llm`, `session` (the last two are additive fields; the first seven are the original plugin API)

What the loop does with them: it publishes facts on `bus`, calls the model through `llm`, runs tools through `tool_registry.dispatch`, asks `gate` and `permission_evaluator` before a tool runs, and asks `memory_slot` for the recalled context it adds to the turn's system prompt. The results it gets back become events again, so the replay log stays complete.

#### Design notes

Two planes exist because they carry different things. Events carry facts, and a fact must be recorded in order so a session can be replayed. A collaborator is a capability, and a capability must be handed to a component by whoever builds it rather than found by the component itself. Martin Fowler's article gave this second idea its common name, dependency injection, with constructor injection as one of its three forms, and framed it as "separating configuration from use" ([Fowler, 2004](https://martinfowler.com/articles/injection.html)); in LocalHarness, `cli/start_cmd.py` is the configuration and `AgentLoop` is the use. Channels and the memory slot follow the ports-and-adapters idea from Alistair Cockburn ([Cockburn, 2005](https://alistair.cockburn.us/hexagonal-architecture/)): the core defines the port (`ChannelAdapter`, the memory slot's interface), and a technology-specific adapter (the terminal, Zed, Discord, the phone app, the memory plugin) plugs into it, so the core stays ignorant of which one is there. The import-direction test is what keeps the wiring plane pointing one way: core never imports a plugin, so a plugin can only receive capabilities, never be reached into.

### Enforcement

When writing a new component:
1. It publishes the facts it produces as events, and reads other components' facts by subscribing
2. It receives every collaborator it calls as a constructor parameter, and nothing else is wired; adding one means adding it to the list above
3. It does not reach into another component's internals (private attributes, module globals)
4. A core module never imports a plugin module; a plugin reaches core only through `ctx` (its `PluginContext`). `plugins/builtin.py` is the one core module allowed to import plugins. `tests/unit/test_import_direction.py` scans every import (including lazy and type-checking imports) and fails on a new edge
5. Its wiring section here stays equal to the code: `tests/unit/test_docs_convergence.py` fails if a constructor parameter or `PluginContext` field is added or removed without this list changing

---

## 6. Data Flows

### 6.1 Setup: `localharness init`

```
User runs: localharness init
      │
      ▼
CLI (cli/init_cmd.py)
      │
      ├── Detector (provider/detector.py) probes the local ports in parallel:
      │       8081 harness-managed vLLM, 8000 stock vLLM, 11434 Ollama,
      │       1234 LM Studio, 8080 llama.cpp  →  the first answer names the backend
      │
      ├── No answer: guided setup from a reference-architecture profile
      │       (provider/refarch.py), or an import of a running llama-server
      │       into a launch record (cli/init_import.py)
      │
      ├── Config loader writes <config dir>/config.yaml (owner-only)
      │
      └── CLI prints what it found and what it wrote
```

### 6.2 Start: `localharness start`

```
User runs: localharness start [--preset <name>]
      │
      ▼
CLI (cli/start_cmd.py)
      │
      ├── Trust: which workspace layer applies (cli/workspace.py), one question per new
      │   workspace (cli/session_trust.py), machine-level-only settings never read from a project
      │
      ├── What serves: the one ladder (cli/model_ops.resolve_serving) names the record;
      │   the sensor says what answers; cli/start_serving.decide picks
      │       attach           — a server already answering for that record
      │       swap             — stop the incumbent, launch the target (lifecycle.swap)
      │       launch           — nothing serves: sense → plan → confirm → launch
      │   through strategy_for(record, unit=…): a spawned process, a daemon, LM Studio,
      │   managed vLLM, or the harness's own systemd user unit when autostart is on.
      │   Nothing the owner has not seen in the machine record is ever launched.
      │
      ├── Session wiring:
      │       EventBus(persist_path=…)         core/bus.py, one JSONL line per event
      │       ToolRegistry + HookSystem        built-in tools registered, MCP servers wrapped
      │       plugins: resolve → create → configure → tools → start   (plugins/lifecycle.py)
      │       MemorySlot                       occupied by the memory plugin, or empty
      │       ContextManager                   the window fit from autosense, the compaction pipeline
      │       PermissionEvaluator + PermissionGate
      │       AgentLoop(…)                     every collaborator passed in (section 5)
      │       TerminalChannel                  subscribes to the bus
      │
      └── OrchestratorREPL (cli/repl.py) reads input until exit; stop_plugins in reverse order
```

### 6.3 A Turn

```
User types a line
      │
      ▼
REPL (cli/repl.py)
      │
      ├── A slash command (the one table, cli/slash_commands.py)? run it, done
      ├── An agent-creation workflow active, or creation intent? drive the state machine
      │
      └── Otherwise: publish UserMessage, call AgentLoop.run_turn(text)
            │
            ▼
Agent Loop (agent/loop.py)
      │
      ├── Publish TurnStarted; BudgetTracker and StuckDetector for this turn
      ├── Memory slot: the recalled section for the system prompt (contained; a failure costs
      │   the section, not the turn)
      ├── Context manager: build the message list; the compaction pipeline keeps it inside the
      │   window (tool-result cap → boundary guard → summary compaction → full auto-compact);
      │   evicted bodies stay addressable through the ContentStore and tool_result_get
      │
      └── WHILE LOOP:
            ├── Budget check (iterations, duration, actions) → TurnFailed if exhausted
            ├── LLM client streams the completion; tool calls parsed natively or from the XML fallback
            ├── Publish Action (the model's reply)
            ├── No tool calls → BREAK
            └── For each tool call:
                  ├── Verdict (agent/verdict.py): deny patterns, the shell classifier, the
                  │   capability floor → DENY | ASK | ALLOW
                  ├── ASK → the gate (agent/gate.py) asks the human through the channel and
                  │   remembers the answer (config/grants.py); off a terminal it fails closed
                  ├── Denied → publish Observation(permission denied)
                  ├── tool_registry.dispatch: pre hooks → tool.run → post hooks
                  ├── Publish Observation (the result; large bodies stored, a handle in context)
                  └── The bus persists each event as it is published
      │
      ├── StuckDetector: repeated action signatures → recovery, StuckRecovered
      ├── Model server gone mid-turn → ServerRecovery waits once, ProviderLost if it stays gone
      │
      └── On BREAK: publish TaskComplete, then TurnCompleted (or TurnFailed)
            └── The channel renders the reply; the task record keeps the working state
```

### 6.4 Agent Creation

```
User types: "create an agent that monitors Hacker News"
      │
      ▼
REPL detects creation intent
      │
      └── AgentCreationFlow.begin_agent_creation (orchestrator/router.py)
            │
            ▼
AgentCreationWorkflow (orchestrator/workflow.py), a state machine driven turn by turn:
      │
      ├── Discuss: the model asks what the agent is for; the user answers until confirmed
      │   (a cancel phrase at the start of a line ends it, deterministically)
      ├── Configure: an AgentConfig is written to <config dir>/agents/<name>.yaml
      └── Deploy: the Agent Card registry gains the agent; the `agent` tool's roster grows
```

---

## 7. Project Structure

The repository by directory. The source tree is the authority and `python scripts/gen_index.py` lists every file; this is the shape.

```
localharness/
├── pyproject.toml            # entry point: localharness = localharness.cli.app:main; extras; tool settings
├── uv.lock
├── README.md, SECURITY.md, CHANGELOG.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md, LICENSE
├── docs/
│   ├── specs/                # these specs (00 is the map; the rest are the territory)
│   ├── plugins/              # one guide per bundled plugin
│   ├── runtimes/             # running on llama.cpp, LM Studio, Ollama
│   └── reference-architectures/   # hardware profiles behind guided setup
├── examples/
│   ├── agents/               # example agent YAML
│   ├── plugin-template/      # a starting point for a third-party plugin (a separate distribution)
│   └── workflows/            # a worked multi-agent workflow
├── bench/                    # bench.yaml, categories, fixtures, scenarios/train, scenarios/holdout (sealed)
├── subagents/                # specialist YAML the `agent` tool can delegate to
├── scripts/                  # maintainer scripts: the index generator, icon and bench utilities
├── .github/                  # issue and PR templates; CI runs the suite and installs the built wheel
│
├── src/localharness/
│   ├── core/                 # Layer 1
│   ├── config/               # Layer 1 (models.py) and Layer 2
│   ├── provider/             # Layer 2: the wire client and the model-server lifecycle
│   ├── tools/                # Layer 2: base, registry, hooks, mcp, capabilities
│   │   └── builtin/          # the built-in tools; the image plugin's tool and workflow live here too
│   ├── plugins/              # Layer 2: the plugin API, the one bundled list, discovery, plan, lifecycle, slot
│   ├── registry/             # Layer 2: the component catalogue
│   ├── agent/                # Layer 3
│   ├── orchestrator/         # Layer 4
│   ├── channels/             # Layer 5: base, terminal, acp, input router, keys
│   │   └── mobile/           # plugin: the phone app
│   ├── cli/                  # Layer 5: app, start, repl, model, doctor, init, ...; plugin command modules
│   ├── bench/                # Layer 5: the benchmark runner
│   ├── memory/               # plugin: memory
│   ├── dispatch/             # plugin: dispatch (adapters/discord.py)
│   ├── autoresearch/         # plugin: autoresearch
│   └── __main__.py           # `python -I -m localharness serve`, the body of the autostart unit
│
└── tests/
    ├── unit/                 # the suite, including the structural pins of section 11
    ├── integration/
    ├── fixtures/
    └── conftest.py           # the socket guard, the service-manager seam stub, fake_home
```

---

## 8. Technology Stack

Constraints as `pyproject.toml` declares them; `uv.lock` holds the resolved versions. A test checks the list both ways: every third-party import is declared, and every declared main dependency is imported.

| Library | Constraint | Purpose | Layer |
|---------|-----------|---------|-------|
| Python | 3.12+ | The language | all |
| pydantic | >=2.13,<3 | Event models, config models, tool schemas | 1 |
| pydantic-yaml, PyYAML | >=1.4,<2; >=6,<7 | YAML parse and round-trip | 2 |
| openai, httpx | >=1,<2; >=0.27 | The OpenAI-compatible client and the HTTP transport everything else shares | 2 |
| mcp | >=1.27.1,<2 | The MCP client SDK (2.0 renamed the streamable-HTTP client; raise only with a shim) | 2 |
| huggingface_hub | >=0.26 | Model downloads and metadata | 2 |
| tiktoken | >=0.13 | Token counting where the server has no tokenizer endpoint | 3 |
| aiosqlite | >=0.22,<1 | The memory plugin's facts store | plugin |
| pluggy | >=1.5,<2 | The two tool hooks | 2 |
| structlog, anyio | >=25.5,<26; >=4,<5 | Structured logging; the bus's JSONL persistence | 1 |
| ddgs | >=6 | Keyless web search | 2 |
| scipy, numpy | >=1.17,<2; >=1.26,<3 | Bench aggregation and the sequential stopping rule | 5 |
| Typer, click, Rich, prompt_toolkit | >=0.25,<1; >=8,<9; >=15,<16; >=3,<4 | CLI, terminal output, REPL input | 5 |
| agent-client-protocol | >=0.12,<0.13 | The Zed adapter (pre-1.0 SDK, pinned to its method signatures) | 5 |
| packaging | >=23.2 | `localharness update` version comparison | 5 |

Optional extras:

| Extra | Packages | Enables |
|-------|----------|---------|
| `dispatch` | discord.py | The Discord adapter |
| `mobile` | starlette, uvicorn, py-vapid, http-ece, segno, cryptography | The phone app's server, Web Push and the enrolment QR |
| `embeddings` | sentence-transformers, torch | The memory plugin's local embedding model (a hashing fallback runs without it) |
| `exact-tokenizer` | llama-cpp-python | Exact token counts from the served model's own GGUF vocabulary |
| `json-repair` | json-repair | Last-resort repair of malformed JSON from a local model |
| `dev` | pytest, pytest-asyncio, ruff, mypy, pillow | The test suite and tooling |

Build and tooling: `uv` manages the environment and lockfile (`uv sync --extra dev`; a plain `uv sync` strips the dev tools), `hatchling` builds the wheel (`src/localharness` only) and a trimmed sdist. There is no compiled code.

---

## 9. Design Principles

### P1: Event-Sourced State

Events are the source of truth. The bus persists every event as one JSONL line, in order, as it is published; `EventBus.replay` reads them back and skips a torn line rather than crashing. The log is not just audit, it is the session.

Source: OpenHands V1 (November 2025) replaced their original pub/sub after it caused thread and async issues with few guarantees on message order. LocalHarness adopted the ordered-log model from day one.

### P2: Ordered Append Log, Not Pub/Sub

Events have sequence numbers. Subscribers process events in order. Replay is deterministic. Unordered multi-subscriber pub/sub is rejected.

### P3: Lean Coordinator, Fat Worker

A coordinator passes paths and briefs, never contents. Each child agent starts with a fresh context window and a bounded budget, and reports back a summary. Agent Cards exist so a coordinator can hold many agents' capabilities in a small context.

### P4: ReAct While-Loop, Not a Graph

The agent loop is a plain Python while-loop with explicit break conditions. Graph-based execution engines add indirection without benefit for a single agent.

### P5: Minimal Tool Interface

Every tool implements `info() -> ToolSchema` and `run(**kwargs) -> ToolResult`. MCP-discovered tools and plugin tools are wrapped in the same interface. A tool can refuse through `ToolVetoed`; the registry's `dispatch` runs the pre and post hooks around every call, so no tool runs unhooked.

### P6: Deny First, Then Ask, Then Allow

A tool call is judged before it runs, by a pure function and then an effectful gate. The verdict ladder is DENY, then an ASK that nothing can grant, then a remembered grant, then a grantable ASK, then ALLOW. Deny patterns are config; the shell classifier reads a command's structure rather than its text; the capability floor keeps untrusted-ingest and host-dangerous tools out of one agent's hands. The gate asks the human through whichever channel is live and records "always" and "never here" answers per workspace; off a terminal, it fails closed. `SECURITY.md` is the authority on this gate and on what runs unasked.

### P7: Config Over Code

Users define agents in YAML. The harness interprets config. Agent behavior changes through YAML, not Python.

### P8: Machine-Level Trust

A project checkout can lower a setting but never raise one, and a fixed list of settings may be set only in the machine-level config: what gets launched, with which binary, under which environment, which hooks run, where files go. The list lives in `SECURITY.md` and a test holds it equal to the code. A launch entry the owner has not seen in the machine's record is never launched; the harness asks once and remembers.

### P9: One Owner Per Fact

Anything that could be spelled twice is spelled once and read from there: one list of bundled plugins, one slash-command table, one selection ladder for what serves, one answer to where the config dir is, one place every `/proc` read lives, one writer for each private store. The structural tests in section 11 pin several of these.

### P10: Model Agnostic, Zero Cloud

No model-specific assumptions in harness code. Detection handles endpoint discovery; the XML fallback handles models without native function calling; all model interaction goes through `LLMClient` over the OpenAI-compatible wire. Everything runs on local hardware: no telemetry, no external API calls that the user did not configure.

### P11: Pure Python, Port When Measured

There is no compiled code. A hot path is rewritten in another language only after a profiler names it under real load.

---

## 10. Anti-Patterns

### AP1: Reaching Past the Wiring

**Violation:** a core module importing a plugin (for example `from localharness.memory.sqlite import ...` inside `agent/loop.py`), or any component reaching into another's internals (its private attributes or module globals) instead of using what it was handed.

**Why wrong:** The import makes the plugin impossible to switch off, because core now needs it to load. Reaching into internals couples two components to each other's implementation, so neither can change or be tested alone. Either way, the dependency appears nowhere in the wiring list, so nobody reviewing the constructor can see it.

**Correct:** Receive the collaborator as a constructor parameter and call it there. The loop's `tool_registry` parameter is the sanctioned form: `agent/loop.py` runs every tool through `self._tools.dispatch(...)` on the registry it was given, then publishes the result as an `Observation` event. Memory reaches the loop the same way, through the `memory_slot` parameter, never by import.

### AP2: Graph-Based Agent Execution

**Violation:** Using a graph engine or a state machine to drive agent execution steps.

**Why wrong:** Adds indirection without benefit. The agent loop is a plain loop.

**Correct:** Plain Python `while True` with explicit `break` conditions. (The agent-creation workflow is a state machine on purpose: it is a conversation with fixed stages, not an agent's reasoning.)

### AP3: Fat Coordinator

**Violation:** A coordinator reads file contents, loads every agent's context, or does the domain reasoning itself.

**Why wrong:** Context explosion on the first multi-agent task.

**Correct:** The coordinator reads Agent Cards and summaries, passes paths and a brief, and lets a bounded child do the work.

### AP4: Tight LLM Coupling

**Violation:** Calling a vendor SDK directly, or assuming `tool_calls` is present in a response.

**Why wrong:** Breaks the model-agnostic requirement.

**Correct:** All model calls go through `LLMClient`; `provider/fn_call.py` handles the function-calling difference.

### AP5: A Second Spelling

**Violation:** A second list of bundled plugins, a second slash table, a second place that decides what serves, a second `/proc` reader, a second pidfile writer.

**Why wrong:** The two drift, and the test that pins the first does not see the second.

**Correct:** Read the one owner (P9). If it does not fit, change the owner.

### AP6: Unordered Pub/Sub

**Violation:** Firing events to multiple subscribers in any order, with no sequence guarantee.

**Why wrong:** Replay becomes non-deterministic and the log stops being the session.

**Correct:** The ordered append log. Events have sequence numbers.

### AP7: Conversation-as-Memory

**Violation:** Treating the model's conversation history as the agent's memory.

**Why wrong:** The context window is bounded. Cross-session continuity is impossible. This is the most common harness failure mode.

**Correct:** Memory is explicit external state behind the memory slot (the memory plugin's store, MEMORY.md, the JSONL history), and the per-turn injection is explicit and bounded.

### AP8: A Doc That Duplicates the Tree

**Violation:** Listing every module and its line count in a hand-written document.

**Why wrong:** It is stale by the next release and nobody notices.

**Correct:** Generate it (`scripts/gen_index.py`); keep the hand-written docs at the level that changes slowly, and let tests hold the few facts that must stay exact (section 11).

---

## 11. Enforced Boundaries

The dependency rules that are tests, not intentions. Each fails on the first new edge.

| Rule | Test |
|------|------|
| No core module imports a plugin module; `plugins/builtin.py` is the one exception. Every `.py` is classified core or plugin by an explicit list; lazy imports, `TYPE_CHECKING` imports and `import_module` literals all count | `tests/unit/test_import_direction.py` |
| The bench never imports the autoresearch plugin (the plugin calls the bench, never the reverse) | `tests/unit/test_import_direction.py` |
| `provider/procfs.py` imports nothing from `provider/server.py`, `provider/lifecycle.py`, `provider/units.py` or `cli/` | `tests/unit/test_procfs_import_direction.py` |
| `config/trust.py` reaches the provider layer only through function-local imports; importing config pulls no provider module in | `tests/unit/test_trust_autostart_structural.py` |
| No plugin, bundled or the template, can reach `cli/start_cmd.py` | `tests/unit/test_plugin_never_imports_start_cmd.py` |
| Listing a plugin never imports it (discovery is metadata only) | `tests/unit/test_plugin_discovery.py`, `tests/unit/test_plugins_cli.py` |
| Section 5's wiring list equals `AgentLoop.__init__` and `PluginContext`; `SECURITY.md`'s machine-level-only list and launch keys equal the code; public docs carry no stale names or internal ids; every relative link resolves | `tests/unit/test_docs_convergence.py` |
| Every third-party import is declared and every declared main dependency is imported | `tests/unit/test_declared_dependencies.py` |
| Writes to the private stores go through one writer each; the launch lock has a fixed set of openers; no unasked `ask=True` under `cli/` outside the autostart switch | the structural pins beside each store and command in `tests/unit/` |

---

## 12. Directions

Where the architecture leaves room, without committing a release to it.

| Concern | Today | Room |
|---------|-------|------|
| Event bus | In-process asyncio, one JSONL log per agent | Several agents on separate logs; no change to the event contract |
| Tool execution | Sequential within a turn | Parallel tool calls within one turn, behind the same `dispatch` |
| Channels | Terminal and Zed in core; the phone and Discord as plugins | Further chat platforms as `dispatch` adapters |
| Permissions | Deny patterns, the verdict ladder, the human gate, the capability floor | Sandboxing of the shell tool |
| Memory | One slot, one occupant, the bundled plugin | Another occupant behind the same slot interface |
| Model servers | One server per config dir, launched by strategy, optionally through a user unit | Presets served on another device over a private network (advisory in spec 20) |

---

## 13. Cross-References

| Topic | Document |
|-------|----------|
| Trust boundaries, the approval gate, machine-level-only settings, the threat model | [SECURITY.md](../../SECURITY.md) |
| How to contribute, test conventions | [CONTRIBUTING.md](../../CONTRIBUTING.md) |
| Event types, `EventBus` | [01-event-bus.md](01-event-bus.md) |
| Provider detection, the LLM client | [02-provider.md](02-provider.md) |
| The agent loop, the permission evaluator | [03-agent-loop.md](03-agent-loop.md) |
| Tool interface, registry, hooks | [04-tool-system.md](04-tool-system.md) |
| MCP | [04b-mcp-integration.md](04b-mcp-integration.md) |
| Memory (historical design; the module docstrings are current) | [05-memory.md](05-memory.md), [docs/plugins/memory.md](../plugins/memory.md) |
| Config YAML schema, config models | [06-config.md](06-config.md) |
| Orchestrator, agent creation, Agent Cards | [07-orchestrator.md](07-orchestrator.md) |
| Context management (historical design; the module docstrings are current) | [08-context-management.md](08-context-management.md) |
| Hooks and the plugin API | [09-hooks-plugins.md](09-hooks-plugins.md) |
| CLI commands | [10-cli.md](10-cli.md) |
| Channels | [11-channels.md](11-channels.md) |
| Audit and observability | [12-audit.md](12-audit.md) |
| Provider support and the server lifecycle | [13-provider-support.md](13-provider-support.md) |
| Co-author consent | [14-coauthor-consent.md](14-coauthor-consent.md) |
| `/mcp`: in-session MCP server management | [15-mcp-slash-command.md](15-mcp-slash-command.md) |
| Model presets, window autosense, `/effort` | [16-model-presets-autosense-effort.md](16-model-presets-autosense-effort.md) |
| Loop hardening: reconnect, stuck detection, budgets, project instructions | [17-loop-hardening.md](17-loop-hardening.md) |
| Tuned presets, `start --preset`, autostart | [20-tuned-presets-start-on-preset-autostart.md](20-tuned-presets-start-on-preset-autostart.md) |
| Terminal keys | [21-terminal-keys.md](21-terminal-keys.md) |
| The phone app, Zed, the runtimes | [docs/mobile.md](../mobile.md), [docs/zed.md](../zed.md), [docs/runtimes/](../runtimes/) |
| Task context, references, delegation and resume | [docs/task-context.md](../task-context.md) |
| Hardware profiles | [docs/reference-architectures/](../reference-architectures/) |
