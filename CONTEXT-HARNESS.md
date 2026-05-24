# Context: Hierarchical Agent Harness Design

## Vision

An open-source, model-agnostic hierarchical agent harness for local LLMs. Specialized agents organized into divisions, each with isolated context/memory, communicating via Discord subchannels, orchestrated by a thin routing layer with automatic cross-agent indexing. Designed so anyone running local LLMs can configure it for their own use case.

## Project Principles

- **Open-source, clean repo** — no personal data or workloads baked in. Structured git history, READMEs, docs.
- **The harness IS the product** — 98.4% operational infrastructure, 1.6% AI decision logic (from Claude Code source analysis). The LLM is interchangeable.
- **Configurable, not coded** — users define agents/divisions/tools in YAML, not Python.
- **Harness effect is real** — same model swings 36 points (42%→78%) depending on harness. Same Opus: 77% default Claude Code → 93% in Cursor's harness. Harness engineering is where the leverage lives.

---

## Research Sources

Architecture informed by deep analysis of 5 harnesses, 8 multi-agent frameworks, and production enforcement patterns:

### Harnesses analyzed (source code read)

| Harness | Key Extraction |
|---------|---------------|
| **OpenHands** (68.6K stars, MIT) | Event-driven bus, Action/Observation duality, fn_call_converter for non-function-calling models, LLMSummarizingCondenser, StuckDetector, microagent keyword triggering, LiteLLM provider abstraction |
| **OpenCode** (160K stars, MIT, Go) | Typed pubsub broker, minimal tool interface (Info+Run), permission-as-blocking-channel, SummaryMessageID compaction, sub-agent-via-tool, local model auto-discovery, LSP as first-class tool, SQLite sessions with parent-child |
| **Claw Code / Claude Code** (160K stars, clean-room rewrite) | ReAct while-loop core, deny-first permission evaluation, 5-layer compaction pipeline, tool-use/tool-result boundary guard, recovery recipes with structured ledger, session health probes, summary-only subagent returns, append-only JSONL, OpenAI-compat provider cascade |
| **GSD** (Claude Code skill) | Lean orchestrator/fat subagent, discuss→research→plan→execute→verify workflow, single-call init, goal-backward verification (must_haves), wave-based parallelism, model profile system, atomic artifact commits, context restoration (pause/resume) |

### Multi-agent frameworks researched

| Framework | Key Pattern |
|-----------|------------|
| **Anthropic (Claude Agent SDK)** | 5 canonical patterns: Generator-Verifier, Orchestrator-Subagent, Agent Teams, Message Bus, Shared State. Planner-Generator-Evaluator triad for quality. Isolated context windows, summary-only returns. |
| **LangGraph** | Graph-based orchestration, state checkpointing after each node, reducer-based state updates, PostgresSaver for production |
| **CrewAI** | Role-based agents, hierarchical mode with manager delegation, task-scoped tool access |
| **AutoGen/AG2** | GroupChat conversation patterns, speaker selection strategies, Microsoft Agent Framework merger |
| **Google ADK** | Tree-structured agent hierarchy, AgentTool wrapping, 8 built-in patterns, native A2A + MCP |
| **OpenAI Swarm → Agents SDK** | Handoff primitives, agents-as-tools, Temporal durable execution |
| **MetaGPT** | SOP-driven roles, assembly-line paradigm, structured document passing |
| **Mem0** | Multi-scope memory: user_id + agent_id + session_id + org_id tagging, scope-composed retrieval |

### Enforcement patterns researched

PEV patterns, permission models (Claude Code 3-tier, Codex Guardian, OpenHands inline risk, Cline HITL), computational controls (lint/typecheck/test gates), inferential controls (guardian subagent, adversarial review), stuck detection (action signature hashing, error normalization), sandboxing (bubblewrap/Landlock/Seatbelt, Docker, network proxy), audit logging (hash-chained JSONL, decision provenance schema, GUARDRAILS.md).

---

## Architecture

### Hierarchy

```
Orchestrator (thin: route, index, escalate, synthesize)
├── Division: Financial
│   ├── Agent: Morning Briefing
│   ├── Agent: Portfolio
│   └── Agent: Afternoon Reconciliation
├── Division: Engineering
│   ├── Agent: Coder
│   └── Agent: Infra
├── Division: Research
│   ├── Agent: Model Research
│   └── Agent: Memory Research
└── (extensible via YAML)
```

Each agent has: own Discord subchannel, own context/memory (SQLite + .md + chat history), own tool set (inherited + scoped), own permission policy (inherited, can only narrow).

### Core Components

#### 1. Event Bus

Central typed event stream. All components communicate through events, never direct calls. From OpenHands EventStream + OpenCode typed pubsub.

- **Event types:** Action, Observation, Delegation, DelegateResult (summary only), Escalation, Heartbeat
- **Subscribers:** Orchestrator, Runtime, Memory, Audit, Channel
- **Persistence:** Append-only JSONL per agent

#### 2. Agent Definition (YAML config, not code)

```yaml
name: morning-briefing
division: financial
role: "Generate daily morning market intelligence report"
model: qwen3.5-122b-a10b  # or inherit from division
channel: discord://morning-briefing
tools:
  inherit: [division]
  add: [exa_search, exa_crawl]
  deny: [execute_bash]
permissions:
  mode: auto  # allow everything except self-config modification
  deny_patterns:
    - "write(*/settings.json)"
    - "write(*/.env)"
    - "bash(sudo:*)"
  budget:
    max_actions: 100
    max_duration_minutes: 30
memory:
  sqlite: ~/.agents/financial/morning-briefing/memory.db
  persistent_md: ~/.agents/financial/morning-briefing/MEMORY.md
  shared_read: [division]
schedule:
  cron: "30 5 * * 1-5"
  timezone: America/New_York
context:
  system_prompt_file: prompts/morning-briefing.md
  microagents: [market-data, portfolio-context]
  max_context_tokens: 200000
```

#### 3. Agent Loop (ReAct while-loop, from Claude Code)

```
run_turn(agent, input):
    1. Load agent config + division config
    2. Load context: system prompt + microagents + memory
    3. Push user/task message to session
    4. LOOP:
        a. Check iteration/budget limits
        b. Build request (system_prompt + session.messages)
        c. Stream LLM response → extract tool calls
        d. Push assistant message to session
        e. IF no tool calls → BREAK
        f. FOR EACH tool_call:
            - Permission check (auto mode: allow unless deny-pattern match)
            - Execute tool
            - Post-hook: lint/typecheck gates (if applicable)
            - Push tool_result to session
        g. Check stuck detector (action signature hashing)
    5. Auto-compact if needed
    6. Return summary to orchestrator
```

NOT a graph. Graphs add complexity without proportional benefit for agent loops.

#### 4. Tool System (from OpenCode)

Minimal interface: `info()` returns schema, `run()` executes. Tools self-describe with JSON schema parameters.

**Scoping:** Global (read-only: glob, grep, view, ls) → Division (shared: portfolio_query) → Agent (specific) → MCP (discovered dynamically).

**Function calling:** Native OpenAI-compatible for models that support it. XML-like fallback for models that don't (OpenHands fn_call_converter pattern).

#### 5. Context Management

**Lean orchestrator / fat subagent** (from GSD — the single most important pattern): Orchestrator stays at ~10-15% context. Passes FILE PATHS to subagents, never contents. Each subagent reads files with fresh context.

**Compaction pipeline:**
1. Tool result budget — cap output size
2. Boundary guard — don't orphan tool_result without tool_use (critical for OpenAI-compat)
3. Summary compaction — at 80% window, summarize middle preserving first N + last N
4. Full auto-compact — comprehensive LLM summary and context reset

**Configurable per model:** Users set `max_context_tokens` in agent config. Context management adapts to the model's actual window size. Critical for local models with varying context lengths.

**Stuck detection:** Action signature hashing over sliding window. Recovery: 1 retry with forced different approach, then escalate.

#### 6. Orchestrator (thin)

Does NOT do work. Four functions:
1. **Route** — match task to agent via Agent Cards (JSON capability descriptions from A2A protocol)
2. **Index** — SQLite FTS5 across all agent memories, updated on write events
3. **Escalate** — stuck/failed/over-budget → retry, delegate, or escalate to human
4. **Synthesize** — combine multi-agent summaries for compound tasks

Agent-to-agent communication is NEVER direct. Always through orchestrator. Enforces hub-and-spoke audit trail.

#### 7. Memory (three-tier hierarchical, from Mem0)

```
Organization level (all agents read)
  ├── GUARDRAILS.md (persistent failure memory)
  ├── shared tools index
  └── global config

Division level (division agents read)
  ├── DIVISION.md
  ├── shared.db (SQLite)
  └── shared tool configs

Agent level (this agent only)
  ├── MEMORY.md
  ├── memory.db (SQLite)
  ├── chat history (append-only JSONL)
  └── session state
```

All writes tagged with `{agent_id, division_id, org_id, timestamp}`. Retrieval composes scopes automatically.

#### 8. Permissions (auto mode default)

Default behavior: **allow everything except a short deny list** (like Claude Code's auto mode). No annoying approval prompts for normal operations.

**Default deny list:**
- Modifying own config/settings
- Writing to credential files (.env, secrets, tokens)
- sudo / privilege escalation
- Self-modifying agent definitions

**Defense in depth (layers available, not all required by default):**
- Layer 0: OS sandbox (bubblewrap) — optional, recommended for untrusted code
- Layer 1: Deterministic gates — lint/typecheck post-hooks, budget enforcement
- Layer 2: Inline risk annotation — zero-cost LLM self-report on every tool call
- Layer 3: Guardian review — separate agent context for flagged operations
- Layer 4: Kill switch — `touch KILL` file → immediate stop

Subagents inherit parent policy, can only narrow.

#### 9. Workflow System (from GSD patterns)

**Discuss → Research → Plan → Execute → Verify → Document**

Each stage produces externalized artifacts (files, not in-context reasoning):

| Stage | Artifact | Purpose |
|-------|----------|---------|
| Discuss | CONTEXT.md | User decisions, locked choices, "you decide" zones |
| Research | RESEARCH.md | Domain research, codebase analysis |
| Plan | PLAN.md (per wave) | Executable steps with must_haves for verification |
| Execute | SUMMARY.md (per plan) | What was done, deviations, decisions |
| Verify | VERIFICATION.md | Goal-backward verification (exists→substantive→wired) |

**Key patterns:**
- **Goal-backward verification:** Plans have `must_haves` — truths (observable behaviors), artifacts (files with constraints), key_links (wiring between components). Verification checks goal achievement, not task completion.
- **Wave-based parallelism:** Plans assigned to waves at planning time. Wave 1 = no dependencies, parallel. Wave 2 depends on wave 1.
- **Revision loop with ceiling:** Planner → checker → revision, max 3 iterations, then user override.
- **Atomic artifact commits:** Every document committed immediately. Context loss leaves artifacts intact.
- **Model profiles:** quality/balanced/budget — critical reasoning gets best model, execution can use cheaper.
- **Single-call init:** One call per workflow returns all metadata as JSON. Zero multi-step state gathering.

#### 10. Channel System (pluggable, Discord primary)

Each agent gets a Discord subchannel. Interface is pluggable — Discord, Slack, stdout, file, webhook.

**Discord-specific:** Agent sends deliverables/status to its channel. Human sends commands/feedback. Orchestrator reads all channels for indexing. Cross-channel communication goes through orchestrator.

#### 11. Audit & Observability

Append-only JSONL with SHA-256 hash chain per agent + org-level aggregate. Every tool call, risk assessment, approval, verification logged with decision provenance (identity, temporal, reasoning, tools, lineage, reversibility).

**GUARDRAILS.md:** Persistent failure memory. When an agent fails the same pattern 3+ times, a guardrail is appended with trigger, instruction, reason, and provenance.

---

## Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python + Rust | Python for agent logic/config, Rust for hot paths |
| LLM Runtime | vLLM | OpenAI-compat API, multi-agent serving, MTP support |
| Primary Model | Qwen3.5-122B-A10B | 72.2% BFCL v4, 51 tok/s on DGX Spark |
| Persistence | SQLite + JSONL | Simple, portable, no external deps |
| Indexing | SQLite FTS5 | Cross-agent memory search |
| Channel | Discord (pluggable) | Subchannels per agent |
| Sandbox | bubblewrap (optional) | OS-level isolation |
| Config | YAML | Agent/division/org definitions |
| Tool extension | MCP (stdio + SSE) | Dynamic tool discovery |

## Key Patterns Adopted (with provenance)

| Pattern | Source | Why |
|---------|--------|-----|
| Event-driven bus | OpenHands | Decouples components, enables replay/debug |
| ReAct while-loop | Claude Code | Simple, proven, debuggable |
| Deny-first permissions | Claw Code | Deny always wins over allow |
| Auto mode default | Claude Code | Don't interrupt agents for routine operations |
| Tool-result boundary guard | Claw Code | Prevents 400 errors on OpenAI-compat |
| fn_call_converter (XML fallback) | OpenHands | Supports models without native function calling |
| Summary compaction | OpenCode | SummaryMessageID — simple, effective |
| Lean orchestrator / fat subagent | GSD | Pass paths not contents, fresh context per subagent |
| Discuss before plan | GSD | Capture decisions as structured CONTEXT.md |
| Goal-backward verification | GSD | Verify goal achievement, not task completion |
| Wave-based parallelism | GSD | Pre-computed dependencies, parallel execution |
| Model profiles | GSD | Quality/cost tradeoff per agent role |
| Atomic artifact commits | GSD | Context loss leaves work intact |
| Agent Cards | A2A Protocol | Self-documenting capabilities for routing |
| Multi-scope memory | Mem0 | Hierarchical agent/division/org scoping |
| Orchestrator-Subagent | Anthropic | Thin orchestrator, isolated subagent contexts |
| Summary-only returns | Anthropic + Claw Code | Prevent context explosion across agents |
| Inline risk annotation | OpenHands | Zero-cost security per tool call |
| Recovery recipes | Claw Code | Structured failure→recovery with escalation |
| Action signature hashing | PEV research | Stuck detection via sliding window |
| Hash-chained audit | Hermes proposal | Tamper-evident logging |
| LSP as tool | OpenCode | Compiler-grade diagnostics |
| Single-call init | GSD | All metadata in one JSON call |
| Context restoration | GSD | pause-work / resume-work handoff |

---

## Open Decisions (for next session)

1. **Project name** — needs a name for the repo
2. **Language split** — how much Rust vs Python? Could start pure Python and optimize hot paths later
3. **Build order** — event bus → agent loop → tool system → orchestrator? Or workflow system first?
4. **Discord bot setup** — one bot with subchannel routing, or separate bot instances per agent?
5. **Model profile defaults** — what models for quality/balanced/budget tiers?
6. **Repo structure** — monorepo or workspace packages?

## What This Is NOT

- Not a fork of any existing harness
- Not Claude Code for local models — different architecture (hierarchical, multi-agent)
- Not a framework/SDK — it's a runnable system with YAML configuration
- Not tied to any specific model or provider
- Not a research project — designed for production daily use
