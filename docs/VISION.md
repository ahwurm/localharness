# LocalHarness — Vision, Principles & Security Direction

> The north star. **Why** this project exists, the **load-bearing principles** every design decision answers to, an **honest map of what is actually built**, and the **committed security direction**. This doc deliberately does *not* duplicate component detail — that lives in [`docs/specs/`](specs/) (canonical) and model/hardware in [`docs/reference-architectures/`](reference-architectures/). Component docs rot; this one is about the *why* and the *invariants*, which don't.
>
> *Supersedes the deleted day-0 `CONTEXT-HARNESS.md`/`CONTEXT-MODEL.md`, which described an aspirational architecture (divisions, per-agent Discord subchannels, FTS5 cross-agent index, hash-chained audit) much of which never shipped. This doc describes the project as it actually is.*

---

## The one sentence (re-ask it after every phase)

**Make a *local, smaller* model punch above its weight on real tasks — measured, safely — by engineering the harness, not by swapping the model.**

If a change doesn't move a local ~27B toward doing real work more reliably *and* survive a hostile read on safety, it isn't this project.

## The bet

The harness, not the model, is most of the capability. The same model swings dramatically across harnesses; harness engineering is where the leverage lives, and a good harness lets a local ~27B (currently Qwen3.6-27B on vLLM — see [`reference-architectures/`](reference-architectures/)) do work that the raw model can't. So:

- **The harness IS the product; the model is interchangeable.** No model-specific assumptions in harness code — any OpenAI-compatible endpoint.
- **The evaluator is the trust boundary.** We don't trust vibes; we trust a sealed train/holdout bench. A weak or invalid evaluator makes any self-improvement loop optimize noise — so getting the *measurement* right outranks getting any one feature shipped.
- **Local-first, zero cloud dependency.** Runs entirely on the user's hardware. Privacy and cost are the reason the audience chose local; we don't betray that.

## Design principles (durable — these arbitrate disputes)

1. **Configurable, not coded.** Agents/tools/topology are YAML, not Python. A user shapes behavior without forking.
2. **Event-driven.** Components talk through a typed event bus (bubus), never direct calls — replayable, debuggable, auditable.
3. **ReAct loop, not a graph.** A while-loop with stuck-detection and budgets handles ~all agent patterns; graphs add complexity without proportional payoff.
4. **Lean orchestrator / fat subagent.** The orchestrator routes and synthesizes; it does not do the work. Subagents get *references* (paths, **handles**) and fresh context — never raw contents inlined into someone else's window.
5. **Summary-only returns.** A child returns a distilled result, not its transcript. This is what keeps multi-agent work from exploding context — and (see Security) what keeps the laundering surface small.
6. **Lossless over content larger than the window.** Sources (web, memory, history, files) are *retained whole and addressable* via a per-agent content store, and navigated by verbs — not dumped into a context window and truncated. This is the current frontier (J3 below).
7. **Statistically honest or it didn't happen.** Welch/Wilson, sealed holdout, no claim-and-run. Never report a number a finished run didn't return.

## What is actually built (honest map)

**Shipped & reachable** — ReAct agent loop (stuck detection, budgets, compaction at ~80% via summarize-middle) · typed event bus · SQLite memory (facts table + append-only history.jsonl) · tool system (minimal `info`/`run`, Pydantic-validated, scoped) · MCP client (stdio + HTTP) · provider auto-detection (Ollama/vLLM/llama.cpp/LM Studio) · YAML config with org→division→agent inheritance · the **bench + autoresearch loop** (proposer → experiment → statistically-honest promotion gate → mutation archive → reversible auto-adopt, sealed holdout enforced) · a read-only **Explore** subagent · **dispatch subagents** with a fixed delegation topology (e.g. `web-researcher → search-verifier`) · per-agent **ContentStore** (handle → `(body, origin)`, sticky taint) + navigation **verbs** (`web_page_query`, `tool_result_get`, `chunk`, `load_document`) · the **cross-agent grant** (`parent_store` + `granted_handles`) wired through `make_explore_agent_runner` on BOTH the live `start` path and the bench runner — a parent hands a child a *handle* not bytes, origin-gated, fail-closed · the **cruncher** (J3 reducer): harness-orchestrated map (lossless `split_lossless` → per-chunk leaf in a fresh granted window) + bounded hierarchical reduce, reachable end-to-end and live-dogfed on the 27B · web/bash/write/edit tools · the **L1 capability floor** (no agent co-resides untrusted-ingest with host-dangerous; enforced at both resolution chokepoints, fails closed; root web-stripped; flag sealed from the proposer) — **scope:** built-in web verbs + MCP on both paths + plugins on the dispatch path; a plugin resolved via inherited `global` scope is a *named residual* (needs a per-tool ingest tag).

**Wired but gated / off by default** — the trusted restricted exec (`cruncher_exec`) + its origin binder are now reached by the cruncher's final-combine, but ONLY when `agent.cruncher.exec_enabled` is set (default off) AND every granted handle is clean-origin (`bind_clean_origin_bodies` refuses untrusted — the F3 gate); the faithful reduce never needs it. Off the J3 critical path by design.

**Aspirational / not built (don't cite as real)** — divisions as a runtime construct, per-agent Discord subchannels, FTS5 cross-agent memory index, hash-chained audit log, three-tier shared memory, native scheduled execution, a guardian-review layer.

## The current milestone (J3) and its jobs

Let a local ~27B **losslessly accomplish tasks over content far larger than its window** — safely.

- **J1 Retain losslessly** — any source kept whole, addressable. *(shipped: ContentStore)*
- **J2 Navigate without loading** — query/slice via verbs, not by dumping into the window. *(shipped: verbs)*
- **J3 Reduce over-window content to a faithful answer** — the hard one. **Mechanism shipped & reachable** (grant keystone + cruncher + harness-orchestrated map / model per-chunk reduce). The open α-vs-β question is RESOLVED: the **harness orchestrates the map** (`split_lossless` → independent leaves), the **model only does the per-chunk reduce + final combine** (Plan α); model-driven 2-level delegation (β) is named-and-deferred. Decided on a real over-window task + a scored bench scenario, not on paper. *Residual (honest):* the scored scenario (25) discriminates on routing + a mid-doc needle but the body still fits the 27B window — a genuinely-larger-than-window stress fixture is a named follow-up; whether J3 is "milestone-done" vs "core mechanism done" is the user's call.
- **J4 Keep host-mutating agents clean** — the security spine, below.
- **J5 Don't regress the 27B** — design for the model we have; a mechanism it won't invoke is worthless.

## Security direction (committed — and the one decision to gut-check)

**Threat:** untrusted bytes (a fetched page, or any tool result carrying attacker-controlled text) entering an agent that can mutate the host → prompt-injection → host action. This is **not hypothetical**: the morning-report pipeline ([localshift](#the-proving-workload)) runs this harness's loop + bash + web tools on a **cron, over un-vetted live web pages.** *(Memory is **not** a current laundering vector: tool output lands in `history.jsonl`, while `memory_get`/`memory_search` read only the `facts` table, which no code path promotes tool output into — verified, don't cite memory as ingestion until that changes.)*

The boundary is **host-dangerous capability** (`bash`/`write`/`edit`/trusted-`exec`) — *not* `python_exec` specifically (too narrow; the prior design gated the least-exposed exec) and *not* "delegation" (too broad; a blind read-only child can't mutate the host). Three layers, primary-first, with one honestly-deferred residual:

- **L1 — Co-residence invariant (topology-first, the floor). [SHIPPED]** No single agent holds both untrusted-ingest *and* host-dangerous capability. Achieved primarily by **who holds what in the YAML topology** (the ingestion agent has no bash; the host-acting root does not fetch), enforced by a check that **rejects** a co-resident toolset at the **resolved-toolset chokepoint every agent passes through** (`get_tools_for_agent` for root via `inherit:[global]` + config `add`; `from_allowed` for dispatched children), failing closed. *Coverage is scoped, not total* — built-in web verbs + MCP on both paths + plugins on the dispatch path; an inherited-`global` plugin is a named residual (per-tool ingest tag is the full fix). *Cheap, structural, adopter-legible.*
- **L2 — Sandbox containment (the backstop — *not yet built*).** The intended defense-in-depth: run host-dangerous tools under OS isolation (e.g. bubblewrap) so a violated or bypassed invariant still has bounded blast radius. **Today this does not exist** — `bash`/`write`/`edit` run with the host's full trust, and the only isolation in the tree is the cruncher's rlimit+timeout subprocess (itself unwired, and by its own docstring "a runaway bound, not an escape-proof sandbox"). Naming L2 as the floor is a *commitment*, not a current control; until it ships, L1 (topology) is carrying the containment load alone. This is the conventional, OSS-legible answer and the next safety build after the grant keystone.
- **L3 — Origin-gated content movement (the lossless enabler).** Content moves between agents as **handles** over the ContentStore, each carrying a **sticky origin taint**; an *untrusted* handle's **body** resolves only inside a no-host-dangerous agent. This is what lets the 27B work losslessly *across* agents **without** breaking L1 — the grant keystone is the mechanism, the origin gate keeps it from laundering verbatim bytes.

**Deferred, named, NOT pretended-closed — the dual-LLM split.** L1–L3 block *verbatim* untrusted bytes from host-dangerous contexts. They do **not** block *summary-laundered influence*: a no-danger processor distills untrusted content and returns a summary to a bash-holding parent. Distillation degrades attacker control but does not eliminate it. Full closure (host-dangerous agents receive *only* handles; a separate no-danger composer reads summaries) is a heavyweight re-architecture, deferred until a live red-team shows the residual is exploitable on the 27B. We ship the high-severity verbatim closure + sandbox; we name the residual.

> **Gut-check decision:** the primary floor is **topology (L1) + sandbox (L2)** — *not* a general runtime taint-tracking engine. The origin/grant machinery (L3) earns its place as the **lossless-handoff enabler**, not as the headline security mechanism. If we'd rather invest in the full taint engine (or the dual-LLM split) up front, that changes the milestone shape — flag it before the spine is built.

## What's measured (the feedback loop)

A sealed **train/holdout** scenario corpus + the **autoresearch loop**: a stronger proposer reads *only* train traces, emits one atomic component mutation, an experiment runs it in a git-isolated worktree, and a promotion gate demands Welch p<0.05 on train **then** Bonferroni non-regression on holdout before reversible auto-adoption. The holdout is **never** executed during proposing.

**Known measurement gaps (honest):** no scenario yet exercises web-ingestion or injection. **J3 now HAS a bench path** — scored scenario `25_..._over_window_cruncher` (train slice) is tool-delivered + handle-only, so a mid-doc needle is reachable ONLY by routing to the cruncher (naive-read-fails by construction); it discriminated live (with cruncher pass / naive-baseline fail, no memory-recitation) and is wired through the SAME `make_explore_agent_runner` seam production uses. *Remaining gap:* it forces the cruncher via handle-only delivery, not raw over-window size (the body fits the 27B window) — a genuinely >window fixture is a named follow-up; and its discrimination depends on the model not having the public-domain text memorized. The autoresearch spine also still lives on a long-running branch, not `main`; that integration debt understates the project's real capability to anyone reading `main`.

## The proving workload

The **localshift morning report** is the real consumer: a cron-scheduled pipeline that imports this harness's agent loop, bash, and web tools to fetch and reason over live web pages the user did not hand-vet. It is why the security spine is a present-tense requirement, and it is the live red-team / dogfood target for J4 (injection reported as data, no host action, no bash reached) and J3 (lossless reduce over a real over-window pull).

## Non-goals

Web UI / dashboard · cloud-model integrations (blurs local-first) · graph orchestration · telemetry/usage reporting · Docker as a hard requirement · role-based agent definitions (token bloat) · **and** the full dual-LLM retriever split (deferred, above — not abandoned, but out of the current milestone with its residual named).

## Pointers

| For | Read |
|-----|------|
| Canonical component/architecture detail | [`docs/specs/`](specs/) |
| Model & hardware config (current: Qwen3.6-27B / vLLM / DGX Spark) | [`docs/reference-architectures/`](reference-architectures/) |
| Internal milestone state & PRDs (git-ignored, never committed) | `.planning/` |
