# Setup & Tuning Notes — Reference Architectures

Per-hardware tuning items to run [architecture A](dgx-spark.md) (DGX Spark) and
[architecture B](mac-mini.md) (base Mac mini) with **zero per-agent overrides**. Each item
names the harness default it adjusts and the one-line workaround. Numbering is stable —
other docs reference these as "gaps §N".

## §1 Timeout math on slow single-stream decode — MOSTLY ADDRESSED

**Then:** the provider default was `timeout_seconds: 300.0` against `max_tokens: 4096`. A
full completion at architecture A's slowest measured config (9.5 tok/s) takes ~431s, so
the default could cut a healthy generation off before it finished.

**Now:** the default is `timeout_seconds: 600.0` (`ProviderConfig`, `config/defaults.py`),
which covers 431s with margin, and every measured configuration on both architectures.
No override is needed today.

- **Still open:** the timeout is a fixed constant, not derived. A slower future model or a
  larger `max_tokens` re-opens the same hole silently.
- **Fix:** derive the read timeout from `max_tokens / measured_decode_rate` (with a floor),
  using a decode rate measured once by `init`/`doctor` and stored in provider config.

## §2 Context budget exceeds the served window — RESOLVED

**Was:** the `max_context_tokens` default (now `131_072`, `config/defaults.py`) exceeded a
64k served window, so compaction triggered at 80% of a budget the server would already
have rejected requests against.

**Now:** `init` writes the runtime's served `max_model_len` and `start` validates the budget
against it, so a 64k server is respected without per-agent YAML. (Room for the model's reply is
held back inside that window at runtime by `agent.context.response_reserve`, not subtracted from
the configured value — see #145.)
`context.max_context_tokens` remains available as an explicit **cap** — use it to request a
budget smaller than the server offers, never larger.

- **Residual:** the config value is still a plain number with no validation against the
  served window, so setting it *above* what the server can do is accepted at load time and
  only fails later, at the runtime.

## §3 Runtime coverage — vLLM + Ollama must work out of box

Project requirement: vLLM and Ollama are first-class on both machines. As of 0.11 all four
runtimes have a harness-managed lifecycle and a live round-trip on architecture A — the full
per-dimension state is the [provider support matrix](../../README.md#supported-runtimes). This
row tracks HARDWARE fit on each target:

| Runtime | Arch A (Spark) | Arch B (base mini, Qwen3.5-9B) |
|---------|----------------|--------------------------------|
| vLLM | ✅ tier 1, tested | ✅ via [vllm-metal](https://github.com/vllm-project/vllm-metal) — feature parity (tool parser, quant formats) unvalidated |
| Ollama | ✅ lifecycle validated (CPU round-trip, zero orphans) | ✅ model fits resident — parity unvalidated |
| llama.cpp | ✅ tier 1 — the runtime for arch A's default config (A1) and for DeepSeek V4 Flash (A3); spawn + live cross-framework swap validated | ✅ third option (`:8080`) |
| LM Studio | ✅ lifecycle validated (CPU round-trip) | untested here |

- **Validate:** vllm-metal tool-call parsing and sampling parity with CUDA vLLM (same
  agent YAML must behave identically on both architectures).
- **Harness side:** `doctor` should warn when the configured model cannot sit resident
  in machine RAM. (Lesson from the rejected 35B-A3B-on-16GB config: Ollama loaded it
  fully resident — ~26 GB — and swap-froze the machine.)

## §4 No concurrency policy for single-stream budgets

The orchestrator can run multiple agents concurrently against one endpoint. At ~10 tok/s
single-stream every added stream stretches per-agent latency toward §1 timeouts; on B
the KV budget (2.1 GB @ 64k) is also per-slot.

**Partially addressed:** GPU access is already serialized (an in-process semaphore plus a
cross-process lock), and `provider.inference_queue_wait_seconds` (default 600.0) bounds how
long a request waits for its turn. So requests queue rather than trampling each other.

- **Still open:** there is no configurable concurrency *level* — no
  `max_concurrent_requests` field exists. The policy is fixed serialization, which leaves
  throughput on the table on A-class hardware that could serve a small N in parallel.
- **Fix:** provider-level concurrency setting (1 on B-class, small N on A-class) with
  orchestrator-side queueing.

## §5 Tool-call reliability on current Qwen models

Architecture A's Qwen3.6-27B intermittently drifts from its tool-call format — stray
closing tags ([QwenLM/Qwen3.6#178](https://github.com/QwenLM/Qwen3.6/issues/178)).
Architecture B's Qwen3.5-9B has **no published BFCL score** and is unvalidated through
both vllm-metal's parser and llama.cpp's hermes parser.

Architecture A's current default (Qwen3.8-27B, config A1) routes tool calls through
llama.cpp's `--jinja` path, which uses the model's own chat template rather than a
hand-written parser. That is the right mechanism, but its format fidelity on this model
has **not** been scored against `bench/` yet — the same measurement owed for the models
above.

**Live evidence, short of that score:** the August 2026 live-use test pass recorded **zero
tool-call parse failures** across every session, on both the llama.cpp `--jinja` configs
(A1, A3) and vLLM native tool-calling (A2). That is encouraging but it is not the owed
measurement — it is an absence of observed failures over a set of hand-driven sessions, with
no fixed scenario set, no failure-rate denominator, and no coverage of Qwen3.6-27B (A4), the
model the drift issue was actually filed against. The bench scenarios below are still owed.

- **Fix:** harden the XML fallback parser (`provider/fn_call.py`) to tolerate stray/
  unbalanced tags; have `init`'s capability probe (`CapabilityResult`) record the
  per-architecture `tool_call_mode` plus a drift-tolerance flag; add bench scenarios
  that score tool-call format fidelity per architecture.

## §6 Thinking-token accounting

Qwen 3.5/3.6 preserve reasoning traces across turns (thinking preservation). Reasoning
tokens consume `max_tokens` and the context budget but are not separately tracked by the
token counter, and enable/disable flags differ per runtime (vLLM chat-template kwargs vs
llama.cpp `--jinja` template vars vs Ollama `enable_thinking`/`preserve_thinking`).

- **Fix:** count reasoning tokens in `agent/context.py` budgets; expose a per-provider
  thinking toggle in `LLMConfig`.

## §7 Context-hungry defaults don't scale with the window

**Conflict:** `max_tool_output_chars: 32_000` (`config/models.py:326`) ≈ 8k tokens —
12.5% of a 64k window per tool observation, and proportionally worse if an architecture
ever serves less. `preserve_first/last_n_messages` defaults similarly assume a roomy window.

- **Fix:** scale tool-output and preservation defaults from `max_context_tokens`
  (e.g. tool output ≤ 10% of window) instead of absolute constants.

## §8 Deep-context prefill latency (architecture B)

Prefill at 48–64k prompt depth is compute-bound on the M4 and can take minutes cold —
indistinguishable from a hang to the stuck-detector, and timeout material under §1.
Prefix caching (default in vllm-metal and llama-server) makes it once-per-session, but
the first deep turn and any cache eviction still pay it.

- **Document:** measure TTFT at 8k/32k/64k in the B validation checklist.
- **Fix:** per-architecture stuck-detector thresholds; treat slow first token at depth
  as prefill, not stall; surface prefill progress where the runtime exposes it.

## §9 Bench matrix assumes architecture A

`bench/bench.yaml` and `bench/orchestrator.py:79-80` default to vLLM at
`http://localhost:8000/v1`; regression thresholds were tuned against Spark-class
serving. Architecture B answers on the same port/protocol but with a different decode
profile and backend (vllm-metal).

- **Fix:** add an architecture-B bench profile (64k-budget scenarios, B-class latency
  thresholds, optional llama.cpp `:8080` endpoint variant) so harness changes are
  regression-tested against **both** reference architectures before merge.

## §10 No multimodal input path

Architecture A's default model (Qwen3.8-27B, config A1) is natively multimodal — its
checkpoint ships both image and video preprocessor configs. The harness has **no way to
send an image**: there is no multimodal field anywhere in the config schema, and the
message path assembles text content only.

- **Consequence:** the reference architecture's headline capability is unreachable from the
  harness today. Screenshots, diagrams and UI state all have to be described in words.
- **Fix:** accept image parts in the OpenAI-format content array, add a tool-result shape
  that can carry an image, and decide how images are counted against the context budget
  (they are not free — they expand to a large number of tokens).

## §11 Speculative-decoding setup is opaque to the harness

Both llama.cpp configs on architecture A (A1, A3) depend on speculative decoding, which
roughly doubles their decode rate. The harness has no schema for it: draft model, spec
type and the tuning flags can only be passed as raw strings through
`server.extra_args`.

- **Consequence:** nothing validates them. A typo in `--spec-type` or a missing `-ngld 99`
  silently costs most of the speedup, and `doctor` cannot tell the user their expensive
  drafter is sitting on the CPU. Nothing surfaces the acceptance rate either, which is the
  one number that says whether speculation is helping on the current workload. The August
  2026 pass demonstrated the cost concretely: A3's acceptance varied 0.43–0.80 across
  sessions — the whole explanation for its ~24–47 tok/s live spread — and reading it meant
  tailing `llama-server`'s own log, because the harness surfaces nothing.
- **Fix:** first-class `speculative:` fields on the server config (draft model path, spec
  type, n-max, p-min) with validation, and surface llama.cpp's draft-acceptance telemetry
  in `doctor`.

## §12 What the August 2026 live-use test pass did *not* cover

That pass is the freshest validation behind [dgx-spark.md](dgx-spark.md): real agent
sessions driven through the harness REPL on architecture A, on 0.12.5 code. It is worth
being equally explicit about its edges, because "recently validated" is easy to read as
"validated everywhere". Four holes:

- **A4 (Qwen3.6-27B dense, vLLM) was never brought up.** Its June 2026 numbers — 9.5 tok/s
  single stream — stand unrevalidated against current vLLM builds and the current harness,
  and so does the tool-call drift in §5, which was filed against *that* model.
  **Fix:** re-run A4 and either refresh its numbers or demote it out of the tested set.

- **Cross-session memory continuity was not exercised.** Every session in the pass started
  clean. Restarting the harness against an **existing** memory home — the case where
  recall, activation scoring and consolidation have to survive a process boundary and
  actually earn their keep — was not tested live at all. This is the load-bearing promise of
  the memory subsystem, so an untested restart path is the most consequential hole here.
  **Fix:** a live continuity scenario — seed a home, restart, verify prior-session facts are
  retrieved and scored — plus a `bench/` scenario that fails when they are not.

- **Compaction was observed firing live exactly once (n=1).** That one firing was clean:
  context at 106% of budget dropped to 40% after a restore-heavy turn, on 0.12.5 code. But
  eleven other sessions never crossed the threshold at all, because eviction reaches
  equilibrium first and holds doc-heavy sessions around 62–65%. So the honest state is:
  eviction is well-exercised, **compaction is not** — one clean observation is an anecdote,
  not a validated path, and the workload that reliably reaches it (restore-heavy turns) is
  now the known way to provoke it.
  **Fix:** a deterministic compaction test that forces the threshold rather than waiting for
  a session to drift there, so the path gets exercised every run instead of once a month.

- **Ollama and LM Studio were not exercised in the pass at all**, and one 0.12.5 change
  makes that gap sharper than it looks. The speed ledger now admits only substantive samples
  (≥16 completion tokens over a ≥0.25 s window) on **client-measured** runtimes; llama.cpp
  is exempt because it reports engine-side timings. Ollama and LM Studio are both on the
  client-measured path, so whether real turns on them clear that floor often enough to keep
  a live readout populated is **unverified** — the gate is correct-by-construction but its
  practical sample yield on those two runtimes has never been watched live.
  **Fix:** drive a live session on each and confirm the ledger populates; if sub-threshold
  turns dominate, the floor needs a per-runtime answer rather than one constant.

---

## Priority order (suggested)

1. §12 cross-session memory continuity + a forced compaction test (untested paths beat
   under-tuned ones — and these two are the memory subsystem's actual promise)
2. §5 tool-call hardening (agent-loop correctness)
3. §10 multimodal input path (the default reference model's headline capability is
   unreachable)
4. §3 runtime parity validation + doctor RAM-fit warning
5. §11 speculative-decoding validation + acceptance telemetry
6. §1 timeout derivation (constant works today; re-opens silently on slower models)
7. §4 configurable concurrency level
8. §9 bench profile for B
9. §6 thinking-token accounting
10. §8 prefill-aware thresholds
11. §7 window-scaled defaults

§12's other two items ride along cheaply: A4 re-validation belongs to the next vLLM upgrade
(§3), and the Ollama/LM Studio ledger check belongs to the same runtime-parity pass.

§2 is resolved and kept for its residual note.
