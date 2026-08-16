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

**Now:** `start` derives the effective window from the runtime's served `max_model_len`
minus the output reservation, so a 64k server is respected without per-agent YAML.
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
  one number that says whether speculation is helping on the current workload.
- **Fix:** first-class `speculative:` fields on the server config (draft model path, spec
  type, n-max, p-min) with validation, and surface llama.cpp's draft-acceptance telemetry
  in `doctor`.

---

## Priority order (suggested)

1. §5 tool-call hardening (agent-loop correctness)
2. §10 multimodal input path (the default reference model's headline capability is
   unreachable)
3. §3 runtime parity validation + doctor RAM-fit warning
4. §11 speculative-decoding validation + acceptance telemetry
5. §1 timeout derivation (constant works today; re-opens silently on slower models)
6. §4 configurable concurrency level
7. §9 bench profile for B
8. §6 thinking-token accounting
9. §8 prefill-aware thresholds
10. §7 window-scaled defaults

§2 is resolved and kept for its residual note.
