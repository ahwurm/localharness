# Setup & Tuning Notes — Reference Architectures

Per-hardware tuning items to run [architecture A](dgx-spark.md) (DGX Spark) and
[architecture B](mac-mini.md) (base Mac mini) with **zero per-agent overrides**. Each item
names the harness default it adjusts and the one-line workaround. Numbering is stable —
other docs reference these as "gaps §N".

## §1 Timeout math breaks on slow single-stream decode

**Conflict:** `provider/client.py` — `timeout_seconds: 300.0` (also enforced as
`LOCAL_INFERENCE_TIMEOUT_MIN`), `max_tokens: 4096`.

A full 4096-token completion takes **~431s at architecture A's measured 9.5 tok/s**, so
the default 300s timeout can cut a healthy generation off before it finishes. Architecture B
at 64k depth (est. 10–15 tok/s) lands at 273–410s: the same adjustment applies.

- **Workaround (documented in both arch docs):** per-agent `timeout_seconds: 600` or `max_tokens: 2048`.
- **Fix:** derive the read timeout from `max_tokens / measured_decode_rate` (with floor),
  using a decode rate measured once by `init`/`doctor` and stored in provider config.

## §2 Context budget exceeds the served window

**Conflict:** `max_context_tokens: 128_000` default (`core/events.py:67`,
`orchestrator/cards.py:49`, `agent/context.py:346`).

Both architectures serve 64k (`--max-model-len 65536`). Compaction triggers at 80% of
the *configured* budget (102k with the default) — the server rejects oversized requests
long before the harness ever compacts.

- **Workaround:** per-agent `context.max_context_tokens: 65536`.
- **Fix:** probe the served context length at `init` (vLLM exposes `max_model_len` via
  `/v1/models`; llama.cpp exposes `n_ctx` via `/props`) and clamp the default budget to it.

## §3 Runtime coverage — vLLM + Ollama must work out of box

Project requirement: vLLM and Ollama are first-class on both machines. Current state:

| Runtime | Arch A (Spark) | Arch B (base mini, Qwen3.5-9B) |
|---------|----------------|--------------------------------|
| vLLM | ✅ tier 1, tested | ✅ via [vllm-metal](https://github.com/vllm-project/vllm-metal) — feature parity (tool parser, quant formats) unvalidated |
| Ollama | works (untested here) | ✅ model fits resident — parity unvalidated |
| llama.cpp `--mmap` | n/a | ✅ third option (`:8080`) |

- **Validate:** vllm-metal tool-call parsing and sampling parity with CUDA vLLM (same
  agent YAML must behave identically on both architectures).
- **Harness side:** `doctor` should warn when the configured model cannot sit resident
  in machine RAM. (Lesson from the rejected 35B-A3B-on-16GB config: Ollama loaded it
  fully resident — ~26 GB — and swap-froze the machine.)

## §4 No concurrency policy for single-stream budgets

The orchestrator can run multiple agents concurrently against one endpoint. At ~10 tok/s
single-stream every added stream stretches per-agent latency toward §1 timeouts; on B
the KV budget (2.1 GB @ 64k) is also per-slot.

- **Fix:** provider-level `max_concurrent_requests` (default 1 on B-class, small N on
  A-class) with orchestrator-side queueing instead of timeout pileups.

## §5 Tool-call reliability on current Qwen models

Architecture A's Qwen3.6-27B intermittently drifts from its tool-call format — stray
closing tags ([QwenLM/Qwen3.6#178](https://github.com/QwenLM/Qwen3.6/issues/178)).
Architecture B's Qwen3.5-9B has **no published BFCL score** and is unvalidated through
both vllm-metal's parser and llama.cpp's hermes parser.

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

---

## Priority order (suggested)

1. §2 context clamp (breaks every long session on both archs today)
2. §1 timeout derivation (kills healthy generations on both)
3. §5 tool-call hardening (agent-loop correctness)
4. §3 runtime parity validation + doctor RAM-fit warning
5. §4 concurrency policy
6. §9 bench profile for B
7. §6 thinking-token accounting
8. §8 prefill-aware thresholds
9. §7 window-scaled defaults
