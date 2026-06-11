# Gaps — Out-of-Box Support for Both Reference Architectures

Development items required before [architecture A](dgx-spark.md) (DGX Spark) and
[architecture B](mac-mini.md) (base Mac mini) work with **zero per-agent overrides**.
Each item cites the harness default that conflicts. Numbering is stable — reference
these as "gaps §N" from other docs and issues.

## §1 Timeout math breaks on slow single-stream decode

**Conflict:** `provider/client.py` — `timeout_seconds: 300.0` (also enforced as
`LOCAL_INFERENCE_TIMEOUT_MIN`), `max_tokens: 4096`.

A full 4096-token completion takes **~431s at architecture A's measured 9.5 tok/s** —
the default timeout kills healthy generations. Architecture B (~17 tok/s) completes in
~237s but paging spikes after idle push past 300s.

- **Workaround (documented in both arch docs):** per-agent `timeout_seconds: 600` or `max_tokens: 2048`.
- **Fix:** derive the read timeout from `max_tokens / measured_decode_rate` (with floor),
  using a decode rate measured once by `init`/`doctor` and stored in provider config.

## §2 Context budget exceeds the served window

**Conflict:** `max_context_tokens: 128_000` default (`core/events.py:67`,
`orchestrator/cards.py:49`, `agent/context.py:346`).

Architecture A serves 64k (`--max-model-len 65536`); B serves 8–16k. Compaction triggers
at 80% of the *configured* budget (102k) — the server rejects oversized requests long
before the harness ever compacts. On B the default is ~8× the physical window.

- **Workaround:** per-agent `context.max_context_tokens` matching the server (65536 / 16384).
- **Fix:** probe the served context length at `init` (vLLM exposes `max_model_len` via
  `/v1/models`; llama.cpp exposes `n_ctx` via `/props`) and clamp the default budget to it.

## §3 Runtime coverage — vLLM + Ollama must work out of box

Project requirement: vLLM and Ollama are first-class on both machines. Current reality:

| Runtime | Arch A (Spark) | Arch B (base mini, 35B-A3B) |
|---------|----------------|------------------------------|
| vLLM | ✅ tier 1, tested | ❌ [vllm-metal](https://github.com/vllm-project/vllm-metal): Qwen3.6 MoE experimental, MLX-4bit/AWQ only (≈18–20 GB → needs 24 GB+ Mac), no mmap paging |
| Ollama | works (untested here) | ❌ loads model class fully resident (~26 GB) → swap freeze on 16 GB |
| llama.cpp `--mmap` | n/a | ✅ interim reference runtime (detected on `:8080`) |

- **Track upstream:** vllm-metal low-bit quant support; Ollama mmap/paged loading for MoE GGUFs.
- **Harness side:** `doctor` should warn when the detected runtime + machine RAM cannot
  hold the configured model resident (Ollama-on-16GB foot-gun).

## §4 No concurrency policy for single-stream budgets

The orchestrator can run multiple agents concurrently against one endpoint. At 9.5 tok/s
single-stream (A) every added stream stretches per-agent latency toward §1 timeouts;
llama.cpp (B) defaults to `--parallel 1` and queues, with the KV budget shared across slots.

- **Fix:** provider-level `max_concurrent_requests` (default 1 for B-class, small N for
  A-class) with orchestrator-side queueing instead of timeout pileups.

## §5 Tool-call reliability on Qwen 3.6

Qwen3.6-27B intermittently drifts from its tool-call format — stray closing tags
([QwenLM/Qwen3.6#178](https://github.com/QwenLM/Qwen3.6/issues/178)). 35B-A3B has a
published 67.3% BFCL v4 but is unvalidated through llama.cpp's hermes parser.

- **Fix:** harden the XML fallback parser (`provider/fn_call.py`) to tolerate stray/
  unbalanced tags; have `init`'s capability probe (`CapabilityResult`) record the
  per-architecture `tool_call_mode` plus a drift-tolerance flag; add bench scenarios
  that score tool-call format fidelity per architecture.

## §6 Thinking-token accounting

Qwen 3.6 preserves reasoning traces across turns (thinking preservation). Reasoning
tokens consume `max_tokens` and the context budget but are not separately tracked by the
token counter, and enable/disable flags differ per runtime (vLLM chat-template kwargs vs
llama.cpp `--jinja` template vars vs Ollama `enable_thinking`/`preserve_thinking`).

- **Fix:** count reasoning tokens in `agent/context.py` budgets; expose a per-provider
  thinking toggle in `LLMConfig`.

## §7 Context-hungry defaults on small windows

**Conflict:** `max_tool_output_chars: 32_000` (`config/models.py:326`) ≈ 8k tokens —
**half of architecture B's entire context** in a single tool observation. Similarly,
`preserve_first/last_n_messages` defaults assume a roomy window.

- **Fix:** scale tool-output and preservation defaults from `max_context_tokens`
  (e.g. tool output ≤ 12% of window) instead of absolute constants.

## §8 mmap paging variance (architecture B)

Cold experts page from NVMe after idle periods — first-token latency spikes that look
like a hung agent to the stuck-detector, and look like timeout material to §1.

- **Document:** internal-NVMe requirement; optional warmup request after model (re)load.
- **Fix:** per-architecture stuck-detector thresholds; treat slow-first-token as warmup,
  not stall.

## §9 Bench matrix assumes architecture A

`bench/bench.yaml` and `bench/orchestrator.py:79-80` default to vLLM at
`http://localhost:8000/v1`; regression thresholds were tuned against Spark-class
throughput. Architecture B runs ~2× the decode rate but ~1/4 the context and a different
tool-call parser.

- **Fix:** add a `llamacpp`/architecture-B bench profile (endpoint `:8080`, 16k-budget
  scenarios, adjusted latency thresholds) so harness changes are regression-tested
  against **both** reference architectures before merge.

---

## Priority order (suggested)

1. §2 context clamp (breaks every long session on both archs today)
2. §1 timeout derivation (kills healthy generations on A)
3. §5 tool-call hardening (agent-loop correctness)
4. §7 small-window scaling (B usability)
5. §3 runtime coverage warnings + upstream tracking
6. §4 concurrency policy
7. §9 bench profile for B
8. §6 thinking-token accounting
9. §8 paging-aware thresholds
