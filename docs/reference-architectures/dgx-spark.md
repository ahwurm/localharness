# Reference Architecture A — NVIDIA DGX Spark

**Status: TESTED** (maintainer hardware, June 2026). Decode and KV figures below are
measured on this exact stack; treat them as ground truth for harness defaults.

## Summary

| | |
|---|---|
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, SM 12.1 |
| Memory | 128 GB LPDDR5x unified (119 GiB usable), 273 GB/s |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| Model | [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) — dense 27.8B, Apache 2.0 |
| Quantization | **NVFP4** (~15.6 GB weights) |
| Runtime | vLLM (NVIDIA ARM64 container), OpenAI API on `:8000` |
| Context served | **64k** (`--max-model-len 65536`) |
| Measured decode | **9.5 tok/s single-stream** |

## Model

Qwen3.6-27B (released 2026-04-22) is the dense flagship of the latest Qwen family:

- 262,144-token native context; hybrid attention (Gated DeltaNet linear attention 3:1
  with gated self-attention) keeps KV cost low — ~64 KB/token FP16, ≈ 4.2 GB at 64k.
- Strongest agentic-coding model that fits this hardware: 77.2% SWE-bench Verified
  (see [CONTEXT-MODEL.md](../../CONTEXT-MODEL.md) candidate table).
- Thinking preservation: retains reasoning traces across turns (see
  [gaps.md](gaps.md) §6 for token-accounting implications).
- **No published BFCL tool-calling score**, and a known tool-call format-drift issue
  ([QwenLM/Qwen3.6#178](https://github.com/QwenLM/Qwen3.6/issues/178)): intermittently
  emits stray closing tags around tool calls. The harness XML fallback
  (`provider/fn_call.py`) must tolerate this — [gaps.md](gaps.md) §5.

### NVFP4 checkpoint

Alibaba publishes BF16 and FP8 ([`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8))
but **no official NVFP4 checkpoint**. The tested checkpoint is produced locally with
NVIDIA TensorRT Model Optimizer (ModelOpt) PTQ from the BF16 weights, then served by vLLM.

NVFP4-on-Spark caveats (state of June 2026):

- vLLM supports NVFP4 from v0.25+, but SM 12.x (consumer/Spark Blackwell) kernels
  lag SM 10.x datacenter parts — see the
  [NVIDIA forum PSA](https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069).
- Community results are split: some report AWQ-4bit faster than NVFP4 on Spark; others
  [report NVFP4 ~20% faster](https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83)
  with current kernels. The 9.5 tok/s figure below is what this NVFP4 stack measures
  today; re-benchmark on vLLM upgrades.

## Serving

Follow the [NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)
vLLM recipe (ARM64 container). Tested invocation shape:

```bash
vllm serve <path-to-local-nvfp4-checkpoint> \
  --quantization modelopt_fp4 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Qwen-recommended sampling (also harness defaults): `temperature 0.6, top_p 0.95, top_k 20`.

### Memory budget (119 GiB)

| Component | Size |
|-----------|------|
| Weights (NVFP4) | ~15.6 GB |
| KV cache @ 64k | ~4.2 GB |
| Runtime overhead | a few GB |
| **Headroom** | **~95 GB** — room for secondary models (embeddings, vision) or higher concurrency |

## Measured performance

| Metric | Value |
|--------|-------|
| Decode, single stream | **9.5 tok/s** |
| KV cache | 64k, no eviction issues |

vLLM batching raises aggregate throughput with concurrent requests, but **per-stream
latency is what governs an agent loop** — size harness timeouts from the single-stream
figure ([gaps.md](gaps.md) §1).

## Harness configuration

`localharness init` detects vLLM on `:8000` automatically. Per-agent overrides required
today (until the gaps below are closed):

```yaml
# agent YAML — architecture A profile
model: inherit            # Qwen3.6-27B NVFP4 via vLLM
timeout_seconds: 600.0    # default 300s < 4096 tokens / 9.5 tok/s ≈ 431s
max_tokens: 2048          # or keep 4096 with the 600s timeout
context:
  max_context_tokens: 65536   # MUST match --max-model-len; harness default is 128k
```

Why: harness defaults (`provider/client.py`: `timeout_seconds=300`, `max_tokens=4096`;
`core/events.py`: `max_context_tokens=128_000`) were set before this architecture was
measured. See [gaps.md](gaps.md) §1–2 for the out-of-box fixes.

## Known issues

1. Tool-call format drift (stray closing tags) — [gaps.md](gaps.md) §5.
2. Default timeout math breaks at 9.5 tok/s — [gaps.md](gaps.md) §1.
3. Harness 128k context default exceeds the 64k served window — [gaps.md](gaps.md) §2.
4. NVFP4 kernel maturity on SM 12.x — re-validate decode rate on each vLLM upgrade.
