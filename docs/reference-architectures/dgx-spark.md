# Reference Architecture A — NVIDIA DGX Spark

**Status: TESTED** (maintainer hardware). The recommended Qwen3.6-35B-A3B config below
was measured July 2026; the Qwen3.6-27B alternative was measured June 2026 and remains
documented as-is. Figures in each section are measured on this exact stack; treat them
as ground truth for harness defaults.

## Summary

| | |
|---|---|
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, SM 12.1 |
| Memory | 128 GB LPDDR5x unified (119 GiB usable), 273 GB/s |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |
| **Recommended model** | [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) — MoE, 35B total / ~3B active |
| Quantization | **NVFP4** (~22 GB weights) |
| Runtime | vLLM (`vllm/vllm-openai:nightly`, ≥0.22.1), OpenAI API on `:8000` |
| Context served | **128k** (131,072 tokens exactly — `--max-model-len 131072`) |
| Measured decode | **~78 tok/s single-stream** |
| Alternative | Qwen3.6-27B dense, ~15.6 GB, ~9.5 tok/s single-stream — see below |

## Recommended model: Qwen3.6-35B-A3B (NVFP4)

### Model

Qwen3.6-35B-A3B is the mixture-of-experts member of the Qwen 3.6 family: 35B total
parameters, ~3B active per forward pass (the "A3B" suffix is Qwen's own naming for
"active 3B"). NVIDIA publishes an official NVFP4 checkpoint for it —
[`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
(~22 GB weights) — so, unlike the 27B dense model below, no local quantization step is
needed.

### Serving

Tested recipe:

```bash
docker run -d --name vllm-qwen36-35b --gpus all --restart no --ipc=host \
  -p 8000:8000 \
  -v <MODEL_DIR>:/models/serving:ro \
  vllm/vllm-openai:nightly \
  --model /models/serving --host 0.0.0.0 --port 8000 \
  --quantization modelopt --moe-backend marlin --attention-backend flashinfer \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.5 \
  --max-model-len 131072 --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --served-model-name qwen3.6-35b-a3b --enable-prefix-caching
```

Image: `vllm/vllm-openai:nightly`, tested at digest
`sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9`. **vLLM ≥0.22.1
is required** — older builds cannot load this checkpoint's NVFP4 MoE weights.

Two flags in that recipe are non-obvious and worth calling out:

- `--max-num-batched-tokens 4096` — **required**, not just a tuning knob. This
  checkpoint's hybrid Mamba/GDN layers assert `block_size` (2096) ≤ batched tokens;
  vLLM's own default of 2048 sits below that and crashes at startup.
- `--reasoning-parser qwen3` — routes chain-of-thought into the response's `reasoning`
  field so `content` arrives clean. Without it, thinking text pollutes the answer.

### Measured performance

Method: temperature 0, concurrency 1 (single stream), medians over repeated reps,
warmup run excluded, prefix cache defeated with nonces so every request is a genuine
cache miss.

| Metric | Value |
|--------|-------|
| Decode throughput (steady state) | **~78 tok/s** (median 77.97 tok/s over 5×256-token reps; 512-token group 78.3 tok/s) |
| Time to first token | ~0.09 s |
| Time to first *answer* token (reasoning on, default) | ~7.1 s — the model spends ~600 hidden reasoning tokens before the first visible answer token |
| Prefill | ~0.8 s at ~1k prompt tokens; ~1.0–1.3 s at ~4k (first request at a new depth is slower — warmup effect) |
| Context window | 131,072 tokens (`--max-model-len 131072`) |

Caveats:

- With reasoning on, small `max_tokens` budgets (≤512) can be consumed entirely by
  hidden reasoning, returning empty `content` with `finish_reason: length`. Set
  per-request `chat_template_kwargs: {"enable_thinking": false}` when a direct answer
  (no reasoning) is what you want instead.
- Memory-footprint telemetry is unreliable on GB10's unified-memory architecture —
  `docker stats` undercounts and `nvidia-smi` memory counters return `N/A`. We
  deliberately don't publish a footprint number for this config; the ~22 GB weight size
  above is the only hard number we have.
- Rough bandwidth sanity check: ~3B active params at 4-bit against 273 GB/s of memory
  bandwidth implies a naive ceiling around 180 tok/s. Measured ~78 tok/s is ~43% of
  that — plausible for batch-1 decode once attention and KV overhead are counted in.

For orientation: the 27B dense NVFP4 alternative documented below decodes roughly
10–15 tok/s on this hardware by informal observation — not benchmarked to the same
standard as the numbers above. (That section's own Measured performance table has the
rigorous single-stream figure this doc otherwise treats as ground truth.)

## Alternative: Qwen3.6-27B (dense, NVFP4)

Qwen3.6-27B was the original recommended model for this hardware (tested June 2026). It
remains fully supported and documented below as a smaller-footprint alternative
(~15.6 GB weights vs. ~22 GB for the 35B-A3B recipe above).

### Model

Qwen3.6-27B (released 2026-04-22) is the dense flagship of the latest Qwen family:

- 262,144-token native context; hybrid attention (Gated DeltaNet linear attention 3:1
  with gated self-attention) keeps KV cost low — ~64 KB/token FP16, ≈ 4.2 GB at 64k.
- Strongest agentic-coding model that fits this hardware: 77.2% SWE-bench Verified.
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

### Serving

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

### Measured performance

| Metric | Value |
|--------|-------|
| Decode, single stream | **9.5 tok/s** |
| KV cache | 64k, no eviction issues |

vLLM batching raises aggregate throughput with concurrent requests, but **per-stream
latency is what governs an agent loop** — size harness timeouts from the single-stream
figure ([gaps.md](gaps.md) §1).

## Harness configuration

`localharness init` detects vLLM on `:8081`/`:8000` automatically; with no server
running, its guided setup uses the container route — Docker and the NVIDIA container
toolkit must already be installed — pulling the vLLM image, downloading a checkpoint, and
launching on `:8081` (this architecture).

With the recommended 35B-A3B recipe, **no per-agent overrides are required** — harness
defaults already fit. The served context (131,072 tokens) comfortably covers the
harness's default `max_context_tokens` (128,000), and the default `max_tokens` (4096) at
~78 tok/s decodes in ~53s, well inside the default 300s `timeout_seconds`.

Running the 27B alternative instead still needs the overrides below (unchanged from
when it was the sole reference config):

```yaml
# agent YAML — architecture A profile, 27B alternative
model: inherit            # Qwen3.6-27B NVFP4 via vLLM
timeout_seconds: 600.0    # default 300s < 4096 tokens / 9.5 tok/s ≈ 431s
max_tokens: 2048          # or keep 4096 with the 600s timeout
context:
  max_context_tokens: 65536   # MUST match --max-model-len; harness default is 128k
```

Why: harness defaults (`provider/client.py`: `timeout_seconds=300`, `max_tokens=4096`;
`core/events.py`: `max_context_tokens=128_000`) were set before either config on this
architecture was measured. See [gaps.md](gaps.md) §1–2 for the out-of-box fixes.

## Known issues

The items below are specific to the 27B alternative unless noted otherwise.

1. Tool-call format drift (stray closing tags) on Qwen3.6-27B — [gaps.md](gaps.md) §5.
2. Default timeout math breaks at 9.5 tok/s (27B only — the 35B-A3B recipe's harness
   defaults need no override; see Harness configuration above) — [gaps.md](gaps.md) §1.
3. Harness 128k context default exceeds the 27B config's 64k served window (27B only,
   for the same reason) — [gaps.md](gaps.md) §2.
4. NVFP4 kernel maturity on SM 12.x — re-validate decode rate on each vLLM upgrade. This
   concern was raised against the community PTQ 27B checkpoint on then-current vLLM
   kernels; the 35B-A3B recipe above is separately measured on a specific `nightly`
   digest and a pinned vLLM ≥0.22.1 floor, and should be re-benchmarked the same way on
   future vLLM upgrades.
