# Reference Architecture B — Base Mac mini

**Status: PROPOSED** (June 2026). Config meets the [practicality bar](README.md#practicality-bar)
on paper; performance numbers are estimates until the [validation checklist](#validation-checklist)
is executed on the target machine. The decode-rate gate (≥9.5 tok/s at depth) is the
promotion criterion to TESTED.

## Summary

| | |
|---|---|
| Hardware | Apple Mac mini, **base configuration** — M4, 16 GB unified, 120 GB/s, NVMe SSD ($799, May-2026 lineup) |
| Model | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) — dense 9B, hybrid attention, Apache 2.0 |
| Quantization | MLX 4-bit ([`mlx-community/Qwen3.5-9B-MLX-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)) or GGUF `Q4_K_M` (5.68 GB, [`unsloth/Qwen3.5-9B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)) |
| Runtime | **vLLM via [vllm-metal](https://github.com/vllm-project/vllm-metal)**, OpenAI API on `:8000` — same runtime and port as architecture A |
| Context served | **64k** (`--max-model-len 65536`); KV @ 64k ≈ **2.1 GB** |
| Decode estimate | 22–28 tok/s short-context (published M4 figures); est. 10–15 tok/s at 64k depth — **validate ≥9.5** |

## Why Qwen3.5-9B — the bar math

The practicality bar requires **weights + 64k of KV headroom + macOS inside 16 GB, at
≥9.5 tok/s single-stream**. KV cost per token is set by the attention architecture:
hybrid-attention generations (Qwen 3.5/3.6) keep KV on only ¼ of layers; classic
full-attention models pay on every layer.

| Candidate | Weights (4-bit) | KV @ 64k | Fits + ≥9.5 tok/s on 16 GB / 120 GB/s? |
|-----------|-----------------|----------|-----------------------------------------|
| Qwen3.6-27B dense | 16.8 GB | ~4.2 GB (64 KB/tok) | ❌ weights alone exceed RAM; at Q3 (~12 GB) decode ~7–10 tok/s and no KV headroom |
| Qwen3.6-35B-A3B MoE | 12.3 GB (IQ3_XXS) | ~1.3 GB (20 KB/tok) | ❌ resident only via mmap expert-paging; maintainer-assessed impractical for sustained 64k sessions on 16 GB |
| Qwen3-14B dense (older family) | 8.5 GB | **~10.5 GB (160 KB/tok)** | ❌ classic full attention — 64k KV alone consumes the headroom |
| **Qwen3.5-9B dense** | **5.7 GB** | **~2.1 GB (32 KB/tok)** | ✅ ~11 GB total incl. macOS; ~5 GB spare |

Qwen3.5-9B KV math (from its `config.json`): 32 layers in a 3:1 linear:full pattern → 8
full-attention layers × 4 KV heads × 256 head-dim × K+V × FP16 = **32 KB/token** →
2.1 GB at 65,536 tokens. Native context 262k; the GDN linear layers carry constant-size
state (~tens of MB) regardless of depth.

**Family-policy note:** Qwen 3.6 ships no dense smaller than 27B (verified June 2026 —
collection is 27B + 35B-A3B only). Architecture B therefore runs the **newest dense
model that meets the bar**. When a small Qwen3.6 dense releases, this doc revises to it.

### Memory budget (16 GB) @ 64k

| Component | Resident |
|-----------|----------|
| Weights (4-bit) | ~5.7 GB |
| KV cache @ 64k FP16 | ~2.1 GB |
| Runtime buffers | ~0.5–1 GB |
| macOS floor | ~2.5–3 GB |
| **Headroom** | **~4–5 GB** |

No mmap tricks required — the model is fully resident. Optional: raise the Metal wired
limit for extra margin on a headless box
(`sudo sysctl iogpu.wired_limit_mb=13312`, [reference](https://pixelesque.net/other/snippets/macos/changing-mac-unified-memory-gpu-limit/)).

## Serving

### Primary: vLLM (vllm-metal) — runtime parity with architecture A

[vllm-metal](https://github.com/vllm-project/vllm-metal) (v0.2.0+, MLX compute backend,
paged KV) supports Qwen3.5 with MLX 4-bit checkpoints. Requires native arm64 Python 3.12.

```bash
vllm serve mlx-community/Qwen3.5-9B-MLX-4bit \
  --max-model-len 65536 \
  --port 8000
```

Same OpenAI API, port, and detector path as the DGX Spark — agent YAML is identical
across both architectures except for the context/timeout profile below.

### Alternatives (both fit resident — all three runtimes work on this model)

- **Ollama** (`:11434`): pull the qwen3.5 9B library model; MLX backend on Apple
  Silicon since v0.19.
- **llama.cpp** (`:8080`): `llama-server --model Qwen3.5-9B-Q4_K_M.gguf --ctx-size 65536 --jinja`
  (GGUF is text-only; the vision mmproj file is separate and not needed for the harness).

Qwen-recommended sampling everywhere: `temperature 0.6, top_p 0.95, top_k 20`.

## Performance expectations

| Metric | Estimate | Basis |
|--------|----------|-------|
| Decode, short context | 22–28 tok/s | published M4 benchmarks for this model/quant class |
| Decode @ 64k depth | 10–15 tok/s | per-token reads grow from ~5.7 GB to ~7.8 GB (weights + full-attn KV) on 120 GB/s |
| **Bar** | **≥9.5 tok/s @ 64k** | promotion gate — measure, don't assume |

First-token latency on a deep prompt is compute-bound prefill and can be substantial at
64k; prefix caching (default in vllm-metal and llama-server) makes it a once-per-session
cost in agent loops. See [gaps.md](gaps.md) §8.

## Harness configuration

`localharness init` detects vLLM on `:8081`/`:8000` (or the alternatives on their ports)
automatically. Per-agent overrides required today:

```yaml
# agent YAML — architecture B profile
timeout_seconds: 600.0        # 4096 max_tokens at ~10-15 tok/s ≈ 273-410s; default 300s too tight
max_tokens: 2048
context:
  max_context_tokens: 65536   # match --max-model-len; harness default is 128k
```

## Validation checklist

Run on the actual base mini; promotion to TESTED requires the first two:

- [ ] **Decode ≥9.5 tok/s single-stream at 48–64k prompt depth** (the bar)
- [ ] **64k session with zero swap** (`memory_pressure` nominal, `vm_stat` swapouts = 0)
- [ ] TTFT at 8k/32k/64k cold vs prefix-cached (informs stuck-detector thresholds, gaps §8)
- [ ] Tool-call success rate through vllm-metal's parser vs llama.cpp `--jinja` hermes, against `bench/` scenarios (gaps §5 — no published BFCL for this model)
- [ ] Thinking-mode token accounting (reasoning traces vs `max_tokens`, gaps §6)
- [ ] KV q8_0 variant (llama.cpp `--cache-type-k/v q8_0`): quality delta vs ~1 GB savings
- [ ] Runtime parity: same agent YAML against vllm-metal / Ollama / llama.cpp

**If decode misses the bar at depth:** try GGUF/llama.cpp vs MLX/vllm-metal (backend
throughput differs), then KV q8_0 (cuts attention reads), then `UD-Q3_K_XL` (5.05 GB)
weights. If all miss, the finding is documented here and the hardware floor for this
architecture moves to the 24 GB mini.
