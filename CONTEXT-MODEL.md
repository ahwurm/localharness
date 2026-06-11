# Context: Model & Runtime Selection

> **⚠️ SUPERSEDED (2026-06-11)** by [docs/reference-architectures/](docs/reference-architectures/README.md).
> Current tested config: **Qwen3.6-27B / NVFP4 / vLLM @ 9.5 tok/s, 64k KV cache** on DGX Spark
> ([dgx-spark.md](docs/reference-architectures/dgx-spark.md)), plus a base Mac mini architecture
> running **Qwen3.6-35B-A3B / llama.cpp `--mmap`** ([mac-mini.md](docs/reference-architectures/mac-mini.md)).
> The analysis below (Qwen3.5-122B-A10B selection) is retained as historical evaluation data —
> its candidate table and runtime comparison are still referenced by the current docs.

## Decision

**Primary Model:** Qwen3.5-122B-A10B (MoE, 122B total / 10B active per token)
**Runtime:** vLLM (OpenAI-compatible API)
**Hardware:** NVIDIA DGX Spark, GB10 Blackwell, 119 GiB unified LPDDR5X, 273 GB/s bandwidth

## Why This Model

### The selection process

We evaluated every open-source model that fits 119 GiB across three dimensions: coding benchmarks (SWE-bench), tool-calling accuracy (BFCL), and community validation (Reddit, Twitter/X, DGX Spark forums, expert opinions).

### Candidates evaluated

| Model | Active Params | SWE-bench | BFCL | Memory (Q4) | Verdict |
|-------|-------------|-----------|------|-------------|---------|
| **Qwen3.5-122B-A10B** | 10B | 72.0% | **72.2% (v4)** | 76.5 GB | **SELECTED** |
| Qwen3.6-27B (dense) | 27.8B | 77.2% | untested | 16.8 GB | Higher coding score but no BFCL data, broken in Ollama |
| Qwen3.6-35B-A3B | 3B | 73.4% | 67.3% (v4) | 21 GB | Best efficiency but only 3B active — ceiling on complex tasks |
| Qwen3-Coder-Next 80B-A3B | 3B | 70%+ | untested | 45-50 GB | Coding-specialized but no tool-calling benchmarks |
| Nemotron 3 Super 120B-A12B | 12.7B | 60.5% | unpublished | 87 GB (NVFP4) | Purpose-built for DGX Spark but lower scores |
| Devstral Small 2 24B | 24B | 68.0% | untested | 14 GB | Solid backup, tiny footprint |
| Gemma 4 27B | 27B | ~41.6% | 72.7% (v3) | 16 GB | Best multimodal (text+image+audio) but weak coding |
| Qwen3 32B (dense) | 32B | 69.6% | 75.7% (v3) | 19 GB | Community moved past it — Qwen3.6-27B replaced it |
| GLM-4.5-Air 106B | 12B | untested | 76.4% (v3) | 73 GB | High BFCL but score discrepancy across sources |
| DeepSeek V3/V4 | 37-49B active | 78-80% | strong | 350+ GB | Does NOT fit |

### Why not the smaller, faster models?

The user's priority is **accuracy over speed**, especially tool-calling accuracy. GPT-OSS 120B's poor tool calling was the primary bottleneck in the previous setup. The Qwen3.5-122B-A10B has **72.2% BFCL v4** — the highest published tool-calling score of any open-source model that fits this hardware. It outperforms GPT-5 mini (55.5%) on BFCL v4.

### Why not the Qwen3.6-27B despite higher SWE-bench?

- Qwen3.6-27B scores 77.2% SWE-bench (higher) but has **no published BFCL tool-calling score**
- It's **broken in Ollama** due to separate mmproj vision files
- As a dense 27.8B model, it's **2x slower at decode** (27.6 vs 51 tok/s optimized) despite being smaller
- The 122B-A10B wins on **instruction following** (IFBench 76.1 vs 67.6, +8.5 pts) — critical for agents following structured system prompts

### Community validation

- DGX Spark owners specifically converge on this model
- "The Spark paid for itself in about two months of not paying for cloud API tokens"
- Intelligence Index 42 vs Claude 4.5 Sonnet's 43 — Claude-adjacent intelligence
- Community runs it at 51 tok/s with vLLM + hybrid INT4+FP8 + MTP-2 speculative decoding

### Architecture details

| Spec | Value |
|------|-------|
| Total params | 122B |
| Active per token | ~10B (8 routed + 1 shared expert, 256 total experts per layer) |
| Architecture | Hybrid Mamba-Transformer MoE (GatedDeltaNet + GatedAttention) |
| Layers | 48 (12 attention + 36 recurrent DeltaNet) |
| Context | 262K native, 1M extended (YaRN) |
| KV cache per token | 24 KB (FP16) — 2.67x smaller than Qwen3.6-27B |
| Multimodal | Yes (vision) |
| MTP | Via vLLM speculative decoding |
| License | Apache 2.0 |

### Memory & quantization on 119 GiB

| Quantization | Size | Headroom | Max Context |
|-------------|------|----------|-------------|
| Q5_K_M | 91.5 GB | 17.1 GB | ~696K |
| **Q4_K_M** | **76.5 GB** | **32.1 GB** | **~1.3M** |
| UD-IQ4_XS | 60.2 GB | 48.4 GB | ~2M |

### Measured DGX Spark performance

| Runtime | Quantization | Decode tok/s |
|---------|-------------|-------------|
| vLLM | INT4 baseline | 28.3 |
| vLLM | INT4 + MTP-1 | 36.5 |
| **vLLM** | **Hybrid INT4+FP8 + MTP-2** | **51** |
| llama.cpp | UD-Q5_K_XL | 58.6 |
| llama.cpp | Q5_K | 23 |

### Benchmark suite

| Benchmark | Score |
|-----------|-------|
| SWE-bench Verified | 72.0% |
| BFCL v4 (tool calling) | 72.2% |
| LiveCodeBench v6 | 78.9 |
| Terminal-Bench 2.0 | 49.4 |
| IFBench (instruction following) | 76.1 |
| BrowseComp | 63.8 |
| MMLU-Pro | 86.7 |
| GPQA Diamond | 86.6 |
| TAU2-Bench | 79.5 |

## Why vLLM

| Runtime | Single-user tok/s (this model) | Multi-user | OpenAI-compat API | MTP Support | Setup Complexity |
|---------|-------------------------------|-----------|-------------------|-------------|-----------------|
| **vLLM** | **28-51** | **Yes** | **Yes** | **Yes (MTP-2)** | Medium |
| llama.cpp | 23-58 | Limited | Partial | No | Low |
| SGLang | 52 (MXFP4) | Yes | Yes | Limited | High (CUDA 13.0 custom build) |
| Ollama | ~23 | Poor | Yes | No | Lowest |
| TRT-LLM/NIM | Best for Nemotron only | Yes | Yes | Yes | High |

vLLM selected because:
1. **OpenAI-compatible API** — every harness (OpenHands, OpenCode, Aider, custom) works out of the box
2. **MTP-2 speculative decoding** — 51 tok/s, 80% faster than baseline
3. **Multi-user serving** — can serve multiple agents simultaneously (hierarchical harness needs this)
4. **PagedAttention** — efficient KV cache management for long contexts
5. **Community-validated** on DGX Spark with this exact model

### Configuration for optimal performance

```bash
# vLLM serve command (community-optimized for DGX Spark)
vllm serve Qwen/Qwen3.5-122B-A10B \
  --quantization int4 \
  --speculative-model [draft-model] \
  --num-speculative-tokens 2 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90
```

### Recommended Qwen inference settings (from Qwen docs)

```
temperature: 0.6
top_p: 0.95
top_k: 20
presence_penalty: 0.0
```

## vs. Previous Setup (GPT-OSS 120B + Ollama)

| Dimension | GPT-OSS 120B (old) | Qwen3.5-122B-A10B (new) |
|-----------|-------------------|------------------------|
| SWE-bench | ~42% | 72.0% (+30 pts) |
| Tool calling (BFCL) | unpublished/poor | 72.2% |
| Decode speed | 19.5 tok/s | 51 tok/s (+2.6x) |
| Memory | ~87 GB | 76.5 GB (-12%) |
| Architecture | Dense (all 120B active) | MoE (10B active) |
| Context | Limited | 262K-1M |
| License | Proprietary-ish | Apache 2.0 |
| Runtime | Ollama (single-user) | vLLM (multi-agent capable) |

## Secondary / Backup Models

| Role | Model | Memory | Rationale |
|------|-------|--------|-----------|
| Fast/cheap tasks | Devstral Small 2 24B | 14 GB (Q4) | 68% SWE-bench, trivial footprint, Apache 2.0 |
| Vision tasks | Gemma 4 27B | 16 GB (Q4) | Best multimodal (text+image+audio), Apache 2.0 |
| Embeddings | Qwen3-Embedding 0.6B | <1 GB | Already running on this hardware |

All three could run simultaneously with the primary model: 76.5 + 14 + 16 + 1 = ~108 GB, within budget.
