# Reference Architecture B — Base Mac mini

**Status: PROPOSED** (June 2026). Config is derived from a published production recipe on
identical hardware with the previous-generation sibling model; all performance numbers
are estimates until the [validation checklist](#validation-checklist) is executed.

## Summary

| | |
|---|---|
| Hardware | Apple Mac mini, **base configuration** — M4, 16 GB unified, 120 GB/s, NVMe SSD ($799, May-2026 lineup) |
| Model | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) — MoE, ~3B active per token, Apache 2.0 |
| Quantization | GGUF [`unsloth/Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) **`UD-IQ3_XXS` (12.29 GB)** |
| Runtime | **llama.cpp server with `--mmap`** (required), OpenAI API on `:8080` |
| Context | 8–16k practical |
| Decode estimate | 15–20 tok/s (**17.3 tok/s measured** on the Qwen3.5-35B-A3B sibling, same recipe/hardware) |

## Why this model

Per the project model policy (latest Qwen family on both architectures), the base mini
runs Qwen 3.6. The family has two members; on 16 GB the MoE is the only fit, and it is
the better agent model for this hardware regardless:

| | Qwen3.6-27B dense | **Qwen3.6-35B-A3B MoE** |
|---|---|---|
| Fits 16 GB | Only at ~Q3 (~12 GB) with dense-quant quality loss | Yes — `UD-IQ3_XXS` 12.29 GB, MoE quants degrade more gracefully |
| Decode on 120 GB/s | ~10 tok/s (reads all 27.8B weights/token) | est. 15–20 tok/s (reads ~3B active/token) |
| Tool calling | No published BFCL; format-drift issue | **67.3% BFCL v4** (published) |
| SWE-bench | 77.2% | 73.4% |

(Benchmark rows from the candidate table in [CONTEXT-MODEL.md](../../CONTEXT-MODEL.md).)

## The recipe: MoE + mmap on 16 GB

A 35B model on a 16 GB machine works because of two compounding effects, demonstrated in
a [week-long production run](https://thoughts.jock.pl/p/local-llm-35b-mac-mini-gemma-swap-production-2026)
of the Qwen3.5 sibling on a 16 GB mini (17.3 tok/s, zero swap, 16k context):

1. **MoE sparsity** — only ~3B parameters activate per token. Shared layers
   (attention, embeddings, ~4–6 GB) stay resident; expert weights are touched sparsely.
2. **`--mmap`** — llama.cpp memory-maps the GGUF read-only from NVMe. The OS pages
   experts in on demand and evicts cold ones. **Without `--mmap` the same model loads
   ~26 GB resident and the machine swap-thrashes to a freeze.** This is also why Ollama
   currently fails with this model class on 16 GB ([gaps.md](gaps.md) §3).

### Quant ladder (16 GB budget)

| Quant | Disk | Verdict |
|-------|------|---------|
| `UD-Q2_K_XL` | 11.44 GB | Fallback if IQ3_XXS pages too hard |
| **`UD-IQ3_XXS`** | **12.29 GB** | **Reference** — matches the proven recipe class |
| `UD-Q3_K_M` | 15.46 GB | mmap-only; expect heavier paging — test before adopting |
| `UD-Q4_K_M` | 20.62 GB | Needs a 24 GB+ Mac |

## Serving

```bash
llama-server \
  --model Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf \
  --mmap \
  --ctx-size 16384 \
  --jinja \
  --port 8080
```

- `--jinja` enables the model's chat template (hermes-style tool calls).
- Qwen-recommended sampling: `temperature 0.6, top_p 0.95, top_k 20`.
- Headless GPU memory: macOS caps Metal-wired memory at ~75% of RAM by default. For a
  dedicated headless mini, raise it:
  `sudo sysctl iogpu.wired_limit_mb=13312` (13 GB; [reference](https://pixelesque.net/other/snippets/macos/changing-mac-unified-memory-gpu-limit/)).
  Leave ≥2.5 GB for macOS.
- NVMe matters: experts page from disk. The recipe's zero-swap result assumes the
  internal SSD, not external storage.

### Memory budget (16 GB)

| Component | Resident |
|-----------|----------|
| Shared layers + hot experts (mmap) | ~5–8 GB |
| KV cache @ 16k | ~1–2 GB (validate — per-token KV unpublished for this model) |
| macOS floor | ~2.5–3 GB |
| Recipe-measured outcome | "81% memory free, zero swap" during steady decode |

## Harness configuration

`localharness init` detects llama.cpp on `:8080` automatically. Per-agent overrides
required today:

```yaml
# agent YAML — architecture B profile
timeout_seconds: 600.0        # paging can spike first-token latency after idle
max_tokens: 2048
context:
  max_context_tokens: 16384   # match --ctx-size; harness default is 128k
  max_tool_output_chars: 8000 # default 32k ≈ 8k tokens — half this context window
```

## Alternatives considered (and their gaps)

| Runtime | State on base 16 GB mini | Tracked |
|---------|--------------------------|---------|
| **vLLM ([vllm-metal](https://github.com/vllm-project/vllm-metal))** | Preferred long-term (matches architecture A's runtime). Qwen3.6 MoE support is experimental and only MLX-4bit/AWQ quants are supported — 35B-A3B @ 4-bit ≈ 18–20 GB, needs a 24 GB+ Mac. No mmap-style paging. | [gaps.md](gaps.md) §3 |
| **Ollama** | Fine for ≤14B-class models. Loads this model class fully resident (~26 GB) → swap freeze on 16 GB. | [gaps.md](gaps.md) §3 |
| 27B dense @ Q3 | Fits, but ~10 tok/s and dense 3-bit quality loss; no published tool-calling score. | rejected above |

## Validation checklist

Run on the actual base mini before promoting to TESTED:

- [ ] Decode tok/s, single stream, 1k/8k/16k prompt depths (target: ≥15)
- [ ] First-token latency cold vs warm (expert paging after idle; informs stuck-detector thresholds)
- [ ] Max stable context with zero swap (`memory_pressure`, `vm_stat` swapouts = 0)
- [ ] Tool-call success rate via llama.cpp hermes parser against `bench/` scenarios
- [ ] Thinking-mode token accounting (reasoning traces vs `max_tokens`)
- [ ] Week-long stability under the harness agent loop (recipe proved raw serving only)
- [ ] `UD-Q2_K_XL` vs `UD-IQ3_XXS` quality delta on `bench/` if paging is heavy
