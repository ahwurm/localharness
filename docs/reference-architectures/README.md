# Reference Architectures

LocalHarness is developed and tested against three concrete hardware targets. Every harness
default should work out of the box on all of them; per-hardware setup and tuning notes are in
[gaps.md](gaps.md).

## Practicality bar

A configuration only qualifies as a reference architecture if, on its target machine:

1. **≥64k of KV-cache headroom** — after model weights are resident, enough memory
   remains to hold at least 64k tokens of context (KV headroom = leftover memory usable
   as model context), alongside the OS.
2. **≥9.5 tok/s single-stream decode** — the tested DGX Spark baseline. Agent loops are
   latency-bound on single streams; below this, multi-step tasks stall out.

**Model policy:** each architecture runs the **newest model that meets the bar on its
hardware**. Qwen remains the primary family — it is the model the harness is developed and
benchmarked against — but the policy is family-agnostic where the hardware allows it:
architecture A also has a fully tested DeepSeek V4 Flash configuration. Both architectures
are personally maintained and re-tested by the project maintainer; when a new family
ships, both docs are revised together.

**How architecture C meets bar 1:** on an 8 GB card, 64k of FP16 KV does not fit alongside
the weights — it reaches the 64k figure with `q8_0` KV cache (`-ctk q8_0 -ctv q8_0`), a
one-flag setting that measured no decode or tool-calling cost. Its native FP16 window is
32k. Stated here rather than buried, because it is the one place a reference architecture
clears the bar by a documented setting instead of outright.

| Status | Meaning |
|--------|---------|
| **TESTED** | Numbers measured on the maintainer's hardware; treat as ground truth. |
| **PROPOSED** | Config meets the bar on paper; numbers are estimates until the doc's validation checklist passes. |

## The three architectures

| | A: [DGX Spark](dgx-spark.md) | B: [Base Mac mini](mac-mini.md) | C: [8 GB consumer GPU](rtx-4060.md) |
|---|---|---|---|
| Status | **TESTED** | **PROPOSED** | **PROPOSED** |
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, 128 GB LPDDR5x (119 GiB usable), 273 GB/s | Apple Mac mini (base) — M4, 16 GB unified, 120 GB/s | Discrete 8 GB GPU + 16 GB RAM — reference part RTX 4060, 272 GB/s |
| Default model | Qwen3.8-27B (dense, multimodal) — **3 other tested configs**, see below | `Qwen/Qwen3.5-9B` (dense 9B, hybrid attention) | `empero-ai/Qwen3.8-9B-Distill` (Qwen3.5-9B base, distilled) |
| Quantization | GGUF `UD-Q4_K_XL` (17.9 GB) + MTP head (4.5 GB) | MLX 4-bit / GGUF `Q4_K_M` (5.68 GB) | GGUF `Q4_K_M` (5.38 GiB) |
| Runtime | llama.cpp with MTP speculative decode, OpenAI API on `:8080` | vLLM ([vllm-metal](https://github.com/vllm-project/vllm-metal)), OpenAI API on `:8000` | llama.cpp, OpenAI API on `:8080` |
| Context served | 64k (`-c 65536`) | 64k (`--max-model-len 65536`), KV ≈ 2.1 GB | 32k FP16 KV, or 64k with `q8_0` KV |
| Decode, single stream | **17 tok/s prose, 19.5–21.3 tok/s code (measured)** | est. 10–15 tok/s @ 64k depth (validate ≥9.5) | **~39 tok/s (measured on a bandwidth-equivalent proxy)** |
| Tool calling | llama.cpp `--jinja` (model's own template) | vllm-metal parser / llama.cpp hermes (unvalidated) | llama.cpp `--jinja`, **native — validated end to end** |

Architecture C is the first target aimed at hardware the reader already owns. Its numbers
were measured on the maintainer's Spark standing in for the 8 GB card — the two are within
0.4% on memory bandwidth, which is what single-stream decode is bound by — so decode and
memory-footprint figures transfer but prefill does not. [rtx-4060.md](rtx-4060.md) states
exactly what that proxy does and does not establish.

### Tested configurations on architecture A

The Spark has enough memory to run several very different models well, so the DGX Spark
doc carries four measured configurations rather than one. Summary:

| # | Model | Runtime | Context | Decode (measured) | Why pick it |
|---|-------|---------|---------|-------------------|-------------|
| **A1** | Qwen3.8-27B dense | llama.cpp + MTP speculative decode | 64k | 17 tok/s prose, 19.5–21.3 code | Newest; the only config here that takes images and video; leaves ~85 GB free |
| **A2** | Qwen3.6-35B-A3B MoE | vLLM (NVFP4) | 128k | ~78 tok/s | Fastest decode and the biggest window |
| **A3** | DeepSeek V4 Flash MoE | llama.cpp + DSpark draft model | 64k | 37–40 tok/s | Strongest reasoning model that fits the box at all; uses ~105 of 121 GB |
| **A4** | Qwen3.6-27B dense | vLLM (NVFP4) | 64k | 9.5 tok/s | Smallest footprint; the original reference config |

Those decode figures are **controlled runs** — fixed workload, repeated reps. Real agent
sessions spread wider, because speculative-decoding gain tracks what the model is
generating: the August 2026 live-use test pass measured A1 session medians of 19.9–24.7
tok/s and A3 sessions spanning ~24–47 tok/s. The controlled numbers stay the headline (they
are the reproducible ones); [dgx-spark.md](dgx-spark.md) carries both side by side.

Coverage, stated plainly: **A4 was not re-validated in that pass** — its June 2026 numbers
stand unrevalidated. A2's ~78 tok/s is the July 2026 controlled figure, re-anchored only by
a container-digest check.

**Will not fit:** the Qwen 3.8 MoE flagship (~2.4T total / ~95B active parameters) does
not fit 121 GiB-class hardware at any quantization. Hosted API only — don't start the
download.

Full recipes, flags and caveats: [dgx-spark.md](dgx-spark.md).

## Zero-config detection

`localharness init` already auto-detects all three architectures with no configuration:
`provider/detector.py` probes ports `[8081, 8000, 11434, 1234, 8080]` in priority order
(harness-managed vLLM, stock vLLM, Ollama, LM Studio, llama.cpp). **Agent YAML is
portable across every configuration on this page** — all of them speak the OpenAI API, and
the harness reads the served context window from the runtime at startup rather than
requiring you to restate it. When nothing is running, `init` offers a guided setup:
pick a reference architecture, it installs vLLM (venv or the NVIDIA container route),
downloads the reference model, launches the server on `:8081`, and writes a `server:`
block so `localharness start` restarts it after reboots and the REPL `/model` command
can swap between downloaded models. What does **not** work out of the box yet (context budgets,
timeouts, concurrency) is itemized in [gaps.md](gaps.md).

## Runtime support commitment

vLLM, llama.cpp and Ollama must all work out of the box harness-wide:

- **vLLM** — tier 1 on both architectures: native CUDA build on the Spark (configs A2/A4),
  [vllm-metal](https://github.com/vllm-project/vllm-metal) (MLX backend) on the mini.
- **llama.cpp** — tier 1 on the Spark, where it is the only runtime that currently serves
  configs A1 and A3 (both need speculative decoding); also validated on architecture B
  (`:8080`), and the sole runtime for architecture C, where `init` was confirmed to
  auto-detect it and land on native tool calling.
- **Ollama** — supported on both for models that meet the bar resident (the
  architecture-B model fits comfortably). `doctor` should warn when a configured model
  cannot sit resident in machine RAM — see [gaps.md](gaps.md) §3.

## Documents

- [dgx-spark.md](dgx-spark.md) — architecture A, tested config
- [mac-mini.md](mac-mini.md) — architecture B, proposed config + validation checklist
- [rtx-4060.md](rtx-4060.md) — architecture C, 8 GB consumer GPU, proposed config + validation checklist
- [gaps.md](gaps.md) — development items blocking out-of-box support for all three

---

**Planned:** a Gemma reference architecture (backup family; Qwen remains primary).
