# Reference Architectures

LocalHarness is developed and tested against two concrete hardware targets. Every harness
default should work out of the box on both; anything that doesn't is tracked in
[gaps.md](gaps.md) as a development item.

## Practicality bar

A configuration only qualifies as a reference architecture if, on its target machine:

1. **≥64k of KV-cache headroom** — after model weights are resident, enough memory
   remains to hold at least 64k tokens of context (KV headroom = leftover memory usable
   as model context), alongside the OS.
2. **≥9.5 tok/s single-stream decode** — the tested DGX Spark baseline. Agent loops are
   latency-bound on single streams; below this, multi-step tasks stall out.

**Model policy:** each architecture runs the **newest Qwen model that meets the bar on
its hardware**. Currently that is Qwen 3.6 on architecture A; Qwen 3.6 ships no dense
model small enough for architecture B (the family is 27B dense + 35B-A3B MoE only, June
2026), so B runs Qwen3.5-9B until a small Qwen 3.6 dense releases. Both architectures
are personally maintained and re-tested by the project maintainer; when a new family
ships, both docs are revised together.

| Status | Meaning |
|--------|---------|
| **TESTED** | Numbers measured on the maintainer's hardware; treat as ground truth. |
| **PROPOSED** | Config meets the bar on paper; numbers are estimates until the doc's validation checklist passes. |

## The two architectures

| | A: [DGX Spark](dgx-spark.md) | B: [Base Mac mini](mac-mini.md) |
|---|---|---|
| Status | **TESTED** | **PROPOSED** |
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, 128 GB LPDDR5x (119 GiB usable), 273 GB/s | Apple Mac mini (base) — M4, 16 GB unified, 120 GB/s |
| Model | `Qwen/Qwen3.6-27B` (dense 27.8B) | `Qwen/Qwen3.5-9B` (dense 9B, hybrid attention) |
| Quantization | NVFP4 (~15.6 GB weights) | MLX 4-bit / GGUF `Q4_K_M` (5.68 GB) |
| Runtime | vLLM, OpenAI API on `:8000` | vLLM ([vllm-metal](https://github.com/vllm-project/vllm-metal)), OpenAI API on `:8000` |
| Context served | 64k (`--max-model-len 65536`), KV ≈ 4.2 GB | 64k (`--max-model-len 65536`), KV ≈ 2.1 GB |
| Decode, single stream | **9.5 tok/s (measured)** | est. 10–15 tok/s @ 64k depth (validate ≥9.5) |
| Tool calling | vLLM hermes parser (format drift — see doc) | vllm-metal parser / llama.cpp hermes (unvalidated) |

## Zero-config detection

`localharness init` already auto-detects both architectures with no configuration:
`provider/detector.py` probes ports `[8000, 11434, 1234, 8080]` in priority order
(vLLM, Ollama, LM Studio, llama.cpp). **Both architectures answer on `:8000` with the
same runtime family (vLLM)** — agent YAML is identical across machines except for the
context/timeout profile. What does **not** work out of the box yet (context budgets,
timeouts, concurrency) is itemized in [gaps.md](gaps.md).

## Runtime support commitment

vLLM and Ollama must both work out of the box harness-wide:

- **vLLM** — tier 1 on both architectures: native CUDA build on the Spark,
  [vllm-metal](https://github.com/vllm-project/vllm-metal) (MLX backend) on the mini.
- **Ollama** — supported on both for models that meet the bar resident (the
  architecture-B model fits comfortably). `doctor` should warn when a configured model
  cannot sit resident in machine RAM — see [gaps.md](gaps.md) §3.
- **llama.cpp** — additionally validated on architecture B (`:8080`).

## Documents

- [dgx-spark.md](dgx-spark.md) — architecture A, tested config
- [mac-mini.md](mac-mini.md) — architecture B, proposed config + validation checklist
- [gaps.md](gaps.md) — development items blocking out-of-box support for both

---

**Planned:** a Gemma reference architecture (backup family; Qwen remains primary).
