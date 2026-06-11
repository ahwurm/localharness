# Reference Architectures

LocalHarness is developed and tested against two concrete hardware targets. Every harness
default should work out of the box on both; anything that doesn't is tracked in
[gaps.md](gaps.md) as a development item.

**Model policy:** both architectures track the **latest Qwen family** (currently Qwen 3.6,
April 2026). When a new family ships, both reference docs are revised together. Both
architectures are personally maintained and re-tested by the project maintainer.

| Status | Meaning |
|--------|---------|
| **TESTED** | Numbers measured on the maintainer's hardware; treat as ground truth. |
| **PROPOSED** | Config derived from published recipes/benchmarks; numbers are estimates until the validation checklist in the doc is executed. |

## The two architectures

| | A: [DGX Spark](dgx-spark.md) | B: [Base Mac mini](mac-mini.md) |
|---|---|---|
| Status | **TESTED** | **PROPOSED** |
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, 128 GB LPDDR5x (119 GiB usable), 273 GB/s | Apple Mac mini (base) — M4, 16 GB unified, 120 GB/s |
| Model | `Qwen/Qwen3.6-27B` (dense 27.8B) | Qwen3.6-35B-A3B (MoE, ~3B active) |
| Quantization | NVFP4 (~15.6 GB weights) | GGUF `UD-IQ3_XXS` (12.29 GB) |
| Runtime | vLLM, OpenAI API on `:8000` | llama.cpp server `--mmap`, OpenAI API on `:8080` |
| Context served | 64k (`--max-model-len 65536`) | 8–16k practical |
| Decode, single stream | **9.5 tok/s (measured)** | est. 15–20 tok/s (17.3 measured on the Qwen3.5 sibling) |
| Tool calling | vLLM hermes parser (format drift — see doc) | llama.cpp hermes parser (unvalidated) |

## Zero-config detection

`localharness init` already auto-detects both architectures with no configuration:
`provider/detector.py` probes ports `[8000, 11434, 1234, 8080]` in priority order
(vLLM, Ollama, LM Studio, llama.cpp). Architecture A answers on `:8000`, architecture B
on `:8080`. What does **not** work out of the box yet (context budgets, timeouts,
concurrency, runtime alternatives) is itemized in [gaps.md](gaps.md).

## Runtime support commitment

vLLM and Ollama must both work out of the box harness-wide:

- **vLLM** — tier 1 on DGX Spark (architecture A). On the base Mac mini, vLLM via
  [vllm-metal](https://github.com/vllm-project/vllm-metal) cannot yet host the 35B-A3B
  (no low-bit quant support) — tracked in [gaps.md](gaps.md) §3.
- **Ollama** — works on both machines for models that fit resident memory (≤ ~14B class
  on the base mini). Known failure loading 35B-A3B on 16 GB — tracked in
  [gaps.md](gaps.md) §3.

## Documents

- [dgx-spark.md](dgx-spark.md) — architecture A, tested config
- [mac-mini.md](mac-mini.md) — architecture B, proposed config + validation checklist
- [gaps.md](gaps.md) — development items blocking out-of-box support for both
- [../../CONTEXT-MODEL.md](../../CONTEXT-MODEL.md) — historical model-selection analysis (superseded)
