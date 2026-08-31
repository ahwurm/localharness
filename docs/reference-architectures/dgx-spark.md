# Reference Architecture A — NVIDIA DGX Spark

**Status: TESTED** (maintainer hardware). Four configurations are supported on this box
and each was measured on this exact stack at the date given below. Numbers labelled
*measured* were produced here; anything else is labelled as an estimate or as unverified.

Two provenance labels recur below, and they are not interchangeable:

- **Controlled runs** — a fixed workload, repeated reps, warmup excluded (`llama-bench`,
  scripted long completions). Tight spread, reproducible.
- **The August 2026 live-use test pass** — real agent sessions driven through the harness
  REPL, where workload, generation length and context depth all vary turn to turn. Wider
  spread by construction.

Where the two differ, **the controlled number stays this doc's headline** (its workload is
pinned, so it is the one you can reproduce) and the live range is reported alongside it to
say what the config actually feels like in use. Each table below states which it is.

## Hardware

| | |
|---|---|
| Hardware | NVIDIA DGX Spark — GB10 Grace Blackwell, SM 12.1 |
| Memory | 128 GB LPDDR5x unified (119 GiB usable), 273 GB/s |
| CPU | 20-core ARM (10× Cortex-X925 + 10× Cortex-A725) |

## Supported configurations

All four clear the [practicality bar](README.md#practicality-bar) (≥64k context, ≥9.5
tok/s single-stream). Pick by what the workload needs — there is no single winner.

| # | Model | Runtime | Quantization | Context | Decode, single stream (measured) | Measured |
|---|-------|---------|--------------|---------|----------------------------------|----------|
| **A1** | **Qwen3.8-27B** — dense, multimodal, MTP head | llama.cpp + MTP speculative decode | GGUF `UD-Q4_K_XL` 17.9 GB + MTP head 4.5 GB | 64k | **17 tok/s** prose, **19.5–21.3 tok/s** code (11.3 tok/s synthetic baseline with MTP not engaged — see the section for why these two are not directly comparable) | Aug 2026 |
| **A2** | Qwen3.6-35B-A3B — MoE, 35B total / ~3B active | vLLM | NVFP4 ~22 GB | 128k | **~78 tok/s** | Jul 2026 |
| **A3** | DeepSeek V4 Flash — MoE, MLA attention | llama.cpp + DSpark draft model | GGUF `UD-Q2_K_XL` ~90 GB + drafter 10.9 GB | 64k | **37–40 tok/s** (live agent sessions span ~24–47, set by draft acceptance — see the section) | Aug 2026 |
| **A4** | Qwen3.6-27B — dense | vLLM | NVFP4 ~15.6 GB | 64k | **9.5 tok/s** | Jun 2026 — **not** re-validated in the Aug 2026 pass |

**Which one:**

- **A1 (default recommendation)** — newest model, the only one here that takes images and
  video, and it leaves ~85 GB of the box free. Slowest decode of the four, but well over
  the bar.
- **A2** — fastest decode and the largest context window (128k). Text only. Pick this when
  turn latency matters more than model recency.
- **A3** — the strongest reasoning model that fits this machine at all, at the price of
  filling it (~105 of 121 GB with the drafter loaded). Currently **parked** on the
  maintainer's box in favour of A1, but the recipe is complete and was measured here.
- **A4** — the original reference config. Smallest footprint; kept for reproducibility of
  older results.

The full recipes follow, **grouped by runtime rather than by number** — the two llama.cpp
configs (A1, A3) first, then the two vLLM ones (A2, A4) — because the setup work is shared
within each pair.

**Does not fit — don't start the download:** the Qwen 3.8 MoE flagship (~2.4T total
parameters, ~95B active) does not fit 121 GiB-class hardware at **any** quantization.
Estimated weight sizes are ~1.2 TB at NVFP4 and still ~600 GB heavily quantized to 2 bits —
an order of magnitude past the machine, not a near miss that a smaller quant or offload
trick rescues. (Those two figures are sizing estimates from the parameter count, not
measurements; nothing was downloaded.) Use that model through a hosted API or not at all.

## A1 (recommended): Qwen3.8-27B — llama.cpp with MTP speculative decode

### Model

Qwen3.8-27B is the dense flagship of the Qwen 3.8 family: 64 transformer layers, 262,144-token
native context, natively multimodal (the checkpoint ships image **and** video preprocessor
configs, not a bolted-on projector), and it publishes a **multi-token-prediction (MTP)
head** — a small extra module trained to guess the next few tokens, which llama.cpp can
drive as a speculative-decoding draft model. That head is what turns 11.3 tok/s into
17–21 tok/s below.

### Serving

**llama.cpp must be a master build from mid-August 2026 or newer.** Support for this
model's graph and the MTP draft path (`--spec-type draft-mtp`) is that recent. The failure
mode on an older binary is nasty because it is not a clean refusal: the model *loads* (it
gets mapped onto the older Qwen 3.5 architecture) and short benchmarks look plausible, then
`ggml_abort` fires during decode once prompts or generations get longer. If you see that,
rebuild from master before debugging anything else — and discard any numbers the old build
produced. Keeping the new build as a separate checkout is the low-drama option.

Two files are needed — the quantized weights and the separately-packaged MTP head:

| File | Size | Purpose |
|------|------|---------|
| `Qwen3.8-27B-UD-Q4_K_XL.gguf` (`unsloth/Qwen3.8-27B-GGUF`) | 17.9 GB (16.7 GiB) | target model |
| `Qwen3.8-27B-MTP-ONLY-Q8_0.gguf` (`a4lg/Qwen3.8-27B-MTP-ONLY-GGUF`) | 4.5 GB | draft head |

The MTP head ships as a separate repack because it is not part of the standard GGUF
conversion — you cannot extract it from the main file.

```bash
llama-server \
  -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  -md Qwen3.8-27B-MTP-ONLY-Q8_0.gguf \
  --spec-type draft-mtp \
  -ngl 99 -ngld 99 -fa on \
  --host 127.0.0.1 --port 8000 -c 65536 --jinja -a qwen3.8-27b
```

- `--spec-type draft-mtp` tells llama.cpp the "draft model" is this checkpoint's own MTP
  head rather than an independent small model.
- `-ngld 99` puts the draft head fully on the GPU as well; leaving it on CPU throws away
  most of the speedup.
- `-fa on` takes an explicit value in current builds — a bare `-fa` is not the same thing.
- `--jinja` is required for tool calling — it uses the model's own chat template instead
  of llama.cpp's generic one.
- `-a qwen3.8-27b` sets the name the server advertises, which is what you put in
  `provider.default_model`.
- Check the startup banner for `n_ctx_slot` rather than assuming: llama-server's slot
  semantics for `-c` have changed between releases, and what you care about is that one
  request can use the full 65,536 tokens.

### Measured performance (GB10, `UD-Q4_K_XL`, August 2026)

Read the two halves of this table separately — **they are different measurements**, and
the difference matters more than the ratio between them.

*Synthetic, via `llama-bench` at zero context depth, prefill excluded from the decode rows:*

| Metric | Value |
|--------|-------|
| Prefill, 512-token prompt | **~816–820 tok/s** |
| Prefill, 2048-token prompt | **~821 tok/s** |
| Decode, MTP not engaged (`tg32`/`tg128`/`tg256`) | **11.3 tok/s** (±0.02) |

*End-to-end wall clock, MTP engaged, 256-token generations with prefill included:*

| Workload | Value |
|----------|-------|
| Prose | **17.0–17.1 tok/s** |
| Code | **19.5–21.3 tok/s** |
| Mean accepted draft length | **2.45 tokens** |
| Total resident footprint | **~34 GB** (weights + MTP head + KV/buffers) |

So MTP is worth roughly **1.5–1.9×** on this hardware. Treat that as the honest shape of
the effect rather than a precise multiplier: the 11.3 baseline excludes prefill and the
17–21 figures include it, so a like-for-like decode-only comparison would differ somewhat.

*Live agent sessions — the August 2026 live-use test pass, in-REPL decode readout:*

| Metric | Value |
|--------|-------|
| Per-session median decode | **19.9 / 24.7 / 22.3 tok/s** (independent sessions) |
| Individual samples across those sessions | **18.25–26.84 tok/s** |

This is a **third** measurement class, not a correction of the two above: real agent turns
at varying context depth and generation length, versus fixed 256-token generations. It
lands at or above the controlled 17.0–21.3 band rather than below it, and the likely reason
is the same acceptance effect the controlled numbers already show — an agent loop emits
tool calls and structured output, which the MTP head drafts better than free prose, so live
sessions skew toward the code end. **The controlled figures stay the headline for A1**; read
the session medians as the in-use spread, not as a competing claim. Neither set changes the
one honest caveat: the gain is task-dependent, so a session's position in the range is set
by what that session was doing.

Speculative decoding is lossless with respect to output — rejected draft tokens are thrown
away, so the text is what the 27B would have produced anyway. The **gain** varies with how
predictable the next tokens are, which is why code beats prose. Per-workload draft
*acceptance rates* are deliberately not published here: the maintainer's acceptance
telemetry for this config was not retained in a form that survives audit, so only the mean
accepted draft length above is stated as measured. (A3 below *does* publish an acceptance
range — that is not an inconsistency but a difference in evidence: A3's came off the
`llama-server` log during the August 2026 pass and was kept. The same figures were never
captured for A1, and no A1 number is inferred from A3's.)

~34 GB of 119 GiB leaves roughly 85 GB free — room for a second model, an embedding
server, or higher concurrency.

Prefill and decode were both measured at zero context depth. **Decode at 32k–64k depth has
not been measured for this config** — expect it to be slower, as on any attention model.

### The vLLM / NVFP4 lane: not yet

An NVFP4 checkpoint for this model exists (`unsloth/Qwen3.8-27B-NVFP4` — 22.6 GB of
weights plus a 0.85 GB MTP head), which on paper is the faster route on Blackwell.
**No Spark-compatible vLLM build could serve it as of August 2026.** Two attempts, both
maintainer-reported and neither with a retained log, so treat these as pointers for the
next person rather than as reproducible findings:

- The community ARM/Spark vLLM 0.17 build rejects the checkpoint's quantization config.
  The checkpoint declares `quant_method: compressed-tensors` with
  `format: mixed-precision`, which that version does not accept.
- The NGC 26.07 container fails during engine-core initialization on this model.

The next NGC container drop (or an updated community tag) is the thing to retest. Until
then **A1 is a llama.cpp config.** No performance claim is made for the NVFP4 path here,
because nothing has successfully run it on this hardware — the checkpoint is worth
downloading only if you intend to be the person who gets it working.

## A3: DeepSeek V4 Flash — llama.cpp with the DSpark draft model

### Model

DeepSeek V4 Flash is a large MoE model using MLA (multi-head latent attention), which
compresses the KV cache hard — the practical consequence on this box is that context is
nearly free (a 128k window costs only a few GB over the weights). It is the strongest
reasoning model that fits this machine at all, and it fits only at ~2-bit.

DeepSeek publishes an official **DSpark draft checkpoint** (MIT-licensed) for speculative
decoding, and llama.cpp has first-class support for it.

**Status: parked, not deprecated.** The maintainer's box currently runs A1 instead — this
config leaves ~16 GB of headroom and reloads ~90 GB on every restart. The recipe below is
complete and the numbers are measured here. Parked does not mean stale: it was brought back
up and driven live during the August 2026 test pass, and the live-session numbers below are
from that.

### Serving

Requires a llama.cpp build with the DSpark speculative-decoding merge (mainline since
early August 2026 — build b10269 or newer).

```bash
llama-server \
  -m DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf \
  -md dspark-DeepSeek-V4-Flash-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 5 --spec-draft-p-min 0.3 \
  --fit off -ngl 99 -ngld 99 -fa on \
  --host 127.0.0.1 --port 8000 -c 65536 --jinja -a deepseek-v4-flash --parallel 1
```

Reloading ~90 GB takes a while and this box runs close to full with this config — restart
it deliberately, not in a loop.

Every flag here is load-bearing:

- `--spec-type draft-dspark`, **not** `draft-dflash` — the latter drafts shifted by one
  token and produces wrong results.
- `--spec-draft-n-max 5` explicitly. The default of 3 wastes a drafter trained on blocks
  of 5. Do **not** raise it further: larger blocks measured *slower* than the no-draft
  baseline on GB10.
- `--spec-draft-p-min 0.3` — the confidence floor below which the drafter stops proposing.
- `--fit off` — the auto-fit memory planner mis-accounts for two resident models. If your
  build rejects the flag, drop it and watch the load output for the real allocation.
- `-ngld 99` puts the drafter on the GPU. Do not use `-devd` for this; it is documented
  broken.

### Measured performance (GB10, `UD-Q2_K_XL`, August 2026)

| Metric | Value |
|--------|-------|
| Decode, no speculation | **17.0 tok/s** (Aug 2026: 8 samples, 16.75–17.13 — a notably tight spread, and the harness's speed ledger, `llama-server`'s own log and the in-REPL readout all agree. Matches the 17.35 tok/s independently published for this model/quant/box.) |
| Decode with the drafter, 64k context | **37–40 tok/s** aggregate (40.6 tok/s best in that round) |
| Decode with the drafter, by task type (32k-era measurements) | code ~30, structured ~36, prose ~22 tok/s |
| Resident footprint with drafter @ 64k | **~105 GB of 121 GiB** |

*Live agent sessions with the drafter — the August 2026 live-use test pass:*

| Metric | Value |
|--------|-------|
| Per-session median decode | **31.9 / 33.0 / 33.5 / 40.8 tok/s** (independent sessions) |
| Individual samples across those sessions | **23.75–47.45 tok/s** |
| Draft acceptance rate (from the `llama-server` log) | **0.43–0.80** |
| 400-token code-generation sanity check | **26.07 tok/s** |

The honest live range for this config is therefore **~24–47 tok/s** (floor = the lowest
recorded sample, 23.75). The 37–40 headline sits
inside it and stands as the aggregate figure; the spread around it is not noise, it is the
0.43–0.80 acceptance spread showing through — those two rows are the same fact measured two
ways.

Three honesty notes:

- The 37–40 tok/s headline and the per-task breakdown come from **different measurement
  rounds** (64k vs an earlier 32k pass) — treat the per-task row as the shape of the
  effect, not as directly comparable numbers.
- The gain is entirely acceptance-rate driven and therefore task-dependent, same as A1.
  Code and agent/structured output gain most; free prose gains least.
- **Don't read that task mapping too tightly.** The 400-token code generation above measured
  26.07 tok/s — near the *bottom* of the live range, not the top. Task type is a tendency,
  not a predictor; acceptance on the specific tokens being generated is what decides.

**Why 64k and not 128k:** MLA makes the KV cache cheap enough that 128k costs only a few
GB, so the window is not what forces the choice — the 10.9 GB drafter plus its own KV is.
Without speculative decoding, 128k fits fine.

## A2: Qwen3.6-35B-A3B (NVFP4, vLLM) — fastest decode

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
`sha256:a671d5fcda70fe9ac6f245f9780821de459fb4ee22c018fd07a0f10a55279bf9` — re-checked
against the running image during the August 2026 live-use test pass and still the same
digest, so the performance numbers below describe the image documented here. **vLLM ≥0.22.1
is required** — older builds cannot load this checkpoint's NVFP4 MoE weights.

Two flags in that recipe are non-obvious and worth calling out:

- `--max-num-batched-tokens 4096` — **required**, not just a tuning knob. This
  checkpoint's hybrid Mamba/GDN layers assert `block_size` (2096) ≤ batched tokens;
  vLLM's own default of 2048 sits below that and crashes at startup.
- `--reasoning-parser qwen3` — routes chain-of-thought into the response's `reasoning`
  field so `content` arrives clean. Without it, thinking text pollutes the answer.

### Measured performance

Method (**controlled run**, July 2026): temperature 0, concurrency 1 (single stream),
medians over repeated reps, warmup run excluded, prefix cache defeated with nonces so every
request is a genuine cache miss. The August 2026 pass re-verified the image digest for this
config but did **not** re-run this measurement — see the readout caveat below for why its
live sessions produced no usable replacement number.

| Metric | Value |
|--------|-------|
| Decode throughput (steady state) | **~78 tok/s** (median 77.97 tok/s over 5×256-token reps; 512-token group 78.3 tok/s) |
| Time to first token | ~0.09 s |
| Time to first *answer* token (reasoning on, default) | ~7.1 s — the model spends ~600 hidden reasoning tokens before the first visible answer token |
| Prefill | ~0.8 s at ~1k prompt tokens; ~1.0–1.3 s at ~4k (first request at a new depth is slower — warmup effect) |
| Context window | 131,072 tokens (`--max-model-len 131072`) |

Caveats:

- **The harness's in-REPL speed readout was unreliable on vLLM before 0.12.5 — discard any
  vLLM tok/s figure you saw in the REPL from that period.** On client-measured runtimes the
  harness timed chunk *arrival*, and the short bursty generations an agent loop emits
  between tool calls arrive coalesced, so the arithmetic divided a real token count by a
  near-zero window. Displayed values ran as high as **152 tok/s** on this ~78 tok/s config.
  That was the measurement, not the model. 0.12.5 fixed it with a sample-substance gate: a
  client-measured sample is admitted only if it carries ≥16 completion tokens over a ≥0.25 s
  window. **~78 tok/s remains the number for A2** — it comes from the controlled July 2026
  method above, which never went through the affected path.
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

For orientation: the A4 27B dense NVFP4 config documented below decodes roughly
10–15 tok/s on this hardware by informal observation — not benchmarked to the same
standard as the numbers above. (That section's own Measured performance table has the
rigorous single-stream figure this doc otherwise treats as ground truth.)

## A4: Qwen3.6-27B (dense, NVFP4, vLLM)

Qwen3.6-27B was the original recommended model for this hardware (tested June 2026). It
remains fully supported and documented below as a smaller-footprint config
(~15.6 GB weights vs. ~22 GB for the 35B-A3B recipe above).

**Not re-validated in the August 2026 live-use test pass** — this config was not brought up
at all during it. Everything in this section is the June 2026 measurement, standing
unrevalidated against the current vLLM builds and the current harness.

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

### Measured performance (June 2026 — unrevalidated since)

| Metric | Value |
|--------|-------|
| Decode, single stream | **9.5 tok/s** |
| KV cache | 64k, no eviction issues |

vLLM batching raises aggregate throughput with concurrent requests, but **per-stream
latency is what governs an agent loop** — size harness timeouts from the single-stream
figure ([gaps.md](gaps.md) §1).

## Harness configuration

`localharness init` detects a running server automatically — vLLM on `:8081`/`:8000`,
llama.cpp on `:8080` — and writes the `provider:` block for you. With no server running,
its guided setup can install and launch vLLM itself (container route; Docker and the
NVIDIA container toolkit must already be installed).

**No per-agent overrides are required for any of the four configs.** At startup the
harness reads the served context length from the runtime (vLLM `/v1/models`, llama.cpp
`/props`) and uses it as the context budget, holding room for the model's reply back inside
that window, so a 64k server is respected without you restating 64k in YAML. The provider default
`timeout_seconds` is 600.0, which covers a full 4096-token completion at every decode
rate in the table above (the slowest, A4 at 9.5 tok/s, needs ~431s).

Set `context.max_context_tokens` only when you deliberately want a budget **smaller**
than what the server offers — for example to cap cost or force earlier compaction:

```yaml
# <config_dir>/agents/<name>.yaml — name and role are required
name: coder
role: Writes and reviews code.
model: inherit
context:
  max_context_tokens: 65536   # optional cap; never set this ABOVE the served window
```

If you swap between the configs above on the same box, note that `max_context_tokens` is a
single scalar and cannot be correct for two models with different served windows. Pin the
odd one out by exact model name instead of re-editing the scalar on every swap:

```yaml
context:
  max_context_tokens: 65536       # every model without a pin
  model_context_overrides:
    <served-model-name>: 40000    # this model only; name must match the server EXACTLY
```

The key must be the model name the runtime reports, character for character — there is no
globbing or prefix matching, and a mistyped key silently falls back to the scalar. A pin
larger than the served window aborts `start` with an error naming the pin. See
[spec 06](../specs/06-config.md#per-model-context-pins-contextmodel_context_overrides).

Config files live under `~/.localharness` by default (override with `LOCALHARNESS_DIR`).
Note that agent-level keys like `context:` and `timeout_seconds:` belong in an **agent**
file; the root `config.yaml` has a different shape and rejects unknown keys outright.

### Letting the harness own the llama.cpp server (A1/A3)

For the two llama.cpp configs, a `server:` block in `config.yaml` makes `localharness
start` bring the server up and take it down with the session. Speculative-decoding flags
are passed straight through — the harness has no schema of its own for them:

```yaml
# <config_dir>/config.yaml (excerpt)
server:
  runtime: llamacpp
  launch: binary
  binary: ~/llama.cpp/build/bin/llama-server
  model: ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf
  port: 8000
  extra_args:
    - "-md"
    - "~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-MTP-ONLY-Q8_0.gguf"
    - "--spec-type"
    - "draft-mtp"
    - "-ngl"
    - "99"
    - "-ngld"
    - "99"
    - "-fa"
    - "on"
    - "-c"
    - "65536"
    - "--jinja"
    - "--parallel"
    - "1"
```

The same block works for A3 by swapping the model paths and using `--spec-type
draft-dspark` with its `--spec-draft-n-max 5 --spec-draft-p-min 0.3 --fit off` flags.

## Known issues

1. **Tool-call format drift** on Qwen3.6-27B (A4) — intermittent stray closing tags;
   [gaps.md](gaps.md) §5. Not seen in the August 2026 live-use test pass, which recorded
   **zero tool-call parse failures** in every session — across both llama.cpp configs
   (A1, A3, `--jinja` path) and vLLM native tool-calling (A2). That is live evidence, not a
   bench score, and it does **not** clear A4: A4 was never brought up during that pass, so
   the model this issue was actually reported against remains untested against it.
2. **NVFP4 kernel maturity on SM 12.x** — re-validate decode rate on each vLLM upgrade.
   The concern was raised against the community PTQ 27B checkpoint (A4) on then-current
   kernels; A2 is separately measured on a specific `nightly` digest and a pinned vLLM
   ≥0.22.1 floor, and should be re-benchmarked the same way on future upgrades.
3. **Qwen3.8-27B on vLLM does not work yet** (A1) — see the NVFP4 lane note above. The
   llama.cpp path is the only tested one for this model on this hardware.
4. **Speculative-decoding gain is task-dependent** (A1, A3) — the measured ranges assume
   a mix of code and prose. A prose-only workload lands at the bottom of each range.
5. **Memory-footprint telemetry is unreliable** on GB10's unified memory — `docker stats`
   undercounts and `nvidia-smi` memory counters return `N/A`. Footprint figures in this
   doc come from process-level accounting at load time, not from GPU counters.
