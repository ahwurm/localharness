# Running LocalHarness on llama.cpp

*See the provider support matrix in the README for the full per-provider status.*

[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` is one of the four
OpenAI-compatible runtimes LocalHarness speaks to (alongside vLLM, Ollama, and LM Studio) — and
the one the harness spawns and owns most directly: a single process it forks, tracks by pid, and
tears down itself.

**Support status:** detection, the managed lifecycle, tool-calling, and token counting are all
TESTED live (including a real cross-framework heavy-swap on a DGX Spark); only the bench-matrix
entry needs your served model id filled in before it will run.

## Install

Build `llama-server` from source (there's no single blessed package across distros):

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # swap for your backend: -DGGML_METAL=ON, or omit for CPU-only
cmake --build build --config Release -j
```

The binary lands at `build/bin/llama-server`. Flag choice is hardware-specific — follow
llama.cpp's own build docs for your GPU/CPU combination.

## Serve a model

llama.cpp serves a single GGUF file directly — no registry, no pull step:

```bash
llama-server -m <model.gguf> --host 127.0.0.1 --port 8080 --jinja -c 32768 -ngl 99 -a <served-name>
```

- `-c` — context size in tokens.
- `-ngl 99` — offload (effectively) all layers to GPU; lower it on a memory-constrained box.
- `--jinja` — render the GGUF's chat template, required for tool-call syntax to work at all.
- `-a` — the served model id/alias exposed over the API (your choice).

OpenAI-compatible surface at `http://127.0.0.1:8080/v1`; native `/props` reports `n_ctx`, native
`/tokenize` gives exact token counts.

**Load-bearing gotcha:** recent `llama-server` builds default to multiple parallel slots and
silently divide `-c` among them (e.g. `-c 32768` across 4 slots = an 8k window per request). The
harness is single-stream — pass `--parallel 1` so one slot owns the whole context. `init`/`doctor`
read the *per-slot* figure from `/props`; if it reports less context than you launched with, check
the slot count first.

## Point LocalHarness at it

Two separate axes show up below: `provider_type` (the wire protocol — drives token counting) and
`runtime` (the launch strategy — how the harness starts/stops the process). They share the same
four names by coincidence, not because they're the same field.

### Attach (already running)

Point the primary provider at a `llama-server` you started yourself:

```yaml
# ~/.localharness/config.yaml
provider:
  provider_type: llamacpp
  base_url: http://127.0.0.1:8080/v1
  default_model: qwen3.6-35b-a3b   # whatever you passed to -a
  api_key: none
```

Or add it as a peer the `/model` tree can switch to, alongside a different primary:

```yaml
extra_endpoints:
  - name: llamacpp-local
    base_url: http://127.0.0.1:8080/v1
    provider_type: llamacpp
    gpu: true   # does this peer occupy the accelerator? no inference — say so explicitly
```

### Harness-managed

Let the harness launch, watch, and stop `llama-server` itself:

```yaml
server:
  runtime: llamacpp
  binary: /home/you/llama.cpp/build/bin/llama-server   # absolute path — see Troubleshooting
  model: /home/you/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
  port: 8080
  gpu: true
  extra_args: ["-c", "32768", "-ngl", "99", "--jinja", "-a", "qwen3.6-35b-a3b", "--parallel", "1"]
```

Or as a cross-framework `/model`-tree peer the harness can cold-launch on the freed accelerator
(the same shape, nested under `lifecycle:`):

```yaml
extra_endpoints:
  - name: llamacpp-peer
    base_url: http://127.0.0.1:8080/v1
    provider_type: llamacpp
    gpu: true
    lifecycle:
      runtime: llamacpp
      binary: /home/you/llama.cpp/build/bin/llama-server
      model: /home/you/models/Qwen3.6-35B-A3B-Q4_K_M.gguf
      port: 8080
      gpu: true
      extra_args: ["-c", "32768", "-ngl", "99", "--jinja", "-a", "qwen3.6-35b-a3b"]
```

## What the harness does

`SpawnedProcessStrategy` reuses the exact launch/readiness/stop primitives vLLM's binary mode
uses:

- **Activate** — builds the `llama-server` command from your config, launches it detached (its
  own session, a pidfile written, stdout+stderr appended to a log), then polls `{base_url}/models`
  until it answers. If the process dies mid-startup, this fails fast with the log tail instead of
  polling a corpse for 30 minutes.
- **Stop** — SIGTERM to the whole process group, SIGKILL if it hasn't exited after the grace
  window. Pid-group teardown is correct here because a spawned `llama-server` is a single tracked
  process — never a docker client — so there's no name-based liveness dance to do.
- **Liveness** — `is the pidfile's pid alive?`. Also correct here: unlike vLLM's docker mode
  (where the pidfile holds the *client* pid, not the container), the llama.cpp pidfile pid **is**
  the real server process.

## Support status

- Detect (response-shape) — TESTED
- Init: model + context window (`/props` n_ctx) — TESTED
- Managed lifecycle (harness spawns llama-server, verified process-group stop) — TESTED
  (live-proven; also drove a live cross-framework heavy-swap on a DGX Spark)
- Tool-calling (taught XML dialect injected + parsed; tool-name sanitization) — TESTED
- Token counting — TESTED: exact via `/tokenize`; whole requests via `/apply-template` +
  `/tokenize` (message structure AND tools block) — verified live equal to
  `usage.prompt_tokens`. Older llama-server builds without `/apply-template` keep exact
  content counts and estimate message overhead, disclosed at start
- Bench run — a `bench.yaml` matrix entry is provided as a template (`model_id` is a
  placeholder — llama.cpp serves whatever GGUF is loaded); fill in your served id before running
- Decode-speed readout (**0.12.5 behavior note**) — llama.cpp is the one runtime whose rate
  the harness does **not** measure itself: `llama-server` streams `timings.predicted_per_second`,
  computed inside its own decode loop, and the speed ledger takes that. It is therefore exempt
  from the sample-substance floors 0.12.5 applies to client-measured runtimes (≥16 completion
  tokens over a ≥0.25 s window) — chunk-arrival timing can't distort an engine-side number.
  Practical upshot: llama.cpp populates the readout on short turns that vLLM/Ollama/LM Studio
  would skip

## Troubleshooting

- **Absolute paths only.** `binary:` and `model:` go straight into a subprocess call with no
  shell — a literal `~` is never expanded. Use a full path (`/home/you/...`), not `~/...`.
- **Multi-slot context division** (see above) — if the served context is smaller than you
  launched with, check for multiple slots eating your `-c` and add `--parallel 1`.
- **400s on tool calls** — MCP/plugin tool names like `mcp:fetch` violate the OpenAI
  function-name grammar; llama.cpp 400s the *entire* request if one is sent raw. The harness
  already sanitizes names on the wire and restores them on parse, so this should be invisible —
  if you see it anyway, confirm `--jinja` is set and that the GGUF's chat template actually
  declares tool-call syntax.
- **`-ngl` too high for the box** — an OOM at load shows up in the harness's serve log
  (`~/.localharness/vllm/serve.log` — yes, that directory is named `vllm` even for a
  llama.cpp-managed server; it's a shared, inherited path, not a bug).
- **`runtime: llamacpp` requires `binary`** — there's no PATH fallback like Ollama/LM Studio get;
  the config is rejected at load if it's missing.
- **Port defaults to 8081** (vLLM's convention) if you omit `port:` — always set it to `8080`
  explicitly for llama.cpp, as shown above.
