# Running LocalHarness on Ollama

*See the provider support matrix in the README for the full per-provider status.*

[Ollama](https://ollama.com) is a model-management daemon: it pulls, stores, and lazily loads
GGUF checkpoints behind its own API. LocalHarness runs the agent loop on top of Ollama's
OpenAI-compatible surface — Ollama serves the model, the harness is the thing calling it in a
loop, managing tools, and watching the context budget.

**Support status:** detection, the managed lifecycle (spawn-and-own the daemon), native
tool-calling, context-window discovery (the loaded model's served window, via `/api/ps`), and exact
token counting (from the model's own GGUF vocab) are all TESTED live.

## Install

The standard Ollama install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

(Or your distro's package.) This installs the `ollama` binary and, on most systems, a
`systemd` service. The daemon listens on `:11434`.

## Serve a model

```bash
ollama pull qwen2.5:7b
ollama serve   # only needed if the daemon isn't already running as a service
```

OpenAI-compatible surface at `http://127.0.0.1:11434/v1`. Native API: `/api/tags` (pulled
models), `/api/ps` (currently loaded models + their served context window), `/api/generate`
(the raw generation/load endpoint).

## Point LocalHarness at it

Two separate axes show up below: `provider_type` (the wire protocol — drives token counting) and
`runtime` (the launch strategy — how the harness starts/stops the process). They share the same
four names by coincidence, not because they're the same field.

### Attach (already running)

Point the primary provider at a daemon you already run yourself:

```yaml
# ~/.localharness/config.yaml
provider:
  provider_type: ollama
  base_url: http://127.0.0.1:11434/v1
  default_model: qwen2.5:7b
  api_key: none
```

Or add it as a peer the `/model` tree can switch to — the common case here is a CPU-light Ollama
coexisting with a GPU-heavy primary (e.g. vLLM):

```yaml
extra_endpoints:
  - name: ollama-local
    base_url: http://127.0.0.1:11434/v1
    provider_type: ollama
    gpu: false   # explicit — no inference from provider_type. CPU peers can coexist with a heavy primary.
```

### Harness-managed

Let the harness spawn and own the `ollama serve` daemon itself:

```yaml
server:
  runtime: ollama
  model: qwen2.5:7b
  port: 11434
  gpu: false   # true lets the daemon see the GPU (CUDA_VISIBLE_DEVICES unset instead of blanked)
  # binary: ollama   # optional — defaults to PATH `ollama` if omitted
```

Or as a cross-framework `/model`-tree peer the harness can cold-launch on the freed accelerator:

```yaml
extra_endpoints:
  - name: ollama-peer
    base_url: http://127.0.0.1:11434/v1
    provider_type: ollama
    gpu: true
    lifecycle:
      runtime: ollama
      model: gpt-oss:120b
      port: 11434
      gpu: true
```

## What the harness does

`DaemonStrategy` reuses `SpawnedProcessStrategy`'s launch/pidfile/stop primitives, with
Ollama-specific handling wrapped around them:

- **Activate** — fails fast if a daemon is already answering at the target address (the harness
  manages only daemons *it* starts — attaching to a pre-existing one isn't supported yet, so a
  stray `ollama serve` from a system service will block harness-managed mode until you stop it).
  Otherwise it spawns `ollama serve` with `OLLAMA_HOST`/`OLLAMA_KEEP_ALIVE=-1` set in its
  environment (plus `CUDA_VISIBLE_DEVICES=` when `gpu: false`), waits for the daemon to answer,
  pulls the model if it isn't already on disk, then **warm-loads** it with an empty-prompt
  `/api/generate` call — a load with no generation, so it respects giving the model room to think
  once real turns start.
- **Stop kills the whole daemon.** This is a deliberate ruling, not the default Ollama pattern: a
  soft `keep_alive: 0` unload is a *request* to free memory, not a verified guarantee. Killing the
  daemon process (SIGTERM, SIGKILL on timeout) is the same verified-free class as a container stop
  or a plain process kill — the daemon owns its own process group, so this also reaps the runner
  children actually holding the weights.
- **Liveness** — pid-based (`is the pidfile's pid alive?`). Correct here because the harness
  spawns the daemon itself as a tracked foreground process, not a docker client.

## Support status

- Detect (response-shape) — TESTED
- Init: model id — TESTED; context window — TESTED: read from the loaded model via `/api/ps`
  (`context_length`, the served window — not `/api/show`'s over-reporting ceiling). Ollama
  lazy-loads, so before the model is resident `init` falls back to config with an honest
  "approximate" flag; the in-session refit reads `/api/ps` once loaded
- Managed lifecycle (spawn+own `ollama serve`, warm-load, whole-daemon stop) — TESTED
  (live-proven, CPU, zero orphaned processes)
- Tool-calling (native) — TESTED (a real tool_call returned via qwen2.5:7b)
- Token counting — TESTED (exact): Ollama serves no `/tokenize`, so the harness loads the served
  model's own GGUF vocab + chat template in-process (`llama-cpp-python` vocab-only, the
  `exact-tokenizer` extra — install with `uv sync --extra exact-tokenizer`, or
  `pip install "localharness[exact-tokenizer]"`) and counts to the token — verified equal to
  Ollama's own `prompt_eval_count`. Falls back to a labeled approximate estimate only when no
  local GGUF is reachable
- Bench run — opt-in `bench.yaml` matrix entries are provided but UNVERIFIED (no recorded run in
  this repo)

## Troubleshooting

- **"an ollama daemon is already listening"** — harness-managed mode refuses to start if
  something is already bound to the target port, including your own system service. Stop it
  (`systemctl stop ollama` or equivalent) before using `server: runtime: ollama`, or use Attach
  mode instead and skip harness-managed entirely.
- **Port defaults to 8081** (vLLM's convention) if you omit `port:` — always set `port: 11434`
  explicitly, as shown above; it also drives `OLLAMA_HOST` for the spawned daemon.
- **First activate can be slow.** If the model isn't pulled yet, activate pulls it before
  warm-loading — a large model's pull time counts against the same activation timeout as the
  warm-load, so don't be surprised by a multi-minute first swap.
- **Logs and pidfile land under `~/.localharness/vllm/`** even though `runtime: ollama` — that
  directory name is a shared, inherited path, not a bug. Check `serve.log` there if activation
  times out.
- **`binary:` paths aren't `~`-expanded** — if you set it explicitly instead of relying on the
  PATH fallback, use an absolute path; the config passes it straight into a subprocess call with
  no shell.
