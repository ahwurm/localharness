# Running LocalHarness on LM Studio

*See the provider support matrix in the README for the full per-provider status.*

[LM Studio](https://lmstudio.ai) runs headless via its `lms` CLI and a persistent `llmster`
daemon — no GUI required. LocalHarness runs the agent loop on top of LM Studio's
OpenAI-compatible server; LM Studio owns the model and the engine, the harness drives `lms` to
bring that server up and down.

**Support status:** detection, the managed lifecycle, native tool-calling, and exact token counting
(from the model's own GGUF vocab) are all TESTED live (certified on an NVIDIA DGX Spark, CPU-mode).

## Install

Headless install (no GUI needed):

```bash
export PATH=/usr/sbin:$PATH   # lms's installer needs ldconfig on PATH first
curl -fsSL https://lmstudio.ai/install.sh | bash
```

This installs the native `lms` and `llmster` binaries to `~/.lmstudio/bin` — add that to your
PATH. NVIDIA is an LM Studio launch partner, so Linux aarch64 (the DGX Spark's architecture) is
supported; note LM Studio's own docs currently call Ubuntu >22 "not well tested" — in practice
it's been validated here on Ubuntu 24.04.

## Serve a model

Get a model — the staff-pick hub search takes a short name:

```bash
lms get -y qwen2.5-7b-instruct
```

For an arbitrary HuggingFace GGUF, the hub proxy needs the **full URL** (bare repo ids don't
resolve):

```bash
lms get -y https://huggingface.co/lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF
```

Then bring the server up manually:

```bash
lms daemon up
lms load <model-key> --gpu off -c 4096 -y   # --gpu off = CPU, --gpu max = full GPU offload
lms server start --port 1234 --bind 127.0.0.1
```

OpenAI-compatible surface at `http://127.0.0.1:1234/v1`; native REST at `/api/v0/models`. Stop
with `lms daemon down` — this stops the *whole daemon*, not just the HTTP listener (see below).

## Point LocalHarness at it

Two separate axes show up below: `provider_type` (the wire protocol — drives token counting) and
`runtime` (the launch strategy — how the harness starts/stops the process). They share the same
four names by coincidence, not because they're the same field.

### Attach (already running)

Point the primary provider at a server you already started with `lms`:

```yaml
# ~/.localharness/config.yaml
provider:
  provider_type: lmstudio
  base_url: http://127.0.0.1:1234/v1
  default_model: qwen2.5-7b-instruct   # the loaded model key
  api_key: none
```

Or add it as a peer the `/model` tree can switch to:

```yaml
extra_endpoints:
  - name: lmstudio-local
    base_url: http://127.0.0.1:1234/v1
    provider_type: lmstudio
    gpu: false   # explicit — no inference. true if this peer holds the accelerator.
```

### Harness-managed

Let the harness drive `lms` itself — daemon up, load, server start:

```yaml
server:
  runtime: lmstudio
  binary: /home/you/.lmstudio/bin/lms   # absolute path — see Troubleshooting
  model: qwen2.5-7b-instruct
  port: 1234
  gpu: false   # derives `lms load --gpu off`; true -> --gpu max. NEVER put --gpu in extra_args.
```

Or as a cross-framework `/model`-tree peer the harness can cold-launch on the freed accelerator:

```yaml
extra_endpoints:
  - name: lmstudio-peer
    base_url: http://127.0.0.1:1234/v1
    provider_type: lmstudio
    gpu: true
    lifecycle:
      runtime: lmstudio
      binary: /home/you/.lmstudio/bin/lms
      model: qwen2.5-7b-instruct
      port: 1234
      gpu: true
```

## What the harness does

`LmsStrategy` drives the headless `lms` CLI directly — it never touches vLLM's pidfile/log
machinery, because every `lms` subcommand backgrounds itself and returns immediately (there's no
foreground process to track):

- **Activate** — fails fast if a server is already answering at the target endpoint (the harness
  manages only servers *it* starts). Otherwise: `lms daemon up` → `lms load <model> --gpu
  {off|max}` (derived from your config's `gpu:` flag) → `lms server start --port <port> --bind
  127.0.0.1` → polls until the OpenAI endpoint actually answers.
- **Stop is `lms daemon down` — not `lms server stop`.** Settled empirically: `lms server stop`
  only closes the HTTP listener while `llmster` keeps the model resident (a soft signal, the LM
  Studio analog of Ollama's `keep_alive: 0`). `lms daemon down` reaps the whole daemon and engine —
  the verified GPU-free teardown. Both the listener and `lms daemon status` flip to "down" *before*
  the process holding the weights actually exits, so neither is trustworthy alone — the harness
  polls for the `llmster` process itself to be gone (by name; there's no pidfile) and raises rather
  than report a false accelerator-free if it's still alive after a few seconds.
- **Liveness** — endpoint-based (`GET {base_url}/models`), since there's no pidfile to check.

## Support status

- Detect (response-shape) — TESTED
- Init: model + context window (`loaded_context_length` from `/api/v0/models`) — TESTED
- Managed lifecycle (daemon up/load/server start, verified `lms daemon down` stop) — TESTED
  (live-certified on a DGX Spark, CPU-mode, 2026-07-29)
- Tool-calling (native OpenAI function-calling) — TESTED (confirmed live: a real tool_call
  returned)
- Token counting — TESTED (exact): LM Studio serves no tokenize endpoint, so the harness loads the
  served model's own GGUF vocab + chat template in-process (`llama-cpp-python` vocab-only, the
  `exact-tokenizer` extra — install with `uv sync --extra exact-tokenizer`, or
  `pip install "localharness[exact-tokenizer]"`) and counts to the token — verified equal to
  LM Studio's own `usage.prompt_tokens`. Falls back to a labeled approximate estimate only when
  no local GGUF is reachable
- Bench run — an opt-in `bench.yaml` matrix entry is provided; no bench run is recorded in this
  repo yet (pull your own model + run it)

## Troubleshooting

- **Installer needs `ldconfig` on PATH** — if the install script fails oddly, run
  `export PATH=/usr/sbin:$PATH` first, then retry.
- **`lms` isn't on PATH by default.** Either add `~/.lmstudio/bin` to your shell's PATH, or set
  `binary:` to the full path. Either way, use an **absolute path** if you set `binary:` explicitly
  — it's passed straight into a subprocess call with no shell, so a literal `~/...` is never
  expanded.
- **Never put `--gpu` in `extra_args`.** The config loader rejects it outright at load time — the
  flag is derived from `gpu:` (`--gpu max`/`--gpu off`), and a manual one would silently desync the
  GPU-lock bookkeeping and risk two heavy servers up at once on a later swap.
- **A stop that raises "still running after `lms daemon down`"** means the harness is refusing to
  report a false accelerator-free — check `lms ps` / `pgrep llmster` before retrying, don't just
  relaunch on top of it.
- **Driving `lms` manually alongside the harness?** Follow `lms server stop` with `lms daemon
  down` — the first alone leaves the model resident even though the port looks closed.
- **No pidfile.** LmsStrategy never writes `~/.localharness/vllm/server.pid` — check `lms ps` /
  `lms daemon status` directly instead.
- An install issue on ARM64 hardware other than a DGX Spark may be genuinely uncharted territory
  upstream (see the Ubuntu >22 caveat above), not a LocalHarness bug.
