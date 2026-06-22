# LocalHarness

[![GitHub stars](https://img.shields.io/github/stars/ahwurm/localharness?style=social)](https://github.com/ahwurm/localharness/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**An open-source agent harness for local LLMs** — run AI agents on local models, defined in YAML, against any OpenAI-compatible endpoint. LocalHarness is the *agent layer* that runs on top of your inference engine (vLLM, Ollama, LM Studio, llama.cpp) — not another inference engine.

It's model-agnostic and hierarchical: define agents in YAML — system prompt, tools, permissions, memory — and run them as a coordinated org (orchestrator → divisions → agents) against any OpenAI-compatible local endpoint. The thesis: the harness, not the model, is where most of the capability lives — the same model can swing tens of benchmark points depending on the harness around it.

![LocalHarness — init detects your local model, start drops you into a ready agent, and it researches a question live with web search and multi-step tool calls](assets/demo.gif)

> `localharness init` auto-detects your running endpoint (here, vLLM serving Qwen) and probes its tool-calling. Then `localharness start` is zero-config: it creates a default general-purpose agent and drops you straight into the REPL. Ask it a real question and watch the agent work — here it chains `web_search` → `web_fetch` across several iterations to research the best open-source model for a 128 GB machine, the tool-call loop visible the whole way.

## Why local

Frontier coding agents are great when you're sitting there driving them, but the metering and rate limits make them an awkward fit for the routine, recurring jobs you'd actually want an agent to own: the nightly report, the scheduled cleanup, the watch-and-react task. LocalHarness keeps the Claude Code / OpenCode workflow you already know and points it at a model running on hardware you control.

- **No metering.** A job that fires every hour runs on hardware you already own, with no per-token bill.
- **Your data stays put.** Code, files, and prompts never leave the machine.
- **Always on.** No quota or rate caps to budget around for unattended runs.
- **Familiar.** Same agent, tool, and permission model as the cloud tools, just local.

A frontier agent like Claude Code is still the easy way to set the harness up and compose a bespoke subagent for a task. The split that works: frontier to design, local to run.

**Migrating existing headless work?** [LocalShift](https://github.com/ahwurm/localshift) is the companion project. Point Claude Code at a cron job, skill, or bare prompt and it builds a per-workload quality eval, proves the local model is good enough (or honestly says keep-frontier), then cuts the job over to run claude-free on LocalHarness.

## Features

- **YAML-defined agents** — add an agent, division, or tool policy without writing Python
- **Event-bus core** — components communicate via a typed event stream, persisted as append-only JSONL per agent
- **Isolated memory per agent** — SQLite-backed, scoped per agent
- **Deny-first permissions** — policies inherit down the hierarchy and can only narrow
- **Tool-call fallback** — native function calling where the model supports it, XML/Hermes fallback where it doesn't
- **MCP support** — connect Model Context Protocol servers and expose their tools to agents
- **Built-in tools** — read, write, edit, glob, grep, bash, python, web search/fetch, and subagent delegation
- **Benchmark suite** — scenario corpus in `bench/` for measuring harness changes against your own model
- **Autoresearch loop** — propose → gate → promote mutation archive for harness self-improvement experiments
- **Pluggable channels** — CLI today; Discord adapter in development

## How it compares

LocalHarness is an *agent layer* — not an inference engine, and not a cloud SaaS. It sits on top of whatever serves your model and gives that model agents, tools, memory, and permissions.

| | What it is | LocalHarness relationship |
|---|---|---|
| **Ollama / vLLM / LM Studio / llama.cpp** | Inference engines — they *serve* a model over an API | LocalHarness runs on top; point it at their endpoint |
| **Cloud agent frameworks** (hosted assistants / SaaS) | Agents that run against a vendor's metered API | Same agent / tool / permission model, but against a model on *your* hardware — no metering, data stays local |
| **Agent libraries** (write-your-own in Python) | Code-first SDKs for building agents | Config-first: agents, divisions, and permissions in YAML, no Python required |

If you already serve a model with Ollama or vLLM and want to run real agents against it — with tools, isolated memory, and deny-first permissions — that's the gap LocalHarness fills.

## Requirements

- Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/)
- A local LLM server with an OpenAI-compatible API (vLLM, Ollama, LM Studio, or llama.cpp)

## Quick start

```bash
git clone https://github.com/ahwurm/localharness.git
cd localharness
uv sync

uv run localharness init    # probes vLLM :8000, Ollama :11434, LM Studio :1234, llama.cpp :8080
uv run localharness start   # interactive session
```

`init` detects your endpoint and models, probes tool-calling capability, and writes `~/.localharness/config.yaml`. Non-standard setup: `localharness init --endpoint http://host:port/v1`. A repo-local `.localharness/` directory overlays the global config.

> Got it running? If LocalHarness saved you an API bill, a [star](https://github.com/ahwurm/localharness/stargazers) helps other local-LLM folks find it.

### Running the harness on a different machine than the model

The harness and the model server are separate processes talking HTTP — they don't need to
share a machine. A laptop can run agents against a model served elsewhere on your network:
`localharness init --endpoint http://<server-ip>:8000/v1`. Two things to know:

- **Tools run where the harness runs.** bash/file tools execute on the client machine; the
  model server only sees text in, text out. Pointing a harness at a server doesn't let
  anyone act on the server.
- **Secure the endpoint.** Inference servers ship with no authentication by default. On a
  network with untrusted devices, start the server with an API key (e.g. vLLM `--api-key`)
  and set `provider.api_key` to match; for access from outside your LAN use a private
  overlay network (Tailscale/WireGuard). Never port-forward a bare endpoint to the internet.

## CLI

| Command | Purpose |
|---------|---------|
| `init` | Detect endpoint/model, write config |
| `start` | Interactive session |
| `doctor` | Diagnose config/endpoint issues |
| `validate` | Validate agent/org YAML |
| `agent …` | Manage agent definitions |
| `bench …` | Run the scenario benchmark |
| `components …` | Autoresearch component registry |
| `autoresearch …` | Run the self-improvement loop |
| `experiment …` | Gated experiment runs |
| `propose` | Propose a harness mutation |

## Testing

```bash
uv sync --extra dev
uv run pytest                                          # hermetic — no model server needed
LOCALHARNESS_LIVE_VLLM=1 uv run pytest -m live_vllm    # opt-in tests against a live endpoint
```

Some bench scenarios read fixture files from `/tmp/bench_fixtures/`. pytest stages these automatically from `tests/fixtures/bench/`; before standalone `bench run` invocations, run the test suite once or copy that directory there yourself.

## Reference architectures

LocalHarness is developed against two maintainer-tested hardware targets. Both must meet
the practicality bar — **64k of KV-cache headroom and ≥9.5 tok/s single-stream** — with
the newest Qwen model that fits it:

| | Hardware | Model / Runtime | Status |
|---|---|---|---|
| [A: DGX Spark](docs/reference-architectures/dgx-spark.md) | GB10, 128 GB unified | Qwen3.6-27B NVFP4 / vLLM, 64k ctx, 9.5 tok/s | TESTED |
| [B: Base Mac mini](docs/reference-architectures/mac-mini.md) | M4, 16 GB unified | Qwen3.5-9B 4-bit / vLLM (vllm-metal), 64k ctx | PROPOSED |

Start at [docs/reference-architectures/](docs/reference-architectures/README.md);
known out-of-box gaps are tracked in [gaps.md](docs/reference-architectures/gaps.md).

## Documentation

- [docs/reference-architectures/](docs/reference-architectures/README.md) — supported hardware targets and gaps
- [docs/specs/](docs/specs/) — component specs

## Status

Early stage (v0.1). Interfaces and config schema may change without notice.

## License

[MIT](LICENSE)
