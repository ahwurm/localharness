# LocalHarness

Model-agnostic hierarchical agent harness for local LLMs.

Define agents in YAML — system prompt, tools, permissions, memory — and run them as a coordinated org (orchestrator → divisions → agents) against any OpenAI-compatible local endpoint: vLLM, Ollama, LM Studio, llama.cpp. The thesis: the harness, not the model, is where most of the capability lives — the same model can swing tens of benchmark points depending on the harness around it.

## Features

- **YAML-defined agents** — add an agent, division, or tool policy without writing Python
- **Event-bus core** — components communicate via a typed event stream, persisted as append-only JSONL per agent
- **Isolated memory per agent** — SQLite-backed, scoped per agent
- **Deny-first permissions** — policies inherit down the hierarchy and can only narrow
- **Tool-call fallback** — native function calling where the model supports it, XML fallback where it doesn't
- **Benchmark suite** — scenario corpus in `bench/` for measuring harness changes against your own model
- **Autoresearch loop** — propose → gate → promote mutation archive for harness self-improvement experiments
- **Pluggable channels** — CLI today; Discord adapter in development

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

## Documentation

- [CONTEXT-HARNESS.md](CONTEXT-HARNESS.md) — architecture and design rationale
- [CONTEXT-MODEL.md](CONTEXT-MODEL.md) — model selection notes for the reference hardware (NVIDIA DGX Spark)
- [docs/specs/](docs/specs/) — component specs

## Status

Early stage (v0.1). Interfaces and config schema may change without notice.

## License

[MIT](LICENSE)
