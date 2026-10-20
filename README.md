<p align="center">
  <img src="https://raw.githubusercontent.com/ahwurm/localharness/main/docs/assets/logo.svg" alt="LocalHarness logo" width="96" height="96">
</p>

# LocalHarness

[![GitHub stars](https://img.shields.io/github/stars/ahwurm/localharness?style=social)](https://github.com/ahwurm/localharness/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Run AI agents on the models you already run locally.**

LocalHarness is an open-source agent harness for local LLMs. It does not serve models: it connects to one you already serve over an OpenAI-compatible API (vLLM, llama.cpp, Ollama or LM Studio) and gives it agents, defined in YAML, with tools, a permission gate in front of every tool call, and helper agents that each work in a fresh context with only the tools they are allowed. A small core runs the agent loop; memory, the phone app, Discord, image generation, the self-improvement loop and a headless browser for JavaScript-rendered pages are plugins that you turn on or off with one command.

![LocalHarness: init detects your local model, start drops you into a ready agent, and it researches a question live with web search and multi-step tool calls](assets/demo.gif)

## Why LocalHarness

- **Built for local models, not ported to them.** The harness sizes a launch from the model's own files: it senses the served, trained and fitted context windows, prices a llama.cpp launch from the GGUF tensor table, keeps an MoE's experts in RAM when the GPU cannot hold them, refuses a model that fits nowhere in one sentence, and never runs below the window the agent loop needs. Tool calling works on a model without native support through an XML fallback, and `/effort off|low|high|max` bounds how much a model may think per request.
- **One command owns the model stack.** `localharness start` finds a model on your machine or offers one to fetch, writes it a tuned preset (KV cache type, drafting and parsers derived at launch), serves it, and brings back your last choice next time. `/model` lists every launch this box can make, each priced with its runtime, where its weights sit and its window, and switches in place. `localharness model --autostart on` brings the server up with the harness, `localharness model --script` prints the launch as a shell script you can read, and a turn survives a server restart.
- **Your data, your trace, your code.** MIT licensed end to end. Memory is SQLite in your home directory, config is YAML you can read, and every session rides one event bus whose stream, logs and bench results stay on your disk. Plugins, pre- and post-tool hooks and the phone app read the same events through the same public plugin API as the six bundled plugins. No account, no telemetry, no relay through anyone's servers; web search runs over DuckDuckGo with no key.
- **Safe by construction.** One permission gate stands in front of every tool call, helper agents included. An agent that reads untrusted content, a web page or an MCP result, is never given tools that change your machine; the check runs when an agent's tools resolve and fails closed. Where the harness connects, which credential it uses and who may talk to it are machine-level settings that a cloned repository cannot change. [SECURITY.md](SECURITY.md) is the full threat model.

## Quick start

Python 3.12 or later and [uv](https://docs.astral.sh/uv/). A model server is not required: the first `localharness start` finds a model on your machine or offers one to fetch, and sets everything up.

```bash
uv tool install 'localharness[embeddings,mobile]'  # memory's CPU model + the phone app
localharness start                   # finds a model (or offers one), sets up, opens a session

localharness plugins list            # every plugin and whether it is on
localharness plugins disable autoresearch
localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188
localharness update                  # the latest release from PyPI
```

Out of the box: agents in YAML with read, write, edit, glob, grep, bash, python, web search and fetch, delegation to helper agents and MCP servers. Drive a session from the terminal, from Zed ([`localharness acp`](docs/zed.md)), from your phone on your own network ([`localharness mobile`](docs/mobile.md)) or from Discord ([`--channel discord`](docs/plugins/dispatch.md)); with a vision model, drop a screenshot on the terminal, paste one into Zed or the phone page, or attach one on Discord.

`plugins enable` and `plugins disable` write `overrides.yaml`, never your `config.yaml`; `--workspace` changes a bundled plugin for one project only. Inside a session, `/plugins enable <name>` asks what the plugin needs, checks it, and restarts with your conversation kept. Multi-line input, Shift+Enter and the chords: [docs/terminal-keys.md](docs/terminal-keys.md).

## Supported runtimes

Every runtime is reached through one OpenAI-compatible client; `init` can also attach to any endpoint that is already running. Live-validated means a real end-to-end run on the [DGX Spark](docs/reference-architectures/dgx-spark.md) reference machine; recorded bench runs exist for vLLM only.

| Runtime | Harness starts the server | Tool calling | Token counting | Live-validated |
|---|---|---|---|---|
| **vLLM** | docker or binary | native | exact | reference machine and bench |
| **llama.cpp** | spawns `llama-server` | XML / Hermes fallback | exact | reference machine |
| **Ollama** | spawns `ollama serve` | native | exact with the `exact-tokenizer` extra, else estimated | CPU round-trip |
| **LM Studio** | drives headless `lms` | native | exact with the `exact-tokenizer` extra, else estimated | CPU round-trip |

A session with no temperature set sends none, so the server's own sampling defaults govern; pin one on the server (llama-server's `--temp`, vLLM's `--generation-config`). Setup: [llama.cpp](docs/runtimes/llamacpp.md), [Ollama](docs/runtimes/ollama.md), [LM Studio](docs/runtimes/lmstudio.md), vLLM ([reference architectures](docs/reference-architectures/README.md)); the code is [the provider layer](src/localharness/provider/).

## Plugins

| Name | What it does | Default | How to switch |
|------|--------------|---------|---------------|
| `image` | makes pictures with ComfyUI | off | `localharness plugins enable image --set comfyui_url=<url>` |
| `mobile` | Mobile: the phone page, served with its event API by `localharness mobile` | on; needs `localharness[mobile]` | `localharness plugins disable mobile` |
| `memory` | persistent memory: facts recalled into each turn, memory tools, background consolidation | on | `localharness plugins disable memory` |
| `dispatch` | chat: Discord | on; needs `localharness[dispatch]` | `localharness plugins disable dispatch` |
| `autoresearch` | experiment loop | on | `localharness plugins disable autoresearch` |
| `browser` | reads JavaScript-rendered pages through a sandboxed headless Chromium | off; needs `localharness[browser]` | `uv tool install 'localharness[browser]'` (name the other extras you use too: a reinstall keeps only the ones it names), then `localharness plugins enable browser` |

Pages: [image](docs/plugins/image.md) · [mobile](docs/plugins/mobile.md) · [memory](docs/plugins/memory.md) · [dispatch](docs/plugins/dispatch.md) · [autoresearch](docs/plugins/autoresearch.md) · [browser](docs/plugins/browser.md), and on the site: [image](https://localharness.dev/plugins/image/) · [mobile](https://localharness.dev/plugins/mobile/) · [memory](https://localharness.dev/plugins/memory/) · [dispatch](https://localharness.dev/plugins/dispatch/) · [autoresearch](https://localharness.dev/plugins/autoresearch/) · [browser](https://localharness.dev/plugins/browser/).

A plugin you install is found on your machine and never in a project: a Python package with a `localharness.plugins` entry point beside LocalHarness, or a folder `~/.localharness/plugins/<name>/`. None of its code is imported until `localharness plugins enable <name>`, which only your machine-level config can do. A plugin you turn on is trusted code that runs with your privileges: read [SECURITY.md](SECURITY.md#plugins) before you enable one.

## Write a plugin

Copy [the example plugin](examples/plugin-template/README.md): one tool, one command, one slash command, one `doctor` check and two settings, exercised by the test suite and in CI. The API is [`plugins/api.py`](src/localharness/plugins/api.py); its version is `"1"`, and changes since its first release have only added optional fields. Plugins written for the previous plugin API no longer load; [the example plugin](examples/plugin-template/README.md#plugins-written-for-015) says how to port one.

## Status

Early stage (v0.20.1, pre-1.0). Interfaces and config schema may change without notice. Each release's known limitations are listed in the [CHANGELOG](CHANGELOG.md).

## Links

- [SECURITY.md](SECURITY.md): trust boundaries, the permission gate, the prompt-injection threat model
- [src/localharness/](src/localharness/): the components, one package each; the CLI is [`cli/`](src/localharness/cli/)
- [docs/task-context.md](docs/task-context.md): how a session keeps a working record for substantive work, with an [example workflow](examples/workflows/research-note/)
- [docs/reference-architectures/](docs/reference-architectures/README.md): tested hardware and setup notes
- [docs/running-agents-locally.md](docs/running-agents-locally.md): where LocalHarness fits among local agent tools, and where it is behind
- [LocalShift](https://github.com/ahwurm/localshift): the companion project that moves a headless Claude Code job onto LocalHarness once a local model proves good enough
- [localharness.dev](https://localharness.dev)

## License

[MIT](LICENSE)
