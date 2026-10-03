<p align="center">
  <img src="https://raw.githubusercontent.com/ahwurm/localharness/main/docs/assets/logo.svg" alt="LocalHarness logo" width="96" height="96">
</p>

# LocalHarness

[![GitHub stars](https://img.shields.io/github/stars/ahwurm/localharness?style=social)](https://github.com/ahwurm/localharness/stargazers) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Run AI agents on the models you already run locally.**

LocalHarness is an open-source agent harness for local LLMs. It does not serve models: it connects to one you already serve over an OpenAI-compatible API (vLLM, llama.cpp, Ollama or LM Studio) and gives it agents, defined in YAML, with tools, a permission gate in front of every tool call, and helper agents that each work in a fresh context with only the tools they are allowed. A small core runs the agent loop; memory, the phone app, Discord, image generation and the self-improvement loop are plugins that you turn on or off with one command.

![LocalHarness: init detects your local model, start drops you into a ready agent, and it researches a question live with web search and multi-step tool calls](assets/demo.gif)

## A small core, five plugins

- **Core** owns the agent loop, the built-in tools (read, write, edit, glob, grep, bash, python, web search and fetch, delegation to helper agents), MCP servers, config, the permission gate, the bench and the CLI.
- **Plugins** add tools, commands, slash commands, `doctor` checks, settings and chat channels through one plugin API. The five that ship with LocalHarness use the same API as a plugin you write yourself. Turning one off removes what it adds; turning off memory or autoresearch leaves their files on disk (the memories, the experiment archive).

## Quick start

You need Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and a model server running.

```bash
uv tool install 'localharness[embeddings,web]'  # memory's CPU model + the phone app
localharness init                    # finds your model server, writes your config
localharness start                   # an interactive session

localharness plugins list            # every plugin and whether it is on
localharness plugins disable autoresearch
localharness plugins enable image --set comfyui_url=http://127.0.0.1:8188
```

`plugins enable` and `plugins disable` write `overrides.yaml`, never your `config.yaml`. Add `--workspace` to change a bundled plugin for one project only. Add the `dispatch` extra for Discord.

`init` ends with every plugin and the command that sets it up. Inside a running session, `/plugins enable <name>` does the same and turns the plugin on right away: the first time, or while the plugin is not set up yet, it asks what the plugin needs and checks it, then it restarts the session with your conversation kept. A first `localharness start` with no config runs `init`'s setup and goes on into the session.

## Supported runtimes

Every runtime is reached through one OpenAI-compatible client; `init` can also attach to any endpoint that is already running. Live-validated means a real end-to-end run on the [DGX Spark](docs/reference-architectures/dgx-spark.md) reference machine; recorded bench runs exist for vLLM only.

| Runtime | Harness starts the server | Tool calling | Token counting | Live-validated |
|---|---|---|---|---|
| **vLLM** | docker or binary | native | exact | reference machine and bench |
| **llama.cpp** | spawns `llama-server` | XML / Hermes fallback | exact | reference machine |
| **Ollama** | spawns `ollama serve` | native | exact with the `exact-tokenizer` extra, else estimated | CPU round-trip |
| **LM Studio** | drives headless `lms` | native | exact with the `exact-tokenizer` extra, else estimated | CPU round-trip |

Setup: [llama.cpp](docs/runtimes/llamacpp.md), [Ollama](docs/runtimes/ollama.md), [LM Studio](docs/runtimes/lmstudio.md), vLLM ([reference architectures](docs/reference-architectures/README.md)). Details: [spec 13](docs/specs/13-provider-support.md).

## Security

An agent that reads untrusted content, such as a web page or an MCP tool's result, is never given tools that change your machine; this separation is on by default, checked when an agent's tools are resolved, and fails closed. Every tool call of every agent, helper agents included, passes one permission gate. A plugin you install has its tools' gate declarations clamped, except for a tool it registers directly on the tool registry instead of returning it from `tools()`, which skips that rule ([SECURITY.md](SECURITY.md#plugins)). Settings that say where the harness connects, which credential it uses or who may talk to it can be set only in your machine-level config, never by a repository you clone. A plugin you turn on is trusted code that runs with your privileges, and nothing contains a plugin that means harm: read [SECURITY.md](SECURITY.md) before you enable one.

## Plugins

| Name | What it does | Default | How to switch |
|------|--------------|---------|---------------|
| `image` | makes pictures with ComfyUI | off | `localharness plugins enable image --set comfyui_url=<url>` |
| `web` | Mobile: the phone page, served with its event API by `localharness web` | on; needs `localharness[web]` | `localharness plugins disable web` |
| `memory` | persistent memory: facts recalled into each turn, memory tools, background consolidation | on | `localharness plugins disable memory` |
| `dispatch` | chat: Discord | on; needs `localharness[dispatch]` | `localharness plugins disable dispatch` |
| `autoresearch` | experiment loop | on | `localharness plugins disable autoresearch` |

Each plugin has its own page:

- **image**: a `generate_image` tool and `localharness generate-image`, against a ComfyUI server you run yourself. [Page](docs/plugins/image.md) · [site](https://localharness.dev/plugins/image/) · [Docs](docs/reference-architectures/image-generation.md)
- **Mobile (the `web` plugin)**: drive a session from your phone on your own private network. [Page](docs/plugins/mobile.md) · [site](https://localharness.dev/plugins/mobile/) · [Docs](docs/web.md)
- **memory**: per-agent SQLite memory recalled into each turn, with idle consolidation. [Page](docs/plugins/memory.md) · [site](https://localharness.dev/plugins/memory/) · [Docs](docs/specs/05-memory.md)
- **dispatch**: `localharness start --channel discord` drives a session from allowlisted Discord messages. [Page](docs/plugins/dispatch.md) · [site](https://localharness.dev/plugins/dispatch/) · [Docs](docs/specs/11-channels.md#the-dispatch-plugin-chat-platforms-discord-today)
- **autoresearch**: propose one harness change, run it through a statistical gate, adopt or reject it. [Page](docs/plugins/autoresearch.md) · [site](https://localharness.dev/plugins/autoresearch/) · [Docs](docs/specs/09-hooks-plugins.md#the-bundled-plugins)

A plugin you install is found in two places, both on your machine and never in a project: a Python package with a `localharness.plugins` entry point in the same environment as LocalHarness, or a folder `~/.localharness/plugins/<name>/`. It stays off, with none of its code imported, until `localharness plugins enable <name>`, which only your machine-level config can do.

## Write a plugin

Copy [the example plugin](examples/plugin-template/README.md): one tool, one command, one slash command, one `doctor` check and two settings, exercised by the test suite and in CI. The API is [spec 09](docs/specs/09-hooks-plugins.md); its version is `"1"`, and changes since its first release have only added optional fields. Plugins written for the previous plugin API no longer load; [spec 09](docs/specs/09-hooks-plugins.md#plugins-written-for-015) says how to port one.

## Status and known limitations

Early stage (v0.16.0, pre-1.0). Interfaces and config schema may change without notice. The limits a new user is most likely to meet:

- The Discord plugin is tested offline against a stand-in for the Discord library and has not been run against a live Discord server. Files people upload to the bot are not passed to the model.
- The phone app ships a bare reference page, not a finished chat app: one live session at a time, no chat list, no resume. Its incognito switch only keeps pictures off the phone; memory, sessions and pictures are still written to disk.
- The image plugin does not install or start ComfyUI or download its model files. The terminal does not display pictures; the phone page and Discord do.
- With memory on, `localharness doctor` fails until the `embeddings` extra and the embedding model are installed.
- The Discord token and the autoresearch proposer's API key are stored as plain text in your global `overrides.yaml`.
- `pre_tool` and `post_tool` hooks do not fire for a helper agent's tool calls, and a hook written as `async def` never runs.
- `localharness validate` does not check a plugin's own settings, and `plugins enable` and `plugins disable` write no audit event.
- `/plugins enable` and `/plugins disable` work in a terminal session only. The restart keeps the conversation, but not a `/model` switch to another endpoint, `/reasoning` and `/verbose`, or an MCP server's own state; and if the model server goes away during the restart, the conversation is lost.

The full list is under "Known limitations" in each [CHANGELOG](CHANGELOG.md) release.

## Links

- [SECURITY.md](SECURITY.md): trust boundaries, the permission gate, the prompt-injection threat model
- [docs/specs/](docs/specs/): component specs, starting with the [architecture overview](docs/specs/00-architecture-overview.md); the CLI is [spec 10](docs/specs/10-cli.md)
- [docs/zed.md](docs/zed.md): use LocalHarness from Zed's agent panel
- [docs/reference-architectures/](docs/reference-architectures/README.md): tested hardware and setup notes
- [docs/running-agents-locally.md](docs/running-agents-locally.md): where LocalHarness fits among local agent tools, and where it is behind
- [LocalShift](https://github.com/ahwurm/localshift): the companion project that moves a headless Claude Code job onto LocalHarness once a local model proves good enough
- [localharness.dev](https://localharness.dev)

## License

[MIT](LICENSE)
