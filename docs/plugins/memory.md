# The memory plugin

The `memory` plugin gives each agent a persistent SQLite memory: facts are recalled into every turn's prompt, the agent can search and write memory with three tools, and while the session is idle a consolidation pass strengthens what proved relevant and groups related facts. It ships with LocalHarness and is on by default; similarity is judged by a small local embedding model that runs on CPU, so install the `embeddings` extra.

## Turn it on and off

```bash
uv tool install 'localharness[embeddings]'   # the embedding model's package
localharness plugins disable memory           # or per project: add --workspace
localharness plugins enable memory
```

The switch is `memory.enabled`. With memory off, no memory store is opened, no memory tool or `/memory` command exists, and memory files already on disk stay.

`localharness plugins enable memory` on a terminal, or `/plugins enable memory` in a running session, asks no questions about settings. While the embedding model is missing, it asks "Download the embedding model now (about 1.2 GB)?" and on yes downloads it into the Hugging Face cache; once the model is there, it asks nothing. Without the `embeddings` package it still asks, then downloads nothing and names the install line.

## What it adds

- Tools: `memory_search`, `memory_get` and `remember`.
- The `Division Context` and `Agent Memory` sections of each turn's prompt, inside the context budget core gives it.
- `/memory` in a session: browse by tag, show, search, forget, and `promote` one fact from a project's memory to your machine-global memory.
- `localharness memory list | show | edit | rm | archive | restore`.
- `doctor` checks: each memory database, opened read-only, and whether the embedding model is in the local cache. Nothing is loaded or downloaded.

Inside a project with a `.localharness/` folder, memory lives with the project, so projects do not read each other's memories.

## Settings

All under `agent.memory.*` in an agent's file or `overrides.yaml`; none is machine-level only. The ones most people touch:

| Setting | Meaning |
|---|---|
| `memory.enabled` | Turns the plugin on or off. |
| `agent.memory.recall_scope` | `workspace` (default), `global` or `both`: which store recall reads in a project. Writes always go to the session's own store. |
| `agent.memory.consolidation.enabled` | Idle consolidation. Default on. |
| `agent.memory.archival.enabled` | Moves facts that stopped earning their place to an archive, restorable with `localharness memory restore`; nothing is deleted. Default off. |
| `agent.memory.embedding_model` | Default `Qwen/Qwen3-Embedding-0.6B`. |

`localharness components list` shows every setting, marked `(plugin: memory)`.

## Not there yet

- With memory on, `localharness doctor` fails until the `embeddings` extra and the embedding model are installed.
- A helper agent's cruncher run gets no write access to memory, so its gists are not stored.
- What `remember` returns is treated as trusted text, while what `memory_search` and `memory_get` return is fenced as untrusted.
- `agent.memory.inject_into_context` is accepted but does nothing.
- Five memory event types (`MemoryGateFired`, `ExpectationAttached`, `OutcomeObserved`, `SurpriseScored`, `TurnEndMicroPassCompleted`) are defined, but nothing sends them.
- The phone's memory screen has no promote button. A promoted fact is not filed in the destination's tag tree.
- The spec lists further honest limits of recall and capture.

## More

- [Spec 05, memory](../specs/05-memory.md) and its [honest limits](../specs/05-memory.md#10-honest-limits-named-not-hidden)
- [Spec 09, the memory slot](../specs/09-hooks-plugins.md#the-memory-slot): how another memory plugin can take its place
- [localharness.dev/plugins/memory](https://localharness.dev/plugins/memory/)
- [Write your own plugin](../../examples/plugin-template/README.md)
