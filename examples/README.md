# Examples

Runnable, validated examples of LocalHarness configuration, and a plugin to copy.

## `agents/hn-monitor.yaml`

A small read-only agent that turns raw Hacker News items into a tight,
link-preserving digest. It demonstrates the anatomy of every agent: identity
and role, model knobs, inherited tool access with a `deny` (no shell),
deny-first permissions, and capability keywords for orchestrator routing.

Validate it any time — no model server needed:

```bash
uv run localharness validate examples/agents/hn-monitor.yaml
```

To run it for real, point LocalHarness at your local model and make the agent
available to your config directory:

```bash
uv run localharness init                              # detect your endpoint + model
cp examples/agents/hn-monitor.yaml ~/.localharness/agents/
uv run localharness start                             # interactive session
```

See [docs/specs/](../docs/specs/) for how agents, divisions, and the orchestrator
fit together, and the tool model used by `tools.add` / `tools.mcp_servers`.

## `plugin-template/`

The copyable plugin. It adds one of each thing a plugin can add: a declared tool that saves an image
artifact, a command, a slash command, a `doctor` check and its own settings. It is also a fixture:
the development install includes it and LocalHarness's tests exercise it, so it cannot drift from the
plugin API.

Install it and see it listed — no model server needed:

```bash
uv sync --extra dev                          # installs the example plugin with the dev tools
uv run localharness plugins list             # "example" shows as available, not yet on
```

To turn it on and use it:

```bash
uv run localharness plugins enable example   # a machine-level setting
uv run localharness init                     # first time on this machine: needs your model server
uv run localharness start                    # ask for a swatch
```

The agent can tell you where the swatch was saved: the tool's result names the file.

See [plugin-template/README.md](plugin-template/README.md) for what each part does and a checklist
for copying it.

## `workflows/research-note/`

A fictional customer-profile research note for trying the working record on substantive work: an
instruction file with stages and checks, two sources, a voice sample, a deterministic lint, and a
read-only reviewer plus a writer. Copy the folder outside this repository first, then make the
specialists available, because LocalHarness loads project agents from `.localharness/agents/`:

```bash
cp -r examples/workflows/research-note ~/research-note && cd ~/research-note
mkdir -p .localharness/agents && cp agents/*.yaml .localharness/agents/
localharness start
```

See [the example workflow](../docs/task-context.md#the-example-workflow) for what to ask and what
the record should show.
