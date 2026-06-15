# Examples

Runnable, validated examples of LocalHarness configuration.

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

See [CONTEXT-HARNESS.md](../CONTEXT-HARNESS.md) for how agents, divisions, and
the orchestrator fit together, and the tool model used by `tools.add` /
`tools.mcp_servers`.
