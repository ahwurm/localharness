# Contributing

Thanks for considering a contribution to LocalHarness.

## Dev setup

```sh
uv sync --extra dev                                    # Python 3.12+
uv run pytest                                          # hermetic — no model server needed
uv run localharness validate examples/agents/hn-monitor.yaml   # validates an agent YAML
```

The test suite is hermetic and needs no model server. A few bench scenarios read
fixtures from `/tmp/bench_fixtures/`; both pytest and `localharness bench run` stage
these automatically from `tests/fixtures/bench/` — no manual copy needed. Live-model
tests are opt-in:

```sh
LOCALHARNESS_LIVE_VLLM=1 uv run pytest -m live_vllm    # needs a local OpenAI-compatible endpoint
```

## Hard rules

- **Every change ships a test.** Behavior without a test is behavior we can't keep.
  `uv run pytest` must stay green.
- **One concern per PR.** Bundled changes get closed, not untangled.
- **Honest claims only.** No invented numbers, no "tested" without the output, no
  benchmark claims that didn't come from this repo's own runs. Exit codes, JSONL
  traces, and pytest output are the evidence standard — PRs meet the same bar.
- **The sealed holdout (`bench/scenarios/holdout/`) is never run while proposing.**
  The autoresearch proposer reads train traces only. Don't "fix" this.

## What we won't accept

- Drive-by AI slop: untested generated changes, blank-template PRs, reformatting noise.
- Feature sprawl outside scope. LocalHarness is the agent runtime — the YAML-defined
  hierarchy, tool dispatch, permissions, context management, and the harness benchmark.
  Per-request model routing, proxies, and fallback chains are out of scope.

## PR flow

1. Fork and branch from `main`.
2. Make the change with a test; keep `uv run pytest` green.
3. Open a PR using the template. Disclose AI assistance; paste real test output.
4. A maintainer reviews. CI runs the full suite on every push and pull request.

AI-generated or AI-assisted PRs are welcome and must disclose it — and pass the same
bar: real test evidence, one concern, honest claims.
