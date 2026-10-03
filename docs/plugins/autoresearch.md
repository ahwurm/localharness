# The autoresearch plugin

The `autoresearch` plugin is LocalHarness's self-improvement loop for experiments: a proposer model reads failed benchmark traces and proposes one change to the harness, a statistical gate tests it on the training scenarios and then on a sealed holdout set, and you decide what to adopt. It ships with LocalHarness and is on by default; it adds nothing to a normal session.

## Turn it on and off

```bash
localharness plugins disable autoresearch
localharness plugins enable autoresearch
```

Turning it off removes its commands from `localharness --help` and its settings from `components list`, and deletes nothing: the experiment archive and your settings stay. The bench and its sealed holdout are part of core and work either way.

`localharness plugins enable autoresearch` on a terminal, or `/plugins enable autoresearch` in a running session while no proposer is set up, asks the proposer's address and model id and writes both at once; a model equal to your main one is refused, and nothing is written. Then it asks the proposer for its model list once, to check that it answers and serves that model. It reads the address and key from your machine-level config only, as `propose` does, and prints the host before it sends the key; a different address in the project's own config is named and ignored. `doctor` still never contacts it.

## What it adds

- `localharness propose`: generate one typed change, a diff and its rationale, for one component, from failed training traces.
- `localharness experiment run`: put a proposal through the promotion gate (a Welch test on the training scenarios, then a Bonferroni-corrected test on the holdout).
- `localharness autoresearch run | review | adopt | report | sentinel | archive`.
- A `doctor` check naming the configured proposer model and address. It never shows the key and never contacts the endpoint.

While the plugin is off, running one of its commands prints how to turn it on and exits 4.

## Settings

It keeps the core settings it had before it became a plugin, under the same names. `proposer.base_url` and `proposer.api_key` are machine-level only, because the setup check sends the key to that address: a project's own value for either is ignored with a startup warning, and with no proposer address in your global config a project's whole `proposer:` section is ignored.

| Setting | Meaning |
|---|---|
| `proposer.base_url` | OpenAI-compatible address of the proposer model. |
| `proposer.model` | The proposer model; it must differ from `provider.default_model`. |
| `proposer.api_key` | Shown as `**********` everywhere; `none` for a local model. |
| `proposer.is_local`, `proposer.timeout_seconds`, `proposer.temperature`, `proposer.max_tokens` | How the proposer is called. |
| `sentinel.*` | Thresholds for the sentinel that watches the loop for overfitting, duplicate proposals and saturation. |

`propose` and `autoresearch run` need a proposer; `experiment`, `report` and `archive` do not.

## Not there yet

- `components set proposer.api_key …` writes the key as plain text to your global `overrides.yaml`.
- Two cases give a script an exit code that `experiment run` also uses for a verdict: when the config cannot be read, an off plugin's command gets `No such command` and exit 2 (reject-holdout); when an on plugin's command fails to import, it exits 1 (reject-train).
- With the plugin off, `components list` still shows its `autoresearch.enabled` row.

## More

- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [Spec 10, exit codes](../specs/10-cli.md#exit-codes)
- [localharness.dev/plugins/autoresearch](https://localharness.dev/plugins/autoresearch/)
- [Write your own plugin](../../examples/plugin-template/README.md)
