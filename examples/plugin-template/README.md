# The example plugin

A small, real LocalHarness plugin that you can copy. It adds one of each thing a plugin can add:

- a tool, `example_swatch`, that renders a solid-color square PNG and saves it as an artifact
- a command, `localharness example`
- a slash command, `/example`, inside a session
- a check in `localharness doctor`
- two settings, `example.color` and `agent.example.size`

LocalHarness's development install includes this plugin, and its test suite exercises each of those.
So it cannot quietly fall behind the plugin API: if the API changes, the tests fail until this example
is updated.

## Try it

From a checkout of the LocalHarness repository:

```bash
uv sync --extra dev                          # installs this plugin along with the dev tools
uv run localharness plugins list             # "example" is listed as available, not yet on
uv run localharness plugins enable example   # turn it on (writes your machine-level overrides.yaml)
uv run localharness example                  # the plugin's own command
uv run localharness init                     # first time on this machine: needs your model server
uv run localharness doctor                   # its check shows in the plugins section
uv run localharness start                    # ask for a swatch; type /example to see its settings
```

`plugins list`, `plugins enable` and `localharness example` need no model server and no config.
`init` writes the config that `doctor` and `start` read, and `init` and `start` need your model
server running: without it, `init` writes nothing. Ask for a swatch and the agent can tell you the
file it saved (see [Artifacts](#artifacts)). To turn the plugin off again:
`uv run localharness plugins disable example`.

## How LocalHarness finds a plugin

There are two ways, and both are on your machine, never in a project:

1. An installed package that declares a `localharness.plugins` entry point, in the same Python
   environment as LocalHarness. This plugin's `pyproject.toml` declares:

   ```toml
   [project.entry-points."localharness.plugins"]
   example = "localharness_plugin_example:ExamplePlugin"
   ```

2. A folder `~/.localharness/plugins/<name>/` whose `__init__.py` binds the name `plugin` to the
   plugin class: `plugin = YourPlugin`.

Finding a plugin is not running it. LocalHarness lists what it found from package metadata and folder
names alone, and imports a plugin's code only once that plugin is enabled. Enabling a plugin you
brought yourself is a machine-level setting (`localharness plugins enable <name>`, never per
project), and it is you vouching for what the plugin says its tools do. Plugin code runs inside the
harness with your privileges, so treat it like any program you install.

## The four declarations every tool makes

LocalHarness's safety checks read what a tool declares, never its name. Every declaration has a
default, and every default assumes the worst, so a tool that declares nothing is treated as the
riskiest kind on all four:

- `ingest`: does the tool bring outside content (web pages, files, another service's replies) into
  the conversation? `"none"`, or `"untrusted"` (the default). An agent that holds a tool which
  ingests untrusted content is kept from also holding tools that can change your machine, so text
  injected into a page never reaches, word for word, an agent that can run commands. A summary of
  it still can; SECURITY.md names that gap.
- `host`: can it change your machine (run commands, write files wherever it likes)? `"safe"`, or
  `"dangerous"` (the default).
- `result_origin`: is its result text your code wrote (`"trusted"`), or outside content
  (`"untrusted"`, the default)? An untrusted result is stored marked as outside content. The agent
  can read it back later, but the step that runs code over stored results (the cruncher) refuses
  to take it as input.
- `gate_family`: which rule of the permission gate decides whether to ask you before the tool runs:
  `"write"`, `"shell"`, `"code"`, `"delegate"`, `"network"` or `"allow"`. The default, no family,
  is asked about once per workspace in `guarded` mode. For a plugin you install, a declared family
  counts only if it asks at least as often as no family would, so a plugin cannot declare its way
  out of being asked.

`example_swatch` declares `ingest="none"`, `host="safe"`, `result_origin="trusted"` and no gate
family: it reads nothing from outside, writes only into the folder core gives it, and returns text
it wrote itself.

## Settings

A plugin's settings live under its name, and core checks them with the plugin's own models before
the plugin sees them:

- `ConfigModel` (here `ExampleConfig`) checks the harness-level section `example:`.
  `example.color` is the swatch color as `#rrggbb` (default `#4a90d9`). Like core's settings, it can
  be set for the whole machine or for one project.
- `AgentConfigModel` (here `ExampleAgentConfig`) checks the agent-level section `agent.example`.
  `agent.example.size` is the swatch's width and height in pixels (default 8, from 1 to 256).

```bash
uv run localharness components set example.color "#d94a4a"
uv run localharness components set agent.example.size 16
```

Both models forbid unknown keys, so a misspelled setting is an error instead of being silently
ignored. The plugin receives the checked values as `ctx.config` and `ctx.agent_config`.

A setting that names a network endpoint, a credential or an access list should be machine-only, so
that a project you open can never point your plugin somewhere else. Mark such a field with
`GLOBAL_ONLY` from `localharness.plugins.api`:

```python
url: str = Field("", json_schema_extra=GLOBAL_ONLY)
```

A project's value for that field is then dropped with a warning, and the machine's value stands.
This plugin has no such setting.

## Artifacts

An artifact is a file a plugin makes for you, like the swatch image.

- Set `wants_artifacts = True` on the plugin class. Core then works out the plugin's folder,
  `<state dir>/artifacts/<name>/`, and passes it in as `ctx.paths.artifact_dir` (`None` for a
  plugin that did not ask). A plugin never chooses where its artifacts live.
- Save with `write_artifact(root, "<name>", data, mime)` from `localharness.core.artifacts`. Core
  gives the file an id of its own making (`art-YYYYMMDD-HHMMSS-xxxxxx`), writes exactly one new
  file, never overwrites one, and returns an `ArtifactRef`.
- Only three types are allowed: `image/png`, `image/jpeg` and `image/webp`. Anything else is refused
  before a byte is written.
- Neither the terminal nor the reference phone page displays an artifact yet. So the swatch tool's
  result names the file it wrote, `<state dir>/artifacts/example/<id>.png`, and the agent can tell
  you where the swatch is. The state dir is the `.localharness/` folder of the project you started
  in, if it has one (`localharness init --workspace` makes it), and `~/.localharness/` otherwise.
- The web channel serves an artifact to your signed-in phone at `/api/artifacts/<name>/<id>`, if
  you open that URL.
- The swatch tool takes no arguments, so asking for another color changes nothing. A different
  color is a settings change (`localharness components set example.color '#ff7f50'`), read when a
  session starts.

## Removing a plugin

1. Remove it. A folder plugin: delete its folder from `~/.localharness/plugins/`. An installed
   package, if you installed LocalHarness with `uv tool`: run `uv tool install` again without that
   plugin's `--with`, naming your extras and every plugin you keep (each run replaces the install's
   plugins and extras with the ones it names). In a virtual environment: activate it and run
   `uv pip uninstall <distribution>`.
2. Delete its `<name>:` section, and its entry under `agent:` if you set one, from `overrides.yaml`
   and `config.yaml`. LocalHarness refuses settings for a plugin it cannot find, exactly as it
   refuses a misspelled key, so a leftover section is an error until you remove it.

## Copy checklist

1. Copy this folder and rename three things: the distribution (`name` in `pyproject.toml`), the
   package (`src/localharness_plugin_example/`, the `packages` line and every import of it), and the
   entry point (`example = ...`). Then search the package for `example` and rename each remaining
   use, including the plugin name passed to `write_artifact`.
2. The entry point's name must equal `manifest.name`.
3. The plugin name becomes a settings key (`<name>:` and `agent.<name>`). It must be a lower-case
   letter followed by up to 63 lower-case letters, digits, `_` or `-`, and it must not be a key core
   already uses: a plugin named like a core setting is refused.
4. Tool names share one namespace with core's tools and every other plugin's. Prefix yours with your
   plugin name, as `example_swatch` does.
5. Delete the import-sentinel lines in `__init__.py`. They exist only for LocalHarness's own tests.
6. Keep the four declarations honest for what your tool really does. They are all the safety checks
   know about it.
7. Set `requires_localharness` to the LocalHarness releases you have tested against.
8. Install it into the same Python environment as LocalHarness, then turn it on with
   `localharness plugins enable <name>`. If you installed LocalHarness with `uv tool`, run
   `uv tool install --with-editable path/to/your-plugin localharness`, naming your extras (as in
   `'localharness[web]'`) and every other plugin's `--with` again. In a virtual environment,
   activate it and run `uv pip install -e path/to/your-plugin`.

## How this repository installs it

The root `pyproject.toml` lists `localharness-plugin-example` in a dependency group named `dev`
(`[dependency-groups]`), with a `[tool.uv.sources]` entry that points at this folder as an editable
install. uv installs the `dev` group by default, so `uv sync --extra dev` installs the plugin, and
the test suite finds it through real package metadata.

It is deliberately not in the `dev` extra (`[project.optional-dependencies]`). Extras are published
in the package's metadata: the wheel would carry
`Requires-Dist: localharness-plugin-example; extra == "dev"`, and `pip install "localharness[dev]"`
would then fetch that name from PyPI, where anyone could register it. Dependency groups are never
published.

## Files

- `pyproject.toml`: the distribution and its `localharness.plugins` entry point
- `src/localharness_plugin_example/__init__.py`: exports the plugin class (and holds the test sentinel)
- `src/localharness_plugin_example/plugin.py`: the manifest, the two settings models, `tools()` and
  `doctor()`
- `src/localharness_plugin_example/tool.py`: the `example_swatch` tool and its PNG writer
- `src/localharness_plugin_example/cli.py`: `localharness example`
- `src/localharness_plugin_example/slash.py`: `/example`
