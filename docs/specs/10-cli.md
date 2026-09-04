# Spec 10: CLI

**Component:** `src/localharness/cli/`
**Requirements:** CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, SETUP-01, SETUP-02, SETUP-03, SETUP-04, CHAN-01, CHAN-02
**Status:** v1

---

## Purpose

The CLI is the user's entry point to LocalHarness. It provides:

1. `localharness init` — one-time setup (auto-detect LLM, write config)
2. `localharness start` — launch the orchestrator REPL
3. `localharness agent create|list` — agent management (`run`, `delete`: **planned**, not yet implemented — see below)
4. `localharness doctor` — prerequisite checks
5. `localharness validate` — config validation

Built with Typer 0.25.1 (commands + subcommand groups), Rich 15.0.0 (formatted output, streaming), prompt_toolkit 3.0.52 (REPL input, history, completion).

The CLI does not contain business logic. It parses arguments, sets up the event bus and orchestrator, and delegates. All heavy work happens in the orchestrator and agent loop components.

---

## App Structure

```python
# src/localharness/cli/app.py

import typer
from localharness.cli.init_cmd import init_app
from localharness.cli.start_cmd import start_app
from localharness.cli.agent_cmd import agent_app
from localharness.cli.doctor_cmd import doctor
from localharness.cli.validate_cmd import validate

app = typer.Typer(
    name="localharness",
    help="Model-agnostic hierarchical agent harness for local LLMs.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.add_typer(agent_app, name="agent")
app.command()(init_app)  # localharness init → flat command (not subgroup)
app.command()(start_app)  # localharness start
app.command()(doctor)     # localharness doctor
app.command()(validate)   # localharness validate

def main() -> None:
    """Entry point registered in pyproject.toml."""
    app()
```

```toml
# pyproject.toml entry point
[project.scripts]
localharness = "localharness.cli.app:main"
```

---

## Commands

### `localharness init`

```python
# src/localharness/cli/init_cmd.py

import typer
from typing import Annotated
from localharness.provider.detector import AutoDetector, DetectedProvider

def init_app(
    endpoint: Annotated[
        str | None,
        typer.Option(
            "--endpoint", "-e",
            help="Override auto-detection. Full base URL: http://localhost:8000/v1",
            envvar="LOCALHARNESS_ENDPOINT",
        )
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m",
            help="Override model selection (use with --endpoint).",
            envvar="LOCALHARNESS_MODEL",
        )
    ] = None,
    config_dir: Annotated[
        str,
        typer.Option(
            "--config-dir",
            help="Directory for LocalHarness config and agent data.",
            envvar="LOCALHARNESS_DIR",
        )
    ] = "~/.localharness",
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Overwrite existing config without prompting.",
        )
    ] = False,
) -> None:
    """
    Auto-detect local LLM and write initial configuration.
    
    Probes known ports in order: vLLM (:8000), Ollama (:11434),
    LM Studio (:1234), llama.cpp (:8080). Writes config to
    ~/.localharness/config.yaml on success.
    
    Must complete in under 5 seconds (SETUP-03). Uses 1s timeout per probe.
    """
    ...
```

**Behavior:**

1. If `~/.localharness/config.yaml` already exists and `--force` not set, print a warning and ask: "Config already exists. Re-run init? [y/N]". Default N.
2. Run `AutoDetector.probe()` — probes all known ports with 1s timeout each. Total timeout: 4s maximum (4 ports × 1s, parallel with `asyncio.gather`).
3. If `--endpoint` is provided, skip probing and use that endpoint directly.
4. Display detected provider and model list using Rich table.
5. If multiple models found, prompt user to select one (Rich prompt, numbered list).
6. Write `~/.localharness/config.yaml` with provider settings.
7. Print confirmation: `✓ LocalHarness configured. Run 'localharness start' to begin.`

**Auto-detection display:**

```
Probing for local LLM...
  ✓ vLLM found at http://localhost:8000/v1
  
Available models:
  1. Qwen/Qwen3.5-122B-A10B
  2. Qwen/Qwen3-Embedding-0.6B
  
Select model [1]: _
```

**On failure:**
```
✗ No local LLM detected.

Checked:
  http://localhost:8000  (vLLM)    — connection refused
  http://localhost:11434 (Ollama)  — connection refused
  http://localhost:1234  (LM Studio) — connection refused
  http://localhost:8080  (llama.cpp) — connection refused

Start your LLM server and run 'localharness init' again, or use:
  localharness init --endpoint http://your-host:port/v1 --model your-model-name
```

Exit code 1 on failure.

**`--workspace` (v0.13): a different command in the same name.** `localharness init --workspace`
does not configure the machine. It scaffolds `./.localharness/` for the project you are standing in
— an `agents/` directory and an all-comments `config.yaml`, nothing else, and never a `provider:`
block. It probes nothing, asks nothing, and refuses an existing workspace with **exit 1** rather
than plain `init`'s interactive exit 0, so a script can tell "created" from "already there" by the
exit code. It cannot be combined with `--endpoint`, `--model` or `--config-dir` (exit 2, naming the
flag). Spec 06 §2 has what the scaffolded file says and why it is empty.

```
✓ Workspace created at /home/you/proj/.localharness
  Config:  /home/you/proj/.localharness/config.yaml (all comments — nothing is set yet)
  Agents:  /home/you/proj/.localharness/agents
  Next:    run `localharness start` from anywhere in this project — its memory, sessions and logs now stay here.
```

---

### `localharness start`

```python
# src/localharness/cli/start_cmd.py

import asyncio
import typer
from typing import Annotated

def start_app(
    config_dir: Annotated[
        str,
        typer.Option(
            "--config-dir",
            help="LocalHarness config directory.",
            envvar="LOCALHARNESS_DIR",
        )
    ] = "~/.localharness",
    agent: Annotated[
        str | None,
        typer.Option(
            "--agent", "-a",
            help="Start directly in a specific agent's context (skip orchestrator REPL).",
        )
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug logging (structured JSON to stderr).",
            envvar="LOCALHARNESS_DEBUG",
        )
    ] = False,
) -> None:
    """
    Launch the orchestrator REPL.
    
    Starts the event bus, orchestrator, and terminal channel adapter.
    Enters an interactive prompt_toolkit loop for user input.
    Streams agent output in real time as it arrives.
    
    Exit: Ctrl-C or Ctrl-D, or type 'exit' / 'quit'.
    """
    asyncio.run(_start_async(config_dir, agent, debug))
```

**REPL Architecture:**

The REPL is a prompt_toolkit `PromptSession` inside an asyncio event loop. Input is read on one coroutine; event bus output is written on another. They share the terminal through Rich's `Live` context manager (not directly — see threading note below).

```python
# src/localharness/cli/repl.py

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
import asyncio

class OrchestratorREPL:
    """
    Interactive REPL for orchestrator conversation.
    
    Uses prompt_toolkit for input (history, completion, multi-line).
    Uses Rich Console for output (formatted text, streaming, panels).
    
    Threading model:
      - prompt_toolkit runs on the asyncio event loop via PromptSession.prompt_async()
      - Rich output is written to stdout between prompts (not during input)
      - When an agent is running, input is suspended; output streams via Rich Live
      - When agent completes, input is re-enabled with a new prompt
    """

    def __init__(
        self,
        console: Console,
        history_file: str,
        bus: EventBus,
    ) -> None: ...

    async def run(self) -> None:
        """
        Main REPL loop. Runs until the user exits.
        
        Loop:
          1. Display prompt: "you> "
          2. Read line via PromptSession.prompt_async()
          3. If input is empty or whitespace, continue
          4. If input is 'exit' or 'quit', break
          5. Publish UserMessage to event bus
          6. Enter streaming mode (suspend prompt, show Rich Live output)
          7. Wait for terminal channel to signal completion
          8. Resume prompt
        """
        ...

    def _get_prompt(self) -> HTML:
        """
        Generate the prompt string.
        Default: HTML('<ansigreen>you</ansigreen><b>></b> ')
        """
        ...
```

**Input handling during agent execution:**

When an agent is running, `PromptSession.prompt_async()` is not called. Instead, the REPL displays streaming output. If the user presses Ctrl-C during execution, a `KeyboardInterrupt` is caught, the REPL creates the KILL file for the current agent (`~/.localharness/agents/{agent_id}/KILL`), and prints: `Interrupt signal sent. Agent will stop at the next tool boundary.`

---

### `localharness agent`

```python
# src/localharness/cli/agent_cmd.py

import typer
from typing import Annotated

agent_app = typer.Typer(
    name="agent",
    help="Manage LocalHarness agents.",
    no_args_is_help=True,
)
```

#### `localharness agent create`

```python
@agent_app.command("create")
def agent_create(
    name: Annotated[
        str,
        typer.Argument(help="Agent name (alphanumeric and hyphens, max 32 chars).")
    ],
    role: Annotated[
        str,
        typer.Option("--role", "-r", help="Agent role description (what does it do).")
    ],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model to use. Inherits org default if not set.")
    ] = None,
    division: Annotated[
        str,
        typer.Option("--division", "-d", help="Division ID for this agent.")
    ] = "default",
    tools: Annotated[
        list[str] | None,
        typer.Option("--tool", "-t", help="Tool to add (repeat for multiple). E.g. --tool glob --tool bash")
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Write YAML to this path instead of default location.")
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print generated YAML without writing.")
    ] = False,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
) -> None:
    """
    Create a new agent from CLI arguments (non-conversational path).
    
    Generates YAML config from provided arguments and writes to
    ~/.localharness/agents/{name}.yaml (or --output path).
    
    For the conversational creation path, use 'localharness start'
    and ask the orchestrator to create an agent.
    
    Exit codes:
      0: Agent created successfully
      1: Name validation failed (invalid characters, name already exists)
      2: Config write failed
    """
    ...
```

#### `localharness agent list`

```python
@agent_app.command("list")
def agent_list(
    division: Annotated[
        str | None,
        typer.Option("--division", "-d", help="Filter by division.")
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", "-s", help="Filter by status: active|inactive|error")
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON array instead of table.")
    ] = False,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
) -> None:
    """
    List all configured agents.
    
    Reads Agent Cards from ~/.localharness/agents/*/agent_card.json.
    Displays as a Rich table by default.
    
    Table columns: Name | Division | Model | Status | Success Rate | Last Run
    """
    ...
```

**Rich table output:**

```
┌─────────────────────┬──────────────┬──────────────────────┬────────┬──────────────┬─────────────────────┐
│ Name                │ Division     │ Model                │ Status │ Success Rate │ Last Run            │
├─────────────────────┼──────────────┼──────────────────────┼────────┼──────────────┼─────────────────────┤
│ morning-briefing    │ financial    │ qwen3.5-122b-a10b    │ active │ 95%          │ 2026-05-23 05:30    │
│ portfolio           │ financial    │ qwen3.5-122b-a10b    │ active │ 88%          │ 2026-05-22 16:00    │
│ hn-monitor          │ research     │ qwen3.5-122b-a10b    │ error  │ 72%          │ 2026-05-23 14:12    │
└─────────────────────┴──────────────┴──────────────────────┴────────┴──────────────┴─────────────────────┘
```

**`--json` is machine output.** It is written with `typer.echo`, not through the Rich console, so
the payload is emitted without markup interpretation and without wrapping — a name containing
`[brackets]` survives intact, and no newline is injected mid-string at any terminal width. Pipe it
to `jq` directly. Inside a workspace the roster is the union of both layers, and `--json` never
stops to ask about an untrusted workspace: an undecided one is simply left out, so a scripted run
cannot spend your one-time answer for you.

#### Editing or removing an agent (today)

`agent run` and `agent delete` below are **planned** and not yet implemented. Until they
ship, manage an existing agent directly on disk:

- **Edit:** open `<config-dir>/agents/<name>.yaml` (global) or `./.localharness/agents/<name>.yaml`
  (project) and change `role`, `tools`, `permissions`, etc.
- **Remove:** delete that YAML file (and, if you also want its memory gone,
  `<config-dir>/agents/<name>/` which holds `memory.db`, `events.jsonl`, `MEMORY.md`).
- **Pick up changes:** restart the session (`localharness start`). Edits and deletions of an
  *existing* agent are read at startup. (A newly *created* agent — via `localharness start`'s
  conversational flow — is registered into the running session immediately and needs no restart;
  edits/removals still do.)
- **Run a one-off task now:** `localharness start` and ask the orchestrator to delegate to the
  agent by name.

#### `localharness agent run`

**Status: Planned — not yet implemented.** The design below is the intended surface; the command
does not exist yet. To run an agent today, see "Editing or removing an agent (today)" above.

```python
@agent_app.command("run")
def agent_run(
    agent_id: Annotated[
        str,
        typer.Argument(help="Agent ID to run.")
    ],
    task: Annotated[
        str | None,
        typer.Option("--task", "-t", help="Task description. If not set, prompts interactively.")
    ] = None,
    task_file: Annotated[
        str | None,
        typer.Option("--task-file", "-f", help="Path to file containing task description.")
    ] = None,
    max_actions: Annotated[
        int | None,
        typer.Option("--max-actions", help="Override budget: max tool calls.")
    ] = None,
    max_minutes: Annotated[
        int | None,
        typer.Option("--max-minutes", help="Override budget: max duration in minutes.")
    ] = None,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Suppress streaming output; show only final result.")
    ] = False,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
) -> None:
    """
    Run a specific agent with a task.
    
    Streams agent output to terminal in real time (unless --no-stream).
    Exits when the agent completes, times out, or encounters an error.
    
    Exit codes:
      0: Agent completed successfully (exit_reason='complete')
      1: Agent hit budget limit (exit_reason='budget')
      2: Agent got stuck (exit_reason='stuck')
      3: Agent error (exit_reason='error')
      4: Agent not found
      5: Delegation timeout
    """
    ...
```

#### `localharness agent delete`

**Status: Planned — not yet implemented.** The design below is the intended surface; the command
does not exist yet. To remove an agent today, delete its YAML (see "Editing or removing an agent
(today)" above).

```python
@agent_app.command("delete")
def agent_delete(
    agent_id: Annotated[
        str,
        typer.Argument(help="Agent ID to delete.")
    ],
    keep_memory: Annotated[
        bool,
        typer.Option("--keep-memory", help="Keep memory.db and history.jsonl; only remove config.")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation prompt.")
    ] = False,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
) -> None:
    """
    Delete an agent and optionally its memory.
    
    Removes: {config_dir}/agents/{agent_id}.yaml, agent_card.json.
    If not --keep-memory, also removes: memory.db, history.jsonl, MEMORY.md.
    
    Prompts for confirmation unless --yes is set.
    
    Exit codes:
      0: Deleted successfully
      1: Agent not found
      2: Delete failed (file I/O error)
    """
    ...
```

---

### `localharness doctor`

```python
# src/localharness/cli/doctor_cmd.py

import typer
from typing import Annotated

def doctor(
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
    fix: Annotated[
        bool,
        typer.Option("--fix", help="Attempt to auto-fix detected issues.")
    ] = False,
) -> None:
    """
    Run prerequisite checks and report system health.
    
    Checks (in order):
      1. Python version >= 3.12
      2. Required packages installed (pydantic, typer, rich, aiosqlite, etc.)
      3. Config file exists and is valid YAML
      4. LLM endpoint reachable (HTTP GET /v1/models, 5s timeout)
      5. Model name in config matches available models
      6. Config directory writable
      7. Agents directory exists and is writable
      8. SQLite available (Python built-in, should always pass)
      9. For each agent: YAML config parseable, memory.db integrity check
    
    Each check is PASS/FAIL with a one-line description.
    Exit code 0 if all pass, 1 if any fail.
    """
    ...
```

**Output format:**

```
LocalHarness Doctor
──────────────────────────────────────────────
✓ Python 3.12.3 (required: >=3.12)
✓ All packages installed
✓ Config file: ~/.localharness/config.yaml
✓ LLM endpoint reachable: http://localhost:8000/v1
✓ Model available: Qwen/Qwen3.5-122B-A10B
✓ Config directory writable
✓ Agents directory exists and writable
✓ SQLite available

Agents (2):
  ✓ morning-briefing     config OK | memory OK
  ✗ hn-monitor           config OK | memory CORRUPT
    → Run: localharness doctor --fix to attempt repair
    → Or: delete ~/.localharness/agents/hn-monitor/memory.db (loses facts)

──────────────────────────────────────────────
1 issue found. Run with --fix to attempt repair.
```

**`--fix` behavior:** Attempts to repair each detected issue. Repairable: corrupted SQLite (delete and recreate empty), missing directories (create). Non-repairable: LLM unreachable (report only), config parse error (report with line number).

**Two v0.13 additions to the output.**

*The layer report, printed only inside a workspace.* After the config-file line, `doctor` names both
layer directories and then lists every key this workspace actually changes, with the value it
displaced:

```
✓ Workspace layer: /home/you/proj/.localharness
       Global layer:    /home/you/.localharness
       1 key(s) overridden by this workspace:
         org.name = 'THIS-PROJECT'  [workspace-config]  (global: 'MACHINE-DEFAULT-ORG')
```

It compares values, not files: a workspace that restates a key with the value the global layer
already had produces no row. A workspace that changes nothing prints
`No overrides — the global config governs every key.` instead of an empty section. None of these
lines appears at all when no workspace applies. A config error here names the file that set the
offending key and that file's own line number, so an error you cannot find in the global config
tells you which workspace file to open.

*The security-defaults state, printed on every run.* One informational line — never a failure, and
it counts toward no issue total:

```
i Security defaults: revision 1 (current)
       Last migrated 2026-02-14 15:30; backup at /home/you/.localharness/config.yaml.bak-20260214-153000
```

When the config is behind the shipped revision it says so and names the two ways forward
(`localharness start` will fold them in on its next run, or `localharness config migrate` now). The
backup line is omitted entirely when no `config.yaml.bak-*` file exists — no invented date. This
exists because `start` performs that fold-in automatically and its announcement scrolls away in a
long session; see SECURITY.md.

---

### `localharness validate`

```python
# src/localharness/cli/validate_cmd.py

import typer
from typing import Annotated

def validate(
    path: Annotated[
        str | None,
        typer.Argument(help="Path to a specific agent YAML to validate. If not set, validates all.")
    ] = None,
    config_dir: Annotated[
        str,
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR")
    ] = "~/.localharness",
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat warnings as errors.")
    ] = False,
) -> None:
    """
    Validate agent YAML configuration files.
    
    Loads each YAML file through the Pydantic config loader.
    Reports: parse errors (line number, field name, error message),
    inheritance resolution failures (division/org config not found),
    unknown tool names (warn if tool not in registry).
    
    Exit code 0 if all pass, 1 if any errors (or warnings with --strict).
    """
    ...
```

**Output format:**

```
Validating agent configs...

  morning-briefing.yaml    ✓ valid
  portfolio.yaml           ✓ valid
  hn-monitor.yaml          ✗ invalid
    Line 7: tools.add[0]: 'exa_search_v2' is not a registered tool
             (did you mean 'exa_search'?)
    Line 12: permissions.budget.max_actions: value 0 is not allowed (must be >= 1)

──────────────────────────────
2 configs valid, 1 invalid.
```

---

### `localharness config`

Two subcommands: `show` (v0.13) and `migrate`.

#### `localharness config show`

`Print the effective config and the file that set each key.` One command for the question four
config files make hard to answer. It prints a header naming every file in the merge order, lowest
priority first, each marked `present` or `missing`, then a table of the values in force:

```
Config layers, lowest priority first — each one wins any key the ones above it set:
  global-config        present  /home/you/.localharness/config.yaml
  global-overrides     present  /home/you/.localharness/overrides.yaml
  workspace-config     present  /home/you/proj/.localharness/config.yaml
  workspace-overrides  present  /home/you/proj/.localharness/overrides.yaml
                             Effective config
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ key                     ┃ value                   ┃ set by              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ org.default_temperature │ 0.3                     │ global-overrides    │
│ org.log_level           │ 'debug'                 │ workspace-overrides │
│ org.name                │ 'payments-platform'     │ workspace-config    │
│ provider.base_url       │ 'http://127.0.0.1:9/v1' │ global-config       │
└─────────────────────────┴─────────────────────────┴─────────────────────┘
Showing the 7 key(s) some file sets, of 182 total. Run with --all to include the 175 compiled-in defaults.
```

The three columns are `key`, `value` and `set by`; values are shown as `repr()`, so strings arrive
quoted — the same convention `components list` uses.

| Option | Effect |
|---|---|
| (none) | only the keys some file sets, plus the footer counting the rest |
| `--all` | the whole catalogue, compiled-in defaults included; no footer |
| `--json` | `{"layers": [...], "keys": [...]}`, `indent=2` |
| `--config-dir` | read that directory instead, and switch workspace discovery off with it |

`--json` has exactly two top-level keys. `layers` is ordered lowest-priority-first with
`layer` / `path` / `exists` per entry; `keys` carries `path` / `value` / `type` / `layer` /
`default`:

```json
{
  "layers": [
    { "layer": "global-config", "path": "/home/you/.localharness/config.yaml", "exists": true },
    { "layer": "workspace-config", "path": "/home/you/proj/.localharness/config.yaml", "exists": true }
  ],
  "keys": [
    { "path": "org.default_temperature", "value": 0.3, "type": "float", "layer": "global-overrides", "default": 0.6 }
  ]
}
```

With no workspace there are **two** layer rows, not four with two marked missing: someone who never
made a workspace sees no trace of the feature. Exit 2 if no config can be loaded at all.

#### `localharness config migrate`

`Additively sync the shipped default deny patterns into your config.yaml.` `init` bakes the resolved
`org.permissions.deny_patterns` into `config.yaml`, so a later growth of the shipped default list
never reaches an existing install. This appends the missing ones and stamps the config's defaults
revision. Additive only: it never removes or reorders your own entries, touches no other key, and —
because it is revision-gated — never re-adds a default you deliberately deleted. A timestamped
backup is written before the config is updated.

`--dry-run` reports what would change and writes nothing:

```
24 shipped default deny pattern(s) missing from /home/you/.localharness/config.yaml (defaults revision 0 → 1):
  + write(*/.env)
  + bash_exec(*sudo *)
  ...
Additive only — if you deliberately removed a default, re-remove it after migrating.

i --dry-run: nothing written.
```

`localharness start` runs this same engine automatically on the first start after an upgrade — see
SECURITY.md, and `doctor` for the state it leaves behind.

---

### `localharness components`

Read and change any registered config value by dot-path, without opening a YAML file.

#### `localharness components list`

`List every mutable component with its current value and winning layer.` Columns: `path`, `type`,
`current value`, `layer`.

`--layer <band>` filters to the entries a given layer won. The accepted names are the merge bands
plus `default`:

| Band | Means |
|---|---|
| `default` | no file set it; the compiled-in value is in force |
| `global-config` | `~/.localharness/config.yaml` |
| `global-overrides` | `~/.localharness/overrides.yaml` |
| `workspace-config` | `{project}/.localharness/config.yaml` |
| `workspace-overrides` | `{project}/.localharness/overrides.yaml` |
| `experiment` | an in-memory overlay a running gate applies; it has no file of its own, so nothing on disk sets it and `--layer experiment` normally matches nothing |

`--json` emits the same rows with the field names `path` / `type` / `current_value` / `layer`.

#### `localharness components get <path>`

Prints one path's value, its type, the layer that won it, and its compiled-in default:

```
org.name = 'THIS-PROJECT'
  type:    str
  layer:   workspace-config
  default: 'default'
```

`--json` gives `path` / `value` / `type` / `layer` / `default`.

#### `localharness components set <path> <value>`

One path at a time; the value is coerced to the path's declared type.

**`set` always writes the global `~/.localharness/overrides.yaml`, in every project.** There is no
per-project write target in v0.13, so the command says what it did rather than leaving you to find
out: it prints the file it wrote, warns that the setting applies machine-wide, and points at the
workspace `config.yaml` as the place a per-project value belongs. If the workspace already sets that
key, it also tells you the value you just wrote is not the one in force where you are standing:

```
set org.log_level = 'debug' (was: 'info')
  wrote: /home/you/.localharness/overrides.yaml  (the global overrides.yaml)
  note: this is a MACHINE-WIDE setting. It applies in every project, including this one.
        A per-project value goes in /home/you/proj/.localharness/config.yaml.
```

All three subcommands take `--config-dir`, which — as everywhere else — replaces the config
directory outright and switches workspace discovery off with it.

---

## REPL Interface

### Input Loop

```python
# src/localharness/cli/repl.py (continued)

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.formatted_text import HTML

HISTORY_FILE = "~/.localharness/.repl_history"

def build_session() -> PromptSession:
    """
    Build a prompt_toolkit session with history and auto-suggest.
    
    Key bindings:
      Ctrl-C: Send interrupt (kill current agent if running)
      Ctrl-D: Exit REPL
      Up/Down: History navigation
      Tab: Auto-complete from history (AutoSuggestFromHistory)
      Enter: Submit input
      Alt-Enter: Insert newline (multi-line input)
    """
    kb = KeyBindings()

    @kb.add("c-c")
    def _(event):
        event.app.current_buffer.set_document(
            event.app.current_buffer.document, bypass_readonly=True
        )
        raise KeyboardInterrupt

    return PromptSession(
        history=FileHistory(HISTORY_FILE),
        auto_suggest=AutoSuggestFromHistory(),
        key_bindings=kb,
        multiline=False,
    )
```

### Streaming Output

Agent output is streamed to the terminal as it arrives. The terminal channel adapter (see spec 11) publishes token events to the bus; the REPL subscribes and writes tokens to stdout using `rich.console.Console.print`.

Streaming output is displayed inside a Rich `Panel` with the agent name as the title:

```
╭─ morning-briefing ────────────────────────────────────────╮
│ Searching for today's market news...                      │
│ [tool call: exa_search {"query": "SPX May 23 2026"}]     │
│ Found 5 results. Analyzing...                             │
│                                                           │
│ **Morning Briefing — May 23, 2026**                       │
│                                                           │
│ Markets opened flat...                                    │
╰───────────────────────────────────────────────────────────╯
```

Tool calls are rendered inline in a distinct style to distinguish agent reasoning from tool activity:

- Agent text: `white`
- Tool call line: `dim cyan` prefixed with `⚙`
- Tool result summary: `dim` prefixed with `↳`
- Final result panel: `green` border

### Output During Long-Running Tasks

For tasks that run longer than 10 seconds without new output, the REPL displays a spinner: `⠋ Agent is working... (42s)`. This uses `rich.progress.Progress` in `Live` mode.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Configuration error / agent not found / init failed |
| 2 | Agent completed with error or budget exhausted |
| 3 | Agent stuck (stuck detection triggered) |
| 4 | Provider unreachable (LLM server not responding) |
| 5 | Delegation timeout |
| 130 | Interrupted by Ctrl-C (standard SIGINT convention) |

Typer does not set exit codes automatically — use `raise typer.Exit(code=N)` to exit with a specific code. Use `typer.echo(message, err=True)` for error messages (writes to stderr, not stdout, so they don't pollute piped output).

---

## Error Formatting

All user-facing errors follow this format:

```
Error: {short description}

  {detail lines, indented 2 spaces}
  {may span multiple lines}

{recovery suggestion if applicable}
```

Printed with `rich.console.Console(stderr=True).print("[bold red]Error:[/bold red] ...")`.

Internal exceptions (Python tracebacks) are only shown when `--debug` is set. Without `--debug`, they are caught at the CLI boundary and converted to user-friendly error messages.

---

## Shell Completion

Typer provides shell completion automatically via Click's `--install-completion` and `--show-completion` flags. These are added automatically to the app when `add_completion=True` is set.

```bash
# Install completion (Bash)
localharness --install-completion bash
source ~/.bashrc

# Install completion (Zsh)
localharness --install-completion zsh

# Install completion (Fish)
localharness --install-completion fish
```

Custom completions for agent names (planned for the not-yet-implemented `agent run`, `agent delete`) are provided via Typer's `typer.Completion` mechanism:

```python
def complete_agent_id(ctx: typer.Context, param: typer.CallbackParam, incomplete: str):
    """Return list of agent IDs matching the incomplete string."""
    config_dir = ctx.params.get("config_dir", "~/.localharness")
    agent_ids = _load_agent_ids(config_dir)
    return [a for a in agent_ids if a.startswith(incomplete)]
```

---

## Configuration Precedence

Every configurable value follows this precedence order (highest to lowest):

```
1. CLI flag (--endpoint, --model, --config-dir, etc.)
2. Environment variable (LOCALHARNESS_ENDPOINT, LOCALHARNESS_MODEL, LOCALHARNESS_DIR)
3. Config file (~/.localharness/config.yaml)
4. Built-in defaults (hardcoded in config/defaults.py)
```

Implementation: Typer handles CLI flags and `envvar=` on each `Option`. The config file is loaded by `ConfigLoader` after CLI parsing. Defaults are in `ToolConfig()`, `PermissionConfig()`, etc. as Pydantic field defaults.

**Workspace layer (v0.13).** `--config-dir`, `LOCALHARNESS_DIR` and `LOCALHARNESS_HOME` are a full replacement: naming a config directory also switches workspace discovery off, so no workspace layer applies. With none of them set, the harness looks for the nearest `.localharness/` directory at or above your current directory and reads agent and division files from it ahead of the global directory. It applies that layer without asking when the workspace belongs to the project you are in — your own directory, or at or below the root of the git repository containing you. A workspace from outside that project loads only after a one-time confirmation, and is ignored with a notice on stderr when there is no terminal to ask. `doctor` and `start` name the layer they chose; see spec 06 for the full search order.

When a workspace applies, the config file at step 3 above is four files, merged in this order —
lowest priority first, each winning any key the ones before it also set:

```
1. ~/.localharness/config.yaml         global-config       (written by `localharness init`)
2. ~/.localharness/overrides.yaml      global-overrides    (written by `components set`)
3. {project}/.localharness/config.yaml    workspace-config
4. {project}/.localharness/overrides.yaml workspace-overrides
```

Spec 06 §"How the two layers combine" states the rule this order encodes, and the exceptions to it
(`deny_patterns` accumulate; a workspace `overrides.yaml`'s `agent:` section is not read).
`localharness config show` prints these four files, in this order, with the layer that set each
effective key.

Precedence merging:

```python
def resolve_config(
    cli_overrides: dict[str, Any],
    env_overrides: dict[str, Any],
    file_config: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge configs in precedence order. CLI > env > file > defaults.
    None values in CLI/env do not override lower-precedence values.
    """
    result = {**defaults}
    for overrides in [file_config, env_overrides, cli_overrides]:
        result.update({k: v for k, v in overrides.items() if v is not None})
    return result
```

---

## Implementation Notes

- All async CLI functions use `asyncio.run()` at the top level. Typer callbacks are synchronous; the async entry point is wrapped with `asyncio.run(_async_main(...))`.
- `rich.console.Console()` is instantiated once at CLI startup and passed to all components. Never create multiple Console instances — it causes formatting conflicts.
- `prompt_toolkit` and `rich` do not share terminal state automatically. The REPL must pause `rich.Live` before calling `PromptSession.prompt_async()` and resume it afterward. Pattern: `with Live(...) as live: ... live.stop() ... await session.prompt_async() ... live.start()`.
- `localharness start` checks for the KILL file (`~/.localharness/KILL`) on startup and removes it if present (cleanup from a previous interrupted session).
- The `--config-dir` option is repeated on every command rather than being a top-level app option. This is intentional — Typer's callback mechanism for top-level options has edge cases with subcommand groups. Repetition is the safer pattern.
- (Planned) `agent run` with `--task-file` will read the file and pass the path (not the contents) to the orchestrator delegation call. This is the lean context pattern applied at the CLI level.
