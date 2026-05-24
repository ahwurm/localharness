# Spec 10: CLI

**Component:** `src/localharness/cli/`
**Requirements:** CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, SETUP-01, SETUP-02, SETUP-03, SETUP-04, CHAN-01, CHAN-02
**Status:** v1

---

## Purpose

The CLI is the user's entry point to LocalHarness. It provides:

1. `localharness init` — one-time setup (auto-detect LLM, write config)
2. `localharness start` — launch the orchestrator REPL
3. `localharness agent create|list|run|delete` — agent management
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

#### `localharness agent run`

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

Custom completions for agent names (used in `agent run`, `agent delete`) are provided via Typer's `typer.Completion` mechanism:

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
- `agent run` with `--task-file` reads the file and passes the path (not the contents) to the orchestrator delegation call. This is the lean context pattern applied at the CLI level.
