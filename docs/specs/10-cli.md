# Spec 10: CLI

**Component:** `src/localharness/cli/`
**Requirements:** CLI-01, CLI-02, CLI-03, CLI-04, CLI-05, CLI-06, SETUP-01, SETUP-02, SETUP-03, SETUP-04, CHAN-01, CHAN-02
**Status:** v1

---

## Purpose

The CLI is the user's entry point to LocalHarness. The everyday path is four commands:

1. `localharness init` — one-time setup (auto-detect LLM, write config)
2. `localharness start` — launch the orchestrator REPL
3. `localharness doctor` — prerequisite checks
4. `localharness validate` — config validation

The full command list is under [App Structure](#app-structure).

Built with Typer (commands + subcommand groups), Rich (formatted output, streaming) and prompt_toolkit (REPL input, history, completion). `pyproject.toml` holds the supported ranges — `typer>=0.25,<1`, `rich>=15.0,<16`, `prompt-toolkit>=3.0,<4`.

The CLI does not contain business logic. It parses arguments, sets up the event bus and orchestrator, and delegates. All heavy work happens in the orchestrator and agent loop components.

---

## App Structure

`src/localharness/cli/app.py` builds one Typer app and registers thirteen top-level commands —
seven flat commands and six subcommand groups. `localharness --help` prints exactly this list:

| Command | What it does |
|---|---|
| `init` | Auto-detect local LLM and write initial configuration. |
| `start` | Launch the agent REPL. Zero to chatting in one command. |
| `doctor` | Run prerequisite checks and report system health. |
| `validate` | Validate agent YAML configuration files. |
| `model` | List available models, or switch the persisted default with `localharness model <name>`. |
| `propose` | Generate ONE typed mutation `{diff, rationale}` for ONE component from failed TRAIN traces. |
| `update` | Upgrade LocalHarness to the latest release on PyPI. |
| `agent` *(group)* | Manage LocalHarness agents — `create`, `list`. |
| `bench` *(group)* | Run scenario benchmarks; compare runs for regressions. Matrix is opt-in (`--matrix`). Subcommands: `compare`, `pack`. |
| `components` *(group)* | List, inspect, and mutate harness components (registry) — `list`, `get`, `set`. |
| `config` *(group)* | Inspect and maintain your LocalHarness configuration — `show`, `migrate`. |
| `autoresearch` *(group)* | Autoresearch loop tools. |
| `experiment` *(group)* | Run a proposal through the promotion gate (train Welch → holdout Bonferroni). |

Sections below document `init`, `start`, `agent`, `doctor`, `validate`, `config` and `components`
in detail. `bench`, `autoresearch`, `experiment`, `propose`, `model` and `update` are documented by
their own `--help`, which is generated from the same source that defines them.

```python
# src/localharness/cli/app.py (abridged — the registrations)

app = typer.Typer(
    name="localharness",
    help="Model-agnostic hierarchical agent harness for local LLMs.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)

app.command("init")(init_app)
app.command("start")(start_app)
app.command("doctor")(doctor)
app.command("validate")(validate)
app.command("model")(model)
app.command("propose")(propose)
app.command("update")(update)
app.add_typer(agent_app, name="agent")
app.add_typer(bench_app, name="bench")
app.add_typer(components_app, name="components")
app.add_typer(config_app, name="config")
app.add_typer(autoresearch_app, name="autoresearch")
app.add_typer(experiment_app, name="experiment")

def main() -> None:
    """Entry point registered in pyproject.toml."""
    app()
```

A root callback adds `--version`, which prints the in-source version and exits.

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
        str | None,
        typer.Option(
            "--config-dir",
            help="Directory for LocalHarness config and agent data. Default: "
                 "$LOCALHARNESS_DIR, else $LOCALHARNESS_HOME, else ~/.localharness.",
            envvar="LOCALHARNESS_DIR",
        )
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Overwrite existing config without prompting.",
        )
    ] = False,
    workspace: Annotated[
        bool,
        typer.Option(
            "--workspace",
            help="Scaffold ./.localharness/ for THIS project instead of configuring the "
                 "machine. Non-interactive; never writes a provider block; never overwrites "
                 "an existing one.",
        )
    ] = False,
) -> None:
    """
    Auto-detect local LLM and write initial configuration.

    Probes known ports in order: vLLM (:8081), vLLM (:8000), Ollama (:11434),
    LM Studio (:1234), llama.cpp (:8080). Writes config to
    <config-dir>/config.yaml on success.
    """
    ...
```

The probe order is one list, `DEFAULT_PORTS` in `provider/detector.py` — `[8081, 8000, 11434, 1234,
8080]`. Both the `--help` text and the "no server detected" message are derived from it, so all
three cannot drift apart.

**`--workspace` is a different command in the same skin.** It configures a *project*, not the
machine: it creates `./.localharness/`, `./.localharness/agents/`, and a `./.localharness/config.yaml`
that is entirely comments — nothing set. It probes nothing, prompts for nothing, never writes a
`provider:` block, and refuses (rather than overwriting) if `./.localharness/` already exists or if
you are standing in the machine's own global config directory.

**Behavior:**

1. If `~/.localharness/config.yaml` already exists and `--force` not set, print a warning and ask: "Config already exists. Re-run init? [y/N]". Default N.
2. Run `detect_provider()` (`provider/detector.py`) — probes the five ports above with a short
   timeout each.
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

When an agent is running, `PromptSession.prompt_async()` is not called. Instead, the REPL displays streaming output. Ctrl-C during a turn does not touch the KILL file: the REPL installs its own SIGINT handler for the turn's duration that **cancels the turn task**, which propagates cancellation through the loop and closes the in-flight streaming HTTP call. On vLLM that disconnect aborts generation engine-side, so no ghost request is left running; other runtimes end the stream on their own terms. The session itself survives, and you are back at the prompt. A second Ctrl-C while the turn is already cancelling restores the default handler, so pressing it again hard-exits.

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

```
Usage: localharness agent create [OPTIONS] NAME

  Create a new agent YAML config.

Arguments:
  NAME                  Agent name (lowercase alphanumeric + hyphens)  [required]

Options:
  --role, -r    TEXT    Agent role description  [default: General-purpose agent]
  --model, -m   TEXT    Model name. Inherits org default if not set.
  --global              Add agent to global config (~/.localharness/agents/)
  --project             Add agent to this project's workspace (nearest .localharness/agents/
                        up-tree, else ./.localharness/agents/). Not available with an explicit
                        config directory.
  --dry-run             Print YAML without writing
  --force               Overwrite an existing agent with the same name (default refuses)
  --no-input            Never ask about an untrusted workspace: skip its config layer, say so,
                        and record nothing. For hooks, CI, and any run with no one watching.
  --config-dir  TEXT    Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                        else ~/.localharness.  [env var: LOCALHARNESS_DIR]
```

There is a conversational path too: run `localharness start` and ask the orchestrator to create an
agent.

**Which layer it writes to.** `--global` writes `<config-dir>/agents/<name>.yaml`. `--project`
writes into this project's workspace — the nearest `.localharness/agents/` at or above your current
directory, creating `./.localharness/agents/` if there is none yet. Passing both is an error. With
neither, the command asks ("Add globally or to this project?", default `global`) — unless
`--no-input` is set, which refuses instead of guessing, because writing an agent to the wrong layer
is precisely the mistake that flag exists to prevent.

**`--project` refuses an explicit config directory.** If `--config-dir`, `$LOCALHARNESS_DIR` or
`$LOCALHARNESS_HOME` is set, naming a config directory switches workspace discovery off entirely
(spec 06, "Search Order for Config Files") — an explicit directory is a full replacement, not a base
to layer on. So there is no project for `--project` to resolve, and an agent written to one would be
invisible to every command run the same way. The command exits 2 with a message that names the
setting you actually used, and points at the two ways forward: drop that setting, or pass `--global`.

**It never silently overwrites.** An existing `<name>.yaml` is refused (exit 1) unless `--force` is
passed. The refusal happens before anything is created, so a command that refuses has written
nothing. If a workspace exists but is not in use for this command (outside your project and
untrusted, or undecided with no terminal to ask), `--project` says so on stderr and names both paths
rather than quietly minting a second `.localharness/` beside the one you already have.

**Exit codes:** 0 created (or `--dry-run` printed); 1 invalid name, both scope flags, an invalid
answer to the prompt, an existing agent without `--force`, or a write failure; 2 `--no-input` with
no scope, or `--project` with an explicit config directory.

#### `localharness agent list`

```
Usage: localharness agent list [OPTIONS]

  List all configured agents.

Options:
  --json                Output as JSON array
  --verbose, -v         Show full details
  --config-dir  TEXT    Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                        else ~/.localharness.  [env var: LOCALHARNESS_DIR]
```

It reads the agent YAML files themselves, across both layers, keyed by filename — so a workspace
`agents/foo.yaml` replaces a global one of the same name, and everything else from both layers is
listed. An unparseable file is skipped with a warning naming it, never silently.

**Rich table output.** Two columns by default, three with `--verbose`:

```
                Agents
┌──────────────────┬───────────────────────────┐
│ Name             │ Role                      │
├──────────────────┼───────────────────────────┤
│ orchestrator     │ General-purpose agent     │
│ morning-briefing │ Generate a daily report…  │
└──────────────────┴───────────────────────────┘
```

`--verbose` adds a `Model` column, showing `inherit` where the agent does not pin one.

**`--json` is machine output.** It emits the raw agent dictionaries — whatever keys each YAML file
declares, plus `name` filled in from the filename when the file omits it. An empty roster emits `[]`,
not prose, so a caller's `json.loads` always has an answer. It is written with `typer.echo`, not through the Rich console, so
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
  `<config-dir>/agents/<name>/` which holds `memory.db`, `history.jsonl`, `bus-events.jsonl`
  and `MEMORY.md`).
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

Implemented in `localharness.cli.doctor_cmd.doctor`.

```
Usage: localharness doctor [OPTIONS]

Options:
  --config-dir  TEXT    Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                        else ~/.localharness.  [env var: LOCALHARNESS_DIR]
  --fix                 Create a missing agents directory (doctor's only auto-fix today).
  --no-input            Never ask about an untrusted workspace: skip its config layer, say so,
                        and record nothing. For hooks, CI, and any run with no one watching.
```

**What it checks, in the order it prints them:**

1. **Python version** — `>=3.12`.
2. **Config file** — that `<config-dir>/config.yaml` exists. If it does not *and* no workspace layer
   applies, doctor stops here and reports the machine as unconfigured.
3. **The layer report** — workspace directory, global directory, and the keys this workspace
   overrides. Printed only inside a workspace; see below.
4. **Config valid** — the merged config parses and validates.
5. **Security defaults** — an informational line, never a failure. Printed whenever the config
   loaded; a missing or invalid config skips it, because there is nothing to report a revision for.
   See below.
6. **LLM endpoint reachable** — the configured `base_url`.
7. **Model available** — the configured default model is in the endpoint's model list.
8. **Context budget vs. served window** — whether the configured context budget exceeds, badly
   undershoots, or fits the window the server actually reports.
9. **Token counting** — which counting mode this runtime resolves to (exact server-side, exact from
   a local GGUF vocab, or a labeled approximation), plus whether the tokenizer endpoint supports
   message-level counting.
10. **Runtime advisories** — llama.cpp only, informational: slot-window and AMD Vulkan-vs-HIP notes.
11. **Config directory writable.**
12. **Agents directory exists** — the one check `--fix` acts on.
13. **Workspace agents** — the workspace's own `agents/` directory and the names in it. Workspace
    only.
14. **Tool calling** — native, XML fallback, or not yet probed.
15. **Web search** — whether `ddgs` is installed.

It does **not** check individual agents' memory databases, and there is no per-agent health table.

**Output format:**

```
─────────────────────────── LocalHarness Doctor ───────────────────────────
✓ Python 3.12.3 (required: >=3.12)
✓ Config file: /home/you/.localharness/config.yaml
✓ Config valid
i Security defaults: revision 1 (current)
✓ LLM endpoint reachable: http://localhost:8081/v1
✓ Model available: Qwen/Qwen3-32B
✓ Context budget 131,072 fits served window 131,072
✓ Token counting: exact — server-side /tokenize (vllm contract).
✓ Tokenizer endpoint reachable (/tokenize) — exact counts, message-level (chat template applied server-side)
✓ Config directory writable
✓ Agents directory exists
i  Tool calling: native
✓ Web search ready (ddgs installed)

───────────────────────────────────────────────────────────────────────────
All checks passed.
```

A failing run ends with `N issue(s) found.` and exits 1.

**`--fix` creates a missing agents directory. That is all it does.** There is no repair for a
corrupt database, an unreachable endpoint, or an invalid config — those are reported and left to
you. The flag's help string says the same thing, and both are the implementation: two `mkdir`
calls on the agents directory.

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

Implemented in `localharness.cli.validate_cmd.validate`.

```
Usage: localharness validate [OPTIONS] [PATH]

  Validate agent YAML configuration files.

  Reports parse errors, field validation failures with line numbers.
  Exit code 0 if all valid, 1 if any invalid, 2 if no config files found.

Arguments:
  PATH                  Path to specific YAML to validate. If not set, validates all.

Options:
  --config-dir  TEXT    Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                        else ~/.localharness.  [env var: LOCALHARNESS_DIR]
  --strict              Reserved. No warning-level checks exist yet; currently identical to
                        the default.
  --no-input            Never ask about an untrusted workspace: skip its config layer, say so,
                        and record nothing. For hooks, CI, and any run with no one watching.
```

That is the whole flag surface — **there is no `--json`, `--agent` or `--verbose`**, and `--strict`
does nothing today beyond printing a note saying so. It is reserved for warning-level checks that
have not been written yet; when they exist, `--strict` will promote them to errors.

With no `PATH`, it validates every config file across both layers — the global directory and, when
one applies, the project workspace. Each file is validated **at its own path**: a workspace
`agents/foo.yaml` and a global `agents/foo.yaml` are two files and get two verdicts, rather than one
of them being checked twice. One asymmetry is deliberate: on a machine that was never `init`ed, a
workspace `config.yaml` is reported only when it fails to parse. It is an overlay, partial by
design, and the global file is still required to start a session — so calling it valid would
green-light a machine that cannot run. With a `PATH`, only that file is validated, and the model used is
chosen from the filename and its parent directory (`config.yaml` → harness, `org.yaml` → org,
anything under `divisions/` → division, otherwise → agent).

**Output format.** With no workspace layer, rows show the bare filename, exactly as before v0.13:

```
Validating configs...

  config.yaml                         ✓ valid
  morning-briefing.yaml               ✓ valid
  hn-monitor.yaml                     ✗ invalid
    Line 12: permissions.budget.max_actions: Input should be greater than or equal to 1
      value: 0

──────────────────────────────────────────────────────────────────────────
2 config(s) valid, 1 invalid.
```

Inside a workspace, where two files can share a name, rows show the **full path** instead, and an
error's row is keyed to the file that actually owns the error. Where that differs from the file the
walk was iterating, the detail lines name it explicitly (`in <path>`).

**Exit codes:** 0 all valid; 1 one or more invalid, a `PATH` that does not exist, or an unreadable
config file; 2 no configuration files found at all.

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

**This is also where autoresearch adoptions land.** When the loop adopts a mutation that cleared the
promotion gate, it writes the new value into that same global `overrides.yaml`, through the same
primitives `set` uses, and emits a `ComponentMutated` audit event. It does not touch git — nothing
is staged, committed, or written into your project. Two consequences worth knowing before you run
the loop: the change is **machine-wide**, applying in every project on the box; and undoing one is
`localharness components set <path> <old value>` (the archive keeps the previous value on the
adoption record), or editing `overrides.yaml` directly.

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

There is no single exit-code table for the whole CLI, and a doc that claimed one would misdescribe
half of it. `0` always means success. Everything else is per command, and the research commands use
the exit code as *structured output* rather than as a pass/fail flag.

| Command | Codes |
|---|---|
| `doctor` | 0 all checks passed; 1 one or more failed |
| `validate` | 0 all valid; 1 one or more invalid, or a named path that cannot be read; 2 no config files found |
| `init` | 0 written (or an overwrite you declined); 1 detection, prompt or write failure; 2 conflicting flags, or `--workspace` in the global config directory |
| `start` | 0 normal exit; 1 config/startup failure; 2 usage error |
| `agent create` / `agent list` | 0 done; 1 invalid input or write failure; 2 usage error (see `agent create` above) |
| `config show` / `config migrate` | 0 done; 1 failure; 2 usage error |
| `model`, `propose` | 0 done; 2 any error |
| `components` | 0 done; 2 any error |
| `bench` | 0 success; 2 infrastructure failure (config missing, empty corpus, no runs) |
| `bench compare` | 0 stable; 1 regressed; 2 infrastructure failure; 3 unstable |
| `bench pack` | 0 built; 1 the pack failed to build |
| `experiment` | **the code is the gate verdict** — 0 promote, 1 reject-train, 2 reject-holdout, 3 inconclusive; ≥4 a structural refusal (the experiment did not run) |
| `autoresearch` | 0 done; 2 any error |
| `update` | 1 if PyPI is unreachable or `uv` is missing from PATH; otherwise the exit code of the upgrade subprocess it runs |

Typer does not set exit codes automatically — use `raise typer.Exit(code=N)`. Use
`typer.echo(message, err=True)`, or a stderr Rich console, for error messages, so they don't pollute
piped output.

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
2. Config file (~/.localharness/config.yaml)
3. Built-in defaults (Pydantic field defaults in config/models.py)
```

Implementation: Typer handles CLI flags. The config file is loaded by `ConfigLoader` after CLI
parsing. Defaults live on the models themselves — `ToolConfig()`, `PermissionConfig()` and the rest.

**Environment variables are not a general precedence layer.** Four env vars exist, and each one is
tied to a specific flag or to the config directory: `LOCALHARNESS_DIR` and `LOCALHARNESS_HOME`
choose the config directory, and `LOCALHARNESS_ENDPOINT` and `LOCALHARNESS_MODEL` are `init`'s two
overrides. Three of them are read by Typer's `envvar=`, which is why they behave exactly like
typing the flag; `LOCALHARNESS_HOME` is read instead by `config_dir_env_override()` in
`config/paths.py`, as the legacy alias consulted only when `LOCALHARNESS_DIR` is unset. There is no
environment override for arbitrary config keys — see spec 06 §"Environment variables".

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

**How the merge actually happens.** `ConfigLoader.load_harness()` reads those files as raw
dictionaries in that order and folds them together with a recursive `deep_merge`, so a later file
wins any key it sets and inherits every key it does not mention. The merged dictionary is then
validated once, by `HarnessConfig.model_validate` — which is where the compiled-in defaults come
from, as Pydantic field defaults, not as a separate defaults dictionary. `deny_patterns` is the one
exception to "later wins": it is unioned across all four files first, so a workspace can add deny
patterns but never drop one. Environment variables are not a merge layer at all — see the note
above.

When validation fails, the loader maps the failing field back to the file that set it and reports
that file's own line numbers, so an error you cannot find in the global config tells you which
workspace file to open.

---

## Implementation Notes

- All async CLI functions use `asyncio.run()` at the top level. Typer callbacks are synchronous; the async entry point is wrapped with `asyncio.run(_async_main(...))`.
- `rich.console.Console()` is instantiated once at CLI startup and passed to all components. Never create multiple Console instances — it causes formatting conflicts.
- `prompt_toolkit` and `rich` do not share terminal state automatically. The REPL must pause `rich.Live` before calling `PromptSession.prompt_async()` and resume it afterward. Pattern: `with Live(...) as live: ... live.stop() ... await session.prompt_async() ... live.start()`.
- The KILL file is one flat file in the **global** config directory — `~/.localharness/KILL` in a default setup — not one per agent, and never inside a workspace: it exists to stop every session on the machine. Its value comes from `permissions.budget.kill_file`, read from the global config layer only; a workspace that sets it is ignored, with a line in the log. The harness only reads that file. It does not create it and does not remove it — creating it is how you stop a session, and deleting it again is yours to do.
- The `--config-dir` option is repeated on every command rather than being a top-level app option. This is intentional — Typer's callback mechanism for top-level options has edge cases with subcommand groups. Repetition is the safer pattern.
- (Planned) `agent run` with `--task-file` will read the file and pass the path (not the contents) to the orchestrator delegation call. This is the lean context pattern applied at the CLI level.
