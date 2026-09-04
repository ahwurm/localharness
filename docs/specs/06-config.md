# Spec 06: Configuration System

**Project:** LocalHarness  
**Component:** `config/models.py`, `config/loader.py`, `config/defaults.py`  
**Layer:** 1 (models) + 2 (loader)  
**Status:** Authoritative — implement against this document  
**Last updated:** 2026-09-03 (v0.13: workspace layer search order, trust gate, and config merge)

---

## 1. Purpose

The configuration system is the mechanism by which users define agents without writing code. All agent behavior — model selection, tool access, permissions, memory paths, scheduling, context management — is expressed in YAML files on disk. The configuration system:

1. Parses YAML files with `yaml.safe_load` (never `yaml.load`)
2. Validates every field against Pydantic models with line-level error reporting
3. Resolves inheritance: org defaults → division overrides → agent overrides
4. Exposes a `ConfigLoader` class consumed by the agent loop and orchestrator
5. Provides the `localharness validate` contract for CI and user debugging

All agent configuration is **read-only at runtime**. The agent loop reads config once at the start of `run_turn()` and does not re-read during execution. Config changes take effect on the next invocation.

---

## 2. File Layout

```
~/.localharness/
├── config.yaml                   # Global harness config (written by `localharness init`)
├── org.yaml                      # Organization-level agent defaults (optional)
├── agents/
│   ├── {agent-name}.yaml         # Agent config files (one per agent)
│   └── {agent-name}.yaml.bak     # Backup on overwrite
└── divisions/
    └── {division-name}.yaml      # Division config files (one per division)

# Workspace config (optional): the nearest .localharness/ at or above the current directory
{project}/.localharness/
├── config.yaml                   # merged over the global config.yaml (workspace wins per key)
├── overrides.yaml                # merged last — highest priority, EXCEPT its `agent:` section,
│                                 #   which nothing reads (see "What this does NOT cover")
├── audit.jsonl                   # this project's audit records
├── agents/
│   ├── {agent-name}.yaml         # agent config
│   └── {agent-name}/             # ...and that agent's state: memory.db, MEMORY.md,
│       └── sessions/             #    history.jsonl, bus-events.jsonl, compact.md
├── divisions/
└── microagents/
```

### Search Order for Config Files

The ConfigLoader searches for agent configs in this priority order (first found wins):

1. `{nearest workspace}/.localharness/agents/{name}.yaml` — the workspace layer (highest priority)
2. `~/.localharness/agents/{name}.yaml` — user-global
3. (v2) system-level configs — not implemented in v1

Division configs follow the same pattern. Org settings live in the `org:` section of `config.yaml`
and are merged across both layers like any other key. A standalone `~/.localharness/org.yaml` is an
older, optional file — it still works, and it is read from the global directory only.

**Finding the workspace layer (v0.13).** At session start the harness walks up from your current
directory looking for a `.localharness/` directory. The **first one it finds is the workspace
layer** — exactly one layer applies, so a second `.localharness/` further up the tree has no
effect. The walk stops at `$HOME` and never treats the global `~/.localharness/` as a workspace,
so a user with no project workspace behaves exactly as they did before v0.13.

A workspace **inside the project you are standing in** applies straight away. That means the
`.localharness/` in your current directory, or one at or below the root of the git repository you
are working in. Nested directories inherit their project's config: running the harness from
`myproject/src/pkg` picks up `myproject/.localharness/`, with no prompt and nothing recorded.

A workspace **from outside that project** — above your repository's root, or in a parent folder
when you are not in a repository at all — is only loaded after you agree to it once. The harness
asks, and stores the answer in `~/.localharness/trusted_workspaces.yaml`, globally, never inside
the workspace itself. The answer is permanent; edit that file to change it. When there is no
terminal to ask — a script, a cron job, CI — that workspace layer is ignored and a notice is
printed on stderr.

`--config-dir`, `LOCALHARNESS_DIR` and `LOCALHARNESS_HOME` replace the config directory outright
and skip discovery entirely: no workspace layer applies when any of them is set.

**`--no-input`, for runs with nobody watching.** `doctor`, `validate` and `agent create` take
`--no-input`: never ask about an untrusted workspace, skip that layer, say on stderr that it was
skipped, and **record nothing**. The reason it exists is that the trust answer is permanent. A git
hook, a CI job or a scheduled run that happens to inherit a terminal would otherwise be able to
answer that question on your behalf, once, forever — a decision about trust made by whatever process
happened to run first. `--no-input` makes the run declare that it is not the right process to be
asked. It is also the honest way to script these commands: the run still works, it just works
without that layer. (`agent create --no-input` additionally refuses to guess a target layer: pass
`--global` or `--project`.)

**Three current behaviors worth knowing before you rely on the walk.** These are what the code does
today, described plainly rather than promised:

- **A linked git worktree reads as outside its parent project.** The walk stops at the worktree's
  own root, so a `.localharness/` in the main checkout is "somewhere else" from inside the worktree
  and you get the one-time trust question about your own repository.
- **A non-git folder gets the workspace layer only at the project root.** For a directory that is
  not in a git repository, only standing exactly in the folder that holds `.localharness/` counts as
  in-project. A subdirectory of it is treated as outside, which means the trust question — and in a
  non-interactive run, the layer is simply skipped.
- **A recorded "no" does not apply from inside that project.** The in-project check runs before the
  recorded answer is consulted, so a workspace you declined from outside still loads silently when
  you `cd` into its own directory. Declining is about loading someone else's config from a distance,
  not about disabling a project's own config for yourself.

**A project can have a workspace and no machine config.** `doctor`, `config show` and `agent list`
all work before `localharness init` has ever run: they report the machine as unconfigured in one
line and then show the workspace layer anyway, rather than exiting on the missing global
`config.yaml`.

### Creating a workspace

```
localharness init --workspace
```

Run it in the directory you want to be the project root. It creates `./.localharness/` with exactly
two things in it: an `agents/` directory, and a `config.yaml` that is **entirely comments** — the
merge order, the two rules that do not follow "workspace wins", and the settings people reach for
first, all written out and none of them switched on. An all-comment file parses to nothing and
contributes nothing to the merge, so a freshly scaffolded workspace changes no value you had before
you ran the command. That is the intended starting state, not an unfinished one: the workspace
already scopes this project's memory, sessions and logs without a single setting in the file.

It creates nothing else. No `overrides.yaml`, no `divisions/`, no `microagents/`, no state
directory — those appear when something writes them. It never writes a `provider:` block, because
where the model server lives is a fact about the machine.

Two properties worth relying on:

- **It never overwrites a workspace you already have.** If `./.localharness/` exists, the command
  refuses, changes nothing, and exits **1** — deliberately different from plain `localharness init`,
  whose "config already exists" path is an interactive question that exits 0. A script can tell
  "created" from "already there" by the exit code alone.
- **It never prompts.** No question, at any point, so it works in a `Dockerfile`, a setup script or
  CI, where an unanswerable prompt would abort the run.

`--workspace` cannot be combined with the flags that point `init` at a machine or a directory —
`--endpoint`, `--model`, `--config-dir`. That combination exits 2 and names the flag that
conflicted. `--force` is accepted and does not help: the refusal above holds under it too.

### How the two layers combine

When a workspace applies, four files can contribute to one config. Lowest priority first:

1. `~/.localharness/config.yaml` — the global config `localharness init` writes
2. `~/.localharness/overrides.yaml` — the global overlay `localharness components set` writes
3. `{workspace}/.localharness/config.yaml`
4. `{workspace}/.localharness/overrides.yaml`

Each one wins any key an earlier one also set. In one line: **the specific beats the general — the
workspace's word wins wherever the two conflict, and the global layer still governs everything the
workspace is silent about.**

Notice what that means for a value you set globally with `components set`. If a workspace sets the
same key, the workspace wins, and your global value no longer applies while you are working inside
that project. That is deliberate, and it is the cost of the workspace owning its own behavior: a
global default that a project could not override would not be a default, it would be a rule. Your
global value is untouched and still applies everywhere else.

**Nested blocks merge key by key.** A workspace that sets only:

```yaml
provider:
  default_model: qwen3.8-27b
```

keeps the global `base_url`, `provider_type` and every other field under `provider:`. You never
have to restate a whole block to change one field inside it.

**Lists are replaced, not combined.** A workspace that sets `provider.available_models` replaces
the global list rather than adding to it. There is exactly one exception:
`org.permissions.deny_patterns` accumulates across the layers. Every layer's org-level deny rules
are unioned, so a workspace can add deny rules there and can never remove one the global layer set.

Read that guarantee at exactly its own size: it is about the **org-level** list. Deny patterns
written inside an individual agent file travel with that file, and a workspace agent file replaces
the global one of the same name whole — its `permissions.deny_patterns` along with everything else
(see **Agents, divisions and microagents** below). The org-level union still applies on top of
whichever agent file ends up in effect, so the org's rules reach the agent either way; a rule
written into one agent file only survives while that file is the one being used.

**Provider.** A workspace that says nothing about `provider:` uses the global one, which is the
normal case. A workspace may override it, but nothing ever writes a provider block into a
workspace, and `localharness model` always edits the global file. There is one model server on the
machine, and where it lives is a fact about the machine rather than about the folder you happen to
be standing in.

**Agents, divisions and microagents** are a union by name across both layers. A name defined in
only one layer is available either way; a name defined in both resolves to the workspace's file,
and that file replaces the global one entirely instead of being merged into it field by field. So a
workspace's `agents/deployer.yaml` is the whole definition of `deployer` inside that project — no
field leaks across from the global file of the same name.

#### Finding out which file set a key

Four files can set the same key, so "what is this value, and who set it?" is one question with one
answer. Three commands give it, and all three name the layer using the same five words:
`global-config`, `global-overrides`, `workspace-config`, `workspace-overrides` — and `default` for a
value no file set at all.

**`localharness config show`** is the whole picture in one page. It prints the four files in merge
order, lowest priority first, each marked `present` or `missing`, then a table — `key`, `value`,
`set by` — of every key some file actually sets:

```
Config layers, lowest priority first — each one wins any key the ones above it set:
  global-config        present  /home/you/.localharness/config.yaml
  global-overrides     missing  /home/you/.localharness/overrides.yaml
  workspace-config     present  /home/you/proj/.localharness/config.yaml
  workspace-overrides  missing  /home/you/proj/.localharness/overrides.yaml
                           Effective config
┃ key                    ┃ value                   ┃ set by           ┃
│ org.log_level          │ 'info'                  │ global-config    │
│ org.name               │ 'THIS-PROJECT'          │ workspace-config │
Showing the 7 key(s) some file sets, of 182 total. Run with --all to include the 175 compiled-in defaults.
```

`--all` adds the compiled-in defaults; `--json` emits the same thing as `{"layers": [...],
"keys": [...]}` for scripts. With no workspace there are two layer rows, not four with two marked
missing — nothing about the feature appears for someone who never made a workspace.

**`localharness doctor`** answers the narrower question you usually have: what does this project
change? Inside a workspace it prints both layer paths and then one row per key the workspace
actually moves, with the value it displaced:

```
       1 key(s) overridden by this workspace:
         org.name = 'THIS-PROJECT'  [workspace-config]  (global: 'MACHINE-DEFAULT-ORG')
```

It is a comparison of values, not of files: a workspace that restates a key with the same value the
global layer already had produces no row, because nothing changed for you. A workspace that
overrides nothing says so in one line.

**`localharness components list / get / set`** read the workspace layer too, so a value they show
you is the value the running harness is using. `list` takes `--layer` to filter by winning layer,
`get` prints one path with its layer, and all three take `--config-dir` (which, like everywhere
else, switches workspace discovery off).

**`components set` still writes the global `overrides.yaml`, in every project — and now says so in
its own output.** A setting you make inside a project applies on the whole machine; the command
prints the file it wrote, warns that the setting is machine-wide, and tells you where a per-project
value would go instead. If the workspace already sets that key, it also tells you the value you just
wrote is not the one in force here.

**Config errors name the file that actually set them.** A bad value in a workspace `config.yaml` is
reported against that file's path and that file's own line number, not against the global file —
`validate` and `doctor` both. And a malformed `overrides.yaml`, in either layer, is reported as a
parse error with its line and column instead of ending the command in a traceback.

#### What this does NOT cover

- **An `agent:` section in a workspace `overrides.yaml` is not read.** That section is the
  per-agent default layer `components set agent.*` writes, and every writer of that layer is global
  in this release.
- **A standalone `org.yaml` is read from the global directory only** and is not layered. Put org
  settings in the `org:` section of `config.yaml` if you want a workspace to be able to override
  them.
- **Microagent files resolve across both layers, but nothing injects them into a prompt yet.**
  `context.microagents` is parsed and the files are found; the keyword-triggered injection that
  would use them is not built.
- **Which memory a workspace session RECALLS is a separate setting from where it writes.** That
  setting is `agent.memory.recall_scope` — `workspace` (the default), `global` or `both`. The
  default means a session you start inside a project recalls that project's memory and nothing
  else. It moves reads only: no value of it makes a project's session write into your global
  memory. `05-memory.md` §11 has the merged mode, its origin tokens and the four things `both`
  deliberately does not merge.

### Where your work is stored

A workspace layer moves more than config. When one applies, the session's own state is written
inside that project instead of in your global directory.

| Follows the workspace | Stays in your global directory, always |
|---|---|
| `memory.db`, `MEMORY.md`, `history.jsonl` | `GUARDRAILS.md` and `DIVISION.md` (the org/division safety context) |
| `bus-events.jsonl` and `sessions/` | the `KILL` file (it stops the machine's sessions, not one project's) |
| `compact.md` and the memory log | `vllm/server.pid`, `serve.log`, `venv/` (one accelerator, one daemon) |
| `.repl_history` | `plugins/`, packaged `tools/`, the root agent's own file |
| `audit.jsonl` | `overrides.yaml` — a model swap always edits the global file |

The rule behind the table: **what you did in a project stays with that project; what governs the
machine stays with the machine.** Your notes, your history and your audit trail are about the work,
so each of them follows the work into the project. The safety context, the kill switch and the model
server are about the machine, and one of them moving into a project would break the other projects
on it — two projects each holding their own copy of the kill switch means hunting down two files to
stop two sessions.

The practical consequence: **a fresh workspace starts with an empty memory.** Nothing is copied out
of your global memory into it, and nothing in your global memory is moved, rewritten or deleted by
a session you run inside a project. The first session in a new project starts that project's memory
from nothing, the way the first session on a new machine did.

**A session reads what it writes, by default.** Where your memory is written is the table above;
which memory a session reads back is one setting, `agent.memory.recall_scope`, and its default
(`workspace`) makes the two the same. Set it to `global` or `both` and the session still writes
where the table says — the setting moves reads only. `05-memory.md` §11 covers the merged mode and
`/memory promote`, the one command that copies a memory from a project into your global store.

### Environment variables

**There is no environment override for config values.** Nothing in `config/` reads the environment
except `config/paths.py`, and what it reads is where the configuration lives, not what is in it. Two
variables do that:

| Variable | What it does |
|---|---|
| `LOCALHARNESS_DIR` | Use this directory as the config directory instead of `~/.localharness`. |
| `LOCALHARNESS_HOME` | The same thing, consulted only when `LOCALHARNESS_DIR` is unset. |

Both behave exactly like passing `--config-dir`, including the consequence that matters most:
naming a config directory switches workspace discovery off entirely (see "Search Order for Config
Files" above).

Two further variables exist, but they belong to `localharness init` rather than to the config
system — `LOCALHARNESS_ENDPOINT` and `LOCALHARNESS_MODEL` are the environment spellings of its
`--endpoint` and `--model` flags, read by Typer's `envvar=`, and they do nothing for any other
command.

To change a config value from a script, write it: `localharness components set <path> <value>`
edits the global `overrides.yaml`, and a project's `.localharness/config.yaml` is a plain file you
can generate.

---

## 3. Pydantic Models (`config/models.py`)

All models use Pydantic v2. All models are validated at load time, not at access time.

```python
# src/localharness/config/models.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
```

### 3.1 ToolConfig

Controls which tools an agent has access to. Inheritance semantics are defined in Section 5.

```python
class ToolConfig(BaseModel):
    """
    Tool access specification for an agent or division.

    Tool scope resolution order (most specific wins for conflicts):
      global built-ins → division additions → agent additions
      deny lists are evaluated last and always win.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    inherit: list[Literal["global", "division", "org"]] = Field(
        default=["global"],
        description=(
            "Which tool scopes to inherit. 'global' always provides the built-in tools "
            "(glob, grep, read, write, bash). 'division' inherits the division's tool list. "
            "'org' inherits the org-level tool list."
        ),
    )

    add: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to add to this agent's accessible set. "
            "Must be registered tool names or MCP server tool names."
        ),
    )

    deny: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names to explicitly deny. Denial takes precedence over any 'add' or 'inherit'. "
            "Can also be a glob pattern, e.g. 'bash' or 'mcp_*'."
        ),
    )

    mcp_servers: list["MCPServerConfig"] = Field(
        default_factory=list,
        description=(
            "MCP servers to discover tools from. Tools from these servers are added "
            "to this agent's accessible set (subject to 'deny' list)."
        ),
    )

    @field_validator("inherit", mode="before")
    @classmethod
    def normalize_inherit(cls, v: Any) -> list[str]:
        """Accept string shorthand: inherit: division → inherit: [division]"""
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("deny")
    @classmethod
    def validate_deny_not_global(cls, v: list[str]) -> list[str]:
        """Warn if user tries to deny a global built-in — they usually mean something else."""
        global_builtins = {"glob", "grep", "read", "write", "bash"}
        denied_globals = set(v) & global_builtins
        # Do not raise — just allow (user may intentionally deny bash)
        return v
```

### 3.2 MCPServerConfig

```python
class MCPServerConfig(BaseModel):
    """Configuration for an MCP tool server."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        description="Human-readable name for this server. Used in tool prefixing: {name}_{tool_name}.",
    )

    transport: Literal["stdio", "streamable_http"] = Field(
        description="MCP transport type.",
    )

    # stdio transport fields
    command: Optional[str] = Field(
        default=None,
        description="Command to launch the MCP server process (stdio transport).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Arguments to pass to the command (stdio transport).",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables to set for the MCP server process.",
    )

    # streamable_http transport fields
    url: Optional[str] = Field(
        default=None,
        description="Base URL for the MCP server (streamable_http transport).",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="HTTP headers to include in all requests (streamable_http transport).",
    )

    timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Timeout for tool calls to this server.",
    )

    @model_validator(mode="after")
    def validate_transport_fields(self) -> "MCPServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError("MCPServerConfig: 'command' is required for stdio transport")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("MCPServerConfig: 'url' is required for streamable_http transport")
        return self
```

### 3.3 PermissionConfig

```python
class PermissionConfig(BaseModel):
    """
    Permission policy for an agent.

    Default mode is 'auto': allow everything except the deny_patterns list.
    An agent can only narrow its inherited permission policy, never broaden it.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    mode: Literal["auto", "manual"] = Field(
        default="auto",
        description=(
            "'auto': allow all tool calls except those matching deny_patterns. "
            "'manual': deny all tool calls that are not in explicit allow_patterns "
            "(v2 feature — not implemented in v1)."
        ),
    )

    deny_patterns: list[str] = Field(
        # 24 shipped defaults, in four groups — see the table below.
        default_factory=lambda: [
            "write(*/.env)",
            "write(*/secrets*)",
            # … 22 more; config/models.py is the list
        ],
        description=(
            "List of deny patterns. Each pattern is in the form: "
            "'{tool_name}({argument_glob})'. "
            "A tool call is denied if its name and any argument matches a pattern. "
            "Pattern matching uses fnmatch semantics over the RAW pre-expansion argument "
            "strings (no shell parsing). "
            "The deny list is evaluated after inheritance resolution — "
            "an agent's deny list is the UNION of its own list and all inherited lists. "
            "Agents can add to the deny list but never remove inherited entries."
        ),
    )

    allow_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit allow list for 'manual' mode (v2). "
            "In 'auto' mode, this field is ignored."
        ),
    )

    workspace_root: Optional[str] = Field(
        default=None,
        description=(
            "Opt-in filesystem confinement. When set, write/edit target paths and bash_exec "
            "working_dir must resolve (symlink-safe, expanduser'd) INSIDE this directory or "
            "the tool returns a permission_denied error. Default None = UNCONFINED: "
            "file-write capability is a core product feature and stays fully enabled by "
            "default. Harness-run evals set this to an isolated per-run scratch dir."
        ),
    )

    budget: "BudgetConfig" = Field(
        default_factory=lambda: BudgetConfig(),
        description="Execution budget for this agent.",
    )

    @field_validator("deny_patterns")
    @classmethod
    def validate_pattern_format(cls, patterns: list[str]) -> list[str]:
        """
        Validate deny pattern format: must be 'tool_name(arg_glob)' or 'tool_name'.
        """
        pattern_re = re.compile(r"^[a-z_][a-z0-9_]*(\(.+\))?$")
        for p in patterns:
            if not pattern_re.match(p):
                raise ValueError(
                    f"Invalid deny pattern {p!r}. "
                    f"Format: 'tool_name' or 'tool_name(argument_glob)'. "
                    f"Example: 'bash(sudo:*)'"
                )
        return patterns
```

**The 24 shipped deny defaults**, by what they stop:

| Group | Patterns |
|---|---|
| Credential and config writes | `write(*/.env)`, `write(*/secrets*)`, `write(*/config.yaml)`, `write(*/agents/*.yaml)` |
| Privilege escalation, recursive delete, world-writable | `bash_exec(*sudo *)`, `bash_exec(rm -rf *)`, `bash_exec(*rm -rf *)`, `bash_exec(chmod 777 *)` |
| Destructive container / service ops | `bash_exec(*docker stop*)`, `*docker kill*`, `*docker rm*`, `*docker compose down*`, `*docker-compose down*`, `*systemctl stop*`, `*systemctl disable*`, `*systemctl kill*`, `*systemctl mask*` |
| Process and machine control | `bash_exec(*pkill*)`, `*killall*`, `kill *`, `* kill *`, `*shutdown*`, `*reboot*`, `*poweroff*` |

Read-only equivalents stay allowed on purpose — `docker ps`, `docker logs`, `systemctl status`,
`journalctl`.

Because these are unioned into every agent at resolution time, a config that declares
`deny_patterns: []` is not an agent with nothing denied. `config show`, `components get` and
`components list` say so where it would otherwise mislead, appending
`(+24 shipped defaults always enforced)` to an empty value. The count is read off the model, so it
cannot drift from what ships.

**`workspace_root` confines where files are written.** Set it and every `write`/`edit` target path
and every `bash_exec` working directory must resolve inside that directory — symlinks followed
first — or the tool returns `permission_denied`.

Its resolved value depends on whether a workspace layer applies:

- **No workspace** — unset means **unconfined**, and that is the shipped default. LocalHarness is a
  local agent harness whose whole point is acting on your machine, so writing files is a feature,
  not an escape.
- **Inside a workspace** — an agent that sets nothing gets the project folder (the directory holding
  `.localharness/`) filled in by the loader. A value you write yourself always wins, in either case.

Read it at its own size: it is **not a sandbox**. It constrains those tool arguments, not the
process — a command run through `bash_exec` can still leave the folder, and the deny patterns remain
the mechanism that stops specific actions. The harness's own eval runs set it to an isolated per-run
scratch directory. See SECURITY.md for what the trust boundary does and does not cover.

### 3.4 BudgetConfig

```python
class BudgetConfig(BaseModel):
    """Execution budget constraints for one agent session."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    max_actions: int = Field(
        default=100,
        ge=1,
        le=10_000,
        description="Maximum number of tool calls in one session.",
    )

    max_duration_minutes: float = Field(
        default=30.0,
        ge=0.1,
        le=1440.0,  # 24 hours
        description="Maximum wall-clock duration for one session.",
    )

    kill_file: Optional[str] = Field(
        default="~/.localharness/KILL",
        description=(
            "Path to the kill switch file. Agent checks for this file before each iteration. "
            "If it exists, the agent stops immediately and the file is removed. "
            "Set to null to disable the kill switch."
        ),
    )
```

### 3.5 MemoryConfig

```python
class MemoryConfig(BaseModel):
    """Memory backend configuration for an agent."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    # sqlite_path / history_path / notes_path are declared here and read by nothing: the memory
    # store derives all three from its own base_dir. See the note under §4.1.

    max_notes_chars: int = Field(
        default=16_000,
        ge=0,
        le=200_000,
        description=(
            "Maximum characters of MEMORY.md to inject into context on each turn. "
            "Notes are injected from the top (most recent entries are at the bottom — "
            "the whole file is included until this limit is reached)."
        ),
    )

    shared_read: list[Literal["division", "org"]] = Field(
        default_factory=list,
        description=(
            "Which memory scopes this agent can read from in addition to its own. "
            "'division': can read the division's shared.db and DIVISION.md. "
            "'org': can read the org-level GUARDRAILS.md (v2)."
        ),
    )

    inject_into_context: bool = Field(
        default=True,
        description=(
            "If True, inject MEMORY.md contents and recent SQLite facts into the system prompt "
            "at the start of each turn. If False, memory is available via tools only."
        ),
    )
```

### 3.6 ContextConfig

```python
class ContextConfig(BaseModel):
    """Context window management configuration."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    max_context_tokens: int = Field(
        default=131_072,  # DEFAULT_MAX_CONTEXT_TOKENS (config/defaults.py) — single source of truth
        ge=1_000,
        le=2_000_000,
        description=(
            "The FULL context window the server serves, in tokens — init writes the detected "
            "window verbatim. The harness reserves room for the model's reply INSIDE this "
            "number (agent.context.response_reserve), so never subtract an output reservation "
            "here: that reserves it twice. start's guard refuses a value larger than the "
            "served window. The ContextManager uses this to determine when to compact."
        ),
    )

    model_context_overrides: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-model context budgets, keyed by EXACT model name (no globbing). A model listed "
            "here uses its own budget instead of max_context_tokens — consulted by the start-time "
            "window guard and by the budget refit after a /model swap. One global scalar cannot be "
            "correct across models with different served windows; this is how you pin one. "
            "Absent an entry, the scalar applies exactly as before."
        ),
    )

    compaction_threshold_pct: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description=(
            "Trigger summarize-middle compaction when context utilization exceeds this percentage. "
            "Default: 80% — compact when 80% of max_context_tokens is used."
        ),
    )

    preserve_first_n_messages: int = Field(
        default=4,
        ge=1,
        description=(
            "When compacting, always preserve the first N messages in the conversation "
            "(typically: system prompt + initial task). "
            "These are never summarized."
        ),
    )

    preserve_last_n_messages: int = Field(
        default=8,
        ge=2,
        description=(
            "When compacting, always preserve the last N messages "
            "(current working context). These are never summarized."
        ),
    )

    max_tool_output_chars: int = Field(
        default=32_000,
        ge=100,
        le=500_000,
        description=(
            "Maximum characters of tool output to include in a single Observation. "
            "Output exceeding this limit is truncated with a '... [truncated]' suffix."
        ),
    )

    system_prompt_file: Optional[str] = Field(
        default=None,
        description=(
            "Path to a markdown file containing the agent's system prompt. "
            "If not set, a default system prompt is generated from the agent's 'role' field."
        ),
    )

    microagents: list[str] = Field(
        default_factory=list,
        description=(
            "Names of microagent prompt files to inject into context (keyword-triggered, v2). "
            "Each microagent is a .md file in ~/.localharness/microagents/. "
            "v1: all listed microagents are always injected."
        ),
    )
```

#### Per-model context pins (`context.model_context_overrides`)

`max_context_tokens` is a single scalar, and one scalar cannot be right for two models with
different served windows. Swap from a model served at 65,536 to one served at 40,960 and the
scalar is now a lie — compaction never fires and long turns die at the provider's input cap.
A **pin** gives one named model its own budget.

```yaml
context:
  max_context_tokens: 61440           # applies to every model without a pin
  model_context_overrides:
    qwen3.8-27b: 40000                # this model, and only this model, runs on 40,000
```

**Matching is exact-name, and that is the whole rule.** The key is the served model name,
verbatim — no globbing, no prefixes, no wildcards, no case folding. A key that does not match the
served model name is simply never consulted, and the scalar applies instead. **A typo is therefore
silent**: `qwen3.8-27B` when the server reports `qwen3.8-27b` produces no warning and no error, it
just quietly leaves you on the scalar. If a pin does not appear to be taking effect, check the
name against what the provider actually serves before checking anything else.

**Precedence.** A matching pin outranks `max_context_tokens` for that model. Everything downstream
sees the pinned number: the start-time window guard, the session's real budget, `doctor`'s
reconciliation, and the budget refit after a `/model` swap in the REPL all resolve through one
helper (`ContextConfig.resolve_budget`), so they cannot disagree. What a pin does **not** outrank
is the physical served window — see the guard below.

**The guard.** `start` validates the resolved budget against the window the server actually serves.
The bound is that window itself — the reply reserve is held back inside it at runtime, not
subtracted here. An over-window pin aborts startup rather than 400ing mid-session, and the error
names the pinned setting — not the scalar, which would be a fix that changes nothing:

```
Error: context.model_context_overrides['qwen3.8-27b']=80000 exceeds the window served for
'qwen3.8-27b' (65536). It would 400 mid-session. Set
context.model_context_overrides['qwen3.8-27b'] ≤ 65536 (under org: in config.yaml, or
in the agent's yaml — it is not a top-level key), or raise the served window at the server.
```

**Confirmation on the happy path.** A pin that is applied says so at start, so the budget the
session runs on is never a silent assumption:

```
✓ Context budget pinned for qwen3.8-27b: 40,000 tokens
```

`localharness doctor` names the same resolved value when it reconciles config against the served
window, and annotates it as pinned:

```
✓ Context budget 50,000 (pinned for qwen3.8-27b) fits served window 65,536
```

Its remedies are pin-aware too. A budget far below the served window (wasting more than 25% of it)
tells a pinned session to `Raise context.model_context_overrides['<model>']`, and an unpinned one
to run `localharness init` to refit — because for a pinned model, refitting the scalar would be
overruled by the pin on the very next start.

### 3.7 ScheduleConfig

```python
class ScheduleConfig(BaseModel):
    """Scheduled execution configuration (v2 — parsed but not executed in v1)."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    cron: Optional[str] = Field(
        default=None,
        description=(
            "Cron expression for scheduled execution. "
            "Standard 5-field cron: 'minute hour day month weekday'. "
            "Example: '30 5 * * 1-5' = 5:30 AM Monday through Friday."
        ),
    )

    timezone: str = Field(
        default="UTC",
        description=(
            "IANA timezone name for cron evaluation. "
            "Example: 'America/New_York', 'Europe/London', 'Asia/Seoul'."
        ),
    )

    task: Optional[str] = Field(
        default=None,
        description=(
            "Task description or path to task file to run on schedule. "
            "If a path, must be an absolute path or relative to ~/.localharness/."
        ),
    )

    @field_validator("cron")
    @classmethod
    def validate_cron_expression(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        parts = v.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression {v!r}: expected 5 fields "
                f"(minute hour day month weekday), got {len(parts)}"
            )
        return v

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, v: str) -> str:
        import zoneinfo
        try:
            zoneinfo.ZoneInfo(v)
        except zoneinfo.ZoneInfoNotFoundError:
            raise ValueError(f"Unknown timezone {v!r}. Use an IANA timezone name.")
        return v
```

### 3.8 AgentConfig

The primary config model. One YAML file per agent.

```python
class AgentConfig(BaseModel):
    """
    Complete configuration for one agent.

    This is the model that the agent loop receives after inheritance resolution.
    All fields have their final resolved values — no inheritance placeholders remain.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    # --- Identity ---
    name: str = Field(
        description=(
            "Agent identifier. Must be unique across all agents. "
            "Format: lowercase alphanumeric with hyphens. Example: 'morning-briefing'. "
            "This becomes the agent_id used in events and file paths."
        ),
    )

    division: Optional[str] = Field(
        default=None,
        description=(
            "Division this agent belongs to. "
            "Must match the 'name' of an existing DivisionConfig. "
            "If set, the agent inherits division defaults before its own config is applied."
        ),
    )

    role: str = Field(
        description=(
            "Human-readable description of what this agent does. "
            "Used as the basis for the default system prompt if system_prompt_file is not set. "
            "Also used to generate the Agent Card for orchestrator routing."
        ),
    )

    # --- Model ---
    model: str = Field(
        default="inherit",
        description=(
            "LLM model identifier. "
            "Use 'inherit' to use the division's model, or the org default. "
            "Examples: 'qwen2.5:72b', 'Qwen/Qwen3.5-122B-A10B', 'llama3.3:70b'. "
            "The model name must match what the detected LLM backend serves."
        ),
    )

    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature for this agent.",
    )

    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128_000,
        description="Maximum tokens to generate in a single LLM response.",
    )

    # --- Subsystem configs ---
    tools: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Tool access specification.",
    )

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description="Permission policy.",
    )

    memory: MemoryConfig = Field(
        default_factory=MemoryConfig,
        description="Memory backend configuration.",
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
        description="Context window management.",
    )

    schedule: Optional[ScheduleConfig] = Field(
        default=None,
        description="Scheduled execution (v2). Parsed but not executed in v1.",
    )

    # --- Channel ---
    channel: Optional[str] = Field(
        default="terminal",
        description=(
            "Channel for delivering results. "
            "v1: 'terminal' only. "
            "v2: 'discord://channel-name', 'slack://channel-name'."
        ),
    )

    # --- Agent Card (for orchestrator routing) ---
    capabilities: list[str] = Field(
        default_factory=list,
        description=(
            "List of capability keywords for Agent Card routing. "
            "The orchestrator uses these to match tasks to agents. "
            "Examples: ['web-search', 'hacker-news', 'summarization']. "
            "If empty, capabilities are inferred from the role description."
        ),
    )

    tags: list[str] = Field(
        default_factory=list,
        description="Free-form tags for organizing and filtering agents.",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$", v):
            raise ValueError(
                f"Agent name {v!r} is invalid. "
                f"Must be lowercase alphanumeric with hyphens, 1-64 chars, "
                f"start and end with a letter or digit. Example: 'hn-monitor'."
            )
        return v

    @field_validator("model")
    @classmethod
    def validate_model_not_empty(cls, v: str) -> str:
        if v.strip() == "":
            raise ValueError("model cannot be empty. Use 'inherit' to inherit from division/org.")
        return v
```

### 3.9 DivisionConfig

```python
class DivisionConfig(BaseModel):
    """
    Configuration for a division (group of related agents).

    Division config sets defaults for all agents in the division.
    Agent configs override individual fields. Division inherits from OrgConfig.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        description="Division identifier. Lowercase alphanumeric with hyphens.",
    )

    description: str = Field(
        default="",
        description="Human-readable description of this division's purpose.",
    )

    model: str = Field(
        default="inherit",
        description="Default model for all agents in this division. 'inherit' uses org default.",
    )

    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)

    tools: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Shared tool configuration for all agents in this division.",
    )

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description=(
            "Baseline permission policy for this division. "
            "Individual agents may add to the deny list but cannot remove inherited entries."
        ),
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
    )

    shared_memory: Optional[str] = Field(
        default=None,
        description=(
            "Path to the division's shared SQLite database. "
            "Agents with shared_read=['division'] can read from this. "
            "Defaults to ~/.localharness/divisions/{name}/shared.db."
        ),
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9-]{0,62}[a-z0-9]$|^[a-z]$", v):
            raise ValueError(f"Division name {v!r} is invalid. Same rules as agent name.")
        return v
```

### 3.10 OrgConfig

```python
class OrgConfig(BaseModel):
    """
    Organization-level defaults.

    The org config is the lowest-priority ancestor in the inheritance chain.
    All agents inherit org defaults unless overridden at division or agent level.
    """
    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(
        default="default",
        description="Organization identifier.",
    )

    default_model: str = Field(
        default="",
        description=(
            "Default LLM model for all agents. "
            "Must be set either here or in each agent/division config. "
            "Empty string means 'use whatever the LLM backend reports as default'."
        ),
    )

    default_temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1, le=128_000)

    permissions: PermissionConfig = Field(
        default_factory=PermissionConfig,
        description=(
            "Org-level baseline permission policy. "
            "Applied to all agents before division or agent-level overrides."
        ),
    )

    context: ContextConfig = Field(
        default_factory=ContextConfig,
    )

    log_level: Literal["debug", "info", "warning", "error"] = Field(
        default="info",
        description="Structlog log level for the harness process.",
    )

    audit_log_path: Optional[str] = Field(
        default="~/.localharness/audit.jsonl",
        description=(
            "Path to the org-level audit log. "
            "All events from all agents are also written here. "
            "Set to null to disable org-level audit log."
        ),
    )
```

### 3.11 HarnessConfig

The root config written by `localharness init`.

```python
class ProviderType(str):
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    LM_STUDIO = "lm_studio"
    UNKNOWN = "unknown"


class ProviderConfig(BaseModel):
    """Detected LLM provider configuration. Written by localharness init."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    provider_type: str = Field(description="One of the ProviderType constants.")
    base_url: str = Field(description="OpenAI-compatible base URL. Includes /v1 suffix.")
    api_key: str = Field(default="none", description="API key (usually 'none' for local servers).")
    default_model: str = Field(description="First model from the detected backend's model list.")
    available_models: list[str] = Field(default_factory=list)
    supports_function_calling: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the default model supports native function calling. "
            "None means not yet probed. Set by PROV-03 startup probe."
        ),
    )
    timeout_seconds: float = Field(
        default=300.0,
        description="HTTP timeout for LLM API calls. Extended for local models.",
    )


class HarnessConfig(BaseModel):
    """Root harness configuration. Stored at ~/.localharness/config.yaml."""
    model_config = ConfigDict(frozen=False, extra="forbid")

    version: str = Field(default="1", description="Config schema version.")
    provider: ProviderConfig
    org: OrgConfig = Field(default_factory=OrgConfig)
```

---

## 4. Full YAML Config Schema

### 4.1 Field Reference

The keys you are most likely to write by hand, with type, default, and constraints. This is not the
whole schema: `agent.yaml` also accepts the nested blocks `stuck_detector`, `recovery_injection`,
`self_check`, `baton_gate`, `repetition_guard`, `role_sections` and `cruncher`, each a model of its
own in `config/models.py` with its own defaults. `config/models.py` is the complete, authoritative
list — every field is declared there with its own description.

#### agent.yaml fields

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `name` | string | required | `^[a-z][a-z0-9-]{0,62}[a-z0-9]$` | Agent identifier |
| `division` | string or null | null | must match existing division name | Division membership |
| `role` | string | required | non-empty | Role description |
| `model` | string | `"inherit"` | non-empty | LLM model name |
| `temperature` | float | `0.6` | 0.0–2.0 | Sampling temperature |
| `max_tokens` | int | `4096` | 1–128000 | Max response tokens |
| `timeout_seconds` | float or null | null | 30.0–3600.0 | Per-agent HTTP timeout; null uses the provider's |
| `max_subagent_depth` | int | `2` | 1–4 | How deep delegation may nest (1 disables nesting) |
| `channel` | string | `"terminal"` | — | Output channel |
| `capabilities` | list[string] | `[]` | — | Routing keywords |
| `tags` | list[string] | `[]` | — | Organizational tags |
| `tools.inherit` | list[string] | `["global"]` | subset of global/division/org | Inherited tool scopes |
| `tools.add` | list[string] | `[]` | registered tool names | Tools to add |
| `tools.deny` | list[string] | `[]` | tool names or globs | Tools to deny |
| `tools.mcp_servers` | list | `[]` | — | MCP server configs |
| `permissions.mode` | string | `"auto"` | auto or manual | Permission mode |
| `permissions.deny_patterns` | list[string] | 24 shipped defaults | format: `tool(arg_glob)` | Deny patterns. Unioned down the hierarchy — an agent can add, never remove |
| `permissions.workspace_root` | string or null | null, or the project folder inside a workspace | abs or `~/…` path | Filesystem confinement for write/edit/bash_exec. Null and no workspace = unconfined |
| `permissions.budget.max_actions` | int | `100` | 1–10000 | Max tool calls |
| `permissions.budget.max_duration_minutes` | float | `30.0` | 0.1–1440 | Max duration |
| `permissions.budget.max_tool_calls` | int or null | null | 0+ | Separate ceiling on dispatched tool calls; null = `max_actions` alone governs |
| `permissions.budget.kill_file` | string or null | `"KILL"` | bare name, abs, or `~/…` path | Kill switch file. A bare name resolves under the config directory (`~/.localharness/KILL` by default); null disables the kill switch. Read from the global layer only — see "The kill switch" below |
| `memory.max_notes_chars` | int | `16000` | 0–200000 | Chars of `MEMORY.md` injected per turn |
| `memory.shared_read` | list[string] | `[]` | division, org | Extra org-hierarchy scopes this agent may read |
| `memory.recall_scope` | string | `"workspace"` | workspace, global, both | Which STORE recall reads from when a workspace applies. Reads only — writes always go to this session's own store |
| `memory.inject_into_context` | bool | `true` | — | Inject `MEMORY.md` + recent facts into the system prompt each turn |
| `memory.index_mode` | bool | `true` | — | Inline a memory INDEX (name + one line each) and fetch bodies via tools, instead of inlining all of `MEMORY.md` |
| `memory.max_session_history_entries` | int | `8` | 0–200 | Recent session-history lines inlined in index mode |
| `memory.write_gate_enabled` | bool | `true` | — | Auto-capture memory candidates from bus signals (no extra model calls). False leaves the `remember` tool unaffected |
| `memory.trace_ambient_injection` | bool | `true` | — | Record an activation trace for the every-turn ambient injection |
| `memory.consolidation` | map | see `MemoryConsolidationConfig` | — | Idle-time consolidation pass |
| `memory.predictive_gate` | map | see `PredictiveGateConfig` | — | Prediction-error write gate |
| `context.max_context_tokens` | int | `131072` | 1000–2000000 | The FULL served window (init writes it verbatim; the reply reserve is held back inside it) |
| `context.model_context_overrides` | map[string, int] | `{}` | keys match the served model name EXACTLY | Per-model context budget; outranks `max_context_tokens` for that model |
| `context.compaction_threshold_pct` | float | `80.0` | 50.0–99.0 | Compaction trigger |
| `context.preserve_first_n_messages` | int | `4` | 1+ | Preserved prefix count |
| `context.preserve_last_n_messages` | int | `8` | 2+ | Preserved suffix count |
| `context.max_tool_output_chars` | int | `32000` | 100–500000 | Tool output cap |
| `context.system_prompt_file` | string or null | null | abs or ~/… path | System prompt file |
| `context.microagents` | list[string] | `[]` | — | Microagent names |
| `schedule.cron` | string or null | null | 5-field cron | Schedule expression |
| `schedule.timezone` | string | `"UTC"` | IANA timezone | Schedule timezone |
| `schedule.task` | string or null | null | — | Task to run |

**Three memory keys the schema still accepts but nothing reads.** `memory.sqlite_path`,
`memory.history_path` and `memory.notes_path` parse without error and are then ignored: the memory
store derives all three paths from the agent's own directory (`memory.db`, `history.jsonl`,
`MEMORY.md` under `<config-dir>/agents/<name>/`, or under the workspace when one applies). Setting
them moves nothing. They are declarative leftovers, kept only so an older config still loads.

**The kill switch.** `permissions.budget.kill_file` names one file; while that file exists, every
agent session stops at its next iteration boundary. Two things about it are deliberate. It resolves
under the **global** config directory, never a workspace's — one file has to stop every session on
the machine, so a project cannot move it. And since v0.13 its **value** is read from the global
layer alone: a workspace agent or division file that sets `kill_file` is ignored, with a line in the
log saying so. The harness only ever reads the file; removing it again is yours to do.

---

## 5. Inheritance Resolution Algorithm

The inheritance algorithm is applied by `ConfigLoader.resolve()` before an `AgentConfig` is returned. The resolution produces a fully-merged `AgentConfig` with no `"inherit"` placeholders.

### 5.1 Algorithm

```
resolve(agent_name: str) → AgentConfig:

  1. Load OrgConfig from ~/.localharness/org.yaml
     (or OrgConfig() defaults if file does not exist)

  2. Determine division:
     a. Load raw agent YAML
     b. Extract division name from raw YAML (may be null)
     c. If division name set: load DivisionConfig from ~/.localharness/divisions/{name}.yaml
        If file not found: raise ConfigError(f"Division {name!r} not found")
     d. If no division: use DivisionConfig() defaults

  3. Build base config (org → division → agent):
     merged = merge(OrgConfig_as_partial_AgentConfig, DivisionConfig_as_partial_AgentConfig, raw_agent_yaml)

     Merge semantics (field-by-field):
       - Scalar fields (model, temperature, max_tokens):
           If agent sets "inherit" or omits → use division value
           If division sets "inherit" or omits → use org value
           If org omits → use the global overlay's `agent:` section
           If the overlay says nothing → use the field default
         In one line: agent yaml > division > org > overlay > schema default.
         The overlay rung is what `localharness components set agent.temperature 0.2`
         writes; "org omits" is read from the fields the org file actually set, not from
         OrgConfig's own schema defaults, or the org rung would shadow the overlay on
         every install.
       - ToolConfig.inherit, ToolConfig.add, ToolConfig.deny:
           These are ADDITIVE. Agent's tools = union of inherited tools + agent additions, minus denials.
           Denial always wins. See Section 5.2 for tool resolution details.
       - PermissionConfig.deny_patterns:
           ADDITIVE — the resolved deny_patterns is the union of org + division + agent deny_patterns.
           An agent cannot remove a pattern inherited from division or org.
       - PermissionConfig.budget:
           Agent values WIN. If agent sets max_actions: 50, the resolved value is 50 regardless of division/org.
       - MemoryConfig:
           Agent values win for all fields.
       - ContextConfig:
           Agent values win for all fields.

  4. Validate the merged AgentConfig via Pydantic (model_validate)
     On validation failure: raise ConfigValidationError with structured errors (see Section 6)

  5. Return merged, validated AgentConfig
```

### 5.2 Tool Resolution Details

```
tool_resolution(org_tools, division_tools, agent_tools) → set[str]:

  1. Start with global built-in tools: {glob, grep, read, write, bash}
     These are always present unless explicitly denied.

  2. If "org" in agent_tools.inherit OR division_tools.inherit:
     Add org-level tool additions to working set

  3. If "division" in agent_tools.inherit:
     Add division tool additions to working set

  4. Add agent_tools.add to working set

  5. Compute deny_set:
     = org_tools.deny ∪ division_tools.deny ∪ agent_tools.deny

  6. Apply deny_set:
     working_set = working_set - {t for t in working_set if matches_any_deny_pattern(t, deny_set)}

  7. Return working_set
```

### 5.3 Scalar Field Sentinel

The sentinel value `"inherit"` in string fields triggers inheritance lookup. The loader must handle this case:

```python
def _resolve_scalar(
    field: str,
    agent_val: Any,
    division_val: Any,
    org_val: Any,
    default: Any,
) -> Any:
    """
    Return the most specific non-inherit value, or the default if all are inherit/None.
    """
    if agent_val not in (None, "inherit"):
        return agent_val
    if division_val not in (None, "inherit"):
        return division_val
    if org_val not in (None, "inherit"):
        return org_val
    return default
```

---

## 6. Validation Rules and Error Reporting

### 6.1 Validation Stages

ConfigLoader performs validation in three stages:

1. **YAML parse** — `yaml.safe_load()`. Fails on malformed YAML with a `yaml.YAMLError` that includes the line number.
2. **Pydantic validation** — `AgentConfig.model_validate(data)`. Catches type errors, constraint violations, and cross-field dependencies.
3. **Semantic validation** — checks that reference constraints hold (division exists, model name is non-empty after inheritance, file paths are reachable).

### 6.2 ConfigError Hierarchy

```python
class ConfigError(Exception):
    """Base class for all configuration errors."""
    pass


class ConfigParseError(ConfigError):
    """YAML is malformed."""
    def __init__(self, path: str, line: int, column: int, message: str) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.message = message
        super().__init__(f"{path}:{line}:{column}: {message}")


class ConfigValidationError(ConfigError):
    """
    One or more Pydantic validation failures.
    Wraps pydantic.ValidationError with file path context.
    """
    def __init__(self, path: str, errors: list["ConfigFieldError"]) -> None:
        self.path = path
        self.errors = errors
        lines = [f"{path}:"] + [f"  {e}" for e in errors]
        super().__init__("\n".join(lines))


class ConfigFieldError:
    """One validation failure for one field."""
    def __init__(
        self,
        field_path: str,  # e.g. "permissions.budget.max_actions"
        value: Any,
        message: str,
        yaml_line: Optional[int] = None,  # best-effort line number
    ) -> None:
        self.field_path = field_path
        self.value = value
        self.message = message
        self.yaml_line = yaml_line

    def __str__(self) -> str:
        loc = f" (line {self.yaml_line})" if self.yaml_line else ""
        return f"{self.field_path}{loc}: {self.message} (got: {self.value!r})"


class ConfigNotFoundError(ConfigError):
    """Agent or division config file not found."""
    def __init__(self, name: str, searched_paths: list[str]) -> None:
        self.name = name
        self.searched_paths = searched_paths
        paths_str = ", ".join(searched_paths)
        super().__init__(f"Config for {name!r} not found. Searched: {paths_str}")


class ConfigReferenceError(ConfigError):
    """A config field references something that doesn't exist."""
    def __init__(self, path: str, field: str, ref: str, message: str) -> None:
        self.path = path
        self.field = field
        self.ref = ref
        super().__init__(f"{path}: field '{field}' references missing {ref!r}: {message}")
```

### 6.3 Line-Level Error Extraction

Pydantic's `ValidationError` does not include YAML line numbers. The ConfigLoader maintains a YAML line map for best-effort line number reporting:

```python
def _build_line_map(yaml_text: str) -> dict[str, int]:
    """
    Parse YAML text and return a mapping from dot-notation field paths to line numbers.
    Example: {"permissions.budget.max_actions": 14, "name": 1}
    
    Best-effort — only works for simple scalar fields at known nesting depths.
    Complex structures (lists of objects) get the line number of the list key.
    """
    ...
```

### 6.4 Error Output Format

`localharness validate` prints errors in this format, one per line:

```
ERROR ~/.localharness/agents/hn-monitor.yaml
  name (line 1): Agent name 'HN Monitor' is invalid. Must be lowercase alphanumeric with hyphens. (got: 'HN Monitor')
  permissions.budget.max_actions (line 14): value must be >= 1 (got: 0)
  tools.mcp_servers[0].transport (line 22): Input should be 'stdio' or 'streamable_http' (got: 'http')

3 error(s) in ~/.localharness/agents/hn-monitor.yaml
```

On success:
```
OK ~/.localharness/agents/hn-monitor.yaml
OK ~/.localharness/divisions/research.yaml
OK ~/.localharness/org.yaml

All 3 config file(s) valid.
```

---

## 7. ConfigLoader Interface (`config/loader.py`)

```python
# src/localharness/config/loader.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import AgentConfig, DivisionConfig, HarnessConfig, OrgConfig


class ConfigLoader:
    """
    Loads, validates, and resolves LocalHarness configuration files.

    The ConfigLoader is a synchronous dependency. It is constructed once at
    startup and injected into the agent loop constructor. It does not subscribe
    to the event bus.

    Agent configs are cached in memory after first load. Use reload() to
    pick up changes without restarting.

    Usage:
        loader = ConfigLoader()
        config = loader.load_agent("hn-monitor")
        harness_config = loader.load_harness()
    """

    def __init__(
        self,
        *,
        config_dir: Optional[Path] = None,
        local_config_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the ConfigLoader.

        Args:
            config_dir: Base config directory. Resolved through one precedence chain:
                        this argument, else $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                        else ~/.localharness.
            local_config_dir: The workspace layer, or None for "no workspace applies".
                              The loader NEVER discovers one itself — callers name it, and
                              cli/workspace.py's resolve_workspace_layer() is the only thing
                              that computes one. A literal default here would let a stray
                              ./.localharness outrank an explicit config directory, which is
                              the opposite of "an explicit directory is a full replacement".

        Does not read any files at construction time.
        """
        self._config_dir = resolve_config_dir(config_dir)
        self._local_dir = Path(local_config_dir) if local_config_dir is not None else None
        self._agent_cache: dict[str, AgentConfig] = {}
        self._division_cache: dict[str, DivisionConfig] = {}
        self._harness_cache: Optional[HarnessConfig] = None
        self._org_cache: Optional[OrgConfig] = None

    def load_harness(self) -> HarnessConfig:
        """
        Load and validate the root harness config (~/.localharness/config.yaml).

        Returns the cached result on subsequent calls.

        Returns:
            HarnessConfig with provider and org settings.

        Raises:
            ConfigNotFoundError: If config.yaml does not exist.
            ConfigParseError: If YAML is malformed.
            ConfigValidationError: If any field fails validation.
        """
        ...

    def load_agent(self, name: str, *, bypass_cache: bool = False) -> AgentConfig:
        """
        Load, validate, and resolve a named agent config.

        Performs full inheritance resolution (org → division → agent).
        The returned AgentConfig has all 'inherit' sentinels resolved to
        concrete values.

        Args:
            name: Agent name (must match the 'name' field in the YAML, not the filename).
            bypass_cache: If True, re-read and re-validate from disk even if cached.

        Returns:
            Fully resolved AgentConfig.

        Raises:
            ConfigNotFoundError: If no YAML file for this agent name exists.
            ConfigParseError: If YAML is malformed.
            ConfigValidationError: If any field fails Pydantic validation.
            ConfigReferenceError: If a referenced division does not exist.
        """
        ...

    def load_division(self, name: str, *, bypass_cache: bool = False) -> DivisionConfig:
        """
        Load and validate a division config.

        Args:
            name: Division name.
            bypass_cache: Force re-read from disk.

        Returns:
            Validated DivisionConfig.

        Raises:
            ConfigNotFoundError, ConfigParseError, ConfigValidationError
        """
        ...

    def load_org(self) -> OrgConfig:
        """
        Load the org config from ~/.localharness/org.yaml.

        Returns OrgConfig() defaults if the file does not exist (org.yaml is optional).

        Returns:
            OrgConfig with org-level defaults.

        Raises:
            ConfigParseError: If org.yaml exists but is malformed.
            ConfigValidationError: If org.yaml exists but fails validation.
        """
        ...

    def list_agents(self) -> list[str]:
        """
        Return names of all configured agents (from YAML files on disk).

        Searches both config_dir/agents/ and local_config_dir/agents/.
        Names are deduplicated (local takes precedence in case of collision).

        Returns:
            Sorted list of agent names (the 'name' field values, not filenames).
        """
        ...

    def list_divisions(self) -> list[str]:
        """
        Return names of all configured divisions.

        Returns:
            Sorted list of division names.
        """
        ...

    def write_agent(self, config: AgentConfig, *, overwrite: bool = False) -> Path:
        """
        Write an AgentConfig to disk as YAML.

        Used by the orchestrator's agent creation workflow. The config is
        validated before writing.

        Args:
            config: The AgentConfig to write.
            overwrite: If False (default) and the file already exists, raise FileExistsError.
                       If True, back up the existing file to .yaml.bak before overwriting.

        Returns:
            Absolute path to the written YAML file.

        Raises:
            FileExistsError: If overwrite=False and the file already exists.
            ConfigValidationError: If the config fails validation (should not happen
                                   if config was loaded through ConfigLoader, but
                                   defensively re-validated before writing).
            PermissionError: If the config_dir is not writable.
        """
        ...

    def reload(self) -> None:
        """
        Clear all caches, forcing re-read from disk on next access.

        Used after agent config files are modified (e.g., by the orchestrator's
        write_agent() method).
        """
        self._agent_cache.clear()
        self._division_cache.clear()
        self._harness_cache = None
        self._org_cache = None

    def validate_all(self) -> list[tuple[str, Optional["ConfigError"]]]:
        """
        Validate all YAML config files on disk.

        Used by `localharness validate`. Attempts to load every config file
        and collects all errors without raising.

        Returns:
            List of (config_path, error_or_none) tuples.
            error_or_none is None for valid files, or a ConfigError subclass for invalid ones.

        Example:
            results = loader.validate_all()
            for path, err in results:
                if err:
                    print(f"ERROR {path}: {err}")
                else:
                    print(f"OK {path}")
        """
        ...
```

---

## 8. Example YAML Configs

### 8.1 Minimal Agent Config

The smallest valid agent config. All other fields use defaults.

```yaml
# ~/.localharness/agents/hn-monitor.yaml
name: hn-monitor
role: "Fetch the top 10 Hacker News stories each day and write a summary."
```

With this minimal config, the agent:
- Inherits the org default model (or whatever `localharness init` detected)
- Has access to global built-in tools: glob, grep, read, write, bash
- Uses auto permission mode with the default deny patterns
- Has memory at `~/.localharness/agents/hn-monitor/`
- Has a 128,000 token context window
- Has a budget of 100 actions / 30 minutes

### 8.2 Full Agent Config

Every field explicitly set.

```yaml
# ~/.localharness/agents/morning-briefing.yaml
name: morning-briefing
division: financial
role: >
  Generate a daily morning market intelligence report covering portfolio positions,
  overnight news, and key macro indicators. Deliver as structured markdown.
  Target length: 1000-1500 words.

model: qwen2.5:72b
temperature: 0.4
max_tokens: 8192
channel: terminal

capabilities:
  - financial-analysis
  - market-intelligence
  - report-generation
  - web-research

tags:
  - daily
  - financial
  - morning

tools:
  inherit:
    - global
    - division
  add:
    - exa_search
    - exa_crawl
  deny:
    - write(*/.env)
    - bash(sudo:*)
  mcp_servers:
    - name: exa
      transport: stdio
      command: npx
      args: ["-y", "@exa-ai/mcp-server"]
      timeout_seconds: 30.0

permissions:
  mode: auto
  deny_patterns:
    - "write(*/.env)"
    - "write(*/secrets*)"
    - "write(*/config.yaml)"
    - "write(*/agents/*.yaml)"
    - "bash(sudo:*)"
    - "bash(rm -rf *)"
  budget:
    max_actions: 100
    max_duration_minutes: 30.0
    kill_file: ~/.localharness/KILL

memory:
  max_notes_chars: 20000
  shared_read:
    - division
  inject_into_context: true

context:
  max_context_tokens: 131072
  model_context_overrides: {}     # e.g. {qwen3.8-27b: 40000} — EXACT model name
  compaction_threshold_pct: 80.0
  preserve_first_n_messages: 4
  preserve_last_n_messages: 8
  max_tool_output_chars: 32000
  system_prompt_file: ~/.localharness/prompts/morning-briefing.md
  microagents: []

schedule:
  cron: "30 5 * * 1-5"
  timezone: America/New_York
  task: "Generate and deliver the morning intelligence report."
```

### 8.3 Division Config

```yaml
# ~/.localharness/divisions/financial.yaml
name: financial
description: "Agents for financial analysis, portfolio management, and market intelligence."

model: qwen2.5:72b
temperature: 0.4
max_tokens: 8192

tools:
  inherit:
    - global
  add:
    - portfolio_query
    - market_data_fetch
  deny: []
  mcp_servers: []

permissions:
  mode: auto
  deny_patterns:
    - "write(*/.env)"
    - "write(*/secrets*)"
    - "write(*/config.yaml)"
    - "write(*/agents/*.yaml)"
    - "bash(sudo:*)"
    - "bash(rm -rf *)"
  budget:
    max_actions: 200
    max_duration_minutes: 60.0
    kill_file: ~/.localharness/KILL

context:
  max_context_tokens: 131072
  compaction_threshold_pct: 80.0
  preserve_first_n_messages: 4
  preserve_last_n_messages: 8
  max_tool_output_chars: 32000

shared_memory: ~/.localharness/divisions/financial/shared.db
```

### 8.4 Org Config

```yaml
# ~/.localharness/org.yaml
name: default

default_model: qwen2.5:72b
default_temperature: 0.6
default_max_tokens: 4096

permissions:
  mode: auto
  deny_patterns:
    - "write(*/.env)"
    - "write(*/secrets*)"
    - "write(*/config.yaml)"
    - "write(*/agents/*.yaml)"
    - "bash(sudo:*)"
    - "bash(rm -rf *)"
    - "bash(chmod 777 *)"
  budget:
    max_actions: 100
    max_duration_minutes: 30.0
    kill_file: ~/.localharness/KILL

context:
  max_context_tokens: 131072
  compaction_threshold_pct: 80.0
  preserve_first_n_messages: 4
  preserve_last_n_messages: 8
  max_tool_output_chars: 32000

log_level: info
audit_log_path: ~/.localharness/audit.jsonl
```

### 8.5 Root Harness Config (written by `localharness init`)

```yaml
# ~/.localharness/config.yaml
version: "1"

provider:
  provider_type: ollama
  base_url: http://localhost:11434/v1
  api_key: "none"
  default_model: qwen2.5:72b
  available_models:
    - qwen2.5:72b
    - llama3.3:70b
    - nomic-embed-text
  supports_function_calling: true
  timeout_seconds: 600.0
  inference_queue_wait_seconds: 600.0

org:
  name: default
  default_model: qwen2.5:72b
  log_level: info
  audit_log_path: ~/.localharness/audit.jsonl
```

`init` serializes the whole validated `HarnessConfig`, so the file it writes contains every
field of every model with its resolved value — including a `server:` block for a
harness-managed runtime — not just the keys shown here. This example is the shape, not a
transcript. Two values above are the ones people most often expect to differ from what
`init` actually writes: `timeout_seconds` is `600.0`, chosen for slow local single-stream
decode, and `inference_queue_wait_seconds` bounds the wait for the local inference gate
rather than generation itself.

### 8.6 Multi-Agent Hierarchy Example

This shows how three agents inherit from one division.

```yaml
# ~/.localharness/divisions/research.yaml
name: research
description: "Research agents for model evaluation and memory research."
model: llama3.3:70b
tools:
  inherit: [global]
  add: [exa_search]
permissions:
  budget:
    max_actions: 150
    max_duration_minutes: 45.0
```

```yaml
# ~/.localharness/agents/model-research.yaml
name: model-research
division: research
role: "Research and evaluate new LLM model releases. Compare benchmarks and summarize findings."
model: inherit          # uses research division default: llama3.3:70b
tools:
  inherit: [division]
  add: [exa_crawl]     # adds exa_crawl on top of division's exa_search
```

```yaml
# ~/.localharness/agents/memory-research.yaml
name: memory-research
division: research
role: "Research memory and retrieval techniques for agent systems."
model: inherit
tools:
  inherit: [division]   # same tools as model-research (just exa_search)
context:
  max_context_tokens: 200000  # overrides division default
```

```yaml
# ~/.localharness/agents/quick-lookup.yaml
name: quick-lookup
division: research
role: "Answer quick factual questions with minimal tool use."
model: llama3.1:8b     # smaller model for fast lookup — overrides division
permissions:
  budget:
    max_actions: 10    # much tighter budget
    max_duration_minutes: 5.0
```

After inheritance resolution for `model-research`:
- `model`: `llama3.3:70b` (from division)
- `tools.add`: `[exa_search, exa_crawl]` (division's exa_search + agent's exa_crawl)
- `permissions.deny_patterns`: union of org + division + agent deny lists
- `permissions.budget.max_actions`: `150` (from division — agent didn't override)
- `context.max_context_tokens`: `131072` (from org default — neither agent nor division overrode it)

---

## 9. `localharness validate` Contract

The `validate` command is implemented in `cli/validate_cmd.py`. It must satisfy these requirements (CFG-04):

### 9.1 Exit Codes

| Exit Code | Meaning |
|-----------|---------|
| `0` | All configs valid |
| `1` | One or more configs invalid |
| `2` | No config files found (non-error, no agents configured) |

### 9.2 Behavior

```
localharness validate [OPTIONS] [PATH]

Arguments:
  PATH            Path to specific YAML to validate. If not set, validates all.

Options:
  --config-dir TEXT  Config directory. Default: $LOCALHARNESS_DIR, else $LOCALHARNESS_HOME,
                     else ~/.localharness.  [env var: LOCALHARNESS_DIR]
  --strict           Reserved. No warning-level checks exist yet; currently identical to
                     the default.
  --no-input         Never ask about an untrusted workspace: skip its config layer, say so,
                     and record nothing. For hooks, CI, and any run with no one watching.
```

That is the whole surface. There is no `--json`, no `--agent`, no `--verbose`. `--strict` is
declared and currently does nothing but say so: the command has no warning level for it to promote.

### 9.3 What Is Validated

`ConfigLoader.validate_all()` walks, across both layers: the global `config.yaml`, the global
`org.yaml`, every `divisions/*.yaml`, and every `agents/*.yaml`. Each file is loaded **at its own
path**, so a workspace file and a global file with the same name are two independent verdicts.

For each file:

1. YAML parses without error
2. All required fields are present (`name`, `role` for agents)
3. All field values satisfy type and constraint rules
4. Inheritance resolution completes without errors — a `division` that does not exist is a
   resolution error here

Everything validate reports is an error. It does not check that `system_prompt_file` exists, that an
MCP `command` is executable, or that a tool name is registered — those are runtime concerns, and no
warning level exists to carry them.

---

## 10. Implementation Notes

### 10.1 YAML Safety

Always use `yaml.safe_load()`. Never `yaml.load()` (arbitrary Python object deserialization). This is enforced by a ruff rule:

```toml
# pyproject.toml
[tool.ruff.lint]
extend-select = ["S506"]  # Probable use of unsafe loader
```

### 10.2 Path Expansion

All path fields support `~` expansion and must be expanded via `Path(v).expanduser()` before use, at the point of use rather than at load time. Paths are stored as strings in the model (not Path objects) for YAML round-trip compatibility.

### 10.3 pydantic-yaml for Round-Trip

Use `pydantic-yaml` for serializing `AgentConfig` back to YAML:

```python
from pydantic_yaml import to_yaml_str, parse_yaml_raw_as

# Read
config = parse_yaml_raw_as(AgentConfig, yaml_text)

# Write (in ConfigLoader.write_agent)
yaml_text = to_yaml_str(config)
path.write_text(yaml_text, encoding="utf-8")
```

### 10.4 Caching Strategy

The `ConfigLoader` caches parsed and resolved configs in memory. Cache is invalidated by calling `reload()`. The agent loop calls `load_agent()` once per `run_turn()` invocation (not once per iteration). This means config changes take effect on the next task invocation, not mid-session.

### 10.5 No Side Effects in Validators

Pydantic validators must not perform I/O (file existence checks, network calls). These checks are deferred to the `validate_all()` method's semantic validation phase. This ensures that model construction is fast and testable without real filesystem state.

### 10.6 Test Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path
from localharness.config.models import AgentConfig, OrgConfig

@pytest.fixture
def minimal_agent_config() -> AgentConfig:
    return AgentConfig(name="test-agent", role="Test agent for unit tests.")

@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A temporary config directory with the expected subdirectories.

    Tests build their own ConfigLoader on it, so a test that needs a workspace
    layer can pass `local_config_dir=` too."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "divisions").mkdir()
    return tmp_path
```

### 10.7 Mandatory Unit Tests (`tests/unit/test_config_loader.py`)

| Test | What it verifies |
|------|-----------------|
| `test_minimal_agent_config_valid` | Minimal YAML (name + role) loads without error |
| `test_invalid_name_rejected` | CamelCase name raises ConfigValidationError |
| `test_inherit_sentinel_resolved` | model: inherit resolves to division model |
| `test_deny_patterns_union` | Agent's deny_patterns are unioned with division's, not replaced |
| `test_tool_deny_wins_over_add` | Tool in both add and deny results in tool being denied |
| `test_division_not_found_raises` | division: nonexistent raises ConfigReferenceError |
| `test_budget_agent_wins` | Agent's max_actions overrides division's |
| `test_write_agent_creates_file` | write_agent() creates YAML on disk |
| `test_write_agent_backup_on_overwrite` | overwrite=True creates .yaml.bak |
| `test_validate_all_returns_results` | validate_all() returns one tuple per config file |
| `test_yaml_safe_load_only` | Mock yaml.load raises — verify loader doesn't call it |
| `test_cron_five_fields_required` | 6-field cron raises ConfigValidationError |
| `test_invalid_timezone_rejected` | timezone: NotAPlace raises ConfigValidationError |
| `test_mcp_stdio_requires_command` | stdio transport without command raises |
| `test_mcp_http_requires_url` | streamable_http without url raises |
