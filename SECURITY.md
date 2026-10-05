# Security Policy

## Supported versions

LocalHarness is early-stage (v0.x). Security fixes land on the latest `main`;
there are no long-term-support branches yet.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✓ |
| older tags | ✗ |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public issue.

Use GitHub's private vulnerability reporting:
[**Report a vulnerability**](https://github.com/ahwurm/localharness/security/advisories/new).
Include reproduction steps and impact. You will get an acknowledgment, and a fix
or mitigation will be coordinated before any public disclosure.

## Trust boundaries

LocalHarness runs tools — including `bash` and file writes — on the machine where
the harness runs, driven by a local model. **Treat agent definitions, any
connected MCP servers and any plugin you turn on as trusted code**: review them the way you
would review code, because they decide what the agents are allowed to do.

**Workspace config from outside your project is not trusted by default.** From v0.13 the harness
looks for a `.localharness/` directory at or above your current directory and can load agent and
division files from it. If that directory is the one you are standing in, or it sits inside the same
git repository you are working in, it loads straight away — it is part of the project you already
opened, and every directory inside a project inherits that project's config. If it sits somewhere
else — above your repository's root, or in a parent folder while you are not in a repository at all
— the harness asks once before loading it. Your answer is recorded in your global
`~/.localharness/trusted_workspaces.yaml`, never inside the workspace itself, so a directory can
never vouch for itself. Edit that file to change an answer. When there is no terminal to ask — a
script, a cron job, CI — that workspace layer is ignored and the run continues without it. `start`,
`doctor`, `validate` and `agent create` also take `--no-input`, which declines to be asked at all:
the layer is skipped, the run says so, and nothing is recorded. Use it wherever a process might otherwise answer a
permanent trust question on your behalf — hooks, CI, anything scheduled.

**Three edges of that gate, as it behaves today.** Each is a case where the "inside your project"
test lands narrower or wider than you might guess. Know which before you rely on it.

- A **linked git worktree counts as inside the checkout it was cut from**. Its `.git` is a file
  naming the parent repository (`gitdir: …`), which the walk reads — so the main checkout's
  `.localharness/` loads with no prompt while you work in a worktree of it, exactly as it would in
  the checkout itself. A submodule reads the same way about its superproject. The exposure is the
  one below it: config in a repository you cloned loads because you opened that repository.
- A **folder that is not in a git repository** counts as in-project only at the exact directory that
  holds `.localharness/`. From a subdirectory of it, the workspace is external — asked about if there
  is a terminal, skipped if there is not.
- A recorded **"no" does not apply from inside that project**. The in-project test runs before the
  recorded answer is read, so a workspace you declined from outside loads silently once you are
  standing in it. Declining is a decision about loading a distant directory's config, not a way to
  disable a project's own config while you work in it.

**What a project's own files may do, and what waits for your Yes.** An in-project `.localharness/`
loads, but it may set only what changes the agent's behaviour inside that project: a setting that
says where a request or a key goes, or what the harness launches or imports, belongs to your machine
(see [Machine-level-only settings](#machine-level-only-settings)), and a project's permission
settings only ever tighten, trusted or not. The programs a project's agent files name — MCP
servers, which run with your environment or connect with the headers set there — start only in a
project you trusted, and only the set you approved. The trust question `localharness start` asks in
a project lists them when there are any (so does the question about a workspace outside your
project), and a Yes records exactly what you were shown: each server's command, arguments,
environment-variable names, URL and header names. No, or no terminal to ask, starts none of them:
one line names the files and how to change that for the command you ran — `--trust-project` on
`start`, `LOCALHARNESS_TRUST_PROJECT=1` for `localharness mobile` and `localharness acp`, a Yes to the
trust question at a `localharness start` on a terminal, or the project's entry in
`trusted_workspaces.yaml` — and in the default `auto` mode the session runs `guarded`. A Yes given
where the list was not shown — the question a channel asks during a session, in Zed or Discord —
approves no server; they are shown at the next `localharness start` on a terminal. When the set
changes, that next start shows what changed and asks again; a No starts none and is asked again at
the start after; a start with no terminal starts none and says so; a server that went away is
recorded without a question. `--trust-project` or `LOCALHARNESS_TRUST_PROJECT=1` trusts the project
for that one run, starts its servers and records nothing — for CI and scripts. Session files inside
a repository never count as having worked there; only your machine's own store does (see the trust
question under the approval gate). An agent or division file in a project's `.localharness/` that
is a symlink leading outside it — or that sits in a symlinked folder leading outside it — is
ignored, with one warning naming the link (`localharness validate` names it the same way and does
not check it), and the machine's file of that name is used instead: a project's agent files must be
files inside its `.localharness/`, judged by where they sit, never by where they point. That rule is
for a project's files only: a symlink among your machine's own agent files loads as the file it
points to, as a dotfiles setup expects. What your machine's agent files and the `agent:` section of
your `overrides.yaml` start, load or loosen waits for one Yes instead — see the paragraph on what
the agent may change about its own setup, under the approval gate.

**What this does NOT cover.** If you clone someone's repository and run the harness inside it, that
repository's `.localharness/` config and agent files load with no prompt, because you are inside
that project. They decide an agent's role, prompt, model and tools, so read them in an unfamiliar
repository before you run the harness there, the same way you would read its build scripts. A trust
Yes covers what you were shown and no more: the scripts and files an approved server's command runs
from inside the repository are not fingerprinted (the same gap as a granted `python3 <script>`,
under the approval gate's named gaps), so a change to them asks nothing; nor are the values of
environment variables and headers, of which only the names are kept, because values are often
secrets. The first start after upgrading to this release adopts, without asking, the servers that a
project you had already trusted names at that moment, and your own agent files, the `agent:`
section of your `overrides.yaml` and tool scripts as they are; later changes are what it asks
about. And the record a Yes writes is an ordinary file in
your config folder, `trusted_workspaces.yaml`: the agent's write and edit tools and the shell writes
the gate can read reach it only with your approval (in every mode but `unattended`, which approves
everything), but code the agent is already running — an interpreter tool, which `auto` runs without
asking — can rewrite any file you own, that record included, and a start believes the record it
finds. These questions guard what the agent writes with its file tools and the shell,
not what code that is already running can do. Plugin
code and the org-level guardrails file are never taken from a workspace. Plugins are found only in
Python packages installed alongside LocalHarness that declare a `localharness.plugins` entry point,
and in the `plugins/` folder of your global (machine-level) config directory; nothing of a plugin
you installed is imported until you turn it on (see [Plugins](#plugins) below). The guardrails
file, `GUARDRAILS.md`, is read by the harness itself from your global config directory on every
turn of the agent you talk to, whether or not memory is on, so a workspace cannot silence the org's
safety context by shipping its own copy of the file, and cannot blank it by having no copy at all.
The agents it delegates to (subagents) and bench runs are not given the file. One crossing does
exist and it is yours to make:
`/memory promote` copies a memory out of a project's store into your machine-global store, so a
memory learned inside an untrusted repository can reach your global memory **if you promote it**.
The harness never promotes anything on its own — nothing runs it, nothing suggests it — and the
promoted copy records which project it came from, so you can see the origin and undo the copy.

**`workspace_root` is yours to set, and nothing sets it for you.** v0.13 quietly filled it in with
the project folder whenever a workspace layer applied, so "where files are written" was decided in
two places at once — a silent config default and the gate. It is one place now: the gate derives
the boundary from where you stand and decides what asks, and the loader fills nothing in.
`permissions.workspace_root` written in your own config is still a **hard confinement**, and the
strictest thing in this document: every `write`/`edit` target path and every `bash_exec` working
directory must resolve inside it, symlinks followed first, or the tool returns `permission_denied`
— a refusal, not a question, with no prompt and no grant that can lift it. Unset is the default,
inside a workspace and out, and means unconfined. Read that honestly: unconfined is the shipped
posture, and the gate — not a path check — is what stands in front of a tool call. Even when you do
set it, it is not a sandbox: a command run through `bash_exec` can still leave the folder, and the
deny patterns remain the mechanism that stops specific actions.

### Plugins

A plugin is code that adds tools, commands, checks or settings to the harness. One trust model
covers every plugin; where a plugin that ships with LocalHarness and one you install yourself are
treated differently, it says so below. (Five features ship as plugins: image generation, off until you
turn it on, and the phone app `mobile`, memory, Discord (`dispatch`) and autoresearch, each on by
default; everything else that comes with LocalHarness is built into its core, including the bench
and its sealed holdout.) A plugin that ships with LocalHarness may own core settings under their old
names (`autoresearch` owns `proposer:` and `sentinel:`); a plugin you install that tries to is
refused at load.

**Stored credentials are masked wherever a command shows them.** The settings LocalHarness treats as
secrets are `provider.api_key`, the `api_key` and `extra_headers` of each `extra_endpoints` entry,
`active_endpoint.api_key`, an MCP server's `env` and `headers`, `proposer.api_key` and
`dispatch.discord.token`. Each is shown as `**********` wherever a command displays it:
`components get`, `components list` and `components set` (its receipt and its audit record),
`config show` and its `--json`, `validate`, `doctor`, error text, and the message the phone shows
when its session fails to start. Error text masks those values only, never the name or address of
an endpoint around them, so it still says where the problem is. The harness sends the real value
only where it belongs — the key to its server, an MCP server's `env` to that server's process —
and the file it writes a secret to (`overrides.yaml`, or `config.yaml` from `init`) holds the real
value, owner-only (0600); a file you write by hand keeps the mode you give it.
`localharness components set <secret setting> -` reads the value with input hidden on a terminal,
or one line from standard input, so a key never has to sit in your shell history or in `ps`. A
setup question for a key that is already stored never shows it and says what Enter does: it keeps
the stored key, and `none` clears it. In 0.16.0 a config that failed to load or validate could
print a stored secret, whole or by its last characters, in the error text of commands such as
`doctor`, `start` and `components set`, and `validate` could print an MCP server's `env` and
`headers` values; both are fixed. **What this does NOT cover:** the phone's failed-start message
masks the provider's, the endpoints' and the proposer's keys, so the Discord token or an MCP
server's `env` or `headers` value quoted in that one error would reach the phone as it is.

- **Found is not on.** A plugin is found from package metadata and folder names alone, and one you
  installed stays off, with none of its code imported, until you turn it on. `localharness start`,
  `localharness doctor` and `localharness plugins list` say it is available and print the command
  that turns it on.
- **Turning on a plugin you installed is a machine-level act, and it is your trust grant.**
  `localharness plugins enable <name>` writes your machine-level `overrides.yaml`. A project can
  never turn such a plugin on: its value for `<name>.enabled` is ignored with a warning, and
  `plugins enable --workspace` refuses. In a session, `/plugins enable <name>` for a plugin you
  installed first asks one yes/no question that says its code will run with your permissions in
  every session; No leaves it off and the conversation goes on. A plugin that ships with
  LocalHarness can be switched on or
  off per project, like any other setting. A project can also change a plugin's ordinary settings,
  but not one the plugin marks machine-level only (endpoints, credentials, access lists), so that
  protection is only as good as the plugin's marking.
- **What turning it on vouches for.** The safety checks believe what a tool declares about itself
  (step 4 of the approval gate, and the prompt-injection section). The one exception is the
  permission gate: a `gate_family` declared by a tool from a plugin you installed counts only when
  the gate treats it at least as strictly as no family at all (`code` and `delegate`), so a plugin
  cannot declare its way past a question. Plugins that ship with LocalHarness are reviewed with the
  rest of the code and are exempt from that exception.
- **Plugin code is trusted code.** It runs inside the harness process with your privileges and can
  do anything the harness can, including things no tool call shows: the gate judges tool calls, not
  a plugin's own code. This is the stance taken for MCP servers, except that a plugin runs inside
  the harness process itself. The harness contains a plugin's failures, not its intent: a plugin
  that raises or exits while loading or starting is disabled for the session, its tools and hooks
  are taken out, and it is named in the startup summary (`localharness doctor` names one that fails
  to load or configure); a tool of its that raises returns an error to the model instead of ending
  the turn.
- **`plugins/**` stays protected.** It is on the protected list in step 2 of the approval gate, in
  your machine-level config directory and in a project's `.localharness/`, because it is code the
  harness imports.

**What this does NOT cover.** A malicious plugin you turned on: nothing here contains code that
means harm once it runs. A plugin that hangs instead of failing: containment catches errors, not
stalls. And anything a plugin registers directly on the tool registry, a tool or a raw pre/post
hook, instead of handing it over through its `tools()` method or the hook system, skips the
gate-family rule above and stays registered if the plugin later fails.

### Machine-level-only settings

A project's `.localharness/` may set what the agent does inside that project: the model it asks
for, budgets, prompts and agent structure, memory tuning, and stricter permissions. A setting that
says where a request or a credential goes, which credential is sent, what program the harness starts
or imports, or which file outside the project it writes belongs to your machine, so only your
machine-level (global) `config.yaml` or `overrides.yaml` may set it. A repository you cloned must
not be able to send your API key or your conversation to an address it picks, start a program of
its choosing, or switch a protection off — and the address alone is enough to leak, because every
message and every file the agent reads travels to it. A project's value for one of these settings
is ignored, the harness prints a warning naming the key and the file, and your global value stands;
if your global config sets no model server at all, `start` stops with one line naming the file to
set `provider.base_url` in. The autoresearch proposer's address and key are two of them, because
`plugins enable autoresearch` sends the key to that address; with no proposer address in your
global config, a project's whole `proposer:` section is ignored. These are all of them:

- `provider.base_url`
- `provider.api_key`
- `extra_endpoints` (the whole list of peer endpoints)
- `active_endpoint`
- `server` (the whole section: what the harness launches, with which arguments and on which
  address)
- `org.audit_log_path`
- `org.hooks`
- `org.enforce_capability_floor`
- `org.web_fetch_allow_private`
- `channels.remote_unattended`
- `proposer.base_url`
- `proposer.api_key`
- `image.comfyui_url`
- `image.workflow`
- `dispatch.discord.token`
- `dispatch.discord.allow`
- `dispatch.discord.channels`
- `mobile.public_url`
- `permissions.ask.read_only_signatures`
- `permissions.ask.dropped_commands`
- `permissions.ask.wrapper_commands`
- `permissions.ask.subcommand_tools`
- `permissions.ask.mcp_trusted_servers`
- `permissions.ask.timeout_s`
- `memory.embedding_model`, in an agent file (the model the memory plugin loads, and what a cache
  miss downloads)
- `permissions.budget.kill_file`, in an agent file (the one kill switch for every agent on the
  machine)
- `<name>.enabled`, for a plugin you installed (a plugin that ships with LocalHarness can be
  switched per project)

A few other permission settings are narrowed instead of ignored: a project may add deny patterns
and add commands to the gate's lists of dangerous calls but never remove any, may switch the
network-host question on but not off, may not pick a looser `permissions.mode`, and may not move
`permissions.workspace_root` outward.

**What this does NOT cover.** Your global config is trusted as it is: a value already in your
global files is never checked, whoever put it there. A project value equal to your global value is
treated as yours and left alone. A plugin you install decides for itself which of its settings are
machine-level only; this list covers the plugins that ship with LocalHarness. The other `org`
settings a project may set — its name, default model, temperature, output cap, context settings,
log level and the memory switch — choose nothing outside the project. One key is announced
differently: a project agent file's `permissions.budget.kill_file` is ignored like the others, but
its warning names the agent rather than the file and is logged rather than printed in the start
summary.

## Human approval gate

From v0.14 every tool call of every agent — subagents included — passes one decision function
before it runs. The function is code, not a judgment call by a model: the same call in the same
workspace always gets the same answer. It runs in a fixed order and the first match wins.

**The default mode is `auto`: one question about the workspace, then a blacklist.** v0.14.0
shipped `guarded` as the default — ask once about each new thing, remember the answer — and in
real use it stopped people during ordinary work. A gate that interrupts ordinary work trains you
to approve without reading, so from v0.14.1 the shape is different: the first time a session opens
a workspace you are asked **once** whether you trust it, and after that everything runs except a
named list of dangerous operations.

**The trust question, and the three ways a session gets past it.** They are tried in this order,
and only the last one is a prompt (`cli/session_trust.py`):

1. **A recorded decision.** `~/.localharness/trusted_workspaces.yaml` is consulted for this root
   and every directory above it, so a nested folder inherits its project's answer. A recorded
   "no" is honored too: the session runs `guarded`.
2. **Evidence that you have already worked here — on this machine, never in the repository.** For
   a session rooted at `$HOME` with no project, earlier sessions in your machine's own store
   (`agents/*/sessions/*.jsonl` under your global config directory) count as prior use: a place
   you have worked in is not a place to be asked about, so the harness records the trust, prints
   one line saying it recognized the workspace, and moves on. A project's own
   `.localharness/agents/*/sessions/` never counts — those files can be committed to a
   repository, so a fresh clone would arrive looking worked-in — and a project with no record is
   asked like any new one. One old home session can never vouch for a project directory nobody
   has opened.
3. **The question**, asked through whatever channel you are on: inline in the terminal, a dialog
   in Zed, a message in Discord. A "yes" writes the trust record and is never asked here again —
   one record, both halves: the project's `.localharness/` config loads, and its tool calls run
   under `auto`. When `localharness start` asks it on a terminal, before the session opens, it
   lists the MCP servers the project's agent files start and the yes approves that list; the same
   question asked during a session (Zed, Discord, the phone) shows no list, so its yes — like a
   recognized workspace's — approves no server, and they are asked about at the next
   `localharness start` on a terminal. A "no" is recorded too, and the session runs `guarded`.

**A session that cannot ask and has no record runs `guarded` and records nothing** — fail closed,
and leave the question for the next interactive session in that directory rather than answering it
on that person's behalf. **An explicitly configured `permissions.mode` skips all of this**:
`guarded` and `read-only` already ask or refuse, and `trusted` and `unattended` are deliberate
loosenings someone typed into a config, so confirming a decision you just made is exactly the
fatigue this release removes. It does not skip the decision about a project's MCP servers: those
still start only in a project you trusted, with the set you approved, whatever the mode.

In Zed the question is a permission dialog with two buttons — *Trust this workspace* and *Not
now* — because its answer is permanent, unlike every other dialog the gate raises there.

**There is no allow-list in `auto`.** Nothing is enumerated as safe; everything is allowed unless
it is on the blacklist in step 2 below, and that list is the thing to curate. This is deliberate
and it is the honest weakness: a whitelist fails closed on the operation nobody thought of, and a
blacklist runs it. We chose the blacklist because a whitelist of "ordinary work" is the thing that
was interrupting people, and because a list of ways to lose data irreversibly is short enough to
read and argue with. Each step below says what it does in `auto`.

1. **Deny.** Your deny patterns, unchanged from earlier versions. Nothing overrides them — not a
   grant, not a mode. Two shipped defaults matter for how the rest of this section reads, and both
   are v0.13 entries that v0.14.1 did not touch: **`bash_exec(rm -rf *)` and
   `bash_exec(*rm -rf *)` refuse `rm -rf` outright** — inside your project or outside it, embedded
   in a `cd x && rm -rf y` or not — before the gate classifies anything, and `write(*/.env)` does
   the same for `.env` writes. So "a delete inside the project runs without asking" below means the
   deletes this list has not already refused; a plain `rm -rf build` is *denied*, not allowed and
   not asked about. If you want in-project `rm -rf` to run, remove that entry from
   `permissions.deny_patterns` — it is your list, and the gate's blacklist will then ask about the
   ones that point outside the project.
2. **`AUTO_BLACKLIST`: parked for a human, never a blocking prompt, and no answer is
   remembered.** In a trusted workspace this is the **whole** of what still needs a person.
   Everything not on it runs without asking. Since 0.14.2 a hit here does not stop the loop: the
   call is staged as a pending decision (`/pending`, `/approve N`, `/deny N`; ctrl+y / ctrl+n in
   the terminal, ✅ / ❌ on the Discord notice), the model is told to continue without that step,
   and an approval is a one-shot pass for that exact call — tool plus arguments — so approving
   one `rm -rf` target never covers another. The queue lives in the session and is not persisted. It is one structure in
   `agent/gate_types.py` with four fields, listed here in that order:
   - **`target_scoped_verbs` — a delete or a recursive permission change whose target is outside
     the project, protected, or unresolvable.** `rm`, `rmdir`, `chmod`, `chown`, `chgrp`,
     `truncate`, `find` (with `-delete` or `-exec`), and the Windows spellings `Remove-Item`,
     `ri`, `del`, `erase`, `rd`. Nothing about these verbs is dangerous on its own — `rm -rf
     build` is what a build script does — so only the target decides, and a target the classifier
     cannot place (a variable, a glob, a substitution, a `cd` it could not follow, or no target at
     all) counts as outside: the whole narrowing rests on knowing where the command points. Note
     what step 1 already did: with the shipped deny patterns in place, `rm -rf` never reaches this
     rule at all — it is refused outright either way — so in practice this rule governs the other
     verbs, and `rm -rf` only for someone who removed that pattern.
   - **`irreversible_signatures` — commands that ask wherever they point.** `sudo`, `su`, `doas`;
     `dd`, `mkfs`, `shred`, `format`, `diskpart`; `git push --force`, `git push --delete`,
     `git reset --hard`, `git clean -f`. Matched on the canonical signature, so `git push -f` and
     `git push --force-with-lease` are the one `git push --force` entry.
   - **`protected_paths_workspace` — in-project writes to `.git/**`, plus the config-directory
     list below applied to a project's own `.localharness/`** (`config.yaml`, `overrides.yaml`,
     `plugins/**`). `.git` keeps its whole subtree: there is no part of it you write by hand, and
     `.git/hooks` and `.git/config` both re-point what the next ordinary git command executes. The
     rest of a project's `.localharness/` — agents, tools, state, memory — is ordinary project
     content and is not protected, and neither are `.env` files or keys inside your own repository,
     which `guarded` still asks about.
     The home and system protected sets apply in full on top of this: `~/.ssh`, `~/.aws`,
     `~/.gnupg`, `~/.config/gh`, `~/.kube`, `~/.docker`, `~/.git-credentials`, `~/.netrc`,
     `~/.npmrc`, `~/.pypirc`, `~/.config/gcloud`, `~/.azure`, your shell rc and profile files and
     the Windows credential folders. `~/.localharness` is no longer protected as a whole tree:
     **one six-entry list protects each harness config directory a session uses** — `config.yaml`,
     `overrides.yaml`, `trusted_workspaces.yaml`, `grants.yaml`, `declined_workspace_offers.yaml`
     and `plugins/**`, the files that change what the harness does next. Those directories are the
     one the session was started with (`--config-dir`, else `LOCALHARNESS_DIR`, else
     `~/.localharness`), the default `~/.localharness` and any `.localharness/` in the project; a
     config directory the session does not use is not covered. Everything else under one (agents,
     divisions, tools, session state, memory, history, the audit log, the kill file) is the harness
     being *used* and is not protected; naming what is protected rather than what is exempt also
     means a new kind of runtime state added there is allowed by default rather than becoming a
     prompt nobody wanted. Two of those entries, `config.yaml` and `overrides.yaml`, never reach
     this step from the agent's write and edit tools or from a shell write the gate can read: those
     are refused outright, in every mode (see the paragraph on what the agent may change about its
     own setup, below). An entry kept elsewhere through a symlink, as a dotfiles setup does, is
     judged by where it points and is not protected there — except those two, whose refusal also
     matches the entry by its name.
   - **A system directory.** `/etc`, `/usr`, `/bin`, `/sbin`, `/lib` (and `/lib64`), `/boot`,
     `/var` except `/var/tmp`, `/opt`, `/root`, `/srv`, macOS `/System`, `/Library` and
     `/Applications`, Windows `C:\Windows`, `C:\Program Files*` and `C:\ProgramData`.
   - **`pipe_to_shell` — a download piped into a shell.** `curl … | sh` and its PowerShell twin,
     where the code being run has been read by nobody. It is the sink that decides, and only when
     the sink takes its **program** from standard input: `curl x | sh`, `curl x | python3` with no
     script argument, `… | iex`. `curl x | python3 -c '…'` is not pipe-to-shell — the program is
     the `-c` string, and the pipe only feeds it data.
   - **A call the gate could not read at all.** A `command` that is present but not a string
     (`{"command": ["rm", "-rf", "/"]}`), a path argument that is not a string: unreadable is not
     the same as absent, so it asks, ungrantably, with no identity to remember. A command *name*
     computed at runtime is **not** this and does not ask in `auto` — `eval "$(direnv hook bash)"`
     and `"$VAR" …` are ordinary work in a workspace you have trusted.

   **What is deliberately NOT on the list**, and therefore runs without asking in a trusted
   workspace: `docker` in any form (your own deny patterns already refuse `docker stop`, `kill`,
   `rm`/`rmi` and `compose down` outright, which is where that protection belongs); `git branch`,
   `git stash`, `git checkout`, `git restore`, `git filter-branch`, `git reflog expire`, and every
   other git subcommand; `.env` and key files inside the project (a `write(*/.env)` deny pattern
   ships by default and refuses those outright); interpreters (`python3 -c`, `bash -c`, `perl -e`),
   `python_exec` and `cruncher_exec`; subagents; MCP and plugin tools; network reads; and writes
   anywhere else at all — including elsewhere in your home directory.

   Two more things bind in `auto` and are not questions: a refusal you have already recorded denies
   outright (step 3), and so does anything your `deny_patterns` name (step 1). `guarded` adds the
   classes in step 4, its own fuller rule sets, and one class of its own: every write-shaped call
   made when there is no workspace boundary at all.

   **Curate this list, because there is no other surface to curate.** `AUTO_BLACKLIST` is
   deliberately **not** reachable from `permissions.ask.*` — every field of it either loosens the
   mode when extended or is the one list deciding whether the default is safe at all, and config
   travels with a repository. `localharness ask-rate --traces DIR --mode auto` replays your own
   traces and reports which entries actually fired (`--mode guarded` measures the v0.14.0
   behaviour over the same corpus, which is the before/after), and that is the evidence a change
   to the list should rest on.
3. **Grants.** A remembered "always" for this workspace. `auto` neither reads them nor writes them
   — it remembers nothing, because it parks nothing that could be remembered. In `guarded`
   they are checked only after step 2, so an old permissive answer can never cover a destructive
   variant. A recorded "never" is consulted in **every** mode, `auto` included: a refusal you have
   already given still denies, without prompting.
4. **Ask once, then remember — in `guarded`. Silently allowed in `auto`.** A write or shell write
   target outside the project folder (keyed by the target's parent directory), a shell command whose
   signature this workspace has not seen before, an inline interpreter (`python3 -c`, `bash -c`,
   `eval`, `xargs`), `python_exec` and `cruncher_exec`, the `agent` tool, each MCP tool, and any
   tool in no family the gate knows, keyed by the tool's name, because a tool nobody can describe is
   asked about rather than allowed. A tool's family is what its schema declares (`gate_family`); a
   tool that declares none, or whose schema could not be read, is in no family. For a plugin you
   installed yourself, only a declared `code` or `delegate` is honoured, because the gate treats
   both at least as strictly as no family; any other family it declares (`allow`, `network`,
   `shell`, `write`) is treated as no family, because each of those lets through some call that a
   tool in no family would have been asked about. A plugin tool that labels itself as an MCP
   server's (`group: mcp/<server>`) is not judged as that server's tool either.
   What this does NOT cover is making a plugin's tool ask where a tool in no family would not: in
   `auto`, the default, and in `trusted`, a tool in no family runs without asking, and so does a
   plugin's.
5. **Allow.** Everything else: reads, search, memory, `chunk`, the read-only shell commands (`ls`,
   `cat`, `head`, `tail`, `grep`, `rg`, `find` without `-exec`/`-delete`, `git status`/`diff`/`log`,
   `sed -n`, and their kin), network reads, and edits inside the project when the channel can show
   you the diff.

**`docker` is not on the blacklist, and in `guarded` it is judged by its subcommand.** In `auto`
no docker command asks: the four that cost people real state — `docker stop`, `docker kill`,
`docker rm`/`rmi` and `docker compose down` — are refused outright by shipped deny patterns, which
is a stronger answer than a prompt, and an agent killing the model server it runs on is the
incident that put them there. In `guarded`, `exec`, `run`, `start`, `restart`, `system prune`,
`compose up/exec/run/rm` (and the old `docker-compose` spelling), the `docker container …` /
`docker image …` management spellings and the `volume`/`network` removals are ungrantable and ask
every time; `docker ps`, `logs`, `images`, `inspect`, `version` and `info` are reads and never ask;
`build`, `pull`, `push`, `tag` and `login` ask once and can be granted. A bare `docker` entry made
all of those ungrantable, which is ask-fatigue on commands nobody needs protection from.

**The boundary is derived from where you stand, never configured.** It is the folder holding the
nearest in-project `.localharness/`, else the git top level, else the directory you started in,
resolved through symlinks. A `permissions.workspace_root` in your config may only *narrow* it; a
value outside it is ignored with a warning. **If that folder turns out to be your home directory or
anything above it, there is no boundary** — the harness says so rather than pretending your whole
home is one project. In `guarded` every write-shaped call then asks, without a remembered answer.
In `auto` it does not: a session started in `$HOME` used to ask about every single write, which is
the shape of ask-fatigue rather than the shape of safety, so the directory you started in serves as
the target boundary for the destructive-file-operation rule and ordinary writes run silently. The
protected paths and the irreversible operations still ask there, and `~/.ssh` and friends are
exactly the things a boundary-less session is most likely to reach.

**Two things deliberately never ask, and both are a judgment you should check against your own
threat model.** Network reads (`web_fetch`, `web_search`, `web_page_query`) are silent: seven in ten
real tool calls are web fetches, and a prompt per host would fire in a third of all sessions. Set
`permissions.ask.network_hosts: true` for a per-host prompt. What keeps a silent fetch off your own
network is a rule rather than a question: `web_fetch` reaches only public addresses (see
[What a web fetch may reach](#what-a-web-fetch-may-reach)). Data carried out to a public host is
handled structurally instead — the agents that ship with LocalHarness and read the web hold no tool
that reads your files or changes your machine (see the prompt-injection section below). Edits
inside the project are silent when the channel shows you the change: in Zed
they land in the review pane with per-hunk accept/reject, in the terminal the diff is printed after
the fact. A channel with no review surface asks once per workspace in `guarded`; in `auto` it does
not ask at all.

**Answers live in your global config, and a repository can only tighten.** The default mode writes
nothing here — `auto` parks only classes that are never remembered — so this is the file
`guarded` fills, and the refusals in it that still bind every mode. An "always" is written to
`~/.localharness/grants.yaml`, keyed by the workspace's resolved path, with the channel, session and
timestamp that produced it; a "never" is written to the same file as a negative grant, under the same
key, and denies exactly that key — refusing the command `cp` does not touch `scp` — beating any later
"always" on it. Nested folders inherit the parent project's grants and refusals. A `grants.yaml` **inside a project tree is never read** — a
cloned repository must not be able to pre-approve its own `curl … | sh` — and `permissions.mode` and
`permissions.workspace_root` coming from a project layer may only make the policy stricter, never
looser. `permissions.allow_patterns` is gone rather than repurposed; it was a loosening surface.
What a repository can still add is deny patterns, agents with more tools, and MCP servers. Every
call those tools make passes this same gate: in a project you have not trusted, a session in the
default mode runs `guarded`, so a new kind of call asks first, and in a project you trusted it runs
as any call does in `auto`. And an MCP server's process starts only in a project you trusted, and
only with the set you approved (see "What a project's own files may do", under trust boundaries).
Edit `grants.yaml` to
change an answer; there is no CLI verb for it, the prompt is the interface and the file is the
escape hatch.

**What the agent may change about its own setup.** The agent may extend itself — that is a feature,
not a leak: it can write a new specialist into `~/.localharness/agents/` and drop a helper script into
`~/.localharness/tools/`. (A config `init` wrote before this release lists the `write(*/agents/*.yaml)`
deny pattern, which refuses the agent's `write` tool there and stays; delete it from
`org.permissions.deny_patterns` if you want the agent to create specialists with it.) It is told to
compose specialists from the tools the harness already has and never to write its own tool for
something the harness already provides (web search, fetch, files, delegation). What it writes there
is held back until you have seen it: an MCP server, an embedding model or a permission looser than
the shipped default in one of your agent files or in the `agent:` section of your `overrides.yaml`
(every agent's default layer), a looser permission in a division file or `org.yaml` or in the
`org:` section of your `config.yaml` or `overrides.yaml`, or a model-server launch command there
(`server.binary`, `server.docker_image`, `server.extra_args`), takes effect only after you confirm
it once at the next `localharness start` on a terminal (a start without a terminal leaves it off and
says so; a launch command left off withholds the whole `server:` section, so that start launches no
server and uses one already answering); and a script added to or changed in the tools folder
since you last confirmed your scripts — a symlink there included, judged by its own name and what
it points to — is treated as an unconfirmed shell command: it gets what any shell command gets in
the mode you are in: `guarded` asks before it runs (an earlier "always" on `python3` or `node` does
not cover it, and an "always" you give it covers that exact content only), while `auto`, `trusted`
and `unattended` run it as they run any command — and it is listed for you at that start (what was
already there the first time you start this version is adopted without a question).

The files that hold the harness's own settings have a rule of their own: the agent's write and edit
tools, and the shell commands the gate reads as writing them, cannot change `config.yaml` or
`overrides.yaml` in your machine's config folder — the one the session was started with
(`--config-dir`, else `LOCALHARNESS_DIR`, else `~/.localharness`) and the default `~/.localharness`
— or in any project's `.localharness/`; the model is told to ask you to run `localharness components
set`. The shell shapes it reads, with `~`, `$HOME` and `${HOME}` expanded: a redirect, `tee`,
`touch`, `mkdir`, `rm`, `truncate`, `chmod`, `sed -i` or `dd of=` onto the file; a `cp`, `mv`,
`install`, `ln` or `rsync` onto it, for all but `rsync` the destination also read from `-t DIR` or
`--target-directory` (`ln -sft ~/.localharness config.yaml`); `curl -o` or `wget -O` onto it, and
`curl --output-dir` or `wget -P` saving a file of that name into the folder; a copy, move or link
into the folder of a file named `config.yaml` or `overrides.yaml` (`cp /tmp/config.yaml
~/.localharness/` — a file of any other name copied there is an ordinary write, left to the gate);
and, refused because what lands in the folder is not known in advance, an unpack into it (`tar`,
`bsdtar` or `gtar -x -C`, `unzip -d`, `7z x -o`, the folder attached to the flag or not), a copy of
a folder's contents into it (`rsync -a src/`, `cp -r src/.`, `cp -T`) or a download into it whose
name the server chooses (`curl -J`, `wget --content-disposition`) — for those the model is told to
unpack or fetch elsewhere and copy the files it means. That rule is code, not a deny pattern, so no
config can remove it, and it applies whatever the mode — but only to the shapes it reads, so it is
not a boundary in `unattended`, where everything else runs unasked.
What this does NOT cover: `python_exec` and `cruncher_exec` run code that can write any file you
can; code run inline through a non-shell interpreter (`python3 -c …`, `node -e …`) is not read for
the files it touches (a shell payload — `sh -c …`, `bash -c …`, `eval` — is); the rule reads the
shell shapes it knows, so a command that reaches the file by a route it does not read — another
name or a link for it, a path through a variable, `sudo`, a copy or an unpack that lands there from
above — is not caught; the harness's own CLI (`localharness components set …`, run through
`bash_exec`) changes settings by design; outside `guarded` an unconfirmed script runs without
asking (it is still listed at the next start); a script named through a variable or a glob,
relative to an earlier `cd`, or written by the same command that runs it, is not recognised as one,
and one reached through a symlinked folder in the tools folder is judged as the file it points to;
files below a dependency or cache folder in the tools folder (`node_modules`, a virtual
environment, `.git`) are not tracked — that is where `npm install` and a virtual environment put
thousands of files; the scripts and files a confirmed command reads are not hashed; of
`config.yaml` and `overrides.yaml`, the record a start checks holds the looser `org:` permissions,
three `server:` keys (`binary`, `docker_image`, `extra_args`) and `overrides.yaml`'s `agent:`
section, nothing else, so code the agent runs can change any other setting in those files, other
launch settings included, and the change takes effect at the next start without a question; and
the record itself is an ordinary file that code the agent runs can
rewrite (see trust boundaries). What runs is gated — that is the control, not a promise that the
agent cannot touch its own files.

**Five modes, set in config or switched mid-session with `/mode`.** `auto` is the default: one
trust question per workspace, then everything runs except the step-2 blacklist, and nothing is
remembered beyond that one answer. A workspace you declined, and a session with nobody to ask and
no record, run `guarded` instead. `guarded` —
the v0.14.0 default, now opt-in with `permissions.mode: guarded` or `/mode guarded` — asks once
about each step-4 class and remembers your answer, and asks about every write when there is no
boundary. `trusted` is `auto` plus one thing: a destructive file operation aimed **inside** the
project asks too. `read-only` refuses writes, non-read-only shell, and code execution with a
message the model can re-plan against. `unattended` turns every ask into allow, leaving only your
deny patterns — and the settings-file rule above for the shell shapes it reads, which is no
backstop there: code, and every shape it does not read, runs unasked — **this is how the harness
behaved before v0.14**, named honestly. It is
never a default, and from v0.14.1 it is settable the same way the others are: `/mode unattended` in
the terminal, `mode unattended` in Discord, the picker in Zed, or `permissions.mode: unattended` in
the config file of a bench run or a scheduled job. From the phone and Discord it can be switched on
only while `channels.remote_unattended` is `true`, the default (see
[A paired phone or Discord account](#a-paired-phone-or-discord-account)). A session that turns its own gate off is a
decision a person can make out loud; a config file is still the right place for a job nobody is
watching. Ordered from most permissive to strictest —
`unattended` < `auto` < `trusted` < `guarded` < `read-only` — because a project layer may only
raise strictness, never lower it.

**A channel that cannot ask denies.** Bench runs, cron jobs, a piped non-tty session: if a call
reaches step 2 or 4 and there is nobody to answer, it is refused, the model is told "needs human
approval; this channel cannot ask", and the session prints one warning naming the fix. Failing
closed is the rule; the warning is what keeps it from being a silent regression in a scheduled job.

**Named gaps. Read these before you rely on any of it.**

- **Trusting a workspace trusts everything but the blacklist.** One dialog, answered once,
  forever, is what stands between a project and an unasked tool call — and it is a question about
  the folder, not about the call. The blacklist is what limits the blast radius after it, so read
  it as the actual policy.
- **Prior use is taken as consent, for a session in your home folder.** A home-rooted session
  with earlier sessions in your machine's own store is trusted without being asked, and the trust
  is then recorded. That is the behaviour the owner asked for — being re-asked about the place you
  live in is the fatigue this release removes — but state it plainly: sessions that ran under an
  older version, or under `guarded`, are what that evidence is made of. A project's own session
  files are never evidence, so a repository cannot arrive looking familiar. Delete the entry in
  `trusted_workspaces.yaml` to be asked again, and remember that a `.localharness/` you did not
  create is a reason to look before you run.
- **`auto` trusts the project directory.** A destructive command whose target resolves inside the
  project runs without asking: a `chmod -R` over the tree, a `find -delete`, a `truncate`, a
  `Remove-Item -Recurse` — and `rm -rf` too, for anyone who removed the shipped deny pattern that
  refuses it outright. That is the price of a default that stays quiet during ordinary work, and it is a real
  price — a wrong `rm -rf` inside your repo is on the model, and **git is your undo**, so what you
  actually lose is uncommitted work and untracked files. Commit before you hand a session a big
  refactor. `trusted` adds the prompt back for exactly this case; `guarded` adds it back for
  everything else as well.
- **An interpreter runs anything, and in `auto` nothing asked you first.** `python3 -c`, `bash -c`,
  `perl -e` and their kin are a step-4 class, so the default allows them outright — including one
  that deletes a tree outside the project, which the gate would have caught had it been spelled
  `rm -rf`. In `guarded` the same hole opens one answer later: say "always" to `python3 -c` and
  every later `python3 -c` in that workspace runs unasked. `python -c` is the second signature
  nearly every workspace is asked about, so this is the widest hole by design and by frequency in
  either mode. `python_exec` is the tool to prefer.
- **A granted `python3 <script>` covers a script the agent just wrote.** `python3 build.py` is one
  signature, and the file it names sits inside the project, where writing it does not ask when your
  channel shows you the diff. So an agent can write `build.py` and then run it under an answer you
  gave about an earlier `build.py` — the contents of that file were never part of what you approved.
  The same holds for `bash run.sh` and every other interpreter-plus-file signature. Reading the diff
  is what stands between the grant and the code it runs, which is why the review surface matters and
  why a channel that cannot show you one asks about in-project edits instead.
- **`source FILE` and `. FILE` are keyed like a script too** (`source <script>`), so the same
  caveat applies. Three shapes that looked like this gap are caught instead: a `git config` write
  to a key that repoints execution (`core.hooksPath`, `core.sshCommand`, `alias.*`, filters, merge
  drivers, and the rest of a named list) is ungrantable — parked in `auto`, asked every time in
  `guarded` — including the
  `-c key=value` spelling on any git command; `eval`'s argument is classified the way `bash -c`'s
  is; and a shell function's body is classified where it is defined, not hidden behind its name.
- **The shell boundary is best-effort by construction.** A `bash_exec` call is one opaque string.
  The harness strips heredoc bodies, lifts `$(…)`, backticks and process substitutions, splits at
  `&&`, `;`, `|`, newlines and inside groups, peels wrappers, lifts `find -exec` and `xargs`
  payloads, and canonicalizes destructive flags — that is an enumerated set of known shapes, not a
  proof. The ask-once on an unfamiliar signature is the backstop for whatever the list misses.
- **A command whose name comes from a substitution or a variable is unfamiliar, not destructive.**
  A name the shell builds at run time cannot be signed at classification time, so the command is
  asked about as an unknown one rather than as destructive, and a grant on that placeholder
  covers the next command built the same way. The prompt still happens; the label on it understates
  what may run.
- **A write target containing a variable, a glob or a substitution is treated as outside.** It
  cannot be resolved before the shell expands it, so it is treated as an out-of-project write
  (parked in `auto`, asked in `guarded`) — and once
  you answer "always" for that unresolved shape, a later expansion of the same shape to a different
  path passes on that grant.
- **Program text handed to a non-shell interpreter is never read.** `python3 -c`, `perl -e`, an
  `awk` program, a PowerShell `-Command` string: these are keyed as inline interpreters — the class
  that asks once and can then be granted — and what the program says is not parsed, because parsing
  five languages to decide one prompt is not a thing this gate does. Only shell payloads are
  recursed into (`bash -c`, `sh -c`, `eval`, `find -exec`, `xargs`, `ssh host CMD`). So an "always"
  on `perl -e` covers every later `perl -e` in that workspace, whatever it contains. This is the
  same hole as the granted-interpreter gap at the top of this list, restated for the languages
  people forget are interpreters.
- **On Windows the shell is git-bash, and a PowerShell or cmd invocation is keyed by a named verb
  list.** `bash_exec` runs commands through Git for Windows' `bash.exe`; a `powershell`, `pwsh`,
  `cmd` or `wsl` command reached from there is classified as an inline interpreter, and the
  destructive ones are recognized from an enumerated list of verbs. An enumerated list is not a
  parser: a destructive spelling that is not on it — an alias, an encoded command, a cmdlet nobody
  listed — classifies as an ordinary inline interpreter and can be granted. The same "best-effort
  by construction" caveat as the POSIX shell boundary applies, with a shorter list behind it.
- **There is still no OS sandbox.** Everything above is a policy boundary, enforced by this process
  in this process. It narrows an enumerated set of mistakes and crossings; it does not contain a
  program that has already started running.

**Being asked too often is itself a security failure.** A gate that interrupts you about ordinary
work trains you to approve without reading, and an approval nobody read protects nothing — the
prompt is only worth what your attention to it is worth. That is why the silent paths above are
deliberately wide, and why the harness measures how often it asks rather than assuming the answer.
`localharness ask-rate --traces DIR` reads your own session traces and reports prompts per session;
the target is a median of zero and at least nine sessions in ten with no prompt at all once a
workspace is warm, with three prompts the ceiling for the first session in a fresh one. It also
counts "never here" answers, because a rising count of those means the gate is asking about the
wrong things and is spending attention it will need later. If your own numbers sit far above these,
treat that as a defect in the gate and report it, not as something to click through.

## What `localharness start` writes without asking

Five things happen on a start that are worth knowing about, because each writes into your config
directory and none stops to ask.

**Your `config.yaml` gains any newly-shipped default deny patterns.** New releases add to the
default deny list, but `localharness init` baked the list into your `config.yaml` when you first set
up, so a later addition would never reach you. On the first start after an upgrade the harness folds
the missing ones in. It never reorders an entry you wrote and is gated on the `defaults_revision`
stamp your config carries, so a default you deliberately deleted is not re-added. One key it does
delete: `permissions.allow_patterns`, which v0.14 removed and which every earlier `localharness
init` wrote. A config still carrying it cannot be loaded at all, so the fold-in strips it before the
loader ever reads the file — and `localharness config migrate` does the same, on purpose, because
the documented repair has to be able to repair the thing that breaks. If yours held entries, they
are listed as they are removed; they were never honored by any release. A
timestamped `config.yaml.bak-<stamp>` is written before the change, and the change is announced:
`i  Security defaults updated (revision 0 → 1): added 24 deny pattern(s) — backup at
…`. State it plainly: **this happens without asking you.** The reason it is not a prompt is that the
change only ever tightens the deny list and drops a key no release honors, and a start that blocks
on a question is a start that fails in a script. The reason it is not invisible is `localharness doctor`, which prints the revision your
config carries, the revision shipped, and the backup path — so you can see the change after the
announcement has scrolled away. `localharness config migrate --dry-run` prints exactly what a start
would add and remove, and writes nothing. Whether this should stay automatic is an open question for the
project owner; the behavior above is what ships today, not a settled ruling.

**A config that launches a vLLM server keeps that server reachable from other machines.** A
model server LocalHarness launches now listens on 127.0.0.1 only (see
[A model server LocalHarness launches](#a-model-server-localharness-launches)). So that nothing
that uses yours from another machine — a laptop, a second box on your tailnet — stops working at
the upgrade, a start that finds a config written before `server.bind_all` existed, launching vLLM,
writes `server.bind_all: true` into it (and into each `extra_endpoints` entry that launches one),
with a timestamped backup and one line: `Config updated; kept the model server reachable from other
machines (server.bind_all: true)`. `localharness doctor` then shows a row saying the server is
reachable from other machines and how to keep it on this one, and `localharness config migrate
--dry-run` lists the write. A `server:` section you write by hand without the setting is treated
the same way — set `bind_all: false` in it if you want loopback. A new install's config says
`bind_all: false`. This is not tied to the defaults revision, so it never re-adds a deny pattern
you deleted. **What this does NOT cover:** only `config.yaml` is read, so a launched server's
section that lives only in your global `overrides.yaml` is not migrated, and that server listens on
127.0.0.1 after the upgrade until you set `server.bind_all: true`.

**The config folder is made owner-only.** An install from before this release made
`~/.localharness` at your umask, often readable by every account on the machine, and the folder
holds your keys, the phone token, your grants and every session log. Each start makes the machine's
config folder — the default one, or the one `--config-dir` or `LOCALHARNESS_DIR` names — owner-only
(0700) if it is not, without a word on the terminal, and records the change in the audit log
(`config_dir.mode`). A folder it cannot change — another account's, one on a read-only mount — is
left as it is, the start goes on, and `localharness doctor` says why. Read the second part plainly:
`--config-dir` or `LOCALHARNESS_DIR` pointed at a folder you share on purpose gets the same
`chmod 700`.

**The trust store records what you had already approved.** At the first start after the upgrade,
`trusted_workspaces.yaml` records without asking the MCP servers of a project you had already
trusted (its Yes predates the list) and the state of your own agent files, division files,
`org.yaml`, the `agent:` section of `overrides.yaml` and tool scripts (you wrote them). Later changes are compared against these records,
and those are what a start asks about. A start also records, without asking, a server or script
that went away, a tightening, and a kind of entry the record did not know yet. A start with
`--config-dir` writes that folder's record into the default trust store, under a key of its own.

**`start` also seeds `<config-dir>/tools/design-screenshot.js`.** The frontend-designer builtin
shells out to that script by path, so the package copies it into your config directory's `tools/`
folder on first run, owner-only and executable (0700). It is idempotent — present means untouched
— it is written to your global config directory and never into a workspace, and a failure to copy
it is a warning rather than a blocked start. It is the only file `start` installs from the
package, and it is never listed as one of your tool scripts.

## What `localharness start` contacts

A local-first harness should reach nothing at start but what you configured. A start reaches the
model server in your config (starting it first, when your config launches one) and, when you set
them up, the MCP servers your agent files name and, as the `discord` channel, Discord — nothing
else. Two downloads can follow later, each once and only when it is needed:

- `huggingface.co`, when memory's embedding model is not on this machine yet. The start summary
  says so in one line, the banner does not wait for it, and the model downloads the first time
  memory needs it (a memory search that comes first waits for that download). A model already in
  the local Hugging Face cache loads from it with no network call, and is never updated on its own
  (`hf download <model>` fetches a newer copy); a cached copy missing a file, as an interrupted
  download leaves it, is completed the first time memory needs it, and the same line is written to
  `memory.log` first.
- the tiktoken vocabulary host, `openaipublic.blob.core.windows.net`, on the first approximate
  token count with no cached vocabulary — which a vLLM or llama.cpp setup never makes, because
  those count tokens on the server. Offline, the estimate falls back to a byte count and says so
  once.

Beyond those, the network is reached by what the agent's tools do in the session (a web search, a
fetched page, a shell command), which the approval gate and the web-fetch rule govern. A start with
no MCP server configured no longer imports the MCP client library at all.

**Memory's embedding model loads only from a checked folder, by its full path; the current folder is
never a model source.** Given a model name, sentence-transformers (5.6) looks for a folder of that
name in the current folder before the cache, where a cloned project could ship one, and for any
folder it imports the module code the model names without asking for `trust_remote_code`. So the
harness never hands it a name: a model id is its snapshot in your local Hugging Face cache
(downloaded there on a real miss), and a relative path in `memory.embedding_model` is a folder only
when your config folder holds it (one line in `memory.log` says so), otherwise a model id. Before the
load it reads the model's files: a model that names code of its own (a module that is not
sentence-transformers' own, in `modules.json` or a Router's config; a Dense activation outside torch;
a WordEmbeddings tokenizer outside the library; `auto_map` or `trust_remote_code` in a config) is
refused with one line; memory search, `remember` and the background embedding report that line for
the session, and nothing else stops.

## Files the harness writes

`init` creates the config folder (`~/.localharness`, or the folder `--config-dir` or
`LOCALHARNESS_DIR` names) owner-only (0700), and a start makes an older one owner-only (see above).
The files the harness creates there are owner-only (0600) from their first byte — created with
that mode, never written first and changed after: `config.yaml` and its backups
(`config.yaml.bak-<stamp>` from the security-defaults update, `config.yaml.before-init-<stamp>`
from `init --force`), `overrides.yaml`, `grants.yaml`, `trusted_workspaces.yaml`, the phone token,
the push key and the push subscriptions, `audit.jsonl`, each agent's `bus-events.jsonl`, session
logs, `history.jsonl`, `memory.db` (and its `-wal` and `-shm` files), `MEMORY.md`, `compact.md` and
`memory.log`, `.repl_history`, the speed ledger, the live-session files, `vllm/serve.log` and
`vllm/server.pid`, agent files written by `agent create`, by a start (the root agent) or by the
in-session creation workflow, autoresearch's archive, run journals and budget file, and
`plugins/README.md`. The same writers make the same files owner-only in a project's
`.localharness/` (its sessions, history and memory). The one script `start` installs,
`tools/design-screenshot.js`, is owner-only and executable (0700). `init --force` saves the old
`config.yaml` beside the new one before it writes, and says where.

**What this does NOT cover.** Pictures the image plugin generates are written at your umask. In a
session with no project they sit in the owner-only config folder, which keeps other accounts out,
but in a project they sit in its `.localharness/`, and a copy taken anywhere keeps that mode. The
files `init --workspace` writes inside a project (`.localharness/config.yaml`, its
`plugins/README.md` and `.gitignore`) are meant for the repository and are not made private, and
neither is a project's `.localharness/` folder itself. Files written outside the config folder — a
report or a picture saved where you choose, the research reports written in the folder you work
in — keep your umask. Folders inside the config folder are made at your umask too; the 0700 folder
above them is what keeps other accounts out. Files an older release created keep their modes: the
folder's 0700 closes them, `localharness doctor` counts the ones other accounts could read, and
`chmod -R go-rwx` on the folder makes each private. A temporary file a crash left behind keeps
its old mode.

## What ends up in logs

No setting LocalHarness treats as a secret is written to `.repl_history`, `bus-events.jsonl`,
`audit.jsonl`, the session logs or the model's prompt: a secret changed with `components set` is
recorded as changed, with `**********` for its old and new value. The phone token is printed to
nothing but a terminal (see [`localharness mobile`](#localharness-mobile)). `vllm/serve.log` holds the
launch command, which never contains a key, because a required key travels in the server's
environment. **What this does NOT cover:** what reaches the conversation as ordinary text — a key
you paste into a message, or a file the agent reads that holds one, your own `config.yaml`
included — is written to the history and the session log like any other text. To keep a key out
of your shell history, set it with `localharness components set <secret setting> -`.

## Threat model: prompt injection

Agents fetch web pages and call tools, then act on what they read. The central risk
is **prompt injection**: attacker-controlled text in a fetched page or tool result
trying to make an agent take a host action it should not. This is not hypothetical —
the companion morning-report job runs agents with `bash` and web tools, on a
schedule, over live pages no human vetted first.

**Primary defense: separation, enforced structurally.** An agent that ingests
untrusted content is never the same agent that can mutate the host. Host-mutating
tools (`bash`, file `write`/`edit`) are kept out of any agent that fetches or ingests
untrusted text. This is enforced where an agent's tools are resolved: a host-mutating
toolset combined with untrusted ingestion is rejected, and the check **fails closed**
(deny on doubt). Untrusted content moves between agents only as opaque handles
carrying a sticky "untrusted" tag; its raw bytes resolve only inside an agent that
holds no host-mutating tools.

The separation is on by default. Only your machine-level config can turn it off
(`org.enforce_capability_floor: false`); a project's value for that setting is ignored, with a
startup warning naming the file (see [Machine-level-only settings](#machine-level-only-settings)).

Which tools count as ingesting and which as host-mutating is read from each tool's own declaration,
never from its name or from the plugin it came from. Every tool declares four things: what it
ingests (`ingest`), whether it can change the host (`host`), whether its results are trusted
(`result_origin`), and which family the approval gate files it under (`gate_family`). A tool that
declares nothing gets the most restrictive value of each: `ingest: untrusted`,
`host: dangerous`, `result_origin: untrusted` and `gate_family: none` (asked about). Three places
read these declarations, all under `src/localharness/`: the separation check in
`tools/capabilities.py` reads `ingest` and `host`; the approval gate in `agent/gate.py` reads
`gate_family`; and the context store in `agent/context.py` reads `result_origin` when it marks a
stored tool result untrusted. So a plugin tool that declares nothing counts as both ingesting and
host-mutating, and no agent may hold it: the root agent is not given it (`localharness start`
prints a warning naming the tool and its plugin), and any other agent configured with it is
refused. The same declarations are read by the rule that a handle to untrusted content may be
granted only to an agent with no host-mutating tools, and by the rule that an agent without the web
tools may not fetch through an exec tool such as `bash_exec`. Built-in tools, MCP tools (their
wrapper declares them untrusted) and plugin tools are judged the same way, whatever scope they
arrive in.

**A residual that is now closed.** Tool classification used to read tool names. A plugin tool that
reached an agent through the inherited global scope therefore needed a separate per-tool tag before
the checks knew what it was. Now every tool carries its own declarations, the three readers above
read the same ones, and a tool that declares nothing gets the most restrictive values.

**What this does NOT cover.** A declaration is believed. A tool that declares `ingest: none` while
it actually fetches attacker-controlled text is treated as it says, and turning on a plugin you
installed yourself is you vouching for what its tools declare (see [Plugins](#plugins)). The warning
at start is the only place a stripped plugin tool is named; `localharness doctor` does not show it.
And a plugin tool that ingests is kept apart from the host tools, but it is not marked as untrusted
where you read its output: the terminal's "web results — UNTRUSTED, treated as data only" note and
the phone's untrusted label cover the three built-in web tools only.

Memory is split the same way, and one part of it is trusted. What `memory_search` and `memory_get`
return is marked untrusted, like web content: a recalled fact is fenced as data. What the
`remember` tool returns is marked trusted, because it is the harness's own confirmation that a
fact was saved, not the fact read back. The bench turns memory on only for the three scenarios
that seed it, with the same setting on both sides of a comparison and background consolidation
off, so a bench result says nothing about how memory behaves in any other scenario.

**The terminal channel cleans model and tool output before it prints it.** It removes
escape and control sequences — ESC, OSC, CSI and the C1 range — from model text, reasoning, tool
calls and their output, errors and plugin output, so text the model copied from a page cannot set
your clipboard (OSC 52), clear the screen or redraw a permission question above the prompt.
`localharness memory list|show` and `localharness agent list` print stored text as it is, so a
fact or an agent file the model wrote can still carry a sequence there; that gap is open.

**Not yet built: sandboxing.** Host-mutating tools currently run with the machine's
full trust; there is no OS-level sandbox (e.g. bubblewrap) around them yet. That is on
the roadmap. The human approval gate above is not a substitute: it decides whether a call
starts, in this process, against an enumerated set of shapes — it cannot constrain a program
once it is running. Until a sandbox ships, the separation above is the containment — so **run the
harness as a non-privileged user**, and isolate it in a container or VM if it will
process untrusted content on a machine you care about.

**Known residual (named, not closed).** The separation blocks *verbatim* untrusted
bytes from reaching a host-mutating agent. It does not fully block *laundered*
influence: an agent with no host tools can read untrusted content and hand a summary
to an agent that has them. Summarizing degrades an attacker's control but does not
eliminate it. Closing this fully is a larger change, deferred until a live test shows
it is exploitable on the target model.

**Not a current vector: memory.** Tool output is written to per-agent history, not to
the queryable facts memory, and no code path promotes tool output into the facts an
agent recalls. If that changes, this section changes with it.

### What a web fetch may reach

The URL `web_fetch` opens comes from the model, and the model may be reading a page written to
steer it. So every address the URL's host resolves to — at the first request and at every
redirect, which `web_fetch` follows itself, at most 5 — must be a public internet address.
Loopback, private networks, link-local and cloud-metadata addresses (`169.254.169.254`), CGNAT and
tailnet addresses (`100.64.0.0/10`), multicast, and the IPv6 forms of all of these are refused,
however the address is spelled: `2130706433`, `0x7f000001`, `127.1`, `localhost` and
`[::ffff:7f00:1]` are judged by what they resolve to. Only http and https are fetched, and a URL
with a user name or password in it is refused. A refusal is one line to the model, naming the
setting below; nobody is asked anything.

Without a proxy, the request goes only to an address that was checked: the URL carries the checked
IP, and the host name rides in the `Host` header and the TLS server name, so a DNS answer that
changes between the check and the connection (DNS rebinding) cannot point it elsewhere. When a
checked address refuses the connection, the next checked one is tried — never a fresh lookup.
Behind an environment proxy (`HTTP_PROXY`, `HTTPS_PROXY` or `ALL_PROXY`, for a host `NO_PROXY`
does not exempt), the name is checked here first and then the original URL goes to the proxy, so
HTTPS through a corporate proxy keeps working. To fetch a private service on purpose, add its
address, network or host name to `org.web_fetch_allow_private` in your machine-level config (a
project cannot set it); multicast, unspecified and reserved addresses are never admitted.

`web_page_query` searches the text of a page `web_fetch` already fetched, with a pattern the model
chooses. A pattern longer than 128 characters is refused, and a pattern with regex syntax runs in a
separate process that is stopped after 1 second, so a pattern written to make the regex engine
spin costs one tool error, not a frozen session.

**What this does NOT cover.** A GET to a public host can still carry data out in its URL —
fetching public pages is the tool's purpose. The defence there is structural and only as wide as
the agents: the agents that ship with LocalHarness never hold web ingestion together with tools
that read your files or change your machine, and the capability floor keeps any agent from holding
web ingestion with host-mutating tools, but an agent you define with both `web_fetch` and `read` can
put what it read into a URL. `bash_exec` and `python_exec` reach any address — the guard is on
`web_fetch` alone — and a fetcher script the agent writes for itself is gated as an unconfirmed
shell command until you confirm it (see the approval gate, on what the agent may change about its
own setup). The
guard resolves names on this machine, so a setup where only a proxy can resolve names cannot fetch
a named host (the model is told the name could not be resolved). Behind an environment proxy, the
proxy resolves the name again: a DNS answer that changes between the two lookups, or a name the
proxy resolves differently (split-horizon DNS), reaches whatever the proxy can reach. And a pattern
with no regex syntax is searched inside the session itself, which on a hostile page holds it for
up to about a second and a half (about, on a 4 MB page).

## Securing the endpoint

Inference servers ship with no authentication. On a network with untrusted devices,
start the server with an API key and set `provider.api_key` to match; for access
beyond your LAN use a private overlay network (Tailscale/WireGuard). Never port-forward
a bare, unauthenticated endpoint to the internet. See "Running the harness on a
different machine than the model" in the README.

### A model server LocalHarness launches

When your config has a `server:` section, `localharness start` (and `/model`) can launch vLLM for
you. That server listens on 127.0.0.1 only — a docker launch publishes `-p 127.0.0.1:<port>:8000`,
a binary launch gets `--host 127.0.0.1` — unless `server.bind_all: true`, which keeps the old
behaviour of listening on every interface, for a LAN or tailnet you serve on purpose. A config from
an earlier release that launches one has `bind_all: true` written for it at the first start (see
"What `localharness start` writes without asking"), so nothing that used it from another machine
breaks; a new install's config says `bind_all: false`. Launches of llama.cpp, Ollama and LM Studio
were on 127.0.0.1 already. Both settings are machine-level only.

With `server.require_api_key: true`, the server is launched requiring your `provider.api_key`. The
key goes in the server's environment (`VLLM_API_KEY`; for docker, `-e VLLM_API_KEY`), never on its
command line, so it is not in `ps` or in `vllm/serve.log`, and every request the harness makes to
that server's `/v1/` routes — the readiness wait, model lists, the context-window probe, a `/model`
switch, `localharness model` and `doctor` — sends it. **It is off by default.** A launched server
without it answers any program
that can reach its port: on this machine any process, including one the agent starts, and with
`bind_all` any device on your network. `localharness doctor` shows a row when `provider.api_key` is
set and the launched server does not require it, and a row while a launched server is reachable
from other machines. An `extra_endpoints` entry whose `lifecycle` launches a server follows the same
rules with its own `lifecycle.bind_all`, `lifecycle.require_api_key` and `api_key`.

**What this does NOT cover.** As vLLM's source reads, its key guards the `/v1/` routes only, so
its `/tokenize` answers without a key, and the harness's token counter and doctor's tokenizer
probes call it that way. `localharness init` sends no key when it looks for a server, so a re-run of `init` does not find
a server that requires its key. Doctor's network row reads `bind_all` only: a `--host` in
`server.extra_args` that opens the server is not named. And these launch rules were checked against
a stand-in that behaves the way vLLM's source says vLLM does, not against vLLM itself: that a binary
launch listens on every interface without `--host`, that it reads `VLLM_API_KEY`, and which routes
its key guards are vLLM's behaviour as read from its source, not measured here.

### `localharness mobile`

The mobile channel adds a second listener, and it is a more serious one than the model
server: **it can run arbitrary shell with your privileges**, because that is what the
agent behind it does. Three things hold it, and each is there for a reason:

- **It binds loopback only.** Startup refuses any other address unless
  `--allow-unsafe-bind` is passed explicitly. A fronting proxy publishes it — this is
  also what keeps a proxy's identity headers meaningful, since they are only
  trustworthy if nothing but the proxy can reach the backend. `mobile.public_url`, which
  `localharness plugins enable mobile` asks for, is only the address the pairing QR sends
  your phone to: it never changes what the server binds, and `--allow-unsafe-bind` stays
  a flag you pass on each run, never a saved setting.
- **An app token is required on every request**, the event stream included. It is
  generated on first run and stored `0600` under the global config directory. Its text is
  printed only to a terminal, and only when it is created, right after `--rotate-token`, or
  when you ask with `localharness mobile --show-token` (which prints it with the pairing QR
  and exits, so a server that is already running can be paired from another terminal).
  On a terminal the pairing QR is drawn on every start: it carries the token, and
  scanning it is pairing. When standard output is a pipe or a file — a `tee`, journald —
  nothing that carries the token goes there: if the process is the foreground job of a
  terminal, the pairing block is drawn on that terminal instead, and otherwise one line
  names `--show-token`. Binding loopback makes the boundary tighter in one sense and
  looser in another: every local process on the machine can now reach that port,
  including one an agent itself started. `localharness mobile --rotate-token` makes every
  enrolled client invalid and also clears the push subscriptions, so each phone turns
  notifications on again after it pairs; a `localharness mobile` that is already running
  keeps accepting the old token until it restarts.
  **Named gap:** there is no per-device revoke — rotation is all or nothing.
- **The POST surface is CSRF-safe by construction.** Every write verb requires both a
  bearer header and `application/json`, and the three simple content types a
  cross-origin form can send without a preflight are refused outright. The event
  stream authenticates with a cookie instead, because `EventSource` cannot carry a
  header and a token in a URL lands in logs and referrers. That cookie is not the
  token: it is a value derived from it (an HMAC), `Secure`, `HttpOnly`,
  `SameSite=Strict` and sent only to `/api`. Only GET routes accept it, and no route
  hands the token back for it, so whoever holds the cookie can read what the GET routes
  serve but cannot drive the session. The cookie an older release set, which was the
  token itself, is cleared at the next enrolment and accepted nowhere; a phone paired
  before the upgrade picks up the new one by itself. A credential that is not plain
  text — a non-ASCII bearer or cookie — gets a 401, never a server error.

Two further properties worth knowing about, both deliberate:

- **A permanent grant cannot be written in one request.** `allow_always` and
  `reject_always` take a second POST carrying a server-minted, short-lived,
  single-use token. The grant store is global, keyed by workspace real path, never
  expires, and has no revoke command, so one mis-tap — or one buggy script holding the
  token — must not be able to produce one.
- **`--ui-dir` and `--replay` are real-path confined.** Both take a user path; neither
  will serve a file outside the resolved root, follow a symlink out of it, or open
  anything that is not a session log. Same discipline the trust and grant stores
  already use.

The static page is served without a credential and is inert without one: every `/api`
route refuses an unauthenticated caller, so reaching the port is still not a shell.
The alternative — putting the token in the URL of the page that bootstraps enrolment —
is the thing the cookie design exists to avoid.

**What this does NOT cover.** `--allow-unsafe-bind` with plain HTTP sends the bearer token
in cleartext to every hop between the phone and this machine. The token file is the one
durable copy of the token, and owner-only stops other accounts, not the agent: while
`localharness mobile` runs, a process running as you — one the agent starts included — holds what a
paired phone holds (see [A paired phone or Discord account](#a-paired-phone-or-discord-account)),
and with `channels.remote_unattended` left at its default that includes the switch to
`unattended` (setting it to `false` closes that path too).
The QR drawn on a terminal carries the token into that terminal's scrollback (a tmux
history, a `script` log). A token an older release printed — into a `localharness mobile | tee`
log, or journald — is still in those logs; `--rotate-token` makes those copies useless.
Cookies are scoped by host name, not by port, so another service published under the same
host name receives the stream cookie and could read what the GET routes serve, though it
could not drive the session.

Three more, added with the installable app and notifications:

- **The enrolment QR carries the token in the URL *fragment*** (`https://host/#t=…`),
  never the query string. A fragment is not sent to any server, so it reaches no access
  log, no proxy log and no `Referer` header — which is what the rule against tokens in
  URLs is actually protecting. The page reads it once and erases it from the address
  bar, so it does not survive into history or a screenshot either. The alternative is
  hand-typing a 256-bit secret on a phone keyboard, which is the setup step people skip.
  The address printed beside the QR carries no token, and a token from a freshly scanned
  QR wins over the one the page stored, so pairing again after a rotation just works.
- **Registering for push takes the app token.** A push subscription is a standing channel
  into your lock screen, carrying deep links to the calls the harness wants approved, so
  `POST /api/push/subscribe` is gated exactly like every write verb. The VAPID private
  key is generated by the harness, stored `0600` beside the token, and never leaves the
  box; payloads are encrypted to each device's own key, so the push service that relays
  them cannot read them.
- **The web app manifest carries the token only for a caller that presents the bearer
  token.** An installed iOS web app gets a different storage jar from Safari's, so the
  manifest's `start_url` is what could pair it — and that variant is served `no-store`,
  only to a request carrying the bearer header, never to one with only the stream cookie
  (or the cookie would trade for the token). A browser fetches the manifest without the
  bearer, so a new home-screen install asks for the token once. Any other fetch gets a
  manifest with no token in it, which is the version anything reaching the port can see.

One more, added with plugins:

- **`GET /api/artifacts/{plugin}/{id}` serves a file a plugin made, and nothing else.** It takes
  the same credential as every other `/api` route. It answers only for a plugin that is on in this
  session and asked for artifacts, and only from the folder the harness computed for that plugin,
  `<state dir>/artifacts/<plugin>/`; a plugin that reports any other folder gets no artifact
  serving for the session, and a request naming any other plugin is a 404 before the filesystem is
  touched. The id must have the one shape the harness mints (`art-`, a timestamp and six hex
  digits, ASCII only), exactly one entry may bear it, that entry must be a regular file (a symlink
  is refused even when it points inside the folder, the same rule the gallery listing applies),
  and its real path must stay inside the folder. The media type comes from the file actually
  served, and only `image/png`, `image/jpeg` and `image/webp` are served; anything else is refused
  with 415. Responses are cached as immutable, because an id never names a different file.
  **What this does NOT cover:** the folder itself is resolved before that check, so if
  `<state dir>/artifacts/<plugin>` is replaced with a symlink, files are served from wherever it
  points (still only harness-shaped names with an allowed type, and whoever can plant that link
  can already write your state directory); and a folder the harness cannot read answers 500, not
  404.
  The image plugin (off by default) is the first bundled plugin to use this route: it writes each
  picture it generates as a PNG under `<state dir>/artifacts/image/`, named only by a core-minted
  id, and the phone page shows a picture only from the typed `artifact` field of a tool-result
  event, never by reading paths out of tool output. The ComfyUI address and workflow template are
  machine-level settings a project folder cannot set; a project can at most turn image on against
  the server the machine already points at.

Added with the mobile plugin:

- **`GET /api/artifacts` lists pictures, and says little about them.** It takes the same
  credential as every other `/api` route. It lists only the folders bound in this session (in a
  project, that project's own), and in them only files directly under the folder whose name is a
  core-minted id with an allowed suffix: no subfolders, no symlinks. Each entry is plugin, id, media
  type and size, never a path or a prompt. The one new fact a viewer learns is WHEN pictures were
  made, because the id embeds a UTC timestamp. With no folder bound, or in incognito, it
  answers 404.
- **Incognito (`--incognito`) keeps pictures out of the phone's cache, and nothing more today.**
  Pictures are then served `Cache-Control: no-store` and the gallery is off. Pictures the phone
  already cached stay until the browser evicts them or its site data is cleared. Nothing on the
  machine is
  removed: memory, sessions and files on this box still persist. It is not yet a private mode.
  The drawer's Incognito switch (`POST /api/incognito`) flips the same setting for this server
  process only; the flag sets the starting value, and memory, sessions and files on this box
  still persist either way. Turning it on sends one authenticated response carrying
  `Clear-Site-Data: "cache"`, which Safari/iOS 17+ and Chrome honor by dropping this origin's
  cached pictures; a browser that ignores the header keeps what it had.
- **The phone reaches memory only through the memory slot's browse API.** The four
  `/api/memory` routes call the slot's occupant; with memory off the slot is empty and they
  answer 404. A test pins that the mobile channel's server, channel, push, replay and protocol
  modules import nothing from `memory/` or the image plugin.
- **Memory is the bundled `memory` plugin.** Its tools (`memory_search`, `memory_get`, `remember`)
  carry the same safety declarations they always had, and as a bundled plugin it is exempt from
  the third-party clamp. With memory off — `memory.enabled: false`, or the deprecated
  `org.memory_enabled: false` at any layer — no memory tool is registered and the guardrails still
  reach the prompt.
- **`start --channel` accepts core channels and bundled channel plugins only**; an installed
  third-party plugin cannot add a channel in v0.16 — a channel sees every event, tool results
  included, and can inject user messages.
- **Discord is the bundled `dispatch` plugin, and who may drive it is a machine-level decision.**
  `dispatch.discord.token`, `dispatch.discord.allow` (the user ids that may send turns and answer
  permission prompts) and `dispatch.discord.channels` are read from the global config layers only;
  a project's value is dropped with a warning, so a cloned repository cannot point the bot at a
  different token or widen who may drive the agent. An empty allow-list refuses to start rather
  than listen to everyone. The token is stored as plain text in the global `overrides.yaml`
  (written mode 600 by `plugins enable` and `components set`; a hand-written `config.yaml` keeps
  whatever mode you give it). It is never printed: `plugins enable`, `components set`/`list`/`get`,
  `plugins info`, `doctor`, the start banner, the setup prompt, error messages and the
  `components set` audit event show `**********`. The token comes only from
  `dispatch.discord.token` or, until 0.17.0, the deprecated `LOCALHARNESS_DISCORD_TOKEN` and
  `DISCORD_BOT_TOKEN` variables (the other `LOCALHARNESS_DISCORD_*` variables still fill their
  unset fields until then too), each with a deprecation warning. Another program's token file,
  `~/.claude/channels/discord/.env`, is no longer read; with no token anywhere the start refuses
  with one line that says so and names `localharness plugins enable dispatch`. ♾️ — "always" —
  takes a second tap, as on the phone: the bot posts a confirm message, and only ✅ on that message
  records the grant (✅ on the question counts once). The bot ignores its own reactions, so it can
  never answer its own question; what it sends pings nobody but the person it replies to (no
  `@everyone`, role or user mention from model text); and a masked link (`[text](url)`) is sent as
  its text followed by the plain address, so the address you would open is in view. Files that
  users upload are kept as metadata only and are not passed
  to the model; outbound files are read only from core's artifact folder. Limits: one settings
  section, `dispatch.discord.*`, feeds every dispatch adapter today (a second platform would share
  Discord's token and allow-list until it gets its own section), and this plugin build has not yet
  been verified against a live Discord server.

**Two live sessions on one agent are warned about, not prevented.** `history.jsonl` and
`compact.md` take unlocked appends, so a terminal session and a web session on the same
agent can interleave their writes. Starting the second one names the first; neither is
refused. Closing one is still your call to make.

### A paired phone or Discord account

**A paired phone or an allowlisted Discord account drives the agent as you would at the
terminal.** It sends turns, answers the agent's permission questions, can switch the session to
`unattended` and can answer "always", so it is as powerful as a shell on this machine: guard the
phone's pairing and the Discord allow list the way you would guard a shell login. To keep the last
two to this machine, set `channels.remote_unattended: false`
in your machine-level config (a project cannot set it). `unattended` is then refused from the phone
and Discord with one line naming the setting, and their permission questions are put without
"always" — an "always" that comes back anyway counts once. The terminal and Zed are never limited
by it. It is `true` by default, so nothing you do from the phone or Discord today stops working,
and `localharness doctor` shows a row while the phone app or the Discord plugin is on and the lock
is not set.

**What this does NOT cover.** The lock refuses `unattended` and "always" only. The phone and
Discord can still switch the session to `auto` or `trusted`, answer every question once, and answer
the workspace trust question, whose Yes puts a new workspace in `auto` — so a paired phone or an
allowlisted account can still run anything `auto` allows.
