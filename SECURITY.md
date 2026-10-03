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

**What this does NOT cover.** If you clone someone's repository and run the harness inside it, that
repository's `.localharness/agents/` loads with no prompt, because you are inside that project.
Agent files decide an agent's role, model and tool permissions, so read them in an unfamiliar
repository before you run the harness there, the same way you would read its build scripts. Plugin
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
turn it on, and the phone app `web`, memory, Discord (`dispatch`) and autoresearch, each on by
default; everything else that comes with LocalHarness is built into its core, including the bench
and its sealed holdout.) A plugin that ships with LocalHarness may own core settings under their old
names (`autoresearch` owns `proposer:` and `sentinel:`); a plugin you install that tries to is
refused at load. `proposer.api_key` is shown as `**********` wherever a command displays it; like
the Discord token, the file `components set` writes holds the real value as plain text. In 0.16.0 a
config that failed to load or validate could still print a stored secret, whole or by its last
characters, in the error text of commands such as `doctor`, `start` and `components set`; this is
fixed after 0.16.0, and error text now names only where the problem is and what is wrong, with
every setting LocalHarness treats as a secret masked: `proposer.api_key` and the Discord token
(`dispatch.discord.token`). Other credentials are plain-text settings, not treated as secrets yet,
and nothing masks them: `provider.api_key`, `active_endpoint.api_key`, the `api_key` and
`extra_headers` of each `extra_endpoints` entry, and an MCP server's `env` and `headers`.
`components get`, `components list` and `config show` display the provider and endpoint ones,
`components set` prints them and writes them to its audit log as they are, and the error text of a
config that fails validation can show any of them.

- **Found is not on.** A plugin is found from package metadata and folder names alone, and one you
  installed stays off, with none of its code imported, until you turn it on. `localharness start`,
  `localharness doctor` and `localharness plugins list` say it is available and print the command
  that turns it on.
- **Turning on a plugin you installed is a machine-level act, and it is your trust grant.**
  `localharness plugins enable <name>` writes your machine-level `overrides.yaml`. A project can
  never turn such a plugin on: its value for `<name>.enabled` is ignored with a warning, and
  `plugins enable --workspace` refuses. A plugin that ships with LocalHarness can be switched on or
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

Some settings say where the harness connects, which credential it uses or who may talk to it, and
one says whether the separation described under
[Threat model: prompt injection](#threat-model-prompt-injection) is enforced at all. A repository you
cloned must not be able to point those somewhere else or switch that separation off, so only your
machine-level (global) config may set them. The autoresearch proposer's address and key are two of
them, because `plugins enable autoresearch` sends the key to that address to check that the
proposer answers; with no proposer address in your global config, a project's whole `proposer:`
section is ignored. A project's value for one of them is ignored, the harness prints a warning
naming the key and the file, and your global value stands. These are all of them:

- `image.comfyui_url`
- `image.workflow`
- `dispatch.discord.token`
- `dispatch.discord.allow`
- `dispatch.discord.channels`
- `web.public_url`
- `proposer.base_url`
- `proposer.api_key`
- `org.enforce_capability_floor`
- `permissions.ask.read_only_signatures`
- `permissions.ask.dropped_commands`
- `permissions.ask.wrapper_commands`
- `permissions.ask.subcommand_tools`
- `permissions.ask.mcp_trusted_servers`
- `permissions.ask.timeout_s`
- `<name>.enabled`, for a plugin you installed (a plugin that ships with LocalHarness can be
  switched per project)

A few other permission settings are narrowed instead of ignored: a project may add deny patterns
and add commands to the gate's lists of dangerous calls but never remove any, may switch the
network-host question on but not off, may not pick a looser `permissions.mode`, and may not move
`permissions.workspace_root` outward.

**What this does NOT cover.** Your global config is trusted as it is: a value already in your
global files is never checked, whoever put it there. A project value equal to your global value is
treated as yours and left alone. A plugin you install decides for itself which of its settings are
machine-level only; this list covers the plugins that ship with LocalHarness. Apart from the
settings above, a project's `org` settings (such as `org.audit_log_path`) still merge over your
global ones; they have not yet been checked one by one for whether a project value can loosen a
protection.

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
2. **Evidence that you have already worked here.** Any `agents/*/sessions/*.jsonl` under the
   workspace's own `.localharness/` — or under the global store, for a session rooted at `$HOME`
   with no project — counts as prior use. A place you have worked in is not a place to be asked
   about, so the harness records the trust, prints one line saying it recognized the workspace,
   and moves on. (The global store only vouches for the home-rooted case; one old home session
   can never vouch for a project directory nobody has opened.)
3. **The question**, asked through whatever channel you are on: inline in the terminal, a dialog
   in Zed, a message in Discord. A "yes" writes the trust record and is never asked here again —
   one record, both halves: the project's `.localharness/` config loads, and its tool calls run
   under `auto`. A "no" is recorded too, and the session runs `guarded`.

**A session that cannot ask and has no record runs `guarded` and records nothing** — fail closed,
and leave the question for the next interactive session in that directory rather than answering it
on that person's behalf. **An explicitly configured `permissions.mode` skips all of this**:
`guarded` and `read-only` already ask or refuse, and `trusted` and `unattended` are deliberate
loosenings someone typed into a config, so confirming a decision you just made is exactly the
fatigue this release removes.

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
     **one six-entry list protects every harness config directory**, global or in-project —
     `config.yaml`, `overrides.yaml`, `trusted_workspaces.yaml`, `grants.yaml`,
     `declined_workspace_offers.yaml` and `plugins/**`, the files that change what the harness does
     next. Everything else under one (agents, divisions, tools, session state, memory, history, the
     audit log, the kill file) is the harness being *used* and is not protected; naming what is
     protected rather than what is exempt also means a new kind of runtime state added there is
     allowed by default rather than becoming a prompt nobody wanted.
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
`permissions.ask.network_hosts: true` for a per-host prompt. Exfiltration is handled structurally
instead — an agent that ingests untrusted text holds no host-mutating tools (see the prompt-injection
section below). Edits inside the project are silent when the channel shows you the change: in Zed
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
What a repository *can* still add is deny patterns, MCP servers, and agents with more tools — and
because every one of those tools passes this same gate, what it has added are things that **ask**,
not things that run. Edit `grants.yaml` to change an answer; there is no CLI verb for it, the prompt
is the interface and the file is the escape hatch.

**Five modes, set in config or switched mid-session with `/mode`.** `auto` is the default: one
trust question per workspace, then everything runs except the step-2 blacklist, and nothing is
remembered beyond that one answer. A workspace you declined, and a session with nobody to ask and
no record, run `guarded` instead. `guarded` —
the v0.14.0 default, now opt-in with `permissions.mode: guarded` or `/mode guarded` — asks once
about each step-4 class and remembers your answer, and asks about every write when there is no
boundary. `trusted` is `auto` plus one thing: a destructive file operation aimed **inside** the
project asks too. `read-only` refuses writes, non-read-only shell, and code execution with a
message the model can re-plan against. `unattended` turns every ask into allow, leaving only your
deny patterns — **this is exactly how the harness behaved before v0.14**, named honestly. It is
never a default, and from v0.14.1 it is settable the same way the others are: `/mode unattended` in
the terminal, `mode unattended` in Discord, the picker in Zed, or `permissions.mode: unattended` in
the config file of a bench run or a scheduled job. A session that turns its own gate off is a
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
- **Prior use is taken as consent.** A workspace with earlier LocalHarness sessions in its state
  store is trusted without ever being asked about, and the trust is then recorded. That is the
  behaviour the owner asked for — being re-asked about the project you live in is the fatigue this
  release removes — but state it plainly: sessions that ran under an older version, or under
  `guarded`, or that someone else's run left behind in a shared checkout, are what that evidence
  is made of. Delete the entry in `trusted_workspaces.yaml` to be asked again, and remember that a
  `.localharness/` you did not create is a reason to look before you run.
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
  `$(echo rm) -rf build` and `"$RM" -rf build` cannot be signed at classification time, so they are
  asked about as unknown commands rather than as destructive ones, and a grant on that placeholder
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

Two things happen on a start that are worth knowing about, because both write into your config
directory and neither stops to ask.

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

**`start` also seeds `<config-dir>/tools/design-screenshot.js`.** The frontend-designer builtin
shells out to that script by path, so the package copies it into your config directory's `tools/`
folder on first run. It is idempotent — present means untouched — it is written to your global
config directory and never into a workspace, and a failure to copy it is a warning rather than a
blocked start. It is the only file `start` installs from the package.

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

## Securing the endpoint

Inference servers ship with no authentication. On a network with untrusted devices,
start the server with an API key and set `provider.api_key` to match; for access
beyond your LAN use a private overlay network (Tailscale/WireGuard). Never port-forward
a bare, unauthenticated endpoint to the internet. See "Running the harness on a
different machine than the model" in the README.

### `localharness web`

The web channel adds a second listener, and it is a more serious one than the model
server: **it can run arbitrary shell with your privileges**, because that is what the
agent behind it does. Three things hold it, and each is there for a reason:

- **It binds loopback only.** Startup refuses any other address unless
  `--allow-unsafe-bind` is passed explicitly. A fronting proxy publishes it — this is
  also what keeps a proxy's identity headers meaningful, since they are only
  trustworthy if nothing but the proxy can reach the backend. `web.public_url`, which
  `localharness plugins enable web` asks for, is only the address the pairing QR sends
  your phone to: it never changes what the server binds, and `--allow-unsafe-bind` stays
  a flag you pass on each run, never a saved setting.
- **An app token is required on every request**, the event stream included. It is
  generated on first run and stored `0600` under the global config directory.
  Binding loopback makes the boundary tighter in one sense and looser in another:
  every local process on the machine can now reach that port, including one an agent
  itself started. `localharness web --rotate-token` invalidates every enrolled client.
  **Named gap:** there is no per-device revoke — rotation is all or nothing.
- **The POST surface is CSRF-safe by construction.** Every write verb requires both a
  bearer header and `application/json`, and the three simple content types a
  cross-origin form can send without a preflight are refused outright. The event
  stream uses a `Secure`/`HttpOnly`/`SameSite=Strict` cookie, because `EventSource`
  cannot carry a header and a token in a URL lands in logs and referrers.

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

Three more, added with the installable app and notifications:

- **The enrolment QR carries the token in the URL *fragment*** (`https://host/#t=…`),
  never the query string. A fragment is not sent to any server, so it reaches no access
  log, no proxy log and no `Referer` header — which is what the rule against tokens in
  URLs is actually protecting. The page reads it once and erases it from the address
  bar, so it does not survive into history or a screenshot either. The alternative is
  hand-typing a 256-bit secret on a phone keyboard, which is the setup step people skip.
- **Registering for push takes the app token.** A push subscription is a standing channel
  into your lock screen, carrying deep links to the calls the harness wants approved, so
  `POST /api/push/subscribe` is gated exactly like every write verb. The VAPID private
  key is generated by the harness, stored `0600` beside the token, and never leaves the
  box; payloads are encrypted to each device's own key, so the push service that relays
  them cannot read them.
- **The web app manifest is served with the token only to an authenticated caller.** An
  installed iOS web app gets a different storage jar from Safari's, so the manifest's
  `start_url` is what can pair it — and that variant is served `no-store`, only to a
  request that already presents the credential. An anonymous fetch gets a manifest with
  no token in it, which is the version anything reaching the port can see.

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

Added with the web plugin:

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
  answer 404. A test pins that the web channel's server, channel, push, replay and protocol
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
  `components set` audit event show `**********`. Until 0.17.0 the old `LOCALHARNESS_DISCORD_*`
  variables, `DISCORD_BOT_TOKEN` and `~/.claude/channels/discord/.env` still fill an unset field,
  with a deprecation warning. Files that users upload are kept as metadata only and are not passed
  to the model; outbound files are read only from core's artifact folder. Limits: one settings
  section, `dispatch.discord.*`, feeds every dispatch adapter today (a second platform would share
  Discord's token and allow-list until it gets its own section), and this plugin build has not yet
  been verified against a live Discord server.

**Two live sessions on one agent are warned about, not prevented.** `history.jsonl` and
`compact.md` take unlocked appends, so a terminal session and a web session on the same
agent can interleave their writes. Starting the second one names the first; neither is
refused. Closing one is still your call to make.
