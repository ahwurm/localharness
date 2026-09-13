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
the harness runs, driven by a local model. **Treat agent definitions and any
connected MCP servers as trusted code**: review them the way you would review code,
because they decide what the agents are allowed to do.

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
repository before you run the harness there, the same way you would read its build scripts. Plugins
and the org-level guardrails file are never taken from a workspace — they load from your global
config directory only. For the guardrails file that is a mechanism rather than a side effect: the
memory store is given the global directory as a separate input from the directory its own state
lives in, so a workspace cannot silence the org's safety context by shipping its own copy of the
file, and cannot blank it by having no copy at all. One crossing does exist and it is yours to make:
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
   tool in no family the gate knows — a plugin's tool, or one whose schema could not be read —
   keyed by the tool's name, because a tool nobody can describe is asked about rather than allowed.
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
holds no host-mutating tools. This covers built-in web and tool-result ingestion and
MCP tools today; one known gap remains — a plugin pulled in through inherited global
scope still needs a per-tool ingestion tag to be caught.

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
