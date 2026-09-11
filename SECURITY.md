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

**Inside a workspace, file writes and commands default to the project folder.** When a workspace
layer applies, the write and edit tools and the working directory for `bash_exec` default to the
folder that contains `.localharness/` — the project you are standing in. Outside a workspace the
default is unchanged and those tools are unconfined, exactly as before. A `workspace_root` you set
in your own config still wins either way. Read this honestly: it is a default that narrows what the
tools reach by accident, not a sandbox. A command run through `bash_exec` can still leave that
folder, and the deny patterns remain the mechanism that stops specific actions.

## Human approval gate

From v0.14 every tool call of every agent — subagents included — passes one decision function
before it runs. The function is code, not a judgment call by a model: the same call in the same
workspace always gets the same answer. It runs in a fixed order and the first match wins.

1. **Deny.** Your deny patterns, unchanged from earlier versions. Nothing overrides them — not a
   grant, not a mode.
2. **Ask, and no answer is remembered.** Destructive shell commands (`rm -rf`, `git push --force`,
   `git reset --hard`, `chmod -R`, `sudo`, `curl … | sh`, `find -delete`, `docker`, `dd` and the
   rest), any write whose target is a protected path (`~/.ssh`, `~/.aws`, `~/.gnupg`, `~/.config/gh`,
   your shell rc files, `~/.localharness`, and inside the project `.git/**`, `.localharness/**`,
   `.env*`, `*.pem`, `id_*`), and every write-shaped call made when there is no workspace boundary
   at all. These ask **every time**, on purpose. The flags are part of what is matched, so `rm file`
   and `rm -rf dir` are different things and an answer about one is never an answer about the other.
3. **Grants.** A remembered "always" for this workspace. Checked only after step 2, so an old
   permissive answer can never cover a destructive variant.
4. **Ask once, then remember.** A write or shell write target outside the project folder (keyed by
   the target's parent directory), a shell command whose signature this workspace has not seen
   before, an inline interpreter (`python3 -c`, `bash -c`, `eval`, `xargs`), `python_exec` and
   `cruncher_exec`, the `agent` tool, each MCP tool, and any tool in no family the gate knows —
   a plugin's tool, or one whose schema could not be read — keyed by the tool's name, because a
   tool nobody can describe is asked about rather than allowed.
5. **Allow.** Everything else: reads, search, memory, `chunk`, the read-only shell commands (`ls`,
   `cat`, `head`, `tail`, `grep`, `rg`, `find` without `-exec`/`-delete`, `git status`/`diff`/`log`,
   `sed -n`, and their kin), network reads, and edits inside the project when the channel can show
   you the diff.

**The boundary is derived from where you stand, never configured.** It is the folder holding the
nearest in-project `.localharness/`, else the git top level, else the directory you started in,
resolved through symlinks. A `permissions.workspace_root` in your config may only *narrow* it; a
value outside it is ignored with a warning. **If that folder turns out to be your home directory or
anything above it, there is no boundary** — the harness says so rather than pretending your whole
home is one project, and every write-shaped call then asks without a remembered answer.

**Two things deliberately never ask, and both are a judgment you should check against your own
threat model.** Network reads (`web_fetch`, `web_search`, `web_page_query`) are silent: seven in ten
real tool calls are web fetches, and a prompt per host would fire in a third of all sessions. Set
`permissions.ask.network_hosts: true` for a per-host prompt. Exfiltration is handled structurally
instead — an agent that ingests untrusted text holds no host-mutating tools (see the prompt-injection
section below). Edits inside the project are silent when the channel shows you the change: in Zed
they land in the review pane with per-hunk accept/reject, in the terminal the diff is printed after
the fact. A channel with no review surface asks once per workspace instead.

**Answers live in your global config, and a repository can only tighten.** An "always" is written to
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

**Four modes, set in config or switched mid-session with `/mode`.** `guarded` is the default and is
the list above. `trusted` turns step 4 into allow while step 2 still asks. `read-only` refuses
writes, non-read-only shell, and code execution with a message the model can re-plan against.
`unattended` turns every ask into allow, leaving only your deny patterns — **this is exactly how the
harness behaved before v0.14**, named honestly. It is never a default and cannot be set from a chat
or terminal command; write it in config, which is what the benchmark runner and scheduled jobs do.

**A channel that cannot ask denies.** Bench runs, cron jobs, a piped non-tty session: if a call
reaches step 2 or 4 and there is nobody to answer, it is refused, the model is told "needs human
approval; this channel cannot ask", and the session prints one warning naming the fix. Failing
closed is the rule; the warning is what keeps it from being a silent regression in a scheduled job.

**Named gaps. Read these before you rely on any of it.**

- **A granted interpreter runs anything.** Answer "always" to `python3 -c` and every later
  `python3 -c` in that workspace runs unasked, including one that deletes a tree. `python -c` is the
  second signature nearly every workspace is asked about, so this is the widest hole by design and
  by frequency. `python_exec` is the tool to prefer; `trusted` mode is the honest alternative to
  granting interpreters one at a time.
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
  drivers, and the rest of a named list) is ungrantable and asks every time, including the
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
  cannot be resolved before the shell expands it, so it asks as an out-of-project write — and once
  you answer "always" for that unresolved shape, a later expansion of the same shape to a different
  path passes on that grant.
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
