# Use LocalHarness in Zed

Zed talks to outside agents through one mechanism: the [Agent Client
Protocol](https://agentclientprotocol.com) (ACP), JSON-RPC over stdin and stdout. LocalHarness
ships that as a subcommand — `localharness acp` — so it appears in Zed's agent panel next to any
other agent, running entirely on your own model server.

## Install

```bash
uv tool install localharness
localharness init     # detects your model server and writes ~/.localharness/config.yaml
```

`localharness acp` needs a working `localharness start`. If `start` cannot reach your model
server, neither can Zed.

## Register it with Zed

Open Zed's `settings.json` (`cmd-shift-p` → *open settings*) and add:

```json
{
  "agent_servers": {
    "LocalHarness": {
      "type": "custom",
      "command": "localharness",
      "args": ["acp"],
      "env": {}
    }
  }
}
```

Use the absolute path to the binary (`~/.local/bin/localharness`, or the output of
`which localharness`) if Zed's environment does not have it on `PATH`.

Then open the agent panel, click **New Thread**, and pick **LocalHarness**.

To point one Zed thread at a different config directory, add it to `args`:
`["acp", "--config-dir", "/path/to/config"]`.

## What to expect

**Open a project folder first.** LocalHarness derives a workspace boundary from where the
session stands — the nearest in-project `.localharness/`, else the git top level, else the
folder itself. If that resolves to your home directory or above there is no boundary at all, and
the first message you get back says so and asks you to open a project folder instead of a bare
directory. Nothing runs until you do.

**The first message takes as long as your model server does.** Zed gets a session id
immediately; the model server and the agent come up on your first prompt, which is where
progress can actually be shown. A warm server costs nothing; a cold one streams a
`Starting the session — 40s so far…` line every fifteen seconds until it is ready.

**Four modes, in Zed's mode picker.**

| Mode | What it does |
|---|---|
| **Auto** (default) | Asks once whether you trust this project, then stays out of the way. A dialog appears only for a protected or system path, a destructive command aimed outside the project, or an irreversible one — `sudo`, `curl … \| sh`, a force push, `git reset --hard`. Nothing is remembered. |
| **Guarded** | The v0.14.0 default: asks once about each new thing — anything leaving the project folder, a protected path, a command this workspace has never allowed — and remembers the answer. Reads and web fetches never ask. |
| **Trusted** | Auto plus one prompt: a destructive command whose target is inside the project asks too. |
| **Read only** | Writes, edits, code execution and non-read-only shell commands are refused with an explanation the model can re-plan against. |

**What to expect in Auto.** Opening a project you have not used before, the first dialog is the
trust question: trust this workspace? Answer yes and ordinary work — reading, editing, running
your build, an unfamiliar command, an MCP tool — never raises another dialog. Answer no and the
thread runs in Guarded, which asks about each new thing and remembers it. After that the only
dialogs you should see in a session are the dangerous ones in the table above, and they come back
every time, because an "always" on them would be a lie. If you are getting more than that, treat
it as a defect worth reporting.

There is a fifth mode, `unattended`, which turns every question into a yes. It is config-only
(`permissions.mode: unattended`) and deliberately not in the picker.

**The permission dialog.** When the gate decides a call needs a human, Zed shows its own
permission dialog with up to four buttons: *Allow once*, *Always allow in this workspace*, *No*,
*Never allow in this workspace*. Some calls — destructive shell commands, protected paths —
offer only the two "once" buttons, because they are designed to ask every single time and an
"always" there would be a lie. An "always" answer is written to `~/.localharness/grants.yaml`,
keyed by this workspace, and it holds in the terminal and Discord too — one gate, one memory.
Dismissing the dialog with Escape refuses that one call and remembers nothing.

The first time you open a project, you get the workspace trust dialog described above. It is one
question covering both halves of trust: whether that workspace's `.localharness/` config is loaded
— it defines roles, models and tool permissions, so treat it like code you are about to run — and
whether its tool calls run without asking. Answered once, permanently, in
`~/.localharness/trusted_workspaces.yaml`.

**Edits go through Zed.** Because Zed advertises filesystem access, `read` sees the buffer you
are actually looking at (unsaved edits included) and `write`/`edit` hand their changes to Zed
rather than writing to disk — so they land in the review pane where you can accept or reject
them. That review surface is also why an in-workspace edit does not ask for permission at all:
you are going to see it.

This routing is all-or-nothing, and it takes **both** ACP filesystem capabilities —
`fs/read_text_file` *and* `fs/write_text_file`. Every editor-backed write is a read first: ACP
has no append, so appending rewrites the whole file, and `edit` matches its `old_string` against
the buffer. A client offering only the write half would therefore append onto (or diff against)
the stale copy on disk and throw away your unsaved lines, so the harness wires neither hook for
it: `read`/`write`/`edit` touch the disk exactly as they do in a terminal, there is no review
surface, and an in-workspace edit asks once per workspace instead. Zed advertises both, so in
Zed you get the review pane; the fallback is for other ACP clients.

**Cancel works.** The stop button cancels the running turn, the same path `Ctrl-C` takes in the
terminal.

## Not yet

Honest list of what this adapter does not do in its first version.

- **No session resume.** `session/load` is not advertised, because session ids are fresh on
  every start and there is nothing to resume. Reopening a thread starts a new session.
- **Thinking is shown as ordinary text.** The streaming callback carries no phase yet, so
  reasoning arrives as message chunks rather than Zed's collapsible thought blocks.
- **One thread per agent process.** ACP allows an editor to run several threads over one agent
  process ("Each connection can support several concurrent sessions"), and this adapter does not.
  The harness derives its boundary, config layer, memory and state directory from one folder the
  process changes into, and the loop, the permission gate and the running turn all belong to that
  one session — a second thread would be served by the first one's session under a different id,
  which is the kind of quiet wrong answer the whole boundary exists to prevent. So a second
  thread on the same agent server is refused with a message saying so, whether it opens the same
  folder or another one. If your editor gives a second thread its own process (Zed's docs do not
  say either way), you will not notice this at all; if it does not, run a second LocalHarness
  agent server entry in `settings.json` for the second thread.
- **Zed's terminal capability is unused.** `bash_exec` runs the command itself; it does not
  appear as a Zed terminal you can watch or kill.
- **@-mentions and attachments are announced, not read.** Only the text of a prompt reaches the
  agent. Everything else — an `@file` mention, a pasted image, an attached resource — arrives as
  one line saying `[attachment: <name> — not read by this agent]`, in the place you put it. So
  the agent knows something was attached and can ask you for it (paste the text, or give it the
  path and let `read` open it), instead of answering about a file it never saw.
- **MCP servers Zed passes with the thread are not connected.** Zed hands its own MCP server list
  to the agent when a thread opens; this version does not start them, and says so in the panel on
  your first message and in the server log. LocalHarness connects the MCP servers declared in its
  own config (`tools.mcp_servers`), which are shared by Zed, the terminal and Discord alike.
- **Not in the ACP registry yet**, so this is a manual `settings.json` entry rather than a
  one-click install. That is a later phase.
- **No per-turn token usage** is reported back to Zed.

## When something goes wrong

The protocol owns stdout, so everything LocalHarness would normally print goes to stderr. In
Zed, the agent panel's menu has **View Server Logs** — startup failures (no config, a config
this version rejects, a model server that is not answering) print their real explanation there.
The panel itself will tell you a session could not be started and point you at that log.
