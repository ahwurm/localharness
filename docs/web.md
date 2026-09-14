# Use LocalHarness from a phone

LocalHarness ships a fourth channel: a small HTTP server that puts the whole session on a web
page. `localharness web` serves two things — a JSON event API, and a static page that consumes
it — so a phone on your own private network can drive a real session: streaming answer text, live
tool calls and their results, the permission gate, the pending queue, and the instruments the
terminal footer shows.

**Read this before you install it.** What ships today is a **bare, intentionally unstyled
reference page**. It is a worked example whose job is to exercise every part of the wire so the
contract is demonstrable and copyable. It is **not a finished chat app**, and it is not trying to
be one. The intended use is that you fork the page and build the interface you want on top of an
API that is already complete.

**One live session at a time.** `localharness web` binds to the directory you launch it in — the
same rule `localharness start` and `localharness acp` follow — and serves exactly one live session
from that directory. There is no chat list, no session switching and no resume yet. Close the
page, come back, and you are in the same session you left; restart the process and it is a new
one.

## Install

```bash
uv tool install 'localharness[web]'
localharness init          # detects your model server and writes ~/.localharness/config.yaml
```

The `web` extra pulls the ASGI app and the server (`starlette`, `uvicorn`). `localharness web`
needs a working `localharness start`: if `start` cannot reach your model server, neither can the
phone.

## Run it

```bash
cd ~/your-project
localharness web
```

It prints the address, the UI directory and an **app token**, generated on first run and stored
`0600` under your config directory. The token is required on every request, the event stream
included. Open the page and paste it once; it is kept in the browser's local storage from then on.

```bash
localharness web --rotate-token     # invalidate every enrolled client
```

## Reaching it from the phone

The server **binds loopback only**. That is deliberate: this endpoint runs shell commands with
your privileges, so it refuses to bind anything else unless you pass `--allow-unsafe-bind` and
mean it. A proxy publishes it.

| Topology | TLS | Home-screen install | Notes |
|---|---|---|---|
| **`tailscale serve` (recommended)** | Tailscale terminates it and **auto-renews** the certificate | yes | Also the only option that can tell you which device drove a session. |
| `tailscale cert` + your own HTTPS listener | 90-day certificates **you renew** — Tailscale does not renew files it handed you | yes | The fallback if the serve proxy ever buffers the event stream. |
| Plain LAN, or any reverse proxy you already run (WireGuard, Caddy, nginx, an SSH tunnel) | yours | only with a certificate the phone trusts | **Genuinely degraded, not equivalent** — see below. |

```bash
tailscale serve --bg 8765
```

**The plain-HTTP path is a browser tab, not an app.** Over plain HTTP on a non-localhost origin
the browser will not register a service worker, so there is no home-screen install and no push
notification. That is a browser rule, not a LocalHarness limitation, and it is said here rather
than left to be discovered. The app token is required in all three topologies, precisely because
the network boundary differs between them.

## Which channel should I use?

A fourth channel with no map is the predictable confusion. Honestly:

| | Terminal | Discord | Zed (ACP) | **Web** |
|---|---|---|---|---|
| Streams the answer as it generates | no | no | yes (in Zed) | **yes** |
| Shows tool calls with real arguments and results | yes | **no** — deliberate silence | yes | **yes** |
| Shows the model's reasoning live | yes | no | partly | **yes** |
| Can ask a permission question | yes | yes | yes | **yes** |
| Holds a question open with no deadline | yes | no | yes | no — a pocket is not a person |
| Reviews an edit as a diff | yes, after the fact | no | yes, per hunk | **no** (so in-project edits ask once per workspace) |
| Past chats / resume | no | scrollback | per Zed thread | **not yet** |
| Reachable before the model server is up | no | yes | yes | **yes** |
| Memory browsing (`/memory`) | best — a real tree | flattened | flattened | flattened into a `<pre>` |

The short version: the **terminal** is the full-fidelity surface and the only one with a proper
memory view. **Discord** is for driving a box you are not sitting at, and it shows the least.
**Zed** is for editing with an agent beside you. The **web** channel is the only one that streams
answer text *and* renders tool calls *and* can ask permission — which makes it the best surface
for *watching a turn happen*, and currently the worst for looking at anything that already
happened.

## What to expect

**The server is up before the session is.** Open the page with the model server cold and you can
still read the pending queue and the protocol. Connecting is itself the signal to start bringing
the session up in the background, so on a warm box the session is usually ready by the time you
have finished typing.

**With the model server actually cold, connecting does nothing** — deliberately. A backgrounded
page reconnects on its own, and bringing a session up can start a harness-managed model server,
so a phone in a pocket would otherwise be able to spin your GPU with nobody asking for anything.
Sending a message always brings the session up; opening the app only does so when the provider is
already answering. If you send before the session is ready, the page names the stage it is waiting
on and gives you a way out of a build that has stuck — a rising number tells you how long you have
waited, not whether anything is wrong.

**Permission questions are a queue, not a modal.** In `auto` — the default — a blacklisted call is
*parked*: the model is told to carry on without that step, the turn keeps running, and the page
shows a badge. Answer it from the phone, the terminal or Discord; whichever you use, it closes
everywhere. An approval means "run it when the model re-issues it", never "ran".

`guarded` and `trusted` ask with a real blocking question instead, and you should know what that
costs on a phone: the gate puts a deadline on it (25 seconds for a web tool, 60 for a shell
command), and buzz → notice → unlock → open → tap is not reliably a 25-second chain from a pocket.
Short-timeout tools will often expire into a denial before you can reach them. That is a
documented consequence, not a defect — `auto` exists so this is not the daily path.

**"Always allow here" takes two taps, and the second one is checked by the server.** A permanent
grant is global, keyed by your project's real path, never expires, and has no revoke command, so a
single request can never write one. `GET /api/grants` lists what you have allowed forever.

**Stop has an undo window.** Cancel sits next to the composer where a distracted thumb lands, and
the GPU is the scarce resource here.

## Building your own UI

The page is served **from a directory, live**. Edit the file, pull to refresh. No build step, no
bundler, no server restart.

```bash
localharness web --ui-dir ~/my-ui
```

The wire is the harness's own event bus, verbatim: every event reaches the page as the same bytes
the session log gets, with the bus sequence number as the stream's event id. Two endpoints
describe it, both generated from the code so they cannot drift:

- `GET /api/protocol` — the verbs, the event list (dead event types flagged), which frames are
  live-only, the slash commands, the permission modes, and the rendering rules that decide whether
  a transcript is correct.
- `GET /api/schema` — JSON Schema for every event and frame.

**Build it with the box asleep.** A recorded session replays as if it were live:

```bash
localharness web --replay ~/.localharness/agents/orchestrator/sessions/<id>.jsonl --speed 4
```

The transcript, the tool calls and the parked queue replay for real. The live-progress frames —
streaming text, the instrument cluster — are *synthesized* from the log and flagged as such, so
you can build the streaming bubble offline without ever mistaking a replayed rate for a
measurement. A blocking permission question has no persisted form at all, so it comes from a
fixtures file:

```bash
localharness web --replay <log>.jsonl --fixtures ./my-fixtures.json
```

```json
{"frames": [
  {"after_seq": 42, "frame": {
    "frame_type": "BlockingAsk", "request_id": "fx-1", "tool_name": "bash_exec",
    "tool_params": {"command": "rm -rf build/"}, "klass": "shell-destructive",
    "grantable": false, "display": "bash_exec: rm -rf build/"}}
]}
```

Fixture questions are genuinely answerable — they appear in `GET /api/permissions` and complete
the same round trip a real one does.

### Three rules that decide whether your client is correct

1. **Render the answer once.** It arrives three times — as streaming frames, as an `Action`, and
   as `TaskComplete.summary`. `TaskComplete` is the one you render. A tool-less `llm_response`
   `Action` must **not** be drawn; it is the same text.
2. **Everything is text, never markup.** A tool result that can execute script in your page is a
   tool result that can rewrite the permission dialog asking about it.
3. **Persist the last sequence number you saw, on every event.** iOS reclaims a backgrounded page,
   so the app relaunches into a fresh context with the browser's own resume state gone. Without
   your own cursor the reopened app jumps silently to the end and misses the answer.

`GET /api/protocol` ships all of these as data.

## Not yet

- **No home-screen install and no push notifications.** The manifest, the service worker and Web
  Push are the next slice of work. Until then this is a browser tab, and a long task is something
  you have to come back and look at.
- **No chat list, no titles, no search, no resume.** One live session, from the directory you
  started in. Past sessions are files on disk; nothing browses them yet.
- **No concurrent sessions.** Not a scheduling convenience: two sittings under one agent append to
  the same unlocked per-agent history file, and the per-agent summary is last-writer-wins, so one
  chat can inherit another's prior context. Fixing that comes before concurrency, not after.
- **Nothing stops you running the terminal and the web channel at once** on the same agent, which
  reopens exactly that path. Nothing warns you about it yet either.
- **No image or file upload.** The attachment field exists in the event schema and nothing has
  ever produced or consumed it.
- **No diff review.** In-project edits therefore ask once per workspace rather than never.
- **The memory view is worse than the terminal's.** `/memory` gives you the terminal's tree
  flattened into preformatted text. A real memory endpoint is the first thing after the chat list.
- **No per-device revoke.** `--rotate-token` is all or nothing; every other device re-enrols.
- **No grant revocation.** You can see what you have permanently allowed; removing one means
  editing `~/.localharness/grants.yaml` by hand.
- **A disk write that fails can leave a hole** in the log a reconnecting client replays from. The
  event bus logs the failure and delivers the event anyway, so a live page saw it and a fresh one
  cannot. The page shows a visible gap marker rather than pretending nothing happened.

## When something goes wrong

**401 on everything.** The token is wrong or was rotated. `localharness web` prints the current
one on startup.

**The page loads but nothing streams.** Something between the phone and the process is buffering
`text/event-stream`. `tailscale serve` is the tested path; a reverse proxy of your own needs
buffering off for that content type.

**"model server unreachable".** That is a real TCP probe of your provider endpoint, not a guess —
the model server is down or the URL is wrong. "cold" is different and means the session simply has
not been built yet; send a message and it will be.

**A question expired before you got to it.** The gate's deadline, not the page's. Use `auto`, where
calls park and wait indefinitely, or answer from the terminal.

**Two sessions on one agent.** If you are also running `localharness start` on the same agent,
stop one. Nothing enforces this yet and the failure is quiet.
