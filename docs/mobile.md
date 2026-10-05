# Use LocalHarness from a phone

LocalHarness ships a fourth channel: a small HTTP server that puts the whole session on a web
page. `localharness mobile` serves two things — a JSON event API, and a static page that consumes
it — so a phone on your own private network can drive a real session: streaming answer text, live
tool calls and their results, the permission gate, the pending queue, and the instruments the
terminal footer shows.

**Read this before you install it.** What ships today is a **single-file reference page** — one
HTML file with one inline module, no framework and no build step. It wears the project's palette
and is meant to be usable one-handed, but its job is to exercise every part of the wire so the
contract is demonstrable and copyable. It is **not a finished chat app**, and it is not trying to
be one. The intended use is that you fork the page and build the interface you want on top of an
API that is already complete.

**One live session at a time.** `localharness mobile` binds to the directory you launch it in — the
same rule `localharness start` and `localharness acp` follow — and serves exactly one live session
from that directory. There is no switching between live chats. Close the page, come back, and you
are in the conversation you left: a session nobody has used for thirty minutes goes to sleep (its
conversation is written beside its session log and the session is torn down), and your next
message wakes it and continues the same conversation. Restart the process and it continues the
same way, from that file. The `+` button starts a fresh chat; the old one stays in the drawer.

## Install

```bash
uv tool install 'localharness[mobile]'
localharness init          # detects your model server and writes ~/.localharness/config.yaml
```

The `mobile` extra pulls the ASGI app and the server (`starlette`, `uvicorn`). `localharness mobile`
needs a working `localharness start`: if `start` cannot reach your model server, neither can the
phone.

## Run it

```bash
cd ~/your-project
localharness mobile
```

It prints the address and the UI directory, and on a terminal it draws a **QR code that encodes
both the URL and the app token**, so you never type the secret into a phone. The token is generated on
first run and stored `0600` under your config directory, and it is required on every request, the
event stream included. Its text is printed only on that first run, right after `--rotate-token`, or
when you ask for it:

```bash
localharness mobile --show-token      # print the token and the QR on this terminal, then exit
```

`--show-token` pairs a phone with a server that is already running, without stopping it. When
standard output is not a terminal — a pipe, a `tee`, journald — nothing that carries the token is
printed there: if the process runs in a terminal, the pairing QR is drawn on that terminal
instead, and otherwise one line says to run `localharness mobile --show-token` on a terminal.

Tell it the address the phone will actually use:

```bash
localharness mobile --public-url https://your-machine.your-tailnet.ts.net
```

or save it once with `localharness plugins enable mobile` (the machine-level setting `mobile.public_url`);
`--public-url` still wins for one run. Without either, it asks `tailscale status` for a guess and labels it as a guess; if there is no
answer it prints the loopback URL and says plainly that no phone can reach it.

```bash
localharness mobile --rotate-token     # invalidate every enrolled client and push subscription, print a new QR
```

After a rotation every phone pairs again and turns notifications on again. A `localharness mobile`
that is already running keeps accepting the old token until you restart it.

## Put it on the home screen

Do these in order. **The order is not cosmetic** — iOS grants notifications only to a web app
that is already on the Home Screen, and asking before that produces a subscription that can never
deliver anything.

1. **Publish it over HTTPS** (`tailscale serve --bg 8765`). Over plain HTTP there is no install
   and no notifications, and that is the browser's rule, not ours.
2. **Scan the QR** with the phone's camera. Safari opens the page already paired.
3. **Share → Add to Home Screen.** You now have an icon that opens straight into a composer. iOS
   keeps the installed app's storage apart from Safari's, so the app asks for the token once:
   `localharness mobile --show-token` prints it.
4. **Open it from the icon** and tap **turn on notifications**. Before the install, that button
   tells you to install first instead of asking.

The icon is a deliberate placeholder — three bars on a dark square. Replace it by dropping your
own `icon-192.png`, `icon-512.png` and `icon-180.png` into the UI directory (`--ui-dir`, or the
packaged one the banner prints). No build step, no restart.

## Notifications

The phone buzzes for exactly two things:

- **A turn finished while you were away.** Only if nothing is watching — with the page open and
  visible you get no notification — and only for a turn longer than the measured median, because
  a turn you could have stood and waited for is not news.
- **Something needs you:** a parked call, a blocking permission question, or the harness's own
  "stuck, needs a human" signal.

Nothing else, ever. A twenty-tool-call turn produces no notifications at all. Repeats within the
same session collapse onto one notification whose badge counts them, so parking three calls in ten
minutes buzzes once and shows three — a phone that buzzes on every tool call is a phone whose
owner turns notifications off, and then the permission badge reaches nobody.

Tapping one opens that chat with that item in view.

The VAPID keys are generated by the harness on first use and stored `0600` beside the token. You
are never sent to an external site to mint push keys, and the payloads are encrypted to each
device, so the push service relaying them cannot read them.

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

**Plain HTTP on a LAN address does not work at all — it is not a reduced version of this.**
Two browser rules, not LocalHarness limitations, and the second is the fatal one:

- No service worker on a non-secure origin, so no home-screen install and no notifications.
- **The event stream cannot authenticate.** Enrolment sets a `Secure` cookie (a value derived from
  the token, which only the GET routes accept), browsers refuse to
  keep a `Secure` cookie on a plain-http origin, and `EventSource` cannot send the bearer header
  instead — so every connection attempt is rejected and the page retries forever. It looks
  exactly like bad wifi. The page now says so on load rather than letting you debug your router.

`http://localhost` is exempt from both: browsers treat it as a secure context, which is why the
development loop works without any of this. For a phone, put TLS in front of the port.

The app token is required in all three topologies, precisely because the network boundary differs
between them.

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
| Past chats / resume | no | scrollback | per Zed thread | **yes** — a drawer, and a sleeping chat resumes |
| Notifies you when a long turn finishes | no | yes (it is a chat app) | no | **yes** — lock-screen push |
| Reachable before the model server is up | no | yes | yes | **yes** |
| Memory browsing (`/memory`) | best — a real tree | flattened | flattened | flattened into a `<pre>` |

The short version: the **terminal** is the full-fidelity surface and the only one with a proper
memory view. **Discord** is for driving a box you are not sitting at, and it shows the least.
**Zed** is for editing with an agent beside you. The **mobile** channel is the only one that streams
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

**A session nobody is using goes to sleep, and your next message wakes it.** After thirty minutes
with no phone attached, no turn running and nothing waiting (`--sleep-after MINUTES`; `0` never
sleeps), the live session writes its conversation, its prior context and its permission mode to
`sessions/asleep.json` beside its session log, owner-only, and tears itself down through its
normal shutdown: memory closes the sitting, and the model client, the memory store, the embedding
model and the consolidation timer go with the session. The server stays up holding the channel
and the token: on the reference box, 83 MB before any session, 1.8 GB with a session and the
embedding model loaded, 0.9 GB asleep (the model's memory is released; the libraries it needed
stay imported). The ribbon says *asleep — wakes on your next message*. Your next message brings a
session up from that file and the conversation continues where it left off, as a new sitting: the
model has every earlier turn, and memory sees one sitting end and another begin. Opening the app
does not wake it; only a message does — a backgrounded page reconnects on its own, and the point
of sleeping was that nobody's pocket keeps the box busy. The first reply after a wake pays the
warm-up. Ctrl-C with a live session puts it to sleep the same way, so a server restart continues
the conversation too; the `+` button is how you leave a sleeping conversation behind. A wake that
fails before a session exists — the model server down, say — puts the file back for the next
message to try again. The default leaves memory's dreaming pass (ten minutes into the quiet) room
to run first; below about thirteen minutes it waits for the next wake. A parked permission call
or an open question keeps the session awake: both live only in that session's gate.

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

**A paired phone can do what you can do at the terminal** — including switching the session to
`unattended` and answering "always". To keep those two to the terminal and Zed, set
`channels.remote_unattended: false` in your machine config: the phone then gets one line naming the
setting, and its questions offer no "always". The default, `true`, changes nothing, and
`localharness doctor` shows a row while it is on. See
[SECURITY.md](../SECURITY.md#a-paired-phone-or-discord-account).

**Memory and Pictures appear only when the session has them.** The drawer's Memory button shows
when memory is on: list and search facts, open one with its history, edit it, forget it (two
taps). There is no promote button. The Pictures button shows when a plugin that saves pictures is
on: a grid of this session's pictures, newest first, 60 at a time, and a full-screen viewer that
shows only the picture's time (UTC). `localharness mobile --incognito` (or the drawer's Incognito
switch) hides Pictures and stops the phone keeping any picture; it does not make the machine
forget anything — memory, sessions and files on the box still persist.

**Stop has an undo window.** Cancel sits next to the composer where a distracted thumb lands, and
the GPU is the scarce resource here.

## Building your own UI

The page is served **from a directory, live**. Edit the file, pull to refresh. No build step, no
bundler, no server restart.

```bash
localharness mobile --ui-dir ~/my-ui
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
localharness mobile --replay ~/.localharness/agents/orchestrator/sessions/<id>.jsonl --speed 4
```

The transcript, the tool calls and the parked queue replay for real. The live-progress frames —
streaming text, the instrument cluster — are *synthesized* from the log and flagged as such, so
you can build the streaming bubble offline without ever mistaking a replayed rate for a
measurement. A blocking permission question has no persisted form at all, so it comes from a
fixtures file:

```bash
localharness mobile --replay <log>.jsonl --fixtures ./my-fixtures.json
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

- **The installed app asks for the token once.** iOS gives a home-screen web app its own
  storage, separate from Safari's, so pairing the tab does not pair the app. The manifest carries
  the token only for a request that presents the bearer token, which a browser's manifest fetch
  never does, so the app asks you for the token once (`localharness mobile --show-token` prints it)
  and then remembers it.
- **One live session, from the directory you started in.** The drawer lists past chats and
  searches what was said, and a sleeping chat resumes on your next message, but there is no
  switching between live chats and no reopening an ended one. After a server restart, the earlier
  turns of a woken conversation are in the drawer, not on the screen above the composer; the model
  has them. A tool result the sleeping session had evicted from its context comes back as its
  stub, and the model re-fetches it if it needs it: the eviction store is not in the file.
- **No concurrent sessions.** Not a scheduling convenience: two sittings under one agent append to
  the same unlocked per-agent history file, and the per-agent summary is last-writer-wins, so one
  chat can inherit another's prior context. Fixing that comes before concurrency, not after.
- **Nothing stops you running the terminal and the mobile channel at once** on the same agent, which
  reopens exactly that path. You are now *warned* when you start the second one, and the phone
  shows it too — but nothing prevents it, and that is the owner's call to make, not the harness's.
- **No image or file upload.** The attachment field exists in the event schema and nothing has
  ever produced or consumed it.
- **No diff review.** In-project edits therefore ask once per workspace rather than never.
- **The memory view is worse than the terminal's.** The Memory screen is a flat list with no
  promote, and `/memory` gives you the terminal's tree flattened into preformatted text.
- **No per-device revoke.** `--rotate-token` is all or nothing; every other device re-enrols, and
  a server that is already running keeps the old token until it restarts.
- **No grant revocation.** You can see what you have permanently allowed; removing one means
  editing `~/.localharness/grants.yaml` by hand.
- **A disk write that fails can leave a hole** in the log a reconnecting client replays from. The
  event bus logs the failure and delivers the event anyway, so a live page saw it and a fresh one
  cannot. The page shows a visible gap marker rather than pretending nothing happened.

## When something goes wrong

**401 on everything.** The token is wrong or was rotated. Scan the QR again (`localharness mobile`
draws it on every start on a terminal), or print the token with `localharness mobile --show-token`.
After a rotation, a server that was already running keeps the old token until it restarts.

**The page loads but nothing streams.** Something between the phone and the process is buffering
`text/event-stream`. `tailscale serve` is the tested path; a reverse proxy of your own needs
buffering off for that content type.

**"model server unreachable".** That is a real TCP probe of your provider endpoint, not a guess —
the model server is down or the URL is wrong. "cold" is different and means the session simply has
not been built yet; send a message and it will be. "asleep" means the conversation is on disk and
your next message continues it.

**A question expired before you got to it.** The gate's deadline, not the page's. Use `auto`, where
calls park and wait indefinitely, or answer from the terminal.

**Two sessions on one agent.** Starting the second one now names the first (channel, pid and
directory), and the phone shows the same thing. Nothing is refused: they share that agent's
unlocked `history.jsonl` and `compact.md`, so the writes can interleave. Close one, or carry on
knowing that.

**"add to Home Screen first".** The notifications button says that when the page is running as a
browser tab. iOS will not grant push to a tab; install it and open it from the icon.

**No notifications on the phone, and no button either.** The page is not on HTTPS. A service
worker — and therefore the install and push — needs a secure context; put `tailscale serve` (or
any TLS proxy) in front of the port and reopen the URL.

**Notifications were on and stopped.** Rotating the token clears every push subscription, and
reinstalling the app or clearing site data drops that device's. Turn them on again from the app
after it pairs; a subscription the push service reports gone is dropped automatically.
