# Mobile (the phone app)

Mobile lets you drive a LocalHarness session from your phone on your own private network: `localharness web` serves a JSON event API and a bare reference page that uses it. It is the `web` plugin in the CLI. It ships with LocalHarness, is on by default, and needs the `web` install extra; the server and the page are in [`channels/web/`](../../src/localharness/channels/web/), and the plugin class is [`cli/web_plugin.py`](../../src/localharness/cli/web_plugin.py).

## Turn it on and off

```bash
uv tool install 'localharness[web]'      # the extra: starlette, uvicorn, push and QR support
localharness plugins disable web          # removes the `web` command
localharness plugins enable web
```

Without the extra, `localharness plugins list` shows it as `on (install localharness[web] to use it)`.

`localharness plugins enable web` on a terminal asks the phone address: the URL your phone opens. Press Enter to leave it empty and let `localharness web` guess it. In a running session, `/plugins enable web` asks it too, until `localharness web` has run once or an address is saved.

## What it adds

- `localharness web`, run from the project folder you want the session in. It prints a pairing QR code so you never type the app token. Options include `--port` (default 8765), `--incognito`, `--rotate-token`, `--allow-unsafe-bind` (it binds to loopback only without it), and `--replay` / `--fixtures` to build a UI with the model server down.
- On the phone: streaming answers, live tool calls, the permission gate and the pending queue, a memory screen (when memory is on), a picture gallery (when a plugin makes pictures), home-screen install and a notification when a long turn finishes or the gate needs you.
- A `doctor` check: whether the phone is paired, the address the server binds, and that the token file is mode 600.

## Settings

| Setting | Meaning |
|---|---|
| `web.public_url` | **Machine-level only.** The address your phone opens, put in the pairing QR. Empty: `localharness web` guesses it from Tailscale. `--public-url` still wins for one run. |

The bind address and `--allow-unsafe-bind` stay flags of `localharness web`, never settings. A token is required on every request; it is generated on first run and stored in your global config folder.

## Not there yet

- It is a reference page to fork, **not a finished chat app**.
- One live session at a time, from the folder you started in: no chat list, no search, no resume, no concurrent sessions. Nothing stops you running the terminal and the phone on the same agent at once, which can mix their history; you are warned.
- Incognito only keeps pictures off the phone. Memory, sessions and pictures are still written to disk.
- No image or file upload, no diff review, no per-device revoke (`--rotate-token` re-pairs every device), and the memory screen has no promote button.
- The memory screen now reads memory through the memory plugin; this is tested on the server only, not yet on a phone.
- `/plugins` works only in the terminal: a plugin's setup questions are never asked over the phone.
- Once `localharness web` has run once, or an address is saved, `/plugins enable web` answers "web is already on.": change the phone address from a shell with `localharness plugins enable web`.

## More

- [docs/web.md](../web.md): install, pairing, notifications, reaching it from the phone, building your own UI
- [SECURITY.md, `localharness web`](../../SECURITY.md#localharness-web)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [localharness.dev/plugins/mobile](https://localharness.dev/plugins/mobile/)
- [Write your own plugin](../../examples/plugin-template/README.md)
