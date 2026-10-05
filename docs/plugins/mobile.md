# Mobile (the phone app)

Mobile lets you drive a LocalHarness session from your phone on your own private network: `localharness mobile` serves a JSON event API and a bare reference page that uses it. It is the `mobile` plugin in the CLI. It ships with LocalHarness, is on by default, and needs the `mobile` install extra; the server and the page are in [`channels/mobile/`](../../src/localharness/channels/mobile/), and the plugin class is [`cli/mobile_plugin.py`](../../src/localharness/cli/mobile_plugin.py).

## Turn it on and off

```bash
uv tool install 'localharness[mobile]'      # the extra: starlette, uvicorn, push and QR support
localharness plugins disable mobile          # removes the `mobile` command
localharness plugins enable mobile
```

Without the extra, `localharness plugins list` shows it as `on (install localharness[mobile] to use it)`.

`localharness plugins enable mobile` on a terminal asks the phone address: the URL your phone opens. Press Enter to leave it empty and let `localharness mobile` guess it. In a running session, `/plugins enable mobile` asks it too, until `localharness mobile` has run once or an address is saved. Press Enter there and the address stays unset: the session comes back saying "mobile: on, but not set up yet — not enrolled yet" until `localharness mobile` has run once, the normal state of a new install.

## What it adds

- `localharness mobile`, run from the project folder you want the session in. On a terminal it draws a pairing QR code so you never type the app token; when its output goes to a pipe or a log, it prints one line instead. Options include `--port` (default 8765), `--show-token` (print the token and the QR on a terminal and exit, to pair a phone with a server that is already running), `--incognito`, `--rotate-token` (every device pairs again and turns notifications on again; a running server keeps the old token until it restarts), `--allow-unsafe-bind` (it binds to loopback only without it), and `--replay` / `--fixtures` to build a UI with the model server down.
- On the phone: streaming answers, live tool calls, the permission gate and the pending queue, a memory screen (when memory is on), a picture gallery (when a plugin makes pictures), home-screen install and a notification when a long turn finishes or the gate needs you.
- A `doctor` check: whether the phone is paired, the address the server binds, and that the token file is mode 600.

## Settings

| Setting | Meaning |
|---|---|
| `mobile.public_url` | **Machine-level only.** The address your phone opens, put in the pairing QR. Empty: `localharness mobile` guesses it from Tailscale. `--public-url` still wins for one run. |

The bind address and `--allow-unsafe-bind` stay flags of `localharness mobile`, never settings. A token is required on every request; it is generated on first run and stored in your global config folder, and its text is printed only to a terminal: when it is created, after `--rotate-token`, or with `--show-token`. The phone's event stream uses a cookie derived from the token, which only the GET routes accept. An app added to the home screen asks for the token once.

A paired phone can do what you can do at the terminal, including switching the session to `unattended` and answering "always". The machine-level setting `channels.remote_unattended: false` keeps those two to the terminal and Zed; the default, `true`, changes nothing. See [SECURITY.md, a paired phone or Discord account](../../SECURITY.md#a-paired-phone-or-discord-account).

## Not there yet

- It is a reference page to fork, **not a finished chat app**.
- One live session at a time, from the folder you started in: no chat list, no search, no resume, no concurrent sessions. Nothing stops you running the terminal and the phone on the same agent at once, which can mix their history; you are warned.
- Incognito only keeps pictures off the phone. Memory, sessions and pictures are still written to disk.
- No image or file upload, no diff review, no per-device revoke (`--rotate-token` re-pairs every device and clears push subscriptions), and the memory screen has no promote button.
- The memory screen now reads memory through the memory plugin; this is tested on the server only, not yet on a phone.
- `/plugins` works only in the terminal: a plugin's setup questions are never asked over the phone.
- Once `localharness mobile` has run once, or an address is saved, `/plugins enable mobile` answers "mobile is already on.": change the phone address from a shell with `localharness plugins enable mobile`.

## More

- [docs/mobile.md](../mobile.md): install, pairing, notifications, reaching it from the phone, building your own UI
- [SECURITY.md, `localharness mobile`](../../SECURITY.md#localharness-mobile)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [localharness.dev/plugins/mobile](https://localharness.dev/plugins/mobile/)
- [Write your own plugin](../../examples/plugin-template/README.md)
