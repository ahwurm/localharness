# The dispatch plugin (Discord)

The `dispatch` plugin lets you drive a LocalHarness session from Discord: `localharness start --channel discord` turns messages from allowlisted users into turns and posts the replies, permission questions and pictures back to the conversation. It ships with LocalHarness, is on by default and needs the `dispatch` install extra; Discord is its only chat platform today.

## Turn it on and off

```bash
uv tool install 'localharness[dispatch]'    # discord.py
localharness plugins enable dispatch --set discord.token=… --set discord.allow=<your user id>
localharness start --channel discord
localharness plugins disable dispatch
```

On a terminal, `plugins enable dispatch` with no `--set`, or `/plugins enable dispatch` in a running session while it is not set up, asks for the bot token (nothing shows as you type, and a stored token is never shown) and your Discord user id(s); Enter on a question writes nothing. It then checks the settings, without logging in to Discord, and ends with the next step: `localharness start --channel discord`. When the check does not pass it also prints a prompt to paste into your coding agent, and that prompt never holds the token. Without the `dispatch` extra it names the install command and asks nothing. In the Discord Developer Portal, turn on your bot's Message Content intent and invite it to your server; your user id comes from Discord's Developer Mode (right-click your name, Copy User ID). `start --channel discord` refuses when the plugin is off, the extra is missing, the token is empty or the allow list is empty, and it never falls back to the terminal.

## What it adds

- The `discord` channel for `localharness start --channel`. No tool and no command.
- The bot reacts with the ack emoji when it takes a message, replies in 2000-character pieces, and asks permission questions as a message you answer with a reaction; only users on the allow list can answer. ✅ allows once and ❌ refuses; ♾️ ("always") takes a second tap, as on the phone: the bot posts a confirm message, and only ✅ on that message records the grant. The bot ignores its own reactions, so it never answers its own question. A plain `mode <name>` message switches the permission mode.
- What the bot sends pings nobody but the person it replies to (no `@everyone`, role or user mention from the model's text), and a masked link (`[text](url)`) is sent as its text followed by the plain address, so you see where it goes.
- An allowlisted account can do what you can do at the terminal, including `mode unattended` and "always". The machine-level setting `channels.remote_unattended: false` keeps those two to the terminal and Zed: `mode unattended` then gets one line naming the setting, and questions offer no ♾️. The default, `true`, changes nothing.
- Pictures a tool makes (the `image` plugin) are posted as files.
- Pictures you attach (PNG, JPEG, GIF, WebP; up to 4 per message, 20 MiB each) go to the model with your message; a picture alone is a message too. One that cannot be read (too big, past the fourth, a download failure, or not really an image) is named in the message text as `[attachment <name> not read: <reason>]`, never dropped silently. Pictures are downloaded only from people on the allow list. Seeing them needs a vision model; large ones are fitted to `context.max_image_tokens` like any other image.
- A `doctor` check: whether Discord is configured and how many users are allowed. It never prints the token and never logs in.

## Settings

| Setting | Meaning |
|---|---|
| `dispatch.discord.token` | **Machine-level only.** The bot token. Never printed (shown as `**********`). |
| `dispatch.discord.allow` | **Machine-level only.** The Discord user ids that may talk to the bot and answer its questions. Required. |
| `dispatch.discord.channels` | **Machine-level only.** Channel ids to listen in; empty means any channel the bot can see. |
| `dispatch.discord.ack` | The reaction added to a message the agent takes; default ✅, `""` for none. A project may set it. |

A repository you clone cannot point the bot somewhere else or widen who may drive it: a project's value for the three machine-level settings is dropped with a warning.

The token comes only from `dispatch.discord.token`. The deprecated environment variables of 0.16.x are no longer read (deleted in 0.17.1); set each value with `localharness components set dispatch.discord.<key> …`. Claude Code's Discord token file is not read either: with no token, `start --channel discord` refuses with one line naming `localharness plugins enable dispatch`, which says so.

## Not there yet

- **Not yet run against a live Discord server.** The plugin is tested offline against a stand-in for the Discord library: logging in, the reactions and a real file upload are unverified.
- Attached pictures are tested against the stand-in only: a real Discord upload reaching the model has not been run. Other files (PDFs, text) are still not passed to the model; the turn sees their name nowhere, only the message text.
- The token is stored as it is in your global `overrides.yaml` (owner-only, mode 600, when LocalHarness writes it); every command shows it as `**********`.
- There is one settings section, `dispatch.discord.*`. A second chat platform added today would share Discord's token, allow list and channels.
- `localharness start --help` still lists `discord` while the plugin is off.

## More

- [Spec 11, the dispatch plugin](../specs/11-channels.md#the-dispatch-plugin-chat-platforms-discord-today): the channel, the adapter protocol, how a second platform is added
- [SECURITY.md, machine-level-only settings](../../SECURITY.md#machine-level-only-settings)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [localharness.dev/plugins/dispatch](https://localharness.dev/plugins/dispatch/)
- [Write your own plugin](../../examples/plugin-template/README.md)
