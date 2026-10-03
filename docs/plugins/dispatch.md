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
- The bot reacts with the ack emoji when it takes a message, replies in 2000-character pieces, and asks permission questions as a message you answer with a reaction; only users on the allow list can answer. A plain `mode <name>` message switches the permission mode.
- Pictures a tool makes (the `image` plugin) are posted as files.
- A `doctor` check: whether Discord is configured and how many users are allowed. It never prints the token and never logs in.

## Settings

| Setting | Meaning |
|---|---|
| `dispatch.discord.token` | **Machine-level only.** The bot token. Never printed (shown as `**********`). |
| `dispatch.discord.allow` | **Machine-level only.** The Discord user ids that may talk to the bot and answer its questions. Required. |
| `dispatch.discord.channels` | **Machine-level only.** Channel ids to listen in; empty means any channel the bot can see. |
| `dispatch.discord.ack` | The reaction added to a message the agent takes; default ✅, `""` for none. A project may set it. |

A repository you clone cannot point the bot somewhere else or widen who may drive it: a project's value for the three machine-level settings is dropped with a warning.

The older `LOCALHARNESS_DISCORD_*` and `DISCORD_BOT_TOKEN` environment variables and the file `~/.claude/channels/discord/.env` are deprecated: they still fill a setting you have not set, with a warning at start and in `doctor`, and they stop working in 0.17.0. Move each to its setting with `localharness components set dispatch.discord.<key> …`.

## Not there yet

- **Not yet run against a live Discord server.** The plugin is tested offline against a stand-in for the Discord library: logging in, the reactions and a real file upload are unverified.
- Files people upload to the bot are not passed to the model; the turn sees the message text only.
- The token is stored as plain text in your global `overrides.yaml` (mode 600 when LocalHarness writes it).
- There is one settings section, `dispatch.discord.*`. A second chat platform added today would share Discord's token, allow list and channels.
- `localharness start --help` still lists `discord` while the plugin is off.

## More

- [Spec 11, the dispatch plugin](../specs/11-channels.md#the-dispatch-plugin-chat-platforms-discord-today): the channel, the adapter protocol, how a second platform is added
- [SECURITY.md, machine-level-only settings](../../SECURITY.md#machine-level-only-settings)
- [Spec 09, the bundled plugins](../specs/09-hooks-plugins.md#the-bundled-plugins)
- [localharness.dev/plugins/dispatch](https://localharness.dev/plugins/dispatch/)
- [Write your own plugin](../../examples/plugin-template/README.md)
