"""The dispatch plugin: chat platforms as a channel — `start --channel discord` drives the agent from
allowlisted chat messages, replies posted back.

On by default; needs the `dispatch` install extra (discord.py). plugins/builtin.py imports this
module for every `--help`, `doctor` and `plugins list`, so at module level it imports only the
plugin API and its own settings; the channel core and the adapters are imported inside the methods
(importing anything under localharness.channels runs channels/__init__.py, which pulls in
prompt_toolkit). start() opens no network: the gateway connects when the REPL starts the channel.
The legacy env variables are read in one place, `_effective`, through `env_fallback`."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from localharness.dispatch.config import DiscordSettings, DispatchConfig, env_fallback
from localharness.plugins.api import Availability, Check, Plugin, PluginContext, PluginManifest, SetupField

if TYPE_CHECKING:
    from localharness.channels.base import ChannelAdapter
    from localharness.core.bus import EventBus

NOT_CONFIGURED_HINT = ("localharness plugins enable dispatch --set discord.token=… "
                       "--set discord.allow=<your user id>")


def _effective(ctx: PluginContext) -> tuple[DiscordSettings, list[str]]:
    """The settings with the deprecated env sources folded in, and one line per deciding source."""
    cfg = ctx.config if isinstance(ctx.config, DispatchConfig) else DispatchConfig()
    return env_fallback(cfg.discord, os.environ, Path.home())


class DispatchPlugin(Plugin):
    """chat: Discord

    Allowlisted messages become turns; replies and permission asks go back to the chat."""

    manifest = PluginManifest(
        name="dispatch", version="0.1.0", kind="channel", enabled_by_default=True,
        requires_extra="dispatch", channels=("discord",),
        setup=(SetupField(key="discord.token", prompt="Discord bot token", secret=True),
               SetupField(key="discord.allow", prompt="Your Discord user id(s), comma-separated")),
        setup_help=(
            "In the Discord Developer Portal, open your bot and turn on the Message Content intent.\n"
            "Invite the bot to your server (OAuth2 > URL Generator, scope 'bot').\n"
            "In Discord, turn on Developer Mode (Settings > Advanced), then right-click your name > "
            "Copy User ID.\n"
            "Then: localharness start --channel discord"),
    )
    ConfigModel = DispatchConfig
    AgentConfigModel = None
    wants_artifacts = False

    def __init__(self) -> None:
        self.startup_warnings: list[str] = []
        self._settings: DiscordSettings | None = None
        self._state_dir: Path | None = None

    async def configure(self, ctx: PluginContext) -> Availability:
        """Always ready: an unconfigured Discord refuses at `start --channel discord`, never here,
        so a terminal session with the plugin on is unaffected."""
        return "ready"

    async def start(self, ctx: PluginContext) -> None:
        """Resolve the effective settings once and put each deprecation line on startup_warnings.
        No network."""
        self._settings, lines = _effective(ctx)
        self._state_dir = ctx.paths.state_dir
        self.startup_warnings.extend(lines)

    def channels(self) -> dict[str, type[ChannelAdapter]]:
        from localharness.dispatch.adapters import ADAPTERS
        from localharness.dispatch.channel import DispatchChannel
        return {name: DispatchChannel for name in ADAPTERS}

    def make_channel(self, name: str, bus: EventBus) -> ChannelAdapter:
        """The channel for `name`, built from the settings start() resolved."""
        from localharness.dispatch.adapters import ADAPTERS, load_adapter
        if name not in ADAPTERS:
            raise ValueError(f"dispatch has no {name!r} adapter (it has: {', '.join(ADAPTERS)})")
        if self._settings is None:
            from localharness.channels.errors import ChannelStartError
            raise ChannelStartError("the dispatch plugin was not started — no settings to build a channel")
        from localharness.dispatch.channel import DispatchChannel
        s = self._settings
        adapter: Any = load_adapter(name)(token=s.token.get_secret_value())
        return DispatchChannel(bus, {"adapter": adapter, "allow": set(s.allow), "channels": set(s.channels),
                                     "ack": s.ack, "state_dir": self._state_dir})

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """Configured or not, from the effective settings (doctor runs without start()); one warn row
        per deprecated env source. Never the token, and no login."""
        s, lines = _effective(ctx)
        token, allow = bool(s.token.get_secret_value()), bool(s.allow)
        if token and allow:
            where = f"{len(s.channels)} channel(s)" if s.channels else "any channel the bot can see"
            rows = [Check(name="dispatch", status="pass",
                          detail=f"Discord configured — {len(s.allow)} allowed user(s); listens in {where}")]
        else:
            missing = [k for k, ok in (("token", token), ("allow", allow)) if not ok]
            detail = "Discord not configured" + ("" if len(missing) == 2 else
                                                 f" — dispatch.discord.{missing[0]} is empty")
            rows = [Check(name="dispatch", status="skip", detail=detail, hint=NOT_CONFIGURED_HINT)]
        return rows + [Check(name="dispatch", status="warn", detail=line) for line in lines]
