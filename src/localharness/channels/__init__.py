"""Channel system: ChannelAdapter ABC, TerminalChannel, and error types."""
from localharness.channels.base import ChannelAdapter
from localharness.channels.errors import (
    ChannelError,
    ChannelInputError,
    ChannelOutputError,
    ChannelStartError,
    NotInteractiveError,
)
from localharness.channels.discord import DiscordChannel, discord_config_from_env
from localharness.channels.terminal import TerminalChannel

__all__ = [
    "ChannelAdapter",
    "TerminalChannel",
    "DiscordChannel",
    "discord_config_from_env",
    "ChannelError",
    "ChannelStartError",
    "ChannelOutputError",
    "ChannelInputError",
    "NotInteractiveError",
]
