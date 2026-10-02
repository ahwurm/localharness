"""Channel system: ChannelAdapter ABC, TerminalChannel, and error types."""
from localharness.channels.base import ChannelAdapter
from localharness.channels.errors import (
    ChannelError,
    ChannelInputError,
    ChannelOutputError,
    ChannelStartError,
    NotInteractiveError,
)
from localharness.channels.terminal import TerminalChannel

__all__ = [
    "ChannelAdapter",
    "TerminalChannel",
    "ChannelError",
    "ChannelStartError",
    "ChannelOutputError",
    "ChannelInputError",
    "NotInteractiveError",
]
