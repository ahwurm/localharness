"""Channel error types."""


class ChannelError(Exception):
    """Base class for channel errors."""


class ChannelStartError(ChannelError):
    """Channel failed to start (resource unavailable, bus subscription failed)."""


class ChannelOutputError(ChannelError):
    """Write to channel output failed (stdout closed, socket disconnected, etc.)."""

    def __init__(self, channel_id: str, underlying: Exception) -> None:
        super().__init__(f"Output error on channel {channel_id!r}: {underlying}")
        self.channel_id = channel_id
        self.underlying = underlying


class ChannelInputError(ChannelError):
    """Read from channel input failed."""

    def __init__(self, channel_id: str, underlying: Exception) -> None:
        super().__init__(f"Input error on channel {channel_id!r}: {underlying}")
        self.channel_id = channel_id
        self.underlying = underlying


class NotInteractiveError(ChannelError):
    """Raised by read_input() on non-interactive channels."""

    def __init__(self, channel_id: str) -> None:
        super().__init__(f"Channel {channel_id!r} does not support interactive input")
        self.channel_id = channel_id
