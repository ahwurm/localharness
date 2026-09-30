"""`/example` in a session — imported the first time someone types it."""
from localharness.plugins.api import PluginContext


async def run(ctx: PluginContext, args: str) -> str:
    """Show the swatch settings this session uses. `args` (the text after `/example`) is unused;
    the returned text is shown to the user."""
    return f"example plugin: swatches render in {ctx.config.color} at {ctx.agent_config.size}px"
