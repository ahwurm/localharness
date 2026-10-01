"""The web plugin: the phone app — `localharness web` serves the web UI and its event API.

On by default; needs the `web` install extra (starlette, uvicorn, ...). plugins/builtin.py imports this
module for every `--help`, `doctor` and `plugins list`, so it imports only the plugin API at module
level; the channel, the server and the token helpers are imported when a caller asks for them.
It lives in cli/, not channels/web/: importing anything under localharness.channels runs
channels/__init__.py (discord, prompt_toolkit) and channels/web/__init__.py (starlette), which would
cost every --help ~0.12 s and break the whole CLI without the extra. Phase 49 may move it once
channels/__init__.py is lazy. The server's life belongs to `localharness web`; start/stop are no-ops."""
from __future__ import annotations

from typing import TYPE_CHECKING

from localharness.plugins.api import CliDescriptor, Plugin, PluginManifest

if TYPE_CHECKING:
    from localharness.channels.base import ChannelAdapter


class WebPlugin(Plugin):
    """the phone app: `localharness web` serves the web UI and its event API"""

    manifest = PluginManifest(
        name="web", version="0.1.0", kind="channel", enabled_by_default=True, requires_extra="web",
        cli=(CliDescriptor(name="web", help="Serve the phone UI and its event API (see docs/web.md).",
                           target="localharness.cli.web_cmd:app"),),
    )
    ConfigModel = None  # no settings this phase: resolve() strips `enabled` before validating

    def channels(self) -> dict[str, type[ChannelAdapter]]:
        from localharness.channels.web.channel import WebChannel
        return {"web": WebChannel}
