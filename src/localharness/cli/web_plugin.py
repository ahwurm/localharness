"""The web plugin: the phone app — `localharness web` serves the web UI and its event API.

On by default; needs the `web` install extra (starlette, uvicorn, ...). plugins/builtin.py imports this
module for every `--help`, `doctor` and `plugins list`, so it imports only the plugin API at module
level; the channel, the server and the token helpers are imported when a caller asks for them.
It lives in cli/, not channels/web/: importing anything under localharness.channels runs
channels/__init__.py (prompt_toolkit, through the terminal channel) and channels/web/__init__.py
(starlette), which would cost every --help ~0.12 s and break the whole CLI without the extra; moving
this file is deferred. The server's life belongs to `localharness web`; start/stop are no-ops."""
from __future__ import annotations

from typing import TYPE_CHECKING

from localharness.plugins.api import Check, CliDescriptor, Plugin, PluginContext, PluginManifest

if TYPE_CHECKING:
    from localharness.channels.base import ChannelAdapter

WEB_DEFAULT_PORT = 8765
"""An unassigned port in the registered range, above every service this box is known to run.
The one literal: cli/web_cmd imports it as DEFAULT_PORT (`--port` overrides it) and doctor reports
it. It lives here, not in web_cmd, because importing web_cmd from the plugin would put
cli/start_cmd.py on the plugin's import chain (PAPI-03)."""


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

    def doctor(self, ctx: PluginContext) -> list[Check]:
        """Enrolment, the bind the server enforces, and the token file's mode — never the token, and
        no live-port probe (the server is a separate command and may be down on purpose). The address
        comes from the constants the server enforces (the port pinned equal by a test) (WEBCH-13)."""
        from localharness.channels.web.auth import LOOPBACK_HOSTS, token_path

        path = token_path(ctx.paths.global_config_dir)
        if not path.exists():
            return [Check(name="web", status="skip", detail="not enrolled yet",
                          hint="`localharness web` generates its app token on first run")]
        rows = [Check(name="web", status="pass",
                      detail=f"enrolled; binds {sorted(LOOPBACK_HOSTS)[0]}:{WEB_DEFAULT_PORT} (loopback only "
                             f"unless --allow-unsafe-bind). A token is required on every request. "
                             f"Token file: {path}")]
        mode = path.stat().st_mode & 0o777
        rows.append(Check(name="web-token", status="pass", detail="token file is mode 600") if mode == 0o600
                    else Check(name="web-token", status="fail",
                               detail=f"Web app token is mode {mode:o}, expected 600 — anyone who can "
                                      f"read it can drive your agent.", hint=f"chmod 600 {path}"))
        return rows
