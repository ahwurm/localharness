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

from pydantic import BaseModel, ConfigDict, Field, field_validator

from localharness.plugins.api import (
    GLOBAL_ONLY, Check, CliDescriptor, Plugin, PluginContext, PluginManifest, SetupField,
)

if TYPE_CHECKING:
    from localharness.channels.base import ChannelAdapter

WEB_DEFAULT_PORT = 8765
"""An unassigned port in the registered range, above every service this box is known to run.
The one literal: cli/web_cmd imports it as DEFAULT_PORT (`--port` overrides it) and doctor reports
it. It lives here, not in web_cmd, because importing web_cmd from the plugin would put
cli/start_cmd.py on the plugin's import chain (PAPI-03)."""


class WebConfig(BaseModel):
    """The `web:` section: one setting. public_url is machine-level only — it is the address the
    pairing QR sends a phone to, so a project folder must never choose where a phone is pointed. The
    bind address and --allow-unsafe-bind are flags of `localharness web`, never settings: a saved
    unsafe bind would turn a deliberate per-run choice into a standing one (SECURITY.md)."""

    model_config = ConfigDict(extra="forbid")
    public_url: str = Field("", json_schema_extra=GLOBAL_ONLY,
                            description="The address your phone opens this machine at, e.g. "
                                        "https://yourbox.your-tailnet.ts.net. Empty: `localharness web` "
                                        "guesses it from Tailscale.")

    @field_validator("public_url")
    @classmethod
    def _clean_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("web.public_url must start with http:// or https://")
        return v


class WebPlugin(Plugin):
    """Mobile: the phone page, served with its event API by `localharness web`"""

    manifest = PluginManifest(
        name="web", version="0.1.0", kind="channel", enabled_by_default=True, requires_extra="web",
        cli=(CliDescriptor(name="web", help="Serve the phone UI and its event API (see docs/web.md).",
                           target="localharness.cli.web_cmd:app"),),
        setup=(SetupField(key="public_url",
                          prompt="Phone address, the URL your phone opens (Enter: `localharness web` guesses it)"),),
        next_steps="Run `localharness web`, then scan its pairing QR with your phone.",
        agent_prompt=(
            "Set up the LocalHarness phone app on this machine. Install LocalHarness with its web extra,\n"
            "keeping the extras I already use. Run `localharness web` and leave it running: it serves the\n"
            "page on this machine only, so do not pass --allow-unsafe-bind. To reach it from my phone,\n"
            f"put it behind a private network I already use, for example `tailscale serve --bg {WEB_DEFAULT_PORT}`,\n"
            "rather than opening a port to the internet. You are done when `localharness doctor` shows\n"
            "web enrolled and my phone has scanned the pairing QR."),
    )
    ConfigModel = WebConfig

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
