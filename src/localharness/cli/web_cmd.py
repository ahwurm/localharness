"""`localharness web` — serve the event API and the PWA shell on a private network boundary.

The shape of this command is the web PRD's §6.0 hybrid, and the ORDER is the point:

    the HTTP server starts FIRST and is reachable before any session exists,

so a phone can open the app, read the pending queue and browse a past log with the model server
cold. Then the SSE connection itself — not the first prompt — starts session bring-up in the
background, because this is the one channel with a connect event the others lack, and by the time
a thumb has finished typing the session is usually already up.

When bring-up runs it calls `_start_async(channel_mode="web", web_channel=...)`, which hands the
already-built channel to `OrchestratorREPL`. That is Discord's drive loop, and taking it is what
earns the web channel slash commands, the input router, the pending resolver and `UserMessage`
publishing for free. An ACP-shaped web channel would have re-implemented the first three and
shipped a history with no user turns in it.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.markup import escape

log = logging.getLogger(__name__)
console = Console()

DEFAULT_PORT = 8765
"""An unassigned port in the registered range, above every service this box is known to run.

Named rather than sprinkled: `--port` overrides it, `doctor` reports the effective bind, and the
docs quote this one constant.
"""

MISSING_DEPENDENCY = (
    "the web channel needs its optional extra — install it with:\n"
    "    uv pip install 'localharness[web]'\n"
    "(or `uv sync --extra web` in a checkout). It pulls starlette and uvicorn, which are the "
    "ASGI app and the server behind `localharness web`."
)

BANNER = """LocalHarness web channel
  serving   http://{host}:{port}
  UI        {ui_dir}
  workspace {cwd}
  session   builds on first connect (or first message)

Put a TLS-terminating proxy in front of it before reaching it from a phone:
  tailscale serve --bg {port}            (recommended: auto-renewing certs, and the only
                                          topology that also offers tailnet identity)
Plain HTTP on a non-localhost origin works as a browser TAB only — no service worker, no
home-screen install, no Web Push. That is a browser rule, not a harness limitation.

Enrol a client with this token (it is required on every request, including the stream):
  {token}
"""

TOKEN_NEW_NOTE = "A new app token was generated on this first run and stored 0600 at {path}."

TOKEN_ROTATED = (
    "App token rotated. Every enrolled client is now invalid and must re-enrol with:\n  {token}\n"
    "Stored 0600 at {path}.\n"
    "Named gap: rotation is all-or-nothing — there is no per-device revoke."
)

REPLAY_BANNER = """LocalHarness web channel — REPLAY (no model server, no GPU, deterministic)
  serving   http://{host}:{port}
  log       {log}
  speed     {speed}x{fixtures}

Persisted surfaces replay for real: the transcript, tool calls and results, and the parked
queue. The live-progress frames are SYNTHESIZED and flagged as such, because no log contains
them. A BlockingAsk has no persisted analog at all, so it comes only from --fixtures.
"""


def web_cmd(
    config_dir: Annotated[
        Optional[str],
        typer.Option("--config-dir", envvar="LOCALHARNESS_DIR", show_default=False,
                     help="Config directory. Default: $LOCALHARNESS_DIR, else ~/.localharness."),
    ] = None,
    host: Annotated[str, typer.Option(
        "--host",
        help="Bind address. Loopback only unless --allow-unsafe-bind: this endpoint runs shell "
             "with your privileges, so a fronting proxy publishes it, not the app.",
    )] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Bind port.")] = DEFAULT_PORT,
    allow_unsafe_bind: Annotated[bool, typer.Option(
        "--allow-unsafe-bind",
        help="Permit a non-loopback bind. You are publishing a shell-running endpoint; put "
             "authentication and a network boundary in front of it yourself.",
    )] = False,
    ui_dir: Annotated[Optional[str], typer.Option(
        "--ui-dir",
        help="Serve the UI from this directory instead of the packaged reference page. Served "
             "LIVE — edit a file, pull to refresh, no restart and no build step.",
    )] = None,
    replay: Annotated[Optional[str], typer.Option(
        "--replay",
        help="Play a recorded session log (sessions/<id>.jsonl) as if it were live: real events, "
             "real ordering, zero GPU. Builds the UI with the box asleep.",
    )] = None,
    fixtures: Annotated[Optional[str], typer.Option(
        "--fixtures",
        help="Inject scripted frames during --replay (BlockingAsk, PermissionStaged, StatusTick) "
             "so the permission UI and instrument cluster can be built offline too.",
    )] = None,
    speed: Annotated[float, typer.Option("--speed", help="Replay speed multiplier.")] = 1.0,
    rotate_token: Annotated[bool, typer.Option(
        "--rotate-token",
        help="Mint a new app token, invalidating every enrolled client, and exit.",
    )] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed session view.")] = False,
    agent: Annotated[Optional[str], typer.Option("--agent", "-a", help="Start a specific agent.")] = None,
) -> None:
    """Serve the phone UI and its event API (see docs/web.md)."""
    from localharness.channels.web import auth as web_auth

    if rotate_token:
        token = web_auth.rotate_token(config_dir)
        console.print(escape(TOKEN_ROTATED.format(
            token=token, path=web_auth.token_path(config_dir),
        )), soft_wrap=True)
        raise typer.Exit(0)

    try:
        web_auth.check_bind(host, allow_unsafe=allow_unsafe_bind)
    except ValueError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
        raise typer.Exit(2) from exc

    try:
        import starlette  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:
        console.print(f"[red]{escape(MISSING_DEPENDENCY)}[/red]", soft_wrap=True)
        raise typer.Exit(1) from exc

    token, created = web_auth.load_or_create_token(config_dir)
    if created:
        console.print(escape(TOKEN_NEW_NOTE.format(path=web_auth.token_path(config_dir))),
                      style="dim", soft_wrap=True)

    try:
        asyncio.run(_serve(
            config_dir=config_dir, host=host, port=port, token=token, ui_dir=ui_dir,
            replay=replay, fixtures=fixtures, speed=speed, verbose=verbose, agent=agent,
        ))
    except KeyboardInterrupt:
        console.print("\nGoodbye.")


async def _serve(
    *,
    config_dir: Optional[str],
    host: str,
    port: int,
    token: str,
    ui_dir: Optional[str],
    replay: Optional[str],
    fixtures: Optional[str],
    speed: float,
    verbose: bool,
    agent: Optional[str],
) -> None:
    import uvicorn

    from localharness.channels.web.channel import WebChannel
    from localharness.channels.web.replay import ReplayDriver, ReplayFixtures
    from localharness.channels.web.server import PACKAGED_UI_DIR, WebServer
    from localharness.core.bus import EventBus

    resolved_ui = Path(ui_dir).expanduser().resolve() if ui_dir else PACKAGED_UI_DIR
    if not resolved_ui.is_dir():
        console.print(f"[red]--ui-dir is not a directory: {escape(str(resolved_ui))}[/red]",
                      soft_wrap=True)
        raise typer.Exit(2)

    driver: Any = None
    session_task: Optional[asyncio.Task] = None

    if replay is not None:
        # A bus with NO persist path: replay must never write into a real session log. The
        # channel's own fan-out is the only thing it drives.
        channel = WebChannel(bus=EventBus(), config={})
        await channel.start()
        try:
            log_path = ReplayDriver.resolve(replay)
        except ValueError as exc:
            console.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
            raise typer.Exit(2) from exc
        loaded = None
        if fixtures:
            try:
                loaded = ReplayFixtures.load(Path(fixtures).expanduser().resolve())
            except (OSError, ValueError) as exc:
                console.print(f"[red]{escape(str(exc))}[/red]", soft_wrap=True)
                raise typer.Exit(2) from exc
        driver = ReplayDriver(channel, log_path, speed=speed, fixtures=loaded)
        server = WebServer(channel, token=token, ui_dir=resolved_ui, replay=driver)
        console.print(escape(REPLAY_BANNER.format(
            host=host, port=port, log=log_path, speed=speed,
            fixtures=f"   fixtures  {fixtures}" if fixtures else "",
        )), soft_wrap=True)
    else:
        channel = WebChannel(bus=EventBus(), config={})

        def _begin_session() -> None:
            """Start session bring-up once, in the background, and never twice.

            Held in a closure rather than on the channel because the channel must not be able to
            start a session — it is the surface, and one process serves one LIVE session at a
            time (LOCKED for the whole milestone; §6.6's unlocked `history.jsonl` appends are
            what that rule keeps closed, not merely scheduling convenience).
            """
            nonlocal session_task
            if session_task is not None and not session_task.done():
                return
            session_task = asyncio.ensure_future(
                _bring_up(channel, config_dir=config_dir, verbose=verbose, agent=agent)
            )
            channel.set_bringup_abort(session_task.cancel)

        server = WebServer(
            channel, token=token, ui_dir=resolved_ui, on_first_message=_begin_session,
        )
        console.print(escape(BANNER.format(
            host=host, port=port, ui_dir=resolved_ui, cwd=Path.cwd(), token=token,
        )), soft_wrap=True)

    config = uvicorn.Config(
        server.app, host=host, port=port, log_level="warning", access_log=False,
        # The SSE stream is long-lived by design; uvicorn's default graceful shutdown would sit
        # waiting for every attached phone to hang up.
        timeout_graceful_shutdown=2,
    )
    # The replay is NOT started here: it starts when a client connects, so "edit the page,
    # pull to refresh, watch it again" replays from the beginning instead of joining a playback
    # that has been running since the server booted. Found by driving the real command.
    runner = uvicorn.Server(config)
    try:
        await runner.serve()
    finally:
        if driver is not None:
            await driver.stop()
        if session_task is not None and not session_task.done():
            session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session_task
        await channel.stop()


async def _bring_up(
    channel: Any, *, config_dir: Optional[str], verbose: bool, agent: Optional[str]
) -> None:
    """Build the session and hand the channel to the REPL, naming each stage as it goes.

    The stage names are WEBCH-43's whole point: a rising number reports elapsed time, not health,
    and a wedged memory lock, a failing MCP server and a sibling process holding the inference
    `flock` all look identical to a healthy slow start. So the screen says which stage it is in,
    and the build has its own abort — cancelling a TURN is a different state machine and there
    is otherwise no way out of a build that stuck.
    """
    import time

    from localharness.cli.start_cmd import _start_async

    started = time.monotonic()
    channel.set_bringup("starting the session", elapsed=0.0)
    try:
        await _start_async(
            agent, verbose, False, config_dir,
            channel_mode="web", web_channel=channel,
        )
    except asyncio.CancelledError:
        channel.set_bringup("cancelled", detail="the build was abandoned from the client.",
                            failed=True, elapsed=time.monotonic() - started)
        raise
    except Exception as exc:  # noqa: BLE001 — a failed build reports itself; it never kills the server
        log.warning("web session bring-up failed", exc_info=True)
        channel.set_bringup(
            "failed", detail=str(exc), failed=True, elapsed=time.monotonic() - started,
        )
    else:
        channel.set_bringup("ready", elapsed=time.monotonic() - started)
