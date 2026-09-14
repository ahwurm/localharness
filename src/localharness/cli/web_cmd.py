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
import io
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
Plain HTTP on a non-localhost origin does not work at all: no service worker (so no
home-screen install and no Web Push), and the event stream cannot authenticate, because a
browser refuses to keep the Secure session cookie there. Both are browser rules. localhost
is exempt, which is why the development loop needs none of this.

Enrol a client with this token (it is required on every request, including the stream):
  {token}
"""

TOKEN_NEW_NOTE = "A new app token was generated on this first run and stored 0600 at {path}."

TOKEN_ROTATED = (
    "App token rotated. Every enrolled client is now invalid and must re-enrol with:\n  {token}\n"
    "Stored 0600 at {path}.\n"
    "Named gap: rotation is all-or-nothing — there is no per-device revoke."
)

PUSH_UNAVAILABLE = (
    "Web Push is unavailable (its `web` extra packages are missing or the VAPID key could not "
    "be written). Everything else works; the phone just will not buzz."
)

QR_ERROR_LEVEL = "l"
"""Error correction for the enrolment QR: the lowest, on purpose.

The higher levels exist for a code printed on a box that will be scratched, photographed at an
angle or faded by sunlight. This one is on a screen, forty centimetres from a phone, for about
four seconds — so the redundancy buys nothing and costs modules, and modules are terminal rows.
"""

TAILSCALE_PROBE_TIMEOUT_S = 2.0
"""Budget for the `tailscale status` call that guesses the phone-reachable URL. Bounded because
this runs before the server starts serving: a wedged CLI must not hold up start-up, and the
answer is a convenience — `--public-url` is the authoritative one."""

ENROLMENT_HEADER = """Pair a phone: scan this with the camera (it carries the URL and the token).
  {url}
"""

ENROLMENT_LOOPBACK_NOTE = (
    "That is a LOOPBACK url, which no phone can reach. Publish the port, then re-run with the "
    "address the phone will use:\n"
    "    tailscale serve --bg {port}\n"
    "    localharness web --public-url https://<your-machine>.<your-tailnet>.ts.net"
)

ENROLMENT_GUESSED_NOTE = (
    "Address guessed from `tailscale status`. It is right only if something is publishing this "
    "port on 443 (`tailscale serve --bg {port}`). Pass --public-url to say it exactly."
)

ENROLMENT_NO_QR = (
    "No QR: the `segno` package is missing (it ships with the `web` extra). Open the URL above "
    "on the phone by hand — the part after the # is the token, and it never reaches the server."
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
    public_url: Annotated[Optional[str], typer.Option(
        "--public-url",
        help="The URL a PHONE reaches this box at (e.g. https://spark.tail1234.ts.net). Used for "
             "the enrolment QR. Auto-detected from `tailscale status` when omitted.",
    )] = None,
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
        print_enrolment(token, public_url=public_url, host=host, port=port)
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
            public_url=public_url,
        ))
    except KeyboardInterrupt:
        console.print("\nGoodbye.")


def detect_public_url(port: int, *, runner: Any = None) -> Optional[str]:
    """Best-effort guess at the URL a phone reaches this box at, from `tailscale status --json`.

    A guess, and labelled as one wherever it is printed. Tailscale is the SUPPORTED topology,
    not a requirement (§7.1) — plain LAN and any reverse proxy are legitimate — so this never
    fails a start-up and never overrides `--public-url`. It exists because the alternative for a
    first-time user is looking up their own MagicDNS name before they can pair a phone.
    """
    import json as _json
    import subprocess

    run = runner or (lambda cmd: subprocess.run(
        cmd, capture_output=True, text=True, timeout=TAILSCALE_PROBE_TIMEOUT_S, check=False))
    try:
        result = run(["tailscale", "status", "--json"])
        if getattr(result, "returncode", 1) != 0:
            return None
        name = (_json.loads(result.stdout).get("Self") or {}).get("DNSName") or ""
    except Exception:  # noqa: BLE001 — no tailscale, no network, bad JSON: all just "no guess"
        return None
    name = name.rstrip(".")
    return f"https://{name}" if name else None


def enrolment_url(token: str, *, public_url: Optional[str], host: str, port: int) -> tuple[str, str]:
    """The URL the QR encodes, and which KIND of address it turned out to be.

    The token rides in the FRAGMENT. A fragment is never sent to a server, so it lands in no
    access log, no proxy log and no `Referer` — which is what §7.3's "never put the token in a
    URL" is actually protecting. The page reads it once and erases it from the address bar.
    """
    kind = "given"
    base = public_url
    if not base:
        base = detect_public_url(port)
        kind = "guessed" if base else "loopback"
    if not base:
        base = f"http://{host}:{port}"
    return f"{base.rstrip('/')}/#t={token}", kind


def render_qr(url: str) -> Optional[str]:
    """The QR as terminal art, or None when `segno` is not installed.

    Half-block compaction keeps it to about 21 rows for an enrolment URL, which fits a terminal
    nobody has resized. Degrading to None rather than raising is the point: a missing optional
    package must cost you a convenience, not the command.
    """
    try:
        import segno
    except ImportError:
        return None
    buffer = io.StringIO()
    segno.make(url, error=QR_ERROR_LEVEL).terminal(buffer, compact=True)
    return buffer.getvalue().rstrip("\n")


def print_enrolment(token: str, *, public_url: Optional[str], host: str, port: int) -> None:
    """Print the pairing QR and its URL. Nobody hand-types a 256-bit secret into a phone."""
    url, kind = enrolment_url(token, public_url=public_url, host=host, port=port)
    console.print(escape(ENROLMENT_HEADER.format(url=url)), soft_wrap=True)
    art = render_qr(url)
    if art is None:
        console.print(escape(ENROLMENT_NO_QR), style="dim", soft_wrap=True)
    else:
        # No markup, no highlighting and no wrapping: this is a picture made of block
        # characters, and Rich reflowing it would turn it into a QR that does not scan.
        console.print(art, markup=False, highlight=False, soft_wrap=True)
    if kind == "loopback":
        console.print(escape(ENROLMENT_LOOPBACK_NOTE.format(port=port)), style="dim",
                      soft_wrap=True)
    elif kind == "guessed":
        console.print(escape(ENROLMENT_GUESSED_NOTE.format(port=port)), style="dim",
                      soft_wrap=True)


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
    public_url: Optional[str] = None,
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
        # The real builtin registry, no session required: without it `/api/tools` is empty in
        # replay, every group renders as "unknown", and the condensed view a client is being
        # DEVELOPED against never groups a thing — the one rendering the replay loop exists to
        # exercise. Built-ins only; a live session may add more, and that difference is honest.
        from localharness.tools.builtin import register_builtin_tools
        from localharness.tools.registry import ToolRegistry

        registry = ToolRegistry()
        await register_builtin_tools(registry)
        channel._tool_registry = registry
        server = WebServer(channel, token=token, ui_dir=resolved_ui, replay=driver,
                           config_dir=config_dir)
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

        async def _new_session() -> None:
            """The + button's other half: end the live session, then begin a fresh one.

            Cancellation is the mechanism — the session lives inside `session_task` — and
            `_bring_up` tells the two cancellations apart: a live session ending on purpose
            publishes "ended", only a genuinely abandoned BUILD publishes the red "cancelled".
            The old session's log stays on disk, which is what keeps it in the drawer.
            """
            nonlocal session_task
            if session_task is not None and not session_task.done():
                session_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await session_task
            channel.reset_session()
            _begin_session()

        server = WebServer(
            channel, token=token, ui_dir=resolved_ui, on_first_message=_begin_session,
            on_new_session=_new_session, config_dir=config_dir,
        )
        # Web Push, on the live path only. A `--replay` run must never buzz a phone about a
        # session that finished last week.
        try:
            from localharness.channels.web.push import PushService

            channel.set_push(PushService.build(config_dir))
        except Exception:  # noqa: BLE001 — no push is a missing convenience, not a failed start
            console.print(escape(PUSH_UNAVAILABLE), style="dim", soft_wrap=True)

        console.print(escape(BANNER.format(
            host=host, port=port, ui_dir=resolved_ui, cwd=Path.cwd(), token=token,
        )), soft_wrap=True)
        print_enrolment(token, public_url=public_url, host=host, port=port)

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
    """Build the session and hand the channel to the REPL, reporting the build to the client.

    What WEBCH-43 actually delivers here is the ABORT: cancelling a TURN is a different state
    machine, so without this there is no way out of a build that stuck, and the phone's only
    option is to kill the app. That half is real — `set_bringup` publishes an abortable stage and
    the client's button reaches it.

    The stage NAMES are not yet. There is exactly one non-terminal name, "starting the session",
    covering the whole of `_start_async`; the other three are its terminals. So a wedged memory
    lock, a failing MCP server and a sibling process holding the inference `flock` still look
    identical from the client — the screen says the build is running and can be abandoned, not
    which part of it is slow. Naming the wedges means instrumenting `_start_async` itself, and
    that is the fast-follow this docstring is not allowed to pretend already happened.
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
        # Two very different cancels share this except: a bound session means the NEW-CHAT verb
        # ended a live session on purpose (quiet, not red); no session yet means the build
        # itself was abandoned (the give-up button's job, and worth the red row).
        if channel.session_id is not None:
            channel.set_bringup("ended", detail="this chat was closed for a new one.",
                                elapsed=time.monotonic() - started)
        else:
            channel.set_bringup("cancelled", detail="the build was abandoned from the client.",
                                failed=True, elapsed=time.monotonic() - started)
        raise
    except Exception as exc:  # noqa: BLE001 — a failed build reports itself; it never kills the server
        log.warning("web session bring-up failed", exc_info=True)
        channel.set_bringup(
            "failed", detail=str(exc), failed=True, elapsed=time.monotonic() - started,
        )
    else:
        # A clean return means the session is OVER (the task runs its whole life). "ready" is
        # published where readiness actually begins — bind_runtime — not here.
        channel.set_bringup("ended", elapsed=time.monotonic() - started)
