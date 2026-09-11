"""`localharness acp` — serve the Agent Client Protocol on stdin/stdout (PRD §4).

This is the command a Zed `agent_servers` entry runs. It speaks JSON-RPC and nothing else: the
process's stdout IS the protocol channel, so a single stray `print`, rich banner or structlog
line on it corrupts the stream and the editor sees a dead agent with no error.

Guarding that is the whole job of this module beyond wiring the adapter, and it is done by
construction rather than by auditing every code path for prints (`_ProtocolStdout`): the harness
writes megabytes of startup output through `rich` consoles and structlog, all of which resolve
`sys.stdout` at call time, and a rule that says "remember not to print" is a rule that lasts
until the next commit.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Annotated, Any, Optional

import typer

log = logging.getLogger(__name__)

TEARDOWN_SIGNALS: tuple[int, ...] = tuple(
    s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)) if s is not None
)
"""The two signals an editor closing the agent panel can send.

Zed sends SIGTERM when the panel closes (and on quit); SIGINT is what a person typing Ctrl-C at
a hand-started agent server sends. Both used to kill the process outright — exit -15 — which
skipped `serve()`'s teardown entirely: MCP servers left running, memory consolidation and the WAL
checkpoint never run, the session's last minutes lost. Derived with `getattr` because the names
are POSIX; a platform missing one simply gets the other."""

SHUTDOWN_NOTICE = "acp: %s received — shutting the session down"
"""What the server log shows when a signal starts an orderly shutdown. It is also the line that
tells a user reading 'View Server Logs' that the exit was clean and not a crash."""


class _ProtocolStdout:
    """Text output goes to stderr; `.buffer` stays the real stdout (the JSON-RPC channel).

    Everything that prints in this process — `rich.Console` (which resolves `sys.stdout` on every
    write), `structlog`'s `PrintLogger` (which binds `sys.stdout` on first use), a bare `print` —
    goes through the TEXT layer and lands on stderr, where Zed shows it as agent-server logs.
    The ACP SDK's stdio transport writes bytes through `sys.stdout.buffer`
    (`acp/stdio.py:_StdoutTransport`), which this object leaves pointing at the real stdout.

    Everything not named here (`encoding`, `isatty`, `fileno`, `flush`, `writelines`, …)
    delegates to stderr, so anything probing the stream sees a real, writable text stream.
    """

    def __init__(self, real_stdout: Any, err: Any) -> None:
        # Instance attributes on purpose: they shadow __getattr__, so these are the ONLY things
        # that still reach the real stdout.
        self.buffer = real_stdout.buffer
        self._real = real_stdout
        self._err = err

    def fileno(self) -> int:
        """The REAL stdout's descriptor. `acp/stdio.py` builds its writer with
        `loop.connect_write_pipe(..., sys.stdout)` on POSIX, which resolves the stream by
        `fileno()` — return stderr's here and every JSON-RPC response goes to the log instead of
        the editor, which looks exactly like an agent that never answers."""
        return self._real.fileno()

    def isatty(self) -> bool:
        """False, always: this stream is a protocol channel. It also keeps `rich` from painting
        ANSI into the log output that rides on stderr."""
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._err, name)


def _protect_stdout() -> None:
    """Make stdout unwritable by accident (see :class:`_ProtocolStdout`) and pin logging to
    stderr, overriding any handler an earlier import installed (`force=True`)."""
    sys.stdout = _ProtocolStdout(sys.stdout, sys.stderr)  # type: ignore[assignment]
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(message)s",
        force=True,
    )


def _install_shutdown_handlers(loop: asyncio.AbstractEventLoop, task: asyncio.Task) -> None:
    """Turn :data:`TEARDOWN_SIGNALS` into a cancel of the protocol task (D3).

    Cancelling — rather than exiting — is the whole point: the cancel unwinds `run_agent`, and
    `_serve`'s `finally` then runs `aclose()`, which releases `serve()` and lets `_start_async`'s
    ordered teardown happen exactly as it does when stdin hits EOF. The default disposition kills
    the process mid-turn instead, which is what left MCP servers behind.

    `loop.add_signal_handler` is the correct primitive on POSIX (it wakes the loop). Windows has
    no implementation of it, so there the fallback is `signal.signal` with
    `call_soon_threadsafe`, which is the only safe way into a running loop from a C-level signal
    handler. A platform that refuses both is left with the default disposition rather than a
    crash at startup — a harness that will not START is worse than one that exits abruptly.
    """
    def _request_shutdown(signum: int) -> None:
        log.warning(SHUTDOWN_NOTICE, signal.Signals(signum).name)
        task.cancel()

    for signum in TEARDOWN_SIGNALS:
        try:
            loop.add_signal_handler(signum, _request_shutdown, signum)
        except (NotImplementedError, RuntimeError, ValueError, AttributeError, OSError):
            try:
                signal.signal(
                    signum,
                    lambda num, _frame: loop.call_soon_threadsafe(_request_shutdown, num),
                )
            except (ValueError, OSError):  # not the main thread, or not a real signal here
                log.warning("acp: no shutdown handler for %s on this platform", signum)


async def _serve(agent: Any) -> None:
    """Run the protocol until stdin closes or a signal arrives, then tear the session down."""
    import acp

    task = asyncio.ensure_future(acp.run_agent(agent))
    _install_shutdown_handlers(asyncio.get_running_loop(), task)
    try:
        await task
    except asyncio.CancelledError:
        # A signal asked for this. The exit is clean and the code is 0 — the teardown below is
        # the reason the signal was caught at all, and reporting a failure after running it
        # would misreport an orderly shutdown as a crash.
        pass
    finally:
        await agent.aclose()


def acp_cmd(
    config_dir: Annotated[
        Optional[str],
        typer.Option("--config-dir", help="Config directory (default: ~/.localharness)"),
    ] = None,
) -> None:
    """Run as an ACP agent server for Zed (see docs/zed.md)."""
    _protect_stdout()

    from localharness.channels.acp import LocalHarnessAcpAgent

    asyncio.run(_serve(LocalHarnessAcpAgent(config_dir=config_dir)))
