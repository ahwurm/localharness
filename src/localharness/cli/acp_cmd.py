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
import sys
from typing import Annotated, Any, Optional

import typer


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


async def _serve(agent: Any) -> None:
    """Run the protocol until stdin closes, then tear the harness session down."""
    import acp

    try:
        await acp.run_agent(agent)
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
