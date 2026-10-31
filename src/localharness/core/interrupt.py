"""Per-step SIGINT: the first Ctrl-C cancels the step's task, the second hands SIGINT back to the host."""
from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager


@asynccontextmanager
async def sigint_cancels(
    task: asyncio.Task, *, on_stopping: Callable[[], None] | None = None,
    cancel: Callable[[], None] | None = None,
) -> AsyncIterator[Callable[[], bool]]:
    """Yields `cancelled()`: true in the awaiting task's `except CancelledError` only for a user stop."""
    previous = signal.getsignal(signal.SIGINT)
    if previous is signal.SIG_IGN:
        yield lambda: False
        return
    loop = asyncio.get_running_loop()
    baseline = asyncio.current_task().cancelling()
    by_user, timer = False, None

    def _restore() -> None:
        try:
            loop.remove_signal_handler(signal.SIGINT)
            # never Python's default: under `localharness mobile` previous is uvicorn's shutdown
            if previous is not None:
                signal.signal(signal.SIGINT, previous)
        except (NotImplementedError, RuntimeError, ValueError, TypeError):
            pass

    def _on_sigint() -> None:
        nonlocal by_user, timer
        if by_user:
            _restore()
            return
        by_user = True
        (cancel or task.cancel)()
        if on_stopping is not None:
            timer = loop.call_later(1, on_stopping)

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False
    try:
        yield lambda: by_user and asyncio.current_task().cancelling() == baseline
    finally:
        if timer is not None:
            timer.cancel()
        if installed:
            _restore()
