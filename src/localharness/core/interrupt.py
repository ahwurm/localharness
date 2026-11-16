"""Per-step SIGINT (the first Ctrl-C cancels the step's task, the second hands it back to the host) and a private session's TERM/HUP guard."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
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


DEADLINE_S = 10.0
JOIN_S = 3.0


def _within(seconds: float, fn: Callable[[], None]) -> None:
    worker = threading.Thread(target=fn, daemon=True)
    worker.start()
    worker.join(seconds)


class Guard:
    """Ends a private session on TERM/HUP: cancel the REPL task, let teardown finish, else exit at the deadline.

    A handler never ends the run itself while the loop can: only the idle state and the deadline do, both on a helper thread with a bounded join."""

    def __init__(self, signums: list[int], on_idle: Callable[[], None]) -> None:
        self._signums, self._on_idle = signums, on_idle
        self._previous = {n: signal.getsignal(n) for n in signums}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._received: int | None = None
        self._timer: threading.Timer | None = None

    def _first(self, signum: int) -> None:
        if self._received is None:
            self._received = signum
            self._timer = threading.Timer(DEADLINE_S, self._expire)
            self._timer.daemon = True
            self._timer.start()

    def _expire(self) -> None:
        _within(JOIN_S, self._on_idle)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os._exit(128 + (self._received or 0))

    def _use(self, handler: Callable[[int], None]) -> None:
        for n in self._signums:
            self._loop.add_signal_handler(n, handler, n)

    def _drop_loop_handlers(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            for n in self._signums:
                self._loop.remove_signal_handler(n)
        self._loop = None

    def idle(self) -> None:
        self._drop_loop_handlers()
        for n in self._signums:
            signal.signal(n, self._on_idle_signal)

    def _on_idle_signal(self, signum: int, _frame: object) -> None:
        self._first(signum)
        _within(JOIN_S, self._on_idle)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def repl(self, task: asyncio.Task) -> None:
        self._loop = asyncio.get_running_loop()

        def cancel(signum: int) -> None:
            self._first(signum)
            self._loop.call_soon_threadsafe(lambda: task.done() or task.cancel())

        self._use(cancel)

    def teardown(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._use(self._first)

    def restore(self) -> None:
        self._drop_loop_handlers()
        for n, handler in self._previous.items():
            signal.signal(n, handler if handler is not None else signal.SIG_DFL)
        if self._received is not None:
            signal.signal(self._received, signal.SIG_DFL)
            os.kill(os.getpid(), self._received)


def install(signals: tuple[str, ...], on_idle: Callable[[], None]) -> Guard:
    """A guard for the named signals that exist here and are not ignored (nohup); it starts in the idle state."""
    signums = [n for n in (getattr(signal, name, None) for name in signals)
               if n is not None and signal.getsignal(n) is not signal.SIG_IGN]
    guard = Guard(signums, on_idle)
    guard.idle()
    return guard
