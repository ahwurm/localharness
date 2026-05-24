"""EventBus wrapper providing publish/subscribe/unsubscribe/replay/wait_for/history.

bubus 1.5.6 requires events to inherit from bubus.BaseEvent. Our events use plain Pydantic
BaseModel with frozen=True for immutability. Because bubus.BaseEvent uses validate_assignment=True
which conflicts with frozen=True (bubus writes event_processed_at post-dispatch), we implement
our own subscriber registry and JSONL persistence via anyio directly.

JSONL persistence: O_APPEND writes under PIPE_BUF (4096 bytes) are atomic on POSIX.
All event records are well under this limit.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

import anyio
import structlog

from .events import AnyEvent, BaseEvent, deserialize_event
from .types import AgentID, EventSeq, SessionID

log = structlog.get_logger(__name__)

E = TypeVar("E", bound=BaseEvent)
AsyncHandler = Callable[[Any], Coroutine[Any, Any, None]]


class SubscriptionHandle:
    """Opaque handle returned by EventBus.subscribe(). Pass to unsubscribe() to cancel."""

    def __init__(self, event_key: str, wrapped_handler: AsyncHandler) -> None:
        self._event_key = event_key
        self._wrapped_handler = wrapped_handler


class EventBus:
    """
    Central ordered event stream for LocalHarness.

    All components communicate exclusively through this bus. No component holds
    a reference to any other component.

    Thread safety: all public methods are async-safe within a single asyncio event loop.
    """

    def __init__(
        self,
        persist_path: Optional[Path] = None,
        *,
        replay_on_start: bool = False,
        handler_timeout_seconds: float = 30.0,
    ) -> None:
        self._next_seq: int = 0
        self._seq_lock: asyncio.Lock = asyncio.Lock()
        self._history: list[AnyEvent] = []
        self._max_history: int = 10_000
        self._handler_timeout: float = handler_timeout_seconds
        # _subscriptions: event_key -> list of wrapped async handlers
        self._subscriptions: dict[str, list[AsyncHandler]] = {}
        # persist_path for JSONL
        if persist_path is not None:
            self._persist_path: Optional[Path] = Path(persist_path).expanduser()
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._session_dir: Optional[Path] = self._persist_path.parent / "sessions"
        else:
            self._persist_path = None
            self._session_dir = None

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def publish(self, event: BaseEvent) -> AnyEvent:
        """Assign seq, persist, deliver to subscribers. Returns sequenced event."""
        if event.seq is not None:
            raise ValueError(f"Event already published: seq={event.seq}")

        async with self._seq_lock:
            seq = EventSeq(self._next_seq)
            self._next_seq += 1

        sequenced: AnyEvent = event.model_copy(update={"seq": seq})  # type: ignore[assignment]

        # Append to in-memory history
        if len(self._history) < self._max_history:
            self._history.append(sequenced)

        # Persist to JSONL
        if self._persist_path is not None:
            await self._append_jsonl(self._persist_path, sequenced)
            if sequenced.session_id is not None and self._session_dir is not None:
                session_file = self._session_dir / f"{sequenced.session_id}.jsonl"
                self._session_dir.mkdir(parents=True, exist_ok=True)
                await self._append_jsonl(session_file, sequenced)

        # Deliver to subscribers
        event_key = type(sequenced).__name__
        await self._deliver(event_key, sequenced)

        return sequenced

    async def _append_jsonl(self, path: Path, event: BaseEvent) -> None:
        line = event.model_dump_json() + "\n"
        try:
            async with await anyio.open_file(str(path), "a", encoding="utf-8") as f:
                await f.write(line)
        except Exception as exc:
            log.error("persist_failed", event_id=event.id, seq=event.seq, path=str(path), error=str(exc))

    async def _deliver(self, event_key: str, event: AnyEvent) -> None:
        """Deliver event to all matching subscribers. Exceptions are isolated."""
        handlers = list(self._subscriptions.get(event_key, []))
        for handler in handlers:
            try:
                await asyncio.wait_for(handler(event), timeout=self._handler_timeout)
            except asyncio.TimeoutError:
                log.warning("handler_timeout", event_key=event_key, seq=event.seq)
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception:
                log.exception("subscriber_error", event_type=event_key, seq=event.seq)

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------

    def on(
        self,
        event_type: Type[E],
        *,
        agent_id: Optional[AgentID] = None,
        session_id: Optional[SessionID] = None,
    ) -> Callable[[AsyncHandler], AsyncHandler]:
        """Decorator that registers an async handler for a specific event type."""

        def decorator(handler: AsyncHandler) -> AsyncHandler:
            self.subscribe(event_type, handler, agent_id=agent_id, session_id=session_id)
            return handler

        return decorator

    def subscribe(
        self,
        event_type: Type[E],
        handler: AsyncHandler,
        *,
        agent_id: Optional[AgentID] = None,
        session_id: Optional[SessionID] = None,
    ) -> SubscriptionHandle:
        """Programmatic subscription. Returns a handle for unsubscription."""
        event_key = event_type.__name__

        async def filtered(event: AnyEvent) -> None:
            if agent_id is not None and getattr(event, "agent_id", None) != agent_id:
                return
            if session_id is not None and getattr(event, "session_id", None) != session_id:
                return
            await handler(event)

        self._subscriptions.setdefault(event_key, []).append(filtered)
        return SubscriptionHandle(event_key, filtered)

    def unsubscribe(self, handle: SubscriptionHandle) -> None:
        """Remove subscription. Safe to call during event delivery."""
        handlers = self._subscriptions.get(handle._event_key, [])
        try:
            handlers.remove(handle._wrapped_handler)
        except ValueError:
            pass  # already removed — idempotent

    # ------------------------------------------------------------------
    # wait_for
    # ------------------------------------------------------------------

    async def wait_for(
        self,
        event_type: Type[E],
        *,
        timeout: float = 30.0,
        predicate: Optional[Callable[[E], bool]] = None,
        agent_id: Optional[AgentID] = None,
        session_id: Optional[SessionID] = None,
    ) -> E:
        """Await the next matching event. Raises asyncio.TimeoutError if not found in time."""
        result_event: asyncio.Future[E] = asyncio.get_event_loop().create_future()

        async def one_shot(event: E) -> None:
            if result_event.done():
                return
            if predicate is not None and not predicate(event):
                return
            result_event.set_result(event)

        handle = self.subscribe(event_type, one_shot, agent_id=agent_id, session_id=session_id)  # type: ignore[arg-type]
        try:
            return await asyncio.wait_for(asyncio.shield(result_event), timeout=timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            self.unsubscribe(handle)

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    async def replay(
        self,
        *,
        session_id: Optional[SessionID] = None,
        from_seq: Optional[EventSeq] = None,
        to_seq: Optional[EventSeq] = None,
        event_types: Optional[list[Type[BaseEvent]]] = None,
    ) -> AsyncIterator[AnyEvent]:
        """Replay events from JSONL file. Yields events in seq order. Skips corrupt lines."""
        if self._persist_path is None:
            raise RuntimeError("replay() requires persist_path to be set at construction")
        if not self._persist_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self._persist_path}")

        async with await anyio.open_file(str(self._persist_path), "r", encoding="utf-8") as f:
            async for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = deserialize_event(line)
                except (json.JSONDecodeError, ValueError):
                    continue  # skip partial/corrupt lines
                if session_id is not None and event.session_id != session_id:
                    continue
                if from_seq is not None and (event.seq is None or event.seq < from_seq):
                    continue
                if to_seq is not None and (event.seq is None or event.seq > to_seq):
                    continue
                if event_types is not None and not any(isinstance(event, t) for t in event_types):
                    continue
                yield event

    async def replay_and_resubmit(
        self,
        *,
        session_id: SessionID,
        from_seq: Optional[EventSeq] = None,
    ) -> None:
        """Replay events from JSONL and republish to current subscribers (crash recovery)."""
        if self._persist_path is None:
            raise RuntimeError("replay_and_resubmit() requires persist_path")
        async for event in self.replay(session_id=session_id, from_seq=from_seq):
            event_key = type(event).__name__
            await self._deliver(event_key, event)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def history(
        self,
        *,
        session_id: Optional[SessionID] = None,
        limit: Optional[int] = None,
        event_types: Optional[list[Type[BaseEvent]]] = None,
    ) -> list[AnyEvent]:
        """Return in-memory event history. Sorted by seq."""
        result = list(self._history)
        if session_id is not None:
            result = [e for e in result if e.session_id == session_id]
        if event_types is not None:
            result = [e for e in result if any(isinstance(e, t) for t in event_types)]
        if limit is not None:
            result = result[-limit:]
        return result

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Total events published since construction."""
        return len(self._history)

    @property
    def subscriber_count(self) -> int:
        """Number of active subscriptions."""
        return sum(len(handlers) for handlers in self._subscriptions.values())
