"""`--replay` and `--fixtures`: build the UI with the GPU cold and the model server down (WIN-B).

`bus.replay()` already exists, so replaying a session is nearly free. **Replaying it RAW is half
a promise**, and that is the part this module exists for: a persisted log contains none of the
SSE-only frames, which are exactly the interactive parts a UI author most needs to iterate on.
Play a log back untouched and you get a silent, non-streaming, permission-free session.

So replay does two more things:

1. **Synthesizes the progress frames.** Each `llm_response` `Action.content` is re-emitted as a
   `TokenDelta` stream BEFORE the Action itself, so the provisional-supersede path of §4.2.1 —
   the client logic most likely to be wrong — is exercised offline. `StatusTick` is likewise
   synthesized from persisted timestamps. Both are PLAUSIBLE rather than real, and the replay
   `Hello` carries `synthetic: true` so nobody mistakes a replayed tok/s for a measurement.
2. **Takes a fixtures file.** Scripted `BlockingAsk` / `PermissionStaged` / `StatusTick` frames
   injected at chosen points, so the permission UI, the pending queue and the instrument cluster
   can be built with the box asleep. Answers posted back against a fixture resolve locally.

**What replay still cannot give you, stated plainly:** a blocking ask that never happened. It has
no persisted analog — `PermissionAsked` is payload-free by design — so `BlockingAsk` comes only
from fixtures, never from a log. The parked queue, by contrast, replays for real:
`PermissionStaged`/`PermissionResolved` are ordinary bus events carrying the whole `PendingCall`.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import structlog

from .channel import WebChannel
from .protocol import FRAME_TYPES, StatusTick, TokenDelta, WireFrame

log = structlog.get_logger(__name__)

MAX_STEP_S = 2.0
"""The longest pause replay will honour between two persisted events.

Real gaps between turns run to minutes (the measured median between turns in a multi-turn chat
is 3.7 minutes) and nobody iterating on a stylesheet wants to wait them out. Capped rather than
dropped, so ordering and the FEEL of a pause survive while the dead time does not.
"""

TOKEN_CHUNK_CHARS = 12
"""How much text one synthesized `TokenDelta` carries.

Sized to look like a real decode rather than to be fast: a local model emits a handful of
characters per chunk, and a synthetic stream that arrives in two enormous pieces would not
exercise the incremental-render path this exists to exercise.
"""

SYNTHETIC_TPS = 30.0
"""The tok/s a synthesized `StatusTick` reports.

The bottom of the terminal's GREEN band (`provider/speed_stats.py`), so a replayed session looks
healthy rather than alarming — and it is flagged `synthetic` on every frame besides, because a
replayed rate is not a measurement and a UI that treats it as one is reading a fiction.
"""

REPLAY_NOT_A_LOG = (
    "--replay takes a session log (sessions/<id>.jsonl written by the event bus), not an "
    "arbitrary file: {path}"
)


class ReplayFixtures:
    """Scripted frames injected at chosen points, plus the asks they make answerable.

    File shape::

        {"frames": [{"after_seq": 42, "frame": {"frame_type": "BlockingAsk", ...}}, ...]}

    `after_seq` is the persisted `seq` this frame follows; a frame with no `after_seq` is emitted
    at connect. Unknown `frame_type` values are refused loudly rather than skipped — a typo in a
    fixture file that silently produces nothing is a morning lost.
    """

    def __init__(self, entries: list[tuple[Optional[int], WireFrame]]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: Path) -> "ReplayFixtures":
        by_name = {f.__name__: f for f in FRAME_TYPES}
        data = json.loads(path.read_text(encoding="utf-8"))
        entries: list[tuple[Optional[int], WireFrame]] = []
        for item in data.get("frames") or []:
            spec = dict(item.get("frame") or {})
            name = spec.get("frame_type")
            model = by_name.get(name or "")
            if model is None:
                raise ValueError(
                    f"unknown frame_type {name!r} in {path}; known frames: "
                    + ", ".join(sorted(by_name))
                )
            entries.append((item.get("after_seq"), model.model_validate(spec)))
        return cls(entries)

    def at(self, seq: Optional[int]) -> list[WireFrame]:
        return [frame for after, frame in self.entries if after == seq]


class ReplayDriver:
    """Plays one persisted session at the channel, as if it were live.

    It pushes through the SAME fan-out a live session uses, so the SSE loop, the seam de-dup and
    the client's reducer are all exercised by the identical code path. A replay that took a side
    door would test a thing nobody runs.
    """

    def __init__(
        self,
        channel: WebChannel,
        path: Path,
        *,
        speed: float = 1.0,
        fixtures: Optional[ReplayFixtures] = None,
    ) -> None:
        self.channel = channel
        self.path = path
        self.speed = max(speed, 0.01)
        self.fixtures = fixtures
        self._task: Optional[asyncio.Task] = None
        # Named at construction, not when playback starts: the `Hello` frame goes out on connect
        # and playback starts a moment later, so a session id set in `run()` would reach the
        # client as null on the one frame that is supposed to tell it which session this is.
        self.channel.session_id = self.channel.session_id or _session_id_of(path)

    @staticmethod
    def resolve(raw: str) -> Path:
        """Realpath-resolve the log and refuse anything that is not one (WEBCH-40).

        `--replay` reads only a session log, never an arbitrary file rendered into a page. The
        check is on the resolved path and on the content shape, not on the string the user typed,
        because `../../secrets` and a symlink both look fine as strings.
        """
        path = Path(raw).expanduser().resolve()
        if not path.is_file() or path.suffix != ".jsonl":
            raise ValueError(REPLAY_NOT_A_LOG.format(path=path))
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    probe = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(REPLAY_NOT_A_LOG.format(path=path)) from exc
                if not isinstance(probe, dict) or "event_type" not in probe:
                    raise ValueError(REPLAY_NOT_A_LOG.format(path=path))
                break
        return path

    def start(self) -> None:
        """Begin (or re-begin) playback. A connect while one is already running joins it.

        Restarting a finished replay is the point — it is what makes a refresh replay the session
        from the top. Restarting a RUNNING one is not: two drivers on one channel would interleave
        two copies of the same log into the same transcript.
        """
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def run(self) -> None:
        """Walk the log once, synthesizing the frames the log does not contain."""
        for frame in (self.fixtures.at(None) if self.fixtures else []):
            self._inject(frame)

        previous: Optional[datetime] = None
        for line, event in _rows(self.path):
            await self._pause(previous, event.get("timestamp"))
            previous = _parse_time(event.get("timestamp")) or previous
            seq = event.get("seq")

            if _is_llm_response(event):
                # BEFORE the Action, exactly as a live turn orders it: the deltas populate a
                # provisional bubble and the Action supersedes it.
                await self._stream(event.get("content") or "")
            self.channel._emit(event.get("event_type") or "Event", seq, line)
            if event.get("event_type") in ("Heartbeat", "Observation"):
                self.channel.push(self._tick(event))
            for frame in (self.fixtures.at(seq) if self.fixtures else []):
                self._inject(frame)

    def _inject(self, frame: WireFrame) -> None:
        """Emit a fixture frame, and make a fixture ask genuinely answerable.

        Without the second half, `--fixtures` would draw a permission dialog whose buttons do
        nothing — which would let a UI author build a modal that looks right and has never once
        completed its own round trip.
        """
        stamped = frame.model_copy(update={"session_id": self.channel.session_id})
        if stamped.frame_type == "BlockingAsk":
            self.channel.register_fixture_ask(stamped)  # type: ignore[arg-type]
            return
        self.channel.push(stamped)

    async def _stream(self, content: str) -> None:
        if not content:
            return
        stream_id = uuid.uuid4().hex
        self.channel._stream_id = stream_id
        for start in range(0, len(content), TOKEN_CHUNK_CHARS):
            self.channel.push(TokenDelta(
                session_id=self.channel.session_id,
                stream_id=stream_id,
                text=content[start:start + TOKEN_CHUNK_CHARS],
                phase="writing",
            ))
            await asyncio.sleep(TOKEN_CHUNK_CHARS / (SYNTHETIC_TPS * self.speed))

    def _tick(self, event: dict) -> StatusTick:
        return StatusTick(
            session_id=self.channel.session_id,
            phase="tool_call" if event.get("event_type") == "Observation" else "thinking",
            tps=SYNTHETIC_TPS,
            tps_verified=False,
            context_pct=event.get("context_utilization_pct"),
            synthetic=True,
        )

    async def _pause(self, previous: Optional[datetime], stamp: Any) -> None:
        now = _parse_time(stamp)
        if previous is None or now is None:
            return
        delta = (now - previous).total_seconds()
        if delta <= 0:
            return
        await asyncio.sleep(min(delta, MAX_STEP_S) / self.speed)


def _rows(path: Path) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append((line, json.loads(line)))
        except json.JSONDecodeError:
            continue  # a torn line is skipped, exactly as the bus's own replay does
    return rows


def _is_llm_response(event: dict) -> bool:
    return (
        event.get("event_type") == "Action"
        and event.get("action_type") == "llm_response"
        and not event.get("parent_id")
    )


def _parse_time(stamp: Any) -> Optional[datetime]:
    if not isinstance(stamp, str):
        return None
    try:
        return datetime.fromisoformat(stamp)
    except ValueError:
        return None


def _session_id_of(path: Path) -> str:
    return path.stem
