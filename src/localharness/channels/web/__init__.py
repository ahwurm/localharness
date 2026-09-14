"""The web channel: a JSON event API and the static shell that consumes it.

`localharness web` serves two things on a private network boundary — the bus event stream over
SSE, and the deliberately-ugly reference page that exercises every wire feature. The wire
protocol is not invented: the bus already assigns a monotonic `seq` to every event and persists
it before delivering it, so the channel forwards those exact bytes and three properties fall out.

**Nothing is hidden.** Tool calls, results, denials, parked calls, compaction, parse failures and
turn accounting are all already typed bus events, and they arrive TYPED — so the page can tell
"the answer" from "a system notice", a distinction the terminal has no typed way to make.

**Reconnect is cheap.** The replay log is already on disk with a monotonic cursor, so a phone
that sleeps mid-turn resumes from `seq`. Cheap is not free: the stitching is new code and it has
one honest hole in it, which `GapDetected` makes visible rather than inherits silently.

**A closed phone cannot hang the agent.** In `auto` — the default — the gate PARKS a gated call
instead of blocking, so the phone's permission surface is a queue with a badge, not a modal.

See `.planning/web-channel-PRD.md` for the full design and `docs/web.md` for the user-facing
account, including the honest "not yet" list.
"""
from .channel import WebChannel
from .protocol import PROTOCOL_VERSION
from .server import WebServer

__all__ = ["PROTOCOL_VERSION", "WebChannel", "WebServer"]
