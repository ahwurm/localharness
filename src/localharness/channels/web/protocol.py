"""The web channel's wire contract: the SSE-only frames, and the generated schema behind them.

Read §4.2 of the web-channel PRD before changing anything here. The one rule everything else
hangs off:

    **The wire IS the bus event stream.** Every persisted event reaches the client as the exact
    bytes `event.model_dump_json()` writes to `sessions/<id>.jsonl`, with the bus `seq` as the
    SSE `id:`. One serialization, three consumers (disk, phone, any future analysis tool), and a
    new event type added anywhere in the harness reaches the phone with no channel change.

The frames in THIS module are the deliberate exception: live-progress signals that are **not**
published to the bus and therefore **not persisted**. Persisting every token would bloat the
session log enormously and the final `Action.content` / `TaskComplete.summary` carries the same
text deduplicated. So:

    **The wire is a superset of the log, and the difference is exactly these frames.**

A replayed session renders the persisted events and shows no deltas. That is written down here,
and served at `GET /api/protocol` as `sse_only[]`, so a UI author never wonders why a replayed
chat looks quieter than a live one.

The corollary is the rule that keeps live and replay on ONE client code path: an SSE-only frame
populates only **provisional** state, and a persisted event always supersedes it. Nothing
load-bearing is ever carried by a frame alone — which is why this set is progress signals plus
the two things that have no persisted analog at all (a blocking ask, and a cancellation).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

PROTOCOL_VERSION = 1
"""Bumped on any breaking change to the wire.

Enforced rather than remembered: `tests/unit/channels/test_web_protocol.py` snapshots every
event and frame schema into a checked-in fixture and fails if a schema moves without this
integer moving in the same diff. A project that auto-generates its schema specifically because
it distrusts humans to keep two things in sync should not then trust a human to remember the
version — and a forgotten bump degrades a stale client SILENTLY (`undefined` renders as a blank
cell, not a crash), which is the worst failure for somebody returning to their own UI a few
evenings later.

It tracks FRAME semantics, not the route list: ADDING an endpoint does not bump it. A client
written against version 1 keeps rendering every frame correctly when a new route appears
beside the ones it already calls, so a bump there would only teach clients to distrust a
number that had not changed meaning. Changing or removing a frame field is what moves it.

The snapshot digests a field's DESCRIPTION along with its type, so correcting wrong prose also
moves it. That is a re-record without a bump, called out in the commit message: the frame is
byte-for-byte what it was, and a client written against version 1 renders it identically.
"""

NEVER_FIRED_EVENTS: frozenset[str] = frozenset({
    "SystemReady",
    "AgentCreated",
    "AgentDeleted",
    "TaskRequest",
    "DelegationRequest",
    "DelegationResult",
})
"""Event types that are declared but never published anywhere in the harness (verified by grep,
2026-09-14).

Served with `never_fires: true` rather than quietly listed beside the real ones: a client author
reading `/api/protocol` would otherwise write and test rendering code for six events that can
never arrive, and never find out.
"""

FINISH_REASONS_KNOWN: tuple[str, ...] = (
    "stop", "length", "tool_calls", "content_filter", "function_call", "degenerate",
)
"""The `Action.finish_reason` values seen in practice, for a client that wants to branch on them.

**Open, not closed.** The PRD asked for this field to be tightened from `str` to a `Literal`.
It is deliberately NOT: the value comes straight off the provider's response
(`agent/loop.py:1629`, `getattr(response_message, "finish_reason", None)`), so the set is
whatever the serving runtime emits — vLLM, llama.cpp, Ollama and LM Studio do not agree, and a
`Literal` would turn an unfamiliar runtime's perfectly ordinary reply into a validation error on
deserialize, breaking replay of logs already on disk. `degenerate` is the one value the harness
mints itself (`agent/loop.py:1527`). The intent behind the request — "a UI author can know the
value space" — is served by enumerating it here and shipping it in `/api/protocol`, which costs
a client nothing and cannot break a runtime we have not met yet.
"""

COLLAPSIBLE_GROUPS: tuple[str, ...] = ("fs.read", "web", "memory")
"""The ONLY tool groups a client may collapse into a counter line (WEBCH-04).

Opt-in by safe group, never opt-out by dangerous group. A tool in a group the client has never
heard of is itemized, so a newly-added dangerous tool cannot silently start collapsing because
nobody remembered to add it to a deny list. Served in `/api/protocol` so the reference page and
the owner's own UI read the same list from one place.
"""

UNTRUSTED_INGEST_LABEL = "untrusted_ingest"
"""The provenance label a frame carries for the web tools (`tools/capabilities.UNTRUSTED_INGEST`).

The UI marks it the way the terminal's once-per-turn note does. It is a label, not a filter:
§4.2 forbids the wire dropping anything.
"""


class WireFrame(BaseModel):
    """Base for every SSE-only frame. Immutable, like the bus events it travels beside.

    `frame_type` mirrors `BaseEvent.event_type`'s job — it is what a client switches on — and is
    spelled differently on purpose. A client that cannot tell a frame from an event at a glance
    will eventually treat one as the other, and the whole provisional-vs-authoritative rule of
    §4.2.1 rests on telling them apart.
    """

    model_config = ConfigDict(frozen=True)

    frame_type: str
    session_id: Optional[str] = Field(
        default=None,
        description="The session this frame belongs to. Every frame carries it so one multiplexed "
                    "stream can serve several sessions without the transport changing (§4.1).",
    )


class Hello(WireFrame):
    """The first frame on every connect, live or replayed."""

    frame_type: Literal["Hello"] = "Hello"
    protocol_version: int = Field(
        default=PROTOCOL_VERSION,
        description="Bumped on any breaking wire change. A client that does not recognise it must "
                    "say so loudly rather than rendering blanks.",
    )
    agent_id: Optional[str] = Field(default=None, description="Whose session this is.")
    mode: Optional[str] = Field(
        default=None,
        description="The gate's permission mode right now (auto/guarded/trusted/read-only/"
                    "unattended). None before a session exists.",
    )
    resume_seq: Optional[int] = Field(
        default=None,
        description="The `seq` this connection resumed from, i.e. the client's own cursor as the "
                    "server honoured it. None on a fresh connect with no cursor.",
    )
    turn_in_progress: bool = Field(
        default=False,
        description="True when a turn is running RIGHT NOW. It exists so a client reconnecting "
                    "mid-turn knows at once, instead of inferring it by scanning backwards for an "
                    "unterminated TurnStarted.",
    )
    model_state: str = Field(
        default="unknown",
        description="ready | cold | unreachable | building | unknown — what `GET /api/health` "
                    "would say, delivered on connect so the first screen never has to ask.",
    )
    session_live: bool = Field(
        default=False,
        description="True once the session is built and can take a turn. False means the server "
                    "is reachable but the session is still coming up (or has not started): "
                    "reading history works, sending starts bring-up.",
    )
    synthetic: bool = Field(
        default=False,
        description="True in --replay. Says that TokenDelta and StatusTick frames on this "
                    "connection were SYNTHESIZED from a persisted log, not measured — so nobody "
                    "mistakes a replayed tok/s for a real one (§8).",
    )


class TokenDelta(WireFrame):
    """One chunk of the model's answer as it generates (§4.3).

    Populates a **provisional** bubble which the next tool-less `llm_response` Action supersedes.
    The ordering is guaranteed, not hoped for: `stream_complete(on_token=...)` is awaited at
    `agent/loop.py:1510` and the Action is published afterwards at `agent/loop.py:1669`.
    """

    frame_type: Literal["TokenDelta"] = "TokenDelta"
    stream_id: str = Field(description="Groups the deltas of one generation. Matched by StreamClosed.")
    text: str = Field(description="The raw delta. Render as TEXT, never as markup (§5.7).")
    phase: str = Field(
        default="writing",
        description="Which phase produced it, mirroring `provider.stream_snapshot()['phase']`.",
    )


class StreamClosed(WireFrame):
    """The provisional bubble's hand-off to its authoritative Action.

    Emitted when the channel sees the `llm_response` Action that supersedes a stream, so the
    client never has to INFER the hand-off — inferring it is how an answer ends up rendered
    twice, which §4.2.1 calls the single most likely way a client ships broken.
    """

    frame_type: Literal["StreamClosed"] = "StreamClosed"
    stream_id: str
    superseded_by_seq: Optional[int] = Field(
        default=None,
        description="The seq of the Action that now owns this text — drop the provisional bubble, "
                    "that Action renders it. None when the stream ended without one (a cancelled "
                    "or failed turn): KEEP the text and drop only its 'streaming' label. No "
                    "Action is coming and none was persisted, so the bubble is the only copy of "
                    "what the model wrote (WEBCH-37).",
    )


class ReasoningDelta(WireFrame):
    """One chunk of the model's reasoning as it thinks (§4.3).

    Live only. `Action.reasoning_chars` is a COUNT — the text is stripped before the event
    exists — so a REPLAYED turn can never show what the model thought. The UI must say so rather
    than showing an empty pane (§6.4's fidelity limits).
    """

    frame_type: Literal["ReasoningDelta"] = "ReasoningDelta"
    text: str = Field(description="Render as TEXT, never as markup.")


class StatusTick(WireFrame):
    """The instrument cluster (§5.4), pushed only while a turn is running.

    None of this is on the bus: `model_source` / `tps_source` / `progress_source` are direct
    callables the terminal gets wired inside an `isinstance(TerminalChannel)` block, so this
    frame is genuinely new plumbing rather than a subscription. An idle phone holds a silent
    stream — the ticker starts on TurnStarted and stops on the turn's end.
    """

    frame_type: Literal["StatusTick"] = "StatusTick"
    phase: str = Field(default="waiting", description="waiting | thinking | writing | tool_call.")
    thinking_tokens: int = 0
    answer_tokens: int = 0
    tool_call_tokens: int = 0
    elapsed: float = Field(default=0.0, description="Seconds since the current request started.")
    silent: float = Field(
        default=0.0,
        description="Seconds since the last delta. The terminal flags a stall at 10s "
                    "(`_SILENCE_NOTE_SECONDS`); mirror that band rather than inventing one.",
    )
    tps: Optional[float] = Field(default=None, description="Measured decode rate; None when unknown.")
    tps_verified: bool = Field(
        default=False,
        description="False means tps is a live estimate that self-corrects at stream end; True "
                    "means it is the previous stream's measured rate. Bands: green >=30, "
                    "yellow >=20 (`provider/speed_stats.py`).",
    )
    model: Optional[str] = Field(default=None, description="The model serving this turn, swap-safe.")
    context_pct: Optional[float] = Field(
        default=None,
        description="Latest Heartbeat.context_utilization_pct. Bands <50/<65/<80/>=80, where 80 "
                    "is where compaction actually FIRES — so red means 'compacting now', not "
                    "'nearly full'.",
    )
    synthetic: bool = Field(
        default=False,
        description="True when this tick was synthesized by --replay from persisted timestamps. "
                    "Plausible, not measured.",
    )


class AskOption(BaseModel):
    """One button on a blocking ask."""

    model_config = ConfigDict(frozen=True)

    kind: str = Field(description="allow_once | allow_always | reject_once | reject_always.")
    name: str = Field(description="The label to draw. Comes from the server so a class with its "
                                  "own wording (workspace-trust) cannot be mislabelled by a page.")
    confirm_required: bool = Field(
        default=False,
        description="True for the two `_always` kinds: the FIRST POST of this kind writes no "
                    "grant and returns a short-lived token; only a second POST carrying it "
                    "produces the durable decision. Enforced on the server (§5.3a) — this flag "
                    "only lets a UI warn in advance.",
    )


class BlockingAsk(WireFrame):
    """A permission question that is holding a tool call, carrying its own answer handle (§5.2).

    This is a channel-minted frame, NOT the `PermissionAsked` bus event — that event is audit
    only and carries no request payload. The names are one suffix apart on purpose-avoidance
    grounds: an earlier draft called this `PermissionAsk`, and two semantically unrelated frames
    sharing a 14-character prefix is a `startsWith` branch waiting to merge them at 11pm.

    **The question is state, not a message.** `GET /api/permissions` returns every open ask, so a
    client that reconnects, connects fresh, or is a second device renders it without ever having
    received this frame.
    """

    frame_type: Literal["BlockingAsk"] = "BlockingAsk"
    request_id: str = Field(description="Answer with POST /api/permissions/{request_id}/answer. "
                                        "Idempotent: a duplicate or late answer returns the "
                                        "recorded outcome, never a second decision.")
    tool_name: str
    tool_params: dict[str, Any] = Field(
        default_factory=dict,
        description="The model's own arguments. Render as TEXT, never markup — a rendered-HTML "
                    "tool argument can rewrite the very dialog that is asking about it (§5.7).",
    )
    klass: str = Field(description="The AskClass. `workspace-trust` is the one that is permanent "
                                   "despite arriving grantable=False (§5.6).")
    key: Optional[str] = Field(default=None, description="The grant key an `_always` answer "
                                                          "would be remembered under.")
    grantable: bool = Field(description="False means only the `_once` pair is offered.")
    reason: str = ""
    display: str = Field(default="", description="The gate's one-line human rendering, already "
                                                  "stripped of terminal control characters.")
    agent_id: Optional[str] = Field(default=None, description="WHO is asking — one gate serves "
                                                               "the orchestrator and every subagent.")
    call_id: Optional[str] = Field(default=None, description="The tool call this question is "
                                                              "about (`Action.tool_call_id`).")
    options_legend: Optional[str] = None
    options: list[AskOption] = Field(default_factory=list)
    untrusted_ingest: bool = Field(
        default=False,
        description="The asking tool ingests attacker-controllable bytes (web_search/web_fetch/"
                    "web_page_query). Mark the provenance.",
    )
    deadline_s: Optional[float] = Field(
        default=None,
        description="Roughly how long the GATE will wait before recording reject_once. Advisory "
                    "only — the deadline is the gate's, and a second one in the client would be "
                    "a second place to get it wrong.",
    )


class AskExpired(WireFrame):
    """A blocking ask stopped mattering before anybody answered it.

    Pushed when the gate cancels the wait. The Discord lesson (`PERMISSION_TIMEOUT_LINE`) is that
    a question still showing live buttons after it was recorded as a denial is a lie somebody
    will tap.
    """

    frame_type: Literal["AskExpired"] = "AskExpired"
    request_id: str
    decision: str = Field(default="reject_once", description="What was recorded instead. Fail closed.")
    waited_s: float = Field(default=0.0, description="What the person actually waited — measured, "
                                                      "not the configured budget.")


class TurnCancelled(WireFrame):
    """A turn was cancelled from a client.

    It exists because the loop publishes NEITHER `TurnCompleted` NOR `TurnFailed` on cancel
    (`agent/loop.py`; the REPL prints its own line). Without this the phone shows a turn that
    never ends.
    """

    frame_type: Literal["TurnCancelled"] = "TurnCancelled"
    reason: str = "cancelled"


class BringUpStage(WireFrame):
    """Which stage of session bring-up is running (WEBCH-43).

    A rising number reports elapsed time, not health: a wedged memory lock, a failing MCP server
    and a sibling process holding the inference flock all look identical to a healthy slow start.
    So the ticker NAMES the stage, and the build window gets its own abort affordance — cancelling
    a TURN is a different state machine and there is otherwise no way out of a stuck build.
    """

    frame_type: Literal["BringUpStage"] = "BringUpStage"
    stage: str = Field(description="A short human label: loading config, opening memory, starting "
                                   "MCP servers, waiting for the model server, ready, failed.")
    detail: str = ""
    elapsed: float = 0.0
    abortable: bool = Field(default=True, description="Whether POST /api/bringup/abort applies.")
    failed: bool = False


class Notice(WireFrame):
    """Prose the harness addressed to the human rather than to the model.

    Slash-command output, mode changes, the workspace-trust notices, a compaction line, an error
    — everything a channel's `send_message` / `send_error` / `send_renderable` carries. It exists
    because without it the entire slash-command surface would be invisible on the phone: the REPL
    talks to a person exclusively through those three methods.

    `style` is the REPL's own `metadata["style"]` intent (`system.info`, `system.error`), which
    is a DEAD key today — no channel reads it, so info-versus-error intent is currently thrown
    away at the terminal. Carrying it is the cheap win §2.2 names.
    """

    frame_type: Literal["Notice"] = "Notice"
    text: str = Field(description="Render as TEXT, never markup.")
    style: str = Field(default="system.info", description="system.info | system.error | renderable.")
    detail: Optional[str] = None
    agent_id: Optional[str] = None
    preformatted: bool = Field(
        default=False,
        description="True for `send_renderable` output (the /memory tree): box-drawing that only "
                    "survives in a <pre>. This is the one place the web channel is WORSE than the "
                    "terminal, and memory legibility is the project's north star — a JSON memory "
                    "endpoint is the first thing after Phase B, not vague later work.",
    )


class GapDetected(WireFrame):
    """Events a live client saw are missing from the log a reconnecting client replays from.

    Not hypothetical: `EventBus._append_jsonl` swallows write failures — it logs and returns —
    and `publish()` delivers to subscribers regardless. So a disk hiccup produces an event that
    was rendered live and is permanently absent from the file. WEBCH-06's diff-against-the-JSONL
    test would still PASS while describing a client that silently missed something.

    Fixing the swallow inside the bus is a larger change than this channel should make
    unilaterally. Surfacing it is not: a visible gap marker beats an invisible one, and the
    server logs it loudly besides.
    """

    frame_type: Literal["GapDetected"] = "GapDetected"
    from_seq: int = Field(description="First seq known to be missing from the log.")
    to_seq: int = Field(description="Last seq known to be missing (inclusive).")
    detail: str = Field(
        default="",
        description="Why the gap is believed to exist, in words a person can act on.",
    )


class Lagged(WireFrame):
    """This client fell too far behind and its buffer was dropped rather than grown.

    The honest alternative to an unbounded queue: a phone that sleeps through a 20-tool-call turn
    must not be able to make the server hold the whole thing in memory forever. The client
    re-syncs with `GET /api/sessions/{id}/events?from={seq}` — the same cursor reconnect uses —
    so the recovery path is one that is already tested.
    """

    frame_type: Literal["Lagged"] = "Lagged"
    from_seq: Optional[int] = Field(
        default=None, description="Resync from here; None means resync from the beginning."
    )
    dropped: int = 0


FRAME_TYPES: tuple[type[WireFrame], ...] = (
    Hello,
    TokenDelta,
    StreamClosed,
    ReasoningDelta,
    StatusTick,
    BlockingAsk,
    AskExpired,
    TurnCancelled,
    BringUpStage,
    Notice,
    GapDetected,
    Lagged,
)
"""Every SSE-only frame, in the order `/api/protocol` lists them.

Derived from here rather than spelled out again in the server, so a frame added without being
described is impossible: the contract test walks this tuple.
"""


def frame_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema for every SSE-only frame, generated from the models themselves."""
    return {f.__name__: f.model_json_schema() for f in FRAME_TYPES}


def event_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schema for every bus event type, generated from the models themselves.

    Auto-generated means it cannot drift from the code. It also means the rendering rules that
    matter — the `has_tool_calls` double-print rule above all — only reach a client if they are
    written as `Field(description=...)` on the field itself, which is why §8 requires that of
    every field carrying one.
    """
    from localharness.core.events import EVENT_TYPE_MAP

    return {name: model.model_json_schema() for name, model in sorted(EVENT_TYPE_MAP.items())}


TRANSCRIPT_RULES: tuple[str, ...] = (
    "Action(action_type='llm_response', has_tool_calls=True) is INTERSTITIAL NARRATION — the "
    "model talking while it calls tools. Render it as a narration bubble.",
    "Action(action_type='llm_response', has_tool_calls=False) is the FINAL ANSWER and MUST NOT "
    "be rendered. The rule is written into the field's own declaration in core/events.py: a "
    "tool-less llm_response is rendered via TaskComplete and must never be echoed as narration.",
    "TaskComplete.summary is where the answer is rendered FROM. Getting this backwards is the "
    "single most likely way a client ships with every answer printed twice.",
    "TokenDelta populates a PROVISIONAL bubble that the next llm_response Action supersedes; "
    "StreamClosed tells you which seq took it over, so you never have to infer the hand-off.",
    "Deltas only ever come from the ROOT turn. Every subagent call site invokes run_turn with no "
    "on_token, so a child's generation cannot interleave with the root's stream.",
    "Events carrying parent_id belong to a SUBAGENT. The wire forwards them verbatim — including "
    "a child's TaskComplete and TurnFailed, which the base adapter's RENDERING policy swallows "
    "and downgrades. That swallow is a client default, not a transport filter: a filter at the "
    "wire would be invisible and impossible for a UI to undo. Collapse them into an expandable "
    "child block; do not drop them.",
    "Collapse ONLY the groups in collapsible_groups. A tool in a group you have never seen is "
    "itemized. Never key a collapse rule off Action.risk_level — it is dead and null on every "
    "Action ever recorded.",
    "Every model- or tool-derived string renders as TEXT, never markup. A tool result that can "
    "execute script in the page is a tool result that can operate the permission UI.",
    "SSE-only frames are absent on replay. A replayed session is quieter by construction, not "
    "broken.",
)
"""The rendering rules a client cannot derive from the schema, served at `GET /api/protocol`.

`model_json_schema()` cannot see a `#` comment, and the rules that decide whether a transcript is
correct live in comments today. Rather than hope a UI author reads the source, they are shipped
as data beside the schema they qualify.
"""
