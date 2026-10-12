"""Shared primitive types used across all LocalHarness components."""
from dataclasses import dataclass
from typing import Any, Literal, NewType, TypedDict

from localharness.core.content import Content, strip_private

AgentID = NewType("AgentID", str)
SessionID = NewType("SessionID", str)
EventSeq = NewType("EventSeq", int)
ToolCallID = NewType("ToolCallID", str)
DivisionID = NewType("DivisionID", str)
OrgID = NewType("OrgID", str)

# Message type used by LLMClient — OpenAI-compat format
Message = dict[str, Any]


class MessageProvenance(TypedDict):
    """Internal source information; never inferred from model-visible text.

    A user message's private ``_lh`` is this; a tool message's is a
    :class:`ToolProvenance` (see ``Provenance``).
    """

    origin: Literal["human", "harness"]
    subtype: str


class ToolOutcome(TypedDict):
    """What became of one tool call, written by the agent loop and never shown to the model.

    ``call`` is the call-only signature of the tool name and arguments, ``ok`` is
    whether the call ran and succeeded, ``denied`` whether the permission gate refused
    it and ``staged`` whether the gate staged it for a human.
    """

    call: str
    ok: bool
    denied: bool
    staged: bool


class ToolProvenance(TypedDict):
    """A tool message's private ``_lh``.

    It deliberately carries no ``origin``: an origin would add a source to every
    compaction summary that reads the key.
    """

    outcome: ToolOutcome


# The documented type of a message's private ``_lh`` key.
Provenance = MessageProvenance | ToolProvenance


def human_message(content: Content, subtype: str = "request") -> Message:
    """A human turn: a ``str``, or OpenAI content parts when an image rides with the text
    (see :mod:`localharness.core.content`)."""
    return {"role": "user", "content": content,
            "_lh": MessageProvenance(origin="human", subtype=subtype)}


def harness_message(content: str, subtype: str) -> Message:
    """A control message stays user-role for strict local chat templates."""
    return {"role": "user", "content": content,
            "_lh": MessageProvenance(origin="harness", subtype=subtype)}


def is_harness_message(message: Message) -> bool:
    metadata = message.get("_lh")
    return isinstance(metadata, dict) and metadata.get("origin") == "harness"


def provider_messages(messages: list[Message]) -> list[Message]:
    """Render provenance and remove private fields without mutating canonical history.

    Call before XML role merging so human and harness content retain their boundaries.
    Text that merely quotes the marker remains ordinary user content.
    """
    rendered = []
    for message in messages:
        wire = {key: value for key, value in message.items() if key != "_lh"}
        if is_harness_message(message):
            subtype = message["_lh"].get("subtype", "control")
            marker = f"[Harness control: {subtype}; not human feedback]\n"
            content = wire.get("content") or ""
            wire["content"] = (
                marker + content if isinstance(content, str)
                else [{"type": "text", "text": marker}, *content]
            )
        if isinstance(wire.get("content"), list):
            wire["content"] = strip_private(wire["content"])  # part-level _lh (image facts) stays home
        rendered.append(wire)
    return rendered

# Tool schema — JSON Schema format as expected by OpenAI API
ToolSchema = dict[str, Any]


# Parsed tool call from model response
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""
