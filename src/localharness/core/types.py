"""Shared primitive types used across all LocalHarness components."""
from dataclasses import dataclass
from typing import Any, Literal, NewType, TypedDict

AgentID = NewType("AgentID", str)
SessionID = NewType("SessionID", str)
EventSeq = NewType("EventSeq", int)
ToolCallID = NewType("ToolCallID", str)
DivisionID = NewType("DivisionID", str)
OrgID = NewType("OrgID", str)

# Message type used by LLMClient — OpenAI-compat format
Message = dict[str, Any]


class MessageProvenance(TypedDict):
    """Internal source information; never inferred from model-visible text."""

    origin: Literal["human", "harness"]
    subtype: str


def human_message(content: str, subtype: str = "request") -> Message:
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
