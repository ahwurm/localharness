"""Shared primitive types used across all LocalHarness components."""
from dataclasses import dataclass, field
from typing import Any, NewType

AgentID = NewType("AgentID", str)
SessionID = NewType("SessionID", str)
EventSeq = NewType("EventSeq", int)
ToolCallID = NewType("ToolCallID", str)

# Message type used by LLMClient — OpenAI-compat format
Message = dict[str, Any]

# Tool schema — JSON Schema format as expected by OpenAI API
ToolSchema = dict[str, Any]


# Parsed tool call from model response
@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = ""
