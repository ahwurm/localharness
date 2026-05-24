"""XML function call converter for models without native tool support."""
from __future__ import annotations

import json
import logging
import re
import uuid

from localharness.core.types import ToolCall, ToolSchema

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction regexes
# ---------------------------------------------------------------------------

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<name>([\w\-]+)</name>\s*<parameters>(.*?)</parameters>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_TOOL_CALL_PARTIAL = re.compile(
    r"<tool_call>\s*<name>([\w\-]+)</name>\s*<parameters>([^<]*)",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


class FnCallConverter:
    """Convert between OpenAI native tool_calls format and XML text format.

    Stateless — all methods are pure functions of their inputs.
    """

    def extract_tool_calls(self, response_text: str) -> list[ToolCall]:
        """Parse tool calls from model text response. Never raises."""
        calls: list[ToolCall] = []

        matches = _TOOL_CALL_PATTERN.findall(response_text)
        if matches:
            for name, raw_params in matches:
                parsed = self._repair_json(raw_params.strip())
                if parsed is None:
                    log.warning("Could not parse tool call parameters for %s — skipping", name)
                    continue
                calls.append(ToolCall(name=name, arguments=parsed, id=str(uuid.uuid4())))
            return calls

        # Fallback: partial (truncated) matches
        partial_matches = _TOOL_CALL_PARTIAL.findall(response_text)
        for name, raw_params in partial_matches:
            parsed = self._repair_json(raw_params.strip())
            if parsed is None:
                log.warning("Could not parse partial tool call parameters for %s — skipping", name)
                continue
            calls.append(ToolCall(name=name, arguments=parsed, id=str(uuid.uuid4())))

        return calls

    def build_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block for system prompt injection."""
        if not tools:
            return ""

        tool_xml = "\n".join(self.schema_to_xml_tool(t) for t in tools)
        return (
            "You have access to the following tools. To call a tool, output a tool_call XML block\n"
            "exactly as shown. Only output tool_call blocks — do not describe tool calls in prose.\n\n"
            f"<tools>\n{tool_xml}\n</tools>\n\n"
            "To call a tool:\n"
            "<tool_call>\n"
            "<name>tool_name</name>\n"
            '<parameters>{"param_name": "value"}</parameters>\n'
            "</tool_call>\n\n"
            "You may call multiple tools in sequence. Wait for tool results before calling the next tool.\n"
            "After you have all necessary information, respond in plain text without a tool_call block."
        )

    def schema_to_xml_tool(self, schema: ToolSchema) -> str:
        """Convert a single ToolSchema to the <tool> XML block."""
        name = schema.get("name", "")
        description = schema.get("description", "")
        params = schema.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            is_required = "true" if param_name in required else "false"
            param_lines.append(
                f'      <parameter name="{param_name}" type="{param_type}" required="{is_required}">\n'
                f"        {param_desc}\n"
                f"      </parameter>"
            )

        params_xml = "\n".join(param_lines)
        return (
            f'  <tool name="{name}">\n'
            f"    <description>{description}</description>\n"
            f"    <parameters>\n{params_xml}\n    </parameters>\n"
            f"  </tool>"
        )

    def tool_calls_to_messages(self, tool_calls: list[ToolCall]) -> list[dict]:
        """Convert extracted ToolCall list to OpenAI message format dicts."""
        messages = []
        for tc in tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id or str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                ],
            })
        return messages

    def _repair_json(self, raw: str) -> dict | None:
        """Attempt to parse JSON with lightweight repair. Returns None on total failure."""
        # 1. Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # 2. Remove trailing commas before }
        repaired = re.sub(r",\s*}", "}", raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 3. Single quotes -> double quotes
        repaired = re.sub(r"'", '"', raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 4. Append } to close truncated objects
        repaired = raw + "}"
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

        # 5. jsonrepair if available
        try:
            from jsonrepair import repair_json  # type: ignore[import-untyped]
            return json.loads(repair_json(raw))
        except (ImportError, Exception):
            pass

        return None
