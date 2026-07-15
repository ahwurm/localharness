"""XML function call converter for models without native tool support."""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from localharness.core.types import ToolCall, ToolSchema

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thinking tag stripper (Qwen, DeepSeek, etc.)
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from model output."""
    if not text:
        return text
    text = _THINK_PATTERN.sub("", text)
    # Also strip orphaned tags (opening <think> in a prior iteration)
    text = text.replace("</think>", "").replace("<think>", "")
    return text.lstrip()


def truncate_after_last_tool_call(text: str) -> str:
    """Drop prose trailing the last tool-call block.

    Text a model writes AFTER a tool call is speculation by construction — no result existed
    yet (observed live: a 4B model narrating invented file contents after its <tool_call>,
    then trusting that narration over the real tool result on the next turn). Intended for the
    HISTORY copy of an assistant turn only; the event stream keeps the full text. Returns the
    text unchanged when no closing tag is present.
    """
    if not text:
        return text
    end = -1
    for marker in ("</tool_call>", "</function>"):
        idx = text.rfind(marker)
        if idx != -1:
            end = max(end, idx + len(marker))
    return text[:end] if end != -1 else text


# Untaught tool-call conventions a model may reach for despite never being taught the harness's
# XML syntax (Gemma's ```tool_code fences are the observed case on llama.cpp) — recognized so a
# parse failure is nudged with the correct format instead of silently reading as "no tool call
# intended" (which leaves parse_failures at 0 and hides the miss).
_FENCED_BLOCK_PATTERN = re.compile(
    r"```(?:tool_code|tool_call|json)\b(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE,
)
# Cheap "looks like a call" body check for a fenced block above: a bareword call (Gemma's
# tool_code fence is Python-call-style, e.g. list_files(path="/tmp")) or a JSON "name" key. No
# specific verb/tool name required — the fence tag itself is already the strong signal.
_FENCED_CALL_BODY_PATTERN = re.compile(r'\b[a-zA-Z_]\w*\s*\(|"name"\s*:')
# Bare inline JSON call shape (no fence needed): "name" co-occurring with "arguments"/"parameters"
# within 200 chars of each other, either key order.
_JSON_CALL_SHAPE_PATTERN = re.compile(
    r'"name"\s*:\s*"[^"]*".{0,200}?"(?:arguments|parameters)"\s*:'
    r'|"(?:arguments|parameters)"\s*:.{0,200}?"name"\s*:\s*"[^"]*"',
    re.DOTALL,
)


def has_tool_call_attempt(text: str) -> bool:
    """Check if text contains what looks like an attempted tool call.

    Recognizes the harness's taught XML syntax (<tool_call>, <function=...><parameter=...>) AND
    untaught-but-plausible conventions a model may reach for anyway: a fenced ```tool_code/
    ```tool_call/```json block whose body looks like a call, or bare inline JSON shaped like
    {"name": ..., "arguments"/"parameters": ...}.
    """
    if not text:
        return False
    if "<tool_call" in text or ("<function=" in text and "<parameter=" in text):
        return True
    for body in _FENCED_BLOCK_PATTERN.findall(text):
        if _FENCED_CALL_BODY_PATTERN.search(body):
            return True
    return bool(_JSON_CALL_SHAPE_PATTERN.search(text))


# ---------------------------------------------------------------------------
# Extraction regexes
# ---------------------------------------------------------------------------

# Qwen 3 native format: <function=NAME><parameter=PARAM>value</parameter></function>
_QWEN_TOOL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_QWEN_PARAM_PATTERN = re.compile(
    r"<parameter=([\w\-]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)

# Hermes JSON-inside-XML: <tool_call>{"name": "X", "arguments": {...}}</tool_call>
# Used by Qwen 3 base/instruct models (non-Coder)
_HERMES_TOOL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)

# Legacy OpenHands format: <name>X</name><parameters>{JSON}</parameters>
_LEGACY_TOOL_PATTERN = re.compile(
    r"<tool_call>\s*<name[>=]([\w\-]+)</name>\s*<parameters?>(.*?)</parameters?>\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_LEGACY_PARTIAL = re.compile(
    r"<tool_call>\s*<name[>=]([\w\-]+)</name>\s*<parameters?>([^<]*)",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Tolerant closer normalization (Qwen3.6 / llama.cpp thinking-mode drift)
# ---------------------------------------------------------------------------

# Runtimes intermittently emit stray, duplicated, or orphan </tool_call> closers
# (QwenLM/Qwen3.6#178, ggml-org/llama.cpp#20837 & #22684). The extraction regexes
# require balanced <tool_call>...</tool_call> pairs, so such noise drops the call.
# This pre-pass removes </tool_call> closers that have no body-bearing open wrapper
# to close — orphan closers (no matching open) and closers of a still-empty wrapper
# (the stray/duplicated shapes). It is run only as a fallback after the normal
# pipeline finds nothing, so it can never alter a successful parse; balanced,
# well-formed input contains no such closers and is returned unchanged.
_TOOLCALL_TOKEN = re.compile(r"<tool_call>|</tool_call>|<function=|<name[>=]|\{")


def _drop_spurious_toolcall_closers(text: str) -> str:
    """Drop orphan/empty-wrapper </tool_call> closers; return text unchanged if none."""
    stack: list[bool] = []  # one body-seen flag per open <tool_call>
    drops: list[tuple[int, int]] = []
    for m in _TOOLCALL_TOKEN.finditer(text):
        tok = m.group(0)
        if tok == "<tool_call>":
            stack.append(False)
        elif tok == "</tool_call>":
            if stack and stack[-1]:
                stack.pop()  # closes a body-bearing wrapper — keep
            else:
                drops.append(m.span())  # orphan or empty-wrapper closer — drop
        elif stack:
            stack[-1] = True  # body token (<function=/<name/{) inside open wrapper
    if not drops:
        return text
    out: list[str] = []
    prev = 0
    for s, e in drops:
        out.append(text[prev:s])
        prev = e
    out.append(text[prev:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------


# TODO: Model-family recognizer system — at startup, identify the model family
# (Qwen 3.6 Coder, Qwen 3 base, Llama, DeepSeek, etc.) and select an optimized
# tool call parser per family instead of the current try-all-formats chain.
# This would eliminate wasted regex passes and allow family-specific prompt
# injection (e.g. Hermes JSON for Qwen 3 base, XML params for Qwen 3.6 Coder).


# Stable prefix of build_system_injection's output — client.py's XML-mode injection fold checks
# for this marker to detect "already injected" and avoid appending the tool-syntax block twice
# when _complete_xml's BadRequestError fallback reuses already-injected messages.
_TOOL_INJECTION_MARKER = "You have access to the following tools. To call a tool, output a tool_call XML block"


class FnCallConverter:
    """Convert between OpenAI native tool_calls format and XML text format.

    Stateless — all methods are pure functions of their inputs.
    """

    def extract_tool_calls(self, response_text: str) -> list[ToolCall]:
        """Parse tool calls from model text response. Never raises.

        Tries Qwen native format first (<function=NAME><parameter=P>v</parameter>),
        then legacy OpenHands format (<name>X</name><parameters>{JSON}</parameters>).
        """
        cleaned = strip_thinking_tags(response_text)
        calls = self._extract_all(cleaned)
        if calls:
            return calls
        # Fallback: drop stray/duplicated/orphan </tool_call> closers that break the
        # balanced-pair regexes (Qwen3.6 / llama.cpp drift) and retry. Reached only
        # when the normal parse found nothing, so working parses are never altered —
        # only currently-zero inputs can change outcome.
        normalized = _drop_spurious_toolcall_closers(cleaned)
        if normalized != cleaned:
            return self._extract_all(normalized)
        return calls

    def _extract_all(self, cleaned: str) -> list[ToolCall]:
        """Run the format extractors in priority order; first non-empty wins."""
        # 1. Qwen 3.5/3.6 Coder: <function=NAME><parameter=P>value</parameter>
        # 2. Hermes JSON: <tool_call>{"name": "X", "arguments": {...}}</tool_call>
        # 3. Legacy OpenHands: <name>X</name><parameters>{JSON}</parameters>
        for extractor in (self._extract_qwen_native, self._extract_hermes_json, self._extract_legacy):
            calls = extractor(cleaned)
            if calls:
                return calls
        # 4. Partial legacy (truncated output)
        return self._extract_legacy_partial(cleaned)

    def _extract_qwen_native(self, text: str) -> list[ToolCall]:
        """Parse Qwen 3 native format: <function=NAME><parameter=P>v</parameter>."""
        calls: list[ToolCall] = []
        for match in _QWEN_TOOL_PATTERN.finditer(text):
            fn_name = match.group(1)
            body = match.group(2)
            args: dict[str, Any] = {}
            for pm in _QWEN_PARAM_PATTERN.finditer(body):
                param_name = pm.group(1)
                raw_value = pm.group(2).strip()
                args[param_name] = self._coerce_value(raw_value)
            calls.append(ToolCall(name=fn_name, arguments=args, id=str(uuid.uuid4())))
        return calls

    def _extract_hermes_json(self, text: str) -> list[ToolCall]:
        """Parse Hermes JSON-inside-XML: <tool_call>{"name":"X","arguments":{...}}</tool_call>."""
        calls: list[ToolCall] = []
        for match in _HERMES_TOOL_PATTERN.finditer(text):
            raw = match.group(1).strip()
            parsed = self._repair_json(raw)
            if parsed is None or not isinstance(parsed, dict):
                continue
            name = parsed.get("name", "")
            args = parsed.get("arguments", {})
            if not name:
                continue
            if isinstance(args, str):
                args = self._repair_json(args) or {}
            calls.append(ToolCall(name=name, arguments=args, id=str(uuid.uuid4())))
        return calls

    def _extract_legacy(self, text: str) -> list[ToolCall]:
        """Parse legacy OpenHands XML: <name>X</name><parameters>{JSON}</parameters>."""
        calls: list[ToolCall] = []
        for name, raw_params in _LEGACY_TOOL_PATTERN.findall(text):
            parsed = self._repair_json(raw_params.strip())
            if parsed is None:
                log.warning("Could not parse tool call parameters for %s — skipping", name)
                continue
            calls.append(ToolCall(name=name, arguments=parsed, id=str(uuid.uuid4())))
        return calls

    def _extract_legacy_partial(self, text: str) -> list[ToolCall]:
        """Parse truncated legacy format."""
        calls: list[ToolCall] = []
        for name, raw_params in _LEGACY_PARTIAL.findall(text):
            parsed = self._repair_json(raw_params.strip())
            if parsed is None:
                log.warning("Could not parse partial tool call parameters for %s — skipping", name)
                continue
            calls.append(ToolCall(name=name, arguments=parsed, id=str(uuid.uuid4())))
        return calls

    @staticmethod
    def _coerce_value(raw: str) -> Any:
        """Coerce a string value from XML parameter to appropriate Python type."""
        if not raw:
            return ""
        # Try JSON parse first (handles numbers, booleans, arrays, objects)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def build_system_injection(self, tools: list[ToolSchema]) -> str:
        """Serialize tools as XML schema block for system prompt injection."""
        if not tools:
            return ""

        tool_xml = "\n".join(self.schema_to_xml_tool(t) for t in tools)
        return (
            f"{_TOOL_INJECTION_MARKER}\n"
            "exactly as shown. Only output tool_call blocks — do not describe tool calls in prose.\n\n"
            f"<tools>\n{tool_xml}\n</tools>\n\n"
            "To call a tool:\n"
            "<tool_call>\n"
            "<name>tool_name</name>\n"
            '<parameters>{"param_name": "value"}</parameters>\n'
            "</tool_call>\n\n"
            "You may call multiple tools in one response. Do NOT wrap tool calls in <think> tags.\n"
            "After you have all necessary information, respond in plain text without a tool_call block."
        )

    def schema_to_xml_tool(self, schema: ToolSchema) -> str:
        """Convert a single ToolSchema to the <tool> XML block."""
        # Support both ToolSchema objects and plain dicts
        if isinstance(schema, dict):
            name = schema.get("name", "")
            description = schema.get("description", "")
            params = schema.get("parameters", {})
        else:
            name = schema.name
            description = schema.description
            params = schema.parameters
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
