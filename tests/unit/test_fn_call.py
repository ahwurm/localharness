"""Tests for XML function call converter (fn_call.py)."""
import pytest

from localharness.core.types import ToolCall, ToolSchema
from localharness.provider.fn_call import FnCallConverter


@pytest.fixture
def converter() -> FnCallConverter:
    return FnCallConverter()


# ---------------------------------------------------------------------------
# extract_tool_calls
# ---------------------------------------------------------------------------


def test_extract_single_tool_call(converter: FnCallConverter):
    text = """
<tool_call>
<name>list_files</name>
<parameters>{"path": "/tmp"}</parameters>
</tool_call>
"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_extract_multiple_tool_calls(converter: FnCallConverter):
    text = """
<tool_call>
<name>read_file</name>
<parameters>{"path": "/etc/hosts"}</parameters>
</tool_call>
<tool_call>
<name>glob</name>
<parameters>{"pattern": "src/**/*.py"}</parameters>
</tool_call>
"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "read_file"
    assert calls[1].name == "glob"


def test_extract_no_tool_calls(converter: FnCallConverter):
    text = "This is a plain text response with no tool calls."
    calls = converter.extract_tool_calls(text)
    assert calls == []


def test_extract_partial_tool_call(converter: FnCallConverter):
    """Truncated tool_call (missing closing tag) still extracted via partial regex."""
    text = '<tool_call>\n<name>my_tool</name>\n<parameters>{"key": "value"}'
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "my_tool"


# ---------------------------------------------------------------------------
# _repair_json
# ---------------------------------------------------------------------------


def test_json_repair_trailing_comma(converter: FnCallConverter):
    result = converter._repair_json('{"a": 1,}')
    assert result == {"a": 1}


def test_json_repair_single_quotes(converter: FnCallConverter):
    result = converter._repair_json("{'a': 'b'}")
    assert result == {"a": "b"}


def test_json_repair_truncated(converter: FnCallConverter):
    result = converter._repair_json('{"a": 1')
    assert result == {"a": 1}


def test_json_repair_valid(converter: FnCallConverter):
    result = converter._repair_json('{"x": 42}')
    assert result == {"x": 42}


def test_json_repair_unfixable(converter: FnCallConverter):
    """Totally malformed JSON returns None."""
    result = converter._repair_json("this is not json at all }{")
    assert result is None


# ---------------------------------------------------------------------------
# build_system_injection
# ---------------------------------------------------------------------------


def test_build_system_injection_empty(converter: FnCallConverter):
    result = converter.build_system_injection([])
    assert result == ""


def test_build_system_injection(converter: FnCallConverter):
    tools: list[ToolSchema] = [
        {
            "name": "list_files",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory path"}},
                "required": ["path"],
            },
        },
        {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "File path"}},
                "required": ["path"],
            },
        },
    ]
    result = converter.build_system_injection(tools)
    assert "<tools>" in result
    assert 'name="list_files"' in result
    assert 'name="read_file"' in result
    assert "<tool_call>" in result


# ---------------------------------------------------------------------------
# schema_to_xml_tool
# ---------------------------------------------------------------------------


def test_schema_to_xml_tool(converter: FnCallConverter):
    schema: ToolSchema = {
        "name": "my_tool",
        "description": "Does something useful",
        "parameters": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "How many",
                },
            },
            "required": ["count"],
        },
    }
    xml = converter.schema_to_xml_tool(schema)
    assert '<tool name="my_tool">' in xml
    assert "<description>Does something useful</description>" in xml
    assert 'name="count"' in xml
    assert 'type="integer"' in xml


# ---------------------------------------------------------------------------
# tool_calls_to_messages
# ---------------------------------------------------------------------------


def test_tool_calls_to_messages(converter: FnCallConverter):
    calls = [
        ToolCall(name="search", arguments={"query": "hello"}, id="tc-1"),
        ToolCall(name="list_files", arguments={"path": "/"}, id="tc-2"),
    ]
    messages = converter.tool_calls_to_messages(calls)
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "assistant"
