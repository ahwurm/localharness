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


# ---------------------------------------------------------------------------
# strip_thinking_tags / has_tool_call_attempt
# ---------------------------------------------------------------------------


def test_strip_thinking_tags():
    from localharness.provider.fn_call import strip_thinking_tags
    text = "<think>\nI should search for files.\n</think>\n\nHere is the result."
    assert strip_thinking_tags(text) == "Here is the result."


def test_strip_thinking_tags_empty():
    from localharness.provider.fn_call import strip_thinking_tags
    assert strip_thinking_tags("") == ""
    assert strip_thinking_tags(None) is None


def test_strip_thinking_tags_preserves_non_think():
    from localharness.provider.fn_call import strip_thinking_tags
    text = "No thinking here."
    assert strip_thinking_tags(text) == "No thinking here."


def test_has_tool_call_attempt():
    from localharness.provider.fn_call import has_tool_call_attempt
    assert has_tool_call_attempt("<tool_call>\n<name>glob</name>") is True
    assert has_tool_call_attempt("<function=grep>\n<parameter=pattern>test</parameter>") is True
    assert has_tool_call_attempt("Just plain text.") is False
    assert has_tool_call_attempt("") is False


# ---------------------------------------------------------------------------
# Permissive XML parsing
# ---------------------------------------------------------------------------


def test_extract_name_equals_syntax(converter: FnCallConverter):
    """Qwen sometimes emits <name=X</name> instead of <name>X</name>."""
    text = """<tool_call>
<name=grep</name>
<parameters>{"pattern": "morning"}</parameters>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "grep"
    assert calls[0].arguments == {"pattern": "morning"}


def test_extract_singular_parameter_tag(converter: FnCallConverter):
    """Some models emit <parameter> instead of <parameters>."""
    text = """<tool_call>
<name>glob</name>
<parameter>{"pattern": "*.py"}</parameter>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "glob"


def test_extract_tool_calls_with_thinking_tags(converter: FnCallConverter):
    """Tool calls wrapped in thinking tags should still be extracted."""
    text = """<think>
I need to search for files.
</think>

<tool_call>
<name>glob</name>
<parameters>{"pattern": "*.py"}</parameters>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "glob"


# ---------------------------------------------------------------------------
# Qwen 3 native format: <function=NAME><parameter=P>value</parameter>
# ---------------------------------------------------------------------------


def test_extract_qwen_native_single(converter: FnCallConverter):
    """Qwen 3 native tool call format."""
    text = """<tool_call>
<function=list_files>
<parameter=path>
/tmp
</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_extract_qwen_native_multi_params(converter: FnCallConverter):
    """Qwen 3 native format with multiple parameters."""
    text = """<tool_call>
<function=grep_search>
<parameter=file_type>
.py
</parameter>
<parameter=path>
/home
</parameter>
<parameter=pattern>
import
</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "grep_search"
    assert calls[0].arguments == {"file_type": ".py", "path": "/home", "pattern": "import"}


def test_extract_qwen_native_multiple_calls(converter: FnCallConverter):
    """Multiple Qwen 3 native tool calls in one response."""
    text = """<tool_call>
<function=glob>
<parameter=pattern>*.py</parameter>
</function>
</tool_call>
<tool_call>
<function=bash_exec>
<parameter=command>pwd</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "glob"
    assert calls[0].arguments == {"pattern": "*.py"}
    assert calls[1].name == "bash_exec"
    assert calls[1].arguments == {"command": "pwd"}


def test_extract_qwen_native_with_thinking(converter: FnCallConverter):
    """Qwen 3 native format with thinking tags."""
    text = """<think>
I should list files first.
</think>

<tool_call>
<function=list_files>
<parameter=path>/tmp</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"


def test_extract_qwen_native_numeric_value(converter: FnCallConverter):
    """Qwen 3 native format with numeric parameter values."""
    text = """<tool_call>
<function=read_file>
<parameter=path>/etc/hosts</parameter>
<parameter=limit>50</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments["path"] == "/etc/hosts"
    assert calls[0].arguments["limit"] == 50


def test_extract_qwen_native_boolean_value(converter: FnCallConverter):
    """Qwen 3 native format with boolean parameter values."""
    text = """<tool_call>
<function=bash_exec>
<parameter=command>ls</parameter>
<parameter=background>true</parameter>
</function>
</tool_call>"""
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments["background"] is True


# ---------------------------------------------------------------------------
# Hermes JSON-inside-XML format (Qwen 3 base/instruct)
# ---------------------------------------------------------------------------


def test_extract_hermes_json(converter: FnCallConverter):
    """Hermes format: JSON inside <tool_call> tags."""
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "SF"}


def test_extract_hermes_json_multiple(converter: FnCallConverter):
    text = (
        '<tool_call>\n{"name": "glob", "arguments": {"pattern": "*.py"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "read", "arguments": {"path": "/tmp/x.py"}}\n</tool_call>'
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0].name == "glob"
    assert calls[1].name == "read"


# ---------------------------------------------------------------------------
# Stray / unbalanced closing-tag tolerance (issue #14 — Qwen3.6 / llama.cpp drift)
#
# Runtimes emit stray, duplicated, or orphan </tool_call> closers that break the
# balanced-pair regexes. A fallback pre-pass drops spurious closers and retries,
# but ONLY when the normal parse found nothing — so working parses are untouched.
# ---------------------------------------------------------------------------


def test_stray_duplicate_closer_after_qwen(converter: FnCallConverter):
    """(1) Valid call followed by a stray duplicate </tool_call> still parses."""
    text = (
        "<tool_call>\n<function=list_files>\n<parameter=path>/tmp</parameter>\n"
        "</function>\n</tool_call>\n</tool_call>"
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_orphan_closer_before_opening(converter: FnCallConverter):
    """(2) Stray </tool_call> BEFORE the opener (leaked from a thinking block)."""
    text = (
        "</tool_call>\n<tool_call>\n<function=list_files>\n"
        "<parameter=path>/tmp</parameter>\n</function>\n</tool_call>"
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_duplicated_closer_immediately_repeated(converter: FnCallConverter):
    """(3) Duplicated closer immediately repeated (no whitespace between)."""
    text = (
        "<tool_call>\n<function=list_files>\n<parameter=path>/tmp</parameter>\n"
        "</function>\n</tool_call></tool_call>"
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"


def test_balanced_input_parses_identically(converter: FnCallConverter):
    """(4) Control: balanced input is byte-identical through the pre-pass and parses the same."""
    from localharness.provider.fn_call import _drop_spurious_toolcall_closers
    text = (
        "<tool_call>\n<function=list_files>\n<parameter=path>/tmp</parameter>\n"
        "</function>\n</tool_call>"
    )
    # Pre-pass must be a no-op on well-formed input (conservatism guarantee).
    assert _drop_spurious_toolcall_closers(text) == text
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_plain_text_with_lone_stray_closer(converter: FnCallConverter):
    """(5) Plain text with a lone stray closer and NO tool call -> zero, no crash."""
    text = "Here is my final answer.</tool_call> Nothing else to do."
    calls = converter.extract_tool_calls(text)
    assert calls == []


def test_stray_closer_between_wrapper_and_function_qwen(converter: FnCallConverter):
    """Observed drift: stray </tool_call> between the wrapper open and the function."""
    text = (
        "<tool_call>\n</tool_call>\n<function=list_files>\n"
        "<parameter=path>/tmp</parameter>\n</function>\n</tool_call>"
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_stray_closer_between_wrapper_and_json_hermes(converter: FnCallConverter):
    """Observed drift: stray </tool_call> between the wrapper open and the Hermes JSON."""
    text = (
        '<tool_call>\n</tool_call>\n'
        '{"name": "get_weather", "arguments": {"city": "SF"}}\n</tool_call>'
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "get_weather"
    assert calls[0].arguments == {"city": "SF"}


def test_stray_closer_between_wrapper_and_name_legacy(converter: FnCallConverter):
    """Observed drift: stray </tool_call> between the wrapper open and the legacy <name>."""
    text = (
        '<tool_call>\n</tool_call>\n<name>list_files</name>\n'
        '<parameters>{"path": "/tmp"}</parameters>\n</tool_call>'
    )
    calls = converter.extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"path": "/tmp"}


def test_prepass_noop_on_balanced_multi_call(converter: FnCallConverter):
    """Pre-pass leaves balanced multi-call input byte-identical (never fires on success)."""
    from localharness.provider.fn_call import _drop_spurious_toolcall_closers
    text = (
        "<tool_call>\n<function=glob>\n<parameter=pattern>*.py</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=bash_exec>\n<parameter=command>pwd</parameter>\n</function>\n</tool_call>"
    )
    assert _drop_spurious_toolcall_closers(text) == text
    assert len(converter.extract_tool_calls(text)) == 2
