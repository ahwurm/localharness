"""Unit tests for tools package: base types, registry, Pydantic dispatch, and built-ins."""
import asyncio
import pytest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from localharness.config.models import ToolConfig
from localharness.tools import (
    Tool,
    ToolProtocol,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    ToolParameter,
    ToolVetoed,
)


# ---------------------------------------------------------------------------
# Minimal test tool
# ---------------------------------------------------------------------------


class _EchoTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="echo",
            description="Returns input as output.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo."},
                    "count": {"type": "integer", "description": "Optional repeat.", "default": 1},
                },
                "required": ["message"],
            },
        )

    async def _execute(self, message: str, count: int = 1) -> ToolResult:
        return self.ok(message * count, chars=len(message) * count)


class _SlowTool(Tool):
    timeout_s = 0.05  # 50ms

    def info(self) -> ToolSchema:
        return ToolSchema(
            name="slow",
            description="Sleeps forever.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        await asyncio.sleep(10)
        return self.ok("done")


class _BoomTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="boom",
            description="Always raises.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# Task 1: Tool interface tests
# ---------------------------------------------------------------------------


def test_tool_abc_info_returns_schema():
    tool = _EchoTool()
    schema = tool.info()
    assert isinstance(schema, ToolSchema)
    assert schema.name == "echo"
    assert schema.description


def test_tool_protocol_runtime_checkable():
    tool = _EchoTool()
    assert isinstance(tool, ToolProtocol)


def test_tool_protocol_rejects_plain_object():
    assert not isinstance(object(), ToolProtocol)


@pytest.mark.asyncio
async def test_tool_ok_helper():
    tool = _EchoTool()
    result = tool.ok("hello", foo="bar")
    assert result.success is True
    assert result.output == "hello"
    assert result.metadata["foo"] == "bar"


@pytest.mark.asyncio
async def test_tool_err_helper():
    tool = _EchoTool()
    result = tool.err("bad thing", error_type="execution_error")
    assert result.success is False
    assert result.error == "bad thing"
    assert result.error_type == "execution_error"
    assert result.output == ""


@pytest.mark.asyncio
async def test_tool_run_returns_tool_result():
    tool = _EchoTool()
    result = await tool.run(message="hi", count=2)
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "hihi"


@pytest.mark.asyncio
async def test_tool_run_timeout():
    tool = _SlowTool()
    result = await tool.run()
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_tool_run_catches_exception():
    tool = _BoomTool()
    result = await tool.run()
    assert result.success is False
    assert result.error_type == "execution_error"
    assert "kaboom" in result.error


# ---------------------------------------------------------------------------
# Task 1: ToolRegistry tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_register_global():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    assert "echo" in reg._tools["global"]


@pytest.mark.asyncio
async def test_registry_register_duplicate_raises():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    with pytest.raises(ValueError, match="already registered"):
        await reg.register(_EchoTool(), scope="global")


@pytest.mark.asyncio
async def test_registry_register_division_requires_id():
    reg = ToolRegistry()
    with pytest.raises(ValueError, match="division_id required"):
        await reg.register(_EchoTool(), scope="division")


@pytest.mark.asyncio
async def test_registry_register_type_error_on_bad_tool():
    reg = ToolRegistry()
    with pytest.raises(TypeError):
        await reg.register(object(), scope="global")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_registry_get_tools_for_agent_global():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    tools = reg.get_tools_for_agent("agent-1", "div-1", config)
    assert "echo" in tools
    assert isinstance(tools["echo"], ToolSchema)


@pytest.mark.asyncio
async def test_registry_get_tools_deny_removes():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    config = ToolConfig(inherit=["global"], deny=["echo"])
    tools = reg.get_tools_for_agent("agent-1", "div-1", config)
    assert "echo" not in tools


@pytest.mark.asyncio
async def test_registry_get_tools_add_force_includes():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    # Start with empty inherit, but force-add echo
    config = ToolConfig(inherit=[], add=["echo"])
    tools = reg.get_tools_for_agent("agent-1", "div-1", config)
    assert "echo" in tools


@pytest.mark.asyncio
async def test_registry_dispatch_valid_args():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("echo", {"message": "hello"}, "agent-1", "div-1", config)
    assert result.success is True
    assert result.output == "hello"
    assert result.duration_ms is not None


@pytest.mark.asyncio
async def test_registry_dispatch_invalid_args_validation_error():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    # message is required; passing integer for message
    result = await reg.dispatch("echo", {}, "agent-1", "div-1", config)
    assert result.success is False
    assert result.error_type == "validation_error"


@pytest.mark.asyncio
async def test_registry_dispatch_not_found():
    reg = ToolRegistry()
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("nonexistent", {}, "agent-1", "div-1", config)
    assert result.success is False
    assert result.error_type == "not_found"


@pytest.mark.asyncio
async def test_registry_dispatch_truncates_large_output():
    class _BigTool(Tool):
        def info(self) -> ToolSchema:
            return ToolSchema(
                name="big",
                description="Returns huge output.",
                parameters={"type": "object", "properties": {}, "required": []},
            )

        async def _execute(self, **kwargs: Any) -> ToolResult:
            return self.ok("x" * 100_000)

    reg = ToolRegistry(result_size_cap_chars=50_000)
    await reg.register(_BigTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("big", {}, "agent-1", "div-1", config)
    assert result.truncated is True
    assert len(result.output) == 50_000
    assert result.original_length == 100_000


@pytest.mark.asyncio
async def test_registry_dispatch_sets_duration_ms():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("echo", {"message": "hi"}, "agent-1", "div-1", config)
    assert result.duration_ms is not None
    assert result.duration_ms >= 0


@pytest.mark.asyncio
async def test_registry_dispatch_pre_hook_veto():
    reg = ToolRegistry()
    await reg.register(_EchoTool(), scope="global")

    def veto_hook(**kwargs: Any) -> None:
        raise ToolVetoed("not allowed")

    reg.register_pre_hook(veto_hook)
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("echo", {"message": "hi"}, "agent-1", "div-1", config)
    assert result.success is False
    assert result.error_type == "permission_denied"
    assert "not allowed" in result.error


def test_build_validator_model_required_and_optional():
    from localharness.tools.registry import _build_validator_model

    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Required name."},
            "count": {"type": "integer", "description": "Optional count.", "default": 5},
        },
        "required": ["name"],
    }
    model_cls = _build_validator_model("test_tool", parameters)

    # Required field present
    instance = model_cls(name="alice")
    assert instance.name == "alice"
    assert instance.count == 5  # default

    # Required field missing raises
    with pytest.raises(Exception):
        model_cls()


# ---------------------------------------------------------------------------
# Task 2: Built-in tools tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_tool_info_schema():
    from localharness.tools.builtin.glob_tool import GlobTool

    tool = GlobTool()
    schema = tool.info()
    assert schema.name == "glob"
    props = schema.parameters["properties"]
    assert "pattern" in props
    assert "base_dir" in props
    assert "limit" in props
    assert "pattern" in schema.parameters.get("required", [])


@pytest.mark.asyncio
async def test_glob_tool_finds_files(tmp_path: Path):
    from localharness.tools.builtin.glob_tool import GlobTool

    (tmp_path / "a.py").write_text("# a")
    (tmp_path / "b.py").write_text("# b")
    (tmp_path / "c.txt").write_text("# c")
    tool = GlobTool()
    result = await tool.run(pattern="*.py", base_dir=str(tmp_path))
    assert result.success is True
    assert "a.py" in result.output
    assert "b.py" in result.output
    assert "c.txt" not in result.output


@pytest.mark.asyncio
async def test_glob_tool_nonexistent_dir():
    from localharness.tools.builtin.glob_tool import GlobTool

    tool = GlobTool()
    result = await tool.run(pattern="*.py", base_dir="/nonexistent/path/xyz")
    assert result.success is False
    assert "does not exist" in result.error


@pytest.mark.asyncio
async def test_grep_tool_finds_matching_lines(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    f = tmp_path / "code.py"
    f.write_text("def test_foo():\n    pass\ndef bar():\n    pass\n")
    tool = GrepTool()
    result = await tool.run(pattern="def test", path=str(tmp_path))
    assert result.success is True
    assert "def test_foo" in result.output


@pytest.mark.asyncio
async def test_grep_tool_invalid_regex():
    from localharness.tools.builtin.grep_tool import GrepTool

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tool = GrepTool()
        result = await tool.run(pattern="[invalid", path=d)
        assert result.success is False
        assert result.error_type == "validation_error"


@pytest.mark.asyncio
async def test_grep_tool_context_lines(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    f = tmp_path / "file.txt"
    f.write_text("line1\nline2\nTARGET\nline4\nline5\n")
    tool = GrepTool()
    result = await tool.run(pattern="TARGET", path=str(f), context_lines=1)
    assert result.success is True
    assert "line2" in result.output
    assert "line4" in result.output


@pytest.mark.asyncio
async def test_read_tool_returns_numbered_lines(tmp_path: Path):
    from localharness.tools.builtin.read_tool import ReadTool

    f = tmp_path / "file.txt"
    f.write_text("alpha\nbeta\ngamma\n")
    tool = ReadTool()
    result = await tool.run(path=str(f))
    assert result.success is True
    assert "1\talpha" in result.output
    assert "2\tbeta" in result.output
    assert "3\tgamma" in result.output


@pytest.mark.asyncio
async def test_read_tool_not_found():
    from localharness.tools.builtin.read_tool import ReadTool

    tool = ReadTool()
    result = await tool.run(path="/nonexistent/path/file.txt")
    assert result.success is False
    assert result.error_type == "not_found"


@pytest.mark.asyncio
async def test_read_tool_offset_and_limit(tmp_path: Path):
    from localharness.tools.builtin.read_tool import ReadTool

    f = tmp_path / "file.txt"
    f.write_text("line1\nline2\nline3\nline4\nline5\n")
    tool = ReadTool()
    result = await tool.run(path=str(f), offset=2, limit=2)
    assert result.success is True
    assert "2\tline2" in result.output
    assert "3\tline3" in result.output
    assert "line1" not in result.output
    assert "line4" not in result.output


@pytest.mark.asyncio
async def test_write_tool_creates_file(tmp_path: Path):
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    out = tmp_path / "output.txt"
    result = await tool.run(path=str(out), content="hello world")
    assert result.success is True
    assert out.read_text() == "hello world"
    assert "bytes" in result.output


@pytest.mark.asyncio
async def test_write_tool_append_mode(tmp_path: Path):
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    out = tmp_path / "output.txt"
    out.write_text("first\n")
    result = await tool.run(path=str(out), content="second\n", mode="append")
    assert result.success is True
    assert out.read_text() == "first\nsecond\n"


@pytest.mark.asyncio
async def test_write_tool_blocks_env_files(tmp_path: Path):
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    result = await tool.run(path=str(tmp_path / ".env"), content="SECRET=abc")
    assert result.success is False
    assert result.error_type == "permission_denied"


@pytest.mark.asyncio
async def test_bash_exec_tool_runs_command():
    from localharness.tools.builtin.bash_tool import BashExecTool

    tool = BashExecTool()
    result = await tool.run(command="echo hello")
    assert result.success is True
    assert "hello" in result.output
    assert result.metadata.get("exit_code") == 0


@pytest.mark.asyncio
async def test_bash_exec_tool_timeout():
    from localharness.tools.builtin.bash_tool import BashExecTool

    tool = BashExecTool()
    result = await tool.run(command="sleep 10", timeout_s=0.1)
    assert result.success is False
    assert result.error_type == "timeout_error"


@pytest.mark.asyncio
async def test_bash_exec_tool_captures_stderr():
    from localharness.tools.builtin.bash_tool import BashExecTool

    tool = BashExecTool()
    result = await tool.run(command="echo errout >&2")
    assert result.success is True
    assert "errout" in result.output


@pytest.mark.asyncio
async def test_register_builtin_tools_registers_all():
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    names = set(reg._tools["global"].keys())
    assert {"glob", "grep", "read", "write", "edit", "bash_exec", "web_search", "web_fetch"} == names


# ---------------------------------------------------------------------------
# web_fetch pagination
# ---------------------------------------------------------------------------


def _fake_httpx_client(monkeypatch, page_text: str):
    """Patch web_tool's httpx.AsyncClient to return a fixed text/plain body."""
    from localharness.tools.builtin import web_tool

    class _Resp:
        text = page_text
        headers = {"content-type": "text/plain"}
        url = "https://example.test/page"
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return _Resp()

    monkeypatch.setattr(web_tool.httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_web_fetch_short_page_no_clip(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    _fake_httpx_client(monkeypatch, "short page")
    result = await WebFetchTool().run(url="https://example.test/page")
    assert result.success is True
    assert result.output == "short page"
    assert not result.truncated


@pytest.mark.asyncio
async def test_web_fetch_clips_with_cursor_notice(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    _fake_httpx_client(monkeypatch, "x" * 12000)
    result = await WebFetchTool().run(url="https://example.test/page", max_chars=5000)
    assert result.success is True
    assert result.truncated is True
    assert "start_index=5000" in result.output
    assert result.metadata["next_start_index"] == 5000
    # window itself is 5000 chars of body
    assert result.output.startswith("x" * 100)


@pytest.mark.asyncio
async def test_web_fetch_resumes_from_start_index(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    page = "a" * 5000 + "b" * 3000
    _fake_httpx_client(monkeypatch, page)
    result = await WebFetchTool().run(
        url="https://example.test/page", max_chars=5000, start_index=5000
    )
    assert result.success is True
    assert "[chars 5000-8000 of 8000]" in result.output
    assert "b" * 3000 in result.output
    assert "start_index=" not in result.output.split("]", 1)[1]  # no further-read notice
    assert result.truncated is False
    assert result.metadata["next_start_index"] is None


@pytest.mark.asyncio
async def test_web_fetch_start_index_past_end_errors(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    _fake_httpx_client(monkeypatch, "tiny")
    result = await WebFetchTool().run(url="https://example.test/page", start_index=999)
    assert result.success is False
    assert "past the end" in result.error


@pytest.mark.asyncio
async def test_file_tools_expand_tilde(tmp_path, monkeypatch):
    """Models routinely pass ~ paths (observed live: read + glob both failed on them)."""
    import os
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text("tilde works")

    from localharness.tools.builtin.read_tool import ReadTool
    r = await ReadTool().run(path="~/notes/a.txt")
    assert r.success and "tilde works" in r.output

    from localharness.tools.builtin.glob_tool import GlobTool
    g = await GlobTool().run(pattern="~/notes/*.txt")
    assert g.success and "a.txt" in g.output
