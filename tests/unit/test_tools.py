"""Unit tests for tools package: base types, registry, Pydantic dispatch, and built-ins."""
import asyncio
import os
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
async def test_glob_tool_trailing_double_star_finds_nested_files(tmp_path: Path):
    """#74 exact live miss: pathlib's Path.glob() yields DIRECTORIES ONLY for a trailing
    bare '**' — '<agents>/**' listed the subdir but never the .yaml files under it."""
    from localharness.tools.builtin.glob_tool import GlobTool

    agents = tmp_path / "agents"
    agents.mkdir()
    (agents / "coder.yaml").write_text("name: coder")
    (agents / "planner.yaml").write_text("name: planner")
    tool = GlobTool()
    result = await tool.run(pattern="agents/**", base_dir=str(tmp_path))
    assert result.success is True
    assert "coder.yaml" in result.output
    assert "planner.yaml" in result.output


@pytest.mark.asyncio
async def test_glob_tool_bare_double_star_finds_files(tmp_path: Path):
    """#74: a bare '**' rooted at base_dir must include files at every depth, not just dirs."""
    from localharness.tools.builtin.glob_tool import GlobTool

    (tmp_path / "top.txt").write_text("x")
    nested = tmp_path / "d"
    nested.mkdir()
    (nested / "deep.txt").write_text("x")
    tool = GlobTool()
    result = await tool.run(pattern="**", base_dir=str(tmp_path))
    assert result.success is True
    assert "top.txt" in result.output
    assert "deep.txt" in result.output


@pytest.mark.asyncio
async def test_glob_tool_absolute_trailing_double_star_finds_files(tmp_path: Path):
    """#74: an absolute '/…/**' pattern (base_dir ignored) also matches files at depth >=1."""
    from localharness.tools.builtin.glob_tool import GlobTool

    sub = tmp_path / "nest"
    sub.mkdir()
    (sub / "found.md").write_text("x")
    tool = GlobTool()
    result = await tool.run(pattern=f"{tmp_path}/**")
    assert result.success is True
    assert "found.md" in result.output


@pytest.mark.asyncio
async def test_glob_tool_tilde_trailing_double_star_finds_files(tmp_path, monkeypatch):
    """#74: a '~/…/**' pattern expands home AND matches files at depth >=1 (the live shape)."""
    import os  # noqa: F401
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows' Path.expanduser() prefers this over HOME
    agents = tmp_path / ".localharness" / "agents"
    agents.mkdir(parents=True)
    (agents / "mine.yaml").write_text("name: mine")

    from localharness.tools.builtin.glob_tool import GlobTool

    result = await GlobTool().run(pattern="~/.localharness/agents/**")
    assert result.success is True
    assert "mine.yaml" in result.output


@pytest.mark.asyncio
async def test_glob_tool_single_level_pattern_unchanged(tmp_path: Path):
    """#74 guard: normalization must NOT touch ordinary patterns — '*.py' stays depth-0 only."""
    from localharness.tools.builtin.glob_tool import GlobTool

    (tmp_path / "top.py").write_text("x")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "deep.py").write_text("x")
    tool = GlobTool()
    result = await tool.run(pattern="*.py", base_dir=str(tmp_path))
    assert result.success is True
    assert "top.py" in result.output
    assert "deep.py" not in result.output  # single-level '*' must not recurse


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


# ---------------------------------------------------------------------------
# grep bounded-walk guards (#21): exclusions, size/binary skips, scan caps.
# Caps are module-level constants so these tests can patch them low.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grep_excludes_hidden_and_vendor_dirs_opt_in_with_flag(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("GITMATCH here\n")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "mod.py").write_text("VENVMATCH here\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("NODEMATCH here\n")

    tool = GrepTool()
    default = await tool.run(pattern="MATCH", path=str(tmp_path))
    assert default.success is True
    assert default.output == "(no matches)"  # hidden + vendor dirs pruned at walk time

    revealed = await tool.run(pattern="MATCH", path=str(tmp_path), include_hidden=True)
    assert revealed.success is True
    assert "GITMATCH" in revealed.output       # hidden dir revealed by the flag
    assert "VENVMATCH" in revealed.output       # hidden dir revealed by the flag
    assert "NODEMATCH" not in revealed.output   # non-hidden vendor dir stays excluded


@pytest.mark.asyncio
async def test_grep_skips_oversized_files(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    big = tmp_path / "big.log"
    big.write_bytes(b"filler\n" * 200_000 + b"BIGTOKEN match here\n")  # ~1.4MB > 1MB
    tool = GrepTool()
    result = await tool.run(pattern="BIGTOKEN", path=str(tmp_path))
    assert result.success is True
    assert "BIGTOKEN" not in result.output
    assert result.output == "(no matches)"


@pytest.mark.asyncio
async def test_grep_skips_binary_files(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    (tmp_path / "data.bin").write_bytes(b"BINTOKEN\x00\x00\x00payload here\n")
    tool = GrepTool()
    result = await tool.run(pattern="BINTOKEN", path=str(tmp_path))
    assert result.success is True
    assert "BINTOKEN" not in result.output
    assert result.output == "(no matches)"


@pytest.mark.asyncio
async def test_grep_file_count_cap_returns_partial_with_note(tmp_path: Path, monkeypatch):
    from localharness.tools.builtin import grep_tool as grep_mod
    from localharness.tools.builtin.grep_tool import GrepTool

    for i in range(60):
        (tmp_path / f"f{i:02d}.txt").write_text("CAPTOKEN line\n")
    monkeypatch.setattr(grep_mod, "SCAN_FILE_CAP", 5)
    tool = GrepTool()
    result = await tool.run(pattern="CAPTOKEN", path=str(tmp_path))
    assert result.success is True
    assert "scan capped" in result.output              # honest truncation note
    assert result.truncated is True  # real field since the ok() lift (#133)
    assert "CAPTOKEN" in result.output                  # found-so-far returned, never nothing
    assert result.output.count("CAPTOKEN") <= 5         # stopped at the cap


@pytest.mark.asyncio
async def test_grep_time_budget_returns_without_hang(tmp_path: Path, monkeypatch):
    from localharness.tools.builtin import grep_tool as grep_mod
    from localharness.tools.builtin.grep_tool import GrepTool

    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("TIMETOKEN\n")
    monkeypatch.setattr(grep_mod, "SCAN_TIME_BUDGET_S", 0.0)
    tool = GrepTool()
    result = await tool.run(pattern="TIMETOKEN", path=str(tmp_path))
    assert result.success is True
    assert "scan capped" in result.output       # capped note even at zero budget
    assert result.truncated is True  # real field since the ok() lift (#133)


@pytest.mark.asyncio
async def test_grep_small_tree_matches_path_line_format_unchanged(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    (tmp_path / "a.py").write_text("nope\nKEEPTOKEN yes\n")
    tool = GrepTool()
    result = await tool.run(pattern="KEEPTOKEN", path=str(tmp_path))
    assert result.success is True
    assert "KEEPTOKEN" in result.output
    import re as _re

    assert _re.search(r":\d+: ", result.output)   # path:line: content format preserved


@pytest.mark.asyncio
async def test_grep_clean_miss_reports_no_matches(tmp_path: Path):
    from localharness.tools.builtin.grep_tool import GrepTool

    (tmp_path / "a.py").write_text("alpha\nbeta\n")
    tool = GrepTool()
    result = await tool.run(pattern="ZZZ_ABSENT", path=str(tmp_path))
    assert result.success is True
    assert result.output == "(no matches)"


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
async def test_read_tool_refuses_binary_file(tmp_path: Path):
    """Live incident (2026-08-30): `read` on a SQLite .db file silently decoded raw bytes with
    errors='replace' and returned ~287,000 chars of replacement-character soup in one call —
    the offset/limit (line-count) bound never applied since binary content has almost no
    newlines. `read` must refuse binary content outright instead of dumping it as text."""
    from localharness.tools.builtin.read_tool import ReadTool

    f = tmp_path / "data.db"
    f.write_bytes(b"SQLite format 3\x00" + b"\x01\x02\x03" * 2000)
    tool = ReadTool()
    result = await tool.run(path=str(f))
    assert result.success is False
    assert result.error_type == "validation_error"
    assert "binary" in result.error.lower()
    assert "sqlite3" in result.error.lower()  # names the actual remedy for a .db file


@pytest.mark.asyncio
async def test_read_tool_binary_guard_agrees_with_grep(tmp_path: Path):
    """`read` and `grep` must agree on what counts as binary — both sniff for a NUL byte,
    so a file grep silently skips is a file read refuses outright rather than dumping."""
    from localharness.tools.builtin.grep_tool import _read_text_guarded
    from localharness.tools.builtin.read_tool import ReadTool

    f = tmp_path / "data.bin"
    f.write_bytes(b"BINTOKEN\x00\x00\x00payload here\n")
    assert _read_text_guarded(str(f)) is None  # grep's own guard skips it

    result = await ReadTool().run(path=str(f))
    assert result.success is False
    assert result.error_type == "validation_error"


@pytest.mark.asyncio
async def test_read_tool_caps_a_single_oversized_line(tmp_path: Path):
    """Defense in depth independent of the binary guard: a single enormous line (no newlines
    at all, e.g. minified output) would otherwise bypass the offset/limit (line-count) bound
    entirely. A char cap backstops ANY oversized single result, binary or not."""
    from localharness.tools.builtin.read_tool import MAX_RETURNED_CHARS, ReadTool

    f = tmp_path / "huge_line.txt"
    f.write_text("x" * (MAX_RETURNED_CHARS * 2))  # one line, well over the cap, zero newlines
    tool = ReadTool()
    result = await tool.run(path=str(f))
    assert result.success is True
    assert result.truncated is True
    assert result.original_length is not None and result.original_length > MAX_RETURNED_CHARS
    assert len(result.output) < result.original_length
    assert "truncated" in result.output.lower()


@pytest.mark.asyncio
async def test_read_tool_small_file_reports_no_truncation(tmp_path: Path):
    """Control: an ordinary small text file is untouched by the new char cap."""
    from localharness.tools.builtin.read_tool import ReadTool

    f = tmp_path / "small.txt"
    f.write_text("hello\nworld\n")
    result = await ReadTool().run(path=str(f))
    assert result.success is True
    assert result.truncated is False
    assert result.original_length is None


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
async def test_write_tool_created_message_for_new_file(tmp_path: Path):
    """#80: overwrite of a brand-new path reports 'Created … (N bytes)'."""
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    out = tmp_path / "new.txt"
    result = await tool.run(path=str(out), content="hello")
    assert result.success is True
    p = result.metadata["path"]
    assert result.output == f"Created {p} (5 bytes)"
    assert result.metadata["bytes_written"] == 5
    assert result.metadata.get("unchanged") is not True


@pytest.mark.asyncio
async def test_write_tool_overwrote_message_for_changed_content(tmp_path: Path):
    """#80: overwrite with DIFFERENT content reports the old→new byte delta."""
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    out = tmp_path / "f.txt"
    await tool.run(path=str(out), content="aaaa")          # 4 bytes
    result = await tool.run(path=str(out), content="bbbbbb")  # 6 bytes
    assert result.success is True
    p = result.metadata["path"]
    assert result.output == f"Overwrote {p} (was 4 bytes, now 6 bytes)"
    assert result.metadata["bytes_written"] == 6
    assert result.metadata.get("unchanged") is not True
    assert out.read_text() == "bbbbbb"


@pytest.mark.asyncio
async def test_write_tool_no_change_when_identical(tmp_path: Path):
    """#80: overwrite with byte-identical content is a no-op STOP signal (unchanged=True),
    not another 'success' line the model reacts to by rewriting the file again."""
    from localharness.tools.builtin.write_tool import WriteTool

    tool = WriteTool()
    out = tmp_path / "f.txt"
    await tool.run(path=str(out), content="same content")
    result = await tool.run(path=str(out), content="same content")
    assert result.success is True
    p = result.metadata["path"]
    n = len("same content".encode())
    assert result.output == (
        f"No change: {p} already contains exactly this content ({n} bytes). "
        "The file is already written — do not rewrite it; take the next step."
    )
    assert result.metadata["unchanged"] is True
    assert result.metadata["bytes_written"] == n
    assert result.metadata["path"] == p


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


async def _wait_gone(pid: int, tries: int = 300) -> bool:
    """Poll until `pid` is neither alive NOR a zombie. os.kill(pid, 0) still succeeds for an
    unreaped zombie, so ProcessLookupError proves killed AND reaped. On Windows os.kill(pid, 0)
    is NOT a probe — any signal but CTRL_* events TerminateProcess-es the target — so the
    liveness check goes through tasklist there (_pid_alive below)."""
    import os
    import subprocess
    for _ in range(tries):
        if os.name == "nt":
            if not _pid_alive(pid):
                return True
        else:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
        await asyncio.sleep(0.01)
    if os.name == "nt":  # never leak a stray sleep out of a RED run
        subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    else:
        os.kill(pid, 9)
    return False


@pytest.mark.asyncio
async def test_bash_exec_tool_cancelled_kills_process(tmp_path):
    """#153: a cancelled turn must not abandon the child. CancelledError is a BaseException, so it
    sailed past the tool's `except asyncio.TimeoutError` — the child kept running with its pipes
    open, one leaked process per mid-turn Ctrl+C (shipped since v0.5.0).

    `exec sleep` makes the harness's DIRECT child be the sleep itself, so the pid it writes with
    `$$` is the one the tool holds. Identity is by pid — never a pgrep/pkill pattern match, which
    happily matches the test runner's own cmdline."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    pidfile = tmp_path / "child.pid"
    # as_posix(): bash strips the backslashes of a raw Windows path (see the dash test above).
    # /proc/$$/winpid is the Windows pid under MSYS; POSIX falls back to $$ itself.
    task = asyncio.ensure_future(
        BashExecTool().run(
            command=f"(cat /proc/$$/winpid 2>/dev/null || echo $$) > {pidfile.as_posix()}; "
                    "exec sleep 30",
            timeout_s=30,
        )
    )
    for _ in range(300):  # let the child spawn and publish its pid
        await asyncio.sleep(0.01)
        if pidfile.exists() and pidfile.read_text().strip():
            break
    assert pidfile.exists() and pidfile.read_text().strip(), "the child never started"
    pid = int(pidfile.read_text())

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert await _wait_gone(pid), f"cancelled bash_exec left child {pid} alive (#153)"


class _InnerTimeoutTool(Tool):
    """Mirrors bash_exec's shape: a per-call timeout_s with the tool's OWN inner
    wait_for + cleanup path. The instance timeout_s is SMALLER than the call's."""
    timeout_s = 0.2

    def __init__(self) -> None:
        self.inner_cleanup_ran = False

    def info(self) -> ToolSchema:
        return ToolSchema(name="inner_timeout", description="x",
                          parameters={"type": "object", "properties": {}, "required": []})

    async def _execute(self, timeout_s: float = 0.5) -> ToolResult:
        try:
            await asyncio.wait_for(asyncio.sleep(30), timeout=timeout_s)
        except asyncio.TimeoutError:
            self.inner_cleanup_ran = True  # the proc.kill-equivalent cleanup path
            return self.err(f"inner timeout after {timeout_s}s", error_type="timeout_error")
        return self.ok("done")


@pytest.mark.asyncio
async def test_outer_timeout_never_beats_inner_cleanup_path():
    """Timeout-inversion regression (36-08 pre-flight MAJOR 7): when a call passes its own
    timeout_s, the base-class outer wait_for must be sized ABOVE it — otherwise the outer
    cancel fires first, the inner kill/cleanup path never runs, and (for bash_exec) the
    subprocess is orphaned. The inner cleanup must always win the race."""
    tool = _InnerTimeoutTool()
    result = await tool.run(timeout_s=0.5)  # call timeout > instance timeout_s=0.2
    assert tool.inner_cleanup_ran is True, "outer wait_for cancelled _execute before inner cleanup"
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert "inner timeout" in (result.error or "")  # the INNER path produced the result


@pytest.mark.asyncio
async def test_bash_exec_tool_captures_stderr():
    from localharness.tools.builtin.bash_tool import BashExecTool

    tool = BashExecTool()
    result = await tool.run(command="echo errout >&2")
    assert result.success is True
    assert "errout" in result.output


@pytest.mark.asyncio
async def test_bash_exec_runs_under_bash_not_dash(tmp_path: Path):
    """#79: bash_exec must launch bash, not the platform /bin/sh (dash on Ubuntu).
    `mkdir -p {a,b}` is a bash brace expansion — under dash it makes ONE literal
    directory named '{a,b}' instead of 'a' and 'b'. Fails on any dash host until the
    subprocess passes executable=bash."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    tool = BashExecTool()
    # as_posix(): a raw Windows str(tmp_path) embeds backslashes that bash strips as escapes,
    # mangling the target dir; the forward-slash form is what git-bash/MSYS expects and is
    # identical to str() on POSIX.
    result = await tool.run(command=f"mkdir -p {tmp_path.as_posix()}/{{a,b}}")
    assert result.success is True
    assert (tmp_path / "a").is_dir()
    assert (tmp_path / "b").is_dir()
    # dash leaves a literal brace directory ('{a,b}'); bash expands it away.
    assert not any("{" in p.name for p in tmp_path.iterdir())


def test_find_bash_prefers_git_bin_wrapper_and_skips_wsl_stubs(monkeypatch, tmp_path: Path):
    """Windows discovery order. `Git\\usr\\bin\\bash.exe` launched directly inherits the
    harness PATH (no /usr/bin from PowerShell) so coreutils vanish; the `Git\\bin\\bash.exe`
    wrapper sets PATH itself and must win. Both WSL stubs (System32, Store WindowsApps
    alias) are rejected even when `which` returns them. Simulated on every OS."""
    from localharness.tools.builtin import bash_tool

    wrapper = tmp_path / "Git" / "bin" / "bash.exe"
    inner = tmp_path / "Git" / "usr" / "bin" / "bash.exe"
    for exe in (wrapper, inner):
        exe.parent.mkdir(parents=True)
        exe.write_bytes(b"")
    monkeypatch.setattr(bash_tool.os, "name", "nt")
    monkeypatch.delenv("LOCALHARNESS_BASH", raising=False)
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.delenv("ProgramFiles(x86)", raising=False)
    monkeypatch.setenv("LocalAppData", str(tmp_path / "absent"))
    stub = str(tmp_path / "WindowsApps" / "bash.exe")
    monkeypatch.setattr(bash_tool.shutil, "which", lambda _name: stub)

    assert bash_tool._find_bash() == str(wrapper)


@pytest.mark.asyncio
async def test_bash_exec_coreutils_resolve_without_git_on_path(monkeypatch):
    """Live regression: from a PowerShell-started harness every `mkdir -p` came back
    '/usr/bin/bash: line 1: mkdir: command not found' — and was reported ✓. Strip Git
    entries from PATH so discovery (not the caller's shell) has to supply coreutils; a
    no-op on POSIX, where the test still pins that coreutils resolve under bash_exec."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    monkeypatch.delenv("LOCALHARNESS_BASH", raising=False)
    stripped = [p for p in os.environ.get("PATH", "").split(os.pathsep) if "git" not in p.lower()]
    monkeypatch.setenv("PATH", os.pathsep.join(stripped))

    result = await BashExecTool().run(command="command -v mkdir")
    assert result.success is True, result.error
    assert "mkdir" in result.output


@pytest.mark.asyncio
async def test_bash_exec_nonzero_exit_is_failure_and_keeps_output():
    """A non-zero exit used to be success=True with the code tucked in metadata: the
    terminal showed ✓ and the model read the result as done. It is a failure now, and
    because the loop forwards .error (not .output) on failure, the command's own output
    must travel inside the error message."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    result = await BashExecTool().run(command="echo boom >&2; exit 3")
    assert result.success is False
    assert result.error_type == "execution_error"
    assert result.metadata.get("exit_code") == 3
    assert "boom" in result.output
    assert "exit code 3" in (result.error or "")
    assert "boom" in (result.error or "")


@pytest.mark.asyncio
async def test_bash_exec_stdin_is_never_the_terminal():
    """A command that reads stdin gets EOF at once (stdin=DEVNULL) — never the harness's
    own terminal. Observed live: `cmd /c "…"` under git-bash opened an interactive cmd on
    the inherited stdin and sat there for the full 65s timeout."""
    from localharness.tools.builtin.bash_tool import BashExecTool

    result = await BashExecTool().run(command="cat; echo done", timeout_s=10)
    assert result.success is True, result.error
    assert "done" in result.output


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.asyncio
async def test_bash_exec_timeout_kills_process_tree_promptly(tmp_path: Path):
    """The inner timeout must kill the whole tree and return on ITS path. A grandchild
    holding the stdout pipe (`sleep 30 &`) kept the post-kill communicate() blocked, so
    the base-class outer timeout fired instead (the inversion) and the grandchild lived on.
    On Windows `taskkill /T` does not reach git-bash's forked children (verified), hence the
    job object — so the test checks the grandchild is really dead, not just that we returned."""
    import time
    from localharness.tools.builtin.bash_tool import BashExecTool

    pidfile = (tmp_path / "child.pid").as_posix()
    # /proc/<pid>/winpid is the Windows pid under MSYS; POSIX falls back to $! itself.
    command = f'sleep 30 & (cat /proc/$!/winpid 2>/dev/null || echo $!) > "{pidfile}"; wait'
    t0 = time.monotonic()
    result = await BashExecTool().run(command=command, timeout_s=1)
    elapsed = time.monotonic() - t0
    assert result.success is False
    assert result.error_type == "timeout_error"
    assert "Command timed out" in result.output  # the inner path, not base.run's outer one
    assert elapsed < 10, f"took {elapsed:.1f}s — the tree was not killed"
    child = int((tmp_path / "child.pid").read_text().strip())
    for _ in range(20):
        if not _pid_alive(child):
            break
        await asyncio.sleep(0.25)
    assert not _pid_alive(child), f"grandchild {child} survived the timeout"


@pytest.mark.asyncio
async def test_register_builtin_tools_registers_all():
    from localharness.tools.builtin import register_builtin_tools

    reg = ToolRegistry()
    await register_builtin_tools(reg)
    names = set(reg._tools["global"].keys())
    assert {"glob", "grep", "read", "write", "edit", "bash_exec",
            "web_search", "web_fetch", "web_page_query", "chunk", "load_document"} == names


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
        encoding = "utf-8"
        def raise_for_status(self): pass
        async def aiter_bytes(self):
            yield page_text.encode("utf-8")

    class _Stream:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return _Resp()
        def stream(self, method, url, **k): return _Stream()

    monkeypatch.setattr(web_tool.httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_web_fetch_short_page_no_clip(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    from localharness.tools.builtin.web_tool import _UNTRUSTED

    _fake_httpx_client(monkeypatch, "short page")
    result = await WebFetchTool().run(url="https://example.test/page")
    assert result.success is True
    assert result.output.startswith(_UNTRUSTED + "short page")
    assert "fetch_id=" in result.output  # full page retained + queryable even when fully shown
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
    # window itself is 5000 chars of body (after the untrusted-content banner)
    assert ("x" * 5000) in result.output


@pytest.mark.asyncio
async def test_web_fetch_resumes_from_start_index(monkeypatch):
    from localharness.tools.builtin.web_tool import WebFetchTool

    page = "a" * 5000 + "b" * 3000
    _fake_httpx_client(monkeypatch, page)
    result = await WebFetchTool().run(
        url="https://example.test/page", max_chars=5000, start_index=5000
    )
    assert result.success is True
    assert "chars 5000-8000 of 8000" in result.output
    assert "b" * 3000 in result.output
    assert "start_index=" not in result.output  # last window: no further-read cursor
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
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows' Path.expanduser() prefers this over HOME
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.txt").write_text("tilde works")

    from localharness.tools.builtin.read_tool import ReadTool
    r = await ReadTool().run(path="~/notes/a.txt")
    assert r.success and "tilde works" in r.output

    from localharness.tools.builtin.glob_tool import GlobTool
    g = await GlobTool().run(pattern="~/notes/*.txt")
    assert g.success and "a.txt" in g.output


def test_write_tool_description_steers_large_files_to_smaller_writes():
    """#77 belt-and-suspenders: the write tool's own description must nudge large files
    toward several smaller calls (first write, then mode=append) so an oversized argument
    isn't cut off at the output-token limit."""
    from localharness.tools.builtin.write_tool import WriteTool
    desc = WriteTool().info().description.lower()
    assert "append" in desc
    assert "output-token" in desc or "cut off" in desc


def test_resolve_user_path_maps_posix_tmp_to_os_tempdir():
    """Windows: `/tmp/...` must land where git-bash mounts /tmp (%TEMP%), not `<drive>\tmp` —
    otherwise the write tool and bash_exec operate on two different trees (live: write landed
    C:\tmp, `python3 /tmp/...` under bash couldn't find it, 3 identical retry cycles)."""
    import os
    import tempfile
    from localharness.tools.builtin.paths import resolve_user_path

    if os.name != "nt":
        import pytest as _pytest
        _pytest.skip("Windows-only remap; POSIX resolution is pass-through")
    tmp = Path(tempfile.gettempdir()).resolve()
    assert resolve_user_path("/tmp/x/y.txt") == tmp / "x" / "y.txt"
    assert resolve_user_path("/tmp") == tmp
    assert resolve_user_path("C:/Windows") == Path("C:/Windows").resolve()
    assert resolve_user_path("some/rel/dir").is_absolute()


def test_find_bash_rejects_wsl_stub(monkeypatch):
    """PowerShell PATH order resolves `bash` to the System32 WSL stub, which prints a
    UTF-16LE error without a distro (observed live: mojibake observations, stuck loop).
    _find_bash must skip it and search git-bash locations instead."""
    import os as _os
    from localharness.tools.builtin import bash_tool

    if _os.name != "nt":
        import pytest as _pytest
        _pytest.skip("Windows-only resolution rules")
    monkeypatch.delenv("LOCALHARNESS_BASH", raising=False)
    monkeypatch.setattr(bash_tool.shutil, "which", lambda _: r"C:\Windows\System32\bash.exe")
    monkeypatch.setattr(bash_tool.os.path, "isfile", lambda p: False)
    assert bash_tool._find_bash() is None  # stub rejected, no git-bash found -> clear error path

    monkeypatch.setattr(
        bash_tool.os.path, "isfile", lambda p: p.endswith(r"Git\usr\bin\bash.exe")
    )
    found = bash_tool._find_bash()
    assert found and found.endswith(r"Git\usr\bin\bash.exe")

    monkeypatch.setenv("LOCALHARNESS_BASH", r"D:\custom\bash.exe")
    assert bash_tool._find_bash() == r"D:\custom\bash.exe"


def test_decode_output_handles_utf16le():
    from localharness.tools.builtin.bash_tool import _decode_output

    utf16 = "Windows Subsystem for Linux has no installed distributions.".encode("utf-16-le")
    out = _decode_output(utf16)
    assert "\x00" not in out
    assert "no installed distributions" in out
    assert _decode_output("plain utf-8 ✓".encode("utf-8")) == "plain utf-8 ✓"


def test_ok_lifts_truncated_onto_the_real_field():
    """#133 critic finding: Tool.ok(..., truncated=True) buried the flag in metadata, so
    grep's limit-capped results published truncated=False. ok() now lifts truncated /
    original_length onto the ToolResult fields the audit trail actually reads."""
    r = _EchoTool().ok("... (limit 200 reached)", match_count=200, truncated=True)
    assert r.truncated is True
    assert r.original_length is None  # unknown at producer — honest, not invented
    assert r.metadata == {"match_count": 200}  # lifted keys don't linger in metadata


class _HugeErrTool(Tool):
    def info(self) -> ToolSchema:
        return ToolSchema(
            name="hugeerr",
            description="Always fails with an enormous error payload.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(output="", success=False, error="x" * 60_000,
                          error_type="execution_error")


@pytest.mark.asyncio
async def test_registry_caps_error_payloads_too():
    """#133 critic finding: .error was never capped at dispatch — MCP tools mirror whole
    server responses into it, and the old event slice bounded it only by accident."""
    reg = ToolRegistry(result_size_cap_chars=50_000)
    await reg.register(_HugeErrTool(), scope="global")
    config = ToolConfig(inherit=["global"])
    result = await reg.dispatch("hugeerr", {}, "agent-1", "div-1", config)
    assert result.success is False
    assert len(result.error) <= 50_000 + 100
    assert "[error truncated: 60,000 chars total]" in result.error
