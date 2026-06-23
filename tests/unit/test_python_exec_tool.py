"""Unit tests for PythonExecTool — the stateful REPL (re-pointed as the trusted cruncher exec)."""
import pytest

from localharness.tools.base import ToolResult
from localharness.tools.builtin.python_tool import PythonExecTool


@pytest.mark.asyncio
async def test_stdout_captured():
    tool = PythonExecTool()
    r = await tool.run(code="print('hello', 1 + 1)")
    assert isinstance(r, ToolResult)
    assert r.success is True
    assert "hello 2" in r.output


@pytest.mark.asyncio
async def test_state_persists_across_calls():
    tool = PythonExecTool()
    await tool.run(code="x = 41")
    r = await tool.run(code="print(x + 1)")
    assert r.success is True
    assert "42" in r.output


@pytest.mark.asyncio
async def test_imports_persist_across_calls():
    tool = PythonExecTool()
    await tool.run(code="import re")
    r = await tool.run(code=r"print(re.findall(r'\d+', 'a1b22'))")
    assert r.success is True
    assert "['1', '22']" in r.output


@pytest.mark.asyncio
async def test_seeded_ctx_namespace():
    """A caller seeds the input as `ctx`; the model reads it with code."""
    tool = PythonExecTool(namespace={"ctx": "the vault access code is MAGIC-7731"})
    r = await tool.run(code=r"import re; print(re.search(r'MAGIC-\d+', ctx).group())")
    assert r.success is True
    assert "MAGIC-7731" in r.output


@pytest.mark.asyncio
async def test_exception_surfaced_as_traceback():
    """A code error is returned as a traceback (success=True) so the model can self-correct."""
    tool = PythonExecTool()
    r = await tool.run(code="1 / 0")
    assert r.success is True  # the TOOL ran fine; the code raised
    assert "ZeroDivisionError" in r.output


@pytest.mark.asyncio
async def test_no_output_placeholder():
    tool = PythonExecTool()
    r = await tool.run(code="y = 5")
    assert r.success is True
    assert r.output == "(no output)"


@pytest.mark.asyncio
async def test_two_instances_have_isolated_namespaces():
    a = PythonExecTool()
    b = PythonExecTool()
    await a.run(code="secret = 1")
    r = await b.run(code="print('secret' in dir())")
    assert "False" in r.output
