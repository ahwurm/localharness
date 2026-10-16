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


# --- WR-05: a child python_exec spawns carries the agent-command marker ---------------------------

_PROBE = "import subprocess, sys; print(subprocess.run([sys.executable, '-c', " \
         "'import os; print(os.environ.get(\"LOCALHARNESS_COMMIT\"))'], capture_output=True, text=True).stdout.strip())"


@pytest.mark.asyncio
async def test_wr05_a_child_python_exec_spawns_inherits_the_marker_and_the_process_is_left_clean(monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    import os
    r = await PythonExecTool().run(code=_PROBE)
    assert r.success is True and r.output.strip() == "1"
    assert "LOCALHARNESS_COMMIT" not in os.environ


@pytest.mark.asyncio
async def test_wr05_a_child_makes_the_cli_say_not_a_local_operator(monkeypatch):
    """The CLI's own question, asked in a child python_exec spawned: the marker alone decides it
    (a tty cannot be faked here, so both terminal checks are patched in the child's code)."""
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    child = (
        "import localharness.cli.workspace as w\n"
        "w._stdin_is_a_terminal = lambda: True\n"
        "w._stdout_is_a_terminal = lambda: True\n"
        "print(w.local_operator(False))"
    )
    code = (
        "import subprocess, sys\n"
        f"print(subprocess.run([sys.executable, '-c', {child!r}], capture_output=True, text=True).stdout.strip())\n"
    )
    assert (await PythonExecTool().run(code=code)).output.strip() == "False"
    import subprocess
    import sys
    bare = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True).stdout.strip()
    assert bare == "True"  # the same child without the marker is a local operator: the marker is what matters


@pytest.mark.asyncio
async def test_wr05_a_marker_that_was_already_set_is_never_removed(monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    import os
    await PythonExecTool().run(code="pass")
    assert os.environ["LOCALHARNESS_COMMIT"] == "1"


@pytest.mark.asyncio
async def test_wr05_overlapping_calls_keep_the_marker_until_the_last_one_ends(monkeypatch):
    import asyncio
    import os
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    slow = PythonExecTool()
    quick = PythonExecTool()
    both = asyncio.gather(
        slow.run(code="import time, os; time.sleep(0.4); print(os.environ.get('LOCALHARNESS_COMMIT'))"),
        quick.run(code="pass"))
    r_slow, _ = await both
    assert r_slow.output.strip() == "1"
    assert "LOCALHARNESS_COMMIT" not in os.environ
