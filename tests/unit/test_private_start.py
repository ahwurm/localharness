"""`start --private`: its refusals, co-author off, state in the run folder and the texts that name the private memory."""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from localharness.agent.context import ContentStore
from localharness.cli import start_cmd
from localharness.core import run_folder
from localharness.core.run_folder import RunFolder, begin, end
from tests.unit.test_web_fetch_guard import _serve, _text, public_web

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")


@pytest.fixture
def cfg(tmp_path):
    d = tmp_path / "cfg"
    d.mkdir()
    (d / "config.yaml").write_text("{}")
    return d


@pytest.fixture
def private_run(cfg):
    run = begin(cfg, private=True)
    yield run
    end(run)


@pytest.fixture
def started(monkeypatch):
    calls: list[tuple] = []

    async def fake(*args, **kw):
        calls.append((args, kw))

    monkeypatch.setattr(start_cmd, "_start_async", fake)
    monkeypatch.setattr(start_cmd, "_models_that_arrived", lambda *a, **k: None)
    return calls


def invoke(*args: str):
    app = typer.Typer()
    app.command()(start_cmd.start_app)
    return CliRunner().invoke(app, list(args))


def no_run_made(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("begin must not run before the refusal")
    monkeypatch.setattr(run_folder, "begin", boom)


@pytest.mark.parametrize("channel, line", [
    ("mobile", start_cmd.PRIVATE_MOBILE_LINE),
    ("acp", start_cmd.PRIVATE_ACP_LINE),
    ("discord", start_cmd.PRIVATE_PLATFORM_LINE.format(name="discord")),
])
def test_a_channel_with_no_private_form_is_refused_before_any_run_folder(cfg, monkeypatch, started, channel, line):
    no_run_made(monkeypatch)
    result = invoke("--private", "--channel", channel, "--config-dir", str(cfg))
    assert result.exit_code == 2
    assert " ".join(result.output.split()) == line
    assert not started


def test_an_unknown_channel_still_gets_the_unknown_channel_refusal(cfg, monkeypatch, started):
    no_run_made(monkeypatch)
    result = invoke("--private", "--channel", "nope", "--config-dir", str(cfg))
    assert result.exit_code == 2
    assert "unknown channel" in result.output
    assert not started


def test_the_mobile_line_sends_a_person_to_the_phone():
    assert start_cmd.PRIVATE_MOBILE_LINE == "run localharness mobile and tap Private chat"


def test_the_hidden_mobile_flag_prints_the_same_line():
    from localharness.cli.mobile_cmd import mobile_cmd
    app = typer.Typer()
    app.command()(mobile_cmd)
    result = CliRunner().invoke(app, ["--private"])
    assert result.exit_code == 2
    assert start_cmd.PRIVATE_MOBILE_LINE in result.output


@pytest.mark.parametrize("cause, native, line", [
    (run_folder.LINKED, True, run_folder.PRIVATE_LINKED_RUN_LINE),
    (run_folder.FOREIGN, True, run_folder.PRIVATE_LINKED_RUN_LINE),
    (run_folder.UNLOCKED, True, run_folder.PRIVATE_UNLOCKED_LINE),
    (None, True, run_folder.PRIVATE_UNLOCKED_LINE),
    (None, False, run_folder.PRIVATE_NEEDS_POSIX_LINE),
])
def test_the_temp_fallback_is_refused_with_its_cause_and_ended(cfg, monkeypatch, started, cause, native, line):
    run = RunFolder(cfg / "run" / "x", private=True)
    ended = []
    monkeypatch.setattr(run_folder, "begin", lambda c, private=False: run)
    monkeypatch.setattr(run_folder, "end", lambda r, say=None: ended.append(r))
    monkeypatch.setattr(run_folder, "fallback_cause", lambda c: cause)
    monkeypatch.setattr(run_folder, "native", lambda: native)
    result = invoke("--private", "--config-dir", str(cfg))
    assert result.exit_code == 2
    assert " ".join(result.output.split()) == " ".join(line.format(path=cfg / "run").split())
    assert (str(cfg / "run") in " ".join(result.output.split())) is (line != run_folder.PRIVATE_NEEDS_POSIX_LINE)
    assert ended == [run] and not started


def test_a_private_start_that_got_no_run_folder_is_refused(cfg, monkeypatch, started):
    monkeypatch.setattr(run_folder, "begin", lambda c, private=False: None)
    result = invoke("--private", "--config-dir", str(cfg))
    assert result.exit_code == 2
    assert run_folder.PRIVATE_NO_RUN_LINE in " ".join(result.output.split())
    assert not started


def test_list_models_and_a_missing_config_ignore_private(tmp_path, cfg, monkeypatch, started):
    monkeypatch.setattr(run_folder, "begin", lambda c, private=False: None)
    assert invoke("--private", "--list-models", "--config-dir", str(cfg)).exit_code == 0
    assert invoke("--private", "--no-input", "--config-dir", str(tmp_path / "none")).exit_code == 0
    assert len(started) == 2


def test_a_private_start_hands_its_run_folder_to_the_session(cfg, started):
    assert invoke("--private", "--config-dir", str(cfg)).exit_code == 0
    run = started[0][1]["run"]
    assert run.private and run.state.name == "state"
    assert [p.name for p in (cfg / "run").iterdir() if p.name != ".gitignore"] == []


@pytest.mark.parametrize("scratch_free", [False, True])
def test_a_private_run_marks_every_command_private(private_run, scratch_free):
    env = run_folder.command_env(private_run, "/proj" if scratch_free else None)
    assert env["LOCALHARNESS_COMMIT"] == "private"
    assert ("LOCALHARNESS_SCRATCH" in env) is not scratch_free


def test_a_normal_run_marks_every_command_with_one(cfg):
    run = begin(cfg)
    try:
        assert run_folder.command_env(run)["LOCALHARNESS_COMMIT"] == "1"
        assert run_folder.command_env(run, "/proj") == {"LOCALHARNESS_COMMIT": "1"}
    finally:
        end(run)


async def test_the_bash_tool_gives_its_child_the_private_marker(private_run):
    from localharness.tools.builtin.bash_tool import BashExecTool
    out = await BashExecTool(run=private_run).run(command='echo "marker=$LOCALHARNESS_COMMIT"')
    assert "marker=private" in out.output


async def test_python_exec_sets_private_while_the_code_runs(private_run, monkeypatch):
    from localharness.tools.builtin.python_tool import PythonExecTool
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    out = await PythonExecTool().run(code="import os; print(os.environ['LOCALHARNESS_COMMIT'])")
    assert out.output.strip() == "private"


async def test_a_private_session_installs_no_hook_and_asks_no_consent(private_run, tmp_path, monkeypatch):
    from localharness.cli import coauthor as cli_coauthor
    from localharness.config import coauthor
    repo = tmp_path / "proj"
    (repo / ".git").mkdir(parents=True)

    def boom(*a, **k):
        raise AssertionError("a private session reaches for co-author state")
    monkeypatch.setattr(cli_coauthor, "install_hook", boom)
    monkeypatch.setattr(coauthor, "read_pending_marker", boom)
    await cli_coauthor.settle_coauthor_startup(None, repo)
    await cli_coauthor.settle_pending_coauthor(None)


async def test_a_page_saved_in_a_private_run_lands_in_its_state_and_the_receipt_names_it(private_run, monkeypatch):
    from localharness.agent.taint import under_downloads
    from localharness.tools.builtin import register_builtin_tools
    from localharness.tools.builtin.web_tool import WebFetchTool
    public_web(monkeypatch)
    _serve(monkeypatch, lambda request: _text("a page"))
    registry = SimpleNamespace(registered=[], result_size_cap_chars=1000)

    async def register(tool, scope=None):
        registry.registered.append(tool)
    registry.register = register
    await register_builtin_tools(registry, boundary=Path("/elsewhere"), run=private_run)
    fetch = next(t for t in registry.registered if isinstance(t, WebFetchTool))
    assert fetch.save_root == private_run.state
    res = await WebFetchTool(ContentStore(), save_root=fetch.save_root).run(url="http://example.com/p", save_to="p.md")
    saved = private_run.state / ".localharness" / "downloads" / "p.md"
    assert saved.read_text() == "a page" and under_downloads(saved)
    assert f"to {saved} (fetch_id=" in res.output


def test_the_verification_ledger_lands_in_the_private_state(private_run, monkeypatch):
    from localharness.agent.subagent import _verification_ledger_path
    monkeypatch.delenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", raising=False)
    assert _verification_ledger_path().is_relative_to(private_run.state / "reports")


def test_every_loop_on_a_private_gate_carries_the_private_line(private_run):
    from localharness.agent.loop import PRIVATE_PROMPT_LINE, AgentLoop
    private, normal = object.__new__(AgentLoop), object.__new__(AgentLoop)
    private._gate = SimpleNamespace(run=private_run)
    normal._gate = SimpleNamespace(run=SimpleNamespace(private=False))
    assert private._private_line() == "\n\n" + PRIVATE_PROMPT_LINE
    assert normal._private_line() == ""


def test_the_private_line_says_nothing_reaches_a_later_session_and_remember_is_this_sessions():
    from localharness.agent.loop import PRIVATE_PROMPT_LINE
    assert "a later session will not know anything said here" in PRIVATE_PROMPT_LINE
    assert "never offer to save or restore it there" in PRIVATE_PROMPT_LINE
    assert "`remember` keeps a fact for this session only" in PRIVATE_PROMPT_LINE
    assert "unless the person asks for a file in a message" in PRIVATE_PROMPT_LINE
    assert "files or memory" not in PRIVATE_PROMPT_LINE


async def test_memory_says_it_is_this_sessions_and_promotes_nothing(private_run):
    from localharness.cli import memory_cmd
    from localharness.tools.builtin.memory_tools import MemoryRememberTool

    class Empty:
        async def list_groups(self, **k):
            return []

        async def recent_facts(self, **k):
            return []

    assert await memory_cmd.dispatch(Empty(), "") == memory_cmd.PRIVATE_EMPTY
    assert await memory_cmd.dispatch(Empty(), "promote 3") == memory_cmd.PRIVATE_PROMOTE
    assert "this private session's memory" in MemoryRememberTool(None).info().description
    end(private_run)
    assert "private" not in MemoryRememberTool(None).info().description
