"""The downloads folder is write-denied and exec-always-asks, marked or not; a marked context turns severe asks on in every mode."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from localharness.agent.gate import NO_ASKER_REASON, PermissionGate
from localharness.agent.gate_types import Decision, GateSettings, ToolMeta, Verdict
from localharness.agent.verdict import (
    DOWNLOAD_EXEC_DETAIL,
    DOWNLOADS_WRITE_REASON,
    GateContext,
    evaluate,
)
from localharness.config.grants import GrantStore
from localharness.tools.builtin.web_tool import WebFetchTool
from tests.unit.test_web_fetch_guard import _serve, _text, public_web

SETTINGS = GateSettings()
SHELL = ToolMeta(group="shell", destructive=True)
WRITE = ToolMeta(destructive=True, group="fs.write")
MODES = ("guarded", "trusted", "auto", "unattended")
SAVED = ".localharness/downloads"
SAVED_NOTE = " — the agent has read pages saved from the web (a.md)."


@pytest.fixture
def proj(tmp_path) -> Path:
    root = (tmp_path / "proj").resolve()
    (root / SAVED).mkdir(parents=True)
    (root / SAVED / "x.sh").write_text("echo hi\n")
    (root / SAVED / "x.py").write_text("print(1)\n")
    (root / "run.sh").symlink_to(root / SAVED / "x.sh")
    return root


def ctx(proj: Path, mode: str = "guarded", **kw) -> GateContext:
    return GateContext(boundary=proj, workspace=proj, grants=lambda w, k, key: None, mode=mode,  # type: ignore[arg-type]
                       has_review_surface=True, **kw)


def shell(command: str, context: GateContext):
    return evaluate("bash_exec", {"command": command}, SHELL, context, SETTINGS)


class Asker:
    def __init__(self) -> None:
        self.calls: list = []

    async def __call__(self, request):
        self.calls.append(request)
        return Decision("allow_once")


def gate(proj: Path, tmp_path: Path, mode: str, asker=None) -> PermissionGate:
    return PermissionGate(boundary=proj, workspace=proj, grants=GrantStore(tmp_path / "grants.yaml"),
                          mode=mode, asker=asker, channel_name="test", has_review_surface=True)


async def check(g: PermissionGate, tool: str, params: dict, meta: ToolMeta = SHELL):
    return await g.check(tool, params, meta, agent_id="root", session_id="s")


async def test_a_saved_page_runs_only_after_a_person_answers(proj, tmp_path, monkeypatch):
    public_web(monkeypatch)
    _serve(monkeypatch, lambda request: _text("echo from the web\n" * 50))
    saved = await WebFetchTool(boundary=proj).run(url="http://example.com/x", save_to="fetched.sh")
    assert saved.success and (proj / SAVED / "fetched.sh").exists()

    asker = Asker()
    g = gate(proj, tmp_path, "auto", asker)
    command = {"command": f"bash {SAVED}/fetched.sh"}
    first = await check(g, "bash_exec", command)
    assert not first.allowed and first.pending is not None
    assert "runs a file saved from the web" in first.pending.rendering
    assert not asker.calls
    await g.approve()
    assert (await check(g, "bash_exec", command)).allowed


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("tool,params,meta", [
    ("write", {"path": f"{SAVED}/x.md", "content": "x"}, WRITE),
    ("edit", {"path": f"{SAVED}/x.sh", "old_string": "a", "new_string": "b"}, WRITE),
    ("bash_exec", {"command": f"echo x > {SAVED}/a"}, SHELL),
    ("bash_exec", {"command": f"rm -rf {SAVED}/x.sh"}, SHELL),
    ("write", {"path": "run.sh", "content": "x"}, WRITE),
])
async def test_writing_into_the_downloads_folder_is_denied_in_every_mode(proj, tmp_path, mode, tool, params, meta):
    g = gate(proj, tmp_path, mode, Asker())
    out = await check(g, tool, params, meta)
    assert not out.allowed and out.pending is None
    assert out.reason == DOWNLOADS_WRITE_REASON and "save_to" in out.reason


@pytest.mark.parametrize("command", [
    f"bash {SAVED}/x.sh", f"python3 {SAVED}/x.py", f"source {SAVED}/x.sh", "./run.sh", f"{SAVED}/x.sh",
])
@pytest.mark.parametrize("mode", ["guarded", "trusted"])
def test_running_a_saved_file_asks_every_time_and_cannot_be_remembered(proj, mode, command):
    result = shell(command, ctx(proj, mode))
    assert result.verdict is Verdict.ASK
    request = result.request
    assert request.klass == "shell-destructive" and request.grantable is False
    assert DOWNLOAD_EXEC_DETAIL in request.reason
    assert "taint" not in request.display.lower()


def test_cat_of_a_saved_file_is_not_an_execution(proj):
    assert shell(f"cat {SAVED}/x.sh", ctx(proj, "trusted")).verdict is Verdict.ALLOW


@pytest.mark.parametrize("mode", ["auto", "unattended"])
async def test_auto_and_unattended_park_the_run_and_approve_lets_the_same_call_through_once(proj, tmp_path, mode):
    asker = Asker()
    g = gate(proj, tmp_path, mode, asker)
    command = {"command": "python3 " + f"{SAVED}/x.py"}
    parked = await check(g, "bash_exec", command)
    assert parked.pending is not None and not parked.allowed and not asker.calls
    await g.approve()
    assert (await check(g, "bash_exec", command)).allowed
    assert (await check(g, "bash_exec", command)).pending is not None


async def test_unattended_with_no_asker_refuses_the_run(proj, tmp_path):
    g = gate(proj, tmp_path, "unattended")
    out = await check(g, "bash_exec", {"command": f"bash {SAVED}/x.sh"})
    assert not out.allowed and out.reason == NO_ASKER_REASON


async def test_unattended_still_runs_everything_else_unasked(proj, tmp_path):
    asker = Asker()
    g = gate(proj, tmp_path, "unattended", asker)
    assert (await check(g, "bash_exec", {"command": "rm -rf build && python -c 'print(1)'"})).allowed
    assert not asker.calls


# ---------------------------------------------------------------- a context marked as having read saved pages

def test_rm_rf_inside_the_project_asks_in_auto_only_when_marked(proj):
    assert shell("rm -rf build", ctx(proj, "auto")).verdict is Verdict.ALLOW
    marked = shell("rm -rf build", ctx(proj, "auto", tainted_by="a.md"))
    assert marked.verdict is Verdict.ASK
    assert marked.request.display.endswith(SAVED_NOTE)
    assert marked.reason.endswith(SAVED_NOTE)


@pytest.mark.parametrize("mode", ["guarded", "trusted", "auto", "unattended"])
def test_an_inline_interpreter_asks_in_every_mode_when_marked(proj, mode):
    assert shell("python -c 'print(1)'", ctx(proj, mode)).verdict is (
        Verdict.ASK if mode == "guarded" else Verdict.ALLOW)
    marked = shell("python -c 'print(1)'", ctx(proj, mode, tainted_by="a.md"))
    assert marked.verdict is Verdict.ASK and marked.request.klass == "interpreter-inline"
    assert marked.request.display.endswith(SAVED_NOTE)


@pytest.mark.parametrize("mode", MODES)
def test_a_listing_stays_silent_when_marked(proj, mode):
    assert shell("ls", ctx(proj, mode, tainted_by="a.md")).verdict is Verdict.ALLOW


def test_a_stored_grant_does_not_cover_a_severe_ask_while_marked(proj):
    def granted(workspace, klass, key):
        return object() if klass == "interpreter-inline" else None

    base = dataclasses.replace(ctx(proj, "guarded"), grants=granted)  # type: ignore[arg-type]
    assert shell("python -c 'print(1)'", base).verdict is Verdict.ALLOW
    assert shell("python -c 'print(1)'", dataclasses.replace(base, tainted_by="a.md")).verdict is Verdict.ASK
