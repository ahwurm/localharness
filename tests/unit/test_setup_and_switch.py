"""A `(not set up)` entry is one question from serving: `localharness model <name>` asks, writes and adopts the peer, then switches to it."""
from __future__ import annotations

import shutil

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import interview, model_cmd, model_ops, workspace
from localharness.cli.app import app
from localharness.config import trust
from tests.unit.interview_support import UNREACHED, scripted
from tests.unit.test_plan_entries_library import NVIDIA, cached, snapshot, with_vllm
from tests.unit.test_serve_cmd import PRIMARY, serve_world

runner = CliRunner()


def terminal(monkeypatch, on=True):
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    monkeypatch.setattr(workspace, "_stdin_is_a_terminal", lambda: on)
    monkeypatch.setattr(workspace, "_stdout_is_a_terminal", lambda: on)


@pytest.fixture
def box(tmp_path, monkeypatch):
    world = serve_world(tmp_path, monkeypatch)
    repo = snapshot(tmp_path, NVIDIA, "nvidia-qwen3.8-27b-nvfp4")
    world.snapshot = repo.revisions[0].snapshot_path
    cached(monkeypatch, repo)
    with_vllm(tmp_path, monkeypatch)
    monkeypatch.setattr(model_ops, "list_live_models", lambda *a, **k: ([PRIMARY], True))
    monkeypatch.setattr(model_cmd, "_own_unit", lambda cfg: None)
    monkeypatch.setattr(workspace, "port_holder", lambda port: None)
    terminal(monkeypatch)
    world.exe = tmp_path / "bin" / "vllm"
    return world


def switch(box, name=NVIDIA):
    return runner.invoke(app, ["model", name, "--config-dir", str(box.g)])


def written(box):
    path = box.g / "overrides.yaml"
    return path.read_bytes() if path.exists() else b""


def machine(box):
    return trust.recorded_machine(box.g)


def peers(box):
    return (yaml.safe_load(written(box)) or {}).get("extra_endpoints", [])


def said(result):
    return " ".join(result.output.split())


def test_a_snapshot_is_set_up_adopted_and_selected_after_one_yes(box, monkeypatch):
    asked = scripted(monkeypatch, "set-up")

    result = switch(box)

    assert result.exit_code == 0, result.output
    [question] = asked.asked
    assert question.question == "Set it up and switch?"
    assert [o.label for o in question.options] == ["Set it up and switch", "Not now"]
    assert (question.default, question.cancel) == (0, "not-now")
    [peer] = peers(box)
    assert {k: peer["lifecycle"][k] for k in ("runtime", "launch", "binary", "model")} == {
        "runtime": "vllm", "launch": "binary", "binary": str(box.exe), "model": NVIDIA}
    prefix = f"extra_endpoints.{peer['name']}.lifecycle."
    assert {e["name"] for e in machine(box)["entries"] if e["name"].startswith(prefix)} >= {
        prefix + "runtime", prefix + "model"}
    out = said(result)
    assert f"← {NVIDIA}" in out and out.index(f"← {NVIDIA}") < out.index(f"/model {peer['name']} switches in a session;")
    assert f"Selected {peer['name']} — the next start serves it" in out
    assert yaml.safe_load(written(box))["pending_preset"] == peer["name"]


def test_not_now_writes_nothing_and_selects_nothing(box, monkeypatch):
    scripted(monkeypatch, "not-now")
    before = written(box), machine(box)

    result = switch(box)

    assert result.exit_code == 0, result.output
    assert (written(box), machine(box)) == before
    assert "Selected" not in result.output


OFF_TERMINAL = f"{NVIDIA} is not set up — localharness model --setup sets it up"


def untouched(box, result, asked):
    assert result.exit_code == 2, result.output
    assert OFF_TERMINAL in said(result)
    assert asked.asked == []
    assert (written(box), machine(box)) == box.before


@pytest.fixture
def before(box):
    box.before = written(box), machine(box)
    return box


@pytest.mark.parametrize("name", [NVIDIA, "2"])
def test_off_a_terminal_nothing_is_set_up(before, monkeypatch, name):
    terminal(monkeypatch, on=False)
    asked = scripted(monkeypatch)

    untouched(before, switch(before, name), asked)


def test_an_agents_command_on_a_terminal_sets_nothing_up(before, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    asked = scripted(monkeypatch)

    untouched(before, switch(before), asked)


def test_a_question_nobody_could_answer_sets_nothing_up(before, monkeypatch):
    asked = scripted(monkeypatch, UNREACHED)

    result = switch(before)

    assert result.exit_code == 2 and OFF_TERMINAL in said(result)
    assert len(asked.asked) == 1
    assert (written(before), machine(before)) == before.before


def test_a_snapshot_gone_since_the_listing_is_not_set_up(before, monkeypatch):
    def vanish(*a, **k):
        shutil.rmtree(before.snapshot)
        return interview.Answer(("set-up",), False, True)

    monkeypatch.setattr(interview, "ask", vanish)

    result = switch(before)

    assert result.exit_code == 2, result.output
    assert "is gone — nothing set up" in said(result)
    assert (written(before), machine(before)) == before.before
