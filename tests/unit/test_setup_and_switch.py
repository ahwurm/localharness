"""A `(not set up)` entry is one question from serving: `localharness model <name>` asks, writes and adopts the peer, then switches to it."""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.agent.gate_types import Decision
from localharness.cli import interview, model_cmd, model_ops, workspace
from localharness.cli.app import app
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.provider import lifecycle
from tests.unit.interview_support import UNREACHED, scripted
from tests.unit.test_plan_entries_library import NVIDIA, cached, snapshot, with_vllm
from tests.unit.test_launch_paths_repl import _liveness_not_launched, _no_machine_probes  # noqa: F401 — autouse
from tests.unit.test_repl_model_cmd import _xrepl
from tests.unit.test_serve_cmd import PRIMARY, PRIMARY_URL, serve_world

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


PEER_URL = "http://127.0.0.1:8082/v1"


@pytest.fixture
def session(box):
    harness = ConfigLoader(config_dir=box.g, local_config_dir=None).load_harness()
    repl, channel, agent, _ = _xrepl(box.g, harness, {PRIMARY_URL: [PRIMARY], PEER_URL: "unreachable"})
    agent._llm.config.base_url, agent._llm.config.model = PRIMARY_URL, PRIMARY
    agent._llm.config.provider_type = "llamacpp"
    channel.requests = []
    return harness, repl, channel


def answering(channel, *kinds, local_operator=True, can_ask=True):
    channel.local_operator, channel.can_ask = local_operator, can_ask
    kinds = list(kinds)

    async def ask(request):
        channel.requests.append(request)
        return Decision(kinds.pop(0))

    channel.ask_permission = ask


@pytest.fixture
def swaps(monkeypatch):
    targets = []

    async def fake_swap(incumbent, target, **kw):
        targets.append(target)
        return lifecycle.SwapOutcome(None, None, None, RuntimeError("stop here"), "launch")

    monkeypatch.setattr(lifecycle, "swap", fake_swap)
    return targets


@pytest.mark.asyncio
async def test_the_repl_asks_through_the_channel_and_launches_what_it_set_up(box, session, swaps):
    harness, repl, channel = session
    answering(channel, "allow_once")

    await repl._handle_slash(f"/model {NVIDIA}")

    [request] = channel.requests
    assert (request.grantable, request.reason) == (False, "Set it up and switch?")
    assert request.tool_params == {"model": NVIDIA}
    [peer] = peers(box)
    assert any(m.startswith(f"• {peer['name']} (vLLM) ← {NVIDIA}") for m in channel.messages)
    assert any(f"/model {peer['name']} switches in a session;" in m for m in channel.messages)
    assert peer["name"] in [ep.name for ep in harness.extra_endpoints]
    [target] = swaps
    assert (target.spec.runtime, target.spec.model, target.spec.port) == ("vllm", NVIDIA, peer["lifecycle"]["port"])
    prefix = f"extra_endpoints.{peer['name']}.lifecycle."
    assert any(e["name"].startswith(prefix) for e in machine(box)["entries"])


@pytest.mark.asyncio
async def test_a_rejection_in_the_repl_writes_and_launches_nothing(before, session, swaps):
    _, repl, channel = session
    answering(channel, "reject_once")

    await repl._handle_slash(f"/model {NVIDIA}")

    assert len(channel.requests) == 1 and swaps == []
    assert (written(before), machine(before)) == before.before


@pytest.mark.asyncio
@pytest.mark.parametrize("operator,can_ask", [(False, True), (True, False), (False, False)])
async def test_a_channel_with_no_local_operator_is_told_where_to_set_it_up(before, session, swaps, operator, can_ask):
    _, repl, channel = session
    answering(channel, local_operator=operator, can_ask=can_ask)

    await repl._handle_slash(f"/model {NVIDIA}")

    assert channel.messages[-1] == OFF_TERMINAL
    assert channel.requests == [] and swaps == []
    assert (written(before), machine(before)) == before.before


@pytest.mark.asyncio
async def test_a_setup_entry_posted_from_a_phone_is_sent_to_the_terminal(before, session, swaps):
    _, repl, channel = session
    answering(channel, local_operator=False, can_ask=True)
    picked = []
    channel.can_run_picker = lambda: True
    channel.run_picker = lambda options, current: picked.append(options)
    repl._gate = SimpleNamespace(remote_unattended=True)

    await repl._handle_slash(f"/model {NVIDIA}")

    assert channel.messages[-1] == OFF_TERMINAL
    assert channel.requests == [] and picked == [] and swaps == []
    assert (written(before), machine(before)) == before.before


@pytest.mark.asyncio
async def test_a_snapshot_gone_since_the_listing_is_not_launched_by_the_repl(before, session, swaps):
    _, repl, channel = session

    async def vanish(request):
        shutil.rmtree(before.snapshot)
        return Decision("allow_once")

    answering(channel)
    channel.ask_permission = vanish

    await repl._handle_slash(f"/model {NVIDIA}")

    assert "is gone — nothing set up" in channel.messages[-1] and swaps == []
    assert (written(before), machine(before)) == before.before
