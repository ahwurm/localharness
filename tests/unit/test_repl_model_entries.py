"""REPL /model: the listing is `model_ops.listing`, a pick resolves through `Entries.pick`."""
from __future__ import annotations

import asyncio
import threading

import pytest

from localharness.cli import model_ops
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import EndpointRef
from localharness.provider import lifecycle
from tests.unit import test_model_foreign_incumbent as foreign
from tests.unit.launch_support import stub_liveness
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_launch_paths_repl import (  # noqa: F401 — the autouse fixtures
    PEER_URL,
    PRIMARY_URL,
    _launchable_peer,
    _liveness_not_launched,
    _llama_primary,
    _no_machine_probes,
)
from tests.unit.test_repl_model_cmd import _harness, _xrepl
from tests.unit.units_support import fake_systemd  # noqa: F401 — the fixture

SWITCH_HINT = "Switch with /model <name|number>, or scroll the menu and press Enter."


def _session(tmp_path, *, peers=None, live=None, current="qwen"):
    peers = [_launchable_peer(name="flash", served="flash")] if peers is None else peers
    harness = _harness(server=_llama_primary(served_name="qwen"), extra_endpoints=peers)
    harness.provider.provider_type = "llamacpp"
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness, live if live is not None else {PRIMARY_URL: ["qwen"], PEER_URL: "unreachable"})
    agent._llm.config.model = current
    agent._llm.config.provider_type = "llamacpp"
    return harness, repl, channel, agent


@pytest.mark.asyncio
async def test_bare_model_sends_the_one_list_and_the_switch_hint(tmp_path):
    harness, repl, channel, _ = _session(tmp_path)

    await repl._handle_slash("/model")

    sent = channel.messages[-1].split("\n")
    entries, lines = model_ops.listing(
        harness, live={model_ops._norm_url(PRIMARY_URL): ("qwen",), model_ops._norm_url(PEER_URL): None},
        config_dir=tmp_path, trust=repl._machine_trust, floor=None, current=(PRIMARY_URL, "qwen"))
    assert sent == [*lines, SWITCH_HINT]
    assert sent[:4] == ["Models:", "▸ llama.cpp · localhost:8081", "  1. qwen  (serving)  ● [active]",
                        "▸ this box can launch"]
    assert sent[4].startswith("  2. flash  llama.cpp")
    assert [e.name for e in entries.numbered()] == ["qwen", "flash"]


@pytest.mark.asyncio
async def test_a_cold_peer_pick_by_number_launches_that_peer(tmp_path, monkeypatch):
    _, repl, channel, agent = _session(tmp_path)
    targets = []

    async def fake_swap(incumbent, target, **kw):
        targets.append(target)
        return lifecycle.SwapOutcome(None, None, None, RuntimeError("stop here"), "launch")

    monkeypatch.setattr(lifecycle, "swap", fake_swap)

    await asyncio.wait_for(repl._handle_slash("/model 2"), 5)

    assert [t.spec.extra_args for t in targets] == [["-a", "flash"]]
    assert agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_the_active_entry_answers_that_it_is_active(tmp_path):
    _, repl, channel, agent = _session(tmp_path)

    await repl._handle_slash("/model qwen")

    assert channel.messages[-1] == "qwen is already active."
    assert agent._llm.rebinds == [] and agent._llm.config.model == "qwen"


@pytest.mark.asyncio
async def test_a_live_entry_on_the_current_endpoint_hot_swaps(tmp_path):
    _, repl, channel, agent = _session(
        tmp_path, peers=[], live={PRIMARY_URL: ["qwen", "qwen-b"]})

    await repl._handle_slash("/model 2")

    assert agent._llm.config.model == "qwen-b" and agent._llm.rebinds == []
    assert "Switched to qwen-b" in channel.messages[-1]


@pytest.mark.asyncio
async def test_plan_entries_runs_off_the_event_loop(tmp_path, monkeypatch):
    _, repl, _, _ = _session(tmp_path)
    threads = []
    real = model_ops.plan_entries

    def spy(*a, **kw):
        threads.append(threading.get_ident())
        return real(*a, **kw)

    monkeypatch.setattr(model_ops, "plan_entries", spy)

    await repl._handle_slash("/model")

    assert threads and threading.get_ident() not in threads


@pytest.mark.asyncio
async def test_a_pick_of_nothing_the_list_holds_is_unknown(tmp_path):
    _, repl, channel, agent = _session(tmp_path)

    await repl._handle_slash("/model nope")

    assert channel.messages[-1] == "Unknown model 'nope'. /model lists what's available."
    assert agent._llm.config.model == "qwen"


@pytest.mark.asyncio
async def test_an_unreachable_server_with_nothing_to_launch_says_so(tmp_path):
    _, repl, channel, _ = _session(tmp_path, peers=[], live={PRIMARY_URL: "unreachable"})
    repl._harness.server = None

    await repl._handle_slash("/model")

    assert channel.messages[-1] == f"Could not reach the model server at {PRIMARY_URL}. Is it running?"


def test_the_repl_builds_no_list_of_its_own():
    for name in ("_grouped_model_lines", "_discover_peer_models", "_cold_lifecycle_targets", "_cold_served_name"):
        assert not hasattr(OrchestratorREPL, name)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["flash", "flash-next"])
async def test_naming_an_entry_a_foreign_holder_blocks_sends_its_sentence_and_moves_nothing(
        tmp_path, monkeypatch, fake_proc, fake_systemd, name):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=foreign._show())
    foreign._foreign(fake_proc, cgroup=foreign._cgroup())
    repl, channel, agent = foreign._build(tmp_path, foreign._harness(foreign._llama_primary()))
    moved = []

    async def swap(*a, **k):
        moved.append("swap")

    async def free(*a, **k):
        moved.append("free_accelerator")

    async def stop(*a, **k):
        moved.append("stop")

    monkeypatch.setattr(lifecycle, "swap", swap)
    monkeypatch.setattr(lifecycle, "free_accelerator", free)
    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "stop", stop)

    await repl._handle_slash(f"/model {name}")

    assert channel.messages == [foreign.UNIT_STOP]
    assert moved == [] and agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_a_down_lifecycle_peer_is_one_entry_and_no_note(tmp_path):
    _, repl, channel, _ = _session(tmp_path)

    await repl._handle_slash("/model")

    out = channel.messages[-1]
    assert out.count("flash") == 1 and "skipped" not in out


@pytest.mark.asyncio
async def test_a_down_attach_only_peer_is_one_note(tmp_path):
    attach = EndpointRef(name="attached", base_url="http://127.0.0.1:1234/v1", provider_type="lmstudio")
    _, repl, channel, _ = _session(
        tmp_path, peers=[attach], live={PRIMARY_URL: ["qwen"], attach.base_url: "unreachable"})

    await repl._handle_slash("/model")

    notes = [line for line in channel.messages[-1].split("\n") if "skipped" in line]
    assert notes == ["  · skipped attached (http://127.0.0.1:1234/v1): unreachable"]


@pytest.mark.asyncio
async def test_cached_repos_never_reach_the_listing(tmp_path, monkeypatch):
    from localharness.provider import server as managed_server

    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["org/cached-repo"])
    _, repl, channel, _ = _session(tmp_path)

    await repl._handle_slash("/model")

    out = channel.messages[-1]
    assert "(downloaded" not in out and "(cold on" not in out and "cached-repo" not in out
