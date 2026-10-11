"""Swap back to the primary by its name (Phase 5, plan 13: D-34, D-58, D-70, PRESET-06, PRESET-02).

After `/model <peer>` stopped the primary for a cold peer, `/model <primary served name>` relaunches the
record as written; a typed `.gguf` path relaunches that file under its own name; anything else on a
llama.cpp primary (an HF-cache or `local_models` name) is answered in one line before anything moves.
The process boundaries are the recorders `test_launch_paths_repl.py` uses: no server, no HTTP.
"""
# ruff: noqa: F811  (the `boundaries` fixture is imported by name, as test_procfs.py does)
from __future__ import annotations

import pytest

from localharness.config.models import ManagedServerConfig
from localharness.config.overlay import load_overlay
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.launch_support import harness_launched, stub_liveness  # noqa: F401
from tests.unit.test_launch_paths_repl import (  # noqa: F401
    PEER_URL,
    PRIMARY_URL,
    NO_LLAMA_PRIMARY,
    _launchable_peer,
    _on_the_cold_peer,
    boundaries,
)
from tests.unit.test_repl_model_cmd import _harness, _xrepl


@pytest.fixture(autouse=True)
def _liveness_not_launched(monkeypatch, request):
    """No REPL build here reaches a real liveness probe (D-67): the answer is "not up" unless the
    test requests `harness_launched`."""
    if "harness_launched" not in request.fixturenames:
        stub_liveness(monkeypatch, False)


@pytest.fixture(autouse=True)
def _no_machine_probes(monkeypatch):
    from localharness.agent import context as agent_context
    from localharness.provider import autosense

    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    monkeypatch.setattr(agent_context, "probe_window",
                        lambda *a, **k: agent_context.WindowProbe(None, None))


def _llama(*, binary: str | None = "/x/llama-server", **extra) -> ManagedServerConfig:
    return ManagedServerConfig(
        runtime="llamacpp", binary=binary, model="/x/primary.gguf", port=8081, gpu=True, **extra)


def _on_a_peer(tmp_path, server, *, primary_live="unreachable", peer_live=("flash-next",)):
    """A session on a launched cold peer; the primary's endpoint answers `primary_live`."""
    peer = _launchable_peer(name="flash", served="flash-next")
    harness = _harness(server=server, extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness, {PRIMARY_URL: primary_live, PEER_URL: list(peer_live)})

    async def _refresh(model, *, base_url=None, provider_type=None):
        return ""

    repl._refresh_token_counter = _refresh
    _on_the_cold_peer(repl, agent, peer)
    agent._llm.config.model = "flash-next"
    return harness, repl, channel, agent, peer


def _ready_as(monkeypatch, *names: str) -> None:
    async def wait_ready(base_url, config_dir=None, **kw):
        return list(names)

    monkeypatch.setattr(managed_server, "wait_ready", wait_ready)


def _record_activations(monkeypatch) -> list[dict]:
    """What each strategy was asked to activate: the record's model and served name at that instant."""
    seen: list[dict] = []
    for cls in (lifecycle.SpawnedProcessStrategy, lifecycle.ManagedVllmStrategy):
        real = cls.activate

        async def spy(self, spec, config_dir, base_url, _real=real, **kw):
            seen.append({"model": spec.model, "served_name": spec.served_name,
                         "runtime": spec.runtime})
            return await _real(self, spec, config_dir, base_url, **kw)

        monkeypatch.setattr(cls, "activate", spy)
    return seen


def _flag_value(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# --- the tracer: the primary comes back by its name --------------------------------------------- #


@pytest.mark.asyncio
async def test_the_down_primarys_served_name_is_listed_and_swaps_back(tmp_path, boundaries, monkeypatch):
    primary = _llama(served_name="qwen-27b")
    harness, repl, channel, agent, peer = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "qwen-27b")
    seen = _record_activations(monkeypatch)

    await repl._handle_slash("/model")
    assert any("qwen-27b" in m for m in channel.messages)

    await repl._handle_slash("/model qwen-27b")

    assert not any("Unknown model" in m for m in channel.messages)
    assert seen == [{"model": "/x/primary.gguf", "served_name": "qwen-27b", "runtime": "llamacpp"}]
    cmd = boundaries.cmds[-1]
    assert _flag_value(cmd, "-m") == "/x/primary.gguf" and _flag_value(cmd, "-a") == "qwen-27b"
    assert len(boundaries.stops) >= 1  # the cold peer, then the primary's own (no-op) stop
    assert primary.model == "/x/primary.gguf" and primary.served_name == "qwen-27b"
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert agent._llm.config.base_url == PRIMARY_URL and agent._llm.config.model == "qwen-27b"
    assert any(m.startswith("Restarting managed llama.cpp with qwen-27b") for m in channel.messages)


@pytest.mark.asyncio
async def test_the_served_name_resolves_by_its_listed_number(tmp_path, boundaries, monkeypatch):
    primary = _llama(served_name="qwen-27b")
    _, repl, channel, agent, _ = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "qwen-27b")

    await repl._handle_slash("/model")
    listing = next(m for m in channel.messages if "qwen-27b" in m)
    number = next(line.split(".")[0].strip() for line in listing.splitlines()
                  if "qwen-27b" in line and line.strip()[:1].isdigit())

    await repl._handle_slash(f"/model {number}")

    assert boundaries.cmds and _flag_value(boundaries.cmds[-1], "-a") == "qwen-27b"
    assert agent._llm.config.base_url == PRIMARY_URL


@pytest.mark.asyncio
async def test_without_a_served_name_the_alias_in_extra_args_names_the_primary(
        tmp_path, boundaries, monkeypatch):
    primary = _llama(extra_args=["-a", "alias-x"])
    _, repl, channel, _, _ = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "alias-x")
    seen = _record_activations(monkeypatch)

    await repl._handle_slash("/model alias-x")

    assert not any("Unknown model" in m for m in channel.messages)
    assert seen and seen[0]["model"] == "/x/primary.gguf" and seen[0]["served_name"] is None
    assert boundaries.cmds[-1].count("-a") == 1 and _flag_value(boundaries.cmds[-1], "-a") == "alias-x"


@pytest.mark.asyncio
async def test_with_neither_the_providers_default_model_names_the_primary(
        tmp_path, boundaries, monkeypatch):
    primary = _llama()
    _, repl, channel, _, _ = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "model-a")
    seen = _record_activations(monkeypatch)

    await repl._handle_slash("/model model-a")  # _harness's provider.default_model

    assert not any("Unknown model" in m for m in channel.messages)
    assert seen and seen[0]["model"] == "/x/primary.gguf"
    assert "-a" not in boundaries.cmds[-1]


@pytest.mark.asyncio
async def test_a_primary_served_live_is_not_added_twice(tmp_path, boundaries):
    primary = _llama(served_name="qwen-27b")
    harness = _harness(server=primary, extra_endpoints=[])
    repl, channel, agent, _ = _xrepl(tmp_path, harness, {PRIMARY_URL: ["qwen-27b"]})
    agent._llm.config.model = "qwen-27b"

    await repl._handle_slash("/model")
    listing = channel.messages[-1]
    rows = [line for line in listing.splitlines() if "qwen-27b" in line]
    assert len(rows) == 1 and "(serving)" in rows[0]

    await repl._handle_slash("/model qwen-27b")

    assert channel.messages[-1] == "qwen-27b is already active."
    assert boundaries.cmds == [] and boundaries.stops == []


# --- D-70: a vLLM alias never becomes the vLLM model --------------------------------------------- #


def _vllm_alias_primary() -> ManagedServerConfig:
    return ManagedServerConfig(
        launch="docker", docker_image="vllm:latest", model="Qwen/Qwen3-27B",
        extra_args=["--served-model-name", "qwen-27b"])


@pytest.mark.asyncio
async def test_a_vllm_primary_swaps_back_by_its_alias_and_keeps_its_weights(
        tmp_path, boundaries, monkeypatch):
    primary = _vllm_alias_primary()
    harness, repl, channel, agent, _ = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "qwen-27b")
    seen = _record_activations(monkeypatch)

    await repl._handle_slash("/model")
    assert any("qwen-27b" in m for m in channel.messages)
    await repl._handle_slash("/model qwen-27b")

    assert not any("Unknown model" in m for m in channel.messages)
    assert seen == [{"model": "Qwen/Qwen3-27B", "served_name": None, "runtime": "vllm"}]
    assert primary.model == "Qwen/Qwen3-27B"  # the alias never replaced the weights, even persisted
    assert agent._llm.config.base_url == PRIMARY_URL
    assert "server" not in (load_overlay(tmp_path / "overrides.yaml") or {})


@pytest.mark.asyncio
async def test_a_vllm_primary_still_restarts_on_a_cached_model(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    primary = _vllm_alias_primary()
    _, repl, _, _, _ = _on_a_peer(tmp_path, primary)
    _ready_as(monkeypatch, "cached-b")
    seen = _record_activations(monkeypatch)

    await repl._handle_slash("/model cached-b")

    assert seen == [{"model": "cached-b", "served_name": None, "runtime": "vllm"}]
    assert primary.model == "cached-b"
