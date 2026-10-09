"""REPL launch paths 2-4 meet the one launch decision (Phase 1, plan 06).

Path 2 is `/model`'s cold branch, path 3 the managed restart that is also the swap-back to the
primary, path 4 `_restore_incumbent`. Each asks `launch_decision` before it stops, announces or
rebinds, passes `peer=` for a peer, and ends a loopback launch's argv with `--host 127.0.0.1`.
"""
from __future__ import annotations

import pytest

from localharness.config.models import EndpointRef, ManagedServerConfig
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.test_repl_model_cmd import (
    _harness,
    _heavy_primary,
    _xrepl,
)

PEER_URL = "http://127.0.0.1:8080/v1"
PRIMARY_URL = "http://localhost:8081/v1"
NO_LLAMA_PEER = "no llama-server — set extra_endpoints[llamacpp-local].lifecycle.binary"
NO_LLAMA_PRIMARY = "no llama-server — set server.binary"


class Boundaries:
    """Recorders for the three process boundaries a launch crosses."""

    def __init__(self) -> None:
        self.stops: list[str] = []
        self.cmds: list[list[str]] = []
        self.peers: list[str | None] = []
        self.waits: list[str] = []


@pytest.fixture
def boundaries(monkeypatch):
    """stop_server / start_server / wait_ready recorded; list_cached_models empty; no settle."""
    b = Boundaries()
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    monkeypatch.setattr(managed_server, "stop_server",
                        lambda cfg, launch="binary": b.stops.append(launch) or True)
    monkeypatch.setattr(managed_server, "start_server",
                        lambda cfg, cmd, **kw: b.cmds.append(list(cmd)) or 4321)

    async def fake_wait_ready(base_url, config_dir=None, **kw):
        b.waits.append(base_url)
        return ["qwen-gguf"]

    monkeypatch.setattr(managed_server, "wait_ready", fake_wait_ready)
    return b


def _declared_only_peer(name="llamacpp-local", served="qwen-gguf", **extra) -> EndpointRef:
    """A llama.cpp peer that names its record but no binary: declared, not launchable."""
    return EndpointRef(
        name=name, base_url=PEER_URL, provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", model="/x/m.gguf", port=8080, gpu=True,
            extra_args=["-a", served], **extra),
    )


def _launchable_peer(name="llamacpp-local", served="qwen-gguf", api_key="none", **life) -> EndpointRef:
    return EndpointRef(
        name=name, base_url=PEER_URL, provider_type="llamacpp", gpu=True, api_key=api_key,
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080, gpu=True,
            extra_args=["-a", served], **life),
    )


def _session(tmp_path, peer, server=None):
    harness = _harness(server=server if server is not None else _heavy_primary(),
                       extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness, {PRIMARY_URL: ["model-a"], PEER_URL: "unreachable"})

    async def _refresh(model, *, base_url=None, provider_type=None):
        return ""

    repl._refresh_token_counter = _refresh

    async def _persist_active(endpoint, model):
        return None

    repl._persist_active_endpoint = _persist_active
    return harness, repl, channel, agent


# --- path 2: /model's cold branch -------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_declared_only_cold_peer_answers_one_line_and_touches_nothing(tmp_path, boundaries):
    harness, repl, channel, agent = _session(tmp_path, _declared_only_peer())
    seeded = repl._active_heavy
    assert seeded is not None and seeded[0] is harness.server

    await repl._handle_slash("/model qwen-gguf")

    assert channel.messages[-1] == NO_LLAMA_PEER
    assert boundaries.stops == [] and boundaries.cmds == [] and boundaries.waits == []
    assert repl._active_heavy is seeded
    assert agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_a_refused_cold_launch_announces_nothing(tmp_path, boundaries):
    _, repl, channel, _ = _session(tmp_path, _declared_only_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert not any("Launching" in m for m in channel.messages)
    assert not any("Could not free" in m for m in channel.messages)


@pytest.mark.asyncio
async def test_an_exposed_cold_launch_announces_its_posture_once(tmp_path, boundaries):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(bind_all=True))

    await repl._handle_slash("/model qwen-gguf")

    exposed = [m for m in channel.messages if "every interface, no key" in m]
    assert len(exposed) == 1
    assert exposed[0].startswith(
        "Launching llamacpp-local (every interface, no key) — the swap can take several minutes...")


@pytest.mark.asyncio
async def test_a_keyed_cold_launch_carries_no_exposure_phrase(tmp_path, boundaries):
    _, repl, channel, _ = _session(
        tmp_path, _launchable_peer(api_key="sk-PEER", bind_all=True, require_api_key=True))

    await repl._handle_slash("/model qwen-gguf")

    assert not any("every interface, no key" in m for m in channel.messages)
    assert any(m.startswith("Launching llamacpp-local — the swap can take") for m in channel.messages)


@pytest.mark.asyncio
async def test_an_unreadable_key_file_is_sent_as_its_own_line(tmp_path, boundaries):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    _, repl, channel, _ = _session(
        tmp_path,
        _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key_dir)))

    await repl._handle_slash("/model qwen-gguf")

    expected = (f"--api-key-file {key_dir} unreadable: not a regular file — "
                "launched on 127.0.0.1 for this run")
    assert channel.messages.count(expected) == 1
    # the withheld widening launched loopback, so the announcement carries no exposure phrase
    assert not any("every interface, no key" in m for m in channel.messages)
    assert boundaries.cmds and boundaries.cmds[0][-2:] == ["--host", "127.0.0.1"]


@pytest.mark.asyncio
async def test_the_cold_launch_passes_the_peer_name_to_activate(tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)
    seen: list[str | None] = []
    real = managed_server.launch_decision

    def spy(spec, keyed, *, peer=None):
        seen.append(peer)
        return real(spec, keyed, peer=peer)

    monkeypatch.setattr(managed_server, "launch_decision", spy)

    await repl._handle_slash("/model qwen-gguf")

    # the REPL's own decision and the strategy's enforcement both name the peer
    assert seen and set(seen) == {"llamacpp-local"}
    assert len(seen) >= 2


# --- path 3: the managed restart / swap-back --------------------------------------------------- #


def _llama_primary(*, binary: str | None = "/x/llama-server", **extra) -> ManagedServerConfig:
    return ManagedServerConfig(
        runtime="llamacpp", binary=binary, model="/x/primary.gguf", port=8081, gpu=True, **extra)


def _on_the_cold_peer(repl, agent, peer) -> None:
    """The session sits on a launched cold peer: it holds the accelerator and the client is bound to it."""
    repl._active_heavy = (peer.lifecycle, peer.base_url)
    agent._llm.config.base_url = peer.base_url
    agent._llm.config.model = "qwen-gguf"


@pytest.mark.asyncio
async def test_a_swap_back_to_a_declared_only_primary_refuses_before_its_rebind(
        tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    peer = _launchable_peer()
    _, repl, channel, agent = _session(tmp_path, peer, server=_llama_primary(binary=None))
    _on_the_cold_peer(repl, agent, peer)
    held = repl._active_heavy

    await repl._handle_slash("/model cached-b")

    assert channel.messages[-1] == NO_LLAMA_PRIMARY
    assert agent._llm.rebinds == []
    assert boundaries.stops == [] and boundaries.cmds == [] and boundaries.waits == []
    assert repl._active_heavy is held
    assert agent._llm.config.base_url == peer.base_url


@pytest.mark.asyncio
async def test_a_llama_cpp_restart_names_the_runtime(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=_llama_primary())

    await repl._handle_slash("/model cached-b")

    assert any(m.startswith(
        "Restarting managed llama.cpp with cached-b — model load can take several minutes...")
        for m in channel.messages)


@pytest.mark.asyncio
async def test_a_vllm_restart_announcement_is_unchanged(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model cached-b")

    assert any(m.startswith(
        "Restarting managed vLLM with cached-b — model load can take several minutes...")
        for m in channel.messages)
    assert not any("every interface" in m for m in channel.messages)


# --- path 4: _restore_incumbent ----------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_restoring_nothing_gives_nothing(tmp_path, boundaries):
    _, repl, _, _ = _session(tmp_path, _launchable_peer())

    assert await repl._restore_incumbent(None) == (None, None)


@pytest.mark.asyncio
async def test_a_restored_peer_returns_its_label_and_the_decisions_note(tmp_path, boundaries):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key_dir))
    _, repl, _, _ = _session(tmp_path, peer)
    repl._active_heavy = None

    label, why = await repl._restore_incumbent((peer.lifecycle, PEER_URL))

    assert label == peer.lifecycle.model
    assert why == (f"--api-key-file {key_dir} unreadable: not a regular file — "
                   "launched on 127.0.0.1 for this run")
    assert repl._active_heavy == (peer.lifecycle, PEER_URL)


@pytest.mark.asyncio
async def test_a_restore_without_a_note_gives_no_why(tmp_path, boundaries):
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)

    assert await repl._restore_incumbent((peer.lifecycle, PEER_URL)) == (peer.lifecycle.model, None)


@pytest.mark.asyncio
async def test_a_declared_only_primary_is_refused_not_swallowed(tmp_path, boundaries):
    primary = _llama_primary(binary=None)
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    repl._active_heavy = None

    assert await repl._restore_incumbent((primary, PRIMARY_URL)) == (None, NO_LLAMA_PRIMARY)
    assert repl._active_heavy is None
    assert boundaries.cmds == []


@pytest.mark.asyncio
async def test_a_declared_only_peer_is_refused_by_name(tmp_path, boundaries):
    peer = _declared_only_peer(name="flash-next")
    _, repl, _, _ = _session(tmp_path, peer)
    repl._active_heavy = None

    assert await repl._restore_incumbent((peer.lifecycle, PEER_URL)) == (
        None, "no llama-server — set extra_endpoints[flash-next].lifecycle.binary")
    assert repl._active_heavy is None


@pytest.mark.asyncio
async def test_a_spec_that_is_not_the_peers_own_lifecycle_is_not_named_as_the_peer(
        tmp_path, boundaries):
    peer = _declared_only_peer(name="flash-next")
    _, repl, _, _ = _session(tmp_path, peer)
    stranger = peer.lifecycle.model_copy()

    assert await repl._restore_incumbent((stranger, PEER_URL)) == (None, NO_LLAMA_PRIMARY)


@pytest.mark.asyncio
async def test_a_failed_cold_launch_reports_why_the_incumbent_was_not_restored(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary(binary=None)
    peer = _launchable_peer()
    _, repl, channel, agent = _session(tmp_path, peer, server=primary)

    async def never_ready(base_url, config_dir=None, **kw):
        raise TimeoutError("llama-server never became ready")

    monkeypatch.setattr(managed_server, "wait_ready", never_ready)

    await repl._handle_slash("/model qwen-gguf")

    last = channel.messages[-1]
    assert f"Could not restore it: {NO_LLAMA_PRIMARY}" in last
    assert "NO model serving" not in last
    assert agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_a_failed_swap_back_reports_why_the_peer_was_not_restored(
        tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    peer = _declared_only_peer(name="flash-next")  # a vLLM primary lends it no binary
    _, repl, channel, agent = _session(tmp_path, peer)
    _on_the_cold_peer(repl, agent, peer)

    async def never_ready(base_url, config_dir=None, **kw):
        raise TimeoutError("llama-server never became ready")

    monkeypatch.setattr(managed_server, "wait_ready", never_ready)

    await repl._handle_slash("/model cached-b")

    last = channel.messages[-1]
    assert last.startswith("Model swap failed:")
    assert ("Could not restore it: no llama-server — set extra_endpoints[flash-next]"
            ".lifecycle.binary") in last
    assert "NO model serving" not in last
