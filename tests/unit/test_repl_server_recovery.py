# ruff: noqa: F401, F811
"""The REPL's reconnect supplier and the launch lock around its swaps (Phase 7, plan 05).

`OrchestratorREPL.server_recovery()` is what the loop asks when the model server drops: one
`ServerRecovery` per (client address, active spec, active address), relaunchable only for the spec the
harness launched at the address the client is using, with that endpoint's own key and its strategy's
config dir. Everything the three process boundaries cross is stubbed; every config dir is `tmp_path`.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from localharness.config.models import ManagedServerConfig
from localharness.provider import client as provider_client
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.test_launch_paths_repl import (
    PEER_URL,
    PRIMARY_SERVED,
    PRIMARY_URL,
    SENTINEL,
    _declared_only_peer,
    _on_the_cold_peer,
    _session,
    _capture_start_kw,
    _launchable_peer,
    _liveness_not_launched,
    _llama_primary,
    _no_machine_probes,
    boundaries,
)
from tests.unit.test_repl_model_cmd import _harness, _heavy_primary, _peer, _xrepl

PEER_KEY = "KEY-SENTINEL-peer-own"
PRIMARY_KEY = "KEY-SENTINEL-primary-own"
OTHER_URL = "http://localhost:8082/v1"


def _build(tmp_path, *, server=None, peers=(), active=None, on=None, provider_key=None):
    """A REPL over a harness with `server` and `peers`, `_active_heavy` set to `active`, and the client
    bound to `on` (default: the primary's address)."""
    harness = _harness(server=server, extra_endpoints=list(peers))
    if provider_key is not None:
        harness.provider.api_key = SecretStr(provider_key)
    repl, channel, agent, _ = _xrepl(tmp_path, harness, {})
    repl._active_heavy = active
    if on is not None:
        agent._llm.config.base_url = on
    return harness, repl, channel, agent


def _ollama_primary() -> ManagedServerConfig:
    return ManagedServerConfig(runtime="ollama", model="llama3", port=8081, gpu=True)


# --- the memo ------------------------------------------------------------------------------------ #


def test_the_supplier_never_returns_none_and_returns_the_same_pair_until_something_changes(tmp_path):
    primary = _llama_primary()
    _, repl, _, agent = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL))

    first = repl.server_recovery()
    assert first is not None and repl.server_recovery() is first

    repl._active_heavy = (_llama_primary(), PRIMARY_URL)  # another spec object at the same address
    second = repl.server_recovery()
    assert second is not first and repl.server_recovery() is second

    repl._active_heavy = (repl._active_heavy[0], OTHER_URL)  # the same spec at another address
    third = repl.server_recovery()
    assert third is not second

    agent._llm.rebind_endpoint(PEER_URL)  # the client moves
    assert repl.server_recovery() is not third


def test_a_hand_run_primary_gets_a_pair_that_only_polls(tmp_path):
    _, repl, _, _ = _build(tmp_path, server=None)

    pair = repl.server_recovery()

    assert pair is not None
    assert pair.relaunch is None and pair.config_dir is None
    assert pair.base_url == PRIMARY_URL


# --- the address rule ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("primary", [_llama_primary(), _heavy_primary(), _ollama_primary()],
                         ids=["llamacpp", "vllm", "ollama"])
def test_the_launched_primary_at_the_clients_address_is_relaunchable(tmp_path, primary):
    _, repl, _, agent = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL))

    pair = repl.server_recovery()

    assert pair.relaunch is not None
    assert pair.config_dir == repl._server_config_dir
    assert pair.runtime == primary.runtime
    assert pair.base_url == agent._llm.config.base_url == PRIMARY_URL


def test_an_attach_only_peer_never_gets_the_primarys_relaunch(tmp_path):
    primary = _llama_primary()
    attach = _peer("attach-only", PEER_URL, "llamacpp")
    _, repl, _, agent = _build(tmp_path, server=primary, peers=[attach],
                               active=(primary, PRIMARY_URL))
    agent._llm.rebind_endpoint(PEER_URL)  # the hot-swap leaves _active_heavy at the primary

    assert repl._active_heavy == (primary, PRIMARY_URL)
    pair = repl.server_recovery()

    assert pair.relaunch is None and pair.config_dir is None
    assert pair.base_url == PEER_URL


def test_the_address_test_is_exact_string_equality(tmp_path):
    primary = _llama_primary()
    _, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL), on=OTHER_URL)
    assert repl.server_recovery().relaunch is None

    for near in (PRIMARY_URL + "/", PRIMARY_URL.replace("/v1", "/v2"),
                 PRIMARY_URL.replace("localhost", "127.0.0.1")):
        _, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL), on=near)
        assert repl.server_recovery().relaunch is None, near


def test_a_gpu_false_managed_primary_is_polled_with_its_dir_and_never_relaunched(tmp_path):
    cpu = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf",
                              port=8081, gpu=False)
    _, repl, _, _ = _build(tmp_path, server=cpu, active=None)

    pair = repl.server_recovery()

    assert repl._active_heavy is None
    assert pair.relaunch is None
    assert pair.config_dir == repl._server_config_dir and pair.runtime == "llamacpp"


def test_an_lm_studio_primary_has_no_config_dir_to_poll_with(tmp_path):
    lms = ManagedServerConfig(runtime="lmstudio", model="qwen", port=8081, gpu=False)
    _, repl, _, _ = _build(tmp_path, server=lms, active=None)

    pair = repl.server_recovery()

    assert pair.relaunch is None and pair.config_dir is None and pair.runtime == "lmstudio"


def test_a_peer_that_is_not_the_active_spec_is_polled_not_relaunched(tmp_path):
    primary = _llama_primary()
    peer = _launchable_peer()  # launchable, but not what the harness launched
    _, repl, _, _ = _build(tmp_path, server=primary, peers=[peer], active=(primary, PRIMARY_URL),
                           on=PEER_URL)

    pair = repl.server_recovery()

    assert pair.relaunch is None and pair.config_dir is None


def test_a_config_less_session_has_nothing_to_relaunch_with(tmp_path):
    primary = _llama_primary()
    _, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL))
    repl._config_dir = None

    pair = repl.server_recovery()

    assert pair.relaunch is None and pair.config_dir is None


# --- the key ------------------------------------------------------------------------------------- #


def test_a_required_key_that_resolves_to_nothing_is_polled_never_relaunched(tmp_path):
    primary = _llama_primary(require_api_key=True)
    _, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL))  # key "none"

    pair = repl.server_recovery()

    assert pair.relaunch is None
    assert pair.config_dir == repl._server_config_dir


@pytest.mark.asyncio
async def test_a_keyed_peer_is_relaunched_with_its_own_key_while_the_primary_is_keyless(
        tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer(api_key=PEER_KEY, require_api_key=True)
    _, repl, _, _ = _build(tmp_path, server=_llama_primary(), peers=[peer],
                           active=(peer.lifecycle, PEER_URL), on=PEER_URL)
    started = _capture_start_kw(monkeypatch)

    pair = repl.server_recovery()
    assert pair.relaunch is not None and pair.runtime == "llamacpp"
    await pair.relaunch()

    (call,) = started
    assert call["api_key"] == PEER_KEY
    assert PEER_KEY not in " ".join(call["cmd"])


@pytest.mark.asyncio
async def test_a_keyed_primary_never_lends_its_key_to_a_peer(tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer(api_key=PEER_KEY, require_api_key=True)
    _, repl, _, _ = _build(tmp_path, server=_llama_primary(require_api_key=True), peers=[peer],
                           active=(peer.lifecycle, PEER_URL), on=PEER_URL,
                           provider_key=PRIMARY_KEY)
    started = _capture_start_kw(monkeypatch)

    await repl.server_recovery().relaunch()

    (call,) = started
    assert call["api_key"] == PEER_KEY != PRIMARY_KEY

    # and a peer that requires a key it does not hold is not given the primary's
    bare = _launchable_peer(api_key="none", require_api_key=True)
    _, repl2, _, _ = _build(tmp_path, server=_llama_primary(require_api_key=True), peers=[bare],
                            active=(bare.lifecycle, PEER_URL), on=PEER_URL,
                            provider_key=PRIMARY_KEY)
    assert repl2.server_recovery().relaunch is None


@pytest.mark.asyncio
async def test_the_primary_is_relaunched_with_the_providers_key_fixed_when_the_pair_is_built(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary(require_api_key=True)
    harness, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL),
                                 provider_key=PRIMARY_KEY)
    started = _capture_start_kw(monkeypatch)
    pair = repl.server_recovery()
    harness.provider.api_key = SecretStr("rotated-later")

    await pair.relaunch()

    assert started[0]["api_key"] == PRIMARY_KEY


@pytest.mark.asyncio
async def test_a_peer_whose_key_file_changed_is_relaunched_with_the_new_key(tmp_path, boundaries, monkeypatch):
    """WR-55: a cold /model launch re-reads the peer's key file and leaves `_active_heavy` at the same
    object and address; the supplier must not hand back a pair whose relaunch still carries the old key,
    or the relaunched server would require a key the client no longer sends."""
    peer = _launchable_peer(api_key=PEER_KEY, require_api_key=True)
    _, repl, _, _ = _build(tmp_path, server=_llama_primary(), peers=[peer],
                           active=(peer.lifecycle, PEER_URL), on=PEER_URL)
    started = _capture_start_kw(monkeypatch)
    before = repl.server_recovery()
    assert repl.server_recovery() is before  # nothing changed: the same pair

    peer.api_key = SecretStr("KEY-SENTINEL-peer-rotated")  # the key file said something new
    after = repl.server_recovery()
    assert after is not before
    await after.relaunch()

    (call,) = started
    assert call["api_key"] == "KEY-SENTINEL-peer-rotated"
    assert PEER_KEY not in repr(repl._recovery_memo[0])
    assert "KEY-SENTINEL-peer-rotated" not in repr(repl._recovery_memo[0])  # only a digest is kept


@pytest.mark.asyncio
async def test_the_relaunch_activates_the_very_object_in_active_heavy(tmp_path, boundaries, monkeypatch):
    primary = _llama_primary()
    _, repl, _, _ = _build(tmp_path, server=primary, active=(primary, PRIMARY_URL))
    seen: list = []
    real = lifecycle.SpawnedProcessStrategy.activate

    async def spy(self, spec, config_dir, base_url, **kw):
        seen.append((spec, config_dir, base_url, kw))
        return await real(self, spec, config_dir, base_url, **kw)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", spy)

    await repl.server_recovery().relaunch()

    (spec, config_dir, base_url, kw), = seen
    assert spec is primary and spec is repl._active_heavy[0]
    assert (config_dir, base_url) == (repl._server_config_dir, PRIMARY_URL)
    assert kw["peer"] is None


# --- the launch lock around every REPL swap (D-27, D-47) ------------------------------------------ #


class LockWatch:
    """What the launch lock saw during one test: how many times the REPL asked for it (`entries`,
    nested asks included), how many times the lock file was actually opened (`opens`: a nested ask by
    the holder opens nothing), and the held locks at each `start_server`."""

    def __init__(self, monkeypatch, repl) -> None:
        self.entries = 0
        self.opens = 0
        self.held_at_start: list[set] = []
        lock = str(managed_server.lock_path(repl._server_config_dir))
        real_open = os.open

        def counting_open(path, *args, **kw):
            if str(path) == lock:
                self.opens += 1
            return real_open(path, *args, **kw)

        monkeypatch.setattr(lifecycle.os, "open", counting_open)
        real_lock = repl._launch_lock

        def counting_lock():
            self.entries += 1
            return real_lock()

        repl._launch_lock = counting_lock
        real_start = managed_server.start_server
        monkeypatch.setattr(
            managed_server, "start_server",
            lambda cfg, cmd, **kw: (self.held_at_start.append(set(lifecycle._HELD)),
                                    real_start(cfg, cmd, **kw))[1])
        self.key = managed_server.lock_path(repl._server_config_dir).resolve()


def _wait_ready_fails_at(monkeypatch, bad_url: str) -> None:
    async def wait_ready(base_url, config_dir=None, **kw):
        if base_url == bad_url:
            raise TimeoutError("llama-server never became ready")
        return ["qwen-gguf"]

    monkeypatch.setattr(managed_server, "wait_ready", wait_ready)


@pytest.mark.asyncio
async def test_a_cold_launch_takes_the_lock_once_and_holds_it_while_the_server_starts(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary()
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)
    repl._active_heavy = (primary, PRIMARY_URL)
    watch = LockWatch(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model qwen-gguf"), 5)

    assert watch.entries == 1 and watch.opens == 1
    assert watch.held_at_start == [{watch.key}]
    assert lifecycle._HELD == {}
    assert any(m.startswith("Switched to qwen-gguf on llamacpp-local") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_failed_cold_launch_restores_inside_the_held_lock_without_deadlock(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary()
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)
    repl._active_heavy = (primary, PRIMARY_URL)
    _wait_ready_fails_at(monkeypatch, PEER_URL)
    watch = LockWatch(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model qwen-gguf"), 5)

    # One entry: `lifecycle.swap` restores the incumbent itself, inside the REPL's lock, without
    # asking for it again (R-13-25). The managed swap-back below still restores through
    # `_restore_incumbent`, which nests, so its count stays 2.
    assert watch.entries == 1
    assert watch.opens == 1
    assert watch.held_at_start == [{watch.key}, {watch.key}]  # the peer's start, and the restore's
    assert "Launch of llamacpp-local failed" in channel.messages[-1] and "Restored" in channel.messages[-1]
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_the_managed_restart_that_is_the_swap_back_takes_the_lock_once(
        tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    primary = _llama_primary()
    _, repl, channel, agent = _session(tmp_path, peer, server=primary)
    _on_the_cold_peer(repl, agent, peer)
    watch = LockWatch(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash(f"/model {PRIMARY_SERVED}"), 5)

    assert watch.entries == 1 and watch.opens == 1
    assert watch.held_at_start == [{watch.key}]
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_a_failed_swap_back_restores_the_peer_inside_the_held_lock(
        tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    primary = _llama_primary()
    _, repl, channel, agent = _session(tmp_path, peer, server=primary)
    _on_the_cold_peer(repl, agent, peer)
    _wait_ready_fails_at(monkeypatch, PRIMARY_URL)
    watch = LockWatch(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash(f"/model {PRIMARY_SERVED}"), 5)

    assert watch.entries == 2 and watch.opens == 1
    assert channel.messages[-1].startswith("Model swap failed:") and "Restored" in channel.messages[-1]
    assert repl._active_heavy == (peer.lifecycle, PEER_URL)
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_a_direct_restore_takes_its_own_lock(tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)
    watch = LockWatch(monkeypatch, repl)

    assert await repl._restore_incumbent((peer.lifecycle, PEER_URL)) == (peer.lifecycle.model, None)

    assert watch.entries == 1 and watch.opens == 1
    assert watch.held_at_start == [{watch.key}]
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_a_refused_cold_launch_never_touches_the_lock(tmp_path, boundaries, monkeypatch):
    _, repl, channel, _ = _session(tmp_path, _declared_only_peer())
    watch = LockWatch(monkeypatch, repl)

    await repl._handle_slash("/model qwen-gguf")

    assert watch.entries == 0 and watch.opens == 0
    assert not managed_server.lock_path(repl._server_config_dir).exists()


@pytest.mark.asyncio
async def test_a_lock_held_by_another_process_makes_the_swap_wait_and_say_so_once(
        tmp_path, boundaries, monkeypatch):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    monkeypatch.setattr(provider_client, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    path = managed_server.lock_path(repl._server_config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    foreign = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(foreign, fcntl.LOCK_EX)
    asyncio.get_running_loop().call_later(0.3, os.close, foreign)

    await asyncio.wait_for(repl._handle_slash("/model qwen-gguf"), 5)

    assert channel.messages.count(lifecycle.LAUNCH_LOCK_WAIT_LINE) == 1
    assert len(boundaries.cmds) == 1  # then the launch went ahead
    assert any(m.startswith("Switched to qwen-gguf") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_lock_held_by_this_sessions_own_launch_never_says_another_localharness(
        tmp_path, monkeypatch):
    """A relaunch still unwinding after Esc (or running for another caller) holds the lock from a task of
    this very process: the person waiting is told it is this session's launch, not another process's."""
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    monkeypatch.setattr(provider_client, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    inside, release = asyncio.Event(), asyncio.Event()

    async def own_launch():
        async with lifecycle.launch_lock(repl._server_config_dir):
            inside.set()
            await release.wait()

    holder = asyncio.create_task(own_launch())
    await inside.wait()
    asyncio.get_running_loop().call_later(0.3, release.set)

    async def waiter():
        async with repl._launch_lock():
            pass

    await asyncio.wait_for(waiter(), 5)
    await holder

    assert channel.messages.count(lifecycle.LAUNCH_LOCK_LOCAL_WAIT_LINE) == 1
    assert lifecycle.LAUNCH_LOCK_WAIT_LINE not in channel.messages


def test_the_wait_line_names_the_holder_by_process(tmp_path):
    assert lifecycle.launch_lock_wait_line(tmp_path) == lifecycle.LAUNCH_LOCK_WAIT_LINE
    key = managed_server.lock_path(tmp_path).resolve()
    lifecycle._HELD[key] = [object(), 1]
    try:
        assert lifecycle.launch_lock_wait_line(tmp_path) == lifecycle.LAUNCH_LOCK_LOCAL_WAIT_LINE
    finally:
        lifecycle._HELD.pop(key, None)


@pytest.mark.asyncio
async def test_a_session_with_no_config_dir_launches_with_no_lock_file_anywhere(tmp_path, monkeypatch):
    """`/model` is unavailable without a config dir, but the restore and the relaunch can still be driven
    on such a REPL (the model-swap tests build one): they take no lock and create no file."""
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)
    repl._config_dir = None
    calls: list = []

    async def activate(self, spec, config_dir, base_url, **kw):
        calls.append((config_dir, dict(lifecycle._HELD)))
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["qwen-gguf"], handle=1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    before = sorted(str(p) for p in tmp_path.rglob("*"))

    assert await asyncio.wait_for(repl._restore_incumbent((peer.lifecycle, PEER_URL)), 5) == (
        peer.lifecycle.model, None)

    assert calls == [(None, {})]
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert not list(tmp_path.rglob("launch.lock"))
