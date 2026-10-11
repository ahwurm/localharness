# ruff: noqa: F401, F811
"""The REPL's reconnect supplier and the launch lock around its swaps (Phase 7, plan 05).

`OrchestratorREPL.server_recovery()` is what the loop asks when the model server drops: one
`ServerRecovery` per (client address, active spec, active address), relaunchable only for the spec the
harness launched at the address the client is using, with that endpoint's own key and its strategy's
config dir. Everything the three process boundaries cross is stubbed; every config dir is `tmp_path`.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.test_launch_paths_repl import (
    PEER_URL,
    PRIMARY_URL,
    SENTINEL,
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
