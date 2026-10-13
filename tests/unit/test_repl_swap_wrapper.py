# ruff: noqa: F401, F811
"""The REPL's cold `/model <peer>` swap runs on `lifecycle.swap`, said and remembered the REPL's way
(plan 13-08, spec 20 §8.2, D-07, D-11, D-19..D-22, SEC-02, SEC-04).

The three process boundaries a swap crosses (`stop_server`, `start_server`, `wait_ready`) record into one
ordered event list; the channel records the style of each line; the config dir is `tmp_path`. Nothing is
started, no service manager is asked, no real port is dialled.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from localharness.cli import model_ops
from localharness.cli.repl import GPU_HOLDER_CLAUSE, PERSIST_SELECTION_FAILED
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import load_overlay
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from localharness.provider.speed_stats import default_speed_stats_path, last_served
from tests.unit.test_launch_paths_repl import (
    PEER_URL,
    PRIMARY_SERVED,
    PRIMARY_URL,
    _launchable_peer,
    _liveness_not_launched,
    _llama_primary,
    _no_machine_probes,
)
from tests.unit.test_repl_model_cmd import FakeChannel, _harness, _heavy_primary, _xrepl

PROVIDER_KEY = "sk-prov-7d3f9a"
PLANTED_TAIL = (
    "llama-server exited: Authorization: Bearer sk-planted-91ab\n"
    "main: api_keys: ****abcd\n"
    "listening on http://u:p@127.0.0.1:8081\n"
    f"key {PROVIDER_KEY}"
)
PLANTED = ("sk-planted-91ab", "abcd", "u:p", PROVIDER_KEY)


class StyledChannel(FakeChannel):
    """A channel that keeps the style each line was sent with."""

    def __init__(self) -> None:
        super().__init__()
        self.styled: list[tuple[str, str | None]] = []

    async def send_message(self, text, metadata=None):
        await super().send_message(text, metadata=metadata)
        self.styled.append((text, (metadata or {}).get("style")))


class Rig:
    """stop / start / wait recorded in order; `wait_raises` maps a base URL to what its wait raises."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.wait_raises: dict[str, BaseException] = {}
        self.stop_raises: list[BaseException] = []

    @property
    def waits(self) -> list[str]:
        return [e for e in self.events if e.startswith("wait:")]

    @property
    def starts(self) -> list[str]:
        return [e for e in self.events if e.startswith("start:")]


@pytest.fixture
def rig(monkeypatch):
    r = Rig()
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])

    def fake_stop(cfg, launch="binary", **kw):
        r.events.append("stop")
        if r.stop_raises:
            raise r.stop_raises.pop(0)
        return True

    def fake_start(cfg, cmd, **kw):
        r.events.append("start:" + " ".join(cmd))
        return 4321

    async def fake_wait(base_url, config_dir=None, **kw):
        r.events.append(f"wait:{base_url}")
        if base_url in r.wait_raises:
            raise r.wait_raises[base_url]
        return ["flash-next" if base_url == PEER_URL else "model-a"]

    monkeypatch.setattr(managed_server, "stop_server", fake_stop)
    monkeypatch.setattr(managed_server, "start_server", fake_start)
    monkeypatch.setattr(managed_server, "wait_ready", fake_wait)
    return r


def _build(tmp_path, *, provider_key=None, primary=None, **repl_kw):
    """A REPL on a llama.cpp primary that the harness launched, with the cold peer `flash-next`.
    Persistence is the real one (no stub): the overlay and the ledger land under `tmp_path`."""
    peer = _launchable_peer(name="flash-next", served="flash-next")
    primary = primary if primary is not None else _llama_primary()
    harness = _harness(server=primary, extra_endpoints=[peer])
    if provider_key is not None:
        harness.provider.api_key = SecretStr(provider_key)
    repl, _, agent, _ = _xrepl(
        tmp_path, harness, {PRIMARY_URL: ["model-a"], PEER_URL: "unreachable"}, **repl_kw)
    channel = StyledChannel()
    repl._channel = channel

    async def _refresh(model, *, base_url=None, provider_type=None):
        return ""

    repl._refresh_token_counter = _refresh
    repl._active_heavy = (primary, PRIMARY_URL)
    return harness, repl, channel, agent, peer, primary


def _spy_swap(monkeypatch, repl):
    """lifecycle.swap spied: every call's arguments, whether the REPL's launch lock was held during it,
    and the outcome it returned."""
    seen = SimpleNamespace(calls=[], held=[], outcomes=[])
    real = lifecycle.swap
    key = managed_server.lock_path(repl._server_config_dir).resolve()

    async def spy(incumbent, target, **kw):
        seen.calls.append((incumbent, target, kw))
        seen.held.append(key in lifecycle._HELD)
        out = await real(incumbent, target, **kw)
        seen.outcomes.append(out)
        return out

    monkeypatch.setattr(lifecycle, "swap", spy)
    return seen


def _count_lock(repl):
    entries = []
    real = repl._launch_lock

    def counting():
        entries.append(1)
        return real()

    repl._launch_lock = counting
    return entries


# --- the tracer ----------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_cold_swap_stops_the_primary_launches_the_peer_says_who_held_the_gpu_and_lands(
        tmp_path, rig, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    model_ops.write_pending_preset(harness, "flash-next", config_dir=tmp_path)
    assert load_overlay(tmp_path / "overrides.yaml")["pending_preset"] == "flash-next"

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert [e.split(":")[0] for e in rig.events] == ["stop", "start", "wait"]
    assert "/x/m.gguf" in rig.events[1] and rig.events[2] == f"wait:{PEER_URL}"
    launching = next(m for m in channel.messages if m.startswith("Launching flash-next"))
    assert "the swap can take several minutes..." in launching
    assert GPU_HOLDER_CLAUSE.format(holder="qwen-primary") in launching
    assert launching.index("several minutes...") < launching.index("was stopped to free the GPU.")
    assert channel.messages[-1].startswith("Switched to flash-next on flash-next")
    assert repl._active_heavy == (peer.lifecycle, PEER_URL)
    assert agent._llm.config.base_url == PEER_URL
    # persisted through persist_landed: the selection, the ledger entry, the pending preset gone
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["active_endpoint"]["base_url"] == PEER_URL
    assert overlay["active_endpoint"]["model"] == "flash-next"
    assert "pending_preset" not in overlay and harness.pending_preset is None
    assert last_served(default_speed_stats_path(tmp_path)) == "flash-next"


@pytest.mark.asyncio
async def test_the_swap_is_called_once_for_the_trust_resolved_peer_inside_the_launch_lock(
        tmp_path, rig, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    seen = _spy_swap(monkeypatch, repl)
    entries = _count_lock(repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert len(seen.calls) == 1 and seen.held == [True]
    incumbent, target, kw = seen.calls[0]
    assert target.peer == "flash-next" and target.base_url == PEER_URL
    assert target.spec is peer.lifecycle
    assert incumbent.spec is primary and incumbent.base_url == PRIMARY_URL
    assert len(entries) == 1
    assert kw["config_dir"] == repl._server_config_dir and callable(kw["redact"])
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_a_swap_target_the_machine_withholds_is_refused_before_anything_is_stopped(
        tmp_path, rig, monkeypatch):
    from localharness.config.trust import LAUNCH_KIND

    trust = MachineTrust(
        withheld={"config.yaml": frozenset({(LAUNCH_KIND, "extra_endpoints.flash-next.lifecycle")})},
        line="Not applying 1 change(s) in the machine file that you have not confirmed")
    harness, repl, channel, agent, peer, primary = _build(tmp_path, machine_trust=trust)
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert seen.calls == [] and rig.events == []
    assert channel.styled[-1] == (trust.line, "system.error")
    assert repl._active_heavy == (primary, PRIMARY_URL)
