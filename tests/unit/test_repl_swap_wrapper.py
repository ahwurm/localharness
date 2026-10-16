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
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
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


# --- failure, restore, cancellation, redaction -------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_incumbent_that_will_not_stop_is_reported_and_nothing_launches(tmp_path, rig):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    rig.stop_raises.append(RuntimeError("won't die"))

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert channel.styled[-1] == (
        "Could not free the GPU for flash-next: won't die. No change made — the current model "
        "should still be serving.", "system.error")
    assert rig.starts == [] and rig.waits == []
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert not any(m.startswith("Launching") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_failed_launch_restores_the_incumbent_and_the_planted_secrets_never_reach_the_channel(
        tmp_path, rig, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(tmp_path, provider_key=PROVIDER_KEY)
    rig.wait_raises[PEER_URL] = RuntimeError(PLANTED_TAIL)
    seen = _spy_swap(monkeypatch, repl)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    text, style = channel.styled[-1]
    assert style == "system.error"
    assert text.startswith("Launch of flash-next failed:")
    assert f" Restored {primary.model}." in text
    for secret in PLANTED:
        assert secret not in text
        assert not any(secret in m for m in channel.messages)
    # the swap's own outcome carries no secret and no raw tail either
    (outcome,) = seen.outcomes
    for secret in PLANTED:
        assert secret not in str(outcome.error) and secret not in repr(outcome.error)
    assert outcome.error.__cause__ is None and outcome.error.__context__ is None
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_a_planted_key_in_a_timeout_is_masked_too(tmp_path, rig):
    harness, repl, channel, agent, peer, primary = _build(tmp_path, provider_key=PROVIDER_KEY)
    rig.wait_raises[PEER_URL] = TimeoutError(PLANTED_TAIL)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    text = channel.messages[-1]
    assert text.startswith("Launch of flash-next failed:")
    assert not any(secret in text for secret in PLANTED)


@pytest.mark.asyncio
async def test_the_managed_restart_failure_is_redacted_the_same_way(tmp_path, rig, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(
        tmp_path, provider_key=PROVIDER_KEY, primary=_heavy_primary())
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    rig.wait_raises[PRIMARY_URL] = RuntimeError(PLANTED_TAIL)
    repl._active_heavy = (primary, PRIMARY_URL)

    await asyncio.wait_for(repl._handle_slash("/model cached-b"), 5)

    text, style = channel.styled[-1]
    assert style == "system.error" and text.startswith("Model swap failed:")
    assert not any(secret in text for secret in PLANTED)


@pytest.mark.asyncio
async def test_a_failed_cold_restart_stops_the_launched_model_before_the_record_is_reset(
        tmp_path, rig, monkeypatch, fake_proc):
    """The REPL's cold `/model <other model>` restart mutates the shared record to the new model, then
    the launch dies (or times out, the process still alive). The half-started target is stopped as it
    was launched: the verified stop reads the record's own model, so it runs BEFORE the reset to the old
    one. The pidfile is gone before any incumbent is restored, and the record ends on its old model."""
    harness, repl, channel, agent, peer, primary = _build(tmp_path, primary=_heavy_primary())
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    pidfile = managed_server.server_dir(repl._server_config_dir) / "server.pid"
    launched: list[str] = []

    def fake_start(cfg, cmd, **kw):
        rig.events.append("start")
        launched.append(cmd[cmd.index("--model") + 1])
        fake_proc.process(4321, argv=tuple(cmd), exe="/usr/bin/docker", start_time=900)
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("4321 900", encoding="utf-8")
        return 4321

    def fake_stop(cfg, launch="binary", **kw):
        rig.events.append("stop")
        pidfile.unlink(missing_ok=True)
        return True

    gone_at_restore: list[bool] = []

    async def spy_restore(freed):
        gone_at_restore.append(not pidfile.exists())
        return None, None

    monkeypatch.setattr(managed_server, "start_server", fake_start)
    monkeypatch.setattr(managed_server, "stop_server", fake_stop)
    repl._restore_incumbent = spy_restore
    rig.wait_raises[PRIMARY_URL] = TimeoutError("not ready")
    repl._active_heavy = (primary, PRIMARY_URL)

    await asyncio.wait_for(repl._handle_slash("/model cached-b"), 5)

    assert launched == ["cached-b"]
    # the strategy's own stop before the launch, then the half-started target's after the failure
    assert rig.events[0] == "stop" and rig.events[-1] == "stop" and rig.events.count("stop") == 2
    assert not pidfile.exists() and gone_at_restore == [True]
    assert primary.model == "model-a"                   # reset only after the target was stopped
    assert channel.styled[-1][0].startswith("Model swap failed:")


@pytest.mark.asyncio
async def test_a_cancellation_during_the_peers_wait_leaves_nothing_of_the_target_behind(
        tmp_path, rig):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    rig.wait_raises[PEER_URL] = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await repl._handle_slash("/model flash-next")

    assert repl._active_heavy is None
    assert rig.starts and len(rig.starts) == 1          # the peer's start only: no restore launched
    assert rig.waits == [f"wait:{PEER_URL}"]
    assert rig.events.count("stop") == 2                # the primary freed, the half-started peer stopped
    assert agent._llm.rebinds == []
    assert lifecycle._HELD == {}


# --- the seed ------------------------------------------------------------------------------------- #


def _liveness(monkeypatch, up):
    """Every strategy's liveness answered by `up(spec)` (no docker, pidfile or HTTP is asked)."""
    asked: list = []

    def liveness(self, spec, config_dir=None, **kw):
        asked.append(spec)
        alive = bool(up(spec))
        return lifecycle.Liveness(alive, "pid 1" if alive else None)

    for cls in (lifecycle.SpawnedProcessStrategy, lifecycle.ManagedVllmStrategy,
                lifecycle.DaemonStrategy, lifecycle.LmsStrategy):
        monkeypatch.setattr(cls, "liveness", liveness)
    return asked


def test_serving_heavy_that_is_alive_seeds_the_accelerator_occupant(tmp_path, monkeypatch):
    peer = _launchable_peer(name="flash-next", served="flash-next")
    harness = _harness(server=_llama_primary(), extra_endpoints=[peer])
    _liveness(monkeypatch, lambda spec: True)

    repl, _, _, _ = _xrepl(tmp_path, harness, {}, serving_heavy=(peer.lifecycle, PEER_URL))

    assert repl._active_heavy == (peer.lifecycle, PEER_URL)


def test_a_cpu_only_serving_record_is_not_the_accelerator_occupant(tmp_path, monkeypatch):
    """D-33: the lock tracks GPU heavies. A live `gpu: false` server must not be seeded, or the next
    cold `/model <gpu peer>` would stop a CPU server it was never meant to touch."""
    peer = _launchable_peer(name="flash-next", served="flash-next")
    cpu_spec = peer.lifecycle.model_copy(update={"gpu": False})
    harness = _harness(server=None, extra_endpoints=[peer])
    asked = _liveness(monkeypatch, lambda spec: True)

    repl, _, _, _ = _xrepl(tmp_path, harness, {}, serving_heavy=(cpu_spec, PEER_URL))

    assert repl._active_heavy is None
    assert asked == []


def test_serving_heavy_that_is_dead_falls_back_to_the_primary_seed(tmp_path, monkeypatch):
    peer = _launchable_peer(name="flash-next", served="flash-next")
    primary = _llama_primary()
    harness = _harness(server=primary, extra_endpoints=[peer])
    asked = _liveness(monkeypatch, lambda spec: spec is primary)

    repl, _, _, _ = _xrepl(tmp_path, harness, {}, serving_heavy=(peer.lifecycle, PEER_URL))

    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert asked[0] is peer.lifecycle


def test_no_serving_heavy_keeps_todays_seed_and_no_trust_means_a_machine_trust(tmp_path, monkeypatch):
    primary = _llama_primary()
    harness = _harness(server=primary)
    _liveness(monkeypatch, lambda spec: True)

    repl, _, _, _ = _xrepl(tmp_path, harness, {})

    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert repl._machine_trust == MachineTrust()


def test_a_machine_trust_passed_in_is_the_one_the_session_holds(tmp_path):
    trust = MachineTrust(line="withheld")
    repl, _, _, _ = _xrepl(tmp_path, _harness(), {}, machine_trust=trust)
    assert repl._machine_trust is trust


# --- the wrappers delegate ------------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_restore_and_safe_stop_delegate_to_lifecycle(tmp_path, rig, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    calls: list = []
    real_restore, real_stop = lifecycle.restore_incumbent, lifecycle.safe_stop

    async def restore(heavy, **kw):
        calls.append(("restore", heavy.spec, heavy.base_url, heavy.peer))
        return await real_restore(heavy, **kw)

    async def stop(spec, config_dir, **kw):
        calls.append(("stop", spec, config_dir))
        return await real_stop(spec, config_dir, **kw)

    monkeypatch.setattr(lifecycle, "restore_incumbent", restore)
    monkeypatch.setattr(lifecycle, "safe_stop", stop)
    repl._active_heavy = None

    label, note = await repl._restore_incumbent((primary, PRIMARY_URL))
    await repl._safe_stop(primary)

    assert label == primary.model and note is None
    assert calls[0] == ("restore", primary, PRIMARY_URL, None)
    assert calls[1] == ("stop", primary, repl._server_config_dir)
    assert repl._active_heavy == (primary, PRIMARY_URL)
    assert await repl._restore_incumbent(None) == (None, None)


# --- persistence ---------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_unknown_base_url_persists_nothing(tmp_path):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)

    saved = await repl._persist_landed("http://elsewhere.invalid:9/v1", "x")

    assert saved is False
    assert not (tmp_path / "overrides.yaml").exists()
    assert last_served(default_speed_stats_path(tmp_path)) is None


@pytest.mark.asyncio
async def test_a_primary_landing_clears_the_active_endpoint_and_records_last_served(tmp_path):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    await model_ops.persist_active_endpoint(harness, peer, "flash-next", config_dir=tmp_path)
    assert "active_endpoint" in load_overlay(tmp_path / "overrides.yaml")

    saved = await repl._persist_landed(PRIMARY_URL, "model-b")

    assert saved is True
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert "active_endpoint" not in overlay
    assert overlay["provider"]["default_model"] == "model-b"
    assert last_served(default_speed_stats_path(tmp_path)) == PRIMARY_SERVED


@pytest.mark.asyncio
async def test_a_failed_selection_write_is_said_redacted_and_the_session_carries_on(
        tmp_path, monkeypatch):
    harness, repl, channel, agent, peer, primary = _build(tmp_path, provider_key=PROVIDER_KEY)

    async def boom(*a, **kw):
        raise ValueError(f"cannot write {PROVIDER_KEY}")

    monkeypatch.setattr(model_ops, "persist_landed", boom)
    seen_default: list = []

    async def default(model):
        seen_default.append(model)
        return True

    repl._persist_default_model = default

    saved = await repl._persist_landed(PRIMARY_URL, "model-b")

    assert saved is True and seen_default == ["model-b"]
    text, style = channel.styled[-1]
    assert style == "system.error"
    assert text.startswith(PERSIST_SELECTION_FAILED.split("{")[0])
    assert PROVIDER_KEY not in text


@pytest.mark.asyncio
async def test_a_peer_sharing_the_primarys_address_is_persisted_as_the_peer(tmp_path):
    """The address names the primary first, so a `/model <peer>` landing at a shared address must go by
    the endpoint switched to, or it would clear the selection instead of writing the peer."""
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    peer.base_url = PRIMARY_URL  # the peer shares the primary's address

    saved = await repl._persist_landed(PRIMARY_URL, "flash-next", endpoint=peer)

    assert saved is False  # a peer selection is not a persisted default model
    active = load_overlay(tmp_path / "overrides.yaml")["active_endpoint"]
    assert (active["name"], active["model"]) == ("flash-next", "flash-next")
    assert last_served(default_speed_stats_path(tmp_path)) == "flash-next"


@pytest.mark.asyncio
async def test_persisting_an_active_endpoint_at_a_shared_address_lands_the_peer(tmp_path):
    harness, repl, channel, agent, peer, primary = _build(tmp_path)
    peer.base_url = PRIMARY_URL

    await repl._persist_active_endpoint(peer, "flash-next")

    assert load_overlay(tmp_path / "overrides.yaml")["active_endpoint"]["name"] == "flash-next"
    assert last_served(default_speed_stats_path(tmp_path)) == "flash-next"
