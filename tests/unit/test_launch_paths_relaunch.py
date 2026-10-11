# ruff: noqa: F401, F811
"""The reconnect's relaunch is the REPL's sixth launch path (Phase 7, plan 05, D-22..D-25, D-27(6), D-47).

`OrchestratorREPL._build_relaunch` returns the zero-argument closure a recovery flight calls when the
server the harness launched has died. It is the same launch every other path makes: `activate` of the
spec object it was built from, the cold branch's `peer=` and `autosense=`, the key in the child's
environment and `--host 127.0.0.1` last in the argv, under the launch lock with a liveness re-check.
Everything the three process boundaries cross (`stop_server`, `start_server`, `wait_ready`) is stubbed.
"""
from __future__ import annotations

import asyncio
import fcntl
import os

import pytest

from localharness.provider import client as provider_client
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from localharness.provider.server import LaunchRefused
from tests.unit.launch_support import stub_liveness
from tests.unit.reconnect_support import write_live_pidfile
from tests.unit.test_launch_paths_repl import (
    NO_LLAMA_PRIMARY,
    PEER_URL,
    PRIMARY_URL,
    SENTINEL,
    _capture_start_kw,
    _declared_only_peer,
    _launchable_peer,
    _liveness_not_launched,
    _llama_primary,
    _no_machine_probes,
    _session,
    boundaries,
)


def _primary_relaunch(repl, spec, *, launch_key=None, peer=None, ptype="llamacpp", url=PRIMARY_URL):
    return repl._build_relaunch(spec, url, repl._server_config_dir, launch_key or {}, peer, ptype)


@pytest.mark.asyncio
async def test_a_relaunch_starts_one_server_loopback_last(tmp_path, boundaries):
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)

    live = await _primary_relaunch(repl, primary)()

    assert len(boundaries.cmds) == 1
    assert boundaries.cmds[0][-2:] == ["--host", "127.0.0.1"]
    assert live is not None and live.served_models == ["qwen-gguf"]


@pytest.mark.asyncio
async def test_a_keyed_relaunch_sends_the_key_in_the_environment_never_on_argv(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary(require_api_key=True)
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    started = _capture_start_kw(monkeypatch)

    await _primary_relaunch(repl, primary, launch_key={"api_key": SENTINEL})()

    (call,) = started
    assert call["api_key"] == SENTINEL
    assert SENTINEL not in " ".join(call["cmd"])


@pytest.mark.asyncio
async def test_a_relaunch_passes_the_cold_branchs_peer_autosense_and_the_identical_spec(
        tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    harness, repl, _, _ = _session(tmp_path, peer)
    seen: list[tuple] = []
    real = lifecycle.SpawnedProcessStrategy.activate

    async def spy(self, spec, config_dir, base_url, **kw):
        seen.append((spec, config_dir, base_url, kw))
        return await real(self, spec, config_dir, base_url, **kw)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", spy)

    await repl._handle_slash("/model qwen-gguf")  # the cold branch
    (cold,) = seen
    assert repl._active_heavy == (peer.lifecycle, PEER_URL)

    live = await repl._build_relaunch(
        peer.lifecycle, PEER_URL, repl._server_config_dir, {}, repl._peer_name_for(peer.lifecycle, PEER_URL),
        "llamacpp")()

    assert live is not None and len(seen) == 2
    spec, cfg_dir, base_url, kw = seen[1]
    assert spec is peer.lifecycle is cold[0]
    assert (cfg_dir, base_url) == (cold[1], cold[2])
    assert kw["peer"] == cold[3]["peer"] == "llamacpp-local"
    assert kw["autosense"] is cold[3]["autosense"] is harness.autosense
    assert kw["timeout_seconds"] == 1800.0  # relaunch_budget() outside a flight


@pytest.mark.asyncio
async def test_a_relaunch_surfaces_the_decisions_note_once(tmp_path, boundaries):
    key_dir = tmp_path / "keys"
    key_dir.mkdir()
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key_dir))
    _, repl, channel, _ = _session(tmp_path, peer)

    await repl._build_relaunch(
        peer.lifecycle, PEER_URL, repl._server_config_dir, {"api_key": SENTINEL}, "llamacpp-local",
        "llamacpp")()
    await repl._drain_relaunch_reports()  # the report is sent after the relaunch returns (WR-17)

    expected = (f"--api-key-file {key_dir} unreadable: not a regular file — "
                "launched on 127.0.0.1 for this run")
    assert channel.messages.count(expected) == 1
    assert boundaries.cmds[0][-2:] == ["--host", "127.0.0.1"]


@pytest.mark.asyncio
async def test_a_refused_relaunch_raises_its_refusal_and_starts_nothing(tmp_path, boundaries):
    primary = _llama_primary(binary=None)
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)

    with pytest.raises(LaunchRefused) as raised:
        await _primary_relaunch(repl, primary)()

    assert str(raised.value) == NO_LLAMA_PRIMARY
    assert boundaries.cmds == [] and boundaries.stops == [] and boundaries.waits == []
    assert channel.messages == []


@pytest.mark.asyncio
async def test_the_first_launchs_pending_lines_are_sent_once_and_a_refusal_sends_none(
        tmp_path, boundaries):
    primary = _llama_primary()
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)
    repl._pending_launch_lines = ["first-launch line"]

    await _primary_relaunch(repl, primary)()
    await _primary_relaunch(repl, primary)()

    assert channel.messages.count("first-launch line") == 1
    assert repl._pending_launch_lines == []

    refused = _llama_primary(binary=None)
    _, repl2, channel2, _ = _session(tmp_path, _launchable_peer(), server=refused)
    repl2._pending_launch_lines = ["first-launch line"]
    with pytest.raises(LaunchRefused):
        await _primary_relaunch(repl2, refused)()
    assert channel2.messages == [] and repl2._pending_launch_lines == ["first-launch line"]


@pytest.mark.asyncio
async def test_the_window_is_announced_with_the_endpoint_and_served_model(tmp_path, boundaries):
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    seen: list[tuple] = []

    async def record(live, provider_type, model_id):
        seen.append((live, provider_type, model_id))

    repl._announce_window = record

    live = await _primary_relaunch(repl, primary, ptype="llamacpp")()
    await _primary_relaunch(repl, primary, ptype=None)()
    await repl._drain_relaunch_reports()  # the report is sent after the relaunch returns (WR-17)

    assert seen[0] == (live, "llamacpp", "qwen-gguf")
    assert seen[1][1:] == ("llamacpp", "qwen-gguf")  # no endpoint type: the spec's runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [RuntimeError("discord send failed"), ConnectionResetError("phone gone"),
                                   ValueError("bad window line")])
async def test_a_report_that_fails_never_fails_the_relaunch_that_succeeded(tmp_path, boundaries, fault):
    """WR-08/WR-53: the window lines, the decision's note and the first-launch lines are sent from inside
    the closure the flight awaits; a channel that fails while saying them costs the lines, never the
    launch: the relaunch still returns its live endpoint."""
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    repl._pending_launch_lines = ["first-launch line"]

    async def broken(*_a, **_k):
        raise fault

    repl._send_info = broken
    repl._announce_window = broken

    live = await _primary_relaunch(repl, primary)()

    assert live is not None and live.served_models == ["qwen-gguf"]
    assert len(boundaries.cmds) == 1


@pytest.mark.asyncio
async def test_a_slow_report_is_not_part_of_the_relaunch(tmp_path, boundaries):
    """WR-17: the relaunch returns as soon as activate does; its window lines are sent afterwards, so a
    slow channel never holds the recovery flight (and its bound) open."""
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    sent: list[str] = []

    async def slow_window(live, provider_type, model_id):
        await asyncio.sleep(0.5)
        sent.append(model_id)

    repl._announce_window = slow_window
    loop = asyncio.get_running_loop()
    started = loop.time()
    live = await _primary_relaunch(repl, primary)()
    assert loop.time() - started < 0.4
    assert live is not None and sent == []
    await repl._drain_relaunch_reports()
    assert sent == ["qwen-gguf"]  # still said, best-effort, after the relaunch


@pytest.mark.asyncio
async def test_a_relaunch_whose_report_crosses_the_bound_is_still_a_success(tmp_path, boundaries, monkeypatch):
    """WR-17 through a real flight: activate returns at about 0.3 s of a 0.5 s bound and the report takes
    0.5 s more. The wait ends well; it is not "the model server did not come back"."""
    from tests.unit.reconnect_support import write_dead_pidfile

    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    cfg = repl._server_config_dir
    monkeypatch.setattr(provider_client, "_QUEUE_VISIBILITY_SECONDS", 0.02)
    write_dead_pidfile(cfg)
    launched: list[int] = []

    async def slow_activate(self, spec, config_dir, base_url, **kw):
        await asyncio.sleep(0.3)  # activate's own readiness wait saw /models answer at about 0.3 s
        write_live_pidfile(cfg)
        launched.append(1)
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["qwen-gguf"], handle=1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", slow_activate)

    async def wait_ready(base_url, config_dir=None, **kw):
        if managed_server.server_pid(cfg) is None:
            if config_dir is not None:
                raise RuntimeError("llama.cpp exited during startup. Log tail:\n")
            raise TimeoutError("nothing answers")  # the look at the address before relaunching
        return ["qwen-gguf"]

    monkeypatch.setattr(managed_server, "wait_ready", wait_ready)

    async def slow_window(live, provider_type, model_id):
        await asyncio.sleep(0.5)  # crosses the 0.5 s bound

    repl._announce_window = slow_window
    pair = lifecycle.ServerRecovery(PRIMARY_URL, cfg, _primary_relaunch(repl, primary), "llamacpp")

    async def nobody():
        return None

    await asyncio.wait_for(pair.recover(base_url=PRIMARY_URL, api_key="none", wait_seconds=0.5,
                                        on_lost=nobody), 5)
    await repl._drain_relaunch_reports()
    assert launched == [1]


@pytest.mark.asyncio
async def test_draining_with_a_bound_lets_a_quick_report_finish_and_cancels_a_stuck_one(tmp_path):
    """IN-25 / IN-63: at exit a quick report still gets said; one that would hang is cancelled, never
    left to asyncio.run's teardown."""
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=_llama_primary())
    said: list[str] = []

    async def quick():
        await asyncio.sleep(0.05)
        said.append("quick")

    async def stuck():
        await asyncio.Event().wait()

    loop = asyncio.get_running_loop()
    tasks = [loop.create_task(quick()), loop.create_task(stuck())]
    repl._relaunch_reports.update(tasks)
    for task in tasks:
        task.add_done_callback(repl._relaunch_reports.discard)

    started = loop.time()
    await repl._drain_relaunch_reports(timeout=0.3)
    assert loop.time() - started < 1.0
    assert said == ["quick"]
    assert tasks[1].cancelled()
    assert repl._relaunch_reports == set()


@pytest.mark.asyncio
async def test_leaving_the_repl_drains_the_relaunch_reports(tmp_path, monkeypatch):
    """IN-25 / IN-63: the REPL's exit path drains the pending reports, bounded."""
    from localharness.cli import repl as repl_mod

    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=_llama_primary())
    seen: list[float | None] = []

    async def drain(timeout=None):
        seen.append(timeout)

    async def nothing():
        return None

    repl._drain_relaunch_reports = drain
    repl._use_input_box = lambda: False
    repl._run_classic = nothing
    repl._prefetch_model_cache = nothing
    repl._subscribe_pending = lambda: None  # this session's stand-in bus has no subscribe
    repl._unsubscribe_pending = lambda: None
    await repl.run()
    assert seen == [repl_mod.RELAUNCH_REPORT_DRAIN_SECONDS]


@pytest.mark.asyncio
async def test_a_live_pidfile_under_the_lock_means_no_launch(tmp_path, boundaries):
    primary = _llama_primary()
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)
    write_live_pidfile(repl._server_config_dir)

    assert await _primary_relaunch(repl, primary)() is None
    assert boundaries.cmds == [] and channel.messages == []


@pytest.mark.asyncio
async def test_a_strategy_that_says_alive_under_the_lock_means_no_launch(
        tmp_path, boundaries, monkeypatch):
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    stub_liveness(monkeypatch, True)

    assert await _primary_relaunch(repl, primary)() is None
    assert boundaries.cmds == []


@pytest.mark.asyncio
async def test_a_liveness_that_raises_counts_as_not_alive(tmp_path, boundaries, monkeypatch):
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)

    def broken(self, spec, config_dir, **kw):
        raise OSError("cannot ask")

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "liveness", broken)

    assert await _primary_relaunch(repl, primary)() is not None
    assert len(boundaries.cmds) == 1


@pytest.mark.asyncio
async def test_the_launch_lock_is_held_while_the_server_starts(tmp_path, monkeypatch, boundaries):
    primary = _llama_primary()
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)
    held: list[set] = []
    monkeypatch.setattr(managed_server, "start_server",
                        lambda cfg, cmd, **kw: held.append(set(lifecycle._HELD)) or 4321)

    await _primary_relaunch(repl, primary)()

    key = managed_server.lock_path(repl._server_config_dir).resolve()
    assert held == [{key}]
    assert lifecycle._HELD == {}


@pytest.mark.asyncio
async def test_no_config_dir_means_no_lock_and_nothing_on_disk(tmp_path):
    _, repl, _, _ = _session(tmp_path, _launchable_peer())
    repl._config_dir = None
    before = sorted(p.name for p in tmp_path.iterdir())

    async with repl._launch_lock():
        assert lifecycle._HELD == {}

    assert sorted(p.name for p in tmp_path.iterdir()) == before


@pytest.mark.asyncio
async def test_a_lock_held_elsewhere_tells_the_person_once(tmp_path, monkeypatch):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    monkeypatch.setattr(provider_client, "_QUEUE_VISIBILITY_SECONDS", 0.05)
    path = managed_server.lock_path(repl._server_config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    foreign = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(foreign, fcntl.LOCK_EX)
    asyncio.get_running_loop().call_later(0.3, os.close, foreign)

    async with repl._launch_lock():
        pass

    assert channel.messages.count(lifecycle.LAUNCH_LOCK_WAIT_LINE) == 1


def test_a_peer_name_is_given_only_for_the_peers_own_lifecycle(tmp_path):
    peer = _declared_only_peer()
    _, repl, _, _ = _session(tmp_path, peer)

    assert repl._peer_name_for(peer.lifecycle, PEER_URL) == "llamacpp-local"
    assert repl._peer_name_for(peer.lifecycle.model_copy(), PEER_URL) is None
    assert repl._peer_name_for(repl._harness.server, PRIMARY_URL) is None
