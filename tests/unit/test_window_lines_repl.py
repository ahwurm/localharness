"""The REPL's swap report names the window a launch got (Phase 2, plan 07, D-16 / D-22 / D-36).

The cold `/model` branch and the managed restart pass `autosense` into `activate`, send the launch's
window line (one line, never an undo line, whatever the numbers) and record the triple;
`_restore_incumbent` passes `autosense` and announces nothing. A fake strategy stands in for the
process boundary: no server, no HTTP, no nvidia-smi.
"""
from __future__ import annotations

import pytest

from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from localharness.provider.autosense import WindowReport
from localharness.provider.lifecycle import LiveEndpoint
from localharness.provider.speed_stats import default_speed_stats_path, last_window_report
from tests.unit.test_launch_paths_repl import (
    PEER_URL,
    PRIMARY_URL,
    _launchable_peer,
    _llama_primary,
    _on_the_cold_peer,
    _session,
)

FIT_LINE = "window: served 14,950 · trained 262,144 · fits 14,950 (free 6.0 GiB of 32.6; weights 18.5 GiB)"


def _report(**over) -> WindowReport:
    base = dict(served=14950, trained=262144, fits=14950, free_bytes=6442450944,
                total_bytes=35004153856, weights_bytes=19864223744, asked=14950, mode="fit",
                verified=True)
    base.update(over)
    return WindowReport(**base)


class FakeStrategy:
    """A strategy whose `activate` returns the scripted LiveEndpoint and records its kwargs."""

    def __init__(self, live: LiveEndpoint | None = None, fail: Exception | None = None) -> None:
        self.live = live
        self.fail = fail
        self.activations: list[dict] = []

    async def stop(self, spec, config_dir, **kw) -> None:
        return None

    async def activate(self, spec, config_dir, base_url, **kw):
        self.activations.append(kw)
        if self.fail is not None:
            raise self.fail
        return self.live


@pytest.fixture
def fake(monkeypatch):
    """Route every strategy lookup to one FakeStrategy the test scripts through `.live`."""
    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    strat = FakeStrategy()
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: strat)
    return strat


def _live(window: WindowReport | None, served: str = "qwen") -> LiveEndpoint:
    return LiveEndpoint(base_url=PEER_URL, served_models=[served], handle=1, window=window)


def _window_lines(channel) -> list[str]:
    return [m for m in channel.messages if m.startswith("window:") or m.lstrip().startswith("→")]


def _ledger(tmp_path, ptype, model):
    return last_window_report(default_speed_stats_path(tmp_path), ptype, model)


# --- the cold branch ---------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_cold_launch_sends_its_one_window_line_after_launching_and_records_it(
        tmp_path, fake):
    fake.live = _live(_report())
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    launching = next(i for i, m in enumerate(channel.messages) if m.startswith("Launching "))
    assert channel.messages[launching + 1:].count(FIT_LINE) == 1
    assert channel.messages.index(FIT_LINE) > launching
    assert not any(m.lstrip().startswith("→") for m in channel.messages)  # D-36: one line
    rec = _ledger(tmp_path, "llamacpp", "qwen")
    assert (rec.served, rec.trained, rec.fits) == (14950, 262144, 14950)


@pytest.mark.asyncio
async def test_the_window_line_precedes_the_switched_line(tmp_path, fake):
    fake.live = _live(_report())
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    at = channel.messages.index(FIT_LINE)
    assert channel.messages[at + 1].startswith("Switched to qwen on llamacpp-local")


@pytest.mark.asyncio
async def test_the_cold_launch_hands_activate_the_harness_autosense(tmp_path, fake):
    fake.live = _live(_report())
    harness, repl, _, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert fake.activations[-1]["autosense"] is harness.autosense


@pytest.mark.asyncio
async def test_a_stand_in_harness_without_autosense_launches_with_none(tmp_path, fake):
    fake.live = _live(None)
    harness, repl, channel, _ = _session(tmp_path, _launchable_peer())

    class NoAutosense:
        def __getattr__(self, name):
            if name == "autosense":
                raise AttributeError(name)
            return getattr(harness, name)

    repl._harness = NoAutosense()

    await repl._handle_slash("/model qwen-gguf")

    assert fake.activations[-1]["autosense"] is None
    assert any(m.startswith("Switched to qwen") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_launch_with_an_unverified_window_ends_not_verified_and_completes(tmp_path, fake):
    fake.live = _live(_report(served=None, verified=False))
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    line = next(m for m in channel.messages if m.startswith("window:"))
    assert line.endswith(" (not verified)")
    assert not any("failed" in m for m in channel.messages)
    assert any(m.startswith("Switched to qwen") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_launch_that_served_less_than_asked_names_what_was_asked_and_completes(
        tmp_path, fake):
    fake.live = _live(_report(served=8192))
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    line = next(m for m in channel.messages if m.startswith("window:"))
    assert line.endswith(" (asked 14,950)")
    assert not any("failed" in m.lower() for m in channel.messages)
    assert any(m.startswith("Switched to qwen") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_launch_without_a_window_sends_no_line_and_writes_no_ledger(tmp_path, fake):
    fake.live = _live(None)
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert _window_lines(channel) == []
    assert not default_speed_stats_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_a_launch_that_learned_nothing_keeps_the_recorded_baseline(tmp_path, fake):
    """WR-06: an all-unknown report must not overwrite a good record with Nones."""
    from localharness.provider.speed_stats import record_window_report

    record_window_report(default_speed_stats_path(tmp_path), "llamacpp", "qwen", 65536, 262144, 114688)
    fake.live = _live(_report(served=None, trained=None, fits=None, verified=False))
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    rec = _ledger(tmp_path, "llamacpp", "qwen")
    assert (rec.served, rec.trained, rec.fits) == (65536, 262144, 114688)
    assert any(m.startswith("Switched to qwen") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_ledger_write_error_never_breaks_the_swap(tmp_path, fake, monkeypatch):
    from localharness.provider import speed_stats

    def boom(*a, **k):
        raise OSError("read-only state dir")

    monkeypatch.setattr(speed_stats, "record_window_report", boom)
    fake.live = _live(_report())
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert FIT_LINE in channel.messages
    assert any(m.startswith("Switched to qwen") for m in channel.messages)
    assert not any("failed" in m.lower() for m in channel.messages)


@pytest.mark.asyncio
async def test_a_failed_cold_launch_announces_no_window(tmp_path, fake):
    fake.fail = TimeoutError("llama-server never became ready")
    fake.live = _live(_report())
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert _window_lines(channel) == []
    assert any(m.startswith("Launch of llamacpp-local failed") for m in channel.messages)


# --- the managed restart ------------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_a_fit_by_server_restart_sends_its_line_and_records_it(tmp_path, fake, monkeypatch):
    primary = _llama_primary()
    served = primary.served_name  # the record's own name, so the two files cannot drift (D-66)
    fake.live = LiveEndpoint(
        base_url=PRIMARY_URL, served_models=[served], handle=1,
        window=_report(served=65536, trained=None, fits=None, free_bytes=None, total_bytes=None,
                       weights_bytes=None, asked=None, mode="fit-by-server"))
    harness, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary)

    await repl._handle_slash(f"/model {served}")

    expected = "window: served by llama-server's --fit · served 65,536"
    assert channel.messages.count(expected) == 1
    restarting = next(i for i, m in enumerate(channel.messages) if m.startswith("Restarting managed"))
    assert channel.messages.index(expected) > restarting
    assert fake.activations[-1]["autosense"] is harness.autosense
    rec = _ledger(tmp_path, harness.provider.provider_type, served)
    assert rec is not None and rec.served == 65536


@pytest.mark.asyncio
async def test_a_restart_with_no_window_sends_no_line(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    fake.live = LiveEndpoint(base_url=PRIMARY_URL, served_models=["cached-b"], handle=1)
    harness, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model cached-b")

    assert _window_lines(channel) == []
    # no window was recorded; the landing's own `last_served` entry (persist_landed) may be in the file
    assert _ledger(tmp_path, harness.provider.provider_type, "cached-b") is None


# --- _restore_incumbent ------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_restore_passes_autosense_but_announces_and_records_nothing(tmp_path, fake):
    peer = _launchable_peer()
    fake.live = _live(_report())
    harness, repl, channel, _ = _session(tmp_path, peer)
    repl._active_heavy = None

    label, why = await repl._restore_incumbent((peer.lifecycle, PEER_URL))

    assert label == peer.lifecycle.model and why is None
    assert fake.activations[-1]["autosense"] is harness.autosense
    assert _window_lines(channel) == []
    assert not default_speed_stats_path(tmp_path).exists()


@pytest.mark.asyncio
async def test_a_swap_back_failure_restores_with_autosense_and_stays_silent_on_the_window(
        tmp_path, fake, monkeypatch):
    primary = _llama_primary()
    peer = _launchable_peer()
    harness, repl, channel, agent = _session(tmp_path, peer, server=primary)
    _on_the_cold_peer(repl, agent, peer)
    calls: list[dict] = []
    ok = _live(_report())

    async def activate(spec, config_dir, base_url, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise TimeoutError("never ready")
        return ok

    fake.activate = activate

    await repl._handle_slash(f"/model {primary.served_name}")

    assert len(calls) == 2 and all(c["autosense"] is harness.autosense for c in calls)
    assert _window_lines(channel) == []
    assert any(m.startswith("Model swap failed") for m in channel.messages)
