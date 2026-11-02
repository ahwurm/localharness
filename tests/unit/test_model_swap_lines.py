"""A model swap reads as steps: one `◆ model <state> <name> · clause · …` line as each step happens.

The swap runs the REPL's real code over `lifecycle.swap`, with the three process boundaries a swap crosses recorded
(tests/unit/test_repl_swap_wrapper.py); the channel keeps every line the REPL sends.
"""
# ruff: noqa: F401, F811
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from localharness.channels.terminal import _colorize_rate_notes
from localharness.cli import repl as repl_mod
from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from localharness.provider.autosense import WindowReport, window_lines
from localharness.provider.lifecycle import LiveEndpoint
from localharness.provider.speed_stats import default_speed_stats_path, record_tps
from tests.unit.test_launch_paths_repl import _launchable_peer, _session
from tests.unit.test_launch_paths_repl import PEER_URL as LAUNCH_PEER_URL
from tests.unit.test_launch_paths_repl import _liveness_not_launched, _no_machine_probes  # noqa: F401 — autouse fixtures
from tests.unit.test_repl_swap_wrapper import PEER_URL, _build, rig  # noqa: F401 — the rig is a fixture
from tests.unit.test_spill_report import SCREENSHOT, short_by

GIB = 2**30


def steps(channel) -> list[str]:
    return [m for m in channel.messages if m.startswith("◆ model")]


def seed_speeds(tmp_path, incumbent: float, target: float) -> None:
    ledger = default_speed_stats_path(tmp_path)
    record_tps(ledger, "llamacpp", "qwen-primary", incumbent)
    record_tps(ledger, "llamacpp", "flash-next", target)


@pytest.mark.asyncio
async def test_a_swap_is_launching_stopped_ready_switched_each_line_naming_a_model_once_and_the_speed_once(tmp_path, rig):
    _, repl, channel, _, _, _ = _build(tmp_path)
    seed_speeds(tmp_path, 125.4, 50.9)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert steps(channel) == [
        "◆ model launching flash-next · a few minutes",
        "◆ model stopped qwen-primary · GPU freed",
        "◆ model ready flash-next · 50.9 t/s measured · window: served — · trained — · fits — (not verified)",
        f"◆ model switched · {PEER_URL} · tool calling native",
    ]
    said = "\n".join(channel.messages)
    assert said.count("50.9 t/s measured") == 1 and "125.4" not in said  # the target's speed, once; no arrow, no incumbent's
    assert "Speed:" not in said and "→" not in said
    assert all(m.count("flash-next") <= 1 for m in channel.messages)  # a peer named for its model: once to a line
    assert channel.messages[-1] == steps(channel)[-1]


@pytest.mark.asyncio
async def test_a_swap_that_stopped_nothing_has_no_stopped_line(tmp_path, rig):
    _, repl, channel, _, _, _ = _build(tmp_path)
    repl._active_heavy = None

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert [m.split(" · ")[0] for m in steps(channel)] == [
        "◆ model launching flash-next", "◆ model ready flash-next", "◆ model switched"]


@pytest.mark.asyncio
async def test_a_swap_that_could_not_stop_the_incumbent_says_no_launching(tmp_path, rig):
    _, repl, channel, _, _, primary = _build(tmp_path)
    rig.stop_raises.append(RuntimeError("won't die"))

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert steps(channel) == [] and rig.starts == []
    assert repl._active_heavy[0] is primary


@pytest.mark.asyncio
async def test_a_name_a_server_reports_prints_inert_on_its_own_line(tmp_path, rig, monkeypatch):
    hostile = "evil\n◆ model switched\x1b[2J\x1b]52;c;aGk=\x07 · pwned"

    async def wait(base_url, config_dir=None, **kw):
        return [hostile]

    monkeypatch.setattr(managed_server, "wait_ready", wait)
    _, repl, channel, _, _, _ = _build(tmp_path)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    ready = next(m for m in steps(channel) if m.startswith("◆ model ready"))
    assert ready.startswith("◆ model ready evil ◇ model switched") and ready.count("◆") == 1
    assert not any(c in ready for c in ("\x1b", "\x07", "\n"))
    assert [m.split(" ")[2] for m in steps(channel)] == ["launching", "stopped", "ready", "switched"]


# --- the `ready` line: the window, the speed once, one placement fact ------------------------------- #

SPILLED = short_by(SCREENSHOT, int(1.5 * GIB))
HEALTHY = dataclasses.replace(SCREENSHOT, held=SPILLED.held + int(1.5 * GIB))
RATE = "50.9 t/s measured"


def ready(report: WindowReport | None, rate: str | None = RATE, served: str = "gemma") -> str:
    lines = window_lines(report, AutosenseConfig()) if report is not None else []
    return repl_mod._ready_line(served, lines, rate)


def test_ready_puts_the_window_first_the_speed_after_it_and_the_placement_last():
    assert ready(HEALTHY) == f"◆ model ready gemma · 58,368 window · {RATE} · all on the GPU (21.6 of 26.4 GiB free)"


def test_ready_says_the_spill_and_not_all_on_the_gpu_for_a_launch_that_held_less_than_it_was_priced():
    said = ready(SPILLED)
    assert said == f"◆ model ready gemma · 58,368 window · {RATE} · 1.5 GiB of layers on the CPU, decode will be slow"
    assert "all on the GPU" not in said


@pytest.mark.parametrize(("line", "tail"), [
    ("window: served 14,950 · trained 262,144 · fits 14,950 (free 6.0 GiB of 32.6; weights 18.5 GiB)",
     "window: served 14,950 · trained 262,144 · fits 14,950 (free 6.0 GiB of 32.6; weights 18.5 GiB)"),
    ("window: served 8,192 · trained 40,960 · Ollama decides the window", "window: served 8,192 · trained 40,960 · Ollama decides the window"),
    ("window: served by llama-server's --fit · served 65,536", "window: served by llama-server's --fit · served 65,536"),
], ids=["unpriced fit", "ollama", "fit by server"])
def test_a_report_with_no_window_head_keeps_its_words_after_the_speed(line, tail):
    assert repl_mod._ready_line("gemma", [line], RATE) == f"◆ model ready gemma · {RATE} · {tail}"


def test_a_head_with_nothing_after_it_and_a_launch_with_no_report_or_no_speed():
    assert repl_mod._ready_line("gemma", ["window: 8,192"], RATE) == f"◆ model ready gemma · 8,192 window · {RATE}"
    assert repl_mod._ready_line("gemma", [], RATE) == f"◆ model ready gemma · {RATE}"
    assert repl_mod._ready_line("gemma", [], None) == "◆ model ready gemma"
    assert repl_mod._ready_line("gemma", ["window: 8,192 · all on the GPU (1.0 of 2.0 GiB free)"], None) == (
        "◆ model ready gemma · 8,192 window · all on the GPU (1.0 of 2.0 GiB free)")


def test_the_one_speed_in_a_ready_line_is_banded():
    text = _colorize_rate_notes(ready(SPILLED, "50.9 t/s measured"))
    assert [str(span.style) for span in text.spans] == ["green"]
    assert text.plain[text.spans[0].start:text.spans[0].end] == "50.9 t/s measured"


@pytest.mark.asyncio
async def test_the_ready_line_through_the_repl_says_one_placement_fact(tmp_path):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())
    live = LiveEndpoint(base_url=LAUNCH_PEER_URL, served_models=["gemma"], handle=1, window=SPILLED)

    await repl._announce_ready(live, "llamacpp", "gemma", RATE)

    assert channel.messages == [
        f"◆ model ready gemma · 58,368 window · {RATE} · 1.5 GiB of layers on the CPU, decode will be slow"]


# --- the approved screenshot's swap, through the REPL's own code and the real terminal channel ---------------- #

NAME = "gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M"
HOLDER = "qwen-27b-swift-1.5"


@pytest.mark.asyncio
async def test_the_screenshots_swap_prints_the_approved_rows_through_the_terminal_channel(tmp_path, monkeypatch):
    """Peer and served name are one 57-character string, 125.4 t/s gives way to 50.9, the launch held 1.5 GiB less on the GPU than
    it was priced: what a person's terminal prints, row for row, at 120 columns."""
    from io import StringIO

    from rich.console import Console

    from localharness.channels.terminal import TERMINAL_THEME, TerminalChannel
    from localharness.cli.model_switch import ModelSwitch
    from localharness.config.models import EndpointRef, ManagedServerConfig
    from localharness.core.bus import EventBus
    from localharness.provider import lifecycle
    from tests.unit.test_launch_paths_repl import PRIMARY_URL, _llama_primary
    from tests.unit.test_repl_model_cmd import _harness, _xrepl

    url = "http://127.0.0.1:8083/v1"
    peer = EndpointRef(name=NAME, base_url=url, provider_type="llamacpp", gpu=True, api_key="none", lifecycle=ManagedServerConfig(
        runtime="llamacpp", binary="/x/llama-server", model="/x/gemma.gguf", port=8083, gpu=True, extra_args=["-a", NAME]))
    primary = _llama_primary(served_name=HOLDER)
    harness = _harness(server=primary, extra_endpoints=[peer])
    repl, _, agent, _ = _xrepl(tmp_path, harness, {PRIMARY_URL: [HOLDER], url: "unreachable"})
    out = StringIO()
    channel = TerminalChannel(EventBus(), {})
    channel._console = Console(file=out, force_terminal=False, width=120, theme=TERMINAL_THEME, highlight=False)
    channel.local_operator = True
    repl._channel = channel
    repl._switch = ModelSwitch(agent, repl._bus, channel, harness, repl._session_gate, tmp_path)

    async def refit(model, *, base_url=None, provider_type=None):
        return "context budget refit to 58,368"

    repl._refresh_token_counter = refit
    repl._active_heavy = (primary, PRIMARY_URL)
    ledger = default_speed_stats_path(tmp_path)
    record_tps(ledger, "llamacpp", HOLDER, 125.4)
    record_tps(ledger, "llamacpp", NAME, 50.9)

    class Server:
        def launch_identity(self, spec):
            return None

        async def stop(self, spec, config_dir, **kw):
            return None

        async def activate(self, spec, config_dir, base_url, **kw):
            return LiveEndpoint(base_url=url, served_models=[NAME], handle=1, window=SPILLED)

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec, **kw: Server())

    await asyncio.wait_for(repl._handle_model_cmd(NAME), 5)

    assert out.getvalue().splitlines() == [
        f"◆ model launching {NAME} · a few minutes",
        f"◆ model stopped {HOLDER} · GPU freed",
        f"◆ model ready {NAME} · 58,368 window · 50.9 t/s measured · 1.5 GiB of",
        "  layers on the CPU, decode will be slow",
        f"◆ model switched · {url} · tool calling native · context budget refit to 58,368",
    ]


# --- the other ends of a switch: a model already served, a peer already up --------------------------------------- #


@pytest.mark.asyncio
async def test_a_hot_swap_closes_with_switched_naming_the_model_once_with_its_url_and_the_saved_default(tmp_path):
    from tests.unit.test_repl_model_cmd import _harness, _repl

    repl, channel, _ = _repl(tmp_path, _harness(), live=["model-a", "model-b"])
    record_tps(default_speed_stats_path(tmp_path), "vllm", "model-b", 31.0)

    await repl._handle_slash("/model model-b")

    assert channel.messages == [
        "◆ model switched model-b · http://localhost:8081/v1 · tool calling native · 31.0 t/s measured"
        " · saved as default — persists across restarts"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("peer_name", "served", "line"), [
    ("ollama-local", "gpt-oss:20b",
     "◆ model switched gpt-oss:20b · on ollama-local · http://localhost:11434/v1 · tool calling native"),
    ("gpt-oss", "gpt-oss", "◆ model switched gpt-oss · http://localhost:11434/v1 · tool calling native"),
], ids=["a peer named otherwise", "a peer named for its model: once"])
async def test_a_switch_to_a_peer_already_up_names_the_model_and_the_peer_only_when_they_differ(tmp_path, peer_name, served, line):
    from tests.unit.test_repl_model_cmd import _harness, _peer, _xrepl

    repl, channel, agent, _ = _xrepl(tmp_path, _harness(extra_endpoints=[_peer(name=peer_name)]),
                                     {"http://localhost:8081/v1": ["model-a"], "http://localhost:11434/v1": [served]})

    async def refit(model, *, base_url=None, provider_type=None):
        return ""

    async def persist(endpoint, model):
        return None

    repl._refresh_token_counter = refit
    repl._persist_active_endpoint = persist

    await repl._handle_slash(f"/model {served}")

    assert channel.messages == [line]


@pytest.mark.asyncio
async def test_every_step_hands_the_channel_the_one_name_it_prints_and_the_speed_to_band(tmp_path, rig):
    _, repl, channel, _, _, _ = _build(tmp_path)
    seed_speeds(tmp_path, 125.4, 50.9)
    sent = []
    send = channel.send_message

    async def spy(text, metadata=None):
        sent.append((text.split(" · ")[0], (metadata or {}).get("names"), (metadata or {}).get("colorize")))
        await send(text, metadata=metadata)

    channel.send_message = spy

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert [s for s in sent if s[0].startswith("◆ model")] == [
        ("◆ model launching flash-next", ["flash-next"], None), ("◆ model stopped qwen-primary", ["qwen-primary"], None),
        ("◆ model ready flash-next", ["flash-next"], "rates"), ("◆ model switched", None, None)]

# --- what a swap must not do: print a secret, break a channel written against the old signature, stop silently ----- #


@pytest.mark.asyncio
async def test_a_url_with_credentials_prints_without_them_on_every_switched_line(tmp_path):
    from tests.unit.test_repl_model_cmd import _harness, _repl

    url = "https://admin:s3cret@gw.example.com:8443/v1"
    harness = _harness()
    harness.provider.base_url = url
    repl, channel, agent = _repl(tmp_path, harness, live=["model-a", "model-b"])
    agent._llm.config.base_url = url

    await repl._handle_slash("/model model-b")

    (said,) = channel.messages
    assert said.startswith("◆ model switched model-b · https://gw.example.com:8443/v1 · tool calling native")
    assert "s3cret" not in said and "admin" not in said


@pytest.mark.asyncio
async def test_a_channel_that_overrides_send_notice_with_the_old_one_argument_signature_gets_the_lines(tmp_path, rig):
    _, repl, channel, _, _, _ = _build(tmp_path)
    got = []

    async def send_notice(line):  # the signature the base class documented before it took metadata
        got.append(line)

    channel.send_notice = send_notice

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert [m.split(" · ")[0] for m in got] == [
        "◆ model launching flash-next", "◆ model stopped qwen-primary", "◆ model ready flash-next", "◆ model switched"]
    assert repl._active_heavy[1] == PEER_URL  # and the swap landed


@pytest.mark.asyncio
async def test_a_server_whose_model_id_is_not_a_string_still_lands_the_swap(tmp_path, rig, monkeypatch):
    async def wait(base_url, config_dir=None, **kw):
        return [123]

    monkeypatch.setattr(managed_server, "wait_ready", wait)
    _, repl, channel, agent, _, _ = _build(tmp_path)

    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)

    assert "◆ model ready 123" in " ".join(steps(channel)) and agent._llm.config.base_url == PEER_URL


@pytest.mark.asyncio
async def test_a_swap_back_that_stops_a_peer_says_so_between_launching_and_ready(tmp_path, monkeypatch):
    from tests.unit.test_launch_paths_repl import PRIMARY_URL
    from tests.unit.test_model_swap_back import _llama, _on_a_peer, _ready_as

    monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    monkeypatch.setattr(managed_server, "stop_server", lambda cfg, launch="binary", **kw: True)
    monkeypatch.setattr(managed_server, "start_server", lambda cfg, cmd, **kw: 4321)
    _ready_as(monkeypatch, "qwen-27b")
    primary = _llama(served_name="qwen-27b")
    _, repl, channel, _, _ = _on_a_peer(tmp_path, primary)

    await asyncio.wait_for(repl._handle_slash("/model qwen-27b"), 5)

    assert [m.split(" · ")[0] for m in steps(channel)] == [
        "◆ model launching qwen-27b", "◆ model stopped flash", "◆ model ready qwen-27b", "◆ model switched"]
    assert steps(channel)[1] == "◆ model stopped flash · GPU freed"
    assert steps(channel)[3].startswith(f"◆ model switched · {PRIMARY_URL} · tool calling native")
