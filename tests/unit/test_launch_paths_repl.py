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


@pytest.fixture(autouse=True)
def _no_machine_probes(monkeypatch):
    """A launch through the real `activate` now sizes its window (Phase 2): the fit asks
    nvidia-smi for free memory and the post-launch probe reads the served window back over HTTP.
    No test here makes either call; the window itself is pinned in test_window_lines_repl.py."""
    from localharness.agent import context as agent_context
    from localharness.provider import autosense

    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    monkeypatch.setattr(agent_context, "probe_window",
                        lambda *a, **k: agent_context.WindowProbe(None, None))


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


def _session(tmp_path, peer, server=None, **repl_kw):
    harness = _harness(server=server if server is not None else _heavy_primary(),
                       extra_endpoints=[peer])
    repl, channel, agent, _ = _xrepl(
        tmp_path, harness, {PRIMARY_URL: ["model-a"], PEER_URL: "unreachable"}, **repl_kw)

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


@pytest.mark.asyncio
async def test_the_cold_launch_passes_the_harness_autosense_to_activate(
        tmp_path, boundaries, monkeypatch):
    harness, repl, _, _ = _session(tmp_path, _launchable_peer())
    seen: list[dict] = []
    real = lifecycle.SpawnedProcessStrategy.activate

    async def spy(self, spec, config_dir, base_url, **kw):
        seen.append(kw)
        return await real(self, spec, config_dir, base_url, **kw)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", spy)

    await repl._handle_slash("/model qwen-gguf")

    assert seen and seen[-1]["autosense"] is harness.autosense


SENTINEL = "KEY-SENTINEL-peer-1"


def _capture_start_kw(monkeypatch) -> list[dict]:
    """What `start_server` was handed besides the command: the key reaches the child's environment
    through `api_key=`."""
    seen: list[dict] = []
    monkeypatch.setattr(managed_server, "start_server",
                        lambda cfg, cmd, **kw: seen.append({"cmd": list(cmd), **kw}) or 4321)
    return seen


@pytest.mark.asyncio
async def test_a_file_keyed_cold_peer_launches_keyed_and_rebinds_with_the_files_key(
        tmp_path, boundaries, monkeypatch):
    key = tmp_path / "peer.key"
    key.write_text(SENTINEL + "\n", encoding="utf-8")
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key))
    _, repl, channel, agent = _session(tmp_path, peer)
    started = _capture_start_kw(monkeypatch)

    await repl._handle_slash("/model qwen-gguf")

    (call,) = started
    assert call["api_key"] == SENTINEL
    assert call["cmd"][call["cmd"].index("--api-key-file") + 1] == str(key)
    assert "0.0.0.0" in call["cmd"] and SENTINEL not in " ".join(call["cmd"])
    (rebind,) = [r for r in agent._llm.rebinds if r[0] == PEER_URL]
    assert rebind[1].get_secret_value() == SENTINEL
    assert not any("every interface, no key" in m for m in channel.messages)
    assert not any("this session sends no key" in m for m in channel.messages)
    assert not any(SENTINEL in m for m in channel.messages)
    # the key is held for this peer in memory, so a later probe of it carries it too
    from localharness.cli.repl import _launch_key
    assert _launch_key(repl._harness, PEER_URL) == {"api_key": SENTINEL}


@pytest.mark.asyncio
async def test_a_refused_peer_key_file_is_one_line_and_the_peer_launches_as_phase_1_would(
        tmp_path, boundaries, monkeypatch):
    key = tmp_path / "peer.key"
    key.write_bytes(SENTINEL.encode() + b"\r\n")
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key))
    _, repl, channel, agent = _session(tmp_path, peer)
    started = _capture_start_kw(monkeypatch)

    await repl._handle_slash("/model qwen-gguf")

    line = (f"extra_endpoints[llamacpp-local].lifecycle.api_key_file {key}: "
            "byte outside printable ASCII — this session sends no key")
    assert channel.messages.count(line) == 1
    (call,) = started
    assert "api_key" not in call
    assert "--api-key-file" not in call["cmd"] and call["cmd"][-2:] == ["--host", "127.0.0.1"]
    (rebind,) = [r for r in agent._llm.rebinds if r[0] == PEER_URL]
    assert rebind[1].get_secret_value() == "none"
    assert not any(SENTINEL in m for m in channel.messages)


@pytest.mark.asyncio
async def test_cr01_a_refused_peer_key_file_with_a_configured_key_stays_off_argv_and_launches_loopback(
        tmp_path, boundaries, monkeypatch):
    """The peer holds its own configured `api_key`; the byte rule refuses the CRLF file. Decision, argv
    and the printed line agree: the file is not on argv, the launch is loopback although `bind_all` is
    true, the configured key still reaches the child and the rebind, and the line says the file is not
    used (never `sends no key`)."""
    key = tmp_path / "peer.key"
    key.write_bytes(SENTINEL.encode() + b"\r\n")
    peer = _launchable_peer(api_key="CONFIGURED", bind_all=True, require_api_key=True,
                            api_key_file=str(key))
    _, repl, channel, agent = _session(tmp_path, peer)
    started = _capture_start_kw(monkeypatch)

    await repl._handle_slash("/model qwen-gguf")

    line = (f"extra_endpoints[llamacpp-local].lifecycle.api_key_file {key}: "
            "byte outside printable ASCII — the file is not used; "
            "this peer keeps sending the key it is configured with")
    assert channel.messages.count(line) == 1
    assert not any("sends no key" in m for m in channel.messages)
    (call,) = started
    assert call["api_key"] == "CONFIGURED"
    assert "--api-key-file" not in call["cmd"] and "0.0.0.0" not in call["cmd"]
    assert call["cmd"][-2:] == ["--host", "127.0.0.1"]
    (rebind,) = [r for r in agent._llm.rebinds if r[0] == PEER_URL]
    assert rebind[1].get_secret_value() == "CONFIGURED"
    assert not any(SENTINEL in m for m in channel.messages)


def test_cr01_an_earlier_launchs_file_key_does_not_survive_a_later_refusal(tmp_path):
    """A peer's first cold launch reads a good file (its key is held on the endpoint); the file is then
    replaced by one the byte rule refuses. The second launch must hold the CONFIGURED key again, keep
    the refused file off argv and on loopback, and say so."""
    from localharness.cli.repl import _hold_peer_session_key, _peer_session_key
    key = tmp_path / "peer.key"
    key.write_text("EARLIER-FILE-KEY\n", encoding="utf-8")
    peer = _launchable_peer(api_key="CONFIGURED", bind_all=True, require_api_key=True,
                            api_key_file=str(key))

    first, first_line = _peer_session_key(peer)
    _hold_peer_session_key(peer, first)
    assert first_line is None and peer.api_key.get_secret_value() == "EARLIER-FILE-KEY"
    assert managed_server.launch_decision(peer.lifecycle, True, peer=peer.name).key_file_ok is True

    key.write_bytes(b"EARLIER-FILE-KEY\r\n")
    second, second_line = _peer_session_key(peer)
    _hold_peer_session_key(peer, second)
    assert second is None and "the file is not used" in second_line
    assert peer.api_key.get_secret_value() == "CONFIGURED"
    decision = managed_server.launch_decision(peer.lifecycle, True, peer=peer.name)
    assert (decision.loopback, decision.key_file_ok) == (True, False)
    assert "--api-key-file" not in managed_server.serve_command(
        peer.lifecycle, keyed=True, loopback=decision.loopback, key_file_ok=decision.key_file_ok)

    key.write_text("LATER-GOOD\n", encoding="utf-8")  # a later good read lifts the refusal
    third, _ = _peer_session_key(peer)
    _hold_peer_session_key(peer, third)
    assert peer.api_key.get_secret_value() == "LATER-GOOD"
    assert managed_server.launch_decision(peer.lifecycle, True, peer=peer.name).key_file_ok is True


def test_cr01_a_peer_with_no_configured_key_falls_back_to_none_after_a_refusal(tmp_path):
    from localharness.cli.repl import _hold_peer_session_key, _peer_session_key
    key = tmp_path / "peer.key"
    key.write_text("EARLIER-FILE-KEY\n", encoding="utf-8")
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key))
    first, _ = _peer_session_key(peer)
    _hold_peer_session_key(peer, first)
    key.write_bytes(b"a,b\n")
    second, line = _peer_session_key(peer)
    _hold_peer_session_key(peer, second)
    assert line.endswith("— this session sends no key")
    assert peer.api_key.get_secret_value() == "none"


@pytest.mark.asyncio
async def test_a_refused_launch_decision_still_reads_nothing_it_has_to_print(tmp_path, boundaries):
    key = tmp_path / "peer.key"
    key.write_bytes(b"a,b\n")
    peer = EndpointRef(
        name="llamacpp-local", base_url=PEER_URL, provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080, gpu=True,
                                      extra_args=["-a", "qwen-gguf"], require_api_key=True,
                                      api_key_file=str(key)))
    _, repl, channel, _ = _session(tmp_path, peer)

    await repl._handle_slash("/model qwen-gguf")

    assert channel.messages[-1] == NO_LLAMA_PEER
    assert not any("this session sends no key" in m for m in channel.messages)


@pytest.mark.asyncio
async def test_restoring_a_file_keyed_primary_reuses_the_sessions_key_with_no_second_read(
        tmp_path, boundaries, monkeypatch):
    from pydantic import SecretStr
    from tests.unit.launch_support import record_opens
    key = tmp_path / "primary.key"
    key.write_text("NOT-READ-AGAIN\n", encoding="utf-8")
    primary = _llama_primary(bind_all=True, require_api_key=True, api_key_file=str(key))
    harness = _harness(server=primary, extra_endpoints=[])
    harness.provider.api_key = SecretStr(SENTINEL)  # what `start` read into the session
    repl, _, _, _ = _xrepl(tmp_path, harness, {PRIMARY_URL: ["model-a"]})
    started = _capture_start_kw(monkeypatch)

    with record_opens() as opened:
        await repl._restore_incumbent((primary, harness.provider.base_url))

    assert started and started[0]["api_key"] == SENTINEL
    assert str(key) not in opened


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


# --- the cold peer's menu name, and --host last on paths 2-4 ------------------------------------- #


def _with_lifecycle(**record) -> EndpointRef:
    """A peer whose record is built from `record`; it names nothing but what a test sets."""
    return EndpointRef(
        name="flash", base_url=PEER_URL, provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080, gpu=True,
            **record))


def _name_of(ep) -> str:
    from localharness.cli.repl import OrchestratorREPL
    return OrchestratorREPL._cold_served_name(ep)


def test_served_name_is_read_before_any_alias_flag():
    assert _name_of(_with_lifecycle(served_name="flash-next", extra_args=["-a", "other"])) == "flash-next"


@pytest.mark.parametrize("args,expected", [
    (["-a", "x"], "x"),
    (["--alias", "y"], "y"),
    (["--served-model-name", "z"], "z"),
])
def test_without_a_served_name_the_alias_flags_are_read(args, expected):
    assert _name_of(_with_lifecycle(extra_args=args)) == expected


def test_with_neither_the_endpoint_name_is_the_served_name():
    assert _name_of(_with_lifecycle()) == "flash"
    assert _name_of(_with_lifecycle(served_name=None, extra_args=["-ngl", "99"])) == "flash"


def test_a_peer_with_no_lifecycle_gives_the_endpoint_name():
    attach = EndpointRef(name="ollama-local", base_url="http://localhost:11434/v1",
                         provider_type="ollama")
    assert _name_of(attach) == "ollama-local"


@pytest.mark.asyncio
async def test_a_non_ascii_served_name_is_returned_and_matched_verbatim(tmp_path, boundaries):
    name = "qwen-äö"
    peer = _with_lifecycle(served_name=name)
    assert _name_of(peer) == name
    _, repl, channel, _ = _session(tmp_path, peer)
    assert repl._cold_lifecycle_targets(["model-a"], {}) == {name: peer}

    await repl._handle_slash(f"/model {name}")

    assert boundaries.cmds, channel.messages
    assert any(m.startswith("Launching flash") for m in channel.messages)


@pytest.mark.asyncio
async def test_launch_path_2_cold_branch_host_is_last(tmp_path, boundaries):
    peer = _launchable_peer()
    peer.lifecycle.extra_args = ["--host", "0.0.0.0", "-a", "qwen-gguf"]
    assert peer.lifecycle.bind_all is False
    _, repl, _, _ = _session(tmp_path, peer)

    await repl._handle_slash("/model qwen-gguf")

    assert boundaries.cmds
    assert boundaries.cmds[0][-2:] == ["--host", "127.0.0.1"]


@pytest.mark.asyncio
async def test_launch_path_3_swap_back_host_is_last(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    peer = _launchable_peer()
    primary = _llama_primary(extra_args=["--host", "0.0.0.0"])
    _, repl, _, agent = _session(tmp_path, peer, server=primary)
    _on_the_cold_peer(repl, agent, peer)

    await repl._handle_slash("/model cached-b")

    assert boundaries.cmds
    assert boundaries.cmds[-1][-2:] == ["--host", "127.0.0.1"]
    assert repl._active_heavy == (primary, PRIMARY_URL)


@pytest.mark.asyncio
async def test_launch_path_4_restore_host_is_last(tmp_path, boundaries):
    primary = _llama_primary(extra_args=["--host", "0.0.0.0"])
    _, repl, _, _ = _session(tmp_path, _launchable_peer(), server=primary)

    label, _ = await repl._restore_incumbent((primary, PRIMARY_URL))

    assert label == primary.model
    assert boundaries.cmds and boundaries.cmds[-1][-2:] == ["--host", "127.0.0.1"]


# --- a binary that is set but missing: the failure every activate caller restores from ----------- #

_REAL_START_SERVER = managed_server.start_server  # the `boundaries` fixture replaces it per test


@pytest.mark.asyncio
async def test_a_cold_launch_with_a_missing_binary_restores_the_incumbent_in_one_failure_line(
        tmp_path, boundaries, monkeypatch):
    class _Child:
        pid = 4242

    def popen(cmd, **kw):
        if cmd[0] == "/x/llama-server":
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        return _Child()

    # the real start_server, so the Popen failure travels the real activate path
    monkeypatch.setattr(managed_server, "start_server", _REAL_START_SERVER)
    monkeypatch.setattr(managed_server.subprocess, "Popen", popen)
    harness, repl, channel, agent = _session(tmp_path, _launchable_peer())
    primary = harness.server

    await repl._handle_slash("/model qwen-gguf")

    last = channel.messages[-1]
    assert last.startswith("Launch of llamacpp-local failed: could not launch /x/llama-server: "
                           "No such file or directory")
    assert f"Restored {primary.model}." in last
    assert "\n" not in last and "NO model serving" not in last
    assert repl._active_heavy is not None and repl._active_heavy[0] is primary
    assert agent._llm.rebinds == []


@pytest.mark.asyncio
async def test_a_failed_restore_launch_returns_its_cause_not_nothing(tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)
    repl._active_heavy = None

    async def never_ready(base_url, config_dir=None, **kw):
        raise TimeoutError("llama-server never became ready")

    monkeypatch.setattr(managed_server, "wait_ready", never_ready)

    assert await repl._restore_incumbent((peer.lifecycle, PEER_URL)) == (
        None, "llama-server never became ready")
    assert repl._active_heavy is None


@pytest.mark.asyncio
async def test_a_failed_restore_that_is_not_a_launch_error_keeps_its_type_name(
        tmp_path, boundaries, monkeypatch):
    peer = _launchable_peer()
    _, repl, _, _ = _session(tmp_path, peer)
    repl._active_heavy = None

    async def broken(base_url, config_dir=None, **kw):
        raise KeyError("served")

    monkeypatch.setattr(managed_server, "wait_ready", broken)

    assert await repl._restore_incumbent((peer.lifecycle, PEER_URL)) == (None, "KeyError: 'served'")


@pytest.mark.asyncio
async def test_a_cold_launch_whose_restore_fails_names_the_cause_in_the_one_failure_line(
        tmp_path, boundaries, monkeypatch):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    async def never_ready(base_url, config_dir=None, **kw):
        raise TimeoutError("llama-server never became ready")

    monkeypatch.setattr(managed_server, "wait_ready", never_ready)

    await repl._handle_slash("/model qwen-gguf")

    last = channel.messages[-1]
    assert "Could not restore it: llama-server never became ready" in last
    assert "NO model serving" not in last


# --- the pending launch lines `start` hands over (D-15, D-26) ---------------------------------- #

LINE = ("server.bind_all: true not applied — launched on 127.0.0.1 for this run; "
        "a terminal start asks")


def _when_sent(channel, boundaries) -> list[tuple[str, int]]:
    """Each message with the number of processes started when it went out."""
    seen: list[tuple[str, int]] = []
    send = channel.send_message

    async def recording(text, metadata=None):
        seen.append((text, len(boundaries.cmds)))
        await send(text, metadata=metadata)

    channel.send_message = recording
    return seen


@pytest.mark.asyncio
async def test_the_first_cold_launch_prints_the_pending_line_once_before_its_announcement(
        tmp_path, boundaries):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), launch_lines=(LINE,))
    sent = _when_sent(channel, boundaries)

    await repl._handle_slash("/model qwen-gguf")

    assert [m for m in channel.messages if m == LINE] == [LINE]
    assert channel.messages.index(LINE) < next(
        i for i, m in enumerate(channel.messages) if m.startswith("Launching llamacpp-local"))
    assert [started for text, started in sent if text == LINE] == [0]  # before activate
    assert repl._pending_launch_lines == []


@pytest.mark.asyncio
async def test_a_repl_built_without_the_keyword_prints_nothing_extra(tmp_path, boundaries):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer())

    await repl._handle_slash("/model qwen-gguf")

    assert repl._pending_launch_lines == []
    assert not any("not applied" in m for m in channel.messages)
    assert any(m.startswith("Launching llamacpp-local") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_swap_back_that_is_the_sessions_first_launch_prints_the_line_once_before_its_announcement(
        tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=_llama_primary(),
                                   launch_lines=(LINE,))
    sent = _when_sent(channel, boundaries)

    await repl._handle_slash("/model cached-b")

    assert channel.messages.count(LINE) == 1
    assert channel.messages.index(LINE) < next(
        i for i, m in enumerate(channel.messages) if m.startswith("Restarting managed llama.cpp"))
    assert [started for text, started in sent if text == LINE] == [0]
    assert repl._pending_launch_lines == []


@pytest.mark.asyncio
async def test_a_restore_that_is_the_sessions_first_launch_prints_the_line(tmp_path, boundaries):
    primary = _llama_primary()
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), server=primary, launch_lines=(LINE,))
    sent = _when_sent(channel, boundaries)

    label, _ = await repl._restore_incumbent((primary, PRIMARY_URL))

    assert label == primary.model
    assert channel.messages == [LINE] and [started for _, started in sent] == [0]
    assert repl._pending_launch_lines == []


@pytest.mark.asyncio
async def test_the_sessions_second_launch_prints_none_of_the_lines(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    peer = _launchable_peer()
    primary = _llama_primary()
    _, repl, channel, agent = _session(tmp_path, peer, server=primary, launch_lines=(LINE,))

    await repl._handle_slash("/model qwen-gguf")  # the cold launch: the first
    await repl._handle_slash("/model cached-b")   # the swap-back: the second
    await repl._restore_incumbent((primary, PRIMARY_URL))  # and a restore: the third

    assert channel.messages.count(LINE) == 1
    assert any(m.startswith("Restarting managed llama.cpp") for m in channel.messages)


@pytest.mark.asyncio
async def test_a_refused_cold_launch_prints_none_of_the_lines_and_keeps_them_for_the_next_launch(
        tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    _, repl, channel, _ = _session(tmp_path, _declared_only_peer(), launch_lines=(LINE,))

    await repl._handle_slash("/model qwen-gguf")

    assert channel.messages[-1] == NO_LLAMA_PEER and LINE not in channel.messages
    assert repl._pending_launch_lines == [LINE]

    await repl._handle_slash("/model cached-b")

    assert channel.messages.count(LINE) == 1 and repl._pending_launch_lines == []


@pytest.mark.asyncio
async def test_a_refused_swap_back_and_a_refused_restore_keep_the_lines(tmp_path, boundaries, monkeypatch):
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: ["cached-b"])
    peer = _launchable_peer()
    declared = _llama_primary(binary=None)
    _, repl, channel, agent = _session(tmp_path, peer, server=declared, launch_lines=(LINE,))
    _on_the_cold_peer(repl, agent, peer)

    await repl._handle_slash("/model cached-b")
    restored = await repl._restore_incumbent((declared, PRIMARY_URL))

    assert channel.messages == [NO_LLAMA_PRIMARY] and restored[0] is None
    assert repl._pending_launch_lines == [LINE]


@pytest.mark.asyncio
async def test_a_launch_with_no_pending_lines_sends_no_extra_message(tmp_path, boundaries):
    _, repl, channel, _ = _session(tmp_path, _launchable_peer(), launch_lines=())

    await repl._handle_slash("/model qwen-gguf")

    first = next(i for i, m in enumerate(channel.messages) if m.startswith("Launching llamacpp-local"))
    assert first == 0


@pytest.mark.asyncio
async def test_wr06_a_peer_whose_base_url_is_not_this_machine_never_has_its_key_file_read(
        tmp_path, boundaries, monkeypatch):
    from tests.unit.launch_support import record_opens
    key = tmp_path / "peer.key"
    key.write_text(SENTINEL + "\n", encoding="utf-8")
    peer = _launchable_peer(bind_all=True, require_api_key=True, api_key_file=str(key))
    peer.base_url = "http://192.168.1.50:8080/v1"
    _, repl, channel, agent = _session(tmp_path, peer)
    started = _capture_start_kw(monkeypatch)

    with record_opens() as opened:
        await repl._handle_slash("/model qwen-gguf")

    assert str(key) not in opened
    line = (f"extra_endpoints[llamacpp-local].lifecycle.api_key_file {key}: "
            "base_url 192.168.1.50:8080 is not this machine, so the file is not read — "
            "this session sends no key")
    assert channel.messages.count(line) == 1
    (call,) = started
    assert "api_key" not in call
    assert "--api-key-file" not in call["cmd"] and call["cmd"][-2:] == ["--host", "127.0.0.1"]
    assert not any(SENTINEL in m for m in channel.messages)


@pytest.mark.asyncio
async def test_in01_an_escape_sequence_in_a_peers_name_and_key_path_prints_escaped(
        tmp_path, boundaries, monkeypatch):
    key = tmp_path / "\x1b[2Jkey"
    peer = _launchable_peer(name="p\x1b[1;1H", bind_all=True, require_api_key=True, api_key_file=str(key))
    _, repl, channel, _ = _session(tmp_path, peer)
    _capture_start_kw(monkeypatch)

    await repl._handle_slash("/model qwen-gguf")

    (line,) = [m for m in channel.messages if "api_key_file" in m and "sends no key" in m]
    assert "\x1b" not in line
    assert line.startswith("extra_endpoints[p\\x1b[1;1H].lifecycle.api_key_file ") and "\\x1b[2Jkey" in line
