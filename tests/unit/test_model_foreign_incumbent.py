"""`/model <peer>` never launches onto a GPU a foreign process holds (Phase 5, plan 11, D-32 / D-33 /
D-58 / D-69), and the accelerator lock is seeded only from what the harness launched.

A foreign listener on the primary's port is answered in one sentence naming the unit, the pid and
the `systemctl` command; nothing is launched, stopped or signalled. The fakes: a `FakeProc` tree
for `/proc`, `FakeSystemd` for `systemctl`, and the REPL's start/stop/free seams made to raise.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from localharness.cli import workspace
from localharness.cli.repl import OrchestratorREPL
from localharness.config.models import (
    EndpointRef,
    HarnessConfig,
    ManagedServerConfig,
    OrgConfig,
    ProviderConfig,
)
from localharness.provider import lifecycle
from localharness.provider import server as managed_server
from tests.unit.launch_support import harness_launched, stub_liveness  # noqa: F401
from tests.unit.procfs_support import FakeProc, fake_proc  # noqa: F401
from tests.unit.test_repl_model_cmd import FakeChannel, FakeLLM
from tests.unit.units_support import FakeSystemd, fake_systemd  # noqa: F401

PRIMARY_URL = "http://localhost:8080/v1"
PEER_URL = "http://127.0.0.1:8090/v1"
UNIT = "llama-server.service"
SENTENCE_STOP = ("qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
                 "systemctl --user stop llama-server.service, then /model flash-next")


def _cgroup(unit: str = UNIT) -> str:
    return f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}"


def _show(unit: str = UNIT, enabled: str = "disabled", load: str = "loaded") -> str:
    return (f"Id={unit}\nLoadState={load}\nFragmentPath=/home/u/.config/systemd/user/{unit}\n"
            f"UnitFileState={enabled}\nDropInPaths=\n")


def _llama_primary(**extra) -> ManagedServerConfig:
    return ManagedServerConfig(
        runtime="llamacpp", binary="/x/llama-server", model="/x/primary.gguf", port=8080,
        gpu=True, served_name="qwen-27b", **extra)


def _peer() -> EndpointRef:
    return EndpointRef(
        name="flash", base_url=PEER_URL, provider_type="llamacpp", gpu=True, api_key="none",
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", binary="/x/llama-server", model="/x/flash.gguf", port=8090, gpu=True,
            extra_args=["-a", "flash-next"]))


def _harness(server, *, base_url=PRIMARY_URL, peers=None) -> HarnessConfig:
    return HarnessConfig(
        provider=ProviderConfig(provider_type="llamacpp", base_url=base_url,
                                default_model="qwen-27b", available_models=["qwen-27b"]),
        org=OrgConfig(default_model="qwen-27b"),
        server=server,
        extra_endpoints=peers if peers is not None else [_peer()],
    )


def _build(tmp_path, harness, *, base_url=PRIMARY_URL):
    channel = FakeChannel()
    agent = SimpleNamespace(_llm=FakeLLM(model="qwen-27b", base_url=base_url, provider_type="llamacpp"))
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=agent, channel=channel, bus=SimpleNamespace(),
        config_dir=tmp_path, harness_config=harness)

    async def _live_models(url):
        return (["qwen-27b"], True) if url == base_url else ([], False)

    repl._live_models = _live_models
    return repl, channel, agent


class Spies:
    """Every way the cold branch could act on a foreign process, made to fail loudly."""

    def __init__(self) -> None:
        self.hits: list[str] = []


@pytest.fixture
def spies(monkeypatch) -> Spies:
    s = Spies()

    def boom(name):
        def _raise(*a, **k):
            s.hits.append(name)
            raise AssertionError(f"{name} must not be called for a foreign incumbent")
        return _raise

    async def aboom(*a, **k):
        s.hits.append("async")
        raise AssertionError("must not be called for a foreign incumbent")

    monkeypatch.setattr(managed_server, "start_server", boom("start_server"))
    monkeypatch.setattr(managed_server, "stop_server", boom("stop_server"))
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
    monkeypatch.setattr(lifecycle, "free_accelerator", aboom)
    monkeypatch.setattr(os, "kill", boom("os.kill"))
    monkeypatch.setattr(os, "killpg", boom("os.killpg"))
    from localharness.cli import repl as repl_module
    monkeypatch.setattr(repl_module, "_peer_session_key", boom("_peer_session_key"), raising=False)
    return s


def _foreign(fake_proc: FakeProc, *, port: int = 8080, pid: int = 1234, inode: int = 5551,
             cgroup: str | None = None, argv=("/usr/bin/llama-server", "-m", "/x.gguf")) -> None:
    fake_proc.listen(port, inode, os.getuid())
    fake_proc.process(pid, argv=argv, cgroup=cgroup, fds={7: inode})


# --- the sentence ------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_foreign_unit_on_the_primarys_port_is_one_sentence_and_nothing_is_touched(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show())
    _foreign(fake_proc, cgroup=_cgroup())
    harness = _harness(_llama_primary())
    repl, channel, agent = _build(tmp_path, harness)
    held = repl._active_heavy

    await repl._handle_slash("/model flash-next")

    assert channel.messages == [SENTENCE_STOP]
    assert spies.hits == []
    assert repl._active_heavy is held
    assert agent._llm.rebinds == []
    assert agent._llm.config.base_url == PRIMARY_URL and agent._llm.config.model == "qwen-27b"


@pytest.mark.asyncio
async def test_an_enabled_unit_is_disabled_now(tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show(enabled="enabled"))
    _foreign(fake_proc, cgroup=_cgroup())
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary()))

    await repl._handle_slash("/model flash-next")

    assert channel.messages == [
        "qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
        "systemctl --user disable --now llama-server.service, then /model flash-next"]
    assert spies.hits == []


@pytest.mark.asyncio
async def test_a_unit_systemd_could_not_be_asked_about_says_so(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    _foreign(fake_proc, cgroup=_cgroup())  # fake_systemd answers nothing: "not checked"
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary()))

    await repl._handle_slash("/model flash-next")

    assert len(channel.messages) == 1
    assert channel.messages[0].startswith(
        "qwen-27b is served by llama-server.service (unit file not checked) (pid 1234), "
        "not by the harness;")
    assert spies.hits == []


@pytest.mark.asyncio
async def test_a_holder_in_no_unit_is_named_by_pid_and_command(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    _foreign(fake_proc, cgroup=f"/user.slice/user-{os.getuid()}.slice/session-3.scope")
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary()))

    await repl._handle_slash("/model flash-next")

    assert channel.messages == [
        "qwen-27b is served by pid 1234 (llama-server), not by the harness; "
        "stop it, then /model flash-next"]
    assert spies.hits == []


@pytest.mark.asyncio
async def test_a_listener_of_another_uid_is_named_by_uid(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_proc.listen(8080, 5551, 0)
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary()))

    await repl._handle_slash("/model flash-next")

    assert channel.messages == [
        "qwen-27b is served by uid 0's process on :8080, not by the harness; "
        "stop it, then /model flash-next"]
    assert spies.hits == []


@pytest.mark.asyncio
async def test_with_no_server_record_the_providers_port_is_the_primarys_port(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    """D-58: no `server:` → provider.base_url's port; `served` is the provider's default model."""
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show())
    _foreign(fake_proc, cgroup=_cgroup())
    repl, channel, _ = _build(tmp_path, _harness(None))

    await repl._handle_slash("/model flash-next")

    assert channel.messages == [SENTENCE_STOP]
    assert spies.hits == []


@pytest.mark.asyncio
async def test_a_remote_primary_is_not_judged_by_a_local_listener_on_its_port(
        tmp_path, fake_proc, fake_systemd, monkeypatch):
    """No `server:` and a base_url that is not this machine: a local listener on that port number
    says nothing about the primary, so the cold branch proceeds."""
    stub_liveness(monkeypatch, False)
    _foreign(fake_proc, cgroup=_cgroup())
    repl, _, _ = _build(tmp_path, _harness(None, base_url="http://gpu-box.lan:8080/v1"),
                        base_url="http://gpu-box.lan:8080/v1")

    assert repl._foreign_incumbent(None, "flash-next") is None


# --- the managed-restart branch is guarded the same way (WR-52) ------------------------------------ #


@pytest.mark.asyncio
async def test_a_typed_gguf_restart_is_one_sentence_while_a_foreign_unit_holds_the_port(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show())
    _foreign(fake_proc, cgroup=_cgroup())
    typed = tmp_path / "typed.gguf"
    typed.write_bytes(b"GGUF")
    repl, channel, agent = _build(tmp_path, _harness(_llama_primary()))
    primary = repl._harness.server

    await repl._handle_slash(f"/model {typed}")

    assert channel.messages == [
        "qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
        f"systemctl --user stop llama-server.service, then /model {typed}"]
    assert spies.hits == [] and agent._llm.rebinds == []
    assert primary.model == "/x/primary.gguf" and primary.served_name == "qwen-27b"


@pytest.mark.asyncio
async def test_swapping_back_to_the_primarys_name_is_one_sentence_while_a_foreign_unit_holds_the_port(
        tmp_path, fake_proc, fake_systemd, spies, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show())
    _foreign(fake_proc, cgroup=_cgroup())
    repl, channel, agent = _build(tmp_path, _harness(_llama_primary()), base_url=PEER_URL)
    agent._llm.config.model = "flash-next"

    async def _live_models(url):
        return (["flash-next"], True) if url == PEER_URL else ([], False)

    repl._live_models = _live_models

    await repl._handle_slash("/model qwen-27b")

    assert channel.messages == [
        "qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
        "systemctl --user stop llama-server.service, then /model qwen-27b"]
    assert spies.hits == [] and agent._llm.rebinds == []
    assert agent._llm.config.base_url == PEER_URL


# --- nothing foreign: the cold branch proceeds exactly as before --------------------------------- #


def test_a_claimed_port_is_no_incumbent(tmp_path, fake_proc, fake_systemd, monkeypatch):
    stub_liveness(monkeypatch, True)
    _foreign(fake_proc, cgroup=_cgroup())
    repl, _, _ = _build(tmp_path, _harness(_llama_primary()))

    assert repl._foreign_incumbent(repl._harness.server, "flash-next") is None


def test_a_free_port_is_no_incumbent(tmp_path, fake_proc, monkeypatch):
    stub_liveness(monkeypatch, False)
    repl, _, _ = _build(tmp_path, _harness(_llama_primary()))

    assert repl._foreign_incumbent(repl._harness.server, "flash-next") is None


def test_a_liveness_that_raises_leaves_the_listener_foreign(
        tmp_path, fake_proc, fake_systemd, monkeypatch):
    stub_liveness(monkeypatch, False)
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show())
    _foreign(fake_proc, cgroup=_cgroup())
    repl, _, _ = _build(tmp_path, _harness(_llama_primary()))

    def raising(*a, **k):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "liveness", raising)

    assert repl._foreign_incumbent(repl._harness.server, "flash-next") == SENTENCE_STOP


# --- D-69: the listener is looked up first ------------------------------------------------------- #


def _declared_only_peer() -> EndpointRef:
    """A peer the harness cannot launch (no binary): the cold branch past the foreign check ends in
    its one refusal line, so a test can watch the check without a launch."""
    return EndpointRef(
        name="flash", base_url=PEER_URL, provider_type="llamacpp", gpu=True, api_key="none",
        lifecycle=ManagedServerConfig(
            runtime="llamacpp", model="/x/flash.gguf", port=8090, gpu=True,
            extra_args=["-a", "flash-next"]))


@pytest.mark.asyncio
async def test_a_free_port_asks_no_strategy_liveness(tmp_path, fake_proc, monkeypatch):
    stub_liveness(monkeypatch, False)
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary(), peers=[_declared_only_peer()]))
    repl._active_heavy = None  # nothing to free: the question is only the foreign check's liveness
    asked: list[str] = []
    stub_liveness(monkeypatch, lambda: asked.append("asked") or False)
    launched: list[str] = []

    def no_launch(*a, **k):
        launched.append("start")
        raise RuntimeError("no launch in tests")

    monkeypatch.setattr(managed_server, "start_server", no_launch)
    monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])

    await repl._handle_slash("/model flash-next")

    assert asked == []
    assert not any("served by" in m for m in channel.messages)
    assert launched == ["start"]  # the cold branch went on to its launch (stubbed, never run)


@pytest.mark.asyncio
async def test_a_foreign_listener_asks_the_liveness_exactly_once(
        tmp_path, fake_proc, fake_systemd, monkeypatch, spies):
    stub_liveness(monkeypatch, False)
    _foreign(fake_proc, cgroup=_cgroup())
    repl, channel, _ = _build(tmp_path, _harness(_llama_primary()))
    asked: list[str] = []
    stub_liveness(monkeypatch, lambda: asked.append("asked") or False)

    await repl._handle_slash("/model flash-next")

    assert asked == ["asked"]
    assert len(channel.messages) == 1 and "served by" in channel.messages[0]


# --- the Holder fillers -------------------------------------------------------------------------- #


def test_port_holder_fills_the_holder_from_procfs_and_units(fake_proc, fake_systemd):
    fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=_show(enabled="enabled"))
    _foreign(fake_proc, cgroup=_cgroup())

    holder = workspace.port_holder(8080)

    assert (holder.unit, holder.pid, holder.command, holder.uid) == (
        UNIT, 1234, "llama-server", os.getuid())
    assert holder.state.unit_file_state == "enabled"


def test_port_holder_on_a_free_port_is_none(fake_proc):
    assert workspace.port_holder(8080) is None


def test_port_holder_for_another_uid_knows_only_the_uid(fake_proc):
    fake_proc.listen(8080, 5551, 0)

    assert workspace.port_holder(8080) == workspace.Holder(None, None, None, 0)


def test_port_holder_with_no_visible_pid_knows_only_our_uid(fake_proc):
    fake_proc.listen(8080, 5551, os.getuid())  # nobody holds the inode

    assert workspace.port_holder(8080) == workspace.Holder(None, None, None, os.getuid())


def test_holder_of_takes_the_command_from_argv0_then_exe(fake_proc, fake_systemd):
    from localharness.provider import procfs

    _foreign(fake_proc, argv=("/opt/bin/llama-server", "-m", "x"), pid=2001, inode=7001)
    rec = procfs.read_process(2001)
    assert workspace.holder_of(rec).command == "llama-server"

    fake_proc.process(2002, argv=(), exe="/opt/bin/other-server", cmdline=b"")
    rec = procfs.read_process(2002)
    assert not isinstance(rec, procfs.ProcRefusal)
    assert workspace.holder_of(rec).command == "other-server"


# --- the seed ------------------------------------------------------------------------------------ #


def test_a_llamacpp_primary_with_no_harness_pidfile_is_not_seeded(tmp_path, monkeypatch):
    monkeypatch.setattr(managed_server, "server_pid", lambda cfg: None)
    repl, _, _ = _build(tmp_path, _harness(_llama_primary()))

    assert repl._active_heavy is None


def test_a_harness_launched_primary_is_seeded(tmp_path, harness_launched):
    harness = _harness(_llama_primary())
    repl, _, _ = _build(tmp_path, harness)

    assert repl._active_heavy == (harness.server, harness.provider.base_url)


def test_an_lm_studio_primary_seeds_through_its_endpoint(tmp_path, monkeypatch):
    import httpx

    server = ManagedServerConfig(runtime="lmstudio", binary="/x/lms", model="m", port=1234, gpu=True)
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(status_code=200))
    harness = _harness(server, peers=[])
    repl, _, _ = _build(tmp_path, harness)
    assert repl._active_heavy == (server, harness.provider.base_url)

    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    repl, _, _ = _build(tmp_path, harness)
    assert repl._active_heavy is None


def test_a_liveness_that_raises_means_no_seed_and_the_repl_still_builds(tmp_path, monkeypatch):
    def raising(self, spec, config_dir):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "liveness", raising)
    server = ManagedServerConfig(launch="docker", docker_image="vllm:latest", model="m", port=8080)
    repl, _, _ = _build(tmp_path, _harness(server, peers=[]))

    assert repl._active_heavy is None


def test_no_config_dir_means_no_seed_and_liveness_is_never_asked(monkeypatch):
    asked: list[str] = []
    stub_liveness(monkeypatch, lambda: asked.append("asked") or True)
    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=SimpleNamespace(_llm=FakeLLM()),
        channel=FakeChannel(), bus=SimpleNamespace(), config_dir=None,
        harness_config=_harness(_llama_primary()))

    assert repl._active_heavy is None and asked == []


def test_a_cpu_server_or_no_provider_asks_no_liveness(tmp_path, monkeypatch):
    asked: list[str] = []
    stub_liveness(monkeypatch, lambda: asked.append("asked") or True)
    cpu = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf",
                              port=8080, gpu=False)
    repl, _, _ = _build(tmp_path, _harness(cpu))
    assert repl._active_heavy is None

    repl = OrchestratorREPL(
        orchestrator=SimpleNamespace(), agent_loop=SimpleNamespace(_llm=FakeLLM()),
        channel=FakeChannel(), bus=SimpleNamespace(), config_dir=tmp_path,
        harness_config=SimpleNamespace(server=_llama_primary(), provider=None))
    assert repl._active_heavy is None
    assert asked == []
