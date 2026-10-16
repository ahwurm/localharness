"""The REPL launches through the same autostart unit `start` sensed (spec 20 §9.4, §13 repl row, §8.2; D-19..D-21; R-14-29).

`OrchestratorREPL(..., unit_state=own)` inherits the state `start` sensed once: every strategy, swap, free, stop
and restore the session builds takes it, so a cold `/model flash-next` is a restart of the unit with the same
manager verbs and the same `unit-target` as `start --preset flash-next`. The session never senses the unit
itself, and what the unit cannot serve (a typed `.gguf`, a model that is not the record's) is refused before
anything is stopped.

Nothing real runs: the world is `unit_strategy_support.unit_world` (a faked service manager, a `FakeServe` thread
for the unit's `serve`, `fake_proc` for `/proc`, the HTTP wait through `server._TRANSPORT`); `server.start_server`
is patched to fail in the unit cases and to a recorder in the no-unit ones.
"""
from __future__ import annotations

import sys

import pytest
import yaml

from localharness.cli import start_serving
from localharness.cli.repl import OrchestratorREPL
from localharness.config.loader import ConfigLoader
from localharness.config.models import ManagedServerConfig
from localharness.provider import lifecycle, server, units
from tests.unit.launch_support import stub_liveness
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_repl_model_cmd import FakeChannel, _xrepl
from tests.unit.unit_strategy_support import PEER_URLS, PRIMARY, PRIMARY_URL, UNIT, unit_world

fcntl = pytest.importorskip("fcntl")
DOCKER_URL = "http://127.0.0.1:8090/v1"
OUTSIDE = lifecycle.RUNS_OUTSIDE_AUTOSTART


class StyledChannel(FakeChannel):
    """A channel that keeps the style each line was sent with."""

    def __init__(self) -> None:
        super().__init__()
        self.styled: list[tuple[str, str | None]] = []

    async def send_message(self, text, metadata=None):
        await super().send_message(text, metadata=metadata)
        self.styled.append((text, (metadata or {}).get("style")))


class ReplWorld:
    """A `unit_world` plus the REPL built on it, the sensing seam counted, and the launch boundaries guarded."""

    def __init__(self, world, monkeypatch) -> None:
        self.world = world
        self.monkeypatch = monkeypatch
        self.seam_calls = 0
        self.own_calls = 0
        self.strategy_calls: list[tuple[str, object]] = []
        self.identity_calls: list[object] = []
        self.started: list[str] = []
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(server, "list_cached_models", lambda: [])
        monkeypatch.setattr(server, "server_pid", lambda _d: None)
        stub_liveness(monkeypatch, False)

        def seam():
            self.seam_calls += 1
            raise AssertionError("the REPL sensed the unit itself")

        def own(state, config_dir):
            self.own_calls += 1
            raise AssertionError("the REPL turned a raw state into an own-unit state itself")

        monkeypatch.setattr(start_serving, "unit_runtime_state", seam)
        monkeypatch.setattr(start_serving, "harness_unit", own)
        real = lifecycle.strategy_for

        def spy(spec, **kw):
            # `record_identity` asks with unit=None on purpose (identity does not depend on the owner): its
            # asks are not the session's launches, so they are kept apart
            if sys._getframe(1).f_code.co_name == "record_identity":
                self.identity_calls.append(kw.get("unit", "<missing>"))
            else:
                self.strategy_calls.append((spec.runtime, kw.get("unit", "<missing>")))
            return real(spec, **kw)

        monkeypatch.setattr(lifecycle, "strategy_for", spy)

    # ------------------------------------------------------------------ building

    def build(self, *, unit, harness=None, **repl_kw):
        w = self.world
        harness = harness if harness is not None else w.harness
        urls = {PRIMARY_URL: [PRIMARY], DOCKER_URL: "unreachable",
                **{url: "unreachable" for url in PEER_URLS.values()}}
        repl, _, agent, _ = _xrepl(w.g, harness, urls, unit_state=unit, **repl_kw)
        channel = StyledChannel()
        repl._channel = channel
        agent._llm.config.base_url = PRIMARY_URL
        agent._llm.config.model = PRIMARY

        async def refresh(model, *, base_url=None, provider_type=None):
            return ""

        repl._refresh_token_counter = refresh
        return repl, channel, agent

    # ------------------------------------------------------------------ boundaries

    def no_server_start(self) -> None:
        self.monkeypatch.setattr(server, "start_server",
                                 lambda *a, **kw: pytest.fail("the harness started a server beside the unit"))

    def spawn_records(self) -> None:
        def fake_start(config_dir, cmd, **kw):
            self.started.append(" ".join(cmd))
            return 4321

        async def fake_wait(base_url, config_dir=None, **kw):
            return ["flash-next"]

        self.monkeypatch.setattr(server, "start_server", fake_start)
        self.monkeypatch.setattr(server, "wait_ready", fake_wait)

    def nothing_stops(self) -> None:
        """Every way a REPL restart stops something fails the test."""
        async def free(*a, **kw):
            pytest.fail("free_accelerator ran before the refusal")

        async def stop(self_, *a, **kw):
            pytest.fail("a strategy stop ran before the refusal")

        self.monkeypatch.setattr(lifecycle, "free_accelerator", free)
        for cls in (lifecycle.UnitStrategy, lifecycle.SpawnedProcessStrategy, lifecycle.ManagedVllmStrategy):
            self.monkeypatch.setattr(cls, "stop", stop)
        self.monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("a server was stopped"))

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        return [v for v in self.world.manager.verbs if v[0] in ("reset-failed", "restart")]


@pytest.fixture
def rw(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    world = unit_world(tmp_path, monkeypatch, fake_proc, primary_extra={"gpu": True})
    made = ReplWorld(world, monkeypatch)
    yield made
    world.stop_servers()


def add_docker_peer(world) -> object:
    """A vLLM docker peer `vllm-box`: a record the unit cannot serve. Returns the reloaded harness."""
    path = world.g / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["extra_endpoints"].append({
        "name": "vllm-box", "base_url": DOCKER_URL, "provider_type": "vllm", "gpu": True,
        "lifecycle": {"gpu": True, "runtime": "vllm", "launch": "docker", "docker_image": "vllm/vllm-openai:x",
                      "model": "/m/ckpt", "port": 8090}})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return ConfigLoader(config_dir=world.g).load_harness()


def docker_activate(rw) -> list[dict]:
    seen: list[dict] = []

    async def activate(self, spec, config_dir, base_url, **kw):
        seen.append({"target": units.read_target(config_dir), "peer": kw.get("peer")})
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["vllm-box"], handle=5555)

    rw.monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "activate", activate)
    return seen


# ======================================================================================== the tracer


@pytest.mark.asyncio
async def test_a_cold_model_swap_under_the_unit_restarts_the_unit_with_the_verbs_start_records(rw):
    w = rw.world
    rw.no_server_start()
    w.fast_polls()
    w.fast_loan()
    w.probe()
    w.http(served="flash-next")
    serving = w.serve_on_restart(w.peer("flash-next"))
    repl, channel, agent = rw.build(unit=w.state)

    await repl._handle_slash("/model flash-next")

    serving.join()
    assert serving.error is None and serving.took_lock.is_set()
    # the same ordered verbs and the same unit-target as `start --preset flash-next` in this world
    assert rw.verbs == [("reset-failed", UNIT), ("restart", UNIT)]
    assert units.read_target(w.g) == "flash-next"
    assert channel.messages[-1].startswith("Switched to flash-next on flash-next")
    assert agent._llm.config.base_url == PEER_URLS["flash-next"]
    assert OUTSIDE not in " ".join(channel.messages)
    assert (rw.seam_calls, rw.own_calls) == (0, 0)


@pytest.mark.asyncio
async def test_every_strategy_the_repl_builds_takes_the_state_it_was_given(rw):
    w = rw.world
    rw.no_server_start()
    w.fast_polls()
    w.fast_loan()
    w.probe()
    w.http(served="flash-next")
    w.serve_on_restart(w.peer("flash-next"))
    repl, _channel, _agent = rw.build(unit=w.state)
    assert repl._unit_state is w.state

    await repl._handle_slash("/model flash-next")

    assert rw.strategy_calls, "the swap built no strategy at all"
    assert all(unit is w.state for _runtime, unit in rw.strategy_calls), rw.strategy_calls
    assert (rw.seam_calls, rw.own_calls) == (0, 0)


def test_the_constructor_seeds_ask_the_strategy_with_the_inherited_state(rw):
    w = rw.world
    held = (w.primary, PRIMARY_URL)

    OrchestratorREPL(
        orchestrator=object(), agent_loop=object(), channel=FakeChannel(), bus=object(), config_dir=w.g,
        harness_config=w.harness, serving_heavy=held, unit_state=w.state)
    OrchestratorREPL(
        orchestrator=object(), agent_loop=object(), channel=FakeChannel(), bus=object(), config_dir=w.g,
        harness_config=w.harness, unit_state=w.state)

    # the held record's liveness, the primary seed it falls back to when that record is not up, and the
    # second session's primary seed
    assert len(rw.strategy_calls) == 3
    assert all(unit is w.state for _runtime, unit in rw.strategy_calls)
    assert (rw.seam_calls, rw.own_calls) == (0, 0)


def test_a_repl_built_without_a_unit_state_asks_every_strategy_for_none(rw):
    w = rw.world
    repl, _channel, _agent = rw.build(unit=None)

    OrchestratorREPL(
        orchestrator=object(), agent_loop=object(), channel=FakeChannel(), bus=object(), config_dir=w.g,
        harness_config=w.harness, serving_heavy=(w.primary, PRIMARY_URL))

    assert repl._unit_state is None
    assert rw.strategy_calls and all(unit is None for _runtime, unit in rw.strategy_calls)
    assert not isinstance(lifecycle.strategy_for(w.primary, unit=None), lifecycle.UnitStrategy)


# =============================================================================== the typed refusal


@pytest.mark.asyncio
async def test_a_typed_gguf_restart_under_the_unit_is_refused_before_anything_is_stopped(rw, tmp_path):
    w = rw.world
    rw.no_server_start()
    rw.nothing_stops()
    typed = tmp_path / "other.gguf"
    typed.write_bytes(b"GGUF")
    repl, channel, agent = rw.build(unit=w.state)
    repl._active_heavy = (w.primary, PRIMARY_URL)
    before = (w.primary.model, w.primary.served_name)

    await repl._handle_slash(f"/model {typed}")

    line = lifecycle.UNIT_TYPED_LINE.format(path=str(typed))
    assert channel.styled[-1] == (line, "system.error")
    assert line == ("under autostart /model takes preset names — put " + str(typed) +
                    " under autosense.model_roots and run localharness init to add it as a peer")
    assert w.manager.verbs == []                                  # no verb: nothing reset, nothing restarted
    assert units.read_target(w.g) is None                         # nothing written
    assert repl._active_heavy == (w.primary, PRIMARY_URL)         # the session still tracks the incumbent
    assert (w.primary.model, w.primary.served_name) == before
    assert agent._llm.config.base_url == PRIMARY_URL              # not even re-pointed


@pytest.mark.asyncio
async def test_a_vllm_binary_restart_to_a_model_that_is_not_the_records_is_refused_the_same_way(rw, monkeypatch):
    w = rw.world
    rw.no_server_start()
    rw.nothing_stops()
    record = ManagedServerConfig(runtime="vllm", launch="binary", binary="/x/vllm", model="org/rec",
                                 port=8080, gpu=True)
    harness = w.harness.model_copy(update={"server": record})
    monkeypatch.setattr(server, "list_cached_models", lambda: ["other-model"])
    repl, channel, _agent = rw.build(unit=w.state, harness=harness)
    repl._live_models = _live([])
    repl._active_heavy = (record, PRIMARY_URL)

    await repl._handle_slash("/model other-model")

    assert channel.styled[-1] == (lifecycle.UNIT_TYPED_LINE.format(path="other-model"), "system.error")
    assert w.manager.verbs == [] and units.read_target(w.g) is None
    assert record.model == "org/rec"                              # the shared record keeps its weights
    assert repl._active_heavy == (record, PRIMARY_URL)


def _live(models):
    async def live(base_url):
        return list(models), True

    return live


@pytest.mark.asyncio
async def test_without_a_unit_a_typed_gguf_restart_goes_on_to_stop_and_launch_as_before(rw, tmp_path, monkeypatch):
    rw.spawn_records()
    seen: list[str] = []

    async def stop(self, spec, config_dir):
        seen.append("stop")

    async def activate(self, spec, config_dir, base_url, **kw):
        seen.append(f"activate:{spec.model}")
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=[spec.served_name], handle=1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "stop", stop)
    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    typed = tmp_path / "other.gguf"
    typed.write_bytes(b"GGUF")
    repl, channel, _agent = rw.build(unit=None)
    repl._live_models = _live([])

    await repl._handle_slash(f"/model {typed}")

    assert seen == ["stop", f"activate:{typed}"]
    assert not [m for m, _s in channel.styled if "under autostart" in m]


# ============================================================================ outside the unit


@pytest.mark.asyncio
async def test_a_cold_docker_peer_under_the_unit_says_the_next_boot_serves_nothing(rw):
    w = rw.world
    harness = add_docker_peer(w)
    seen = docker_activate(rw)
    units.write_target(w.g, "flash-next")                          # the older choice the next boot would serve
    repl, channel, _agent = rw.build(unit=w.state, harness=harness)

    await repl._handle_slash("/model vllm-box")

    launching = next(m for m in channel.messages if m.startswith("Launching vllm-box"))
    assert OUTSIDE in launching
    assert launching.index("several minutes...") < launching.index(OUTSIDE)
    assert seen == [{"target": None, "peer": "vllm-box"}]         # unit-target taken away before its own launch
    assert units.read_target(w.g) is None
    assert w.manager.verbs == []                                  # the unit was not touched


@pytest.mark.asyncio
async def test_the_same_docker_peer_without_a_unit_carries_no_clause(rw):
    w = rw.world
    harness = add_docker_peer(w)
    docker_activate(rw)
    repl, channel, _agent = rw.build(unit=None, harness=harness)

    await repl._handle_slash("/model vllm-box")

    launching = next(m for m in channel.messages if m.startswith("Launching vllm-box"))
    assert OUTSIDE not in launching
    assert channel.messages[-1].startswith("Switched to vllm-box")


# ============================================================ free, stop and restore reach the unit


@pytest.mark.asyncio
async def test_the_managed_restart_frees_the_accelerator_through_the_unit_state(rw):
    w = rw.world
    peer = w.peer("flash-next")
    seen: list[object] = []

    async def free(incumbent, target, config_dir, **kw):
        seen.append(kw.get("unit", "<missing>"))
        raise RuntimeError("held")                                # ends the restart cleanly: nothing else runs

    rw.monkeypatch.setattr(lifecycle, "free_accelerator", free)
    repl, channel, agent = rw.build(unit=w.state)
    repl._live_models = _live([])                                 # the primary is down: a swap back restarts it
    agent._llm.config.base_url = PEER_URLS["flash-next"]          # the session sits on the launched peer
    agent._llm.config.model = "flash-next"
    repl._active_heavy = (peer, PEER_URLS["flash-next"])

    await repl._handle_slash(f"/model {PRIMARY}")

    assert seen == [w.state], channel.styled
    assert channel.styled[-1][1] == "system.error" and "Could not free the GPU: held" in channel.styled[-1][0]
    assert w.manager.verbs == []


@pytest.mark.asyncio
async def test_safe_stop_and_restore_go_to_the_lifecycle_with_the_state_the_repl_holds(rw):
    w = rw.world
    seen: dict[str, object] = {}

    async def safe_stop(spec, config_dir, **kw):
        seen["stop"] = kw.get("unit", "<missing>")

    async def restore(heavy, **kw):
        seen["restore"] = kw.get("unit", "<missing>")
        return heavy.spec.model, None

    rw.monkeypatch.setattr(lifecycle, "safe_stop", safe_stop)
    rw.monkeypatch.setattr(lifecycle, "restore_incumbent", restore)
    repl, _channel, _agent = rw.build(unit=w.state)

    await repl._safe_stop(w.primary)
    label, note = await repl._restore_incumbent((w.primary, PRIMARY_URL))

    assert seen == {"stop": w.state, "restore": w.state}
    assert label == w.primary.model and note is None
    assert (rw.seam_calls, rw.own_calls) == (0, 0)


@pytest.mark.asyncio
async def test_the_recovery_pair_asks_the_strategy_with_the_inherited_state(rw):
    w = rw.world
    repl, _channel, _agent = rw.build(unit=w.state)
    before = len(rw.strategy_calls)

    repl._active_heavy = (w.primary, PRIMARY_URL)
    pair = repl.server_recovery()

    assert pair is not None and len(rw.strategy_calls) > before
    assert all(unit is w.state for _runtime, unit in rw.strategy_calls)


# ================================================================== what start hands the session


@pytest.fixture
def started(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    from tests.unit.test_start_unit import StartWorld

    world = unit_world(tmp_path, monkeypatch, fake_proc)
    made = StartWorld(world, monkeypatch, tmp_path)
    yield made
    world.stop_servers()


@pytest.mark.asyncio
async def test_start_hands_the_session_the_one_state_it_sensed(started):
    started.no_server_start()
    serving = started.unit_serves("flash-next")

    await started.start(preset="flash-next")

    serving.join()
    (_raw, own), = started.harness_unit_calls
    assert own is not None and own.own is True
    assert started.seam_calls == 1                                # sensed once, by `start`
    assert [repl._unit_state for repl in started.repls] == [own]  # the very value, handed over, not re-derived


@pytest.mark.asyncio
async def test_start_hands_the_session_none_when_the_unit_is_not_the_harnesss_own(started):
    started.world.manager.base["FragmentPath"] = "/home/someone/.config/systemd/user/localharness-llm.service"
    started.spawn_records()

    await started.start(preset="flash-next")

    assert started.seam_calls == 1
    assert [repl._unit_state for repl in started.repls] == [None]
