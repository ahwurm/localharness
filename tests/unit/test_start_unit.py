"""`start` under the harness's own autostart unit (spec 20 §8.2, §9.3, §9.4; D-18, D-19, D-21, D-22; AUTO-09).

`_start_async` senses the unit ONCE (the zero-argument seam `start_serving.unit_runtime_state`, then
`start_serving.harness_unit` for the identity-checked state), and hands that state to every strategy, swap,
verdict and boot clause it builds. Under the harness's own unit a launch is a restart of that unit
(`UnitStrategy`), never `server.start_server`.

Nothing real runs. The world is `unit_strategy_support.unit_world` (a faked service manager, a `FakeServe` thread
for the unit's `serve`, `fake_proc` for `/proc`, the HTTP wait through `server._TRANSPORT`); the rest of what
`start` crosses (the probe, the sensor, the REPL, the console) is faked as tests/unit/test_start_serving_e2e.py
fakes it. `server.start_server` is patched to fail in the unit cases and to a recorder in the no-unit ones.
"""
from __future__ import annotations

import json
import os

import pytest
import typer
import yaml

from localharness.cli import model_ops, start_cmd, start_serving
from localharness.cli.start_cmd import _start_async
from localharness.config import trust
from localharness.config.overlay import load_overlay
from localharness.provider import lifecycle, server, speed_stats, units
from tests.unit.launch_support import stub_liveness
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries
from tests.unit.unit_strategy_support import (
    PEER_URLS, PRIMARY, PRIMARY_URL, UNIT, UNIT_CGROUP, FakeServe, install_process, unit_world, vanish)

fcntl = pytest.importorskip("fcntl")
DOCKER_URL = "http://127.0.0.1:8090/v1"
SERVE_ARGV = ("/usr/bin/python3", "-I", "-m", "localharness", "serve")


class StartWorld:
    """One `start`'s surroundings on top of a `unit_world`."""

    def __init__(self, world, monkeypatch, tmp_path) -> None:
        self.world = world
        self.monkeypatch = monkeypatch
        self.answering: set[str] = set()
        self.err: list[str] = []
        self.repls: list = []
        self.activated: list[dict] = []
        self.seam_calls = 0
        self.launched = lambda: server.pid_path(world.g).exists()  # a launch this start made has recorded
        self.harness_unit_calls: list = []
        scratch = tmp_path / "scratch"
        scratch.mkdir()

        async def keep(repl):
            self.repls.append(repl)

        _stub_start_boundaries(scratch, monkeypatch, repl_run=keep)
        self.out = _capture_start_console(monkeypatch)
        monkeypatch.setattr(start_cmd.err_console, "print",
                            lambda *a, **k: self.err.append(" ".join(str(x) for x in a)))
        monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: None)
        monkeypatch.setattr(start_cmd, "_props_model_path", lambda *a, **k: None)
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(server, "list_cached_models", lambda: [])
        stub_liveness(monkeypatch, False)
        # the real `server_pid` asks the OS about a pid and unlinks a stale pidfile: a fake `/proc` pidfile must
        # survive the start, so the question is answered "no" and nothing else reads the OS
        monkeypatch.setattr(server, "server_pid", lambda _d: None)
        monkeypatch.setattr(model_ops, "list_live_models",
                            lambda base_url, timeout=3.0, **kw: (["m"], True) if base_url in self.answering else ([], False))

        async def probe(llm, max_retries=3, delay=2.0):
            url = llm.config.base_url
            if url in self.answering or (url != PRIMARY_URL and self.launched()):
                return (True, "native", 262_144, None)
            return (False, None, None, "Connection error.")

        monkeypatch.setattr(start_cmd, "_probe_llm", probe)

        real_seam = start_serving.unit_runtime_state
        real_own = start_serving.harness_unit

        def counting_seam():
            self.seam_calls += 1
            return real_seam()

        def counting_own(state, config_dir):
            got = real_own(state, config_dir)
            self.harness_unit_calls.append((state, got))
            return got

        monkeypatch.setattr(start_serving, "unit_runtime_state", counting_seam)
        monkeypatch.setattr(start_serving, "harness_unit", counting_own)

    # ------------------------------------------------------------------ the launch side

    def unit_serves(self, name: str, *, choose=None) -> FakeServe:
        """The unit's `serve` for `name`, started by the restart, answering at that record's URL."""
        w = self.world
        w.fast_polls()
        w.fast_loan()
        w.probe()
        w.http(served=name)
        return w.serve_on_restart(w.peer(name) if name != PRIMARY else w.primary, choose=choose)

    def spawn_records(self) -> None:
        """No unit: `server.start_server` is a recorder and the HTTP wait answers at once."""
        def fake_start(config_dir, cmd, **kw):
            self.activated.append({"start": " ".join(cmd)})
            server.pid_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
            server.pid_path(config_dir).write_text("4321 1 8081", encoding="utf-8")
            return 4321

        async def fake_wait(base_url, config_dir=None, **kw):
            return ["flash-next"]

        self.monkeypatch.setattr(server, "start_server", fake_start)
        self.monkeypatch.setattr(server, "wait_ready", fake_wait)
        self.world.probe()

    def no_server_start(self) -> None:
        self.monkeypatch.setattr(server, "start_server", lambda *a, **kw: pytest.fail("the harness started a server"))

    # ------------------------------------------------------------------ driving

    async def start(self, **kw) -> list[str]:
        before = len(self.out)
        await _start_async(None, False, False, str(self.world.g), **kw)
        return self.out[before:]

    async def start_exit(self, **kw) -> int | None:
        try:
            await self.start(**kw)
        except typer.Exit as exc:
            return exc.exit_code
        return None

    @property
    def overlay(self) -> dict:
        path = self.world.g / "overrides.yaml"
        return load_overlay(path) if path.exists() else {}

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        return self.world.manager.verbs

    @property
    def last_served(self) -> str | None:
        return speed_stats.last_served(speed_stats.default_speed_stats_path(self.world.g))


@pytest.fixture
def sw(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    world = unit_world(tmp_path, monkeypatch, fake_proc)
    made = StartWorld(world, monkeypatch, tmp_path)
    yield made
    world.stop_servers()


def add_docker_peer(world) -> None:
    """A vLLM docker peer `vllm-box`: a record the unit cannot serve."""
    path = world.g / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["extra_endpoints"].append({
        "name": "vllm-box", "base_url": DOCKER_URL, "provider_type": "vllm", "gpu": True,
        "lifecycle": {"gpu": True, "runtime": "vllm", "launch": "docker", "docker_image": "vllm/vllm-openai:x",
                      "model": "/m/ckpt", "port": 8090}})
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    trust.record_machine(world.g, trust.machine_snapshot(world.g))  # the person confirmed the new peer too


def docker_activate(sw):
    """Patch the docker strategy's `activate` to a recorder that sees what `unit-target` is at that moment."""

    async def activate(self, spec, config_dir, base_url, **kw):
        sw.activated.append({"docker": True, "target": units.read_target(config_dir), "peer": kw.get("peer")})
        server.pid_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
        server.pid_path(config_dir).write_text("5555 1 8090", encoding="utf-8")
        on_plan = kw.get("on_plan")
        if on_plan is not None:
            on_plan(None)
        return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=5555)

    sw.monkeypatch.setattr(lifecycle.ManagedVllmStrategy, "activate", activate)


# ======================================================================================== the tracer


async def test_preset_under_the_own_unit_launches_through_the_unit_and_senses_once(sw):
    w = sw.world
    sw.no_server_start()
    serving = sw.unit_serves("flash-next")

    lines = await sw.start(preset="flash-next")

    serving.join()
    assert serving.error is None and serving.took_lock.is_set()
    assert units.read_target(w.g) == "flash-next"                              # written by UnitStrategy.activate
    assert [v for v in sw.verbs if v[0] in ("reset-failed", "restart")] == [
        ("reset-failed", UNIT), ("restart", UNIT)]
    assert sw.seam_calls == 1                                                  # sensed once per start
    assert len(sw.harness_unit_calls) == 1
    _raw, own = sw.harness_unit_calls[0]
    assert own is not None and own.own is True
    assert sw.overlay["active_endpoint"]["name"] == "flash-next"               # the launch landed
    assert sw.last_served == "flash-next"
    assert not [x for x in lines if "runs outside autostart" in x]
    assert not [x for x in sw.err if "failed to start" in x]


async def test_a_plain_start_that_launches_the_primary_also_goes_through_the_unit(sw):
    w = sw.world
    sw.no_server_start()
    w.fast_polls()
    w.fast_loan()
    w.probe()
    w.http(served=PRIMARY)
    serving = w.serve_on_restart(w.primary)

    async def probe(llm, max_retries=3, delay=2.0):
        up = server.pid_path(w.g).exists()
        return (True, "native", 262_144, None) if up else (False, None, None, "Connection error.")

    sw.monkeypatch.setattr(start_cmd, "_probe_llm", probe)
    await sw.start()
    serving.join()
    assert serving.error is None
    assert units.read_target(w.g) == units.PRIMARY_TARGET
    assert [v[0] for v in sw.verbs if v[0] in ("reset-failed", "restart")] == ["reset-failed", "restart"]
    assert sw.seam_calls == 1


async def test_pending_preset_outstanding_the_unit_is_served_the_swap_target_and_the_landing_removes_it(sw):
    w = sw.world
    sw.no_server_start()
    model_ops.write_pending_preset(w.harness, "qwen-27b", config_dir=w.g)
    chosen: list[tuple[str, str]] = []

    def choose():
        from localharness.cli.workspace import MachineTrust
        from localharness.config.loader import ConfigLoader

        got = model_ops.resolve_serving(
            ConfigLoader(config_dir=w.g).load_harness(), MachineTrust(), None, config_dir=w.g)
        chosen.append((got.record.name, got.rung))
        return got.record.spec

    serving = sw.unit_serves("flash-next", choose=choose)
    await sw.start(preset="flash-next")
    serving.join()
    assert serving.error is None
    assert chosen == [("flash-next", model_ops.RUNG_UNIT)]                     # never the pending_preset's qwen-27b
    assert "pending_preset" not in sw.overlay                                  # removed by the landing's persist_landed
    assert sw.overlay["active_endpoint"]["name"] == "flash-next"


async def test_with_no_unit_the_launch_is_phase_13s_start_server(sw, monkeypatch):
    monkeypatch.setattr(units, "_run_manager", lambda argv, *, timeout, cap, **kw: None)  # no manager to ask
    sw.spawn_records()
    await sw.start(preset="flash-next")
    assert sw.seam_calls == 1
    assert [a["start"].split()[0] for a in sw.activated] == ["/x/llama-server"]
    assert sw.verbs == [] and units.read_target(sw.world.g) is None
    assert sw.overlay["active_endpoint"]["name"] == "flash-next"


async def test_a_unit_that_is_not_the_harness_own_selects_nothing(sw):
    """A raw, identity-unchecked state never selects `UnitStrategy` (T-14-29): a unit loaded from another file
    answers a real state, `harness_unit` finds it foreign, and the launch is a spawn."""
    sw.world.manager.base["FragmentPath"] = "/home/someone/.config/systemd/user/localharness-llm.service"
    sw.spawn_records()
    await sw.start(preset="flash-next")
    (_raw, own), = sw.harness_unit_calls
    assert isinstance(_raw, units.UnitRuntimeState) and own is None
    assert [a["start"].split()[0] for a in sw.activated] == ["/x/llama-server"]
    assert [v for v in sw.verbs if v[0] in ("reset-failed", "restart")] == []
    assert units.read_target(sw.world.g) is None


async def test_start_asks_the_manager_exactly_once_when_it_only_attaches(sw):
    sw.answering = {PEER_URLS["flash-next"]}
    sw.no_server_start()
    await sw.start(preset="flash-next")
    assert sw.seam_calls == 1
    assert sw.world.manager.shows() == 1                                       # one `show` for the whole start
    assert sw.verbs == []


async def test_preset_swap_under_the_own_unit_frees_the_unit_incumbent_through_the_unit_then_restarts_it(sw):
    """`start --preset flash-next` while the unit serves the primary: the swap stops the primary through the
    unit (owner `unit`), restarts the unit for flash-next, and no process is stopped or started by the harness."""
    w = sw.world
    sw.no_server_start()
    sw.monkeypatch.setattr(server, "stop_server", lambda *a, **kw: pytest.fail("the unit's server was stopped by the harness"))
    pid = 3000
    install_process(w.proc, pid, argv=FakeServe.record_argv(w.primary), exe=str(w.primary.binary), start_time=4000,
                    cgroup=UNIT_CGROUP)
    server.pid_path(w.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(w.g).write_text(f"{pid} 4000 {w.primary.port}", encoding="utf-8")
    w.manager.base["MainPID"] = pid
    manager = w.manager

    def run(argv, *, timeout, cap, **kw):
        done = manager(argv, timeout=timeout, cap=cap, **kw)
        if tuple(argv)[:3] == ("systemctl", "--user", "stop") and done is not None and done.code == 0:
            vanish(w.proc, pid)
            server.pid_path(w.g).unlink(missing_ok=True)
            manager.base["MainPID"] = 0
        return done

    manager.answer_prefix("systemctl", "--user", "stop", out="")
    sw.monkeypatch.setattr(units, "_run_manager", run)
    w.fast_polls()
    w.fast_loan()
    w.probe()
    w.http(served="flash-next")
    serving = w.serve_on_restart(w.peer("flash-next"), main_pid=False)
    manager.on_restart.append(lambda: manager.base.update(MainPID=serving.pid))
    sw.answering = {PRIMARY_URL}

    sw.launched = serving.recorded.is_set  # the incumbent's pidfile does not make the target answer

    lines = await sw.start(preset="flash-next")

    serving.join()
    assert serving.error is None
    assert sw.verbs == [("reset-failed", UNIT), ("stop", UNIT), ("reset-failed", UNIT), ("restart", UNIT)]
    assert units.read_target(w.g) == "flash-next" and sw.overlay["active_endpoint"]["name"] == "flash-next"
    assert [x for x in lines if x.startswith("stopping qwen")]
    assert sw.seam_calls == 1


# ===================================================================================== the starting branch


async def test_the_units_serve_is_waited_on_for_the_states_timeout_and_then_named_never_signalled(sw, monkeypatch):
    """The real MainPID and the cgroup mark the unit's own `serve`: `start` waits for it (the bound is the state's
    TimeoutStartSec, 1800 s) and, when nothing answers, names the pid and exits 1 without signalling it."""
    w = sw.world
    w.proc.process(4242, argv=SERVE_ARGV, exe="/usr/bin/python3", start_time=777, cgroup=UNIT_CGROUP)
    server.pid_path(w.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(w.g).write_text("4242 777", encoding="utf-8")
    w.manager.base["MainPID"] = 4242
    waited: list[float] = []
    real_wait = start_serving.wait_starting

    async def quick(probe, *, seconds, poll_seconds=3.0):
        """The bound `start` computed is printed and reported; the wait itself is shrunk to a blink."""
        waited.append(seconds)
        return await real_wait(probe, seconds=0.05, poll_seconds=0.01)

    monkeypatch.setattr(start_serving, "wait_starting", quick)
    killed: list = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))
    monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a))
    sw.no_server_start()

    code = await sw.start_exit()

    assert code == 1
    assert "autostart is starting qwen (pid 4242) — waiting up to 1800s" in sw.out
    assert "autostart's server (pid 4242) did not answer within 1800s — not touched" in sw.err
    assert waited == [1800] and killed == [] and sw.verbs == []
    assert sw.seam_calls == 1                                                  # the verdict took the sensed state


async def test_the_units_serve_that_finishes_loading_is_attached_not_launched_beside(sw, monkeypatch):
    w = sw.world
    w.proc.process(4242, argv=SERVE_ARGV, exe="/usr/bin/python3", start_time=777, cgroup=UNIT_CGROUP)
    server.pid_path(w.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(w.g).write_text("4242 777", encoding="utf-8")
    w.manager.base["MainPID"] = 4242

    real_wait = start_serving.wait_starting

    async def quick(probe, *, seconds, poll_seconds=3.0):
        asked = {"n": 0}

        async def loading():
            asked["n"] += 1
            if asked["n"] == 2:
                sw.answering.add(PRIMARY_URL)                                 # the unit's server finishes loading
            return await probe()

        return await real_wait(loading, seconds=5, poll_seconds=0.01)

    monkeypatch.setattr(start_serving, "wait_starting", quick)
    sw.no_server_start()
    lines = await sw.start()
    assert "autostart is starting qwen (pid 4242) — waiting up to 1800s" in lines
    assert sw.verbs == [] and sw.last_served == PRIMARY


# ========================================================================================== the boot clause


async def test_an_attach_confirms_the_boot_marker_through_the_live_unit_state(sw, monkeypatch):
    w = sw.world
    spec = w.peer("flash-next")
    key = lifecycle.record_key(spec)
    w.proc.process(4242, argv=("/x/llama-server", "-a", "flash-next", "--port", "8081"), exe="/x/llama-server",
                   start_time=777)
    server.pid_path(w.g).parent.mkdir(parents=True, exist_ok=True)
    server.pid_path(w.g).write_text("4242 777 8081", encoding="utf-8")
    marker = server.server_dir(w.g) / units.BOOT_MARKER_NAME
    marker.write_text(json.dumps(dict(
        record=key, binary=str(spec.binary), invocation="inv-1", cause="death", shape="retry",
        planned_context=40_000, full_fit=80_000, free_bytes=20 * 2**30, dropped=["draft-mtp"],
        at="2026-10-13T11:59:30+00:00")), encoding="utf-8")
    sw.answering = {PEER_URLS["flash-next"]}
    sw.no_server_start()
    seen: list = []
    real = start_serving.boot_clause

    def spy(config_dir, record, **kw):
        seen.append(kw["unit_state"])
        return real(config_dir, record, **kw)

    monkeypatch.setattr(start_serving, "boot_clause", spy)

    await sw.start(preset="flash-next")

    (state,) = seen                                                           # the sensed state, not a second ask
    assert isinstance(state, units.UnitRuntimeState)
    assert (state.invocation_id, state.n_restarts) == ("inv-1", 0)
    assert not marker.exists()                                                # confirm_launch(marker=) consumed it
    assert sw.seam_calls == 1


# ========================================================================================= outside autostart


async def test_a_launch_outside_the_unit_removes_the_target_and_says_the_next_boot_serves_nothing(sw):
    w = sw.world
    add_docker_peer(w)
    units.write_target(w.g, "flash-next")
    docker_activate(sw)
    sw.no_server_start()

    lines = await sw.start(preset="vllm-box")

    assert sw.activated == [{"docker": True, "target": None, "peer": "vllm-box"}]   # removed BEFORE the launch
    assert units.read_target(w.g) is None and sw.verbs == []
    (line,) = [x for x in lines if "is not running — starting it" in x]
    assert line.endswith(f"model load can take several minutes)... {lifecycle.RUNS_OUTSIDE_AUTOSTART}")


async def test_a_swap_to_a_record_outside_the_unit_says_so_in_its_launch_line(sw):
    w = sw.world
    add_docker_peer(w)
    units.write_target(w.g, "flash-next")
    docker_activate(sw)
    sw.no_server_start()
    sw.answering = {PRIMARY_URL}

    lines = await sw.start(preset="vllm-box")

    (line,) = [x for x in lines if x.startswith("stopping qwen")]
    assert "starting vllm-box (" in line and line.endswith(lifecycle.RUNS_OUTSIDE_AUTOSTART)
    assert units.read_target(w.g) is None
    assert sw.activated and sw.activated[0]["target"] is None


async def test_a_record_the_unit_serves_never_carries_the_outside_clause(sw):
    sw.no_server_start()
    serving = sw.unit_serves("flash-next")
    lines = await sw.start(preset="flash-next")
    serving.join()
    assert not [x for x in lines + sw.err if "runs outside autostart" in x]


async def test_with_no_unit_a_docker_launch_has_no_clause_and_leaves_the_target_alone(sw):
    w = sw.world
    w.manager.base["FragmentPath"] = "/elsewhere/localharness-llm.service"
    add_docker_peer(w)
    units.write_target(w.g, "flash-next")
    docker_activate(sw)
    sw.no_server_start()
    lines = await sw.start(preset="vllm-box")
    assert not [x for x in lines if "runs outside autostart" in x]
    assert sw.activated[0]["target"] == "flash-next" and units.read_target(w.g) == "flash-next"


# ============================================================================================ harness_unit


def test_harness_unit_makes_an_own_state_only_from_a_state_that_passed_the_identity_tests(sw):
    w = sw.world
    raw = units.UnitRuntimeState(None, 0, fragment_path=str(units.harness_unit_path()), active_state="active")
    got = start_serving.harness_unit(raw, w.g)
    assert got is not None and got.own is True and got.fragment_path == raw.fragment_path
    assert raw.own is False                                                   # the sensed value is never mutated


@pytest.mark.parametrize("state", [
    None, "a state", object(), {"main_pid": 1},
    units.UnitRuntimeState(None, 0, fragment_path="/home/someone/localharness-llm.service"),
    units.UnitRuntimeState(None, 0, fragment_path="", drop_in_paths="/x/override.conf")])
def test_harness_unit_is_none_for_anything_that_is_not_the_harness_own_unit(sw, state):
    assert start_serving.harness_unit(state, sw.world.g) is None


def test_harness_unit_is_none_for_a_unit_file_that_is_not_byte_equal_to_the_render(sw):
    path = units.harness_unit_path()
    path.write_text(path.read_text(encoding="utf-8") + "ExecStartPost=/bin/true\n", encoding="utf-8")
    raw = units.UnitRuntimeState(None, 0, fragment_path=str(path), active_state="active")
    assert start_serving.harness_unit(raw, sw.world.g) is None
