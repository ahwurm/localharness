# ruff: noqa: F401, F811
"""`_start_async` end to end on the ladder and the serving decision (plan 13-10, spec 20 §8.2, §8.4,
§1 item 3, D-13..D-18, D-29, SEC-04).

Driven as tests/unit/test_launch_paths_start.py and tests/unit/test_window_lines_start.py drive it: the
probe, the sensor's `list_live_models`, the three process boundaries a launch crosses (`stop_server`,
`start_server`, `wait_ready`), the window's free memory and the `/proc` side are faked through their module
seams, and the config dir is `tmp_path`. No socket is opened, no process started, no service manager asked
(the owner's server on :8080 is never touched). Test fakes take `**kw`.
"""
from __future__ import annotations

import asyncio
import os
import contextlib
from types import SimpleNamespace

import pytest
import typer
import yaml

from localharness.agent.context import WindowProbe
from localharness.cli import model_ops, start_cmd, start_serving
from localharness.cli.start_cmd import _start_async
from localharness.config.overlay import load_overlay
from localharness.provider import autosense, lifecycle, procfs, speed_stats, units
from localharness.provider import server as managed_server
from localharness.provider.autosense import FreeMemory
from tests.unit.launch_support import stub_liveness
from tests.unit.procfs_support import fake_proc  # noqa: F401
from tests.unit.test_launch_paths_start import _boot, _drive  # noqa: F401
from tests.unit.test_sense_serving import (  # noqa: F401
    FLASH_ACTIVE,
    PEER_URL,
    PRIMARY,
    PRIMARY_URL,
    PROVIDER_KEY,
    Probes,
    home,
    load,
    peer,
    seed,
)
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

GIB = 2**30
LLAMA = "/x/llama-server"
FLASH_LOW = {**FLASH_ACTIVE, "effort": "low"}
RESUME_LINE = "Serving flash-next (your last /model choice)"


class World:
    """One start's faked surroundings. ``answering`` is the set of base URLs whose server answers (both
    the sensor's model-list probe and start's own capability probe read it); ``events`` records the three
    process boundaries in order; ``clients`` every ``LLMConfig`` an ``LLMClient`` was built from."""

    def __init__(self, home, monkeypatch) -> None:
        self.home = home
        self.monkeypatch = monkeypatch
        self.answering: set[str] = set()
        self.served_ids: dict[str, list[str]] = {}
        self.events: list[str] = []
        self.waits: list[dict] = []
        self.starts: list[dict] = []
        self.stops: list[dict] = []
        self.clients: list = []
        self.agents: list = []
        self.repls: list = []
        self.probe_urls: list[str] = []
        self.wait_raises: dict[str, BaseException] = {}
        self.alive: dict[str, bool] = {}
        self.err: list[tuple[tuple, dict]] = []
        self.window_served = 65536
        self.out_at_wait: list[int] = []

        async def keep(repl):
            self.agents.append(repl._agent)
            self.repls.append(repl)

        _stub_start_boundaries(home, monkeypatch, repl_run=keep)
        self.out = _capture_start_console(monkeypatch)
        monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: self.err.append((a, k)))
        monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: None)
        monkeypatch.setattr(start_cmd, "_props_model_path", lambda *a, **k: None)
        stub_liveness(monkeypatch, False)
        # the real `server_pid` asks the OS about a pid and unlinks a stale pidfile: a fake `/proc` pidfile
        # must survive the start, so the question is answered "no" and nothing else reads the OS
        monkeypatch.setattr(managed_server, "server_pid", lambda _d: None)
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
        monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(6 * GIB, 32 * GIB, "nvidia-smi"))
        monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: 18 * GIB)
        monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "24576, 32768\n")
        monkeypatch.setattr(
            autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})
        monkeypatch.setattr(
            "localharness.agent.context.probe_window",
            lambda base_url, model, provider_type=None, **kw: WindowProbe(self.window_served, 262144))
        self.real_wait_ready = managed_server.wait_ready  # for the test that runs the real death path
        self.sensor = Probes()
        monkeypatch.setattr(model_ops, "list_live_models", self._list_live_models)

        async def probe(llm, max_retries=3, delay=2.0):
            url = llm.config.base_url
            self.probe_urls.append(url)
            if url in self.answering:
                return (True, "native", 262_144, None)
            return (False, None, None, "Connection error.")

        monkeypatch.setattr(start_cmd, "_probe_llm", probe)

        from localharness.provider.client import LLMClient
        real_init = LLMClient.__init__

        def spy_init(client, config, *a, **kw):
            self.clients.append(config)
            return real_init(client, config, *a, **kw)

        monkeypatch.setattr(LLMClient, "__init__", spy_init)
        self._boundaries()

    def _list_live_models(self, base_url, timeout=3.0, **kw):
        self.sensor.calls.append({"url": base_url, "timeout": timeout, "kw": kw})
        if base_url in self.answering:
            return (self.served_ids.get(base_url, []), True)
        return ([], False)

    def _boundaries(self) -> None:
        def fake_stop(config_dir, launch="binary", timeout_seconds=30.0, **kw):
            self.events.append("stop")
            self.stops.append(kw)
            for name in self.alive:
                self.alive[name] = False
            return True

        def fake_start(config_dir, cmd, **kw):
            joined = " ".join(cmd)
            self.events.append("start:" + joined)
            self.starts.append({"cmd": cmd, **kw})
            return 4321

        async def fake_wait(base_url, config_dir=None, **kw):
            self.events.append(f"wait:{base_url}")
            self.out_at_wait.append(len(self.out))
            self.waits.append({"url": base_url, "kw": kw, "flash_alive": self.alive.get("flash-next", False)})
            if base_url in self.wait_raises:
                raise self.wait_raises[base_url]
            self.answering.add(base_url)
            return [PRIMARY if base_url == PRIMARY_URL else "flash-next"]

        @contextlib.asynccontextmanager
        async def spy_lock(config_dir, *, on_wait=None):
            self.events.append("lock-enter")
            try:
                yield
            finally:
                self.events.append("lock-exit")

        self.monkeypatch.setattr(managed_server, "stop_server", fake_stop)
        self.monkeypatch.setattr(managed_server, "start_server", fake_start)
        self.monkeypatch.setattr(managed_server, "wait_ready", fake_wait)
        self.monkeypatch.setattr(lifecycle, "launch_lock", spy_lock)

    def config(self, **kw) -> None:
        # a flash-next whose launches are told apart by `-a flash-next` (the record's served name)
        kw.setdefault("peers", [peer("flash-next", served_name="flash-next")])
        seed(self.home, **kw)

    def kinds(self) -> list[str]:
        """The recorded boundary events, kind only, the lock's enter and exit left out."""
        return [e.split(":")[0] for e in self.events if not e.startswith("lock")]

    def reset(self) -> None:
        self.events.clear()
        self.waits.clear()
        self.starts.clear()
        self.stops.clear()
        self.out_at_wait.clear()
        self.answering.clear()

    async def start(self, *args, **kw) -> list[str]:
        """Run one `start`; the lines it printed on stdout (this start only)."""
        before = len(self.out)
        await _start_async(None, False, False, str(self.home), *args, **kw)
        return self.out[before:]

    async def start_exit(self, *args, **kw) -> int | None:
        try:
            await _start_async(None, False, False, str(self.home), *args, **kw)
        except typer.Exit as exc:
            return exc.exit_code
        return None

    @property
    def err_lines(self) -> list[str]:
        return [" ".join(str(x) for x in a) for a, _ in self.err]

    @property
    def loop(self):
        return self.agents[-1]

    @property
    def ledger(self):
        return speed_stats.default_speed_stats_path(self.home)

    @property
    def last_served(self) -> str | None:
        return speed_stats.last_served(self.ledger)

    @property
    def overlay(self) -> dict:
        return load_overlay(self.home / "overrides.yaml") if (self.home / "overrides.yaml").exists() else {}

    def session_urls(self) -> set[tuple[str, str]]:
        return {(c.base_url, c.model) for c in self.clients}

    def session_endpoint(self) -> tuple[str, str]:
        """The endpoint of the session's own client (the last one built)."""
        return (self.clients[-1].base_url, self.clients[-1].model)


@pytest.fixture
def world(home, monkeypatch, fake_proc):
    return World(home, monkeypatch)


# ------------------------------------------------------------------------------- the tracer


async def test_plain_start_resumes_the_peer_the_last_model_choice_named(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)
    world.answering = {PEER_URL}

    lines = await world.start()

    assert [x for x in lines if x.startswith("Serving ")] == [RESUME_LINE]
    assert world.starts == [] and not [e for e in world.events if e.startswith("start")]
    # both LLMConfig builds (the probe client and the session's) are the peer's, on the peer's served name
    assert world.session_urls() == {(PEER_URL, "flash-next")}
    assert len(world.clients) >= 2
    assert (world.loop.effective_effort, world.loop.effort_rung) == ("low", "endpoint")
    assert world.last_served == "flash-next"
    assert world.overlay["active_endpoint"]["name"] == "flash-next"  # an attach persists no selection


async def test_a_second_start_the_same_way_prints_no_resume_clause(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)
    world.answering = {PEER_URL}
    assert RESUME_LINE in await world.start()
    again = await world.start()
    assert not [x for x in again if x.startswith("Serving ")]
    assert world.last_served == "flash-next"


async def test_the_session_runs_on_the_peers_endpoint_and_leaves_the_primary_provider_alone(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)
    world.answering = {PEER_URL}
    await world.start()
    repl = world.repls[-1]
    assert repl._harness.provider.base_url == PRIMARY_URL  # the primary's identity is never rewritten
    assert repl._harness.provider.default_model == PRIMARY
    assert world.probe_urls and set(world.probe_urls) == {PEER_URL}


async def test_a_config_with_no_peers_and_no_stores_starts_exactly_as_before(tmp_path, monkeypatch):
    """COMPAT-01 / R-13-32: the existing driver of tests/unit/test_launch_paths_start.py, unchanged."""
    from tests.unit.test_launch_paths_start import DECLARED_ONLY, REFUSAL, _err_text

    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    assert await _drive(tmp_path) == 1
    assert [x for x in _err_text(seen.err) if "llama-server" in x] == [REFUSAL]
    assert seen.started == []


# ------------------------------------------------------------------------------- helpers for Task 2

SERVE_ARGV = ("/usr/bin/python3", "-I", "-m", "localharness", "serve")
FLASH_ARGV = (LLAMA, "-a", "flash-next", "--port", "8081")


def _pidfile(home, text: str) -> None:
    directory = managed_server.server_dir(home)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / procfs.PIDFILE_NAME).write_text(text, encoding="utf-8")


def _flash_launch(fake_proc, home, *, pid=4242) -> None:
    """A verified launch of the `flash-next` record: its process in the fake `/proc`, its three-token
    pidfile naming :8081."""
    fake_proc.process(pid, argv=FLASH_ARGV, exe=LLAMA, start_time=777)
    _pidfile(home, f"{pid} 777 8081")


def _flash(home):
    return next(r for r in model_ops.serving_records(load(home)) if r.name == "flash-next")


def _primary_record(home):
    return model_ops.serving_records(load(home))[0]


# ------------------------------------------------------------------------------- the flag


def _runner_app():
    from localharness.cli.app import app
    return app


def test_preset_and_the_hidden_scripts_alias_hand_start_async_the_same_name(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    (tmp_path / "config.yaml").write_text("version: '1'\n", encoding="utf-8")
    seen: list[str | None] = []

    async def fake_start(*a, **kw):
        seen.append(kw.get("preset"))

    monkeypatch.setattr(start_cmd, "_start_async", fake_start)
    for argv in (["--preset", "flash-next"], ["-p", "flash-next"], ["--scripts", "flash-next"],
                 ["--preset", "flash-next", "--scripts", "flash-next"]):
        result = CliRunner().invoke(_runner_app(), ["start", *argv, "--config-dir", str(tmp_path)])
        assert result.exit_code == 0, (argv, result.output)
    assert seen == ["flash-next"] * 4
    result = CliRunner().invoke(_runner_app(), ["start", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0 and seen[-1] is None


def test_two_different_presets_name_a_conflict_and_exit_2(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    (tmp_path / "config.yaml").write_text("version: '1'\n", encoding="utf-8")
    printed: list[str] = []
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: printed.append(" ".join(map(str, a))))

    async def never(*a, **kw):
        raise AssertionError("a start that cannot tell which preset was asked for starts nothing")

    monkeypatch.setattr(start_cmd, "_start_async", never)
    result = CliRunner().invoke(
        _runner_app(), ["start", "--preset", "a", "--scripts", "b", "--config-dir", str(tmp_path)])
    assert result.exit_code == 2
    assert printed == ["--preset and --scripts name different presets (a, b)"]


def test_start_help_lists_preset_and_not_the_hidden_scripts():
    from typer.testing import CliRunner

    result = CliRunner().invoke(_runner_app(), ["start", "--help"])
    assert result.exit_code == 0
    assert "--preset" in result.output and "-p" in result.output
    assert "--scripts" not in result.output


# ------------------------------------------------------------------------------- the explicit swap


async def test_preset_swaps_with_the_same_sequence_as_slash_model(world, tmp_path):
    """`start --preset flash-next` crosses the three boundaries exactly as the REPL's `/model flash-next`
    does on the same fakes: stop the incumbent, start the peer, wait on the peer's address."""
    from tests.unit.test_repl_swap_wrapper import _build

    harness, repl, channel, agent, peer_ep, primary = _build(tmp_path)
    await asyncio.wait_for(repl._handle_slash("/model flash-next"), 5)
    repl_kinds = world.kinds()
    repl_wait = world.waits[-1]["url"]
    assert repl_kinds == ["stop", "start", "wait"] and repl_wait == peer_ep.base_url
    world.reset()

    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    await world.start(preset="flash-next")

    assert world.kinds() == repl_kinds
    assert world.waits[-1]["url"] == PEER_URL and "flash-next.gguf" in world.events[1 + (world.events[0] == "lock-enter")]


async def test_preset_says_who_is_stopped_and_what_starts_before_the_wait_and_lands_the_swap(world):
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}

    lines = await world.start(preset="flash-next")

    swap = [x for x in lines if x.startswith("stopping qwen-27b-swift-1.5…, starting flash-next (")]
    assert len(swap) == 1 and swap[0].endswith("model load can take several minutes)...")
    assert lines.index(swap[0]) < world.out_at_wait[0]  # said before the wait
    assert world.kinds() == ["stop", "start", "wait"]
    overlay = world.overlay
    assert overlay["active_endpoint"]["name"] == "flash-next"
    assert "pending_preset" not in overlay
    assert world.last_served == "flash-next"
    assert world.session_urls() == {(PEER_URL, "flash-next")}
    assert not [x for x in lines if x.startswith("Serving ")]  # an explicit swap says no attach clause


async def test_preset_holds_the_launch_lock_around_the_swap(world):
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    await world.start(preset="flash-next")
    assert world.events[0] == "lock-enter" and world.events[-1] == "lock-exit"
    assert world.events.count("lock-enter") == 1


async def test_preset_naming_the_serving_record_just_attaches(world):
    world.config(last_served=PRIMARY)
    world.answering = {PEER_URL}
    lines = await world.start(preset="flash-next")
    assert world.events == [] and world.starts == []
    assert not [x for x in lines if x.startswith("Serving ")]
    assert world.last_served == "flash-next"


async def test_preset_with_model_keeps_model_session_only(world):
    world.config(last_served=PRIMARY)
    world.answering = {PEER_URL}
    await world.start(preset="flash-next", model_override="flash-next")
    assert world.session_urls() == {(PEER_URL, "flash-next")}
    assert world.overlay.get("provider") is None  # --model never persists
    assert "flash-next" not in (world.overlay.get("provider") or {})


async def test_model_without_preset_is_checked_against_what_start_serves(world):
    world.config(active=FLASH_LOW, last_served=PRIMARY)
    world.answering = {PEER_URL}
    world.served_ids[PEER_URL] = ["flash-next"]
    assert await world.start_exit(model_override="elsewhere") == 1
    assert any("--model 'elsewhere' is not currently served at " + PEER_URL in x for x in world.err_lines)


async def test_an_unknown_preset_exits_2_with_the_list_and_launches_nothing(world):
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    assert await world.start_exit(preset="nope") == 2
    line = model_ops.UNKNOWN_PRESET_LINE.format(name="nope", names=f"flash-next, {PRIMARY}")
    assert world.err_lines == [line]
    assert world.events == [] and world.clients == []


async def test_a_withheld_preset_exits_2_with_spec_16s_line_once_and_nothing_launched_or_cleared(
        world, monkeypatch):
    from localharness.cli import workspace
    from localharness.config.trust import LAUNCH_KIND

    line = "Not applying 1 change(s) in the machine file that you have not confirmed"
    trust = workspace.MachineTrust(
        withheld={"config.yaml": frozenset({(LAUNCH_KIND, "extra_endpoints.flash-next.lifecycle")})}, line=line)
    monkeypatch.setattr(workspace, "decide_machine_trust", lambda *a, **k: trust)
    world.config(overrides={"pending_preset": "flash-next"}, last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    before = (world.home / "overrides.yaml").read_bytes()

    assert await world.start_exit(preset="flash-next") == 2

    assert world.err_lines.count(line) == 1
    assert world.events == [] and world.clients == []
    assert (world.home / "overrides.yaml").read_bytes() == before  # nothing cleared
    assert world.last_served == PRIMARY


# ------------------------------------------------------------------------------- attach variants


async def test_pending_preset_with_the_preset_answering_attaches_and_consumes_the_key(world):
    world.config(overrides={"pending_preset": "flash-next"}, last_served=PRIMARY)
    world.answering = {PEER_URL}
    lines = await world.start()
    assert world.events == []
    assert not [x for x in lines if x.startswith("Serving ")]  # the pending rung carries no resume clause
    assert "pending_preset" not in world.overlay
    assert world.last_served == "flash-next"
    assert world.session_urls() == {(PEER_URL, "flash-next")}


async def test_another_preset_answering_is_attached_with_one_clause_and_no_selection_persisted(world):
    world.config(active=FLASH_ACTIVE)
    world.answering = {PRIMARY_URL}
    world.served_ids[PRIMARY_URL] = [PRIMARY]
    lines = await world.start()
    assert [x for x in lines if x.startswith("serving ")] == [
        f"serving {PRIMARY}; your last choice was flash-next — /model flash-next swaps"]
    assert world.events == [] and world.starts == []
    assert world.overlay["active_endpoint"] == FLASH_ACTIVE  # D-15: no selection written
    assert world.last_served == PRIMARY
    assert world.session_endpoint() == (PRIMARY_URL, PRIMARY)
    assert set(world.probe_urls) == {PEER_URL, PRIMARY_URL}  # probed where it was asked, then where it answers


async def test_a_hand_run_primary_in_the_incumbents_place_is_attached_to_never_stopped_or_launched_beside(
        world, fake_proc):
    """`--preset flash-next` while a process the harness did not launch holds :8080 (D-17): the
    incumbent sentence, an attach to it, nothing stopped and nothing started."""
    fake_proc.listen(8080, 555, os.getuid())
    fake_proc.process(5000, argv=("/usr/bin/python3", "-m", "http.server", "8080"),
                      exe="/usr/bin/python3", fds={3: 555})
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    world.served_ids[PRIMARY_URL] = [PRIMARY]

    await world.start(preset="flash-next")

    assert world.events == [] and world.starts == []
    (sentence,) = [x for x in world.err_lines if "not by the harness" in x]
    assert sentence.startswith(f"{PRIMARY} is served by ") and sentence.endswith("then /model flash-next")
    assert world.session_endpoint() == (PRIMARY_URL, PRIMARY)
    assert world.last_served == PRIMARY
    assert "active_endpoint" not in world.overlay
    assert world.session_endpoint() == (PRIMARY_URL, PRIMARY)


async def test_a_boot_retry_marker_is_closed_on_attach_and_said_exactly_once(world, tmp_path, fake_proc, monkeypatch):
    from tests.unit.test_start_serving import Boot

    boot = Boot(world.home, tmp_path, fake_proc)
    boot.plant()
    (world.home / "overrides.yaml").write_text(yaml.safe_dump({"active_endpoint": FLASH_ACTIVE}), encoding="utf-8")
    speed_stats.record_last_served(world.ledger, "flash-next")
    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: units.UnitRuntimeState("inv-1", 1))
    world.window_served = 40_000
    world.answering = {PEER_URL}

    lines = await world.start()

    clause = ("retried at boot: mtp off, -c halved · "
              "extra_endpoints.flash-next.lifecycle.speculative: off keeps the full window")
    assert [x for x in lines if x.startswith("Serving ")] == [f"Serving flash-next ({clause})"]
    assert boot.retry() is not None  # confirm_launch(marker=) wrote the ledger's retry record
    again = await world.start()
    assert not [x for x in again if x.startswith("Serving ")]  # the marker was consumed: said once


async def test_with_no_boot_marker_nothing_is_confirmed(world, monkeypatch):
    world.config(active=FLASH_LOW, last_served="flash-next")
    world.answering = {PEER_URL}
    calls: list[int] = []
    monkeypatch.setattr(lifecycle, "confirm_launch", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: units.UnitRuntimeState("inv-1", 1))
    await world.start()
    assert calls == []


# ------------------------------------------------------------------------------- nothing answers


async def test_a_verified_loading_launch_of_the_selected_record_is_waited_on_with_its_own_url_hang_regression(
        world, fake_proc, monkeypatch):
    """The 30-minute hang: nothing answers, the pidfile is a live launch of the selected peer. Start waits
    on THAT record's address with `expect=record_identity`, and never on the primary's."""
    stub_liveness(monkeypatch, True)
    world.config(active=FLASH_LOW, last_served="flash-next")
    _flash_launch(fake_proc, world.home)

    lines = await world.start()

    assert "Managed llama.cpp (flash-next) is still loading — waiting..." in lines
    assert [w["url"] for w in world.waits] == [PEER_URL]
    assert world.waits[0]["kw"]["expect"] == lifecycle.record_identity(_flash(world.home).spec)
    assert PRIMARY_URL not in [w["url"] for w in world.waits]
    assert world.starts == [] and "stop" not in world.events
    assert world.last_served == "flash-next"  # a wait on a loading launch settles as an attach


async def test_another_records_live_launch_is_stopped_not_waited_on_the_30_minute_hang(world, fake_proc):
    """Nothing answers, the primary is selected, and the pidfile is a verified launch of flash-next that
    answers nowhere: the swap stops flash-next and launches the primary; no wait is ever entered on the
    primary's address while flash-next's pid is live."""
    world.config()
    _flash_launch(fake_proc, world.home)
    world.alive["flash-next"] = True

    lines = await world.start()

    assert world.kinds() == ["stop", "start", "wait"]
    assert "/m/a.gguf" in world.events[2] and "flash-next.gguf" not in world.events[2]
    assert [w["url"] for w in world.waits] == [PRIMARY_URL]
    assert all(not w["flash_alive"] for w in world.waits)
    assert all("expect" not in w["kw"] for w in world.waits)  # never the "still loading" wait
    assert [x for x in lines if x.startswith("stopping flash-next…, starting qwen-27b-swift-1.5 (")]
    flash = _flash(world.home).spec
    assert world.stops[0]["expect"] == lifecycle.strategy_for(flash).launch_identity(flash)  # the strategy's own stop
    assert world.last_served == PRIMARY and "active_endpoint" not in world.overlay


async def test_nothing_answering_and_nothing_live_launches_the_selected_peer_and_lands_it(world, fake_proc):
    world.config(active=FLASH_LOW, last_served=PRIMARY)

    lines = await world.start()

    assert world.kinds() == ["start", "wait"] and "flash-next.gguf" in world.events[1]
    assert [x for x in lines if x.startswith("Managed llama.cpp (flash-next) is not running — starting it (")]
    assert [w["url"] for w in world.waits] == [PEER_URL]
    assert world.overlay["active_endpoint"]["name"] == "flash-next"
    assert world.last_served == "flash-next"
    assert world.session_urls() == {(PEER_URL, "flash-next")}


async def test_a_retried_peer_launch_names_the_peers_own_undo_field(tmp_path, monkeypatch):
    from localharness.provider.tuning import Tuned

    out: list[str] = []
    monkeypatch.setattr(start_cmd.console, "print", lambda *a, **k: out.append(" ".join(map(str, a))))
    monkeypatch.setattr(managed_server, "server_pid", lambda _d: None)
    plan = SimpleNamespace(tuned=Tuned(), shape="retry", cause="boot", dropped=("draft-mtp",),
                           remembered_at=None, report=None, full_fit=None, dropped_parser_fields=())

    class Strategy:
        plans_launch = True

        def liveness(self, spec, config_dir, **kw):
            return lifecycle.Liveness(False)

        def launch_identity(self, spec):
            return None

        def recovery_config_dir(self, config_dir):
            return config_dir

        async def activate(self, spec, config_dir, base_url, *, on_plan=None, **kw):
            on_plan(plan)
            return lifecycle.LiveEndpoint(base_url=base_url, served_models=["m"], handle=1)

    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: Strategy())
    spec = peer("flash-next", served_name="flash-next")["lifecycle"]
    from localharness.config.models import ManagedServerConfig
    srv = ManagedServerConfig(**spec)
    await start_cmd._start_managed_server(srv, tmp_path, PEER_URL, {}, peer="flash-next")
    (launch_line,) = [x for x in out if "is not running" in x]
    assert "extra_endpoints.flash-next.lifecycle.speculative: off" in launch_line
    assert "server.speculative" not in launch_line.replace("lifecycle.speculative", "")
    assert launch_line.startswith("Managed llama.cpp (flash-next) is not running — starting it (")


async def test_the_autostart_units_server_is_waited_for_then_attached_never_launched_beside(
        world, fake_proc, monkeypatch):
    cgroup = (f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/"
              f"{procfs.HARNESS_UNIT}")
    fake_proc.process(4242, argv=SERVE_ARGV, exe="/usr/bin/python3", start_time=777, cgroup=cgroup)
    _pidfile(world.home, "4242 777")

    class State:
        main_pid = 4242
        timeout_start_seconds = 90
        invocation_id = None
        n_restarts = 0

    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: State())
    slept: list[float] = []

    async def fast_sleep(seconds):
        slept.append(seconds)
        world.answering.add(PEER_URL)  # the unit's server finishes loading while start waits

    monkeypatch.setattr(start_serving.asyncio, "sleep", fast_sleep)
    world.config(active=FLASH_LOW, last_served="flash-next")

    lines = await world.start()

    assert "autostart is starting flash-next (pid 4242) — waiting up to 90s" in lines
    assert slept and world.events == [] and world.starts == []
    assert world.session_urls() == {(PEER_URL, "flash-next")}
    assert world.last_served == "flash-next"


async def test_an_autostart_server_that_never_answers_is_named_and_left_alone_exit_1(
        world, fake_proc, monkeypatch):
    cgroup = (f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/"
              f"{procfs.HARNESS_UNIT}")
    fake_proc.process(4242, argv=SERVE_ARGV, exe="/usr/bin/python3", start_time=777, cgroup=cgroup)
    _pidfile(world.home, "4242 777")

    class State:
        main_pid = 4242
        timeout_start_seconds = 1

    monkeypatch.setattr(start_serving, "unit_runtime_state", lambda: State())

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(start_serving.asyncio, "sleep", no_sleep)
    killed: list[tuple] = []
    monkeypatch.setattr(os, "kill", lambda *a: killed.append(a))
    monkeypatch.setattr(os, "killpg", lambda *a: killed.append(a))
    world.config(active=FLASH_LOW, last_served="flash-next")

    assert await world.start_exit() == 1

    assert "autostart's server (pid 4242) did not answer within 1s — not touched" in world.err_lines
    assert world.events == [] and killed == []


async def test_a_foreign_holder_of_the_selected_port_ends_the_start_with_the_incumbent_sentence(
        world, fake_proc):
    fake_proc.listen(8081, 556, os.getuid())
    fake_proc.process(5001, argv=("/usr/bin/python3", "-m", "http.server", "8081"),
                      exe="/usr/bin/python3", fds={3: 556})
    world.config(active=FLASH_LOW, last_served="flash-next")

    assert await world.start_exit() == 1

    (sentence,) = [x for x in world.err_lines if "not by the harness" in x]
    assert sentence.startswith("flash-next is served by ")
    assert world.events == [] and world.starts == []


# ------------------------------------------------------------------------------- the loading gate (D-18)


def _running(spec, home) -> bool:
    return start_cmd._managed_server_running(lifecycle.strategy_for(spec), spec, home)


def test_a_live_launch_of_another_record_is_not_this_record_already_serving(world, fake_proc, monkeypatch):
    stub_liveness(monkeypatch, True)
    world.config()
    _flash_launch(fake_proc, world.home)
    assert _running(_primary_record(world.home).spec, world.home) is False  # the pidfile is flash-next's
    assert _running(_flash(world.home).spec, world.home) is True


def test_a_live_pidfile_that_cannot_be_read_keeps_todays_answer(world, monkeypatch):
    stub_liveness(monkeypatch, True)
    world.config()
    assert _running(_primary_record(world.home).spec, world.home) is True  # alive, no pidfile to read


def test_a_dead_strategy_is_not_running_whatever_the_pidfile_says(world, fake_proc, monkeypatch):
    stub_liveness(monkeypatch, False)
    world.config()
    _flash_launch(fake_proc, world.home)
    assert _running(_flash(world.home).spec, world.home) is False


def test_a_strategy_with_no_launch_identity_and_a_managed_lm_studio_keep_todays_answer(world, fake_proc):
    from localharness.config.models import ManagedServerConfig

    world.config()
    _flash_launch(fake_proc, world.home)
    bare = SimpleNamespace(liveness=lambda spec, d: lifecycle.Liveness(True))  # no launch_identity at all
    assert start_cmd._managed_server_running(bare, _primary_record(world.home).spec, world.home) is True
    lms = ManagedServerConfig(runtime="lmstudio", model="qwen", port=1234)
    assert start_cmd._managed_server_running(bare, lms, world.home) is True


# ------------------------------------------------------------------------------- the console is a channel (SEC-04)

PLANTED_LOG = (
    "request: Authorization: Bearer sk-planted\n"
    "main: api_keys: ****abcd\n"
    "GET http://u:p@127.0.0.1:8080/v1 200\n"
    f"the key {PROVIDER_KEY} was here\n"
)
SECRETS = ("sk-planted", "abcd", "u:p", PROVIDER_KEY)


async def test_a_start_time_launch_that_dies_over_a_planted_log_shows_only_a_scrubbed_tail_in_the_error_and_on_the_console(
        world, monkeypatch):
    """The REAL `wait_ready` over a planted serve.log: the RuntimeError `_start_managed_server` raises and
    the line `start` prints carry no secret and end with the log reference; serve.log keeps the raw lines."""
    import re

    monkeypatch.setattr(managed_server, "wait_ready", world.real_wait_ready)
    world.config()  # a keyed primary: require_api_key: true, the provider key is the session's
    log = managed_server.log_path(world.home)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(PLANTED_LOG, encoding="utf-8")
    raised: list[BaseException] = []
    real = start_cmd._start_managed_server

    async def spy(*a, **kw):
        try:
            return await real(*a, **kw)
        except BaseException as exc:  # noqa: BLE001 — recorded, then re-raised for start to word
            raised.append(exc)
            raise

    monkeypatch.setattr(start_cmd, "_start_managed_server", spy)

    assert await world.start_exit() == 1

    (exc,) = raised
    text = str(exc)
    assert isinstance(exc, RuntimeError)
    for secret in SECRETS:
        assert secret not in text, secret
    assert re.search(re.escape(f"(see log {log}, ref ") + r"[0-9a-f]{8}\)$", text), text
    (failed,) = [x for x in world.err_lines if "failed to start:" in x]
    assert "managed llama.cpp failed to start:" in failed
    for secret in SECRETS:
        assert secret not in failed, secret
    raw = log.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret in raw, secret  # the raw lines stay in serve.log


async def test_the_channel_redacts_the_start_failure_line_even_when_the_error_text_is_not_scrubbed(
        world, monkeypatch):
    """The channel's own second pass: a launch error that still quotes a configured secret is masked."""
    world.config()
    world.wait_raises[PRIMARY_URL] = RuntimeError(f"llama.cpp exited. tail: Bearer sk-planted {PROVIDER_KEY}")
    assert await world.start_exit() == 1
    (failed,) = [x for x in world.err_lines if "failed to start:" in x]
    assert "sk-planted" not in failed and PROVIDER_KEY not in failed


async def test_a_failed_explicit_swap_redacts_the_planted_token_and_key_in_the_line_and_the_error(
        world, monkeypatch):
    world.config(last_served=PRIMARY)
    world.answering = {PRIMARY_URL}
    world.wait_raises[PEER_URL] = RuntimeError(
        f"the model server exited. Log tail:\nBearer sk-planted ... {PROVIDER_KEY} ... done")
    outcomes: list = []
    real = start_serving.run_swap

    async def spy(*a, **kw):
        out = await real(*a, **kw)
        outcomes.append(out)
        return out

    monkeypatch.setattr(start_serving, "run_swap", spy)

    assert await world.start_exit(preset="flash-next") == 1

    (outcome,) = outcomes
    assert outcome.error is not None and outcome.stage == "launch"
    failed = [x for x in world.err_lines if x.startswith("Launch of flash-next failed:")]
    assert len(failed) == 1
    for secret in ("sk-planted", PROVIDER_KEY):
        assert secret not in failed[0] and secret not in str(outcome.error)
        assert not any(secret in line for line in world.err_lines)
    assert world.last_served == PRIMARY and "active_endpoint" not in world.overlay  # nothing landed
