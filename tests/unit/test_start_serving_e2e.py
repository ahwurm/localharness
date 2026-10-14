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

        async def keep(repl):
            self.agents.append(repl._agent)
            self.repls.append(repl)

        _stub_start_boundaries(home, monkeypatch, repl_run=keep)
        self.out = _capture_start_console(monkeypatch)
        monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: self.err.append((a, k)))
        monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: None)
        monkeypatch.setattr(start_cmd, "_props_model_path", lambda *a, **k: None)
        stub_liveness(monkeypatch, False)
        monkeypatch.setattr(lifecycle, "GPU_FREE_SETTLE_SECONDS", 0.0)
        monkeypatch.setattr(managed_server, "list_cached_models", lambda: [])
        monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(6 * GIB, 32 * GIB, "nvidia-smi"))
        monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: 18 * GIB)
        monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "24576, 32768\n")
        monkeypatch.setattr(
            autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})
        monkeypatch.setattr(
            "localharness.agent.context.probe_window",
            lambda base_url, model, provider_type=None, **kw: WindowProbe(65536, 262144))
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
            self.waits.append({"url": base_url, "kw": kw})
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
        seed(self.home, **kw)

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
