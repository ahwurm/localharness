"""Shared fixtures for the tests of `lifecycle.UnitStrategy` (spec 20 §9.4, R-14-37).

An importable helper module (its name does not start with `test_`, so pytest collects nothing from it):

    from tests.unit.unit_strategy_support import unit_world, ScriptedManager, FakeServe

- `ScriptedManager`: `units_support.FakeManager` for the autostart unit's runner, with a `show` answer that
  changes after the unit's `restart` (successive scripted steps, the last one repeating) and a hook that
  runs when `restart` is answered.
- `FakeServe`: the unit's `serve`, as a thread. It waits until it is allowed to, takes the launch lock on a
  SECOND file descriptor (flock conflicts exactly like a second process would, so it succeeds only while the
  lock is lent), writes the pidfile, releases, and from then on presents the record's own llama-server argv
  as the unit's main process in the fake `/proc`.
- `unit_world(tmp_path, monkeypatch, fake_proc)`: a global config dir with a llama.cpp primary `qwen` and
  peers `flash-next` and `qwen-27b` (real GGUF files), the unit's file written under a redirected
  `XDG_CONFIG_HOME`, the scripted manager installed as the unit runner, sensing faked, and the test process
  registered in the fake `/proc`.

Nothing here starts a process, runs a service manager, dials a port or reads the host's `/proc`: the runner is
faked, `/proc` is `procfs_support.fake_proc`, the HTTP wait answers through `server._TRANSPORT`, every file is
under tmp. Later plans (14-10, 14-13, 14-14) import this module instead of copying its fixtures.
"""
from __future__ import annotations

import errno
import fcntl
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
import yaml

from localharness.agent.context import WindowProbe
from localharness.config.loader import ConfigLoader
from localharness.config.models import ManagedServerConfig
from localharness.provider import autosense, lifecycle, procfs, server, units
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import FakeProc
from tests.unit.test_tuning_derive import FLASH_NEXT, SWIFT
from tests.unit.units_support import FakeManager

GIB = 2**30
INTERPRETER = "/opt/lh/bin/python"
UNIT = procfs.HARNESS_UNIT
PRIMARY = "qwen"
PRIMARY_URL = "http://127.0.0.1:8080/v1"
PEER_URLS = {"flash-next": "http://127.0.0.1:8081/v1", "qwen-27b": "http://127.0.0.1:8082/v1"}
UNIT_CGROUP = (f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{UNIT}")
SERVE_TOKENS = (INTERPRETER, "-I", "-m", "localharness", "serve", "--config-dir")
RESTART_ARGV = ("systemctl", "--user", "restart", UNIT)
RESET_ARGV = ("systemctl", "--user", "reset-failed", UNIT)
LOCK_BUSY = (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES)


# ------------------------------------------------------------------------------------------ the manager


class ScriptedManager(FakeManager):
    """The unit runner for a test. Before the `restart` verb is answered, `show` answers `base`. After it,
    `show` answers `base` updated by the next scripted step: `after_restart({"NRestarts": 1}, {...})` queues
    the steps, each used for ONE `show`, the last repeating forever (`None` as a step is the runner's None:
    the manager could not be asked). `on_restart` callables run when `restart` is answered."""

    def __init__(self, fragment: str) -> None:
        super().__init__()
        self.base: dict[str, object] = {
            "FragmentPath": fragment, "DropInPaths": "", "ActiveState": "active", "UnitFileState": "enabled",
            "NRestarts": 0, "ExecMainStatus": 0, "MainPID": 0, "InvocationID": "inv-1", "Result": "success"}
        self.steps: list[dict | None] = []
        self.on_restart: list[Callable[[], None]] = []
        self.restarted = False
        self.show_count = 0
        self.shown: dict[str, object] = dict(self.base)
        self._mine = threading.Lock()
        self.answer_prefix("systemctl", "--user", "reset-failed", out="")
        self.answer_prefix("systemctl", "--user", "restart", out="")

    def after_restart(self, *steps: dict | None) -> None:
        self.steps = list(steps)

    def journal(self, text: str, *, code: int = 0) -> None:
        self.answer_prefix("journalctl", out=text, code=code)

    def __call__(self, argv, *, timeout, cap, **kw):
        key = tuple(argv)
        if key[:3] == ("systemctl", "--user", "show"):
            self.calls.append(key)
            self.limits.append((timeout, cap))
            return self._show()
        done = super().__call__(argv, timeout=timeout, cap=cap, **kw)
        if key[:3] == ("systemctl", "--user", "restart") and done is not None and done.code == 0:
            self.restarted = True
            for hook in list(self.on_restart):
                hook()
        return done

    def _show(self):
        with self._mine:
            fields = dict(self.base)
            if self.restarted and self.steps:
                step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
                if step is None:
                    self.show_count += 1
                    return None
                fields.update(step)
            self.shown = fields
            self.show_count += 1
        text = "".join(f"{k}={v}\n" for k, v in fields.items())
        return units.ManagerResult(0, text, "")

    @property
    def verbs(self) -> list[tuple[str, ...]]:
        """Every non-`show`, non-journal call as `(verb, unit)`, in order."""
        return [(c[2], *c[3:]) for c in self.calls if c[:2] == ("systemctl", "--user") and c[2] != "show"]

    def shows(self) -> int:
        return sum(1 for c in self.calls if c[:3] == ("systemctl", "--user", "show"))


# ---------------------------------------------------------------------------------------- fake /proc edits


def install_process(proc: FakeProc, pid: int, *, argv, exe: str, start_time: int, cgroup: str | None,
                    comm: str = "llama-server") -> None:
    """Make `pid` appear in the fake `/proc` all at once (staged beside the tree, then renamed in), so a
    reader on another thread never sees a half-written directory."""
    stage_root = proc.root.parent / f"stage-{pid}-{time.monotonic_ns()}"
    stage_root.mkdir()
    FakeProc(stage_root).process(pid, argv=argv, exe=exe, start_time=start_time, cgroup=cgroup, comm=comm)
    target = proc.root / str(pid)
    if target.exists():
        vanish(proc, pid)
    os.rename(stage_root / str(pid), target)
    shutil.rmtree(stage_root, ignore_errors=True)


def change_argv(proc: FakeProc, pid: int, argv) -> None:
    """The process `exec`s: its command line is replaced in one atomic step (same pid, same start time)."""
    target = proc.root / str(pid) / "cmdline"
    fresh = target.with_name("cmdline.new")
    fresh.write_bytes(b"".join(a.encode() + b"\0" for a in argv))
    os.replace(fresh, target)


def vanish(proc: FakeProc, pid: int) -> None:
    """The process is gone: its directory leaves the tree in one atomic rename."""
    gone = proc.root.parent / f"gone-{pid}-{time.monotonic_ns()}"
    os.rename(proc.root / str(pid), gone)
    shutil.rmtree(gone, ignore_errors=True)


# --------------------------------------------------------------------------------------------- the serve


class FakeServe:
    """The unit's `serve` for one MainPID, as a thread (spec 20 §9.3): once `when()` is true it appears as the
    unit's main process running `python -I -m localharness serve`, takes the launch lock on its own fd (only
    possible while the lock is lent), writes the pidfile `"<pid> <start> <port>"`, releases the lock, and
    presents the record's llama-server argv. `took_lock` is set when it held the lock; `error` carries
    anything the thread raised."""

    def __init__(self, g: Path, proc: FakeProc, spec: ManagedServerConfig, *, pid: int = 4242,
                 start_time: int = 5000, when: Callable[[], bool] = lambda: True, exec_argv=None,
                 choose: "Callable[[], ManagedServerConfig] | None" = None) -> None:
        self.g, self.proc, self.spec = g, proc, spec
        self.pid, self.start_time, self.when = pid, start_time, when
        self.exec_argv = exec_argv
        self.choose = choose
        self.chosen: ManagedServerConfig | None = None
        self.took_lock = threading.Event()
        self.recorded = threading.Event()
        self.error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"fake-serve-{pid}")

    def start(self) -> "FakeServe":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def join(self, timeout: float = 10.0) -> None:
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            while not self.when():
                if self._stop.is_set():
                    return
                time.sleep(0.002)
            self.chosen = self.choose() if self.choose is not None else self.spec
            spec = self.chosen
            install_process(self.proc, self.pid, argv=(*SERVE_TOKENS, str(self.g)), exe=str(spec.binary),
                            start_time=self.start_time, cgroup=UNIT_CGROUP)
            path = server.lock_path(self.g)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in LOCK_BUSY:
                            raise
                        if self._stop.is_set():
                            return
                        time.sleep(0.002)
                self.took_lock.set()
                server.pid_path(self.g).write_text(f"{self.pid} {self.start_time} {spec.port}", encoding="utf-8")
                self.recorded.set()
            finally:
                os.close(fd)
            change_argv(self.proc, self.pid, self.exec_argv or self.record_argv(spec))
        except BaseException as exc:  # noqa: BLE001 - surfaced by the test through `error`
            self.error = exc

    @staticmethod
    def record_argv(spec: ManagedServerConfig) -> tuple[str, ...]:
        return tuple(server.serve_command(spec, keyed=False, loopback=True, key_file_ok=False,
                                          context=None, cache_ram=None))


# ---------------------------------------------------------------------------------------------- the world


class Clock:
    """A fake `time` for `lifecycle`: every `monotonic()` call advances it by `step` until it reaches
    `freeze_at` (then it stands still), so a test can say "this much time passed" without sleeping."""

    def __init__(self, *, step: float = 100.0, freeze_at: float | None = None, start: float = 1000.0) -> None:
        self.now, self.step, self.freeze_at = start, step, freeze_at

    def monotonic(self) -> float:
        value = self.now
        if self.freeze_at is None or self.now < self.freeze_at:
            self.now += self.step
        return value

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class UnitWorld:
    tmp: Path
    g: Path
    manager: ScriptedManager
    proc: FakeProc
    monkeypatch: pytest.MonkeyPatch
    state: units.UnitRuntimeState
    harness: object
    requests: list[str] = field(default_factory=list)
    servers: list[FakeServe] = field(default_factory=list)

    # ------------------------------------------------------------------ records

    @property
    def primary(self) -> ManagedServerConfig:
        return self.harness.server

    def peer(self, name: str) -> ManagedServerConfig:
        return next(e.lifecycle for e in self.harness.extra_endpoints if e.name == name)

    def url(self, name: str | None = None) -> str:
        return PRIMARY_URL if name is None else PEER_URLS[name]

    def strategy(self, spec: ManagedServerConfig) -> "lifecycle.UnitStrategy":
        got = lifecycle.strategy_for(spec, unit=self.state)
        assert isinstance(got, lifecycle.UnitStrategy)
        return got

    # ------------------------------------------------------------------ the unit's side

    def serve(self, spec: ManagedServerConfig, **kw) -> FakeServe:
        """A `FakeServe` for `spec` that starts when the unit is restarted (`when` overrides)."""
        kw.setdefault("when", lambda: self.manager.restarted)
        made = FakeServe(self.g, self.proc, spec, **kw)
        self.servers.append(made)
        return made

    def serve_on_restart(self, spec: ManagedServerConfig, *, pid: int = 4242, main_pid: bool = True,
                         **kw) -> FakeServe:
        """Start a `FakeServe` when `restart` is answered, and make the unit report its pid as MainPID."""
        made = self.serve(spec, pid=pid, **kw)
        self.manager.on_restart.append(made.start)
        if main_pid:
            self.manager.base["MainPID"] = pid
        return made

    def stop_servers(self) -> None:
        for made in self.servers:
            made.stop()
            assert made.error is None, made.error

    # ------------------------------------------------------------------ the caller's lock

    def third_party_can_lock(self) -> bool:
        """Can another file description take the launch lock right now (is it free)?"""
        fd = os.open(server.lock_path(self.g), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in LOCK_BUSY:
                return False
            raise
        finally:
            os.close(fd)
        return True

    # ------------------------------------------------------------------ the HTTP wait

    def http(self, *, answer_after: int = 2, served: str = PRIMARY, hook: Callable[[int], None] | None = None,
             needs_pidfile: bool = True, gate: Callable[[], bool] | None = None) -> None:
        """`/models` answers 503 for the first `answer_after - 1` polls and a well-formed 200 after; `hook(n)`
        runs on poll n (before the answer) and may raise. A server only answers once the unit's `serve` has
        recorded its pidfile (`needs_pidfile`), as the real one can only be running after that; `gate()`, when
        given, must also be true."""
        seen = {"n": 0}

        def answer(request: httpx.Request) -> httpx.Response:
            seen["n"] += 1
            self.requests.append(str(request.url))
            if hook is not None:
                hook(seen["n"])
            if (seen["n"] < answer_after or (needs_pidfile and not server.pid_path(self.g).exists())
                    or (gate is not None and not gate())):
                return httpx.Response(503)
            return httpx.Response(200, json={"data": [{"id": served}]})

        self.monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(answer))

    def fast_polls(self, seconds: float = 0.005) -> None:
        """`wait_ready` polls every `seconds` instead of every three."""
        real = server.wait_ready

        async def quick(*args, **kw):
            kw.setdefault("poll_seconds", seconds)
            return await real(*args, **kw)

        self.monkeypatch.setattr(server, "wait_ready", quick)

    def fast_loan(self, seconds: float = 0.005) -> None:
        self.monkeypatch.setattr(lifecycle, "LEND_POLL_SECONDS", seconds)

    def fake_clock(self, **kw) -> Clock:
        clock = Clock(**kw)
        self.monkeypatch.setattr(lifecycle, "time", clock)
        return clock

    def probe(self, served: int | None = 65536, trained: int | None = 262144) -> list[tuple]:
        """The window probe answers `(served, trained)`; the list records its calls."""
        seen: list[tuple] = []

        def fake(base_url, model, provider_type=None, *, api_key=None):
            seen.append((base_url, model, provider_type, api_key))
            return WindowProbe(served, trained)

        self.monkeypatch.setattr("localharness.agent.context.probe_window", fake)
        return seen

    def ready(self) -> None:
        """Everything a successful activate needs besides a `FakeServe`: fast polls and loan, an HTTP answer."""
        self.fast_polls()
        self.fast_loan()
        self.http()
        self.probe()

    def target(self) -> str | None:
        return units.read_target(self.g)


def patch_sensing(monkeypatch: pytest.MonkeyPatch) -> None:
    """What `sense_launch` reads from the machine, faked: 40 GiB free of 48, 60 of 64 GiB of RAM, 18 GiB of weights."""
    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})
    monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: 18 * GIB)
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "24576, 32768\n", raising=False)


def _peer(name: str, port: int, model: Path) -> dict:
    return {"name": name, "base_url": PEER_URLS[name], "provider_type": "llamacpp", "gpu": True,
            "lifecycle": {"gpu": True, "runtime": "llamacpp", "binary": "/x/llama-server",
                          "model": str(model), "port": port}}


def unit_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_proc: FakeProc, *,
               primary_extra: dict | None = None, primary_header: dict | None = None) -> UnitWorld:
    """The world a `UnitStrategy` runs in (see the module docstring). `primary_extra` adds keys to the primary
    `server:` record (a `context:` for the no-boot-retry cases); `primary_header` is its GGUF header (default
    SWIFT, which derives MTP drafting and so has a boot retry)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(sys, "executable", INTERPRETER)
    g = Path(os.environ["LOCALHARNESS_HOME"])
    models = tmp_path / "models"
    models.mkdir()
    qwen = write_gguf(models / "qwen.gguf", primary_header or SWIFT)
    flash = write_gguf(models / "flash-next.gguf", FLASH_NEXT)
    big = write_gguf(models / "qwen-27b.gguf", FLASH_NEXT)
    record = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": str(qwen), "served_name": PRIMARY,
              "port": 8080, **(primary_extra or {})}
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1",
        "provider": {"provider_type": "llamacpp", "base_url": PRIMARY_URL, "default_model": PRIMARY,
                     "available_models": [PRIMARY]},
        "org": {"default_model": PRIMARY, "audit_log_path": str(g / "audit.jsonl")},
        "extra_endpoints": [_peer("flash-next", 8081, flash), _peer("qwen-27b", 8082, big)],
        "server": record}), encoding="utf-8")
    (g / "overrides.yaml").unlink(missing_ok=True)
    patch_sensing(monkeypatch)

    body = units.render_harness_unit(sys.executable, g, {"PATH": str(tmp_path / "bin")},
                                     home=units.harness_home(), version="0.19.0")
    units.write_harness_unit(body)
    manager = ScriptedManager(str(units.harness_unit_path()))
    monkeypatch.setattr(units, "_run_manager", manager)
    fake_proc.process(os.getpid(), argv=("python", "-m", "pytest"), exe=INTERPRETER, start_time=111,
                      cgroup=None, comm="python")
    state = units.own_unit(
        units.UnitRuntimeState(None, 0, fragment_path=str(units.harness_unit_path()), active_state="active"), g)
    assert state is not None and state.own
    harness = ConfigLoader(config_dir=g).load_harness()
    return UnitWorld(tmp_path, g, manager, fake_proc, monkeypatch, state, harness)

