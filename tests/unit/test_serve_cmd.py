"""`localharness serve` — the autostart unit's body (spec 20 §9.3, D-14..D-17, AUTO-07, AUTO-08).

`serve_world` builds the one fixture world every `serve` test shares, as a plain importable function (the
fixture `world` wraps it; later plans import it and add peers): a global dir with a llama.cpp `server:`
record whose binary is a real executable file under tmp, a machine record made by one
`decide_machine_trust(ask=False)`, the harness's own unit file written under a redirected
`XDG_CONFIG_HOME` and its `autostart.env` entry adopted, a fake service manager that answers the state
`show` for that unit, and `server.exec_server` replaced by a recorder that raises a sentinel.

Nothing here starts a process, runs a service manager, reads the host's `/proc` for a foreign pid or dials a
port: the manager is `units_support.FakeManager`, `/proc` is `procfs_support.fake_proc`, the exec is the
recorder, and every file is under tmp.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import serve_cmd
from localharness.cli.app import app
from localharness.cli.workspace import decide_machine_trust
from localharness.config import paths as config_paths
from localharness.config import trust
from localharness.provider import autosense, lifecycle, server, speed_stats, units
from localharness.provider.autosense import FreeMemory
from tests.unit.gguf_support import write_gguf
from tests.unit.procfs_support import fake_proc  # noqa: F401 — the fixture
from tests.unit.test_server_pidfile_start_time import pidfile, proc_says
from tests.unit.test_tuning_derive import SWIFT
from tests.unit.trust_support import never_asked
from tests.unit.units_support import FakeManager

GIB = 2**30
INTERPRETER = "/opt/lh/bin/python"
PRIMARY = "swift"
PRIMARY_URL = "http://127.0.0.1:8080/v1"
UNIT = units.HARNESS_UNIT


class _Exec(Exception):
    """What the exec recorder raises in place of replacing the process."""


@dataclass
class ServeWorld:
    tmp: Path
    g: Path
    home: Path
    bindir: Path
    binary: Path
    model: Path
    captured: dict
    manager: FakeManager
    monkeypatch: pytest.MonkeyPatch
    runtime: str = "llamacpp"
    execs: list = field(default_factory=list)
    err: list = field(default_factory=list)
    lock_free_at_exec: list = field(default_factory=list)
    pidfile_at_exec: list = field(default_factory=list)

    # ------------------------------------------------------------------ running

    def run(self, *args: str, config_dir: str | None = None):
        return CliRunner().invoke(app, ["serve", "--config-dir", config_dir or str(self.g), *args])

    # ------------------------------------------------------------------ the world's parts

    @property
    def config_path(self) -> Path:
        return self.g / "config.yaml"

    def edit_config(self, change) -> None:
        data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        change(data)
        self.config_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    def set_overrides(self, data: dict) -> None:
        (self.g / "overrides.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")

    def rerecord(self) -> None:
        """Re-record the machine after a config edit, as a terminal Yes would, and adopt the unit's entry."""
        trust.record_machine(self.g, trust.machine_snapshot(self.g))

    def write_unit(self, captured: dict | None = None, *, adopt: bool = False) -> str:
        body = units.render_harness_unit(
            sys.executable, self.g, self.captured if captured is None else captured,
            home=units.harness_home(), version="0.19.0")
        units.write_harness_unit(body)
        if adopt:
            trust.adopt_entries(self.g, trust.autostart_entries(self.g))
        return body

    def manager_says(self, *, n_restarts: int = 0, main_pid: int | None = None, fragment: str | None = None,
                     drop_ins: str = "", invocation: str = "inv-1") -> None:
        text = "\n".join([
            f"FragmentPath={units.harness_unit_path() if fragment is None else fragment}",
            f"DropInPaths={drop_ins}", "ActiveState=activating", "UnitFileState=enabled",
            f"NRestarts={n_restarts}", "ExecMainStatus=0",
            f"MainPID={os.getpid() if main_pid is None else main_pid}",
            f"InvocationID={invocation}", "Result=success", ""])
        self.manager.answer_prefix("systemctl", "--user", "show", out=text)

    def manager_silent(self) -> None:
        self.manager.answer_prefix("systemctl", "--user", "show", out=None)

    # ------------------------------------------------------------------ reading what a run left

    @property
    def pid_file(self) -> Path:
        return server.pid_path(self.g)

    @property
    def marker_file(self) -> Path:
        return server.server_dir(self.g) / units.BOOT_MARKER_NAME

    def marker(self):
        return units.read_marker(self.g)

    @property
    def record(self) -> str:
        return speed_stats.speed_key(self.runtime, PRIMARY if self.runtime == "llamacpp" else str(self.model))

    def said(self) -> str:
        return "\n".join(self.err)

    def untouched_by(self, result, code: int) -> None:
        """A refusal or a nothing-to-do: the exit code, no exec, no pidfile, no marker."""
        assert result.exit_code == code, (result.exit_code, self.err, result.exception)
        assert not isinstance(result.exception, _Exec) and self.execs == []
        assert not self.pid_file.exists() and not self.marker_file.exists()


def llama_peer(name: str, port: int, *, binary: str = "/x/llama-server") -> dict:
    return {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "provider_type": "llamacpp", "gpu": True,
            "lifecycle": {"gpu": True, "runtime": "llamacpp", "binary": binary,
                          "model": f"/m/{name}.gguf", "port": port}}


def docker_peer(name: str = "dock", port: int = 8082) -> dict:
    return {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "provider_type": "vllm", "gpu": True,
            "lifecycle": {"gpu": True, "runtime": "vllm", "launch": "docker", "docker_image": "vllm/vllm-openai:x",
                          "model": "org/model", "port": port}}


def ollama_peer(name: str = "oll", port: int = 11434) -> dict:
    return {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "provider_type": "ollama", "gpu": True,
            "lifecycle": {"gpu": True, "runtime": "ollama", "model": "llama3:8b", "port": port}}


def attach_peer(name: str = "attached", port: int = 1234) -> dict:
    return {"name": name, "base_url": f"http://127.0.0.1:{port}/v1", "provider_type": "lmstudio"}


def serve_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, peers=(), runtime: str = "llamacpp",
                binary_name: str | None = None, captured: dict | None = None, provider_key: str | None = None,
                server_extra: dict | None = None, create_binary: bool = True) -> ServeWorld:
    """The world `serve` runs in (see the module docstring). ``runtime`` is "llamacpp" or "vllm" (a binary
    record); ``peers`` are raw `extra_endpoints` dicts (`llama_peer`, `docker_peer`, ...); ``captured`` is
    the unit's `# env` block (PATH is the world's bin dir by default); ``provider_key`` makes the primary
    require that key; ``create_binary=False`` leaves the bin dir without the record's binary."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("INVOCATION_ID", "inv-1")
    monkeypatch.setattr(sys, "executable", INTERPRETER)
    g = Path(os.environ["LOCALHARNESS_HOME"])
    monkeypatch.setenv("LOCALHARNESS_DIR", str(g))  # restored at teardown, whatever serve sets it to
    never_asked(monkeypatch)

    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / (binary_name or ("llama-server" if runtime == "llamacpp" else "vllm"))
    if create_binary:
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
    if runtime == "llamacpp":
        model = write_gguf(tmp_path / "model.gguf", SWIFT)
        record = {"runtime": "llamacpp", "binary": binary_name or str(binary), "model": str(model),
                  "served_name": PRIMARY, "port": 8080}
    else:
        model = tmp_path / "ckpt"
        model.mkdir()
        (model / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
        record = {"runtime": "vllm", "launch": "binary", "binary": binary_name or str(binary),
                  "model": str(model), "port": 8000}
    if provider_key is not None:
        record["require_api_key"] = True
    record.update(server_extra or {})
    provider = {"provider_type": "llamacpp" if runtime == "llamacpp" else "vllm",
                "base_url": PRIMARY_URL, "default_model": PRIMARY, "available_models": [PRIMARY]}
    if provider_key is not None:
        provider["api_key"] = provider_key
    (g / "config.yaml").write_text(yaml.safe_dump({
        "version": "1", "provider": provider,
        "org": {"default_model": PRIMARY, "audit_log_path": str(g / "audit.jsonl")},
        "extra_endpoints": list(peers), "server": record}), encoding="utf-8")
    (g / "overrides.yaml").unlink(missing_ok=True)

    monkeypatch.setattr(autosense, "free_memory", lambda ratios: FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})
    monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: 18 * GIB)
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "24576, 32768\n", raising=False)

    manager = FakeManager()
    monkeypatch.setattr(units, "_run_manager", manager)
    world = ServeWorld(tmp_path, g, home, bindir, binary, Path(model), captured or {"PATH": str(bindir)},
                       manager, monkeypatch, runtime)

    decide_machine_trust(g, ask=False)  # the machine record: one adoption of the disk as it is
    world.write_unit(adopt=True)
    world.manager_says()

    def exec_recorder(path, argv, env):
        world.execs.append((path, list(argv), dict(env)))
        world.pidfile_at_exec.append(world.pid_file.read_text(encoding="utf-8") if world.pid_file.exists() else None)
        fd = os.open(server.lock_path(g), os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            world.lock_free_at_exec.append(True)
        except OSError:
            world.lock_free_at_exec.append(False)
        finally:
            os.close(fd)
        raise _Exec

    monkeypatch.setattr(server, "exec_server", exec_recorder)
    monkeypatch.setattr(serve_cmd.err_console, "print", lambda line, *a, **kw: world.err.append(str(line)))
    return world


@pytest.fixture
def world(tmp_path, monkeypatch) -> ServeWorld:
    return serve_world(tmp_path, monkeypatch)


def _launched(world: ServeWorld, result) -> tuple[str, list[str], dict]:
    assert isinstance(result.exception, _Exec), (result.exit_code, world.err, result.exception)
    assert len(world.execs) == 1
    return world.execs[0]


# ====================================================================== the tracer


def test_serve_on_a_recorded_machine_plans_the_record_and_reaches_the_exec(world):
    result = world.run()

    path, argv, _env = _launched(world, result)
    assert path == str(world.binary) and argv[0] == str(world.binary)
    assert argv[argv.index("-m") + 1] == str(world.model)
    assert "--port" in argv and argv[argv.index("--port") + 1] == "8080"
    assert world.said().endswith("serve: launching swift (full)")


def test_serve_writes_its_own_pid_into_the_pidfile_before_the_exec(world):
    world.run()

    text = world.pid_file.read_text(encoding="utf-8")
    assert text == world.pidfile_at_exec[0]
    pid, start, port = text.split()
    assert int(pid) == os.getpid() and start.isdigit() and port == "8080"


def test_serve_writes_the_boot_marker_for_the_record(world):
    world.run()

    marker = world.marker()
    assert marker is not None
    assert marker.record == world.record == "llamacpp:swift"
    assert marker.binary == str(world.binary)
    assert marker.invocation == "inv-1"
    assert marker.cause == "none" and marker.shape == "full" and marker.dropped == ()
    assert marker.planned_context and marker.full_fit == marker.planned_context
    assert marker.free_bytes == 40 * GIB
    assert datetime.fromisoformat(marker.at).tzinfo is not None


def test_serve_has_released_the_launch_lock_when_it_execs(world):
    world.run()

    assert world.lock_free_at_exec == [True]


def test_serve_is_hidden_from_the_help_but_has_a_help_of_its_own(world):
    top = CliRunner().invoke(app, ["--help"])
    own = CliRunner().invoke(app, ["serve", "--help"])

    assert top.exit_code == 0 and "│ start" in top.output and "│ serve" not in top.output
    assert own.exit_code == 0 and "--config-dir" in own.output


def test_serve_needs_a_config_dir(world):
    result = CliRunner().invoke(app, ["serve"])

    assert result.exit_code == 2 and world.execs == []


