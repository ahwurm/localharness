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


# ====================================================================== 78: refused


def _no_record(w: ServeWorld) -> None:
    trust.trust_store_path().unlink()


def _record_withheld(w: ServeWorld) -> None:
    w.edit_config(lambda d: d["server"].update(binary="/other/llama-server"))


def _pending_peer_withheld(w: ServeWorld) -> None:
    w.edit_config(lambda d: d["extra_endpoints"][0]["lifecycle"].update(binary="/other/llama-server"))
    w.set_overrides({"pending_preset": "flash"})


def _path_line_edited(w: ServeWorld) -> None:
    w.write_unit({"PATH": f"{w.bindir}:/usr/bin"})  # not adopted: a changed environment


def _env_block_unparsable(w: ServeWorld) -> None:
    body = units.read_harness_unit()
    units.write_harness_unit(body.replace(f"# env PATH={w.bindir}", "# env PATH=relative/bin"))


def _no_unit_file(w: ServeWorld) -> None:
    units.remove_harness_unit()


def _fragment_elsewhere(w: ServeWorld) -> None:
    w.manager_says(fragment="/etc/systemd/user/localharness-llm.service")


def _drop_in_present(w: ServeWorld) -> None:
    w.manager_says(drop_ins="/home/u/.config/systemd/user/localharness-llm.service.d/x.conf")


def _one_byte_of_exec_start(w: ServeWorld) -> None:
    body = units.read_harness_unit()
    assert " -I -m localharness" in body
    units.write_harness_unit(body.replace(" -I -m localharness", " -u -m localharness"))


def _no_binary(w: ServeWorld) -> None:
    w.edit_config(lambda d: d["server"].pop("binary"))
    w.rerecord()
    w.write_unit(adopt=True)


REFUSALS = [
    pytest.param(_no_record, "no confirmed record", id="no-machine-record"),
    pytest.param(_record_withheld, "server.binary (config.yaml)", id="target-record-withheld"),
    pytest.param(_pending_peer_withheld, "extra_endpoints.flash.lifecycle.binary", id="pending-preset-names-a-withheld-peer"),
    pytest.param(_path_line_edited, "autostart unit's environment", id="autostart-env-withheld"),
    pytest.param(_env_block_unparsable, "no usable autostart environment", id="env-block-unparsable"),
    pytest.param(_no_unit_file, "no readable unit file", id="no-unit-file"),
    pytest.param(_fragment_elsewhere, "FragmentPath", id="fragment-path-elsewhere"),
    pytest.param(_drop_in_present, "drop-ins", id="drop-in-present"),
    pytest.param(_one_byte_of_exec_start, "differs from what localharness writes", id="exec-start-edited"),
    pytest.param(_no_binary, "no llama-server", id="launch-decision-refused"),
]


@pytest.mark.parametrize("mutate,line", REFUSALS)
def test_serve_refuses_with_78_and_launches_nothing(tmp_path, monkeypatch, mutate, line):
    w = serve_world(tmp_path, monkeypatch, peers=[llama_peer("flash", 8081)])
    mutate(w)

    result = w.run()

    w.untouched_by(result, 78)
    assert line in w.said()


def test_serve_with_a_withheld_pending_preset_clears_nothing(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, peers=[llama_peer("flash", 8081)])
    _pending_peer_withheld(w)
    before = (w.g / "overrides.yaml").read_bytes()

    result = w.run()

    w.untouched_by(result, 78)
    assert (w.g / "overrides.yaml").read_bytes() == before
    assert "pending_preset" in (w.g / "overrides.yaml").read_text(encoding="utf-8")


def test_serve_refuses_a_changed_environment_the_record_does_not_cover_with_78(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch)
    w.write_unit({"PATH": str(w.bindir), "HF_HOME": "/h/hf"})  # a new value, never adopted

    result = w.run()

    w.untouched_by(result, 78)


def test_serve_refuses_a_recorded_entry_that_no_longer_matches_the_unit(tmp_path, monkeypatch):
    # the record holds an older environment while the machine decision reports nothing withheld: the
    # compare itself is the last line of defence
    w = serve_world(tmp_path, monkeypatch)
    entry = trust.recorded_autostart_entry(w.g)
    stale = {**entry, "shown": json.dumps({"PATH": "/usr/bin"}, separators=(",", ":"))}
    monkeypatch.setattr(trust, "recorded_autostart_entry", lambda g: stale)

    result = w.run()

    w.untouched_by(result, 78)
    assert "differs from the machine record" in w.said()


def test_serve_never_asks_a_question(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch)
    _record_withheld(w)  # a changed launch entry: a start in a terminal would ask here

    result = w.run()  # never_asked(monkeypatch) raises if anything asks

    w.untouched_by(result, 78)


# ====================================================================== 0: nothing to do


def _by_pending(name: str):
    def mutate(w: ServeWorld) -> None:
        w.set_overrides({"pending_preset": name})
    return mutate


def _by_active(w: ServeWorld) -> None:
    w.set_overrides({"active_endpoint": {
        "name": "attached", "base_url": "http://127.0.0.1:1234/v1", "provider_type": "lmstudio",
        "model": "attached"}})


NOTHING = [
    pytest.param([docker_peer()], _by_pending("dock"), "dock is docker — not served by the unit", id="docker"),
    pytest.param([ollama_peer()], _by_pending("oll"), "oll is Ollama — not served by the unit", id="ollama"),
    pytest.param([attach_peer()], _by_active, "attached is attach-only — not served by the unit",
                 id="attach-only"),
]


@pytest.mark.parametrize("peers,mutate,line", NOTHING)
def test_serve_has_nothing_to_do_for_a_record_the_unit_does_not_serve(tmp_path, monkeypatch, peers, mutate, line):
    w = serve_world(tmp_path, monkeypatch, peers=peers)
    mutate(w)

    result = w.run()

    w.untouched_by(result, 0)
    assert w.said() == line


def test_serve_says_already_serving_for_a_verified_live_launch_of_the_record(tmp_path, monkeypatch, fake_proc):  # noqa: F811
    w = serve_world(tmp_path, monkeypatch)
    fake_proc.process(4242, argv=(str(w.binary), "-m", str(w.model), "-a", PRIMARY, "--port", "8080"),
                      exe=str(w.binary), start_time=777)
    pidfile(w.g, "4242 777 8080")

    result = w.run()

    assert result.exit_code == 0 and w.execs == [] and not w.marker_file.exists()
    assert w.said() == "already serving (pid 4242) — not started twice"
    assert w.pid_file.read_text(encoding="utf-8") == "4242 777 8080"


# ====================================================================== 75: a live pidfile


def test_serve_exits_75_and_leaves_the_pidfile_when_an_unverified_process_holds_it(
        tmp_path, monkeypatch, fake_proc):  # noqa: F811
    w = serve_world(tmp_path, monkeypatch)
    fake_proc.process(4242, argv=("bash",), comm="bash", exe="/usr/bin/bash", start_time=777)
    monkeypatch.setattr(server.os, "kill", lambda pid, sig: None)
    pidfile(w.g, "4242 777")
    proc_says(monkeypatch, 777)

    result = w.run()

    assert result.exit_code == serve_cmd.EXIT_LIVE_PIDFILE == 75
    assert w.execs == [] and not w.marker_file.exists()
    assert w.pid_file.read_text(encoding="utf-8") == "4242 777"
    assert w.said() == server.LIVE_LAUNCH_LINE.format(pid=4242, exe="bash", pidfile=server.pid_path(w.g))


# ====================================================================== 1: transient


def test_serve_exits_1_when_the_service_manager_cannot_be_asked(world):
    world.manager_silent()

    result = world.run()

    world.untouched_by(result, 1)
    assert world.said() == serve_cmd.NO_MANAGER_LINE


def test_serve_exits_1_when_the_launch_lock_wait_times_out(world, monkeypatch):
    real = lifecycle.launch_lock_sync
    monkeypatch.setattr(lifecycle, "launch_lock_sync", lambda cfg, *, timeout_seconds: real(cfg, timeout_seconds=0.2))
    server.server_dir(world.g).mkdir(parents=True, exist_ok=True)
    held = os.open(server.lock_path(world.g), os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        result = world.run()
    finally:
        os.close(held)

    world.untouched_by(result, 1)
    assert "another LocalHarness is launching the model server" in world.said()


def test_serve_exits_1_when_the_pidfile_cannot_be_written(world, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(server, "write_pidfile", refuse)

    result = world.run()

    world.untouched_by(result, 1)
    assert "could not record the launch: No space left on device" in world.said()


def test_serve_exits_1_when_the_config_ini_home_cannot_be_prepared(world, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(errno.EACCES, "Permission denied")

    monkeypatch.setattr(server, "prepare_config_ini_home", refuse)

    result = world.run()

    world.untouched_by(result, 1)
    assert "could not prepare the llama.cpp config home: Permission denied" in world.said()


def test_serve_exits_1_for_a_relative_binary_that_is_not_on_the_captured_path(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, binary_name="no-such-server", create_binary=False)
    monkeypatch.setenv("PATH", f"{w.tmp}/manager/bin:/usr/bin")  # the manager's PATH is not consulted

    result = w.run()

    w.untouched_by(result, 1)
    assert w.said() == serve_cmd.NOT_ON_PATH_LINE.format(binary="no-such-server", path=str(w.bindir))


def test_serve_exits_1_for_an_absolute_binary_that_is_not_executable(world):
    world.binary.chmod(0o644)

    result = world.run()

    world.untouched_by(result, 1)
    assert world.said() == serve_cmd.NOT_EXECUTABLE_LINE.format(binary=str(world.binary))


def test_serve_exits_1_when_the_config_will_not_load(world):
    world.config_path.write_text("server: [unclosed\n", encoding="utf-8")

    result = world.run()

    world.untouched_by(result, 1)
    assert "cannot load the config" in world.said()


def test_serve_exits_1_when_the_exec_itself_fails(world, monkeypatch):
    def broken(path, argv, env):
        raise OSError(errno.ENOENT, "No such file or directory")

    monkeypatch.setattr(server, "exec_server", broken)

    result = world.run()

    assert result.exit_code == 1 and "could not start" in world.said()


def test_serve_resolves_a_relative_binary_on_the_captured_path_only(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, binary_name="llama-server")
    decoy = tmp_path / "manager" / "bin"
    decoy.mkdir(parents=True)
    (decoy / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    (decoy / "llama-server").chmod(0o755)
    monkeypatch.setenv("PATH", f"{decoy}:/usr/bin")

    result = w.run()

    path, argv, env = _launched(w, result)
    assert path == str(w.bindir / "llama-server") and argv[0] == "llama-server"
    assert env["PATH"] == str(w.bindir)


# ====================================================================== the pin comes first


def test_serve_pins_the_config_dir_before_any_resolver_runs(world, monkeypatch):
    seen: list = []
    real = config_paths.resolve_config_dir

    def watching(config_dir=None):
        seen.append(os.environ.get("LOCALHARNESS_DIR"))
        return real(config_dir)

    monkeypatch.setattr(config_paths, "resolve_config_dir", watching)
    monkeypatch.setenv("LOCALHARNESS_DIR", "/planted/elsewhere")

    result = world.run()

    _launched(world, result)
    assert seen and set(seen) == {str(world.g)}


def test_serve_pins_a_relative_config_dir_as_its_absolute_form(world, monkeypatch):
    monkeypatch.chdir(world.g.parent)

    result = world.run(config_dir=world.g.name)

    _launched(world, result)
    assert os.environ["LOCALHARNESS_DIR"] == str(world.g)


def test_serve_sets_the_pin_in_its_first_statement():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(serve_cmd.serve))
    body = [n for n in tree.body[0].body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert ast.unparse(body[0]).startswith("pinned = ")
    assert ast.unparse(body[1]) == "os.environ['LOCALHARNESS_DIR'] = pinned"


# ====================================================================== the boot marker closes the loop


def _plant(world: ServeWorld, *, record: str | None = None, binary: str | None = None, cause: str = "death",
           shape: str = "full", age: int = 60) -> None:
    at = (datetime.now(timezone.utc) - timedelta(seconds=age)).isoformat()
    units.write_marker(world.g, units.Marker(
        record=record or world.record, binary=binary or str(world.binary), invocation="inv-0", cause=cause,
        shape=shape, planned_context=1000, full_fit=1000, free_bytes=1, dropped=(), at=at))


def test_a_boot_that_died_is_retried_in_the_retry_shape_and_the_marker_says_so(world):
    first = world.run()
    _launched(world, first)
    assert "--spec-type" in world.execs[0][1]  # the full shape drafts with MTP
    world.execs.clear()
    world.pid_file.unlink()  # the first server died: nothing holds the pidfile any more
    world.manager_says(n_restarts=1)

    second = world.run()

    _path, argv, _env = _launched(world, second)
    assert "--spec-type" not in argv
    marker = world.marker()
    assert marker.cause == "death" and marker.shape == "retry" and "draft-mtp" in marker.dropped
    assert marker.record == world.record and marker.binary == str(world.binary)


def test_a_planted_death_marker_with_a_restarted_unit_plans_the_boot_retry(world):
    _plant(world, cause="death", shape="retry")
    world.manager_says(n_restarts=1)

    result = world.run()

    _path, argv, _env = _launched(world, result)
    assert "--spec-type" not in argv
    assert world.marker().cause == "death"


def test_a_marker_for_another_record_is_replaced_by_serves_own(world):
    _plant(world, record="llamacpp:someone-else")
    world.manager_says(n_restarts=1)

    result = world.run()

    _path, argv, _env = _launched(world, result)
    assert "--spec-type" in argv  # the foreign marker was ignored
    marker = world.marker()
    assert marker.record == world.record and marker.cause == "none" and marker.shape == "full"


def test_an_unparsable_marker_is_replaced(world):
    server.server_dir(world.g).mkdir(parents=True, exist_ok=True)
    world.marker_file.write_text("{not json", encoding="utf-8")

    result = world.run()

    _launched(world, result)
    assert world.marker() is not None and world.marker().cause == "none"


def test_a_marker_that_cannot_be_written_does_not_stop_the_launch(world, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError(errno.EROFS, "Read-only file system")

    monkeypatch.setattr(units, "write_marker", refuse)

    result = world.run()

    _launched(world, result)
    assert "the boot marker was not written (Read-only file system) — launching anyway" in world.said()


def test_a_stale_death_marker_past_the_load_window_is_not_a_retry(world):
    _plant(world, cause="death", age=units.TIMEOUT_START_SECONDS + 5)
    world.manager_says(n_restarts=1)

    result = world.run()

    _path, argv, _env = _launched(world, result)
    assert "--spec-type" in argv


# ====================================================================== nothing else is written


def _files(w: ServeWorld) -> dict:
    watched = [w.g / "overrides.yaml", w.config_path, speed_stats.default_speed_stats_path(w.g),
               trust.trust_store_path(), units.harness_unit_path()]
    return {p: (p.read_bytes() if p.exists() else None) for p in watched}


def test_serve_writes_only_the_pidfile_the_marker_and_the_config_home(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch, peers=[llama_peer("flash", 8081)])
    w.set_overrides({"pending_preset": "gone-for-good"})  # a selection that names nothing
    before = _files(w)
    ledger = speed_stats.default_speed_stats_path(w.g)

    result = w.run()

    _launched(w, result)
    assert _files(w) == before  # the overlay (pending_preset kept), config, ledger, trust record, unit
    assert not ledger.exists()
    assert w.pid_file.exists() and w.marker_file.exists()
    assert (w.g / "xdg" / "8080").is_dir()


def test_serve_does_not_adopt_a_change_and_does_not_rewrite_the_record(world):
    before = trust.trust_store_path().read_bytes()

    result = world.run()

    _launched(world, result)
    assert trust.trust_store_path().read_bytes() == before
    assert trust.recorded_autostart_entry(world.g) is not None


# ====================================================================== the ladder is the only chooser


def test_serve_lands_on_the_pending_preset_the_ladder_names(tmp_path, monkeypatch):
    w = serve_world(tmp_path, monkeypatch,
                    peers=[llama_peer("flash", 8081, binary=str(tmp_path / "bin" / "llama-server"))])
    w.set_overrides({"pending_preset": "flash"})

    result = w.run()

    path, argv, _env = _launched(w, result)
    assert argv[argv.index("-m") + 1] == "/m/flash.gguf" and argv[argv.index("--port") + 1] == "8081"
    assert w.marker().record == "llamacpp:/m/flash.gguf"
    assert w.pid_file.read_text(encoding="utf-8").split()[2] == "8081"
