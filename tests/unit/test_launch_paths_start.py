"""Launch path 1 of 5: `start`'s down-server branch (spec §5, D-08/D-09/D-13/D-20).

`_start_managed_server` asks `launch_decision` before it announces anything. A declared-only
llama.cpp record prints one refusal line to the error console and `start` exits 1 with no
`Error:` line and no `localharness doctor` pointer; a launchable one names its runtime and carries
the exposure phrase computed from the effective argv. No process is started and no socket opened.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import typer
import yaml

from localharness.cli import start_cmd
from localharness.cli.start_cmd import _start_async, _start_managed_server
from localharness.config.models import ManagedServerConfig
from localharness.provider import autosense, server
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries  # noqa: F401

URL = "http://127.0.0.1:8080/v1"
REFUSAL = "no llama-server — set server.binary"
PHRASE = "every interface, no key"


def _llama(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def _write_config(tmp_path, *, server_block: dict | None) -> None:
    data: dict = {"version": "1",
                  "provider": {"provider_type": "llamacpp", "base_url": URL, "api_key": "none",
                               "default_model": "qwen"}}
    if server_block is not None:
        data["server"] = server_block
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _record_err(monkeypatch) -> list[tuple[tuple, dict]]:
    seen: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: seen.append((a, k)))
    return seen


def _err_text(seen) -> list[str]:
    return [" ".join(str(x) for x in a) for a, _ in seen]


def _boot(tmp_path, monkeypatch, *, server_block, path_binary=None, probe_ok=False):
    """The real `_start_async` with the boundaries stubbed, a failing probe and a start recorder."""
    _stub_start_boundaries(tmp_path, monkeypatch)
    _write_config(tmp_path, server_block=server_block)

    async def probe(llm, max_retries=3, delay=2.0):
        return (True, "native", 262_144, None) if probe_ok else (False, None, None, "Connection error.")

    monkeypatch.setattr(start_cmd, "_probe_llm", probe)
    monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: path_binary)
    started: list[tuple] = []

    def start_server(config_dir, cmd, **kw):
        started.append((cmd, kw))
        return 4321

    monkeypatch.setattr(server, "start_server", start_server)
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    # D-31: a launch through the real `activate` sizes the window, which asks for free memory. Fake the
    # two seams so no test in this file shells out to the driver or reads the host (WR-04).
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "24576, 32768\n")
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": 64 * 2**30, "MemAvailable": 60 * 2**30})
    return SimpleNamespace(started=started, err=_record_err(monkeypatch),
                           out=_capture_start_console(monkeypatch))


async def _drive(tmp_path) -> int | None:
    try:
        await asyncio.wait_for(_start_async(None, False, False, str(tmp_path)), timeout=60)
    except typer.Exit as exc:
        return exc.exit_code
    return None


DECLARED_ONLY = {"runtime": "llamacpp", "model": "/x/m.gguf", "port": 8080}


# ------------------------------------------------------------------------------- tracer


async def test_a_declared_only_primary_prints_one_refusal_line_and_exits_1(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    assert await _drive(tmp_path) == 1
    lines = _err_text(seen.err)
    assert [line for line in lines if "llama-server" in line] == [REFUSAL]
    assert not [line for line in lines if "Error:" in line or "localharness doctor" in line]
    assert seen.started == []


async def test_the_refusal_is_printed_verbatim_not_as_markup(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary="/opt/[bold]x/llama-server")
    assert await _drive(tmp_path) == 1
    (args, kwargs), = [pair for pair in seen.err if "llama-server" in str(pair[0])]
    assert args == (f"{REFUSAL} (PATH has /opt/[bold]x/llama-server)",)
    assert kwargs.get("markup") is False and kwargs.get("soft_wrap") is True


async def test_a_path_binary_is_named_in_the_refusal_text_only(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY, path_binary="/opt/llama/llama-server")
    assert await _drive(tmp_path) == 1
    assert [line for line in _err_text(seen.err) if "llama-server" in line] == [
        "no llama-server — set server.binary (PATH has /opt/llama/llama-server)"]
    assert seen.started == []


async def test_a_refusal_announces_nothing_and_never_activates(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    activated: list[int] = []
    from localharness.provider import lifecycle

    async def activate(self, *a, **k):
        activated.append(1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    assert await _drive(tmp_path) == 1
    assert activated == []
    assert not [line for line in seen.out if "starting it" in line or "still loading" in line]


async def test_an_attach_only_config_keeps_the_unreachable_line_and_the_doctor_pointer(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block=None)
    assert await _drive(tmp_path) == 1
    lines = _err_text(seen.err)
    assert any("model endpoint unreachable at" in line for line in lines)
    assert any("Run `localharness doctor` to diagnose" in line for line in lines)
    assert not [line for line in lines if REFUSAL in line]


async def test_a_declared_only_peer_is_not_touched_by_the_primary_sensing(tmp_path, monkeypatch):
    """The sensing line asks PATH once per start; it never launches what it finds."""
    asked: list[str] = []
    seen = _boot(tmp_path, monkeypatch, server_block=DECLARED_ONLY)
    monkeypatch.setattr(start_cmd, "which_outside_cwd", lambda name: asked.append(name))
    assert await _drive(tmp_path) == 1
    assert asked == ["llama-server"]
    assert seen.started == []


def test_the_path_is_sensed_before_the_machine_trust_decision():
    text = open(start_cmd.__file__, encoding="utf-8").read()
    assert text.index('which_outside_cwd("llama-server")') < text.index("decide_machine_trust(cfg_path")


# ------------------------------------------------------------------------------- the helper


async def test_the_helper_returns_the_refusal_and_prints_and_activates_nothing(tmp_path, monkeypatch):
    out = _capture_start_console(monkeypatch)
    err = _record_err(monkeypatch)
    from localharness.provider import lifecycle

    activated: list[int] = []

    async def activate(self, *a, **k):
        activated.append(1)

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    spec = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080)
    assert await _start_managed_server(spec, tmp_path, URL, {}) == REFUSAL
    assert out == [] and err == [] and activated == []


async def test_the_helper_appends_the_path_only_for_llama_cpp(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    spec = ManagedServerConfig(runtime="llamacpp", model="/x/m.gguf", port=8080)
    assert await _start_managed_server(spec, tmp_path, URL, {}, path_binary="/p/llama-server") == (
        f"{REFUSAL} (PATH has /p/llama-server)")


# ------------------------------------------------------------------------------- the launch lines


class _Launch:
    """Console, error console and `start_server` recorders sharing one ordering log."""

    def __init__(self, monkeypatch, *, pid=None):
        self.log: list[str] = []
        self.out: list[tuple[tuple, dict]] = []
        self.err: list[tuple[tuple, dict]] = []
        self.cmds: list[list[str]] = []
        self.kw: list[dict] = []

        def printer(sink, tag):
            def _print(*a, **k):
                sink.append((a, k))
                self.log.append(f"{tag}:{a[0] if a else ''}")
            return _print

        def start(config_dir, cmd, **kw):
            self.cmds.append(list(cmd))
            self.kw.append(kw)
            self.log.append("start_server")
            return 4321

        async def wait_ready(base_url, **kw):
            self.log.append("wait_ready")
            return ["m"]

        monkeypatch.setattr(start_cmd.console, "print", printer(self.out, "out"))
        monkeypatch.setattr(start_cmd.err_console, "print", printer(self.err, "err"))
        monkeypatch.setattr(server, "server_pid", lambda _d: pid)
        monkeypatch.setattr(server, "docker_container_running", lambda _n: False)
        monkeypatch.setattr(server, "start_server", start)
        monkeypatch.setattr(server, "wait_ready", wait_ready)

    @property
    def lines(self) -> list[str]:
        return [" ".join(str(x) for x in a) for a, _ in self.out]

    @property
    def err_lines(self) -> list[str]:
        return [" ".join(str(x) for x in a) for a, _ in self.err]


async def test_an_extra_args_host_never_widens_the_launch_and_the_line_names_llama_cpp(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    spec = _llama(extra_args=["--host", "0.0.0.0"])
    assert await _start_managed_server(spec, tmp_path, URL, {}) is None
    assert seen.cmds[0][-2:] == ["--host", "127.0.0.1"]
    assert seen.lines == ["Managed llama.cpp is not running — starting it (model load can take several minutes)..."]


async def test_every_interface_no_key_rides_the_announcement_once_and_before_the_launch(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    assert await _start_managed_server(_llama(bind_all=True), tmp_path, URL, {}) is None
    assert seen.lines == [f"Managed llama.cpp is not running — starting it ({PHRASE}; "
                          "model load can take several minutes)..."]
    assert len([line for line in seen.lines if PHRASE in line]) == 1
    assert seen.log.index(f"out:{seen.lines[0]}") < seen.log.index("start_server")
    assert seen.cmds[0][-2:] != ["--host", "127.0.0.1"] and "0.0.0.0" in seen.cmds[0]


async def test_a_held_key_means_the_line_carries_no_exposure_phrase(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    spec = _llama(bind_all=True, require_api_key=True)
    assert await _start_managed_server(spec, tmp_path, URL, {"api_key": "K"}) is None
    assert not [line for line in seen.lines if PHRASE in line]
    assert seen.lines == ["Managed llama.cpp is not running — starting it (model load can take several minutes)..."]


async def test_a_vllm_docker_record_keeps_its_0_17_line_byte_for_byte(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    spec = ManagedServerConfig(launch="docker", docker_image="img:tag", model="org/repo", port=8080,
                               bind_all=False)
    assert await _start_managed_server(spec, tmp_path, URL, {}) is None
    assert seen.lines == ["Managed vLLM is not running — starting it (model load can take several minutes)..."]


async def test_an_unreadable_key_file_prints_its_note_verbatim_and_launches_on_loopback(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    spec = _llama(bind_all=True, require_api_key=True, api_key_file=str(tmp_path))
    assert await _start_managed_server(spec, tmp_path, URL, {}) is None
    (args, kwargs), = seen.err
    assert args == (f"--api-key-file {tmp_path} unreadable: not a regular file — "
                    "launched on 127.0.0.1 for this run",)
    assert kwargs.get("markup") is False and kwargs.get("soft_wrap") is True
    assert seen.cmds[0][-2:] == ["--host", "127.0.0.1"] and "--api-key-file" not in seen.cmds[0]
    assert PHRASE not in seen.lines[0]


async def test_a_bracketed_note_path_reaches_the_error_console_verbatim(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch)
    bad = tmp_path / "[bold]keys"
    bad.mkdir()
    spec = _llama(bind_all=True, require_api_key=True, api_key_file=str(bad))
    assert await _start_managed_server(spec, tmp_path, URL, {}) is None
    assert f"--api-key-file {bad} unreadable" in seen.err_lines[0]


async def test_a_server_already_loading_is_waited_for_with_no_decision_and_no_posture(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch, pid=4321)

    def boom(*a, **k):
        raise AssertionError("a start that only waits asks no launch decision")

    monkeypatch.setattr(server, "launch_decision", boom)
    assert await _start_managed_server(_llama(bind_all=True), tmp_path, URL, {}) is None
    assert seen.lines == ["Managed llama.cpp is still loading — waiting..."]
    assert seen.cmds == [] and seen.log == ["out:Managed llama.cpp is still loading — waiting...", "wait_ready"]


async def test_a_vllm_record_that_is_loading_keeps_its_0_17_wait_line(tmp_path, monkeypatch):
    seen = _Launch(monkeypatch, pid=4321)
    spec = ManagedServerConfig(launch="binary", binary="/opt/vllm", model="org/repo", port=8080)
    assert await _start_managed_server(spec, tmp_path, URL, {}) is None
    assert seen.lines == ["Managed vLLM is still loading — waiting..."]


@pytest.mark.parametrize(("runtime", "label"), [("vllm", "vLLM"), ("llamacpp", "llama.cpp"),
                                                 ("ollama", "Ollama"), ("lmstudio", "LM Studio")])
def test_the_runtime_label_comes_from_the_one_table(runtime, label):
    assert server.RUNTIME_LABELS[runtime] == label


async def test_a_failed_launch_names_the_runtime_in_the_error_line(tmp_path, monkeypatch):
    seen = _boot(tmp_path, monkeypatch, server_block={"runtime": "llamacpp", "binary": "/x/llama-server",
                                                      "model": "/x/m.gguf", "port": 8080})

    def boom(config_dir, cmd, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "start_server", boom)
    assert await _drive(tmp_path) == 1
    assert [line for line in _err_text(seen.err) if "Error:" in line][0] == (
        "[bold red]Error:[/bold red] managed llama.cpp failed to start: boom")
    assert not [line for line in _err_text(seen.err) if REFUSAL in line]


async def test_a_failed_launch_never_reaches_a_real_nvidia_smi(tmp_path, monkeypatch):
    """WR-04 (D-31): the unit suite is hermetic. Were the real `_run_nvidia_smi` reached, the resolved
    executable would run and this `subprocess.run` would fail the test, on any box."""
    seen = _boot(tmp_path, monkeypatch, server_block={"runtime": "llamacpp", "binary": "/x/llama-server",
                                                      "model": "/x/m.gguf", "port": 8080})

    def no_subprocess(*a, **k):
        raise AssertionError("a unit test ran a real subprocess for free memory")

    monkeypatch.setattr(autosense, "which_outside_cwd", lambda name: "/fake/nvidia-smi")
    monkeypatch.setattr(autosense.subprocess, "run", no_subprocess)

    def boom(config_dir, cmd, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(server, "start_server", boom)
    assert await _drive(tmp_path) == 1
    assert [line for line in _err_text(seen.err) if "Error:" in line]


async def test_a_failed_launch_message_with_brackets_reaches_the_error_console_verbatim(tmp_path, monkeypatch):
    # Real Rich console on the error stream: a raw `[/x]` in the exception text used to raise
    # MarkupError while `start` was reporting the failure.
    import io

    from rich.console import Console

    seen = _boot(tmp_path, monkeypatch, server_block={"runtime": "llamacpp", "binary": "/x/llama-server",
                                                      "model": "/x/m.gguf", "port": 8080})
    buf = io.StringIO()
    monkeypatch.setattr(start_cmd, "err_console", Console(file=buf, width=500, force_terminal=False))
    message = "log tail: [bold]warn[/bold] [/oops] [red]x"

    def boom(config_dir, cmd, **kw):
        raise RuntimeError(message)

    monkeypatch.setattr(server, "start_server", boom)
    assert await _drive(tmp_path) == 1
    assert f"Error: managed llama.cpp failed to start: {message}" in buf.getvalue()


# ------------------------------------------------------------------------------- autosense / endpoint_out


class _Recording:
    """A lifecycle strategy stand-in: records the keywords `activate` was called with."""

    def __init__(self, endpoint=None):
        self.calls: list[dict] = []
        self.endpoint = endpoint

    async def activate(self, spec, config_dir, base_url, **kw):
        self.calls.append(kw)
        return self.endpoint


async def test_the_helper_hands_autosense_to_activate_and_collects_the_endpoint(tmp_path, monkeypatch):
    from localharness.config.models import AutosenseConfig
    from localharness.provider import lifecycle
    from localharness.provider.lifecycle import LiveEndpoint

    _Launch(monkeypatch)
    live = LiveEndpoint(base_url=URL, served_models=["m"], handle=4321, dropped_env=("A",))
    fake = _Recording(live)
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: type(
        "S", (), {"liveness": lambda self, s, d: lifecycle.Liveness(False), "activate": fake.activate})())
    cfg, out = AutosenseConfig(), []
    assert await _start_managed_server(_llama(), tmp_path, URL, {}, autosense=cfg, endpoint_out=out) is None
    assert fake.calls == [{"autosense": cfg}]
    assert out == [live]


async def test_the_new_keywords_default_to_none(tmp_path, monkeypatch):
    from localharness.provider import lifecycle

    _Launch(monkeypatch)
    fake = _Recording()
    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec: type(
        "S", (), {"liveness": lambda self, s, d: lifecycle.Liveness(False), "activate": fake.activate})())
    assert await _start_managed_server(_llama(), tmp_path, URL, {}) is None
    assert fake.calls == [{"autosense": None}]


async def test_a_server_already_loading_collects_no_endpoint_and_never_activates(tmp_path, monkeypatch):
    from localharness.provider import lifecycle

    _Launch(monkeypatch, pid=4321)

    async def activate(self, *a, **k):
        raise AssertionError("a start that only waits does not activate")

    monkeypatch.setattr(lifecycle.SpawnedProcessStrategy, "activate", activate)
    out: list = []
    assert await _start_managed_server(_llama(), tmp_path, URL, {}, endpoint_out=out) is None
    assert out == []
