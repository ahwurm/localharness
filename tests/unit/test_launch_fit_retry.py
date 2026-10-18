# ruff: noqa: F811  (the `popen` fixture is imported by name, then taken as a parameter)
"""AUTO-04/05/06 (launch half): `activate(autosense=)` sizes `-c` and `--cache-ram`, verifies the served
window after readiness and pays one halve-once retry for a process that dies on a wrong fit (D-16..D-20,
D-25, D-30, D-31, D-37).

Nothing here starts a process or reads the host: `subprocess.Popen` is the `PopenRecorder`, free memory,
`/proc/meminfo`, the weights and the probe are faked through their module seams, and `wait_ready` is a
scripted fake (except in the D-37 tests, where the real `wait_ready`, `start_server` and `log_tail` run
against a process stand-in that dies).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from localharness.agent import context as agent_context
from localharness.provider import gguf_header
from localharness.agent.context import WindowProbe
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import autosense, lifecycle, server
from localharness.provider.autosense import FreeMemory, cache_ram_mib
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import PopenRecorder, popen, record_chain  # noqa: F401

GIB = 2**30
RATIOS = AutosenseConfig(kv_cache_type="f16", draft_mtp=False)  # the 0.18 shape these tests pin
BASE = "http://127.0.0.1:8080/v1"

HEADER_A = {"general.architecture": "qwen3", "qwen3.block_count": 64, "qwen3.attention.head_count_kv": 8,
            "qwen3.attention.key_length": 128, "qwen3.context_length": 262144}
WEIGHTS_A = 19864223744  # 18.5 GiB: with FREE free the fit is 14848
FREE = 24 * GIB + int(RATIOS.compute_reserve * GIB)
MEMINFO = {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB}
CACHE_RAM = cache_ram_mib(MEMINFO["MemAvailable"], MEMINFO["MemTotal"], WEIGHTS_A, "gpu", [], RATIOS)


def _gguf(tmp_path: Path) -> Path:
    return write_gguf(tmp_path / "model.gguf", HEADER_A)


def _spec(tmp_path: Path, **over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": str(_gguf(tmp_path)), "port": 8080, **over})


class Box:
    """The faked world one activation sees: the memory probes, the scripted wait_ready, the probe, and
    the ordered log of what happened."""

    def __init__(self, monkeypatch):
        self.events: list[str] = []
        self.probe_calls: list[tuple] = []
        self.ready_script: list = [["qwen"]]
        self.probe_result = WindowProbe(14950, 262144)
        self.stops: list[tuple] = []
        self.stop_kw: list[dict] = []
        mem = FreeMemory(FREE, 32 * GIB, "nvidia-smi")
        self.mem: FreeMemory | None = mem
        self.meminfo: dict | None = dict(MEMINFO)
        self.weights: int | None = WEIGHTS_A

        def free_memory(ratios):
            self.events.append("free_memory")
            return self.mem

        monkeypatch.setattr(autosense, "free_memory", free_memory)
        monkeypatch.setattr(autosense, "_read_meminfo", lambda: self.meminfo)
        monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: self.weights)

        async def wait_ready(base_url, **kw):
            self.events.append("wait_ready")
            step = self.ready_script.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step

        def probe_window(base_url, model, provider_type=None, *, api_key=None):
            self.events.append("probe_window")
            self.probe_calls.append((base_url, model, provider_type, api_key))
            return self.probe_result

        def stop_server(config_dir, launch="binary", timeout_seconds=30.0, **kw):
            self.events.append("stop_server")
            self.stops.append((config_dir, launch))
            self.stop_kw.append(kw)
            return True

        monkeypatch.setattr(server, "wait_ready", wait_ready)
        monkeypatch.setattr(agent_context, "probe_window", probe_window)
        monkeypatch.setattr(server, "stop_server", stop_server)


@pytest.fixture
def box(monkeypatch):
    return Box(monkeypatch)


def _argv(popen_calls, i=0) -> list[str]:
    return popen_calls[i]["cmd"]


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    return any(argv[i] == flag and argv[i + 1] == value for i in range(len(argv) - 1))


async def _activate(tmp_path, spec, **kw):
    return await lifecycle.SpawnedProcessStrategy().activate(
        spec, tmp_path, BASE, autosense=kw.pop("autosense", RATIOS), **kw)


# ------------------------------------------------------------------------------- the tracer


async def test_a_record_without_a_context_launches_with_the_fit_and_the_cache_size(tmp_path, popen, box):
    live = await _activate(tmp_path, _spec(tmp_path))
    argv = _argv(popen)
    assert _has_pair(argv, "-c", "14848")
    assert _has_pair(argv, "--cache-ram", str(CACHE_RAM))
    assert argv[-2:] == ["--host", "127.0.0.1"]
    w = live.window
    assert (w.served, w.trained, w.fits, w.asked, w.mode, w.verified, w.retried) == (
        14950, 262144, 14848, 14848, "fit", True, False)
    assert w.weights_bytes == WEIGHTS_A and w.free_bytes == FREE and w.total_bytes == 32 * GIB
    assert live.served_models == ["qwen"] and live.handle == 4242


async def test_the_first_launch_comes_before_any_probe(tmp_path, popen, box):
    await _activate(tmp_path, _spec(tmp_path))
    assert len(popen) == 1
    assert box.events == ["free_memory", "wait_ready", "probe_window"]
    assert box.probe_calls == [(BASE, "qwen", "llamacpp", None)]


async def test_the_sizing_runs_off_the_event_loop(tmp_path, popen, box, monkeypatch):
    """WR-02: free memory (nvidia-smi, up to 5 s), the header read and the weights stat are blocking
    work; they must not run on the loop's thread, and the module-attribute seams still fire."""
    import threading

    loop_thread = threading.get_ident()
    seen: list[int] = []

    def free_memory(ratios):
        seen.append(threading.get_ident())
        return box.mem

    monkeypatch.setattr(autosense, "free_memory", free_memory)
    live = await _activate(tmp_path, _spec(tmp_path))
    assert seen and all(t != loop_thread for t in seen)
    assert live.window.fits == 14848  # the faked memory was the one used


async def test_the_key_reaches_the_probe_and_nothing_else_does(tmp_path, popen, box):
    await _activate(tmp_path, _spec(tmp_path), api_key="sk-secret")
    assert box.probe_calls == [(BASE, "qwen", "llamacpp", "sk-secret")]
    assert "sk-secret" not in " ".join(_argv(popen))


# ------------------------------------------------------------------------------- who wins


async def test_a_record_context_wins_and_the_fit_stays_off_argv(tmp_path, popen, box):
    live = await _activate(tmp_path, _spec(tmp_path, context=4096))
    argv = _argv(popen)
    assert [t for t in argv if t == "-c"] == ["-c"] and _has_pair(argv, "-c", "4096")
    assert live.window.mode == "record" and live.window.asked == 4096
    assert live.window.fits == 14848  # still computed, for the report


@pytest.mark.parametrize("extra", [["-c", "8192"], ["--ctx-size=8192"], ["--ctx-size", "8192"]])
async def test_a_c_in_extra_args_wins_with_no_fit_keyword(tmp_path, popen, box, extra):
    live = await _activate(tmp_path, _spec(tmp_path, extra_args=extra))
    argv = _argv(popen)
    assert "14848" not in argv
    assert argv.count("-c") == extra.count("-c")  # the extra_args -c only, never a second one
    assert live.window.mode == "extra_args" and live.window.asked is None


async def test_an_extra_args_c_never_triggers_a_retry(tmp_path, popen, box):
    box.ready_script = [RuntimeError("llama.cpp exited during startup. Log tail:\nx")]
    with pytest.raises(RuntimeError, match="Log tail"):
        await _activate(tmp_path, _spec(tmp_path, extra_args=["-c", "8192"]))
    assert len(popen) == 1 and box.stops == []


async def test_an_unknown_free_memory_launches_with_no_c(tmp_path, popen, box):
    box.mem = None
    live = await _activate(tmp_path, _spec(tmp_path))
    assert "-c" not in _argv(popen) and "--fit-ctx" not in _argv(popen)
    assert live.window.mode == "not-computed" and live.window.asked is None and live.window.fits is None


async def test_weights_beyond_the_free_memory_are_refused_before_anything_starts(tmp_path, popen, box):
    box.mem = FreeMemory(1 * GIB, 32 * GIB, "nvidia-smi")
    with pytest.raises(server.LaunchRefused, match="needs 18.5 GiB of GPU memory and 1.0 GiB is free"):
        await _activate(tmp_path, _spec(tmp_path))
    assert popen == [] and box.events == ["free_memory"]


async def test_an_unreadable_header_is_not_computed_not_a_failure(tmp_path, popen, box):
    spec = _spec(tmp_path)
    Path(spec.model).write_bytes(b"not a gguf")
    live = await _activate(tmp_path, spec)
    assert "-c" not in _argv(popen)
    assert live.window.mode == "fit-by-server" and live.window.fits is None


async def test_ill_typed_header_values_are_not_computed_and_never_crash_the_launch(tmp_path, popen, box):
    """WR-01: a GGUF that declares `block_count` as a string and `context_length` as a list must not
    reach `offload_kind` / `window_lines` as such (a TypeError there escaped `activate` after the
    incumbent was stopped)."""
    bad = {**HEADER_A, "qwen3.block_count": "abc", "qwen3.context_length": [1]}
    spec = _spec(tmp_path, context=4096, extra_args=["-ngl", "10"])
    write_gguf(Path(spec.model), bad)
    box.probe_result = WindowProbe(4096, None)
    live = await _activate(tmp_path, spec)
    assert live.window.trained is None and live.window.fits is None
    (line,) = autosense.window_lines(live.window, RATIOS)
    assert line.startswith("window: served 4,096 · trained — · fits —")


async def test_a_missing_model_file_is_not_computed_not_a_failure(tmp_path, popen, box):
    spec = _spec(tmp_path)
    Path(spec.model).unlink()
    box.weights = None
    live = await _activate(tmp_path, spec)
    assert "-c" not in _argv(popen) and live.window.mode == "fit-by-server"


async def test_a_vllm_record_never_computes_a_fit(tmp_path, monkeypatch, box):
    def boom(*a, **k):
        raise AssertionError("a vLLM launch must not size a window")

    monkeypatch.setattr(gguf_header, "read_header", boom)
    monkeypatch.setattr(autosense, "free_memory", boom)
    calls = record_chain(monkeypatch)
    spec = ManagedServerConfig(launch="binary", binary="/opt/vllm", model="org/repo", port=8081)
    live = await lifecycle.ManagedVllmStrategy().activate(
        spec, tmp_path, "http://127.0.0.1:8081/v1", autosense=RATIOS)
    assert calls["serve_command"]["context"] is None and calls["serve_command"]["cache_ram"] is None
    assert live.window is None


# ------------------------------------------------------------------------------- --cache-ram


async def test_a_record_cache_ram_of_zero_is_sent_as_zero(tmp_path, popen, box):
    await _activate(tmp_path, _spec(tmp_path, cache_ram=0))
    argv = _argv(popen)
    assert _has_pair(argv, "--cache-ram", "0") and argv.count("--cache-ram") == 1
    assert _has_pair(argv, "-c", "14848")  # the fit path is independent of it


async def test_a_record_cache_ram_wins_over_the_formula(tmp_path, popen, box):
    await _activate(tmp_path, _spec(tmp_path, cache_ram=512))
    assert _has_pair(_argv(popen), "--cache-ram", "512")


async def test_unreadable_meminfo_sends_no_cache_ram(tmp_path, popen, box):
    box.meminfo = None
    await _activate(tmp_path, _spec(tmp_path))
    assert "--cache-ram" not in _argv(popen)


async def test_empty_extra_args_and_unknown_memory_send_nothing_extra(tmp_path, popen, box):
    box.meminfo = None
    box.mem = None
    await _activate(tmp_path, _spec(tmp_path, extra_args=[]))
    argv = _argv(popen)
    assert "-c" not in argv and "--cache-ram" not in argv


@pytest.mark.parametrize("extra", [["--cache-ram", "100"], ["--cache-ram=100"]])
async def test_a_cache_ram_in_extra_args_is_not_doubled(tmp_path, popen, box, extra):
    await _activate(tmp_path, _spec(tmp_path, extra_args=extra))
    argv = _argv(popen)
    assert sum(t == "--cache-ram" or t.startswith("--cache-ram=") for t in argv) == 1


# ------------------------------------------------------------------------------- verification


async def test_a_probe_that_answers_nothing_is_unverified_not_a_failure(tmp_path, popen, box):
    box.probe_result = WindowProbe(None, None)
    live = await _activate(tmp_path, _spec(tmp_path))
    assert live.window.verified is False and live.window.served is None
    assert live.window.trained == 262144  # the header's number survives a silent probe


async def test_a_served_window_smaller_than_asked_is_a_mismatch_not_a_failure(tmp_path, popen, box):
    box.probe_result = WindowProbe(8192, 262144)
    live = await _activate(tmp_path, _spec(tmp_path))
    assert live.window.served == 8192 < live.window.asked == 14848
    assert live.window.verified is True


async def test_the_probe_names_the_served_alias_when_the_server_lists_none(tmp_path, popen, box):
    box.ready_script = [[]]
    await _activate(tmp_path, _spec(tmp_path, served_name="alias"))
    assert box.probe_calls[0][1] == "alias"


# ------------------------------------------------------------------------------- ordering


async def test_free_memory_is_read_inside_activate_after_the_callers_stop(tmp_path, popen, box):
    strategy = lifecycle.SpawnedProcessStrategy()
    spec = _spec(tmp_path)
    await strategy.stop(spec, tmp_path)
    await strategy.activate(spec, tmp_path, BASE, autosense=RATIOS)
    assert box.events[:2] == ["stop_server", "free_memory"]


# ------------------------------------------------------------------------------- autosense=None


async def test_without_autosense_the_phase_1_launch_is_unchanged(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("autosense=None must not size or probe anything")

    monkeypatch.setattr(agent_context, "probe_window", boom)
    monkeypatch.setattr(autosense, "free_memory", boom)
    monkeypatch.setattr(gguf_header, "read_header", boom)
    calls = record_chain(monkeypatch)
    spec = ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model="/x/m.gguf", port=8080)
    live = await lifecycle.SpawnedProcessStrategy().activate(spec, tmp_path, BASE)
    assert calls["serve_command"] == {"keyed": False, "loopback": True, "key_file_ok": False,
                                      "context": None, "cache_ram": None}
    assert live.window is None and live.dropped_env == ()


async def test_without_autosense_a_dead_process_is_not_retried(tmp_path, popen, box):
    box.ready_script = [RuntimeError("llama.cpp exited during startup. Log tail:\nx")]
    with pytest.raises(RuntimeError):
        await _activate(tmp_path, _spec(tmp_path), autosense=None)
    assert len(popen) == 1 and box.stops == [] and "-c" not in _argv(popen)


# ------------------------------------------------------------------------------- dropped environment


def test_start_server_collects_the_dropped_names_for_the_caller(tmp_path, popen, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    names: list[str] = []
    server.start_server(tmp_path, ["llama-server"], runtime="llamacpp", dropped_out=names)
    assert names == ["LLAMA_ARG_HOST"]
    server.start_server(tmp_path, ["llama-server"], runtime="llamacpp")  # the old signature still works
    assert "LLAMA_ARG_HOST" not in popen[-1]["env"]


async def test_the_dropped_names_ride_the_live_endpoint(tmp_path, popen, box, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    live = await _activate(tmp_path, _spec(tmp_path))
    assert live.dropped_env == ("LLAMA_ARG_HOST",)
    box.ready_script = [["qwen"]]
    live = await _activate(tmp_path, _spec(tmp_path), autosense=None)
    assert live.dropped_env == ("LLAMA_ARG_HOST",)  # D-25: both strategies, with or without a fit


async def test_a_vllm_launch_reports_its_dropped_names_too(tmp_path, monkeypatch, popen):
    monkeypatch.setenv("VLLM_API_KEY", "left-over")

    async def ready(base_url, **kw):
        return ["m"]

    monkeypatch.setattr(server, "wait_ready", ready)
    spec = ManagedServerConfig(launch="binary", binary="/opt/vllm", model="org/repo", port=8081)
    live = await lifecycle.ManagedVllmStrategy().activate(spec, tmp_path, "http://127.0.0.1:8081/v1")
    assert live.dropped_env == ("VLLM_API_KEY",)


def test_the_live_endpoint_defaults_are_the_phase_1_shape():
    live = lifecycle.LiveEndpoint(base_url="u", served_models=[], handle=None)
    assert live.window is None and live.dropped_env == ()
    assert inspect.signature(server.start_server).parameters["dropped_out"].default is None


# ------------------------------------------------------------------------------- the halve-once retry

DEAD = "llama.cpp exited during startup. Log tail:\nsrv: exiting"
_REAL_WAIT_READY = server.wait_ready  # captured before any fixture replaces it


def _launches(box, monkeypatch):
    """Popen as a recorder that also logs each launch into the box's ordered events."""
    class EventPopen(PopenRecorder):
        def __init__(self, cmd, **kw):
            box.events.append("popen")
            super().__init__(cmd, **kw)

    PopenRecorder.calls = []
    monkeypatch.setattr(server.subprocess, "Popen", EventPopen)
    return PopenRecorder.calls


async def test_a_dead_process_on_a_fit_c_is_stopped_then_relaunched_at_half(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), ["qwen"]]
    live = await _activate(tmp_path, _spec(tmp_path))
    assert len(calls) == 2
    assert _has_pair(_argv(calls, 0), "-c", "14848") and _has_pair(_argv(calls, 1), "-c", "7424")
    assert box.stops == [(tmp_path, "binary")]
    assert live.window.retried is True and live.window.asked == 7424
    assert live.window.mode == "fit"


async def test_the_stop_comes_between_the_two_launches_and_no_probe_before_the_first(tmp_path, box, monkeypatch):
    _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), ["qwen"]]
    await _activate(tmp_path, _spec(tmp_path))
    assert box.events == ["free_memory", "popen", "wait_ready", "stop_server", "popen", "wait_ready",
                          "probe_window"]


async def test_the_cache_ram_and_the_rest_of_the_argv_survive_the_halving(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), ["qwen"]]
    await _activate(tmp_path, _spec(tmp_path))
    first, second = _argv(calls, 0), _argv(calls, 1)
    strip = lambda argv: [t for t in argv if t not in ("14848", "7424")]  # noqa: E731
    assert strip(first) == strip(second) and _has_pair(second, "--cache-ram", str(CACHE_RAM))


async def test_two_dead_processes_stop_twice_launch_twice_and_surface_the_tail(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), RuntimeError(DEAD)]
    with pytest.raises(RuntimeError, match="Log tail:"):
        await _activate(tmp_path, _spec(tmp_path))
    assert len(calls) == 2  # exactly one retry, never three
    assert len(box.stops) == 2
    assert box.events.count("probe_window") == 0


async def test_a_failing_second_stop_keeps_the_second_death_tail_as_the_message(tmp_path, box, monkeypatch):
    """IN-05: `stop_server` raises RuntimeError when a pid survives SIGKILL; that must not replace the
    log tail the user needs."""
    _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), RuntimeError(DEAD)]
    stops: list[int] = []

    def stop_server(config_dir, launch="binary", timeout_seconds=30.0, **kw):
        stops.append(1)
        if len(stops) == 2:
            raise RuntimeError("pid 4242 survived SIGKILL")
        return True

    monkeypatch.setattr(server, "stop_server", stop_server)
    with pytest.raises(RuntimeError, match="Log tail:"):
        await _activate(tmp_path, _spec(tmp_path))
    assert len(stops) == 2


async def test_a_failing_first_stop_surfaces_the_original_death_and_does_not_relaunch(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD)]

    def stop_server(config_dir, launch="binary", timeout_seconds=30.0, **kw):
        raise RuntimeError("pid 4242 survived SIGKILL")

    monkeypatch.setattr(server, "stop_server", stop_server)
    with pytest.raises(RuntimeError, match="Log tail:"):
        await _activate(tmp_path, _spec(tmp_path))
    assert len(calls) == 1  # an unstopped survivor makes a second launch unsafe


async def test_a_timeout_is_never_retried(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.ready_script = [TimeoutError("llama.cpp not ready after 1800s.")]
    with pytest.raises(TimeoutError):
        await _activate(tmp_path, _spec(tmp_path))
    assert len(calls) == 1 and box.stops == []


@pytest.mark.parametrize("over", [
    {"context": 4096},  # mode "record"
    {"extra_args": ["-c", "8192"]},  # mode "extra_args"
])
async def test_nothing_is_halved_without_a_fit_keyword(tmp_path, box, monkeypatch, over):
    calls = _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD)]
    with pytest.raises(RuntimeError, match="Log tail:"):
        await _activate(tmp_path, _spec(tmp_path, **over))
    assert len(calls) == 1 and box.stops == []


async def test_a_launch_the_server_sized_itself_is_not_retried(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    box.mem = None
    box.ready_script = [RuntimeError(DEAD)]
    with pytest.raises(RuntimeError):
        await _activate(tmp_path, _spec(tmp_path))
    assert len(calls) == 1 and box.stops == [] and "-c" not in _argv(calls)


async def test_a_fit_of_one_halves_to_one_not_zero(tmp_path, box, monkeypatch):
    calls = _launches(box, monkeypatch)
    monkeypatch.setattr(autosense, "place_llamacpp", lambda *a, **k: autosense.Placement(
        1, "gpu", 0, 0, 0, None, 0, 0, None, None))
    box.ready_script = [RuntimeError(DEAD), ["qwen"]]
    live = await _activate(tmp_path, _spec(tmp_path))
    assert _has_pair(_argv(calls, 1), "-c", "1") and live.window.asked == 1 and live.window.retried


async def test_the_retried_launch_reports_the_dropped_names_once(tmp_path, box, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_HOST", "0.0.0.0")
    _launches(box, monkeypatch)
    box.ready_script = [RuntimeError(DEAD), ["qwen"]]
    live = await _activate(tmp_path, _spec(tmp_path))
    assert live.dropped_env == ("LLAMA_ARG_HOST",)


# ------------------------------------------------------------------------------- the runtime in the messages


# the reference line every message that quotes the log ends with (D-22): where the raw lines are
_REF_LINE = r"\(see log {tmp}/vllm/serve\.log, ref [0-9a-f]{{8}}\)"


async def test_wait_ready_names_the_runtime_that_died(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    monkeypatch.setattr(server, "log_tail", lambda config_dir, lines=15, log=None: "the tail")
    with pytest.raises(RuntimeError) as named:
        await _REAL_WAIT_READY(BASE, config_dir=tmp_path, runtime="llamacpp")
    assert re.fullmatch(r"llama\.cpp exited during startup\. Log tail:\nthe tail\n" + _REF_LINE.format(tmp=tmp_path),
                        str(named.value))
    with pytest.raises(RuntimeError) as bare:
        await _REAL_WAIT_READY(BASE, config_dir=tmp_path)
    assert re.fullmatch(r"vLLM exited during startup\. Log tail:\nthe tail\n" + _REF_LINE.format(tmp=tmp_path),
                        str(bare.value))


async def test_wait_ready_names_the_runtime_that_timed_out(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "server_pid", lambda config_dir: 1)
    monkeypatch.setattr(server, "log_tail", lambda config_dir, lines=15, log=None: "the tail")
    with pytest.raises(TimeoutError, match=r"^llama\.cpp not ready after 0s"):
        await _REAL_WAIT_READY(BASE, config_dir=tmp_path, timeout_seconds=0, runtime="llamacpp")
    with pytest.raises(TimeoutError, match=r"^vLLM not ready after 0s"):
        await _REAL_WAIT_READY(BASE, config_dir=tmp_path, timeout_seconds=0)


# ------------------------------------------------------------------------------- D-37: the load failure IS the death

LOAD_FAILURE = (
    "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 16384.00 MiB on device 0: cudaMalloc failed: out of memory\n"
    "llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache\n")


def _dying_popen(monkeypatch, *, write_failure: bool) -> list[dict]:
    """A process stand-in that records its argv, writes (or not) a load failure into the serve.log
    handle `start_server` gave it, and is gone by the time anyone asks (server_pid -> None)."""
    launched: list[dict] = []

    class DyingPopen:
        def __init__(self, cmd, **kwargs):
            launched.append({"cmd": cmd})
            self.pid = 4242
            if write_failure:
                kwargs["stdout"].write(LOAD_FAILURE.encode())
                kwargs["stdout"].flush()

    monkeypatch.setattr(server.subprocess, "Popen", DyingPopen)
    monkeypatch.setattr(server, "server_pid", lambda config_dir: None)
    return launched


@pytest.mark.parametrize("write_failure", [True, False],
                         ids=["load-failure-in-the-log", "no-line-in-the-log"])
async def test_the_death_not_the_log_text_triggers_the_retry(tmp_path, box, monkeypatch, write_failure):
    monkeypatch.setattr(server, "wait_ready", _REAL_WAIT_READY)  # the real dead-process detection
    launched = _dying_popen(monkeypatch, write_failure=write_failure)
    with pytest.raises(RuntimeError) as dead:
        await _activate(tmp_path, _spec(tmp_path))
    assert len(launched) == 2
    assert _has_pair(launched[0]["cmd"], "-c", "14848") and _has_pair(launched[1]["cmd"], "-c", "7424")
    assert len(box.stops) == 2
    assert str(dead.value).startswith("llama.cpp exited during startup. Log tail:")
    # the abandon quotes the log's last lines: the failure when there was one, otherwise the launch line
    assert ("failed to initialize the context" in str(dead.value)) is write_failure
    assert "=== launch:" in str(dead.value)


async def test_load_failure_in_the_log_is_the_dead_process(tmp_path, box, monkeypatch):
    monkeypatch.setattr(server, "wait_ready", _REAL_WAIT_READY)
    launched = _dying_popen(monkeypatch, write_failure=True)
    with pytest.raises(RuntimeError) as dead:
        await _activate(tmp_path, _spec(tmp_path))
    text = str(dead.value)
    assert text.startswith("llama.cpp exited during startup. Log tail:")
    assert "failed to initialize the context" in text and "cudaMalloc failed" in text
    assert len(launched) == 2 and _has_pair(launched[1]["cmd"], "-c", "7424")
    assert [call[1] for call in box.stops] == ["binary", "binary"]
