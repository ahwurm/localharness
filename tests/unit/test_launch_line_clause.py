"""TUNE-10 (spec 20 §6.5, D-16, D-17, D-23): `start`'s launch line carries what the launch derived.

The existing parenthesis gains the derived clause (`kv q8_0 · mtp (1 layer)`), never the fitted `-c`; a
retried launch says what it dropped and how to keep the full window; a remembered shape says when it was
learned. The line is printed from `activate`'s `on_plan` callback, so it names the plan that really
starts. Nothing here starts a process or reads the host: `start_server`, `wait_ready`, free memory, the
weights and the probe are faked (`Box`), the consoles are recorders, the ledger is a file in `tmp_path`.
"""
from __future__ import annotations

import pytest

from localharness.cli import start_cmd
from localharness.cli.start_cmd import LaunchNotes, _start_managed_server
from localharness.config.models import AutosenseConfig
from localharness.provider import lifecycle, server, speed_stats
from localharness.provider.autosense import FreeMemory
from localharness.provider.tuning import NOTHING_DERIVED, Tuned, derived_clause
from tests.unit.test_launch_fit_retry import BASE, Box
from tests.unit.test_launch_plan import FLASH_NEXT, SWIFT, _gguf_spec, _vllm_spec

GIB = 2**30
KNOBS = AutosenseConfig()
PHRASE = "every interface, no key"
TAIL = "model load can take several minutes)..."
DEATH = "llama.cpp exited during startup. Log tail:\nfirst death"
MTP = Tuned(cache_type="q8_0", spec_type="draft-mtp", draft_p_min=0.6, mtp_layers=1)
UNDO = "server.speculative: off keeps the full window"


def _line(inside: str) -> str:
    return f"Managed llama.cpp is not running — starting it ({inside}{TAIL}"


# ------------------------------------------------------------------- the clause helper


def test_a_full_shape_names_the_kv_type_and_the_mtp_layers():
    assert derived_clause(MTP) == "kv q8_0 · mtp (1 layer)"
    assert derived_clause(Tuned(cache_type="q8_0", spec_type="draft-mtp", mtp_layers=2)) == "kv q8_0 · mtp (2 layers)"
    assert derived_clause(Tuned(cache_type="q8_0")) == "kv q8_0"
    assert derived_clause(NOTHING_DERIVED) == ""


def test_the_full_shape_clause_never_carries_the_fitted_window():
    assert "-c" not in derived_clause(MTP, context=65536)


def test_vllm_parsers_read_as_reasoning_and_tools():
    tuned = Tuned(reasoning_parser="qwen3", tool_call_parser="qwen3_coder")
    assert derived_clause(tuned) == "reasoning qwen3 · tools qwen3_coder"
    assert derived_clause(Tuned(tool_call_parser="llama3_json")) == "tools llama3_json"


def test_a_retry_after_a_death_names_what_it_dropped_and_how_to_keep_the_full_window():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="death", dropped=("draft-mtp",),
                            context=40000)
    assert clause == f"mtp off, -c 40000 (retried: the full shape died) · {UNDO}"


def test_a_retry_that_only_halved_the_window_has_no_mtp_undo():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="death", context=40000)
    assert clause == "-c 40000 (retried: the full shape died)"


PINNED = "server.speculative: off keeps drafting pinned off"


def test_a_retry_that_dropped_mtp_with_no_halved_window():
    """WR-05: no window was halved, so `off` keeps nothing but drafting off (no "full window")."""
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="death", dropped=("draft-mtp",))
    assert clause == f"mtp off (retried: the full shape died) · {PINNED}"


def test_a_boot_retry_that_halved_no_window_says_so():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="boot", dropped=("draft-mtp",),
                            halved=False)
    assert clause == f"retried at boot: mtp off · {PINNED}"


def test_a_remembered_shape_with_mtp_dropped_says_when_it_was_learned():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="remembered",
                            dropped=("draft-mtp",), remembered_at="2026-10-01T09:00:00+00:00")
    assert clause == "kv q8_0 · mtp off (remembered 2026-10-01)"


def test_a_remembered_shape_without_mtp_dropped_says_the_window_was_halved():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="remembered",
                            remembered_at="2026-10-01T09:00:00+00:00")
    assert clause == "kv q8_0 · window halved (remembered 2026-10-01)"


def test_the_remembered_date_is_sliced_to_ten_characters():
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="remembered", dropped=("draft-mtp",),
                            remembered_at="2026-10-01[bold]ZZZZZZ")
    assert clause == "mtp off (remembered 2026-10-01)"


@pytest.mark.parametrize("planted", ["\x1b[2J\x1b[H\x1b]0;x", "2026-10-01\x1b[2J", "\x07\x9b2026-10-01T",
                                     "‮2026-10-01", "\x00\r\n\t"])
def test_a_planted_ledger_date_reaches_the_line_as_printable_ascii_only(planted):
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="remembered", dropped=("draft-mtp",),
                            remembered_at=planted)
    assert clause.startswith("mtp off (remembered") and clause.endswith(")")
    assert all(" " <= c <= "~" or c == "·" for c in clause)


def test_a_date_that_filters_to_nothing_prints_no_date():
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="remembered", dropped=("draft-mtp",),
                            remembered_at="\x1b\x1b\x07")
    assert clause == "mtp off (remembered)"


def test_a_vllm_parser_retry_names_the_fields_that_pin_the_parsers_off():
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="death", dropped=("parsers",),
                            parser_fields=("reasoning_parser", "tool_call_parser"))
    assert clause == ("parsers off (retried: the full shape died) · server.reasoning_parser: off / "
                      "server.tool_call_parser: off keeps them pinned off")
    one = derived_clause(NOTHING_DERIVED, shape="retry", cause="death", dropped=("parsers",),
                         parser_fields=("tool_call_parser",),
                         undo_field="extra_endpoints.x.lifecycle.speculative")
    assert one.endswith("· extra_endpoints.x.lifecycle.tool_call_parser: off keeps them pinned off")


def test_a_boot_retry_uses_the_spec_wording():
    clause = derived_clause(Tuned(cache_type="q8_0"), shape="retry", cause="boot", dropped=("draft-mtp",))
    assert clause == f"retried at boot: mtp off, -c halved · {UNDO}"


def test_the_undo_field_is_the_callers():
    clause = derived_clause(NOTHING_DERIVED, shape="retry", cause="death", dropped=("draft-mtp",),
                            context=8, undo_field="extra_endpoints.x.lifecycle.speculative")
    assert clause.endswith("· extra_endpoints.x.lifecycle.speculative: off keeps the full window")


# ------------------------------------------------------------------- start's launch line


class _World:
    """The launch's consoles and `start_server` recorded in the one ordering log `Box` already keeps."""

    def __init__(self, monkeypatch, box: Box):
        self.box = box
        self.cmds: list[list[str]] = []

        def printer(tag):
            def _print(*a, **k):
                assert k.get("markup") is False
                box.events.append(f"{tag}:{a[0] if a else ''}")
            return _print

        def start(config_dir, cmd, **kw):
            self.cmds.append(list(cmd))
            box.events.append("start_server")
            return 4321

        monkeypatch.setattr(start_cmd.console, "print", printer("out"))
        monkeypatch.setattr(start_cmd.err_console, "print", printer("err"))
        monkeypatch.setattr(server, "server_pid", lambda _d: None)
        monkeypatch.setattr(server, "docker_container_running", lambda _n: False)
        monkeypatch.setattr(server, "start_server", start)

    @property
    def out(self) -> list[str]:
        return [e[4:] for e in self.box.events if e.startswith("out:")]

    @property
    def err(self) -> list[str]:
        return [e[4:] for e in self.box.events if e.startswith("err:")]


@pytest.fixture
def box(monkeypatch):
    world = Box(monkeypatch)
    world.mem = FreeMemory(40 * GIB, 48 * GIB, "nvidia-smi")
    world.weights = 18 * GIB
    return world


@pytest.fixture
def world(monkeypatch, box):
    return _World(monkeypatch, box)


def _argv_c(cmd: list[str]) -> int:
    return int(cmd[cmd.index("-c") + 1])


async def _start(tmp_path, spec, **kw):
    kw.setdefault("autosense", KNOBS)
    return await _start_managed_server(spec, tmp_path, BASE, kw.pop("launch_key", {}), **kw)


async def test_the_swift_preset_line_names_kv_and_mtp_before_the_server_starts(tmp_path, world):
    assert await _start(tmp_path, _gguf_spec(tmp_path, SWIFT)) is None
    assert world.out == ["Managed llama.cpp is not running — starting it "
                         "(kv q8_0 · mtp (1 layer); model load can take several minutes)..."]
    events = world.box.events
    assert events.index(f"out:{world.out[0]}") < events.index("start_server")


async def test_the_line_never_carries_the_fitted_window(tmp_path, world):
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT))
    window = _argv_c(world.cmds[0])
    assert window > 1000
    assert " -c" not in world.out[0] and str(window) not in world.out[0]


async def test_the_flash_next_preset_reads_kv_only(tmp_path, world):
    await _start(tmp_path, _gguf_spec(tmp_path, FLASH_NEXT))
    assert world.out == [_line("kv q8_0; ")]


async def test_a_vllm_binary_preset_reads_the_derived_parsers(tmp_path, world):
    await _start(tmp_path, _vllm_spec(tmp_path))
    assert world.out == ["Managed vLLM is not running — starting it "
                         "(reasoning qwen3 · tools qwen3_coder; model load can take several minutes)..."]


async def test_the_posture_phrase_stays_first(tmp_path, world):
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT, bind_all=True))
    assert world.out == [_line(f"{PHRASE}; kv q8_0 · mtp (1 layer); ")]


async def test_a_launch_with_nothing_derived_prints_todays_line(tmp_path, world):
    off = AutosenseConfig(kv_cache_type="f16", draft_mtp=False)
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT), autosense=off)
    assert world.out == [_line("")]
    world.box.events.clear()
    world.box.ready_script = [["qwen"]]
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT, bind_all=True), autosense=off)
    assert world.out == [_line(f"{PHRASE}; ")]


async def test_an_untuned_launch_prints_todays_line_with_no_clause(tmp_path, world):
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT), autosense=None)
    assert world.out == [_line("")]


# ------------------------------------------------------------------- the retry line


async def test_a_retried_launch_prints_the_retry_line_before_the_second_start(tmp_path, world, box):
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    notes = LaunchNotes(["a note line"])
    assert await _start(tmp_path, _gguf_spec(tmp_path, SWIFT), notes=notes) is None

    halved = _argv_c(world.cmds[1])
    assert halved == _argv_c(world.cmds[0]) // 2
    assert world.out == [
        _line("kv q8_0 · mtp (1 layer); "),
        f"Managed llama.cpp is not running — starting it (mtp off, -c {halved} (retried: the full shape "
        f"died) · {UNDO}; model load can take several minutes)..."]
    events = box.events
    second = f"out:{world.out[1]}"
    assert events.index("stop_server") < events.index(second) < len(events) - 1
    starts = [i for i, e in enumerate(events) if e == "start_server"]
    assert events.index(second) < starts[1] and events.index(f"out:{world.out[0]}") < starts[0]
    # the notes ride the first line only
    assert world.err == ["a note line"] and notes.printed is True


async def test_a_retried_launch_prints_the_decision_note_once(tmp_path, world, box):
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    spec = _gguf_spec(tmp_path, SWIFT, bind_all=True, require_api_key=True, api_key_file=str(tmp_path))
    await _start(tmp_path, spec)
    assert len(world.out) == 2 and len(world.err) == 1
    assert world.err[0].endswith("launched on 127.0.0.1 for this run")


async def test_a_retry_that_could_only_halve_the_window_names_no_undo(tmp_path, world, box):
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    await _start(tmp_path, _gguf_spec(tmp_path, FLASH_NEXT))
    halved = _argv_c(world.cmds[1])
    assert world.out[1] == _line(f"-c {halved} (retried: the full shape died); ")
    assert "server.speculative" not in world.out[1]


async def test_a_retry_of_a_record_context_names_no_window_and_pins_drafting_off(tmp_path, world, box):
    """WR-05 (R-10 ruling): the record's own `context:` is never halved, so the retry drops only the
    derived spec type and the line carries no `-c` and no "keeps the full window"."""
    box.ready_script = [RuntimeError(DEATH), ["qwen"]]
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT, context=65536))
    assert _argv_c(world.cmds[1]) == _argv_c(world.cmds[0]) == 65536
    assert "--spec-type" in world.cmds[0] and "--spec-type" not in world.cmds[1]
    assert world.out[1] == _line(f"mtp off (retried: the full shape died) · {PINNED}; ")


async def test_a_boot_plan_of_a_record_context_names_no_halved_window(tmp_path, box):
    spec = _gguf_spec(tmp_path, SWIFT, context=65536)
    sensed = lifecycle.sense_launch(spec, tmp_path, KNOBS)
    sensed = lifecycle.dataclasses.replace(sensed, prior_death=True)
    plan = lifecycle.plan_launch(spec, sensed, server.launch_decision(spec, False), autosense=KNOBS,
                                 api_key_held=False)
    assert plan.cause == "boot"
    assert start_cmd._plan_clause(plan) == f"retried at boot: mtp off · {PINNED}"


async def test_a_vllm_parser_retry_prints_its_line_before_the_second_start(tmp_path, world, box):
    box.ready_script = [RuntimeError("vLLM exited during startup. Log tail:\nx"), ["qwen"]]
    await _start(tmp_path, _vllm_spec(tmp_path))
    assert len(world.cmds) == 2 and "--reasoning-parser" not in world.cmds[1]
    assert world.out[1] == (
        "Managed vLLM is not running — starting it (parsers off (retried: the full shape died) · "
        "server.reasoning_parser: off / server.tool_call_parser: off keeps them pinned off; "
        "model load can take several minutes)...")


# ------------------------------------------------------------------- the remembered line


def _remember(tmp_path, spec, *, dropped, full_fit=10_000_000) -> str:
    path = speed_stats.default_speed_stats_path(tmp_path)
    key = lifecycle.record_key(spec)
    speed_stats.record_retry(path, key, full_fit=full_fit, free_bytes=24 * GIB, dropped=dropped,
                             binary=str(spec.binary), kv_cache_type="q8_0", draft_p_min=0.6)
    return speed_stats.last_retry(path, key).at[:10]


async def test_a_remembered_shape_with_mtp_dropped_reads_with_its_date(tmp_path, world):
    spec = _gguf_spec(tmp_path, SWIFT)
    date = _remember(tmp_path, spec, dropped=["draft-mtp"])
    await _start(tmp_path, spec)
    assert world.out == [_line(f"kv q8_0 · mtp off (remembered {date}); ")]
    assert "--spec-type" not in world.cmds[0]


async def test_a_remembered_shape_prints_its_clause_on_every_launch(tmp_path, world):
    """R-10c: the clause is a pure function of the plan; no on-change gate."""
    spec = _gguf_spec(tmp_path, SWIFT)
    date = _remember(tmp_path, spec, dropped=["draft-mtp"])
    await _start(tmp_path, spec)
    world.box.ready_script = [["qwen"]]
    await _start(tmp_path, spec)
    assert world.out == [_line(f"kv q8_0 · mtp off (remembered {date}); ")] * 2


async def test_a_remembered_window_only_shape_says_the_window_was_halved(tmp_path, world):
    spec = _gguf_spec(tmp_path, FLASH_NEXT)
    date = _remember(tmp_path, spec, dropped=[])
    await _start(tmp_path, spec)
    assert world.out == [_line(f"kv q8_0 · window halved (remembered {date}); ")]


# ------------------------------------------------------------------- the strategies that plan nothing


async def test_a_strategy_without_plans_launch_prints_before_activate_with_no_clause(tmp_path, monkeypatch):
    log: list[str] = []
    monkeypatch.setattr(start_cmd.console, "print", lambda *a, **k: log.append(f"out:{a[0]}"))
    monkeypatch.setattr(start_cmd.err_console, "print", lambda *a, **k: log.append(f"err:{a[0]}"))
    monkeypatch.setattr(server, "server_pid", lambda _d: None)
    monkeypatch.setattr(server, "docker_container_running", lambda _n: False)

    class Plain:
        def recovery_config_dir(self, config_dir):
            return config_dir

        async def activate(self, spec, config_dir, base_url, **kw):
            log.append(f"activate:{sorted(kw)}")
            return lifecycle.LiveEndpoint(base_url=base_url, served_models=[], handle=1)

    monkeypatch.setattr(lifecycle, "strategy_for", lambda spec, **kw: Plain())
    monkeypatch.setattr(start_cmd, "_managed_server_running", lambda *a, **k: False)
    notes = LaunchNotes(["a note"])
    await _start(tmp_path, _gguf_spec(tmp_path, SWIFT), notes=notes)
    assert log[:2] == [f"out:{_line('')}", "err:a note"]
    assert log[2].startswith("activate:") and "on_plan" not in log[2]
    assert notes.printed is True
