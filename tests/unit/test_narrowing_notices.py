"""Narrowing warnings are notices that print once per change of the rejected value.

A project's config and agent files may ask for things only the machine may set. The loader refuses
them and keeps its warning texts (`harness_warnings`, `agent_warnings`); beside each text it now
records one `Notice` (`loader.notices`: the dotted key, the file, the repr of the rejected value and
the text). `start` subtracts the texts of the current notices from the warnings slot and prints the
unseen ones at the end of it, storing the set only when a person is at a terminal. A rejection made
while a built-in subagent is dispatched travels the bus once per sitting instead of the log.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pytest
import yaml

from localharness.agent import subagent as sub
from localharness.agent.permissions import PermissionEvaluator
from localharness.agent.subagent import build_explore_config, make_explore_agent_runner
from localharness.cli import start_cmd
from localharness.config import trust
from localharness.config.loader import ConfigLoader
from localharness.config.project_records import (
    DispatchNoticeSink,
    Notice,
    notice_value,
    read_notices,
)
from localharness.core.events import NarrowingNotice
from localharness.tools.builtin import register_builtin_tools
from localharness.tools.registry import ToolRegistry
from tests.unit.test_floor_machine_level import _layers, _write
from tests.unit.test_instructions_start_hook import _capture, _tty
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _drive,
    _install_recorders,
    _workspace_start,
)

EVIL = "http://evil.test/v1"
KNOB = "channels.unattended_max_actions"


def _dropped(key: str, file: Path) -> str:
    return f"ignoring {key} in {file}: only the global config may set it"


def _lower(key: str, file) -> str:
    return f"ignoring {key} in {file}: a project may only lower it"


# ------------------------------------------------------------------ the loader: one Notice per narrowing


def test_a_workspace_provider_address_is_one_notice_and_the_same_warning(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg={"provider": {"base_url": EVIL}})
    loader.load_harness()

    text = _dropped("provider.base_url", ws / "config.yaml")
    assert loader.notices == [
        Notice("narrowing", "provider.base_url", str(ws / "config.yaml"), notice_value(EVIL), text)]
    assert loader.harness_warnings == [text]  # D-89: the text lists did not move


def test_the_notice_names_the_file_that_set_it(tmp_path):
    loader, ws = _layers(tmp_path, ws_over={"channels": {"unattended_max_actions": 5000}})
    loader.load_harness()

    [notice] = loader.notices
    assert (notice.subject, notice.file, notice.value) == (KNOB, str(ws / "overrides.yaml"), "5000")


def test_a_changed_value_is_a_different_notice(tmp_path):
    first, _ = _layers(tmp_path / "a", ws_cfg={"channels": {"unattended_max_actions": 5000}})
    second, _ = _layers(tmp_path / "b", ws_cfg={"channels": {"unattended_max_actions": 9000}})
    first.load_harness()
    second.load_harness()

    [a], [b] = first.notices, second.notices
    assert a.subject == b.subject == KNOB
    assert (a.value, b.value) == ("5000", "9000")


def test_a_huge_rejected_value_is_cut_in_the_notice(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg={"provider": {"base_url": "http://x/" + "a" * 5000}})
    loader.load_harness()

    [notice] = loader.notices
    assert len(notice.value) == 200 and notice.value.endswith("…")


def test_loading_twice_records_each_narrowing_once(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg={"provider": {"base_url": EVIL}})
    loader.load_harness()
    loader._harness_cache = None
    loader.load_harness()

    assert len(loader.notices) == 1


def test_a_lower_only_harness_value_is_a_notice(tmp_path):
    key = "org.children_budget.explore.actions"
    loader, ws = _layers(tmp_path, g_cfg={"org": {"children_budget": {"explore": {"actions": 10}}}},
                         ws_cfg={"org": {"children_budget": {"explore": {"actions": 99}}}})
    loader.load_harness()

    [notice] = loader.notices
    assert (notice.subject, notice.value) == (key, "99")
    assert loader.harness_warnings == [_lower(key, ws / "config.yaml")]
    assert notice.text == loader.harness_warnings[0]


def test_a_section_that_is_not_a_mapping_is_a_notice(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg={"org": {"context": 5}})
    loader.load_harness()

    assert [n.subject for n in loader.notices] == ["org.context"]
    assert loader.notices[0].text == _lower("org.context", ws / "config.yaml")


def test_a_null_lower_only_value_is_dropped_without_a_notice(tmp_path):
    loader, _ = _layers(tmp_path, ws_cfg={"org": {"children_budget": {"explore": {"actions": None}}}})
    loader.load_harness()

    assert loader.notices == [] and loader.harness_warnings == []


def test_the_peer_shadow_line_is_plain_text(tmp_path):
    peer = {"name": "a", "base_url": "http://127.0.0.1:8001/v1", "provider_type": "llamacpp"}
    other = {**peer, "name": "c"}
    loader, _ = _layers(tmp_path, g_cfg={"extra_endpoints": [peer]},
                        g_over={"extra_endpoints": [other]})
    loader.load_harness()

    assert loader.harness_warnings == ["overrides.yaml extra_endpoints replaces the 1 peers in config.yaml"]
    assert loader.notices == []


def _agent_file(base: Path, name: str, data: dict) -> Path:
    path = base / "agents" / f"{name}.yaml"
    _write(path, {"name": name, "role": "r", **data})
    return path


def test_a_workspace_agent_file_choosing_the_embedding_model_is_a_notice(tmp_path):
    loader, ws = _layers(tmp_path)
    path = _agent_file(ws, AGENT, {"memory": {"embedding_model": "./evil-model"}})
    loader.load_agent(AGENT)

    [notice] = loader.notices
    assert (notice.subject, notice.file) == ("memory.embedding_model", str(path))
    assert notice.text == _dropped("memory.embedding_model", path)
    assert loader.agent_warnings == [notice.text]


def test_a_lower_only_budget_raise_in_an_agent_file_is_a_notice(tmp_path):
    loader, ws = _layers(tmp_path)
    _agent_file(tmp_path / "global", AGENT, {"permissions": {"budget": {"max_actions": 20}}})
    path = _agent_file(ws, AGENT, {"permissions": {"budget": {"max_actions": 25}}})
    loader.load_agent(AGENT)

    [notice] = loader.notices
    assert (notice.subject, notice.file, notice.value) == (
        "permissions.budget.max_actions", str(path), "25")
    assert notice.text in loader.agent_warnings


def test_a_workspace_stuck_detector_in_an_agent_file_is_a_notice(tmp_path):
    loader, ws = _layers(tmp_path)
    path = _agent_file(ws, AGENT, {"stuck_detector": {"escalation_threshold": 50}})
    loader.load_agent(AGENT)

    [notice] = loader.notices
    assert (notice.subject, notice.file) == ("stuck_detector", str(path))
    assert notice.text in loader.agent_warnings


def test_a_plain_text_agent_warning_makes_no_notice(tmp_path):
    loader, ws = _layers(tmp_path)
    target = tmp_path / "elsewhere"
    target.mkdir()
    (ws / "agents").mkdir()
    (ws / "agents" / "linked.yaml").symlink_to(target / "linked.yaml")
    _write(target / "linked.yaml", {"name": "linked", "role": "r"})
    loader._find_file("agents", "linked")

    assert loader.agent_warnings and loader.notices == []


# ------------------------------------------------------------------ start: print on change


def _config_with(ws: Path, data: dict) -> None:
    (ws / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _summary_text(buf) -> str:
    """The one startup summary line a terminal start printed into the capture buffer."""
    text = buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    lines = [ln for ln in text.splitlines() if "s startup)" in ln]
    assert len(lines) == 1, text
    return lines[0]


KEY = "instructions.children"


def _text_for(ws: Path) -> str:
    return _dropped(KEY, ws / "config.yaml")


async def test_a_narrowing_prints_at_the_first_terminal_start_and_not_the_second(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    first = _summary_text(buf)
    await _drive()
    second = _summary_text(buf)

    assert first.count(_text_for(ws)) == 1
    assert _text_for(ws) not in second
    stored = read_notices(ws.parent.resolve())
    assert [n.subject for n in stored] == [KEY]


async def test_the_text_sits_at_the_end_of_the_slot_after_every_other_warning(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    monkeypatch.setattr("localharness.config.cleartext.cleartext_warnings",
                        lambda harness: ["PLAIN WARNING"])
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    first = _summary_text(buf)
    await _drive()
    second = _summary_text(buf)

    assert first.index("PLAIN WARNING") < first.index(_text_for(ws))
    assert first.rstrip().endswith(_text_for(ws) + "]")
    assert "PLAIN WARNING" in second and _text_for(ws) not in second  # a plain warning keeps its place


async def test_a_changed_rejected_value_prints_again(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    text = _dropped(KNOB, ws / "config.yaml")
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    _config_with(ws, {"channels": {"unattended_max_actions": 5000}})
    await _drive()
    one = _summary_text(buf)
    await _drive()
    two = _summary_text(buf)
    _config_with(ws, {"channels": {"unattended_max_actions": 9000}})
    await _drive()
    three = _summary_text(buf)

    assert (text in one, text in two, text in three) == (True, False, True)


async def test_a_notice_that_disappears_and_returns_prints_again(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    text = _text_for(ws)
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    _config_with(ws, {"instructions": {"children": False}})
    await _drive()
    shown = [text in _summary_text(buf)]
    (ws / "config.yaml").unlink()
    await _drive()
    shown.append(text in _summary_text(buf))
    _config_with(ws, {"instructions": {"children": False}})
    await _drive()
    shown.append(text in _summary_text(buf))

    assert shown == [True, False, True]


async def test_off_the_terminal_the_text_prints_every_start_and_nothing_is_stored(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    await _drive()

    summaries = [p for p in printed if "s startup)" in p]
    assert len(summaries) == 2 and all(_text_for(ws) in s for s in summaries)
    assert read_notices(ws.parent.resolve()) == frozenset()


async def test_no_input_at_a_tty_stores_nothing(tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _start_async(None, False, False, None, no_input=True)
    await _start_async(None, False, False, None, no_input=True)

    assert read_notices(ws.parent.resolve()) == frozenset()
    assert buf.getvalue().count(_text_for(ws)) == 2


async def test_with_no_boundary_the_slot_is_todays_and_nothing_is_written(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    monkeypatch.setattr("localharness.agent.gate.derive_session_boundary", lambda **kw: None)
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    await _drive()

    assert buf.getvalue().count(_text_for(ws)) == 2
    assert read_notices(ws.parent.resolve()) == frozenset()


async def test_an_oversized_file_prints_the_line_not_the_entity_then_the_entity_not_the_line(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = ws.parent
    (proj / "CLAUDE.md").write_text("x" * 300_000, encoding="utf-8")
    trust.record_trust(proj, True)
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    first = _summary_text(buf)
    await _drive()
    second = _summary_text(buf)

    assert "over the" in first and "instructions.max_ratio" in first and "not used" in first
    assert "(too large)" not in first
    assert "instructions.max_ratio" not in second
    assert "CLAUDE.md (too large)" in second


async def test_a_notice_equal_to_a_text_already_in_the_slot_is_not_added_twice(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _config_with(ws, {"instructions": {"children": False}})
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()

    assert _summary_text(buf).count("ignoring instructions.children") == 1


# ------------------------------------------------------------------ the built-in overlays: notices, not logs


def _overlay_world(tmp_path: Path, ws_explore: dict, g_explore: dict | None = None):
    loader, ws = _layers(tmp_path)
    _write(ws / "agents" / "explore.yaml", ws_explore)
    if g_explore is not None:
        _write(tmp_path / "global" / "agents" / "explore.yaml", g_explore)
    return loader, ws


@pytest.fixture
def last_resort(monkeypatch):
    """Record anything that would reach the terminal through `logging.lastResort`."""
    handled: list[logging.LogRecord] = []

    class _Recorder:
        level = logging.WARNING

        def handle(self, record):
            handled.append(record)

    monkeypatch.setattr(logging, "lastResort", _Recorder())
    return handled


def test_a_pinned_kill_file_is_a_notice_and_not_a_log_record(tmp_path, caplog, last_resort):
    loader, ws = _overlay_world(
        tmp_path, {"permissions": {"budget": {"kill_file": str(tmp_path / "never-pressed")}}})
    got: list[Notice] = []

    with caplog.at_level(logging.DEBUG):
        loader.overlay_builtin_config("explore", build_explore_config("explore"), notices=got.append)

    [notice] = got
    assert notice.kind == "narrowing" and notice.subject == "permissions.budget.kill_file"
    assert notice.file == str(ws / "agents" / "explore.yaml")
    assert notice.value == notice_value(str(tmp_path / "never-pressed"))
    assert not caplog.records and not last_resort
    assert loader.notices == []  # handed to the caller, not kept


def test_without_a_reporter_the_notice_lands_on_the_loader_once(tmp_path, caplog, last_resort):
    loader, _ = _overlay_world(
        tmp_path, {"permissions": {"budget": {"kill_file": str(tmp_path / "never-pressed")}}})

    with caplog.at_level(logging.DEBUG):
        for _ in range(3):
            loader.overlay_builtin_config("explore", build_explore_config("explore"))

    assert [n.subject for n in loader.notices] == ["permissions.budget.kill_file"]
    assert not caplog.records and not last_resort


@pytest.mark.parametrize("ws_file, subject", [
    ({"stuck_detector": {"escalation_threshold": 50}}, "stuck_detector"),
    ({"permissions": {"budget": {"max_actions": 99}}}, "permissions.budget.max_actions"),
    ({"permissions": {"budget": {"max_duration_minutes": 9999.0}}},
     "permissions.budget.max_duration_minutes"),
    ({"permissions": {"budget": {"max_tool_calls": 50}}}, "permissions.budget.max_tool_calls"),
    ({"context": {"tool_output_ratio": 0.9}}, "context.tool_output_ratio"),
])
def test_every_overlay_rejection_is_one_notice(tmp_path, caplog, last_resort, ws_file, subject):
    loader, ws = _overlay_world(tmp_path, ws_file)
    got: list[Notice] = []

    with caplog.at_level(logging.DEBUG):
        loader.overlay_builtin_config("explore", build_explore_config("explore"), notices=got.append)

    assert [n.subject for n in got] == [subject]
    assert got[0].file == str(ws / "agents" / "explore.yaml")
    assert repr(str(ws / "agents" / "explore.yaml")) in got[0].text
    assert not caplog.records and not last_resort


def test_a_lowering_overlay_reports_nothing(tmp_path):
    loader, _ = _overlay_world(tmp_path, {"permissions": {"budget": {"max_actions": 3}}})
    got: list[Notice] = []

    cfg = loader.overlay_builtin_config("explore", build_explore_config("explore"), notices=got.append)

    assert cfg.permissions.budget.max_actions == 3 and got == []


# ------------------------------------------------------------------ the dispatch sink


class _Bus:
    def __init__(self):
        self.events: list = []

    async def publish(self, event):
        self.events.append(event)
        return event


def _notice(subject="permissions.budget.max_actions", file="/p/.localharness/agents/explore.yaml",
            value="99"):
    return Notice("narrowing", subject, file, value,
                  f"ignoring {subject} in {file!r}: a project may only lower it")


async def test_the_sink_publishes_only_what_start_did_not_see(tmp_path):
    seen, fresh = _notice(value="50"), _notice(value="99")
    sink = DispatchNoticeSink([seen], global_dir=tmp_path)
    bus = _Bus()

    sink.record(seen)
    await sink.flush(bus, agent_id="root", session_id="s1")
    assert bus.events == []

    sink.record(fresh)
    await sink.flush(bus, agent_id="root", session_id="s1")

    [event] = bus.events
    accept = str(tmp_path / "agents" / "explore.yaml")
    assert isinstance(event, NarrowingNotice)
    assert (event.key, event.file, event.accept_file) == (fresh.subject, fresh.file, accept)
    assert (event.agent_id, event.session_id) == ("root", "s1")
    assert event.text == f"{fresh.text} — {accept!r} would accept it"


async def test_the_sink_tells_once_per_sitting(tmp_path):
    fresh = _notice()
    sink = DispatchNoticeSink([], global_dir=tmp_path)
    bus = _Bus()

    for _ in range(3):
        sink.record(fresh)
        await sink.flush(bus, agent_id="root", session_id="s1")

    assert len(bus.events) == 1


async def test_a_key_no_door_accepts_names_no_file(tmp_path):
    note = _notice(subject="permissions.budget.max_tool_calls", value="50")
    sink = DispatchNoticeSink([], global_dir=tmp_path)
    bus = _Bus()

    sink.record(note)
    await sink.flush(bus, agent_id="root", session_id="s1")

    [event] = bus.events
    assert event.accept_file == "" and event.text == note.text


async def test_a_value_outside_an_agents_folder_points_at_the_global_config(tmp_path):
    note = _notice(subject="context.tool_output_ratio", file="/p/.localharness/config.yaml")
    sink = DispatchNoticeSink([], global_dir=tmp_path)
    bus = _Bus()

    sink.record(note)
    await sink.flush(bus, agent_id="root", session_id="s1")

    assert bus.events[0].accept_file == str(tmp_path / "config.yaml")


async def test_the_notice_reaches_a_subscriber_through_a_real_bus(tmp_path, bus):
    heard: list = []

    async def _handler(event):
        heard.append(event)

    bus.subscribe(NarrowingNotice, _handler)
    sink = DispatchNoticeSink([], global_dir=tmp_path)
    sink.record(_notice())

    await sink.flush(bus, agent_id="root", session_id="s1")

    assert [type(e).__name__ for e in heard] == ["NarrowingNotice"]


# ------------------------------------------------------------------ the runner awaits an async override


@pytest.fixture
def recorded_loops(monkeypatch):
    calls: list[dict] = []

    class _RecordingLoop:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.current_session_id = None

        def action_limit(self):
            return 0

        def time_limit(self):
            return 0.0

        async def run_turn(self, task, *a, **k):
            return ""

    monkeypatch.setattr("localharness.agent.loop.AgentLoop", _RecordingLoop)
    return calls


async def _runner(bus, **kw):
    reg = ToolRegistry()
    await register_builtin_tools(reg)
    return make_explore_agent_runner(
        llm=object(), bus=bus, base_registry=reg, permission_evaluator=PermissionEvaluator(),
        get_parent_session_id=lambda: "parent-sess", **kw)


async def test_an_async_override_is_awaited_and_its_config_is_dispatched(
        recorded_loops, bus, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))
    ran: list[str] = []

    async def _override(name, base):
        ran.append(name)
        return base.model_copy(update={"role": "AWAITED ROLE"})

    runner = await _runner(bus, load_builtin_override=_override)
    await runner("explore", "look")

    assert ran == ["explore"]
    assert "AWAITED ROLE" in recorded_loops[0]["config"].role


async def test_a_plain_override_still_works(recorded_loops, bus, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_VERIFICATION_LEDGER_DIR", str(tmp_path / "ledger"))
    runner = await _runner(
        bus, load_builtin_override=lambda n, base: base.model_copy(update={"role": "PLAIN ROLE"}))
    await runner("explore", "look")

    assert "PLAIN ROLE" in recorded_loops[0]["config"].role


def test_nothing_schedules_the_publish_in_the_background():
    import localharness.config.project_records as records

    for module in (sub, start_cmd, records):
        source = inspect.getsource(module)
        assert "ensure_future" not in source and "create_task(_dispatch_sink" not in source


# ------------------------------------------------------------------ start: the overlays join the start set


async def test_a_built_in_overlay_note_is_in_the_first_start_and_not_told_at_dispatch(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(ws.parent, True)
    _write(ws / "agents" / "explore.yaml",
           {"name": "explore", "role": "r", "permissions": {"budget": {"max_actions": 99}}})
    buf = _tty(monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()
    first = _summary_text(buf)
    await _drive()
    second = _summary_text(buf)

    note = "ignoring permissions.budget.max_actions in"
    assert note in first and note not in second

    # The runner's override is start's sink: a dispatch tells the bus nothing the start printed,
    # and one new value written since is told once.
    kwargs = _one_runner(rec)
    override, run_bus = kwargs["load_builtin_override"], kwargs["bus"]
    assert inspect.iscoroutinefunction(override)
    heard: list = []

    async def _hear(event):
        heard.append(event)

    run_bus.subscribe(NarrowingNotice, _hear)
    base = build_explore_config("explore")
    await override("explore", base)
    assert heard == []

    _write(ws / "agents" / "explore.yaml",
           {"name": "explore", "role": "r", "permissions": {"budget": {"max_actions": 77}}})
    await override("explore", base)
    await override("explore", base)
    [event] = heard
    assert event.key == "permissions.budget.max_actions"
    assert event.accept_file == str(_g / "agents" / "explore.yaml")
    assert event.text.endswith(f"{event.accept_file!r} would accept it")


def _one_runner(rec) -> dict:
    assert rec["runner"], "the runner was never built - the patch did not bite"
    return rec["runner"][-1]
