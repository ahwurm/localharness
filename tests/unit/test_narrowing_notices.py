"""Narrowing warnings are notices that print once per change of the rejected value.

A project's config and agent files may ask for things only the machine may set. The loader refuses
them and keeps its warning texts (`harness_warnings`, `agent_warnings`); beside each text it now
records one `Notice` (`loader.notices`: the dotted key, the file, the repr of the rejected value and
the text). `start` subtracts the texts of the current notices from the warnings slot and prints the
unseen ones at the end of it, storing the set only when a person is at a terminal. A rejection made
while a built-in subagent is dispatched travels the bus once per sitting instead of the log.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from localharness.config import trust
from localharness.config.project_records import Notice, notice_value, read_notices
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
