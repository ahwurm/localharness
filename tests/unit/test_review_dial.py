"""The review level a person sets: the per-project store, the /review command and its completer."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.document import Document

from localharness.channels.terminal import SlashCommandCompleter
from localharness.cli.repl import (
    REVIEW_PICKER_PROMPT,
    REVIEW_SAVE_FAILED_TEMPLATE,
    REVIEW_SAVED_LINE,
    REVIEW_SESSION_ONLY_LINE,
    REVIEW_UNAVAILABLE_LINE,
    REVIEW_UNKNOWN_TEMPLATE,
)
from localharness.config import project_records as store
from localharness.orchestrator.task_flow.record import LIMITS, REVIEW_LEVELS
from tests.unit.test_repl_effort_cmd import Chan, MobileDouble, TerminalDouble
from tests.unit.test_repl_unknown_slash import _build_repl

INFO = {"style": "system.info"}
ERROR = {"style": "system.error"}
CEILING = LIMITS["revisions.reply"].ceiling


def put(config_dir, root, value):
    with store._store_transaction(config_dir) as data:
        store._entry_for_write(data, store._normalize_root(root))["review"] = value


def test_an_absent_entry_reads_as_the_fallback(tmp_path):
    assert store.read_review_level(tmp_path / "proj", 1, config_dir=tmp_path / "cfg") == 1


@pytest.mark.parametrize("stored, read", [(0, 0), (2, 2), (CEILING, CEILING)])
def test_a_stored_level_inside_the_row_is_read_and_only_a_stored_off_gives_zero(tmp_path, stored, read):
    put(tmp_path / "cfg", tmp_path / "proj", stored)
    assert store.read_review_level(tmp_path / "proj", 3, config_dir=tmp_path / "cfg") == read


@pytest.mark.parametrize("stored", [True, "2", -1, CEILING + 1, 2.0, None, [2]])
def test_a_stored_value_that_is_not_an_int_inside_the_row_falls_back(tmp_path, stored):
    put(tmp_path / "cfg", tmp_path / "proj", stored)
    assert store.read_review_level(tmp_path / "proj", 3, config_dir=tmp_path / "cfg") == 3


def test_a_store_that_cannot_be_read_falls_back_and_never_raises(tmp_path, monkeypatch):
    def locked(*args, **kwargs):
        raise store.StoreUnavailable("locked")

    monkeypatch.setattr(store, "_store_transaction", locked)
    assert store.read_review_level(tmp_path, 1, config_dir=tmp_path) == 1


def test_a_damaged_store_file_falls_back(tmp_path):
    (tmp_path / "cfg").mkdir()
    store.store_path(tmp_path / "cfg").write_bytes(b"\xff\x00: [")
    assert store.read_review_level(tmp_path / "proj", 1, config_dir=tmp_path / "cfg") == 1


def test_writing_a_level_keeps_the_entrys_other_records(tmp_path):
    root, cfg = tmp_path / "proj", tmp_path / "cfg"
    with store._store_transaction(cfg) as data:
        entry = store._entry_for_write(data, store._normalize_root(root))
        entry["instructions"], entry["notices"] = {"state": "yes"}, []
    store.write_review_level(root, 2, config_dir=cfg)
    with store._store_transaction(cfg) as data:
        assert store._entry(data, store._normalize_root(root)) == {"instructions": {"state": "yes"}, "notices": [], "review": 2}
    assert store.read_review_level(root, 1, config_dir=cfg) == 2


def repl_at(tmp_path, channel=None, level=1, boundary=True):
    channel = channel if channel is not None else Chan([])
    repl, agent, _ = _build_repl(channel)
    agent.reply_panel = SimpleNamespace(limits=SimpleNamespace(revisions={"reply": level}))
    repl._gate = SimpleNamespace(boundary=tmp_path / "proj" if boundary else None)
    repl._config_dir = tmp_path / "cfg"
    return repl, agent.reply_panel.limits.revisions, channel


def saved(tmp_path):
    return store.read_review_level(tmp_path / "proj", -1, config_dir=tmp_path / "cfg")


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("arg, level", [("2", 2), ("max", CEILING), ("off", 0)])
def test_review_sets_the_session_level_saves_it_per_project_and_says_so(tmp_path, arg, level):
    repl, revisions, channel = repl_at(tmp_path)
    assert run(repl._handle_slash(f"/Review {arg.upper()}")) is True
    assert revisions["reply"] == level and saved(tmp_path) == level
    ((text, meta),) = channel.sent
    assert meta == INFO and text.startswith(f"review: {arg} — ") and text.endswith(REVIEW_SAVED_LINE)
    assert (level == 0) == ("send the draft as written" in text)
    assert (level == CEILING) == ("until it passes, up to 10 checks" in text)


def test_review_one_names_the_default_and_the_unrechecked_last_fix(tmp_path):
    repl, _, channel = repl_at(tmp_path, level=3)
    run(repl._handle_review_cmd("1"))
    assert "check, then fix (default)" in channel.sent[0][0]


def test_review_show_prints_the_level_and_what_it_does_and_changes_nothing(tmp_path):
    repl, revisions, channel = repl_at(tmp_path, level=3)
    run(repl._handle_review_cmd("show"))
    assert channel.sent == [("review: 3 — check and fix, three times", INFO)]
    assert revisions["reply"] == 3 and saved(tmp_path) == -1


def test_an_unknown_level_names_the_levels_and_changes_nothing(tmp_path):
    repl, revisions, channel = repl_at(tmp_path)
    for arg in ("bogus", "11", "-1"):
        run(repl._handle_review_cmd(arg))
    assert channel.sent[0] == (REVIEW_UNKNOWN_TEMPLATE.format(arg="bogus"), ERROR)
    assert "off, 1, 2, 3, max" in channel.sent[0][0] and "/review show" in channel.sent[0][0]
    assert len(channel.sent) == 3 and revisions["reply"] == 1 and saved(tmp_path) == -1


def test_a_failing_save_keeps_the_session_change_and_says_saving_failed(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(store, "write_review_level", boom)
    repl, revisions, channel = repl_at(tmp_path)
    run(repl._handle_review_cmd("3"))
    assert revisions["reply"] == 3
    assert channel.sent[0] == (REVIEW_SAVE_FAILED_TEMPLATE.format(exc="disk full"), ERROR)
    assert REVIEW_SAVED_LINE not in channel.sent[-1][0]


def test_with_no_project_the_level_holds_for_the_session_only(tmp_path):
    repl, revisions, channel = repl_at(tmp_path, boundary=False)
    run(repl._handle_review_cmd("2"))
    assert revisions["reply"] == 2 and saved(tmp_path) == -1
    assert channel.sent[0][0].endswith(REVIEW_SESSION_ONLY_LINE)


def test_without_a_reply_panel_review_is_unavailable(tmp_path):
    repl, _, channel = repl_at(tmp_path)
    repl._agent.reply_panel = None
    run(repl._handle_review_cmd("2"))
    assert channel.sent == [(REVIEW_UNAVAILABLE_LINE, INFO)]


def test_bare_review_without_a_picker_lists_the_levels_with_the_current_bracketed(tmp_path):
    repl, revisions, channel = repl_at(tmp_path, level=2)
    run(repl._handle_review_cmd(""))
    assert channel.sent[0][0].startswith("review: off  1  [2]  3  max") and revisions["reply"] == 2


def test_the_terminal_picker_is_a_row_of_levels_with_the_highlighted_one_described_below(tmp_path, monkeypatch):
    channel = TerminalDouble("3")
    repl, revisions, _ = repl_at(tmp_path, channel)
    monkeypatch.setattr(repl, "_use_input_box", lambda: True)
    run(repl._handle_review_cmd(""))
    ((options, current, notes),) = channel.picker_calls
    assert options == list(REVIEW_LEVELS) and current == "1"
    assert notes[0].endswith("send the draft as written") and "(default)" in notes[1]
    assert revisions["reply"] == 3 and saved(tmp_path) == 3


def test_the_phone_picker_carries_the_prompt_and_none_changes_nothing(tmp_path):
    channel = MobileDouble("max")
    repl, revisions, _ = repl_at(tmp_path, channel)
    run(repl._handle_review_cmd(""))
    assert channel.picker_calls[0][2] == REVIEW_PICKER_PROMPT and revisions["reply"] == CEILING
    none = MobileDouble(None)
    repl, revisions, _ = repl_at(tmp_path, none)
    run(repl._handle_review_cmd(""))
    assert revisions["reply"] == 1 and none.sent == []


def completions(text):
    return [c.text for c in SlashCommandCompleter().get_completions(Document(text, len(text)), None)]


def test_the_completer_offers_the_levels_and_show_after_review():
    assert completions("/review ") == [*REVIEW_LEVELS, "show"]
    assert completions("/review m") == ["max"] and completions("/Review S") == ["show"]
    assert completions("/review max ") == [] and completions("/review z") == []
