"""The first run is one question: every model found is listed best first, Enter sets them all up with the best
serving, and every record adopted is one whose bullet was printed before the question. Each GGUF is a real
header under a faked HOME; the interview is scripted, nothing reaches a terminal, a network or a process."""
from __future__ import annotations

import io
import os

import pytest
from rich.console import Console

import localharness.cli.init_cmd as init_cmd
from localharness.cli import interview, library
from localharness.cli.interview import Answer
from localharness.cli.workspace import shown_path
from localharness.config import trust
from localharness.provider import model_library
from tests.unit.interview_support import CANCEL, UNREACHED, scripted
from tests.unit.test_init_core_setup import _exited, _flat, _hermetic, _init  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import (  # noqa: F401  (fixtures and helpers)
    _model,
    _peers,
    _written,
    box,
)


def _first_run(box, **kw) -> library.Sync:
    box.cfg.mkdir(exist_ok=True)
    return library.sync(box.cfg, scope="all", person=kw.pop("person", True), first_run=True, **kw)


def _before_the_question(monkeypatch, answer: str = "all") -> tuple[io.StringIO, list[str]]:
    """What was printed when each question was asked, and a console that holds it."""
    buf = io.StringIO()
    monkeypatch.setattr(library, "console", Console(file=buf, width=300))
    printed: list[str] = []

    def ask(question, options, **_kw):
        printed.append(buf.getvalue())
        return Answer((answer,), False, True)

    monkeypatch.setattr(interview, "ask", ask)
    return buf, printed


def test_enter_sets_up_both_models_the_best_serving_and_the_other_a_peer(box):
    result = _init(box.cfg)

    _exited(result, 0)
    [asked] = box.asked.asked
    assert asked.question == "Set up these 2 models?" and asked.default == 0 and asked.cancel == "skip"
    assert [o.key for o in asked.options] == ["all", "choose", "skip"]
    assert _written(box.cfg)["server"]["model"] == str(box.qwen)
    assert [p["lifecycle"]["model"] for p in _peers(box.cfg)] == [str(box.swift)]
    assert not (box.cfg / "config.yaml.before-init").exists()


def test_every_adopted_record_is_one_whose_bullet_was_printed_before_the_question(box, monkeypatch):
    buf, printed = _before_the_question(monkeypatch)

    synced = _first_run(box)

    [before] = printed
    lines = [line for line in before.splitlines() if line.startswith("• ")]
    assert len(lines) == synced.shown == 2
    for model in (box.qwen, box.swift):
        assert shown_path(model) in before
    assert f"+{box.proj.name})" in before
    assert synced.records.confirmed and synced.records.block_in_receipt
    assert [s.model for s in (synced.records.server, *[p.lifecycle for p in synced.records.peers])] == [
        str(box.qwen), str(box.swift)]


def test_what_init_adopts_is_exactly_what_it_showed(box):
    result = _init(box.cfg)

    entries = trust.recorded_machine(box.cfg)["entries"]
    paths = {e["shown"] for e in entries if e["name"].endswith((".model", ".mmproj"))}
    assert paths == {str(box.qwen), str(box.swift), str(box.proj)}
    for path in (box.qwen, box.swift):
        assert shown_path(path) in result.output
    assert box.proj.name in result.output


def test_without_a_person_nothing_is_asked_and_no_library_file_is_written(box):
    synced = _first_run(box, person=False)

    assert box.asked.asked == [] and synced.shown == 2
    assert not synced.records.confirmed
    assert not model_library.library_path(box.cfg).exists()


def test_an_interview_that_could_not_be_asked_writes_the_records_unadopted(box, monkeypatch):
    scripted(monkeypatch, UNREACHED)

    result = _init(box.cfg)

    _exited(result, 0)
    assert library.NEXT_START_LINE in _flat(result)
    assert _written(box.cfg)["server"]["model"] == str(box.qwen) and len(_peers(box.cfg)) == 1
    assert trust.recorded_machine(box.cfg) is None
    assert not model_library.library_path(box.cfg).exists()


def test_a_cancel_writes_nothing_declines_nothing_and_says_so(box, monkeypatch):
    scripted(monkeypatch, CANCEL)

    result = _init(box.cfg)

    _exited(result, 0)
    assert library.SKIP_LINE in _flat(result)
    assert not (box.cfg / "config.yaml").exists() and not model_library.library_path(box.cfg).exists()


def test_a_root_holding_no_model_has_nothing_to_show(tmp_path, monkeypatch, fake_home):
    fake_home(tmp_path / "home", clear_overrides=False)
    (tmp_path / "cfg").mkdir()

    assert library.sync(tmp_path / "cfg", scope="all", person=True, first_run=True) == library.Sync()


def test_a_name_with_an_escape_byte_prints_escaped_on_the_bullet(box, monkeypatch):
    _model(box.root / "z\x1b[2Jz.gguf", mtime=3_000_000)
    buf, _ = _before_the_question(monkeypatch)

    _first_run(box)

    assert "\x1b" not in buf.getvalue() and "z\\x1b[2Jz" in buf.getvalue()


@pytest.mark.parametrize("gone", ["_discover_presets", "_found_models", "_peer_records", "_free_ports",
                                  "_no_binary_line", "PRIMARY_PORT"])
def test_init_cmd_has_no_discovery_of_its_own(gone):
    assert not hasattr(init_cmd, gone)
