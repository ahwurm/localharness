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
from tests.unit import test_init_discovery_records as world
from tests.unit.test_init_discovery_records import _model, _peers, _written

box = world.box


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


# ------------------------------------------------------------------ the other answers


@pytest.fixture
def trio(box):
    """Qwen-27B (A, the best), Swift-27B-Q8 (B) and Third-7B (C), llama-server on PATH, a scripted interview."""
    box.third = _model(box.root / "Third-7B.gguf", embedding=4096, mtime=500_000)
    box.keys = {"A": _real(box.qwen), "B": _real(box.swift), "C": _real(box.third)}
    return box


def _real(path) -> str:
    return os.path.realpath(str(path))


def _library(box) -> dict:
    return model_library.Library.load(model_library.library_path(box.cfg)).entries


def _nothing_written(box) -> None:
    assert not (box.cfg / "config.yaml").exists() and not (box.cfg / "overrides.yaml").exists()
    assert not model_library.library_path(box.cfg).exists()


def _served(box) -> tuple[str, list[str]]:
    return _written(box.cfg)["server"]["served_name"], [p["name"] for p in _peers(box.cfg)]


def test_choose_which_pre_selects_every_file_and_asks_which_serves_first(trio, monkeypatch):
    k = trio.keys
    script = scripted(monkeypatch, "choose", {k["A"], k["C"]}, k["C"])

    _exited(_init(trio.cfg), 0)

    first, which, serve = script.asked
    assert first.question == "Set up these 3 models?"
    assert which.question == "Set up which?" and which.multi and which.default == set(k.values())
    assert [o.key for o in which.options] == [k["A"], k["B"], k["C"]] and which.cancel == "cancel"
    assert serve.question == "Serve which first?" and not serve.multi and serve.default == 0
    assert [o.key for o in serve.options] == [k["A"], k["C"]]
    assert _served(trio) == ("Third-7B", ["Qwen-27B"])
    assert [e.declined for r, e in _library(trio).items() if r == k["B"]] == [True]
    assert "Swift" not in (trio.cfg / "config.yaml").read_text() + (trio.cfg / "overrides.yaml").read_text()


def test_one_file_selected_asks_no_order_and_declines_the_others(trio, monkeypatch):
    k = trio.keys
    script = scripted(monkeypatch, "choose", {k["B"]})

    _exited(_init(trio.cfg), 0)

    assert [a.question for a in script.asked] == ["Set up these 3 models?", "Set up which?"]
    assert _served(trio) == ("Swift-27B-Q8", [])
    assert {r for r, e in _library(trio).items() if e.declined} == {k["A"], k["C"]}


def test_nothing_selected_declines_all_names_the_library_and_writes_no_config(trio, monkeypatch):
    k = trio.keys
    scripted(monkeypatch, "choose", set())

    result = _init(trio.cfg)

    _exited(result, 0)
    assert f"Nothing set up — 3 declined, remembered in {shown_path(model_library.library_path(trio.cfg))}" \
        in _flat(result)
    assert "run localharness start again to choose" in _flat(result)
    assert not (trio.cfg / "config.yaml").exists()
    assert {r for r, e in _library(trio).items() if e.declined} == set(k.values())


def test_a_rerun_after_declining_everything_offers_choose_and_skip_only(trio, monkeypatch):
    scripted(monkeypatch, "choose", set())
    _init(trio.cfg)
    script = scripted(monkeypatch, "skip")

    result = _init(trio.cfg)

    _exited(result, 0)
    [asked] = script.asked
    assert [o.key for o in asked.options] == ["choose", "skip"] and asked.default == 0
    assert result.output.count(library.DECLINED_TAG) == 3
    assert not (trio.cfg / "config.yaml").exists()


def test_without_a_person_everything_declined_sets_nothing_up_and_exits_zero(trio, monkeypatch):
    scripted(monkeypatch, "choose", set())
    _init(trio.cfg)

    result = _init(trio.cfg, "--no-input")

    _exited(result, 0)
    assert not (trio.cfg / "config.yaml").exists() and "declined" in _flat(result)


def test_all_sets_up_only_the_files_not_declined_earlier(trio, monkeypatch):
    k = trio.keys
    scripted(monkeypatch, "choose", {k["A"], k["B"]}, k["A"])
    _exited(_init(trio.cfg), 0)
    for name in ("config.yaml", "overrides.yaml"):
        (trio.cfg / name).unlink()
    os.utime(trio.third, (9_000_000, 9_000_000))
    script = scripted(monkeypatch, "all")

    result = _init(trio.cfg)

    _exited(result, 0)
    [asked] = script.asked
    assert asked.question == "Set up these 3 models?"
    assert asked.options[0].label == "All, serving Qwen-27B first"
    assert asked.options[0].hint == library.ALL_HINT_EXCEPT_DECLINED.format(best="Qwen-27B")
    assert result.output.count(library.DECLINED_TAG) == 1
    assert _served(trio) == ("Qwen-27B", ["Swift-27B-Q8"])
    assert "Third" not in (trio.cfg / "config.yaml").read_text() + (trio.cfg / "overrides.yaml").read_text()
    assert [r for r, e in _library(trio).items() if e.declined] == [k["C"]]


def test_one_model_asks_set_it_up_and_the_later_line_has_no_model_clause(box, monkeypatch):
    box.swift.unlink()
    box.proj.unlink()
    script = scripted(monkeypatch, "set-up")

    result = _init(box.cfg)

    _exited(result, 0)
    [asked] = script.asked
    assert asked.question == "Set up Qwen-27B?" and asked.default == 0 and asked.cancel == "not-now"
    assert [o.label for o in asked.options] == ["Set it up", "Not now"]
    assert _written(box.cfg)["server"]["served_name"] == "Qwen-27B" and not (box.cfg / "overrides.yaml").exists()
    assert library.LATER_LINE in _flat(result) and "/model" not in _flat(result).split("Later:")[1].split("★")[0]


def test_not_now_on_one_model_writes_nothing_and_says_so(box, monkeypatch):
    box.swift.unlink()
    box.proj.unlink()
    scripted(monkeypatch, "not-now")

    result = _init(box.cfg)

    _exited(result, 0)
    assert library.SKIP_LINE in _flat(result)
    _nothing_written(box)


@pytest.mark.parametrize("answers", [["skip"], [CANCEL], ["choose", CANCEL], ["choose", "ALL_THREE", CANCEL]],
                         ids=["skip", "cancel", "cancel-at-which", "cancel-at-order"])
def test_skip_and_a_cancel_at_any_step_write_nothing_and_decline_nothing(trio, monkeypatch, answers):
    scripted(monkeypatch, *[set(trio.keys.values()) if a == "ALL_THREE" else a for a in answers])

    result = _init(trio.cfg)

    _exited(result, 0)
    assert library.SKIP_LINE in _flat(result)
    _nothing_written(trio)


def test_an_interview_lost_midway_writes_the_records_unadopted(trio, monkeypatch):
    scripted(monkeypatch, "choose", UNREACHED)

    result = _init(trio.cfg)

    _exited(result, 0)
    assert library.NEXT_START_LINE in _flat(result)
    assert _served(trio)[0] == "Qwen-27B" and trio.cfg.joinpath("config.yaml").exists()
    assert trust.recorded_machine(trio.cfg) is None and not model_library.library_path(trio.cfg).exists()


def test_a_preset_naming_a_found_file_serves_it_first_with_no_order_step(trio, monkeypatch):
    k = trio.keys
    script = scripted(monkeypatch, "choose", {k["A"], k["B"]})

    out = init_cmd.core_setup(str(trio.cfg), endpoint=None, model=None, force=False, interactive=True,
                              preset="swift-27b-q8")

    assert out.server_ready
    assert [a.question for a in script.asked] == ["Set up these 3 models?", "Set up which?"]
    assert _served(trio) == ("Swift-27B-Q8", ["Qwen-27B"])


def test_a_preset_changes_the_best_the_all_answer_names(trio, monkeypatch):
    script = scripted(monkeypatch, "all")

    init_cmd.core_setup(str(trio.cfg), endpoint=None, model=None, force=False, interactive=True,
                        preset="Third-7B")

    assert script.asked[0].options[0].label == "All, serving Third-7B first"
    assert _served(trio) == ("Third-7B", ["Qwen-27B", "Swift-27B-Q8"])


def test_a_peer_write_that_fails_keeps_config_yaml_and_leaves_the_peer_file_seen(trio, monkeypatch):
    monkeypatch.setattr(library, "append_peers", lambda *a, **k: None)

    result = _init(trio.cfg)

    _exited(result, 0)
    assert library.PEERS_NOT_WRITTEN_LINE.format(n=2) in _flat(result)
    assert _written(trio.cfg)["server"]["served_name"] == "Qwen-27B"
    seen = _library(trio)
    assert not seen.keys() & {trio.keys["A"]} and {trio.keys["B"], trio.keys["C"]} <= seen.keys()
    assert not any(e.declined for e in seen.values())


def test_pruning_never_creates_a_library_file(tmp_path):
    assert library.prune_recorded(tmp_path) is False and not model_library.library_path(tmp_path).exists()


def test_recorded_paths_are_the_models_and_projectors_both_files_name(tmp_path):
    (tmp_path / "config.yaml").write_text("server:\n  runtime: llamacpp\n  model: /m/a.gguf\n  mmproj: /m/p.gguf\n",
                                          encoding="utf-8")
    (tmp_path / "overrides.yaml").write_text(
        "extra_endpoints:\n- name: b\n  base_url: http://127.0.0.1:1/v1\n  lifecycle:\n    model: /m/b.gguf\n"
        "- not-a-mapping\n", encoding="utf-8")

    assert library.recorded_paths(tmp_path) == {"/m/a.gguf", "/m/p.gguf", "/m/b.gguf"}


@pytest.mark.parametrize("size, text", [(180, "180 B"), (1536, "1.5 KB"), (5 << 20, "5.0 MB"),
                                        (17 << 30, "17.0 GB"), (3 << 40, "3.0 TB")])
def test_a_size_reads_in_the_largest_unit_it_fills(size, text):
    assert library.format_size(size) == text
