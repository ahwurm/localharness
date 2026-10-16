"""The later run of `library.sync`: a file that arrives after the first run is a first sighting once, set up as a
peer on Enter, kept quiet by `Not now`, declined by being left out of `Choose which`; nothing without a person.
Real GGUF headers under a faked HOME, the interview scripted, no walk of a real root."""
from __future__ import annotations

import io
import json
import os

import pytest
import yaml
from rich.console import Console

from localharness.cli import library
from localharness.provider import model_library
from tests.unit import test_init_discovery_records as world
from tests.unit.interview_support import CANCEL, UNREACHED, scripted
from tests.unit.test_init_core_setup import _exited, _hermetic, _init  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import _model, _peers

box = world.box


@pytest.fixture
def later(box, monkeypatch):
    """The two-model box after its first run: Qwen serves, Swift is a peer, the library file exists."""
    _exited(_init(box.cfg), 0)
    box.printed = io.StringIO()
    monkeypatch.setattr(library, "console", Console(file=box.printed, width=300))
    return box


def _add(box, name: str, mtime: int = 5_000_000):
    return _model(box.root / f"{name}.gguf", mtime=mtime)


def _key(box, name: str) -> str:
    return os.path.realpath(box.root / f"{name}.gguf")


def _entries(box) -> dict:
    return json.loads(model_library.library_path(box.cfg).read_text(encoding="utf-8"))["entries"]


def _sync(box, **kw) -> library.Sync:
    return library.sync(box.cfg, scope=kw.pop("scope", "new"), person=kw.pop("person", True), **kw)


def _declined(box) -> set[str]:
    return {os.path.basename(k) for k, e in _entries(box).items() if e["declined"]}


def test_without_a_person_a_later_run_walks_nothing_and_writes_nothing(later, monkeypatch):
    _add(later, "C")
    before = model_library.library_path(later.cfg).read_text(encoding="utf-8")

    def walk(*_a, **_k):
        raise AssertionError("walk_ggufs called")

    monkeypatch.setattr(model_library, "walk_ggufs", walk)

    assert _sync(later, person=False) == library.Sync()
    assert model_library.library_path(later.cfg).read_text(encoding="utf-8") == before


def test_one_new_model_is_one_question_and_enter_appends_it_adopted(later, monkeypatch):
    _add(later, "C")
    asked = scripted(monkeypatch, "set-up")

    synced = _sync(later)

    [question] = asked.asked
    assert question.question == "New model C — set it up?" and not question.multi
    assert [(o.key, o.label) for o in question.options] == [("set-up", "Set it up"), ("not-now", "Not now")]
    assert question.cancel == "not-now"
    assert synced.appended == 1 and [p["name"] for p in _peers(later.cfg)] == ["Swift-27B-Q8", "C"]
    assert "/model C switches in a session; localharness start --preset C starts on it" in later.printed.getvalue()
    assert "C.gguf" not in " ".join(_entries(later))


def test_two_new_models_offer_set_them_up_not_now_and_choose_which(later, monkeypatch):
    _add(later, "C", 5_000_000)
    _add(later, "D", 4_000_000)
    asked = scripted(monkeypatch, "set-up")

    synced = _sync(later)

    [question] = asked.asked
    assert question.question == "2 new models — set them up?"
    assert [(o.key, o.label) for o in question.options] == [
        ("set-up", "Set them up"), ("not-now", "Not now"), ("choose", "Choose which")]
    assert synced.appended == 2 and {p["name"] for p in _peers(later.cfg)} == {"Swift-27B-Q8", "C", "D"}
    ports = [p["lifecycle"]["port"] for p in _peers(later.cfg)]
    assert len(set(ports)) == 3 and 8081 not in ports


def test_not_now_appends_nothing_and_says_what_was_left(later, monkeypatch):
    _add(later, "C", 5_000_000)
    _add(later, "D", 4_000_000)
    scripted(monkeypatch, "not-now")

    synced = _sync(later)

    assert synced.appended == 0 and len(_peers(later.cfg)) == 1
    assert "Not set up: C, D — localharness model --setup lists them" in later.printed.getvalue()
    assert _declined(later) == set()


def test_choose_which_sets_up_the_selected_and_declines_the_rest_for_good(later, monkeypatch):
    _add(later, "C", 5_000_000)
    _add(later, "D", 4_000_000)
    asked = scripted(monkeypatch, "choose", {_key(later, "C")})

    synced = _sync(later)

    pick = asked.asked[1]
    assert pick.question == "Set up which?" and pick.multi and pick.default == {_key(later, "C"), _key(later, "D")}
    assert synced.appended == 1 and [p["name"] for p in _peers(later.cfg)] == ["Swift-27B-Q8", "C"]
    assert _declined(later) == {"D.gguf"}
    assert "Not set up: D — localharness model --setup lists them" in later.printed.getvalue()
    again = scripted(monkeypatch)

    assert _sync(later) == library.Sync() and again.asked == []


def test_a_declined_model_that_is_resized_stays_declined_and_silent(later, monkeypatch):
    d = _add(later, "D")
    scripted(monkeypatch, "choose", set())
    _sync(later)
    assert _declined(later) == {"D.gguf"}
    d.write_bytes(d.read_bytes() + b"\0" * 64)
    again = scripted(monkeypatch)

    assert _sync(later) == library.Sync() and again.asked == []
    assert _declined(later) == {"D.gguf"}


def test_a_cancel_inside_choose_which_is_not_now(later, monkeypatch):
    _add(later, "C")
    _add(later, "D", 4_000_000)
    scripted(monkeypatch, "choose", CANCEL)

    synced = _sync(later)

    assert synced.appended == 0 and _declined(later) == set()
    assert "Not set up: C, D" in later.printed.getvalue()


def test_a_busy_overlay_keeps_the_files_new_so_they_are_asked_again(later, monkeypatch):
    _add(later, "C")
    scripted(monkeypatch, "set-up")
    real = library.append_peers
    monkeypatch.setattr(library, "append_peers", lambda *a, **k: None)

    synced = _sync(later)

    assert synced.appended == 0 and "C.gguf" not in " ".join(_entries(later))
    again = scripted(monkeypatch, "not-now")
    monkeypatch.setattr(library, "append_peers", real)
    _sync(later)
    assert [a.question for a in again.asked] == ["New model C — set it up?"]


def test_an_interview_that_cannot_be_asked_writes_nothing_and_keeps_the_file_new(later, monkeypatch):
    _add(later, "C")
    scripted(monkeypatch, UNREACHED)

    synced = _sync(later)

    assert synced == library.Sync(shown=1) and len(_peers(later.cfg)) == 1
    again = scripted(monkeypatch, "not-now")
    _sync(later)
    assert [a.question for a in again.asked] == ["New model C — set it up?"]


def test_a_name_a_preset_already_has_is_numbered_on_the_question_and_in_the_file(later, monkeypatch):
    _model(later.root / "other" / "Swift-27B-Q8.gguf", mtime=5_000_000)
    asked = scripted(monkeypatch, "set-up")

    _sync(later)

    assert asked.asked[0].question == "New model Swift-27B-Q8-2 — set it up?"
    assert [p["name"] for p in _peers(later.cfg)] == ["Swift-27B-Q8", "Swift-27B-Q8-2"]


def test_recorded_names_and_ports_are_the_merged_views(later):
    names, ports = library.recorded_names_ports(later.cfg)

    assert names == {"Qwen-27B", "Swift-27B-Q8"} and ports == {8081, 8082}
    overrides = later.cfg / "overrides.yaml"
    assert yaml.safe_load(overrides.read_text(encoding="utf-8"))["extra_endpoints"][0]["name"] in names
