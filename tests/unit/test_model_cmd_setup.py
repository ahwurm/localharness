"""`localharness model --setup` is the way back to a `Not now` or a decline: the multi-select over every model the box
holds that has no preset, for a person only. Real GGUF headers, the interview scripted, no walk without a person."""
from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from localharness.cli import library, model_cmd, workspace
from localharness.cli.app import app
from localharness.provider import model_library
from tests.unit.interview_support import scripted
from tests.unit.test_init_core_setup import _exited, _flat, _hermetic  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import _peers
from tests.unit.test_library_later_runs import _add, _declined, _entries, _key, _sync, box, later  # noqa: F401


def _setup(box, *flags: str):
    return CliRunner().invoke(app, ["model", "--setup", "--config-dir", str(box.cfg), *flags])


def _seen_and_declined(later, monkeypatch) -> None:
    _add(later, "B", 3_000_000)
    scripted(monkeypatch, "not-now")
    _sync(later)
    _add(later, "D", 2_000_000)
    scripted(monkeypatch, "choose", set())
    _sync(later)
    later.printed.truncate(0)
    later.printed.seek(0)


def test_setup_lists_the_seen_and_the_declined_preselects_nothing_and_a_selected_decline_is_cleared(later, monkeypatch):
    _seen_and_declined(later, monkeypatch)
    asked = scripted(monkeypatch, {_key(later, "D")})

    result = _setup(later)

    _exited(result, 0)
    [pick] = asked.asked
    assert pick.question == "Set up which?" and pick.multi and pick.default == set() and pick.cancel == "cancel"
    assert [o.label for o in pick.options] == ["B", "D (declined earlier)"]
    assert [p["name"] for p in _peers(later.cfg)] == ["Swift-27B-Q8", "D"]
    assert _declined(later) == {"B.gguf"}
    out = later.printed.getvalue()
    assert "/model D switches in a session; localharness start --preset D starts on it" in out
    assert "Not set up: B — localharness model --setup lists them" in out


def test_setup_preselects_a_first_sighting_and_not_a_seen_file(later, monkeypatch):
    _add(later, "B", 3_000_000)
    scripted(monkeypatch, "not-now")
    _sync(later)
    _add(later, "C", 5_000_000)
    asked = scripted(monkeypatch, set())

    _exited(_setup(later), 0)

    [pick] = asked.asked
    assert pick.default == {_key(later, "C")} and [o.label for o in pick.options] == ["C", "B"]
    assert _declined(later) == {"B.gguf", "C.gguf"}


def test_setup_with_nothing_unrecorded_says_so_and_asks_nothing(later, monkeypatch):
    asked = scripted(monkeypatch)

    result = _setup(later)

    _exited(result, 0)
    assert asked.asked == [] and library.NOTHING_TO_SET_UP_LINE in later.printed.getvalue()


def test_setup_over_a_missing_library_with_nothing_to_list_writes_the_baseline(later, monkeypatch):
    model_library.library_path(later.cfg).unlink()
    asked = scripted(monkeypatch)

    _exited(_setup(later), 0)

    assert asked.asked == [] and model_library.library_path(later.cfg).exists()
    assert library.NOTHING_TO_SET_UP_LINE in later.printed.getvalue()


def test_setup_off_a_terminal_refuses_with_one_line_exit_2_and_walks_nothing(later, monkeypatch):
    monkeypatch.setattr(workspace, "_stdout_is_a_terminal", lambda: False)
    _add(later, "C")

    def walk(*_a, **_k):
        raise AssertionError("walk_ggufs called")

    monkeypatch.setattr(model_library, "walk_ggufs", walk)
    asked = scripted(monkeypatch)

    result = _setup(later)

    assert result.exit_code == 2 and model_cmd.SETUP_TERMINAL_LINE in _flat(result)
    assert asked.asked == [] and len(_peers(later.cfg)) == 1


def test_setup_without_a_config_says_start_sets_one_up(box, monkeypatch):
    asked = scripted(monkeypatch)

    result = _setup(box)

    assert result.exit_code == 2 and model_cmd.SETUP_NO_CONFIG_LINE in _flat(result)
    assert asked.asked == [] and not (box.cfg / "config.yaml").exists()


@pytest.mark.parametrize("flags", [["NAME"], ["--script"], ["--download", "a/b"], ["--autostart", "status"],
                                   ["--file", "x.gguf"]],
                         ids=["name", "script", "download", "autostart", "file"])
def test_setup_takes_nothing_else(later, monkeypatch, flags):
    asked = scripted(monkeypatch)

    result = _setup(later, *flags)

    assert result.exit_code == 2 and model_cmd.SETUP_ALONE_LINE in _flat(result) and asked.asked == []


def test_a_setup_that_is_cancelled_is_not_now(later, monkeypatch):
    from tests.unit.interview_support import CANCEL
    _add(later, "C")
    scripted(monkeypatch, CANCEL)

    _exited(_setup(later), 0)

    assert len(_peers(later.cfg)) == 1 and _declined(later) == set()
    assert os.path.realpath(later.root / "C.gguf") in _entries(later)
