"""`/plugins` in a terminal session (RULINGS R1, R2, R3, R10, R14).

1. The restart decision: `plugins_cmd.switch_decision` and `plugins_overview`, over stub plugins
   swapped into BUILTIN_PLUGINS and a real config dir. Does `/plugins enable|disable NAME` restart
   the session (None), or answer one line and stay? The decision is made BEFORE teardown, from a
   fresh plan over the session's own layers, and asks and writes nothing.

Tests that call asyncio.run are plain `def`: asyncio_mode is "auto".
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from localharness.cli import plugins_cmd
from localharness.plugins import builtin, discovery
from tests.unit.test_plugins_enable_setup import _CONFIG, Needy, Stub

_REAL_DISCOVER = discovery.discover
NEEDY_LINE = ("needy needs its install extra first: install `localharness[nosuchextra]`, "
              "then /plugins enable needy again")
USAGE_TAIL = "/plugins enable <name> turns one on here; /plugins disable <name> turns one off."


# --- 1. the restart decision -------------------------------------------------------------------

@pytest.fixture
def g(tmp_path: Path, monkeypatch) -> Path:
    """stub (off by default, one question) and needy (on by default, its extra missing) as the
    bundled plugins, over a real global config dir — returned."""
    monkeypatch.setattr(builtin, "BUILTIN_PLUGINS", (Stub, Needy))
    monkeypatch.setattr(discovery, "discover", lambda global_config_dir: [
        f for f in _REAL_DISCOVER(global_config_dir) if f.source == "folder"])
    g = tmp_path / "g"
    g.mkdir()
    (g / "config.yaml").write_text(yaml.safe_dump(_CONFIG), encoding="utf-8")
    return g


def _overrides(directory: Path, data: dict) -> None:
    (directory / "overrides.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def _decide(name: str, on: bool, running: set[str], g: Path, workspace: Path | None = None):
    return asyncio.run(plugins_cmd.switch_decision(name, on, frozenset(running), g, workspace))


def test_decision_unknown_name_is_one_line(g):
    assert _decide("nope", True, set(), g) == "Unknown plugin: nope. /plugins lists them."


def test_decision_running_and_set_up_is_already_on(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    assert _decide("stub", True, {"stub"}, g) == "stub is already on."


def test_decision_not_running_restarts(g):
    assert _decide("stub", True, set(), g) is None


def test_decision_running_but_its_step_still_has_work_restarts(g):
    # on, but its question has no answer: the step's questions are asked outside the input box
    _overrides(g, {"stub": {"enabled": True}})
    assert _decide("stub", True, {"stub"}, g) is None


def test_decision_needs_its_extra_is_one_line(g):
    assert _decide("needy", True, set(), g) == NEEDY_LINE


def test_decision_needs_its_extra_even_when_turned_off(g):
    # the plan reads `off` now; the decision asks for the extra directly, as the step does
    _overrides(g, {"needy": {"enabled": False}})
    assert _decide("needy", True, set(), g) == NEEDY_LINE


def test_decision_a_project_that_turns_it_off_still_wins(g, tmp_path):
    ws = tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"stub": {"enabled": False}}), encoding="utf-8")
    assert _decide("stub", True, set(), g, ws) == (
        f"this project turns stub off ({ws}), and that still wins here — "
        "`localharness plugins enable stub --workspace` changes it for this project")


def test_decision_disable_of_an_enabled_plugin_restarts(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    assert _decide("stub", False, {"stub"}, g) is None


def test_decision_disable_of_a_plugin_not_enabled_is_already_off(g):
    assert _decide("stub", False, set(), g) == "stub is already off."


def test_decision_asks_and_writes_nothing(g, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the decision must not ask or write")
    monkeypatch.setattr(plugins_cmd.typer, "prompt", boom)
    monkeypatch.setattr(plugins_cmd, "atomic_write_overlay", boom)
    _overrides(g, {"stub": {"enabled": True}})
    before = sorted(p.name for p in g.iterdir())
    assert _decide("stub", True, {"stub"}, g) is None
    assert _decide("stub", False, {"stub"}, g) is None
    assert sorted(p.name for p in g.iterdir()) == before


def test_overview_lists_every_plugin_and_how_to_switch_one(g):
    text = plugins_cmd.plugins_overview(frozenset(), g, None)
    assert text.splitlines() == [
        "Plugins:",
        "  stub   off — turn on: localharness plugins enable stub",
        "  needy  on (install `localharness[nosuchextra]` to use it)",
        USAGE_TAIL,
    ]


def test_overview_says_which_run_in_this_session(g):
    _overrides(g, {"stub": {"enabled": True, "url": "http://ok"}})
    lines = plugins_cmd.plugins_overview(frozenset({"stub"}), g, None).splitlines()
    assert lines[1] == "  stub   running in this session"
    assert lines[-1] == USAGE_TAIL


def test_decision_disable_of_a_plugin_running_here_restarts_even_when_its_flag_is_off(g):
    # its flag was turned off on disk mid-session (a shell `plugins disable`): it still runs here,
    # so "already off" would be false — the restart is what stops it
    assert _decide("stub", False, {"stub"}, g) is None


def test_decision_enable_of_a_plugin_running_here_whose_flag_is_off_restarts(g):
    # running and set up, but its flag on disk is off: "already on" would leave it off at the
    # next start — the restart's step is what writes the flag
    _overrides(g, {"stub": {"url": "http://ok"}})
    assert _decide("stub", True, {"stub"}, g) is None
