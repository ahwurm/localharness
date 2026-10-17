"""The foreign-unit sentence: one builder says what a unit the harness did not write means, the speed
ledger remembers it per (unit, served), and `init` prints it on change."""
from __future__ import annotations

import json
from types import SimpleNamespace

from localharness.cli import workspace
from localharness.cli.workspace import Holder
from localharness.provider import speed_stats
from tests.unit.test_init_core_setup import _exited, _init
from tests.unit.test_init_live_import import UNIT, block_of, live, show  # noqa: F401  (fixture)
from tests.unit.test_init_unit_import import STOPPED_LINE, _stopped_unit, _systemd
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)
from tests.unit.units_support import fake_systemd  # noqa: F401  (fixture)

SERVED = "qwen-27b-swift-1.5"
TAIL = "hands the GPU to the harness (localharness model --autostart on brings your choice up at login)."


def unit_holder(unit: str = "llm.service", state: str = "enabled") -> Holder:
    return Holder(unit, 4242, "llama-server", 1000, SimpleNamespace(unit_file_state=state, load_state="loaded"))


def test_an_enabled_unit_is_disabled_now_to_hand_the_gpu_over():
    assert workspace.foreign_unit_sentence(unit_holder(), served=SERVED, port=8080) == (
        f"llm.service serves {SERVED} and is yours, so the harness lists models but never stops it; "
        f"systemctl --user disable --now llm.service {TAIL}")


def test_a_unit_that_is_not_enabled_is_stopped():
    for state in ("disabled", "static", "transient", "enabled-runtime"):
        sentence = workspace.foreign_unit_sentence(unit_holder(state=state), served=SERVED, port=8080)
        assert "systemctl --user stop llm.service " in sentence


def test_a_control_character_in_the_unit_is_shown_as_an_escape_and_quoted_in_the_command():
    sentence = workspace.foreign_unit_sentence(unit_holder("llm\x1b[2J my.service"), served=SERVED, port=8080)

    assert sentence.startswith("llm\\x1b[2J my.service serves ")
    assert "systemctl --user disable --now 'llm\\x1b[2J my.service' " in sentence
    assert "\x1b" not in sentence


def test_a_holder_with_no_unit_is_named_by_its_process_and_has_no_systemctl_command():
    holder = Holder(None, 4242, "llama-server", 1000)

    assert workspace.foreign_unit_sentence(holder, served="qwen", port=8080) == (
        "pid 4242 (llama-server) serves qwen and is yours, so the harness lists models but never stops it; "
        "stopping it hands the GPU to the harness.")


def test_the_ledger_remembers_a_unit_and_what_it_serves(tmp_path):
    ledger = tmp_path / "speed_stats.json"
    assert speed_stats.foreign_unit_changed(ledger, "llm.service", "qwen")

    speed_stats.record_foreign_unit(ledger, "llm.service", "qwen")

    assert not speed_stats.foreign_unit_changed(ledger, "llm.service", "qwen")
    assert speed_stats.foreign_unit_changed(ledger, "llm.service", "flash")
    assert speed_stats.foreign_unit_changed(ledger, "other.service", "qwen")
    assert set(json.loads(ledger.read_text())) == {speed_stats.FOREIGN_UNIT_KEY}


def test_the_foreign_unit_record_is_no_model_entry(tmp_path):
    ledger = tmp_path / "speed_stats.json"
    speed_stats.record_foreign_unit(ledger, "llm.service", "qwen")

    assert speed_stats.model_paths(ledger) == set()
    assert speed_stats.all_median_tps(ledger) == {}


# ------------------------------------------------------------------ init


def running_sentence(unit: str, state: str) -> str:
    verb = "disable --now" if state == "enabled" else "stop"
    return (f"{unit} serves qwen-27b and is yours, so the harness lists models but never stops it; "
            f"systemctl --user {verb} {unit} {TAIL}")


def test_init_against_a_running_foreign_unit_prints_the_sentence_once(live):  # noqa: F811
    live.fake_systemd.answer_prefix("systemctl", "--user", "show", stdout=show(enabled="enabled"))
    live.serve()

    first = _init(live.cfg, "--no-input")

    _exited(first, 0)
    assert running_sentence(UNIT, "enabled") in block_of(first)
    assert not any("holds :8080" in line for line in block_of(first))
    assert not speed_stats.foreign_unit_changed(live.ledger(), UNIT, "qwen-27b")

    for name in ("config.yaml", "overrides.yaml"):
        (live.cfg / name).unlink()
    second = _init(live.cfg, "--no-input")

    _exited(second, 0)
    assert not any("serves qwen-27b and is yours" in line for line in block_of(second))


def test_with_a_window_the_sentence_rides_on_the_line_before_the_window_clause(live):  # noqa: F811
    live.serve("-c", "65536")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    block = block_of(result)
    at = block.index(running_sentence(UNIT, "disabled"))
    assert block[at + 1].startswith(f"{UNIT} serves -c 65536; the harness launches")


# ------------------------------------------------------------------ the stopped unit init imported


def imported_sentence(enabled: bool) -> str:
    clause = f" — {DISABLE} keeps it from starting at login beside the harness" if enabled else ""
    return (f"llm.service is yours and was imported as the server: record; from now on the harness "
            f"launches qwen-27b itself and never touches the unit{clause}.")


DISABLE = "systemctl --user " + "disable llm.service"


def stopped(live, enabled: str) -> None:  # noqa: F811
    _stopped_unit(live)
    _systemd(live, ("llm.service", enabled, str(live.home / ".config/systemd/user/llm.service"), ""))


def test_an_enabled_unit_the_import_took_says_how_to_keep_it_from_starting_beside_the_harness(live):  # noqa: F811
    stopped(live, "enabled")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    block = block_of(result)
    assert block.count(imported_sentence(True)) == 1
    assert block.index(imported_sentence(True)) < block.index(STOPPED_LINE)


def test_a_unit_that_is_not_enabled_gets_no_systemctl_clause(live):  # noqa: F811
    stopped(live, "disabled")

    result = _init(live.cfg, "--no-input")

    _exited(result, 0)
    assert imported_sentence(False) in block_of(result)
    assert "keeps it from starting at login" not in result.output


def test_a_later_init_on_the_written_config_does_not_say_it_again(live):  # noqa: F811
    stopped(live, "enabled")
    _exited(_init(live.cfg, "--no-input"), 0)

    again = _init(live.cfg, "--no-input")

    assert "was imported as the server" not in again.output


def test_a_control_character_in_the_imported_unit_is_escaped_and_quoted():
    sentence = workspace.imported_unit_sentence(unit_holder("llm\x1b[2J my.service"), served=SERVED)

    assert sentence.startswith("llm\\x1b[2J my.service is yours and was imported")
    assert "disable 'llm\\x1b[2J my.service' keeps it" in sentence
    assert "\x1b" not in sentence
