"""`terminal.keys` — the person's own chords for the nudge and the line break (spec 21 §5, §10).

This half pins the config and the loader: the closed vocabulary, the string-or-list reading, the
precedence of a list the person set over one left at its default, the one refusal for two set lists
that share a chord, and the machine-only drop. Everything here is hermetic: pydantic and the
loader's layers on `tmp_path`, no tty, no network.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from localharness.config.loader import (
    HARNESS_GLOBAL_ONLY_FIELDS,
    ConfigValidationError,
    is_harness_global_only,
)
from localharness.config.models import HarnessConfig, TerminalKeysConfig
from tests.unit.test_docs_convergence import _read, check_machine_only_list
from tests.unit.test_floor_machine_level import _layers, _warnings

SPEC_REFUSAL = (
    "terminal.keys: alt-enter is in both nudge and newline — set one list without it first "
    "(e.g. localharness components set terminal.keys.nudge ctrl-o), then retry"
)


def test_defaults():
    keys = TerminalKeysConfig()
    assert keys.nudge == ["alt-enter"]
    assert keys.newline == ["shift-enter"]
    assert HarnessConfig.model_fields["terminal"].default_factory is not None
    terminal = HarnessConfig.model_fields["terminal"].default_factory()
    assert terminal.keys == TerminalKeysConfig()


def test_a_set_newline_list_takes_its_chord_from_the_defaulted_nudge():
    keys = TerminalKeysConfig(newline=["alt-enter"])
    assert keys.nudge == []
    assert keys.newline == ["alt-enter"]


def test_terminal_keys_is_one_machine_only_member():
    assert "terminal.keys" in HARNESS_GLOBAL_ONLY_FIELDS
    assert "terminal.keys.nudge" not in HARNESS_GLOBAL_ONLY_FIELDS
    assert "terminal.keys.newline" not in HARNESS_GLOBAL_ONLY_FIELDS
    assert is_harness_global_only("terminal.keys.nudge")
    assert is_harness_global_only("terminal.keys.newline")


def test_security_md_lists_terminal_keys():
    assert check_machine_only_list(_read("SECURITY.md")) == []


# --- the vocabulary and the reading of a value ---------------------------------------------------


@pytest.mark.parametrize("name", [
    "ctrl-x", "", "enter", "ctrl-j", "ctrl-enter", "Alt-Enter", "alt+enter",
])
@pytest.mark.parametrize("field", ["nudge", "newline"])
def test_an_unknown_chord_name_is_refused(field, name):
    with pytest.raises(ValidationError):
        TerminalKeysConfig(**{field: [name]})


@pytest.mark.parametrize("text, expected", [
    ("ctrl-o", ["ctrl-o"]),
    ("alt-enter, ctrl-o", ["alt-enter", "ctrl-o"]),
    ("alt-enter,ctrl-o", ["alt-enter", "ctrl-o"]),
    ("ctrl-o, alt-enter", ["ctrl-o", "alt-enter"]),
    ("", []),
    ("  ", []),
    ("ctrl-o,", ["ctrl-o"]),
])
def test_a_comma_string_reads_as_the_list(text, expected):
    assert TerminalKeysConfig(nudge=text).nudge == expected
    # the same string for the other list reads the same list (it is set, so nothing is trimmed from it)
    assert TerminalKeysConfig(newline=text).newline == expected


def test_a_comma_string_with_an_unknown_name_is_refused():
    with pytest.raises(ValidationError):
        TerminalKeysConfig(nudge="ctrl-o, ctrl-x")


def test_a_repeated_chord_is_kept_once():
    assert TerminalKeysConfig(nudge=["ctrl-o", "ctrl-o"]).nudge == ["ctrl-o"]
    assert TerminalKeysConfig(nudge=["ctrl-o", "ctrl-g", "ctrl-o"]).nudge == ["ctrl-o", "ctrl-g"]


def test_a_non_hashable_element_is_a_validation_error_not_a_type_error():
    with pytest.raises(ValidationError):
        TerminalKeysConfig(nudge=[["ctrl-o"]])  # type: ignore[list-item]


# --- precedence ----------------------------------------------------------------------------------


def test_a_set_nudge_list_takes_its_chord_from_the_defaulted_newline():
    keys = TerminalKeysConfig(nudge=["shift-enter"])
    assert keys.nudge == ["shift-enter"]
    assert keys.newline == []


def test_each_list_keeps_the_order_it_was_written_in():
    kept = TerminalKeysConfig(newline=["ctrl-g", "alt-enter"])
    assert kept.newline == ["ctrl-g", "alt-enter"]
    assert kept.nudge == []
    both = TerminalKeysConfig(nudge=["ctrl-o", "alt-enter"], newline=["ctrl-g", "shift-enter"])
    assert (both.nudge, both.newline) == (["ctrl-o", "alt-enter"], ["ctrl-g", "shift-enter"])


def test_a_set_empty_list_trims_nothing():
    assert TerminalKeysConfig(nudge=[]).newline == ["shift-enter"]
    assert TerminalKeysConfig(newline=[]).nudge == ["alt-enter"]
    both = TerminalKeysConfig(nudge=[], newline=[])
    assert (both.nudge, both.newline) == ([], [])


def test_the_chord_a_set_list_names_leaves_the_defaulted_list_silently(recwarn):
    keys = TerminalKeysConfig(newline=["alt-enter"])
    assert keys.nudge == []
    assert len(recwarn) == 0


@pytest.mark.parametrize("kwargs, line", [
    pytest.param(
        {"nudge": ["alt-enter"], "newline": ["alt-enter"]}, SPEC_REFUSAL, id="spec-alt-enter"),
    pytest.param(
        {"nudge": ["shift-enter"], "newline": ["shift-enter"]},
        "terminal.keys: shift-enter is in both nudge and newline — set one list without it first "
        "(e.g. localharness components set terminal.keys.newline ctrl-o), then retry",
        id="shift-enter-names-newline"),
    pytest.param(
        {"nudge": ["alt-enter", "ctrl-o"], "newline": ["alt-enter", "ctrl-g"]},
        "terminal.keys: alt-enter is in both nudge and newline — set one list without it first, "
        "then retry",
        id="no-free-chord"),
    pytest.param(
        {"nudge": ["ctrl-g", "alt-enter"], "newline": ["alt-enter", "ctrl-g"]},
        "terminal.keys: ctrl-g is in both nudge and newline — set one list without it first "
        "(e.g. localharness components set terminal.keys.nudge ctrl-o), then retry",
        id="two-shared-first-in-nudge-order"),
    pytest.param(
        {"nudge": ["ctrl-o"], "newline": ["ctrl-o"]},
        "terminal.keys: ctrl-o is in both nudge and newline — set one list without it first "
        "(e.g. localharness components set terminal.keys.nudge ctrl-g), then retry",
        id="free-chord-skips-ctrl-o-when-used"),
])
def test_two_set_lists_sharing_a_chord_are_refused_with_the_fix(kwargs, line):
    with pytest.raises(ValidationError) as exc:
        TerminalKeysConfig(**kwargs)
    assert line in str(exc.value)


def test_the_spec_refusal_is_the_whole_message():
    with pytest.raises(ValidationError) as exc:
        TerminalKeysConfig(nudge=["alt-enter"], newline=["alt-enter"])
    messages = [e["msg"] for e in exc.value.errors()]
    assert messages == [f"Value error, {SPEC_REFUSAL}"]


def test_two_set_disjoint_lists_are_accepted():
    keys = TerminalKeysConfig(nudge=["ctrl-o"], newline=["alt-enter"])
    assert (keys.nudge, keys.newline) == (["ctrl-o"], ["alt-enter"])
    both_default_values = TerminalKeysConfig(nudge=["alt-enter"], newline=["shift-enter"])
    assert (both_default_values.nudge, both_default_values.newline) == (["alt-enter"], ["shift-enter"])


@pytest.mark.parametrize("kwargs, written", [
    ({}, set()),
    ({"nudge": ["ctrl-o"]}, {"nudge"}),
    ({"nudge": ["shift-enter"]}, {"nudge"}),
    ({"newline": ["alt-enter"]}, {"newline"}),
    ({"nudge": []}, {"nudge"}),
    ({"nudge": ["ctrl-o"], "newline": ["alt-enter"]}, {"nudge", "newline"}),
])
def test_resolution_leaves_model_fields_set_as_written(kwargs, written):
    assert TerminalKeysConfig(**kwargs).model_fields_set == written


@pytest.mark.parametrize("kwargs", [
    {},
    {"nudge": ["ctrl-o"]},
    {"nudge": ["shift-enter"]},
    {"newline": ["alt-enter"]},
    {"newline": ["ctrl-g", "alt-enter"]},
    {"nudge": []},
    {"newline": []},
    {"nudge": [], "newline": []},
    {"nudge": ["ctrl-o"], "newline": ["alt-enter"]},
    {"nudge": "alt-enter, ctrl-o"},
])
def test_every_resolved_state_round_trips(kwargs):
    keys = TerminalKeysConfig(**kwargs)
    assert TerminalKeysConfig.model_validate(keys.model_dump()) == keys


def test_a_trimmed_list_is_not_written_back_by_plain_assignment():
    # the structural half of the same promise: the resolution never assigns, so the set stays put
    import inspect

    from localharness.config import models

    source = inspect.getsource(models.TerminalKeysConfig)
    assert "object.__setattr__(" in source
    assert "self.nudge =" not in source
    assert "self.newline =" not in source


# --- the loader ----------------------------------------------------------------------------------


def test_the_loader_refuses_an_unknown_name_naming_the_file_and_key(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg={"terminal": {"keys": {"nudge": ["ctrl-x"]}}})
    with pytest.raises(ConfigValidationError) as exc:
        loader.load_harness()
    text = str(exc.value)
    assert "config.yaml" in text
    assert "terminal.keys.nudge" in text


def test_the_loader_refuses_an_unknown_newline_name_naming_the_key(tmp_path):
    loader, _ = _layers(tmp_path, g_cfg={"terminal": {"keys": {"newline": "ctrl-enter"}}})
    with pytest.raises(ConfigValidationError) as exc:
        loader.load_harness()
    assert "terminal.keys.newline" in str(exc.value)


def test_a_workspace_terminal_keys_is_dropped_and_the_machine_wins(tmp_path):
    loader, ws = _layers(
        tmp_path,
        g_cfg={"terminal": {"keys": {"nudge": ["ctrl-o"]}}},
        ws_cfg={"terminal": {"keys": {"nudge": ["ctrl-g"]}}},
    )
    h = loader.load_harness()
    assert h.terminal.keys.nudge == ["ctrl-o"]
    assert _warnings(loader) == [
        f"ignoring terminal.keys in {ws / 'config.yaml'}: only the global config may set it"
    ]


def test_a_workspace_overrides_file_is_dropped_too(tmp_path):
    loader, ws = _layers(
        tmp_path,
        g_cfg={"terminal": {"keys": {"nudge": ["ctrl-o"]}}},
        ws_over={"terminal": {"keys": {"newline": ["alt-enter"]}}},
    )
    h = loader.load_harness()
    assert (h.terminal.keys.nudge, h.terminal.keys.newline) == (["ctrl-o"], ["shift-enter"])
    assert _warnings(loader) == [
        f"ignoring terminal.keys in {ws / 'overrides.yaml'}: only the global config may set it"
    ]


def test_a_workspace_block_restating_the_defaults_is_warned_too(tmp_path):
    loader, ws = _layers(
        tmp_path,
        ws_cfg={"terminal": {"keys": {"nudge": ["alt-enter"], "newline": ["shift-enter"]}}},
    )
    h = loader.load_harness()
    assert h.terminal.keys == TerminalKeysConfig()
    assert _warnings(loader) == [
        f"ignoring terminal.keys in {ws / 'config.yaml'}: only the global config may set it"
    ]


def test_the_machines_overrides_file_sets_one_list_with_no_warning(tmp_path):
    loader, _ = _layers(tmp_path, g_over={"terminal": {"keys": {"newline": "ctrl-o"}}})
    h = loader.load_harness()
    assert h.terminal.keys.newline == ["ctrl-o"]
    assert h.terminal.keys.nudge == ["alt-enter"]
    assert _warnings(loader) == []


def test_a_dropped_workspace_block_never_counts_as_set(tmp_path):
    # the machine sets only newline; a workspace nudge is dropped before validation, so the
    # machine's own newline still takes alt-enter off the defaulted nudge and nothing is refused
    loader, _ = _layers(
        tmp_path,
        g_cfg={"terminal": {"keys": {"newline": ["alt-enter"]}}},
        ws_cfg={"terminal": {"keys": {"nudge": ["alt-enter"]}}},
    )
    h = loader.load_harness()
    assert (h.terminal.keys.nudge, h.terminal.keys.newline) == ([], ["alt-enter"])


# --- the catalogue -------------------------------------------------------------------------------


def test_both_lists_are_catalogue_paths(components_home):
    from localharness.config.models import AgentConfig
    from localharness.registry.catalogue import build_catalogue
    from tests.unit.test_registry_catalogue import _make_minimal_harness_cfg

    entries = build_catalogue(
        _make_minimal_harness_cfg(), overlays={}, agent_cfg=AgentConfig(name="x", role="y"),
        plugins=(),
    )
    assert "terminal.keys.nudge" in entries
    assert "terminal.keys.newline" in entries
