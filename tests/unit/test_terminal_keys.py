"""`terminal.keys` — the person's own chords for the nudge and the line break (spec 21 §5, §10).

This half pins the config and the loader: the closed vocabulary, the string-or-list reading, the
precedence of a list the person set over one left at its default, the one refusal for two set lists
that share a chord, and the machine-only drop. Everything here is hermetic: pydantic and the
loader's layers on `tmp_path`, no tty, no network.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import pytest
from prompt_toolkit.application import Application, create_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import vt100_parser
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import DummyOutput
from pydantic import ValidationError

from localharness.channels import keys as keys_mod
from localharness.channels.keys import (
    KEYS,
    SHIFT_ENTER_KEY,
    SHIFT_ENTER_SEQUENCES,
    install_wire_sequences,
)
from localharness.config.chords import CHORD_NAMES, LABELS
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


# --- the chords in code: the table, the borrowed key and the guarded install ----------------------

XTERM_FORM = "\x1b[27;2;13~"
KITTY_FORM = "\x1b[13;2u"
SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


def test_keys_table_covers_the_vocabulary_and_every_name_has_a_label():
    assert KEYS.keys() == set(CHORD_NAMES)
    assert set(LABELS) == set(CHORD_NAMES)
    assert CHORD_NAMES == ("alt-enter", "shift-enter", "ctrl-g", "ctrl-o")
    assert len(set(KEYS.values())) == 4
    assert KEYS["alt-enter"] == ("escape", "enter")
    assert KEYS["shift-enter"] == (SHIFT_ENTER_KEY,)
    assert KEYS["ctrl-g"] == ("c-g",)
    assert KEYS["ctrl-o"] == ("c-o",)
    assert SHIFT_ENTER_KEY is Keys.ControlF3


def test_premise_controlf3_unproduced_and_prefix_cache_present():
    # The borrowed key is only safe while no table entry yields it, and the install only works
    # while prompt_toolkit keeps the private prefix cache: an upgrade that changes either must fail
    # here, not in the person's input.
    assert Keys.ControlF3 not in ANSI_SEQUENCES.values()
    assert hasattr(vt100_parser, "_IS_PREFIX_OF_LONGER_MATCH_CACHE")
    assert ANSI_SEQUENCES[XTERM_FORM] == Keys.ControlM
    assert KITTY_FORM not in ANSI_SEQUENCES


def test_install_wire_sequences_is_idempotent():
    assert install_wire_sequences() is True
    assert install_wire_sequences() is True
    assert sorted(k for k, v in ANSI_SEQUENCES.items() if v == SHIFT_ENTER_KEY) == sorted(
        SHIFT_ENTER_SEQUENCES
    )
    assert sum(1 for v in ANSI_SEQUENCES.values() if v == SHIFT_ENTER_KEY) == 2
    assert len(vt100_parser._IS_PREFIX_OF_LONGER_MATCH_CACHE) == 0


def test_install_replaces_the_xterm_entry_and_leaves_ctrl_enter():
    install_wire_sequences()
    assert ANSI_SEQUENCES[XTERM_FORM] == SHIFT_ENTER_KEY
    assert ANSI_SEQUENCES[KITTY_FORM] == SHIFT_ENTER_KEY
    assert ANSI_SEQUENCES["\x1b[27;5;13~"] == Keys.ControlM
    assert ANSI_SEQUENCES["\x1b[27;6;13~"] == Keys.ControlM


def test_install_without_the_prefix_cache_installs_nothing_and_logs_once(monkeypatch, caplog):
    monkeypatch.delattr(vt100_parser, "_IS_PREFIX_OF_LONGER_MATCH_CACHE")
    before = dict(ANSI_SEQUENCES)
    with caplog.at_level(logging.INFO, logger="localharness.channels.keys"):
        assert install_wire_sequences() is False
    assert ANSI_SEQUENCES == before
    records = [r for r in caplog.records if r.name == "localharness.channels.keys"]
    assert len(records) == 1
    assert "shift+enter" in records[0].getMessage().lower()


async def test_both_shift_enter_encodings_reach_a_binding_on_the_borrowed_key():
    assert install_wire_sequences() is True
    lines: list[str] = []
    buf = Buffer(multiline=False)
    kb = KeyBindings()

    @kb.add(*KEYS["shift-enter"])
    def _newline(event):
        event.current_buffer.insert_text("\n")

    @kb.add("enter")
    def _submit(event):
        lines.append(event.current_buffer.text)
        event.current_buffer.reset()

    @kb.add("c-d")
    def _exit(event):
        event.app.exit()

    with create_pipe_input() as inp:
        with create_app_session(input=inp, output=DummyOutput()):
            app = Application(
                layout=Layout(Window(BufferControl(buf))), key_bindings=kb,
                input=inp, output=DummyOutput(),
            )
            inp.send_text("a" + KITTY_FORM + "b" + XTERM_FORM + "c" + "\r" + "\x04")
            await asyncio.wait_for(app.run_async(), timeout=10.0)
    assert lines == ["a\nb\nc"]


def test_ctrl_f3_is_named_only_in_channels_keys():
    named = sorted(
        str(p.relative_to(SRC)) for p in SRC.rglob("*.py") if "ControlF3" in p.read_text("utf-8")
    )
    assert named == ["channels/keys.py"]


def test_the_chord_module_never_reads_a_device_or_sets_a_keyboard_mode():
    source = (SRC / "channels" / "keys.py").read_text("utf-8")
    imports = [ln for ln in source.splitlines() if re.match(r"\s*(import|from)\s", ln)]
    assert not [ln for ln in imports if re.search(r"evdev|pyudev|uinput", ln)]
    assert "/dev/input" not in source
    assert not re.search(r"\bopen\(", source)
    assert "sys.stdout.write" not in source
    assert "os.write" not in source
    assert "\x1b[>" not in source and "\x1b[=" not in source


# --- the legend, the /help block and terminal detection (spec 21 §6.6, §6.7, §7, §8) ---------------

KITTY_LINE = r"kitty.conf: map shift+enter send_text all \x1b[13;2u"
GHOSTTY_LINE = r"config: keybind = shift+enter=text:\x1b[13;2u"
WEZTERM_LINE = (
    r'keys: { key = "Enter", mods = "SHIFT", action = wezterm.action.SendString("\x1b[13;2u") }'
)
ALACRITTY_LINE = r'[[keyboard.bindings]] key = "Enter", mods = "Shift", chars = "\u001b[13;2u"'
FOOT_LINE = r"[text-bindings]: \x1b[13;2u=Shift+Return"
ITERM_LINE = "Profiles → Keys → add Shift+Enter → Send Escape Sequence → [13;2u"
VSCODE_LINE = (
    r'keybindings.json: { "key": "shift+enter", "command": "workbench.action.terminal.sendSequence", '
    r'"args": { "text": "\u001b[13;2u" }, "when": "terminalFocus" }'
)
WT_LINE = (
    r'actions: { "command": { "action": "sendInput", "input": "\u001b[13;2u" }, '
    r'"keys": "shift+enter" }'
)
ESC = "\x1b"
COUNT = "send the line into the running turn (a nudge)"


def _keys(**kw):
    return TerminalKeysConfig(**kw)


def _rows(block: str) -> list[str]:
    lines = block.split("\n")
    assert len(lines) == 3 and lines[0] == "Keys:"
    assert lines[1].startswith("  ") and lines[2].startswith("  ")
    return [lines[1][2:], lines[2][2:]]


def test_nudge_legend_for_one_two_and_zero_names():
    assert keys_mod.nudge_legend(["alt-enter"]) == "!text or alt+enter nudge"
    assert keys_mod.nudge_legend(["shift-enter", "ctrl-o"]) == "!text, shift+enter or ctrl+o nudge"
    assert keys_mod.nudge_legend([]) == "!text nudge"
    assert keys_mod.nudge_legend(["alt-enter", "ctrl-g", "ctrl-o"]) == (
        "!text, alt+enter, ctrl+g or ctrl+o nudge"
    )


@pytest.mark.parametrize(
    ("env", "name", "line"),
    [
        ({"KITTY_WINDOW_ID": "1"}, "kitty", KITTY_LINE),
        ({"TERM": "xterm-kitty"}, "kitty", KITTY_LINE),
        ({"TERM_PROGRAM": "ghostty"}, "Ghostty", GHOSTTY_LINE),
        ({"TERM": "xterm-ghostty"}, "Ghostty", GHOSTTY_LINE),
        ({"TERM_PROGRAM": "WezTerm"}, "WezTerm", WEZTERM_LINE),
        ({"TERM": "alacritty"}, "Alacritty", ALACRITTY_LINE),
        ({"ALACRITTY_WINDOW_ID": "1234"}, "Alacritty", ALACRITTY_LINE),
        ({"TERM": "foot"}, "foot", FOOT_LINE),
        ({"TERM": "foot-extra"}, "foot", FOOT_LINE),
        ({"TERM_PROGRAM": "iTerm.app"}, "iTerm2", ITERM_LINE),
        ({"TERM_PROGRAM": "vscode"}, "VS Code", VSCODE_LINE),
        ({"WT_SESSION": "abc"}, "Windows Terminal", WT_LINE),
    ],
)
def test_detect_terminal_names_each_terminal(env, name, line):
    assert keys_mod.detect_terminal(env) == (name, line)


def test_detect_terminal_tmux_first():
    assert keys_mod.detect_terminal({"TMUX": "/tmp/tmux-1/default,1,0", "KITTY_WINDOW_ID": "1"}) is None
    assert keys_mod.detect_terminal({"TERM_PROGRAM": "tmux", "WT_SESSION": "abc"}) is None
    # a variable present with an empty value counts as unset
    assert keys_mod.detect_terminal({"TMUX": "", "KITTY_WINDOW_ID": "1"}) == ("kitty", KITTY_LINE)


def test_detect_terminal_empty_environment():
    assert keys_mod.detect_terminal({}) is None
    assert keys_mod.detect_terminal({"TERM": "xterm-256color"}) is None
    assert keys_mod.detect_terminal({"KITTY_WINDOW_ID": ""}) is None
    assert keys_mod.detect_terminal({"TERM": "gnome-256color", "TERM_PROGRAM": "unknown"}) is None


def test_every_setup_line_is_plain_text():
    lines = [KITTY_LINE, GHOSTTY_LINE, WEZTERM_LINE, ALACRITTY_LINE, FOOT_LINE, ITERM_LINE,
             VSCODE_LINE, WT_LINE]
    seen = set()
    for env in ({"KITTY_WINDOW_ID": "1"}, {"TERM_PROGRAM": "ghostty"}, {"TERM_PROGRAM": "WezTerm"},
                {"TERM": "alacritty"}, {"TERM": "foot"}, {"TERM_PROGRAM": "iTerm.app"},
                {"TERM_PROGRAM": "vscode"}, {"WT_SESSION": "x"}):
        found = keys_mod.detect_terminal(env)
        assert found is not None
        name, line = found
        assert line.isprintable() and ESC not in line
        assert name.isprintable() and ESC not in name
        seen.add(line)
    assert seen == set(lines)


def test_help_block_named_terminal_form():
    block = keys_mod.help_block(
        _keys(), terminal=keys_mod.detect_terminal({"KITTY_WINDOW_ID": "1"}), shift_enter_seen=False
    )
    assert block == (
        "Keys:\n"
        r"  \ then enter       line break. shift+enter submits until kitty is set up — "
        r"kitty.conf: map shift+enter send_text all \x1b[13;2u"
        "\n"
        "  !text, alt+enter   send the line into the running turn (a nudge)"
    )


def test_help_block_seen_form():
    block = keys_mod.help_block(
        _keys(), terminal=keys_mod.detect_terminal({"KITTY_WINDOW_ID": "1"}), shift_enter_seen=True
    )
    assert _rows(block)[0] == r"\ then enter       line break — shift+enter too"
    # seen outranks an unnamed terminal too
    assert _rows(keys_mod.help_block(_keys(), terminal=None, shift_enter_seen=True))[0] == (
        r"\ then enter       line break — shift+enter too"
    )


@pytest.mark.parametrize(
    ("kw", "row1"),
    [
        ({}, r"\ then enter       line break (or give ctrl+o to it: "
             r"localharness components set terminal.keys.newline ctrl-o)"),
        ({"nudge": ["ctrl-o"]}, r"\ then enter       line break (or give ctrl+g to it: "
                                r"localharness components set terminal.keys.newline ctrl-g)"),
        ({"nudge": ["ctrl-o", "ctrl-g"]}, r"\ then enter       line break"),
        ({"newline": ["shift-enter", "ctrl-o"]}, r"\ then enter, ctrl+o   line break"),
        ({"newline": ["ctrl-g"]}, r"\ then enter, ctrl+g   line break"),
        ({"newline": ["alt-enter"]}, r"\ then enter, alt+enter   line break"),
    ],
)
def test_help_block_route_form(kw, row1):
    assert _rows(keys_mod.help_block(_keys(**kw), terminal=None, shift_enter_seen=False))[0] == row1


def test_help_block_route_form_second_row_follows_the_nudge_list():
    rows = _rows(keys_mod.help_block(_keys(nudge=["ctrl-o"]), terminal=None, shift_enter_seen=False))
    assert rows[1] == f"!text, ctrl+o      {COUNT}"


def test_help_block_with_empty_lists():
    rows = _rows(
        keys_mod.help_block(_keys(nudge=[], newline=[]), terminal=None, shift_enter_seen=False)
    )
    assert rows[1] == f"!text              {COUNT}"
    assert rows[0].startswith(r"\ then enter")
    assert "shift+enter" not in rows[0]


def test_help_block_shift_enter_as_nudge_only():
    kitty = keys_mod.detect_terminal({"KITTY_WINDOW_ID": "1"})
    rows = _rows(keys_mod.help_block(_keys(nudge=["shift-enter"]), terminal=kitty,
                                     shift_enter_seen=False))
    assert "shift+enter" not in rows[0]
    assert "(or give ctrl+o to it:" in rows[0]
    assert rows[1].startswith("!text, shift+enter ")


def test_help_block_names_other_newline_chords():
    rows = _rows(keys_mod.help_block(_keys(newline=["shift-enter", "ctrl-g", "alt-enter"],
                                           nudge=["ctrl-o"]),
                                     terminal=None, shift_enter_seen=True))
    assert rows[0].startswith(r"\ then enter, ctrl+g, alt+enter ")
    assert rows[0].endswith("line break — shift+enter too")
    # both columns share the one width
    assert rows[0].index("line break") == rows[1].index("send the line")


def test_help_block_never_points_to_the_readme_or_alt_enter():
    import itertools

    kitty = keys_mod.detect_terminal({"KITTY_WINDOW_ID": "1"})
    cases = 0
    for assignment in itertools.product(("nudge", "newline", None), repeat=4):
        nudge = [n for n, a in zip(CHORD_NAMES, assignment) if a == "nudge"]
        newline = [n for n, a in zip(CHORD_NAMES, assignment) if a == "newline"]
        for terminal in (None, kitty):
            for seen in (False, True):
                cases += 1
                block = keys_mod.help_block(_keys(nudge=nudge, newline=newline),
                                            terminal=terminal, shift_enter_seen=seen)
                assert len(block.split("\n")) == 3
                assert "README" not in block
                assert ESC not in block
                row1 = _rows(block)[0]
                routes = re.findall(r"\(or give (\S+) to it: .*? terminal\.keys\.newline (\S+)\)", row1)
                for label, name in routes:
                    assert name in ("ctrl-o", "ctrl-g") and label == LABELS[name]
                    assert name not in nudge and name not in newline
                if set(newline) & {"ctrl-o", "ctrl-g", "alt-enter"}:
                    assert "(or give" not in block
    assert cases == 324
