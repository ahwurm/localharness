"""The interview app (cli/interview.py): real keys through the real bindings, and the gate that keeps it from a pipe."""
from __future__ import annotations

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.cli import interview, workspace
from localharness.cli.interview import Answer, Option, ask, build_interview
from tests.unit.interview_support import CANCEL, UNREACHED, Asked, scripted

TWO_MODELS = [
    Option("all", "All, serving qwen first", "every file above becomes a preset"),
    Option("choose", "Choose which, and which first", "pick the files"),
    Option("skip", "Skip for now", "nothing is set up"),
]
QUESTION = "Set up these 2 models?"


@pytest.fixture(autouse=True)
def a_person_at_a_capable_terminal(monkeypatch):
    monkeypatch.setattr(interview, "local_operator", lambda no_input: True)
    monkeypatch.setenv("TERM", "xterm-256color")


def _ask(keys: str, options=TWO_MODELS, **kw) -> Answer:
    kw.setdefault("default", 0)
    kw.setdefault("cancel", "skip")
    with create_pipe_input() as inp:
        inp.send_text(keys)
        return ask(QUESTION, options, input=inp, output=DummyOutput(), **kw)


@pytest.mark.parametrize(
    "keys,default,expected",
    [
        ("\r", 0, "all"),
        ("\x1b[B\r", 0, "choose"),
        ("\x1b[A\r", 0, "skip"),
        ("\r", 2, "skip"),
    ],
)
def test_enter_returns_the_highlighted_key(keys, default, expected):
    assert _ask(keys, default=default) == Answer((expected,), False, True)


def test_the_rendered_rows_hold_the_question_labels_hints_and_one_selected_row():
    app = build_interview(QUESTION, TWO_MODELS, default=1, cancel="skip", output=DummyOutput())
    fragments = app._lh_interview_fragments()
    text = "".join(t for _, t in fragments)
    assert QUESTION in text
    for o in TWO_MODELS:
        assert o.label in text and o.hint in text
    selected = [t for style, t in fragments if style == "class:selected"]
    assert len(selected) == 1 and "Choose which" in selected[0]


@pytest.fixture
def tty_streams(monkeypatch):
    class Stream:
        def __init__(self, tty):
            self.tty = tty

        def isatty(self):
            return self.tty

    def set_streams(stdin=True, stdout=True):
        monkeypatch.setattr("sys.stdin", Stream(stdin))
        monkeypatch.setattr("sys.stdout", Stream(stdout))
    monkeypatch.delenv("LOCALHARNESS_COMMIT", raising=False)
    return set_streams


def test_local_operator_needs_a_tty_on_both_ends(tty_streams):
    tty_streams()
    assert workspace.local_operator(False) is True
    tty_streams(stdin=False)
    assert workspace.local_operator(False) is False
    tty_streams(stdout=False)
    assert workspace.local_operator(False) is False


def test_local_operator_refuses_the_marker_no_input_and_a_missing_stdout(tty_streams, monkeypatch):
    tty_streams()
    assert workspace.local_operator(True) is False
    monkeypatch.setattr("sys.stdout", None)
    assert workspace.local_operator(False) is False
    tty_streams()
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "")
    assert workspace.local_operator(False) is False


def test_a_pipe_on_stdin_is_cancelled_unreached_and_no_application_is_built(monkeypatch, tty_streams):
    monkeypatch.setattr(interview, "local_operator", workspace.local_operator)
    tty_streams(stdin=False)
    built = []
    monkeypatch.setattr(interview, "build_interview", lambda *a, **k: built.append(1))
    assert ask(QUESTION, TWO_MODELS, default=0, cancel="skip") == Answer(("skip",), True, reached=False)
    assert built == []


ABC = [Option("a", "Alpha"), Option("b", "Beta"), Option("c", "Gamma")]
TWELVE = [Option(f"k{i}", f"Option {i}") for i in range(1, 13)]
DOWN, UP, SPACE = "\x1b[B", "\x1b[A", " "


def _multi(keys: str, options=ABC, default=frozenset({"a", "c"})) -> Answer:
    return _ask(keys, options, default=default, multi=True)


@pytest.mark.parametrize(
    "keys,expected",
    [
        ("\r", ("a", "c")),
        (SPACE + "\r", ("c",)),
        (DOWN + SPACE + "\r", ("a", "b", "c")),
        (SPACE + DOWN + DOWN + SPACE + "\r", ()),
    ],
)
def test_multi_toggles_with_space_and_returns_keys_in_option_order(keys, expected):
    assert _multi(keys) == Answer(expected, False, True)


def test_multi_in_toggle_order_still_returns_option_order():
    assert _multi(DOWN + DOWN + SPACE + UP + UP + SPACE + "\r", default=frozenset()).keys == ("a", "c")


@pytest.mark.parametrize("keys,expected", [("2\r", "b"), ("0\r", "a"), ("4\r", "a"), ("3\r", "c")])
def test_a_typed_number_jumps_only_inside_the_range(keys, expected):
    assert _ask(keys, ABC, default=0, cancel="a").keys == (expected,)


def test_in_multi_a_typed_number_toggles_that_option():
    assert _multi("2\r", default=frozenset()).keys == ("b",)
    assert _multi("0\r4\r", default=frozenset()).keys == ()


@pytest.mark.parametrize("keys,expected", [("12\r", "k12"), ("9\r", "k9"), ("10\r", "k10"), ("1\r", "k1")])
def test_two_digits_form_one_number_with_ten_or_more_options(keys, expected):
    assert _ask(keys, TWELVE, default=0, cancel="k1").keys == (expected,)


def test_in_multi_the_first_digits_toggle_is_undone_by_the_second():
    assert _multi("12\r", TWELVE, default=frozenset()).keys == ("k12",)
    assert _multi("1" + DOWN + "2\r", TWELVE, default=frozenset()).keys == ("k1", "k2")


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("keys", ["\x1b", "\x03", DOWN + "\x1b"])
def test_escape_and_ctrl_c_cancel_with_the_cancel_key(keys, multi):
    default = frozenset({"a"}) if multi else 0
    assert _ask(keys, ABC, default=default, cancel="c", multi=multi) == Answer(("c",), True, True)


def test_the_hint_line_names_the_keys_for_the_mode():
    single = build_interview(QUESTION, ABC, default=0, cancel="a", output=DummyOutput())
    multi = build_interview(QUESTION, ABC, default=frozenset(), cancel="a", multi=True, output=DummyOutput())
    assert single._lh_interview_fragments()[-1][1] == interview.HINT_SINGLE == "↑↓ move · Enter confirm"
    assert multi._lh_interview_fragments()[-1][1] == interview.HINT_MULTI == "↑↓ move · Space select · Enter confirm"


def test_multi_rows_begin_with_their_checkbox():
    app = build_interview(QUESTION, ABC, default=frozenset({"b"}), cancel="a", multi=True, output=DummyOutput())
    rows = [t for _, t in app._lh_interview_fragments() if t.startswith(("[x]", "[ ]"))]
    assert [r[:3] for r in rows] == ["[ ]", "[x]", "[ ]"]


def test_a_control_sequence_in_a_label_renders_as_text_not_as_a_byte():
    hostile = [Option("a", "Alpha\x1b[2K", "wipe‮"), Option("b", "Beta")]
    text = "".join(t for _, t in build_interview(
        "Q\x1b[H?", hostile, default=0, cancel="b", output=DummyOutput())._lh_interview_fragments())
    assert "\x1b" not in text and "‮" not in text
    assert "Alpha\\x1b[2K" in text and "wipe\\u202e" in text and "Q\\x1b[H?" in text


def _last_render(options, keys, **kw) -> str:
    with create_pipe_input() as inp:
        inp.send_text(keys)
        app = build_interview(QUESTION if "question" not in kw else kw.pop("question"), options,
                              input=inp, output=DummyOutput(), **kw)
        app.run(in_thread=True)
        return "".join(t for _, t in app._lh_interview_fragments())


def test_after_enter_the_render_is_one_summary_line():
    assert _last_render(TWO_MODELS, "\r", default=0, cancel="skip") == "Set up these 2 models? All, serving qwen first"


def test_the_multi_summary_lists_the_chosen_labels_or_none():
    assert _last_render(ABC, "\r", default=frozenset({"a", "c"}), cancel="a", multi=True,
                        question="Set up which?") == "Set up which? Alpha, Gamma"
    assert _last_render(ABC, "\r", default=frozenset(), cancel="a", multi=True,
                        question="Set up which?") == "Set up which? none"


def test_the_cancel_summary_says_cancelled():
    assert _last_render(ABC, "\x1b", default=0, cancel="a") == f"{QUESTION} (cancelled)"


def test_the_app_keeps_its_last_line_in_scrollback():
    assert build_interview(QUESTION, ABC, default=0, cancel="a", output=DummyOutput()).erase_when_done is False


@pytest.mark.parametrize("bad", [
    dict(options=[], default=0),
    dict(options=[Option("a", "A"), Option("a", "B")], default=0),
    dict(options=ABC, default=3),
    dict(options=ABC, default=-1),
    dict(options=ABC, default=0, cancel=""),
    dict(options=ABC, default=frozenset({"zzz"}), multi=True),
    dict(options=ABC, default=0, multi=True),
])
def test_invalid_arguments_raise_before_any_application_exists(bad, monkeypatch):
    monkeypatch.setattr(interview, "build_interview", lambda *a, **k: pytest.fail("built"))
    bad.setdefault("cancel", "a")
    with pytest.raises(ValueError):
        ask(QUESTION, **bad)


def test_an_empty_option_list_names_the_reason():
    with pytest.raises(ValueError, match="interview needs at least one option"):
        ask(QUESTION, [], default=0, cancel="a")


class _Sized(DummyOutput):
    def __init__(self, rows, columns):
        self._size = (rows, columns)

    def get_size(self):
        from prompt_toolkit.data_structures import Size
        return Size(*self._size)


def test_a_dumb_terminal_is_unreached_and_builds_nothing(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setattr(interview, "build_interview", lambda *a, **k: pytest.fail("built"))
    assert ask(QUESTION, ABC, default=0, cancel="c") == Answer(("c",), True, reached=False)


@pytest.mark.parametrize("rows,columns", [(2, 80), (24, 19)])
def test_a_terminal_below_three_rows_or_twenty_columns_is_unreached(rows, columns, monkeypatch):
    monkeypatch.setattr(interview, "build_interview", lambda *a, **k: pytest.fail("built"))
    out = _Sized(rows, columns)
    assert ask(QUESTION, ABC, default=0, cancel="c", output=out) == Answer(("c",), True, reached=False)


def test_a_failure_while_drawing_is_unreached_not_an_exception(monkeypatch):
    class Boom:
        def run(self, **kw):
            raise OSError("no terminal")

    monkeypatch.setattr(interview, "build_interview", lambda *a, **k: Boom())
    assert ask(QUESTION, ABC, default=0, cancel="c") == Answer(("c",), True, reached=False)


def test_a_real_pipe_on_stdin_is_never_read(monkeypatch, tty_streams):
    import os
    monkeypatch.setattr(interview, "local_operator", workspace.local_operator)
    r, w = os.pipe()
    os.write(w, b"\r")
    with os.fdopen(r) as pipe:
        monkeypatch.setattr("sys.stdin", pipe)
        assert ask(QUESTION, ABC, default=0, cancel="c") == Answer(("c",), True, reached=False)
        assert os.read(pipe.fileno(), 1) == b"\r"
    os.close(w)


def test_the_marker_and_no_person_leave_the_question_unasked(monkeypatch, tty_streams):
    monkeypatch.setattr(interview, "local_operator", workspace.local_operator)
    tty_streams()
    monkeypatch.setenv("LOCALHARNESS_COMMIT", "1")
    assert ask(QUESTION, ABC, default=0, cancel="c").reached is False




@pytest.mark.parametrize("keys,index", [(DOWN * 5, 5), (UP, 11), (DOWN * 12, 0), (UP * 12 + DOWN, 1)])
def test_the_options_window_has_the_height_the_output_allows_and_its_cursor_rides_the_highlight(keys, index):
    from prompt_toolkit.layout.dimension import to_dimension
    with create_pipe_input() as inp:
        inp.send_text(keys + "\r")
        app = build_interview(QUESTION, TWELVE, default=0, cancel="k1", input=inp, output=_Sized(6, 80))
        (rows,) = [w for w in app.layout.find_all_windows()
                   if "Option 12" in "".join(t for _, t in w.content.text())]
        assert to_dimension(rows.height).preferred == 3
        app.run(in_thread=True)
        assert app._lh_interview_state()["index"] == index
        assert rows.content.get_cursor_position().y == index


def test_scripted_answers_each_kind_and_records_what_was_asked(monkeypatch):
    script = scripted(monkeypatch, "all", CANCEL, UNREACHED, {"c", "a"})
    assert interview.ask(QUESTION, TWO_MODELS, default=0, cancel="skip") == Answer(("all",), False, True)
    assert script.asked == [Asked(QUESTION, tuple(TWO_MODELS), 0, "skip", False)]
    assert interview.ask(QUESTION, TWO_MODELS, default=0, cancel="skip") == Answer(("skip",), True, True)
    assert interview.ask(QUESTION, TWO_MODELS, default=0, cancel="skip") == Answer(("skip",), True, False)
    assert interview.ask("Which?", ABC, default=frozenset(), cancel="a", multi=True) == Answer(("a", "c"), False, True)
    with pytest.raises(AssertionError, match="asked"):
        interview.ask(QUESTION, TWO_MODELS, default=0, cancel="skip")
