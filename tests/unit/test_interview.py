"""The interview app (cli/interview.py): real keys through the real bindings, and the gate that keeps it from a pipe."""
from __future__ import annotations

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from localharness.cli import interview, workspace
from localharness.cli.interview import Answer, Option, ask, build_interview

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
