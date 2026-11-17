"""A private session on screen: the chip, the hint, the arm, the legend and answer lines, the trust form, /review, /instructions and /model."""
from __future__ import annotations

import os
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from localharness.agent.gate_types import GateSettings, ToolMeta
from localharness.agent.verdict import SCRATCH_OPTIONS_LEGEND, GateContext, evaluate
from localharness.channels.terminal import (
    PERMISSION_ANSWER_LINES,
    PERMISSION_ANSWER_LINES_PRIVATE_TRUST,
    PERMISSION_ANSWER_LINES_SESSION,
    TERMINAL_THEME,
    TerminalChannel,
    press,
)
from localharness.cli import session_trust
from localharness.cli.repl import PRIVATE_SENT_NOTE, REVIEW_PRIVATE_LINE
from localharness.cli.start_cmd import PRIVATE_HINT, _private_chip, _private_hint
from localharness.cli.ui import startup_banner
from localharness.cli.workspace import PRIVATE_TRUST_CLAUSE, TRUST_ONLY_PROMPT, settle_startup_trust
from localharness.core.bus import EventBus
from localharness.core.run_folder import begin, end
from tests.unit.test_instructions_trust_prompts import _answer, _tty, project  # noqa: F401
from tests.unit.test_repl_input_box import FakeBoxChannel, _repl

pytestmark = pytest.mark.skipif(os.name != "posix", reason="descriptor-relative run folders")

LOCAL, REMOTE = "http://127.0.0.1:8000/v1", "https://models.example.org:8443/v1"


@pytest.fixture
def private_run(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "config.yaml").write_text("{}")
    run = begin(cfg, private=True)
    yield run
    end(run)


def llm(url):
    return SimpleNamespace(config=SimpleNamespace(base_url=url))


def terminal() -> TerminalChannel:
    ch = TerminalChannel(EventBus(), {})
    ch._console = Console(file=StringIO(), force_terminal=True, width=120, theme=TERMINAL_THEME, highlight=False)
    return ch


def text(frags) -> str:
    return "".join(t for _, t in frags)


def test_the_chip_reads_private_here_and_private_remote_for_another_machine():
    assert _private_chip(llm(LOCAL)) == "private"
    assert _private_chip(llm(REMOTE)) == "private · remote"


def test_the_first_prompt_hint_names_the_host_only_when_the_chat_leaves_the_machine():
    assert _private_hint(llm(LOCAL)) == "Private: no saved memory; only files written into your project stay."
    assert _private_hint(llm(REMOTE)) == f"{PRIVATE_HINT} This chat goes to models.example.org:8443."


def test_the_banner_info_row_carries_the_chip_only_when_given():
    def row(**kw):
        console = Console(file=StringIO(), width=120)
        console.print(startup_banner("m", True, False, **kw))
        return console.file.getvalue()
    assert "private · remote" in row(chip="private · remote")
    assert "private" not in row()


def test_the_footer_shows_the_chip_live_and_the_armed_hint_replaces_the_left_hint():
    ch = terminal()
    assert "private" not in text(ch._box_instrument_frags())
    url = [LOCAL]
    ch.chip = lambda: _private_chip(llm(url[0]))
    assert text(ch._box_instrument_frags()) == "private"
    url[0] = REMOTE
    assert text(ch._box_instrument_frags()) == "private · remote"
    ch.box_set_armed("Ctrl+C or Ctrl+D again ends this private session (3 files go)")
    assert text(ch._box_hint_frags()).strip() == "Ctrl+C or Ctrl+D again ends this private session (3 files go)"
    ch.box_set_armed("")
    assert "tab commands" in text(ch._box_hint_frags())


def test_a_second_press_inside_a_second_is_ignored_and_one_after_it_exits():
    lines = ("arm", "footer")
    assert press(False, 0.0, 10.0, lines, eof=True) == "arm"
    assert press(True, 10.0, 10.5, lines, eof=False) == "ignore"
    assert press(True, 10.0, 11.5, lines, eof=True) == "exit"


def test_with_nothing_to_lose_ctrl_c_arms_as_before_and_ctrl_d_exits_at_once():
    assert press(False, 0.0, 1.0, None, eof=False) == "arm"
    assert press(False, 0.0, 1.0, None, eof=True) == "exit"
    assert press(True, 1.0, 1.1, None, eof=False) == "exit"


class ArmChannel(FakeBoxChannel):
    def __init__(self, lines):
        super().__init__()
        self._lines, self.armed = lines, []

    def arm_lines(self):
        return self._lines

    def box_set_armed(self, text):
        self.armed.append(text)


async def test_the_box_arms_with_what_goes_then_a_submit_disarms():
    channel = ArmChannel(("Press Ctrl+C or Ctrl+D again to end this private session: out.csv go.", "footer text"))
    repl, _, _ = _repl(channel=channel)
    assert await repl._handle_box_event("eof", None) is True
    assert channel.sent[-1][0].startswith("Press Ctrl+C or Ctrl+D again to end this private session")
    assert channel.armed == ["footer text"]
    assert await repl._handle_box_event("interrupt", None) is True
    assert len(channel.sent) == 1
    await repl._handle_box_event("submit", "")
    assert channel.armed[-1] == "" and not repl._sigint_armed
    repl._sigint_armed, repl._armed_at = True, -10.0
    assert await repl._handle_box_event("eof", None) is False


async def test_with_nothing_that_goes_ctrl_d_ends_at_once():
    repl, _, _ = _repl(channel=ArmChannel(None))
    assert await repl._handle_box_event("eof", None) is False


def test_a_grantable_ask_in_a_private_run_carries_the_session_legend_and_a_normal_run_does_not():
    def ask(private):
        ctx = GateContext(boundary="/p", workspace="/p", grants=lambda *a: None, has_review_surface=True,
                          mode="guarded", leaves_no_trace=private)
        return evaluate("bash_exec", {"command": "mkdir /p/sub"}, ToolMeta(destructive=True, group="shell"), ctx, GateSettings())
    assert ask(True).request.options_legend == SCRATCH_OPTIONS_LEGEND
    assert ask(False).request.options_legend is None


def test_the_session_answer_lines_say_this_session_and_the_private_trust_lines_say_private():
    assert PERMISSION_ANSWER_LINES_SESSION["allow_always"] == "✓ always (this session)"
    assert PERMISSION_ANSWER_LINES_SESSION["reject_always"] == "✗ never (this session)"
    assert PERMISSION_ANSWER_LINES_SESSION["allow_once"] == PERMISSION_ANSWER_LINES["allow_once"]
    assert PERMISSION_ANSWER_LINES_PRIVATE_TRUST["allow_once"] == "✓ trusted for this private session"
    assert PERMISSION_ANSWER_LINES_PRIVATE_TRUST["reject_always"] == "✗ not trusted — guarded for this private session"


def test_the_session_trust_form_swaps_only_the_recording_clause(private_run, tmp_path):
    private = session_trust._request(tmp_path)
    assert PRIVATE_TRUST_CLAUSE in private.display
    assert "records" not in private.display
    assert private.reason in private.display
    assert "guarded mode" in private.display


def test_a_normal_session_trust_form_is_unchanged(tmp_path):
    assert session_trust._request(tmp_path).display.endswith(session_trust.TRUST_QUESTION_DETAIL.format(root=tmp_path))


def test_the_startup_trust_question_in_a_private_run_keeps_the_effect_clauses_and_swaps_the_record(
        project, monkeypatch, private_run):
    _tty(monkeypatch)
    asked = _answer(monkeypatch, True)
    assert settle_startup_trust(None) is None
    assert asked == [f"{TRUST_ONLY_PROMPT} {PRIVATE_TRUST_CLAUSE}"]
    assert not (project / ".localharness").exists()


async def test_review_says_a_private_session_saves_nothing(private_run):
    repl, channel, _ = _repl()
    repl._session_gate = lambda: SimpleNamespace(boundary=None)
    await repl._apply_review(0, {})
    assert REVIEW_PRIVATE_LINE in channel.sent[-1][0]
    assert REVIEW_PRIVATE_LINE == "this session only: a private session saves nothing"


async def test_review_is_unchanged_in_a_normal_session():
    repl, channel, _ = _repl()
    repl._session_gate = lambda: SimpleNamespace(boundary=None)
    await repl._apply_review(0, {})
    assert "there is no project to save it for" in channel.sent[-1][0]


def test_instructions_off_ends_for_this_private_session(private_run):
    from localharness.config.project_records import run_instructions_command
    handle = SimpleNamespace(_declined=False, _set_failure=None, set_state=lambda s: None,
                             state=SimpleNamespace(name="CLAUDE.md", size=1, digest="", path="", state="off"))
    assert run_instructions_command(handle, "off") == "instructions off for this private session; /instructions on to use CLAUDE.md"


def test_the_model_row_says_where_the_chat_goes_only_for_a_changed_remote_endpoint(private_run):
    repl, _, _ = _repl()
    assert repl._sent_note(LOCAL, REMOTE) == PRIVATE_SENT_NOTE == "this private session is sent there"
    assert repl._sent_note(REMOTE, REMOTE) == ""
    assert repl._sent_note(REMOTE, LOCAL) == ""


def test_the_model_row_adds_nothing_in_a_normal_session():
    repl, _, _ = _repl()
    assert repl._sent_note(LOCAL, REMOTE) == ""
