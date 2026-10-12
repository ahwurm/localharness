"""Phase 11-09 — the workspace-open hook in `start`.

One handle per session, built from the session's effective boundary with the basis the trust state
calls for, handed to the root loop, the children, the REPL and ACP, its receipt bound onto the
channel and its dim line riding the terminal's first hint. The drive tests run the REAL
`_start_async` offline with the recorders from test_workspace_state_landing.py; the unit tests at the
top call the module helpers directly.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import yaml
from rich.console import Console

from localharness.cli import start_cmd
from localharness.config import trust
from localharness.config.project_records import read_instructions
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _drive,
    _install_recorders,
    _one,
    _workspace_start,
)

CLAUDE = "# Rules\nUse tabs.\n"


def _project(ws: Path) -> Path:
    return ws.parent


def _put_instructions(proj: Path, text: str = CLAUDE) -> Path:
    path = proj / "CLAUDE.md"
    path.write_text(text, encoding="utf-8")
    return path


def _handle_of(rec) -> object:
    return _one(rec, "loop")["project_instructions"]


# ------------------------------------------------------------------ _open_project_instructions


def _open(tmp_path, monkeypatch, fake_home, *, boundary="proj", **over):
    """Call the helper against a real tmp project; returns (handle_or_None, seen_kwargs)."""
    proj = tmp_path / "proj"
    proj.mkdir(exist_ok=True)
    _put_instructions(proj)
    seen: dict = {}

    def _fake(root, **kw):
        seen["root"] = root
        seen.update(kw)
        return "HANDLE"

    monkeypatch.setattr("localharness.config.project_records.open_project_instructions", _fake)
    args = dict(
        workspace=proj / ".localharness", decision=True, trust_flag=False, sitting_id="s1",
        max_ratio=0.25, no_input=False, channel_mode="terminal", window_tokens=65536,
        config_dir=None,
    )
    args.update(over)
    return start_cmd._open_project_instructions(proj if boundary == "proj" else boundary, **args), seen


def test_no_boundary_builds_nothing(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, boundary=None)
    assert got is None and not seen


def test_a_declined_root_builds_nothing(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, decision=False)
    assert got is None and not seen


def test_basis_layer_when_trusted_with_a_layer(tmp_path, monkeypatch, fake_home):
    got, seen = _open(tmp_path, monkeypatch, fake_home, decision=True)
    assert got == "HANDLE" and seen["basis"] == "layer"
    assert seen["sitting_id"] == "s1" and seen["max_ratio"] == 0.25
    assert seen["window_tokens"] == 65536


def test_basis_trusted_when_trusted_without_a_layer(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=True, workspace=None)
    assert seen["basis"] == "trusted"


def test_basis_trusted_under_the_trust_flag_with_no_record(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=None, trust_flag=True)
    assert seen["basis"] == "trusted"


def test_basis_awaiting_trust_with_no_record_and_no_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, decision=None, trust_flag=False)
    assert seen["basis"] == "awaiting_trust"


def test_acp_internal_no_input_is_not_the_cli_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, channel_mode="acp", no_input=True)
    assert seen["no_input"] is False


def test_terminal_no_input_is_the_cli_flag(tmp_path, monkeypatch, fake_home):
    _got, seen = _open(tmp_path, monkeypatch, fake_home, channel_mode="terminal", no_input=True)
    assert seen["no_input"] is True


def test_a_broken_store_returns_none(tmp_path, monkeypatch, fake_home):
    proj = tmp_path / "proj"
    proj.mkdir()

    def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr("localharness.config.project_records.open_project_instructions", _boom)
    got = start_cmd._open_project_instructions(
        proj, workspace=None, decision=True, trust_flag=False, sitting_id="s", max_ratio=0.25,
        no_input=False, channel_mode="terminal", window_tokens=1000, config_dir=None,
    )
    assert got is None


def test_hint_with_instructions_helper():
    line = "Using CLAUDE.md (1 KB); /instructions off"
    assert start_cmd._hint_with_instructions("", line) == line
    assert start_cmd._hint_with_instructions("/help for commands.", line) == line
    assert start_cmd._hint_with_instructions("Describe a task, or /help for commands.", line) == (
        "Describe a task, or /help for commands. · " + line
    )


# ------------------------------------------------------------------ the drive: scope and basis


async def test_a_trusted_project_with_claude_md_hands_one_handle_to_the_loop(
        tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj)
    trust.record_trust(proj, True)
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None
    assert handle.candidates[0] == proj.resolve() / "CLAUDE.md"
    assert handle._basis == "layer"


async def test_no_instructions_file_means_no_handle(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(_project(ws), True)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert _handle_of(rec) is None


async def test_a_declined_root_gets_no_handle_anywhere(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj)
    trust.record_trust(proj, False)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert _handle_of(rec) is None


async def test_no_record_and_no_flag_is_awaiting_trust(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _put_instructions(_project(ws))
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None and handle._basis == "awaiting_trust"


async def test_the_trust_flag_with_no_record_is_trusted(tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _put_instructions(_project(ws))
    rec = _install_recorders(monkeypatch)

    await _start_async(None, False, False, None, trust_project=True)

    handle = _handle_of(rec)
    assert handle is not None and handle._basis == "trusted"


def test_the_store_is_untouched_by_building_the_handle(tmp_path, monkeypatch, fake_home):
    """read_instructions is importable from here and a fresh root has no record."""
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    assert read_instructions(_project(ws)) is None


# ------------------------------------------------------------------ the drive: wiring (Task 2)

def _tty(monkeypatch) -> StringIO:
    """A person at a terminal: stdin is a tty and the start console is a terminal."""
    import localharness.cli.workspace as ws_mod

    monkeypatch.setattr(ws_mod, "_stdin_is_a_terminal", lambda: True)
    buf = StringIO()
    monkeypatch.setattr(
        start_cmd, "console", Console(file=buf, force_terminal=True, width=400, color_system=None))
    return buf


def _record_terminals(monkeypatch) -> list:
    """Keep each TerminalChannel the drive builds (the REPL is stubbed, so nothing reads it)."""
    import localharness.channels.terminal as term

    made: list = []
    real = term.TerminalChannel.__init__

    def _init(self, *a, **kw):
        made.append(self)
        return real(self, *a, **kw)

    monkeypatch.setattr(term.TerminalChannel, "__init__", _init)
    return made


def _trusted_project(tmp_path, monkeypatch, fake_home, text: str = CLAUDE):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj, text)
    trust.record_trust(proj, True)
    return proj, ws


def _recorded(proj):
    return read_instructions(proj.resolve())


async def _read_bubble(channel) -> str:
    """Consume the channel's first hint the way the REPL does: one input bubble, app faked."""
    from localharness.channels import terminal as term_mod

    seen: list[str] = []

    class _FakeApp:
        async def run_async(self):
            return "typed"

    def _fake_build(history, prompt, hint, context_pct=None, model_names_fn=None, **_kw):
        seen.append(hint)
        return _FakeApp()

    real = term_mod._build_input_app
    term_mod._build_input_app = _fake_build
    try:
        await channel.start()
        await channel.read_input()
        await channel.stop()
    finally:
        term_mod._build_input_app = real
    return seen[0]


async def test_a_first_run_hint_carries_the_line_and_the_bubble_writes_the_yes_record(
        tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    _tty(monkeypatch)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    (channel,) = made
    line = handle.start_line
    assert line and line.startswith("Using CLAUDE.md (") and line.endswith("; /instructions off")
    assert channel.instructions_hint == line
    assert channel.instructions_on_delivered == handle.on_delivered
    assert channel.first_prompt_hint == "Describe a task, or /help for commands. · " + line
    assert _recorded(proj) is None  # nothing is written before the bubble shows the line

    shown = await _read_bubble(channel)

    assert line in shown
    record = _recorded(proj)
    assert record is not None and record.state == "yes"


async def test_a_returning_start_replaces_the_default_hint_with_the_line(
        tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    _tty(monkeypatch)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)
    monkeypatch.setattr("localharness.cli.start_cmd._first_prompt_hint",
                        lambda is_returning: "/help for commands.")

    await _drive()

    (channel,) = made
    assert channel.first_prompt_hint == _handle_of(rec).start_line


async def test_a_resumed_start_still_sets_the_hint_and_the_binding(
        tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async
    from localharness.cli.session_resume import Resume

    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    _tty(monkeypatch)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)
    resume = Resume(
        action=("enable", "example"), agent_name=AGENT, conversation=(), prior_context="",
        eviction_store=None, queued=(), gate_mode="auto", previous_sitting_id="old",
    )

    await _start_async(None, False, False, None, resume=resume)

    handle = _handle_of(rec)
    (channel,) = made
    assert channel.instructions_hint == handle.start_line
    assert channel.first_prompt_hint == handle.start_line
    assert channel.instructions_on_delivered == handle.on_delivered


async def test_no_input_sets_no_hint_and_writes_no_record(tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    _tty(monkeypatch)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _start_async(None, False, False, None, no_input=True)

    handle = _handle_of(rec)
    (channel,) = made
    assert handle is not None and handle._no_input is True
    assert channel.instructions_hint == ""
    assert "CLAUDE.md" not in channel.first_prompt_hint
    assert _recorded(proj) is None


async def test_piped_stdin_sets_no_hint_and_writes_no_record(tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()  # no tty: pytest's stdin is not a terminal

    (channel,) = made
    assert _handle_of(rec) is not None
    assert channel.instructions_hint == "" and channel.first_prompt_hint == ""
    assert _recorded(proj) is None


async def test_the_runner_and_the_repl_hold_the_same_handle(tmp_path, monkeypatch, fake_home):
    _trusted_project(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None
    assert _one(rec, "runner")["project_instructions"] is handle
    assert _one(rec, "repl")["instructions"] is handle


async def test_children_false_takes_the_handle_from_the_runner_only(
        tmp_path, monkeypatch, fake_home):
    _home, g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    proj = _project(ws)
    _put_instructions(proj)
    trust.record_trust(proj, True)
    cfg = g / "config.yaml"
    data = yaml.safe_load(cfg.read_text()) or {}
    data["instructions"] = {"children": False}
    cfg.write_text(yaml.safe_dump(data))
    rec = _install_recorders(monkeypatch)

    await _drive()

    handle = _handle_of(rec)
    assert handle is not None
    assert _one(rec, "runner")["project_instructions"] is None
    assert _one(rec, "repl")["instructions"] is handle


async def test_a_declined_root_hands_nothing_to_the_runner_the_repl_or_the_channel(
        tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    trust.record_trust(proj, False)
    _tty(monkeypatch)
    made = _record_terminals(monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    assert _handle_of(rec) is None
    assert _one(rec, "runner")["project_instructions"] is None
    assert _one(rec, "repl")["instructions"] is None
    (channel,) = made
    assert channel.instructions_on_delivered is None and channel.instructions_hint == ""
    assert _recorded(proj) is None


async def test_acp_hands_the_handle_to_the_trust_question_and_to_serve(
        tmp_path, monkeypatch, fake_home):
    from localharness.cli.start_cmd import _start_async

    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    rec = _install_recorders(monkeypatch)
    trust_kw: dict = {}
    serve_kw: dict = {}

    async def _fake_trust(gate, notice=None, **kw):
        trust_kw.update(kw)
        return "auto"

    monkeypatch.setattr("localharness.cli.session_trust.establish_session_trust", _fake_trust)

    class _Acp:
        instructions_on_delivered = None

        async def serve(self, **kw):
            serve_kw.update(kw)

        def __getattr__(self, name):
            raise AttributeError(name)

    acp = _Acp()
    await _start_async(None, False, False, None, channel_mode="acp", no_input=True,
                       acp_channel=acp)

    handle = _handle_of(rec)
    assert handle is not None
    assert handle._no_input is False  # ACP's internal no_input is not the CLI flag (D-06)
    assert trust_kw["instructions"] is handle
    assert serve_kw["instructions"] is handle
    assert acp.instructions_on_delivered == handle.on_delivered


# ------------------------------------------------------------------ the summary counts (Task 3)


def _summary(printed: list[str]) -> str:
    """The one startup summary line (it carries the startup timing)."""
    (line,) = [p for p in printed if "s startup)" in p]
    return line


def _seed_yes(proj: Path):
    from localharness.config.project_records import accept_instructions, first_candidate

    root = proj.resolve()
    candidate = first_candidate(root)
    assert candidate is not None
    return root, accept_instructions(root, candidate)


async def _printed_summary(printed: list[str]) -> str:
    await _drive()
    return _summary(printed)


def _capture(monkeypatch) -> list[str]:
    printed: list[str] = []
    monkeypatch.setattr(
        start_cmd.console, "print", lambda *a, **k: printed.append(" ".join(str(x) for x in a)))
    return printed


async def test_a_second_start_names_the_file_and_prints_no_instructions_line(
        tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    _seed_yes(proj)
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    summary = await _printed_summary(printed)

    assert "CLAUDE.md" in summary
    assert "(off)" not in summary and "(too large)" not in summary
    assert not [p for p in printed if "Using CLAUDE.md" in p]


async def test_the_first_start_also_names_the_file_without_a_printed_line(
        tmp_path, monkeypatch, fake_home):
    _trusted_project(tmp_path, monkeypatch, fake_home)
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    summary = await _printed_summary(printed)

    assert "CLAUDE.md" in summary
    assert not [p for p in printed if "Using CLAUDE.md" in p]


async def test_a_recorded_off_reads_off(tmp_path, monkeypatch, fake_home):
    from dataclasses import replace

    from localharness.config.project_records import write_instructions

    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    root, record = _seed_yes(proj)
    write_instructions(root, replace(record, state="off"))
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    assert "CLAUDE.md (off)" in await _printed_summary(printed)


async def test_a_model_edit_reads_model_edit(tmp_path, monkeypatch, fake_home):
    from dataclasses import replace

    from localharness.config.project_records import write_instructions

    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    root, record = _seed_yes(proj)
    # The file now holds bytes the model wrote: the record still binds the older digest and
    # names the new bytes as refused.
    new_text = "# Rules\nThe model wrote this.\n"
    (proj / "CLAUDE.md").write_text(new_text, encoding="utf-8")
    import hashlib

    write_instructions(root, replace(
        record, harness_written=hashlib.sha256(new_text.encode()).hexdigest()))
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    assert "CLAUDE.md (model edit; /instructions on)" in await _printed_summary(printed)


async def test_a_file_over_the_window_share_reads_too_large(tmp_path, monkeypatch, fake_home):
    # The start that prints the oversized line shows the line instead of the entity; the next
    # terminal start has seen it and reads "(too large)" in the counts.
    _trusted_project(tmp_path, monkeypatch, fake_home, text="x" * 300_000)
    buf = _tty(monkeypatch)
    _install_recorders(monkeypatch)

    await _drive()
    first = buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    await _drive()

    assert "CLAUDE.md (too large)" not in first and "not used" in first
    assert "CLAUDE.md (too large)" in buf.getvalue()


async def test_no_candidate_means_no_entity(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    trust.record_trust(_project(ws), True)
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    assert "CLAUDE.md" not in await _printed_summary(printed)


async def test_a_declined_root_has_no_entity(tmp_path, monkeypatch, fake_home):
    proj, _ws = _trusted_project(tmp_path, monkeypatch, fake_home)
    trust.record_trust(proj, False)
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    assert "CLAUDE.md" not in await _printed_summary(printed)


async def test_awaiting_trust_with_no_record_has_no_entity(tmp_path, monkeypatch, fake_home):
    _home, _g, ws = _workspace_start(tmp_path, monkeypatch, fake_home)
    _put_instructions(_project(ws))
    printed = _capture(monkeypatch)
    _install_recorders(monkeypatch)

    assert "CLAUDE.md" not in await _printed_summary(printed)
