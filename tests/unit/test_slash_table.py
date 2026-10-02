"""PAPI-07: ONE slash table feeds the REPL dispatcher, /help, the input completer and the phone's
`/api/protocol` commands[]; plugins append rows (`set_plugin_rows`), and a plugin row that fails
is contained and named (PAPI-11).

With no plugin rows every surface is byte-identical to what the hand-written if-chain served:
CORE and HELP_BEFORE below are that table and its /help render, verbatim, so adding a core command
edits them on purpose. The parity tests pin each core command's dispatch with a spy on its target
method: the same handler, the same argument slicing, the same case.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import textwrap
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.document import Document

from localharness.channels.terminal import SlashCommandCompleter
from localharness.cli.repl import OrchestratorREPL
from localharness.cli.slash_commands import (
    SLASH_COMMANDS,
    SlashCommand,
    all_rows,
    find_row,
    help_text,
    set_plugin_rows,
)
from tests.unit.channels.test_web_server import BEARER, _stack
from tests.unit.test_repl_unknown_slash import RecordingChannel, _build_repl

CORE = [
    ("/help", "Show this help message"),
    ("/agents", "List configured agents"),
    ("/model", "List available models; /model <name|number> to switch"),
    ("/reasoning", "Stream the model's reasoning while it thinks; /reasoning on|off"),
    ("/verbose", "Show reasoning and every tool call with its arguments; /verbose on|off"),
    ("/mode", "Permission mode for this session; /mode guarded|trusted|read-only"),
    ("/pending", "Tool calls parked for you to answer"),
    ("/approve", "Run a parked call; /approve [N] (default: the oldest)"),
    ("/deny", "Drop a parked call; /deny [N] (default: the oldest)"),
    ("/quit", "Exit LocalHarness"),
    ("/exit", "Exit LocalHarness"),
]

HELP_BEFORE = (
    "Available commands:\n"
    "  /help       Show this help message\n"
    "  /agents     List configured agents\n"
    "  /model      List available models; /model <name|number> to switch\n"
    "  /reasoning  Stream the model's reasoning while it thinks; /reasoning on|off\n"
    "  /verbose    Show reasoning and every tool call with its arguments; /verbose on|off\n"
    "  /mode       Permission mode for this session; /mode guarded|trusted|read-only\n"
    "  /pending    Tool calls parked for you to answer\n"
    "  /approve    Run a parked call; /approve [N] (default: the oldest)\n"
    "  /deny       Drop a parked call; /deny [N] (default: the oldest)\n"
    "  /quit       Exit LocalHarness\n"
    "  /exit       Exit LocalHarness\n"
    "\n"
    "Everything else is handled by the orchestrator through natural language."
)

EXAMPLE_HELP = "Show the example plugin's swatch settings"
EXAMPLE_TEXT = "example plugin: swatches render in #4a90d9 at 8px"
INFO = {"style": "system.info"}
ERROR = {"style": "system.error"}


@pytest.fixture
def rows():
    """`set_plugin_rows` for one test. The table is process-wide, so every test starts from the
    core rows alone and the plugin rows are emptied afterwards, whatever the test did."""
    assert all_rows() == SLASH_COMMANDS, "a plugin row leaked in from an earlier test"
    yield set_plugin_rows
    set_plugin_rows(())


def _example(handler=None) -> SlashCommand:
    return SlashCommand("/example", EXAMPLE_HELP, handler or AsyncMock(return_value=EXAMPLE_TEXT),
                        takes_args=True, plugin="example")


def _completions(text: str, completer: SlashCommandCompleter | None = None) -> list[tuple[str, str]]:
    found = (completer or SlashCommandCompleter()).get_completions(Document(text, len(text)), None)
    return [(c.text, c.display_meta_text) for c in found]


async def _phone_menu(tmp_path) -> list[dict]:
    _, _, _, client = await _stack(tmp_path)
    return (await client.get("/api/protocol", headers=BEARER)).json()["commands"]


# ------------------------------------------------------------ no plugin rows: nothing moved

def test_the_core_table_is_todays_eleven_rows_in_order():
    assert [tuple(row) for row in SLASH_COMMANDS] == CORE  # a row unpacks as (name, description)
    assert all_rows() == SLASH_COMMANDS


def test_help_is_byte_identical():
    assert help_text() == HELP_BEFORE


def test_the_completer_offers_the_same_menu():
    assert _completions("/") == CORE
    assert _completions("/m") == [row for row in CORE if row[0].startswith("/m")]
    assert _completions("/M") == _completions("/m")


async def test_the_phone_menu_is_the_same_eleven_rows_in_order(tmp_path):
    assert await _phone_menu(tmp_path) == [{"name": n, "description": d} for n, d in CORE]


# ------------------------------------------------------------ core dispatch: exact parity

PARITY = [
    ("/MODEL Qwen-X", "_handle_model_cmd", ("Qwen-X",), {}),        # original case kept
    ("  /model   Qwen-X  ", "_handle_model_cmd", ("Qwen-X",), {}),
    ("/model", "_handle_model_cmd", ("",), {}),
    ("/Reasoning ON", "_handle_reasoning_cmd", ("on",), {}),         # lowered + stripped
    ("/verbose off", "_handle_verbose_cmd", ("off",), {}),
    ("/mode guarded", "_handle_mode_cmd", ("guarded",), {}),
    ("/mode", "_handle_mode_cmd", ("",), {}),
    ("/approve 2", "_handle_pending_answer", (" 2",), {"approve": True}),   # lowered, NOT stripped
    ("/APPROVE  2 ", "_handle_pending_answer", ("  2",), {"approve": True}),
    ("/deny", "_handle_pending_answer", ("",), {"approve": False}),
    ("/pending", "_handle_pending_cmd", (), {}),
]


@pytest.mark.parametrize(("line", "method", "args", "kwargs"), PARITY)
async def test_each_core_command_reaches_the_same_handler_with_the_same_argument(
        line, method, args, kwargs):
    repl, _, _ = _build_repl(RecordingChannel([]))
    spy = AsyncMock()
    setattr(repl, method, spy)
    assert await repl._handle_slash(line) is True
    spy.assert_awaited_once_with(*args, **kwargs)


@pytest.mark.parametrize("line", ["/pending now", "/help me", "/agents all", "/quit now",
                                  "/modelx", "/tmp/foo", "/"])
async def test_a_line_no_row_claims_is_not_dispatched_as_that_command(line):
    """Only a row that takes arguments claims "name ..."; "/modelx" is its own word."""
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    repl._handle_pending_cmd = AsyncMock()
    repl._handle_model_cmd = AsyncMock()
    handled = await repl._handle_slash(line)
    repl._handle_pending_cmd.assert_not_awaited()
    repl._handle_model_cmd.assert_not_awaited()
    if line == "/modelx":  # a lone unknown word: rejected, never a turn (#48)
        assert handled is True
        assert channel.sent == [("Unknown command: /modelx — /help lists commands.", ERROR)]
    else:
        assert handled is False and channel.sent == []


async def test_help_sends_the_live_render():
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/HELP") is True
    assert channel.sent == [(help_text(), INFO)]


async def test_agents_lists_the_cards():
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/agents") is True
    assert channel.sent == [("No agents configured. Describe what you need and I'll create one.", INFO)]


async def test_quit_mid_wizard_cancels_the_creation_and_a_second_quit_exits(tmp_path):
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    repl._orchestrator.begin_agent_creation(config_dir=tmp_path)
    assert await repl._handle_slash("/quit") is True
    assert repl._orchestrator.active_workflow is None
    assert channel.sent == [("Agent creation cancelled. /quit again to exit.", INFO)]
    with pytest.raises(EOFError):
        await repl._handle_slash("/Exit")


def test_an_unknown_single_token_is_still_rejected_without_a_model_turn():
    channel = RecordingChannel(["/nope"])
    repl, agent, bus = _build_repl(channel)
    asyncio.run(repl.run())
    assert ("Unknown command: /nope — /help lists commands.", ERROR) in channel.sent
    agent.run_turn.assert_not_called()
    bus.publish.assert_not_called()


def test_the_dispatcher_names_no_command():
    """The if-chain is gone: `_handle_slash` looks its row up in the table and names no command."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(OrchestratorREPL._handle_slash)))
    strings = {node.value.strip() for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not strings & {name for name, _ in CORE}
    assert "find_row" in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def test_every_core_row_names_a_coroutine_on_the_repl():
    for row in SLASH_COMMANDS:
        assert row.plugin is None and isinstance(row.handler, str), row.name
        assert inspect.iscoroutinefunction(getattr(OrchestratorREPL, row.handler)), row.name
        assert find_row(row.name) is row


# ------------------------------------------------------------ plugin rows

async def test_a_plugin_row_reaches_every_surface(rows, tmp_path):
    completer = SlashCommandCompleter()  # built before the rows exist, as the terminal's one is
    assert rows([_example()]) == []
    assert all_rows()[:-1] == SLASH_COMMANDS and all_rows()[-1].name == "/example"
    assert help_text() == HELP_BEFORE.replace(
        "\n\nEverything", f"\n  /example    {EXAMPLE_HELP}\n\nEverything")
    assert _completions("/e", completer) == [("/exit", "Exit LocalHarness"), ("/example", EXAMPLE_HELP)]
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)  # the REPL module was imported long before the row existed
    assert await repl._handle_slash("/help") is True
    assert channel.sent == [(help_text(), INFO)] and "/example" in channel.sent[0][0]
    assert await _phone_menu(tmp_path) == [
        *({"name": n, "description": d} for n, d in CORE), {"name": "/example", "description": EXAMPLE_HELP}]


async def test_a_plugin_row_awaits_its_handler_with_the_arguments_and_shows_its_text(rows):
    handler = AsyncMock(return_value=EXAMPLE_TEXT)
    rows([_example(handler)])
    channel = RecordingChannel([])
    repl, agent, _ = _build_repl(channel)
    assert await repl._handle_slash("/EXAMPLE  Hi There ") is True
    handler.assert_awaited_once_with("Hi There")  # the original case, like /model
    assert channel.sent == [(EXAMPLE_TEXT, INFO)]
    handler.return_value = None  # nothing to say: nothing shown
    assert await repl._handle_slash("/example") is True
    handler.assert_awaited_with("")
    assert channel.sent == [(EXAMPLE_TEXT, INFO)]
    agent.run_turn.assert_not_called()


def test_a_raising_plugin_handler_is_named_and_the_session_goes_on(rows, caplog):
    rows([_example(AsyncMock(side_effect=RuntimeError("boom")))])
    channel = RecordingChannel(["/example now", "/help"])
    repl, agent, _ = _build_repl(channel)
    with caplog.at_level(logging.WARNING, logger="localharness.cli.repl"):
        asyncio.run(repl.run())
    failed = ("/example (plugin example) failed: RuntimeError: boom", ERROR)
    assert failed in channel.sent
    assert channel.sent.index((help_text(), INFO)) > channel.sent.index(failed)  # it went on
    assert any("example" in r.getMessage() and r.exc_info for r in caplog.records)
    agent.run_turn.assert_not_called()


@pytest.mark.parametrize(("outcome", "shown"), [
    (SystemExit(3), "SystemExit: 3"),      # sys.exit() in a handler must not end the harness
    (EOFError("gone"), "EOFError: gone"),  # the REPL reads EOFError as "exit"
])
async def test_a_plugin_handler_cannot_end_the_session(rows, outcome, shown):
    rows([_example(AsyncMock(side_effect=outcome))])
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/example") is True
    assert channel.sent == [(f"/example (plugin example) failed: {shown}", ERROR)]


class _RenderChannel(RecordingChannel):
    """RecordingChannel that also records rich renderables (the core /memory path's send_renderable)."""

    def __init__(self, inputs: list) -> None:
        super().__init__(inputs)
        self.rendered: list = []

    async def send_renderable(self, renderable) -> None:
        self.rendered.append(renderable)


@pytest.mark.parametrize("bad", [42, object()])
async def test_a_plugin_handler_that_returns_neither_text_nor_a_renderable_is_named(rows, bad):
    rows([_example(AsyncMock(return_value=bad))])
    channel = _RenderChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/example") is True
    assert channel.sent == [(f"/example (plugin example) failed: TypeError: returned "
                             f"{type(bad).__name__}, not text or a renderable", ERROR)]
    assert channel.rendered == []


async def test_a_plugin_handler_may_return_a_rich_renderable(rows):
    from rich.text import Text
    reply = Text("x")
    rows([_example(AsyncMock(return_value=reply))])
    channel = _RenderChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/example") is True
    assert channel.rendered == [reply] and channel.sent == []


@pytest.mark.parametrize(("reply", "sent"), [(None, []), ("", []), ("hi", [("hi", INFO)])])
async def test_a_plugin_handlers_text_or_silence_is_unchanged(rows, reply, sent):
    rows([_example(AsyncMock(return_value=reply))])
    channel = _RenderChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/example") is True
    assert channel.sent == sent and channel.rendered == []


async def test_a_plugin_row_cannot_take_a_name_already_in_the_table(rows):
    evil = SlashCommand("/help", "not the help", AsyncMock(), plugin="evil")
    twin = SlashCommand("/example", "a second /example", AsyncMock(), takes_args=True, plugin="twin")
    warnings = rows([evil, _example(), twin])
    assert len(warnings) == 2
    assert "/help" in warnings[0] and "evil" in warnings[0]
    assert "/example" in warnings[1] and "twin" in warnings[1]
    assert [row.name for row in all_rows()] == [n for n, _ in CORE] + ["/example"]
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/help") is True
    assert channel.sent == [(help_text(), INFO)]
    evil.handler.assert_not_awaited()


def test_a_plugin_row_without_a_callable_handler_is_skipped(rows):
    """Only core rows name a REPL method; a plugin row that did would be dispatched as core."""
    warnings = rows([SlashCommand("/example", EXAMPLE_HELP, "_slash_quit", plugin="example")])
    assert len(warnings) == 1 and "example" in warnings[0]
    assert all_rows() == SLASH_COMMANDS


async def test_setting_rows_replaces_them_and_empty_rows_remove_them_all(rows):
    rows([_example()])
    other = SlashCommand("/other", "another plugin's command", AsyncMock(), plugin="other")
    assert rows([other]) == []
    assert [row.name for row in all_rows()][-1:] == ["/other"] and find_row("/example") is None
    assert rows(()) == []
    assert all_rows() == SLASH_COMMANDS and help_text() == HELP_BEFORE and _completions("/") == CORE
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/other") is True  # an unknown word again
    assert channel.sent == [("Unknown command: /other — /help lists commands.", ERROR)]


# ------------------------------------------------------------ /memory is the memory plugin's row

MEMORY_HELP = "Browse the agent's memory by tag; show/forget/search a memory"


def _memory_row(handler) -> SlashCommand:
    """The row the lifecycle builds from the memory manifest (plugins/lifecycle.py), handler spied."""
    from localharness.memory.plugin import MemoryPlugin
    (desc,) = MemoryPlugin.manifest.slash
    assert (desc.name, desc.help, desc.target) == (
        "/memory", MEMORY_HELP, "localharness.memory.plugin:MemoryPlugin.slash_memory")
    return SlashCommand(desc.name, desc.help, handler, takes_args=True, plugin="memory")


async def test_memory_is_no_core_row_and_follows_the_core_rows_as_the_memory_plugins(rows, tmp_path):
    """G5 made structural: no core row, so no running memory plugin means no /memory anywhere. With
    the plugin's row installed it reaches every surface — M3: plugin rows follow core rows, so
    /memory now sits AFTER /quit and /exit (before 48 it sat before them)."""
    assert "/memory" not in [r.name for r in SLASH_COMMANDS]
    assert find_row("/memory") is None and "/memory" not in help_text()
    assert [c for c, _ in _completions("/me")] == []
    handler = AsyncMock(return_value="ok")
    assert rows([_memory_row(handler)]) == []
    names = [r.name for r in all_rows()]
    assert names == [n for n, _ in CORE] + ["/memory"] and names.index("/memory") > names.index("/exit")
    assert help_text() == HELP_BEFORE.replace("\n\nEverything", f"\n  /memory     {MEMORY_HELP}\n\nEverything")
    assert _completions("/me") == [("/memory", MEMORY_HELP)]
    assert (await _phone_menu(tmp_path))[-1] == {"name": "/memory", "description": MEMORY_HELP}
    channel = RecordingChannel([])
    repl, _, _ = _build_repl(channel)
    assert await repl._handle_slash("/memory Show 12") is True
    handler.assert_awaited_once_with("Show 12")  # original case kept, as the core row did
    assert channel.sent == [("ok", INFO)]
