"""TerminalChannel — Rich-formatted terminal output with streaming and prompt_toolkit input."""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable

import structlog
from prompt_toolkit.application import Application
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.buffer import Buffer, CompletionState
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.filters import Condition, has_completions
from prompt_toolkit.formatted_text.utils import fragment_list_width
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import (
    AfterInput, AppendAutoSuggestion, BeforeInput, ConditionalProcessor,
)
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from rich.theme import Theme

from localharness.channels.base import (
    PERMISSION_DENIED_LINE,
    ChannelAdapter,
    sanitize_for_display,
)
from localharness.channels.errors import ChannelStartError
from localharness.cli.theme import ENTITY_STYLES, SITE_INK
from localharness.core.bus import EventBus
from localharness.channels.input_router import FORCE_PREFIX
from localharness.core.events import (
    Action,
    CompactionTriggered,
    ConsolidationFinished,
    ConsolidationStarted,
    Escalation,
    Heartbeat,
    Observation,
    ParseFailed,
    PermissionStaged,
    TaskComplete,
    TurnFailed,
)
from localharness.tools.capabilities import UNTRUSTED_INGEST

log = structlog.get_logger(__name__)

# Tool call/result display characters (CONTEXT.md locked decisions)
_DIAMOND = "\u25c6"   # ◆  tool call indicator
_CHECK = "\u2713"     # ✓  tool result success
_CROSS = "\u2717"     # ✗  tool result error

# Echo glyph for a type-anytime box submission printed into the scrollback (FIX 1). The
# persistent box resets its buffer on submit, so without a permanent echo the user's own
# prompt would vanish from the transcript. Rendered at column 0 (vs the 2-space tool indent)
# in the user.input style — ink-user / green-agent gives the scrolled-back log a color
# language, so a conversation still shows what the user said and when. (The user used to
# be the green half; the agent took the accent green when the palette moved to the
# architecture-plate hues, and the user moved to primary ink to stay distinguishable.)
_PROMPT_GLYPH = "❯"  # ❯

# Official label for the background-memory ("dreaming") status (#20): middle-dot + ellipsis.
_DREAMING_LABEL = "· dreaming…"   # · dreaming…

# In-turn narration: an interstitial llm_response (content emitted ALONGSIDE tool calls)
# carries the model's progress narration ("pulling the data…"). Rendered as one dim line
# (middle-dot idiom, matching _DREAMING_LABEL / the burst separators) opening the coming
# chunk of tool activity — cropped to the first non-empty line and hard-capped so a chatty
# model can't wall-of-text the turn (per-call truth stays on the bus ledger). A tool-less
# llm_response IS the final answer (rendered by the TaskComplete panel) — never here.
_NARRATE = "·"        # ·  narration line indicator
_MAX_NARRATION = 160       # hard char cap before the ellipsis
_REASON = "⋯"              # ⋯  streamed reasoning line (terminal.show_reasoning)
_REASON_FLUSH_CHARS = 240  # a paragraph with no newline yet still streams in pieces

# Braille spinner frames for the IN-FRAME working glyph. While the persistent input box is
# live, the thinking/burst indicator advances inside the box's bottom border (a
# FormattedTextControl fragment refreshed by app.invalidate) instead of a rich Status/Live —
# rich spinners under patch_stdout glue lines and, worse, FREEZE on Ctrl+C-during-burst.
_SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# Status-row phases while the model generates (see _working_frags). One icon per kind of
# delta: ⋯ thinking (reasoning), ✎ writing (answer text), ◆ tool call (arguments), … waiting
# (no delta yet: queue wait or prefill). The icons double as the tally markers.
_PHASE_ICONS = {"waiting": "…", "thinking": "⋯", "writing": "✎", "tool_call": "◆"}
_PHASE_LABELS = {"waiting": "waiting", "thinking": "thinking", "writing": "writing",
                 "tool_call": "tool call"}
_SILENCE_NOTE_SECONDS = 10.0   # no delta for this long mid-stream → say so on the row

# --- The parked-call row, under the bottom bar (`auto` staging, owner ruling 2026-09-12) ---
PENDING_NOTICE_LINE_TERMINAL = "⏸ needs you  #{id}  {rendering}"
"""The terminal's transcript copy of a parked call: no answering legend, because the row under
the box (:data:`BOX_PENDING_ROW`) carries it for exactly as long as it applies, and the shared
:data:`PENDING_NOTICE_LINE` wrapped its legend to column 0 on a real pane (dogfood 2026-09-12)."""

BOX_PENDING_ROW = "⏸ {count} {noun} pending  #{id}  {rendering}   ctrl+y run · ctrl+n skip · /pending"
BOX_PENDING_ROW_LEGEND_SEP = "   "
"""The three spaces before the legend in :data:`BOX_PENDING_ROW` — where a too-wide row breaks."""
"""The one line a staged call gets at the input, in the owner's own words for it: "ignore or
move to background (under the bottom bar, one decision pending)".

It sits BELOW the footer for exactly that reason — a parked call is not a question holding the
turn open, it is a note the person may leave sitting there all session — and it names the
OLDEST call, which is the one both hotkeys answer, with the rest as a count. Everything needed
to answer it is on the line: the number, the command that was read, and the three ways to
reply."""

BOX_PENDING_NOUNS = ("decision", "decisions")
"""Singular/plural for the row and the away-return line, indexed by ``count != 1``. One tuple so
the two lines can never disagree about how English works."""

PRESENCE_WINDOW_S = 300.0
"""How long after the last keystroke somebody still counts as being at this keyboard.

Source: the freedesktop/GNOME session idle default (``org.gnome.desktop.session idle-delay`` =
300 s) — the desktop's own definition of "the person stepped away", which is the same question
asked about the same person at the same machine, so it is the number to borrow rather than one
invented here. It decides two things and nothing else: whether a staged call rings the bell, and
whether coming back to the keyboard earns the away-return line. Derivable from config later if a
real workflow disagrees with the desktop; it is not a knob today, because nobody has asked for
one and an unused knob is a promise to keep it working."""

AWAY_RETURN_LINE = (
    "while you were away: {count} {noun} pending — ctrl+y runs the oldest, /pending lists them"
)
"""Printed once when the first keystroke arrives after a :data:`PRESENCE_WINDOW_S` gap and
something is parked.

The bell only helps somebody who was there to hear it. Whoever walks back to a finished turn has
a screenful of scrollback and a one-line bar to notice, so the return itself is the moment to
say it — once, at the top of what they are about to type, not on every key."""


def _fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"

# Burst consolidation: consecutive calls from one tool family collapse into a single
# live counter line (`◆ web_search · web_fetch · 30/30`) instead of a line per call —
# a 30-hit research burst is one scrollback line, not 60. Per-call truth (args, errors,
# timings) stays on the bus ledger (bus-events.jsonl); this is display-only. Family
# membership reuses the capability metadata. (Descriptive labels dropped 2026-07-14 —
# owner: cleaner without; the close_note security disclosure stays.)
# Display families (owner, 2026-09-11: "group similar tools like web search — the all-purple
# wall"): the read-only local tools and the memory tools collapse the same way. Side-effecting
# tools stay ITEMIZED on purpose — bash_exec / write / edit / python_exec (HOST_DANGEROUS),
# cruncher_exec and agent delegations print one line per call, because each command, write and
# hand-off is the user's audit trail. The memory family wears the memory hue of the
# architecture plates (cli/theme.py), so a turn's tool section is no longer one color.
# `verbose` (--verbose / /verbose) is the opt-out: every call itemized with its arguments.
LOCAL_READ = frozenset({"read", "glob", "grep", "load_document", "chunk"})
MEMORY_TOOLS = frozenset({"memory_search", "memory_get", "remember"})
_UNTRUSTED_NOTE = "web results — UNTRUSTED, treated as data only"
_BURST_GROUPS: tuple[tuple[frozenset[str], str | None, str], ...] = (
    (UNTRUSTED_INGEST, _UNTRUSTED_NOTE, "tool.call"),
    (frozenset({"tool_result_get"}), None, "tool.call"),
    (LOCAL_READ, None, "tool.call"),
    (MEMORY_TOOLS, None, "memory.call"),
)


@dataclass
class _Burst:
    """Consolidation state for one open family burst (display-only)."""
    family: frozenset[str]
    close_note: str | None       # ✓ line printed once per user turn on close (None = no note)
    style: str = "tool.call"     # rich style of the counter line: the family's entity hue
    tools: list[str] = field(default_factory=list)  # first-use order, deduped
    calls: int = 0
    done: int = 0
    errors: int = 0
    status: Status | None = None  # live spinner while the burst is open (TTY only)


# Input bubble (Claude Code style): an inline prompt_toolkit Application drawing a
# fully closed rounded box around the buffer — bottom border hugs the input line
# (a PromptSession bottom_toolbar would pin it to the screen bottom instead).
INPUT_STYLE = Style.from_dict({
    "frame": "ansibrightblack",
    "caret": f"bold {SITE_INK}",           # ❯ in the box = ❯ of the echoed line in scrollback (one glyph, one ink)
    "input": "ansidefault",                # typed text: terminal default fg (else inherits the dim frame class)
    "auto-suggestion": "ansibrightblack",  # ghost history completion stays dim
    "placeholder": "ansibrightblack italic",  # empty-box guidance (the #49 first-run hint)
    "hint": "ansibrightblack italic",
    "model": ENTITY_STYLES["provider"],    # footer model chip: the model is the provider's name (cli/theme.py)
    # context meter — GSD thresholds (green → yellow → orange → red at compaction)
    "ctx-low": "ansigreen",
    "ctx-mid": "ansiyellow",
    "ctx-high": "#ff8700",
    "ctx-crit": "ansired bold",
    "ctx-track": "ansibrightblack",
    # measured tok/s bands (speed_stats.tps_band: >30 / 20–30 / <20) — status row + /model picker
    "tps-green": "ansigreen",
    "tps-yellow": "ansiyellow",
    "tps-red": "ansired bold",
    # generation phase (status row): the kind of delta arriving right now
    "phase-thinking": "ansiblue",
    "phase-writing": "ansigreen",
    "phase-tool_call": "ansicyan",
    "phase-waiting": "ansiyellow",
    "phase-silent": "ansired bold",
})


def _ctx_segments(pct: float) -> tuple[list[tuple[str, str]], int]:
    """GSD-style 10-cell context meter: ████░░░░░░ 42%. Returns (fragments, plain width).

    Color steps match gsd-statusline thresholds; localharness summary-compaction fires
    at 80% (ctx-crit), so red == "compacting now," not "out of room."
    """
    pct = max(0.0, min(100.0, pct))
    level = ("ctx-low" if pct < 50 else "ctx-mid" if pct < 65
             else "ctx-high" if pct < 80 else "ctx-crit")
    filled = min(10, int(pct // 10))
    label = f" {pct:.0f}%"
    frags = [
        (f"class:{level}", "█" * filled),
        ("class:ctx-track", "░" * (10 - filled)),
        (f"class:{level}", label),
    ]
    return frags, 10 + len(label)


# 'N t/s measured' / '~N t/s measured' notes in /model listings and swap messages.
_RATE_NOTE = re.compile(r"~?(\d+(?:\.\d+)?) t/s measured")
_RATE_STYLE = {"green": "green", "yellow": "yellow", "red": "bold red"}


def _colorize_rate_notes(content: str) -> Text:
    """Plain string → rich Text with every rate note styled by its speed band
    (speed_stats.tps_band). Text SPANS, never markup parsing: arbitrary surrounding
    content (model names, URLs, brackets) stays literal, so this path needs no
    escaping and cannot be style-injected — the safe way to color one substring of
    an otherwise-untrusted line."""
    from localharness.provider.speed_stats import tps_band

    text = Text()
    pos = 0
    for m in _RATE_NOTE.finditer(content):
        text.append(content[pos:m.start()])
        text.append(m.group(0), style=_RATE_STYLE[tps_band(float(m.group(1)))])
        pos = m.end()
    text.append(content[pos:])
    return text


class SlashCommandCompleter(Completer):
    """Complete REPL slash commands when the buffer is a single leading-slash token. Prefix match
    (case-insensitive) against the SLASH_COMMANDS table — the same table /help renders, so the menu
    and /help never drift. Command arguments are left alone (e.g. `/memory show 12`) with ONE
    exception: `/model <partial>` completes its first argument from `model_names_fn` — the
    session's cached live-model list. The supplier must never fetch or block (an empty list just
    means no menu); ids filter case-insensitively but insert case-true (model ids are
    case-sensitive)."""

    def __init__(self, model_names_fn: Callable[[], list[str]] | None = None) -> None:
        from localharness.cli.slash_commands import SLASH_COMMANDS
        self._commands = SLASH_COMMANDS
        self._model_names_fn = model_names_fn

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        head, sep, tail = text.partition(" ")
        if not sep:
            for name, desc in self._commands:
                if name.startswith(text.lower()):
                    yield Completion(name, start_position=-len(text), display=name, display_meta=desc)
            return
        if head.lower() != "/model" or " " in tail or self._model_names_fn is None:
            return
        for item in self._model_names_fn() or []:
            name, meta = item if isinstance(item, tuple) else (item, "model")
            if name.lower().startswith(tail.lower()):
                # style is the one-Enter marker: Enter on a highlighted MODEL completion
                # submits the switch in the same stroke (picking a model is an action,
                # not text editing) — commands keep the accept-then-Enter contract.
                yield Completion(name, start_position=-len(tail), display=name,
                                 display_meta=meta, style="class:model-pick")


def _accept_completion(buf: Buffer) -> "Completion | None":
    """Menu-aware Enter helper. If an item is highlighted, apply it and return the applied
    Completion; the caller consumes Enter for COMMAND completions (the command lands in the
    buffer, nothing submits) but submits in the same stroke for MODEL completions (style
    'model-pick' — the one-Enter picker). Menu open with nothing highlighted: close it and
    return None so the caller submits the fully-typed line (Claude Code feel). No menu → None."""
    cs = buf.complete_state
    if cs is None:
        return None
    comp = cs.current_completion
    if comp is not None:
        buf.apply_completion(comp)
        return comp
    buf.complete_state = None
    return None


def _is_model_pick(comp) -> bool:
    return comp is not None and "model-pick" in (comp.style or "")


# What the /model picker pre-fills into the box (box_open_model_menu). Named here because the
# menu-dismiss Esc must be able to take it back OUT again — declining the picker may not strand
# a `/model ` prefix that turns the user's next line into "Unknown model '<their sentence>'".
_MODEL_PICK_PREFIX = "/model "


def _picker_loan_live(buf: Buffer, prefix: str) -> bool:
    """Is the picker's pre-fill still a LOAN — the app's text, which the user has not made their
    own? True only while the line IS the loan: the bare prefix, or the prefix under the menu's
    preview of a highlighted model (the one-Enter picker leaves that preview in the buffer, and
    it is what the next thing typed used to be appended to). Self-expiring: any other line means
    the user has taken the line over, so the loan is dropped for good and a `/model ` they type
    by hand later is never mistaken for the app's own pre-fill."""
    if not prefix or not getattr(buf, "_lh_picker_loan", False):
        return False
    cs = buf.complete_state
    live = buf.text == prefix or (cs is not None and cs.original_document.text == prefix)
    if not live:
        buf._lh_picker_loan = False
    return live


def _replace_picker_loan(buf: Buffer, data: str) -> None:
    """Take the loan back and enter `data` on the empty line it leaves behind."""
    buf._lh_picker_loan = False
    buf.reset()
    buf.insert_text(data)


def _add_menu_keys(kb: KeyBindings, buf: Buffer, *, injected_prefix: str = "") -> None:
    """Tab and Esc for the completion menu, shared by both input apps. Tab (menu closed) accepts a
    sole match outright, else opens the menu with the first item highlighted; Tab (menu open)
    highlights the first item or advances to the next. Esc dismisses. Arrow navigation is
    prompt_toolkit's built-in behaviour once the menu is open. Tab computes completions
    synchronously so it engages immediately, not only after complete-while-typing has fired.

    `injected_prefix` is text the app itself put in the buffer to open the menu (the /model
    picker's pre-fill): dismissing a menu the user never asked for must leave the line empty."""

    @kb.add("tab")
    def _complete(event) -> None:
        b = event.current_buffer
        if b.complete_state is not None:
            if b.complete_state.current_completion is None:
                b.go_to_completion(0)
            else:
                b.complete_next()
            return
        if b.completer is None:
            return
        comps = list(b.completer.get_completions(b.document, CompleteEvent(completion_requested=True)))
        if not comps:
            return
        if len(comps) == 1:
            b.apply_completion(comps[0])
        else:
            b.complete_state = CompletionState(original_document=b.document, completions=comps)
            b.go_to_completion(0)

    @kb.add("escape", filter=has_completions, eager=True)
    def _dismiss(event) -> None:
        # eager: Esc resolves on its OWN key press. Without it prompt_toolkit holds the press to
        # see whether a second key completes the `escape, enter` (Alt+Enter) chord, so Esc-then-
        # Enter inside the 1s timeout submitted a FORCED NUDGE with the menu still open instead
        # of dismissing and submitting normally.
        b = event.current_buffer
        b.cancel_completion()  # back to the pre-preview line — a highlighted item is not a pick
        if injected_prefix and b.text == injected_prefix:
            b.reset()  # the picker's own pre-fill: dismissing means "never mind", not "/model "

    on_loan = Condition(lambda: _picker_loan_live(buf, injected_prefix))

    @kb.add("<any>", filter=on_loan)
    def _type_over_loan(event) -> None:
        # #135: typing while the pre-fill is on loan REPLACES it — declining by typing is Esc
        # plus typing. Appending is what turned the next line into '/model qwen3.8-27b/quit':
        # an "Unknown model" error where the user typed /quit, and a quit that never happened.
        if len(event.data) != 1 or not event.data.isprintable():
            return  # special key (F-key, a lone Esc flush): never insert its raw sequence
        _replace_picker_loan(event.current_buffer, event.data)

    @kb.add("<bracketed-paste>", filter=on_loan)
    def _paste_over_loan(event) -> None:
        _replace_picker_loan(event.current_buffer, event.data)  # pasting a line is typing it

    @kb.add("backspace", filter=on_loan)
    @kb.add("delete", filter=on_loan)
    def _erase_loan(event) -> None:
        # Backspacing into the loan is declining it too. Deleting one char of text the user
        # never typed would expire the loan and strand '/model qwen3.8-27' for the next line
        # to append to — the same trap as typing. The loan is one unit: erasing erases it all.
        _replace_picker_loan(event.current_buffer, "")


def _menu_float(body) -> FloatContainer:
    """Host a completion menu as a Float over the input body — part of the pt layout, never a
    rich Live/Status overlay (which is what could freeze the screen near the box)."""
    return FloatContainer(
        content=body,
        floats=[Float(xcursor=True, ycursor=True,
                      content=CompletionsMenu(max_height=12, scroll_offset=1))],
    )


PERMISSION_KEYS_GRANTABLE: dict[str, str] = {
    "y": "allow_once",
    "a": "allow_always",
    "n": "reject_once",
    "N": "reject_always",
}
"""The terminal's rendering of PRD §3.5's four option kinds: `[y]es once / [a]lways here /
[n]o / [N]ever here`. One keystroke each, lower/upper case distinguishing "this time" from
"forever" — the shape interactive patch-staging and package-manager prompts already taught
these fingers."""

PERMISSION_KEYS_UNGRANTABLE: dict[str, str] = {"y": "allow_once", "n": "reject_once"}
"""PRD §3.5: "ungrantable classes offer only the `_once` pair" — a destructive or protected-path
call asks every time by construction, so offering "always here" would be a lie."""

PERMISSION_OPTIONS_GRANTABLE = "[y]es once   [a]lways here   [n]o   [N]ever here"
PERMISSION_OPTIONS_UNGRANTABLE = "[y]es once   [n]o     (asks every time — cannot be remembered)"
"""The legend under the question. Written out rather than generated from the key maps because
it is the sentence a human reads at 2am, and the wording is the design."""

PERMISSION_DEFAULT_DECISION = "reject_once"
"""Enter, Escape and Ctrl+C all mean "no, this once" (PRD §3.5 fail-closed). Never
`reject_always`: an accidental keystroke must not write durable state."""

PERMISSION_PROMPT_LABEL = "Permission needed"

PERMISSION_ANSWER_LINES: dict[str, str] = {
    "allow_once": "✓ allowed once",
    "allow_always": "✓ always here",
    "reject_once": "✗ denied",
    "reject_always": "✗ never here",
}
"""What stays on screen after the keystroke — one short line, keyed by `DecisionKind` so the
confirmation and the four options cannot drift apart.

The owner's complaint, 2026-09-11: "I don't like how it persists a chat box every time I answer
(a) (y) or whatever; it should just record my input." Answering used to leave THREE artifacts in
the scrollback — the dead input box the question suspended, the option legend, and then a fresh
live box — so a session with six prompts in it had six dead boxes to scroll past. Now the two
applications erase themselves (:data:`ERASE_APPLICATIONS_WHEN_DONE`) and this line is what is
left: the question above it, the answer below it, nothing else."""

PERMISSION_ANSWER_LINES_BY_CLASS: dict[str, dict[str, str]] = {
    "workspace-trust": {
        "allow_once": "✓ trusted — remembered for this workspace",
        "allow_always": "✓ trusted — remembered for this workspace",
        "reject_once": "✗ not trusted — guarded for this session",
        "reject_always": "✗ not trusted — guarded for this session",
    },
}
"""Per-ask-class overrides for :data:`PERMISSION_ANSWER_LINES`.

The workspace-trust question is the one ask whose answer is not about a tool call, and "✓ allowed
once" was a lie in both directions: a yes is remembered forever, and a no does not deny anything
— it puts the session in ``guarded``. The four kinds collapse to two here because the question
only ever offers two (``cli/session_trust`` builds it ungrantable), and both spellings of each
are listed so a channel that ever offers all four cannot fall through to the wrong sentence."""

ERASE_APPLICATIONS_WHEN_DONE = True
"""Whether a prompt_toolkit application clears its own drawing when it exits.

prompt_toolkit's default is False, which leaves the last frame painted into the scrollback
forever — right for a `prompt()` whose line IS the transcript, wrong for both applications here:
the permission legend is a menu, not a record, and the persistent input box is a live surface
that gets suspended and restarted several times in one turn. Erasing is what makes an answered
prompt collapse to its one confirmation line."""

CANNOT_ASK_NOTICE = (
    "permission prompts are disabled on this channel: {why}; asks will be denied — set "
    "`permissions.mode: unattended` in config for unattended runs"
)
"""PRD §3.5's last row asks for "a loud startup warning naming the fix". The gate logs one for
the model's side; this is the human's, printed once at session start, because a person who ran
`localharness` in a terminal will otherwise meet a denial mid-turn with no idea why."""

CANNOT_ASK_NO_STDIN = "standard input is not a terminal, so nobody can answer a question"
CANNOT_ASK_NO_TTY = "neither standard output nor standard error is a terminal, so a question would be invisible"
"""The two ways a terminal session loses the ability to ask; the notice names which one it is
rather than saying "non-interactive", which is what sent the verifier looking for a bug."""


def _tty(stream: Any) -> bool:
    """Is this stream a real terminal? False for a pipe, a file, a closed or absent stream."""
    try:
        return stream is not None and bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def cannot_ask_reason() -> str | None:
    """Why this process could not put a permission question to a human — None when it can.

    PRD §3.5's non-tty row, corrected by verification A defect D6: an ASK is answered on the
    INPUT side, so the test is `stdin`. `localharness start | tee session.log` keeps a real
    keyboard and a real screen; judging it by `stdout` alone fail-closed every prompt in a
    session where the person was sitting right there. The output side still has to reach a
    terminal somewhere — `ask_permission` draws the question on stderr when stdout is
    redirected — or the question would be invisible and the answer would never come.
    """
    import sys

    if not _tty(sys.stdin):
        return CANNOT_ASK_NO_STDIN
    if not (_tty(sys.stdout) or _tty(sys.stderr)):
        return CANNOT_ASK_NO_TTY
    return None


def _build_permission_app(options: str, grantable: bool, output: Any = None) -> Application:
    """A one-keystroke choice for `TerminalChannel.ask_permission` (PRD §3.5).

    Deliberately NOT a `Buffer`/line-editor like the input bubble: there is nothing to type, and
    a line editor would let a stray Enter on leftover text answer the question. One key, one
    answer, no history, no completion.
    """
    keys = PERMISSION_KEYS_GRANTABLE if grantable else PERMISSION_KEYS_UNGRANTABLE
    kb = KeyBindings()

    def _bind(key: str, kind: str) -> None:
        @kb.add(key, eager=True)
        def _choose(event) -> None:
            event.app.exit(result=kind)

    for key, kind in keys.items():
        _bind(key, kind)

    @kb.add("enter")
    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _default(event) -> None:
        event.app.exit(result=PERMISSION_DEFAULT_DECISION)

    body = Window(
        FormattedTextControl([("class:hint", f" {options} ")]),
        height=1,
        style="class:frame",
    )
    return Application(
        layout=Layout(body),
        key_bindings=kb,
        style=INPUT_STYLE,
        mouse_support=False,
        erase_when_done=ERASE_APPLICATIONS_WHEN_DONE,
        **({"output": output} if output is not None else {}),
    )


def _build_input_app(
    history: FileHistory, prompt: str, hint: str, context_pct: float | None = None,
    model_names_fn: Callable[[], list[str]] | None = None,
) -> Application:
    """Inline application: ╭─╮ │ > input │ ╰─ hint ──── meter ─╯. Exits with the entered line."""
    buf = Buffer(history=history, auto_suggest=AutoSuggestFromHistory(), multiline=False,
                 completer=SlashCommandCompleter(model_names_fn), complete_while_typing=True)
    control = BufferControl(
        buffer=buf,
        input_processors=[
            BeforeInput([("class:caret", f" {prompt} ")]),
            AppendAutoSuggestion(),
        ],
    )

    kb = KeyBindings()
    _add_menu_keys(kb, buf)

    @kb.add("enter")
    def _accept(event) -> None:
        comp = _accept_completion(buf)
        if comp is not None and not _is_model_pick(comp):
            return  # command accepted into the line; don't submit yet
        # no completion applied, or a model was picked -> submit in this stroke
        buf.append_to_history()
        event.app.exit(result=buf.text)

    @kb.add("c-c")
    def _interrupt(event) -> None:
        if buf.text:
            buf.reset()  # first Ctrl+C clears the line, like Claude Code / most shells
        else:
            event.app.exit(exception=KeyboardInterrupt(), style="class:aborting")

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            event.app.exit(exception=EOFError(), style="class:exiting")

    def _wall(char: str) -> Window:
        return Window(width=1, char=char)

    bottom = [_wall("╰"), Window(char="─", height=1, width=1)]
    if hint:
        hint_text = f" {hint} "
        bottom.append(
            Window(FormattedTextControl([("class:hint", hint_text)]), width=len(hint_text), height=1)
        )
    bottom.append(Window(char="─", height=1))  # stretchy filler right-aligns the meter
    if context_pct is not None:
        frags, w = _ctx_segments(context_pct)
        bottom += [
            Window(FormattedTextControl(frags), width=w, height=1),
            Window(char="─", height=1, width=1),
        ]
    bottom.append(_wall("╯"))
    body = HSplit([
        VSplit([_wall("╭"), Window(char="─", height=1), _wall("╮")]),
        VSplit([_wall("│"), Window(control, wrap_lines=True, dont_extend_height=True, style="class:input"), _wall("│")]),
        VSplit(bottom),
    ], style="class:frame")

    return Application(
        layout=Layout(_menu_float(body), focused_element=control),
        key_bindings=kb,
        style=INPUT_STYLE,
        mouse_support=False,
    )


def _build_persistent_input_app(
    history: FileHistory,
    prompt: str,
    *,
    on_submit: Callable[[str], None],
    on_interrupt: Callable[[], None],
    on_eof: Callable[[], None],
    hint_fn: Callable[[], list[tuple[str, str]]],
    right_fn: Callable[[], list[tuple[str, str]]],
    status_fn: Callable[[], list[tuple[str, str]]],
    placeholder_fn: Callable[[], str] = lambda: "",
    model_names_fn: Callable[[], list[str]] | None = None,
    pending_fn: Callable[[], list[tuple[str, str]]] = lambda: [],
    on_pending_answer: Callable[[bool], None] = lambda _approve: None,
    on_keystroke: Callable[[], None] = lambda: None,
) -> Application:
    """Long-lived input box that stays usable while turn output streams above it.

    Differs from _build_input_app in four ways, all load-bearing for the type-anytime box:
      1. Enter SUBMITS without exiting — it hands the line to on_submit and resets the buffer,
         so the same Application services every submission for the whole session (run once via
         asyncio.create_task(app.run_async()) alongside the turn, under patch_stdout(raw=True)).
      2. A FOOTER row under the closed box carries DYNAMIC FormattedTextControl callables
         refreshed by app.invalidate(): left, hint_fn (the key legend for the box's current
         mode / `queued (N)` / the routing-decision flash); right, right_fn (the local-model
         instrument cluster: model · measured tok/s · context meter). placeholder_fn is the
         dim guidance shown inside the box while it is empty.
      3. A one-line working/activity STATUS ROW sits ABOVE the frame (status_fn), so it reads
         as the last line of the log area, not a glyph in the box border (FIX 2). A
         ConditionalContainer collapses it to zero height when status_fn returns [] (idle).
      4. Ctrl+C (empty buffer) / Ctrl+D (empty buffer) call back into REPL policy (on_interrupt
         / on_eof) rather than raising out of run_async — the box owns raw mode for the whole
         session, so these are the only path a signal-suppressed terminal has to interrupt/exit.
      5. A PENDING ROW sits under the footer (pending_fn), collapsing to zero height the same
         way the status row does, and Ctrl+Y / Ctrl+N answer the call it names through
         on_pending_answer. Every key press also stamps on_keystroke, which is how the channel
         knows whether anybody is still at the keyboard.
    """
    buf = Buffer(history=history, auto_suggest=AutoSuggestFromHistory(), multiline=False,
                 completer=SlashCommandCompleter(model_names_fn), complete_while_typing=True)
    control = BufferControl(
        buffer=buf,
        input_processors=[
            BeforeInput([("class:caret", f" {prompt} ")]),
            ConditionalProcessor(  # PromptSession's placeholder, re-created for the hand-built box
                AfterInput(lambda: [("class:placeholder", placeholder_fn())]),
                filter=Condition(lambda: not buf.text),
            ),
            AppendAutoSuggestion(),
        ],
    )

    kb = KeyBindings()
    _add_menu_keys(kb, buf, injected_prefix=_MODEL_PICK_PREFIX)

    @kb.add("enter")
    @kb.add("c-j")  # raw LF: prompt_toolkit's own default treats \n as Enter, because some
    #                 terminals send it for Return (WSL) — binding it to anything else silently
    #                 turns every Enter on those terminals into that something else.
    def _submit(event) -> None:
        comp = _accept_completion(buf)
        if comp is not None and not _is_model_pick(comp):
            return  # command accepted into the line; don't submit yet
        # no completion applied, or a model was picked -> submit in this stroke
        text = buf.text
        if text.strip():
            buf.append_to_history()
            on_submit(text)
        buf.reset()  # ready for the next line; the app stays alive

    @kb.add("escape", "enter")  # Alt+Enter (ESC-prefix Meta — works on every terminal)
    def _submit_nudge(event) -> None:
        # Chord-level twin of the `!` force prefix: submit the line as a FORCED NUDGE, steering
        # the RUNNING turn past the nudge/queue classifier; when no turn runs, strip_force makes
        # it an ordinary submit. Shift+Enter cannot carry this: prompt_toolkit remaps the xterm
        # Shift+Enter sequence to plain Enter (ansi_escape_sequences: "currently unsupported"),
        # and Ctrl+J cannot either (it IS Enter on LF terminals, see above), so ESC-prefixed
        # Alt+Enter is the one reliable cross-terminal spelling. No completion-accept here —
        # the chord means "send exactly this line, now".
        text = buf.text
        if text.strip():
            buf.append_to_history()  # history keeps what was typed, without the synthetic prefix
            on_submit(FORCE_PREFIX + text)
        buf.reset()

    @kb.add("c-c")
    def _interrupt(event) -> None:
        if buf.text:
            buf.reset()  # first Ctrl+C clears the line (Claude Code / shell idiom)
        else:
            on_interrupt()  # empty buffer → REPL cancels the turn, or arms/exits when idle

    @kb.add("c-d")
    def _eof(event) -> None:
        if not buf.text:
            on_eof()

    @kb.add("c-y")
    def _approve_pending(event) -> None:
        # Ctrl+Y / Ctrl+N answer the call named on the pending row — NOW. A slash command typed
        # while a turn runs is queued and replayed after it (`_route_during_turn`), which is the
        # exact wait staging exists to remove: a person sitting there watching the turn should be
        # able to unblock the step it skipped without waiting for the turn to end. Both keys are
        # free here — this app loads only the bindings above (no emacs/vi defaults, where they
        # would be yank and next-history) — and both are dead when nothing is parked, which the
        # channel decides, because the box does not own the queue.
        on_pending_answer(True)

    @kb.add("c-n")
    def _deny_pending(event) -> None:
        on_pending_answer(False)

    def _wall(char: str) -> Window:
        return Window(width=1, char=char)

    def _right_width() -> int:
        return fragment_list_width(right_fn())

    frame = HSplit([
        VSplit([_wall("╭"), Window(char="─", height=1), _wall("╮")]),
        VSplit([_wall("│"), Window(control, wrap_lines=True, dont_extend_height=True, style="class:input"), _wall("│")]),
        VSplit([_wall("╰"), Window(char="─", height=1), _wall("╯")]),
    ], style="class:frame")
    # Footer UNDER the closed box — where Claude Code / Codex / Gemini / Cline put the key hints
    # and the context readout. The border used to carry them, and a width-less hint Window
    # stretched to half the row and padded it with spaces: a hole in the frame. The right
    # cluster is width-fitted (fragment_list_width), so it right-aligns with no padding.
    footer = VSplit([
        Window(FormattedTextControl(hint_fn), height=1),
        Window(FormattedTextControl(right_fn), width=_right_width, height=1),
    ])
    # FIX 2: working/activity status row, ABOVE the frame, so it reads as the last line of the
    # log area (where the tool/agent lines land) rather than a glyph crammed into the box
    # border. The ConditionalContainer collapses it to zero height when status_fn returns []
    # (idle) — no blank line wasted above the box. Still a FormattedTextControl refreshed by
    # app.invalidate() (the _box_tick ticker); NO rich Status/Live near the live box, which is
    # what could freeze the screen on Ctrl+C-during-burst (v0.9.10).
    status_row = ConditionalContainer(
        Window(FormattedTextControl(status_fn), height=1, dont_extend_height=True),
        filter=Condition(lambda: bool(status_fn())),
    )
    # The parked-call row, UNDER the footer (BOX_PENDING_ROW): the last line of the input area,
    # furthest from the work, because a staged call is a note to answer whenever — not a prompt.
    # Same zero-height collapse as the status row, so an empty queue costs no line at all.
    pending_row = ConditionalContainer(
        # No fixed height: one line when it fits, two when the legend has to drop down.
        Window(FormattedTextControl(pending_fn), dont_extend_height=True),
        filter=Condition(lambda: bool(pending_fn())),
    )
    body = HSplit([status_row, frame, footer, pending_row])

    app = Application(
        layout=Layout(_menu_float(body), focused_element=control),
        key_bindings=kb,
        style=INPUT_STYLE,
        mouse_support=False,
        erase_when_done=ERASE_APPLICATIONS_WHEN_DONE,
    )
    app._lh_input_buffer = buf  # box_open_model_menu pre-fills + pops the picker through this
    # Presence, stamped on EVERY key press. Deliberately the key_processor's own hook and not a
    # `@kb.add("<any>")` catch-all: prompt_toolkit sorts the bindings matching a key with the
    # Any-matches LAST and calls `matches[-1]` (KeyProcessor._process), so a catch-all here would
    # take Enter, Ctrl+C and every printable character away from the bindings above rather than
    # passing them through. before_key_press observes; it decides nothing.
    app.key_processor.before_key_press += lambda _sender: on_keystroke()
    return app


# Per-turn session colors, sourced from the ONE entity palette in cli/theme.py (the
# localharness.dev architecture-plate hues) rather than picked per key here. Before this,
# every entity in a turn arrived in the same handful of ANSI colors, so a tool call, an
# agent and a system notice were told apart by their glyph alone.
#
# Two rules hold the map together:
#   - NAMES take the entity color; BODIES stay neutral. agent.text and muted are body
#     text and deliberately keep no hue.
#   - VERDICTS keep green/red and are never entity-colored: success, tool.error and
#     system.error are the vocabulary for "it worked / it did not", not a type.
#
# The user is the one speaker who is NOT on the architecture plates, so user.input takes
# the site's primary INK rather than an entity hue — the agent now owns the accent green
# the plates give the runtime, and the two must stay tellable apart.
TERMINAL_THEME = Theme({
    "agent.name":   f"bold {ENTITY_STYLES['agent']}",
    "agent.text":   "white",
    "tool.call":    ENTITY_STYLES["tool"],
    "memory.call":  ENTITY_STYLES["memory"],   # memory-family burst lines (memory_search/get/remember)
    "tool.result":  f"dim {ENTITY_STYLES['tool']}",
    "tool.error":   "bold red",
    "system.info":  f"dim {ENTITY_STYLES['infra']}",
    "system.error": "bold red",
    "user.input":   f"bold {SITE_INK}",
    "highlight":    "bold yellow",
    "success":      "bold green",
    "warning":      f"bold {ENTITY_STYLES['warning']}",
    "muted":        "dim",
})


def _format_args_compact(arguments: dict[str, Any], max_value_len: int = 60) -> str:
    """Format tool call arguments for inline display.

    Rules:
    - String values: show quoted, truncate to max_value_len with "..."
    - Number values: show as-is
    - List values: show as [N items] if N > 3, else show list
    - Dict values: show as {N keys}
    - Bool values: show as true/false (lowercase)
    - None values: show as null
    """
    parts = []
    for k, v in arguments.items():
        if isinstance(v, bool):
            parts.append(f"{k}={'true' if v else 'false'}")
        elif isinstance(v, str):
            truncated = v[:max_value_len] + "..." if len(v) > max_value_len else v
            parts.append(f'{k}="{truncated}"')
        elif isinstance(v, (int, float)):
            parts.append(f"{k}={v}")
        elif isinstance(v, list):
            if len(v) <= 3:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, dict):
            parts.append(f"{k}={{{len(v)} keys}}")
        elif v is None:
            parts.append(f"{k}=null")
        else:
            parts.append(f"{k}=...")
    return ", ".join(parts)


def _get_key_arg(arguments: dict[str, Any]) -> str:
    """Return the value of the first string argument, or ''."""
    for v in arguments.values():
        if isinstance(v, str):
            return v
    return ""


# write/edit carry a whole file body in their args — the view shows a one-line summary
# (path + line count), never the body. Every other tool's arg preview is capped to one
# line so a big argument can't flood the chat view either.
_FILE_WRITE_TOOLS = frozenset({"write", "edit"})
_MAX_ARG_PREVIEW = 120


def _tool_call_summary(tool_name: str, arguments: dict[str, Any]) -> str:
    """One-line, body-free summary of a tool call (no markup; caller adds ◆ + style).

    write/edit collapse to `<tool> <path> (<n> lines)` so the view never fills with file
    contents; `<n>` is the body the model actually emitted this call. Every other tool
    shows its key argument on a single line, capped to _MAX_ARG_PREVIEW characters."""
    if tool_name in _FILE_WRITE_TOOLS:
        path = str(arguments.get("path", "")).strip()
        body = arguments.get("content") or arguments.get("new_string") or ""
        n = len(body.splitlines()) if isinstance(body, str) else 0
        summary = f"{tool_name} {path}".rstrip()
        return f"{summary} ({n} lines)" if n else summary
    preview = _get_key_arg(arguments).split("\n", 1)[0]
    if len(preview) > _MAX_ARG_PREVIEW:
        preview = preview[: _MAX_ARG_PREVIEW - 1].rstrip() + "…"
    return f"{tool_name} {preview}".rstrip()


def _narration_line(content: str | None, cap: int = _MAX_NARRATION) -> str:
    """First non-empty line of `content`, hard-capped to `cap` chars with an ellipsis.

    Returns '' when there is nothing to narrate — empty/whitespace content, which
    reasoning-parser tool turns legitimately produce (all tokens went to reasoning +
    tool_calls). The caller skips rendering on ''."""
    if not content:
        return ""
    for raw in content.splitlines():
        line = raw.strip()
        if line:
            return line[: cap - 1].rstrip() + "…" if len(line) > cap else line
    return ""


class TerminalChannel(ChannelAdapter):
    """Rich-formatted terminal channel with streaming output.

    State machine:
      IDLE → STREAMING → IDLE
                       ↓
                  WAITING_INPUT → IDLE

    The _output_lock prevents concurrent stdout writes from corrupting terminal output.
    """

    channel_id = "terminal"

    has_review_surface = True
    """PRD §3.1 choice 2: the terminal shows the diff after the fact, so an in-workspace edit is
    reviewable and never asks."""

    ask_holds_dialog = True
    """PRD §3.5, terminal row: "Timeout: none". The question sits in the terminal until the
    person answers it, so the gate must not put a deadline on it — verification A defect D3,
    where an unanswered prompt auto-denied after the tool's own timeout while three docs
    promised it would not. A walk-away must cost a wait, never a silent denial."""

    @property
    def can_ask(self) -> bool:
        """Can a person answer a question here? (PRD §3.5's "non-tty" row.)

        The test is the INPUT side — verification A defect D6. `can_run_input_box()` asks
        whether the persistent box can own STDOUT, which is a different question: piping the
        transcript (`localharness start | tee session.log`) drops the box but keeps the
        keyboard, and judging the ask by stdout fail-closed every prompt in a session where the
        person was sitting right in front of it. :func:`cannot_ask_reason` is the honest test,
        and `ask_permission` draws the question on stderr when stdout is redirected.
        """
        return cannot_ask_reason() is None

    def __init__(
        self,
        bus: EventBus,
        config: dict[str, Any],
        history_file: str = ".repl_history",
    ) -> None:
        # A bare relative default; start_cmd resolves it UNDER the active config dir (#35, so
        # --config-dir isolates REPL history). Absolute/~ values are honored as-is at use.
        super().__init__(bus, config)
        self._console = Console(theme=TERMINAL_THEME, highlight=False)
        self._err_console = Console(stderr=True, theme=TERMINAL_THEME, highlight=False)
        self._history: FileHistory | None = None
        self._history_file = history_file
        self._live: Live | None = None
        self._state: str = "IDLE"
        self._sigint_armed: bool = False  # True after one Ctrl+C on an empty line; second exits
        # #49: hint drawn INSIDE the first input bubble (immune to the box repaint that drops
        # a banner hint in a real TTY). start_cmd sets it for interactive sessions; shown once.
        self.first_prompt_hint: str = ""
        self._output_lock: asyncio.Lock = asyncio.Lock()
        self._thinking: Status | None = None  # rich Status while the model is generating (REPL-02)
        self._dreaming: Status | None = None  # rich Status while a background memory pass runs (#20)
        self._burst: _Burst | None = None  # open family-burst consolidation (display-only)
        # Narration cadence guard: a narration line may open a chunk of tool activity only
        # after a tool result has rendered since the last one — never two dim lines in a row.
        # Starts True so the turn's first narration prints.
        self._tool_result_since_narration: bool = True
        self._last_agent_delegate: str | None = None  # #73: delegate name stashed on an `agent`
        # call, consumed by its completion receipt (the success result carries only the summary)
        self._context_pct: float | None = None  # latest Heartbeat utilization, shown in the input bubble
        # /model picker: the REPL sets this to its cached live-model-list supplier once it
        # owns a provider; None (or an empty list) = no argument menu. Read late-bound by
        # _model_names_for_menu so it works regardless of when the input apps were built.
        self.model_names_fn: Callable[[], list[str]] | None = None
        # --- Persistent type-anytime input box (start_input_box); all inert until then ---
        self._box_active: bool = False           # True while the long-lived box owns the terminal
        # (ctrl_queue, on_interrupt) from start_input_box — ask_permission suspends the box for
        # one keystroke and needs these to put it back exactly as it was.
        self._box_restart_args: tuple[Any, Any] | None = None
        self._box_app: Application | None = None
        self._box_task: asyncio.Task | None = None
        self._box_patch = None                   # patch_stdout(raw=True) ctx, held for the box's life
        self._box_ticker: asyncio.Task | None = None
        self._box_working: bool = False          # working state → status row above the box (thinking/streaming/burst)
        self._box_dreaming: bool = False         # background consolidation ("dreaming") pass → status row (#20)
        self._box_activity: str = ""             # transient status-row note (e.g. /model swap loading line)
        self._queued_count: int = 0              # `queued (N)` shown in the box frame
        # The calls `auto` parked for a human, oldest first — pushed in by the REPL
        # (box_set_pending), read by the row under the footer and by both hotkeys.
        self._pending: list[Any] = []
        # When a key last reached the box (monotonic). 0.0 = nobody has typed yet, which reads as
        # AWAY until start_input_box stamps it: opening the box is somebody being there.
        self._last_keystroke_at: float = 0.0
        self._decision_flash: str = ""           # transient routing-decision line in the box frame
        self._decision_flash_task: asyncio.Task | None = None
        self._first_box_hint: str = ""           # #49 guidance hint, shown in the box until first use
        self.last_activity_summary: str = ""     # latest tool line — tier-2 routing context (box mode)
        # Decode-speed supplier (start_cmd wires LLMClient.gen_speed_snapshot): () -> (tok/s,
        # verified) | None. Drives the colored tok/s readout in the status row / thinking label.
        self.tps_source: Callable[[], tuple[float, bool] | None] | None = None
        # Model-name supplier (start_cmd wires `lambda: llm.config.model`, which every /model
        # swap path reassigns): the footer's model chip.
        self.model_source: Callable[[], str | None] | None = None
        self._notes_shown: set[str] = set()      # burst close notes already printed this user turn
        # Live-request supplier (start_cmd wires LLMClient.stream_snapshot): () -> {phase,
        # thinking_tokens, answer_tokens, tool_call_tokens, elapsed, silent} | None. Drives the
        # phase + tallies in the status row's working text; None → the plain "working".
        self.progress_source: Callable[[], dict | None] | None = None
        # Reasoning stream (terminal.show_reasoning / --show-reasoning / /reasoning). start_cmd
        # points LLMClient.on_reasoning at on_reasoning below; the flag decides whether the
        # deltas print. Line-buffered: a complete line (or _REASON_FLUSH_CHARS of one) prints
        # as a dim ⋯ line; the tail flushes when the response's Action arrives.
        self.show_reasoning: bool = False
        # Verbose view (--verbose / /verbose): the clean default groups read/memory/web calls
        # into family counters and hides reasoning; verbose itemizes EVERY tool call with its
        # arguments and its own result line, and implies the reasoning stream. The per-call
        # truth is always on the bus ledger — this only decides what the screen shows.
        self.verbose: bool = False
        self._reasoning_buf: str = ""
        self._reasoning_open: bool = False       # a line printed since the last flush
        self._action_handle = None
        self._observation_handle = None
        self._task_complete_handle = None
        self._escalation_handle = None
        self._parse_failed_handle = None
        self._heartbeat_handle = None
        self._turn_failed_handle = None
        self._consolidation_started_handle = None
        self._consolidation_finished_handle = None
        self._permission_staged_handle = None

    async def start(self) -> None:
        """Initialize input history, say so if we cannot ask, and subscribe to bus events."""
        why = cannot_ask_reason()
        if why is not None:
            self._err_console.print(
                f"[warning]{escape(CANNOT_ASK_NOTICE.format(why=why))}[/warning]"
            )
        history_path = os.path.expanduser(self._history_file)
        os.makedirs(os.path.dirname(history_path), exist_ok=True)

        self._history = FileHistory(history_path)

        self._action_handle = self.bus.subscribe(Action, self.on_action)
        self._observation_handle = self.bus.subscribe(Observation, self.on_observation)
        self._task_complete_handle = self.bus.subscribe(TaskComplete, self.on_task_complete)
        self._turn_failed_handle = self.bus.subscribe(TurnFailed, self.on_turn_failed)
        self._escalation_handle = self.bus.subscribe(Escalation, self.on_escalation)
        self._parse_failed_handle = self.bus.subscribe(ParseFailed, self.on_parse_failed)
        self._compaction_handle = self.bus.subscribe(
            CompactionTriggered, self.on_compaction_triggered
        )
        self._heartbeat_handle = self.bus.subscribe(Heartbeat, self.on_heartbeat)
        # The only way a channel hears that `auto` parked a call (core/events.PermissionStaged):
        # the loop holds no channel handle, and a queue notice is not about one tool result.
        self._permission_staged_handle = self.bus.subscribe(
            PermissionStaged, self.on_permission_staged
        )
        self._consolidation_started_handle = self.bus.subscribe(
            ConsolidationStarted, self.on_consolidation_started
        )
        self._consolidation_finished_handle = self.bus.subscribe(
            ConsolidationFinished, self.on_consolidation_finished
        )

    async def stop(self) -> None:
        """Unsubscribe from bus, stop any active Live context, flush console."""
        await self.stop_input_box()  # tear the persistent box down first (idempotent)
        for handle in (
            self._action_handle,
            self._observation_handle,
            self._task_complete_handle,
            self._turn_failed_handle,
            self._escalation_handle,
            self._parse_failed_handle,
            self._heartbeat_handle,
            self._consolidation_started_handle,
            self._consolidation_finished_handle,
            self._permission_staged_handle,
        ):
            if handle is not None:
                self.bus.unsubscribe(handle)

        self._stop_thinking()
        self._close_burst()  # teardown: flush an open counter so the count isn't lost

        if self._live is not None:
            self._live.stop()
            self._live = None

        self._console.file.flush()

    async def send_message(
        self,
        content: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Print a message to the terminal. Wraps in Panel if agent_id provided."""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
            self._ensure_idle()
            self._tool_result_since_narration = True  # a full message closes the chunk cadence
            if agent_id:
                self._console.print(Panel(
                    Markdown(content),
                    title=f"[agent.name]{escape(agent_id)}[/agent.name]",
                    border_style="cyan",
                ))
            elif metadata and metadata.get("colorize") == "rates":
                # Trusted-shape opt-in from the REPL's /model surfaces: style the rate notes
                # by band via Text spans (never markup — content stays injection-proof).
                self._console.print(_colorize_rate_notes(content))
            else:
                self._console.print(escape(content))

    async def send_renderable(
        self,
        renderable: Any,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Print a pre-built rich renderable (e.g. the /memory tree) directly, bypassing the
        markup-escaping text path of send_message. Same output-lock + idle preamble, so it composes
        with streaming/burst and renders correctly above the persistent input box (raw patch_stdout
        interprets the ANSI). No rich Live/Status here — a static print, safe near the box."""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
            self._ensure_idle()
            self._tool_result_since_narration = True
            self._console.print(renderable)

    async def send_streaming(
        self,
        token_stream: AsyncIterator[str],
        agent_id: str | None = None,
    ) -> str:
        """Stream tokens to terminal using Rich Live. Returns full assembled text."""
        full_text = ""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()  # rich allows one live display — freeze before Live starts
            self._state = "STREAMING"
            panel_title = f"[agent.name]{escape(agent_id or 'agent')}[/agent.name] [muted]streaming...[/muted]"
            live_panel = Panel("", title=panel_title, border_style="cyan")

            with Live(live_panel, console=self._console, refresh_per_second=20) as live:
                self._live = live
                async for token in token_stream:
                    full_text += token
                    live_panel = Panel(
                        Text(full_text, style="agent.text"),
                        title=panel_title,
                        border_style="cyan",
                    )
                    live.update(live_panel)
                self._live = None

            # Final non-live panel with green border — render markdown (tables,
            # headers, bold) instead of raw text so the answer reads cleanly.
            self._console.print(Panel(
                Markdown(full_text),
                title=f"[agent.name]{escape(agent_id or 'agent')}[/agent.name]",
                border_style="green",
            ))
            self._state = "IDLE"
        return full_text

    async def send_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        agent_id: str | None = None,
    ) -> None:
        """Display tool invocation inline. Format: ◆ tool_name key_arg

        Family tools (_BURST_GROUPS) don't print per-call: they open or extend a
        consolidated burst counter that freezes to one line when the burst ends."""
        async with self._output_lock:
            self._stop_thinking()
            self.last_activity_summary = _tool_call_summary(tool_name, arguments)  # tier-2 context
            group = None if self.verbose else next((g for g in _BURST_GROUPS if tool_name in g[0]), None)
            if group is None:
                self._close_burst()
                if tool_name == "agent":
                    # Stash the delegate for the completion receipt (send_tool_result): a
                    # SUCCESS Observation carries only the summary, not who ran it (#73).
                    d = arguments.get("agent_id")
                    self._last_agent_delegate = d if isinstance(d, str) else None
                summary = _tool_call_summary(tool_name, arguments)
                # no_wrap + ellipsis: a long preview stays ONE physical line (cropped to
                # the terminal), never wrapping the view open (FIX B).
                self._console.print(
                    f"  [tool.call]{_DIAMOND} {escape(summary)}[/tool.call]",
                    no_wrap=True, overflow="ellipsis",
                )
                return
            family, note, style = group
            if self._burst is None or self._burst.family is not family:
                self._close_burst()
                self._burst = _Burst(family=family, close_note=note, style=style)
            burst = self._burst
            if tool_name not in burst.tools:
                burst.tools.append(tool_name)
            burst.calls += 1
            self._burst_refresh(burst)

    async def send_tool_result(
        self,
        tool_name: str,
        result: str,
        is_error: bool,
        agent_id: str | None = None,
    ) -> None:
        """Display tool result inline. Uses checkmark for success, cross for error.

        Results belonging to the open burst tick its counter instead of printing;
        errors are absorbed into the count and annotated on the frozen line (full
        detail stays on the bus ledger)."""
        lines = result.strip().split("\n") if result else [""]
        async with self._output_lock:
            self._stop_thinking()
            self._tool_result_since_narration = True  # a rendered result re-opens narration
            burst = self._burst
            if burst is not None and tool_name in burst.family:
                burst.done += 1
                if is_error:
                    burst.errors += 1
                self._burst_refresh(burst)
                return
            self._close_burst()
            # #73: delegation-outcome receipt. Every `agent` completion prints a truthful,
            # system-style line whose status is taken from the TOOL RESULT (is_error), not the
            # model's later narration, and whose name is the delegate stashed from this call —
            # so a failed delegation can't be re-narrated as the subagent's own answer.
            if tool_name == "agent":
                delegate, self._last_agent_delegate = self._last_agent_delegate, None
                who = f" {escape(delegate)}" if delegate else ""
                if is_error:
                    detail = escape(lines[0].removeprefix("[tool error] ")[:120]) if lines else ""
                    self._console.print(f"  [tool.error]{_DIAMOND} agent{who} — FAILED: {detail}[/tool.error]")
                else:
                    self._console.print(f"  [tool.call]{_DIAMOND} agent{who} — completed[/tool.call]")
                return
            name = escape(tool_name)
            if is_error:
                first = escape(lines[0][:120]) if lines else ""
                self._console.print(f"  [tool.error]{_CROSS} {name} (exit 1): {first}[/tool.error]")
            elif tool_name in _FILE_WRITE_TOOLS:
                # result is a status line ("Written N bytes"), not the lines written —
                # omit the count rather than print a misleading one (FIX C).
                self._console.print(f"  [tool.result]{_CHECK} {name}[/tool.result]")
            else:
                self._console.print(f"  [tool.result]{_CHECK} {name} ({len(lines)} lines)[/tool.result]")
            if tool_name in UNTRUSTED_INGEST and not is_error:
                self._print_close_note(_UNTRUSTED_NOTE)  # itemized (verbose) web calls keep the disclosure

    async def send_permission_denied(
        self,
        tool_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        """One inline line saying why the gate refused this call (PRD §3.5, defect D7).

        Inline on the ordinary console rather than through `send_error`'s stderr path: it
        belongs directly under the `✗ <tool> (exit 1): [DENIED]` line it explains, and a
        denial is not a crash — the model is about to re-plan around it.
        """
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
            self._console.print(
                f"  [tool.error]{PERMISSION_DENIED_LINE.format(tool_name=escape(tool_name), reason=escape(reason))}[/tool.error]"
            )

    async def _print_notice_line(self, text: str) -> None:
        """One indented, muted line in the transcript — `send_permission_denied`'s shape, without
        its error color. Shared by the parked-call notice and the away-return line."""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
            self._console.print(f"  [system.info]{escape(text)}[/system.info]")

    async def send_pending_notice(self, pending: Any, total: int) -> None:
        """One inline line saying a call was parked for the human (:data:`PENDING_NOTICE_LINE`).

        The bar under the box already shows the queue, and the bar is the thing that CLEARS —
        answered, or gone with the next staged call taking its place. This is the transcript's
        permanent copy: scroll back a day later and the turn still says which step it skipped
        and why it could not take it.
        """
        await self._print_notice_line(
            PENDING_NOTICE_LINE_TERMINAL.format(id=pending.id, rendering=pending.rendering)
        )

    async def on_permission_staged(self, event: Any) -> None:
        """PermissionStaged → the transcript line, plus ONE bell if somebody is at the keyboard.

        The bell is what presence tracking is for. A staged call is the one thing in a turn the
        harness cannot finish by itself, and it lands in the middle of streaming output that a
        person watching the screen is not reading word by word. Nobody there, no bell: a terminal
        beeping into an empty room is noise for whoever walks past it, and the away-return line
        (:data:`AWAY_RETURN_LINE`) is what greets the person who actually comes back.
        """
        await self.send_pending_notice(event.pending, event.total)
        if self._present():
            self._ring_bell()

    def _present(self) -> bool:
        """Is somebody at this keyboard right now — a key within :data:`PRESENCE_WINDOW_S`?"""
        return (time.monotonic() - self._last_keystroke_at) < PRESENCE_WINDOW_S

    def _ring_bell(self) -> None:
        """Ring the terminal bell once, through the box's own output (never a raw print).

        Exception-proof like every other box side effect: a terminal that cannot beep must not be
        the reason a staged call takes the session down.
        """
        app = self._box_app
        if app is None or not self._box_active:
            return
        try:
            app.output.bell()
        except Exception:
            pass

    def _note_keystroke(self) -> None:
        """Stamp the presence clock, and welcome somebody back to a queue they never saw.

        Called from the box app's `before_key_press` hook (see `_build_persistent_input_app`), so
        it sees every key — not only the ones that change the buffer. The stamp is taken BEFORE
        the gap is judged, which is what makes the return line print once per return rather than
        once per key: the second key is a hundred milliseconds after the first, not five minutes.
        """
        now = time.monotonic()
        away_for = now - self._last_keystroke_at
        self._last_keystroke_at = now
        if away_for < PRESENCE_WINDOW_S or not self._pending:
            return
        count = len(self._pending)
        line = AWAY_RETURN_LINE.format(count=count, noun=BOX_PENDING_NOUNS[count != 1])
        try:
            asyncio.get_running_loop().create_task(self._print_notice_line(line))
        except RuntimeError:
            pass  # no loop (a unit context): the stamp still moved, which is the part that counts

    async def send_error(
        self,
        error: str,
        detail: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        """Write error to stderr console."""
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()
        self._err_console.print(f"[system.error]Error:[/system.error] {escape(error)}")
        if detail:
            for line in detail.split("\n"):
                self._err_console.print(f"  {escape(line)}")

    def _model_names_for_menu(self) -> list[str]:
        """Late-bound, exception-proof view of model_names_fn for the completion menu — a menu
        callback must never raise into the key processor, and must never block."""
        fn = self.model_names_fn
        try:
            return fn() if fn is not None else []
        except Exception:
            return []

    async def read_input(self, prompt: str = ">") -> str:
        """Read a line from the user inside a rounded input bubble. Raises ChannelStartError if not started."""
        if self._history is None:
            raise ChannelStartError("TerminalChannel.start() must be called before read_input()")
        async with self._output_lock:
            self._stop_thinking()
            self._close_burst()  # freeze any open counter before the input bubble draws
        self._state = "WAITING_INPUT"
        self._notes_shown.clear()  # a new user turn gets the close notes again
        # #49: the first prompt carries the guidance hint in the bubble's bottom border;
        # consume it so later prompts don't repeat it.
        hint, self.first_prompt_hint = self.first_prompt_hint, ""
        app = _build_input_app(
            self._history, prompt, hint=hint, context_pct=self._context_pct,
            model_names_fn=self._model_names_for_menu,
        )
        try:
            line = await app.run_async()
            self._sigint_armed = False
            return (line or "").strip()
        except KeyboardInterrupt:
            if self._sigint_armed:
                raise EOFError  # second consecutive Ctrl+C on an empty line → exit the REPL
            self._sigint_armed = True
            self._console.print("[system.info](Press Ctrl+C again to exit)[/system.info]")
            return ""
        finally:
            self._state = "IDLE"

    # ------------------------------------------------------------------ #
    # Persistent type-anytime input box
    # ------------------------------------------------------------------ #

    def can_run_input_box(self) -> bool:
        """The persistent box needs a real interactive TTY. Non-TTY (pipes, CI, capture)
        falls back to the classic read_input sequencing."""
        return bool(self._console.is_terminal)

    async def start_input_box(
        self, ctrl_queue: asyncio.Queue, on_interrupt: Callable[[], None]
    ) -> None:
        """Launch the long-lived input box as a sibling task, under patch_stdout(raw=True) so
        all turn output streams ABOVE it. Enter enqueues ('submit', text) onto ctrl_queue;
        Ctrl+D (empty) enqueues ('eof', None); Ctrl+C (empty) calls on_interrupt() — the REPL
        owns those policies. raw=True is load-bearing (rich ANSI is interpreted, not escaped)."""
        if self._history is None:
            raise ChannelStartError("TerminalChannel.start() must be called before start_input_box()")
        from prompt_toolkit.patch_stdout import patch_stdout

        # #49: the first prompt carries the guidance hint inside the box; consume it once.
        self._first_box_hint, self.first_prompt_hint = self.first_prompt_hint, ""
        self._box_active = True
        # ask_permission suspends the box and puts it back; it needs the REPL's own callbacks to
        # restart it, and this is the only place they exist.
        self._box_restart_args = (ctrl_queue, on_interrupt)

        def _on_submit(text: str) -> None:
            # #49: the guidance hint is shown "until first use" — a submit IS that first use.
            # Without this the box border keeps repeating it all session, while the classic
            # read_input path consumes it after one prompt (the two modes disagreed).
            self._first_box_hint = ""
            self._last_keystroke_at = time.monotonic()  # a submit is presence, whatever fed it
            ctrl_queue.put_nowait(("submit", text))

        def _on_eof() -> None:
            ctrl_queue.put_nowait(("eof", None))

        def _on_pending_answer(approve: bool) -> None:
            # Ctrl+Y / Ctrl+N. Dead when nothing is parked — a key that silently does something
            # invisible is worse than one that does nothing — and otherwise a control event, so
            # the REPL (which owns the gate) answers it inside its one serialized loop.
            if self._pending:
                ctrl_queue.put_nowait(("approve_pending" if approve else "deny_pending", None))

        # Opening the box IS somebody being there: the person just typed the command that got
        # here. Without this the first staged call of a session would ring no bell.
        self._last_keystroke_at = time.monotonic()
        self._box_app = _build_persistent_input_app(
            self._history, _PROMPT_GLYPH,
            on_submit=_on_submit, on_interrupt=on_interrupt, on_eof=_on_eof,
            hint_fn=self._box_hint_frags, right_fn=self._box_instrument_frags,
            status_fn=self._box_status_frags, placeholder_fn=self._box_placeholder,
            model_names_fn=self._model_names_for_menu,
            pending_fn=self._box_pending_frags, on_pending_answer=_on_pending_answer,
            on_keystroke=self._note_keystroke,
        )
        self._box_patch = patch_stdout(raw=True)
        self._box_patch.__enter__()
        self._box_task = asyncio.create_task(self._box_app.run_async())
        self._box_ticker = asyncio.create_task(self._box_tick())

    async def ask_permission(self, request: Any) -> Any:
        """Put one permission question to the person at the keyboard (PRD §3.5).

        The question is printed through the ordinary console path — so it lands above the
        persistent input box like every other line, wrapped and themed — and only the one-line
        option legend is a prompt_toolkit application. The box is suspended for the keystroke
        and restarted afterwards: two live applications cannot share one terminal, and
        suspending rather than tearing down keeps the REPL's queue and interrupt callbacks
        intact. Ctrl+C, Escape and Enter all answer "no, this once" (fail closed).

        What the keystroke LEAVES on screen is the question and one confirmation line
        (:data:`PERMISSION_ANSWER_LINES`) — nothing else. Both applications erase their own
        drawing on exit (:data:`ERASE_APPLICATIONS_WHEN_DONE`), which is what the owner asked
        for on 2026-09-11: "I don't like how it persists a chat box every time I answer (a) (y)
        or whatever; it should just record my input." Before that, one answer left a dead input
        box, a dead legend and then a new live box in the scrollback, three times per prompt.

        When stdout is redirected (`| tee`, `> file`) the question and its legend go to stderr
        instead, so the person at the keyboard still sees what they are answering — defect D6:
        `can_ask` follows stdin, and a question drawn into a pipe would be a question nobody
        can answer on a channel that now holds its dialog open forever.

        The question and the legend run under `patch_stdout(raw=True)`, the same protection
        `start_input_box` holds for the box's whole life. A turn does not stop while a human
        thinks: subagents keep streaming, and every one of those writes goes through the ordinary
        console path. Without the patch they landed straight on the terminal the legend was
        painting, and the line the person answers from was the corrupted one. The patch cannot be
        the `_output_lock` instead: that lock is held by whichever task is writing, so holding it
        across `run_async()` would stall every writer until a human answered — and `run_async`
        is exactly the wait that can last minutes. The lock stays where it belongs, around the
        print alone; `patch_stdout` is what makes concurrent writers safe, by rendering them
        ABOVE the running application instead of through it.

        The patch is stdout-only on purpose. It replaces `sys.stderr` as well and routes both
        into the app session's output, which on the redirected path is the pipe — the exact
        place this question must never be drawn.
        """
        from contextlib import nullcontext

        from prompt_toolkit.patch_stdout import patch_stdout

        from localharness.agent.gate_types import Decision

        restart = self._box_restart_args if self._box_active else None
        if restart is not None:
            await self.stop_input_box()
        try:
            async with self._output_lock:
                self._stop_thinking()
                self._close_burst()
            on_stdout = self.can_run_input_box()
            console = self._console if on_stdout else self._err_console
            # A request may carry its own legend (the workspace-trust question does — its answer
            # IS remembered, which the ungrantable legend's tail says the opposite of).
            options = getattr(request, "options_legend", None) or (
                PERMISSION_OPTIONS_GRANTABLE if request.grantable
                else PERMISSION_OPTIONS_UNGRANTABLE
            )
            output = None
            if not on_stdout:
                from prompt_toolkit.output.defaults import create_output

                output = create_output(always_prefer_tty=True)
            with (patch_stdout(raw=True) if on_stdout else nullcontext()):
                async with self._output_lock:
                    console.print(
                        f"[system.info]{PERMISSION_PROMPT_LABEL}:[/system.info] "
                        f"{escape(sanitize_for_display(request.display))}"
                    )
                try:
                    kind = await _build_permission_app(
                        options, request.grantable, output
                    ).run_async()
                except (KeyboardInterrupt, EOFError):
                    kind = PERMISSION_DEFAULT_DECISION
                kind = kind or PERMISSION_DEFAULT_DECISION
                # The keystroke erased the legend; this is what replaces it. Printed INSIDE the
                # patch, before the box is restarted, so it lands in the scrollback directly
                # under the question it answers rather than above a freshly redrawn box.
                lines = PERMISSION_ANSWER_LINES_BY_CLASS.get(
                    getattr(request, "klass", ""), PERMISSION_ANSWER_LINES
                )
                async with self._output_lock:
                    console.print(f"[system.info]{lines[kind]}[/system.info]")
            return Decision(kind=kind)
        finally:
            if restart is not None:
                await self.start_input_box(*restart)

    async def stop_input_box(self) -> None:
        """Tear the box down: stop the ticker, exit the app, release patch_stdout. Idempotent."""
        self._box_active = False
        for task_attr in ("_box_ticker", "_decision_flash_task"):
            task = getattr(self, task_attr)
            if task is not None:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                setattr(self, task_attr, None)
        if self._box_app is not None:
            try:
                if self._box_app.is_running:
                    self._box_app.exit()
            except Exception:
                pass
        if self._box_task is not None:
            try:
                await self._box_task
            except (asyncio.CancelledError, EOFError, KeyboardInterrupt, Exception):
                pass
            self._box_task = None
        if self._box_patch is not None:
            try:
                self._box_patch.__exit__(None, None, None)
            except Exception:
                pass
            self._box_patch = None
        self._box_app = None

    def _box_hint_frags(self) -> list[tuple[str, str]]:
        """Footer-left: the transient routing-decision flash, else the key legend for the box's
        CURRENT mode — the type-anytime semantics are invisible otherwise. While a turn runs:
        `queued (N)` when lines wait, then the two keys that matter (alt+enter forces a nudge,
        ctrl+c stops the turn); idle: tab lists the commands. (The first-run hint moved into
        the box as its placeholder — _box_placeholder.)"""
        if self._decision_flash:
            return [("class:caret", f"  {self._decision_flash}")]
        parts: list[str] = []
        if self._queued_count:
            parts.append(f"queued ({self._queued_count})")
        parts.append("alt+enter nudge · ctrl+c stop" if self._box_working else "tab commands · /help")
        return [("class:hint", "  " + " · ".join(parts))]

    def _box_pending_frags(self) -> list[tuple[str, str]]:
        """Content of the row UNDER the footer: what `auto` parked while the turn carried on.

        One muted line (:data:`BOX_PENDING_ROW`) naming the OLDEST parked call — the one both
        hotkeys answer — and counting the rest. Styled like the key legend, not like an error:
        nothing failed, the model was told to continue without the step. Returns [] when nothing
        is parked, so the ConditionalContainer collapses the row to zero height.
        """
        if not self._pending:
            return []
        oldest = self._pending[0]
        count = len(self._pending)
        line = "  " + BOX_PENDING_ROW.format(
            count=count, noun=BOX_PENDING_NOUNS[count != 1], id=oldest.id,
            rendering=oldest.rendering,
        )
        # The rendering carries the class and the reason and can outrun the pane, and the hotkey
        # legend at the END of the row is the part a person must see (dogfood 2026-09-12: a
        # 160-column pane lost `ctrl+n skip · /pending`). A row that does not fit becomes two
        # lines, the legend on its own, rather than an ellipsis eating the command.
        if len(line) > self._columns() > 0:
            head, _, legend = line.rpartition(BOX_PENDING_ROW_LEGEND_SEP)
            line = head + "\n     " + legend
        return [("class:hint", line)]

    @staticmethod
    def _columns() -> int:
        """The pane width the row must fit, from the running application's own output."""
        try:
            from prompt_toolkit.application import get_app

            return get_app().output.get_size().columns
        except Exception:  # noqa: BLE001 — no app (tests, teardown): let the row wrap
            return 0

    def _box_placeholder(self) -> str:
        """Dim text inside the empty box: the #49 guidance hint until first use, then nothing."""
        return self._first_box_hint

    def _box_instrument_frags(self) -> list[tuple[str, str]]:
        """Footer-right: the local-model instrument cluster — `model · 25.3 tok/s · ██░░░░░░░░ 8%`.
        The numbers a local-inference user actually watches, always at the input (no cloud CLI
        shows throughput at all; ours shows the MEASURED rate). The rate here is the RESTING
        one: verified, and only between turns — while a turn runs the status row above owns the
        readout (live `~` rate mid-stream, last verified rate between requests), so one number
        never shows twice on screen. Every supplier is optional and exception-proof."""
        sep = ("class:hint", " · ")
        frags: list[tuple[str, str]] = []
        model = None
        if self.model_source is not None:
            try:
                model = self.model_source()
            except Exception:
                model = None
        if model:
            frags.append(("class:model", model))
        rate = self._tps_text()
        if rate is not None and rate[2] and not self._box_working:
            frags += [sep] * bool(frags) + [(rate[0], rate[1])]
        if self._context_pct is not None:
            meter, _ = _ctx_segments(self._context_pct)
            frags += [sep] * bool(frags) + meter
        return frags

    def _box_status_frags(self) -> list[tuple[str, str]]:
        """FIX 2: content of the one-line status row rendered ABOVE the box (so it reads as the
        log area's last line). The working glyph + the live activity the box already tracks —
        an open tool-burst counter (`tools · done/calls`, in the same column its frozen line
        lands), the between-turns '· dreaming…' consolidation pass, or a plain 'working' while
        the model generates. Returns [] when idle so the ConditionalContainer collapses the row
        to zero height. Refreshed by app.invalidate() from _box_tick — never a rich Status."""
        if not (self._box_working or self._box_dreaming or self._box_activity):
            return []
        glyph = _SPIN_FRAMES[int(time.monotonic() * 8) % len(_SPIN_FRAMES)]
        show_tps = False  # only turn activity shows the rate — never a swap-load or dreaming line
        if self._burst is not None:
            activity = f"{' · '.join(self._burst.tools)} · {self._burst.done}/{self._burst.calls}"
            show_tps = True
        elif self._box_activity and not self._box_working:
            activity = self._box_activity   # e.g. a /model swap's 'loading <model> · 40s'
        elif self._box_dreaming and not self._box_working:
            activity = _DREAMING_LABEL   # · dreaming…
        else:
            frags = [("class:hint", f"  {glyph} ")] + self._working_frags() + [("class:hint", " ")]
            frags.extend(self._tps_frag())
            return frags
        frags = [("class:hint", f"  {glyph} {activity} ")]
        if show_tps:
            frags.extend(self._tps_frag())
        return frags

    def _working_frags(self) -> list[tuple[str, str]]:
        """Status-row text while the model generates — two fragments, then the tok/s one.
        A bare "working" was every phase at once (live: a 12-minute hidden think looked
        exactly like a hang). Fragment 1, colored by phase: the kind of the last delta with
        its icon and its live token tally — `⋯ thinking 2.3k`, `✎ writing 410`, `◆ tool
        call 120`, or `… waiting` before the first delta. Fragment 2, dim: elapsed time,
        which turns into a red `silent 14s` when no delta has arrived for a while
        mid-stream. Without a supplier (or between requests): the plain "working"."""
        snap = None
        if self.progress_source is not None:
            try:
                snap = self.progress_source()
            except Exception:
                snap = None
        if snap is None:
            return [("class:hint", "working")]
        phase = snap.get("phase") or "waiting"
        tally = {
            "thinking": int(snap.get("thinking_tokens") or 0),
            "writing": int(snap.get("answer_tokens") or 0),
            "tool_call": int(snap.get("tool_call_tokens") or 0),
        }.get(phase, 0)
        head = f"{_PHASE_ICONS.get(phase, '')} {_PHASE_LABELS.get(phase, phase)}".strip()
        if tally:
            head += f" {_fmt_tokens(tally)}"
        frags = [(f"class:phase-{phase}", head)]
        silent = float(snap.get("silent") or 0.0)
        if phase != "waiting" and silent >= _SILENCE_NOTE_SECONDS:
            frags.append(("class:phase-silent", f" · silent {_fmt_elapsed(silent)}"))
        else:
            frags.append(("class:hint", f" · {_fmt_elapsed(float(snap.get('elapsed') or 0.0))}"))
        return frags

    def _tps_frag(self) -> list[tuple[str, str]]:
        """Colored tok/s segment for the status row; [] when nothing is known. A live rate
        carries `~` (chunk-approximate while streaming — self-corrects to the exact rate at
        stream end); a verified rate is plain. Bands per speed_stats.tps_band (>30 green,
        20–30 yellow, <20 red). Never raises — the status row must never take the app down."""
        rate = self._tps_text()
        return [] if rate is None else [(rate[0], f"· {rate[1]} ")]

    def _tps_text(self) -> tuple[str, str, bool] | None:
        """(style class, `25.3 tok/s` | `~25 tok/s`, verified) from tps_source; None when unknown."""
        if self.tps_source is None:
            return None
        try:
            snap = self.tps_source()
        except Exception:
            return None
        if not snap:
            return None
        from localharness.provider.speed_stats import tps_band

        tps, verified = snap
        text = f"{tps:.1f} tok/s" if verified else f"~{tps:.0f} tok/s"
        return f"class:tps-{tps_band(tps)}", text, verified

    def _invalidate_box(self) -> None:
        if self._box_app is not None and self._box_active:
            try:
                self._box_app.invalidate()
            except Exception:
                pass

    def box_open_model_menu(self) -> None:
        """Bare /model: pre-fill '/model ' and pop the completion menu with the first model
        highlighted — the one-Enter picker (scroll + Enter switches). No-op without the box.

        Two guards, both about not taking the line away from the user: the listing that precedes
        this call takes seconds of live probes while the box keeps accepting input, so a line
        already being typed is NEVER overwritten (it is not in history yet — overwriting it
        destroys it); and with nothing to pick from the prefix is not injected at all.

        What IS injected is marked a loan (#135): the box takes it straight back the moment the
        user types or pastes anything instead of picking, so the next line is theirs alone."""
        app = self._box_app
        buf = getattr(app, "_lh_input_buffer", None) if app is not None else None
        if buf is None or buf.text:
            return
        if not self._model_names_for_menu():
            return  # empty menu (server unreachable) — a bare '/model ' prefix would only be noise
        buf.text = _MODEL_PICK_PREFIX
        buf._lh_picker_loan = True
        buf.cursor_position = len(buf.text)
        buf.start_completion(select_first=True)
        self._invalidate_box()

    def box_activity(self, text: str | None) -> None:
        """Transient activity note for the status row (e.g. a /model swap's loading line).
        None/empty clears it. Safe to call in any mode — a no-op without the box."""
        self._box_activity = text or ""
        self._invalidate_box()

    def box_set_queued(self, n: int) -> None:
        """Persistent `queued (N)` count shown in the box frame."""
        self._queued_count = max(0, n)
        self._invalidate_box()

    def box_set_pending(self, items: list[Any]) -> None:
        """The calls `auto` parked, OLDEST FIRST — what the row under the box reads.

        A setter rather than a callback into the gate, to match how the REPL already feeds this
        box its other queue (`box_set_queued(len(self._fifo))`): the REPL owns the gate, and a
        pull callback would make `channels` reach into `agent` for a type it deliberately does
        not import (see `ChannelAdapter.send_pending_notice` on why these are typed `Any`).
        Safe in any mode — without the box it just keeps the list.
        """
        self._pending = list(items)
        self._invalidate_box()

    def box_flash_decision(self, text: str, seconds: float = 2.0) -> None:
        """Show a transient routing-decision line ('→ nudging current turn' / 'queued (N)')
        the instant Enter is routed; it clears itself after `seconds`."""
        self._decision_flash = text
        self._invalidate_box()
        if self._decision_flash_task is not None:
            self._decision_flash_task.cancel()
        try:
            self._decision_flash_task = asyncio.get_running_loop().create_task(
                self._clear_flash_after(seconds, text)
            )
        except RuntimeError:
            self._decision_flash_task = None  # no loop (unit context): flash persists till next change

    async def _clear_flash_after(self, seconds: float, text: str) -> None:
        try:
            await asyncio.sleep(seconds)
            if self._decision_flash == text:  # not superseded by a newer decision
                self._decision_flash = ""
                self._invalidate_box()
        except asyncio.CancelledError:
            pass

    async def box_echo_prompt(self, text: str, annotation: str = "") -> None:
        """FIX 1: print one permanent scrollback line for a box submission — ❯ <text> [· note].

        Goes through the SAME patch_stdout-safe console + _output_lock the tool/agent lines use
        (never a rich Live/Status), so a message typed into the persistent box survives its
        buffer reset and stays in the transcript for scroll-back. `annotation` is a short dim
        suffix set at routing time (`queued (N)` / `→ nudge`); omitted for a plain turn-start
        echo. User text is markup-escaped; in box mode the burst counter accumulates silently
        (no live region), so interleaving a mid-turn echo can't corrupt anything."""
        line = f"[user.input]{_PROMPT_GLYPH}[/user.input] {escape(text)}"
        if annotation:
            line += f"  [muted]{_NARRATE} {escape(annotation)}[/muted]"
        async with self._output_lock:
            self._notes_shown.clear()  # a new user turn gets the close notes again
            self._console.print(line)

    def box_notify_working(self, working: bool) -> None:
        """Toggle the working state that drives the status row above the box (FIX 2):
        thinking/streaming/burst active. The glyph itself now renders in the status row
        (see _box_status_frags), no longer in the box's bottom border."""
        self._box_working = working
        self._invalidate_box()

    async def _box_tick(self) -> None:
        """Animate the status-row spinner while the box is live and there's activity — a turn
        working, or a background 'dreaming' pass. No rich Live anywhere: invalidate() +
        FormattedTextControl is the freeze-safe path under patch_stdout."""
        try:
            while self._box_active:
                if self._box_working or self._box_dreaming:
                    self._invalidate_box()
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass

    def _ensure_idle(self) -> None:
        """Log a warning if called while STREAMING (concurrent write detected)."""
        if self._state == "STREAMING":
            self._err_console.print("[warning]Warning: output during streaming state[/warning]")

    def _burst_text(self, burst: _Burst, final: bool) -> str:
        """Render the burst counter line: ◆ tools · done/calls [· N errors]."""
        head = f"{_DIAMOND} {escape(' · '.join(burst.tools))}"
        line = f"  [{burst.style}]{head}[/{burst.style}] [muted]· {burst.done}/{burst.calls}[/muted]"
        if final and burst.errors:
            plural = "s" if burst.errors > 1 else ""
            line += f" [tool.error]· {burst.errors} error{plural}[/tool.error]"
        return line

    def _burst_refresh(self, burst: _Burst) -> None:
        """Tick the live counter. TTY-only spinner; non-TTY counts silently and
        prints just the frozen line on close (captures stay one line per burst)."""
        if self._box_active:
            # Box mode: NO rich Status (it glues lines / freezes on Ctrl+C-during-burst under
            # patch_stdout). The count accumulates in the _Burst and its frozen line prints on
            # close; the live indicator is the in-frame working glyph.
            self.box_notify_working(True)
            return
        text = self._burst_text(burst, final=False)
        if burst.status is not None:
            burst.status.update(text)
        elif self._console.is_terminal and self._state == "IDLE":
            try:
                burst.status = self._console.status(text, spinner="dots")
                burst.status.start()
            except Exception:
                burst.status = None  # a broken spinner must never break output

    def _close_burst(self) -> None:
        """Freeze an open burst into its final scrollback line (+ close note).
        Call with _output_lock held, or from teardown where no writer can race."""
        burst, self._burst = self._burst, None
        if burst is None:
            return
        if burst.status is not None:
            try:
                burst.status.stop()
            finally:
                burst.status = None
        self._console.print(self._burst_text(burst, final=True))
        if burst.close_note and burst.done:
            self._print_close_note(burst.close_note)

    def _print_close_note(self, note: str) -> None:
        """A family's disclosure line, once per user turn: narration splits one research burst
        into several, and the same note under each of them was a third of the tool section
        (2026-09-11). Call with _output_lock held."""
        if note in self._notes_shown:
            return
        self._notes_shown.add(note)
        self._console.print(f"  [tool.result]{_CHECK} {note}[/tool.result]")

    def _start_thinking(self) -> None:
        """Animated indicator while an LLM round-trip is in flight (REPL-02).
        IDLE-only: never over the input bubble, a streaming panel, an open burst
        counter (the burst spinner is already the live indicator), or the dreaming
        status (rich allows one live display \u2014 a turn beginning stops it first)."""
        if self._box_active:
            # Box mode: in-frame glyph, never a rich Status (freeze-safe under patch_stdout).
            self.box_notify_working(True)
            return
        if self._thinking is not None:
            try:  # already spinning (a later iteration's heartbeat): refresh the tok/s suffix
                self._thinking.update(self._thinking_label())
            except Exception:
                pass
            return
        if (self._dreaming is None and self._burst is None and self._state == "IDLE"):
            try:
                self._thinking = self._console.status(self._thinking_label(), spinner="dots")
                self._thinking.start()
            except Exception:
                self._thinking = None  # a broken spinner must never break output

    def _thinking_label(self) -> str:
        """Classic-mode spinner label: 'thinking\u2026' plus the last VERIFIED decode rate, colored
        by band. No per-token ticker outside the box \u2014 the label refreshes once per iteration
        heartbeat, so only the verified (stream-end) figure is shown, never the mid-stream
        approximation."""
        label = "[muted]thinking\u2026[/muted]"
        snap = None
        if self.tps_source is not None:
            try:
                snap = self.tps_source()
            except Exception:
                snap = None
        if snap and snap[1]:
            from localharness.provider.speed_stats import tps_band

            band = tps_band(snap[0])
            label += f" [{band}]{snap[0]:.1f} tok/s[/{band}]"
        return label

    def _start_dreaming(self) -> None:
        """Quiet '\u00b7 dreaming\u2026' status while a background memory consolidation/mining
        pass runs (#20). Extends _start_thinking: same console.status, IDLE-only \u2014 so it can
        never draw over the input bubble (WAITING_INPUT) or a stream, and never opens a second
        live display alongside the thinking spinner or a burst counter."""
        if self._box_active:
            # box mode: no rich Status (freeze-safe). Animate it in the status row above the
            # box instead — closes the v0.9.10 v1 gap where dreaming had no box-mode indicator.
            self._box_dreaming = True
            self._invalidate_box()
            return
        if (self._dreaming is None and self._thinking is None
                and self._burst is None and self._state == "IDLE"):
            try:
                self._dreaming = self._console.status(
                    f"[muted]{_DREAMING_LABEL}[/muted]", spinner="dots"
                )
                self._dreaming.start()
            except Exception:
                self._dreaming = None  # a broken spinner must never break output

    def _stop_dreaming(self) -> None:
        if self._box_dreaming:
            self._box_dreaming = False  # box mode: clear the status-row dreaming indicator
            self._invalidate_box()
        if self._dreaming is not None:
            try:
                self._dreaming.stop()
            finally:
                self._dreaming = None

    def _stop_thinking(self) -> None:
        if self._box_active:
            self.box_notify_working(False)  # box mode: drop the in-frame glyph
        if self._thinking is not None:
            try:
                self._thinking.stop()
            finally:
                self._thinking = None
        self._stop_dreaming()  # #20: real output / input tears the dreaming dot down too (stop-first)

    async def on_action(self, event: Action) -> None:
        """Render interstitial narration for an llm_response that ALSO made tool calls, then
        fall through to the base handler for tool_call display. A tool-less llm_response is
        the final answer (rendered by the TaskComplete panel) — the has_tool_calls
        discriminator keeps it from being echoed here (the double-print regression)."""
        if event.action_type == "llm_response":
            await self.flush_reasoning()  # the generation is over: print its last partial line
            await self._render_narration(event)
            return
        await super().on_action(event)

    async def on_reasoning(self, text: str) -> None:
        """Print streamed reasoning as dim ⋯ lines (no-op unless show_reasoning). Complete
        lines print as they arrive; a long unbroken paragraph prints in _REASON_FLUSH_CHARS
        pieces so a 3-minute think is never silent. Never tears down the box's working glyph
        (thinking continues) — only a classic-mode rich Status, which would glue lines."""
        if not (self.show_reasoning or self.verbose) or not text:
            return
        self._reasoning_buf += text
        lines: list[str] = []
        while "\n" in self._reasoning_buf:
            line, self._reasoning_buf = self._reasoning_buf.split("\n", 1)
            lines.append(line)
        if len(self._reasoning_buf) >= _REASON_FLUSH_CHARS:
            lines.append(self._reasoning_buf)
            self._reasoning_buf = ""
        if lines:
            await self._print_reasoning_lines(lines)

    async def flush_reasoning(self) -> None:
        """Print whatever reasoning is still buffered (the generation ended)."""
        tail, self._reasoning_buf = self._reasoning_buf, ""
        if (self.show_reasoning or self.verbose) and tail.strip():
            await self._print_reasoning_lines([tail])
        self._reasoning_open = False

    async def _print_reasoning_lines(self, lines: list[str]) -> None:
        async with self._output_lock:
            if self._thinking is not None:
                self._stop_thinking()
            if not self._reasoning_open:
                self._close_burst()  # freeze an open tool-burst counter before prose appears
                self._reasoning_open = True
            for line in lines:
                if line.strip():
                    self._console.print(f"  [muted]{_REASON} {escape(line.rstrip())}[/muted]",
                                        soft_wrap=True)

    async def _render_narration(self, event: Action) -> None:
        """One dim line opening a chunk of tool activity. No-op unless the response also made
        tool calls (final answers render elsewhere), its content has a non-empty first line,
        and a tool result has rendered since the last narration — the cadence guard: at most
        one narration line per chunk, never two dim lines in a row."""
        if not event.has_tool_calls:
            return
        line = _narration_line(event.content)
        if not line:
            return
        async with self._output_lock:
            if not self._tool_result_since_narration:
                return
            self._stop_thinking()
            self._close_burst()  # freeze the previous chunk's counter before this one opens
            self._tool_result_since_narration = False
            self._console.print(
                f"  [muted]{_NARRATE} {escape(line)}[/muted]",
                no_wrap=True, overflow="ellipsis",
            )

    async def on_heartbeat(self, event: Heartbeat) -> None:
        """Track context utilization (shown in the next input bubble) and start the
        thinking indicator \u2014 the Heartbeat fires right before each LLM call, so this
        is the 'model is now generating' signal (REPL-02)."""
        self._context_pct = event.context_utilization_pct
        async with self._output_lock:
            self._stop_dreaming()  # a turn is beginning — replace the dreaming dot with thinking
            self._start_thinking()

    async def on_consolidation_started(self, event: ConsolidationStarted) -> None:
        """A background memory consolidation/mining pass began (#20): show the quiet
        '· dreaming…' status if the REPL is idle (never over the input bubble or a stream)."""
        async with self._output_lock:
            self._start_dreaming()

    async def on_consolidation_finished(self, event: ConsolidationFinished) -> None:
        """The background pass ended (#20): clear the dreaming status. Touches only the
        dreaming slot, so a thinking spinner started by a turn mid-pass is left intact."""
        async with self._output_lock:
            self._stop_dreaming()
