"""Tests for the CLI entity palette (cli/theme.py).

The palette is a port of the localharness.dev architecture-plate hues, so the tests that
matter are: the map covers every entity the CLI paints, the hexes still match the site
tokens they were taken from, and the value a style wraps stays LITERAL — the `[old] proj`
rule (see test_cli_markup_guards.py), which is easy to lose the moment a value has to
carry a color as well as be escaped.
"""
from __future__ import annotations

import io

import pytest
from rich.console import Console
from rich.style import Style

from localharness.cli.theme import (
    ENTITY_STYLES,
    SITE_ACCENT,
    SITE_AMBER,
    SITE_INK,
    SITE_MEM,
    SITE_TOOL,
    entity,
    entity_text,
)

# The `[old] proj` fixture, both shapes: `[old]` is silently deleted, `[/red]` raises
# MarkupError. Same constant the start/doctor/config markup guards use.
HOSTILE = "[old] [/red]proj"


def _render(renderable) -> str:
    """Render through a REAL Console so the markup parser actually runs."""
    buf = io.StringIO()
    Console(file=buf, width=200, no_color=True).print(renderable, soft_wrap=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Map completeness
# --------------------------------------------------------------------------- #
def test_every_entity_the_cli_paints_has_a_style():
    """start's summary/roster paint these entities; a missing key is a KeyError at runtime,
    on the startup path, after the session is already wired."""
    for name in ("agent", "orchestrator", "tool", "memory", "provider", "channel",
                 "infra", "config", "bench", "research", "warning"):
        assert name in ENTITY_STYLES, f"{name} missing from ENTITY_STYLES"


def test_every_style_is_a_hex_rich_can_parse():
    """A typo'd hex renders as an unstyled string in some Rich versions and raises in
    others — parse every value up front so it can never reach a terminal."""
    for name, value in ENTITY_STYLES.items():
        assert value.startswith("#") and len(value) == 7, f"{name}={value!r} is not #RRGGBB"
        assert Style.parse(value).color is not None, f"{name}={value!r} did not parse"


def test_hexes_still_match_the_site_tokens():
    """Drift guard: these are localharness.dev's @theme tokens, not free-choice colors.
    If the site repalettes, this test is the place that should fail and be updated."""
    assert ENTITY_STYLES["agent"] == "#56DC85"      # --color-accent
    assert ENTITY_STYLES["orchestrator"] == "#9EACF1"  # --color-orch
    assert ENTITY_STYLES["tool"] == "#BDA8FC"       # --color-tool
    assert ENTITY_STYLES["memory"] == "#47D2E8"     # --color-mem
    assert ENTITY_STYLES["provider"] == "#F0BB3B"   # --color-amber
    assert ENTITY_STYLES["channel"] == "#7FBEF3"    # --color-infra
    assert ENTITY_STYLES["bench"] == "#F7857D"      # --color-bench
    assert ENTITY_STYLES["research"] == "#FB9D59"   # --color-research


def test_entities_the_plates_distinguish_do_not_collide():
    """The whole point is telling types apart at a glance; two entities sharing a hue on
    DIFFERENT plates would silently undo that."""
    distinct = ("agent", "orchestrator", "tool", "memory", "provider", "channel")
    hues = [ENTITY_STYLES[n] for n in distinct]
    assert len(set(hues)) == len(hues), f"hue collision among {distinct}: {hues}"


def test_channel_and_infra_share_the_infra_hue_deliberately():
    """The RuntimePlate legend gives CHANNELS and INFRA the same --color-infra token."""
    assert ENTITY_STYLES["channel"] == ENTITY_STYLES["infra"]


def test_unknown_entity_fails_loudly():
    """No silent fallback to unstyled — a typo'd entity name is a bug, not a color choice."""
    with pytest.raises(KeyError):
        entity("nosuchentity", "x")


def test_banner_accent_is_the_shared_site_token():
    """cli/ui.py used to re-paste the accent hex; it now sources it, so the wordmark and
    the agent entity can never drift apart."""
    from localharness.cli import ui

    assert ui._GREEN == SITE_ACCENT


# --------------------------------------------------------------------------- #
# Escape discipline: the style is ours, the value is the user's
# --------------------------------------------------------------------------- #
def test_entity_renders_a_markup_shaped_value_literally():
    """`[old]` must survive as text. Unescaped, Rich reads it as a tag and DELETES it."""
    out = _render(entity("agent", HOSTILE))
    assert HOSTILE in out, out


def test_entity_does_not_raise_on_an_unopened_closing_tag():
    """`[/red]` with nothing open raises MarkupError and kills the command."""
    assert "[/red]" in _render(entity("memory", "/home/u/[/red]proj/memory.db"))


def test_entity_keeps_its_own_style_tag_live():
    """Escaping the whole string instead of just the value would print the tag verbatim."""
    assert SITE_MEM not in _render(entity("memory", "Memory: in-memory"))
    buf = io.StringIO()
    Console(file=buf, width=200, force_terminal=True, color_system="truecolor").print(
        entity("memory", "Memory: in-memory")
    )
    assert "71;210;232" in buf.getvalue(), buf.getvalue()  # #47D2E8 as truecolor SGR


def test_entity_escapes_an_errno_bracket():
    """The exact shape that made start's warnings vanish: `[Errno 2]` inside the value."""
    assert "[Errno 2]" in _render(entity("warning", "memory: [Errno 2] locked"))


def test_entity_text_is_literal_by_construction_and_carries_the_style():
    """Text cells never touch the markup parser — the safe form for Table cells."""
    text = entity_text("agent", HOSTILE)
    assert text.plain == HOSTILE
    assert text.style == SITE_ACCENT
    assert HOSTILE in _render(text)


def test_tool_and_memory_are_not_the_same_green_they_replaced():
    """Regression on the actual complaint: entities used to read as one undifferentiated
    colour. Tools and memory must not both be the accent."""
    assert ENTITY_STYLES["tool"] != SITE_ACCENT
    assert ENTITY_STYLES["memory"] != SITE_ACCENT
    assert ENTITY_STYLES["tool"] == SITE_TOOL
    assert ENTITY_STYLES["warning"] == SITE_AMBER


# --------------------------------------------------------------------------- #
# The per-turn session surface (channels/terminal.py TERMINAL_THEME) draws from the same
# map, so a tool call in a turn is the same purple as a tool count at startup.
# --------------------------------------------------------------------------- #
def _terminal_theme() -> dict[str, str]:
    """Rendered style strings, lowercased: Rich's Style.__str__ lowercases hex, so a
    case-sensitive `in` would pass vacuously on the negative assertions below."""
    from localharness.channels.terminal import TERMINAL_THEME

    return {k: str(v).lower() for k, v in TERMINAL_THEME.styles.items()}


def _hue(name: str) -> str:
    return ENTITY_STYLES[name].lower()


def test_session_theme_entities_come_from_the_entity_map():
    """agent/tool/system keys are wired to ENTITY_STYLES, not re-picked per key."""
    styles = _terminal_theme()
    assert _hue("agent") in styles["agent.name"]
    assert _hue("tool") in styles["tool.call"]
    assert _hue("tool") in styles["tool.result"]
    assert _hue("infra") in styles["system.info"]
    assert _hue("warning") in styles["warning"]


def test_session_verdict_styles_stay_green_and_red():
    """success/failure keep their verdict vocabulary — entity color never lands here."""
    styles = _terminal_theme()
    assert "green" in styles["success"]
    assert "red" in styles["tool.error"]
    assert "red" in styles["system.error"]
    for verdict in ("success", "tool.error", "system.error"):
        for name in ENTITY_STYLES:
            assert _hue(name) not in styles[verdict], f"{verdict} was entity-colored"


def test_user_and_agent_are_still_tellable_apart():
    """The scrollback's color language: the agent took the accent green the plates give
    the runtime, so the user must NOT also be green."""
    styles = _terminal_theme()
    assert _hue("agent") not in styles["user.input"]
    assert SITE_INK.lower() in styles["user.input"]


def test_tool_call_and_result_share_a_hue_but_not_a_weight():
    """Same entity, different emphasis: the call leads, the result trails."""
    styles = _terminal_theme()
    assert styles["tool.call"] != styles["tool.result"]
    assert "dim" in styles["tool.result"]


def test_body_text_keeps_no_hue():
    """Names take the entity color; bodies stay neutral."""
    styles = _terminal_theme()
    for name in ENTITY_STYLES:
        assert _hue(name) not in styles["agent.text"]
        assert _hue(name) not in styles["muted"]


def test_a_markup_named_agent_renders_literally_through_the_session_theme():
    """The live path: agent_id goes into an [agent.name] panel title. `[/red]` there would
    raise MarkupError and take the turn down; the style tag must stay live while the NAME
    stays literal."""
    from rich.markup import escape as _escape

    from localharness.channels.terminal import TERMINAL_THEME

    buf = io.StringIO()
    console = Console(file=buf, width=200, no_color=True, theme=TERMINAL_THEME)
    console.print(f"[agent.name]{_escape(HOSTILE)}[/agent.name]")
    console.print(f"  [tool.call]◆ {_escape(HOSTILE)}[/tool.call]")

    assert buf.getvalue().count(HOSTILE) == 2, buf.getvalue()
