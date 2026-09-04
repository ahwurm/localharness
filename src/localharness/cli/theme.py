"""CLI entity colors — the localharness.dev architecture-plate palette.

The architecture page on localharness.dev draws the harness as a set of SVG "plates"
(RuntimePlate, MemoryPlateII, AutoresearchPlate, ...), and every architectural entity on
those plates has its own hue. Those hues are OKLCH design tokens declared in the site's
`src/styles/global.css` `@theme` block, where they carry a comment reserving them for
"architecture plates ONLY — the rest of the site stays single-accent green".

The terminal reuses those exact hues so a reader who has seen the plates recognizes the
same entity in `start`'s output: memory is memory-cyan in both places. Values below are
the sRGB conversions of the site's OKLCH tokens, each named after the token it came from
rather than pasted as a bare literal. Rich downsamples truecolor to 256/16-color on its
own, so no terminal-capability branching is needed here.

Two rules this module encodes:

- **Status stays status.** Success/failure glyphs (✓/✗) keep their green/red verdict
  semantics and are NOT entity-colored. The entity color goes on the NAME beside the
  glyph, never on the verdict.
- **Escape inside, style outside.** `entity()` escapes the value it is handed and leaves
  the style tag live, per the repo-wide `[old] proj` rule (see
  tests/unit/test_cli_markup_guards.py): the tag is ours and must stay markup; the value
  is the user's and must render literally.
"""
from __future__ import annotations

from rich.markup import escape
from rich.text import Text

# --- Site tokens (localharness.dev/src/styles/global.css @theme) -----------------------
# Each constant is the sRGB form of the OKLCH token named in its comment.
SITE_ACCENT = "#56DC85"    # --color-accent   agent runtime; also the sitewide accent
SITE_INFRA = "#7FBEF3"     # --color-infra    channels + event bus
SITE_ORCH = "#9EACF1"      # --color-orch     orchestrator
SITE_AMBER = "#F0BB3B"     # --color-amber    provider; also the sitewide warning tone
SITE_TOOL = "#BDA8FC"      # --color-tool     tool system (registry, MCP, plugins, hooks)
SITE_MEM = "#47D2E8"       # --color-mem      memory
SITE_BENCH = "#F7857D"     # --color-bench    bench substrate / corpus split
SITE_RESEARCH = "#FB9D59"  # --color-research autoresearch loop + component registry
SITE_INK = "#E9EBEF"       # --color-ink      primary text — the reader's own voice
SITE_DIM = "#9B9EA6"       # --color-dim      secondary text

# The site paints the CONFIG group in --color-edge (#272B34), a near-black BORDER neutral.
# As a terminal foreground on a dark ground that is unreadable, so config takes the site's
# other neutral, --color-dim. This is the one deliberate divergence from the plates.
_CONFIG = SITE_DIM

ENTITY_STYLES: dict[str, str] = {
    "agent": SITE_ACCENT,
    "orchestrator": SITE_ORCH,
    "tool": SITE_TOOL,
    "memory": SITE_MEM,
    "provider": SITE_AMBER,
    "channel": SITE_INFRA,
    "infra": SITE_INFRA,
    "config": _CONFIG,
    "bench": SITE_BENCH,
    "research": SITE_RESEARCH,
    "warning": SITE_AMBER,
}


def entity(name: str, value: str) -> str:
    """Rich markup painting `value` in entity `name`'s architecture-plate color.

    `value` is escaped, the style tag is not — a path or an exception string carrying
    `[Errno 2]` renders literally instead of being eaten as a tag. Unknown entity names
    raise KeyError on the spot rather than silently rendering unstyled.
    """
    return f"[{ENTITY_STYLES[name]}]{escape(value)}[/]"


def entity_text(name: str, value: str) -> Text:
    """Same color as `entity()`, as a Text span rather than a markup string.

    For Rich Tables, whose cells run their str contents through the markup parser: a Text
    cell carries the color while making the value literal by construction, so it cannot be
    swallowed or style-injected at all.
    """
    return Text(value, style=ENTITY_STYLES[name])
