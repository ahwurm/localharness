"""Pure helpers for a plugin's setup step (`plugins enable`, in a shell or from a session). Core
code: no terminal I/O and nothing imported from a plugin — cli/plugins_cmd.py asks, writes and
prints; this module only shapes text and answers questions about a plugin class."""
from __future__ import annotations

import re
from collections.abc import Mapping

from localharness.plugins.api import AGENT_PROMPT_PLACEHOLDER, Plugin

AGENT_PROMPT_LEAD = "Or paste this into your coding agent to set it up for your hardware:"


def render_agent_prompt(template: str, values: Mapping[str, str]) -> str:
    """The prompt block: AGENT_PROMPT_LEAD, a blank line, then `template` with every {name} filled
    from `values` (a name it lacks renders as nothing) and each line indented two spaces. A
    placeholder that renders empty leaves no double space. "" for an empty template."""
    if not template:
        return ""
    body = AGENT_PROMPT_PLACEHOLDER.sub(lambda m: values.get(m.group(1), ""), template)
    body = re.sub(r"(?<=\S) {2,}(?=\S)", " ", body)
    return AGENT_PROMPT_LEAD + "\n\n" + "\n".join(f"  {line}".rstrip() for line in body.splitlines())


def machine_sentence(gpu: str | None) -> str:
    """What {machine} renders as: "This machine reports <gpu>." — nothing when the machine reports
    no GPU (never placeholder text)."""
    return f"This machine reports {gpu}." if gpu else ""


def has_setup_action(cls: type) -> bool:
    """Does this plugin class override Plugin.setup_action?"""
    return getattr(cls, "setup_action", Plugin.setup_action) is not Plugin.setup_action
