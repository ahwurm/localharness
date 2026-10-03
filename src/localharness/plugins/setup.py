"""Helpers for a plugin's setup step (`plugins enable`, in a shell or from a session). Core code:
no terminal I/O and nothing imported from a plugin — cli/plugins_cmd.py asks, writes and prints;
this module shapes text, answers questions about a plugin class, and asks the machine for its
GPU's name (gpu_name, run only when a prompt that names {machine} is rendered)."""
from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from localharness.core.which import which_outside_cwd
from localharness.plugins.api import AGENT_PROMPT_PLACEHOLDER, Plugin

AGENT_PROMPT_LEAD = "Or paste this into your coding agent to set it up for your hardware:"
_SYSFS_DRM = Path("/sys/class/drm")
_TIMEOUT_S = 2.0
_VENDORS = {"0x10de": "an NVIDIA GPU", "0x1002": "an AMD GPU"}


def render_agent_prompt(template: str, values: Mapping[str, str]) -> str:
    """The prompt block: AGENT_PROMPT_LEAD, a blank line, then `template` with every {name} filled
    from `values` (a name it lacks renders as nothing) and each line indented two spaces. A
    placeholder that renders empty leaves no double space; a filled value is printed exactly as
    given (a folder name keeps its own spaces). "" for an empty template."""
    if not template:
        return ""
    filled: list[str] = []

    def mark(m: re.Match[str]) -> str:  # values go in after the squeeze, so theirs are never touched
        filled.append(values.get(m.group(1), ""))
        return f"\x00{len(filled) - 1}\x00" if filled[-1] else ""

    body = re.sub(r"(?<=\S) {2,}(?=\S)", " ", AGENT_PROMPT_PLACEHOLDER.sub(mark, template))
    body = re.sub("\x00(\\d+)\x00", lambda m: filled[int(m.group(1))], body)
    return AGENT_PROMPT_LEAD + "\n\n" + "\n".join(f"  {line}".rstrip() for line in body.splitlines())


def machine_sentence(gpu: str | None) -> str:
    """What {machine} renders as: "This machine reports <gpu>." — nothing when the machine reports
    no GPU (never placeholder text)."""
    return f"This machine reports {gpu}." if gpu else ""


def has_setup_action(cls: type) -> bool:
    """Does this plugin class override Plugin.setup_action?"""
    return getattr(cls, "setup_action", Plugin.setup_action) is not Plugin.setup_action


def _lines(argv: list[str]) -> list[str]:
    """The non-empty stdout lines of one short command (list argv, no shell); [] on any failure."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()] if out.returncode == 0 else []


def gpu_name() -> str | None:
    """This machine's GPU as the machine itself reports it, for the {machine} sentence of a coding-
    agent prompt — the one fact about the hardware a coding agent should be told. nvidia-smi first
    (the only source that names a GB10: /proc and sysfs give 'Unknown' or a bare PCI id there),
    then the PCI vendor in sysfs, then a Mac's chip; None when nothing answers. Runs only when a
    prompt is rendered."""
    exe = which_outside_cwd("nvidia-smi")
    names = _lines([exe, "--query-gpu=name", "--format=csv,noheader"]) if exe else []
    if names:
        counts = {n: names.count(n) for n in dict.fromkeys(names)}
        return ", ".join(n if c == 1 else f"{c}x {n}" for n, c in counts.items())
    if sys.platform.startswith("linux"):
        try:
            vendors = {p.read_text().strip().lower() for p in _SYSFS_DRM.glob("card[0-9]*/device/vendor")}
        except OSError:
            vendors = set()
        found = next((label for vid, label in _VENDORS.items() if vid in vendors), None)
        if found:
            return found
    if sys.platform == "darwin":
        chip = _lines(["sysctl", "-n", "machdep.cpu.brand_string"])
        return chip[0] if chip else None
    return None
