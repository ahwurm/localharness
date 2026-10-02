"""Importing the dispatch plugin is cheap (49-05, RESEARCH P1): plugins/builtin.py imports it on every
`--help`, so it must load neither discord.py nor prompt_toolkit nor the channel core (anything under
localharness.channels runs channels/__init__.py). A fresh interpreter, so nothing is preloaded."""
from __future__ import annotations

import subprocess
import sys

CODE = (
    "import sys, localharness.dispatch.plugin\n"
    "bad = [m for m in ('discord', 'prompt_toolkit', 'localharness.dispatch.channel',\n"
    "                   'localharness.dispatch.adapters', 'localharness.channels') if m in sys.modules]\n"
    "assert not bad, bad\n"
)


def test_importing_the_plugin_loads_no_channel_and_no_sdk():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
