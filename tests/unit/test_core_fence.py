"""The one fence rule for project text: the tag, the fixed output and the lines that cannot close it."""
from __future__ import annotations

import re
import subprocess
import sys

from localharness.core.fence import TAG, fence_project_text

OPEN = f"<project-instructions {TAG}>"
CLOSE = f"</project-instructions {TAG}>"


def test_the_fence_has_a_fixed_output():
    assert fence_project_text("## H", "Use tabs.\n") == "## H\n<project-instructions " + TAG + ">\nUse tabs.\n</project-instructions " + TAG + ">"


def test_the_tag_is_32_hex_characters():
    assert re.fullmatch(r"[0-9a-f]{32}", TAG)


def test_the_tag_differs_between_processes():
    code = "from localharness.core.fence import TAG as t; print(t)"
    tags = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                       timeout=60).stdout.strip()
        for _ in range(2)
    ]
    assert all(re.fullmatch(r"[0-9a-f]{32}", t) for t in tags)
    assert tags[0] != tags[1]
