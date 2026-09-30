"""The release docs keep two promises the release QA found broken.

- An upgrade note: 0.15.0 documented a plugin API that the next release removes, and the CHANGELOG
  said nothing. The entry for the release that removes it sits above `## [0.15.0]`.
- A link: the notice for a plugin written for 0.15 points at a heading in spec 09. Renaming that
  heading must fail here, not 404 in silence for the user who follows the link.
"""
from __future__ import annotations

import re
from pathlib import Path

from localharness.plugins.discovery import LEGACY_API_DOC

_REPO = Path(__file__).resolve().parents[2]
_BLOB = "https://github.com/ahwurm/localharness/blob/main/"


def _github_anchors(markdown: str) -> set[str]:
    """The anchor GitHub gives each heading outside a code fence: lower-cased, every character other
    than a letter, digit, space, `-` or `_` dropped, and each space turned into `-`."""
    anchors: set[str] = set()
    fenced = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (m := re.match(r"#{1,6} (.+)", line)):
            anchors.add(re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-"))
    return anchors


def test_the_legacy_notice_links_to_a_real_spec_heading() -> None:
    assert LEGACY_API_DOC.startswith(_BLOB), LEGACY_API_DOC
    path, _, anchor = LEGACY_API_DOC.removeprefix(_BLOB).partition("#")
    assert path == "docs/specs/09-hooks-plugins.md"
    assert (_REPO / path).is_file(), _REPO / path
    anchors = _github_anchors((_REPO / path).read_text(encoding="utf-8"))
    assert "hook-specifications" in anchors  # the control: the file's headings were read at all
    assert anchor in anchors, f"#{anchor} names no heading in {path}"


def test_the_changelog_names_the_removed_015_plugin_api() -> None:
    above, found, _ = (_REPO / "CHANGELOG.md").read_text(encoding="utf-8").partition("\n## [0.15.0]")
    assert found, "CHANGELOG.md has no 0.15.0 entry"
    assert re.search(r"^## \[", above, re.M), "no release entry above 0.15.0"
    missing = [s for s in ("localharness.tools", "localharness.hooks", "manifest.yaml",
                           "on_agent_start", "localharness plugins") if s not in above]
    assert not missing, f"the entry above 0.15.0 does not name: {missing}"
