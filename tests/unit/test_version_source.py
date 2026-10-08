"""Version source of truth (#97): banners/--version prefer localharness.__version__, not stale
dist metadata. Editable/live installs read a stale installed version (observed: banner v0.9.16
while source was v0.9.19) — the in-source __version__ is authoritative; metadata is fallback only."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from rich.console import Console

import localharness
from localharness import resolved_version
from localharness.cli.ui import startup_banner

_REPO = Path(__file__).resolve().parents[2]


def test_resolved_version_prefers_source_over_stale_metadata(monkeypatch):
    """__version__ wins even when installed dist metadata says something else (the #97 bug)."""
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.0.1-stale")
    assert resolved_version() == localharness.__version__


def test_resolved_version_falls_back_to_metadata_when_source_absent(monkeypatch):
    """Metadata is the fallback only — reachable when __version__ is somehow empty."""
    monkeypatch.setattr(localharness, "__version__", "")
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "9.9.9-from-metadata")
    assert resolved_version() == "9.9.9-from-metadata"


def test_banner_shows_source_version_not_stale_metadata(monkeypatch):
    """The observed surface of #97: the startup banner printed the stale metadata version."""
    monkeypatch.setattr("importlib.metadata.version", lambda _name: "0.0.1-stale")
    console = Console(width=100, file=None, record=True)
    console.print(startup_banner(model="qwen", is_returning=True, show_hint=False))
    out = console.export_text()
    assert f"v{localharness.__version__}" in out
    assert "0.0.1-stale" not in out


def test_the_release_is_one_number():
    """pyproject.toml, `__version__`, uv.lock, the README's "Early stage" line and the newest
    CHANGELOG entry carry the same version. 0.16.8, 0.16.9 and 0.17.0 bumped pyproject alone, so
    the published 0.17.0 wheel printed v0.16.7 in its banner and `update` offered 0.17.0 to itself
    on every run."""
    pyproject = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    lock = next(p["version"] for p in tomllib.loads((_REPO / "uv.lock").read_text(encoding="utf-8"))["package"]
                if p["name"] == "localharness")
    readme = re.search(r"Early stage \(v(\d+\.\d+\.\d+), pre-1\.0\)",
                       (_REPO / "README.md").read_text(encoding="utf-8"))
    changelog = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (_REPO / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
    assert readme is not None and changelog is not None
    found = {"pyproject.toml": pyproject, "__version__": localharness.__version__, "uv.lock": lock,
             "README.md": readme.group(1), "CHANGELOG.md": changelog.group(1)}
    assert set(found.values()) == {pyproject}, found
