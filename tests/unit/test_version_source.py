"""Version source of truth (#97): banners/--version prefer localharness.__version__, not stale
dist metadata. Editable/live installs read a stale installed version (observed: banner v0.9.16
while source was v0.9.19) — the in-source __version__ is authoritative; metadata is fallback only."""
from __future__ import annotations

from rich.console import Console

import localharness
from localharness import resolved_version
from localharness.cli.ui import startup_banner


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
