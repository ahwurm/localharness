"""The one test seam for a plugin prompt or yes/no answer."""
from __future__ import annotations

from localharness.cli import plugins_cmd


def fake_prompts(monkeypatch, prompt=None, confirm=None) -> None:
    """Install `prompt` and `confirm` as the answers to the plugin questions, each only when given."""
    if prompt is not None:
        monkeypatch.setattr(plugins_cmd.typer, "prompt", prompt)
    if confirm is not None:
        monkeypatch.setattr(plugins_cmd.typer, "confirm", confirm)
