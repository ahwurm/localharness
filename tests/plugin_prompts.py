"""The one test seam for a plugin prompt or yes/no answer."""
from __future__ import annotations

from localharness.cli import plugins_cmd


def fake_prompts(monkeypatch, prompt=None, confirm=None) -> None:
    """Answer the plugin questions with legacy typer-shaped `prompt` and `confirm` fakes, each only when given."""
    if prompt is not None:
        async def ask(text, *, console, default="", secret=False):
            return prompt(text.removesuffix(": "), default=default, hide_input=secret, show_default=bool(default))

        monkeypatch.setattr(plugins_cmd, "prompt_line", ask)
    if confirm is not None:
        async def sure(text, *, console, default, tail=""):
            return confirm(text, default=default)

        monkeypatch.setattr(plugins_cmd, "confirm_line", sure)
