"""Phase 46 (the web channel as a plugin), composed on the surfaces the rig can reach.

The unit tests of 46-02..46-07 prove each part; this file proves they compose through the real
entry points: the Typer app through CliRunner (`--help`, `web`, `plugins list|disable`, `start`);
the plugin resolver and its config layers; `_start_async` run in web mode with a real WebChannel
handed in; the WebServer over httpx's ASGI transport (no socket); the page's reducer run verbatim
under node, fed the exact bodies the server returned.

STUBBED: the LLM probe, the tokenizer and the REPL's read loop (`_stub_start_boundaries`; the
`drive` below stands in for the loop and does its HTTP work while the session is live); ComfyUI
(the image e2e's MockTransport — the image plugin is only here to bind an artifact root).

NOT proven here: the owner's phone (46-09's device checkpoint).
"""
from __future__ import annotations

import asyncio
import sys

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.cli.slash_commands import set_plugin_rows
from localharness.cli.web_cmd import MISSING_DEPENDENCY
from tests.integration.test_image_plugin_e2e import _machine
from tests.unit.test_plugin_cli_mount import _help_rows
from tests.unit.test_start_cmd import _capture_start_console

runner = CliRunner()


@pytest.fixture(autouse=True)
def _process_state():
    yield
    set_plugin_rows(())


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"`localharness {' '.join(args)}` raised {result.exception!r}\n{result.output}")
    return result


def _flat(text: str) -> str:
    """Rich soft-wraps; compare with the wrapping collapsed."""
    return " ".join(text.split())


# --- criterion 1 (CLI half): mounted from the plugin, gone when disabled, hint without the extra --


def test_web_is_a_plugin_command_on_by_default_and_gone_when_disabled(tmp_path, monkeypatch, fake_home):
    global_dir, _ = _machine(tmp_path, monkeypatch, fake_home)
    assert "web" in _help_rows(_invoke("--help").output)
    helped = _invoke("web", "--help")
    assert helped.exit_code == 0, helped.output
    assert "--no-store" in helped.output and "--rotate-token" in helped.output, helped.output
    listed = _invoke("plugins", "list").output
    assert any(line.split()[:1] == ["web"] for line in listed.splitlines()), listed

    disabled = _invoke("plugins", "disable", "web")
    assert disabled.exit_code == 0, disabled.output
    assert "web" not in _help_rows(_invoke("--help").output)
    gone = _invoke("web")
    assert gone.exit_code == 2 and "No such command 'web'" in gone.output, gone.output

    # the terminal is unaffected: a terminal session still boots with web off
    printed = _capture_start_console(monkeypatch)

    async def run() -> None:
        from localharness.cli.start_cmd import _start_async
        await _start_async(None, False, False, None)

    asyncio.run(run())
    assert any("startup)" in line for line in printed), printed
    assert not any("plugin web:" in line for line in printed), printed


def test_without_the_extra_web_prints_the_unchanged_hint(tmp_path, monkeypatch, fake_home):
    from localharness.plugins import resolve

    _machine(tmp_path, monkeypatch, fake_home)
    monkeypatch.setitem(resolve.resolve.__kwdefaults__, "extra_installed", lambda e: False)
    monkeypatch.setitem(sys.modules, "starlette", None)
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert "web" in _help_rows(_invoke("--help").output)  # still mounted: the hint, not "No such command"
    ran = _invoke("web")
    assert ran.exit_code == 1, ran.output
    assert _flat(MISSING_DEPENDENCY) in _flat(ran.output), ran.output
    listed = _invoke("plugins", "list").output
    (row,) = [line for line in listed.splitlines() if line.split()[:1] == ["web"]]
    assert "on (install `localharness[web]` to use it)" in row, row


# --- criterion 2: a channel typo is refused before any plugin loads -------------------------------


def test_a_channel_typo_is_refused_before_any_plugin_loads(tmp_path, monkeypatch, fake_home):
    _machine(tmp_path, monkeypatch, fake_home)
    called: list = []

    def boom(*a, **k):
        called.append(a)
        raise AssertionError("resolve() must not run before the channel gate")

    monkeypatch.setattr("localharness.plugins.resolve.resolve", boom)
    ran = _invoke("start", "--channel", "wbe")
    assert ran.exit_code != 0, ran.output
    assert "unknown channel 'wbe'; choose one of: acp, discord, terminal, web" in _flat(ran.output), ran.output
    assert called == [], "resolve() ran before the channel name was checked"
