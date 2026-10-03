"""The phone token (D5): printed only when it is made or asked for, and only to a terminal.

journald, a tee, a tmux pipe — whatever `localharness web` prints, a log keeps, and the 0600 token
file protects nothing once a copy sits in one of them. So the token TEXT is printed once, when it is
created, or on `--show-token`; the pairing QR (which carries the token, and scanning it is pairing)
is drawn only on a terminal; a run whose stdout is not a terminal prints one line naming
`--show-token` instead. Driven through the real command and the real `_serve` — only uvicorn's
listen is faked — because CliRunner is never a terminal, the check is patched where a terminal is
the case under test.
"""
from __future__ import annotations

import json
import logging
import os
import stat
from types import SimpleNamespace

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli import web_cmd

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

PUBLIC = "https://spark.example.ts.net"
FAKE_QR = "█▀▄ the pairing code ▄▀█"
TOKEN_LINE = "App token (required on every request, the stream included):"
SHOW_TOKEN = "localharness web --show-token"


@pytest.fixture
def web(tmp_path, monkeypatch):
    """`localharness web <args>` through the real command and the real `_serve`. Faked: uvicorn's
    listen (it records that serving began), the QR renderer (it records the URL it was handed) and
    the tailscale guess. `tty` patches the terminal check; None keeps the real one, which under
    CliRunner reads a captured stream — never a terminal."""
    import uvicorn

    seen = SimpleNamespace(served=[], qr=[])

    async def no_listen(self, *a, **kw):
        seen.served.append(self.config.port)

    def qr(url):
        seen.qr.append(url)
        return FAKE_QR

    monkeypatch.setattr(uvicorn.Server, "serve", no_listen)
    monkeypatch.setattr(web_cmd, "render_qr", qr)
    monkeypatch.setattr(web_cmd, "detect_public_url", lambda port, **kw: None)  # no `tailscale`
    real = getattr(web_cmd, "_stdout_is_a_terminal", None)

    def invoke(*args, tty=None):
        monkeypatch.setattr(web_cmd, "_stdout_is_a_terminal",
                            real if tty is None else (lambda: tty), raising=False)
        return CliRunner().invoke(
            web_cmd.app, ["--config-dir", str(tmp_path), "--public-url", PUBLIC, *args],
            env={"COLUMNS": "200"})

    seen.invoke = invoke
    seen.token = lambda: (tmp_path / "web" / "token").read_text(encoding="utf-8").strip()
    return seen


def _flat(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------- when the token is printed

def test_a_run_whose_stdout_is_not_a_terminal_prints_no_token_and_no_qr(web, tmp_path):
    """The first run makes the token (0600) and still serves; it says how to pair instead."""
    result = web.invoke()  # the REAL terminal check: CliRunner's stdout is not one
    assert result.exit_code == 0, result.output
    token = web.token()
    assert stat.S_IMODE(os.stat(tmp_path / "web" / "token").st_mode) == 0o600
    assert token not in result.output
    assert web.qr == [] and FAKE_QR not in result.output, "the QR carries the token: no QR either"
    assert SHOW_TOKEN in result.output
    assert web.served, "a run off a terminal still serves"


def test_on_a_terminal_the_qr_is_drawn_every_start_and_the_token_text_only_when_made(web):
    first = web.invoke(tty=True)
    assert first.exit_code == 0, first.output
    token = web.token()
    assert web.qr == [f"{PUBLIC}/#t={token}"], "the QR carries the token: scanning it is pairing"
    assert FAKE_QR in first.output
    assert TOKEN_LINE in first.output and first.output.index(token) > first.output.index(TOKEN_LINE)

    second = web.invoke(tty=True)
    assert second.exit_code == 0, second.output
    assert web.qr[-1] == f"{PUBLIC}/#t={token}" and FAKE_QR in second.output
    assert token not in second.output, "the token text is printed only when it is created"
    for out in (first.output, second.output):
        assert f"{PUBLIC}/" in out and "#t=" not in out, "the printed address carries no token"
    assert len(web.served) == 2


def test_show_token_prints_and_exits_without_serving(web, tmp_path, monkeypatch):
    """The friction answer: a server already running under systemd or tmux is paired from another
    terminal without stopping it — `--show-token` prints and exits, it never binds the port."""
    from localharness.channels.web import auth

    token, _ = auth.load_or_create_token(tmp_path)
    served: list = []

    async def fake_serve(**kw):
        served.append(kw)

    monkeypatch.setattr(web_cmd, "_serve", fake_serve)
    result = web.invoke("--show-token", tty=True)
    assert result.exit_code == 0, result.output
    assert TOKEN_LINE in result.output and token in result.output
    assert web.qr == [f"{PUBLIC}/#t={token}"] and FAKE_QR in result.output
    assert served == [] and web.served == []


def test_show_token_refuses_a_stdout_that_is_not_a_terminal(web, tmp_path):
    from localharness.channels.web import auth

    token, _ = auth.load_or_create_token(tmp_path)
    result = web.invoke("--show-token")  # the real check
    assert result.exit_code == 1
    assert "--show-token prints the token only to a terminal, and stdout is not one." in result.output
    assert token not in result.output and web.qr == [] and web.served == []


# ---------------------------------------------------------------- rotation

def test_rotation_clears_push_and_says_notifications_must_be_turned_on_again(web, tmp_path):
    """A stolen phone kept receiving notifications after a rotation: its push subscription
    outlived its token. Rotation now deletes every subscription, and its receipt says so."""
    from localharness.channels.web import auth, push

    old, _ = auth.load_or_create_token(tmp_path)
    subs = push.subscriptions_path(tmp_path)
    subs.write_text(json.dumps([{"endpoint": "https://push.example/x", "keys": {}}]), encoding="utf-8")

    result = web.invoke("--rotate-token", tty=True)
    assert result.exit_code == 0, result.output
    new = web.token()
    assert new != old and not subs.exists()
    flat = _flat(result.output)
    assert "turn notifications on again" in flat
    # A server that is already running holds the old token in memory until it restarts.
    assert "until it restarts" in flat
    assert web.qr == [f"{PUBLIC}/#t={new}"] and new in result.output and old not in result.output
    assert web.served == []


def test_rotation_off_a_terminal_prints_no_token_and_names_show_token(web, tmp_path):
    from localharness.channels.web import auth, push

    old, _ = auth.load_or_create_token(tmp_path)
    subs = push.subscriptions_path(tmp_path)
    subs.write_text("[]", encoding="utf-8")

    result = web.invoke("--rotate-token")  # the real check
    assert result.exit_code == 0, result.output
    new = web.token()
    assert new != old and not subs.exists()
    assert new not in result.output and old not in result.output and web.qr == []
    assert SHOW_TOKEN in result.output and "turn notifications on again" in _flat(result.output)


# ---------------------------------------------------------------- the QR address and the no-QR note

def _tailscale(name):
    return lambda cmd: SimpleNamespace(returncode=0, stdout=json.dumps({"Self": {"DNSName": name}}))


@pytest.mark.parametrize("name", [
    "evil/../#x", "a b.ts.net", "", "spark.ts.net@evil.example", "x" * 64 + ".ts.net",
    ("a" * 63 + ".") * 4 + "ts.net",
])
def test_a_dns_name_that_is_not_a_host_name_never_reaches_the_qr(name):
    """tailscaled's answer goes into a URL a phone opens; only a real host name may."""
    assert web_cmd.detect_public_url(8765, runner=_tailscale(name)) is None


def test_a_host_name_with_its_trailing_dot_is_the_guess():
    got = web_cmd.detect_public_url(8765, runner=_tailscale("spark.tail1234.ts.net."))
    assert got == "https://spark.tail1234.ts.net"


def test_without_the_qr_library_the_note_says_to_enter_the_token_and_names_show_token(web, monkeypatch):
    """The printed address carries no fragment any more, so "the part after the #" is gone."""
    monkeypatch.setattr(web_cmd, "render_qr", lambda url: None)
    result = web.invoke(tty=True)
    assert result.exit_code == 0, result.output
    assert _flat(web_cmd.ENROLMENT_NO_QR) in _flat(result.output)
    assert "#" not in web_cmd.ENROLMENT_NO_QR and "the part after" not in web_cmd.ENROLMENT_NO_QR
    assert SHOW_TOKEN in web_cmd.ENROLMENT_NO_QR


# ---------------------------------------------------------------- a failed bring-up

async def test_a_failed_bring_up_tells_the_phone_the_error_with_every_stored_secret_masked(
        tmp_path, monkeypatch, caplog):
    """An error can quote a key. The phone's bring-up row and the server's own log line (stderr,
    often a tee or journald) carry the error's text with every key the machine's files hold
    masked; the phone gets its first line only."""
    from localharness.cli import start_cmd

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"version": "1", "provider": {
        "provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m",
        "api_key": "sk-SENT-BRINGUP-1"}}), encoding="utf-8")
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": [{
        "name": "peer", "base_url": "http://10.0.0.2:8000/v1", "api_key": "sk-SENT-PEER-2"}]}),
        encoding="utf-8")

    async def boom(*a, **kw):
        raise RuntimeError("could not load sk-SENT-BRINGUP-1 (peer sk-SENT-PEER-2)\nsecond line")

    monkeypatch.setattr(start_cmd, "_start_async", boom)
    stages: list = []
    channel = SimpleNamespace(session_id=None,
                              set_bringup=lambda stage, **kw: stages.append((stage, kw)))
    with caplog.at_level(logging.WARNING):
        await web_cmd._bring_up(channel, config_dir=str(tmp_path), verbose=False, agent=None)

    stage, kw = stages[-1]
    assert stage == "failed" and kw["failed"] is True
    assert kw["detail"].startswith("RuntimeError:") and "could not load" in kw["detail"]
    assert "sk-SENT" not in kw["detail"] and "second line" not in kw["detail"]
    assert "bring-up failed" in caplog.text and "could not load" in caplog.text
    assert "sk-SENT" not in caplog.text
