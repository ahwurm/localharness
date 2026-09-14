"""The enrolment QR (WEBCH-27): `localharness web` prints something a phone can scan.

The criterion is "a new user sets up from `docs/web.md` alone, on a phone, without hand-typing a
secret" — so the failure these guard against is a QR that encodes a URL no phone can reach, or
one that omits the token and therefore saves nobody any typing.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from localharness.cli import web_cmd


def test_the_qr_carries_the_token_in_the_fragment():
    """The fragment is the whole reason this is allowed: it is never sent to a server, so it
    lands in no access log, no proxy log and no Referer — which is what §7.3's rule protects."""
    url, kind = web_cmd.enrolment_url(
        "sekrit", public_url="https://spark.example.ts.net", host="127.0.0.1", port=8765)
    assert url == "https://spark.example.ts.net/#t=sekrit"
    assert kind == "given"
    # Never a query string: that one DOES reach the server.
    assert "?t=" not in url and "token=" not in url


def test_a_given_public_url_beats_any_guess():
    url, kind = web_cmd.enrolment_url(
        "s", public_url="https://explicit.example/", host="127.0.0.1", port=8765)
    assert url.startswith("https://explicit.example/#t=")
    assert kind == "given"


def test_the_tailnet_name_is_guessed_when_no_url_is_given():
    def fake_run(cmd):
        assert cmd[:2] == ["tailscale", "status"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"Self": {"DNSName": "spark.tail1234.ts.net."}}),
        )

    # The real function with an injected runner, rather than patching around it.
    assert web_cmd.detect_public_url(8765, runner=fake_run) == "https://spark.tail1234.ts.net"


def test_no_tailscale_is_not_an_error():
    """Tailscale is the supported topology, not a requirement (§7.1). Plain LAN and any reverse
    proxy are legitimate, so a missing CLI must degrade, never fail a start-up."""
    def explode(cmd):
        raise FileNotFoundError("tailscale")

    assert web_cmd.detect_public_url(8765, runner=explode) is None


def test_a_loopback_fallback_says_so(monkeypatch):
    """The honest failure. A QR of `http://127.0.0.1:8765` scans perfectly and then does
    nothing at all on a phone, which is the worst kind of working."""
    monkeypatch.setattr(web_cmd, "detect_public_url", lambda port, **kw: None)
    url, kind = web_cmd.enrolment_url("s", public_url=None, host="127.0.0.1", port=8765)
    assert kind == "loopback"
    assert url == "http://127.0.0.1:8765/#t=s"


def test_the_qr_renders_as_terminal_art():
    art = web_cmd.render_qr("https://spark.example.ts.net/#t=" + "x" * 43)
    assert art is not None
    lines = art.splitlines()
    # Compact half-block rendering: it has to fit a terminal nobody resized.
    assert len(lines) < 30
    assert max(len(line) for line in lines) < 60
    assert "█" in art


def test_a_missing_segno_costs_the_qr_and_nothing_else(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_segno(name, *args, **kwargs):
        if name == "segno":
            raise ImportError("no segno")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_segno)
    assert web_cmd.render_qr("https://example/#t=x") is None


def test_printing_the_enrolment_never_wraps_the_code(capsys, monkeypatch):
    """Rich reflowing a QR turns it into a QR that does not scan, so this one is printed with
    markup off and wrapping off. A narrow console is the case that would expose it."""
    from rich.console import Console

    monkeypatch.setattr(web_cmd, "console", Console(width=20, force_terminal=False))
    web_cmd.print_enrolment("tok", public_url="https://spark.example.ts.net", host="h", port=1)
    out = capsys.readouterr().out
    code = [line for line in out.splitlines() if "█" in line]
    assert code, "no QR was printed"
    assert len(set(len(line) for line in code)) == 1, "the QR was reflowed into ragged rows"
    assert len(code[0]) > 20, "the QR was wrapped to the console width"


def test_the_command_offers_a_public_url_flag():
    """Without it the QR can only ever guess, and WEBCH-27 is a setup a stranger can complete."""
    import inspect

    assert "public_url" in inspect.signature(web_cmd.web_cmd).parameters


def test_rotating_the_token_reprints_a_qr(tmp_path, capsys, monkeypatch):
    """§7.2: rotation is the answer to "my phone was stolen while the app was still enrolled",
    and it is useless if re-pairing means hand-typing the new secret.

    Driven through the real command. The earlier version of this test sliced the function's
    SOURCE and would have stayed green if `raise typer.Exit(0)` moved above the print, making
    the QR dead code — a mutation a reviewer found, not the suite.
    """
    import typer

    from localharness.channels.web import auth as web_auth

    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path))
    before = web_auth.load_or_create_token(tmp_path)[0]
    capsys.readouterr()

    with pytest.raises(typer.Exit) as exit_info:
        web_cmd.web_cmd(config_dir=str(tmp_path), rotate_token=True,
                        public_url="https://spark.example.ts.net")
    assert exit_info.value.exit_code == 0

    out = capsys.readouterr().out
    after = web_auth.load_or_create_token(tmp_path)[0]
    assert after != before, "the token was not actually rotated"
    assert "█" in out, "no QR was printed for the new token"
    assert after in out, "the QR and its URL must carry the NEW token"
    assert before not in out
