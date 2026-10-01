"""WebPlugin: the bundled web channel plugin — manifest, channel, import lightness, the reachable hint."""
from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "HOME": str(tmp_path / "home"), "LOCALHARNESS_DIR": str(tmp_path / "lh")},
    )


def test_web_plugin_manifest() -> None:
    from localharness.cli.web_plugin import WebPlugin
    from localharness.plugins.builtin import bundled_plugins
    from localharness.tools.builtin.image_plugin import ImagePlugin

    m = WebPlugin.manifest
    assert (m.name, m.kind, m.requires_extra, m.enabled_by_default) == ("web", "channel", "web", True)
    assert m.cli[0].name == "web" and m.cli[0].target == "localharness.cli.web_cmd:app"
    assert WebPlugin.ConfigModel is None
    assert bundled_plugins() == (ImagePlugin, WebPlugin)


def test_web_plugin_channels_names_the_web_channel() -> None:
    from localharness.channels.web.channel import WebChannel
    from localharness.cli.web_plugin import WebPlugin

    assert WebPlugin().channels() == {"web": WebChannel}


def test_importing_the_bundled_list_pulls_no_web_stack(tmp_path) -> None:
    r = _run("import sys, localharness.plugins.builtin; bad=[m for m in ('starlette','uvicorn',"
             "'localharness.channels','localharness.channels.web') if m in sys.modules]; print(bad); "
             "raise SystemExit(1 if bad else 0)", tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_without_the_web_stack_still_helps_and_hints(tmp_path) -> None:
    r = _run(
        "import sys\n"
        "sys.modules['starlette'] = None; sys.modules['uvicorn'] = None\n"
        "from typer.testing import CliRunner\n"
        "from localharness.cli.app import app\n"
        "h = CliRunner().invoke(app, ['--help'])\n"
        "print('HELP', h.exit_code); print(h.output)\n"
        "w = CliRunner().invoke(app, ['web'])\n"
        "print('WEB', w.exit_code); print(w.output); print(repr(w.exception))\n",
        tmp_path,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "HELP 0" in out and " web " in out, out
    assert "WEB 1" in out, out
    assert "the web channel needs its optional extra" in out, out


def _ctx(config_dir):
    """doctor reads only ctx.paths.global_config_dir; doctor runs with no session (llm None)."""
    from types import SimpleNamespace
    return SimpleNamespace(paths=SimpleNamespace(global_config_dir=config_dir), llm=None)


def test_the_doctor_port_is_the_port_the_server_binds() -> None:
    """One literal: web_cmd binds the plugin's number, it does not restate it."""
    import inspect

    from localharness.cli import web_cmd
    from localharness.cli.web_plugin import WEB_DEFAULT_PORT

    assert web_cmd.DEFAULT_PORT == WEB_DEFAULT_PORT
    assert str(WEB_DEFAULT_PORT) not in inspect.getsource(web_cmd)


def test_web_doctor_not_enrolled(tmp_path) -> None:
    from localharness.cli.web_plugin import WebPlugin
    from localharness.plugins.api import Check

    assert WebPlugin().doctor(_ctx(tmp_path)) == [
        Check(name="web", status="skip", detail="not enrolled yet",
              hint="`localharness web` generates its app token on first run")]


def test_web_doctor_enrolled_and_token_mode(tmp_path) -> None:
    """The bind the server enforces and the token file's mode — never the token itself (doctor
    output ends up in bug reports)."""
    from localharness.channels.web.auth import rotate_token, token_path
    from localharness.cli.web_plugin import WebPlugin

    rotate_token(tmp_path)
    secret = token_path(tmp_path).read_text().strip()
    web, tok = WebPlugin().doctor(_ctx(tmp_path))
    assert web.name == "web" and web.status == "pass"
    for part in ("binds 127.0.0.1:8765", "loopback only unless --allow-unsafe-bind",
                 "A token is required on every request", f"Token file: {token_path(tmp_path)}"):
        assert part in web.detail, web.detail
    assert (tok.name, tok.status, tok.detail) == ("web-token", "pass", "token file is mode 600")
    assert not any(secret in c.detail + c.hint for c in (web, tok))

    os.chmod(token_path(tmp_path), 0o644)
    _, tok = WebPlugin().doctor(_ctx(tmp_path))
    assert tok.status == "fail" and "mode 644, expected 600" in tok.detail
    assert tok.hint == f"chmod 600 {token_path(tmp_path)}"
