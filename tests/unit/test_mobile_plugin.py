"""MobilePlugin: the bundled mobile channel plugin — manifest, channel, import lightness, the reachable hint."""
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
    from localharness.cli.mobile_plugin import MobileConfig, MobilePlugin
    from localharness.memory.plugin import MemoryPlugin
    from localharness.plugins.builtin import bundled_plugins
    from localharness.tools.builtin.image_plugin import ImagePlugin

    m = MobilePlugin.manifest
    assert (m.name, m.kind, m.requires_extra, m.enabled_by_default) == ("mobile", "channel", "mobile", True)
    assert m.cli[0].name == "mobile" and m.cli[0].target == "localharness.cli.mobile_cmd:app"
    assert MobilePlugin.ConfigModel is MobileConfig and set(MobileConfig.model_fields) == {"public_url"}
    from localharness.autoresearch.plugin import AutoresearchPlugin
    from localharness.dispatch.plugin import DispatchPlugin

    # dispatch (49) is bundled and on by default; autoresearch (50) is bundled and on by default
    assert bundled_plugins() == (ImagePlugin, MobilePlugin, MemoryPlugin, DispatchPlugin, AutoresearchPlugin)


def test_web_plugin_channels_names_the_web_channel() -> None:
    from localharness.channels.mobile.channel import MobileChannel
    from localharness.cli.mobile_plugin import MobilePlugin

    assert MobilePlugin().channels() == {"mobile": MobileChannel}


def test_importing_the_bundled_list_pulls_no_web_stack(tmp_path) -> None:
    r = _run("import sys, localharness.plugins.builtin; bad=[m for m in ('starlette','uvicorn',"
             "'localharness.channels','localharness.channels.mobile') if m in sys.modules]; print(bad); "
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
        "w = CliRunner().invoke(app, ['mobile'])\n"
        "print('WEB', w.exit_code); print(w.output); print(repr(w.exception))\n",
        tmp_path,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "HELP 0" in out and " mobile " in out, out
    assert "WEB 1" in out, out
    assert "the mobile channel needs its optional extra" in out, out


def _ctx(config_dir):
    """doctor reads only ctx.paths.global_config_dir; doctor runs with no session (llm None)."""
    from types import SimpleNamespace
    return SimpleNamespace(paths=SimpleNamespace(global_config_dir=config_dir), llm=None)


def test_the_doctor_port_is_the_port_the_server_binds() -> None:
    """One literal: mobile_cmd binds the plugin's number, it does not restate it."""
    import inspect

    from localharness.cli import mobile_cmd
    from localharness.cli.mobile_plugin import MOBILE_DEFAULT_PORT

    assert mobile_cmd.DEFAULT_PORT == MOBILE_DEFAULT_PORT
    assert str(MOBILE_DEFAULT_PORT) not in inspect.getsource(mobile_cmd)


def test_web_doctor_not_enrolled(tmp_path) -> None:
    from localharness.cli.mobile_plugin import MobilePlugin
    from localharness.plugins.api import Check

    assert MobilePlugin().doctor(_ctx(tmp_path)) == [
        Check(name="mobile", status="skip", detail="not enrolled yet",
              hint="`localharness mobile` generates its app token on first run")]


def test_web_doctor_enrolled_and_token_mode(tmp_path) -> None:
    """The bind the server enforces and the token file's mode — never the token itself (doctor
    output ends up in bug reports)."""
    from localharness.channels.mobile.auth import rotate_token, token_path
    from localharness.cli.mobile_plugin import MobilePlugin

    rotate_token(tmp_path)
    secret = token_path(tmp_path).read_text().strip()
    web, tok = MobilePlugin().doctor(_ctx(tmp_path))
    assert web.name == "mobile" and web.status == "pass"
    for part in ("binds 127.0.0.1:8765", "loopback only unless --allow-unsafe-bind",
                 "A token is required on every request", f"Token file: {token_path(tmp_path)}"):
        assert part in web.detail, web.detail
    assert (tok.name, tok.status, tok.detail) == ("mobile-token", "pass", "token file is mode 600")
    assert not any(secret in c.detail + c.hint for c in (web, tok))

    os.chmod(token_path(tmp_path), 0o644)
    _, tok = MobilePlugin().doctor(_ctx(tmp_path))
    assert tok.status == "fail" and "mode 644, expected 600" in tok.detail
    assert tok.hint == f"chmod 600 {token_path(tmp_path)}"
