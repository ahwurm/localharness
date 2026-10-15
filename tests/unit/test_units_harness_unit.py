"""The harness unit's body, its inverse, the shell capture and the no-follow file helpers (spec 20 §9.2).

Tmp directories only: `XDG_CONFIG_HOME` and `HOME` are redirected for every test, and nothing here asks a
service manager anything."""
from __future__ import annotations

import os
import shlex
import threading

import pytest

from localharness.provider import procfs
from localharness.provider import units as u

INTERPRETER = "/opt/lh/bin/python"
CONFIG_DIR = "/home/u/.localharness"
HOME = "/home/u"
PATH_VALUE = "/usr/bin:/bin"
HF = "/home/u/hf"
LD = "/opt/cuda/lib:/usr/lib"

GOLDEN = (
    "# written by localharness 0.19.0 — `localharness model --autostart off` removes it\n"
    "# env PATH=/usr/bin:/bin\n"
    "# env HF_HOME=/home/u/hf\n"
    "[Unit]\n"
    "Description=LocalHarness model server (serves the preset `localharness model` selected)\n"
    "After=network.target\n"
    "StartLimitIntervalSec=5400\n"
    "StartLimitBurst=3\n"
    "\n"
    "[Service]\n"
    "WorkingDirectory=/home/u\n"
    "ExecStart=/opt/lh/bin/python -I -m localharness serve --config-dir /home/u/.localharness\n"
    "UnsetEnvironment=LOCALHARNESS_DIR LOCALHARNESS_HOME LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH GCONV_PATH "
    "PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE\n"
    "Restart=on-failure\n"
    "RestartSec=5\n"
    "RestartPreventExitStatus=78\n"
    "TimeoutStartSec=1800\n"
    "KillSignal=SIGINT\n"
    "\n"
    "[Install]\n"
    "WantedBy=default.target\n"
)


@pytest.fixture(autouse=True)
def _redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def render(captured=None, **over):
    kw = dict(home=HOME, version="0.19.0")
    kw.update(over)
    return u.render_harness_unit(INTERPRETER, CONFIG_DIR,
                                 {"PATH": PATH_VALUE, "HF_HOME": HF} if captured is None else captured, **kw)


# ----------------------------------------------------------------------------------- the render


def test_the_render_is_the_golden_text_of_the_spec():
    assert render() == GOLDEN


def test_only_path_captured_renders_no_other_env_line():
    body = render({"PATH": PATH_VALUE})
    assert "# env PATH=/usr/bin:/bin\n[Unit]\n" in body
    assert "HF_HOME" not in body and "LD_LIBRARY_PATH=" not in body


def test_all_three_captured_render_in_the_fixed_order():
    body = render({"LD_LIBRARY_PATH": LD, "HF_HOME": HF, "PATH": PATH_VALUE})
    lines = body.split("\n")
    assert lines[1:4] == [f"# env PATH={PATH_VALUE}", f"# env HF_HOME={HF}", f"# env LD_LIBRARY_PATH={LD}"]
    assert lines[4] == "[Unit]"


def test_the_default_version_is_the_running_one():
    from localharness import resolved_version
    assert render(version=None).startswith(f"# written by localharness {resolved_version()} — ")


@pytest.mark.parametrize("captured", [
    {"PATH": PATH_VALUE},
    {"PATH": PATH_VALUE, "HF_HOME": HF},
    {"PATH": PATH_VALUE, "LD_LIBRARY_PATH": LD},
    {"PATH": PATH_VALUE, "HF_HOME": HF, "LD_LIBRARY_PATH": LD},
])
def test_captured_env_round_trips_the_render(captured):
    back = u.captured_env(render(captured))
    assert back == captured
    assert list(back) == [n for n in u.CAPTURED_NAMES if n in captured]


def test_the_rendered_exec_start_is_the_serve_a_unit_process_shows():
    line = next(l for l in render().split("\n") if l.startswith("ExecStart="))
    argv = tuple(shlex.split(line[len("ExecStart="):]))
    rec = procfs.ProcRecord(pid=4242, uid=os.getuid(), argv=argv, exe=INTERPRETER, cwd=HOME, start_time=1,
                            unit=procfs.HARNESS_UNIT, environ=None, mapped_ggufs=())
    assert procfs.serve_starting(rec, main_pid=4242) is True
    assert procfs.serve_starting(rec, main_pid=4243) is False


def test_render_and_captured_env_read_no_environment_and_no_file(monkeypatch):
    import ast
    import inspect
    for fn in (u.render_harness_unit, u.captured_env, u._captured_value_ok, u._unit_value,
               u._absolute_unit_path):
        tree = ast.parse(inspect.getsource(fn).lstrip())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not names & {"environ", "getenv", "open", "read_harness_unit", "expanduser_home"}, fn.__name__

    def boom(*a, **kw):
        raise AssertionError("must stay pure")
    monkeypatch.setattr("builtins.open", boom)
    monkeypatch.setattr(os, "open", boom)
    body = render()
    assert u.captured_env(body) == {"PATH": PATH_VALUE, "HF_HOME": HF}


# ----------------------------------------------------------------- the charset boundary of the render

ALLOWED = "/" + "".join(sorted(set("abcXYZ019/._+@:=-")))


def test_exactly_the_allowed_characters_render():
    assert u.UNIT_VALUE_PATTERN.fullmatch(ALLOWED)
    body = u.render_harness_unit(f"/{ALLOWED}/python", f"/{ALLOWED}", {"PATH": ALLOWED}, home=f"/{ALLOWED}",
                                 version="0.19.0")
    assert f"# env PATH={ALLOWED}\n" in body


@pytest.mark.parametrize("bad", ["a b", "a$b", "a%b", 'a"b', "a'b", "a\nb", "a\\b", "a;b", "é"])
def test_one_character_outside_the_set_refuses_naming_the_value(bad):
    value = f"/x{bad}y"
    for kwargs in (
        dict(interpreter=value, config_dir=CONFIG_DIR, captured={"PATH": PATH_VALUE}, home=HOME),
        dict(interpreter=INTERPRETER, config_dir=value, captured={"PATH": PATH_VALUE}, home=HOME),
        dict(interpreter=INTERPRETER, config_dir=CONFIG_DIR, captured={"PATH": PATH_VALUE}, home=value),
        dict(interpreter=INTERPRETER, config_dir=CONFIG_DIR, captured={"PATH": value}, home=HOME),
        dict(interpreter=INTERPRETER, config_dir=CONFIG_DIR, captured={"PATH": PATH_VALUE, "HF_HOME": value},
             home=HOME),
    ):
        with pytest.raises(ValueError):
            u.render_harness_unit(kwargs["interpreter"], kwargs["config_dir"], kwargs["captured"],
                                  home=kwargs["home"], version="0.19.0")


def test_the_refusal_names_the_offending_value():
    with pytest.raises(ValueError) as exc:
        render({"PATH": "/usr/bin:/opt/my tools"})
    assert "/opt/my tools" in str(exc.value)


def test_relative_paths_and_foreign_names_refuse():
    with pytest.raises(ValueError):
        u.render_harness_unit("python", CONFIG_DIR, {"PATH": PATH_VALUE}, home=HOME)
    with pytest.raises(ValueError):
        u.render_harness_unit(INTERPRETER, CONFIG_DIR, {"PATH": PATH_VALUE}, home="home/u")
    with pytest.raises(ValueError):
        render({"PATH": PATH_VALUE, "HOME": "/home/u"})
    with pytest.raises(ValueError):
        render(version="0.19 0")


def test_a_tilde_config_dir_is_pinned_absolute(tmp_path):
    body = u.render_harness_unit(INTERPRETER, "~/.localharness", {"PATH": PATH_VALUE}, home=HOME,
                                 version="0.19.0")
    assert f"--config-dir {tmp_path / 'home' / '.localharness'}\n" in body


# --------------------------------------------------------------------------- the path of the file


def test_the_unit_path_follows_xdg_config_home_read_at_call_time(tmp_path, monkeypatch):
    assert u.harness_unit_path() == tmp_path / "cfg" / "systemd" / "user" / "localharness-llm.service"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "other"))
    assert u.harness_unit_path().parent == tmp_path / "other" / "systemd" / "user"


@pytest.mark.parametrize("xdg", [None, "", "relative/dir"])
def test_without_a_usable_xdg_the_unit_path_is_under_home_dot_config(tmp_path, monkeypatch, xdg):
    if xdg is None:
        monkeypatch.delenv("XDG_CONFIG_HOME")
    else:
        monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
    assert u.harness_unit_path() == (
        tmp_path / "home" / ".config" / "systemd" / "user" / "localharness-llm.service")
    assert u.harness_unit_path().name == procfs.HARNESS_UNIT
    assert u.harness_home() == str(tmp_path / "home")


def test_write_then_read_the_unit_file():
    assert u.read_harness_unit() is None
    path = u.write_harness_unit(GOLDEN)
    assert path == u.harness_unit_path()
    assert path.read_text(encoding="utf-8") == GOLDEN
    assert (path.stat().st_mode & 0o777) == 0o600
    assert u.read_harness_unit() == GOLDEN
