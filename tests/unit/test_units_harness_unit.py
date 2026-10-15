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


# ------------------------------------------------------------------------- what `on` captures


def test_capture_drops_empty_and_relative_elements_and_names_them():
    got = u.capture_shell_env({"PATH": "./bin:/usr/bin::/bin", "LD_LIBRARY_PATH": "./lib:/opt/cuda/lib"}, HOME)
    assert got.env == {"PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/opt/cuda/lib"}
    assert got.dropped == (("PATH", "./bin"), ("PATH", ""), ("LD_LIBRARY_PATH", "./lib"))
    assert got.absolutized == ()
    assert "HF_HOME" not in got.env


def test_capture_takes_only_path_when_nothing_else_is_set():
    got = u.capture_shell_env({"PATH": "/usr/bin", "HOME": "/x", "PYTHONPATH": "/evil"}, HOME)
    assert got.env == {"PATH": "/usr/bin"}
    assert got.dropped == () and got.absolutized == ()


def test_capture_makes_a_relative_hf_home_absolute_against_the_shell_cwd_and_reports_it():
    got = u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": "models/hf"}, HOME)
    assert got.env["HF_HOME"] == "/home/u/models/hf"
    assert got.absolutized == (("HF_HOME", "models/hf", "/home/u/models/hf"),)
    again = u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": "../hf"}, "/home/u/work")
    assert again.env["HF_HOME"] == "/home/u/hf"


def test_capture_keeps_an_absolute_hf_home_as_written():
    got = u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": "/data/hf"}, HOME)
    assert got.env["HF_HOME"] == "/data/hf" and got.absolutized == ()


def test_capture_omits_an_ld_library_path_with_no_absolute_element_and_an_empty_one():
    got = u.capture_shell_env({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "./lib"}, HOME)
    assert "LD_LIBRARY_PATH" not in got.env and got.dropped == (("LD_LIBRARY_PATH", "./lib"),)
    quiet = u.capture_shell_env({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "", "HF_HOME": ""}, HOME)
    assert quiet.env == {"PATH": "/usr/bin"} and quiet.dropped == ()


def test_capture_refuses_a_kept_value_outside_the_charset_naming_it():
    with pytest.raises(ValueError) as exc:
        u.capture_shell_env({"PATH": "/usr/bin:/opt/my tools"}, "/")
    assert "/opt/my tools" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        u.capture_shell_env({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/a$b"}, "/")
    assert "/opt/a$b" in str(exc.value)


@pytest.mark.parametrize("value", ["/data/$HOME/hf", "/data/50%", "/data/a b"])
def test_capture_refuses_an_hf_home_outside_the_charset(value):
    with pytest.raises(ValueError):
        u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": value}, HOME)
    with pytest.raises(ValueError):  # a relative one is judged after it is made absolute
        u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": value.lstrip("/")}, HOME)


@pytest.mark.parametrize("environ", [{}, {"PATH": ""}, {"PATH": "./a:b"}, {"PATH": ":"}])
def test_capture_refuses_a_path_with_no_absolute_entry(environ):
    with pytest.raises(ValueError) as exc:
        u.capture_shell_env(environ, HOME)
    assert "PATH" in str(exc.value)


def test_capture_refuses_a_relative_cwd_for_a_relative_hf_home():
    with pytest.raises(ValueError):
        u.capture_shell_env({"PATH": "/usr/bin", "HF_HOME": "hf"}, "work")


def test_what_capture_returns_the_render_accepts_and_the_parse_returns():
    got = u.capture_shell_env({"PATH": "./b:/usr/bin", "HF_HOME": "hf", "LD_LIBRARY_PATH": "/l"}, HOME)
    assert u.captured_env(render(got.env)) == got.env


# ------------------------------------------------------------------------ what `serve` will read

UNPARSABLE = {
    "duplicate_path": lambda b: b.replace("# env HF_HOME=/home/u/hf\n", "# env PATH=/bin\n"),
    "unknown_name": lambda b: b.replace("# env HF_HOME=/home/u/hf\n", "# env HOME=/x\n"),
    "relative_element": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH=/usr/bin:relative"),
    "empty_element": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH=/usr/bin::/bin"),
    "trailing_empty_element": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH=/usr/bin:/bin:"),
    "relative_hf_home": lambda b: b.replace("HF_HOME=/home/u/hf", "HF_HOME=hf"),
    "relative_ld_library_path": lambda b: b.replace("HF_HOME=/home/u/hf", "LD_LIBRARY_PATH=/a:b"),
    "comment_in_the_block": lambda b: b.replace("# env HF_HOME=/home/u/hf\n", "# a comment\n"),
    "blank_line_in_the_block": lambda b: b.replace("# env HF_HOME=/home/u/hf\n", "\n"),
    "no_path_line": lambda b: b.replace("# env PATH=/usr/bin:/bin\n", ""),
    "empty_block": lambda b: b.replace("# env PATH=/usr/bin:/bin\n# env HF_HOME=/home/u/hf\n", ""),
    "no_unit_line": lambda b: b.replace("[Unit]\n", "[Unix]\n"),
    "not_the_header": lambda b: b.replace("# written by localharness", "# edited by someone"),
    "value_with_a_space": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH=/usr/bin:/my bin"),
    "empty_value": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH="),
    "no_equals_sign": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH /usr/bin"),
    "carriage_return": lambda b: b.replace("PATH=/usr/bin:/bin", "PATH=/usr/bin:/bin\r"),
    "lowercase_name": lambda b: b.replace("# env PATH=", "# env path="),
    "double_space": lambda b: b.replace("# env PATH=", "# env  PATH="),
}


@pytest.mark.parametrize("name", sorted(UNPARSABLE))
def test_captured_env_is_none_for_an_unparsable_block(name):
    assert u.captured_env(GOLDEN) == {"PATH": PATH_VALUE, "HF_HOME": HF}
    broken = UNPARSABLE[name](GOLDEN)
    assert broken != GOLDEN
    assert u.captured_env(broken) is None


def test_captured_env_none_for_a_body_that_is_not_text():
    assert u.captured_env(None) is None  # type: ignore[arg-type]
    assert u.captured_env("") is None


def test_captured_env_never_reads_an_env_line_after_the_unit_section():
    base = render({"PATH": PATH_VALUE})
    assert u.captured_env(base + "# env HF_HOME=/after\n") == {"PATH": PATH_VALUE}
    only_after = base.replace("# env PATH=/usr/bin:/bin\n", "") + "# env PATH=/usr/bin\n"
    assert u.captured_env(only_after) is None


def test_captured_env_returns_the_fixed_order_whatever_the_file_order():
    body = GOLDEN.replace("# env PATH=/usr/bin:/bin\n# env HF_HOME=/home/u/hf\n",
                          "# env LD_LIBRARY_PATH=/l\n# env HF_HOME=/home/u/hf\n# env PATH=/usr/bin\n")
    assert list(u.captured_env(body)) == ["PATH", "HF_HOME", "LD_LIBRARY_PATH"]


# ---------------------------------------------------------------------------------- unit_serves


def test_unit_serves_the_dir_the_body_was_rendered_with():
    body = render()
    assert u.unit_serves(CONFIG_DIR, body) is True
    assert u.unit_serves(CONFIG_DIR + "/", body) is True  # normalised
    assert u.unit_serves("/home/u/other", body) is False
    assert u.unit_serves("/home/u/.localharness/sub", body) is False


def test_unit_serves_expands_a_tilde_dir(tmp_path):
    body = u.render_harness_unit(INTERPRETER, "~/.localharness", {"PATH": PATH_VALUE}, home=HOME,
                                 version="0.19.0")
    assert u.unit_serves("~/.localharness", body) is True
    assert u.unit_serves(tmp_path / "home" / ".localharness", body) is True


def test_unit_serves_is_false_without_config_dir_on_the_exec_start():
    body = render().replace(" --config-dir /home/u/.localharness", "")
    assert u.unit_serves(CONFIG_DIR, body) is False
    twice = render().replace("serve --config-dir", "serve --config-dir /x --config-dir")
    assert u.unit_serves(CONFIG_DIR, twice) is False
    assert u.unit_serves(CONFIG_DIR, render().replace("ExecStart=", "ExecStart=\"")) is False  # unsplittable


def test_unit_serves_needs_one_exec_start_in_the_service_section():
    body = render()
    assert u.unit_serves(CONFIG_DIR, body.replace("[Install]", "ExecStart=/bin/true\n[Install]")) is False
    only_in_unit = body.replace("ExecStart=", "NotExec=").replace("[Unit]\n", "[Unit]\nExecStart=/x --config-dir "
                                                                  "/home/u/.localharness\n")
    assert u.unit_serves(CONFIG_DIR, only_in_unit) is False


def test_unit_serves_reads_the_file_when_no_body_is_given():
    assert u.unit_serves(CONFIG_DIR) is False  # no file
    u.write_harness_unit(render())
    assert u.unit_serves(CONFIG_DIR) is True
    assert u.unit_serves("/home/u/other") is False


# ------------------------------------------------------------- the hostile filesystem, all three verbs

NAME = procfs.HARNESS_UNIT


def _unit_dir(tmp_path):
    d = tmp_path / "cfg" / "systemd" / "user"
    d.mkdir(parents=True)
    return d


def test_a_symlinked_systemd_user_dir_is_never_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / NAME).write_text("victim\n")
    parent = tmp_path / "cfg" / "systemd"
    parent.mkdir(parents=True)
    (parent / "user").symlink_to(real)
    with pytest.raises(OSError):
        u.write_harness_unit(GOLDEN)
    assert (real / NAME).read_text() == "victim\n"
    assert sorted(p.name for p in real.iterdir()) == [NAME]
    assert u.read_harness_unit() is None
    assert u.remove_harness_unit() is False
    assert (real / NAME).exists()


def test_a_link_planted_at_the_unit_name_is_replaced_not_followed(tmp_path):
    d = _unit_dir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched\n")
    (d / NAME).symlink_to(elsewhere)
    assert u.read_harness_unit() is None
    u.write_harness_unit(GOLDEN)
    assert not (d / NAME).is_symlink()
    assert u.read_harness_unit() == GOLDEN
    assert elsewhere.read_text() == "untouched\n"


def test_remove_unlinks_a_planted_link_itself_not_its_target(tmp_path):
    d = _unit_dir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched\n")
    (d / NAME).symlink_to(elsewhere)
    assert u.remove_harness_unit() is True
    assert not os.path.lexists(d / NAME)
    assert elsewhere.read_text() == "untouched\n"


def test_remove_reports_true_once_then_false(tmp_path):
    assert u.remove_harness_unit() is False  # no directory at all
    u.write_harness_unit(GOLDEN)
    assert u.remove_harness_unit() is True
    assert u.remove_harness_unit() is False
    assert u.read_harness_unit() is None
    assert (tmp_path / "cfg" / "systemd" / "user").is_dir()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_a_fifo_at_the_unit_name_reads_none_without_blocking(tmp_path):
    d = _unit_dir(tmp_path)
    os.mkfifo(d / NAME)
    result = []
    t = threading.Thread(target=lambda: result.append(u.read_harness_unit()), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "reading a FIFO blocked"
    assert result == [None]


def test_a_unit_file_over_the_cap_reads_none(tmp_path):
    d = _unit_dir(tmp_path)
    (d / NAME).write_bytes(b"#" * (u.UNIT_FILE_CAP + 1))
    assert u.read_harness_unit() is None
    (d / NAME).write_bytes(b"#" * u.UNIT_FILE_CAP)
    assert u.read_harness_unit() == "#" * u.UNIT_FILE_CAP


def test_a_unit_file_that_is_not_utf8_reads_none(tmp_path):
    d = _unit_dir(tmp_path)
    (d / NAME).write_bytes(b"[Unit]\n\xff\xfe\n")
    assert u.read_harness_unit() is None


def test_a_failed_write_keeps_the_old_unit_file(monkeypatch):
    u.write_harness_unit("old\n")

    def boom(*a, **kw):
        raise OSError("simulated")
    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", boom)
        with pytest.raises(OSError):
            u.write_harness_unit("new\n")
    assert u.read_harness_unit() == "old\n"
    assert [p.name for p in u.harness_unit_path().parent.iterdir()] == [procfs.HARNESS_UNIT]
