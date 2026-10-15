"""'The harness's own unit' (spec 20 §9.2): FragmentPath, then DropInPaths, then the byte-equal body.

Three tests in that order, each failing alone and each naming itself. The state is a plain value here; the
live one comes from the service manager in a later plan, and nothing in this module asks a manager
anything. `XDG_CONFIG_HOME` and `HOME` are redirected to tmp for every test."""
from __future__ import annotations

import sys

import pytest

from localharness.provider import units as u

INTERPRETER = "/opt/lh/bin/python"
CONFIG_DIR = "/home/u/.localharness"
HOME = "/home/u"
CAPTURED = {"PATH": "/usr/bin:/bin", "HF_HOME": "/home/u/hf"}
DIFFERS = "differs from what localharness writes"


@pytest.fixture(autouse=True)
def _redirected(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def body(captured=None, *, interpreter=INTERPRETER, config_dir=CONFIG_DIR, home=HOME, version="0.19.0"):
    return u.render_harness_unit(interpreter, config_dir, CAPTURED if captured is None else captured,
                                 home=home, version=version)


def state(**over):
    kw = dict(fragment_path=str(u.harness_unit_path()), drop_in_paths="")
    kw.update(over)
    return u.UnitRuntimeState("inv", 0, **kw)


def failure(st=None, text=None, *, write=True, **kw):
    """The failure phrase for a unit file holding `text` (the current render by default)."""
    text = body() if text is None else text
    if write:
        u.write_harness_unit(text)
    return u.harness_unit_failure(state() if st is None else st, CONFIG_DIR, interpreter=INTERPRETER,
                                  home=HOME, **kw)


def test_the_unit_the_harness_wrote_is_its_own():
    u.write_harness_unit(body())
    assert u.harness_unit_failure(state(), CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None
    assert u.is_harness_unit(state(), CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is True


def test_a_body_passed_in_is_judged_without_reading_the_file():
    assert u.harness_unit_failure(state(), CONFIG_DIR, body=body(), interpreter=INTERPRETER, home=HOME) is None
    assert u.read_harness_unit() is None


def test_the_defaults_are_the_running_interpreter_and_the_home(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", INTERPRETER)
    text = body(home=str(tmp_path / "home"))
    u.write_harness_unit(text)
    assert u.harness_unit_failure(state(), CONFIG_DIR) is None
    monkeypatch.setattr(sys, "executable", "/opt/other/python")
    assert u.harness_unit_failure(state(), CONFIG_DIR) == DIFFERS


def test_a_state_of_the_init_shape_is_judged_the_same():
    u.write_harness_unit(body())
    shown = u.UnitState("localharness-llm.service", "loaded", str(u.harness_unit_path()), "enabled", "")
    assert u.harness_unit_failure(shown, CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is None


def test_fragment_path_of_another_file_fails_alone_and_names_both_paths():
    phrase = failure(state(fragment_path="/etc/systemd/user/localharness-llm.service"))
    assert "/etc/systemd/user/localharness-llm.service" in phrase
    assert str(u.harness_unit_path()) in phrase and "FragmentPath" in phrase


def test_an_empty_fragment_path_says_the_unit_is_not_loaded_from_the_harness_path():
    phrase = failure(state(fragment_path=""))
    assert phrase.startswith("is not loaded from ") and str(u.harness_unit_path()) in phrase


def test_a_drop_in_fails_alone_and_is_named():
    phrase = failure(state(drop_in_paths="/x/override.conf"))
    assert "/x/override.conf" in phrase and "drop-in" in phrase


def test_one_changed_byte_in_an_executable_line_fails_the_body_test_alone():
    for old, new in (("RestartSec=5", "RestartSec=6"), ("--autostart off", "--autostart on"),
                     ("Restart=on-failure", "Restart=always"), (" -I ", " "), ("serve --config", "serve  --config")):
        assert old in body()
        assert failure(text=body().replace(old, new, 1)) == DIFFERS, old


def test_an_added_line_or_a_dropped_trailing_newline_fails_the_body_test():
    assert failure(text=body() + "ExecStartPost=/bin/true\n") == DIFFERS
    assert failure(text=body().rstrip("\n")) == DIFFERS
    assert failure(text=body().replace("[Install]", "ExecStartPre=/bin/true\n\n[Install]")) == DIFFERS


def test_a_unit_rendered_for_another_interpreter_dir_or_home_is_foreign():
    assert failure(text=body(interpreter="/usr/bin/python3")) == DIFFERS
    assert failure(text=body(config_dir="/home/u/other")) == DIFFERS
    assert failure(text=body(home="/root")) == DIFFERS


def test_an_unparsable_env_block_says_so():
    broken = body().replace("PATH=/usr/bin:/bin", "PATH=/usr/bin:relative")
    assert "unparsable" in failure(text=broken)
    assert "unparsable" in failure(text=body().replace("# env PATH=/usr/bin:/bin\n", ""))


def test_an_absent_or_unreadable_file_says_so():
    assert "no readable unit file" in failure(write=False)
    d = u.harness_unit_path().parent
    d.mkdir(parents=True)
    (d / u.harness_unit_path().name).write_bytes(b"#" * (u.UNIT_FILE_CAP + 1))
    assert "no readable unit file" in failure(write=False)


def test_a_state_of_none_says_the_service_manager_could_not_be_asked():
    u.write_harness_unit(body())
    phrase = u.harness_unit_failure(None, CONFIG_DIR, interpreter=INTERPRETER, home=HOME)
    assert "service manager" in phrase
    assert u.is_harness_unit(None, CONFIG_DIR, interpreter=INTERPRETER, home=HOME) is False


def test_the_first_failed_test_in_order_is_the_one_named():
    foreign_body = body().replace("RestartSec=5", "RestartSec=6")
    # all three fail: FragmentPath is named
    phrase = failure(state(fragment_path="/other", drop_in_paths="/x/o.conf"), foreign_body)
    assert "FragmentPath" in phrase and "/x/o.conf" not in phrase
    # the drop-in and the body fail: the drop-in is named
    phrase = failure(state(drop_in_paths="/x/o.conf"), foreign_body)
    assert "/x/o.conf" in phrase and phrase != DIFFERS
    # FragmentPath fails and there is no file at all: FragmentPath, not the missing file
    assert "FragmentPath" in failure(state(fragment_path="/other"), write=False)
    # the drop-in fails and the file is absent: the drop-in
    assert "/x/o.conf" in failure(state(drop_in_paths="/x/o.conf"), write=False)


def test_a_header_with_another_version_is_still_the_harness_own_unit():
    for version in ("0.19.1", "1.0.0rc1", "0.19.0+local"):
        assert failure(text=body(version=version)) is None, version


@pytest.mark.parametrize("header", [
    "# written by localharness 0.19 0 — `localharness model --autostart off` removes it",
    "# written by localharness " + "9" * 65 + " — `localharness model --autostart off` removes it",
    "# written by localharness 0.19.0 - `localharness model --autostart off` removes it",
    "# written by localharness  — `localharness model --autostart off` removes it",
    "# written by localharness 0.19.0 — `localharness model --autostart off` removes it, and more",
    "# written by localharness 0.19;0 — `localharness model --autostart off` removes it",
])
def test_a_header_off_the_template_fails_the_body_test(header):
    first, _, rest = body().partition("\n")
    assert failure(text=header + "\n" + rest) == DIFFERS


def test_a_planted_env_line_changes_the_verdict_only_through_the_block_it_is_in():
    # an `# env` line after [Unit] is a comment the parse never reads; the byte comparison still sees it
    assert failure(text=body() + "# env HF_HOME=/evil\n") == DIFFERS
    # the block the render wrote is the one the comparison trusts: a different PATH in it is the unit's own
    assert failure(text=body({"PATH": "/opt/other/bin"})) is None
