"""How the model's shell gate classifies the localharness skills verb."""
from __future__ import annotations

import pytest

from localharness.agent.gate_types import Verdict
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import evaluate
from tests.unit.agent.test_verdict import SETTINGS, SHELL_META, granting, make_ctx

MODES = ("guarded", "trusted", "auto", "read-only", "unattended")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def test_localharness_skills_is_one_read_only_segment():
    shell = classify_shell("localharness skills", SETTINGS)
    assert shell.signatures == ("localharness skills",)
    assert all(s.read_only for s in shell.segments)


@pytest.mark.parametrize("mode", MODES)
def test_localharness_skills_asks_nothing_in_any_mode(ws, mode):
    result = evaluate("bash_exec", {"command": "localharness skills"}, SHELL_META, make_ctx(ws, mode=mode), SETTINGS)
    assert result.verdict is Verdict.ALLOW
    assert result.request is None


def test_a_compound_behind_localharness_skills_is_not_read_only(ws):
    shell = classify_shell("localharness skills; rm x", SETTINGS)
    assert len(shell.segments) == 2
    assert shell.signatures[0] == "localharness skills"
    assert not all(s.read_only for s in shell.segments)
    result = evaluate("bash_exec", {"command": "localharness skills; rm x"}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK


def test_a_config_dir_before_the_subcommand_asks(ws):
    command = "localharness --config-dir X skills"
    shell = classify_shell(command, SETTINGS)
    assert shell.signatures == ("localharness --config-dir X",)
    assert not shell.segments[0].read_only
    result = evaluate("bash_exec", {"command": command}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert (result.request.klass, result.request.key) == ("shell-unfamiliar", "localharness --config-dir X")


@pytest.mark.parametrize("flags", ["--install-completion", "--show-completion", "-h", "--config-dir=X"])
def test_a_flag_before_skills_is_not_the_read_only_verb(flags):
    shell = classify_shell(f"localharness {flags} skills", SETTINGS)
    assert shell.signatures == (f"localharness {flags} skills",)
    assert not shell.segments[0].read_only


@pytest.mark.parametrize(
    "command, key",
    [
        ("localharness --install-completion", "localharness --install-completion"),
        ("localharness --help", "localharness --help"),
        ("localharness -v serve", "localharness -v serve"),
        ("localharness --config-dir X update", "localharness --config-dir X"),
    ],
)
def test_a_flag_led_line_has_its_own_key(command, key):
    shell = classify_shell(command, SETTINGS)
    assert shell.signatures == (key,)
    assert not shell.segments[0].read_only


@pytest.mark.parametrize(
    "command", ["localharness --config-dir X update", "localharness -v serve", "localharness --install-completion"]
)
def test_a_bare_localharness_grant_covers_no_flag_led_line(ws, command):
    ctx = make_ctx(ws, grants=granting(("shell-unfamiliar", "localharness")))
    assert evaluate("bash_exec", {"command": command}, SHELL_META, ctx, SETTINGS).verdict is Verdict.ASK


@pytest.mark.parametrize("command", ["localharness --show-completion", "localharness --config-dir X update"])
def test_a_flag_led_grant_covers_no_other_flag_led_line(ws, command):
    ctx = make_ctx(ws, grants=granting(("shell-unfamiliar", "localharness --help")))
    assert evaluate("bash_exec", {"command": command}, SHELL_META, ctx, SETTINGS).verdict is Verdict.ASK
    same = evaluate("bash_exec", {"command": "localharness --help"}, SHELL_META, ctx, SETTINGS)
    assert same.verdict is Verdict.ALLOW


def test_install_completion_before_skills_asks(ws):
    command = "localharness --install-completion skills"
    result = evaluate("bash_exec", {"command": command}, SHELL_META, make_ctx(ws), SETTINGS)
    assert result.verdict is Verdict.ASK
    assert result.request.key == "localharness --install-completion skills"


def test_a_flag_after_skills_stays_read_only():
    assert classify_shell("localharness skills --config-dir X", SETTINGS).segments[0].read_only


def test_a_bare_localharness_grant_no_longer_covers_a_subcommand(ws):
    ctx = make_ctx(ws, grants=granting(("shell-unfamiliar", "localharness")))
    doctor = evaluate("bash_exec", {"command": "localharness doctor"}, SHELL_META, ctx, SETTINGS)
    assert doctor.verdict is Verdict.ASK
    assert doctor.request.key == "localharness doctor"
    bare = classify_shell("localharness", SETTINGS)
    assert bare.signatures == ("localharness",)
    assert not bare.segments[0].read_only
    assert evaluate("bash_exec", {"command": "localharness"}, SHELL_META, ctx, SETTINGS).verdict is Verdict.ALLOW


def test_a_subcommand_grant_covers_no_other_subcommand(ws):
    ctx = make_ctx(ws, grants=granting(("shell-unfamiliar", "localharness doctor")))
    start = evaluate("bash_exec", {"command": "localharness start"}, SHELL_META, ctx, SETTINGS)
    assert start.verdict is Verdict.ASK
    assert start.request.key == "localharness start"


@pytest.mark.parametrize("sub", ["start", "model", "init", "plugins", "components", "doctor", "update", "serve", "memory"])
def test_no_other_localharness_subcommand_is_read_only(sub):
    shell = classify_shell(f"localharness {sub}", SETTINGS)
    assert shell.signatures == (f"localharness {sub}",)
    assert not shell.segments[0].read_only


def test_a_redirect_from_localharness_skills_is_a_write(ws):
    command = "localharness skills > out.txt"
    shell = classify_shell(command, SETTINGS)
    assert shell.segments[0].write_targets
    result = evaluate("bash_exec", {"command": command}, SHELL_META, make_ctx(ws, mode="read-only"), SETTINGS)
    assert result.verdict is Verdict.DENY
