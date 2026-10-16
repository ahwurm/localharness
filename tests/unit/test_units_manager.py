"""What the harness asks of the user manager besides `show` (spec 20 §9.2, §12, D-09, D-10, D-11, D-20):
the closed verb set, the scrubbed journal tail, the linger stat, the second-launcher scan and the exit
translation. Every call goes through `fake_manager`; nothing here reaches a real service manager."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from localharness.provider import units as u
from tests.unit.units_support import fake_manager  # noqa: F401

UNIT = "localharness-llm.service"
VERBS = ("daemon-reload", "enable", "disable", "reset-failed", "restart", "stop", "start")


# ------------------------------------------------------------------------------------- the verbs


def test_verb_table_is_exactly_the_seven_verbs():
    assert set(u.VERB_TIMEOUTS) == set(VERBS)
    assert all(isinstance(t, (int, float)) and t > 0 for t in u.VERB_TIMEOUTS.values())


def test_verb_table_cannot_be_edited():
    with pytest.raises(TypeError):
        u.VERB_TIMEOUTS["link"] = 1  # type: ignore[index]


def test_verb_daemon_reload_is_one_argv_with_its_own_timeout(fake_manager):
    fake_manager.answer(["systemctl", "--user", "daemon-reload"], "")
    result = u.systemctl_user("daemon-reload")
    assert result == u.ManagerResult(0, "", "")
    assert fake_manager.calls == [("systemctl", "--user", "daemon-reload")]
    assert fake_manager.limits == [(u.VERB_TIMEOUTS["daemon-reload"], u.SYSTEMCTL_OUTPUT_CAP)]


def test_verb_enable_carries_its_arguments_in_order(fake_manager):
    u.systemctl_user("enable", "--now", UNIT)
    assert fake_manager.calls == [("systemctl", "--user", "enable", "--now", UNIT)]


@pytest.mark.parametrize("verb", ["restart", "stop", "disable"])
def test_verb_that_waits_for_a_server_to_exit_allows_90_seconds_or_more(verb):
    assert u.VERB_TIMEOUTS[verb] >= 90


@pytest.mark.parametrize("verb", VERBS)
def test_verb_failure_is_an_answer_and_a_missing_manager_is_none(fake_manager, verb):
    fake_manager.answer_prefix("systemctl", "--user", verb, out="", code=1, err="Failed")
    assert u.systemctl_user(verb, UNIT) == u.ManagerResult(1, "", "Failed")
    fake_manager.answer_prefix("systemctl", "--user", verb, out=None)
    assert u.systemctl_user(verb, UNIT) is None


@pytest.mark.parametrize("verb", [
    "set-environment", "import-environment", "unset-environment", "link", "edit", "set-property",
    "add-wants", "add-requires", "mask", "kill", "show", "is-active", "revert", "", "RESTART",
    "restart --now", "--user", "daemon-reexec",
])
def test_verb_outside_the_closed_set_is_refused_before_anything_runs(fake_manager, verb):
    with pytest.raises(ValueError):
        u.systemctl_user(verb, "X=1")
    assert fake_manager.calls == []


def test_refuse_a_non_string_verb(fake_manager):
    with pytest.raises(ValueError):
        u.systemctl_user(None)  # type: ignore[arg-type]
    assert fake_manager.calls == []


# ----------------------------------------------------------------------------------- the journal


def test_journal_tail_asks_by_the_trusted_fields_only(fake_manager):
    fake_manager.answer_prefix("journalctl", out="line one\nline two\n")
    assert u.journal_tail(invocation="abc") == "line one\nline two"
    (argv,) = fake_manager.calls
    assert argv[:7] == ("journalctl", "--user", "-o", "cat", "-n", "15", "--no-pager")
    assert f"_SYSTEMD_USER_UNIT={UNIT}" in argv
    assert f"_UID={os.getuid()}" in argv
    assert "_SYSTEMD_INVOCATION_ID=abc" in argv
    assert "-u" not in argv and not any(a.startswith("USER_UNIT=") for a in argv)
    assert fake_manager.limits == [(u.JOURNAL_TIMEOUT_S, u.JOURNAL_OUTPUT_CAP)]


def test_journal_tail_without_an_invocation_has_no_invocation_field(fake_manager):
    fake_manager.answer_prefix("journalctl", out="x\n")
    u.journal_tail(lines=5)
    (argv,) = fake_manager.calls
    assert "5" in argv and not any(a.startswith("_SYSTEMD_INVOCATION_ID") for a in argv)


@pytest.mark.parametrize("bad", ["", "a b", "a=b", "x" * 65, "abc\n", "a/b"])
def test_journal_tail_refuses_an_invocation_that_is_not_an_id(fake_manager, bad):
    with pytest.raises(ValueError):
        u.journal_tail(invocation=bad)
    assert fake_manager.calls == []


def test_journal_scrub_masks_every_secret_form(fake_manager):
    fake_manager.answer_prefix("journalctl", out=(
        "loading with sk-planted\n"
        "Authorization: Bearer zzz\n"
        "api_keys: ****9876\n"
        "listening on http://u:p@127.0.0.1:8080/v1\n"))
    tail = u.journal_tail(invocation="abc", secrets=["sk-planted"])
    for leaked in ("sk-planted", "zzz", "9876", "u:p"):
        assert leaked not in tail
    assert "127.0.0.1:8080" in tail and "Bearer" in tail and "api_keys" in tail


def test_journal_scrub_ignores_empty_and_non_string_secrets(fake_manager):
    fake_manager.answer_prefix("journalctl", out="plain line\n")
    assert u.journal_tail(secrets=["", None, 7]) == "plain line"


def test_journal_scrub_masks_a_secret_wherever_it_sits(fake_manager):
    fake_manager.answer_prefix("journalctl", out="a sk-one b sk-one c sk-two\n")
    tail = u.journal_tail(secrets=("sk-one", "sk-two"))
    assert "sk-one" not in tail and "sk-two" not in tail and tail.startswith("a ")


@pytest.mark.parametrize("answer", [
    dict(out=None), dict(out="x", code=1), dict(out=""), dict(out="  \n"),
])
def test_journal_tail_without_a_journal_says_so(fake_manager, answer):
    fake_manager.answer_prefix("journalctl", **answer)
    assert u.journal_tail() == "(no journal)"


def test_journal_tail_with_no_manager_at_all_says_so():
    assert u.journal_tail(invocation="abc") == "(no journal)"


def test_journal_reference_line_names_where_the_raw_lines_are():
    line = u.JOURNAL_REF_LINE.format(invocation="abc")
    assert "journalctl --user -u localharness-llm.service" in line and "abc" in line


# ------------------------------------------------------------------------------------- linger


def test_linger_is_the_existence_of_the_users_file(tmp_path, monkeypatch):
    monkeypatch.setattr(u, "LINGER_DIR", str(tmp_path))
    assert u.linger_on("awurm") is False
    (tmp_path / "awurm").write_text("")
    assert u.linger_on("awurm") is True
    assert u.linger_on("other") is False


def test_linger_defaults_to_the_current_user(tmp_path, monkeypatch):
    monkeypatch.setattr(u, "LINGER_DIR", str(tmp_path))
    monkeypatch.setattr(u, "current_user", lambda: "me")
    assert u.linger_on() is False
    (tmp_path / "me").write_text("")
    assert u.linger_on() is True


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/b", "a\x00b"])
def test_linger_for_a_name_that_is_not_a_user_is_false(tmp_path, monkeypatch, name):
    monkeypatch.setattr(u, "LINGER_DIR", str(tmp_path / "linger"))
    (tmp_path / "linger").mkdir()
    (tmp_path / "x").write_text("")
    assert u.linger_on(name) is False


def test_linger_never_spawns(tmp_path, monkeypatch, fake_manager):
    monkeypatch.setattr(u, "LINGER_DIR", str(tmp_path))
    u.linger_on("awurm")
    assert fake_manager.calls == []


def test_current_user_is_the_password_database_name_or_a_fallback(monkeypatch):
    assert isinstance(u.current_user(), str) and u.current_user()
    import pwd

    def missing(uid):
        raise KeyError(uid)

    monkeypatch.setattr(pwd, "getpwuid", missing)
    monkeypatch.setenv("USER", "fromenv")
    assert u.current_user() == "fromenv"
    monkeypatch.delenv("USER")
    monkeypatch.setenv("LOGNAME", "fromlogname")
    assert u.current_user() == "fromlogname"
    monkeypatch.delenv("LOGNAME")
    assert u.current_user() == str(os.getuid())


# ---------------------------------------------------------------------------- other launchers

LIST_ARGV = ("systemctl", "--user", "list-unit-files", "--type=service", "--plain", "--no-legend",
             "--no-pager")
SHOW_PROPS = "Id,FragmentPath,UnitFileState,ActiveState"


def unit_file(directory: Path, name: str, exec_line: str, *, extra: str = "") -> str:
    path = directory / name
    path.write_text(f"[Service]\nExecStart={exec_line}\n{extra}")
    return str(path)


def show_block(name: str, fragment: str, file_state: str = "enabled", active: str = "active") -> str:
    return f"Id={name}\nFragmentPath={fragment}\nUnitFileState={file_state}\nActiveState={active}\n"


def answer_scan(fake_manager, rows, blocks):
    names = sorted(r.split()[0] for r in rows if "@" not in r.split()[0] and r.split()[1] not in
                   u._SKIPPED_FILE_STATES and r.split()[0] != UNIT)
    fake_manager.answer(LIST_ARGV, "".join(r + "\n" for r in rows))
    fake_manager.answer(("systemctl", "--user", "show", "-p", SHOW_PROPS, "--", *names), "\n".join(blocks))


def test_other_launchers_names_the_enabled_unit_on_the_port(fake_manager, tmp_path):
    llm = unit_file(tmp_path, "llm.service", "/opt/llama-server -m /m.gguf --port 8080")
    idle = unit_file(tmp_path, "idle.service", "/opt/llama-server -m /m.gguf --port 8080")
    own = unit_file(tmp_path, UNIT, "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(fake_manager,
                ["llm.service enabled", "idle.service disabled", f"{UNIT} enabled", "tpl@.service enabled"],
                [show_block("llm.service", llm), show_block("idle.service", idle, "disabled", "inactive")])
    assert own  # the harness's own file is on disk and still not named
    assert u.other_launchers({8080}) == [u.Launcher("llm.service", "enabled", 8080)]
    shown = [c for c in fake_manager.calls if c[:3] == ("systemctl", "--user", "show")]
    assert shown == [("systemctl", "--user", "show", "-p", SHOW_PROPS, "--", "idle.service", "llm.service")]


def test_other_launchers_leaves_out_a_unit_on_another_port(fake_manager, tmp_path):
    other = unit_file(tmp_path, "other.service", "/opt/llama-server -m /m.gguf --port 9000")
    answer_scan(fake_manager, ["other.service enabled"], [show_block("other.service", other)])
    assert u.other_launchers({8080}) == []
    assert u.other_launchers({8080, 9000}) == [u.Launcher("other.service", "enabled", 9000)]


def test_other_launchers_names_an_active_unit_that_is_not_enabled(fake_manager, tmp_path):
    run = unit_file(tmp_path, "run.service", "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(fake_manager, ["run.service disabled"], [show_block("run.service", run, "disabled", "active")])
    assert u.other_launchers({8080}) == [u.Launcher("run.service", "active", 8080)]


def test_other_launchers_leaves_out_a_unit_that_is_neither_enabled_nor_active(fake_manager, tmp_path):
    off = unit_file(tmp_path, "off.service", "/opt/llama-server -m /m.gguf --port 8080")
    answer_scan(fake_manager, ["off.service disabled"], [show_block("off.service", off, "disabled", "inactive")])
    assert u.other_launchers({8080}) == []


def test_other_launchers_counts_a_unit_without_a_port_as_8080(fake_manager, tmp_path):
    bare = unit_file(tmp_path, "bare.service", "/opt/llama-server -m /m.gguf")
    answer_scan(fake_manager, ["bare.service enabled"], [show_block("bare.service", bare)])
    assert u.other_launchers({8080}) == [u.Launcher("bare.service", "enabled", 8080)]
    assert u.other_launchers({8081}) == []


def test_other_launchers_reads_the_port_from_the_units_environment(fake_manager, tmp_path):
    env = unit_file(tmp_path, "env.service", "/opt/llama-server -m /m.gguf",
                    extra="Environment=LLAMA_ARG_PORT=9100\n")
    answer_scan(fake_manager, ["env.service enabled-runtime"], [show_block("env.service", env, "enabled-runtime")])
    assert u.other_launchers({9100}) == [u.Launcher("env.service", "enabled", 9100)]


def test_other_launchers_leaves_out_a_unit_that_is_not_a_llama_server(fake_manager, tmp_path):
    vllm = unit_file(tmp_path, "vllm.service", "/opt/vllm serve m --port 8080")
    answer_scan(fake_manager, ["vllm.service enabled"], [show_block("vllm.service", vllm)])
    assert u.other_launchers({8080}) == []


def test_other_launchers_leaves_out_a_unit_whose_file_cannot_be_read(fake_manager, tmp_path):
    answer_scan(fake_manager, ["gone.service enabled", "none.service enabled"],
                [show_block("gone.service", str(tmp_path / "missing")), show_block("none.service", "")])
    assert u.other_launchers({8080}) == []


def test_other_launchers_skips_what_user_units_skips(fake_manager, tmp_path):
    llm = unit_file(tmp_path, "llm.service", "/opt/llama-server --port 8080")
    answer_scan(fake_manager,
                ["llm.service enabled", "mask.service masked", "gen.service generated", "tpl@.service enabled"],
                [show_block("llm.service", llm)])
    assert [l.unit for l in u.other_launchers({8080})] == ["llm.service"]


def test_other_launchers_ignores_a_show_block_for_a_unit_it_did_not_ask_about(fake_manager, tmp_path):
    rogue = unit_file(tmp_path, "rogue.service", "/opt/llama-server --port 8080")
    answer_scan(fake_manager, ["llm.service enabled"], [show_block("rogue.service", rogue)])
    assert u.other_launchers({8080}) == []


@pytest.mark.parametrize("failure", [dict(out=None), dict(out="", code=1)])
def test_other_launchers_is_empty_when_the_manager_cannot_be_asked(fake_manager, tmp_path, failure):
    fake_manager.answer(LIST_ARGV, **failure)
    assert u.other_launchers({8080}) == []
    llm = unit_file(tmp_path, "llm.service", "/opt/llama-server --port 8080")
    fake_manager.answer(LIST_ARGV, "llm.service enabled\n")
    fake_manager.answer_prefix("systemctl", "--user", "show", **failure)
    assert u.other_launchers({8080}) == []


def test_other_launchers_with_no_ports_asks_nothing(fake_manager):
    assert u.other_launchers(set()) == []
    assert u.other_launchers([None, "8080", True]) == []
    assert fake_manager.calls == []


def test_other_launchers_scans_at_most_the_first_64_units(fake_manager):
    rows = [f"u{i:03d}.service enabled" for i in range(80)]
    fake_manager.answer(LIST_ARGV, "".join(r + "\n" for r in rows))
    u.other_launchers({8080})
    shown = [c for c in fake_manager.calls if c[:3] == ("systemctl", "--user", "show")]
    assert len(shown) == 1 and len(shown[0]) - 6 == u.UNITS_SCANNED_CAP
    assert shown[0][6] == "u000.service" and shown[0][-1] == "u063.service"


# --------------------------------------------------------------------------------- the exit


def state(**over) -> u.UnitRuntimeState:
    kw = dict(active_state="inactive", exec_main_status=0, result="success")
    kw.update(over)
    return u.UnitRuntimeState("abc", 0, **kw)


def test_translate_78_says_the_record_is_withheld():
    line = u.translate_exit(state(exec_main_status=78), record_name="flash-next")
    assert line == "stopped: the machine record withholds flash-next — run localharness start in a terminal"


def test_translate_75_says_a_live_pidfile_held_the_launch():
    line = u.translate_exit(state(exec_main_status=75), record_name="x")
    assert line == u.STOPPED_LIVE_PIDFILE_LINE and "pidfile" in line and "live" in line


def test_translate_start_limit_hit_says_three_failed_loads():
    line = u.translate_exit(state(exec_main_status=1, result="start-limit-hit", active_state="failed"),
                            record_name="x")
    assert line == "stopped after 3 failed loads — journalctl --user -u localharness-llm.service"


def test_translate_75_wins_over_start_limit_hit():
    line = u.translate_exit(state(exec_main_status=75, result="start-limit-hit"), record_name="x")
    assert line == u.STOPPED_LIVE_PIDFILE_LINE


def test_translate_78_wins_over_everything():
    line = u.translate_exit(state(exec_main_status=78, result="start-limit-hit"), record_name="x",
                            not_served="qwen-docker is docker — not served by the unit")
    assert line == u.STOPPED_WITHHELD_LINE.format(record="x")


def test_translate_an_inactive_unit_that_cannot_serve_its_target_says_so():
    said = "qwen-docker is docker — not served by the unit"
    assert u.translate_exit(state(), record_name="x", not_served=said) == said


def test_translate_not_served_applies_to_an_inactive_unit_only():
    said = "qwen-docker is docker — not served by the unit"
    assert u.translate_exit(state(active_state="active", main_pid=4242), record_name="x",
                            not_served=said) is None


def test_translate_an_active_healthy_unit_has_nothing_to_say():
    assert u.translate_exit(state(active_state="active", main_pid=4242), record_name="x") is None


def test_translate_an_inactive_unit_with_nothing_wrong_has_nothing_to_say():
    assert u.translate_exit(state(), record_name="x") is None


def test_translate_no_state_is_none():
    assert u.translate_exit(None, record_name="x", not_served="y") is None


def test_translate_a_failed_unit_with_an_ordinary_exit_has_nothing_to_say():
    assert u.translate_exit(state(active_state="failed", exec_main_status=1, result="exit-code"),
                            record_name="x") is None
