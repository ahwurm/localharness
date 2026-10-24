"""How the model's shell gate classifies the localharness skills verb."""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.agent.gate_types import Verdict
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import evaluate
from localharness.cli.app import app
from localharness.config import skills as skills_config
from tests.unit.agent.test_verdict import SETTINGS, SHELL_META, granting, make_ctx
from tests.unit.test_skills_discovery import _skill

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


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    monkeypatch.setattr(skills_config, "PACKAGE_DIR", tmp_path / "pkg")
    return tmp_path / "pkg"


def test_the_verb_lists_a_machine_skill_with_its_origin_and_quoted_path(fake_home, pkg, monkeypatch):
    home = fake_home()
    path = _skill(home / ".localharness" / "skills", "notes")
    (home / "proj").mkdir()
    monkeypatch.chdir(home / "proj")
    result = CliRunner().invoke(app, ["skills"])
    assert result.exit_code == 0
    assert result.output == f"notes (machine, {str(path)!r})\n"


def test_the_verb_takes_config_dir_after_the_word(fake_home, pkg, monkeypatch, tmp_path):
    home = fake_home()
    _skill(home / ".localharness" / "skills", "notes")
    other = tmp_path / "other"
    path = _skill(other / "skills", "beta")
    (home / "proj").mkdir()
    monkeypatch.chdir(home / "proj")
    result = CliRunner().invoke(app, ["skills", "--config-dir", str(other)])
    assert result.exit_code == 0
    assert result.output == f"beta (machine, {str(path)!r})\n"


def _run(*args):
    result = CliRunner().invoke(app, ["skills", *args])
    assert result.exit_code == 0
    return result.output.splitlines()


def _project(home, monkeypatch):
    proj = home / "proj"
    (proj / ".localharness").mkdir(parents=True)
    monkeypatch.chdir(proj)
    return proj


def _sized(origin_dir, name, length, char="d"):
    return _skill(origin_dir, name, f"---\nname: {name}\ndescription: {char * length}\n---\nBody.\n")


@pytest.fixture
def rows_clear():
    from localharness.cli.slash_commands import set_plugin_rows

    set_plugin_rows(())
    yield
    set_plugin_rows(())


def test_the_verb_lists_in_precedence_then_name_and_says_what_each_shadow_lost_to(fake_home, pkg, monkeypatch):
    home = fake_home()
    proj = _project(home, monkeypatch)
    machine = home / ".localharness" / "skills"
    first = {n: _skill(proj / ".localharness" / "skills", n) for n in ("b", "a")}
    second = {n: _skill(proj / ".agents" / "skills", n) for n in ("a", "c")}
    mine = {n: _skill(machine, n) for n in ("d", "a")}
    bundled = _skill(pkg / "assets" / "skills", "e")
    paths = {"a": first["a"], "b": first["b"], "c": second["c"], "d": mine["d"], "e": bundled}
    origins = {"a": "project", "b": "project", "c": "project", "d": "machine", "e": "bundled"}
    shadow = "  {!r} is shadowed by {!r}. rename yours to keep both."
    assert _run() == [
        *(f"{n} ({origins[n]}, {str(paths[n])!r})" for n in "abcde"),
        "Skipped:",
        shadow.format(str(second["a"].parent), str(first["a"].parent)),
        shadow.format(str(mine["a"].parent), str(first["a"].parent)),
    ]


def test_the_verb_says_each_skip_with_its_reason_and_fix(fake_home, pkg):
    from localharness.config.skills import SkillSet

    home = fake_home()
    machine = home / ".localharness" / "skills"
    _skill(machine, "good")
    _skill(machine, "broken", "no frontmatter")
    for i in range(257):
        (pkg / "assets" / "skills" / f"d{i:03d}").mkdir(parents=True, exist_ok=True)
    found = SkillSet.discover(home / ".localharness", None)
    assert len(found.skipped) == 2
    lines = _run()
    assert lines[1:] == ["Skipped:", *(f"  {s.reason}. {s.fix}." for s in found.skipped)]
    assert lines[0].startswith("good (machine, ")


def test_a_core_name_collision_is_the_start_warning_under_skipped_and_the_skill_stays_listed(fake_home, pkg, rows_clear):
    from localharness.cli.skill_session import skill_row
    from localharness.cli.slash_commands import set_plugin_rows
    from localharness.config.skills import SkillSet

    home = fake_home()
    _skill(home / ".localharness" / "skills", "help")
    record = SkillSet.discover(home / ".localharness", None).skills[0]
    warning = set_plugin_rows([skill_row(record)])[0]
    assert warning == 'skill help: slash command /help is already taken — say "use the help skill", or rename it'
    lines = _run()
    assert lines[0].startswith("help (machine, ")
    assert lines[1:] == ["Skipped:", "  " + warning]


def test_collisions_follow_the_skips_in_skill_order(fake_home, pkg, rows_clear):
    from localharness.config.skills import SkillSet

    home = fake_home()
    machine = home / ".localharness" / "skills"
    for name in ("help", "agents"):
        _skill(machine, name)
    _skill(machine, "broken", "no frontmatter")
    skipped = SkillSet.discover(home / ".localharness", None).skipped[0]
    lines = _run()
    assert [l.split(" (")[0] for l in lines[:2]] == ["agents", "help"]
    assert lines[2:] == [
        "Skipped:",
        f"  {skipped.reason}. {skipped.fix}.",
        '  skill agents: slash command /agents is already taken — say "use the agents skill", or rename it',
        '  skill help: slash command /help is already taken — say "use the help skill", or rename it',
    ]


def test_a_plugin_row_the_process_holds_adds_no_collision_line(fake_home, pkg, rows_clear):
    from localharness.cli.slash_commands import SlashCommand, set_plugin_rows

    home = fake_home()
    _skill(home / ".localharness" / "skills", "notes")
    assert set_plugin_rows([SlashCommand("/notes", "n", "_slash_help", plugin="demo")]) == []
    lines = _run()
    assert len(lines) == 1 and "already taken" not in lines[0]


def test_a_dropped_description_is_marked_with_its_length_the_budget_and_the_fit(fake_home, pkg):
    from localharness.config.skills import LISTING_BUDGET_CHARS, SkillSet, shown

    home = fake_home()
    machine = home / ".localharness" / "skills"
    aa, bb = _sized(machine, "aa", 900), _sized(machine, "bb", 900)
    found = SkillSet.discover(home / ".localharness", None)
    row = shown(found.skills)[0]
    assert (row[2], row[3], len(found.skills[0].description), LISTING_BUDGET_CHARS) == (None, 400, 900, 1300)
    assert _run() == [
        f"aa (machine, {str(aa)!r}) — description dropped from the listing (900 characters; the listing budget is 1300); shorten to 400",
        f"bb (machine, {str(bb)!r})",
    ]


@pytest.mark.parametrize("first, marks", [(650, 0), (651, 1)])
def test_the_mark_starts_one_character_past_the_budget(fake_home, pkg, first, marks):
    machine = fake_home() / ".localharness" / "skills"
    _sized(machine, "aa", first)
    _sized(machine, "bb", 650)
    assert sum("description dropped from the listing" in line for line in _run()) == marks


def test_no_mark_when_every_description_is_kept(fake_home, pkg):
    _sized(fake_home() / ".localharness" / "skills", "aa", 100)
    assert all(" — " not in line for line in _run())


def test_the_length_counts_code_points_and_the_budget_prints_as_the_constant(fake_home, pkg):
    machine = fake_home() / ".localharness" / "skills"
    _sized(machine, "aa", 700, "\U0001f600")
    _sized(machine, "bb", 700)
    line = _run()[0]
    assert line.endswith(" — description dropped from the listing (700 characters; the listing budget is 1300); shorten to 600")


def test_the_mark_lands_on_the_project_line_first(fake_home, pkg, monkeypatch):
    home = fake_home()
    proj = _project(home, monkeypatch)
    _sized(home / ".localharness" / "skills", "m", 800)
    _sized(proj / ".localharness" / "skills", "p", 800)
    lines = _run()
    assert lines[0].startswith("p (project, ") and lines[0].endswith("shorten to 500")
    assert lines[1].startswith("m (machine, ") and " — " not in lines[1]


def test_nothing_found_prints_nothing(fake_home, pkg, monkeypatch):
    home = fake_home()
    (home / "proj").mkdir()
    monkeypatch.chdir(home / "proj")
    result = CliRunner().invoke(app, ["skills"])
    assert (result.exit_code, result.output) == (0, "")


def test_the_verb_never_asks_or_records_for_an_undecided_workspace_above(fake_home, pkg, monkeypatch):
    home = fake_home()
    _skill(home / "outer" / ".localharness" / "skills", "x")
    inner = home / "outer" / "inner"
    inner.mkdir()
    monkeypatch.chdir(inner)

    def boom(*args, **kwargs):
        raise AssertionError("asked or recorded")

    monkeypatch.setattr("localharness.cli.workspace._ask", boom)
    monkeypatch.setattr("localharness.config.trust.record_trust", boom)
    before = sorted(p.relative_to(home) for p in home.rglob("*"))
    result = CliRunner().invoke(app, ["skills"])
    assert result.exit_code == 0
    assert "x (" not in result.output
    assert sorted(p.relative_to(home) for p in home.rglob("*")) == before


def test_a_bracket_in_a_path_survives_and_an_escape_is_written_out(fake_home, pkg, tmp_path):
    fake_home()
    cfg = tmp_path / "cfg [x]"
    path = _skill(cfg / "skills", "notes")
    assert _run("--config-dir", str(cfg)) == [f"notes (machine, {str(path)!r})"]
    evil = _skill(cfg / "skills", "a\x1bb")
    output = CliRunner().invoke(app, ["skills", "--config-dir", str(cfg)]).output
    assert "\x1b" not in output
    assert f"  {str(evil)!r} holds a control character. rename it without control characters." in output.splitlines()


def _agents_only(directory):
    from localharness.cli.skills_cmd import AGENTS_ONLY

    return AGENTS_ONLY.format(dir=repr(str(directory)))


def test_the_agents_only_line_leads_and_the_skills_under_it_are_not_listed(fake_home, pkg, monkeypatch):
    home = fake_home()
    proj = home / "proj"
    _skill(proj / ".agents" / "skills", "x")
    monkeypatch.chdir(proj)
    lines = _run()
    assert lines == [
        f"{str(Path.cwd() / '.agents' / 'skills')!r} is not read: no .localharness/ here; localharness init --workspace "
        "creates it, adds files to this repository, and then loads these skills"
    ]
    assert lines == [_agents_only(Path.cwd() / ".agents" / "skills")]


def test_beside_a_localharness_dir_the_agents_skills_are_listed_and_nothing_is_said(fake_home, pkg, monkeypatch):
    home = fake_home()
    proj = _project(home, monkeypatch)
    path = _skill(proj / ".agents" / "skills", "x")
    assert _run() == [f"x (project, {str(path)!r})"]


@pytest.mark.parametrize("as_file", [False, True])
def test_no_agents_skills_directory_no_line(fake_home, pkg, monkeypatch, as_file):
    home = fake_home()
    (home / "proj" / ".agents").mkdir(parents=True)
    if as_file:
        (home / "proj" / ".agents" / "skills").write_text("")
    monkeypatch.chdir(home / "proj")
    assert _run() == []


def test_a_deleted_working_directory_says_nothing_and_does_not_crash(fake_home, pkg, monkeypatch):
    gone = fake_home() / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    assert _run() == []


def test_a_localharness_dir_made_between_two_runs_changes_only_the_second(fake_home, pkg, monkeypatch):
    home = fake_home()
    proj = home / "proj"
    path = _skill(proj / ".agents" / "skills", "x")
    monkeypatch.chdir(proj)
    first = _run()
    (proj / ".localharness").mkdir()
    assert (first, _run()) == ([_agents_only(Path.cwd() / ".agents" / "skills")], [f"x (project, {str(path)!r})"])


@pytest.mark.parametrize("agents, workspace, count", [(True, False, 1), (False, False, 0), (True, True, 0)])
async def test_start_prints_the_agents_only_line_once_and_only_then(tmp_path, fake_home, pkg, monkeypatch, agents, workspace, count):
    import io

    from rich.console import Console

    from localharness.cli import start_cmd
    from localharness.cli.start_cmd import _start_async
    from tests.unit.test_skills_listing import _start_world

    fake_home(tmp_path, clear_overrides=False)
    _start_world(tmp_path, monkeypatch, skill=False)
    work = tmp_path / "work"
    (work / ".agents" / "skills").mkdir(parents=True) if agents else None
    (work / ".localharness").mkdir() if workspace else None
    out = io.StringIO()
    monkeypatch.setattr(start_cmd, "console", Console(file=out, width=400))
    await _start_async(None, False, False, str(tmp_path))
    assert out.getvalue().count(_agents_only(work / ".agents" / "skills")) == count
