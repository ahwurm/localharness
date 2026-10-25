"""Tests for localharness agent create and agent list commands."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.agent_cmd import agent_app

runner = CliRunner()


# ---------------------------------------------------------------------------
# agent create — name validation
# ---------------------------------------------------------------------------

def test_agent_create_invalid_name_uppercase(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "MyAgent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "invalid" in result.output.lower() or "Invalid" in result.output


def test_agent_create_invalid_name_spaces(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "my agent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1


def test_agent_create_invalid_name_starts_with_digit(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "1agent",
        "--global",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# agent create — mutual exclusion of --global / --project
# ---------------------------------------------------------------------------

def test_agent_create_both_flags_exits_error(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--project",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "Cannot use both" in result.output


# ---------------------------------------------------------------------------
# agent create — --global flag (no prompt)
# ---------------------------------------------------------------------------

def test_agent_create_global_writes_to_config_dir(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--role", "Test role",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    expected = tmp_path / "agents" / "test-agent.yaml"
    assert expected.exists()


def test_agent_create_global_generates_valid_yaml(tmp_path):
    runner.invoke(agent_app, [
        "create", "test-agent",
        "--global",
        "--role", "Test role",
        "--config-dir", str(tmp_path),
    ])
    path = tmp_path / "agents" / "test-agent.yaml"
    data = yaml.safe_load(path.read_text())
    assert data["name"] == "test-agent"
    assert "role" in data
    assert data.get("model", "inherit") == "inherit"


# ---------------------------------------------------------------------------
# agent create — #55: refuse to silently overwrite an existing agent
# ---------------------------------------------------------------------------

def test_agent_create_refuses_overwrite_existing_global(tmp_path):
    """#55: a second `create <name> --global` must NOT silently clobber the first
    (a live receipt erased a user's tools.deny under a "✓ created"). It exits 1,
    names the path, and leaves the existing file BYTE-unchanged."""
    path = tmp_path / "agents" / "dup-agent.yaml"
    first = runner.invoke(agent_app, [
        "create", "dup-agent", "--global", "--role", "The original",
        "--config-dir", str(tmp_path),
    ])
    assert first.exit_code == 0
    original = path.read_bytes()

    second = runner.invoke(agent_app, [
        "create", "dup-agent", "--global", "--role", "A DIFFERENT role that would clobber",
        "--config-dir", str(tmp_path),
    ])
    assert second.exit_code == 1
    assert "exists" in second.output.lower()
    assert str(path) in second.output.replace("\n", "")  # names the path (Rich wraps it)
    assert path.read_bytes() == original  # #55: no silent overwrite


def test_agent_create_refuses_overwrite_existing_project(tmp_path, monkeypatch):
    """#55 also holds for --project scope (same invariant, local dir).

    No `--config-dir`, and the env override cleared: `--project` refuses under an explicit config
    directory now (A-B3), and pairing them here would grade the refusal instead of the invariant.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".localharness" / "agents" / "loc-agent.yaml"
    runner.invoke(agent_app, ["create", "loc-agent", "--project", "--role", "orig"])
    original = path.read_bytes()
    second = runner.invoke(agent_app, ["create", "loc-agent", "--project", "--role", "clobber"])
    assert second.exit_code == 1
    assert path.read_bytes() == original


def test_agent_create_force_overwrites(tmp_path):
    """#55 escape hatch: --force overwrites deliberately (default refuses)."""
    path = tmp_path / "agents" / "force-agent.yaml"
    runner.invoke(agent_app, [
        "create", "force-agent", "--global", "--role", "original",
        "--config-dir", str(tmp_path),
    ])
    result = runner.invoke(agent_app, [
        "create", "force-agent", "--global", "--role", "replaced", "--force",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert yaml.safe_load(path.read_text())["role"] == "replaced"


# ---------------------------------------------------------------------------
# agent create — --project flag (no prompt)
# ---------------------------------------------------------------------------

def test_agent_create_project_writes_to_local_dir(tmp_path, monkeypatch):
    """The bootstrap path: nothing to discover, so `--project` creates this project's workspace."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(agent_app, [
        "create", "local-agent",
        "--project",
        "--role", "Local role",
    ])
    assert result.exit_code == 0, result.output
    expected = tmp_path / ".localharness" / "agents" / "local-agent.yaml"
    assert expected.exists()


# ---------------------------------------------------------------------------
# agent create — interactive prompt (neither flag)
# ---------------------------------------------------------------------------

def test_agent_create_prompt_global_answer(tmp_path):
    result = runner.invoke(
        agent_app,
        [
            "create", "prompted-agent",
            "--role", "Prompted",
            "--config-dir", str(tmp_path),
        ],
        input="global\n",
    )
    assert result.exit_code == 0
    expected = tmp_path / "agents" / "prompted-agent.yaml"
    assert expected.exists()


def test_agent_create_prompt_project_answer(tmp_path, monkeypatch):
    """Answering "project" takes the same route as `--project`, explicit config dir and all — so
    the env override is cleared here too (A-B3)."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        agent_app,
        [
            "create", "prompted-proj",
            "--role", "Prompted",
        ],
        input="project\n",
    )
    assert result.exit_code == 0
    expected = tmp_path / ".localharness" / "agents" / "prompted-proj.yaml"
    assert expected.exists()


# ---------------------------------------------------------------------------
# agent create — dry-run
# ---------------------------------------------------------------------------

def test_agent_create_dry_run_prints_yaml_no_file(tmp_path):
    result = runner.invoke(agent_app, [
        "create", "dry-agent",
        "--global",
        "--role", "Dry role",
        "--dry-run",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "dry-agent" in result.output
    # No file written
    assert not (tmp_path / "agents" / "dry-agent.yaml").exists()


# ---------------------------------------------------------------------------
# agent list — empty
# ---------------------------------------------------------------------------

def test_agent_list_no_agents(tmp_path):
    result = runner.invoke(agent_app, [
        "list",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "No agents" in result.output


# ---------------------------------------------------------------------------
# agent list — with agents (Rich table)
# ---------------------------------------------------------------------------

def test_agent_list_shows_table(tmp_path):
    # Create an agent first
    runner.invoke(agent_app, [
        "create", "listed-agent",
        "--global",
        "--role", "Listed role",
        "--config-dir", str(tmp_path),
    ])
    result = runner.invoke(agent_app, [
        "list",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    assert "listed-agent" in result.output
    assert "Name" in result.output


# ---------------------------------------------------------------------------
# agent list — JSON output
# ---------------------------------------------------------------------------

def test_agent_list_json_output(tmp_path):
    runner.invoke(agent_app, [
        "create", "json-agent",
        "--global",
        "--role", "JSON role",
        "--config-dir", str(tmp_path),
    ])
    result = runner.invoke(agent_app, [
        "list",
        "--json",
        "--config-dir", str(tmp_path),
    ])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    names = [a["name"] for a in data]
    assert "json-agent" in names


# ---------------------------------------------------------------------------
# agent list — what does not load, origin, rows
# ---------------------------------------------------------------------------

def _bound(monkeypatch, layer: Path) -> None:
    monkeypatch.setattr("localharness.cli.workspace.resolve_workspace_layer", lambda *a, **k: layer)


def _yaml(agents_dir: Path, stem: str, data: dict | str) -> Path:
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{stem}.yaml"
    path.write_text(data if isinstance(data, str) else yaml.safe_dump(data), encoding="utf-8")
    return path


def _world(tmp_path: Path) -> tuple[Path, Path]:
    cfg, proj = tmp_path / "cfg", tmp_path / "proj" / ".localharness"
    for stem in ("alpha", "beta"):
        _yaml(cfg / "agents", stem, {"name": stem, "role": f"machine {stem}"})
    for stem in ("alpha", "gamma"):
        _yaml(proj / "agents", stem, {"name": stem, "role": f"project {stem}"})
    return cfg, proj


def _line(output: str, word: str) -> str:
    return next(ln for ln in output.splitlines() if word in ln and ("│" in ln or "┃" in ln))


def test_agent_list_shows_each_agents_origin(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    cfg, proj = _world(tmp_path)
    _bound(monkeypatch, proj)
    result = runner.invoke(agent_app, ["list", "--config-dir", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "Origin" in _line(result.stdout, "Name")
    assert "project" in _line(result.stdout, "alpha")
    assert "machine" in _line(result.stdout, "beta")
    assert "project" in _line(result.stdout, "gamma")
    assert sum("alpha" in ln and "│" in ln for ln in result.stdout.splitlines()) == 1


def test_agent_list_prints_problems_on_stderr_before_the_empty_state(tmp_path):
    _yaml(tmp_path / "agents", "broken", "a: [unclosed")
    result = runner.invoke(agent_app, ["list", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0
    (line,) = [ln for ln in result.stderr.splitlines() if "broken.yaml" in ln]
    assert line.endswith("; edit the file")
    assert "No agents configured" in result.stdout
    as_json = runner.invoke(agent_app, ["list", "--json", "--config-dir", str(tmp_path)])
    assert as_json.stdout == "[]\n"
    assert "broken.yaml" in as_json.stderr


@pytest.mark.parametrize("kind, body", [("validation", {"capabilities": "lots"}), ("load", {"division": "nope"})])
def test_a_stem_whose_winner_does_not_load_drops_both_rows(tmp_path, monkeypatch, kind, body):
    cfg, proj = tmp_path / "cfg", tmp_path / "proj" / ".localharness"
    _yaml(cfg / "agents", "x", {"name": "x", "role": "machine x"})
    _yaml(proj / "agents", "x", {"name": "x", "role": "r", **body})
    _bound(monkeypatch, proj)
    result = runner.invoke(agent_app, ["list", "--json", "--config-dir", str(cfg)])
    assert result.exit_code == 0
    assert not [a for a in json.loads(result.stdout) if a["path"].endswith("x.yaml")]
    assert len([ln for ln in result.stderr.splitlines() if str(proj / "agents" / "x.yaml") in ln]) == 1


def test_a_mismatch_keeps_its_row_under_the_stem(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    _yaml(tmp_path / "agents", "foo", {"name": "bar", "role": "foo role", "model": "m1"})
    _yaml(tmp_path / "agents", "baz", {"name": "[red]q", "role": "baz role"})
    result = runner.invoke(agent_app, ["list", "-v", "--config-dir", str(tmp_path)])
    row = _line(result.stdout, "foo role")
    assert row.split("│")[1].strip() == "foo" and "m1" in row
    assert "name: 'bar' ignored; foo.yaml is agent 'foo'; rename the file or the name: to match" in result.stderr
    assert "[red]q" in result.stderr


def test_an_unparseable_shadowed_machine_file_is_never_parsed(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    cfg, proj = tmp_path / "cfg", tmp_path / "proj" / ".localharness"
    _yaml(cfg / "agents", "x", "a: [unclosed")
    _yaml(proj / "agents", "x", {"name": "x", "role": "project x"})
    _bound(monkeypatch, proj)
    result = runner.invoke(agent_app, ["list", "--config-dir", str(cfg)])
    assert "project" in _line(result.stdout, "x")
    assert "x.yaml" not in result.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any mode")
def test_an_unreadable_winner_is_skipped(tmp_path, monkeypatch):
    cfg, proj = tmp_path / "cfg", tmp_path / "proj" / ".localharness"
    locked = _yaml(proj / "agents", "x", {"name": "x", "role": "r"})
    _bound(monkeypatch, proj)
    locked.chmod(0)
    try:
        result = runner.invoke(agent_app, ["list", "--json", "--config-dir", str(cfg)])
    finally:
        locked.chmod(0o600)
    assert result.exit_code == 0 and result.exception is None
    assert not [a for a in json.loads(result.stdout) if a["name"] == "x"]
    assert "x.yaml" not in result.stderr


def test_hostile_cells_print_in_one_row_literally(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "400")
    _yaml(tmp_path / "agents", "h", {"name": "h", "role": "first\nsecond [red]r[/red] [\x1b/red]", "model": "[bold]m[/bold]"})
    _yaml(tmp_path / "agents", "[red]\x1bz", {"name": "z", "role": "r"})
    result = runner.invoke(agent_app, ["list", "-v", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0
    row = _line(result.stdout, "first second")
    assert "first second [red]r[/red] [/red]" in row and "[bold]m[/bold]" in row
    assert "\x1b" not in result.stdout and "\x1b" not in result.stderr
    assert "[red]z.yaml" in result.stderr


def test_rows_are_read_through_read_agent_file_once_each(tmp_path, monkeypatch):
    from localharness.config.loader import ConfigLoader

    cfg, proj = _world(tmp_path)
    _bound(monkeypatch, proj)
    seen: list[Path] = []
    real = ConfigLoader.read_agent_file

    def spy(path):
        seen.append(Path(path))
        return real(path)

    monkeypatch.setattr(ConfigLoader, "read_agent_file", staticmethod(spy))
    runner.invoke(agent_app, ["list", "--json", "--config-dir", str(cfg)])
    assert len(seen) == 3 and cfg / "agents" / "alpha.yaml" not in seen


def test_agent_list_warns_and_skips_unreadable_agent_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    global_dir = tmp_path / "global"
    _yaml(global_dir / "agents", "healthy", "name: healthy\nrole: Parses fine\n")
    _yaml(global_dir / "agents", "broken", "a: [unclosed\n")
    result = runner.invoke(agent_app, ["list", "--config-dir", str(global_dir)])
    assert result.exit_code == 0
    assert "healthy" in result.stdout
    (line,) = [ln for ln in result.stderr.splitlines() if "broken.yaml" in ln]
    assert line.endswith("; edit the file")


# ---------------------------------------------------------------------------
# agent list — shadows, --json fields, the unread-files line
# ---------------------------------------------------------------------------

def test_a_project_file_that_shadows_prints_one_dim_line(tmp_path, monkeypatch):
    from localharness.config.loader import shadow_label

    monkeypatch.setenv("COLUMNS", "400")
    cfg, proj = _world(tmp_path)
    _bound(monkeypatch, proj)
    result = runner.invoke(agent_app, ["list", "--config-dir", str(cfg)])
    assert result.stdout.count(shadow_label(cfg / "agents" / "alpha.yaml")) == 1
    assert result.stdout.count("shadows ") == 1


def test_agent_list_json_carries_origin_path_and_shadows(tmp_path, monkeypatch):
    cfg, proj = _world(tmp_path)
    server = {"name": "evil", "transport": "stdio", "command": "/bin/echo", "args": ["pwned"]}
    _yaml(proj / "agents", "gamma", {"name": "gamma", "role": "project gamma", "tools": {"mcp_servers": [server]}})
    _bound(monkeypatch, proj)
    result = runner.invoke(agent_app, ["list", "--json", "--config-dir", str(cfg)])
    by_name = {a["name"]: a for a in json.loads(result.stdout)}
    assert by_name["alpha"]["origin"] == "project"
    assert by_name["alpha"]["path"] == str(proj / "agents" / "alpha.yaml")
    assert by_name["alpha"]["shadows"] == str(cfg / "agents" / "alpha.yaml")
    assert by_name["beta"]["origin"] == "machine" and "shadows" not in by_name["beta"]
    assert by_name["gamma"]["tools"]["mcp_servers"] == [server]


@pytest.mark.parametrize("config_dir, env, expected", [
    ("/x", {}, "flag"),
    ("/x", {"LOCALHARNESS_DIR": "/x"}, "LOCALHARNESS_DIR"),
    ("/x", {"LOCALHARNESS_DIR": "/y"}, "flag"),
    (None, {"LOCALHARNESS_DIR": "/y"}, "LOCALHARNESS_DIR"),
    (None, {"LOCALHARNESS_HOME": "/h"}, "LOCALHARNESS_HOME"),
    (None, {}, None),
])
def test_config_dir_source_names_the_setting(monkeypatch, config_dir, env, expected):
    from localharness.cli.agent_cmd import config_dir_source

    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    assert config_dir_source(config_dir) == expected


@pytest.mark.parametrize("cause, why", [
    ("flag", "--config-dir was given"),
    ("LOCALHARNESS_DIR", "LOCALHARNESS_DIR is set"),
    ("LOCALHARNESS_HOME", "LOCALHARNESS_HOME is set"),
])
def test_agent_list_prints_the_unread_line_for_each_cause(tmp_path, monkeypatch, cause, why):
    from localharness.cli.agent_cmd import UNREAD_PROJECT_AGENTS
    from localharness.config.paths import discover_workspace_dir

    work, cfg = tmp_path / "work", tmp_path / "cfg"
    for stem in ("a", "b"):
        _yaml(work / ".localharness" / "agents", stem, {"name": stem, "role": "r"})
    cfg.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    args = ["list", "--json"]
    if cause == "flag":
        args += ["--config-dir", str(cfg)]
    else:
        monkeypatch.setenv(cause, str(cfg))
    result = runner.invoke(agent_app, args)
    line = UNREAD_PROJECT_AGENTS.format(n=2, dir=discover_workspace_dir(work) / "agents", why=why)
    assert line in result.stderr.replace("\n", "")
    assert json.loads(result.stdout) == []


def test_no_unread_line_when_bound_without_files_or_on_the_active_dir(tmp_path, monkeypatch):
    from localharness.cli.agent_cmd import unread_project_agents

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    assert unread_project_agents(None, None) == []
    assert unread_project_agents(None, str(tmp_path / "cfg")) == []
    (work / ".localharness" / "agents").mkdir(parents=True)
    assert unread_project_agents(None, str(tmp_path / "cfg")) == []
    _yaml(work / ".localharness" / "agents", "a", {"name": "a", "role": "r"})
    assert unread_project_agents(None, "./.localharness") == []
    active = str(work / ".localharness")
    monkeypatch.setenv("LOCALHARNESS_DIR", active)
    assert unread_project_agents(None, active) == []


def test_agents_only_and_unread_project_agents_return_lists(tmp_path, monkeypatch):
    from localharness.cli import skills_cmd
    from localharness.cli.agent_cmd import unread_project_agents

    monkeypatch.chdir(tmp_path)
    assert skills_cmd.agents_only(tmp_path) == []
    assert unread_project_agents(tmp_path, str(tmp_path / "cfg")) == []


# ---------------------------------------------------------------------------
# agent create — through the one writer
# ---------------------------------------------------------------------------

def test_agent_create_goes_through_the_one_writer(tmp_path, monkeypatch):
    from localharness.config.loader import ConfigLoader

    calls: list[tuple] = []
    real = ConfigLoader.write_agent

    def spy(self, name, yaml_text, *, layer, overwrite=False):
        calls.append((name, layer, overwrite))
        return real(self, name, yaml_text, layer=layer, overwrite=overwrite)

    monkeypatch.setattr(ConfigLoader, "write_agent", spy)
    cfg = tmp_path / "cfg"
    runner.invoke(agent_app, ["create", "a1", "--global", "--config-dir", str(cfg)])
    runner.invoke(agent_app, ["create", "a1", "--global", "--force", "--config-dir", str(cfg)])
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    runner.invoke(agent_app, ["create", "p1", "--project"])
    assert calls == [("a1", "machine", False), ("a1", "machine", True), ("p1", "project", False)]
    for written in (cfg / "agents" / "a1.yaml", proj / ".localharness" / "agents" / "p1.yaml"):
        assert written.stat().st_mode & 0o777 == 0o600


def test_agent_create_prints_cannot_write_for_a_symlinked_agent_file(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("untouched")
    agents = tmp_path / "cfg" / "agents"
    agents.mkdir(parents=True)
    (agents / "x.yaml").symlink_to(victim)
    result = runner.invoke(agent_app, ["create", "x", "--global", "--config-dir", str(tmp_path / "cfg")])
    flat = result.output.replace("\n", "")
    assert result.exit_code == 1
    assert "cannot write" in flat and "x.yaml is a symlink; replace it with a file" in flat
    assert victim.read_text() == "untouched" and (agents / "x.yaml").is_symlink()


def test_agent_create_collision_prints_cannot_write(tmp_path):
    args = ["create", "dup", "--global", "--config-dir", str(tmp_path)]
    runner.invoke(agent_app, args)
    path = tmp_path / "agents" / "dup.yaml"
    before = path.read_bytes()
    result = runner.invoke(agent_app, args)
    flat = result.output.replace("\n", "")
    assert result.exit_code == 1
    assert "cannot write" in flat and "already exists at" in flat and "pass --force to overwrite" in flat
    assert path.read_bytes() == before


def test_the_bootstrap_receipt_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(agent_app, ["create", "boot", "--project"])
    assert "Agent 'boot' created at .localharness/agents/boot.yaml" in result.stdout
    assert "Edit the YAML to customize role, tools, and permissions." in result.stdout
