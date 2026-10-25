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
    assert sum("alpha" in ln for ln in result.stdout.splitlines()) == 1


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
