"""A-B3/B5/B6 — where `agent create --project` writes, and when it refuses to guess.

`--project` means "this project's workspace", and it resolves that through the same discovery the
readers use. Discovery returns None for three different reasons, and the command used to treat all
three identically — it fell back to a literal `./.localharness/agents`, wrote the file, and printed
a green checkmark:

* **An explicit config dir was given** (`--config-dir`, `LOCALHARNESS_DIR` or `LOCALHARNESS_HOME`).
  Discovery is SKIPPED entirely for those — an explicit dir is a full replacement (LAYR-02). So the
  file landed in a `./.localharness` that the very same invocation's readers do not read: `agent
  list --config-dir X` cannot see it, `start --config-dir X` will not load it. The checkmark was a
  lie about a file nobody would ever open. That combination now refuses.
* **A workspace was found but is not in use** — outside your project and untrusted, or undecided
  with no terminal to ask. Falling back silently minted a SECOND dotdir next to the one the user
  already had. It still falls back (that is the only place left to write), but says so.
* **Nothing was found at all.** The bootstrap case, and the one that stays exactly as it was: a
  fresh project's first agent creates the workspace (LAYR-03).

Every test here clears `LOCALHARNESS_HOME`, which `tests/conftest.py` sets for every test in the
suite. That env var is precisely what hid this bug: with it set, `--project` takes the
explicit-config-dir branch, so no existing test could ever reach the discovery path it claims to
grade.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.agent_cmd import agent_app
from localharness.config.paths import WORKSPACE_DIR_NAME
from localharness.config.trust import record_trust

runner = CliRunner()


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """CWD in a plain project, no env override, a fake $HOME with an empty global dir."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    return proj


# --------------------------------------------------- A-B3: explicit config dir + --project


def test_project_with_config_dir_flag_refuses(project, tmp_path):
    result = runner.invoke(agent_app, [
        "create", "explicit-agent", "--project", "--role", "R",
        "--config-dir", str(tmp_path / "elsewhere"),
    ])

    assert result.exit_code != 0, result.output
    assert "--config-dir" in result.output
    assert "$LOCALHARNESS" not in result.output, "named an env var nobody set"
    # The refusal must leave NOTHING behind — this is the write that used to happen anyway.
    assert not (project / WORKSPACE_DIR_NAME).exists()
    assert not (tmp_path / "elsewhere").exists()


def test_project_with_config_dir_env_refuses(project, tmp_path, monkeypatch):
    """The env spelling is the one that actually bit: nothing on the command line says the dir
    was chosen, so the receipt looked completely ordinary."""
    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path / "elsewhere"))

    result = runner.invoke(agent_app, ["create", "env-agent", "--project", "--role", "R"])

    assert result.exit_code != 0, result.output
    assert not (project / WORKSPACE_DIR_NAME).exists()
    # Names the env var, not the flag. typer fills `config_dir` from LOCALHARNESS_DIR through its
    # own `envvar=`, so the parameter cannot tell them apart and the first version of this message
    # told a user who never typed `--config-dir` to go drop it.
    assert "$LOCALHARNESS_DIR" in result.output
    assert "--config-dir" not in result.output


def test_project_with_legacy_home_env_refuses(project, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "elsewhere"))

    result = runner.invoke(agent_app, ["create", "legacy-agent", "--project", "--role", "R"])

    assert result.exit_code != 0, result.output
    assert not (project / WORKSPACE_DIR_NAME).exists()
    assert "$LOCALHARNESS_HOME" in result.output


def test_refusal_names_both_ways_out(project, tmp_path):
    """A refusal that does not say what to do instead is a dead end."""
    result = runner.invoke(agent_app, [
        "create", "explicit-agent", "--project", "--role", "R",
        "--config-dir", str(tmp_path / "elsewhere"),
    ])

    assert "--global" in result.output


def test_global_with_an_explicit_config_dir_still_works(project, tmp_path):
    """The control: `--global` is what an explicit config dir is FOR. Nothing about it changes."""
    result = runner.invoke(agent_app, [
        "create", "global-agent", "--global", "--role", "R",
        "--config-dir", str(tmp_path / "elsewhere"),
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "elsewhere" / "agents" / "global-agent.yaml").exists()


# --------------------------------------------------- B5/B6: the fallback is announced


def test_untrusted_workspace_fallback_is_announced(tmp_path, monkeypatch):
    """A found-but-unusable workspace means `--project` mints a second dotdir. Say so, and name
    the one that was skipped — otherwise the user has two workspaces and no idea why."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    # A workspace ABOVE the cwd and outside any repository — 39-04's trust-gated shape.
    outside = tmp_path / "proj" / WORKSPACE_DIR_NAME
    (outside / "agents").mkdir(parents=True)
    deep = tmp_path / "proj" / "src"
    deep.mkdir()
    monkeypatch.chdir(deep)
    record_trust(outside.resolve(), False)

    result = runner.invoke(agent_app, ["create", "fallback-agent", "--project", "--role", "R"])

    assert result.exit_code == 0, result.output
    assert (deep / WORKSPACE_DIR_NAME / "agents" / "fallback-agent.yaml").exists()
    assert str(outside.resolve()) in result.output, "the skipped workspace is not named"
    assert str(Path(WORKSPACE_DIR_NAME)) in result.output


def test_bootstrap_with_nothing_to_discover_is_unchanged(project):
    """LAYR-03: a fresh project's first agent creates the workspace, silently, as it always did."""
    result = runner.invoke(agent_app, ["create", "boot-agent", "--project", "--role", "R"])

    assert result.exit_code == 0, result.output
    assert (project / WORKSPACE_DIR_NAME / "agents" / "boot-agent.yaml").exists()
    assert "not in use" not in result.output


# --------------------------------------------------- scaffold order: refuse before you write


def test_a_refused_overwrite_changes_nothing_on_disk(project):
    """A refusal leaves the tree byte-for-byte as it found it — no directory minted on the way to
    deciding not to write. Snapshotted, not spot-checked: "the file is unchanged" would pass with
    a stray sibling directory created next to it."""
    runner.invoke(agent_app, ["create", "kept-agent", "--project", "--role", "orig"])
    root = project / WORKSPACE_DIR_NAME
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    kept = (root / "agents" / "kept-agent.yaml").read_bytes()

    second = runner.invoke(agent_app, ["create", "kept-agent", "--project", "--role", "clobber"])

    assert second.exit_code == 1
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before
    assert (root / "agents" / "kept-agent.yaml").read_bytes() == kept


def test_a_file_where_the_agents_dir_belongs_fails_politely(project):
    """H4: `.localharness/agents` as a FILE. mkdir raises NotADirectoryError/FileExistsError, and
    a traceback is not a message."""
    (project / WORKSPACE_DIR_NAME).mkdir()
    (project / WORKSPACE_DIR_NAME / "agents").write_text("not a directory\n", encoding="utf-8")

    result = runner.invoke(agent_app, ["create", "blocked-agent", "--project", "--role", "R"])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert "agents" in result.output
