"""The WRITE side of workspace discovery: does creation land where discovery reads?

Plan 39-06. Two claims, both about agreement rather than about any single function:

1. `agent create --project` writes into the workspace `agent list` reads from. Before this,
   `--project` wrote to the literal relative `.localharness` — so from `proj/src/pkg` it minted a
   THIRD workspace next to your source files, and `agent list` (which walks up-tree) never saw the
   agent you had just created. Creation and discovery disagreeing is the bug; a test that only
   checks "a file appeared somewhere" cannot see it.
2. `propose` / `experiment` / `autoresearch` share ONE "find my `.localharness`" algorithm. They
   used to hold three byte-identical copies, which is two opportunities to drift.

The archive db adopts the up-tree walk but NOT the trust gate (decision recorded in
`test_archive_db_is_not_trust_gated`). Fixtures are 39-04's `project` shape — deliberately outside
any repository, so the trust gate is live and the tests that turn it on/off actually mean
something.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.agent_cmd import agent_app
from localharness.config.paths import resolve_archive_db_path
from localharness.config.trust import record_trust

runner = CliRunner()


# --------------------------------------------------------------------------- fixtures


def _fake_home(tmp_path, monkeypatch) -> Path:
    """A hermetic `$HOME` holding the GLOBAL layer (and therefore the trust store), with both env
    overrides cleared so discovery is actually allowed to run. Every fixture needs this: both
    walks stop at `$HOME`, and the autouse conftest fixture sets `LOCALHARNESS_HOME`, which the
    resolver counts as an explicit selection."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # keep Rich from wrapping the --json line
    return home


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A discoverable workspace two levels above the cwd, OUTSIDE any repository (39-04's shape).

    No `.git` anywhere on purpose: that absence is what makes this workspace "config reaching in
    from outside the tree you opened", the only trust-gated case. Do not add one — it is what
    gives both `record_trust(ws, True)` and the untrusted `--json` test something to prove.
    """
    _fake_home(tmp_path, monkeypatch)
    ws = tmp_path / "proj" / ".localharness"
    (ws / "agents").mkdir(parents=True)
    deep = tmp_path / "proj" / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    from localharness.config.paths import workspace_is_within_repo

    # Guard: if TMPDIR itself lived in a repository these rows would silently become the
    # in-project case and the gate would never be exercised.
    assert not workspace_is_within_repo(ws, deep)
    return ws.resolve()


@pytest.fixture
def no_workspace(tmp_path, monkeypatch):
    """No `.localharness` anywhere up-tree. `$HOME` is `tmp_path` itself so the walk stays bounded
    (it stops there without inspecting it) instead of climbing to the real filesystem root."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "400")
    here = tmp_path / "empty" / "sub"
    here.mkdir(parents=True)
    monkeypatch.chdir(here)
    return here


class _FakeTTYStdin:
    def isatty(self) -> bool:
        return True


class _FakeSys:
    stdin = _FakeTTYStdin()


def _force_tty(monkeypatch) -> None:
    """Make the resolver believe a terminal is attached, for the whole CLI invocation.

    Patching `sys.stdin.isatty` does NOT survive `CliRunner`: click's isolation reassigns the real
    `sys.stdin` while the command runs and undoes it. The resolver reads `stdin` off its own
    module-level `sys` name, so replacing that name with a stub is the patch that holds.
    """
    monkeypatch.setattr("localharness.cli.workspace.sys", _FakeSys())


def _prompt_must_not_fire(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("prompted on a path whose rule forbids prompting")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _prompt_answers(monkeypatch, answer: bool) -> list:
    asked = []

    def _ask(*args, **kwargs):
        asked.append(args[0] if args else kwargs.get("prompt"))
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _write_agent(agents_dir: Path, name: str) -> None:
    agents_dir.mkdir(parents=True, exist_ok=True)
    (agents_dir / f"{name}.yaml").write_text(
        f"name: {name}\nrole: R\nmodel: inherit\n", encoding="utf-8"
    )


def _names(result) -> list:
    """Names from the JSON on STDOUT. Parsing `result.output` would fold in the resolver's stderr
    notices — and keeping those two channels apart is precisely why 39-04 put notices on stderr."""
    assert result.exit_code == 0, result.output
    return sorted(a["name"] for a in json.loads(result.stdout))


# ------------------------------------------------- agent create --project: the write target


def test_create_project_writes_into_the_discovered_workspace_not_the_cwd(project, monkeypatch):
    """The bug this plan exists to close: from a subdirectory, `--project` used to mint a second
    workspace next to your source files."""
    record_trust(project, True)

    result = runner.invoke(agent_app, ["create", "deep-agent", "--project", "--role", "R"])

    assert result.exit_code == 0, result.output
    assert (project / "agents" / "deep-agent.yaml").exists()
    assert not (Path.cwd() / ".localharness").exists()  # no second workspace scattered


def test_created_project_agent_appears_in_agent_list_from_same_dir(project, monkeypatch):
    """Creation and discovery agree — the whole point. Create, then immediately list, from the
    same deep directory, with nothing in between.

    The workspace holds `ws-agent` BEFORE the create, and that is what makes this test able to
    fail: asserting only "the new name is in the list" passes even with the bug present, because
    a `--project` write to the cwd MINTS `pkg/.localharness`, which then wins the up-tree walk as
    the nearest workspace and lists the agent it just created. Agreement means the create landed
    in the roster you already had — so the pre-existing agent must still be there too.
    """
    record_trust(project, True)
    _write_agent(project / "agents", "ws-agent")

    created = runner.invoke(agent_app, ["create", "agree-agent", "--project", "--role", "R"])
    assert created.exit_code == 0, created.output

    assert _names(runner.invoke(agent_app, ["list", "--json"])) == ["agree-agent", "ws-agent"]


def test_create_project_without_any_workspace_still_writes_to_local_dir(no_workspace):
    """LAYR-03: with nothing to discover, behavior is v0.12's exactly — and this is also the
    bootstrap path, where a fresh project's first agent creates the workspace."""
    result = runner.invoke(agent_app, ["create", "boot-agent", "--project", "--role", "R"])

    assert result.exit_code == 0, result.output
    assert (no_workspace / ".localharness" / "agents" / "boot-agent.yaml").exists()


def test_create_project_with_explicit_config_dir_refuses(project, tmp_path):
    """A-B3, and a reversal of what this file used to call documented behavior.

    An explicit `--config-dir` is a full replacement, so discovery never runs (LAYR-02) — which
    left `--project` with no project to resolve. It fell back to the relative literal, wrote the
    file, and printed a checkmark for an agent that the SAME options cannot read back: `agent list
    --config-dir X` and `start --config-dir X` skip discovery too. Writing somewhere nothing reads
    is worse than refusing, so it refuses; the trusted workspace is left untouched either way.
    """
    record_trust(project, True)

    result = runner.invoke(agent_app, [
        "create", "explicit-agent", "--project", "--role", "R",
        "--config-dir", str(tmp_path / "elsewhere"),
    ])

    assert result.exit_code != 0, result.output
    assert not (Path.cwd() / ".localharness").exists()
    assert not (project / "agents" / "explicit-agent.yaml").exists()


# ------------------------------------------------------------- agent list: --json never blocks


def test_agent_list_json_never_prompts_even_with_a_terminal_attached(project, monkeypatch):
    """`--json` is machine output. A tty-attached script must get JSON, not a blocked prompt, so
    `agent list` passes `interactive=False` for it. The workspace here is undecided, which is the
    only state that would otherwise ask."""
    _force_tty(monkeypatch)
    _prompt_must_not_fire(monkeypatch)
    _write_agent(Path.home() / ".localharness" / "agents", "global-agent")
    _write_agent(project / "agents", "ws-agent")

    result = runner.invoke(agent_app, ["list", "--json"])

    # Global-only roster as clean JSON on stdout: the undecided workspace stays inert...
    assert _names(result) == ["global-agent"]
    # ...and says so on stderr, which is also the proof `interactive=False` got through — that
    # notice is only reachable from the non-interactive branch, and the tty here is forced True.
    assert "no terminal to ask" in result.stderr


def test_agent_list_without_json_does_ask_with_a_terminal_attached(project, monkeypatch):
    """The control for the test above. Without it, `--json` "not prompting" could just mean the
    fake terminal never reached the resolver — this proves the same setup DOES ask on the human
    path, so the difference is the `interactive=False` pass-through and nothing else."""
    _force_tty(monkeypatch)
    asked = _prompt_answers(monkeypatch, False)

    result = runner.invoke(agent_app, ["list"])

    assert result.exit_code == 0, result.output
    assert len(asked) == 1


def test_agent_list_loads_a_trusted_workspace_from_a_deep_directory(project, monkeypatch):
    """LAYR-01/04 through the CLI: the roster follows the up-tree walk, not the cwd."""
    _prompt_must_not_fire(monkeypatch)
    record_trust(project, True)
    _write_agent(Path.home() / ".localharness" / "agents", "global-agent")
    _write_agent(project / "agents", "ws-agent")

    assert _names(runner.invoke(agent_app, ["list", "--json"])) == ["global-agent", "ws-agent"]


# --------------------------------------------------------------- resolve_archive_db_path


def test_archive_db_follows_the_env_override(tmp_path, monkeypatch):
    """The branch every existing archive test runs on, unchanged: env wins over any walk."""
    home = tmp_path / "explicit-home"
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))

    assert resolve_archive_db_path() == home / "archive.db"


def test_archive_db_follows_the_workspace_two_levels_up(project):
    """The new behavior: run `propose` from `proj/src/pkg` and it reaches the project's archive
    instead of minting an empty one beside your source files."""
    assert resolve_archive_db_path() == project / "archive.db"


def test_archive_db_falls_back_to_cwd_when_nothing_is_found(no_workspace):
    """LAYR-03 again: no workspace, no env — byte-identical to v0.12."""
    assert resolve_archive_db_path() == no_workspace / ".localharness" / "archive.db"


def test_archive_db_is_not_trust_gated(project, monkeypatch):
    """DECISION (roadmap-mandated, recorded here): archive.db adopts the up-tree walk but is
    exempt from LAYR-05's trust gate. It holds proposal history — a SQLite file the harness
    WRITES — not agent or config YAML the harness executes, so SECURITY.md's "treat config like
    code you are about to run" concern does not apply. Gating a storage-location choice behind a
    security prompt would be unmotivated and surprising. Hence `discover_workspace_dir()`
    directly, never `resolve_workspace_layer()`.
    """
    ws = project  # the fixture yields the `.localharness` dir itself
    _prompt_must_not_fire(monkeypatch)
    record_trust(ws, False)

    assert resolve_archive_db_path() == ws / "archive.db"


def test_the_three_commands_share_one_archive_algorithm(project, tmp_path, monkeypatch):
    """"One 'find my .localharness' algorithm, not two." Three modules used to carry three
    byte-identical copies; this walks all three through the same three states."""
    from localharness.cli.autoresearch_cmd import _archive_db_path as autoresearch_path
    from localharness.cli.experiment_cmd import _archive_db_path as experiment_path
    from localharness.cli.propose_cmd import _archive_db_path as propose_path

    # State 1: workspace two levels up, no env.
    assert propose_path() == experiment_path() == autoresearch_path() == project / "archive.db"

    # State 2: env override wins over that workspace.
    home = tmp_path / "env-home"
    monkeypatch.setenv("LOCALHARNESS_HOME", str(home))
    assert propose_path() == experiment_path() == autoresearch_path() == home / "archive.db"
    assert propose_path() == resolve_archive_db_path()

    # State 3: no env, nothing to discover — the cwd fallback.
    monkeypatch.delenv("LOCALHARNESS_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    expected = bare / ".localharness" / "archive.db"
    assert propose_path() == experiment_path() == autoresearch_path() == expected
    assert propose_path() == resolve_archive_db_path()
