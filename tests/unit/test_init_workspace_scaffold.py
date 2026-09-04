"""CLI-01 — `localharness init --workspace` scaffolds THIS project's layer, and only this one.

Phases 38-42 built a workspace layer that a session finds, merges, confines and scopes its memory
to — but there was no way to CREATE one except `mkdir .localharness`. `init` had zero workspace
handling: its first executable line resolved the machine-global dir. This file grades the command
that closes that gap.

Four properties, each of which a plausible-looking implementation could miss:

* **It writes to the current directory, not the machine.** A branch placed after
  `config_path = resolve_config_dir(config_dir)` — or one reusing that local — scaffolds a perfectly
  shaped tree into `~/.localharness` and passes every "the files exist" assertion. So the
  fresh-scaffold test asserts the GLOBAL dir is still empty *before* it asserts the workspace tree
  exists: that ordering is the point, not an accident (41-06's assertion-order lesson).
* **It never destroys a config you wrote.** The refusal path is graded by writing a marker into the
  scaffolded `config.yaml` and asserting the marker survives a second run — byte-identity against
  freshly-generated identical bytes would pass even if the file were rewritten.
* **It exits 1, not 0.** Plain `init`'s refusal is an INTERACTIVE `Confirm.ask` that exits 0. This
  path is script-facing and prompt-free, so "already there" must be distinguishable from "created"
  by the exit code alone (orchestrator ruling 3, phase 43). Two different behaviors, deliberately.
* **It never prompts.** Dogfood finding F3: an EOF on a prompt aborts a scripted run. Every test in
  this file makes `Confirm.ask` RAISE, so a prompt appearing fails the test instead of hanging it.

The control at the bottom is LAYR-03 discipline: plain `init` (no `--workspace`) behaves exactly as
it did before phase 43 existed, including creating nothing in the current directory.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml
from rich.prompt import Confirm
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import WORKSPACE_DIR_NAME

runner = CliRunner()

_MARKER = "# marker: the user edited this file by hand\n"


@pytest.fixture(autouse=True)
def _no_prompts(monkeypatch):
    """A prompt on the --workspace path is a FAILURE, not a hang.

    `init_cmd` does `from rich.prompt import Confirm`, so the name it holds IS this class — patching
    the classmethod here covers both modules. Set as a plain function: `Confirm.ask(...)` is called
    on the class, so Python passes the arguments through unbound.
    """

    def _explode(*args, **kwargs):  # pragma: no cover - only runs when the test is failing
        raise AssertionError(f"init --workspace prompted: Confirm.ask{args!r}")

    monkeypatch.setattr(Confirm, "ask", _explode)


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A CWD inside a fake project, with a fake `$HOME` holding an EMPTY global layer.

    All the env moves matter. Both `~` expansion and the discovery walk read `$HOME`, so a real one
    would let the developer's own `~/.localharness` answer these tests; `LOCALHARNESS_DIR` is the
    `--config-dir` option's envvar (conftest's autouse fixture sets its legacy alias), so leaving
    either set would make the conflict guard fire on a command line that never passed the flag.
    `LOCALHARNESS_ENDPOINT` / `LOCALHARNESS_MODEL` bind `--endpoint` / `--model` the same way.
    """
    for var in ("LOCALHARNESS_DIR", "LOCALHARNESS_HOME", "LOCALHARNESS_ENDPOINT", "LOCALHARNESS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    (home / WORKSPACE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # keep rich from wrapping a path out of a message
    proj = tmp_path / "proj" / "src"
    proj.mkdir(parents=True)
    monkeypatch.chdir(proj)
    return proj


def _global_dir(project: Path) -> Path:
    return Path.home() / WORKSPACE_DIR_NAME


def _listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- fresh scaffold


def test_fresh_scaffold_lands_in_cwd_and_leaves_the_machine_alone(project):
    """The shape of the tree, the emptiness of the config, and WHERE it all landed.

    The global-dir assertion runs SECOND, right after the premise: a scaffold that also wrote
    globally — or that wrote ONLY globally — satisfies every other assertion in this test.
    """
    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 0, result.output
    assert _listing(_global_dir(project)) == [], "init --workspace touched the machine-global layer"

    workspace = project / WORKSPACE_DIR_NAME
    config_file = workspace / "config.yaml"
    assert config_file.exists()
    assert (workspace / "agents").is_dir()

    text = config_file.read_text(encoding="utf-8")
    # The claim is "no provider KEY", not "the word never appears": the template deliberately
    # SAYS in a comment why provider belongs to the machine, and a bare `"provider" not in text`
    # grades that sentence rather than the scaffold. An anchored, comment-excluding match is the
    # form that a real `provider:` block would trip.
    assert not re.search(r"^\s*provider:", text, re.M), "a workspace layer never carries provider"
    uncommented = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert uncommented == [], f"the scaffolded config must set nothing at all, got {uncommented}"
    assert yaml.safe_load(text) in (None, {})


def test_scaffold_creates_agents_but_no_overrides_and_no_state_dirs(project):
    """The exact file set, so a later 'while I am here' addition is a deliberate act.

    43-06 drives a real `_start_async` against this tree; anything extra here is something that
    session has to tolerate.
    """
    assert runner.invoke(app, ["init", "--workspace"]).exit_code == 0

    workspace = project / WORKSPACE_DIR_NAME
    assert _listing(workspace) == ["agents", "config.yaml"]


def test_scaffold_does_not_prompt_with_stdin_closed(project):
    """Dogfood F3: an EOF on a prompt aborts a scripted run. This path asks nothing."""
    result = runner.invoke(app, ["init", "--workspace"], input="")

    assert result.exit_code == 0, result.output
    assert (project / WORKSPACE_DIR_NAME / "config.yaml").exists()


# --------------------------------------------------------------------------- refusal


@pytest.mark.parametrize("second_run_flags", [[], ["--force"]], ids=["plain", "force"])
def test_second_run_refuses_with_exit_1_and_destroys_nothing(project, second_run_flags):
    """Ruling 3: exit 1, not plain init's interactive exit 0 — and --force is not an escape hatch.

    The marker is what makes the byte-identity claim real. Without it the file would be compared
    against bytes the command would have regenerated identically, so an overwrite would be
    invisible.
    """
    assert runner.invoke(app, ["init", "--workspace"]).exit_code == 0

    workspace = project / WORKSPACE_DIR_NAME
    config_file = workspace / "config.yaml"
    config_file.write_text(config_file.read_text(encoding="utf-8") + _MARKER, encoding="utf-8")
    before_md5 = _md5(config_file)
    before_listing = _listing(workspace)

    result = runner.invoke(app, ["init", "--workspace", *second_run_flags])

    assert result.exit_code == 1, result.output
    assert str(workspace) in result.output, "the refusal must name the workspace it found"
    assert _md5(config_file) == before_md5, "the second run rewrote a config the user had edited"
    assert _listing(workspace) == before_listing


# --------------------------------------------------------------------------- flag conflicts


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--endpoint", "http://127.0.0.1:9/v1"),
        ("--model", "some-model"),
        ("--config-dir", "/nonexistent-config-dir"),
    ],
)
def test_workspace_refuses_to_be_combined_with_the_global_flags(project, flag, value):
    """Each conflicting flag means the OPPOSITE thing on this path, so it is refused, not ignored.

    --endpoint/--model write a provider block, which a workspace layer never carries; --config-dir
    names the global layer to replace, which is not where a workspace goes.
    """
    result = runner.invoke(app, ["init", "--workspace", flag, value])

    assert result.exit_code == 2, result.output
    assert flag in result.output, "the error must name the flag that conflicted"
    assert not (project / WORKSPACE_DIR_NAME).exists(), "a refused command wrote to disk"


# --------------------------------------------------------------------------- control (LAYR-03)


def test_plain_init_is_untouched_by_this_phase(project):
    """No --workspace: pre-43 behavior exactly — the global layer is the target and CWD is not.

    A dead port makes `init --endpoint` fail at the model listing (`_list_endpoint_models` returns
    None) before it writes anything, so this is a real invocation of the unchanged path rather than
    a stub of it.
    """
    result = runner.invoke(app, ["init", "--endpoint", "http://127.0.0.1:9/v1"])

    assert result.exit_code == 1, result.output
    assert not (project / WORKSPACE_DIR_NAME).exists(), "plain init created a workspace"
    assert not (_global_dir(project) / "config.yaml").exists()


# --------------------------------------------------------------------------- E-M4: $HOME


def test_workspace_in_the_home_directory_says_what_that_directory_is(project, monkeypatch):
    """`init --workspace` in $HOME targets `~/.localharness` — the MACHINE's config dir.

    The refusal used to be "a workspace already exists there", which is a true sentence about the
    wrong thing: that directory is not a workspace, and the advice it gave (edit its config.yaml)
    describes the global config, not a project layer. Naming what the directory actually is, and
    which command sets it up, is the whole fix (E-M4).
    """
    monkeypatch.chdir(Path.home())

    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 2, result.output
    assert "machine-wide config directory" in result.output
    assert "already exists" not in result.output


def test_the_home_check_does_not_fire_one_directory_over(tmp_path, monkeypatch):
    """The control: a project that merely SITS in $HOME is an ordinary project."""
    monkeypatch.chdir(Path.home())
    sibling = Path.home() / "some-project"
    sibling.mkdir()
    monkeypatch.chdir(sibling)

    result = runner.invoke(app, ["init", "--workspace"])

    assert result.exit_code == 0, result.output
    assert (sibling / WORKSPACE_DIR_NAME / "config.yaml").exists()
