"""Phase 39 success criteria, driven through the real CLI.

The unit tests in tests/unit/test_workspace_*.py prove each seam. This file proves the BEHAVIOR:
a user standing in a subdirectory of their project gets that project's agents. A green unit test
on an unreferenced helper is a checkmark on nothing — these invocations go through
localharness.cli.app the way a terminal does.

What each name maps to (ROADMAP Phase 39, plus the owner's 2026-09-03 boundary ruling):

1. `criterion1` — from any subdirectory of a project holding `.localharness/`, that workspace's
   agents are in the roster.
2. `criterion2` — two `.localharness/` up-tree: only the NEAREST applies, and the choice is
   VISIBLE (doctor names it) rather than a silent multi-layer merge.
3. `criterion3` — `--config-dir` / `LOCALHARNESS_DIR` is a full replacement: discovery never runs.
4. `criterion4` — nothing up-tree, nothing changes (LAYR-03).
5. `in_project` — a workspace inside the project you are standing in loads with no prompt and
   writes NOTHING to the trust store.
6. `layr05` — a workspace from OUTSIDE that project is gated: inert without a terminal (and still
   valid JSON on stdout), loadable after one yes, remembered forever after.

Every layout here is built by hand: `.git` markers are plain empty directories, never real
repositories, so the suite needs no git binary and starts no child process (39-01's rule — the
boundary check reads the marker off the filesystem and never shells out).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config import trust
from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

# Bare CliRunner: click 8.4 separates stdout and stderr by default (and no longer accepts
# `mix_stderr`), which is what makes the "JSON on stdout, trust notice on stderr" assertions real.
runner = CliRunner()

# Copied from tests/conftest.py rather than imported: each fixture plants its own $HOME, and this
# stays readable as the whole of what the global layer contains.
_MINIMAL_CONFIG_YAML = (
    "version: '1'\n"
    "provider:\n"
    "  provider_type: vllm\n"
    "  base_url: http://localhost:8000/v1\n"
    "  default_model: test-model\n"
)


# --------------------------------------------------------------------------------- helpers


def _squash(text: str) -> str:
    """Whitespace-stripped comparison — rich wraps at the console width, so a long tmp path
    arrives in the capture with newlines folded into it (39-04's lesson)."""
    return "".join(text.split())


def _write_agent(agents_dir: Path, name: str, role: str = "R") -> Path:
    """The three fields `agent create` writes, so these files are valid AgentConfigs and
    `validate` reports them as valid rather than merely as listed."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    path.write_text(yaml.dump({"name": name, "role": role, "model": "inherit"}), encoding="utf-8")
    return path


def _hermetic(tmp_path: Path, monkeypatch, home: Path) -> Path:
    """A fake `$HOME` holding the GLOBAL layer, with both env overrides cleared.

    Every fixture needs all three moves. Both walks stop at `$HOME`, so a real one would let the
    developer's own `~/.localharness` answer these tests; and the autouse conftest fixture sets
    `LOCALHARNESS_HOME`, which the resolver counts as an explicit selection — leaving it set would
    switch discovery off and every test below would pass for the wrong reason.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "config.yaml").write_text(_MINIMAL_CONFIG_YAML, encoding="utf-8")
    _write_agent(global_dir / "agents", "global-agent", "global role")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # keep rich from wrapping the --json line
    return global_dir


class _FakeStdin:
    def __init__(self, tty: bool) -> None:
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _set_tty(monkeypatch, value: bool) -> None:
    """Pin whether the resolver believes a terminal is attached, in BOTH directions.

    Never left to how the suite was launched. Patching the real `sys.stdin.isatty` does not
    survive `CliRunner` — click's isolation reassigns `sys.stdin` for the duration of the command
    and restores it afterwards. The resolver reads `stdin` off its own module-level `sys` name, so
    replacing that name is the patch that holds (39-06's finding).
    """
    monkeypatch.setattr(
        "localharness.cli.workspace.sys", SimpleNamespace(stdin=_FakeStdin(value))
    )


def _prompt_must_not_fire(monkeypatch) -> None:
    """A rule that forbids a prompt is tested by making the prompt RAISE. Silence proves nothing,
    and an unpatched prompt on a path that should not ask would HANG the runner on stdin."""

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on a path whose rule forbids prompting")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _prompt_answers(monkeypatch, answer: bool) -> list:
    asked: list = []

    def _ask(*args, **kwargs):
        asked.append(args[0] if args else kwargs.get("prompt"))
        return answer

    monkeypatch.setattr("rich.prompt.Confirm.ask", _ask)
    return asked


def _names(result) -> list:
    """Names from the JSON on STDOUT. Parsing `result.output` would fold in the resolver's stderr
    notices — keeping those two channels apart is precisely why 39-04 put notices on stderr."""
    assert result.exit_code == 0, result.output
    return sorted(a["name"] for a in json.loads(result.stdout))


# -------------------------------------------------------------------------------- fixtures


def _build_project(tmp_path: Path, monkeypatch, *, nested: bool = False) -> SimpleNamespace:
    """A repository at `proj/` whose root holds the workspace, with the CWD two levels down.

    The `.git` directory is load-bearing, not decoration: it makes `proj/` the project the test is
    standing in, which is what lets the workspace load with no prompt (the owner-ruled in-project
    rule). `nested=True` adds a SECOND workspace at `proj/src/`, one level nearer the CWD.
    """
    home = tmp_path / "home"
    global_dir = _hermetic(tmp_path, monkeypatch, home)

    proj = tmp_path / "proj"
    ws_dir = proj / ".localharness"
    _write_agent(ws_dir / "agents", "ws-agent", "workspace role")
    (proj / ".git").mkdir(parents=True)

    inner_ws = proj / "src" / ".localharness"
    if nested:
        _write_agent(inner_ws / "agents", "inner-agent", "inner workspace role")

    deep_dir = proj / "src" / "pkg"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)

    # Guards: if either flipped, every assertion below would pass for the wrong reason.
    expected = (inner_ws if nested else ws_dir).resolve()
    assert discover_workspace_dir() == expected
    assert workspace_is_within_repo(expected, deep_dir)
    return SimpleNamespace(
        home_dir=home,
        global_dir=global_dir,
        ws_dir=ws_dir.resolve(),
        inner_ws_dir=inner_ws.resolve(),
        deep_dir=deep_dir,
    )


@pytest.fixture
def workspace_project(tmp_path, monkeypatch):
    """One workspace, at the root of the repository the CWD sits inside."""
    project = _build_project(tmp_path, monkeypatch)
    _prompt_must_not_fire(monkeypatch)
    return project


@pytest.fixture
def nested_workspaces(tmp_path, monkeypatch):
    """Two workspaces up-tree: `proj/` and the nearer `proj/src/`. Both are in-project."""
    project = _build_project(tmp_path, monkeypatch, nested=True)
    _prompt_must_not_fire(monkeypatch)
    return project


@pytest.fixture
def cwd_workspace(tmp_path, monkeypatch):
    """The literal `./.localharness` case, outside any repository — v0.12's exact behavior.

    No `.git` anywhere, and the CWD *is* the workspace's folder, so this exercises the resolver's
    `found.parent == here` branch rather than the repository one.
    """
    home = tmp_path / "home"
    global_dir = _hermetic(tmp_path, monkeypatch, home)

    proj = tmp_path / "proj"
    ws_dir = proj / ".localharness"
    _write_agent(ws_dir / "agents", "ws-agent", "workspace role")
    monkeypatch.chdir(proj)
    _prompt_must_not_fire(monkeypatch)

    assert discover_workspace_dir() == ws_dir.resolve()
    # Guard: no repository, so "in project" here can ONLY mean the exact-cwd rule.
    assert not workspace_is_within_repo(ws_dir, proj)
    return SimpleNamespace(home_dir=home, global_dir=global_dir, ws_dir=ws_dir.resolve())


@pytest.fixture
def workspace_above_repo(tmp_path, monkeypatch):
    """The gated case: the nearest workspace sits ABOVE the repository root.

    `.localharness` at `outer/`, `.git` at `outer/inner/`, CWD `outer/inner/sub`. You opened the
    repository at `outer/inner`; the config is reaching in from a tree you did not open. That is
    the only shape LAYR-05's prompt exists for — do not add a `.git` at `outer/`, it would turn
    every row here into the silent in-project case.
    """
    home = tmp_path / "home"
    global_dir = _hermetic(tmp_path, monkeypatch, home)

    outer = tmp_path / "outer"
    ws_dir = outer / ".localharness"
    _write_agent(ws_dir / "agents", "ws-agent", "workspace role")
    (outer / "inner" / ".git").mkdir(parents=True)

    deep_dir = outer / "inner" / "sub"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)

    assert discover_workspace_dir() == ws_dir.resolve()
    # Guard: being in *a* repository is not enough — the workspace must be outside it.
    assert not workspace_is_within_repo(ws_dir, deep_dir)
    return SimpleNamespace(
        home_dir=home, global_dir=global_dir, ws_dir=ws_dir.resolve(), deep_dir=deep_dir
    )


class _StubCounter:
    """doctor asks the counter for its resolved mode; the real one talks to a server."""

    def __init__(self, base_url=None, model=None, provider_type=None, **_kw):
        self.mode = "vllm"
        self.approximate = False


@pytest.fixture
def doctor_offline(monkeypatch):
    """doctor with no network. The workspace line prints before any of it, but an unmocked run
    would spend real timeouts on localhost:8000."""
    import localharness.agent.context as _ctx

    monkeypatch.setattr(_ctx, "TokenCounter", _StubCounter)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": "test-model", "max_model_len": 131072}]}
    with patch("localharness.cli.doctor_cmd.httpx") as mock_httpx:
        mock_httpx.get.return_value = resp
        mock_httpx.post.return_value = resp
        yield mock_httpx


# ------------------------------------------------------- criterion 1: the roster follows the tree


def test_criterion1_workspace_agent_in_roster_from_deep_subdirectory(workspace_project):
    """The whole phase in one command: `cd proj/src/pkg && localharness agent list` shows the
    project's own agent. `ws-agent` exists ONLY in the workspace, so its presence in the roster is
    the same claim as "the workspace layer was discovered and loaded"."""
    result = runner.invoke(app, ["agent", "list", "--json"])

    assert _names(result) == ["global-agent", "ws-agent"]


def test_criterion1_validate_sees_workspace_agent_files(workspace_project):
    """A second command, so criterion 1 is not one command's accident. `validate` checks the files
    the session will actually load — reporting `ws-agent.yaml` from two levels down is that."""
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0, result.output
    assert "ws-agent.yaml" in result.output


def test_in_project_workspace_loads_without_prompting_or_recording(workspace_project):
    """The owner's 2026-09-03 ruling, observed end to end: your own project is not a trust
    boundary. The layer applies (the fixture's `Confirm.ask` raises, so any prompt fails here),
    and no verdict about it is ever written down — the trust store file is not even created."""
    result = runner.invoke(app, ["agent", "list", "--json"])

    assert "ws-agent" in _names(result)
    assert not (workspace_project.home_dir / ".localharness" / "trusted_workspaces.yaml").exists()
    assert trust.is_trusted(workspace_project.ws_dir) is None


# ------------------------------------------------------------- criterion 2: exactly one layer


def test_criterion2_only_the_nearest_workspace_applies(nested_workspaces):
    """LAYR-04. With `.localharness/` at `proj/` and `proj/src/`, the roster holds the INNER
    workspace's agent and not the outer one's — one layer, chosen by proximity, never a merge of
    both. `ws-agent` (outer) being absent is the load-bearing half."""
    result = runner.invoke(app, ["agent", "list", "--json"])

    assert _names(result) == ["global-agent", "inner-agent"]


def test_criterion2_doctor_names_the_chosen_layer(nested_workspaces, doctor_offline):
    """...and the choice is VISIBLE. A silent pick between two candidate layers is exactly the
    confusion this phase exists to avoid, so doctor prints the one it took."""
    result = runner.invoke(app, ["doctor"])

    assert "Workspace layer:" in result.output
    assert _squash(str(nested_workspaces.inner_ws_dir)) in _squash(result.output)
    assert _squash(str(nested_workspaces.ws_dir)) not in _squash(result.output)


# --------------------------------------------------- criterion 3: an explicit dir is a replacement


def test_criterion3_explicit_config_dir_flag_skips_discovery(workspace_project):
    """LAYR-02: naming a config dir replaces the whole config layer, so the workspace two levels
    up is invisible even though the walk would find it."""
    result = runner.invoke(
        app, ["agent", "list", "--json", "--config-dir", str(workspace_project.global_dir)]
    )

    assert _names(result) == ["global-agent"]


def test_criterion3_localharness_dir_env_skips_discovery(workspace_project, monkeypatch):
    """Same rule through the env form — the variable a user exports once and forgets."""
    monkeypatch.setenv("LOCALHARNESS_DIR", str(workspace_project.global_dir))

    result = runner.invoke(app, ["agent", "list", "--json"])

    assert _names(result) == ["global-agent"]


# ------------------------------------------------------- criterion 4: no workspace, no difference


def test_criterion4_no_workspace_uptree_is_unchanged(tmp_path, monkeypatch, doctor_offline):
    """LAYR-03: a user with no `.localharness/` anywhere up-tree sees v0.12's behavior and none of
    the new vocabulary. The roster is global-only and doctor does not mention a workspace at all —
    the new lines are conditional, not merely empty."""
    home = tmp_path / "home"
    _hermetic(tmp_path, monkeypatch, home)
    here = tmp_path / "empty" / "sub"
    here.mkdir(parents=True)
    monkeypatch.chdir(here)
    _prompt_must_not_fire(monkeypatch)
    assert discover_workspace_dir() is None  # guard: a stray workspace would void this test

    assert _names(runner.invoke(app, ["agent", "list", "--json"])) == ["global-agent"]

    doctored = runner.invoke(app, ["doctor"])
    assert "Workspace layer:" not in doctored.output
    assert "Global layer:" not in doctored.output


def test_workspace_at_the_current_directory_outside_a_repo_loads_silently(cwd_workspace):
    """The literal `./.localharness` read v0.12 always did, preserved exactly. No repository is
    involved, so this is the resolver's exact-cwd branch — and it must stay as quiet as the plain
    directory read it replaces (`Confirm.ask` raises in the fixture)."""
    result = runner.invoke(app, ["agent", "list", "--json"])

    assert _names(result) == ["global-agent", "ws-agent"]
    assert not trust.trust_store_path().exists()


# ------------------------------------------------------------ LAYR-05: config from outside the project


def test_layr05_workspace_above_the_repo_root_is_inert_without_a_tty(
    workspace_above_repo, monkeypatch
):
    """Fail closed, and do not spend the user's one-time answer.

    Undecided workspace + no terminal: the layer is ignored, a notice explains why on STDERR, and
    NOTHING is recorded — a later interactive session in this directory is still asked. The
    stdout/stderr split is the other half of the claim: `--json` is a machine contract, so the
    notice must not land in the document the caller parses.
    """
    _set_tty(monkeypatch, False)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["agent", "list", "--json"])

    # The whole stdout document parses, not merely a substring of it — a notice leaking onto
    # stdout would still contain the right agent names while breaking every caller that pipes
    # this into `jq`. Parsed here directly rather than through `_names`, because the JSON
    # contract surviving a stderr notice IS this test's claim.
    assert [a["name"] for a in json.loads(result.stdout)] == ["global-agent"]
    assert _names(result) == ["global-agent"]  # ...and the workspace stayed out
    assert "no terminal to ask" in result.stderr
    assert "no terminal to ask" not in result.stdout
    assert trust.is_trusted(workspace_above_repo.ws_dir) is None
    assert not trust.trust_store_path().exists()

    # The same fail-closed outcome on a human-facing command, where the tty check is the only
    # thing standing between the user and a prompt.
    validated = runner.invoke(app, ["validate"])
    assert "ws-agent.yaml" not in validated.output


def test_layr05_workspace_above_the_repo_root_loads_after_answering_yes(
    workspace_above_repo, monkeypatch
):
    """One question, on a terminal, and the layer applies. The answer lands in the GLOBAL trust
    store — never inside the workspace being judged, which could otherwise vouch for itself."""
    _set_tty(monkeypatch, True)
    asked = _prompt_answers(monkeypatch, True)

    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0, result.output
    assert len(asked) == 1
    assert "ws-agent.yaml" in result.output
    assert trust.is_trusted(workspace_above_repo.ws_dir) is True
    assert trust.trust_store_path() == (
        workspace_above_repo.global_dir / "trusted_workspaces.yaml"
    )
    assert trust.trust_store_path().exists()
    assert not (workspace_above_repo.ws_dir / "trusted_workspaces.yaml").exists()


def test_layr05_trust_is_remembered_on_the_next_invocation(workspace_above_repo, monkeypatch):
    """"Trust forever after" (owner ruling): asked once, never again. The second invocation runs
    with `Confirm.ask` patched to RAISE, so a re-prompt fails the test rather than passing
    silently on a second yes."""
    _set_tty(monkeypatch, True)
    asked = _prompt_answers(monkeypatch, True)
    first = runner.invoke(app, ["agent", "list"])
    assert first.exit_code == 0, first.output
    assert len(asked) == 1

    _prompt_must_not_fire(monkeypatch)
    second = runner.invoke(app, ["agent", "list", "--json"])

    assert _names(second) == ["global-agent", "ws-agent"]
