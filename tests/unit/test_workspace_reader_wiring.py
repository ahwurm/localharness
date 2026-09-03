"""Phase 39 (LAYR-01/03/04): the four config READERS actually read the discovered workspace layer.

Plans 39-01/02/04 built discovery, the two-layer loader and the decision table, and each of them
deliberately shipped with ZERO callers. That is correct code nothing reaches — which is a
checkmark on nothing. These tests are the proof that `start`, `validate`, `model` and `doctor`
now hand `resolve_workspace_layer()`'s answer to `ConfigLoader`, so a workspace agent is really
in the roster of a command a user runs.

How each wiring is proven, in order of strength:

- `start` and `model` are proven at the CALL: a `ConfigLoader` spy records the kwargs the command
  actually constructed its loader with, and the roster assertions are then built from THOSE
  recorded kwargs. Nothing here is a source-text grep, so a refactor that keeps the string and
  drops the reach would still fail.
- `validate` and `doctor` are proven at the OUTPUT, driven through `CliRunner` end to end.
- Every assertion in this file was mutation-checked (39-05): the `local_config_dir=` kwarg was
  deleted from each of the four command files in turn and a test here went red each time. A
  wiring assertion that cannot fail proves nothing about reachability (38-05's standing rule).

Fixture note: the shape is 39-04's `in_repo_project` — both env overrides cleared (either one
counts as an explicit selection and would switch discovery off), a fake `$HOME` holding the
global layer, the workspace two levels above the CWD, and a `.git` DIRECTORY created by hand at
the project root. That marker is what makes this the IN-PROJECT case, so the layer loads with no
prompt and writes nothing to the trust store — which is why no test here records trust, and why
`Confirm.ask` is patched to raise: a prompt on this path must fail loudly, not hang.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml
from typer.testing import CliRunner

from localharness.cli import start_cmd
from localharness.cli.app import app
from localharness.config.loader import ConfigLoader
from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

runner = CliRunner()

# Copied from tests/conftest.py rather than imported: this fixture plants its own $HOME and must
# stay readable as the whole of what the global layer contains.
_MINIMAL_CONFIG_YAML = (
    "version: '1'\n"
    "provider:\n"
    "  provider_type: vllm\n"
    "  base_url: http://localhost:8000/v1\n"
    "  default_model: test-model\n"
)


def _squash(text: str) -> str:
    """Whitespace-stripped comparison — rich wraps at the console width, so a long tmp path
    arrives in the capture with newlines folded into it (39-04's lesson)."""
    return "".join(text.split())


def _write_agent(agents_dir: Path, name: str, role: str) -> Path:
    """The three fields `agent create` writes, so these files are valid AgentConfigs and
    `validate` reports them as valid rather than merely as listed."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    path.write_text(
        yaml.dump({"name": name, "role": role, "model": "inherit"}), encoding="utf-8"
    )
    return path


def _prompt_must_not_fire(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an IN-PROJECT workspace")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


@pytest.fixture
def in_repo_project(tmp_path, monkeypatch):
    """A global layer under a fake `$HOME`, a workspace two levels above the CWD, inside a repo.

    Returns the workspace, the global dir and the deep CWD. The workspace holds `ws-agent` (only
    it has one) and a colliding `global-agent` with a different role, so both "the workspace adds"
    and "the workspace wins" are observable.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)

    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(_MINIMAL_CONFIG_YAML, encoding="utf-8")
    _write_agent(global_dir / "agents", "global-agent", "global role")
    monkeypatch.setenv("HOME", str(home))

    workspace = tmp_path / "proj" / ".localharness"
    _write_agent(workspace / "agents", "ws-agent", "workspace role")
    _write_agent(workspace / "agents", "global-agent", "workspace override role")
    # A `.git` DIRECTORY by hand: the boundary check reads the marker off the filesystem and
    # never shells out, so no repository has to be created for real.
    (tmp_path / "proj" / ".git").mkdir()

    deep = tmp_path / "proj" / "src" / "pkg"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)
    _prompt_must_not_fire(monkeypatch)

    # Guards: if either of these flipped, the tests below would pass for the wrong reason.
    assert discover_workspace_dir() == workspace.resolve()
    assert workspace_is_within_repo(workspace, deep)
    return SimpleNamespace(
        workspace=workspace.resolve(), global_dir=global_dir, deep=deep
    )


def _spy_config_loader(monkeypatch, target: str) -> list[dict]:
    """Record the kwargs a command builds its ConfigLoader with, then stop the command dead.

    `target` names where that command looks the class up (its own module for a module-level
    import, `localharness.config.loader` for a lazy in-function one). Raising from
    `load_harness` ends the run right after construction, so nothing boots a session or opens a
    socket — and the kwargs collected here are the real ones, not a source-text guess.
    """
    calls: list[dict] = []

    class _Spy(ConfigLoader):
        def __init__(self, **kwargs):
            calls.append(dict(kwargs))
            super().__init__(**kwargs)

        def load_harness(self):
            raise RuntimeError("stopped right after construction (test spy)")

    monkeypatch.setattr(target, _Spy)
    return calls


def _run_start(config_dir: str | None = None) -> None:
    """`start` up to its loader, and no further — the spy makes load_harness fail, which start
    turns into `typer.Exit(1)`."""
    with pytest.raises(typer.Exit):
        asyncio.run(start_cmd._start_async(None, False, False, config_dir))


class _StubCounter:
    """doctor asks the counter for its resolved mode; the real one talks to a server."""

    def __init__(self, base_url=None, model=None, provider_type=None, **_kw):
        self.mode = "vllm"
        self.approximate = False


@pytest.fixture
def doctor_offline(monkeypatch):
    """doctor with no network: a stubbed counter and a mocked httpx. The workspace line prints
    before any of it, but an unmocked run would spend real timeouts on localhost:8000."""
    import localharness.agent.context as _ctx

    monkeypatch.setattr(_ctx, "TokenCounter", _StubCounter)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": "test-model", "max_model_len": 131072}]}
    with patch("localharness.cli.doctor_cmd.httpx") as mock_httpx:
        mock_httpx.get.return_value = resp
        mock_httpx.post.return_value = resp
        yield mock_httpx


# ------------------------------------------------------------------ start: the load-bearing one


def test_start_roster_includes_workspace_agent_from_deep_subdirectory(
    in_repo_project, monkeypatch
):
    """Success criterion 1, end to end: standing two directories deep inside a project, the
    roster `start` builds holds that project's agent as well as the global one."""
    calls = _spy_config_loader(monkeypatch, "localharness.config.loader.ConfigLoader")

    _run_start()

    assert calls, "start never constructed a ConfigLoader"
    assert calls[0]["local_config_dir"] == in_repo_project.workspace
    # The roster is built from the kwargs start ACTUALLY used, so this is start's roster.
    names = {a["name"] for a in ConfigLoader(**calls[0]).discover_agents()}
    assert names == {"global-agent", "ws-agent"}


def test_workspace_agent_wins_a_name_collision(in_repo_project, monkeypatch):
    """LAYR-04, at roster level: same stem in both layers, the workspace's file is the one that
    survives the by-stem overwrite in `discover_agents`."""
    calls = _spy_config_loader(monkeypatch, "localharness.config.loader.ConfigLoader")

    _run_start()

    roster = {a["name"]: a for a in ConfigLoader(**calls[0]).discover_agents()}
    assert roster["global-agent"]["role"] == "workspace override role"
    assert roster["ws-agent"]["role"] == "workspace role"


# ----------------------------------------------------------------------- validate, model, doctor


def test_validate_reports_workspace_agent_file(in_repo_project):
    """`ws-agent.yaml` exists ONLY in the workspace, so naming it in validate's results is the
    same claim as "validate checked the workspace's files"."""
    result = runner.invoke(app, ["validate"])

    assert result.exit_code == 0, result.output
    assert "ws-agent.yaml" in result.output


def test_model_command_passes_the_workspace_layer_to_its_loader(in_repo_project, monkeypatch):
    """`model` reads agents through the same loader; it must see the same layer the others do.

    Proven at the construction, not the output: the real command then probes a live endpoint.
    """
    calls = _spy_config_loader(monkeypatch, "localharness.cli.model_cmd.ConfigLoader")

    runner.invoke(app, ["model"])

    assert calls, "model never constructed a ConfigLoader"
    assert calls[0]["local_config_dir"] == in_repo_project.workspace


def test_doctor_prints_the_workspace_layer_when_one_applies(in_repo_project, doctor_offline):
    """Success criterion 2's visible half: doctor NAMES the layer it chose. A silent merge of
    two config layers is exactly the confusion this phase exists to avoid."""
    result = runner.invoke(app, ["doctor"])

    assert "Workspace layer:" in result.output
    assert _squash(str(in_repo_project.workspace)) in _squash(result.output)
    assert "Global layer:" in result.output
    assert _squash(str(in_repo_project.global_dir)) in _squash(result.output)


def test_doctor_prints_no_workspace_line_when_none_applies(
    in_repo_project, doctor_offline, tmp_path, monkeypatch
):
    """LAYR-03: with nothing up-tree, doctor's output is what v0.12 printed — the new lines are
    conditional, so a workspace-less user sees no new vocabulary at all."""
    elsewhere = tmp_path / "elsewhere" / "deep"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    # Guard: a stray `.localharness` anywhere above tmp would make this pass for no reason.
    assert discover_workspace_dir() is None

    result = runner.invoke(app, ["doctor"])

    assert "Workspace layer:" not in result.output
    assert "Global layer:" not in result.output


# ------------------------------------------------------------------------------ the two carve-outs


def test_explicit_config_dir_hides_the_workspace_agent(in_repo_project, monkeypatch):
    """LAYR-02: naming a config dir is a FULL replacement — discovery never even runs, so the
    workspace agent is absent from every one of these commands."""
    explicit = str(in_repo_project.global_dir)

    validated = runner.invoke(app, ["validate", "--config-dir", explicit])
    assert validated.exit_code == 0, validated.output
    assert "ws-agent.yaml" not in validated.output

    calls = _spy_config_loader(monkeypatch, "localharness.config.loader.ConfigLoader")
    _run_start(explicit)
    assert calls[0]["local_config_dir"] is None
    assert "ws-agent" not in {a["name"] for a in ConfigLoader(**calls[0]).discover_agents()}


def test_in_repo_workspace_loads_without_a_trust_record(in_repo_project):
    """The owner's 2026-09-03 ruling, observed from the CLI: your own project is not a trust
    boundary, so the layer applies AND no verdict about it is ever written down."""
    from localharness.config import trust

    ws = in_repo_project.workspace

    result = runner.invoke(app, ["validate"])

    assert "ws-agent.yaml" in result.output  # the layer applied...
    assert trust.is_trusted(ws) is None  # ...and nothing was recorded
    assert not trust.trust_store_path().exists()
