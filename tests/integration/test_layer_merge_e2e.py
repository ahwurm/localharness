"""Phase 40 success criteria, driven through the real CLI.

The unit tests in tests/unit/test_config_merge_four_source.py, tests/unit/test_deny_union_layers.py,
tests/unit/test_agent_union_layers.py and tests/unit/test_provider_carveout_workspace.py prove each
SEAM — the fold order, the deny union, the wholesale agent collision, the write carve-out. This file
proves the BEHAVIOR: a user standing in a subdirectory of their project gets that project's MERGED
config, from a command they actually type. A green test on a loader method nobody calls from a
command is a checkmark on nothing; these invocations go through `localharness.cli.app` the way a
terminal does.

The rule they encode, in the owner's words: **the SPECIFIC beats the GENERAL** — the workspace's
word wins wherever the two conflict, and the global layer still governs everything the workspace is
silent about.

What each test name maps to (ROADMAP Phase 40 success criteria):

1. `criterion1` — a value that ONLY the workspace `config.yaml` sets reaches a real command. Before
   this phase a workspace `config.yaml` was never read at all.
2. `criterion4` (two tests) — the ruled merge order, end to end: global `config.yaml` < global
   `overrides.yaml` < workspace `config.yaml` < workspace `overrides.yaml`.
3. `criterion5` — `agent list` shows the union of both layers, the workspace winning a collision.
4. `criterion3` — a workspace that overrides only PART of `provider:` still starts, because nested
   blocks merge key by key instead of being swapped wholesale.
5. `explicit_config_dir` — naming a config directory still skips the workspace entirely; the merge
   landing does not weaken the full-replacement rule.
6. `criterion_layr03` — with nothing up-tree, every one of these commands behaves exactly as it did
   before phase 40.

`provider.available_models` is the probe throughout. `start --list-models` reads it off the MERGED
`HarnessConfig` and exits before any agent, session, tool or GPU is involved, and the configured
`base_url` points at a closed port so the live probe fails fast and the command prints exactly the
configured names. It is a LIST, so the winning layer replaces it wholesale — a clean single-value
readout of "which layer won".

Layouts are built by hand: `.git` markers are plain empty directories, never real repositories, so
the suite needs no git binary and starts no child process (39-01's rule). Every workspace here sits
INSIDE the project the test is standing in, so it loads with no trust prompt — and `Confirm.ask` is
patched to RAISE in every layout, so a prompt appearing would fail the test rather than hang it.
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
from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

# Bare CliRunner: click 8.4 separates stdout from stderr, which is what makes `json.loads(stdout)`
# in the agent-union test a real contract rather than a lucky substring.
runner = CliRunner()

# A closed port, deliberately. `start --list-models` probes the configured base_url first; an
# unreachable one makes the command print exactly the CONFIGURED names, which is the merged value
# under measurement. Port 9 (discard) is refused immediately on loopback, so this costs no timeout.
_UNREACHABLE_BASE_URL = "http://127.0.0.1:9/v1"


# --------------------------------------------------------------------------------- helpers


def _squash(text: str) -> str:
    """Whitespace-stripped comparison — rich wraps at the console width, so a long name can arrive
    in the capture with a newline folded into it (39-04's lesson)."""
    return "".join(text.split())


def _full_config(models: list[str]) -> str:
    """A COMPLETE harness config.yaml whose only interesting key is `provider.available_models`."""
    return yaml.dump(
        {
            "version": "1",
            "provider": {
                "provider_type": "vllm",
                "base_url": _UNREACHABLE_BASE_URL,
                "default_model": "test-model",
                "available_models": models,
            },
        },
        sort_keys=False,
    )


def _models_only(models: list[str]) -> str:
    """The shape an overlay has: a PARTIAL tree, no `version`, no required provider fields.

    This is what `localharness components set provider.available_models ...` writes, and it is also
    the workspace `config.yaml` shape in the partial-override test — the same file content proves
    both that later layers win and that they merge per key instead of replacing the block.
    """
    return yaml.dump({"provider": {"available_models": models}}, sort_keys=False)


def _write_agent(agents_dir: Path, name: str, role: str) -> Path:
    """The three fields `agent create` writes, so these files are valid AgentConfigs."""
    agents_dir.mkdir(parents=True, exist_ok=True)
    path = agents_dir / f"{name}.yaml"
    path.write_text(yaml.dump({"name": name, "role": role, "model": "inherit"}), encoding="utf-8")
    return path


def _prompt_must_not_fire(monkeypatch) -> None:
    """A rule that forbids a prompt is tested by making the prompt RAISE. Silence proves nothing,
    and an unpatched prompt on a path that should not ask would HANG the runner on stdin."""

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an in-project workspace, which never asks")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)


def _layout(
    tmp_path: Path,
    monkeypatch,
    *,
    global_models: list[str],
    workspace: bool = True,
) -> SimpleNamespace:
    """A fake `$HOME` holding the GLOBAL layer, plus a project with the CWD two levels down.

        <home>/.localharness/      GLOBAL layer: config.yaml (+ overrides.yaml, per test)
        <home>/proj/.git/          plain empty dir — the project boundary marker
        <home>/proj/.localharness/ WORKSPACE layer: config.yaml / overrides.yaml / agents, per test
        <home>/proj/src/pkg/       the CWD every invocation runs from

    All three env moves are load-bearing. Both walks stop at `$HOME`, so a real one would let the
    developer's own `~/.localharness` answer these tests; and the autouse conftest fixture sets
    `LOCALHARNESS_HOME`, which the resolver counts as an EXPLICIT selection — leaving it set would
    switch discovery off and every test below would pass for the wrong reason.

    The project lives under the fake `$HOME` on purpose: the up-tree walk stops there, so nothing
    outside the fixture can be discovered even if the machine running the suite has a stray
    `.localharness` in a parent of `tmp_path`.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(_full_config(global_models), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")  # keep rich from wrapping the --json line

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    ws_dir = proj / ".localharness"
    if workspace:
        ws_dir.mkdir(parents=True)

    deep_dir = proj / "src" / "pkg"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)
    _prompt_must_not_fire(monkeypatch)

    # Guards: if either flipped, every assertion below would pass for the wrong reason.
    if workspace:
        assert discover_workspace_dir() == ws_dir.resolve()
        assert workspace_is_within_repo(ws_dir.resolve(), deep_dir)
    else:
        assert discover_workspace_dir() is None

    return SimpleNamespace(
        home=home, global_dir=global_dir, ws_dir=ws_dir, proj=proj, deep_dir=deep_dir
    )


class _StubCounter:
    """doctor asks the counter for its resolved mode; the real one talks to a server."""

    def __init__(self, base_url=None, model=None, provider_type=None, **_kw):
        self.mode = "vllm"
        self.approximate = False


@pytest.fixture
def doctor_offline(monkeypatch):
    """doctor with no network — an unmocked run would spend real timeouts on the closed port."""
    import localharness.agent.context as _ctx

    monkeypatch.setattr(_ctx, "TokenCounter", _StubCounter)
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": [{"id": "test-model", "max_model_len": 131072}]}
    with patch("localharness.cli.doctor_cmd.httpx") as mock_httpx:
        mock_httpx.get.return_value = resp
        mock_httpx.post.return_value = resp
        yield mock_httpx


# --------------------------------------------- criterion 1: a workspace config.yaml is finally read


def test_criterion1_workspace_config_value_reaches_a_real_command(tmp_path, monkeypatch):
    """MERG-01 from the command line, and the first invocation in this project's history where a
    workspace `config.yaml` is read at all.

    `workspace-only-model` exists ONLY in `proj/.localharness/config.yaml`, so its appearance in
    `start --list-models` run from `proj/src/pkg` is the same claim as "the workspace config file
    was discovered, read and merged". `global-only-model` being ABSENT is the other half: a
    workspace value that merely appeared alongside the global one would not be an override.

    Asserted around `_auto_migrate_deny_defaults`, which `start` runs BEFORE the loader is built and
    which can rewrite the GLOBAL config.yaml on first run (it folds newer shipped deny defaults into
    `org.permissions.deny_patterns` and stamps a defaults revision). It is not disabled here: the
    test instead pins that it leaves `provider.available_models` alone, never touches the WORKSPACE
    file, and that a second invocation on the rewritten global config still resolves the same way.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["global-only-model"])
    ws_config = layout.ws_dir / "config.yaml"
    ws_config.write_text(_models_only(["workspace-only-model"]), encoding="utf-8")
    ws_bytes_before = ws_config.read_bytes()

    result = runner.invoke(app, ["start", "--list-models"])

    assert result.exit_code == 0, result.output
    assert "workspace-only-model" in _squash(result.stdout)
    assert "global-only-model" not in _squash(result.stdout)

    # The migration is allowed to have run; it is not allowed to have touched either probe.
    global_after = yaml.safe_load((layout.global_dir / "config.yaml").read_text(encoding="utf-8"))
    assert global_after["provider"]["available_models"] == ["global-only-model"]
    assert ws_config.read_bytes() == ws_bytes_before

    # ...and the answer is not a first-run artifact of the pre-migration config file.
    second = runner.invoke(app, ["start", "--list-models"])
    assert second.exit_code == 0, second.output
    assert "workspace-only-model" in _squash(second.stdout)
    assert "global-only-model" not in _squash(second.stdout)


# ------------------------------------------------------- criterion 4: the ruled order, end to end


def test_criterion4_workspace_config_beats_global_overrides_end_to_end(tmp_path, monkeypatch):
    """The owner's ruling (Option A, 2026-09-03) as a user experiences it.

    Three layers set the same key. The ruled order is global `config.yaml` < global
    `overrides.yaml` < workspace `config.yaml`, so the workspace file wins and the GLOBAL OVERLAY —
    the value `localharness components set` writes, and the layer a reasonable person might expect
    to be "most specific" because it is most recently typed — is the one expected to LOSE. That is
    the whole content of the ruling: specificity is about SCOPE, not about recency.

    All three absences are asserted, not just the winner's presence: "the right one is printed"
    would pass with all three printed, which is a merge that resolves nothing.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["from-global-config"])
    (layout.global_dir / "overrides.yaml").write_text(
        _models_only(["from-global-overrides"]), encoding="utf-8"
    )
    (layout.ws_dir / "config.yaml").write_text(
        _models_only(["from-workspace-config"]), encoding="utf-8"
    )

    result = runner.invoke(app, ["start", "--list-models"])

    assert result.exit_code == 0, result.output
    out = _squash(result.stdout)
    assert "from-workspace-config" in out
    assert "from-global-config" not in out
    assert "from-global-overrides" not in out


def test_criterion4_workspace_overrides_beats_workspace_config_end_to_end(tmp_path, monkeypatch):
    """The top of the ladder. Four sources now set the same key and the workspace's own
    `overrides.yaml` is the highest-priority layer, so it wins over the workspace `config.yaml`
    sitting next to it — the same relationship the global pair has, one scope down."""
    layout = _layout(tmp_path, monkeypatch, global_models=["from-global-config"])
    (layout.global_dir / "overrides.yaml").write_text(
        _models_only(["from-global-overrides"]), encoding="utf-8"
    )
    (layout.ws_dir / "config.yaml").write_text(
        _models_only(["from-workspace-config"]), encoding="utf-8"
    )
    (layout.ws_dir / "overrides.yaml").write_text(
        _models_only(["from-workspace-overrides"]), encoding="utf-8"
    )

    result = runner.invoke(app, ["start", "--list-models"])

    assert result.exit_code == 0, result.output
    out = _squash(result.stdout)
    assert "from-workspace-overrides" in out
    assert "from-workspace-config" not in out
    assert "from-global-overrides" not in out
    assert "from-global-config" not in out


# ------------------------------------------------------------ criterion 5: the agent roster union


def test_criterion5_agent_list_shows_the_union_with_the_workspace_winning(tmp_path, monkeypatch):
    """AGNT-01 from a command: the roster is the UNION of both layers, not a replacement of one by
    the other, and a colliding name resolves to the workspace's file.

    `reporter` is global-only, `builder` is workspace-only, `deployer` exists in both. A layer that
    replaced the other wholesale would drop one of the two singletons; a merge that preferred the
    global layer would report `GLOBAL DEPLOYER`.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["global-only-model"])
    _write_agent(layout.global_dir / "agents", "reporter", "GLOBAL REPORTER")
    _write_agent(layout.global_dir / "agents", "deployer", "GLOBAL DEPLOYER")
    _write_agent(layout.ws_dir / "agents", "deployer", "WORKSPACE DEPLOYER")
    _write_agent(layout.ws_dir / "agents", "builder", "WORKSPACE BUILDER")

    result = runner.invoke(app, ["agent", "list", "--json"])

    assert result.exit_code == 0, result.output
    roster = json.loads(result.stdout)
    assert {a["name"] for a in roster} == {"reporter", "deployer", "builder"}
    deployer = next(a for a in roster if a["name"] == "deployer")
    assert deployer["role"] == "WORKSPACE DEPLOYER"


# -------------------------------------------------- criterion 3: a partial provider override works


def test_criterion3_workspace_provider_partial_override_still_starts(tmp_path, monkeypatch):
    """MERG-03's hardware-truth half, proven by the command running at all.

    The workspace sets ONLY `provider.available_models` — no `provider_type`, no `base_url`, no
    `default_model`, all of which `ProviderConfig` requires. If nested blocks were replaced
    wholesale instead of merged key by key, the merged config would be missing three required
    fields, `HarnessConfig` validation would fail and `start` would exit 1 with a config error. So
    exit code 0 is itself an assertion here, and the printed name says which layer supplied the one
    key the workspace did set.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["global-only-model"])
    (layout.ws_dir / "config.yaml").write_text(_models_only(["ws-model"]), encoding="utf-8")

    result = runner.invoke(app, ["start", "--list-models"])

    assert result.exit_code == 0, result.output
    assert "ws-model" in _squash(result.stdout)
    assert "global-only-model" not in _squash(result.stdout)
    assert "Cannot load config" not in result.output


# --------------------------------------- LAYR-02: an explicit config dir still skips the whole walk


def test_explicit_config_dir_still_skips_the_merge(tmp_path, monkeypatch):
    """Naming a config directory replaces the config layer OUTRIGHT — discovery never runs, so the
    workspace two levels up contributes nothing even though the walk would find it.

    The layout is the ruled-order test's, so the global overlay is present and would be beaten by
    the workspace `config.yaml` on the default path. Here it WINS, which is what shows the workspace
    was skipped rather than merely outranked. Landing the merge must not quietly turn
    `--config-dir` back into a partial selection.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["from-global-config"])
    (layout.global_dir / "overrides.yaml").write_text(
        _models_only(["from-global-overrides"]), encoding="utf-8"
    )
    (layout.ws_dir / "config.yaml").write_text(
        _models_only(["from-workspace-config"]), encoding="utf-8"
    )
    argv = ["start", "--list-models", "--config-dir", str(layout.global_dir)]

    result = runner.invoke(app, argv)

    assert result.exit_code == 0, result.output
    out = _squash(result.stdout)
    assert "from-global-overrides" in out
    assert "from-workspace-config" not in out
    assert "Workspacelayer:" not in out


# ------------------------------------------------ LAYR-03: nothing up-tree, nothing changes at all


def test_criterion_layr03_nothing_uptree_is_unchanged(tmp_path, monkeypatch, doctor_offline):
    """The standing invariant at the end of every phase, in its user-visible form for phase 40.

    A user with no `.localharness/` anywhere up-tree gets the global config, the global roster and
    none of the new vocabulary — across all three commands this phase touches. `doctor` printing no
    `Workspace layer:` line is the sharp end: the new lines are CONDITIONAL, not merely empty, so a
    workspace-less user's output is byte-for-byte what it was before phase 40.
    """
    layout = _layout(tmp_path, monkeypatch, global_models=["global-only-model"], workspace=False)
    _write_agent(layout.global_dir / "agents", "reporter", "GLOBAL REPORTER")

    models = runner.invoke(app, ["start", "--list-models"])
    assert models.exit_code == 0, models.output
    assert "global-only-model" in _squash(models.stdout)

    listed = runner.invoke(app, ["agent", "list", "--json"])
    assert listed.exit_code == 0, listed.output
    assert [a["name"] for a in json.loads(listed.stdout)] == ["reporter"]

    doctored = runner.invoke(app, ["doctor"])
    assert "Workspace layer:" not in doctored.output
    assert "Workspace layer:" not in models.output
