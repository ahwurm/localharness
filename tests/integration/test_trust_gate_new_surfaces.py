"""LAYR-05's trust gate, on the surfaces that were wired AFTER it.

`test_workspace_discovery_e2e.py` proves the gate end to end on the two commands that existed when
phase 39 built it: `agent list` and `validate`. Three more readers of the workspace layer have
landed since — `doctor`'s layer report (39-05, extended by 43-05), `components list` / `get`, and
`config show` (43-04) — and every one of them calls `resolve_workspace_layer` for itself rather
than inheriting a decision from those two. "Same helper, therefore the same gate" is an INFERENCE
about source code, not a measurement: a surface that resolved the workspace before the gate, or
handed `discover_workspace_dir()` straight to `layered_catalogue`, would read a stranger's config
while every existing test in this milestone stayed green. This file is the measurement.

The claim, once per surface: standing in a repository whose nearest `.localharness/` sits ABOVE its
root (`workspace_above_repo` — the only shape the gate exists for) with no terminal to ask, the
command still runs and shows nothing that workspace set, and no verdict is recorded.

Both directions, because half of this claim grades nothing. An absent marker is exactly what a
never-implemented feature also produces, so the two controls at the end record trust and watch
`config show` and `components get` pick that same value UP — which is what makes its absence above
mean "the gate held" rather than "the layer was never readable from here".

Every helper AND the fixture are imported from the LAYR-05 e2e, never copied (41-06's rule): a
copied fixture drifts from the one the gate is really graded against, and then grades its copy.
"""
from __future__ import annotations

import json

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config import trust
from localharness.registry import LAYER_DEFAULT, LAYER_WORKSPACE_CONFIG

# The LAYR-05 fixture and the four rules-as-helpers around it, imported whole:
#   `workspace_above_repo` — `.localharness/` at `outer/`, `.git` at `outer/inner/`, CWD
#                            `outer/inner/sub`, under a fake $HOME holding the global layer
#   `doctor_offline`       — doctor's network boundaries stubbed (real timeouts, otherwise)
#   `_set_tty`             — pin whether the resolver believes a terminal is attached; click's
#                            isolation makes patching the real `sys.stdin` useless (39-06)
#   `_prompt_must_not_fire`— a forbidden prompt must RAISE; silence proves nothing, and an
#                            unpatched prompt would hang the runner on stdin
#   `_squash`              — whitespace-stripped comparison, for rich's column padding
# The two fixtures are referenced by NAME in the signatures below, which is how pytest resolves
# them and why they are imported but never called here.
from tests.integration.test_workspace_discovery_e2e import (  # noqa: F401
    _prompt_must_not_fire,
    _set_tty,
    _squash,
    doctor_offline,
    workspace_above_repo,
)

runner = CliRunner()

# One value, set ONLY by the out-of-project workspace, on a key every one of these commands can
# print. Every assertion below is about where this string does and does not appear.
_MARKER = "TRUST-GATE-LEAK-MARKER"
_MARKED_KEY = "org.name"


@pytest.fixture
def marked_workspace(workspace_above_repo):
    """`workspace_above_repo` plus the one value that makes a leak visible.

    The LAYR-05 fixture plants an agent file, which is what `agent list` and `validate` read.
    These three surfaces read CONFIG, so the workspace needs a config key of its own — added here
    rather than in the shared fixture, so the tests already graded against that fixture keep
    grading exactly what they did before.
    """
    (workspace_above_repo.ws_dir / "config.yaml").write_text(
        yaml.dump({"org": {"name": _MARKER}}), encoding="utf-8"
    )
    return workspace_above_repo


# ------------------------------------------------------ inert: the gate reaches all three surfaces


def test_doctor_is_inert_on_a_workspace_from_outside_the_project(
    marked_workspace, doctor_offline, monkeypatch
):
    """doctor names the layer it took (39-05) and reports which file won each key (43-05) — both
    printed from a `resolve_workspace_layer` call of its own, and neither ever composed with the
    gate until now. Untrusted and no terminal: doctor says nothing about a workspace at all.

    No exit-code assertion: doctor's code reports the health of the machine it runs on, which is
    not this file's claim.
    """
    _set_tty(monkeypatch, False)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["doctor"])

    assert "Workspace layer:" not in result.output, result.output
    assert _MARKER not in result.output
    assert "ws-agent" not in result.output
    assert trust.is_trusted(marked_workspace.ws_dir) is None


def test_components_list_is_inert_on_a_workspace_from_outside_the_project(
    marked_workspace, monkeypatch
):
    """`components list` builds its catalogue from the workspace directly, so a gate that skipped
    it would print the stranger's value in the table AND credit it to the `workspace-config`
    band — the CLI stating, in its own vocabulary, that it loaded a layer it must not have."""
    _set_tty(monkeypatch, False)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["components", "list"])

    assert result.exit_code == 0, result.output
    assert _MARKER not in result.output
    assert LAYER_WORKSPACE_CONFIG not in result.output
    assert trust.is_trusted(marked_workspace.ws_dir) is None
    assert not trust.trust_store_path().exists()


def test_components_get_is_inert_on_a_workspace_from_outside_the_project(
    marked_workspace, monkeypatch
):
    """The single-key form, which answers "what is this value, and who set it". Both halves are
    asserted: the value is not the workspace's, and the winning layer is the compiled-in default —
    the answer a machine with no `.localharness/` above it would get."""
    _set_tty(monkeypatch, False)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["components", "get", _MARKED_KEY])

    assert result.exit_code == 0, result.output
    assert _MARKER not in result.output
    assert f"layer:{LAYER_DEFAULT}" in _squash(result.output)
    assert LAYER_WORKSPACE_CONFIG not in result.output


def test_config_show_is_inert_on_a_workspace_from_outside_the_project(
    marked_workspace, monkeypatch
):
    """`config show` prints the merge order itself, so this surface can leak the workspace twice:
    in a value, and in the header row naming the file it came from. Neither appears.

    `--json` is the machine contract (43-02): the whole stdout document must parse, with the
    gate's notice on stderr where a `jq` caller never sees it.
    """
    _set_tty(monkeypatch, False)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert _MARKER not in result.stdout
    layer_paths = " ".join(layer["path"] for layer in payload["layers"])
    assert str(marked_workspace.ws_dir) not in layer_paths
    # "non-interactive", not the old "no terminal to ask": there IS a terminal on a --json run
    # with a tty, and the honest reason nothing was asked is that this is machine output (F11).
    assert "non-interactive" in result.stderr
    assert "non-interactive" not in result.stdout
    assert trust.is_trusted(marked_workspace.ws_dir) is None
    assert not trust.trust_store_path().exists()


# ------------------------------------------------------- controls: the layer IS readable from here


def test_config_show_reads_the_workspace_once_trust_is_recorded(marked_workspace, monkeypatch):
    """The control for every "not in the output" above. With one recorded yes — the same verdict
    the prompt writes — `config show` reports the workspace's value and credits its config.yaml.
    So the marker's absence above is the gate holding, not a value nothing could ever read.

    Trust is already recorded, so a prompt here would be a re-ask of a settled question: it raises.
    """
    trust.record_trust(marked_workspace.ws_dir, True)
    _set_tty(monkeypatch, True)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    key = next(k for k in payload["keys"] if k["path"] == _MARKED_KEY)
    assert key["value"] == _MARKER
    assert key["layer"] == LAYER_WORKSPACE_CONFIG
    assert str(marked_workspace.ws_dir) in " ".join(la["path"] for la in payload["layers"])


def test_components_get_credits_the_workspace_once_trust_is_recorded(
    marked_workspace, monkeypatch
):
    """The same control on the other command family. `components` reaches the workspace by a
    different call than `config show` does, so one control cannot speak for both.
    """
    trust.record_trust(marked_workspace.ws_dir, True)
    _set_tty(monkeypatch, True)
    _prompt_must_not_fire(monkeypatch)

    result = runner.invoke(app, ["components", "get", _MARKED_KEY])

    assert result.exit_code == 0, result.output
    assert _MARKER in result.output
    assert f"layer:{LAYER_WORKSPACE_CONFIG}" in _squash(result.output)
