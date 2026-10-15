"""`components` sees the workspace layer, and `set` says where it writes (dogfood F4).

`components_cmd._build_loader()` was `return ConfigLoader()` — the ONE reader in the CLI that never
adopted the workspace wiring doctor/validate/agent all use. Inside a project it reported the wrong
VALUE and the wrong LAYER, and phase 42 raised the stakes: `recall_scope`'s own field description
tells users to reach for `components set agent.memory.recall_scope`, which — run inside a project —
reconfigures the WHOLE MACHINE while the getter misreports both value and layer.

Global write-targeting is accepted v0.13 policy. The gap this file closes is that nothing SAID so
at the moment it happened, and that `get` lied about what was in force.

Everything here drives the real Typer app through `CliRunner` from a SUBDIRECTORY of the project —
the walk up-tree is the whole point, and a test run from the project root would pass for a
`./.localharness` literal read that never discovers anything. Layout recipe from
`test_config_error_attribution.py`: both discovery walks stop at `$HOME`,
`LOCALHARNESS_DIR`/`LOCALHARNESS_HOME` are cleared (either counts as an explicit selection and
switches discovery OFF, which would make every assertion below pass for the wrong reason), `.git`
is a plain empty directory so no git binary is needed, and `Confirm.ask` RAISES so an unexpected
trust prompt fails the test instead of hanging it on stdin.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config.paths import discover_workspace_dir

runner = CliRunner()

_GLOBAL_NAME = "GLOBAL-NAME-LOSES"
_WORKSPACE_NAME = "WORKSPACE-NAME-WINS"

_GLOBAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:9/v1",
        "default_model": "test-model",
    },
    "org": {"name": _GLOBAL_NAME, "log_level": "info"},
}


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _layout(
    tmp_path: Path,
    monkeypatch,
    fake_home,
    *,
    ws_config: Optional[dict] = None,
    workspace: bool = True,
) -> SimpleNamespace:
    """A fake `$HOME` holding the GLOBAL layer, plus a project whose CWD is two levels down.

        <home>/.localharness/      GLOBAL layer: config.yaml (org.name = GLOBAL-NAME-LOSES)
        <home>/proj/.git/          plain empty dir — the project boundary marker
        <home>/proj/.localharness/ WORKSPACE layer: config.yaml, per test
        <home>/proj/src/pkg/       the CWD every invocation runs from
    """
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.safe_dump(_GLOBAL_CONFIG), encoding="utf-8")
    fake_home(home)
    # rich wraps at the console width; a wrapped path is not a printed path.
    monkeypatch.setenv("COLUMNS", "400")

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    ws_dir = proj / ".localharness"
    if workspace:
        ws_dir.mkdir(parents=True)
        if ws_config is not None:
            (ws_dir / "config.yaml").write_text(yaml.safe_dump(ws_config), encoding="utf-8")

    deep_dir = proj / "src" / "pkg"
    deep_dir.mkdir(parents=True)
    monkeypatch.chdir(deep_dir)

    def _boom(*args, **kwargs):
        raise AssertionError("prompted for trust on an in-project workspace, which never asks")

    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    # Guard: if discovery flipped, every assertion below would pass for the wrong reason.
    assert discover_workspace_dir() == (ws_dir.resolve() if workspace else None)

    return SimpleNamespace(
        home=home,
        global_dir=global_dir,
        global_ovl=global_dir / "overrides.yaml",
        ws_dir=ws_dir,
        proj=proj,
        deep_dir=deep_dir,
    )


# ------------------------------------------------------------------ #
# F4 — `get` from a subdirectory
# ------------------------------------------------------------------ #


def test_f4_get_from_a_subdirectory_reports_the_workspace_value_and_band(tmp_path, monkeypatch, fake_home):
    """The dogfood repro. Three assertions, and all three are load-bearing.

    "the workspace value is printed" also passes when BOTH values are printed, and "the band is
    not global" also passes when the band is `default`. The negative assertion on the global value
    is what proves the workspace one replaced it rather than joining it.
    """
    _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["components", "get", "org.name"])

    assert result.exit_code == 0, result.output
    assert _WORKSPACE_NAME in result.stdout, result.stdout
    assert "workspace-config" in result.stdout, result.stdout
    assert _GLOBAL_NAME not in result.stdout, result.stdout


def test_f4_get_json_twin_carries_the_workspace_band(tmp_path, monkeypatch, fake_home):
    """Machine output tells the same truth. `--json` also means non-interactive: a workspace from
    outside the project must never stop to prompt on a payload stream."""
    _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["components", "get", "--json", "org.name"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["layer"] == "workspace-config"
    assert payload["value"] == _WORKSPACE_NAME


def test_list_layer_filter_selects_exactly_the_workspace_set_paths(tmp_path, monkeypatch, fake_home):
    """`--layer workspace-config` is a band a user can now ask for by name, and it returns the
    paths that file set — exactly those, not "at least" those."""
    _layout(
        tmp_path,
        monkeypatch,
        fake_home,
        ws_config={"org": {"name": _WORKSPACE_NAME, "log_level": "debug"}},
    )

    result = runner.invoke(app, ["components", "list", "--json", "--layer", "workspace-config"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert sorted(e["path"] for e in payload) == ["org.log_level", "org.name"]
    assert {e["current_value"] for e in payload} == {_WORKSPACE_NAME, "debug"}


# ------------------------------------------------------------------ #
# F4's other half — `set` names its target
# ------------------------------------------------------------------ #


def test_set_names_the_global_file_it_wrote_and_says_it_is_machine_wide(tmp_path, monkeypatch, fake_home):
    """The write target is unchanged v0.13 policy; what changes is that the command SAYS so.

    The third assertion is the one that makes the first two mean something: nothing may be written
    into the workspace, so the note is a disclosure and not a description of a split write.
    """
    layout = _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"log_level": "debug"}})
    before = _listing(layout.ws_dir)

    result = runner.invoke(app, ["components", "set", "org.name", "SET-BY-CLI"])

    assert result.exit_code == 0, result.output
    assert str(layout.global_ovl) in result.stdout, result.stdout
    assert "MACHINE-WIDE" in result.stdout, result.stdout
    assert _listing(layout.ws_dir) == before, "set wrote into the workspace"
    assert yaml.safe_load(layout.global_ovl.read_text())["org"]["name"] == "SET-BY-CLI"
    # F2 on this command's own output: no band a user reads may be one of the retired names.
    # `user` survives ONLY in the JSON payload and the audit event (a persisted schema).
    assert "layer: user" not in result.stdout, result.stdout


def test_set_warns_that_the_workspace_still_wins_for_a_workspace_owned_path(tmp_path, monkeypatch, fake_home):
    """A machine-wide write to a key THIS project already owns changes nothing here, and saying
    only "machine-wide" would leave the user watching a value that never moves."""
    layout = _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["components", "set", "org.name", "SET-BY-CLI"])

    assert result.exit_code == 0, result.output
    assert "workspace-config" in result.stdout, result.stdout
    assert str(layout.ws_dir / "config.yaml") in result.stdout, result.stdout

    # …and the effective value really is unmoved, which is what the warning is about.
    after = runner.invoke(app, ["components", "get", "--json", "org.name"])
    assert json.loads(after.stdout)["value"] == _WORKSPACE_NAME


def test_set_never_takes_its_audit_path_from_the_workspace_config(tmp_path, monkeypatch, fake_home):
    """A project no longer chooses where the audit trail goes: `org.audit_log_path` is
    machine-level only (a file written anywhere on disk), so `set`'s workspace-aware loader drops
    the workspace's value and the audit lands at the machine's path — the compiled-in default here,
    under the global config dir. The workspace's other org setting (`log_level`) still merges; the
    command succeeds as before.
    """
    layout = _layout(
        tmp_path,
        monkeypatch,
        fake_home,
        ws_config={"org": {"log_level": "debug", "audit_log_path": "ws-audit.jsonl"}},
    )

    result = runner.invoke(app, ["components", "set", "org.name", "SET-BY-CLI"])

    assert result.exit_code == 0, result.output
    assert (layout.global_dir / "audit.jsonl").exists(), _listing(layout.global_dir)
    assert not (layout.global_dir / "ws-audit.jsonl").exists(), (
        "the audit landed where the project asked — a project chose where the audit trail goes"
    )


def test_set_json_carries_the_write_target_and_keeps_the_audit_vocabulary(tmp_path, monkeypatch, fake_home):
    """`target` is new; `layer` is NOT renamed. It mirrors the ComponentMutated audit event's own
    vocabulary, which this phase deliberately leaves alone (a persisted log's schema)."""
    layout = _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"log_level": "debug"}})

    result = runner.invoke(app, ["components", "set", "--json", "org.name", "SET-BY-CLI"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["target"] == str(layout.global_ovl)
    assert payload["layer"] == "user"


def test_set_with_no_workspace_says_nothing_about_a_project(tmp_path, monkeypatch, fake_home):
    """LAYR-03 on the human output: with nothing up-tree the extra note must not appear at all."""
    layout = _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "set", "org.name", "SET-BY-CLI"])

    assert result.exit_code == 0, result.output
    assert str(layout.global_ovl) in result.stdout, result.stdout
    assert "MACHINE-WIDE" not in result.stdout, result.stdout


# ------------------------------------------------------------------ #
# The two controls: explicit selection, and no workspace at all
# ------------------------------------------------------------------ #


def test_config_dir_skips_discovery_entirely(tmp_path, monkeypatch, fake_home):
    """LAYR-02: an explicit --config-dir is a FULL replacement, not an addition. The workspace is
    two directories up and must not be seen."""
    layout = _layout(tmp_path, monkeypatch, fake_home, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(
        app,
        ["components", "get", "--json", "org.name", "--config-dir", str(layout.global_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["value"] == _GLOBAL_NAME
    assert payload["layer"] == "global-config"


def test_no_workspace_control_is_unchanged_except_for_the_spelling(tmp_path, monkeypatch, fake_home):
    """With nothing up-tree, attribution is exactly what it was before v0.13, honestly spelled."""
    _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "get", "--json", "org.name"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["value"] == _GLOBAL_NAME
    assert payload["layer"] == "global-config"


def test_build_loader_is_untouched_so_its_six_external_callers_are_too():
    """`_build_loader()` has six callers OUTSIDE this module (autoresearch adoption/experiment/loop,
    propose_cmd, autoresearch_cmd x2). Wiring discovery INTO it would make all six workspace-aware —
    a behavior change no requirement asks for, in the autoresearch/experiment path. The three
    `components` commands got a SECOND, explicitly-named constructor instead.

    Asserted on the code, not on the file text: a docstring saying "not workspace-aware" must not
    be readable as the wiring (43-02's lesson, and this docstring says exactly that).
    """
    import ast
    import inspect

    from localharness.cli import components_cmd

    tree = ast.parse(inspect.getsource(components_cmd._build_loader))
    fn = tree.body[0]
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Return), ast.unparse(fn)
    call = body[0].value
    assert isinstance(call, ast.Call) and call.func.id == "ConfigLoader"
    assert not call.args and not call.keywords, (
        "_build_loader() must stay a bare ConfigLoader() — six external callers read it for the "
        f"GLOBAL harness config: {ast.unparse(call)}"
    )


# ------------------------------------------------------------------ #
# Spec 21 §5 / §10: `terminal.keys` changes apply at the next start, and say so
# ------------------------------------------------------------------ #


def test_set_terminal_keys_newline_says_it_applies_at_the_next_start(tmp_path, monkeypatch, fake_home):
    """The box is built once, so the receipt adds the clause inside its existing parenthetical,
    after the file note and a semicolon."""
    layout = _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "set", "terminal.keys.newline", "ctrl-o"])

    assert result.exit_code == 0, result.output
    assert "(the global overrides.yaml; applies at the next start)" in result.stdout, result.stdout
    written = yaml.safe_load(layout.global_ovl.read_text())
    assert written["terminal"]["keys"]["newline"] == "ctrl-o"


def test_set_terminal_keys_json_says_applies_next_start(tmp_path, monkeypatch, fake_home):
    _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "set", "--json", "terminal.keys.nudge", "ctrl-g"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["path"] == "terminal.keys.nudge"
    assert payload["applies"] == "next-start"


@pytest.mark.parametrize(
    ("path", "value"),
    [("org.name", "SET-BY-CLI"), ("terminal.inputbox_enabled", "false")],
)
def test_set_another_path_has_no_next_start_clause(tmp_path, monkeypatch, fake_home, path, value):
    """Only `terminal.keys` paths get the clause and the field; every other receipt, human and
    JSON, is exactly what scripts have always read."""
    _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    human = runner.invoke(app, ["components", "set", path, value])
    assert human.exit_code == 0, human.output
    assert "applies at the next start" not in human.stdout, human.stdout
    assert "(the global overrides.yaml)" in human.stdout, human.stdout

    as_json = runner.invoke(app, ["components", "set", "--json", path, value])
    assert as_json.exit_code == 0, as_json.output
    payload = json.loads(as_json.stdout)
    assert "applies" not in payload
    assert sorted(payload) == ["after", "before", "layer", "path", "target"]


def _read_keys(path: str):
    """`components get --json <path>` -> the effective value."""
    result = runner.invoke(app, ["components", "get", "--json", path])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["value"]


def test_set_terminal_keys_one_at_a_time(tmp_path, monkeypatch, fake_home):
    """DEC21-precedence through the real CLI: the first key set leaves the other list empty of the
    chord it took, and the second set is then free to give that list a chord again."""
    _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    first = runner.invoke(app, ["components", "set", "terminal.keys.newline", "alt-enter"])
    assert first.exit_code == 0, first.output
    assert _read_keys("terminal.keys.nudge") == []
    assert _read_keys("terminal.keys.newline") == ["alt-enter"]

    second = runner.invoke(app, ["components", "set", "terminal.keys.nudge", "ctrl-o"])
    assert second.exit_code == 0, second.output
    assert _read_keys("terminal.keys.nudge") == ["ctrl-o"]
    assert _read_keys("terminal.keys.newline") == ["alt-enter"]


def test_set_newline_ctrl_o_keeps_the_nudge(tmp_path, monkeypatch, fake_home):
    _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "set", "terminal.keys.newline", "ctrl-o"])

    assert result.exit_code == 0, result.output
    assert _read_keys("terminal.keys.nudge") == ["alt-enter"]
    assert _read_keys("terminal.keys.newline") == ["ctrl-o"]


def test_set_newline_alt_enter_against_restated_defaults_succeeds(tmp_path, monkeypatch, fake_home):
    """A config.yaml that restates both defaults has set neither list away from its default, so the
    one-key change wins and takes Alt+Enter from the nudge (a list at its default value yields)."""
    layout = _layout(tmp_path, monkeypatch, fake_home, workspace=False)
    config = dict(_GLOBAL_CONFIG)
    config["terminal"] = {"keys": {"nudge": ["alt-enter"], "newline": ["shift-enter"]}}
    (layout.global_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    result = runner.invoke(app, ["components", "set", "terminal.keys.newline", "alt-enter"])

    assert result.exit_code == 0, result.output
    assert _read_keys("terminal.keys.nudge") == []
    assert _read_keys("terminal.keys.newline") == ["alt-enter"]


def test_set_terminal_keys_nudge_empty_string_empties_the_list(tmp_path, monkeypatch, fake_home):
    layout = _layout(tmp_path, monkeypatch, fake_home, workspace=False)

    result = runner.invoke(app, ["components", "set", "terminal.keys.nudge", ""])

    assert result.exit_code == 0, result.output
    assert "applies at the next start" in result.stdout, result.stdout
    assert _read_keys("terminal.keys.nudge") == []
    assert yaml.safe_load(layout.global_ovl.read_text())["terminal"]["keys"]["nudge"] == ""
