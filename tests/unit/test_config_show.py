"""CLI-03 — `localharness config show`: the merged config, and the file that set each key.

The user problem, in the milestone's own words: a workspace changes what the harness does, and
until now the only way to find out WHICH file changed a given key was to open four files and
re-do the merge in your head. This is the command you run when the harness does something you did
not configure, so it has exactly one job — answer "which file?" in one line per key.

Everything here drives the REAL Typer app through `CliRunner` from `proj/src/pkg`, a SUBDIRECTORY
of the project. That is not incidental: the walk up-tree is the behavior being proven, and a test
run from the project root would pass against an implementation that only ever reads
`./.localharness`. Layout recipe from `test_components_workspace_layer.py`: both discovery walks
stop at `$HOME`, `LOCALHARNESS_DIR`/`LOCALHARNESS_HOME` are cleared (either counts as an explicit
selection and switches discovery OFF, which would make every assertion below pass for the wrong
reason), `.git` is a plain empty directory so no git binary is needed and no child process starts,
and `Confirm.ask` RAISES so an unexpected trust prompt fails the test instead of hanging it.

**The `--json` pin is here from birth, not as a later fix.** F1 in the post-42 dogfood was
`agent list --json` emitting its payload through `console.print`, which wraps at the terminal width
AND eats `[...]` in the data as markup. The two corruptions partially cancel: at a wide terminal the
eaten markup can leave the line UNDER the wrap width, so the output still parses as JSON and
carries silently different data — the failure a parseability check cannot see. So the value-equality
assertion goes BEFORE the parse here, and the fixture is driven at two widths, because one width
would only have caught one of the two bugs.

Ordering of the header is asserted on the PATHS rather than on the band names, deliberately: a band
name also appears in the table's `set by` column, so an index comparison on names could be
satisfied by the table body instead of by the header it is meant to grade.
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
from localharness.config.loader import ConfigLoader
from localharness.config.paths import discover_workspace_dir
from localharness.registry.catalogue import build_catalogue

runner = CliRunner()

_GLOBAL_NAME = "GLOBAL-NAME-LOSES"
_WORKSPACE_NAME = "WORKSPACE-NAME-WINS"

# ASCII only, on purpose: json.dumps escapes non-ASCII (`—` becomes `—`), and the raw-stdout
# assertion below reads the payload BEFORE parsing it. Two markup-shaped tokens, and they are not
# equivalent — Rich's markup tag must start with a lowercase letter, `#`, `/` or `@`, so
# `[bold]`/`[/bold]` are EATEN while `[P1]` survives. A fixture carrying only `[P1]` proves nothing.
# 141 characters, comfortably past width 80 and comfortably under width 200 once indented.
_BRACKETED_NAME = (
    "[bold]payments[/bold] platform, the [P1] on-call rota's project, whose name "
    "is long enough to outrun any terminal width worth testing at all"
)

_GLOBAL_CONFIG = {
    "version": "1",
    "provider": {
        "provider_type": "vllm",
        "base_url": "http://127.0.0.1:9/v1",
        "default_model": "test-model",
    },
    "org": {"name": _GLOBAL_NAME, "log_level": "info"},
}


def _layout(
    tmp_path: Path,
    monkeypatch,
    *,
    ws_config: Optional[dict] = None,
    ws_overrides: Optional[dict] = None,
    workspace: bool = True,
    columns: str = "400",
) -> SimpleNamespace:
    """A fake `$HOME` holding the GLOBAL layer, plus a project whose CWD is two levels down.

        <home>/.localharness/      GLOBAL layer: config.yaml (org.name = GLOBAL-NAME-LOSES)
        <home>/proj/.git/          plain empty dir — the project boundary marker
        <home>/proj/.localharness/ WORKSPACE layer: config.yaml / overrides.yaml, per test
        <home>/proj/src/pkg/       the CWD every invocation runs from
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    home = tmp_path / "home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    (global_dir / "config.yaml").write_text(yaml.safe_dump(_GLOBAL_CONFIG), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    # rich wraps at the console width; a wrapped path is not a printed path.
    monkeypatch.setenv("COLUMNS", columns)

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    ws_dir = proj / ".localharness"
    if workspace:
        ws_dir.mkdir(parents=True)
        if ws_config is not None:
            (ws_dir / "config.yaml").write_text(yaml.safe_dump(ws_config), encoding="utf-8")
        if ws_overrides is not None:
            (ws_dir / "overrides.yaml").write_text(
                yaml.safe_dump(ws_overrides), encoding="utf-8"
            )

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
        global_cfg=global_dir / "config.yaml",
        global_ovl=global_dir / "overrides.yaml",
        ws_dir=ws_dir,
        ws_cfg=ws_dir / "config.yaml",
        ws_ovl=ws_dir / "overrides.yaml",
        proj=proj,
        deep_dir=deep_dir,
    )


def _line_with(output: str, needle: str) -> str:
    """The single output line carrying `needle` — a row-scoped assertion, not a whole-page one."""
    hits = [ln for ln in output.splitlines() if needle in ln]
    assert len(hits) == 1, f"expected exactly one line containing {needle!r}, got {len(hits)}"
    return hits[0]


def _key(payload: dict, path: str) -> dict:
    hits = [k for k in payload["keys"] if k["path"] == path]
    assert len(hits) == 1, f"expected exactly one {path!r} entry, got {len(hits)}"
    return hits[0]


def _combined(result) -> str:
    return (result.output or "") + (result.stderr or "")


# ------------------------------------------------------------------ #
# CLI-03 — which file set this key
# ------------------------------------------------------------------ #


def test_show_from_a_subdirectory_names_the_workspace_as_the_setter(tmp_path, monkeypatch):
    """The command's whole reason to exist, from two directories down.

    Four assertions and all four are load-bearing. "the workspace value is printed" also passes
    when BOTH values are printed, and "the band is workspace-config" is only meaningful on the
    org.name ROW — asserted page-wide it would be satisfied by any other workspace-set key. The
    negative assertion on the global value is what proves the workspace one REPLACED it rather
    than joining it.
    """
    _layout(tmp_path, monkeypatch, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, _combined(result)
    row = _line_with(result.output, "org.name")
    assert _WORKSPACE_NAME in row, "the row does not carry the value actually in force here"
    assert "workspace-config" in row, "the row does not name the file that set it"
    assert _GLOBAL_NAME not in result.output, "the losing global value is still on the page"


def test_the_workspace_overrides_band_wins_and_is_named(tmp_path, monkeypatch):
    """The top of the ruled order, end to end.

    `overrides.yaml` in the workspace is the highest-priority of the four files. It is also the
    band with no writer in v0.13 — nothing in the CLI creates it — so if the display derived its
    bands from what the tools write rather than from the merge, this is the one that would vanish.
    """
    _layout(
        tmp_path,
        monkeypatch,
        ws_config={"org": {"name": _WORKSPACE_NAME, "log_level": "warning"}},
        ws_overrides={"org": {"log_level": "debug"}},
    )

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, _combined(result)
    row = _line_with(result.output, "org.log_level")
    assert "debug" in row, "the highest-priority file did not win"
    assert "workspace-overrides" in row, "the winning band is misnamed"


def test_header_names_all_four_files_in_ruled_order_marking_which_exist(tmp_path, monkeypatch):
    """The header is the merge, rendered: four bands, four paths, lowest priority first.

    Ordering is asserted on the PATHS because band names recur in the table's `set by` column.
    The present/absent marks are asserted on the two files that differ — the global config.yaml
    exists, the global overrides.yaml does not — so a header that hardcoded either mark fails.
    """
    layout = _layout(tmp_path, monkeypatch, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, _combined(result)
    out = result.output
    for band in (
        "global-config",
        "global-overrides",
        "workspace-config",
        "workspace-overrides",
    ):
        assert band in out, f"the header does not name the {band} band"

    positions = [
        out.index(str(layout.global_cfg)),
        out.index(str(layout.global_ovl)),
        out.index(str(layout.ws_cfg)),
        out.index(str(layout.ws_ovl)),
    ]
    assert positions == sorted(positions), (
        "the four files are not printed lowest-priority-first — the display contradicts the merge"
    )
    assert "present" in _line_with(out, str(layout.global_cfg))
    assert "missing" in _line_with(out, str(layout.global_ovl))


def test_bracketed_text_survives_both_the_header_path_and_the_value_cell(tmp_path, monkeypatch):
    """39-05's `[old] proj` lesson, applied to BOTH places this command prints untrusted text.

    Rich would eat `[dim]` from the path and `[bold]` from the value — the first makes the command
    report a path that does not exist, in the one command whose entire job is to say where your
    config comes from; the second silently rewrites the value you came here to read. Two separate
    `escape()` sites, so they are asserted separately.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    home = tmp_path / "[dim] home"
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True)
    bracketed = dict(_GLOBAL_CONFIG, org={"name": _BRACKETED_NAME, "log_level": "info"})
    (global_dir / "config.yaml").write_text(yaml.safe_dump(bracketed), encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, _combined(result)
    assert str(global_dir / "config.yaml") in result.output, "the header path lost its markup"
    assert _BRACKETED_NAME in _line_with(result.output, "org.name"), (
        "the table ate the value's brackets"
    )


# ------------------------------------------------------------------ #
# LAYR-03 — no workspace, no new vocabulary
# ------------------------------------------------------------------ #


def test_no_workspace_introduces_no_workspace_vocabulary(tmp_path, monkeypatch):
    """With nothing up-tree the command prints the two global files and NOTHING about workspaces.

    LAYR-03's contract is that a user who never made a `.localharness/` sees no trace of the
    feature. A header that always printed four bands and marked two of them absent would satisfy
    "names the files in play" and violate this.
    """
    layout = _layout(tmp_path, monkeypatch, workspace=False)

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0, _combined(result)
    assert "workspace-config" not in result.output
    assert "workspace-overrides" not in result.output
    assert str(layout.global_cfg) in result.output


def test_no_workspace_attributes_every_listed_key_to_the_global_config(tmp_path, monkeypatch):
    """No workspace, no overrides: every key some file sets came from the one file there is."""
    _layout(tmp_path, monkeypatch, workspace=False)

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.stdout)
    assert payload["keys"], "the default view listed nothing at all"
    assert {k["layer"] for k in payload["keys"]} == {"global-config"}
    assert [layer["layer"] for layer in payload["layers"]] == [
        "global-config",
        "global-overrides",
    ]


# ------------------------------------------------------------------ #
# F1 — machine output never through Rich
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("width", [80, 200], ids=["width80", "width200"])
def test_json_survives_a_long_bracketed_value_at_two_widths(tmp_path, monkeypatch, width):
    """The payload goes in and comes back out unchanged, at a narrow AND a wide terminal.

    Assertion order is the point (41-06's lesson). The raw-stdout check comes FIRST and needs no
    parser: at width 200 a Rich emitter produces perfectly valid JSON carrying silently eaten
    markup, so a parse-first test would report the wrong failure — or none at all.
    """
    _layout(
        tmp_path,
        monkeypatch,
        ws_config={"org": {"name": _BRACKETED_NAME}},
        columns=str(width),
    )

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, _combined(result)
    assert _BRACKETED_NAME in result.stdout, "the emitter altered the payload's text"
    payload = json.loads(result.stdout)
    assert _key(payload, "org.name")["value"] == _BRACKETED_NAME
    assert _key(payload, "org.name")["layer"] == "workspace-config"


def test_json_layers_carry_the_paths_and_their_existence(tmp_path, monkeypatch):
    """The machine-readable half of the header: four entries, absolute paths, honest `exists`."""
    layout = _layout(tmp_path, monkeypatch, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0, _combined(result)
    payload = json.loads(result.stdout)
    assert [entry["path"] for entry in payload["layers"]] == [
        str(layout.global_cfg),
        str(layout.global_ovl),
        str(layout.ws_cfg),
        str(layout.ws_ovl),
    ]
    assert [entry["exists"] for entry in payload["layers"]] == [True, False, True, False]


# ------------------------------------------------------------------ #
# LAYR-02 — an explicit --config-dir is a full replacement
# ------------------------------------------------------------------ #


def test_config_dir_skips_discovery_entirely(tmp_path, monkeypatch):
    """Naming a config directory replaces the whole thing: no workspace value, no workspace file."""
    layout = _layout(tmp_path, monkeypatch, ws_config={"org": {"name": _WORKSPACE_NAME}})

    result = runner.invoke(
        app, ["config", "show", "--config-dir", str(layout.global_dir)]
    )

    assert result.exit_code == 0, _combined(result)
    assert _WORKSPACE_NAME not in result.output, "an explicit --config-dir still read the workspace"
    assert str(layout.ws_cfg) not in result.output, "a workspace file is in the header anyway"
    assert _GLOBAL_NAME in _line_with(result.output, "org.name")


# ------------------------------------------------------------------ #
# The default view, and --all
# ------------------------------------------------------------------ #


def test_all_lists_the_whole_catalogue_and_the_default_view_lists_less(tmp_path, monkeypatch):
    """Two runs, one comparison, and the expected count is DERIVED rather than typed.

    182 today; 42-01 already tripped over a hardcoded catalogue count once, so the number is read
    from `build_catalogue` in-process instead of being pinned as a literal here.
    """
    layout = _layout(tmp_path, monkeypatch, ws_config={"org": {"name": _WORKSPACE_NAME}})

    default_run = runner.invoke(app, ["config", "show", "--json"])
    all_run = runner.invoke(app, ["config", "show", "--json", "--all"])

    assert default_run.exit_code == 0, _combined(default_run)
    assert all_run.exit_code == 0, _combined(all_run)
    default_keys = json.loads(default_run.stdout)["keys"]
    all_keys = json.loads(all_run.stdout)["keys"]

    expected = len(
        build_catalogue(
            ConfigLoader(
                config_dir=layout.global_dir, local_config_dir=layout.ws_dir
            ).load_harness()
        )
    )
    assert len(all_keys) == expected, "--all is not the whole catalogue"
    assert len(default_keys) < len(all_keys), (
        "the default view is the whole catalogue — the answer is buried in ~180 shipped defaults"
    )
    assert {k["layer"] for k in default_keys} == {"global-config", "workspace-config"}


def test_show_is_registered_beside_migrate(tmp_path, monkeypatch):
    """One sub-app, two commands, and the three flags this one adds."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("COLUMNS", "400")

    group = runner.invoke(app, ["config", "--help"])
    own = runner.invoke(app, ["config", "show", "--help"])

    assert "migrate" in group.output and "show" in group.output
    for flag in ("--all", "--json", "--config-dir"):
        assert flag in own.output, f"{flag} is not on `config show --help`"


# ------------------------------------------------------------------ #
# The failure a first-time user actually hits
# ------------------------------------------------------------------ #


def test_missing_config_exits_nonzero_and_names_init(tmp_path, monkeypatch):
    """#119's rule: a missing config ends with the same next step every other command gives."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    home = tmp_path / "home"
    (home / "empty").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(home / "empty")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code != 0, "a missing config reported success"
    assert "localharness init" in _combined(result)
