"""The autoresearch CLI surface pinned from the EAGER-wiring tree (Phase 50 Wave 1, captured at
d551b7e, before any source edit): `propose`, `autoresearch` (+ `archive` list/show/approve, run,
review, adopt, report, sentinel) and `experiment` (+ `run`), each `--help` through the ROOT `app`,
two offline error paths, the root `--help` rows and order, and the `proposer.*`/`sentinel.*` rows
of `components list`. One JSON golden.

After the commands move behind the autoresearch plugin (50-05) the SAME golden must hold with only
`AUTORESEARCH_IS_A_PLUGIN` flipped. A diff is a finding to explain, never a golden to regenerate.
A `--install-completion` line appearing in a sub-command's help means the moved Typer app lacks
`add_completion=False` (50-RESEARCH P2).

Normalisation: `_norm_paths` (the test's tmp dir and the hermetic home -> `<TMP>` / `<HOME>`) and
`_norm_stamps` (`YYYY-MM-DD HH:MM[:SS]` -> `<STAMP>`). Nothing else is rewritten.

The offline error paths never reach a model, a worktree or the holdout:
- `autoresearch archive list` on a home with no archive.db returns [] before opening a store.
- `experiment run nosuchid`: `_load_and_validate` refuses an unresolvable id (exit >= 4) before
  any worktree or bench run (autoresearch/experiment.py, step 1).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.plugins.builtin import bundled_plugins

pytestmark = pytest.mark.plugin("autoresearch")

# 50-05 flips this when the commands move behind the plugin; the expected deltas below are the
# G6 / AUTO-01 disclosed changes, written as code.
AUTORESEARCH_IS_A_PLUGIN = True
MOVED = ("autoresearch", "experiment", "propose")

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "autoresearch_surfaces" / "cli_golden.json"
REGEN = os.environ.get("LOCALHARNESS_REGEN_GOLDEN") == "1"
ENV = {"COLUMNS": "100", "TERMINAL_WIDTH": "100", "NO_COLOR": "1"}
runner = CliRunner()

CASES = {
    "autoresearch_help": ["autoresearch", "--help"],
    "archive_help": ["autoresearch", "archive", "--help"],
    "archive_list_help": ["autoresearch", "archive", "list", "--help"],
    "archive_show_help": ["autoresearch", "archive", "show", "--help"],
    "archive_approve_help": ["autoresearch", "archive", "approve", "--help"],
    "run_help": ["autoresearch", "run", "--help"],
    "review_help": ["autoresearch", "review", "--help"],
    "adopt_help": ["autoresearch", "adopt", "--help"],
    "report_help": ["autoresearch", "report", "--help"],
    "sentinel_help": ["autoresearch", "sentinel", "--help"],
    "experiment_help": ["experiment", "--help"],
    "experiment_run_help": ["experiment", "run", "--help"],
    "propose_help": ["propose", "--help"],
    "archive_list_empty": ["autoresearch", "archive", "list"],
    "experiment_run_unknown_id": ["experiment", "run", "nosuchid"],
}


def _norm_paths(text: str, tmp_path: Path) -> str:
    home = os.environ.get("LOCALHARNESS_HOME", "")
    text = text.replace(home, "<HOME>") if home else text
    return text.replace(str(tmp_path), "<TMP>")


def _norm_stamps(text: str) -> str:
    return re.sub(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?", "<STAMP>", text)


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _check(key: str, got) -> None:
    if REGEN:
        data = _golden() if GOLDEN.exists() else {}
        data[key] = got
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert got == _golden()[key], f"autoresearch surface {key!r} drifted"


@pytest.mark.parametrize("case", list(CASES))
def test_autoresearch_cli_matches_the_golden(case, tmp_path):
    result = runner.invoke(app, CASES[case], env=ENV)
    _check(case, {"exit": result.exit_code,
                  "output": _norm_stamps(_norm_paths(result.output, tmp_path))})


def _root_rows() -> list[tuple[str, str]]:
    """(command, help line) in display order, wrapped help lines joined, from root `--help`."""
    out = runner.invoke(app, ["--help"], env=ENV).output
    box = out.split("─ Commands ")[1].split("╰")[0]
    rows: list[list[str]] = []
    for line in box.splitlines()[1:]:
        body = line.strip().strip("│").rstrip()
        if not body.strip():
            continue
        if body.startswith(" ") and not body[1:2].isspace():  # `│ name  help`
            name, _, text = body.strip().partition(" ")
            rows.append([name, text.strip()])
        elif rows:  # a wrapped continuation of the previous help line
            rows[-1][1] += " " + body.strip()
    return [(n, t) for n, t in rows]


def _plugin_command_names() -> set[str]:
    return {d.name for p in bundled_plugins() for d in p.manifest.cli}


def _expected_post_order(pre: list[str]) -> list[str]:
    """G6: core names in the pre order minus the three moved ones, then the sorted plugin block
    (the names bundled manifests contribute, plus the three)."""
    block = [n for n in pre if n in _plugin_command_names()]
    core = [n for n in pre if n not in block and n not in MOVED]
    return core + sorted({*block, *MOVED})


def test_root_help_rows():
    """The SET of (command, help line) never changes, before or after the move."""
    rows = _root_rows()
    _check("root_rows", sorted([list(r) for r in rows]))
    assert {n for n, _ in rows} >= set(MOVED)


def test_root_help_order():
    order = [n for n, _ in _root_rows()]
    if REGEN:
        _check("root_order", order)
    pre = _golden()["root_order"]
    assert order == (_expected_post_order(pre) if AUTORESEARCH_IS_A_PLUGIN else pre)


def test_components_rows(components_home):
    """`proposer.*` / `sentinel.*` rows of `components list --json`, with a proposer configured
    (fake endpoint, key "none"). After the move each row carries plugin "autoresearch" (the JSON
    `plugin` field is what the table's `_layer_cell` renders as `(plugin: autoresearch)`; the table
    itself wraps at width 100, so it is not compared), and an `autoresearch.enabled` row exists."""
    cfg = components_home / "config.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8")
                   + "proposer:\n  base_url: http://localhost:11434/v1\n  model: fake-proposer\n"
                     "  api_key: none\n", encoding="utf-8")
    result = runner.invoke(app, ["components", "list", "--json"], env=ENV)
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    ours = [r for r in rows if r["path"].split(".")[0] in ("proposer", "sentinel")]
    if AUTORESEARCH_IS_A_PLUGIN:
        assert all(r["plugin"] == "autoresearch" for r in ours), ours
        assert any(r["path"] == "autoresearch.enabled" for r in rows)
        ours = [{**r, "plugin": None} for r in ours]  # the golden is the pre-move truth
    else:
        assert not any(r["path"].startswith("autoresearch.") for r in rows)
    # G2 (50-04): proposer.api_key is a SecretStr — masked, typed SecretStr. The golden is the
    # pre-G2 truth (`str`, the configured "none"); this is the disclosed delta, written as code.
    key = next(r for r in ours if r["path"] == "proposer.api_key")
    assert (key["type"], key["current_value"]) == ("SecretStr", "**********"), key
    ours = [{**r, "type": "str", "current_value": "none"} if r is key else r for r in ours]
    _check("components_rows", ours)
