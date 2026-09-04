"""Criterion 4 — a workspace the SHIPPED COMMAND created is immediately live.

Phases 39-42 taught a session to find a `.localharness/`, merge it, confine the file tools to the
project around it and keep that project's memory inside it. Phase 43 gave users a command that
creates one. Every other test in this milestone builds `.localharness/` with `mkdir` — which proves
the layering and says nothing about whether the thing a user actually types produces a layer the
harness can then use. That gap is the milestone's one-sentence goal, and it is what this file
closes:

    `localharness init --workspace`  ->  `localharness start`  ->  the 39-42 behaviors apply
    immediately to the fresh scaffold, with zero extra configuration.

So the workspace under test here is **not hand-built**. It is created by invoking the real Typer app
through `CliRunner`, in a scratch directory under a fake `$HOME`, and everything afterwards is
measured against whatever that command actually wrote — including its comment-only `config.yaml`,
which parses to `None` and must therefore contribute nothing to the merge without breaking it.

**Offline by construction, not by trust.** The session half drives the REAL `_start_async` with only
the EXTERNAL boundaries stubbed (`tests/unit/test_start_cmd.py::_stub_start_boundaries`: the LLM
probe, the tokenizer, the REPL loop, plugin discovery). On top of that the fixture rewrites the
harness config's `base_url` to the loopback discard port, so even a boundary that stopped being
stubbed would be refused instantly instead of hanging a test on a timeout. No model is started, no
GPU is touched, no socket is opened.

**Every helper here is imported, never copied** (41-06's rule). A copied fixture drifts away from the
harness the rest of the suite is graded against, and then grades its own copy.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.cli.workspace import resolve_workspace_layer
from localharness.config.paths import (
    WORKSPACE_DIR_NAME,
    discover_workspace_dir,
    workspace_is_within_repo,
)

# 41-05's workspace-drive recipe and 41-06's tool-registration recorder, imported whole:
#   `_hermetic`      — a fake $HOME holding the global layer with BOTH env overrides cleared
#                      (conftest's autouse LOCALHARNESS_HOME counts as an explicit selection and
#                      would switch discovery off entirely, 39-04), and COLUMNS=400
#   `_boom`          — a trust prompt on an IN-PROJECT workspace is a failure, not a hang
#   `_drive`         — `_start_async(None, False, False, None)`; the trailing None is the whole
#                      point, since any explicit config dir skips discovery (LAYR-02)
#   `_file_snapshot` / `_changed` — (size, mtime_ns) per file, and the paths that moved
from tests.unit.test_start_cmd import _stub_start_boundaries, _write_agent
from tests.unit.test_workspace_carveouts import _only, _record_tool_registration
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _boom,
    _changed,
    _drive,
    _file_snapshot,
    _global_only_start,
    _hermetic,
)

runner = CliRunner()

# RFC 863's discard port on loopback. Nothing listens there, so a connection is refused in
# microseconds rather than waiting out a timeout — the safe value for a test that must never
# reach a model.
_DISCARD_URL = "http://127.0.0.1:9/v1"

_BASE_URL_LINE = re.compile(r"(?m)^(\s*base_url:).*$")


def _offline_provider(global_dir: Path) -> None:
    """Point the harness config's provider at the discard port, in place.

    A substitution rather than a rewritten config block: `_stub_start_boundaries` owns the shape of
    that file, and a second copy of it here would drift the first time the harness changed. The
    guard below is what makes the substitution observable — a regex that stopped matching would
    otherwise leave the original endpoint in place silently.
    """
    cfg = global_dir / "config.yaml"
    cfg.write_text(_BASE_URL_LINE.sub(rf"\1 {_DISCARD_URL}", cfg.read_text()))
    assert re.search(
        rf"(?m)^\s*base_url:\s*{re.escape(_DISCARD_URL)}\s*$", cfg.read_text()
    ), "the provider endpoint was not rewritten — this drive is not offline by construction"


def _subdir(proj: Path, monkeypatch) -> Path:
    """Stand somewhere DEEP in the project, which is where a user actually is.

    The workspace sits at the project root; every claim in this file is about what the harness does
    from a directory below it, because a discovery walk that only worked at the root would pass a
    test written at the root.
    """
    sub = proj / "src" / "pkg"
    sub.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(sub)
    return sub


def _scaffolded_project(tmp_path, monkeypatch) -> tuple[Path, Path, Path]:
    """A fake `$HOME` and a git-marked project whose workspace was created by the SHIPPED CLI.

    Returns `(home, global_dir, project_root)`. The `.git` marker makes `proj/` the project the
    caller is standing in, so the workspace is IN-project and loads with no trust prompt (39-04);
    `_boom` turns a prompt into a failure instead of a hang.
    """
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, home)
    _stub_start_boundaries(global_dir, monkeypatch)  # writes the GLOBAL config.yaml
    _offline_provider(global_dir)

    proj = home / "proj"
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    result = CliRunner().invoke(app, ["init", "--workspace"])

    # Premise guards, not tests. `test_init_workspace_scaffold.py` grades this command; if it did
    # not run here, every assertion below would be measuring a project with no workspace at all.
    assert result.exit_code == 0, result.output
    ws = proj / WORKSPACE_DIR_NAME
    assert ws.is_dir() and (ws / "agents").is_dir(), f"the CLI scaffolded {sorted(proj.iterdir())}"
    return home, global_dir, proj


# ------------------------------------------------------------------ 1. it is found from below


def test_the_cli_made_workspace_is_discovered_from_a_subdirectory(tmp_path, monkeypatch):
    """The first link in the chain: the command wrote a layer that discovery recognises.

    `resolve_workspace_layer(None)` rather than `discover_workspace_dir()` alone, because the
    resolver is the gate that could ask for trust — and `_boom` is installed, so a prompt on an
    in-project workspace fails this test rather than silently changing the answer.
    """
    _home, _global_dir, proj = _scaffolded_project(tmp_path, monkeypatch)
    ws = proj / WORKSPACE_DIR_NAME
    _subdir(proj, monkeypatch)

    assert discover_workspace_dir() == ws.resolve(), "the scaffold is not the discovered layer"
    assert workspace_is_within_repo(ws.resolve(), proj), "the scaffold reads as outside its project"
    assert resolve_workspace_layer(None) == ws.resolve(), (
        "the scaffold was discovered but did not APPLY — a layer the gate drops is not a layer"
    )


# ------------------------------------------------------------------ 2. the roster


def test_an_agent_in_the_scaffolded_dir_appears_in_the_roster_from_a_subdirectory(
    tmp_path, monkeypatch
):
    """`agents/` is the directory the scaffold creates for you, so putting a yaml in it must be all
    it takes for `agent list` to see it from anywhere in the project.

    `--json` deliberately: parsing the whole payload is also a live regression guard on 43-02's F1
    fix (`typer.echo`, not a Rich console that eats `[...]` and wraps mid-string), reached from a
    different file and a different topology than the one that pinned it.
    """
    _home, _global_dir, proj = _scaffolded_project(tmp_path, monkeypatch)
    _write_agent(proj / WORKSPACE_DIR_NAME / "agents", AGENT)
    _subdir(proj, monkeypatch)

    result = runner.invoke(app, ["agent", "list", "--json"])

    assert result.exit_code == 0, result.output
    names = [a.get("name") for a in json.loads(result.stdout)]
    assert AGENT in names, f"the workspace roster did not reach `agent list`: {names}"


# ------------------------------------------------------------------ 3. criterion 4 itself


async def test_a_session_in_the_cli_made_workspace_lands_its_state_in_the_project(
    tmp_path, monkeypatch
):
    """The milestone's sentence, executable: scaffold with the command, then run a real session.

    ASSERTION ORDER IS THE DESIGN, and it is not a style preference — a specific assertion sitting
    under a general one that shadows it has bitten this milestone four times (41-06 x3, 42-05 x1),
    and once more in 43-05 where the shadowing line was code the same plan had just added. So:

      1. the did-it-bite guard (`memory.db` under the workspace). If the drive did not run, or ran
         against a session with no workspace, EVERY assertion below is vacuously true;
      2. the other two artifacts a zero-turn drive materialises;
      3. each of the three is ABSENT under `<global>/agents/` — "it exists somewhere" is not the
         claim, "it moved" is;
      4. nothing under `<global>/agents/` was created or modified at all, with the offending paths
         printed in the failure message rather than summarised;
      5. LAST, because it is the weakest thing said here: the workspace tree grew.

    Scoped like 41-05, on purpose: `<global>/agents/` specifically, not the whole global dir. The
    drive legitimately writes elsewhere under the global layer (packaged tools, provider speed
    stats), so "the global dir is unchanged" would be false today and any later relaxation of it
    would be indistinguishable from a regression.
    """
    _home, global_dir, proj = _scaffolded_project(tmp_path, monkeypatch)
    ws = proj / WORKSPACE_DIR_NAME
    # The agent lives in the WORKSPACE, so the roster comes from the workspace and `start`'s
    # root-agent mint branch never fires — nothing writes an agent into the global dir, which is
    # what lets step 4 mean something.
    _write_agent(ws / "agents", AGENT)
    _subdir(proj, monkeypatch)
    assert discover_workspace_dir() == ws.resolve(), "premise: the drive has a workspace to use"

    global_agents = global_dir / "agents"
    before = _file_snapshot(global_dir)

    await _drive()

    ws_agent_dir = ws / "agents" / AGENT
    assert (ws_agent_dir / "memory.db").exists(), (
        f"no memory.db under the scaffolded workspace ({ws_agent_dir}) — the drive did not run, "
        "and nothing below this line proves anything"
    )
    for name in ("MEMORY.md", "history.jsonl"):
        assert (ws_agent_dir / name).exists(), f"{name} did not land in the CLI-made workspace"
    for name in ("memory.db", "MEMORY.md", "history.jsonl"):
        assert not (global_agents / AGENT / name).exists(), (
            f"{name} ALSO landed in the machine's agents tree — the state did not move, it forked"
        )

    intruders = {p for p in _changed(before, _file_snapshot(global_dir)) if global_agents in p.parents}
    assert intruders == set(), (
        f"the session wrote into the machine's agents tree: {sorted(intruders)}"
    )

    assert _file_snapshot(ws), "nothing at all was written under the workspace — nothing was proven"


# ------------------------------------------------------------------ 4. the confinement default


async def test_the_scaffold_confines_the_file_tools_to_the_project_not_the_dotdir(
    tmp_path, monkeypatch
):
    """CONF-01 arrives with the scaffold: no second setup step, and the root is the PROJECT.

    Both halves in one body on purpose. `str(proj)` and `str(ws)` differ by a single path component,
    and confining every agent inside `.localharness/` is a mistake that type-checks, names a real
    directory, and would satisfy any assertion that only checked for not-None (41-03's finding).

    The specific check runs FIRST, mirroring `test_workspace_carveouts.py`: with the general
    equality first, a `.parent`-shaped regression reddens that line and the dotdir assertion below
    can never fire, which would make it decorative.
    """
    _home, _global_dir, proj = _scaffolded_project(tmp_path, monkeypatch)
    ws = proj / WORKSPACE_DIR_NAME
    _write_agent(ws / "agents", AGENT)
    _subdir(proj, monkeypatch)
    calls = _record_tool_registration(monkeypatch)

    await _drive()

    root = _only(calls, "register_builtin_tools")["workspace_root"]
    assert root != str(ws), (
        "the file tools were confined to the config folder INSIDE the project; the leash belongs "
        "around the project itself"
    )
    assert root == str(proj), f"the tools were confined to {root}, not the project root {proj}"


# ------------------------------------------------------------------ 5. the new CLI surface


def test_both_new_surfaces_name_the_scaffolded_file_as_the_source(tmp_path, monkeypatch):
    """Set one key in the file the command wrote, and ask the two new commands who set it.

    This also proves the comment block is INERT rather than merely harmless: the scaffolded config
    parses to `None` (asserted), so `org.name` is the first real key the file carries, and the merge
    has to accept it without the surrounding comments contributing anything.

    Each assertion is scoped to the LINE that carries the value, never to the whole capture.
    `config show` prints `workspace-config` in its layer header too, so a substring check over the
    whole output would be satisfied by the header even if the key row credited the global file —
    that is 43-05's shadow, and it is the failure mode this file is most exposed to.
    """
    _home, _global_dir, proj = _scaffolded_project(tmp_path, monkeypatch)
    cfg = proj / WORKSPACE_DIR_NAME / "config.yaml"
    original = cfg.read_text()
    assert yaml.safe_load(original) is None, "the scaffolded config was not comment-only"
    cfg.write_text(original + "org:\n  name: SCAFFOLD-WINS\n")
    _subdir(proj, monkeypatch)

    got = runner.invoke(app, ["components", "get", "org.name"])
    assert got.exit_code == 0, got.output
    assert "SCAFFOLD-WINS" in got.stdout, f"`components get` read the wrong value: {got.stdout}"
    layer_lines = [ln for ln in got.stdout.splitlines() if ln.strip().startswith("layer:")]
    assert len(layer_lines) == 1, f"expected one layer line, got {layer_lines}"
    assert "workspace-config" in layer_lines[0], (
        f"`components get` credited the wrong file: {layer_lines[0].strip()}"
    )

    shown = runner.invoke(app, ["config", "show"])
    assert shown.exit_code == 0, shown.output
    rows = [ln for ln in shown.stdout.splitlines() if "SCAFFOLD-WINS" in ln]
    assert len(rows) == 1, f"expected one `config show` row for the key, got {rows}"
    assert "org.name" in rows[0] and "workspace-config" in rows[0], (
        f"`config show` credited the wrong file on the row itself: {rows[0].strip()}"
    )


# ------------------------------------------------------------------ 6. the LAYR-03 control


async def test_a_project_that_never_ran_init_workspace_sees_none_of_this(tmp_path, monkeypatch):
    """LAYR-03: skip the one command, and the harness is exactly what it was before v0.13.

    The same fake `$HOME`, the same git-marked project, the same subdirectory, the same offline
    drive — the ONE difference is that `init --workspace` was never run. State lands global, no
    `.localharness/` appears in the project, and neither new surface says the word `workspace-`
    anywhere: a user who never made a workspace sees no trace of the feature.
    """
    _home, global_dir, proj = _global_only_start(tmp_path, monkeypatch)
    _offline_provider(global_dir)
    _subdir(proj, monkeypatch)
    assert discover_workspace_dir() is None, "premise: the control must find no workspace"

    await _drive()

    assert (global_dir / "agents" / AGENT / "memory.db").exists(), (
        "the control's state did not land in the global dir — the drive did not run"
    )
    assert not (proj / WORKSPACE_DIR_NAME).exists(), (
        "a session created a workspace in a project that never asked for one"
    )

    for argv in (["agent", "list", "--json"], ["config", "show"]):
        result = runner.invoke(app, argv)
        assert result.exit_code == 0, result.output
        assert "workspace-" not in result.stdout, (
            f"`{' '.join(argv)}` leaked workspace vocabulary with no workspace: {result.stdout}"
        )
