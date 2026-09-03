"""MEMS-01 — a project's memory stays with the project.

One machine runs many projects. Until this landed, every session on the box wrote its `memory.db`,
its `MEMORY.md`, its history and its event log into the ONE global `~/.localharness/agents/<id>/`,
so a research project's recollections and a client codebase's recollections piled into the same
store and bled into each other's recall. A session started inside a project now lands its per-agent
state under that project's own `.localharness/agents/<id>/`.

This file drives the REAL `_start_async` from inside a project, offline: only the EXTERNAL
boundaries are stubbed (the LLM probe, the tokenizer, the REPL loop, plugin discovery). Everything
memory-side runs for real, and every assertion below says where an artifact actually landed rather
than which expression the source contains.

Two boundaries, stated up front rather than discovered later:

* **This is the state-LOCATION half of MEMS-01.** Whether a workspace agent also RECALLS from the
  global store (`agent.memory.scope`) is phase 42's question, and so is the byte-level proof that a
  workspace session leaves the global MEMORY untouched. The claim made here is narrower and exact:
  **nothing under `<global>/agents/` is created or modified by the drive.** Deliberately NOT "the
  whole global dir is unchanged" — the drive legitimately writes elsewhere under it (packaged tools,
  provider speed stats), so the broad version would be either false today or quietly weakened later.
  Phase 42 (MEMS-03) owns the full byte-proof.
* **The divergence is the point.** A workspace session must move its WORK and leave the
  machine-global CONTROL artifacts where they were, so every assertion names which of the two
  directories a path came from, and both directions are asserted — a path under `<ws>` is also
  asserted NOT to be under `<global>`, and the reverse. A test that only checks "it is somewhere"
  cannot fail.

The no-workspace control at the bottom is LAYR-03: with nothing up-tree, every one of these paths is
exactly what it was before this phase existed.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from localharness.config.paths import discover_workspace_dir, workspace_is_within_repo

# The drive harness is `tests/unit/test_start_cmd.py`'s: `_stub_start_boundaries` writes a
# known-good config.yaml into the dir it is handed and stubs the four external boundaries;
# `_write_agent` writes an agent yaml. Imported rather than copied so this file cannot drift from
# the harness that the rest of the start-path tests are graded against (cross-test imports are
# established practice here — see tests/integration/test_mechanism_e2e.py).
from tests.unit.test_start_cmd import _stub_start_boundaries, _write_agent

AGENT = "solo"


# --------------------------------------------------------------------------------- helpers


def _file_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    """Every file under `root`, keyed by path, valued by (size, mtime_ns).

    `st_mtime_ns` rather than `st_mtime`: a whole drive finishes in well under a second, and
    second-resolution mtimes would report an in-place rewrite as unchanged. Copied from
    `tests/unit/test_provider_carveout_workspace.py` (as 40-03 copied `_hermetic` from the e2e
    file) so this file stays readable as the whole of what it measures.
    """
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def _changed(before: dict, after: dict) -> set[Path]:
    """Paths created or modified between the two snapshots. Deletions would surface as members of
    `before` missing from `after`, which is why the union is taken rather than just the diff."""
    return {p for p, v in after.items() if before.get(p) != v} | (set(before) - set(after))


def _boom(*_args, **_kwargs):
    raise AssertionError("prompted for trust on an IN-PROJECT workspace, which must never ask")


def _hermetic(monkeypatch, home: Path) -> Path:
    """A fake `$HOME` holding the GLOBAL layer, with both env overrides cleared.

    All three moves are required. Both discovery walks stop at `$HOME`, so a real one would let the
    developer's own `~/.localharness` answer these tests; and conftest's autouse fixture sets
    `LOCALHARNESS_HOME`, which `resolve_workspace_layer` counts as an EXPLICIT selection (39-04) —
    leaving it set switches discovery off entirely and every assertion below would pass for the
    wrong reason, against a session that has no workspace at all.
    """
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)
    global_dir = home / ".localharness"
    global_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("COLUMNS", "400")
    return global_dir


def _workspace_start(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """A real workspace session ready to drive: fake `$HOME`, a project inside it, CWD in it.

    The `.git` marker makes `proj/` the project the caller is standing in, so the workspace is
    IN-project and loads with NO trust prompt (39-04's boundary) — an unanswered prompt would either
    hang the drive or silently change the file set.

    The agent yaml lives in the WORKSPACE, so the roster comes from the workspace (phase 40's union)
    and `start`'s mint branch never fires — nothing writes a root agent into the global dir, which
    is what lets the "global agents tree untouched" assertion mean something.

    Returns (home, global_dir, workspace_dir).
    """
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, home)
    _stub_start_boundaries(global_dir, monkeypatch)

    proj = home / "proj"
    ws = proj / ".localharness"
    ws.mkdir(parents=True)
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    _write_agent(ws / "agents", AGENT)

    # Premise guards. A fixture whose topology silently broke is 39-07's near-miss: these two lines
    # make it an error rather than a green test that proved nothing.
    assert discover_workspace_dir() == ws.resolve(), "the workspace is not the discovered layer"
    assert workspace_is_within_repo(ws.resolve(), proj), "the workspace is not in-project"
    return home, global_dir, ws


def _global_only_start(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """The LAYR-03 control: the same recipe with NO `.localharness` anywhere up-tree.

    Same fake `$HOME`, same `.git`-marked project, same CWD — the ONE difference is that the project
    has no `.localharness/` and the agent yaml lives in the global dir. Everything the drive produces
    must therefore be exactly what it was before this phase existed.
    """
    home = tmp_path / "home"
    global_dir = _hermetic(monkeypatch, home)
    _stub_start_boundaries(global_dir, monkeypatch)

    proj = home / "proj"
    proj.mkdir(parents=True)
    (proj / ".git").mkdir(parents=True)
    monkeypatch.chdir(proj)
    monkeypatch.setattr("rich.prompt.Confirm.ask", _boom)

    _write_agent(global_dir / "agents", AGENT)

    assert discover_workspace_dir() is None, "the control found a workspace — it must find none"
    return home, global_dir, proj


def _install_recorders(monkeypatch) -> dict[str, list]:
    """Wrap (never replace) each construction the wiring flows through, and record its kwargs.

    Wrappers rather than doubles on purpose: the real objects still run, so the drive still writes a
    real memory.db and the file-landing assertions stay honest. Each is patched at the MODULE
    attribute because `_start_async` imports every one of these inside the function body.
    """
    rec: dict[str, list] = {
        "migrate": [], "store": [], "loop": [], "runner": [], "channel": [], "repl": [],
    }

    import localharness.agent.loop as _loop_mod
    import localharness.agent.subagent as _sub
    import localharness.channels.terminal as _term
    import localharness.cli.repl as _repl_mod
    import localharness.memory.sqlite as _sqlite

    real_migrate = _sqlite._migrate_legacy_root_agent_dir

    def _rec_migrate(state_dir, agent_id, *args, **kwargs):
        # Pitfall 4: this is the DATA-tree rename, and its global-only lookalike
        # (`_migrate_legacy_root_agent_yaml`) is one identifier away. Recording its first argument
        # is what keeps the swap detectable instead of silent.
        rec["migrate"].append(Path(state_dir))
        return real_migrate(state_dir, agent_id, *args, **kwargs)

    monkeypatch.setattr("localharness.memory.sqlite._migrate_legacy_root_agent_dir", _rec_migrate)

    real_store_init = _sqlite.MemoryStore.__init__

    def _rec_store_init(self, *args, **kwargs):
        rec["store"].append(kwargs)
        return real_store_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.memory.sqlite.MemoryStore.__init__", _rec_store_init)

    real_loop_init = _loop_mod.AgentLoop.__init__

    def _rec_loop_init(self, *args, **kwargs):
        rec["loop"].append(kwargs)
        return real_loop_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.agent.loop.AgentLoop.__init__", _rec_loop_init)

    real_factory = _sub.make_explore_agent_runner

    def _rec_factory(**kwargs):
        rec["runner"].append(kwargs)
        return real_factory(**kwargs)

    monkeypatch.setattr("localharness.agent.subagent.make_explore_agent_runner", _rec_factory)

    real_channel_init = _term.TerminalChannel.__init__

    def _rec_channel_init(self, *args, **kwargs):
        rec["channel"].append(kwargs)
        return real_channel_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.channels.terminal.TerminalChannel.__init__", _rec_channel_init)

    real_repl_init = _repl_mod.OrchestratorREPL.__init__

    def _rec_repl_init(self, *args, **kwargs):
        rec["repl"].append(kwargs)
        return real_repl_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.__init__", _rec_repl_init)

    return rec


def _one(rec: dict[str, list], key: str) -> dict:
    """The single recorded construction for `key`, with the did-it-bite guard.

    Without this guard a mis-targeted monkeypatch reads as an empty list and every downstream
    assertion is skipped rather than failed — a green test on a patch that never fired.
    """
    assert rec[key], f"{key} was never constructed — the patch did not bite"
    return rec[key][0]


async def _drive() -> None:
    """The whole point: `config_dir=None`.

    Any explicit value (flag, LOCALHARNESS_DIR, LOCALHARNESS_HOME) short-circuits discovery before
    the filesystem walk (LAYR-02), so a drive that named its config dir could not have a workspace
    layer at all and would prove nothing about where a workspace session's state lands.
    """
    from localharness.cli.start_cmd import _start_async

    await _start_async(None, False, False, None)


# --------------------------------------------------------------------------------- the workspace


async def test_workspace_session_lands_its_agent_state_in_the_project(tmp_path, monkeypatch):
    """The headline: memory.db is written under the PROJECT, not under the machine's global dir.

    Asserted in both directions — a `memory.db` that exists somewhere proves nothing; a `memory.db`
    that exists under `<ws>` and is ABSENT under `<global>` is the claim.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    ws_agent_dir = ws / "agents" / AGENT
    assert ws_agent_dir.is_dir(), f"no agent state dir under the workspace: {ws_agent_dir}"
    # MEASURED, not assumed: these three are what a zero-turn drive actually materializes. The
    # other artifacts that ride on the same `agent_dir` — bus-events.jsonl, `sessions/` (derived
    # inside EventBus.__init__) and compact.md — are lazy: nothing publishes and no turn compacts,
    # so no file appears. Their PATHS are asserted from the captured constructions instead (see the
    # compact.md assertion in test_root_loop_splits_...); asserting a file into existence here would
    # be a fiction. memory.log is left out for the opposite reason: it is 0 bytes, so its existence
    # says only that logging was configured.
    for name in ("memory.db", "MEMORY.md", "history.jsonl"):
        assert (ws_agent_dir / name).exists(), f"{name} did not land in the workspace"
        assert not (global_dir / "agents" / AGENT / name).exists(), \
            f"{name} ALSO landed in the global dir — the state did not move, it forked"

    # It is a real store the real session wrote, not an empty file: one sessions row, opened and
    # closed by this drive. A path assertion alone cannot tell you the store was ever used.
    con = sqlite3.connect(str(ws_agent_dir / "memory.db"))
    try:
        rows = con.execute("SELECT exit_reason FROM sessions").fetchall()
    finally:
        con.close()
    assert len(rows) == 1 and rows[0][0] == "complete", f"unexpected sessions rows: {rows}"

    # Pitfall 4's guard: the DATA-tree migration must target the tree the store then opens. If it
    # named the global dir instead, it would aim a rename at a directory holding none of the files.
    assert rec["migrate"], "the legacy data-tree migration never ran — the patch did not bite"
    assert rec["migrate"][0] == ws, \
        f"the data-tree migration targeted {rec['migrate'][0]}, not the workspace {ws}"


async def test_workspace_session_touches_nothing_under_the_global_agents_tree(tmp_path, monkeypatch):
    """Scoped on purpose: `<global>/agents/` specifically, not the whole global dir.

    The drive legitimately writes elsewhere under the global layer (packaged tools, provider speed
    stats), so "the global dir is unchanged" would be false today and any later relaxation of it
    would be indistinguishable from a regression. `<global>/agents/` is the tree that holds another
    project's memory, and it is the tree this session must not touch. The byte-level proof that
    global MEMORY CONTENT is untouched belongs to phase 42 (MEMS-03).
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    before = _file_snapshot(global_dir)

    await _drive()

    changed = _changed(before, _file_snapshot(global_dir))
    global_agents = global_dir / "agents"
    intruders = sorted(p for p in changed if global_agents in p.parents)
    assert not intruders, f"the workspace session wrote into the global agents tree: {intruders}"

    # The other direction: the drive DID write, it just wrote somewhere else. Without this the
    # assertion above would be satisfied by a drive that crashed before touching any file.
    assert _file_snapshot(ws), "nothing at all was written under the workspace — nothing was proven"


async def test_root_loop_splits_its_compact_note_from_its_kill_switch(tmp_path, monkeypatch):
    """One construction, two directories: compact.md follows the work, the kill file does not.

    The kill switch is a machine-global CONTROL artifact — one file stops every agent on the box —
    so a per-workspace copy would mean hunting down N files to halt N sessions.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    loop_kwargs = _one(rec, "loop")
    compact = Path(loop_kwargs["compact_md_path"])
    assert compact == ws / "agents" / AGENT / "compact.md", f"compact.md landed at {compact}"
    assert global_dir not in compact.parents, "compact.md stayed in the global dir"

    kill = loop_kwargs["kill_file_path"]
    assert kill is not None, "the root loop got no kill file — the split cannot be graded"
    kill = Path(kill)
    assert kill == global_dir / "KILL", f"the kill file moved to {kill}"
    assert ws not in kill.parents, "the kill switch followed the workspace — it must never"


async def test_subagent_runner_gets_the_workspace_for_state_and_the_global_dir_for_kills(
    tmp_path, monkeypatch
):
    """Asserted twice on purpose: the WIRING, then the CONSEQUENCE.

    The captured kwargs show `start` handed the runner two distinct values. They do NOT show that
    the two still diverge once `_child_runtime_paths` has resolved them — a child's kill file could
    be derived from either. Feeding the captured pair back through the real resolver is what closes
    that gap.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    runner_kwargs = _one(rec, "runner")
    assert runner_kwargs["state_dir"] == ws, \
        f"the runner's state_dir is {runner_kwargs['state_dir']}, not the workspace"
    assert runner_kwargs["config_dir"] == global_dir, \
        f"the runner's config_dir is {runner_kwargs['config_dir']}, not the global dir"

    from localharness.agent.subagent import _child_runtime_paths

    child_cfg = SimpleNamespace(name="child", permissions=None)
    child_kill, child_compact = _child_runtime_paths(
        child_cfg, runner_kwargs["config_dir"], state_dir=runner_kwargs["state_dir"]
    )
    assert child_compact == ws / "agents" / "child" / "compact.md", \
        f"a child's compact.md resolved to {child_compact}"
    assert child_kill == global_dir / "KILL", f"a child's kill file resolved to {child_kill}"
    assert ws not in child_kill.parents, "a child's kill switch followed the workspace"


async def test_memory_store_keeps_the_safety_context_global_while_state_follows_the_work(
    tmp_path, monkeypatch
):
    """Amendment #4, owner-ruled: a workspace may take its memory with it, never the safety voice.

    `base_dir` (memory.db / MEMORY.md / history.jsonl) and `global_base_dir` (DIVISION.md /
    GUARDRAILS.md) are two ctor arguments precisely so a workspace cannot rewrite — or blank by
    omission — the org's safety instructions.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    store_kwargs = _one(rec, "store")
    assert store_kwargs["base_dir"] == str(ws), \
        f"the store's state dir is {store_kwargs['base_dir']}, not the workspace"
    assert store_kwargs["global_base_dir"] == str(global_dir), \
        f"the store's safety-context dir is {store_kwargs['global_base_dir']}, not the global dir"
    assert store_kwargs["base_dir"] != store_kwargs["global_base_dir"], \
        "both ctor arguments got the same value — the split is not observable in this session"


async def test_repl_history_follows_the_work(tmp_path, monkeypatch):
    """The transcript records what you typed while working in THIS project, so it follows the work
    (ruled). Its neighbour two lines up in the source, the kill file, deliberately does not."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    history = Path(_one(rec, "channel")["history_file"])
    assert history == ws / ".repl_history", f"the REPL history landed at {history}"
    assert global_dir not in history.parents, "the REPL history stayed in the global dir"


async def test_repl_carries_the_workspace_layer_for_the_session_lifetime(tmp_path, monkeypatch):
    """41-04 gave the REPL a session-lifetime workspace so `/model` can route its audit record
    without re-walking the filesystem mid-session. `start` is the only thing that can fill it, and
    it must pass the RAW `Optional[Path]` — the REPL's `_audit_base_dir` does its own
    `or self._config_dir` fallback, so `None` there is the signal "no workspace applies"."""
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    repl_kwargs = _one(rec, "repl")
    assert repl_kwargs["workspace"] == ws, \
        f"the REPL's workspace is {repl_kwargs['workspace']}, not {ws}"
    assert repl_kwargs["config_dir"] == global_dir, \
        f"the REPL's config_dir is {repl_kwargs['config_dir']}, not the global dir"


# --------------------------------------------------------------------------------- the control


async def test_with_no_workspace_every_path_is_the_global_dir(tmp_path, monkeypatch):
    """LAYR-03: with nothing up-tree, `state_dir` IS `cfg_path` and the session is byte-identical
    to v0.12.

    This is the assertion that makes `state_dir = workspace if workspace is not None else cfg_path`
    observable rather than assumed. It is also, by construction, blind to every workspace mutation —
    with no workspace the two locals hold the same value, so nothing that swaps one for the other can
    change what this test sees. That blindness is the LAYR-03 argument itself, not a gap in the test.
    """
    _home, global_dir, proj = _global_only_start(tmp_path, monkeypatch)
    rec = _install_recorders(monkeypatch)

    await _drive()

    g_agent_dir = global_dir / "agents" / AGENT
    assert (g_agent_dir / "memory.db").exists(), "memory.db did not land in the global dir"
    assert not (proj / ".localharness").exists(), \
        "the control session created a .localharness in the project — it must create none"

    loop_kwargs = _one(rec, "loop")
    assert Path(loop_kwargs["compact_md_path"]) == g_agent_dir / "compact.md"
    assert Path(loop_kwargs["kill_file_path"]) == global_dir / "KILL"

    runner_kwargs = _one(rec, "runner")
    assert runner_kwargs["state_dir"] == global_dir
    assert runner_kwargs["config_dir"] == global_dir

    store_kwargs = _one(rec, "store")
    assert store_kwargs["base_dir"] == str(global_dir)
    assert store_kwargs["global_base_dir"] == str(global_dir)

    assert Path(_one(rec, "channel")["history_file"]) == global_dir / ".repl_history"

    repl_kwargs = _one(rec, "repl")
    assert repl_kwargs["workspace"] is None, \
        f"the REPL got a workspace ({repl_kwargs['workspace']}) in a session that has none"
    assert repl_kwargs["config_dir"] == global_dir

    assert rec["migrate"] and rec["migrate"][0] == global_dir
