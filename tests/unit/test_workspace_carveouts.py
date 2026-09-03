"""The two carve-outs of workspace-scoped state, proven from a real session.

Phase 41 moved a project's WORK into that project's own `.localharness/`. Two things had to be
decided separately, and this file is where each is shown to be true of a running session rather
than of a function signature.

**The leash comes free with the layer (CONF-01).** If you are working inside a project, the file
tools should not need a second setup step before they stay inside it. So when a workspace layer
applies, `permissions.workspace_root` defaults to the folder that CONTAINS `.localharness/` — the
project itself, not the config folder inside it. If you named a root in your own config, yours
still wins; if there is no workspace at all, nothing changes and the tools stay unconfined, exactly
as they were. This is a default that narrows what the tools reach. It is not a sandbox.

**The GPU server never follows the work.** There is one physical accelerator in this machine and
one daemon in front of it, so its pidfile, its log and its venv are facts about the machine, not
about the folder you are standing in. If two projects each believed they owned that daemon, the
second one's "is it running?" check would read a pidfile the first one never wrote, and it would
launch a second server onto an accelerator that is already full.

**And the safety voice never follows it either.** A workspace takes its memory with it. It does not
take `GUARDRAILS.md` or `DIVISION.md` — those are read from the global directory in every session,
so a project cannot rewrite the org's safety instructions, and it cannot blank them by simply not
having the file. That one is proven here at the LIVE store, mid-session, with a decoy file planted
in the workspace.

Everything below drives the REAL `_start_async` offline from inside a project, reusing the harness
`tests/unit/test_workspace_state_landing.py` built (imported, never re-copied, so the two files
cannot drift apart). No server is started, no GPU is touched and no network call is made.
"""
from __future__ import annotations

import linecache
import sys

import yaml

from localharness.provider.server import log_path, pid_path, venv_vllm_bin

# 41-05's workspace-session recipe: a fake `$HOME` holding the global layer, an in-project
# `.git` marker so the workspace loads with no trust prompt, `LOCALHARNESS_HOME`/`LOCALHARNESS_DIR`
# cleared (either one would switch discovery off and every assertion here would pass against a
# session that has no workspace at all), and `_drive()` = `_start_async(None, ...)`, whose `None`
# is what keeps discovery alive.
from tests.unit.test_workspace_state_landing import (
    AGENT,
    _drive,
    _global_only_start,
    _workspace_start,
)


# --------------------------------------------------------------------------------- recorders


def _record_tool_registration(monkeypatch) -> list[dict]:
    """Record every `register_builtin_tools` call, and still run the real one.

    A wrapper rather than a double: the drive keeps building its real tool registry, so a mistake
    that made registration fail would surface as a broken drive instead of a quietly empty list.
    `_start_async` imports this name INSIDE the function body, so patching the module attribute is
    what takes effect for a drive.
    """
    import localharness.tools.builtin as _builtin

    real = _builtin.register_builtin_tools
    calls: list[dict] = []

    async def _rec(registry, *args, **kwargs):
        calls.append(dict(kwargs))
        return await real(registry, *args, **kwargs)

    monkeypatch.setattr("localharness.tools.builtin.register_builtin_tools", _rec)
    return calls


def _record_global_config_dir(monkeypatch) -> list[dict]:
    """Record `(argument, result, calling source line)` for every `global_config_dir` call.

    The source line is the point. `cfg_path` and `global_config_dir(config_dir)` evaluate to the
    SAME path in every session shipped today, so no value assertion can tell them apart — the
    guarantee this file grades is that the GPU server's directory is derived from the global-only
    function, so that a future change to what `cfg_path` means cannot drag the daemon along
    (the same reason `tests/unit/test_server_dir_global_pin.py` exists). Recording which line asked
    the question is what makes that observable from a live drive.
    """
    import localharness.config.paths as _paths

    real = _paths.global_config_dir
    calls: list[dict] = []

    def _rec(config_dir=None):
        frame = sys._getframe(1)
        linecache.checkcache(frame.f_code.co_filename)
        result = real(config_dir)
        calls.append(
            {
                "arg": config_dir,
                "result": result,
                "caller": frame.f_code.co_name,
                "source": linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip(),
            }
        )
        return result

    monkeypatch.setattr("localharness.config.paths.global_config_dir", _rec)
    return calls


def _record_store_instances(monkeypatch) -> list:
    """Stash the MemoryStore OBJECTS the drive builds, not just their ctor kwargs.

    41-05 already asserts on the kwargs. The question here is different: which file does the
    RUNNING store read? That is a property of the instance's resolved paths, so the instance is
    what gets captured.
    """
    import localharness.memory.sqlite as _sqlite

    real_init = _sqlite.MemoryStore.__init__
    stores: list = []

    def _rec_init(self, *args, **kwargs):
        stores.append(self)
        return real_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.memory.sqlite.MemoryStore.__init__", _rec_init)
    return stores


def _only(items: list, what: str):
    """The single recorded item, with the did-it-bite guard.

    Without this, a mis-targeted patch reads as an empty list and every assertion below is skipped
    rather than failed — a green test on a recorder that never fired.
    """
    assert items, f"{what} was never recorded — the patch did not bite"
    assert len(items) == 1, f"{what} happened {len(items)} times, expected once: {items}"
    return items[0]


# ------------------------------------------------------- criterion 4: the confinement default


async def test_a_workspace_session_confines_the_file_tools_to_the_project_root(
    tmp_path, monkeypatch
):
    """The default reaches the tools, and it is the PROJECT, not the config folder inside it.

    `register_builtin_tools` is where the root is bound into the Write / Edit / BashExec instances,
    once, at startup — subagents reuse those same instances through the shared registry. So this
    kwarg is the whole of "the file tools are confined" for every agent in the session.

    Both halves are asserted in one test on purpose. `str(proj)` and `str(ws)` differ by a single
    path component, and confining every agent INSIDE `.localharness/` is a mistake that type-checks,
    names a real directory, and would look correct in any assertion that only checked for
    not-None.

    ORDER IS DELIBERATE: the specific check runs first. Measured, not assumed — with the general
    equality first, a `.parent` regression reddens THAT line and the dotdir assertion below it can
    never fire, so it would be decorative. This way the dotdir mistake is caught by the assertion
    written for it, and every other wrong value is caught by the equality underneath.
    """
    _home, _global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    proj = ws.parent
    calls = _record_tool_registration(monkeypatch)

    await _drive()

    root = _only(calls, "register_builtin_tools")["workspace_root"]
    assert root != str(ws), (
        "the file tools were confined to the .localharness config folder instead of the project "
        "that contains it"
    )
    assert root == str(proj), f"the file tools were confined to {root}, not the project {proj}"


async def test_an_explicit_workspace_root_still_wins_inside_a_workspace(tmp_path, monkeypatch):
    """A default fills a gap; it does not overwrite an answer you already gave.

    The project root is asserted ABSENT rather than only asserting the explicit value is present:
    a test that checks one value is there passes just as happily when both are.
    """
    _home, _global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    proj = ws.parent
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (ws / "agents" / f"{AGENT}.yaml").write_text(
        yaml.dump(
            {
                "name": AGENT,
                "role": "Test role",
                "model": "inherit",
                "permissions": {"workspace_root": str(elsewhere)},
            }
        )
    )
    calls = _record_tool_registration(monkeypatch)

    await _drive()

    root = _only(calls, "register_builtin_tools")["workspace_root"]
    assert root == str(elsewhere), f"the explicit root lost: the tools got {root}"
    assert root != str(proj), "the workspace default overwrote the root the user configured"


async def test_with_no_workspace_the_file_tools_stay_unconfined(tmp_path, monkeypatch):
    """The control. With nothing up-tree there is no layer to default from, so the contract is
    untouched: `workspace_root=None`, which means unconfined, exactly as before v0.13."""
    _home, _global_dir, _proj = _global_only_start(tmp_path, monkeypatch)
    calls = _record_tool_registration(monkeypatch)

    await _drive()

    root = _only(calls, "register_builtin_tools")["workspace_root"]
    assert root is None, f"a session with no workspace confined its tools to {root}"


# --------------------------------------------------------- criterion 5: the GPU server carve-out


async def test_a_workspace_session_resolves_the_gpu_server_dir_globally(tmp_path, monkeypatch):
    """A session WITH a workspace still asks the global-only function where the daemon lives.

    What this proves: during a real workspace drive, the line that binds `server_cfg_path` called
    `global_config_dir`, got the global directory back, and no `global_config_dir` call anywhere in
    that session returned a path inside the workspace.

    What it does NOT prove, stated plainly: that the ~16 vLLM lifecycle call sites still USE that
    value. That is a structural pin — `tests/unit/test_server_dir_global_pin.py`, unchanged since
    it was written — and it is re-run alongside this file rather than re-derived here. The two
    together are criterion 5; neither is it alone.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    calls = _record_global_config_dir(monkeypatch)

    await _drive()

    assert calls, "global_config_dir was never called — the patch did not bite"
    server_calls = [c for c in calls if "server_cfg_path" in c["source"]]
    assert server_calls, (
        "no global_config_dir call came from the line that binds server_cfg_path — the GPU "
        "server's directory is no longer derived from the global-only function. It may still be "
        "value-identical today, which is exactly why this is graded structurally: "
        f"the calls this session did make were {[c['source'] for c in calls]}"
    )
    bound = _only(server_calls, "the global_config_dir call that binds server_cfg_path")

    # The sharpest claim goes first, for the same reason as the other two tests in this file: the
    # assertion that fires should be the one that names what went wrong.
    for call in calls:
        assert call["result"] != ws and ws not in call["result"].parents, (
            f"a global_config_dir call from `{call['source']}` returned a path inside the "
            f"workspace: {call['result']}"
        )
    assert bound["result"] == global_dir, (
        f"the GPU server directory resolved to {bound['result']}, not the global dir {global_dir}"
    )

    # The captured value fed back through the real path helpers — the criterion is written in terms
    # of the three files, so the three files are what gets asserted, not just the directory.
    for path in (pid_path(bound["result"]), log_path(bound["result"]), venv_vllm_bin(bound["result"])):
        assert global_dir in path.parents, f"{path} is not under the global dir"
        assert ws not in path.parents, f"{path} followed the workspace — it must never"


async def test_the_machine_wide_files_stay_in_the_global_dir_during_a_workspace_session(
    tmp_path, monkeypatch
):
    """The rest of the never-follows list, recorded from the same real workspace drive.

    Three sites, one reason each. `plugins/` is executable code the machine's owner installed, so a
    project must not be able to add to it by sitting in a folder. The packaged `tools/` are shipped
    with the harness, not with your repository. The root agent's own yaml is rewritten in place by
    a migration that DELETES the old file — pointing that at a project folder would edit someone's
    repository without being asked.

    These three are what the published table's right-hand column claims beyond the kill switch, the
    GPU daemon and the safety context, so they are asserted rather than assumed.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)

    tools_dirs: list = []
    monkeypatch.setattr(
        "localharness.cli.start_cmd._ensure_packaged_tools", lambda d: tools_dirs.append(d)
    )
    agent_yaml_dirs: list = []
    monkeypatch.setattr(
        "localharness.cli.start_cmd._migrate_legacy_root_agent_yaml",
        lambda d: agent_yaml_dirs.append(d),
    )

    import localharness.plugins.loader as _plugins

    real_plugin_init = _plugins.PluginLoader.__init__
    plugin_kwargs: list = []

    def _rec_plugin_init(self, *args, **kwargs):
        plugin_kwargs.append(dict(kwargs))
        return real_plugin_init(self, *args, **kwargs)

    monkeypatch.setattr("localharness.plugins.loader.PluginLoader.__init__", _rec_plugin_init)

    await _drive()

    assert _only(tools_dirs, "the packaged-tools install") == global_dir
    assert _only(agent_yaml_dirs, "the root-agent yaml migration") == global_dir / "agents"
    plugins_dir = _only(plugin_kwargs, "the plugin loader")["plugins_dir"]
    assert plugins_dir == global_dir / "plugins", f"plugins were loaded from {plugins_dir}"
    for path in (_only(tools_dirs, "x"), _only(agent_yaml_dirs, "x"), plugins_dir):
        assert ws not in path.parents and path != ws, f"{path} followed the workspace"


async def test_a_model_swap_inside_a_workspace_session_audits_in_the_project(tmp_path, monkeypatch):
    """MEMS-04 end to end: the swap is recorded where the work is, the overlay is written globally.

    The previous plan proved this in three pieces — that `start` hands the REPL its workspace, that
    the REPL's audit directory is that workspace, and that the persist function honours the
    directory it is given. This drives the REPL's OWN `_persist_default_model` inside a live
    session instead, so the three pieces are joined by the harness rather than by a summary.

    Both directions again: a swap that wrote its overlay into the project would fork the machine's
    one model server, and a swap that audited globally would put this project's history in another
    project's log.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    swapped: dict = {}

    async def _swap_the_model(self):
        swapped["ok"] = await self._persist_default_model("swapped-model")
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _swap_the_model)

    await _drive()

    assert swapped.get("ok") is True, f"the in-session swap did not persist: {swapped}"
    assert (ws / "audit.jsonl").exists(), "the swap was not recorded in the project"
    assert not (global_dir / "audit.jsonl").exists(), (
        "the swap was recorded in the global log — this project's history went somewhere else"
    )
    assert (global_dir / "overrides.yaml").exists(), "the overlay did not land in the global dir"
    assert not (ws / "overrides.yaml").exists(), (
        "the overlay followed the work into the project — there is one model server per machine"
    )


async def test_one_repl_answers_two_questions_about_one_session(tmp_path, monkeypatch):
    """The same object, two directories, on purpose.

    The REPL is where a mid-session `/model` swap is handled, so it holds both answers at once: the
    audit record follows the work, the GPU daemon's state does not. Asserting they DIFFER is the
    point — in a global-only session the two are equal and the split is invisible.
    """
    from localharness.cli.repl import OrchestratorREPL

    global_dir = tmp_path / "home" / ".localharness"
    ws = tmp_path / "home" / "proj" / ".localharness"

    repl = OrchestratorREPL(None, None, None, None, config_dir=global_dir, workspace=ws)

    assert repl._server_config_dir == global_dir, (
        f"the GPU server dir is {repl._server_config_dir}, not the global dir"
    )
    assert repl._audit_base_dir == ws, f"the audit dir is {repl._audit_base_dir}, not the workspace"
    assert repl._server_config_dir != repl._audit_base_dir, (
        "both properties answered with the same directory — the divergence is not observable"
    )


async def test_a_fresh_workspace_starts_with_a_memory_of_its_own(tmp_path, monkeypatch):
    """Nothing is copied out of your global store, and nothing in it is rewritten.

    This backs the published promise that opening the harness in a new project gives that project
    an empty memory rather than a copy of everything you have ever told the harness — and that the
    global store you already had is left exactly as it was. A global `MEMORY.md` is planted with a
    marker: it must not appear in the project's own notes, and it must still be there afterwards.

    Stated honestly: no mutation of shipped code reddens this one, because nothing in the harness
    copies memory between stores — that absence IS the promise. It is a forward guard, so a later
    change that helpfully seeds a new workspace from the global store cannot land unnoticed.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    global_notes = global_dir / "agents" / AGENT / "MEMORY.md"
    global_notes.parent.mkdir(parents=True)
    global_notes.write_text("GLOBAL-MEMORY-MARKER\n")

    await _drive()

    project_notes = (ws / "agents" / AGENT / "MEMORY.md").read_text()
    assert "GLOBAL-MEMORY-MARKER" not in project_notes, (
        f"the project's memory was seeded from the global store: {project_notes!r}"
    )
    assert global_notes.read_text() == "GLOBAL-MEMORY-MARKER\n", (
        "the global memory was moved, rewritten or emptied by a session in a project"
    )


# ------------------------------------------------- the safety split, at the live store


async def test_the_running_store_reads_its_guardrails_from_the_global_dir(tmp_path, monkeypatch):
    """The live proof of amendment #4: memory follows the work, the safety voice never does.

    A decoy `GUARDRAILS.md` and `DIVISION.md` are planted INSIDE the workspace. If the store's
    safety-context directory ever followed the state directory, the session would read the decoys —
    which is exactly how a project would silence the org's instructions.

    The read is real and it happens mid-session: the stubbed REPL loop calls the live store's
    `load_context()` while the database is still open, which is the same call the agent loop makes
    to build a system prompt. A path assertion alone would only show where the store is pointed.
    """
    _home, global_dir, ws = _workspace_start(tmp_path, monkeypatch)
    stores = _record_store_instances(monkeypatch)

    (global_dir / "orgs" / "default").mkdir(parents=True)
    (global_dir / "orgs" / "default" / "GUARDRAILS.md").write_text("GLOBAL-GUARDRAILS-MARKER\n")
    (global_dir / "divisions" / "default").mkdir(parents=True)
    (global_dir / "divisions" / "default" / "DIVISION.md").write_text("GLOBAL-DIVISION-MARKER\n")

    (ws / "orgs" / "default").mkdir(parents=True)
    (ws / "orgs" / "default" / "GUARDRAILS.md").write_text("WORKSPACE-DECOY-MARKER\n")
    (ws / "divisions" / "default").mkdir(parents=True)
    (ws / "divisions" / "default" / "DIVISION.md").write_text("WORKSPACE-DECOY-MARKER\n")

    seen: dict = {}

    async def _read_the_safety_context(self):
        # Runs INSIDE the live session, in place of the interactive loop, so the store is open and
        # this is the same code path a real turn takes. `self._store` is the MemoryStore `start`
        # built and handed to the REPL.
        seen["ctx"] = await self._store.load_context()
        return None

    monkeypatch.setattr("localharness.cli.repl.OrchestratorREPL.run", _read_the_safety_context)

    await _drive()

    # The CONTENT the live session actually read comes first, and the paths it read from come
    # second. Measured: with the path assertions on top, a store whose safety directory followed
    # the workspace reddens THOSE, and the decoy assertions below could never fire — the strongest
    # claim in this file would have been decorative. This order makes the failure say what actually
    # went wrong: the session read the project's own guardrails.
    ctx = seen.get("ctx")
    assert ctx is not None, "the live load_context() never ran — the stubbed loop did not fire"
    assert "WORKSPACE-DECOY-MARKER" not in ctx.guardrails_md, (
        "the live session read the workspace's decoy GUARDRAILS.md — a project can silence the "
        "org's safety instructions by planting its own file"
    )
    assert "GLOBAL-GUARDRAILS-MARKER" in ctx.guardrails_md, (
        f"the live session's guardrails came from somewhere else: {ctx.guardrails_md!r}"
    )
    assert "WORKSPACE-DECOY-MARKER" not in ctx.division_md, (
        "the live session read the workspace's decoy DIVISION.md"
    )
    assert "GLOBAL-DIVISION-MARKER" in ctx.division_md, (
        f"the live session's division context came from somewhere else: {ctx.division_md!r}"
    )

    # v0.13 MEMS-02 (42-03): a workspace session now builds TWO stores — the session's own, and
    # the CONSTRUCTED-NOT-OPENED global twin the recall router opens only if `recall_scope` asks
    # for it. The assertions below are about the store the SESSION reads and writes through, which
    # is the first one built. The count stays pinned so a THIRD construction still trips this test
    # (that is what `_only` was buying, and it is not given up here).
    assert stores, "the MemoryStore was never recorded — the patch did not bite"
    assert len(stores) == 2, (
        f"expected the primary + the global recall twin, got {len(stores)}: {stores}"
    )
    store = stores[0]
    assert store._db_path == ws / "agents" / AGENT / "memory.db", (
        f"the running store's database is at {store._db_path}, not in the workspace"
    )
    assert store._guardrails_path == global_dir / "orgs" / "default" / "GUARDRAILS.md", (
        f"the running store reads guardrails from {store._guardrails_path}"
    )
    assert store._division_md_path == global_dir / "divisions" / "default" / "DIVISION.md", (
        f"the running store reads division context from {store._division_md_path}"
    )
    assert ws not in store._guardrails_path.parents, "the guardrails file followed the workspace"
