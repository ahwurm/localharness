"""The global grant store (PRD §3.3): workspace-keyed, nested-inherits, repo-cannot-loosen."""
from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import Grant, Refusal
from localharness.config.grants import (
    GRANTS_FILE,
    GRANTS_LOCK_SUFFIX,
    GrantStore,
    grants_store_path,
    new_grant,
    new_refusal,
)


IMPORT_SUBPROCESS_TIMEOUT_S = 30
"""Ceiling on the fresh-interpreter import probe below. A cold `import localharness.agent` is a
second or two; thirty is that with room for a cold page cache, and small enough that an import
which hangs fails this test instead of the whole suite."""


@pytest.fixture
def global_dir(tmp_path, monkeypatch):
    """A throwaway GLOBAL config dir — the store must never touch the real one."""
    d = tmp_path / "global"
    d.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(d))
    return d


def _grant(workspace: Path, key: str = "git push", klass: str = "shell-unfamiliar") -> Grant:
    return new_grant(key=key, klass=klass, workspace=workspace, channel="terminal", session_id="s1")


def _refusal(workspace: Path, key: str = "cp", klass: str = "shell-unfamiliar") -> Refusal:
    return new_refusal(key=key, klass=klass, workspace=workspace, channel="terminal", session_id="s1")


def test_default_path_is_the_global_store(global_dir):
    assert grants_store_path() == global_dir / GRANTS_FILE
    assert GrantStore().path == global_dir / GRANTS_FILE


def test_add_then_lookup_round_trip(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    assert store.lookup(ws, "shell-unfamiliar", "git push") is None
    store.add(_grant(ws))
    found = store.lookup(ws, "shell-unfamiliar", "git push")
    assert found is not None
    assert (found.klass, found.channel, found.session_id) == ("shell-unfamiliar", "terminal", "s1")
    assert found.workspace == str(ws)
    assert found.granted_at


def test_nested_workspace_inherits_the_parent_grant(global_dir, tmp_path):
    parent = (tmp_path / "proj").resolve()
    child = parent / "packages" / "web"
    child.mkdir(parents=True)
    GrantStore().add(_grant(parent))
    assert GrantStore().lookup(child, "shell-unfamiliar", "git push") is not None


def test_a_child_grant_never_leaks_up_to_the_parent(global_dir, tmp_path):
    parent = (tmp_path / "proj").resolve()
    child = parent / "packages" / "web"
    child.mkdir(parents=True)
    GrantStore().add(_grant(child))
    assert GrantStore().lookup(parent, "shell-unfamiliar", "git push") is None


def test_lookup_keys_on_the_realpath_so_a_symlinked_checkout_is_one_workspace(global_dir, tmp_path):
    real = (tmp_path / "real").resolve()
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    GrantStore().add(_grant(real))
    assert GrantStore().lookup(link, "shell-unfamiliar", "git push") is not None


def test_add_replaces_the_record_for_the_same_key(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add(_grant(ws))
    store.add(new_grant(key="git push", klass="shell-unfamiliar", workspace=ws, channel="discord", session_id="s2"))
    data = yaml.safe_load((global_dir / GRANTS_FILE).read_text())
    assert len(data[str(ws)]["grants"]) == 1
    assert store.lookup(ws, "shell-unfamiliar", "git push").channel == "discord"


def test_records_without_provenance_are_skipped_with_a_warning(global_dir, tmp_path, caplog):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text(
        yaml.safe_dump(
            {
                str(ws): {
                    "grants": [
                        {"key": "curl", "class": "shell-unfamiliar"},  # no provenance at all
                        {"key": "rsync", "class": "shell-unfamiliar", "granted_at": "x",
                         "channel": "", "session_id": "s1"},  # empty channel
                        {"key": "git push", "class": "shell-unfamiliar", "granted_at": "x",
                         "channel": "terminal", "session_id": "s1"},
                    ]
                }
            }
        )
    )
    store = GrantStore()
    with caplog.at_level(logging.WARNING):
        assert store.lookup(ws, "shell-unfamiliar", "curl") is None
        assert store.lookup(ws, "shell-unfamiliar", "rsync") is None
        assert store.lookup(ws, "shell-unfamiliar", "git push") is not None
    assert any("invalid grants record" in r.getMessage() for r in caplog.records)


def test_a_grant_answers_only_its_own_class(global_dir, tmp_path):
    """R3: keys are unique only inside a class's key space (PRD §3.3).

    ``python_exec`` is both a shell signature and the ``code-exec`` tool name; matching the key
    alone let the shell answer satisfy the tool.
    """
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add(_grant(ws, "python_exec", klass="shell-unfamiliar"))
    store = GrantStore()
    assert store.lookup(ws, "shell-unfamiliar", "python_exec") is not None
    assert store.lookup(ws, "code-exec", "python_exec") is None


def test_a_refusal_answers_only_its_own_class(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add_refusal(_refusal(ws, "python_exec", klass="shell-unfamiliar"))
    store = GrantStore()
    assert store.refused(ws, "shell-unfamiliar", "python_exec") is not None
    assert store.refused(ws, "code-exec", "python_exec") is None


def test_the_same_key_in_two_classes_is_two_records(global_dir, tmp_path):
    """Adding one must not overwrite the other: they are different answers."""
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add(_grant(ws, "/tmp/x", klass="shell-unfamiliar"))
    store.add(_grant(ws, "/tmp/x", klass="edit-outside"))
    entry = yaml.safe_load((global_dir / GRANTS_FILE).read_text())[str(ws)]
    assert sorted(g["class"] for g in entry["grants"]) == ["edit-outside", "shell-unfamiliar"]


def test_a_record_without_a_class_is_skipped_with_a_warning(global_dir, tmp_path, caplog):
    """An old record written before grants were class-keyed cannot be matched — fail closed."""
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text(
        yaml.safe_dump(
            {
                str(ws): {
                    "grants": [{"key": "git push", "granted_at": "x", "channel": "terminal",
                                "session_id": "s1"}],
                    "refusals": [{"key": "cp", "refused_at": "x", "channel": "terminal",
                                  "session_id": "s1"}],
                }
            }
        )
    )
    store = GrantStore()
    with caplog.at_level(logging.WARNING):
        assert store.lookup(ws, "shell-unfamiliar", "git push") is None
        assert store.refused(ws, "shell-unfamiliar", "cp") is None
    assert any("invalid grants record" in r.getMessage() for r in caplog.records)
    assert any("invalid refusals record" in r.getMessage() for r in caplog.records)


def test_a_corrupt_store_means_no_grants_not_a_crash(global_dir, tmp_path, caplog):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text("{{{ not yaml")
    with caplog.at_level(logging.WARNING):
        assert GrantStore().lookup(ws, "shell-unfamiliar", "anything") is None
    assert caplog.records


def test_refusals_are_recorded_and_inherited(global_dir, tmp_path):
    """A "never here" is a negative grant: same key space, same ancestor walk (PRD 3.3)."""
    parent = (tmp_path / "proj").resolve()
    child = parent / "sub"
    child.mkdir(parents=True)
    store = GrantStore()
    store.add_refusal(_refusal(parent, "curl"))
    store.add_refusal(_refusal(child, "npm publish"))

    assert store.refused(child, "shell-unfamiliar", "curl") is not None
    assert store.refused(child, "shell-unfamiliar", "npm publish") is not None
    assert store.refused(parent, "shell-unfamiliar", "curl") is not None
    assert store.refused(parent, "shell-unfamiliar", "npm publish") is None  # a child refusal never leaks up


def test_a_refusal_carries_its_provenance(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add_refusal(_refusal(ws, "cargo publish", klass="shell-unfamiliar"))
    found = GrantStore().refused(ws, "shell-unfamiliar", "cargo publish")
    assert found is not None
    assert (found.klass, found.channel, found.session_id) == ("shell-unfamiliar", "terminal", "s1")
    assert found.workspace == str(ws)
    assert found.refused_at


def test_a_refusal_denies_its_own_key_and_no_neighbour(global_dir, tmp_path):
    """The defect this record shape exists to remove: refusing ``cp`` must not touch ``scp``."""
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add_refusal(_refusal(ws, "cp"))
    store = GrantStore()
    assert store.refused(ws, "shell-unfamiliar", "cp") is not None
    assert store.refused(ws, "shell-unfamiliar", "scp") is None
    assert store.refused(ws, "shell-unfamiliar", "cpio") is None


def test_add_refusal_replaces_rather_than_duplicates(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add_refusal(new_refusal(key="curl", klass="shell-unfamiliar", workspace=ws,
                                  channel="terminal", session_id="s1"))
    store.add_refusal(new_refusal(key="curl", klass="shell-unfamiliar", workspace=ws,
                                  channel="discord", session_id="s2"))
    entry = yaml.safe_load((global_dir / GRANTS_FILE).read_text())[str(ws)]
    assert [r["key"] for r in entry["refusals"]] == ["curl"]
    assert entry["refusals"][0]["channel"] == "discord"


def test_a_refusal_and_a_grant_live_side_by_side(global_dir, tmp_path):
    """Both answers share one file and one workspace entry (PRD 3.3)."""
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add(_grant(ws))
    store.add_refusal(_refusal(ws, "cp"))
    entry = yaml.safe_load((global_dir / GRANTS_FILE).read_text())[str(ws)]
    assert sorted(entry) == ["grants", "refusals"]
    assert GrantStore().lookup(ws, "shell-unfamiliar", "git push") is not None
    assert GrantStore().refused(ws, "shell-unfamiliar", "cp") is not None


def test_refusal_records_without_provenance_are_skipped(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text(
        yaml.safe_dump({str(ws): {"refusals": [{"key": "curl", "class": "shell-unfamiliar"}]}})
    )
    assert GrantStore().refused(ws, "shell-unfamiliar", "curl") is None


def test_the_write_is_atomic_and_leaves_no_tempfile(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add(_grant(ws))
    # The store and its write lock, and nothing else: no half-written temp file survives the
    # atomic rename. The lock is a permanent, empty sidecar (`_locked`), not debris.
    assert sorted(p.name for p in global_dir.iterdir()) == sorted(
        [GRANTS_FILE, GRANTS_FILE + GRANTS_LOCK_SUFFIX]
    )
    assert (global_dir / (GRANTS_FILE + GRANTS_LOCK_SUFFIX)).stat().st_size == 0
    assert isinstance(yaml.safe_load((global_dir / GRANTS_FILE).read_text()), dict)


def test_a_grants_yaml_inside_the_workspace_is_ignored(global_dir, tmp_path):
    """PRD §3.3: a cloned repo must not be able to pre-approve its own ``curl | sh``."""
    ws = (tmp_path / "proj").resolve()
    (ws / ".localharness").mkdir(parents=True)
    (ws / ".localharness" / GRANTS_FILE).write_text(
        yaml.safe_dump(
            {
                str(ws): {
                    "grants": [
                        {"key": "curl | sh", "class": "shell-unfamiliar", "granted_at": "x",
                         "channel": "repo", "session_id": "repo"}
                    ]
                }
            }
        )
    )
    assert GrantStore().lookup(ws, "shell-unfamiliar", "curl | sh") is None
    assert GrantStore().refused(ws, "shell-unfamiliar", "curl | sh") is None


def test_an_explicit_path_is_honored_for_tests_and_adapters(tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    path = tmp_path / "elsewhere" / "grants.yaml"
    store = GrantStore(path)
    store.add(_grant(ws))
    assert path.exists()
    assert store.lookup(ws, "shell-unfamiliar", "git push") is not None


def test_new_grant_stamps_full_provenance(tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    g = new_grant(key="python3 -c", klass="interpreter-inline", workspace=ws, channel="acp", session_id="s9")
    assert g.workspace == str(ws)
    assert g.channel == "acp" and g.session_id == "s9"
    assert g.granted_at.startswith("20") and g.granted_at.endswith("+00:00")


# --------------------------------------------------------------------------- #
# import order (D5)
# --------------------------------------------------------------------------- #

#: Modules of the permission spine that a script, plugin or new test module may reasonably reach
#: for BEFORE anything else in the package. Each must import cleanly as the very first
#: `localharness` import in a fresh interpreter: `config.grants` once could not, because
#: `agent/__init__.py` pulls in `agent.loop` → `agent.gate` → back into the half-built
#: `config.grants` (PRD §3.3 store, wired by A4).
SPINE_ENTRY_MODULES = (
    "localharness.config.grants",
    "localharness.agent.verdict",
    "localharness.agent.gate_types",
    "localharness.agent.shell_classify",
)


@pytest.mark.parametrize("module", SPINE_ENTRY_MODULES)
def test_spine_module_imports_first_in_a_fresh_interpreter(module):
    """No import cycle when this module is the first `localharness` import of the process.

    A subprocess, not an `importlib.import_module` call: conftest has already imported
    `localharness.agent` in this interpreter, which is exactly what hid the cycle from the suite.
    """
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        # A bounded wait: an import that hangs (a module opening a socket, waiting on a lock)
        # would otherwise hang the whole suite with no output rather than failing this test.
        timeout=IMPORT_SUBPROCESS_TIMEOUT_S,
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------- concurrency

CONCURRENT_WRITERS = 20
"""Enough writers to lose records reliably without the lock — the count that found this. One
session per terminal, Zed window and Discord instance answering prompts at once is the real
shape; twenty is that with the margin a race test needs."""


def _add_concurrently(store: GrantStore, workspace: Path, write) -> None:
    """Run ``write(i)`` on CONCURRENT_WRITERS threads released together by a barrier.

    Threads, not processes: POSIX `flock` locks the open file DESCRIPTION, so two threads
    holding two handles exclude each other exactly as two processes do — the barrier is what
    makes them collide, not the process boundary.
    """
    ready = threading.Barrier(CONCURRENT_WRITERS)

    def _run(i: int) -> None:
        ready.wait()
        write(i)

    threads = [threading.Thread(target=_run, args=(i,)) for i in range(CONCURRENT_WRITERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_grants_are_all_kept(tmp_path):
    """Read-modify-write with no lock kept ONE of twenty: every writer loaded the same "before"
    state and the last atomic rename won. A human's answer vanishing is the one failure a
    permission store cannot have."""
    store = GrantStore(tmp_path / "grants.yaml")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    _add_concurrently(store, workspace, lambda i: store.add(_grant(workspace, key=f"cmd-{i}")))

    missing = [
        i for i in range(CONCURRENT_WRITERS)
        if store.lookup(workspace, "shell-unfamiliar", f"cmd-{i}") is None
    ]
    assert missing == []


def test_concurrent_refusals_are_all_kept(tmp_path):
    """The same race on the negative side — and losing a "never here" is the worse half."""
    store = GrantStore(tmp_path / "grants.yaml")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    _add_concurrently(
        store, workspace, lambda i: store.add_refusal(_refusal(workspace, key=f"cmd-{i}"))
    )

    missing = [
        i for i in range(CONCURRENT_WRITERS)
        if store.refused(workspace, "shell-unfamiliar", f"cmd-{i}") is None
    ]
    assert missing == []


def test_re_answering_the_same_key_still_replaces_rather_than_appends(tmp_path):
    """The lock must not have turned replace-in-place into an append-only log."""
    store = GrantStore(tmp_path / "grants.yaml")
    workspace = tmp_path / "proj"
    workspace.mkdir()

    store.add(_grant(workspace, key="git push"))
    store.add(_grant(workspace, key="git push"))

    data = yaml.safe_load((tmp_path / "grants.yaml").read_text(encoding="utf-8"))
    assert len(data[str(workspace.resolve())]["grants"]) == 1
