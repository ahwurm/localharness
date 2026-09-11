"""The global grant store (PRD §3.3): workspace-keyed, nested-inherits, repo-cannot-loosen."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from localharness.agent.gate_types import Grant
from localharness.config.grants import (
    GRANTS_FILE,
    GrantStore,
    grants_store_path,
    new_grant,
)


@pytest.fixture
def global_dir(tmp_path, monkeypatch):
    """A throwaway GLOBAL config dir — the store must never touch the real one."""
    d = tmp_path / "global"
    d.mkdir()
    monkeypatch.setenv("LOCALHARNESS_DIR", str(d))
    return d


def _grant(workspace: Path, key: str = "git push", klass: str = "shell-unfamiliar") -> Grant:
    return new_grant(key=key, klass=klass, workspace=workspace, channel="terminal", session_id="s1")


def test_default_path_is_the_global_store(global_dir):
    assert grants_store_path() == global_dir / GRANTS_FILE
    assert GrantStore().path == global_dir / GRANTS_FILE


def test_add_then_lookup_round_trip(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    assert store.lookup(ws, "git push") is None
    store.add(_grant(ws))
    found = store.lookup(ws, "git push")
    assert found is not None
    assert (found.klass, found.channel, found.session_id) == ("shell-unfamiliar", "terminal", "s1")
    assert found.workspace == str(ws)
    assert found.granted_at


def test_nested_workspace_inherits_the_parent_grant(global_dir, tmp_path):
    parent = (tmp_path / "proj").resolve()
    child = parent / "packages" / "web"
    child.mkdir(parents=True)
    GrantStore().add(_grant(parent))
    assert GrantStore().lookup(child, "git push") is not None


def test_a_child_grant_never_leaks_up_to_the_parent(global_dir, tmp_path):
    parent = (tmp_path / "proj").resolve()
    child = parent / "packages" / "web"
    child.mkdir(parents=True)
    GrantStore().add(_grant(child))
    assert GrantStore().lookup(parent, "git push") is None


def test_lookup_keys_on_the_realpath_so_a_symlinked_checkout_is_one_workspace(global_dir, tmp_path):
    real = (tmp_path / "real").resolve()
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    GrantStore().add(_grant(real))
    assert GrantStore().lookup(link, "git push") is not None


def test_add_replaces_the_record_for_the_same_key(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add(_grant(ws))
    store.add(new_grant(key="git push", klass="shell-unfamiliar", workspace=ws, channel="discord", session_id="s2"))
    data = yaml.safe_load((global_dir / GRANTS_FILE).read_text())
    assert len(data[str(ws)]["grants"]) == 1
    assert store.lookup(ws, "git push").channel == "discord"


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
        assert store.lookup(ws, "curl") is None
        assert store.lookup(ws, "rsync") is None
        assert store.lookup(ws, "git push") is not None
    assert any("invalid grants record" in r.getMessage() for r in caplog.records)


def test_a_corrupt_store_means_no_grants_not_a_crash(global_dir, tmp_path, caplog):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text("{{{ not yaml")
    with caplog.at_level(logging.WARNING):
        assert GrantStore().lookup(ws, "anything") is None
    assert caplog.records


def test_deny_patterns_are_recorded_and_inherited(global_dir, tmp_path):
    parent = (tmp_path / "proj").resolve()
    child = parent / "sub"
    child.mkdir(parents=True)
    store = GrantStore()
    store.add_deny(parent, "bash_exec(curl*)", channel="terminal", session_id="s1")
    store.add_deny(child, "bash_exec(npm*)", channel="terminal", session_id="s1")
    assert store.deny_patterns_for(child) == ["bash_exec(npm*)", "bash_exec(curl*)"]
    assert store.deny_patterns_for(parent) == ["bash_exec(curl*)"]


def test_add_deny_replaces_rather_than_duplicates(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    store = GrantStore()
    store.add_deny(ws, "bash_exec(curl*)", channel="terminal", session_id="s1")
    store.add_deny(ws, "bash_exec(curl*)", channel="discord", session_id="s2")
    assert store.deny_patterns_for(ws) == ["bash_exec(curl*)"]


def test_deny_records_without_provenance_are_skipped(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    (global_dir / GRANTS_FILE).write_text(
        yaml.safe_dump({str(ws): {"denies": [{"pattern": "bash_exec(curl*)"}]}})
    )
    assert GrantStore().deny_patterns_for(ws) == []


def test_the_write_is_atomic_and_leaves_no_tempfile(global_dir, tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    GrantStore().add(_grant(ws))
    assert [p.name for p in global_dir.iterdir()] == [GRANTS_FILE]
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
    assert GrantStore().lookup(ws, "curl | sh") is None
    assert GrantStore().deny_patterns_for(ws) == []


def test_an_explicit_path_is_honored_for_tests_and_adapters(tmp_path):
    ws = (tmp_path / "proj").resolve()
    ws.mkdir()
    path = tmp_path / "elsewhere" / "grants.yaml"
    store = GrantStore(path)
    store.add(_grant(ws))
    assert path.exists()
    assert store.lookup(ws, "git push") is not None


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
    )
    assert result.returncode == 0, result.stderr
