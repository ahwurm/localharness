"""config/trust.py — which OUTSIDE-the-project workspaces the user agreed to load (LAYR-05).

Three properties this file exists to hold down:

1. The record lives in the GLOBAL config dir. A workspace that could vouch for itself is not
   a trust boundary — so nothing is ever written inside the directory being judged.
2. Unknown reads as None (undecided), never False. A declined-in-a-script session must not
   harden into a permanent "no"; only an answered prompt records anything.
3. The key is the realpath, so a symlinked checkout and its real path are ONE entry, while a
   git worktree (a real sibling directory) is a separate workspace by design.

The autouse `_isolate_localharness_home` fixture (tests/conftest.py) already points
LOCALHARNESS_HOME at a tmp dir, so the store is hermetic without extra setup.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml


@pytest.fixture(autouse=True)
def _global_dir_is_the_hermetic_home(monkeypatch):
    """LOCALHARNESS_DIR outranks LOCALHARNESS_HOME in resolve_config_dir's chain — clear it so
    a developer who exports it doesn't silently point these tests at their real config dir."""
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)


def _workspace(root: Path) -> Path:
    ws = root / ".localharness"
    ws.mkdir(parents=True)
    return ws


def _expected_store() -> Path:
    return Path(os.environ["LOCALHARNESS_HOME"]) / "trusted_workspaces.yaml"


def test_trust_store_lives_in_the_global_config_dir():
    from localharness.config.trust import WORKSPACE_TRUST_FILE, trust_store_path

    assert WORKSPACE_TRUST_FILE == "trusted_workspaces.yaml"
    assert trust_store_path() == _expected_store()


def test_unknown_workspace_reads_as_undecided(tmp_path):
    """None, not False — 'never asked' is a different answer from 'said no'."""
    from localharness.config.trust import is_trusted

    assert is_trusted(_workspace(tmp_path / "proj")) is None


def test_records_and_reads_back_trusted(tmp_path):
    from localharness.config.trust import is_trusted, record_trust

    ws = _workspace(tmp_path / "proj")
    record_trust(ws, True)
    assert is_trusted(ws) is True


def test_records_and_reads_back_declined(tmp_path):
    from localharness.config.trust import is_trusted, record_trust

    ws = _workspace(tmp_path / "proj")
    record_trust(ws, False)
    assert is_trusted(ws) is False


def test_trust_store_never_written_inside_the_workspace(tmp_path):
    """The judged workspace must not be able to vouch for itself."""
    from localharness.config.trust import record_trust

    ws = _workspace(tmp_path / "proj")
    record_trust(ws, True)

    assert _expected_store().exists()
    assert not (ws / "trusted_workspaces.yaml").exists()
    assert list(ws.iterdir()) == []
    assert list((tmp_path / "proj").iterdir()) == [ws]


def test_symlinked_workspace_is_one_entry(tmp_path):
    """Record through a symlink, read through the real path: same decision, one key."""
    from localharness.config.trust import is_trusted, record_trust, trust_store_path

    real = tmp_path / "real"
    ws = _workspace(real)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)

    record_trust(link / ".localharness", True)

    assert is_trusted(ws) is True
    stored = yaml.safe_load(trust_store_path().read_text(encoding="utf-8"))
    assert list(stored) == [str(ws.resolve())]


def test_worktree_dirs_are_separate_workspaces(tmp_path):
    """`git worktree add` produces a second real directory — trusting one says nothing
    about the other (v0.13 ruling: a worktree is its own workspace)."""
    from localharness.config.trust import is_trusted, record_trust

    main_ws = _workspace(tmp_path / "main")
    wt_ws = _workspace(tmp_path / "wt")

    record_trust(main_ws, True)

    assert is_trusted(main_ws) is True
    assert is_trusted(wt_ws) is None


def test_corrupt_store_reads_as_undecided(tmp_path):
    """A hand-edited store gone wrong degrades to 'ask again', never to a crashed session."""
    from localharness.config.trust import is_trusted, trust_store_path

    ws = _workspace(tmp_path / "proj")
    store = trust_store_path()

    store.write_text("{ this is: not, valid: yaml", encoding="utf-8")
    assert is_trusted(ws) is None

    store.write_text("just a string, not a mapping\n", encoding="utf-8")
    assert is_trusted(ws) is None

    store.write_text(f"{ws.resolve()}: yes-please\n", encoding="utf-8")
    assert is_trusted(ws) is None


def test_recording_a_second_workspace_preserves_the_first(tmp_path):
    from localharness.config.trust import is_trusted, record_trust

    first = _workspace(tmp_path / "one")
    second = _workspace(tmp_path / "two")

    record_trust(first, True)
    record_trust(second, False)

    assert is_trusted(first) is True
    assert is_trusted(second) is False


def test_a_decision_can_be_changed_by_a_later_record(tmp_path):
    """Hand-editing the store is the documented way to change an answer (no CLI verb in
    v0.13) — the writer must overwrite the entry, not append a second one."""
    from localharness.config.trust import is_trusted, record_trust, trust_store_path

    ws = _workspace(tmp_path / "proj")
    record_trust(ws, True)
    record_trust(ws, False)

    assert is_trusted(ws) is False
    stored = yaml.safe_load(trust_store_path().read_text(encoding="utf-8"))
    assert list(stored) == [str(ws.resolve())]
