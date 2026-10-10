"""D-60: outside test_units every test sees an empty /proc and a systemd that answers "not checked"."""
from __future__ import annotations

import os

from localharness.provider import procfs, units


def test_proc_root_is_an_existing_empty_directory():
    assert procfs.PROC_ROOT != "/proc"
    assert os.path.isdir(procfs.PROC_ROOT) and os.listdir(procfs.PROC_ROOT) == []
    assert procfs.listeners(8080) == []


def test_the_systemd_seam_answers_not_checked_without_running_anything(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a test spawned a process")

    monkeypatch.setattr(units.subprocess, "run", boom)
    assert units._run(["systemctl", "--user", "show"], timeout=1, cap=1) is None
    assert units.unit_file_state("llm.service") is None
    assert isinstance(units.read_journal("llm.service"), units.Skip)


def test_xdg_config_home_is_unset(monkeypatch):
    assert "XDG_CONFIG_HOME" not in os.environ
