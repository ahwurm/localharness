"""The overlay lock: exclusive, bounded, fail-closed, private and never nested."""
from __future__ import annotations

import fcntl
import os
import stat

import pytest

from localharness.config import overlay
from localharness.config.overlay import OVERLAY_LOCK_FILE, OverlayBusy, overlay_lock


def test_the_lock_creates_a_private_sidecar_and_releases_it_on_exit(tmp_path):
    with overlay_lock(tmp_path):
        assert stat.S_IMODE((tmp_path / OVERLAY_LOCK_FILE).stat().st_mode) == 0o600
    with overlay_lock(tmp_path):
        pass


def test_a_lock_held_elsewhere_is_busy_after_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(overlay, "OVERLAY_LOCK_BUDGET_SECONDS", 0.1)
    fd = os.open(tmp_path / OVERLAY_LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        with pytest.raises(OverlayBusy):
            with overlay_lock(tmp_path):
                pytest.fail("entered a held lock")
    finally:
        os.close(fd)
    with overlay_lock(tmp_path):
        pass


def test_a_nested_acquisition_on_one_thread_raises_and_the_outer_lock_survives(tmp_path):
    with overlay_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with overlay_lock(tmp_path):
                pass
    with overlay_lock(tmp_path):
        pass


def test_a_symlinked_sidecar_is_refused_as_busy(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("")
    (tmp_path / OVERLAY_LOCK_FILE).symlink_to(target)
    with pytest.raises(OverlayBusy):
        with overlay_lock(tmp_path):
            pytest.fail("followed a symlinked sidecar")
    with overlay_lock(tmp_path / "fresh"):
        pass
