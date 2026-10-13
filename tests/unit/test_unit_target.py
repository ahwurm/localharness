"""The `unit-target` store and the boot marker reader under `<global>/vllm/` (D-06, D-14, D-28)."""
from __future__ import annotations

import os  # noqa: F401
import stat

import pytest

from localharness.provider.units import (
    TARGET_NAME, read_target, remove_target, write_target,
)


def test_target_round_trip(tmp_path):
    write_target(tmp_path, "flash-next")
    f = tmp_path / "vllm" / TARGET_NAME
    assert f.read_text() == "flash-next\n"
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert read_target(tmp_path) == "flash-next"
    assert remove_target(tmp_path) is True
    assert read_target(tmp_path) is None
    assert remove_target(tmp_path) is False


def test_a_second_write_replaces_the_first(tmp_path):
    write_target(tmp_path, "a")
    write_target(tmp_path, "b.c@d")
    assert read_target(tmp_path) == "b.c@d"
    assert [p.name for p in (tmp_path / "vllm").iterdir()] == [TARGET_NAME]
