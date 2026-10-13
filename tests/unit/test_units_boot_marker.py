"""The boot marker's plain-data types and the no-follow remover. Tmp directories and symlinks only:
nothing here touches a service manager."""
from __future__ import annotations

import dataclasses
import os

import pytest

from localharness.provider import units as u


def _marker(**over):
    kw = dict(record="flash-next", binary="/x/llama-server", invocation="abc", cause="fit",
              shape="full", planned_context=114688, full_fit=114688, free_bytes=24 * 2**30,
              dropped=("draft-mtp",), at="2026-10-13T00:00:00+00:00")
    kw.update(over)
    return u.Marker(**kw)


def test_the_marker_name_and_exports():
    assert u.BOOT_MARKER_NAME == "boot-attempt"
    assert {"Marker", "UnitRuntimeState", "remove_marker", "BOOT_MARKER_NAME"} <= set(u.__all__)


def test_marker_is_frozen_plain_data():
    m = _marker()
    assert m.record == "flash-next" and m.dropped == ("draft-mtp",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.record = "other"  # type: ignore[misc]
    assert _marker(invocation=None, planned_context=None, full_fit=None, free_bytes=None).invocation is None


def test_unit_runtime_state_is_frozen():
    s = u.UnitRuntimeState("abc", 1)
    assert (s.invocation_id, s.n_restarts) == ("abc", 1)
    assert u.UnitRuntimeState(None, 0).invocation_id is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.n_restarts = 2  # type: ignore[misc]


def test_remove_marker_unlinks_and_reports_then_reports_absence(tmp_path):
    (tmp_path / "vllm").mkdir()
    marker = tmp_path / "vllm" / "boot-attempt"
    marker.write_text("{}")
    assert u.remove_marker(tmp_path) is True
    assert not marker.exists()
    assert u.remove_marker(tmp_path) is False


def test_remove_marker_leaves_the_other_files_in_the_server_dir(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "boot-attempt").write_text("{}")
    (tmp_path / "vllm" / "keep").write_text("x")
    assert u.remove_marker(tmp_path) is True
    assert (tmp_path / "vllm" / "keep").exists()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_remove_marker_never_follows_a_symlinked_server_dir(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    target = elsewhere / "boot-attempt"
    target.write_text("{}")
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "vllm").symlink_to(elsewhere, target_is_directory=True)
    assert u.remove_marker(config) is False
    assert target.exists()


def test_remove_marker_unlinks_a_planted_link_not_its_target(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("keep me")
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "boot-attempt").symlink_to(secret)
    assert u.remove_marker(tmp_path) is True
    assert secret.read_text() == "keep me"
    assert not (tmp_path / "vllm" / "boot-attempt").is_symlink()


def test_remove_marker_with_no_server_dir_is_false(tmp_path):
    assert u.remove_marker(tmp_path) is False
    assert u.remove_marker(tmp_path / "no" / "such") is False


def test_remove_marker_with_a_file_in_place_of_the_server_dir_is_false(tmp_path):
    (tmp_path / "vllm").write_text("not a dir")
    assert u.remove_marker(tmp_path) is False


def test_remove_marker_accepts_a_str_config_dir(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "boot-attempt").write_text("{}")
    assert u.remove_marker(str(tmp_path)) is True
