"""The boot marker's plain-data types and the no-follow remover. Tmp directories and symlinks only:
nothing here touches a service manager."""
from __future__ import annotations

import dataclasses
import json
import os
import stat

import pytest

from localharness.provider import units as u


def _marker(**over):
    kw = dict(record="flash-next", binary="/x/llama-server", invocation="abc", cause="death",
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


def test_unit_runtime_state_gains_defaulted_fields_and_keeps_its_two_positional_ones():
    s = u.UnitRuntimeState("abc", 1)
    assert (s.main_pid, s.active_state, s.unit_file_state, s.fragment_path, s.drop_in_paths) == (
        None, "", "", "", "")
    assert (s.exec_main_status, s.result, s.own) == (None, "", False)
    assert s.timeout_start_seconds == u.TIMEOUT_START_SECONDS == 1800
    full = u.UnitRuntimeState("inv", 2, main_pid=4242, active_state="active", unit_file_state="enabled",
                              fragment_path="/f", drop_in_paths="/d", exec_main_status=78, result="exit-code",
                              timeout_start_seconds=60, own=True)
    assert (full.main_pid, full.own, full.exec_main_status) == (4242, True, 78)
    assert [f.name for f in dataclasses.fields(u.UnitRuntimeState)][:2] == ["invocation_id", "n_restarts"]
    assert not hasattr(u, "UnitStateRuntime")


def test_unit_state_of_init_is_untouched():
    assert [f.name for f in dataclasses.fields(u.UnitState)] == [
        "unit", "load_state", "fragment_path", "unit_file_state", "drop_in_paths"]


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


# ------------------------------------------------------------------------------- the writer


KEYS = {"record", "binary", "invocation", "cause", "shape", "planned_context", "full_fit", "free_bytes",
        "dropped", "at"}


def test_write_marker_then_read_marker_is_equal(tmp_path):
    m = _marker(dropped=("draft-mtp",))
    u.write_marker(tmp_path, m)
    assert u.read_marker(tmp_path) == m
    both = _marker(dropped=("draft-mtp", "parsers"), shape="retry", cause="remembered")
    u.write_marker(tmp_path, both)
    assert u.read_marker(tmp_path) == both


def test_write_marker_round_trips_a_marker_with_every_optional_none(tmp_path):
    m = _marker(invocation=None, planned_context=None, full_fit=None, free_bytes=None, dropped=(), cause="none")
    u.write_marker(tmp_path, m)
    assert u.read_marker(tmp_path) == m


def test_the_marker_on_disk_holds_exactly_the_ten_keys_and_is_private(tmp_path):
    u.write_marker(tmp_path, _marker(dropped=("parsers",)))
    f = tmp_path / "vllm" / u.BOOT_MARKER_NAME
    data = json.loads(f.read_text(encoding="utf-8"))
    assert set(data) == KEYS and len(KEYS) == len(dataclasses.fields(u.Marker)) == 10
    assert data["dropped"] == ["parsers"]
    assert stat.S_IMODE(f.stat().st_mode) == 0o600
    assert [p.name for p in f.parent.iterdir()] == [u.BOOT_MARKER_NAME]


@pytest.mark.parametrize("over", [
    {"cause": "fit"}, {"cause": ""}, {"shape": "half"}, {"dropped": ("mtp",)}, {"record": ""},
    {"binary": ""}, {"at": ""}, {"planned_context": "1"}, {"full_fit": True}, {"free_bytes": 1.5},
    {"invocation": 7}, {"dropped": ["draft-mtp"]},
])
def test_write_marker_refuses_what_the_reader_would_not_read_back_and_writes_nothing(tmp_path, over):
    with pytest.raises(ValueError):
        u.write_marker(tmp_path, _marker(**over))
    assert not (tmp_path / "vllm").exists()


def test_write_marker_refuses_a_non_marker(tmp_path):
    with pytest.raises(ValueError):
        u.write_marker(tmp_path, {"record": "x"})  # type: ignore[arg-type]
    assert not (tmp_path / "vllm").exists()


def test_write_marker_refuses_json_over_the_cap(tmp_path):
    with pytest.raises(ValueError):
        u.write_marker(tmp_path, _marker(binary="/" + "b" * u.MARKER_CAP))
    assert not (tmp_path / "vllm").exists()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="needs O_NOFOLLOW")
def test_write_marker_refuses_a_symlinked_server_dir(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "vllm").symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        u.write_marker(tmp_path, _marker())
    assert list(real.iterdir()) == []


def test_write_marker_replaces_a_planted_link_and_never_follows_it(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("keep me")
    (d / u.BOOT_MARKER_NAME).symlink_to(secret)
    u.write_marker(tmp_path, _marker())
    assert secret.read_text() == "keep me"
    assert not (d / u.BOOT_MARKER_NAME).is_symlink()
    assert u.read_marker(tmp_path) == _marker()


def test_write_marker_replaces_an_unparsable_marker_that_the_reader_left_in_place(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    (d / u.BOOT_MARKER_NAME).write_text("{not json")
    assert u.read_marker(tmp_path) is None
    assert (d / u.BOOT_MARKER_NAME).exists()  # reading never removes
    u.write_marker(tmp_path, _marker())
    assert u.read_marker(tmp_path) == _marker()


def test_write_marker_accepts_a_str_config_dir(tmp_path):
    u.write_marker(str(tmp_path), _marker())
    assert u.read_marker(str(tmp_path)) == _marker()
