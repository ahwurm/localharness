"""The `unit-target` store and the boot marker reader under `<global>/vllm/` (D-06, D-14, D-28)."""
from __future__ import annotations

import json
import os
import stat

import pytest

from localharness.provider import units
from localharness.provider.units import (
    BOOT_MARKER_NAME, MARKER_CAP, TARGET_CAP, TARGET_NAME, Marker,
    read_marker, read_target, remove_target, write_target,
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


def test_nothing_there_reads_none(tmp_path):
    assert read_target(tmp_path) is None
    assert read_marker(tmp_path) is None
    assert remove_target(tmp_path) is False


@pytest.mark.parametrize("name", ["../x", "", "a b", "-x", "a\nb", "x" * 129])
def test_write_target_refuses_a_name_off_the_charset(tmp_path, name):
    with pytest.raises(ValueError):
        write_target(tmp_path, name)
    assert not (tmp_path / "vllm").exists()


def test_a_symlinked_server_dir_is_never_followed(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / TARGET_NAME).write_text("victim\n")
    (tmp_path / "vllm").symlink_to(real)
    with pytest.raises(OSError):
        write_target(tmp_path, "x")
    assert (real / TARGET_NAME).read_text() == "victim\n"
    assert sorted(p.name for p in real.iterdir()) == [TARGET_NAME]
    assert read_target(tmp_path) is None
    assert remove_target(tmp_path) is False
    assert (real / TARGET_NAME).exists()


def test_a_link_planted_at_unit_target_is_replaced_not_followed(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched\n")
    (d / TARGET_NAME).symlink_to(elsewhere)
    assert read_target(tmp_path) is None
    write_target(tmp_path, "mine")
    assert not (d / TARGET_NAME).is_symlink()
    assert read_target(tmp_path) == "mine"
    assert elsewhere.read_text() == "untouched\n"


def test_remove_target_unlinks_a_planted_link_itself(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.write_text("untouched\n")
    (d / TARGET_NAME).symlink_to(elsewhere)
    assert remove_target(tmp_path) is True
    assert not os.path.lexists(d / TARGET_NAME)
    assert elsewhere.exists()


def test_a_fifo_at_unit_target_reads_none_without_blocking(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    os.mkfifo(d / TARGET_NAME)
    assert read_target(tmp_path) is None


@pytest.mark.parametrize("body", [
    b"x" * (TARGET_CAP + 1), b"bad name\n", b"ab\x00cd\n", b"\xff\xfe\n", b"", b"a\nb\n",
])
def test_a_body_off_the_shape_reads_none(tmp_path, body):
    d = tmp_path / "vllm"
    d.mkdir()
    (d / TARGET_NAME).write_bytes(body)
    assert read_target(tmp_path) is None


def test_a_failed_write_keeps_the_previous_target(tmp_path, monkeypatch):
    write_target(tmp_path, "old")

    def boom(*a, **kw):
        raise OSError("simulated")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError):
        write_target(tmp_path, "new")
    monkeypatch.undo()
    assert read_target(tmp_path) == "old"
    assert [p.name for p in (tmp_path / "vllm").iterdir()] == [TARGET_NAME]


# ------------------------------------------------------------------------------ the boot marker

GOOD = {
    "record": "flash-next", "binary": "/opt/llama/llama-server", "invocation": "inv-1",
    "cause": "death", "shape": "retry", "planned_context": 131072, "full_fit": 262144,
    "free_bytes": 12345, "dropped": ["draft-mtp", "parsers"], "at": "2026-10-13T10:00:00Z",
}


def _put(tmp_path, payload) -> None:
    d = tmp_path / "vllm"
    d.mkdir(exist_ok=True)
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    (d / BOOT_MARKER_NAME).write_bytes(body)


def test_marker_round_trip(tmp_path):
    _put(tmp_path, GOOD)
    assert read_marker(tmp_path) == Marker(
        record="flash-next", binary="/opt/llama/llama-server", invocation="inv-1", cause="death",
        shape="retry", planned_context=131072, full_fit=262144, free_bytes=12345,
        dropped=("draft-mtp", "parsers"), at="2026-10-13T10:00:00Z")


def test_marker_nullable_fields_may_be_none(tmp_path):
    _put(tmp_path, {**GOOD, "invocation": None, "planned_context": None, "full_fit": None,
                    "free_bytes": None, "dropped": [], "cause": "none", "shape": "full"})
    m = read_marker(tmp_path)
    assert m is not None and m.dropped == () and m.invocation is None


def test_marker_symlinked_directory_reads_none(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / BOOT_MARKER_NAME).write_text(json.dumps(GOOD))
    (tmp_path / "vllm").symlink_to(real)
    assert read_marker(tmp_path) is None


def test_marker_planted_link_and_fifo_read_none(tmp_path):
    d = tmp_path / "vllm"
    d.mkdir()
    (tmp_path / "elsewhere").write_text(json.dumps(GOOD))
    (d / BOOT_MARKER_NAME).symlink_to(tmp_path / "elsewhere")
    assert read_marker(tmp_path) is None
    (d / BOOT_MARKER_NAME).unlink()
    os.mkfifo(d / BOOT_MARKER_NAME)
    assert read_marker(tmp_path) is None


def test_marker_over_the_cap_reads_none(tmp_path):
    pad = " " * (MARKER_CAP + 1 - len(json.dumps(GOOD)))
    _put(tmp_path, json.dumps(GOOD).encode() + pad.encode())
    assert read_marker(tmp_path) is None


def _without(key):
    return {k: v for k, v in GOOD.items() if k != key}


@pytest.mark.parametrize("payload", [
    b"{not json", b"\xff\xfe", json.dumps([GOOD]).encode(), _without("binary"), _without("dropped"),
    {**GOOD, "extra": 1}, {**GOOD, "cause": "boot"}, {**GOOD, "shape": "half"},
    {**GOOD, "dropped": ["x"]}, {**GOOD, "dropped": "draft-mtp"}, {**GOOD, "planned_context": "131072"},
    {**GOOD, "full_fit": True}, {**GOOD, "free_bytes": 1.5}, {**GOOD, "record": ""},
    {**GOOD, "at": 5}, {**GOOD, "invocation": 7},
], ids=lambda p: "bytes" if isinstance(p, bytes) else "-".join(sorted(p)) + str(len(p)))
def test_marker_off_its_closed_shape_reads_none(tmp_path, payload):
    _put(tmp_path, payload)
    assert read_marker(tmp_path) is None


def test_nothing_in_units_writes_a_marker():
    import inspect
    src = inspect.getsource(units)
    assert "def write_marker" not in src
