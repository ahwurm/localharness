from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from localharness.provider.lifecycle import FLOOR_FILE, cached_floor, raised_floor, record_floor

SRC = Path(__file__).resolve().parents[2] / "src" / "localharness"


def test_the_floor_round_trips_with_the_roles_and_a_utc_time(tmp_path):
    record_floor(tmp_path, 9_000, {"root": 9_000, "worker": 6_000}, now=datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))

    assert json.loads((tmp_path / FLOOR_FILE).read_text()) == {
        "root": 9_000, "roles": {"root": 9_000, "worker": 6_000}, "at": "2026-01-02T03:04:00+00:00"}
    assert cached_floor(tmp_path) == 9_000


@pytest.mark.parametrize("root, expected", [(1_000, 1_024), (1_024, 1_024), (50_000, 50_000)])
def test_a_cached_root_is_raised_to_the_floor_minimum(tmp_path, root, expected):
    (tmp_path / FLOOR_FILE).write_text(json.dumps({"root": root}))

    assert cached_floor(tmp_path) == expected


@pytest.mark.parametrize("root", [4096.0, True, "4096", [4096], None, 0, -5])
def test_a_root_that_is_not_a_plain_positive_int_reads_as_none(tmp_path, root):
    (tmp_path / FLOOR_FILE).write_text(json.dumps({"root": root}))

    assert cached_floor(tmp_path) is None


@pytest.mark.parametrize("body", [b"not json", b"[]", b'"x"', b"", b"{" + b" " * (64 * 1024) + b'"root": 9000}'])
def test_a_malformed_or_oversized_file_reads_as_none(tmp_path, body):
    (tmp_path / FLOOR_FILE).write_bytes(body)

    assert cached_floor(tmp_path) is None


def test_a_missing_file_and_a_fifo_read_as_none(tmp_path):
    assert cached_floor(tmp_path) is None
    os.mkfifo(tmp_path / FLOOR_FILE)

    assert cached_floor(tmp_path) is None


def test_a_symlink_at_the_name_is_replaced_not_followed(tmp_path):
    target = tmp_path / "elsewhere.txt"
    target.write_text("keep")
    (tmp_path / FLOOR_FILE).symlink_to(target)

    record_floor(tmp_path, 8_192, {})

    assert target.read_text() == "keep"
    assert not (tmp_path / FLOOR_FILE).is_symlink()
    assert cached_floor(tmp_path) == 8_192


def test_a_write_that_cannot_land_does_not_raise(tmp_path):
    record_floor(tmp_path / "missing" / "deeper", 8_192, {})


@pytest.mark.parametrize("live, cached, expected", [
    (None, None, None), (5_000, None, 5_000), (None, 7_000, 7_000), (5_000, 7_000, 7_000), (9_000, 7_000, 9_000)])
def test_the_cache_only_raises_the_live_floor(live, cached, expected):
    assert raised_floor(live, cached) == expected


def test_a_planted_small_cache_never_lowers_the_live_floor(tmp_path):
    (tmp_path / FLOOR_FILE).write_text(json.dumps({"root": 1}))

    assert raised_floor(6_000, cached_floor(tmp_path)) == 6_000



def test_only_the_lifecycle_module_and_the_gate_name_the_floor_file():
    named = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if FLOOR_FILE in p.read_text(encoding="utf-8")}
    assert named == {"provider/lifecycle.py", "agent/gate_types.py"}
