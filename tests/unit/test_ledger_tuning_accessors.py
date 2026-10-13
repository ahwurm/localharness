"""The speed ledger's bounded read and its tuning accessors: the retry record, the sampling record
and the reserved last-served key. The ledger sits where agent-run code can write, so a reader must
never hang, raise or be steered outside a closed set by what it finds there."""
from __future__ import annotations

import json
import os

import pytest

from localharness.provider import speed_stats as ss

KEY = "llamacpp:qwen"


def _retry(p, **over):
    kw = dict(full_fit=114688, free_bytes=24 * 2**30, dropped=["draft-mtp"],
              binary="/x/llama-server", kv_cache_type="q8_0", draft_p_min=0.6)
    kw.update(over)
    ss.record_retry(p, KEY, **kw)


# --- the bounded read ---------------------------------------------------------------------------

def test_cap_is_one_mebibyte_and_droppable_set_is_closed():
    assert ss.LEDGER_CAP == 1 << 20
    assert ss.RETRY_DROPPABLE == frozenset({"draft-mtp"})


def _ledger_of_size(p, size):
    """Valid JSON, exactly `size` bytes: one sampled entry plus ASCII padding."""
    head = json.dumps({"a:b": {"samples": [50.0], "pad": ""}})
    p.write_text(json.dumps({"a:b": {"samples": [50.0], "pad": "x" * (size - len(head))}}))
    assert p.stat().st_size == size


def test_a_ledger_over_the_cap_reads_as_empty(tmp_path):
    p = tmp_path / "speed_stats.json"
    _ledger_of_size(p, ss.LEDGER_CAP + 1)
    assert ss._load(p) == {}
    assert ss.median_tps(p, "a", "b") is None
    assert ss.last_retry(p, "a:b") is None


def test_a_ledger_at_the_cap_still_reads(tmp_path):
    p = tmp_path / "speed_stats.json"
    _ledger_of_size(p, ss.LEDGER_CAP)
    assert ss.median_tps(p, "a", "b") == 50.0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_at_the_ledger_path_reads_as_empty_without_blocking(tmp_path):
    p = tmp_path / "speed_stats.json"
    os.mkfifo(p)
    assert ss._load(p) == {}
    assert ss.last_retry(p, KEY) is None


def test_a_directory_at_the_ledger_path_reads_as_empty(tmp_path):
    p = tmp_path / "speed_stats.json"
    p.mkdir()
    assert ss._load(p) == {}


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_a_symlink_to_a_device_reads_as_empty(tmp_path):
    p = tmp_path / "speed_stats.json"
    p.symlink_to("/dev/zero")
    assert ss._load(p) == {}


def test_deeply_nested_json_reads_as_empty_not_a_recursion_error(tmp_path):
    p = tmp_path / "speed_stats.json"
    p.write_text("[" * 100_000 + "]" * 100_000)
    assert ss._load(p) == {}


def test_non_utf8_and_non_dict_content_reads_as_empty(tmp_path):
    p = tmp_path / "speed_stats.json"
    p.write_bytes(b"\xff\xfe\x00{")
    assert ss._load(p) == {}
    p.write_text("[1, 2]")
    assert ss._load(p) == {}


def test_a_missing_ledger_reads_as_empty(tmp_path):
    assert ss._load(tmp_path / "nope.json") == {}


# --- the retry record ---------------------------------------------------------------------------

def test_retry_round_trips_and_leaves_the_other_fields(tmp_path):
    p = tmp_path / "speed_stats.json"
    ss.record_tps(p, "llamacpp", "qwen", 55.0)
    ss.record_window_report(p, "llamacpp", "qwen", 65536, 262144, 60000)
    _retry(p)
    rec = ss.last_retry(p, KEY)
    assert rec == ss.RetryRecord(114688, 24 * 2**30, ("draft-mtp",), "/x/llama-server", "q8_0",
                                 0.6, rec.at)
    assert isinstance(rec.at, str) and "T" in rec.at
    assert ss.median_tps(p, "llamacpp", "qwen") == 55.0
    assert ss.last_window_report(p, "llamacpp", "qwen").served == 65536


def test_retry_with_no_free_bytes_and_nothing_dropped_round_trips(tmp_path):
    p = tmp_path / "s.json"
    _retry(p, free_bytes=None, dropped=[])
    rec = ss.last_retry(p, KEY)
    assert rec.free_bytes is None and rec.dropped == ()


def test_retry_refuses_a_dropped_outside_the_closed_set_and_writes_nothing(tmp_path):
    p = tmp_path / "s.json"
    with pytest.raises(ValueError):
        _retry(p, dropped=["ngram"])
    assert not p.exists()
    with pytest.raises(ValueError):
        _retry(p, dropped=["draft-mtp", "kv"])
    assert not p.exists()


@pytest.mark.parametrize("field,bad", [
    ("dropped", ["x"]), ("dropped", "draft-mtp"), ("dropped", [1]),
    ("full_fit", True), ("full_fit", -1), ("full_fit", 0), ("full_fit", "big"), ("full_fit", 1.5),
    ("free_bytes", "lots"), ("free_bytes", -5), ("free_bytes", True),
    ("draft_p_min", "0.6"), ("draft_p_min", True), ("draft_p_min", 1.5), ("draft_p_min", -0.1),
    ("binary", 7), ("kv_cache_type", None), ("at", 3),
])
def test_a_hand_edited_retry_reads_as_none(tmp_path, field, bad):
    p = tmp_path / "s.json"
    _retry(p)
    data = json.loads(p.read_text())
    data[KEY]["retry"][field] = bad
    p.write_text(json.dumps(data))
    assert ss.last_retry(p, KEY) is None


def test_a_retry_that_is_not_a_dict_reads_as_none(tmp_path):
    p = tmp_path / "s.json"
    for bad in ([1], "x", 3, None):
        p.write_text(json.dumps({KEY: {"retry": bad}}))
        assert ss.last_retry(p, KEY) is None
    p.write_text(json.dumps({KEY: "not an entry"}))
    assert ss.last_retry(p, KEY) is None


def test_clear_retry_removes_only_the_retry_field(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "qwen", 40.0)
    _retry(p)
    ss.clear_retry(p, KEY)
    assert ss.last_retry(p, KEY) is None
    entry = json.loads(p.read_text())[KEY]
    assert "retry" not in entry and entry["samples"] == [40.0]


def test_clear_retry_with_nothing_to_clear_writes_nothing(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "qwen", 40.0)
    os.utime(p, (1_000_000, 1_000_000))
    ss.clear_retry(p, KEY)
    ss.clear_retry(p, "no:such")
    assert p.stat().st_mtime == 1_000_000
    absent = tmp_path / "absent.json"
    ss.clear_retry(absent, KEY)
    assert not absent.exists()
