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


# --- the sampling record ------------------------------------------------------------------------

def test_sampling_round_trips(tmp_path):
    p = tmp_path / "s.json"
    ss.record_sampling(p, "llamacpp", "qwen", temperature=0.6, rung="org.default_temperature",
                       server_default=1.0)
    rec = ss.last_sampling(p, "llamacpp", "qwen")
    assert rec == ss.SamplingRecord(0.6, "org.default_temperature", 1.0, rec.recorded_at)
    assert isinstance(rec.recorded_at, str)


def test_sampling_none_values_round_trip_as_none(tmp_path):
    p = tmp_path / "s.json"
    ss.record_sampling(p, "llamacpp", "qwen", temperature=None, rung=None, server_default=None)
    rec = ss.last_sampling(p, "llamacpp", "qwen")
    assert (rec.temperature, rec.rung, rec.server_default) == (None, None, None)


def test_sampling_keeps_the_other_fields(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "qwen", 55.0)
    ss.record_sampling(p, "llamacpp", "qwen", temperature=0.7, rung="r", server_default=None)
    ss.record_tps(p, "llamacpp", "qwen", 57.0)
    assert ss.last_sampling(p, "llamacpp", "qwen").temperature == 0.7
    assert ss.median_tps(p, "llamacpp", "qwen") == 56.0


@pytest.mark.parametrize("field,bad", [
    ("temperature", True), ("temperature", "0.6"), ("server_default", "1.0"),
    ("server_default", False), ("rung", 3), ("recorded_at", None), ("recorded_at", 5),
])
def test_a_hand_edited_sampling_record_reads_as_none(tmp_path, field, bad):
    p = tmp_path / "s.json"
    ss.record_sampling(p, "llamacpp", "qwen", temperature=0.6, rung="r", server_default=1.0)
    data = json.loads(p.read_text())
    data["llamacpp:qwen"]["sampling"][field] = bad
    p.write_text(json.dumps(data))
    assert ss.last_sampling(p, "llamacpp", "qwen") is None


def test_sampling_absent_or_not_a_dict_reads_as_none(tmp_path):
    p = tmp_path / "s.json"
    assert ss.last_sampling(p, "a", "b") is None
    p.write_text(json.dumps({"a:b": {"sampling": [1]}}))
    assert ss.last_sampling(p, "a", "b") is None


# --- the last-served record ---------------------------------------------------------------------

def test_last_served_round_trips(tmp_path):
    p = tmp_path / "s.json"
    assert ss.last_served(p) is None
    ss.record_last_served(p, "flash-next")
    assert ss.last_served(p) == "flash-next"
    assert ss.LAST_SERVED_KEY == "_last_served"
    ss.record_last_served(p, "other")
    assert ss.last_served(p) == "other"


@pytest.mark.parametrize("bad", [7, "", None, ["x"], {"x": 1}])
def test_a_non_string_last_served_name_reads_as_none(tmp_path, bad):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"_last_served": {"name": bad}}))
    assert ss.last_served(p) is None
    p.write_text(json.dumps({"_last_served": "flash-next"}))
    assert ss.last_served(p) is None


def test_the_reserved_key_never_appears_among_model_entries(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "qwen", 50.0)
    ss.record_model_path(p, "llamacpp", "qwen", "/m/q.gguf")
    ss.record_last_served(p, "flash-next")
    ss.record_dropped_env(p, "llamacpp", ["FOO"])
    assert set(ss.all_median_tps(p)) == {"llamacpp:qwen"}
    assert ss.model_paths(p) == {"/m/q.gguf"}
    # even an entry the reserved key was hand-edited to carry samples or a path stays out
    data = json.loads(p.read_text())
    data["_last_served"].update(samples=[99.0], model_path="/evil")
    p.write_text(json.dumps(data))
    assert set(ss.all_median_tps(p)) == {"llamacpp:qwen"}
    assert ss.model_paths(p) == {"/m/q.gguf"}
    assert ss.last_served(p) == "flash-next"


def test_last_served_keeps_the_model_entries(tmp_path):
    p = tmp_path / "s.json"
    ss.record_tps(p, "llamacpp", "qwen", 50.0)
    ss.record_last_served(p, "flash-next")
    assert ss.median_tps(p, "llamacpp", "qwen") == 50.0


# --- an unreadable ledger is never written over (WR-03) ----------------------------------------

_WRITERS = {
    "record_tps": lambda p: ss.record_tps(p, "llamacpp", "qwen", 40.0),
    "record_tokens_per_chunk": lambda p: ss.record_tokens_per_chunk(p, "llamacpp", "qwen", 2.0),
    "record_window_report": lambda p: ss.record_window_report(p, "llamacpp", "qwen", 1, 2, 3),
    "record_rejected_effort_keys": lambda p: ss.record_rejected_effort_keys(p, "llamacpp", "qwen", ["k"]),
    "record_model_path": lambda p: ss.record_model_path(p, "llamacpp", "qwen", "/m.gguf"),
    "record_dropped_env": lambda p: ss.record_dropped_env(p, "llamacpp", ["X"]),
    "record_retry": lambda p: _retry(p),
    "clear_retry": lambda p: ss.clear_retry(p, KEY),
    "record_sampling": lambda p: ss.record_sampling(p, "llamacpp", "qwen", temperature=None, rung=None,
                                                    server_default=1.0),
    "record_last_served": lambda p: ss.record_last_served(p, "qwen"),
}


def _over_cap_with_a_retry(p):
    """A ledger one byte over the cap whose content holds a retry record the writers must not lose."""
    body = {KEY: {"retry": {"full_fit": 1}, "pad": ""}}
    head = json.dumps(body)
    body[KEY]["pad"] = "x" * (ss.LEDGER_CAP + 1 - len(head))
    p.write_text(json.dumps(body))
    assert p.stat().st_size == ss.LEDGER_CAP + 1


@pytest.mark.parametrize("writer", sorted(_WRITERS))
@pytest.mark.parametrize("kind", ["over-cap", "malformed", "non-dict"])
def test_no_writer_replaces_an_unreadable_ledger_and_each_warns_once(tmp_path, writer, kind):
    from structlog.testing import capture_logs

    p = tmp_path / "speed_stats.json"
    if kind == "over-cap":
        _over_cap_with_a_retry(p)
    else:
        p.write_text("{not json" if kind == "malformed" else "[1, 2]")
    before = p.read_bytes()
    with capture_logs() as logs:
        _WRITERS[writer](p)
    assert p.read_bytes() == before
    warnings = [e for e in logs if e.get("event") == "speed_ledger_unreadable"]
    assert len(warnings) == 1 and warnings[0]["path"] == str(p) and warnings[0]["log_level"] == "warning"
    assert ss._load(p) == {}  # readers still read it as empty


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_no_writer_replaces_a_fifo(tmp_path):
    p = tmp_path / "speed_stats.json"
    os.mkfifo(p)
    for write in _WRITERS.values():
        write(p)
    import stat
    assert stat.S_ISFIFO(os.lstat(p).st_mode)


def test_a_missing_ledger_is_written_without_a_warning(tmp_path):
    from structlog.testing import capture_logs

    p = tmp_path / "speed_stats.json"
    with capture_logs() as logs:
        _retry(p)
    assert ss.last_retry(p, KEY) is not None and logs == []
