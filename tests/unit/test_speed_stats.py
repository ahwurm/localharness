"""speed_stats: verified decode-rate ledger — banding, math, persistence."""

import pytest

from localharness.provider.speed_stats import (
    SAMPLE_CAP,
    all_median_tps,
    decode_tps,
    median_tps,
    record_tps,
    tps_band,
)


def test_tps_band_boundaries():
    # Owner spec: >30 green, 20–30 yellow, <20 red — both boundary values are yellow.
    assert tps_band(30.01) == "green"
    assert tps_band(30.0) == "yellow"
    assert tps_band(20.0) == "yellow"
    assert tps_band(19.99) == "red"
    assert tps_band(0.0) == "red"


def test_decode_tps_intervals_after_first_token():
    # 31 tokens, first at t=1.0, done at t=4.0 → 30 intervals over 3s = 10 tok/s.
    assert decode_tps(31, 1.0, 4.0) == 10.0


def test_decode_tps_unmeasurable_returns_none():
    assert decode_tps(1, 0.0, 5.0) is None  # single token: no interval
    assert decode_tps(0, 0.0, 5.0) is None
    assert decode_tps(50, 5.0, 5.0) is None  # degenerate window


def test_record_and_median_round_trip(tmp_path):
    path = tmp_path / "speed_stats.json"
    assert median_tps(path, "llamacpp", "m") is None  # missing file → no number
    for tps in (10.0, 30.0, 20.0):
        record_tps(path, "llamacpp", "m", tps)
    assert median_tps(path, "llamacpp", "m") == 20.0
    assert median_tps(path, "vllm", "m") is None  # keyed per provider_type


def test_record_caps_rolling_window(tmp_path):
    path = tmp_path / "speed_stats.json"
    for i in range(SAMPLE_CAP + 5):
        record_tps(path, "vllm", "q", float(i))
    # Oldest dropped: kept samples are 5.0..14.0, median 9.5.
    assert median_tps(path, "vllm", "q") == 9.5


def test_corrupt_file_reads_empty_and_recovers(tmp_path):
    path = tmp_path / "speed_stats.json"
    path.write_text("{not json")
    assert median_tps(path, "vllm", "q") is None
    record_tps(path, "vllm", "q", 25.0)
    assert median_tps(path, "vllm", "q") == 25.0


def test_all_median_tps(tmp_path):
    path = tmp_path / "speed_stats.json"
    record_tps(path, "vllm", "a", 40.0)
    record_tps(path, "ollama", "b", 15.0)
    assert all_median_tps(path) == {"vllm:a": 40.0, "ollama:b": 15.0}


def test_concurrent_writers_do_not_share_one_temp_file(tmp_path, monkeypatch):
    """Two sessions recording at once: os.replace is atomic, but open+truncate+write into a
    temp name SHARED by every process is not. With a fixed `speed_stats.json.tmp`, the other
    writer's replace consumes the file this one is still writing — its own replace then raises
    FileNotFoundError (sample silently dropped, "ledger update failed" logged) or a half-written
    payload gets published and the WHOLE ledger reads as corrupt afterwards. Per-writer names
    give the documented semantics: last writer wins, file always valid JSON."""
    from pathlib import Path

    path = tmp_path / "speed_stats.json"
    record_tps(path, "vllm", "a", 40.0)
    real_write_text = Path.write_text
    state = {"raced": False}

    def racing_write_text(self, data, *args, **kwargs):
        result = real_write_text(self, data, *args, **kwargs)
        if not state["raced"] and self.name.endswith(".tmp"):
            state["raced"] = True  # the OTHER session publishes start→finish mid-write
            record_tps(path, "ollama", "b", 9.0)
        return result

    monkeypatch.setattr(Path, "write_text", racing_write_text)
    record_tps(path, "vllm", "a", 41.0)  # must not raise

    assert state["raced"]
    assert all_median_tps(path)["vllm:a"] == 40.5  # this writer's publish won, ledger readable
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], leftovers  # per-writer name, cleaned up — never a shared fixed one


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, -5.0, 40_000.0])
def test_record_tps_rejects_non_finite_and_out_of_range(tmp_path, bad):
    """#130: the ledger defends its own file. nan poisons this model's median forever and an
    implausible rate is a timing artifact — neither may reach speed_stats.json, whatever the
    caller passes."""
    from localharness.provider.speed_stats import median_tps, record_tps

    p = tmp_path / "speed_stats.json"
    record_tps(p, "vllm", "m", bad)
    assert not p.exists()
    assert median_tps(p, "vllm", "m") is None


def test_record_tps_rejection_never_disturbs_existing_samples(tmp_path):
    """A dropped sample is a no-op, not a rewrite: earlier verified samples survive intact."""
    from localharness.provider.speed_stats import median_tps, record_tps

    p = tmp_path / "speed_stats.json"
    record_tps(p, "vllm", "m", 20.0)
    record_tps(p, "vllm", "m", float("nan"))
    assert median_tps(p, "vllm", "m") == 20.0
