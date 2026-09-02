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


# --- #136: minimum substance for a CLIENT-MEASURED sample ---------------------------------


def test_substantive_sample_admits_a_real_generation():
    """The calibration anchor from the live evidence: 60 tokens over ~1.5s (39.5 tok/s on the
    measured vLLM config) is a decode measurement and must reach the ledger."""
    from localharness.provider.speed_stats import is_substantive_sample

    assert is_substantive_sample(60, 1.5)


@pytest.mark.parametrize("tokens,window,why", [
    (60, 0.012, "a burst flushed in one delta: 59 intervals over 12ms reads as 4,900 tok/s"),
    (200, 0.04, "same artifact at a bigger token count — the window is still transport, not decode"),
    (5, 0.0001, "#130's degenerate window, now refused before the ceiling ever sees it"),
    (11, 2.0, "too few intervals: one stalled delta would set the whole rate"),
    (60, 0.24, "just under the window floor"),
    (15, 3.0, "just under the token floor"),
])
def test_substantive_sample_rejects_insubstantial(tokens, window, why):
    """#136: the ledger's leak was samples the 10k ceiling admits — 1,715..6,788 tok/s recorded
    on a ~78 tok/s config, dragging a displayed median to 152.4 tok/s."""
    from localharness.provider.speed_stats import is_substantive_sample

    assert not is_substantive_sample(tokens, window), why


def test_substantive_sample_without_evidence_is_not_substance():
    """No token count or no window = nothing was measured; absence never admits."""
    from localharness.provider.speed_stats import is_substantive_sample

    assert not is_substantive_sample(None, 1.5)
    assert not is_substantive_sample(60, None)
    assert not is_substantive_sample(60, -1.0)


# ---------------------------------------------------------------------------
# Live tok/s under speculative decoding (MTP): a streamed delta carries every
# accepted draft token, so counting chunks as tokens understates the rate by the
# acceptance length. Measured live: qwen3.8-27b + qwen3_5_mtp ran 23.6 tok/s while
# the chunk-counting readout showed 6.9 (214 tokens delivered in 63 deltas = 3.40).
# ---------------------------------------------------------------------------

def _client():
    from localharness.provider.client import LLMClient, LLMConfig
    return LLMClient(LLMConfig(base_url="http://x/v1", model="m", api_key="none"))


def test_tokens_per_chunk_starts_unknown_not_one():
    """Unknown must not masquerade as 1.0 — that guess reads 3x low on a spec-decode runtime."""
    assert _client()._tokens_per_chunk is None


def test_live_rate_is_suppressed_while_the_ratio_is_unknown():
    """No number beats a wrong number: with no evidence, show nothing rather than ~1/3 of truth."""
    import time as _t
    c = _client()
    c._tokens_per_chunk = None
    c._stream_progress = {"first_at": _t.monotonic() - 9.0, "chunks": 63, "server_tps": None}
    assert c.gen_speed_snapshot() is None


def test_ledger_roundtrip_seeds_a_new_client(tmp_path):
    """A NEW session's FIRST turn is honest only if the ratio survives in the ledger."""
    from localharness.provider.speed_stats import record_tokens_per_chunk, tokens_per_chunk
    p = tmp_path / "speed_stats.json"
    assert tokens_per_chunk(p, "vllm", "qwen3.8-27b") is None
    record_tokens_per_chunk(p, "vllm", "qwen3.8-27b", 3.40)
    assert tokens_per_chunk(p, "vllm", "qwen3.8-27b") == 3.40


def test_ratio_persistence_does_not_clobber_existing_tps_samples(tmp_path):
    from localharness.provider.speed_stats import (
        record_tokens_per_chunk, record_tps, median_tps,
    )
    p = tmp_path / "speed_stats.json"
    record_tps(p, "vllm", "m", 20.0)
    record_tps(p, "vllm", "m", 24.0)
    record_tokens_per_chunk(p, "vllm", "m", 3.0)
    assert median_tps(p, "vllm", "m") == 22.0


def test_sub_one_ratio_is_refused_by_the_ledger(tmp_path):
    from localharness.provider.speed_stats import record_tokens_per_chunk, tokens_per_chunk
    p = tmp_path / "speed_stats.json"
    record_tokens_per_chunk(p, "vllm", "m", 0.4)
    assert tokens_per_chunk(p, "vllm", "m") is None


def test_live_rate_is_scaled_by_the_learned_tokens_per_chunk():
    import time as _t
    c = _client()
    c._tokens_per_chunk = 3.40
    start = _t.monotonic() - 9.0
    c._stream_progress = {"first_at": start, "chunks": 63, "server_tps": None}
    live, verified = c.gen_speed_snapshot()
    assert verified is False, "an in-flight estimate must not claim to be verified"
    # 63 chunks * 3.40 = 214.2 tokens over ~9s -> ~23.7 tok/s, not ~6.9
    assert 22.0 < live < 25.0, live


def test_live_rate_without_scaling_would_understate(monkeypatch):
    """Guard the regression itself: ratio 1.0 on spec-decoded output reads ~1/3.4 of truth."""
    import time as _t
    c = _client()
    c._tokens_per_chunk = 1.0
    c._stream_progress = {"first_at": _t.monotonic() - 9.0, "chunks": 63, "server_tps": None}
    live, _ = c.gen_speed_snapshot()
    assert 6.0 < live < 8.0, live


def test_ratio_is_learned_from_exact_usage_at_stream_end():
    c = _client()
    progress = {"first_at": __import__("time").monotonic() - 9.0, "chunks": 63, "server_tps": None}

    class _Usage:
        completion_tokens = 214

    c._note_gen_speed(progress, _Usage())
    assert round(c._tokens_per_chunk, 2) == 3.40


def test_ratio_never_drops_below_one():
    """A chunk cannot carry less than a token; a sub-1 ratio would slow the readout, not fix it."""
    c = _client()
    progress = {"first_at": __import__("time").monotonic() - 9.0, "chunks": 100, "server_tps": None}

    class _Usage:
        completion_tokens = 40

    c._note_gen_speed(progress, _Usage())
    assert c._tokens_per_chunk == 1.0


def test_recording_tps_does_not_clobber_the_tokens_per_chunk_ratio(tmp_path):
    """The direction my first test missed: a tps write must preserve the ratio on the same
    entry. Both are written in one _note_gen_speed call, ratio first — a replacing write ate it."""
    from localharness.provider.speed_stats import (
        record_tokens_per_chunk, record_tps, tokens_per_chunk,
    )
    p = tmp_path / "speed_stats.json"
    record_tokens_per_chunk(p, "vllm", "m", 3.40)
    record_tps(p, "vllm", "m", 23.6)
    assert tokens_per_chunk(p, "vllm", "m") == 3.40

