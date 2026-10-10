"""The ledger's additive fields (AUTO-08): window_report, model_path and the per-runtime
`_dropped_env` map, all behind one write recipe that updates an entry and never replaces it."""
from __future__ import annotations

import json
import threading

from localharness.provider import speed_stats as ss


def test_window_report_round_trips_and_leaves_the_other_fields(tmp_path):
    p = tmp_path / "speed_stats.json"
    ss.record_tps(p, "llamacpp", "qwen", 55.0)
    ss.record_tokens_per_chunk(p, "llamacpp", "qwen", 2.5)
    ss.record_window_report(p, "llamacpp", "qwen", 65536, 262144, 60000)
    ss.record_model_path(p, "llamacpp", "qwen", "/m/q.gguf")
    entry = json.loads(p.read_text())["llamacpp:qwen"]
    assert entry["samples"] == [55.0]
    assert entry["tokens_per_chunk"] == 2.5
    assert entry["model_path"] == "/m/q.gguf"
    assert set(entry["window_report"]) == {"served", "trained", "fits", "recorded_at"}
    rec = ss.last_window_report(p, "llamacpp", "qwen")
    assert rec == ss.WindowRecord(65536, 262144, 60000, rec.recorded_at)
    assert isinstance(rec.recorded_at, str)


def test_a_later_sample_does_not_disturb_the_new_fields(tmp_path):
    p = tmp_path / "speed_stats.json"
    ss.record_window_report(p, "vllm", "m", 32768, 32768, None)
    ss.record_model_path(p, "vllm", "m", "/w/m")
    ss.record_tps(p, "vllm", "m", 70.0)
    ss.record_tokens_per_chunk(p, "vllm", "m", 1.5)
    assert ss.last_window_report(p, "vllm", "m").served == 32768
    assert ss.model_path(p, "vllm", "m") == "/w/m"
    assert ss.median_tps(p, "vllm", "m") == 70.0


def test_none_fields_round_trip_as_none(tmp_path):
    p = tmp_path / "s.json"
    ss.record_window_report(p, "ollama", "o", None, None, None)
    rec = ss.last_window_report(p, "ollama", "o")
    assert (rec.served, rec.trained, rec.fits) == (None, None, None)


def test_last_window_report_is_none_when_absent_or_hand_edited(tmp_path):
    p = tmp_path / "s.json"
    assert ss.last_window_report(p, "a", "b") is None
    for bad in ({"served": "big", "trained": 1, "fits": 2, "recorded_at": "t"},
                {"served": 1.5, "trained": 1, "fits": 2, "recorded_at": "t"},
                {"served": True, "trained": 1, "fits": 2, "recorded_at": "t"},
                {"served": 1, "trained": 1, "fits": 2, "recorded_at": 7},
                [1, 2, 3], "text"):
        p.write_text(json.dumps({"a:b": {"window_report": bad}}))
        assert ss.last_window_report(p, "a", "b") is None
    p.write_text(json.dumps({"a:b": "not a dict"}))
    assert ss.last_window_report(p, "a", "b") is None


def test_model_path_round_trip_and_non_string_reads_none(tmp_path):
    p = tmp_path / "s.json"
    ss.record_model_path(p, "llamacpp", "qwen", "/m/q.gguf")
    assert ss.model_path(p, "llamacpp", "qwen") == "/m/q.gguf"
    assert ss.model_path(p, "llamacpp", "other") is None
    p.write_text(json.dumps({"llamacpp:qwen": {"model_path": 5}}))
    assert ss.model_path(p, "llamacpp", "qwen") is None


def test_dropped_env_is_sorted_and_keyed_by_runtime(tmp_path):
    p = tmp_path / "s.json"
    ss.record_dropped_env(p, "llamacpp", ["LLAMA_ARG_HOST", "LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_HOST"])
    ss.record_dropped_env(p, "vllm", ["VLLM_PORT"])
    assert ss.DROPPED_ENV_KEY == "_dropped_env"
    assert json.loads(p.read_text())["_dropped_env"]["llamacpp"] == ["LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_HOST"]
    assert ss.last_dropped_env(p, "llamacpp") == ["LLAMA_ARG_CTX_SIZE", "LLAMA_ARG_HOST"]
    assert ss.last_dropped_env(p, "vllm") == ["VLLM_PORT"]
    assert ss.last_dropped_env(p, "ollama") is None
    ss.record_dropped_env(p, "llamacpp", [])
    assert ss.last_dropped_env(p, "llamacpp") == []


def test_dropped_env_survives_a_malformed_slot(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"_dropped_env": ["not", "a", "dict"]}))
    assert ss.last_dropped_env(p, "llamacpp") is None
    ss.record_dropped_env(p, "llamacpp", ["X"])
    assert ss.last_dropped_env(p, "llamacpp") == ["X"]
    p.write_text(json.dumps({"_dropped_env": {"llamacpp": [1, 2]}}))
    assert ss.last_dropped_env(p, "llamacpp") is None


def test_the_reserved_key_is_inert_for_the_speed_readers(tmp_path):
    p = tmp_path / "s.json"
    ss.record_dropped_env(p, "llamacpp", ["A"])
    ss.record_tps(p, "llamacpp", "qwen", 40.0)
    assert ss.all_median_tps(p) == {"llamacpp:qwen": 40.0}
    assert ss.median_tps(p, "_dropped_env", "llamacpp") is None


def test_a_sample_recorded_after_dropped_env_keeps_it(tmp_path):
    p = tmp_path / "s.json"
    ss.record_dropped_env(p, "llamacpp", ["A"])
    ss.record_window_report(p, "llamacpp", "qwen", 1000, 2000, None)
    assert ss.last_dropped_env(p, "llamacpp") == ["A"]


def test_idempotency_the_same_triple_twice_leaves_one_entry(tmp_path):
    p = tmp_path / "s.json"
    ss.record_window_report(p, "llamacpp", "qwen", 65536, 262144, 60000)
    first = ss.last_window_report(p, "llamacpp", "qwen")
    ss.record_window_report(p, "llamacpp", "qwen", 65536, 262144, 60000)
    second = ss.last_window_report(p, "llamacpp", "qwen")
    assert list(json.loads(p.read_text())) == ["llamacpp:qwen"]
    assert (second.served, second.trained, second.fits) == (first.served, first.trained, first.fits)
    assert second.recorded_at >= first.recorded_at


def test_concurrent_window_writers_leave_a_parseable_ledger(tmp_path):
    p = tmp_path / "s.json"
    errors: list[BaseException] = []

    def _writer(i: int) -> None:
        try:
            for j in range(10):
                ss.record_window_report(p, "llamacpp", f"m{i}", 1000 + j, 2000, None)
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert isinstance(json.loads(p.read_text()), dict)
    assert list(tmp_path.glob("*.tmp")) == []


def test_150_names_and_an_awkward_model_id_round_trip(tmp_path):
    p = tmp_path / "s.json"
    names = [f"LLAMA_ARG_{i:03d}" for i in range(150)]
    ss.record_dropped_env(p, "llamacpp", names)
    assert ss.last_dropped_env(p, "llamacpp") == sorted(names)
    model = "org/model:tag"
    ss.record_window_report(p, "ollama", model, 4096, 32768, 4000)
    ss.record_model_path(p, "ollama", model, "/blobs/sha256:abc")
    assert ss.last_window_report(p, "ollama", model).trained == 32768
    assert ss.model_path(p, "ollama", model) == "/blobs/sha256:abc"
    assert "ollama:org/model:tag" in json.loads(p.read_text())
