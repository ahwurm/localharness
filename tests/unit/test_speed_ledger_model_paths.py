"""D-08: `speed_stats.model_paths` — every recorded weights path, for ranking (INIT-02)."""
from __future__ import annotations

import json

from localharness.provider import speed_stats as ss


def test_model_paths_returns_every_recorded_path(tmp_path):
    p = tmp_path / "speed_stats.json"
    ss.record_model_path(p, "llamacpp", "qwen", "/m/q.gguf")
    ss.record_model_path(p, "llamacpp", "other", "/m/o.gguf")
    assert ss.model_paths(p) == {"/m/q.gguf", "/m/o.gguf"}


def test_model_paths_skips_the_reserved_dropped_env_entry(tmp_path):
    p = tmp_path / "speed_stats.json"
    ss.record_model_path(p, "llamacpp", "qwen", "/m/q.gguf")
    ss.record_dropped_env(p, "llamacpp", ["FOO"])
    data = json.loads(p.read_text())
    data[ss.DROPPED_ENV_KEY]["model_path"] = "/must/not/appear"  # even a hostile reserved entry
    p.write_text(json.dumps(data))
    assert ss.model_paths(p) == {"/m/q.gguf"}


def test_model_paths_keeps_only_string_values_and_dict_entries(tmp_path):
    p = tmp_path / "speed_stats.json"
    p.write_text(json.dumps({"a:b": {"model_path": 7}, "c:d": {"model_path": ["x"]}, "e:f": "junk",
                             "g:h": {"samples": [1.0]}, "i:j": {"model_path": "/m/ok.gguf"}}))
    assert ss.model_paths(p) == {"/m/ok.gguf"}


def test_a_missing_or_corrupt_ledger_is_the_empty_set(tmp_path):
    assert ss.model_paths(tmp_path / "nope.json") == set()
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ss.model_paths(bad) == set()
    bad.write_text("[1, 2]")
    assert ss.model_paths(bad) == set()
