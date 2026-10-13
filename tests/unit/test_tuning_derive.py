"""derive on the box headers (spec 20 §6.1, §6.2): the cache type and MTP drafting a llama.cpp launch derives."""
from __future__ import annotations

import pytest

from localharness.agent.gguf_header import read_header
from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider import tuning
from localharness.provider.autosense import FIT_KEYS, fit_header
from localharness.provider.tuning import NOTHING_DERIVED, Tuned, derive, mtp_layers
from tests.unit.gguf_support import write_gguf

SWIFT = {
    "general.architecture": "qwen35", "qwen35.block_count": 65, "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256, "qwen35.context_length": 262144,
    "qwen35.nextn_predict_layers": 1,
}
FLASH_NEXT = {
    "general.architecture": "qwen4exp", "qwen4exp.block_count": 48, "qwen4exp.expert_count": 512,
    "qwen4exp.attention.head_count_kv": 2, "qwen4exp.attention.key_length": 128,
    "qwen4exp.context_length": 131072,
}


def _spec(**over) -> ManagedServerConfig:
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": "/x/m.gguf", "port": 8080, **over})


def _header(tmp_path, kv) -> dict:
    return fit_header(read_header(write_gguf(tmp_path / "m.gguf", kv), FIT_KEYS))


def test_the_header_reader_yields_the_swift_mtp_and_block_counts(tmp_path):
    header = _header(tmp_path, SWIFT)
    assert header["nextn_predict_layers"] == 1 and header["block_count"] == 65


def test_the_flash_next_header_has_experts_and_no_nextn(tmp_path):
    header = _header(tmp_path, FLASH_NEXT)
    assert header["expert_count"] == 512 and "nextn_predict_layers" not in header


def test_swift_derives_q8_0_and_draft_mtp_each_with_one_reason(tmp_path):
    tuned = derive(_spec(), _header(tmp_path, SWIFT), AutosenseConfig())
    reasons = {"cache_type": "autosense.kv_cache_type", "spec_type": "nextn_predict_layers=1"}
    assert tuned == Tuned(cache_type="q8_0", spec_type="draft-mtp", draft_p_min=0.6, mtp_layers=1,
                          reasons=reasons)
    assert dict(tuned.reasons) == reasons


def test_flash_next_derives_q8_0_only(tmp_path):
    header = _header(tmp_path, FLASH_NEXT)
    tuned = derive(_spec(), header, AutosenseConfig())
    assert tuned == Tuned(cache_type="q8_0", reasons={"cache_type": "autosense.kv_cache_type"})
    assert tuned.spec_type is None and mtp_layers(header) is None


def test_nothing_derived_is_the_empty_record():
    assert NOTHING_DERIVED == Tuned() and dict(NOTHING_DERIVED.reasons) == {}
    assert NOTHING_DERIVED.cache_type is None and NOTHING_DERIVED.spec_type is None
    assert tuning.NOTHING_DERIVED is NOTHING_DERIVED
