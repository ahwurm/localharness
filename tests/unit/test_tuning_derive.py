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


# ---- precedence: record > extra_args / env in any spelling > derivation (D-06) --------------------

SWIFT_HEADER = {"block_count": 65, "nextn_predict_layers": 1, "head_count_kv": 4, "key_length": 256}
KNOBS = AutosenseConfig()
Q8 = "autosense.kv_cache_type"

KV_SUPPRESSED = [
    pytest.param({"cache_type_k": "q4_0"}, id="record-cache_type_k-alone"),
    pytest.param({"cache_type_v": "q4_0"}, id="record-cache_type_v-alone"),
    pytest.param({"extra_args": ["-ctk", "q4_0"]}, id="spelling--ctk"),
    pytest.param({"extra_args": ["-ctv", "q4_0"]}, id="spelling--ctv"),
    pytest.param({"extra_args": ["--cache-type-k", "q4_0"]}, id="spelling---cache-type-k"),
    pytest.param({"extra_args": ["--cache-type-v=q4_0"]}, id="spelling---cache-type-v="),
    pytest.param({"extra_args": ["--cache-type-k=q4_0"]}, id="spelling---cache-type-k="),
    pytest.param({"env": {"LLAMA_ARG_CACHE_TYPE_K": "q4_0"}}, id="spelling-env-cache-type-k"),
    pytest.param({"env": {"LLAMA_ARG_CACHE_TYPE_V": "q4_0"}}, id="spelling-env-cache-type-v"),
    pytest.param({"extra_args": ["-fa", "off"]}, id="spelling--fa-off"),
    pytest.param({"extra_args": ["-fa=off"]}, id="spelling--fa=off"),
    pytest.param({"extra_args": ["--flash-attn", "off"]}, id="spelling---flash-attn-off"),
    pytest.param({"extra_args": ["--flash-attn=off"]}, id="spelling---flash-attn=off"),
    pytest.param({"extra_args": ["-fa", "OFF"]}, id="spelling--fa-OFF"),
    *[pytest.param({"env": {"LLAMA_ARG_FLASH_ATTN": v}}, id=f"spelling-env-flash-attn-{v}")
      for v in ("off", "0", "false", "disabled", "OFF", "False", "Disabled")],
]


@pytest.mark.parametrize("over", KV_SUPPRESSED)
def test_precedence_a_pinned_cache_type_or_fa_off_suppresses_only_the_kv_derivation(over):
    tuned = derive(_spec(**over), SWIFT_HEADER, KNOBS)
    assert tuned.cache_type is None and "cache_type" not in tuned.reasons
    assert tuned.spec_type == "draft-mtp" and tuned.draft_p_min == 0.6


@pytest.mark.parametrize("flags", [["-fa", "on"], ["-fa", "auto"], ["--flash-attn=on"], ["-fa"]],
                         ids=lambda f: " ".join(f))
def test_precedence_fa_on_or_auto_does_not_suppress(flags):
    assert derive(_spec(extra_args=flags), SWIFT_HEADER, KNOBS).cache_type == "q8_0"


@pytest.mark.parametrize("value", ["on", "1", "true", "auto"])
def test_precedence_env_fa_on_does_not_suppress(value):
    assert derive(_spec(env={"LLAMA_ARG_FLASH_ATTN": value}), SWIFT_HEADER, KNOBS).cache_type == "q8_0"


def test_precedence_the_f16_knob_derives_no_cache_type():
    tuned = derive(_spec(), SWIFT_HEADER, AutosenseConfig(kv_cache_type="f16"))
    assert tuned.cache_type is None and tuned.spec_type == "draft-mtp"


def test_precedence_another_knob_value_is_the_derived_type():
    assert derive(_spec(), SWIFT_HEADER, AutosenseConfig(kv_cache_type="q4_0")).cache_type == "q4_0"


SPEC_SUPPRESSED = [
    pytest.param({"speculative": "off"}, id="record-speculative-off"),
    pytest.param({"speculative": "draft-mtp"}, id="record-speculative-draft-mtp"),
    pytest.param({"extra_args": ["--spec-type", "ngram"]}, id="spelling---spec-type"),
    pytest.param({"extra_args": ["--spec-type=draft-mtp"]}, id="spelling---spec-type="),
    pytest.param({"extra_args": ["-md", "/m/d.gguf"]}, id="spelling--md"),
    pytest.param({"extra_args": ["--model-draft=/m/d.gguf"]}, id="spelling---model-draft="),
    pytest.param({"extra_args": ["--model-draft", "/m/d.gguf"]}, id="spelling---model-draft"),
    pytest.param({"extra_args": ["--spec-draft-n-max", "3"]}, id="spelling---spec-draft-n-max"),
    pytest.param({"extra_args": ["--spec-draft-p-min=0.5"]}, id="spelling---spec-draft-p-min="),
    pytest.param({"extra_args": ["-hfd", "org/repo"]}, id="spelling--hfd"),
    pytest.param({"extra_args": ["--hf-repo-draft", "org/repo"]}, id="spelling---hf-repo-draft"),
    pytest.param({"extra_args": ["-hfrd", "org/repo"]}, id="spelling--hfrd"),
    pytest.param({"env": {"LLAMA_ARG_SPEC_TYPE": "ngram"}}, id="spelling-env-spec-type"),
    pytest.param({"env": {"LLAMA_ARG_SPEC_DRAFT_N_MAX": "3"}}, id="spelling-env-spec-draft-n-max"),
    pytest.param({"env": {"LLAMA_ARG_MODEL_DRAFT": "/m/d.gguf"}}, id="spelling-env-model-draft"),
]


@pytest.mark.parametrize("over", SPEC_SUPPRESSED)
def test_precedence_a_set_speculative_suppresses_only_the_drafting_derivation(over):
    tuned = derive(_spec(**over), SWIFT_HEADER, KNOBS)
    assert tuned.spec_type is None and tuned.draft_p_min is None and tuned.mtp_layers is None
    assert "spec_type" not in tuned.reasons
    assert tuned.cache_type == "q8_0"


def test_precedence_draft_mtp_false_suppresses_drafting_only():
    tuned = derive(_spec(), SWIFT_HEADER, AutosenseConfig(draft_mtp=False))
    assert tuned.spec_type is None and tuned.cache_type == "q8_0"


def test_precedence_draft_p_min_follows_the_knob():
    assert derive(_spec(), SWIFT_HEADER, AutosenseConfig(draft_p_min=0.25)).draft_p_min == 0.25
    assert derive(_spec(), SWIFT_HEADER, AutosenseConfig(draft_p_min=0)).draft_p_min == 0.0


def test_precedence_an_unrelated_extra_arg_or_env_name_suppresses_nothing():
    spec = _spec(extra_args=["-ngl", "99", "--spec"], env={"LLAMA_ARG_N_GPU_LAYERS": "99"})
    tuned = derive(spec, SWIFT_HEADER, KNOBS)
    assert tuned.cache_type == "q8_0" and tuned.spec_type == "draft-mtp"


# ---- header validation (D-11): hostile ints derive no drafting and never raise --------------------

BAD_HEADERS = [
    pytest.param({"block_count": 1, "nextn_predict_layers": 2}, id="nextn-above-block_count"),
    pytest.param({"block_count": 65, "nextn_predict_layers": True}, id="nextn-bool"),
    pytest.param({"block_count": 65, "nextn_predict_layers": "1"}, id="nextn-string"),
    pytest.param({"block_count": 65, "nextn_predict_layers": -1}, id="nextn-negative"),
    pytest.param({"block_count": 65, "nextn_predict_layers": 0}, id="nextn-zero"),
    pytest.param({"block_count": 65, "nextn_predict_layers": 1.0}, id="nextn-float"),
    pytest.param({"block_count": 65, "nextn_predict_layers": [1]}, id="nextn-list"),
    pytest.param({"block_count": 65, "nextn_predict_layers": None}, id="nextn-none"),
    pytest.param({"block_count": 0, "nextn_predict_layers": 1}, id="block_count-zero"),
    pytest.param({"nextn_predict_layers": 1}, id="block_count-missing"),
    pytest.param({"block_count": True, "nextn_predict_layers": 1}, id="block_count-bool"),
    pytest.param({"block_count": "65", "nextn_predict_layers": 1}, id="block_count-string"),
    pytest.param({}, id="empty"),
]


@pytest.mark.parametrize("header", BAD_HEADERS)
def test_an_invalid_header_derives_no_drafting_and_never_raises(header):
    assert mtp_layers(header) is None
    tuned = derive(_spec(), header, KNOBS)
    assert tuned.spec_type is None and tuned.mtp_layers is None
    assert tuned.cache_type == "q8_0"


def test_nextn_equal_to_block_count_is_valid():
    assert mtp_layers({"block_count": 4, "nextn_predict_layers": 4}) == 4


def test_a_non_mapping_header_derives_no_drafting():
    assert mtp_layers(None) is None  # type: ignore[arg-type]


# ---- other runtimes, and the 0.18 launch -----------------------------------------------------------

def test_a_non_llamacpp_spec_derives_nothing_on_the_llama_cpp_side():
    vllm = ManagedServerConfig(runtime="vllm", launch="binary", binary="/opt/vllm", model="org/repo")
    assert derive(vllm, SWIFT_HEADER, KNOBS) is NOTHING_DERIVED


def test_f16_and_draft_off_on_the_swift_header_derives_nothing():
    assert derive(_spec(), SWIFT_HEADER, AutosenseConfig(kv_cache_type="f16", draft_mtp=False)) \
        is NOTHING_DERIVED


def test_reasons_are_read_only():
    tuned = derive(_spec(), SWIFT_HEADER, KNOBS)
    with pytest.raises(TypeError):
        tuned.reasons["cache_type"] = "x"  # type: ignore[index]
