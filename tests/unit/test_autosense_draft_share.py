"""The fit prices the MTP draft context's extra KV (spec 20 §6.2, D-10)."""
from __future__ import annotations

import pytest

from localharness.config.models import AutosenseConfig, ManagedServerConfig
from localharness.provider.autosense import FreeMemory, compute_fit, fit
from localharness.provider.tuning import draft_share
from tests.unit.gguf_support import write_gguf
from tests.unit.test_tuning_derive import FLASH_NEXT, SWIFT

RATIOS = AutosenseConfig()
GIB = 2**30
SWIFT_HEADER = {"block_count": 65, "head_count_kv": 4, "key_length": 256, "context_length": 262144,
                "nextn_predict_layers": 1}
SHARE = 66 / 65
WEIGHTS = 20 * GIB
FREE = 32 * GIB
# 65 layers x 4 KV heads x 256 dims x (q8_0 K + q8_0 V) bytes per token
KV_Q8 = 65 * 4 * 256 * (34 / 32 + 34 / 32)


def _window(share: float | None = None) -> int | None:
    kw = {} if share is None else {"draft_share": share}
    return fit(SWIFT_HEADER, FREE, RATIOS, cache_type_k="q8_0", cache_type_v="q8_0", weights=WEIGHTS, **kw)


def test_the_draft_share_multiplies_the_kv_bytes_per_token():
    usable = FREE - WEIGHTS - int(RATIOS.compute_margin_ratio * WEIGHTS) - int(RATIOS.compute_reserve * GIB)
    assert _window(SHARE) == int(usable // (KV_Q8 * SHARE))
    assert _window(1.0) == int(usable // KV_Q8)
    assert _window(SHARE) < _window(1.0)


def test_a_fit_without_a_draft_share_is_unchanged():
    assert _window() == _window(1.0)


def test_the_draft_share_prices_an_attach_too():
    base = fit(SWIFT_HEADER, FREE, RATIOS, 10000, cache_type_k="q8_0", cache_type_v="q8_0")
    shared = fit(SWIFT_HEADER, FREE, RATIOS, 10000, cache_type_k="q8_0", cache_type_v="q8_0",
                 draft_share=SHARE)
    assert shared < base


def test_an_unknown_kv_stays_unknown_under_a_share():
    assert fit({}, FREE, RATIOS, weights=WEIGHTS, draft_share=SHARE) is None


def test_swift_draft_share_is_66_over_65():
    assert draft_share(SWIFT_HEADER) == 1 + 1 / 65


def test_flash_next_draft_share_is_one():
    assert draft_share({"block_count": 48, "expert_count": 512}) == 1.0


@pytest.mark.parametrize("header", [
    {"block_count": 1, "nextn_predict_layers": 2}, {"block_count": 65, "nextn_predict_layers": True},
    {"block_count": 65, "nextn_predict_layers": "1"}, {"block_count": 0, "nextn_predict_layers": 1},
    {"nextn_predict_layers": 1}, {"block_count": 65, "nextn_predict_layers": 0}, {},
])
def test_an_invalid_header_has_a_draft_share_of_one(header):
    assert draft_share(header) == 1.0


def _server(tmp_path, kv, **over) -> ManagedServerConfig:
    gguf = write_gguf(tmp_path / "m.gguf", kv)
    return ManagedServerConfig(runtime="llamacpp", binary="/x/llama-server", model=str(gguf), port=8080,
                               **over)


def _memory() -> FreeMemory:
    return FreeMemory(FREE, 36 * GIB, "nvidia-smi")


def test_compute_fit_prices_the_derived_cache_type_when_nothing_else_sets_one(tmp_path):
    srv = _server(tmp_path, SWIFT)
    plain = compute_fit(srv, RATIOS, memory=_memory())
    derived = compute_fit(srv, RATIOS, memory=_memory(), cache_type="q8_0", draft_share=SHARE)
    assert plain.fits is not None and derived.fits is not None
    # q8_0 is smaller than the f16 default; the draft context costs back a sixty-fifth
    assert derived.fits > plain.fits
    only_q8 = compute_fit(srv, RATIOS, memory=_memory(), cache_type="q8_0")
    assert derived.fits < only_q8.fits


def test_compute_fit_prefers_the_record_then_extra_args_over_the_derived_type(tmp_path):
    memory = _memory()
    recorded = compute_fit(_server(tmp_path, SWIFT, cache_type_k="f16", cache_type_v="f16"), RATIOS,
                           memory=memory, cache_type="q8_0")
    assert recorded.fits == compute_fit(_server(tmp_path, SWIFT), RATIOS, memory=memory).fits
    hand = compute_fit(_server(tmp_path, SWIFT, extra_args=["-ctk", "f16", "-ctv", "f16"]), RATIOS,
                       memory=memory, cache_type="q8_0")
    assert hand.fits == recorded.fits


def test_compute_fit_without_the_keywords_is_the_old_fit(tmp_path):
    srv = _server(tmp_path, FLASH_NEXT)
    assert compute_fit(srv, RATIOS, memory=_memory()).fits == compute_fit(
        srv, RATIOS, memory=_memory(), cache_type=None, draft_share=1.0).fits
