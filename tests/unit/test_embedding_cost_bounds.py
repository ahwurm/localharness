"""The embedding model runs on CPU, inside the harness process, when nobody is watching — so its cost
is bounded where it loads (2026-10-04: a phone session idled into a dreaming pass that held every core
for hours). The cut and the thread cap are set on the model once; a model that refuses either still
loads, because a cap is a warning, never a failed memory."""
from __future__ import annotations

import logging
import types

import pytest

from localharness.memory import resonance


def test_the_loaded_model_is_cut_in_tokens_and_capped_in_threads():
    torch = pytest.importorskip("torch")
    model = types.SimpleNamespace(max_seq_length=32_768)
    assert resonance._bound(model) is model
    assert model.max_seq_length == resonance.EMBED_MAX_TOKENS == 1024
    assert torch.get_num_threads() == resonance.EMBED_THREADS == 4


def test_a_model_that_refuses_the_cut_still_loads(caplog: pytest.LogCaptureFixture):
    class Stiff:
        @property
        def max_seq_length(self) -> int:
            return 32_768

    model = Stiff()
    with caplog.at_level(logging.WARNING, logger="localharness.memory.resonance"):
        assert resonance._bound(model) is model
    assert "could not cap the embedding model's input at 1024 tokens" in caplog.text
