"""bench.yaml schema — BenchConfig, MatrixEntry, SamplingConfig, ThresholdSpec + loader.

Lives at repo root (sibling of pyproject.toml). Single source of truth for corpus path,
matrix list, sampling defaults, regression thresholds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_raw_as


# -------------------------------------------------------------------------
# MatrixEntry — one model in the BENCH-06 matrix
# -------------------------------------------------------------------------

class MatrixEntry(BaseModel):
    """One model in the BENCH-06 multi-model matrix.

    `name` is the friendly identifier (also the result-dir name).
    `provider` is the discriminator routed by build_llm_client_factory (e.g. 'ollama').
    `model_id` is the concrete model identifier passed to the provider client.
    `base_url` / `num_ctx` are optional per-entry overrides.
    """

    model_config = ConfigDict(frozen=True)
    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    base_url: Optional[str] = None
    num_ctx: Optional[int] = None


# -------------------------------------------------------------------------
# SamplingConfig — sequential adaptive sampling defaults (locked)
# -------------------------------------------------------------------------

class SamplingConfig(BaseModel):
    """Defaults from CONTEXT.md: ±10% tol, 3-20 runs, 95% confidence."""

    model_config = ConfigDict(frozen=True)
    tolerance: float = Field(gt=0, default=0.10)
    min_runs: int = Field(ge=3, default=3)
    max_runs: int = Field(ge=3, default=20)
    confidence: float = Field(gt=0, lt=1, default=0.95)


# -------------------------------------------------------------------------
# ThresholdSpec — one per-metric regression threshold
# -------------------------------------------------------------------------

class ThresholdSpec(BaseModel):
    """A single per-metric regression threshold.

    Types:
    - 'relative': head/baseline_median > 1 + value triggers regression (e.g. +0.15 = +15%)
    - 'absolute': head_median - baseline_median > value triggers regression (e.g. +1 new)
    - 'absolute_pp': head_rate - baseline_rate < value triggers regression (e.g. -0.05 = -5pp)
    """

    model_config = ConfigDict(frozen=True)
    type: Literal["relative", "absolute", "absolute_pp"]
    value: float


# -------------------------------------------------------------------------
# BenchConfig — top-level bench.yaml model
# -------------------------------------------------------------------------

class BenchConfig(BaseModel):
    """Top-level bench.yaml shape."""

    model_config = ConfigDict(frozen=True)
    corpus_path: Path
    results_path: Path
    matrix: list[MatrixEntry] = Field(default_factory=list)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    thresholds: dict[str, ThresholdSpec] = Field(default_factory=dict)
    # NOTE: per-scenario threshold overrides intentionally not modeled here.
    # Compare-time resolution (11-04) reads scenario YAML `thresholds:` directly.


# -------------------------------------------------------------------------
# Locked default thresholds (CONTEXT.md regression alarm policy)
# -------------------------------------------------------------------------

def default_thresholds() -> dict[str, ThresholdSpec]:
    """Return the 8-entry locked default threshold table from CONTEXT.md.

    Hardcoded so `bench compare` works even without a bench.yaml present.
    """
    return {
        "latency_total":    ThresholdSpec(type="relative", value=0.15),
        "latency_ttft":     ThresholdSpec(type="relative", value=0.10),
        "tokens_out":       ThresholdSpec(type="relative", value=0.10),
        "tokens_in":        ThresholdSpec(type="relative", value=0.10),
        "success_rate":     ThresholdSpec(type="absolute_pp", value=-0.05),
        "iterations":       ThresholdSpec(type="relative", value=0.20),
        "parse_failures":   ThresholdSpec(type="absolute", value=1),
        "stuck_recoveries": ThresholdSpec(type="absolute", value=1),
    }


# -------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------

def load_bench_config(path: Path) -> BenchConfig:
    """Load + validate bench.yaml. Raises pydantic.ValidationError on bad input."""
    path = Path(path)
    return parse_yaml_raw_as(BenchConfig, path.read_text(encoding="utf-8"))
