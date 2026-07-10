"""Reference-architecture profiles for guided setup (`localharness init`).

Distilled from docs/reference-architectures/ — those docs are ground truth;
keep this data in sync with them. Install routes verified June 2026:
- DGX Spark (GB10, SM 12.1): stable PyPI/docker vLLM lacks GB10 support — NVIDIA's
  documented route is the cu130 nightly container (dgx-spark-playbooks).
- Apple Silicon: vllm-metal (pip, native arm64 Python 3.12).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RefArch:
    key: str
    name: str
    status: str  # "tested" | "proposed" — mirrors the doc's status line
    doc: str  # repo-relative path to the ground-truth doc
    platform: str  # sys.platform this hardware runs ("linux" | "darwin")
    launch: str  # preferred route when no vllm binary exists: "docker" | "binary"
    docker_image: str | None
    pip_package: str | None  # installed into the harness venv when launch="binary"
    default_model: str  # published, downloadable HF checkpoint
    model_note: str  # honesty note shown before download
    serve_extra_args: tuple[str, ...]
    context_tokens: int  # served window (--max-model-len)


DGX_SPARK = RefArch(
    key="dgx-spark",
    name="NVIDIA DGX Spark — GB10, 128 GB unified",
    status="tested",
    doc="docs/reference-architectures/dgx-spark.md",
    platform="linux",
    launch="docker",
    docker_image="vllm/vllm-openai:nightly",
    pip_package=None,
    default_model="Qwen/Qwen3.6-27B-FP8",
    model_note=(
        "Published FP8 checkpoint (~28 GB download). The doc's tested NVFP4 variant has no "
        "published checkpoint — quantize locally with TensorRT ModelOpt and enter its path instead."
    ),
    serve_extra_args=(
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.90",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ),
    context_tokens=65536,
)

MAC_MINI = RefArch(
    key="mac-mini",
    name="Apple Mac mini — M4, 16 GB unified",
    status="proposed",
    doc="docs/reference-architectures/mac-mini.md",
    platform="darwin",
    launch="binary",
    docker_image=None,
    pip_package="vllm-metal",
    default_model="mlx-community/Qwen3.5-9B-MLX-4bit",
    model_note="MLX 4-bit (~5.7 GB download). vllm-metal needs native arm64 Python 3.12.",
    serve_extra_args=("--max-model-len", "65536"),
    context_tokens=65536,
)

REF_ARCHS: tuple[RefArch, ...] = (DGX_SPARK, MAC_MINI)
