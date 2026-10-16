"""Reference-architecture rows for the download offer and guided setup (`localharness init`).

Distilled from docs/reference-architectures/ — those docs are ground truth;
keep this data in sync with them. Install routes verified June 2026:
- DGX Spark (GB10, SM 12.1): stable PyPI/docker vLLM lacks GB10 support — NVIDIA's
  documented route is the cu130 nightly container (dgx-spark-playbooks).
- Apple Silicon: vllm-metal (pip, native arm64 Python 3.12).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RefArch:
    key: str
    name: str
    status: str  # "tested" | "proposed" — mirrors the doc's status line
    page: str  # repo-relative path to the ground-truth doc
    runtime: str  # "llamacpp" | "vllm"
    platforms: tuple[str, ...]  # sys.platform prefixes this hardware runs
    needs: str | None  # "nvidia" | "apple-silicon" | None
    repo: str  # published, downloadable HF repo
    file: str | None  # the one file a llama.cpp row fetches; None = whole repo
    bytes: int  # the file's size, or the repo's total
    revision: str | None  # 40-hex commit the file is fetched at; None for a whole-repo row
    sha256: str | None  # the hub's LFS hash of `file` at `revision`
    launch: str  # preferred route when no vllm binary exists: "docker" | "binary"
    docker_image: str | None = None
    pip_package: str | None = None  # installed into the harness venv when launch="binary"
    model_note: str = ""  # honesty note shown before download
    serve_extra_args: tuple[str, ...] = ()
    context_tokens: int | None = None  # served window (--max-model-len)


RTX_4060 = RefArch(
    key="rtx-4060",
    name="8 GB consumer GPU — RTX 4060 class, 16 GB RAM",
    status="proposed",
    page="docs/reference-architectures/rtx-4060.md",
    runtime="llamacpp",
    platforms=("linux", "darwin"),
    needs=None,
    repo="empero-ai/Qwen3.8-9B-Distill-GGUF",
    file="Qwen3.8-9B-Q4_K_M.gguf",
    bytes=5780090176,
    revision="760121cd70bb4c36b2b5ec58eb765e0df5987efe",
    sha256="df13d66021cef676f82be74053220fd75af6bf2a6a7fb77f5222ab9e50744a7a",
    launch="binary",
    model_note="GGUF Q4_K_M (5.8 GB); fully offloaded on an 8 GB card",
)

DGX_SPARK = RefArch(
    key="dgx-spark",
    name="NVIDIA DGX Spark — GB10, 128 GB unified",
    status="tested",
    page="docs/reference-architectures/dgx-spark.md",
    runtime="vllm",
    platforms=("linux",),
    needs="nvidia",
    repo="Qwen/Qwen3.6-27B-FP8",
    file=None,
    bytes=30890040900,
    revision=None,
    sha256=None,
    launch="docker",
    docker_image="vllm/vllm-openai:nightly",
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
    page="docs/reference-architectures/mac-mini.md",
    runtime="vllm",
    platforms=("darwin",),
    needs="apple-silicon",
    repo="mlx-community/Qwen3.5-9B-MLX-4bit",
    file=None,
    bytes=5977074591,
    revision=None,
    sha256=None,
    launch="binary",
    pip_package="vllm-metal",
    model_note="MLX 4-bit (~5.7 GB download). vllm-metal needs native arm64 Python 3.12.",
    serve_extra_args=("--max-model-len", "65536"),
    context_tokens=65536,
)

REF_ARCHS: tuple[RefArch, ...] = (RTX_4060, DGX_SPARK, MAC_MINI)


def needs_met(row: RefArch, *, platform: str, machine: str, has_nvidia: bool) -> bool:
    if row.needs == "nvidia":
        return has_nvidia
    if row.needs == "apple-silicon":
        return platform.startswith("darwin") and machine in ("arm64", "aarch64")
    return True


def default_row(
    rows: tuple[RefArch, ...], *, platform: str, machine: str, has_nvidia: bool,
    on_path: Callable[[str], bool], memory: float | None,
) -> RefArch | None:
    fits = [r for r in rows if any(platform.startswith(p) for p in r.platforms)
            and needs_met(r, platform=platform, machine=machine, has_nvidia=has_nvidia)]
    pool = [r for r in fits if on_path(r.runtime)] or [r for r in fits if r.runtime == "llamacpp"]
    if not pool:
        return None
    within = [r for r in pool if memory is not None and r.bytes <= memory]
    return max(within, key=lambda r: r.bytes) if within else min(pool, key=lambda r: r.bytes)
