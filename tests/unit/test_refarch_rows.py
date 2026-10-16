from __future__ import annotations

import re
from pathlib import Path

import pytest

from localharness.provider import refarch, server
from localharness.provider.refarch import DGX_SPARK, MAC_MINI, REF_ARCHS, RTX_4060

ROOT = Path(__file__).resolve().parents[2]


def test_the_8gb_row_leads_and_the_vllm_rows_follow():
    assert REF_ARCHS == (RTX_4060, DGX_SPARK, MAC_MINI)
    assert (RTX_4060.runtime, DGX_SPARK.runtime, MAC_MINI.runtime) == ("llamacpp", "vllm", "vllm")
    assert (RTX_4060.needs, DGX_SPARK.needs, MAC_MINI.needs) == (None, "nvidia", "apple-silicon")


def test_keys_are_unique():
    assert len({r.key for r in REF_ARCHS}) == len(REF_ARCHS)


@pytest.mark.parametrize("row", [r for r in REF_ARCHS if r.file], ids=lambda r: r.key)
def test_a_single_file_row_is_pinned_to_a_commit_and_a_hash(row):
    assert re.fullmatch("[0-9a-f]{40}", row.revision)
    assert re.fullmatch("[0-9a-f]{64}", row.sha256)
    assert row.file.endswith(".gguf") and row.bytes > 0


@pytest.mark.parametrize("row", [r for r in REF_ARCHS if r.runtime == "vllm"], ids=lambda r: r.key)
def test_a_vllm_row_is_the_whole_repo_unpinned(row):
    assert row.file is None and row.revision is None and row.sha256 is None
    assert row.bytes > 0 and "--max-model-len" in row.serve_extra_args
    assert bool(row.docker_image) == (row.launch == "docker")
    assert bool(row.pip_package) == (row.launch == "binary")


@pytest.mark.parametrize("row", REF_ARCHS, ids=lambda r: r.key)
def test_every_rows_page_exists(row):
    assert (ROOT / row.page).is_file()


@pytest.mark.parametrize("platform, machine, nvidia, expected", [
    ("darwin", "arm64", False, True),
    ("darwin", "aarch64", False, True),
    ("darwin", "x86_64", False, False),
    ("linux", "arm64", False, False),
])
def test_apple_silicon_needs_darwin_on_arm(platform, machine, nvidia, expected):
    assert refarch.needs_met(MAC_MINI, platform=platform, machine=machine, has_nvidia=nvidia) is expected


def test_nvidia_needs_an_answering_gpu_and_no_need_is_always_met():
    assert refarch.needs_met(DGX_SPARK, platform="linux", machine="x86_64", has_nvidia=True)
    assert not refarch.needs_met(DGX_SPARK, platform="linux", machine="x86_64", has_nvidia=False)
    assert refarch.needs_met(RTX_4060, platform="linux", machine="x86_64", has_nvidia=False)


def _pick(platform, machine, nvidia, path, memory):
    return refarch.default_row(
        REF_ARCHS, platform=platform, machine=machine, has_nvidia=nvidia,
        on_path=lambda runtime: runtime in path, memory=memory)


@pytest.mark.parametrize("args, expected", [
    (("linux", "x86_64", False, (), 16e9), RTX_4060),
    (("linux", "x86_64", True, ("vllm",), 120e9), DGX_SPARK),
    (("linux", "x86_64", True, (), 120e9), RTX_4060),
    (("darwin", "arm64", False, ("llamacpp",), 16e9), RTX_4060),
    (("darwin", "arm64", False, (), 16e9), RTX_4060),
    (("darwin", "arm64", False, ("vllm",), 16e9), MAC_MINI),
    (("linux", "x86_64", True, (), 4e9), RTX_4060),
    (("linux", "x86_64", True, ("vllm", "llamacpp"), 4e9), RTX_4060),
    (("linux", "x86_64", True, ("vllm", "llamacpp"), None), RTX_4060),
    (("win32", "AMD64", True, ("vllm",), 64e9), None),
])
def test_default_row_follows_the_one_ordered_rule(args, expected):
    assert _pick(*args) is expected


def test_a_cpu_only_box_is_never_offered_the_spark():
    assert _pick("linux", "x86_64", False, ("vllm",), 128e9) is not DGX_SPARK


def test_the_largest_row_that_fits_wins_among_those_on_path():
    assert _pick("linux", "x86_64", True, ("vllm", "llamacpp"), 40e9) is DGX_SPARK
    assert _pick("linux", "x86_64", True, ("vllm", "llamacpp"), 20e9) is RTX_4060


def _venv_bin(cfg: Path) -> Path:
    path = server.venv_vllm_bin(cfg)
    path.parent.mkdir(parents=True)
    path.write_text("")
    return path


def test_vllm_binary_prefers_the_harness_venv(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "which_outside_cwd", lambda name: "/usr/bin/vllm")
    venv_bin = _venv_bin(tmp_path)
    assert server.vllm_binary(tmp_path) == (str(venv_bin), "harness venv")


def test_vllm_binary_falls_back_to_path(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "which_outside_cwd", lambda name: "/usr/bin/vllm")
    assert server.vllm_binary(tmp_path) == ("/usr/bin/vllm", "on PATH")
    assert server.find_vllm(tmp_path) == "/usr/bin/vllm"


def test_vllm_binary_none_when_neither_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "which_outside_cwd", lambda name: None)
    assert server.vllm_binary(tmp_path) is None
    assert server.find_vllm(tmp_path) is None
