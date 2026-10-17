"""The first-run ranking: at an equal window the file that reads fewer GPU bytes ranks first."""
from __future__ import annotations

from pathlib import Path

from localharness.cli.library import _ranker
from localharness.config.models import AutosenseConfig
from localharness.provider import autosense
from localharness.provider.gguf_header import DiscoveredModel, rank_models
from tests.unit.gguf_support import load_fixture, write_recorded

SWIFT_GPU, NVIDIA_GPU = 18_562_465_856, 18_913_411_136


def dm(path: str, mtime: float = 1.0) -> DiscoveredModel:
    return DiscoveredModel(Path(path), path, Path(path).stem, "llama", 4096, None, mtime, None)


def rank(models, *, fits, gpu):
    return rank_models(models, unit_model=None, ledger_paths=(), fits=fits, gpu_bytes=gpu)


def test_equal_windows_rank_the_smaller_gpu_read_first_even_when_it_is_older():
    swift, nvidia = dm("/m/swift.gguf", 1), dm("/m/nvidia.gguf", 9)
    fits = {str(swift.path): 262_144, str(nvidia.path): 262_144}
    gpu = {str(swift.path): SWIFT_GPU, str(nvidia.path): NVIDIA_GPU}
    assert rank([nvidia, swift], fits=fits, gpu=gpu) == [swift, nvidia]


def test_the_larger_window_outranks_fewer_gpu_bytes():
    a, b = dm("/m/a.gguf"), dm("/m/b.gguf")
    fits = {str(a.path): 200_000, str(b.path): 262_144}
    gpu = {str(a.path): 1, str(b.path): 10**12}
    assert rank([a, b], fits=fits, gpu=gpu) == [b, a]


def test_nothing_ranks_to_nothing():
    assert rank([], fits={}, gpu={}) == []


def test_an_unknown_gpu_size_ranks_after_a_known_one_at_equal_windows():
    known, unknown = dm("/m/z-known.gguf", 1), dm("/m/a-unknown.gguf", 9)
    fits = {str(known.path): 4096, str(unknown.path): 4096}
    assert rank([unknown, known], fits=fits, gpu={str(known.path): 10**12, str(unknown.path): None}) == [known, unknown]


def test_an_unknown_window_ranks_after_a_known_one():
    known, unknown = dm("/m/z.gguf"), dm("/m/a.gguf")
    assert rank([unknown, known], fits={str(known.path): 4096, str(unknown.path): None}, gpu={}) == [known, unknown]


def test_models_equal_on_every_key_are_ordered_by_path_whatever_the_input_order():
    a, b = dm("/m/a.gguf"), dm("/m/b.gguf")
    assert rank([a, b], fits={}, gpu={}) == rank([b, a], fits={}, gpu={}) == [a, b]


def test_the_ranker_over_the_two_recorded_27b_files_puts_swift_first_with_an_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "40960, 40960\n")
    models = []
    for name, mtime in (("swift-27b-nvfp4-q8mix", 1.0), ("nvidia-27b-nvfp4", 9.0)):
        directory = tmp_path / name
        directory.mkdir()
        path = write_recorded(directory, load_fixture(name))[0]
        models.append(DiscoveredModel(path, str(path), name, "qwen35", 5120, None, mtime, None))
    nvidia, swift = models[1], models[0]
    ranked = _ranker(tmp_path / "config.yaml", AutosenseConfig())([nvidia, swift])
    assert [m.name for m in ranked] == ["swift-27b-nvfp4-q8mix", "nvidia-27b-nvfp4"]
