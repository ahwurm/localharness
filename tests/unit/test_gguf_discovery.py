"""INIT-01 / INIT-02 (discovery half): the bounded walk, typing with exact keys, split admission,
projector pairing, preset naming, ranking and the mapped-model typing (D-02..D-05, D-07, D-47, D-48, D-52).

Every file is a real GGUF header written by `write_gguf` under `tmp_path`; nothing reads the owner's
`~/models`. The u16 split keys are `Raw(2, ...)` as llama.cpp writes them.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from localharness.agent import gguf_header
from localharness.agent.gguf_header import (
    DISCOVERY_MAX_DEPTH,
    DISCOVERY_MAX_ENTRIES,
    DISCOVERY_MAX_FILES,
    DiscoveredModel,
    discover,
    preset_name,
    rank_models,
    unique_preset_name,
    walk_ggufs,
)
from tests.unit.gguf_support import Raw, write_gguf


def u16(n: int) -> Raw:
    return Raw(2, struct.pack("<H", n))


def model_file(path: Path, arch: str = "llama", embedding: int = 4096, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    kv = {"general.architecture": arch, "general.type": "model", f"{arch}.embedding_length": embedding,
          f"{arch}.context_length": 32768, **extra}
    return write_gguf(path, kv)


def projector_file(path: Path, dim: int = 4096) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_gguf(path, {"general.architecture": "clip", "general.type": "mmproj",
                             "clip.vision.projection_dim": dim})


def walk_and_discover(*roots: Path):
    return discover(walk_ggufs([str(r) for r in roots]))


# ---- the tracer: one model and its projector ---------------------------------------------------------

def test_a_model_and_its_projector_are_found_typed_and_paired(tmp_path):
    root = tmp_path / "models"
    model = model_file(root / "a" / "Model-Q4.gguf")
    proj = projector_file(root / "a" / "mmproj-Model-F16.gguf")
    found = walk_and_discover(root)
    assert [m.name for m in found.models] == ["Model-Q4"]
    assert found.models[0].path == model and found.models[0].mmproj == proj
    assert found.models[0].arch == "llama" and found.models[0].embedding == 4096
    assert found.models[0].context_length == 32768 and found.models[0].real == str(model.resolve())
    assert found.projectors == ()  # a paired projector is not left over, and is never a preset


def test_a_projector_with_no_dimension_match_stays_unpaired(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Model.gguf", embedding=4096)
    proj = projector_file(root / "mmproj-Other.gguf", dim=2560)
    found = walk_and_discover(root)
    assert found.models[0].mmproj is None and found.projectors == (proj,)


def test_a_header_without_a_type_is_a_model_and_a_clip_architecture_alone_is_a_projector(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    write_gguf(root / "typeless.gguf", {"general.architecture": "llama", "llama.embedding_length": 8})
    write_gguf(root / "old-mmproj.gguf", {"general.architecture": "clip", "clip.vision.projection_dim": 8})
    found = walk_and_discover(root)
    assert [m.name for m in found.models] == ["typeless"]
    assert [m.mmproj for m in found.models] == [root / "old-mmproj.gguf"]


@pytest.mark.parametrize("kind", ["adapter", "imatrix", "something-else"])
def test_other_general_types_are_skipped(tmp_path, kind):
    root = tmp_path / "models"
    root.mkdir()
    write_gguf(root / "x.gguf", {"general.architecture": "llama", "general.type": kind})
    found = walk_and_discover(root)
    assert found.models == () and found.projectors == ()


def test_unreadable_headers_are_skipped_without_blocking_the_rest(tmp_path):
    root = tmp_path / "models"
    model_file(root / "good.gguf")
    (root / "bad.gguf").write_bytes(b"not a gguf at all")
    (root / "empty.gguf").write_bytes(b"")
    found = walk_and_discover(root)
    assert [m.name for m in found.models] == ["good"]


def test_discover_reads_each_candidate_once_with_the_discovery_keys(tmp_path):
    from localharness.provider.autosense import FIT_KEYS

    root = tmp_path / "models"
    model_file(root / "a.gguf")
    model_file(root / "b.gguf")
    calls = []

    def fake(path, want):
        calls.append((os.fspath(path), set(want)))
        return gguf_header.read_header(path, want)

    discover(walk_ggufs([root]), read=fake)
    assert sorted(p for p, _ in calls) == [str(root / "a.gguf"), str(root / "b.gguf")]
    assert all(w == set(gguf_header.DISCOVERY_KEYS) | FIT_KEYS for _, w in calls)


def test_the_named_caps_are_the_decided_numbers():
    assert (DISCOVERY_MAX_DEPTH, DISCOVERY_MAX_FILES, DISCOVERY_MAX_ENTRIES) == (6, 256, 20000)


# ---- ranking (D-07) ----------------------------------------------------------------------------------

def dm(path: str, mtime: float = 1.0, real: str | None = None) -> DiscoveredModel:
    return DiscoveredModel(Path(path), real or path, Path(path).stem, "llama", 4096, None, mtime, None)


def test_a_ledger_model_path_ranks_first_when_nothing_else_differs():
    m1, m2 = dm("/m/a.gguf"), dm("/m/b.gguf")
    assert rank_models([m1, m2], unit_model=None, ledger_paths={str(m2.path)}, fits={}) == [m2, m1]


def test_ranking_order_is_unit_then_ledger_then_fit_then_newest_then_path():
    unit, ledger, fits, newest, older_a, older_b = (dm("/m/unit.gguf", 1), dm("/m/ledger.gguf", 1),
                                                    dm("/m/fits.gguf", 1), dm("/m/newest.gguf", 9),
                                                    dm("/m/a-old.gguf", 2), dm("/m/b-old.gguf", 2))
    pool = [older_b, newest, fits, older_a, ledger, unit]
    ranked = rank_models(pool, unit_model=str(unit.path), ledger_paths={str(ledger.path), str(unit.path)},
                         fits={str(fits.path): 4096, str(newest.path): 0, str(older_a.path): None})
    assert ranked == [unit, ledger, fits, newest, older_a, older_b]


def test_the_unit_model_matches_on_the_resolved_path_too():
    link, other = dm("/m/link.gguf", real="/blobs/abc"), dm("/m/other.gguf", 99)
    assert rank_models([other, link], unit_model="/blobs/abc", ledger_paths=(), fits={})[0] == link


def test_ranking_is_pure_and_does_not_mutate_its_input():
    pool = [dm("/m/b.gguf"), dm("/m/a.gguf")]
    before = list(pool)
    assert [m.name for m in rank_models(pool, unit_model=None, ledger_paths=(), fits={})] == ["a", "b"]
    assert pool == before
