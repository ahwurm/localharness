"""INIT-01 / INIT-02 (discovery half): the bounded walk, typing with exact keys, split admission,
projector pairing, preset naming, ranking and the mapped-model typing (D-02..D-05, D-07, D-47, D-48, D-52).

Every file is a real GGUF header written by `write_gguf` under `tmp_path`; nothing reads the owner's
`~/models`. The u16 split keys are `Raw(2, ...)` as llama.cpp writes them.
"""
from __future__ import annotations

import os
import random
import signal
import struct
from pathlib import Path

import pytest

from localharness.agent import gguf_header
from localharness.agent.gguf_header import (
    DISCOVERY_MAX_DEPTH,
    DISCOVERY_MAX_ENTRIES,
    DISCOVERY_MAX_FILES,
    DiscoveredModel,
    GgufHeaderError,
    discover,
    preset_name,
    rank_models,
    served_model,
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


# ---- the box's seven files (RESEARCH "On the box") ---------------------------------------------------

QWEN, SWIFT = "Qwen3.8-27B-Nvidia-NVFP4.gguf", "Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix.gguf"
FLASH = "Swift-Qwen3.8-Flash-Next-IQ3_XXS"


def box(root: Path) -> Path:
    """`<root>/gguf` and `<root>/gguf/flashnext` as laid out on the owner's machine."""
    gguf, flash = root / "gguf", root / "gguf" / "flashnext"
    model_file(gguf / QWEN, "qwen35", 5120)
    model_file(gguf / SWIFT, "qwen35", 5120)
    projector_file(gguf / "mmproj-F16.gguf", 5120)
    projector_file(gguf / "mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf", 5120)
    model_file(flash / f"{FLASH}-00001-of-00002.gguf", "qwen4exp", 2560,
               **{"split.no": u16(0), "split.count": u16(2)})
    write_gguf(flash / f"{FLASH}-00002-of-00002.gguf", {"split.no": u16(1), "split.count": u16(2)})
    projector_file(flash / "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf", 2560)
    return gguf


def test_the_box_pairs_all_three_projectors_as_the_spec_states(tmp_path):
    gguf = box(tmp_path / "models")
    found = walk_and_discover(tmp_path / "models")
    by_name = {m.name: m for m in found.models}
    assert sorted(by_name) == ["Qwen3.8-27B-Nvidia-NVFP4", "Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix", FLASH]
    assert by_name["Qwen3.8-27B-Nvidia-NVFP4"].mmproj == gguf / "mmproj-F16.gguf"
    assert by_name["Swift-1.5-Qwen3.8-27B-NVFP4-Q8mix"].mmproj == gguf / "mmproj-Swift-1.5-Qwen3.8-27B-F16.gguf"
    assert by_name[FLASH].mmproj == gguf / "flashnext" / "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"
    assert by_name[FLASH].path == gguf / "flashnext" / f"{FLASH}-00001-of-00002.gguf"
    assert found.projectors == ()  # the 00002 shard and every projector are never models


def test_discovery_does_not_depend_on_the_order_the_paths_arrive_in(tmp_path):
    box(tmp_path / "models")
    paths = walk_ggufs([tmp_path / "models"])
    expect = discover(paths)
    for seed in range(5):
        shuffled = list(paths)
        random.Random(seed).shuffle(shuffled)
        assert discover(shuffled) == expect


# ---- split admission (D-03, D-48) --------------------------------------------------------------------

def test_a_split_model_with_a_missing_shard_is_not_admitted(tmp_path):
    gguf = box(tmp_path / "models")
    (gguf / "flashnext" / f"{FLASH}-00002-of-00002.gguf").unlink()
    found = walk_and_discover(tmp_path / "models")
    assert FLASH not in [m.name for m in found.models]
    assert len(found.models) == 2
    # its projector has no model of dimension 2560 left, so it is reported unpaired
    assert found.projectors == (gguf / "flashnext" / "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",)


def test_a_shard_that_is_a_fifo_refuses_admission_without_blocking(tmp_path):
    gguf = box(tmp_path / "models")
    shard2 = gguf / "flashnext" / f"{FLASH}-00002-of-00002.gguf"
    shard2.unlink()
    os.mkfifo(shard2)
    signal.alarm(5)
    try:
        found = walk_and_discover(tmp_path / "models")
    finally:
        signal.alarm(0)
    assert FLASH not in [m.name for m in found.models]


def test_split_no_above_zero_is_never_a_preset_even_with_a_full_header(tmp_path):
    root = tmp_path / "models"
    model_file(root / "m-00002-of-00002.gguf", **{"split.no": u16(1), "split.count": u16(2)})
    assert walk_and_discover(root).models == ()


def test_split_zero_with_siblings_present_admits_once(tmp_path):
    root = tmp_path / "models"
    model_file(root / "m-00001-of-00003.gguf", **{"split.no": u16(0), "split.count": u16(3)})
    for k in (2, 3):
        write_gguf(root / f"m-0000{k}-of-00003.gguf", {"split.no": u16(k - 1), "split.count": u16(3)})
    found = walk_and_discover(root)
    assert [(m.name, m.path.name) for m in found.models] == [("m", "m-00001-of-00003.gguf")]


def test_a_sibling_that_is_a_directory_is_not_a_shard(tmp_path):
    root = tmp_path / "models"
    model_file(root / "m-00001-of-00002.gguf", **{"split.no": u16(0), "split.count": u16(2)})
    (root / "m-00002-of-00002.gguf").mkdir()
    assert walk_and_discover(root).models == ()


def test_split_zero_that_claims_one_part_is_admitted_as_itself(tmp_path):
    root = tmp_path / "models"
    model_file(root / "single.gguf", **{"split.no": u16(0), "split.count": u16(1)})
    assert [m.name for m in walk_and_discover(root).models] == ["single"]


def test_split_zero_whose_name_does_not_match_its_count_is_not_admitted(tmp_path):
    root = tmp_path / "models"
    model_file(root / "odd.gguf", **{"split.no": u16(0), "split.count": u16(4)})
    model_file(root / "off-00001-of-00009.gguf", **{"split.no": u16(0), "split.count": u16(2)})
    assert walk_and_discover(root).models == ()


# ---- pairing: dimension, prefix tie-break, elimination (D-04, D-48) ---------------------------------

def test_two_projectors_and_two_models_of_one_dimension_with_no_prefix_pair_none(tmp_path):
    root = tmp_path / "models"
    for n in ("X", "Y"):
        model_file(root / f"{n}.gguf", embedding=1024)
    for n in ("A", "B"):
        projector_file(root / f"mmproj-{n}.gguf", dim=1024)
    found = walk_and_discover(root)
    assert [m.mmproj for m in found.models] == [None, None] and len(found.projectors) == 2


def test_elimination_pairs_the_last_projector_and_model_of_a_dimension(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Alpha-Q4.gguf", embedding=1024)
    model_file(root / "Beta-Q4.gguf", embedding=1024)
    projector_file(root / "mmproj-Alpha-F16.gguf", dim=1024)  # prefix pairs it with Alpha-Q4
    projector_file(root / "mmproj-zzz.gguf", dim=1024)         # then elimination gives it Beta-Q4
    by_name = {m.name: m.mmproj.name if m.mmproj else None for m in walk_and_discover(root).models}
    assert by_name == {"Alpha-Q4": "mmproj-Alpha-F16.gguf", "Beta-Q4": "mmproj-zzz.gguf"}


def test_a_model_claimed_by_two_projectors_goes_to_the_strictly_longest_prefix(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Gemma-4-12b-it.gguf", embedding=1024)
    projector_file(root / "mmproj-Gemma-4-12b-it-F16.gguf", dim=1024)  # LCP 14
    projector_file(root / "mmproj-Gemma-F16.gguf", dim=1024)           # LCP 6
    found = walk_and_discover(root)
    assert found.models[0].mmproj == root / "mmproj-Gemma-4-12b-it-F16.gguf"
    assert found.projectors == (root / "mmproj-Gemma-F16.gguf",)


def test_a_tie_between_two_projectors_for_one_model_pairs_neither(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Gemma-Q4.gguf", embedding=1024)
    projector_file(root / "mmproj-Gemma-A.gguf", dim=1024)
    projector_file(root / "mmproj-Gemma-B.gguf", dim=1024)
    found = walk_and_discover(root)
    assert found.models[0].mmproj is None and len(found.projectors) == 2


def test_a_projector_claims_a_model_only_when_its_prefix_is_strictly_longest_for_exactly_one(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Gemma-Q4.gguf", embedding=1024)
    model_file(root / "Gemma-Q8.gguf", embedding=1024)
    projector_file(root / "mmproj-Gemma-F16.gguf", dim=1024)  # LCP 6 with both: a tie, no claim
    found = walk_and_discover(root)
    assert [m.mmproj for m in found.models] == [None, None]


def test_a_projector_in_another_directory_never_pairs(tmp_path):
    root = tmp_path / "models"
    model_file(root / "a" / "M.gguf", embedding=1024)
    proj = projector_file(root / "b" / "mmproj-M.gguf", dim=1024)
    found = walk_and_discover(root)
    assert found.models[0].mmproj is None and found.projectors == (proj,)


def test_the_mmproj_lead_is_stripped_case_insensitively_with_any_of_the_three_separators(tmp_path):
    root = tmp_path / "models"
    model_file(root / "Foo-1.gguf", embedding=1024)
    model_file(root / "Bar-1.gguf", embedding=1024)
    projector_file(root / "MMPROJ_Foo-1-F16.gguf", dim=1024)
    projector_file(root / "mmproj.Bar-1-F16.gguf", dim=1024)
    by_name = {m.name: m.mmproj.name for m in walk_and_discover(root).models}
    assert by_name == {"Foo-1": "MMPROJ_Foo-1-F16.gguf", "Bar-1": "mmproj.Bar-1-F16.gguf"}


# ---- names (D-05, D-48) ------------------------------------------------------------------------------

def test_preset_name_strips_the_extension_and_shard_suffix_and_replaces_other_characters():
    assert preset_name("/m/My Model+v2 (q4)-00001-of-00003.gguf") == "My-Model+v2--q4-"
    assert preset_name("/m/x.GGUF") == "x"
    assert preset_name("/m/Qwen3.8-27B_Q4.0.gguf") == "Qwen3.8-27B_Q4.0"
    assert preset_name("/m/naïve:model.gguf") == "na-ve-model"


def test_unique_preset_name_appends_the_first_free_counter():
    assert unique_preset_name("x", {"x", "x-2"}) == "x-3"
    assert unique_preset_name("x", set()) == "x"
    assert unique_preset_name("x", {"y"}) == "x"


# ---- the walk: symlinks, special files and caps (D-02, D-51; T-5-06..08) -----------------------------

def names(paths, root: Path) -> list[str]:
    return [str(p.relative_to(root)) for p in paths]


def test_the_walk_lists_regular_gguf_files_of_any_case_in_name_order(tmp_path):
    root = tmp_path / "models"
    for rel in ("b/z.gguf", "a.GGUF", "a2.txt", "b/y.Gguf", "c/d/e.gguf"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b"")
    assert names(walk_ggufs([root]), root.resolve()) == ["a.GGUF", "b/y.Gguf", "b/z.gguf", "c/d/e.gguf"]


def test_a_missing_root_is_skipped_and_two_names_for_one_root_are_walked_once(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "a.gguf").write_bytes(b"")
    alias = tmp_path / "alias"
    alias.symlink_to(root)
    assert len(walk_ggufs([tmp_path / "nope", root, alias, str(root) + "/../models"])) == 1


def test_a_nested_root_does_not_list_a_file_twice(tmp_path):
    root = tmp_path / "models"
    (root / "inner").mkdir(parents=True)
    (root / "inner" / "a.gguf").write_bytes(b"")
    assert len(walk_ggufs([root, root / "inner"])) == 1


def test_the_root_is_user_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a.gguf").write_bytes(b"")
    assert len(walk_ggufs(["~/models"])) == 1


def test_a_symlinked_directory_is_never_descended(tmp_path):
    one, other = tmp_path / "one", tmp_path / "other"
    other.mkdir()
    (other / "hidden.gguf").write_bytes(b"")
    one.mkdir()
    (one / "linkdir").symlink_to(other, target_is_directory=True)
    (one / "plain.gguf").write_bytes(b"")
    assert names(walk_ggufs([one]), one.resolve()) == ["plain.gguf"]


def test_a_symlinked_gguf_inside_a_root_is_a_candidate_under_the_symlink_path(tmp_path):
    root = tmp_path / "hub"
    (root / "blobs").mkdir(parents=True)
    (root / "snapshots" / "c").mkdir(parents=True)
    (root / "blobs" / "abc").write_bytes(b"")
    link = root / "snapshots" / "c" / "model.gguf"
    link.symlink_to("../../blobs/abc")
    assert walk_ggufs([root]) == [Path(os.path.realpath(root)) / "snapshots" / "c" / "model.gguf"]


def test_a_symlinked_gguf_pointing_outside_every_root_is_not_a_candidate(tmp_path):
    root, outside = tmp_path / "models", tmp_path / "elsewhere"
    root.mkdir()
    outside.mkdir()
    (outside / "real.gguf").write_bytes(b"")
    (root / "x.gguf").symlink_to(outside / "real.gguf")
    (root / "dangling.gguf").symlink_to(outside / "missing.gguf")
    assert walk_ggufs([root]) == []


def test_a_symlink_into_a_sibling_root_is_a_candidate(tmp_path):
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (two / "real.gguf").write_bytes(b"")
    (one / "link.gguf").symlink_to(two / "real.gguf")
    assert names(walk_ggufs([one, two]), tmp_path.resolve()) == ["one/link.gguf", "two/real.gguf"]


def test_a_fifo_named_gguf_is_never_listed_and_never_blocks(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    os.mkfifo(root / "trap.gguf")
    signal.alarm(5)
    try:
        assert walk_ggufs([root]) == []
    finally:
        signal.alarm(0)


def test_a_symlink_to_a_fifo_inside_the_root_reaches_discover_and_is_skipped_without_blocking(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    os.mkfifo(root / "pipe")
    (root / "trap.gguf").symlink_to(root / "pipe")
    model_file(root / "good.gguf")
    signal.alarm(5)
    try:
        listed = walk_ggufs([root])
        found = discover(listed)
    finally:
        signal.alarm(0)
    assert any(p.name == "trap.gguf" for p in listed)
    assert [m.name for m in found.models] == ["good"]


def test_a_tree_deeper_than_max_depth_yields_nothing_below_it(tmp_path):
    root = tmp_path / "models"
    deep = root / "a" / "b" / "c"
    deep.mkdir(parents=True)
    for rel in ("top.gguf", "a/one.gguf", "a/b/two.gguf", "a/b/c/three.gguf"):
        (root / rel).write_bytes(b"")
    assert names(walk_ggufs([root], max_depth=2), root.resolve()) == ["a/b/two.gguf", "a/one.gguf", "top.gguf"]
    assert names(walk_ggufs([root], max_depth=0), root.resolve()) == ["top.gguf"]
    assert len(walk_ggufs([root])) == 4  # the default depth of six reaches all of them


def test_max_files_keeps_what_was_found(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    for n in range(5):
        (root / f"m{n}.gguf").write_bytes(b"")
    assert names(walk_ggufs([root], max_files=2), root.resolve()) == ["m0.gguf", "m1.gguf"]


def test_max_entries_stops_the_walk_keeping_what_was_found(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    for n in range(6):
        (root / f"m{n}.gguf").write_bytes(b"")
    assert names(walk_ggufs([root], max_entries=3), root.resolve()) == ["m0.gguf", "m1.gguf", "m2.gguf"]
    assert names(walk_ggufs([root], max_entries=0), root.resolve()) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any directory")
def test_an_unreadable_directory_is_skipped_silently(tmp_path):
    root = tmp_path / "models"
    (root / "locked").mkdir(parents=True)
    (root / "locked" / "hidden.gguf").write_bytes(b"")
    (root / "ok.gguf").write_bytes(b"")
    (root / "locked").chmod(0)
    try:
        assert names(walk_ggufs([root]), root.resolve()) == ["ok.gguf"]
    finally:
        (root / "locked").chmod(0o700)


def test_the_stores_resolve_gguf_path_knows_are_not_walked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for rel in (".lmstudio/models/pub/repo/lm.gguf", ".ollama/models/blobs/ol.gguf"):
        (tmp_path / rel).parent.mkdir(parents=True)
        (tmp_path / rel).write_bytes(b"")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "mine.gguf").write_bytes(b"")
    assert [p.name for p in walk_ggufs(["~/models"])] == ["mine.gguf"]


# ---- served_model: the mapped path the import types (D-52) -------------------------------------------

def fake_reader(table: dict[str, object]):
    calls: list[str] = []

    def read(path, want):
        calls.append(os.fspath(path))
        value = table.get(os.fspath(path), GgufHeaderError("bad magic"))
        if isinstance(value, Exception):
            raise value
        return value

    read.calls = calls
    return read


def test_served_model_returns_a_blob_that_types_as_a_model():
    read = fake_reader({"/x/blobs/abc": {"type": "model", "architecture": "llama"}})
    assert served_model(["/x/blobs/abc"], read=read) == "/x/blobs/abc"


def test_served_model_rewrites_a_shard_name_to_its_first_shard():
    read = fake_reader({"/x/m-00001-of-00002.gguf": {"architecture": "llama", "split.no": 0}})
    assert served_model(["/x/m-00002-of-00002.gguf"], read=read) == "/x/m-00001-of-00002.gguf"
    assert read.calls == ["/x/m-00001-of-00002.gguf"]


@pytest.mark.parametrize("header", [
    {"type": "mmproj", "architecture": "clip"},
    {"architecture": "clip"},
    {"type": "adapter", "architecture": "llama"},
    {"type": "model", "architecture": "llama", "split.no": 1},
    GgufHeaderError("bad magic"),
    OSError("gone"),
])
def test_served_model_skips_what_is_not_the_served_model(header):
    read = fake_reader({"/x/first.gguf": header, "/x/second.gguf": {"architecture": "llama"}})
    assert served_model(["/x/first.gguf", "/x/second.gguf"], read=read) == "/x/second.gguf"


def test_served_model_stops_at_the_first_qualifying_path_and_none_when_nothing_qualifies():
    read = fake_reader({"/x/a.gguf": {"architecture": "llama"}, "/x/b.gguf": {"architecture": "llama"}})
    assert served_model(["/x/a.gguf", "/x/b.gguf"], read=read) == "/x/a.gguf"
    assert read.calls == ["/x/a.gguf"]
    assert served_model([], read=read) is None
    assert served_model(["/x/nope"], read=read) is None


def test_served_model_with_the_real_reader_refuses_a_non_gguf_and_a_fifo(tmp_path):
    (tmp_path / "text").write_text("hello")
    os.mkfifo(tmp_path / "pipe")
    real = model_file(tmp_path / "blobs" / "abc")
    signal.alarm(5)
    try:
        assert served_model([tmp_path / "text", tmp_path / "pipe", real]) == str(real)
    finally:
        signal.alarm(0)
