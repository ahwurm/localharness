"""AUTO-02/03/04/06: the fit, the memory probe, the cache-ram number, the report and its lines (D-04..D-11, D-17, D-21, D-22, D-36, D-38).

Headers are plain dicts (one is built through `write_gguf` + `read_header` to prove the key shapes agree);
`_run_nvidia_smi` / `_read_meminfo` are faked through the module seam, so no test starts a process or
reads the host's memory; `record_opens()` proves the weights are stat-ed and never opened.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from localharness.agent.gguf_header import read_header
from localharness.config.models import AutosenseConfig
from localharness.provider import autosense
from localharness.provider.autosense import (
    FIT_BY_SERVER_LINE,
    FIT_KEYS,
    FreeMemory,
    WindowReport,
    cache_ram_mib,
    compute_fit,
    extra_cache_types,
    fit,
    fit_header,
    fmt_count,
    free_memory,
    has_cache_ram_arg,
    has_ctx_arg,
    kv_bytes_per_token,
    offload_kind,
    weights_bytes,
    window_changed,
    window_lines,
)
from localharness.provider.speed_stats import WindowRecord
from tests.unit.gguf_support import write_gguf
from tests.unit.launch_support import record_opens

RATIOS = AutosenseConfig()
GIB = 2**30

HEADER_A = {"block_count": 64, "head_count_kv": 8, "key_length": 128, "context_length": 262144}
HEADER_B = {"block_count": 32, "head_count_kv": 8, "embedding_length": 4096, "head_count": 32,
            "context_length": 131072}
HEADER_C = {"block_count": 48, "head_count_kv": [8, 8, 4], "key_length": 128, "context_length": 32768}

WEIGHTS_A = 19864223744  # 18.5 GiB


def report(**kw) -> WindowReport:
    base = dict(served=65536, trained=262144, fits=114688, free_bytes=6442450944,
                total_bytes=35004153856, weights_bytes=WEIGHTS_A, asked=None, mode="fit",
                attached=True)
    return WindowReport(**{**base, **kw})


# ---- fit_header: a GGUF declares its own types, so only well-typed positive ints survive (WR-01) ---

def test_fit_header_keeps_well_typed_values_and_drops_the_rest():
    raw = {"block_count": "abc", "context_length": [1], "key_length": True, "embedding_length": 0,
           "head_count": -3, "head_count_kv": 8}
    assert fit_header(raw) == {"head_count_kv": 8}
    assert fit_header(HEADER_A) == HEADER_A
    assert fit_header({}) == {}


def test_fit_header_filters_a_per_layer_head_count_kv_list():
    assert fit_header({"head_count_kv": [8, "x", 4, False]}) == {"head_count_kv": [8, 4]}
    assert fit_header({"head_count_kv": ["x"]}) == {}
    assert fit_header({"head_count_kv": "8"}) == {}


# ---- compute_fit: the one place the fit inputs are assembled (IN-01) ---------------------------------

def _record(tmp_path, header=None, **over):
    from localharness.config.models import ManagedServerConfig

    gguf = write_gguf(tmp_path / "m.gguf", header or {
        "general.architecture": "qwen3", "qwen3.block_count": 64, "qwen3.attention.head_count_kv": 8,
        "qwen3.attention.key_length": 128, "qwen3.context_length": 262144})
    return ManagedServerConfig(**{"runtime": "llamacpp", "binary": "/x/llama-server",
                                  "model": str(gguf), "port": 8080, **over})


@pytest.fixture
def host(monkeypatch):
    calls = []

    def free(ratios):
        calls.append("free_memory")
        return FreeMemory(24 * GIB, 32 * GIB, "nvidia-smi")

    monkeypatch.setattr(autosense, "free_memory", free)
    monkeypatch.setattr(autosense, "weights_bytes", lambda model, mmproj=None: WEIGHTS_A)
    monkeypatch.setattr(autosense, "_read_meminfo",
                        lambda: calls.append("meminfo") or {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB})
    return calls


def test_compute_fit_gathers_the_inputs_and_the_fit(tmp_path, host):
    fi = compute_fit(_record(tmp_path), RATIOS)
    assert fi.fits == 14950 and fi.weights == WEIGHTS_A and fi.offload == "gpu"
    assert fi.memory == FreeMemory(24 * GIB, 32 * GIB, "nvidia-smi")
    assert fi.header["context_length"] == 262144 and fi.meminfo is None
    assert host == ["free_memory"]  # /proc/meminfo only when asked for


def test_compute_fit_reads_meminfo_when_asked_and_counts_the_served_window_when_attached(tmp_path, host):
    fi = compute_fit(_record(tmp_path), RATIOS, attached_served=65536, meminfo=True)
    assert fi.meminfo == {"MemTotal": 64 * GIB, "MemAvailable": 60 * GIB}
    assert fi.fits == min(262144, 65536 + 24 * GIB // 262144)


def test_compute_fit_without_sizing_reads_only_the_header(tmp_path, host):
    fi = compute_fit(_record(tmp_path, extra_args=["-ngl", "20"]), RATIOS, size=False)
    assert host == [] and fi.fits is None and fi.memory is None and fi.weights is None
    assert fi.header["context_length"] == 262144 and fi.offload == "partial"


def test_compute_fit_types_the_header_and_never_raises_on_a_bad_file(tmp_path, host):
    bad = _record(tmp_path, {"general.architecture": "qwen3", "qwen3.block_count": "abc",
                             "qwen3.context_length": [1]}, extra_args=["-ngl", "10"])
    fi = compute_fit(bad, RATIOS)
    assert fi.header == {} and fi.offload == "unknown" and fi.fits is None
    (tmp_path / "m.gguf").write_bytes(b"not a gguf")
    assert compute_fit(bad, RATIOS).header == {}


@pytest.mark.parametrize("extra", [["-ctk", "q8_0", "-ctv", "q8_0"],
                                   ["--cache-type-k", "q8_0", "--cache-type-v=q8_0"],
                                   ["--cache-type-k=q8_0", "-ctv", "q8_0"]])
def test_compute_fit_reads_the_cache_types_from_extra_args(tmp_path, host, extra):
    """IN-04: `-ctk q8_0 -ctv q8_0` halves the KV cost (262144 -> 139264 bytes/token), so the fit is larger."""
    fi = compute_fit(_record(tmp_path, extra_args=extra), RATIOS)
    # usable 3919157658 // (64 * 8 * 128 * 2 * 34/32 = 139264) = 28141
    assert fi.fits == 28141 and fi.fits > 14950


def test_the_records_cache_types_win_over_extra_args(tmp_path, host):
    rec = _record(tmp_path, extra_args=["-ctk", "q8_0", "-ctv", "q8_0"], cache_type_k="f16", cache_type_v="f16")
    assert compute_fit(rec, RATIOS).fits == 14950


def test_an_unknown_cache_type_in_extra_args_is_not_computed(tmp_path, host):
    assert compute_fit(_record(tmp_path, extra_args=["-ctk", "mystery"]), RATIOS).fits is None


def test_extra_cache_types_takes_the_last_setting_and_none_when_absent():
    assert extra_cache_types(["-c", "4096"]) == (None, None)
    assert extra_cache_types(["-ctk", "f16", "--cache-type-k=q4_0"]) == ("q4_0", None)
    assert extra_cache_types(["-ctv"]) == (None, None)  # a flag with no value names nothing


def test_the_argv_detectors_agree_with_the_forms_llama_server_takes():
    assert all(has_ctx_arg(a) for a in (["-c", "1"], ["--ctx-size", "1"], ["--ctx-size=1"]))
    assert not has_ctx_arg(["-ngl", "1", "--cache-ram", "0"])
    assert all(has_cache_ram_arg(a) for a in (["--cache-ram", "0"], ["-cram", "0"], ["--cache-ram=0"]))
    assert not has_cache_ram_arg(["-c", "4096"])


# ---- the fit: three synthetic headers, hand-computed ------------------------------------------------

def test_header_a_is_the_formula_with_f16_cache():
    assert kv_bytes_per_token(HEADER_A, "f16", "f16") == 262144
    # usable = 24 GiB - 18.5 GiB - int(0.10 * weights) = 3919157658; // 262144 = 14950
    assert fit(HEADER_A, 24 * GIB, RATIOS, cache_type_k="f16", cache_type_v="f16", weights=WEIGHTS_A) == 14950


def test_header_b_derives_head_dim_and_is_capped_at_trained():
    assert kv_bytes_per_token(HEADER_B, "q8_0", "q8_0") == 69632
    # usable 27487790695 // 69632 = 394758, above trained 131072
    assert fit(HEADER_B, 30 * GIB, RATIOS, cache_type_k="q8_0", cache_type_v="q8_0", weights=4 * GIB) == 131072


def test_header_c_weights_beyond_free_memory_is_not_computed():
    assert fit(HEADER_C, 32 * GIB, RATIOS, weights=60 * GIB) is None


def test_attached_counts_the_served_window_plus_what_free_memory_adds():
    # min(262144, 65536 + 6442450944 // 262144) = 65536 + 24576
    assert fit(HEADER_A, 6442450944, RATIOS, 65536) == 90112


def test_attached_needs_no_weights_and_is_capped_at_trained():
    assert fit(HEADER_A, 6442450944, RATIOS, 262000) == 262144


@pytest.mark.parametrize("kw", [
    {"runtime": "vllm"}, {"cache_type_k": "q4_K"}, {"cache_type_v": "q4_K"}, {"free_bytes": None},
    {"weights": None}, {"offload": "partial"}, {"offload": "unknown"},
])
def test_unknown_inputs_give_none(kw):
    args = {"free_bytes": 24 * GIB, "weights": WEIGHTS_A, **kw}
    free = args.pop("free_bytes")
    assert fit(HEADER_A, free, RATIOS, **args) is None


@pytest.mark.parametrize("missing", ["block_count", "head_count_kv", "context_length"])
def test_a_missing_header_value_gives_none(missing):
    header = {k: v for k, v in HEADER_A.items() if k != missing}
    assert fit(header, 24 * GIB, RATIOS, weights=WEIGHTS_A) is None


def test_a_header_with_neither_key_length_nor_embedding_pair_gives_none():
    header = {"block_count": 32, "head_count_kv": 8, "context_length": 4096}
    assert fit(header, 24 * GIB, RATIOS, weights=GIB) is None


def test_the_key_shapes_agree_with_read_header(tmp_path):
    path = tmp_path / "m.gguf"
    write_gguf(path, {"general.architecture": "qwen3", "qwen3.block_count": 64,
                      "qwen3.attention.head_count_kv": 8, "qwen3.attention.key_length": 128,
                      "qwen3.context_length": 262144})
    header = read_header(path, FIT_KEYS)
    assert fit(header, 24 * GIB, RATIOS, weights=WEIGHTS_A) == 14950


def test_a_fit_is_never_zero_or_negative():
    assert fit(HEADER_A, 18 * GIB, RATIOS, weights=WEIGHTS_A) is None
    assert fit(HEADER_A, 0, RATIOS, weights=1) is None


def test_adjacency_k_q8_v_f16_bytes():
    assert kv_bytes_per_token(HEADER_A, "q8_0", "f16") == 64 * 8 * 128 * (34 / 32 + 2)


def test_ordering_the_cap_comes_after_the_division():
    # a huge free memory still returns exactly trained, not trained + anything
    assert fit(HEADER_A, 10_000 * GIB, RATIOS, weights=GIB) == 262144


# ---- offload_kind ------------------------------------------------------------------------------------

@pytest.mark.parametrize("args, n_layer, expect", [
    ([], 48, "gpu"),
    (["-ngl", "99"], 48, "gpu"),
    (["-ngl", "48"], 48, "gpu"),
    (["-ngl", "-1"], 48, "gpu"),
    (["--n-gpu-layers=20"], 48, "partial"),
    (["--n-gpu-layers", "20"], 48, "partial"),
    (["--gpu-layers", "0"], 48, "partial"),
    (["-ot", "exps=CPU"], 48, "unknown"),
    (["--override-tensor=exps=CPU"], 48, "unknown"),
    (["--n-cpu-moe", "8"], 48, "unknown"),
    (["-ncmoe", "8"], 48, "unknown"),
    (["--cpu-moe"], 48, "unknown"),
    (["-ngl", "20"], None, "unknown"),
])
def test_offload_kind(args, n_layer, expect):
    assert offload_kind(args, n_layer) == expect


# ---- weights_bytes: stat only ------------------------------------------------------------------------

def _file(path: Path, size: int) -> Path:
    path.write_bytes(b"x" * size)
    return path


def test_weights_sum_the_shards_and_the_mmproj_without_opening_any(tmp_path):
    first = _file(tmp_path / "m-00001-of-00003.gguf", 10)
    _file(tmp_path / "m-00002-of-00003.gguf", 20)
    _file(tmp_path / "m-00003-of-00003.gguf", 30)
    proj = _file(tmp_path / "mmproj.gguf", 5)
    with record_opens() as opened:
        assert weights_bytes(str(first), str(proj)) == 65
    assert not [p for p in opened if str(tmp_path) in p]


def test_a_plain_model_is_its_own_size(tmp_path):
    model = _file(tmp_path / "m.gguf", 7)
    with record_opens() as opened:
        assert weights_bytes(str(model)) == 7
    assert not [p for p in opened if str(tmp_path) in p]


def test_a_missing_file_or_mmproj_is_none(tmp_path):
    assert weights_bytes(str(tmp_path / "nope.gguf")) is None
    model = _file(tmp_path / "m.gguf", 7)
    assert weights_bytes(str(model), str(tmp_path / "no-proj.gguf")) is None


# ---- free_memory: the only subprocess -----------------------------------------------------------------

def test_nvidia_smi_mib_become_bytes(monkeypatch):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "31649, 32607\n")
    assert free_memory(RATIOS) == FreeMemory(31649 << 20, 32607 << 20, "nvidia-smi")


def test_the_first_gpu_line_is_used(monkeypatch):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "100, 200\n300, 400\n")
    assert free_memory(RATIOS) == FreeMemory(100 << 20, 200 << 20, "nvidia-smi")


def test_na_falls_to_meminfo_less_the_unified_reserve(monkeypatch):
    total, avail = 64139128 * 1024, 61066980 * 1024
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "[N/A], [N/A]\n")
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: {"MemTotal": total, "MemAvailable": avail})
    assert free_memory(RATIOS) == FreeMemory(int(avail - 0.15 * total), total, "meminfo")


def test_na_with_unreadable_meminfo_is_none(monkeypatch):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: "[N/A], [N/A]\n")
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: None)
    assert free_memory(RATIOS) is None


def test_no_nvidia_smi_is_none_not_meminfo(monkeypatch):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: None)
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: pytest.fail("meminfo must not be read"))
    assert free_memory(RATIOS) is None


def test_run_nvidia_smi_goes_through_which_outside_cwd_with_a_timeout(monkeypatch):
    seen = {}
    monkeypatch.setattr(autosense, "which_outside_cwd", lambda cmd: seen.setdefault("cmd", cmd) and "/x/nvidia-smi")

    class Done:
        returncode, stdout = 0, "1, 2\n"

    def run(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        return Done()

    monkeypatch.setattr(autosense.subprocess, "run", run)
    assert autosense._run_nvidia_smi() == "1, 2\n"
    assert seen["cmd"] == "nvidia-smi"
    assert seen["argv"][0] == "/x/nvidia-smi" and seen["kw"]["timeout"] == 5.0
    assert not seen["kw"].get("shell")


def test_run_nvidia_smi_failure_modes_are_none(monkeypatch):
    monkeypatch.setattr(autosense, "which_outside_cwd", lambda cmd: "/x/nvidia-smi")

    class Failed:
        returncode, stdout = 9, "1, 2\n"

    monkeypatch.setattr(autosense.subprocess, "run", lambda *a, **k: Failed())
    assert autosense._run_nvidia_smi() is None

    def hang(*a, **k):
        raise autosense.subprocess.TimeoutExpired("nvidia-smi", 5.0)

    monkeypatch.setattr(autosense.subprocess, "run", hang)
    assert autosense._run_nvidia_smi() is None
    monkeypatch.setattr(autosense, "which_outside_cwd", lambda cmd: None)
    assert autosense._run_nvidia_smi() is None


def test_read_meminfo_parses_kb_into_bytes(monkeypatch):
    text = "MemTotal:       64139128 kB\nMemFree:  1 kB\nMemAvailable:   61066980 kB\n"
    monkeypatch.setattr(Path, "read_text", lambda self, **kw: text)
    assert autosense._read_meminfo() == {"MemTotal": 64139128 * 1024, "MemAvailable": 61066980 * 1024}
    monkeypatch.setattr(Path, "read_text", lambda self, **kw: "MemTotal: nope kB\n")
    assert autosense._read_meminfo() is None


# ---- cache_ram_mib ------------------------------------------------------------------------------------

AVAIL, TOTAL = 61066980 * 1024, 64139128 * 1024


def test_cache_ram_gpu_resident_weights_do_not_count():
    assert cache_ram_mib(AVAIL, TOTAL, WEIGHTS_A, "gpu", [], RATIOS) == 23854


def test_cache_ram_no_mmap_counts_the_weights_as_host_resident():
    assert cache_ram_mib(AVAIL, TOTAL, WEIGHTS_A, "gpu", ["--no-mmap"], RATIOS) == 16276


def test_cache_ram_partial_offload_counts_the_weights():
    assert cache_ram_mib(AVAIL, TOTAL, WEIGHTS_A, "partial", [], RATIOS) == 16276


def test_cache_ram_is_clamped_at_the_ceiling():
    assert cache_ram_mib(TOTAL, 10 * GIB, 0, "gpu", [], RATIOS) == 5120  # 0.50 * 10 GiB


def test_cache_ram_unknown_or_nonpositive_is_none():
    assert cache_ram_mib(None, TOTAL, 0, "gpu", [], RATIOS) is None
    assert cache_ram_mib(AVAIL, None, 0, "gpu", [], RATIOS) is None
    assert cache_ram_mib(GIB, TOTAL, 2 * GIB, "partial", [], RATIOS) is None


# ---- window_lines --------------------------------------------------------------------------------------

LINE1 = ("window: served 65,536 · trained 262,144 · fits 114,688 "
         "(free 6.0 GiB of 32.6 beside the served model; weights 18.5 GiB)")


@pytest.mark.parametrize("undo, second", [
    ("record", "  → unset `server.context` to relaunch with -c 114688"),
    ("extra_args", "  → drop the -c from `server.extra_args` to relaunch with -c 114688"),
    ("hand", "  → the next harness launch uses -c 114688"),
    ("no-record", "  → run `localharness init` to let the harness size the window"),
])
def test_the_attach_block_is_two_lines_by_undo(undo, second):
    assert window_lines(report(), RATIOS, undo) == [LINE1, second]


def test_an_attach_report_without_an_undo_is_line_one_alone():
    assert window_lines(report(), RATIOS) == [LINE1]


def test_a_launch_report_is_one_line_whatever_undo_says():
    launch = report(attached=False, asked=114688)
    expect = ["window: served 65,536 · trained 262,144 · fits 114,688 "
              "(free 6.0 GiB of 32.6; weights 18.5 GiB) (asked 114,688)"]
    for undo in (None, "record", "extra_args", "hand", "no-record"):
        assert window_lines(launch, RATIOS, undo) == expect


def test_an_attach_report_inside_notify_ratio_is_empty():
    assert window_lines(report(served=100000, fits=110000), RATIOS, "record") == []


def test_a_launch_report_is_never_gated():
    launch = report(attached=False, served=100000, fits=110000, asked=100000, mode="record",
                    free_bytes=None, total_bytes=None, weights_bytes=None)
    assert window_lines(launch, RATIOS) == ["window: served 100,000 · trained 262,144 · fits 110,000"]


def test_empty_an_attach_report_with_unknown_served_or_fit_is_empty():
    assert window_lines(report(served=None), RATIOS, "record") == []
    assert window_lines(report(fits=None), RATIOS, "record") == []


def test_fit_by_server_line():
    r = report(attached=False, mode="fit-by-server", served=None, fits=None)
    assert window_lines(r, RATIOS) == [FIT_BY_SERVER_LINE] == ["window: served by llama-server's --fit"]
    verified = report(attached=False, mode="fit-by-server", served=65536, fits=None)
    assert window_lines(verified, RATIOS) == ["window: served by llama-server's --fit · served 65,536"]


def test_empty_a_launch_whose_probe_returned_nothing():
    r = WindowReport(None, 262144, 14950, None, None, None, 14950, "fit", verified=False)
    assert window_lines(r, RATIOS) == ["window: served — · trained 262,144 · fits 14,950 (not verified)"]


def test_a_retry_is_named():
    r = report(attached=False, asked=57344, served=57344, retried=True, fits=57344,
               free_bytes=None, total_bytes=None, weights_bytes=None)
    assert window_lines(r, RATIOS)[0].endswith(" (after one retry at -c 57,344)")


def test_ollama_and_lmstudio_forms():
    ollama = WindowReport(4096, 32768, None, None, None, None, None, "not-computed", runtime="ollama")
    assert window_lines(ollama, RATIOS) == ["window: served 4,096 · trained 32,768 · Ollama decides the window"]
    lm = WindowReport(8192, 32768, None, None, None, None, None, "not-computed", runtime="lmstudio")
    assert window_lines(lm, RATIOS) == ["window: served 8,192 · trained 32,768"]


def test_encoding_thousands_separators_in_the_line_and_none_in_the_undo():
    first, second = window_lines(report(fits=1234567, trained=2000000), RATIOS, "record")
    assert "fits 1,234,567" in first and "trained 2,000,000" in first
    assert second.endswith("-c 1234567")


# ---- window_changed ------------------------------------------------------------------------------------

def rec(served=65536, trained=262144, fits=114688) -> WindowRecord:
    return WindowRecord(served, trained, fits, "2026-10-10T00:00:00+00:00")


def test_window_changed_cases():
    r = report()
    assert window_changed(None, r, 0.25)
    assert not window_changed(rec(), r, 0.25)
    assert window_changed(rec(served=32768), r, 0.25)
    assert window_changed(rec(trained=131072), r, 0.25)
    assert not window_changed(rec(fits=100000), r, 0.25)  # 14.7%
    assert window_changed(rec(fits=80000), r, 0.25)  # 43%
    assert window_changed(rec(fits=100000), report(fits=None), 0.25)
    assert window_changed(rec(fits=None), r, 0.25)
    assert not window_changed(rec(fits=None), report(fits=None), 0.25)


# ---- purity and imports --------------------------------------------------------------------------------

def test_the_module_imports_only_stdlib_and_localharness():
    tree = ast.parse(Path(autosense.__file__).read_text(encoding="utf-8"))
    roots = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    roots |= {n.module.split(".")[0] for n in ast.walk(tree)
              if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
    assert roots <= set(sys.stdlib_module_names) | {"localharness"}, roots


# ---- edge probes (Task 2) -------------------------------------------------------------------------------

def test_the_attach_gate_is_strict_at_the_boundary():
    on_edge = report(served=100000, fits=125000)
    assert window_lines(on_edge, RATIOS, "record") == []
    assert len(window_lines(report(served=100000, fits=125001), RATIOS, "record")) == 2
    assert window_lines(report(served=100000, fits=75000), RATIOS, "record") == []
    assert len(window_lines(report(served=100000, fits=74999), RATIOS, "record")) == 2


@pytest.mark.parametrize("fits", [125000, 125001])
def test_a_launch_report_prints_at_both_sides_of_the_boundary(fits):
    launch = report(attached=False, served=100000, fits=fits, asked=100000, mode="record")
    assert len(window_lines(launch, RATIOS)) == 1


def test_window_changed_is_strict_at_the_boundary():
    assert not window_changed(rec(fits=100000), report(fits=125000), 0.25)
    assert window_changed(rec(fits=100000), report(fits=125001), 0.25)
    assert not window_changed(rec(fits=100000), report(fits=75000), 0.25)
    assert window_changed(rec(fits=100000), report(fits=74999), 0.25)


def test_window_changed_does_not_divide_by_a_zero_fit():
    assert window_changed(rec(fits=0), report(fits=1), 0.25)
    assert not window_changed(rec(fits=0), report(fits=0), 0.25)


def test_a_shard_sum_takes_only_its_own_set(tmp_path):
    first = _file(tmp_path / "m-00001-of-00003.gguf", 10)
    _file(tmp_path / "m-00002-of-00003.gguf", 20)
    _file(tmp_path / "m-00003-of-00003.gguf", 30)
    _file(tmp_path / "m-00001-of-00004.gguf", 1000)  # another split of the same prefix
    _file(tmp_path / "other-00002-of-00003.gguf", 2000)  # another model
    _file(tmp_path / "m.gguf", 4000)
    _file(tmp_path / "m-00002-of-00003.gguf.part", 8000)
    assert weights_bytes(str(first)) == 60


def test_a_plain_name_beside_a_shard_set_is_only_itself(tmp_path):
    model = _file(tmp_path / "m.gguf", 7)
    _file(tmp_path / "m-00001-of-00002.gguf", 100)
    _file(tmp_path / "m-00002-of-00002.gguf", 200)
    assert weights_bytes(str(model)) == 7


def test_a_missing_shard_sibling_directory_entry_is_not_counted(tmp_path):
    (tmp_path / "m-00002-of-00002.gguf").mkdir()  # a directory with a shard-shaped name
    first = _file(tmp_path / "m-00001-of-00002.gguf", 10)
    assert weights_bytes(str(first)) is None


def test_a_model_path_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _file(tmp_path / "m.gguf", 9)
    assert weights_bytes("~/m.gguf") == 9


def test_mixed_cache_types_sum_their_own_bytes():
    assert kv_bytes_per_token(HEADER_A, "q8_0", "q4_0") == 64 * 8 * 128 * (34 / 32 + 18 / 32)
    assert kv_bytes_per_token(HEADER_A, None, None) == 262144  # unset is f16


def test_an_unknown_value_type_is_none_even_with_a_known_key_type():
    assert kv_bytes_per_token(HEADER_A, "q8_0", "q4_K") is None
    assert kv_bytes_per_token(HEADER_A, "q4_K", "q8_0") is None


def test_a_head_count_kv_list_takes_the_maximum():
    assert kv_bytes_per_token(HEADER_C, "f16", "f16") == 48 * 8 * 128 * 4
    zeros = {**HEADER_C, "head_count_kv": [0, 0, 4]}
    assert kv_bytes_per_token(zeros, "f16", "f16") == 48 * 4 * 128 * 4
    assert kv_bytes_per_token({**HEADER_C, "head_count_kv": []}, "f16", "f16") is None
    assert kv_bytes_per_token({**HEADER_C, "head_count_kv": [0, 0]}, "f16", "f16") is None


@pytest.mark.parametrize("args", [["-ngl"], ["--n-gpu-layers=abc"], ["-ngl", "--cpu-moe"], ["--gpu-layers", ""]])
def test_an_unreadable_ngl_is_unknown(args):
    assert offload_kind(args, 48) == "unknown"


def test_the_last_ngl_wins_and_a_hand_placement_wins_over_it():
    assert offload_kind(["-ngl", "10", "-ngl", "99"], 48) == "gpu"
    assert offload_kind(["-ngl", "99", "-ngl", "10"], 48) == "partial"
    assert offload_kind(["-ngl", "99", "--cpu-moe"], 48) == "unknown"


def test_the_parenthesis_is_omitted_unless_all_three_byte_counts_are_known():
    expect = ["window: served 65,536 · trained 262,144 · fits 114,688"]
    for gap in ("free_bytes", "total_bytes", "weights_bytes"):
        assert window_lines(report(**{gap: None}), RATIOS) == expect
    assert window_lines(report(free_bytes=None, total_bytes=None, weights_bytes=None), RATIOS) == expect


def test_window_lines_and_window_changed_are_pure(monkeypatch, tmp_path):
    monkeypatch.setattr(autosense, "_run_nvidia_smi", lambda: pytest.fail("a subprocess was run"))
    monkeypatch.setattr(autosense, "_read_meminfo", lambda: pytest.fail("meminfo was read"))
    r = report()
    with record_opens() as opened:
        first = window_lines(r, RATIOS, "record")
        again = window_lines(r, RATIOS, "record")
        assert window_changed(rec(), r, 0.25) is window_changed(rec(), r, 0.25)
    assert first == again
    assert opened == []


def test_fit_and_the_rest_do_not_open_files_either(tmp_path):
    with record_opens() as opened:
        fit(HEADER_A, 24 * GIB, RATIOS, weights=WEIGHTS_A)
        offload_kind(["-ngl", "20"], 48)
        cache_ram_mib(AVAIL, TOTAL, WEIGHTS_A, "gpu", [], RATIOS)
    assert opened == []


def test_the_report_is_frozen():
    with pytest.raises(AttributeError):
        report().served = 1  # type: ignore[misc]


def test_fmt_count_is_the_one_window_figure_formatter():
    """IN-03: autosense owns it and doctor imports it; a non-int or a bool is an em dash."""
    assert fmt_count(262144) == "262,144" and fmt_count(None) == "—" and fmt_count(True) == "—"
    from localharness.cli import doctor_cmd

    assert not hasattr(doctor_cmd, "_n")
