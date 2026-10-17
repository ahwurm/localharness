"""The window autosense: what the fit is, what it needs, and what the harness says about it.

Three numbers describe a context window. `served` is what the running server answers with,
`trained` is what the model was trained for (its GGUF header), and `fits` is what the free
memory beside the weights could hold: the largest window whose KV cache still fits after the
weights and a compute margin. The fit is one function, `fit`, used by every launch (through
`SpawnedProcessStrategy.activate`), by the attach report `start` makes and by `doctor`.

Everything here is pure except `free_memory` (the only subprocess: `nvidia-smi`, or
`/proc/meminfo` on a unified-memory machine) and `weights_bytes` (a stat of the model files, never
an open). A number that cannot be known is `None`, never a guess, and a fit at or below zero is
`None` too: the caller then launches without `-c` and llama-server's own `--fit` sizes the
window. Lines are returned by `window_lines`, never printed; nothing in this module launches
anything or writes a file.
"""
from __future__ import annotations

import os
import re
import stat
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from localharness.core.which import which_outside_cwd
from localharness.provider.gguf_header import shard_paths
from localharness.provider.runtimes import row_of
from localharness.provider.speed_stats import WindowRecord

if TYPE_CHECKING:
    from localharness.config.models import ManagedServerConfig
    from localharness.provider.lifecycle import Sensed
    from localharness.provider.tuning import Tuned

# What callers hand to `provider.gguf_header.read_header` so the header holds exactly the fit's inputs.
FIT_KEYS = frozenset({
    "block_count", "head_count_kv", "key_length", "embedding_length", "head_count", "context_length",
    "nextn_predict_layers", "expert_count", "full_attention_interval", "indexer.key_length",
    "sliding_window", "sliding_window_pattern", "key_length_swa", "value_length_swa", "value_length",
})

# Bytes per element of a KV cache type, from ggml's block sizes (bytes per block / elements per block).
KV_BYTES: Mapping[str, float] = MappingProxyType({
    "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
    "q5_0": 22 / 32, "q5_1": 24 / 32, "iq4_nl": 18 / 32,
})

FIT_BY_SERVER_LINE = "window: served by llama-server's --fit"

_MIB = 2**20
_SMI_TIMEOUT = 5.0


@dataclass(frozen=True)
class FreeMemory:
    """Memory the weights and the KV cache can still use: free and total bytes, and where they came from."""

    free_bytes: int
    total_bytes: int
    source: str  # "nvidia-smi" or "meminfo"


def _smi(query: str) -> str | None:
    """nvidia-smi's CSV for `query` (MiB), or None when it is absent, fails or hangs."""
    exe = which_outside_cwd("nvidia-smi")
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [exe, query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_SMI_TIMEOUT, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _run_nvidia_smi() -> str | None:
    return _smi("--query-gpu=memory.free,memory.total")


def held_by(pid: int) -> int | None:
    """Bytes of GPU memory nvidia-smi lists for `pid`, summed over its rows; None when it lists none."""
    out = _smi("--query-compute-apps=pid,used_memory")
    mib: list[int] = []
    for line in (out or "").splitlines()[:256]:
        try:
            row_pid, used = (int(field) for field in line.split(","))
        except ValueError:
            continue
        if row_pid == pid:
            mib.append(used)
    return sum(mib) * _MIB if mib else None


def memory_after_stop(mem: "FreeMemory | None", held: int | None) -> "FreeMemory | None":
    """`mem` once the incumbent holding `held` bytes is gone; with `held` unknown, the card less 1 GiB."""
    if mem is None:
        return None
    free = mem.free_bytes + held if held is not None else mem.total_bytes - 2**30
    return replace(mem, free_bytes=max(0, min(free, mem.total_bytes)))


def _read_meminfo() -> dict[str, int] | None:
    """MemTotal and MemAvailable of /proc/meminfo in bytes, or None when unreadable."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="ascii", errors="replace")
        found: dict[str, int] = {}
        for line in text.splitlines():
            name, _, rest = line.partition(":")
            if name in ("MemTotal", "MemAvailable"):
                found[name] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return found if len(found) == 2 else None


def free_memory(ratios: Any) -> FreeMemory | None:
    """Free and total bytes of the first GPU; on a unified-memory machine (`[N/A]`) the available
    RAM less `unified_reserve_ratio` of the total. None when nvidia-smi is absent, fails or hangs."""
    out = _run_nvidia_smi()
    if out is None or not out.strip():
        return None
    try:
        free_mib, total_mib = (int(field.strip()) for field in out.strip().splitlines()[0].split(","))
    except ValueError:
        mi = _read_meminfo()
        if mi is None:
            return None
        # Clamped: with less available than the reserve there is nothing free, not a negative amount.
        free = max(0, int(mi["MemAvailable"] - ratios.unified_reserve_ratio * mi["MemTotal"]))
        return FreeMemory(free, mi["MemTotal"], "meminfo")
    return FreeMemory(free_mib * _MIB, total_mib * _MIB, "nvidia-smi")


def _pos_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def fit_header(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The header values a caller may rely on: only well-typed positive ints survive, and
    `head_count_kv` may stay a list of them and `sliding_window_pattern` a list of bools or a positive int. A GGUF declares its own types, so a value that is a
    string, a bool or a list where an int belongs is dropped here, once, right after `read_header`;
    `offload_kind`, `WindowReport` and `window_lines` then only ever see an int or nothing."""
    out = {k: v for k, v in raw.items() if k not in ("head_count_kv", "sliding_window_pattern") and _pos_int(v) is not None}
    pattern = raw.get("sliding_window_pattern")
    if _pos_int(pattern) is not None or (isinstance(pattern, list) and pattern and all(type(x) is bool for x in pattern)):
        out["sliding_window_pattern"] = pattern
    kv = raw.get("head_count_kv")
    if isinstance(kv, list):
        ints = [x for x in kv if _pos_int(x) is not None]
        if ints:
            out["head_count_kv"] = ints
    elif _pos_int(kv) is not None:
        out["head_count_kv"] = kv
    return out


def head_dim(header: Mapping[str, Any]) -> int | None:
    """`key_length`, else `embedding_length // head_count`."""
    key_length = _pos_int(header.get("key_length"))
    if key_length is not None:
        return key_length
    embedding, heads = _pos_int(header.get("embedding_length")), _pos_int(header.get("head_count"))
    return embedding // heads if embedding is not None and heads is not None else None


def kv_heads(header: Mapping[str, Any]) -> int | None:
    """`head_count_kv` as an int; a per-layer list takes its maximum."""
    value = header.get("head_count_kv")
    if isinstance(value, list):
        ints = [v for v in value if _pos_int(v) is not None]
        return max(ints) if ints else None
    return _pos_int(value)


def swa_pattern(header: Mapping[str, Any]) -> list[bool] | None:
    """Per layer, whether it slides: the header's bool list, or an int period n (all but every nth, as llama.cpp's `set_swa_pattern`); None otherwise."""
    n, pattern = _pos_int(header.get("block_count")), header.get("sliding_window_pattern")
    if n is None or _pos_int(header.get("sliding_window")) is None:
        return None
    if isinstance(pattern, list):
        return list(pattern) if len(pattern) == n and all(type(x) is bool for x in pattern) else None
    period = _pos_int(pattern)
    return [i % period < period - 1 for i in range(n)] if period is not None else None


@dataclass(frozen=True)
class TensorSplit:
    """The model's bytes by where they live, the layers that hold KV, the sliding ones among them and their KV heads."""

    gpu: int
    experts: int
    input: int
    kv_main: int
    kv_draft: int
    indexer: int
    swa: int = 0
    heads_full: int = 0
    heads_swa: int = 0


def read_split(model: str, header: Mapping[str, Any]) -> TensorSplit | None:
    """The model's tensor table split by `classify` and `kv_layers`, or None when unreadable or failing `cross_checked`."""
    from localharness.provider import gguf_header as gh

    try:
        table = gh.read_tensor_table(os.path.expanduser(model))
    except (OSError, ValueError):
        return None
    sums = {"gpu": 0, "experts": 0, "input": 0}
    for e in table.entries:
        sums[gh.classify(e.name)] += e.bytes
    if not any(e.name == "output.weight" for e in table.entries):
        # llama.cpp builds the output projection as a copy of token_embd on the output device (create_tensor ... TENSOR_DUPLICATED)
        sums["gpu"] += sum(e.bytes for e in table.entries if e.name == "token_embd.weight")
    main, draft = gh.kv_layers(table.entries, header)
    if not gh.cross_checked(sums, main, header, table.data_bytes):
        return None
    pattern, heads = swa_pattern(header), header.get("head_count_kv")
    ids = gh.kv_layer_ids(table.entries, header)
    sliding = [i for i in ids if pattern and pattern[i]]
    per_layer = bool(sliding) and isinstance(heads, list) and len(heads) == header["block_count"]
    heads_swa = sum(heads[i] for i in sliding) if per_layer else 0
    heads_full = sum(heads[i] for i in ids if not pattern[i]) if per_layer else 0
    return TensorSplit(sums["gpu"], sums["experts"], sums["input"], main, draft, gh.indexer_layers(table.entries),
                       len(sliding), heads_full, heads_swa)


def kv_bytes_per_token(header: Mapping[str, Any], cache_type_k: str | None, cache_type_v: str | None,
                       split: TensorSplit | None = None) -> float | None:
    """KV cache bytes per token over the split's full-attention layers (else every block); None for an unknown type or a missing header value."""
    k = KV_BYTES.get((cache_type_k or "f16").lower())
    v = KV_BYTES.get((cache_type_v or "f16").lower())
    n_layer = split.kv_main - split.swa if split else _pos_int(header.get("block_count"))
    heads, dim = kv_heads(header), head_dim(header)
    if k is None or v is None or not n_layer or heads is None or dim is None:
        return None
    indexer = split.indexer * (_pos_int(header.get("indexer.key_length")) or 0) * 2 * k if split else 0
    total = split.heads_full if split and split.heads_full else n_layer * heads
    return total * (dim * k + (_pos_int(header.get("value_length")) or dim) * v) + indexer


def swa_bytes_per_cell(header: Mapping[str, Any], cache_type_k: str | None, cache_type_v: str | None,
                       split: TensorSplit | None) -> float:
    """KV cache bytes of one cell of the sliding layers; 0 without any or for an unknown type or a missing header value."""
    k = KV_BYTES.get((cache_type_k or "f16").lower())
    v = KV_BYTES.get((cache_type_v or "f16").lower())
    heads, dim = kv_heads(header), head_dim(header)
    if split is None or not split.swa or k is None or v is None or heads is None or dim is None:
        return 0.0
    key = _pos_int(header.get("key_length_swa")) or dim
    value = _pos_int(header.get("value_length_swa")) or _pos_int(header.get("value_length")) or dim
    return (split.heads_swa or split.swa * heads) * (key * k + value * v)


_UBATCH = 512


def swa_cells_cap(n_swa: int, extra_args: Sequence[str]) -> int | None:
    """llama.cpp's sliding-window cache cells (unified 4 slots unless `-np`); None under `--swa-full`."""
    if _has_flag(extra_args, ("--swa-full",)):
        return None

    def number(flags: Sequence[str], default: int) -> int:
        raw = _flag_value(extra_args, flags)
        return int(raw) if raw is not None and raw.isdecimal() and int(raw) > 0 else default

    ubatch, parallel = number(("-ub", "--ubatch-size"), _UBATCH), number(("-np", "--parallel"), 0)

    def pad(n: int) -> int:
        return -(-n // 256) * 256

    if not parallel:
        return pad(n_swa * 4 + ubatch)
    return pad(n_swa * parallel + ubatch) if _has_flag(extra_args, ("-kvu", "--kv-unified")) else parallel * pad(n_swa + ubatch)


# Flags that place tensors by hand: how many layers sit on the GPU cannot be read from them.
_OFFLOAD_UNKNOWN = ("-ot", "--override-tensor", "--n-cpu-moe", "-ncmoe", "--cpu-moe")
_NGL = ("-ngl", "--n-gpu-layers", "--gpu-layers")


def offload_kind(extra_args: Sequence[str], n_layer: int | None) -> str:
    """"gpu" (weights fully GPU-resident), "partial" (`-ngl` below the layer count) or "unknown"
    (tensors placed by hand, or an `-ngl` that cannot be read). The fit is computed only for "gpu"."""
    kind = "gpu"
    for i, token in enumerate(extra_args):
        if any(token == flag or token.startswith(f"{flag}=") for flag in _OFFLOAD_UNKNOWN):
            return "unknown"
        for flag in _NGL:
            if token == flag:
                raw = extra_args[i + 1] if i + 1 < len(extra_args) else ""
            elif token.startswith(f"{flag}="):
                raw = token[len(flag) + 1:]
            else:
                continue
            try:
                n = int(raw)
            except ValueError:
                return "unknown"
            if n_layer is None:
                return "unknown"
            kind = "partial" if 0 <= n < n_layer else "gpu"
    return kind


def _size(path: Path) -> int:
    st = os.stat(path)
    if not stat.S_ISREG(st.st_mode):
        raise OSError(f"{path} is not a regular file")
    return st.st_size


def checkpoint_bytes(directory: str | os.PathLike[str]) -> int | None:
    """Bytes of one directory's `*.safetensors` files (a link is followed for its size only, as in a Hugging Face snapshot); stat only. None when there are none or the directory cannot be read."""
    try:
        sizes = [_size(p) for p in sorted(Path(directory).glob("*.safetensors")) if not p.is_dir()]
    except OSError:
        return None
    return sum(sizes) if sizes else None


def weights_bytes(model: str, mmproj: str | None = None) -> int | None:
    """Bytes of the GGUF, of its split shards (siblings named `<prefix>-0000N-of-0000M.gguf` when
    the model's own name has that shape) and of `mmproj`. Stat only, never an open. None when any
    of them is missing."""
    try:
        path = Path(os.path.expanduser(model))
        total = sum(_size(p) for p in shard_paths(path))
        if mmproj:
            total += _size(Path(os.path.expanduser(mmproj)))
    except (OSError, ValueError):
        return None
    return total


def fit(header: Mapping[str, Any], free_bytes: int | None, ratios: Any,
        attached_served: int | None = None, *, cache_type_k: str | None = None,
        cache_type_v: str | None = None, weights: int | None = None, offload: str = "gpu",
        runtime: str = "llamacpp", draft_share: float = 1.0, split: TensorSplit | None = None) -> int | None:
    """The largest window the free memory holds, capped at the trained window; None when it cannot
    be known or is not above zero. Beside a running server (`attached_served`) the window already
    served is counted as served plus what the free memory adds; otherwise the weights and a
    `compute_margin_ratio` of them come off the free bytes first.

    `draft_share` multiplies the KV bytes per token (1.0, the default, with drafting off)."""
    kv = kv_bytes_per_token(header, cache_type_k, cache_type_v, split)
    if kv is not None:
        kv = (kv + swa_bytes_per_cell(header, cache_type_k, cache_type_v, split)) * draft_share
    trained = _pos_int(header.get("context_length"))
    row = row_of(runtime)
    if row is None or not row.gguf or offload != "gpu" or free_bytes is None or kv is None or trained is None:
        return None
    if attached_served is not None:
        window = attached_served + int(free_bytes // kv)
    else:
        if weights is None:
            return None
        usable = free_bytes - weights - int(ratios.compute_margin_ratio * weights)
        if usable <= 0:
            return None
        window = int(usable // kv)
    window = min(trained, window)
    return window if window > 0 else None


def cache_ram_mib(mem_available: int | None, mem_total: int | None, weights: int | None,
                  offload: str, extra_args: Sequence[str], ratios: Any) -> int | None:
    """`--cache-ram` in MiB: `cache_ram_ratio` of the available RAM less host-resident weights,
    clamped at `cache_ram_ceiling_ratio` of the total. Weights are host-resident unless fully
    GPU-resident and mmapped. None when memory is unknown or the result is not above zero."""
    if mem_available is None or mem_total is None:
        return None
    host = (weights or 0) if offload != "gpu" or "--no-mmap" in extra_args else 0
    ram = ratios.cache_ram_ratio * (mem_available - host)
    mib = int(min(ram, ratios.cache_ram_ceiling_ratio * mem_total) // _MIB)
    return mib if mib > 0 else None


@dataclass(frozen=True)
class Placement:
    """Where a model's weights live and the window that leaves; `refusal` is the one sentence when nothing holds it."""

    fits: int | None
    where: str | None  # "gpu", "experts in RAM", "vllm" or None when nothing was priced
    gpu_bytes: int
    mmproj: int
    experts_ram: int
    kv_per_token: float | None
    free_bytes: int
    budget_bytes: int
    cache_ram: int | None
    refusal: str | None
    swa_bytes: int = 0


def _window(room: float, kv: float, trained: int, swa: float = 0.0, cap: int | None = None) -> int:
    """The largest multiple of 256 with `w × kv + min(w, cap) × swa` within `room`: the sliding layers' cells stop growing at `cap`."""
    if swa and cap is not None and (room - cap * swa) // kv >= cap:
        w = (room - cap * swa) // kv
    else:
        w = room // (kv + swa)
    return min(trained, max(0, int(w)) // 256 * 256)


def _swa_price(header: Mapping[str, Any], k: str | None, v: str | None, split: TensorSplit | None,
               extra_args: Sequence[str]) -> tuple[float, int | None]:
    """(bytes per sliding cell, the cells the cache holds); the cap is None for a cache the size of the window."""
    swa = swa_bytes_per_cell(header, k, v, split)
    return swa, swa_cells_cap(_pos_int(header.get("sliding_window")) or 0, extra_args) if swa else None


def _cache_ram(meminfo: Mapping[str, int] | None, host: int, ratios: Any) -> int | None:
    if meminfo is None:
        return None
    avail, total = meminfo["MemAvailable"], meminfo["MemTotal"]
    room = avail - ratios.unified_reserve_ratio * total
    cap = min(ratios.cache_ram_ratio * (avail - host), room - host, ratios.cache_ram_ceiling_ratio * total)
    return max(0, int(cap // _MIB))


VRAM_REMEDY = "stop what holds the GPU, or pick a smaller quant"
FLOOR_REMEDY = "free GPU memory, or pick a smaller model"
REMEDY_Q8 = "a q8_0 KV cache fits it (cache_type_k and cache_type_v: q8_0)"
REMEDY_MTP_OFF = "it fits with MTP drafting off (speculative: off)"
REMEDY_Q4 = "a q4_0 KV cache fits it (cache_type_k and cache_type_v: q4_0)"
VRAM_REFUSAL = ("{name} cannot run here: it needs {need} GiB of GPU memory{window} and {free} GiB is free — "
                "{remedy}.")
RAM_REFUSAL = ("{name} cannot run here: its experts need {need} GiB of RAM and only {free} GiB is free beside the "
               "system's reserve — close something that holds RAM, or pick a smaller quant.")
RAM_UNKNOWN_REFUSAL = ("{name} cannot run here: its experts need {need} GiB of RAM and the free RAM cannot be read "
                       "(/proc/meminfo) — pick a quant whose experts fit the GPU.")
NGL_REFUSAL = ("{name} sets -ngl in server.extra_args, which turns llama-server's --fit off, and its experts do not "
               "fit the GPU — drop the -ngl, or add --n-cpu-moe to extra_args.")
CONTEXT_FLOOR_REFUSAL = ("{name} is set to {window} ({where}) and the harness needs at least {floor:,} — raise it, "
                         "or unset it so the harness sizes the window.")
FLOOR_REFUSAL = "{name} fits only {window} here and the harness needs at least {floor:,} — {remedy}."


def window_phrase(n: int) -> str:
    """"a 4,096-token window", "an 8,448-token window": the article follows how the number is read."""
    lead = f"{n:,}".split(",")[0]
    return f"{'an' if lead in ('8', '11', '18') or lead.startswith('8') else 'a'} {n:,}-token window"


def _remedy(room: float, header: Mapping[str, Any], split: TensorSplit | None, k: str | None, v: str | None,
            share: float, trained: int, floor: int, default: str, extra_args: Sequence[str] = ()) -> str:
    """The cheapest change that reaches `floor`: a q8_0 cache (ΔPPL 0.0046 over f16, llama.cpp PR 7412), drafting off, a q4_0 cache."""
    need = max(floor, 1)

    def reaches(cache: str, factor: float) -> bool:
        kv = kv_bytes_per_token(header, cache, cache, split)
        swa, cap = _swa_price(header, cache, cache, split, extra_args)
        return kv is not None and _window(room, kv * factor, trained, swa * factor, cap) >= need

    above_q8 = any(KV_BYTES.get((t or "f16").lower(), 0) > KV_BYTES["q8_0"] for t in (k, v))
    at_q4 = all(KV_BYTES.get((t or "f16").lower(), 0) <= KV_BYTES["q4_0"] for t in (k, v))
    if above_q8 and reaches("q8_0", share):
        return REMEDY_Q8
    kv = kv_bytes_per_token(header, k, v, split)
    swa, cap = _swa_price(header, k, v, split, extra_args)
    if share > 1 and kv is not None and _window(room, kv, trained, swa, cap) >= need:
        return REMEDY_MTP_OFF
    if not at_q4 and reaches("q4_0", share):
        return REMEDY_Q4
    return default


def place_llamacpp(split: TensorSplit | None, header: Mapping[str, Any], *, name: str, weights: int, mmproj: int,
                   free_bytes: int, meminfo: Mapping[str, int] | None, cache_type_k: str | None,
                   cache_type_v: str | None, share: float, autosense: Any, floor: int | None = None,
                   ngl_hand: bool = False, no_mmap: bool = False, extra_args: Sequence[str] = ()) -> Placement:
    """Experts on the GPU first, in RAM when that misses the floor; the whole file on the GPU without a split."""
    gpu = split.gpu + mmproj if split else weights
    experts = split.experts if split else 0
    margin = int(autosense.compute_margin_ratio * gpu) + int(autosense.compute_reserve * 2**30)
    kv = kv_bytes_per_token(header, cache_type_k, cache_type_v, split)
    kv = kv * share if kv is not None else None
    swa, cap = _swa_price(header, cache_type_k, cache_type_v, split, extra_args)
    swa *= share
    trained = _pos_int(header.get("context_length"))
    label = "".join(c for c in name if c.isprintable())

    def made(fits: int, where: str | None, experts_ram: int = 0, refusal: str | None = None) -> Placement:
        cells = fits if cap is None else min(fits, cap)
        return Placement(fits or None, None if refusal else where, gpu, mmproj, experts_ram, kv, free_bytes,
                         free_bytes, _cache_ram(meminfo, weights if no_mmap else experts_ram, autosense), refusal,
                         int(swa * cells))

    def remedy(room: float, default: str) -> str:
        return _remedy(room, header, split, cache_type_k, cache_type_v, share, trained, floor or 0, default, extra_args)

    def decided(window: int, where: str, room: float, experts_ram: int = 0) -> Placement:
        if window <= 0:
            plus = f" plus {window_phrase(floor)}" if floor else ""
            return made(0, where, refusal=VRAM_REFUSAL.format(
                name=label, need=gib(gpu), window=plus, free=gib(free_bytes), remedy=remedy(room, VRAM_REMEDY)))
        if floor and window < floor:
            return made(window, where, refusal=FLOOR_REFUSAL.format(
                name=label, window=window_phrase(window), floor=floor, remedy=remedy(room, FLOOR_REMEDY)))
        return made(window, where, experts_ram)

    if ngl_hand and experts > 0 and gpu + experts > free_bytes:
        return made(0, None, refusal=NGL_REFUSAL.format(name=label))
    if kv is None or trained is None:
        return made(0, None)
    room = free_bytes - gpu - margin
    on_gpu = _window(room - experts, kv, trained, swa, cap)
    if on_gpu >= max(floor or 0, 1) or not (split and experts > 0 and _pos_int(header.get("expert_count"))) or ngl_hand:
        return decided(on_gpu, "gpu", room - experts)
    if meminfo is None:
        return made(0, None, refusal=RAM_UNKNOWN_REFUSAL.format(name=label, need=gib(experts)))
    ram_room = meminfo["MemAvailable"] - autosense.unified_reserve_ratio * meminfo["MemTotal"]
    if experts > ram_room:
        return made(0, None, refusal=RAM_REFUSAL.format(name=label, need=gib(experts), free=gib(max(0, ram_room))))
    return decided(_window(room, kv, trained, swa, cap), "experts in RAM", room, experts)


VLLM_REFUSAL = ("{name} cannot run here: {weights} GiB of {dtype}weights, and vLLM keeps them all on the {total} "
                "GiB GPU — {remedy}.")
VLLM_REMEDY_4BIT = "a 4-bit build of it (a GGUF Q4_K_M for llama.cpp, or NVFP4 for vLLM) would fit"
VLLM_REMEDY = "pick a smaller quant"
_Q4_K_M_SHARE = 0.31
_DTYPE_WORD = {"bfloat16": "bf16 ", "float16": "fp16 "}
_VLLM_GMU = 0.90
_VLLM_BATCHED = 2048


def _config_value(config: Mapping[str, Any], key: str) -> Any:
    """`key` from `text_config` when it holds one, else from the top level."""
    text = config.get("text_config")
    return text[key] if isinstance(text, Mapping) and key in text else config.get(key)


def place_vllm(model_config: Mapping[str, Any], *, name: str, weights: int, total_bytes: int,
               extra_args: Sequence[str], autosense: Any, floor: int | None = None) -> Placement:
    """A vLLM checkpoint's window from its parsed config.json, the stat-only weight sum and the GPU total."""
    raw = _flag_value(extra_args, ("--gpu-memory-utilization",))
    try:
        gmu = float(raw) if raw is not None else _VLLM_GMU
    except ValueError:
        gmu = _VLLM_GMU
    gmu = gmu if 0 < gmu <= 1 else _VLLM_GMU
    budget = int(gmu * total_bytes)
    margin = int(autosense.compute_margin_ratio * weights)
    label = "".join(c for c in name if c.isprintable())

    def made(fits: int, kv: int | None, refusal: str | None = None, swa_bytes: int = 0) -> Placement:
        return Placement(fits or None, None if refusal or not kv else "vllm", weights, 0, 0, kv, total_bytes,
                         budget, None, refusal, swa_bytes)

    if weights + margin >= budget:
        text = model_config.get("text_config")
        quantized = "quantization_config" in model_config or (isinstance(text, Mapping) and "quantization_config" in text)
        raw = _config_value(model_config, "dtype") or _config_value(model_config, "torch_dtype")
        dtype = "" if quantized else _DTYPE_WORD.get(raw, "") if isinstance(raw, str) else ""
        four_bit = int(_Q4_K_M_SHARE * weights)
        fits_4bit = four_bit + int(autosense.compute_margin_ratio * four_bit) < budget
        remedy = VLLM_REMEDY_4BIT if dtype and fits_4bit else VLLM_REMEDY
        return made(0, None, VLLM_REFUSAL.format(
            name=label, weights=gib(weights), dtype=dtype, total=gib(total_bytes), remedy=remedy))
    types = _config_value(model_config, "layer_types")
    layers = sum(t == "full_attention" for t in types) if isinstance(types, list) else 0
    sliding = sum(t == "sliding_attention" for t in types) if isinstance(types, list) else 0
    layers = layers or _pos_int(_config_value(model_config, "num_hidden_layers"))
    heads = _pos_int(_config_value(model_config, "num_key_value_heads"))
    dim = _pos_int(_config_value(model_config, "head_dim"))
    if dim is None:
        hidden, attn = (_pos_int(_config_value(model_config, k)) for k in ("hidden_size", "num_attention_heads"))
        dim = hidden // attn if hidden and attn else None
    if not (layers and heads and dim):
        return made(0, None)
    dtype = (_flag_value(extra_args, ("--kv-cache-dtype",)) or "").lower()
    elem = 1 if dtype.startswith("fp8") else 2
    kv = layers * 2 * (_pos_int(_config_value(model_config, "num_global_key_value_heads")) or heads) \
        * (_pos_int(_config_value(model_config, "global_head_dim")) or dim) * elem
    swa, cap = sliding * 2 * heads * dim * elem, None
    window_tokens = _pos_int(_config_value(model_config, "sliding_window"))
    if window_tokens is None:
        kv, swa = kv + swa, 0
    else:
        raw = _flag_value(extra_args, ("--max-num-batched-tokens",))
        cap = window_tokens + 2 * (int(raw) if raw is not None and raw.isdecimal() and int(raw) > 0 else _VLLM_BATCHED)
    trained = _pos_int(_config_value(model_config, "max_position_embeddings"))
    window = _window(budget - weights - margin, kv, trained or 1 << 40, swa, cap)
    swa_bytes = swa * (window if cap is None else min(window, cap))
    if floor and window < floor:
        return made(window, kv, FLOOR_REFUSAL.format(
            name=label, window=window_phrase(window), floor=floor, remedy=FLOOR_REMEDY), swa_bytes)
    return made(window, kv, swa_bytes=swa_bytes)


def _label(spec: Any) -> str:
    """The record's name as a sentence shows it: `served_name`, else the model's file name, printable characters only."""
    name = getattr(spec, "served_name", None) or Path(os.path.expanduser(str(spec.model))).name
    return "".join(c for c in name if c.isprintable())


def has_ctx_arg(extra_args: Sequence[str]) -> bool:
    """Whether `extra_args` already carry a context size (`-c`, `--ctx-size`, `--ctx-size=N`)."""
    return any(t in ("-c", "--ctx-size") or t.startswith("--ctx-size=") for t in extra_args)


def has_cache_ram_arg(extra_args: Sequence[str]) -> bool:
    """Whether `extra_args` already carry `--cache-ram` / `-cram` (also the `=` form)."""
    return any(t in ("--cache-ram", "-cram") or t.startswith("--cache-ram=") for t in extra_args)


def _flag_value(extra_args: Sequence[str], flags: Sequence[str]) -> str | None:
    """The value of the last of `flags` in `extra_args` (two-token or `=` form), or None."""
    found: str | None = None
    for i, token in enumerate(extra_args):
        for flag in flags:
            if token == flag and i + 1 < len(extra_args):
                found = extra_args[i + 1]
            elif token.startswith(f"{flag}="):
                found = token[len(flag) + 1:]
    return found


def extra_cache_types(extra_args: Sequence[str]) -> tuple[str | None, str | None]:
    """The KV cache types `extra_args` set: `(-ctk | --cache-type-k, -ctv | --cache-type-v)`, each
    None when absent. The record's own `cache_type_k` / `cache_type_v` win over these."""
    return (_flag_value(extra_args, ("-ctk", "--cache-type-k")),
            _flag_value(extra_args, ("-ctv", "--cache-type-v")))


def _has_flag(extra_args: Sequence[str], flags: Sequence[str]) -> bool:
    """Whether any of `flags` appears in `extra_args`, bare (value in the next token) or `flag=value`."""
    return any(t == f or t.startswith(f"{f}=") for t in extra_args for f in flags)


_CACHE_TYPE_FLAGS = ("-ctk", "-ctv", "--cache-type-k", "--cache-type-v")
_CACHE_TYPE_ENV = ("LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V")
_FLASH_ATTN_FLAGS = ("-fa", "--flash-attn")
_FLASH_ATTN_OFF_ENV = frozenset({"off", "0", "false", "disabled"})
_SPEC_FLAGS = ("--spec-type", "-md", "--model-draft", "-hfd", "--hf-repo-draft", "-hfrd")


def has_cache_type_flag(extra_args: Sequence[str], env: Mapping[str, str]) -> bool:
    """Whether the person already pinned a KV cache type: `-ctk` / `-ctv` / `--cache-type-k` /
    `--cache-type-v` (two-token or `=` form) in `extra_args`, or `LLAMA_ARG_CACHE_TYPE_K` / `_V` in `env`."""
    return _has_flag(extra_args, _CACHE_TYPE_FLAGS) or any(name in env for name in _CACHE_TYPE_ENV)


def flash_attn_off(extra_args: Sequence[str], env: Mapping[str, str]) -> bool:
    """Whether flash attention is switched off: `-fa` / `--flash-attn` (next token or `=` form) in
    `extra_args`, or `LLAMA_ARG_FLASH_ATTN` in `env`, set to off / 0 / false / disabled in any case
    (llama.cpp's falsey set, one table for both). A quantized V cache needs flash attention, so off
    suppresses the KV derivation."""
    value = _flag_value(extra_args, _FLASH_ATTN_FLAGS)
    if value is not None and value.strip().lower() in _FLASH_ATTN_OFF_ENV:
        return True
    raw = env.get("LLAMA_ARG_FLASH_ATTN")
    return raw is not None and str(raw).strip().lower() in _FLASH_ATTN_OFF_ENV


def has_spec_flag(extra_args: Sequence[str], env: Mapping[str, str]) -> bool:
    """Whether speculative decoding is already set by hand: `--spec-type`, `-md` / `--model-draft`,
    `-hfd` / `--hf-repo-draft` / `-hfrd` (two-token or `=` form) or any `--spec-draft-*` flag in
    `extra_args`, or any `LLAMA_ARG_SPEC_*` / `LLAMA_ARG_MODEL_DRAFT` name in `env`."""
    if _has_flag(extra_args, _SPEC_FLAGS) or any(t.startswith("--spec-draft-") for t in extra_args):
        return True
    return any(name.startswith("LLAMA_ARG_SPEC_") or name == "LLAMA_ARG_MODEL_DRAFT" for name in env)


# llama.cpp's config.ini grammar (common/preset.cpp): a trailing comment starts at a blank then `;`/`#`.
_INI_TRAILING_COMMENT = re.compile(r"[ \t][;#]")
# The flag spellings the checks above know, by their config.ini key (the flag with its dashes removed).
# Built from the tables above, so no second spelling list exists.
_INI_FLAG_BY_KEY = MappingProxyType({
    flag.lstrip("-").lower(): flag for flag in (*_CACHE_TYPE_FLAGS, *_FLASH_ATTN_FLAGS, *_SPEC_FLAGS)})


def config_ini_items(text: str | None) -> list[tuple[str, str]]:
    """The `key = value` lines of config.ini text, in order: blank lines, comments (`;`, `#`), section
    headers and lines without `=` are skipped, a trailing comment is cut from the value, and key and value
    are stripped. The one walker over a record's `config_ini`: its validator reads the keys, the
    derivation the keys and values. Pure, never raises."""
    items: list[tuple[str, str]] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line[0] in ";#[" or "=" not in line:
            continue
        key, value = line.split("=", 1)
        items.append((key.strip(), _INI_TRAILING_COMMENT.split(value, 1)[0].strip()))
    return items


def hand_set_options(spec: Any) -> tuple[list[str], dict[str, str]]:
    """Every llama-server option a llama.cpp record sets by hand, as `(argv, env)` for the checks above
    (D-06: record field > any explicit flag, in argv, env or config.ini > derivation).

    llama-server reads a config.ini key as an env name verbatim or as a flag name with its dashes removed,
    in any letter case. A `LLAMA_ARG_*` key therefore joins `env` (the record's own `env:` wins a name
    both set) and any other key joins argv as `--key=value` (`-fa=value` for the short spellings the
    tables above know); the config.ini tokens come before `extra_args`, so an argv flag wins a key both
    set, as at launch. Pure."""
    args: list[str] = []
    ini_env: dict[str, str] = {}
    for key, value in config_ini_items(getattr(spec, "config_ini", None)):
        if key.upper().startswith("LLAMA_ARG_"):
            ini_env[key.upper()] = value
        else:
            name = key.lower().lstrip("-")
            args.append(f"{_INI_FLAG_BY_KEY.get(name, f'--{name}')}={value}")
    args += [t for t in (getattr(spec, "extra_args", None) or []) if isinstance(t, str)]
    return args, {**ini_env, **dict(getattr(spec, "env", None) or {})}


def spec_type_flag(extra_args: Sequence[str], env: Mapping[str, str]) -> str | None:
    """The value of the last `--spec-type` in `extra_args`, else env `LLAMA_ARG_SPEC_TYPE`, else None
    (a hand-set `draft-mtp` drafts, so the fit must price it)."""
    value = _flag_value(extra_args, ("--spec-type",))
    if value is not None:
        return value
    raw = env.get("LLAMA_ARG_SPEC_TYPE")
    return str(raw) if raw is not None else None


_UNSET: Any = object()  # compute_fit's `memory=` default: probe free memory as before


@dataclass(frozen=True)
class FitInputs:
    """What `compute_fit` gathered for one llama.cpp record: the typed header, the GPU residency, the
    memory probes, the fit they give, the tensor `split` (None when unreadable or failing its
    cross-checks, then the whole file is priced) and the projector's bytes. Every field is None (or `{}`) when
    it could not be known."""

    header: dict[str, Any]
    offload: str
    memory: FreeMemory | None
    weights: int | None
    meminfo: dict[str, int] | None
    fits: int | None
    split: TensorSplit | None = None
    mmproj: int = 0


def compute_fit(server: Any, ratios: Any, *, attached_served: int | None = None, size: bool = True,
                meminfo: bool = False, memory: FreeMemory | None = _UNSET,
                cache_type: str | None = None, draft_share: float = 1.0,
                as_launched: bool = False) -> FitInputs:
    """The fit for a llama.cpp `server:` record, in one place for every caller (D-04): `activate`
    sizes a launch with it, `start` reports an attach and `doctor` prints its row.

    Reads the GGUF header (through `fit_header`, so every value is an int or absent), works out the
    GPU residency from `extra_args`, probes free memory and stats the weights, then calls `fit` with
    the KV cache types (the record's, else the ones `extra_args` set with `-ctk` / `-ctv`, else
    `cache_type`, the type a launch derives; unset is f16) and the `draft_share` of a drafting launch.
    `as_launched=True` prices exactly what a harness launch of the record runs with instead: the header
    just read and `ratios` (the machine's autosense knobs) go through `tuning.derive`, and
    `tuning.launch_pricing` names the KV types and the draft share (WR-06: the attach report and
    doctor's row agree with the launch; nothing is sensed beyond the header read here).
    `attached_served` is the window an already-running server serves.
    `size=False` stops after the header and the offload (no memory probe: a caller that has no served
    window to add to wants only the header); `meminfo=True` also reads `/proc/meminfo` for
    `--cache-ram`. `memory=` replaces the probe (D-49): `init` runs `free_memory` once and passes that
    one result for every model it ranks; a `FreeMemory` or None is used as given and `free_memory` is
    never called. Starts nothing and writes nothing."""
    from localharness.provider.gguf_header import read_header

    extra = list(server.extra_args or [])
    try:
        header = fit_header(read_header(os.path.expanduser(server.model), FIT_KEYS))
    except (OSError, ValueError):  # GgufHeaderError is a ValueError; NotRegularFile an OSError
        header = {}
    n_layer = header.get("block_count")
    offload = offload_kind(extra, n_layer) if n_layer else "unknown"
    if not size:
        return FitInputs(header, offload, None, None, None, None)
    mem = free_memory(ratios) if memory is _UNSET else memory
    weights = weights_bytes(server.model, server.mmproj)
    mmproj = (weights_bytes(server.mmproj) or 0) if server.mmproj else 0
    split = read_split(server.model, header) if header else None
    mi = _read_meminfo() if meminfo else None
    if as_launched:
        from localharness.provider import tuning  # function-local: tuning imports this module

        k, v, draft_share = tuning.launch_pricing(server, header, tuning.derive(server, header, ratios), split)
    else:
        extra_k, extra_v = extra_cache_types(extra)
        k, v = server.cache_type_k or extra_k or cache_type, server.cache_type_v or extra_v or cache_type
    row = row_of(server.runtime)
    if split is not None and attached_served is None and offload == "gpu" and mem is not None and weights is not None \
            and row is not None and row.gguf:
        ram = mi or (_read_meminfo() if split.experts else None)
        fits = place_llamacpp(
            split, header, name="", weights=weights, mmproj=mmproj, free_bytes=mem.free_bytes, meminfo=ram,
            cache_type_k=k, cache_type_v=v, share=draft_share, autosense=ratios, extra_args=extra).fits
    else:
        fits = fit(
            header, mem.free_bytes if mem else None, ratios, attached_served,
            cache_type_k=k, cache_type_v=v,
            weights=weights, offload=offload, runtime=server.runtime, draft_share=draft_share, split=split)
    return FitInputs(header, offload, mem, weights, mi, fits, split, mmproj)


@dataclass(frozen=True)
class WindowReport:
    """What a launch or an attach learned about the window, handed to `window_lines`."""

    served: int | None
    trained: int | None
    fits: int | None
    free_bytes: int | None
    total_bytes: int | None
    weights_bytes: int | None
    asked: int | None  # the -c sent, or None
    mode: str  # "fit", "record", "extra_args", "fit-by-server" or "not-computed"
    attached: bool = False  # attached_served was used: this is a report on a running server
    retried: bool = False
    verified: bool = True  # the probe answered
    runtime: str = "llamacpp"
    placement: Placement | None = None
    floor: int | None = None


@dataclass(frozen=True)
class Price:
    report: WindowReport | None
    context: int | None
    cache_ram: int | None
    full_fit: int | None
    refusal: str | None = None
    fit_ctx: int | None = None


def price_llamacpp(spec: ManagedServerConfig, sensed: Sensed, tuned: Tuned, autosense: Any,
                   floor: int | None = None) -> Price:
    """The plan's `-c` and `--cache-ram`, each only where neither the record nor `extra_args` set one; a placement that fits nowhere is the refusal."""
    from localharness.provider import tuning

    extra = list(spec.extra_args or [])
    header, mem, weights, mi, split = sensed.header, sensed.memory, sensed.weights, sensed.meminfo, sensed.split
    k, v, share = tuning.launch_pricing(spec, header, tuned, split)
    has_c = has_ctx_arg(extra)
    chosen = spec.context is not None or has_c
    by_hand = offload_kind(extra, header.get("block_count") or 1) != "gpu"
    placed = None
    if not by_hand and mem is not None and (split is not None or weights is not None):
        placed = place_llamacpp(
            split, header, name=_label(spec), weights=weights or 0, mmproj=sensed.mmproj, free_bytes=mem.free_bytes,
            meminfo=mi, cache_type_k=k, cache_type_v=v, share=share, autosense=autosense,
            floor=None if chosen else floor, ngl_hand=_flag_value(extra, _NGL) is not None,
            no_mmap="--no-mmap" in extra, extra_args=extra)
    fits = placed.fits if placed else None
    refusal = fit_ctx = None
    if spec.context is not None:
        mode, asked, ctx_kw = "record", spec.context, None
        if floor and spec.context < floor:
            refusal = CONTEXT_FLOOR_REFUSAL.format(name=_label(spec), window=window_phrase(spec.context), where="context:", floor=floor)
    elif has_c:
        mode, asked, ctx_kw = "extra_args", None, None
        sized = _flag_value(extra, ("-c", "--ctx-size"))
        if floor and sized and sized.isdecimal() and 0 < int(sized) < floor:
            refusal = CONTEXT_FLOOR_REFUSAL.format(
                name=_label(spec), window=window_phrase(int(sized)), where="-c in extra_args", floor=floor)
    elif by_hand or mem is None:
        mode, asked, ctx_kw = "not-computed", None, None
    elif placed is not None and placed.refusal is not None:
        mode, asked, ctx_kw = "fit", None, None
    elif fits:
        mode, asked, ctx_kw = "fit", fits, fits
    else:
        mode, asked, ctx_kw, fit_ctx = "fit-by-server", None, None, floor
    refusal = refusal or (placed.refusal if placed else None)
    cache_set = spec.cache_ram is not None or has_cache_ram_arg(extra)
    if cache_set:
        cache_kw = None
    elif placed is not None and placed.refusal is None:
        cache_kw = placed.cache_ram
    else:
        cache_kw = cache_ram_mib(
            mi["MemAvailable"] if mi else None, mi["MemTotal"] if mi else None, weights, sensed.offload, extra, autosense)
    report = WindowReport(
        served=None, trained=header.get("context_length"), fits=fits,
        free_bytes=mem.free_bytes if mem else None, total_bytes=mem.total_bytes if mem else None,
        weights_bytes=weights, asked=asked, mode=mode, attached=False, retried=False,
        verified=False, runtime="llamacpp", placement=placed, floor=floor)
    return Price(report, ctx_kw, cache_kw, fits if mode == "fit" else None, refusal, fit_ctx)


def price_vllm(spec: ManagedServerConfig, sensed: Sensed, tuned: Tuned, autosense: Any,
               floor: int | None = None) -> Price:
    """`--max-model-len` from the checkpoint's config.json and safetensors bytes; a checkpoint beyond the GPU is the refusal."""
    entry = spec.entry_for(spec.model)
    extra = [*(spec.extra_args or []), *(entry.extra_args if entry is not None else [])]
    mem, config, weights = sensed.memory, sensed.model_config, sensed.weights
    chosen = spec.context is not None or _flag_value(extra, ("--max-model-len",)) is not None
    placed = None
    if mem is not None and config is not None and weights is not None:
        placed = place_vllm(config, name=_label(spec), weights=weights, total_bytes=mem.total_bytes,
                            extra_args=extra, autosense=autosense, floor=None if chosen else floor)
    fits = placed.fits if placed else None
    refusal = None
    if spec.context is not None:
        mode, asked, ctx_kw = "record", spec.context, None
        if floor and spec.context < floor:
            refusal = CONTEXT_FLOOR_REFUSAL.format(name=_label(spec), window=window_phrase(spec.context), where="context:", floor=floor)
    elif chosen:
        mode, asked, ctx_kw = "extra_args", None, None
    elif placed is not None and placed.refusal is not None:
        mode, asked, ctx_kw = "fit", None, None
    elif fits:
        mode, asked, ctx_kw = "fit", fits, fits
    else:
        return Price(None, None, None, None)
    refusal = refusal or (placed.refusal if placed else None)
    trained = _pos_int(_config_value(config, "max_position_embeddings")) if config is not None else None
    report = WindowReport(
        served=None, trained=trained, fits=fits, free_bytes=mem.free_bytes if mem else None,
        total_bytes=mem.total_bytes if mem else None, weights_bytes=weights, asked=asked, mode=mode,
        verified=False, runtime="vllm", placement=placed, floor=floor)
    return Price(report, ctx_kw, None, fits if mode == "fit" else None, refusal)


def gib(n: float) -> str:
    return f"{n / 2**30:.1f}"


def fmt_count(value: int | None) -> str:
    """A window figure with thousands separators, or an em dash for one that is not known."""
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "—"


_UNDO = {
    "record": "  → unset `server.context` to relaunch with -c {fits}",
    "extra_args": "  → drop the -c from `server.extra_args` to relaunch with -c {fits}",
    "hand": "  → the next harness launch uses -c {fits}",
    "no-record": "  → run `localharness init` to let the harness size the window",
}


def _placement_line(r: WindowReport) -> str | None:
    """The launch line for a priced fit: the window, where the weights live and the GPU's share of it."""
    p = r.placement
    window = r.served if r.verified and r.served else r.asked
    if r.attached or r.mode != "fit" or p is None or p.where is None or not window or p.kv_per_token is None:
        return None
    share = gib(p.gpu_bytes + p.kv_per_token * window + p.swa_bytes)
    if p.where == "vllm":
        return f"window: {window:,} · vLLM · all on the GPU ({share} of vLLM's {gib(p.budget_bytes)} GiB)"
    if p.where == "experts in RAM":
        resident = "dense, projector and KV" if p.mmproj else "dense and KV"
        where = f"experts in RAM ({gib(p.experts_ram)} GiB) · {resident} on the GPU"
    else:
        where = "all on the GPU"
    return f"window: {window:,} · {where} ({share} of {gib(p.free_bytes)} GiB free)"


def _fit_line(r: WindowReport) -> str:
    line = f"window: served {fmt_count(r.served)} · trained {fmt_count(r.trained)} · fits {fmt_count(r.fits)}"
    if r.free_bytes is not None and r.total_bytes is not None and r.weights_bytes is not None:
        beside = " beside the served model" if r.attached else ""
        line += f" (free {gib(r.free_bytes)} GiB of {gib(r.total_bytes)}{beside}; weights {gib(r.weights_bytes)} GiB)"
    return line


def window_lines(report: WindowReport, ratios: Any, undo: str | None = None) -> list[str]:
    """The window line(s) for one report; empty when there is nothing worth saying.

    A launch report (`attached=False`) is always exactly one line and never gets a second, whatever
    `undo` says; a launch is printed always, so the on-change gate does not apply to it. Line 2
    (how to undo the harness's choice) exists only for an attach report given an `undo` in
    {"record", "extra_args", "hand", "no-record"}; an attach report with `undo=None` is line 1
    alone. An attach report with a fit or served window unknown, or whose fit is within
    `notify_ratio` of what is served, says nothing."""
    r = report
    row = row_of(r.runtime)
    if row is not None and row.window_note is not None:
        return [f"window: served {fmt_count(r.served)} · trained {fmt_count(r.trained)}{row.window_note}"]
    if r.mode == "fit-by-server":
        floor = f" (at least {r.floor:,})" if r.floor else ""
        return [FIT_BY_SERVER_LINE + floor + (f" · served {r.served:,}" if r.served and r.verified else "")]
    if r.attached and (r.fits is None or r.served is None
                       or abs(r.fits - r.served) <= ratios.notify_ratio * r.served):
        return []
    line = _placement_line(r) or _fit_line(r)
    if r.asked and r.served and r.served < r.asked:
        line += f" (asked {r.asked:,})"
    if not r.verified:
        line += " (not verified)"
    if r.retried and r.asked:
        line += f" (after one retry at -c {r.asked:,})"
    if r.attached and undo in _UNDO:
        return [line, _UNDO[undo].format(fits=r.fits)]
    return [line]


def window_changed(last: WindowRecord | None, report: WindowReport, notify_ratio: float) -> bool:
    """Whether the window differs from the last recorded one: no record, a served or trained
    difference, a fit known against unknown, or a fit that moved by more than `notify_ratio` of
    the recorded fit."""
    if last is None or last.served != report.served or last.trained != report.trained:
        return True
    if last.fits is None or report.fits is None:
        return last.fits is not report.fits
    return abs(report.fits - last.fits) > notify_ratio * last.fits
