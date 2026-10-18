"""Measured decode-speed ledger — rolling tok/s samples per (provider_type, model).

Records VERIFIED decode rates only: an exact completion-token count over the measured
first-token→last-token wall window of a real streamed completion, and only when that
window has enough substance to be a measurement (is_substantive_sample). No estimate ever
enters the file — a model with no recorded sample shows no number anywhere (hard rule:
no placeholder data). The display value is the median of the last SAMPLE_CAP samples:
robust to a one-off stall while still tracking real drift (decode slows as the KV
cache grows, so samples legitimately spread).

Storage: one small JSON file in the harness state dir, written atomically (a PER-WRITER tmp
+ os.replace — a shared tmp name is not atomic across processes). A missing or empty file reads
as empty and is written; a corrupt, oversized or otherwise unreadable one reads as empty and is
never written over (every writer skips it with one warning naming the path, WR-03). Concurrent
sessions may race the file — last writer wins, acceptable for a QOL stat.

The same entry also carries additive fields written by other call sites (window_report,
model_path, the launch `retry` record and the `sampling` record), and reserved top-level keys hold
per-runtime dropped env names (`_dropped_env`) and the last-served record name (`_last_served`). They
share this one file and this one write recipe; no second file exists (AUTO-08). `rejected_effort_keys`
is the AUTO-08 rejected-effort-key field 02-CONTEXT D-24 deferred: the reasoning-effort request keys a
server answered with a 400, per (provider_type, model).

The file is read through `open_regular` under LEDGER_CAP: it sits where agent-run code can write, so a
FIFO, a directory, a device symlink, an oversized file or pathological JSON reads as empty.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median
from typing import NamedTuple
from uuid import uuid4

import structlog

from localharness.core.private_files import open_regular, touch_private

log = structlog.get_logger(__name__)

SAMPLE_CAP = 10
# #130: the ceiling above which a "measured" rate is a timing artifact, not a decode speed.
# decode_tps divides an exact token count by the measured window, so a sub-millisecond window
# yields tens of thousands of tok/s — no local runtime decodes anywhere near this.
MAX_PLAUSIBLE_TPS = 10_000.0
# #136: the ceiling alone was not enough — 1,715..6,788 tok/s samples were recorded on a ~78 tok/s
# vLLM config (one session's display median reached 152.4). A CLIENT-measured sample times chunk
# ARRIVAL, first payload delta → done, so any burst whose deltas land coalesced — the shape the
# short agentic generations between tool calls produced there — measures transport, not decoding.
# Admission therefore needs substance, not just a plausible rate.
MIN_SAMPLE_TOKENS = 16    # >=15 intervals, so no single coalesced or stalled delta sets the rate
MIN_SAMPLE_SECONDS = 0.25  # ~100x local chunk-arrival jitter (single-digit ms) — a window this
                           # long cannot be jitter; every observed artifact implies one under 10ms
# Color bands (owner spec 2026-08-11): >30 green, 20–30 yellow, <20 red.
GREEN_ABOVE_TPS = 30.0
YELLOW_FROM_TPS = 20.0

_LEDGER_NAME = "speed_stats.json"
# Reserved top-level key (leading underscore): a per-runtime map of the env names a launch dropped.
# It holds no samples, so the speed readers skip it; model entries are keyed "provider:model".
DROPPED_ENV_KEY = "_dropped_env"
# Reserved top-level key: the name of the record `start` last served, for spec 20 §8.2's resume
# clause. Not a model entry, so the speed readers skip it.
LAST_SERVED_KEY = "_last_served"
FOREIGN_UNIT_KEY = "_foreign_unit"
STRATA_OFFER_KEY = "_strata_offer"
RESERVED_KEYS = (DROPPED_ENV_KEY, LAST_SERVED_KEY, FOREIGN_UNIT_KEY, STRATA_OFFER_KEY)
# The largest ledger file `_load` will read. A real ledger is a handful of model entries (a few
# KiB); 1 MiB is far above that and far below anything that costs memory or time to parse.
LEDGER_CAP = 1 << 20


def default_speed_stats_path(config_dir: "str | Path | None" = None) -> Path:
    """The ledger's home under the active config dir (same #35 precedence chain as every
    other config-dir-relative path: explicit arg > LOCALHARNESS_DIR > legacy > ~)."""
    from localharness.config.paths import resolve_config_dir

    return resolve_config_dir(config_dir) / _LEDGER_NAME


def tps_band(tps: float) -> str:
    """Rich color name for a decode rate: 'green' | 'yellow' | 'red'."""
    return "green" if tps > GREEN_ABOVE_TPS else "yellow" if tps >= YELLOW_FROM_TPS else "red"


def decode_tps(completion_tokens: int, first_token_at: float, done_at: float) -> float | None:
    """Verified decode rate: token intervals per second after the first token landed.

    None when the sample cannot be measured honestly — fewer than two tokens (no
    interval exists) or a degenerate window.
    """
    window = done_at - first_token_at
    if completion_tokens < 2 or window <= 0:
        return None
    return (completion_tokens - 1) / window


def is_substantive_sample(
    completion_tokens: int | None, window_seconds: float | None
) -> bool:
    """Does a CLIENT-measured sample carry enough substance to be a decode measurement (#136)?

    Both floors, because the artifact appears in both directions: a burst flushed inside a few
    milliseconds states an impossible rate, and a two-token sample states whatever one stall did.
    Missing evidence is never substance. Engine-REPORTED rates (llama.cpp
    timings.predicted_per_second) skip this — they are measured inside the decode loop, where
    chunk-arrival timing cannot reach them.
    """
    if not completion_tokens or window_seconds is None:
        return False
    return completion_tokens >= MIN_SAMPLE_TOKENS and window_seconds >= MIN_SAMPLE_SECONDS


def speed_key(provider_type: str, model: str) -> str:
    return f"{provider_type}:{model}"


def _read(path: Path) -> tuple[dict | None, str | None]:
    """`(ledger, None)` for a ledger that reads, or is absent or empty (`{}`); `(None, reason)` for one
    that is present but refused. The file sits where agent-run code can write, so it is opened through
    `open_regular` (a FIFO, directory or device symlink is refused without blocking) and read at most
    LEDGER_CAP + 1 bytes: a longer file is not read at all. Never raises."""
    try:
        with os.fdopen(open_regular(path), "rb") as fh:
            raw = fh.read(LEDGER_CAP + 1)
    except FileNotFoundError:
        return {}, None
    except OSError as exc:
        return None, f"not readable ({exc.strerror or type(exc).__name__})"
    if len(raw) > LEDGER_CAP:
        return None, f"larger than {LEDGER_CAP} bytes"
    if not raw.strip():
        return {}, None
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        return None, "not valid JSON"
    return (data, None) if isinstance(data, dict) else (None, "not a JSON object")


def _load(path: Path) -> dict:
    """The ledger as a dict for a reader, `{}` for anything absent or unreadable."""
    data, _reason = _read(path)
    return {} if data is None else data


def _load_for_write(path: Path) -> dict | None:
    """The ledger as a dict for a writer, or None when the file is present but unreadable (over the cap,
    not valid JSON, not a regular file, not readable). A writer then writes nothing: replacing the file
    would silently drop every entry it holds (WR-03). One structured warning names the path."""
    data, reason = _read(path)
    if data is None:
        log.warning("speed_ledger_unreadable", path=str(path), reason=reason, action="not written")
    return data


def _write(path: Path, data: dict) -> None:
    """The ledger's one write recipe: a PER-WRITER tmp, then os.replace.

    os.replace is atomic, but open+truncate+write into a name SHARED by every session is not —
    concurrent harnesses either published a byte-mix (unreadable ledger, every model's history
    dropped) or lost the race entirely (the other writer's replace already consumed the file →
    FileNotFoundError). Unique name ⇒ genuine last-writer-wins."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid4().hex[:8]}.tmp")
    try:
        touch_private(tmp)  # the replaced ledger keeps this one's mode
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace; no litter on failure


def _update_entry(path: Path, key: str, **fields) -> None:
    """UPDATE one model's entry with `fields`, never replace it (AUTO-08).

    A wholesale `data[key] = {...}` drops every other field on that entry — it silently ate the
    `tokens_per_chunk` written microseconds earlier in the same _note_gen_speed call, so the ratio
    never survived to seed the next session. Every writer of a model's entry goes through here."""
    data = _load_for_write(path)
    if data is None:
        return
    entry = data.get(key)
    entry = entry if isinstance(entry, dict) else {}
    entry.update(fields)
    entry.setdefault("samples", [])
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    data[key] = entry
    _write(path, data)


def record_tps(path: Path, provider_type: str, model: str, tps: float) -> None:
    """Append one verified sample, keeping the newest SAMPLE_CAP per model.

    #130: the file defends itself. A non-finite (nan/inf) or out-of-range value is DROPPED
    rather than stored — nan poisons this model's median forever, and an implausible rate is a
    degenerate-window artifact. The ledger's contract is verified samples only."""
    if not isfinite(tps) or tps <= 0 or tps > MAX_PLAUSIBLE_TPS:
        return
    data = _load(path)
    key = speed_key(provider_type, model)
    entry = data.get(key)
    samples = entry.get("samples", []) if isinstance(entry, dict) else []
    samples = [s for s in samples if isinstance(s, (int, float))][-(SAMPLE_CAP - 1):]
    samples.append(round(tps, 2))
    # UPDATE the entry, never replace it. A wholesale `data[key] = {...}` drops every other
    # field on that entry — it silently ate the `tokens_per_chunk` written microseconds earlier
    # in the same _note_gen_speed call, so the ratio never survived to seed the next session.
    _update_entry(path, key, samples=samples)


def record_tokens_per_chunk(path: Path, provider_type: str, model: str, ratio: float) -> None:
    """Persist how many tokens a streamed delta carries for this model (>=1.0).

    Stored on the SAME ledger entry as the tps samples, keyed by provider:model, because it is
    a property of that model+runtime pairing — a spec-decode acceptance length, not a per-session
    accident. Persisting it is what lets the FIRST stream of a NEW session show an honest live
    rate; without it every session's opening turn under speculative decoding reads low by the
    acceptance factor, which is exactly the number a user forms their first impression from.
    Additive field: older readers ignore it, and a ledger written before this existed still loads.
    """
    if not isfinite(ratio) or ratio < 1.0:
        return
    _update_entry(path, speed_key(provider_type, model), tokens_per_chunk=round(float(ratio), 3))


def tokens_per_chunk(path: Path, provider_type: str, model: str) -> float | None:
    """Recorded tokens-per-delta for this model, or None when never measured.

    None means "no evidence" and callers must NOT substitute 1.0 — on a spec-decode runtime that
    guess understates the rate threefold. No number beats a wrong number.
    """
    entry = _load(path).get(speed_key(provider_type, model))
    if not isinstance(entry, dict):
        return None
    r = entry.get("tokens_per_chunk")
    return float(r) if isinstance(r, (int, float)) and isfinite(r) and r >= 1.0 else None


class WindowRecord(NamedTuple):
    """The last window the harness saw for one model: what was served, what it was trained for,
    and what the fit settled on (each None when it was never known), and when it was recorded."""

    served: int | None
    trained: int | None
    fits: int | None
    recorded_at: str


def record_window_report(
    path: Path, provider_type: str, model: str,
    served: int | None, trained: int | None, fits: int | None,
) -> None:
    """Record the window triple on this model's entry (idempotent: the same triple again only
    moves `recorded_at`). `start` compares the next probe against it to print a change once."""
    _update_entry(path, speed_key(provider_type, model), window_report={
        "served": served, "trained": trained, "fits": fits,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })


def last_window_report(path: Path, provider_type: str, model: str) -> WindowRecord | None:
    """The recorded window triple, or None when never recorded or hand-edited out of shape."""
    entry = _load(path).get(speed_key(provider_type, model))
    rep = entry.get("window_report") if isinstance(entry, dict) else None
    if not isinstance(rep, dict):
        return None
    nums = [rep.get(k) for k in ("served", "trained", "fits")]
    if any(n is not None and (isinstance(n, bool) or not isinstance(n, int)) for n in nums):
        return None
    at = rep.get("recorded_at")
    return WindowRecord(nums[0], nums[1], nums[2], at) if isinstance(at, str) else None


def record_served_here(path: Path, key: str) -> None:
    """Stamp the entry `key` (a `speed_key`) with the time a launch of it answered on this box."""
    _update_entry(path, key, served_here_at=datetime.now(timezone.utc).isoformat())


def served_here(path: Path, key: str) -> bool:
    """Did a launch of the entry `key` answer here: a `served_here_at` stamp or a recorded window report."""
    entry = _load(path).get(key)
    return isinstance(entry, dict) and (isinstance(entry.get("served_here_at"), str) or isinstance(entry.get("window_report"), dict))


REJECTED_EFFORT_KEYS_FIELD = "rejected_effort_keys"


def record_rejected_effort_keys(
    path: Path, provider_type: str, model: str, keys: Iterable[str],
) -> None:
    """Record, on this model's entry, the effort request keys its server rejected (sorted,
    de-duplicated). Replaces the list; an empty `keys` stores `[]`. Idempotent: the same keys again
    only moves `updated_at`. I/O errors propagate — the caller decides whether to swallow them."""
    _update_entry(path, speed_key(provider_type, model),
                  **{REJECTED_EFFORT_KEYS_FIELD: sorted(set(keys))})


def last_rejected_effort_keys(path: Path, provider_type: str, model: str) -> list[str] | None:
    """The recorded rejected effort keys, or None when never recorded (or hand-edited out of shape)."""
    entry = _load(path).get(speed_key(provider_type, model))
    keys = entry.get(REJECTED_EFFORT_KEYS_FIELD) if isinstance(entry, dict) else None
    return list(keys) if isinstance(keys, list) and all(isinstance(k, str) for k in keys) else None


def record_model_path(path: Path, provider_type: str, model: str, model_path: str) -> None:
    """Record the weights file this model was last launched or measured from."""
    _update_entry(path, speed_key(provider_type, model), model_path=str(model_path))


def model_path(path: Path, provider_type: str, model: str) -> str | None:
    """The recorded weights path, or None when never recorded (or not a string)."""
    entry = _load(path).get(speed_key(provider_type, model))
    value = entry.get("model_path") if isinstance(entry, dict) else None
    return value if isinstance(value, str) else None


def model_paths(path: Path) -> set[str]:
    """Every weights path the ledger recorded, across all entries (INIT-02 ranks on these). The
    reserved `_dropped_env` map and any entry without a string `model_path` are skipped."""
    found: set[str] = set()
    for key, entry in _load(path).items():
        reserved = key in RESERVED_KEYS
        value = entry.get("model_path") if not reserved and isinstance(entry, dict) else None
        if isinstance(value, str):
            found.add(value)
    return found


def record_dropped_env(path: Path, runtime: str, names: Iterable[str]) -> None:
    """Record, per runtime, the env names the last launch dropped (sorted, de-duplicated).

    Lives under the reserved top-level DROPPED_ENV_KEY, not on a model entry: a dropped name is a
    property of the launch environment. Other runtimes' lists and every model entry are kept."""
    data = _load_for_write(path)
    if data is None:
        return
    dropped = data.get(DROPPED_ENV_KEY)
    dropped = dropped if isinstance(dropped, dict) else {}
    dropped[runtime] = sorted(set(names))
    data[DROPPED_ENV_KEY] = dropped
    _write(path, data)


def last_dropped_env(path: Path, runtime: str) -> list[str] | None:
    """The env names the last launch of `runtime` dropped, or None when never recorded."""
    dropped = _load(path).get(DROPPED_ENV_KEY)
    names = dropped.get(runtime) if isinstance(dropped, dict) else None
    return list(names) if isinstance(names, list) and all(isinstance(n, str) for n in names) else None


def median_tps(path: Path, provider_type: str, model: str) -> float | None:
    """Median of the recorded samples for one model, or None if none exist."""
    entry = _load(path).get(speed_key(provider_type, model))
    samples = entry.get("samples", []) if isinstance(entry, dict) else []
    samples = [s for s in samples if isinstance(s, (int, float))]
    return median(samples) if samples else None


def all_median_tps(path: Path) -> dict[str, float]:
    """{speed_key: median tok/s} for every model with at least one sample.

    Reserved keys (leading underscore, RESERVED_KEYS) are not model entries
    and are skipped by name."""
    out: dict[str, float] = {}
    for key, entry in _load(path).items():
        if key in RESERVED_KEYS:
            continue
        samples = entry.get("samples", []) if isinstance(entry, dict) else []
        samples = [s for s in samples if isinstance(s, (int, float))]
        if samples:
            out[key] = median(samples)
    return out


# The only thing a launch retry may remember dropping. A hand-edited ledger can therefore move the
# next launch's threshold only as far as the spec allows (spec 20 §6.4); anything else reads absent.
RETRY_DROPPABLE = frozenset({"draft-mtp"})


class RetryRecord(NamedTuple):
    """What the last launch of one record settled on after a failed full-fit attempt (spec 20 §6.4):
    the fit it measured, the free memory it saw, what it dropped, the binary and the tuning knobs it
    launched with, and when (ISO-8601 UTC)."""

    full_fit: int
    free_bytes: int | None
    dropped: tuple[str, ...]
    binary: str
    kv_cache_type: str
    draft_p_min: float
    at: str


def record_retry(
    path: Path, record_key: str, *, full_fit: int, free_bytes: int | None,
    dropped: Iterable[str], binary: str, kv_cache_type: str, draft_p_min: float,
) -> None:
    """Record the retry on `record_key`'s entry (the `speed_key(runtime, served id)` form).

    Its only production caller is `lifecycle.confirm_launch`, the one writer of the retry record.
    A `dropped` outside RETRY_DROPPABLE raises ValueError and writes nothing."""
    drop = sorted(set(dropped))
    if not set(drop) <= RETRY_DROPPABLE:
        raise ValueError(f"retry may only record dropping {sorted(RETRY_DROPPABLE)}, got {drop}")
    _update_entry(path, record_key, retry={
        "full_fit": full_fit, "free_bytes": free_bytes, "dropped": drop, "binary": binary,
        "kv_cache_type": kv_cache_type, "draft_p_min": draft_p_min,
        "at": datetime.now(timezone.utc).isoformat(),
    })


def _plain_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def last_retry(path: Path, record_key: str) -> RetryRecord | None:
    """The recorded retry, or None when never recorded or hand-edited out of shape (a non-positive
    or bool `full_fit`, a `dropped` outside RETRY_DROPPABLE, any wrong type)."""
    entry = _load(path).get(record_key)
    rec = entry.get("retry") if isinstance(entry, dict) else None
    if not isinstance(rec, dict):
        return None
    full_fit, free, dropped = rec.get("full_fit"), rec.get("free_bytes"), rec.get("dropped")
    p_min = rec.get("draft_p_min")
    if not _plain_int(full_fit) or full_fit <= 0:
        return None
    if free is not None and (not _plain_int(free) or free < 0):
        return None
    if isinstance(p_min, bool) or not isinstance(p_min, (int, float)) or not 0 <= p_min <= 1:
        return None
    if not isinstance(dropped, list) or not all(isinstance(d, str) for d in dropped):
        return None
    if not set(dropped) <= RETRY_DROPPABLE:
        return None
    strs = [rec.get(k) for k in ("binary", "kv_cache_type", "at")]
    if not all(isinstance(v, str) for v in strs):
        return None
    return RetryRecord(full_fit, free, tuple(dropped), strs[0], strs[1], float(p_min), strs[2])


def clear_retry(path: Path, record_key: str) -> None:
    """Drop the retry field from `record_key`'s entry, keeping every other field. Writes nothing
    when there is no retry to drop. Its only production caller is `lifecycle.confirm_launch`."""
    data = _load_for_write(path)
    entry = data.get(record_key) if data is not None else None
    if not isinstance(entry, dict) or "retry" not in entry:
        return
    del entry["retry"]
    _write(path, data)


class SamplingRecord(NamedTuple):
    """The sampling line `start` last printed for one model: the temperature it applied (None when
    the server's own default stood), the rung of the ladder that chose it, the server default seen
    (each None when unknown), and when it was recorded (ISO-8601 UTC)."""

    temperature: float | None
    rung: str | None
    server_default: float | None
    recorded_at: str


def record_sampling(
    path: Path, provider_type: str, model: str, *,
    temperature: float | None, rung: str | None, server_default: float | None,
) -> None:
    """Record the sampling decision on this model's entry. Read by `start`'s sampling line, which
    prints a change once, the way the window report does."""
    _update_entry(path, speed_key(provider_type, model), sampling={
        "temperature": temperature, "rung": rung, "server_default": server_default,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    })


def _number_or_none(v: object) -> bool:
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v))


def last_sampling(path: Path, provider_type: str, model: str) -> SamplingRecord | None:
    """The recorded sampling decision, or None when never recorded or hand-edited out of shape."""
    entry = _load(path).get(speed_key(provider_type, model))
    rec = entry.get("sampling") if isinstance(entry, dict) else None
    if not isinstance(rec, dict):
        return None
    temp, rung, default, at = (rec.get(k) for k in ("temperature", "rung", "server_default",
                                                    "recorded_at"))
    if not (_number_or_none(temp) and _number_or_none(default)):
        return None
    if (rung is not None and not isinstance(rung, str)) or not isinstance(at, str):
        return None
    return SamplingRecord(None if temp is None else float(temp), rung,
                          None if default is None else float(default), at)


def record_last_served(path: Path, name: str) -> None:
    """Record the record name last served, under the reserved top-level LAST_SERVED_KEY. Read by
    spec 20 §8.2's resume clause through `last_served`."""
    data = _load_for_write(path)
    if data is None:
        return
    data[LAST_SERVED_KEY] = {"name": name, "recorded_at": datetime.now(timezone.utc).isoformat()}
    _write(path, data)


def last_served(path: Path) -> str | None:
    """The record name last served, or None when never recorded or not a non-empty string."""
    rec = _load(path).get(LAST_SERVED_KEY)
    name = rec.get("name") if isinstance(rec, dict) else None
    return name if isinstance(name, str) and name else None


def record_foreign_unit(path: Path, unit: str, served: str) -> None:
    """Record the foreign unit and what it serves, so its sentence prints again only when either changes."""
    data = _load_for_write(path)
    if data is None:
        return
    data[FOREIGN_UNIT_KEY] = {"unit": unit, "served": served, "recorded_at": datetime.now(timezone.utc).isoformat()}
    _write(path, data)


def foreign_unit_changed(path: Path, unit: str, served: str) -> bool:
    """True unless the ledger holds exactly this (unit, served) pair."""
    rec = _load(path).get(FOREIGN_UNIT_KEY)
    return not (isinstance(rec, dict) and rec.get("unit") == unit and rec.get("served") == served)


def record_strata_offer(path: Path, name: str, reason: str) -> None:
    """Record the entry and reason the Strata offer was last printed for, so it prints again only when either changes."""
    data = _load_for_write(path)
    if data is None:
        return
    data[STRATA_OFFER_KEY] = {"name": name, "reason": reason, "recorded_at": datetime.now(timezone.utc).isoformat()}
    _write(path, data)


def strata_offer_changed(path: Path, name: str, reason: str) -> bool:
    """True unless the ledger holds exactly this (name, reason) pair."""
    rec = _load(path).get(STRATA_OFFER_KEY)
    return not (isinstance(rec, dict) and rec.get("name") == name and rec.get("reason") == reason)
