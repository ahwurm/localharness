"""Measured decode-speed ledger — rolling tok/s samples per (provider_type, model).

Records VERIFIED decode rates only: an exact completion-token count over the measured
first-token→last-token wall window of a real streamed completion, and only when that
window has enough substance to be a measurement (is_substantive_sample). No estimate ever
enters the file — a model with no recorded sample shows no number anywhere (hard rule:
no placeholder data). The display value is the median of the last SAMPLE_CAP samples:
robust to a one-off stall while still tracking real drift (decode slows as the KV
cache grows, so samples legitimately spread).

Storage: one small JSON file in the harness state dir, written atomically (a PER-WRITER tmp
+ os.replace — a shared tmp name is not atomic across processes). A corrupt or missing file
reads as empty. Concurrent sessions may race the file — last writer wins, acceptable for a
QOL stat.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from statistics import median
from uuid import uuid4

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


def _load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


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
    entry = entry if isinstance(entry, dict) else {}
    entry["samples"] = samples
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    data[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-writer temp name: os.replace is atomic, but open+truncate+write into a name SHARED by
    # every session is not — concurrent harnesses either published a byte-mix (unreadable ledger,
    # every model's history dropped) or lost the race entirely (the other writer's replace already
    # consumed the file → FileNotFoundError). Unique name ⇒ genuine last-writer-wins.
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # no-op after a successful replace; no litter on failure


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
    data = _load(path)
    key = speed_key(provider_type, model)
    entry = data.get(key) if isinstance(data.get(key), dict) else {}
    entry["tokens_per_chunk"] = round(float(ratio), 3)
    entry.setdefault("samples", [])
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    data[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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


def median_tps(path: Path, provider_type: str, model: str) -> float | None:
    """Median of the recorded samples for one model, or None if none exist."""
    entry = _load(path).get(speed_key(provider_type, model))
    samples = entry.get("samples", []) if isinstance(entry, dict) else []
    samples = [s for s in samples if isinstance(s, (int, float))]
    return median(samples) if samples else None


def all_median_tps(path: Path) -> dict[str, float]:
    """{speed_key: median tok/s} for every model with at least one sample."""
    out: dict[str, float] = {}
    for key, entry in _load(path).items():
        samples = entry.get("samples", []) if isinstance(entry, dict) else []
        samples = [s for s in samples if isinstance(s, (int, float))]
        if samples:
            out[key] = median(samples)
    return out
