"""Measured decode-speed ledger — rolling tok/s samples per (provider_type, model).

Records VERIFIED decode rates only: an exact completion-token count over the measured
first-token→last-token wall window of a real streamed completion. No estimate ever
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
    data[key] = {"samples": samples, "updated_at": datetime.now(timezone.utc).isoformat()}
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
