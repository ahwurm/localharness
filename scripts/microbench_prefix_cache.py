#!/usr/bin/env python3
"""GB10 prefix-cache micro-bench (RANK-06) — prices the byte-stability discipline.

STATUS: live run DONE 2026-07-03 (qwen3.6-27b nvfp4 on vLLM, GB10, 5 trials/arm):
L=8k  hit 0.362s / bust 0.366s (delta ~0 — single-chunk prefill, overhead-dominated);
L=32k hit 0.543s / bust 16.692s (**delta 16.1s per one-byte bust**) — the
byte-stability discipline is confirmed load-bearing. L=96k intentionally not run:
repeated cache-bust prefills at that length are a hard-hang risk on unified-memory
boxes — run attended, with a free-memory watchdog, or not at all.

METHOD
------
The injected memory block sits near the TOP of the prompt. vLLM's prefix (KV) cache
skips re-reading a prompt prefix that is byte-identical to a previous request; one
changed byte near the top voids the cache for everything after it. This bench
measures that tax directly, as time-to-first-token (TTFT):

  For each target context length L (tokens):
    1. Build a stable prefix of ~L tokens (repeated corpus text) + short question.
    2. WARM: send once (populates the cache), discard.
    3. HIT:  send the identical prompt N times; record TTFT each time.
    4. BUST: flip one byte near the front of the prefix per trial (unique marker);
             send N times; record TTFT each time.
  Report per-L: median TTFT hit vs bust, the delta, and delta/L.

INTERPRETATION
--------------
- delta small (<~0.5s even at 100k): byte-stability discipline is over-conservative →
  the injected block could re-rank more freely (revisit RANK-04 staging).
- delta large (multi-second at long L): discipline confirmed; consider tightening
  (e.g. move the memory block even earlier / freeze harder).

USAGE
-----
  .venv/bin/python scripts/microbench_prefix_cache.py \
      --base-url http://localhost:8000/v1 --model <served-model> \
      --context-tokens 8000 32000 96000 --trials 5

Requires only the `openai` client already in the project deps. Writes a JSON report
next to this script (microbench_prefix_cache.results.json); record headline numbers
in the STATUS block above when you re-run on new hardware.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

FILLER = (
    "Section {i}. The quick brown fox jumps over the lazy dog near ticker 000660.KS "
    "while P/GP screens at -1.5 sigma; revenue grew and the harness recorded it. "
)


def build_prefix(target_tokens: int) -> str:
    # ~4 chars/token heuristic — close enough for a latency bench.
    chunks, size, i = [], 0, 0
    while size < target_tokens * 4:
        piece = FILLER.format(i=i)
        chunks.append(piece)
        size += len(piece)
        i += 1
    return "".join(chunks)


def ttft(client, model: str, prompt: str) -> float:
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8,
        temperature=0.0,
        stream=True,
    )
    for _chunk in stream:
        return time.perf_counter() - t0  # first token/chunk
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-tokens", nargs="+", type=int, default=[8000, 32000, 96000])
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="local")
    report: dict = {"model": args.model, "trials": args.trials, "lengths": {}}

    for length in args.context_tokens:
        prefix = build_prefix(length)
        question = "\n\nIn one word: what animal jumps over the dog?"
        stable = prefix + question

        ttft(client, args.model, stable)  # warm the cache
        hits = [ttft(client, args.model, stable) for _ in range(args.trials)]
        busts = []
        for t in range(args.trials):
            # One changed byte near the FRONT voids the prefix cache from that point on.
            busted = f"[{t}]" + stable[3 + len(str(t)):]
            busts.append(ttft(client, args.model, busted))

        row = {
            "ttft_hit_median_s": round(statistics.median(hits), 3),
            "ttft_bust_median_s": round(statistics.median(busts), 3),
            "delta_s": round(statistics.median(busts) - statistics.median(hits), 3),
            "hits": [round(x, 3) for x in hits],
            "busts": [round(x, 3) for x in busts],
        }
        report["lengths"][str(length)] = row
        print(f"L={length}: hit {row['ttft_hit_median_s']}s  bust {row['ttft_bust_median_s']}s  "
              f"delta {row['delta_s']}s")

    out = Path(__file__).with_suffix(".results.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
