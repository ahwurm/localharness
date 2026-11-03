# LocalHarness 0.27.5 validation

Gate build: `dc6f1965` (the review dial on top of 0.27.4). Model under test: `qwen-27b-swift-1.5` on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_reply_panel.py` (levels 1, 3, `max` and `off`; disputes, a dispute-only fix turn, the question ending), `tests/unit/test_review_dial.py` (the `/review` command, the per-project store and its fallbacks, the completer), `tests/unit/test_task_flow_panel.py`, `tests/unit/test_task_flow_record.py`, `tests/unit/test_critic_scorers.py`, and `tests/unit/channels/test_mobile_push.py` (the new titles). The full unit suite passed on the gate build except `test_memory_index_tools.py::test_resolve_yesterday_window`, which assumes a 24-hour yesterday and fails on the day after daylight saving ends.

## The live gate

The orchestrator drove `localharness start` through tmux in a throwaway workspace. `/review show` printed `review: 1 — 1 check, each followed by a fix; the last fix goes out not re-checked (default)`. Then:

> can you do a quick network scan to tell me the count of devices on this network

| Build · model | Level | Time | Final line |
| --- | --- | --- | --- |
| 0.27.2 · gemma | 3 rounds | 16.5 min | `◆ reply blocked · 8 blockers` |
| 0.27.4 · qwen | 3 rounds | 3 min 41 s | `◆ reply pass · 21 notes` |
| gate · qwen | 1 | 75 s | `◆ reply fixed · not re-checked · 8 were open` |

One check found 1 blocker and 7 warnings; the round line read `◆ root fixing 8 · disputing 0 · 1/1 · last round`. The reply opened with its path (nmap and arp absent, a ping sweep of the /24, the neighbor table), gave "a lower bound of 12 devices, not a total", named the router's DHCP table as the true source, and closed with `Worth double-checking:`.

Also seen: the model again wrote a results file it was not asked for; it did not use a dispute or a question on this ask, so neither marker was exercised live.
