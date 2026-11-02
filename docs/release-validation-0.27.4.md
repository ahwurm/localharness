# LocalHarness 0.27.4 validation

Gate build: `17f0c9ab` (the review and retelling commits on top of 0.27.3). Models under test: `gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M` and `qwen-27b-swift-1.5`, both on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_critic_scorers.py` (the warning tier, warnings carried into a revision, the path facts, the compose request) and `tests/unit/test_reply_panel.py` (a pass retold once, an unreviewed reply sent at once, the three-round case, the compose message's facts). The full unit suite passed on the gate build except `test_memory_index_tools.py::test_resolve_yesterday_window`, which assumes a 24-hour yesterday and fails on the day after daylight saving ends.

## The live gate

The orchestrator drove `localharness start` through tmux in a throwaway workspace with one line:

> can you do a quick network scan to tell me the count of devices on this network

| Build · model | Time | Final line |
| --- | --- | --- |
| 0.27.2 · gemma | 16.5 min | `◆ reply blocked · 8 blockers`; the reply opened "I apologize; I relied on the ARP cache…" |
| gate · qwen | 3 min 41 s | `◆ reply pass · 21 notes` after one revision |

The qwen reply opened with its path (a ping sweep, then the neighbor table, then noticing the interface is a /16 not the /24 it swept), gave "≥12 devices detected on your 10.0.0.0/16 network", and closed "Worth double-checking: the neighbor-table count is a snapshot … your router's DHCP lease table (or an nmap scan, which isn't installed here) would be the reliable source." No scores, critic names or apology reached the reply.

Also seen: the model wrote a results file it was not asked for; `recency` web-searched MAC prefixes for a LAN count; and on gemma, before the closing line was required, the retold reply called its table "corrected" and left the open points out. Review time is accepted for this release; review dials are next.
