# LocalHarness 0.27.2 validation

Gate build: `e4e3f899` (the feature commit `36503b76` and the to-9 display fix on top of 0.27.1). The model under test was `gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M` on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_critic_scorers.py` (score lines, the stall rule, the round index, provenance, the evicted re-read, the blocker line's fix), `tests/unit/test_task_tool.py` (the model's open: on this turn's message, once a turn, after a finished task) and the panel files `tests/unit/test_task_flow_panel.py` and `tests/unit/test_reply_panel.py`. The full unit suite passed on the rebased tree except `test_memory_index_tools.py::test_resolve_yesterday_window`, which assumes a 24-hour yesterday and fails on the day after daylight saving ends.

## The live gate

A person-emulating agent (`subagents/ux-emulator.yaml`) drove one live `localharness start` session through tmux in a throwaway workspace, with no `/task` verb typed:

> research the three largest merchant acquirers by volume, then write a short comparison, about 200 words, with sources, into comparison.md

| Scenario | Verdict |
| --- | --- |
| model-task | VERDICT: NOT USABLE — the task opens itself and scores show, but it ran 25 minutes without finishing. |

What the screen showed:

- The task opened on its own: `◆ t4 opened by the model · /task clear (removes it)`.
- Reviewers answered in scored items, for example `◆ spec-compliance blocker A4 2/10 Comparison ~200 words`; no `answer` reviewer ran.
- A blocked reply went back with `◆ root revising · 3 blockers`, and `comparison.md` was written (about 170 words).
- The person stopped the turn at 25.6 minutes, while the second round's panel was still reading; `/task show reply` therefore listed round 1 only.

Recorded and accepted by the owner for this release: the time. Every round runs four critics on the local model (jtbd, factual, recency, spec-compliance), each opening its own sources. Review dials and defaults are the next spec.

Also seen: the blocker lines lacked the to-9 fix (fixed before this gate's build was tagged); critics counted the comparison's words differently (176, 146, about 175); one connection error from a critic; and a `1.5 GiB of layers on the CPU` window line on a launch decoding at 59-62 tokens per second, which is the estimate misreading this model rather than a spill.
