# LocalHarness 0.27.0 validation

Gate build: `7451e511`. Commit `4fb3517e` followed the gate. The 0.27.0 release commit changes the version number, the CHANGELOG, a CHANGELOG test and this page. The model under test was `qwen-27b-swift-1.5` on vLLM.

## The live gate

A person-emulating agent (`subagents/ux-emulator.yaml`) drove live `localharness start` sessions through tmux, typing as a person would and reporting what the screen gave it. Each run used a throwaway workspace holding a fake library with nothing to run, and the workspace's `deny_patterns` list refused launches. The agent ends each scenario with one verdict line, `USABLE` or `NOT USABLE`. No `[ACTION-WATCH]` line appeared in any log, and the screen banner still read v0.26.0, as the version was not yet bumped.

| Scenario | Verdict |
| --- | --- |
| hello | VERDICT: USABLE — a one-line greeting in 5 s, though the banner still says v0.26.0. |
| question | VERDICT: USABLE — the right answer in 7 s and a clear "no review yet". |
| web | VERDICT: USABLE — a correct, dated, sourced answer with a matching pass, 36 s over the five-minute mark. |
| steer | VERDICT: NOT USABLE — the first note is fine but the steer was only half honoured (heading kept, extra detail kept) with nothing on screen flagging it. |
| stop | VERDICT: USABLE — Ctrl+C gave my prompt back in seconds and the next turn was clean. |
| wake | VERDICT: NOT USABLE — the restart screen gave no one-line account of the saved task, so I only learned it was there by typing /task. |
| task-note | VERDICT: USABLE — I got the note and a clean close, but only after 344 s, with the harness steering itself and never asking me anything. |
| code-question | VERDICT: USABLE — a grounded three-sentence answer in 48 s with a pass that matches. |
| small-edit | VERDICT: USABLE — the file was made and extended in 11 s each, nothing else changed, reviewers passed. |

Seven of nine were usable. The two that were not were annoyance, not blockage:

- steer: the note arrives in seconds and a person can just ask again, but a plain note ask shows `◆ task new`, and the "no headings" ask is silently missed.
- wake: `/task` on the new session says `task none open · t3 on disk, not carried over · /task show t3/review`, so the truth is one command away; only the first-screen line was missing.

Other things the run showed: `/task new` runs plan, run and done by itself in one long turn (344 s and about 6 min) in both wake and task-note, and the person types nothing and is never asked a question.

Commit `4fb3517e` followed the gate with the first-screen folder line, which prints whether the task folder finished or not, as `/task` does; its unit tests cover it (`tests/unit/test_start_cmd.py`, `tests/unit/test_repl_task.py`).

## Earlier sandbox passes

Two passes on earlier builds shaped the gate's fixes. Scenario and verdict only; the emulation reports are kept with the planning records.

First pass, build `602712a5`:

| Scenario | Verdict |
| --- | --- |
| hello | USABLE |
| question | USABLE |
| web | NOT USABLE |
| steer | NOT USABLE |
| stop | USABLE |
| wake | NOT USABLE |
| task-note | NOT USABLE |
| code-question | USABLE |

Second pass, build `602712a5` for wake and task-note and `1f7fc483` for the rest, on a working tree:

| Scenario | Verdict |
| --- | --- |
| wake | USABLE |
| task-note | USABLE |
| small-edit | USABLE |
| follow-up | USABLE |
| memory | NOT USABLE |
| vague-ask | NOT USABLE |
| long-answer | USABLE |
| web-recency | USABLE |
| stop-during-review | USABLE |
| research-task | USABLE |

The memory and vague-ask findings stand in the CHANGELOG's known limitations.

## Deterministic checks

Unit suite (`uv run pytest -q tests/unit -p no:cacheprovider`) on the release content, with nothing else running: `2 failed, 22664 passed, 10 skipped, 4 xfailed, 98 warnings in 477.74s (0:07:57)`. The two failures are `tests/unit/test_memory_store.py::test_shelf_relative_time_newest_first` and `::test_shelf_day_flip_changes_only_labels`. They fail on this machine on the day daylight-saving time changes, 2026-11-01, and pass with the clock in UTC; the tests seed their sittings at today's local clock time using the current UTC offset. The CHANGELOG entry, its test, the version number and this page are the only differences between the run and the release commit.
