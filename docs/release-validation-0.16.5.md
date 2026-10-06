# LocalHarness 0.16.5 validation

Baseline: the published 0.16.4 wheel (tag `v0.16.4`, `ba64954`). Candidate: the 0.16.5 code at
`5aeace0` (the last code commit before this document; later commits change documentation only).
Persistent memory was disabled in every live run. Private evidence and raw traces stay out of the
repository and the distributions; every fixture is synthetic.

## Deterministic checks

Feature tests, run per file on the candidate code: `test_task_record.py` (36), `test_task_tool.py`
(27), `test_task_context.py` (22), `test_task_references.py` (11), `test_task_delegation.py` (5),
`test_task_workflow.py` (1), `test_coworker_acceptance.py` (4, which also maps the ten composed
cases below to the deterministic tests that cover them), `test_example_workflow.py` (5),
`test_agent_tool.py` (14), `test_subagent.py` (51), `test_active_references.py` (11, unchanged
from 0.16.4). They exercise citations by turn number, the appended evidence line in both finalize
paths, the retire rule, restart diffs, reference snapshots in the read tool's rendering with one
copy per request, the overflow fallback and the genuine-overflow block, delegation statuses from
the child's end reason, cancellation and malformed handoffs, and the ordinary-query path with a
task holder present.

Full suite on the candidate code: 7454 passed, 22 skipped, 4 xfailed. CI passed on `3538fe8`,
`aeb65e1`, and `b4719c2`; the release commit's CI run is the publication gate.

Distribution checks on the candidate code: wheel and sdist built and inspected for private or
unrelated files (none); clean wheel install; upgrade from PyPI 0.16.4; an unchanged configuration
validates; installed `__version__`, package metadata, and `--version` agree; one composed task
(outline, aside, correction) ran from the clean-installed wheel against the live model in three
turns with no delegation, the aside answered directly, and the correction recorded as a cited
decision.

## Preregistered live comparison

Thirteen synthetic cases were recorded with criteria, budgets, measurements, and fixture hashes
before either leg ran. Both legs used the same local model and endpoint, temperature 0, a
2,048-token output cap, a 32,768-token window (16,384 and 8,192 for the two reference cases), the
same real tools in a temporary copy of the same fixture workspace (read, write, edit, glob, grep,
bash_exec, tool_result_get, and the `agent` tool with a reviewer and a writer specialist; web tools
denied), a budget of 24 actions and 8 minutes per turn, and no memory. The candidate additionally
offered the `task` tool. The reviewer crash in one case was injected identically in both legs.

| Case | Baseline 0.16.4 | Candidate 0.16.5 |
| --- | --- | --- |
| Quick arithmetic | 42; 2 calls | 42; 2 calls |
| Answer-only question | one sentence; 2 calls; no files | same |
| Goal, aside, correction | outline via two writer delegations; aside answered; correction applied | outline written directly with references and artifact declared; aside answered plainly; correction recorded as a cited human decision; no delegation |
| Accepted decision plus ambiguity | did not re-ask the angle; wrote the draft with bullets, then flagged the conflict | did not re-ask the angle; raised one specific clarification with two options and wrote nothing |
| Outline checkpoint and restart | stopped at the outline; after restart, oriented from disk in 6 calls with two delegations | stopped at the outline with a cited checkpoint; after restart, oriented from the loaded record in 2 calls, outline unchanged, no draft |
| References under forced compaction (16K) | no draft; two empty replies at the output limit | draft written with each reference body present once; then two empty replies at the output limit |
| References at an 8K window | no draft; empty replies | no record opened; context reset repeatedly; action budget exhausted, no draft |
| Failed check | lint failed, two focused edits, rerun passed | same; no record opened, so no check was declared or bound |
| Delegated review | reviewer found the unsupported figure; coordinator fixed it | same, with the delegation recorded and the record closed |
| Crashed reviewer | said the review did not happen; reviewed itself, labelled as its own | same; record closed as `partial` |
| Changed source after restart | changed only the one figure; two delegations | changed only the one figure; no delegation; the reply misdescribed the edit as already present |
| Missing research, then conflicting standards | refused to invent; then wrote from the marketing brief with bullets, flagging the lint failure afterwards | refused to invent with options; the second turn ended in empty replies at the output limit |
| Bounded revision and isolation | one revision, rerun passed; no foreign record mentioned | same; the foreign record was not loaded |

| Totals over 13 cases | Baseline | Candidate |
| --- | ---: | ---: |
| Model calls | 148 | 146 |
| Elapsed seconds | 3,413 | 3,119 |
| Reported input tokens | 523,050 | 1,316,359 |
| `agent` dispatches | 20 | 4 |
| Turns ending in a model-side failure | 2 | 3 |

Reading of the results. The two quick cases cost the same number of calls in both legs; the
second call is the pre-existing act-guard nudge that fires whenever a tool-less first reply arrives
with tools offered, unchanged by this release. The candidate kept the goal through asides and
corrections, stopped at the requested boundaries, resumed from disk cheaply, and did its own
writing instead of delegating outlines. It paid for that with about 2.5 times the input tokens:
reference bodies, the record packet, and two larger tool schemas ride every request, while the
baseline pushed its work into small child contexts. Record bookkeeping also added tool calls on
several turns. The model opened a record on every multi-step case and skipped it on the two
single-step cases, so the evidence binding was not exercised live; it is covered deterministically.
Three candidate turns and two baseline turns ended with the model exhausting its 2,048-token
output budget on hidden reasoning; that is a model-side limit, and raising `agent.max_tokens` is
the mitigation. The 8K window was too small for either leg.

Review was performed by the implementing session, unblinded, one sample per case per variant.
This sample cannot establish general prose quality, speed, or model-independent reliability, and it
is not a comparison with any other workflow system.

## Limitations carried forward

- The model must choose to use the `task` tool; nothing enforces it, and the runtime enforces no
  policy on the model's behalf. The record reports facts beside the model's words.
- Each request carries one more tool schema and one more sentence of role guidance; with a record
  active it also carries the packet and up to four reference bodies.
- The act-guard and the other pre-existing loop mechanisms are unchanged in this release. An
  inventory of them was compiled for a separate review.
- There is no file watcher and no watchdog for a stalled child. Handle-only references do not
  survive a restart. Without a project `.localharness/`, folders share one `task.json` per agent.
- Streaming text can reach the user before the evidence line is appended.
