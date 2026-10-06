# Task context, references, delegation, and resume (0.16.5)

This page explains how a `localharness start` session keeps a working record for substantive
work. It covers what the record holds, how you inspect and clear it, how it survives a restart,
and what it does not guarantee. The [example workflow](#the-example-workflow) lets you try it.

## In a normal session

There is nothing to set up. Ordinary questions work as before: no task record, no packet, and no
extra preflight call. Each request does carry one more tool schema, `task`, and the role prompt
has one more sentence asking the model to keep the record for substantive work.

For substantive work (artifacts, several steps, delegation), the model may start a record with
the `task` tool. Whether it does depends on the model. While a record is active, the runtime adds
a compact packet with the record to the outgoing request. The packet is sent only in that
request. It is never written to session history.

## The `task` tool

The model changes the record only through this tool. Every change is saved at once.

| Action | What it does |
|--------|--------------|
| `start` | Creates the record: objective, assignment, stop boundary, requested status, instructions file, decisions. It replaces any earlier record, and the tool says so if that record was unfinished. |
| `update` | Changes the assignment, stop boundary, requested status, next action, or open questions. |
| `decide` | Records a decision. |
| `artifact` | Names a workspace file the work produces, such as `draft.md`. |
| `reference` | Names a file (or a `tool_result_get` handle) the work depends on; `drop: true` removes one. |
| `check` | Declares a machine check: which tool call proves it and what result it must give. |
| `judge` | Records an editorial criterion and the model's assessment of it. |
| `waive` | Waives a check or judgment. It must cite the human turn that waives it. |
| `integrate` | Records how a delegated result was used. |
| `close` | Closes as `complete`, `checkpoint`, `partial`, or `blocked`. |
| `show` | Prints the record. |

`start` and `update` also take `artifacts` (key to path) and `references` (a list of source and
path) in the same call. If any item is refused, the whole call is refused and the record stays as
it was.

**The human's words.** The runtime stores each human turn verbatim and numbers it in arrival
order. The last 24 turns are kept, and their numbers never change. A decision, check, waiver, or
checkpoint may cite one with `human_turn`. The runtime checks only that the turn is stored, then
copies its text beside the model's wording, for example
`[human turn 3: "Correction: five sections"] Outline has five sections`. A decision without a
citation is shown as an assumption. A waiver and a checkpoint must cite a turn; a `partial` stop
needs none. The runtime does not judge whether the turn supports the claim. The quoted text is
there for the reader to judge. The packet lists the stored turn numbers, and the `start` reply
echoes the turns on record.

**What the runtime owns.** The model cannot write any of these: check receipts, artifact and
reference hashes, the human turns, delegation statuses, the artifact revision history, and
timestamps.

**Where the policy lives.** The runtime records facts and reports them. It does not enforce
working rules on the model's behalf: when to ask, when to stop, how many revisions or delegations
are reasonable. Those rules live in the role prompt, the guidance text in the packet, and your
workflow's instruction file. The runtime refuses only malformed input (a missing field, an
uncited waiver, a turn number that is not stored, a key clash), input over a size limit, and paths
outside the workspace.

## References

A reference is a file path inside the workspace. Before every request of an active record, the
runtime reads each referenced file again and keeps its current text in view, even after
compaction. A record holds at most 4 references, each at most 200 KiB. A larger file is marked
"too large; narrow the reference".

There is no file watcher. A file that changes is picked up at the next request. A reference by
`tool_result_get` handle has no file behind it, so it does not survive a restart; it is then
marked "unavailable; read it again".

The kept text is the read tool's numbered view of the whole file, the same text `read` returns.
When the model has already read the whole file and that result is still in the request, the
reference is not added a second time. A read result that ends with the run's budget note still
counts. After that result is evicted, the reference comes back in the same numbered form. A file
larger than the read tool's default view (2000 lines or 100,000 characters) was never fully shown
by one `read`, so its reference is always added.

Task references share the four-reference limit with any set the model declares through
`tool_result_get`. A reference that cannot be declared is marked "unprotected: ..." with the
reason.

When the request with its references cannot fit with the reply reserve, the request still goes
out. Each task reference is marked "unprotected: cannot fit with reply reserve; drop or narrow
references" in the packet, and no reference text is added to that request. Any `tool_result_get`
references the model declared are left out of that request as well. The turn is blocked only when
the prompt does not fit even without references.

`reference` with `{source, drop: true}` removes the reference with that source. From the next
request, only the remaining references are kept in view.

## Checks and judgments

A check binds to an actual call with the same tool name whose arguments include every declared
argument with the same value. Extra actual arguments, such as a timeout, are fine. A check with no
declared arguments is refused. For example, the lint check in the example workflow:

```json
{"action": "check", "key": "lint", "description": "Draft passes the lint",
 "tool": "bash_exec", "arguments": {"command": "python3 checks/lint.py draft.md"},
 "result_field": "exit_code", "expected": 0, "depends_on": ["draft"]}
```

Each check has one outcome: passed, failed, unknown, stale, or waived. Stale means the check or a
file it depends on changed after the run. Declare a check before running its command, so that the
run binds as evidence.

A judgment is the model's editorial opinion. It is never evidence. A judgment the human waived
keeps the status `waived`; a later assessment by the model is recorded beside it.

**Artifact revisions.** Each distinct content of an artifact seen at a save gets a revision number
(`draft: rev 3`). The last 16 are kept, and the numbers never change. A check whose dependency
changed after its run shows both: `lint: failed (ran at draft rev 2, draft now rev 3)`. Nothing
counts revisions against a limit.

**When the record retires.** `close complete` is never refused. The record stops being sent only
when it was closed as complete, every check is passed or waived, and every delegation is
integrated. Open judgments do not keep it live. With nothing declared, closing retires it.
Otherwise it stays live and keeps showing what is unresolved. Passing checks alone never retires a
record that was not closed.

### The evidence line

The runtime never replaces the model's reply. When some declared check is not passed, or some
delegation is not integrated, it appends one line to the reply, for example:

```
Task evidence: lint: failed (ran at draft rev 2, draft now rev 3); review: unknown; reviewer (d1): failed: execution_error, not integrated.
```

A waived check is listed as `waived (turn 4: "…")`, with the waiving turn. Nothing is appended
when nothing is declared, or when every check passed and every delegation is integrated.
Judgments are not part of the line. The line is added to the reply you see; session history keeps
the model's own words, and the next request's packet carries the current facts. A bare
`TaskContext` caller gets the same line for its checks, without revision numbers.

## Delegation handoff

The `agent` tool takes optional structured fields: `purpose`, `inputs`, `constraints`,
`expected_output`, `checks`, and `stop_condition`. The runtime builds the child's brief from these
fields. It never copies the conversation into the brief. A plain `agent` call works as before.

The child ends with a HANDOFF block, which is the child's own text, not proof. The runtime itself
observes only the status, the reason the child stopped, its tool-call count, the paths of files it
wrote, and the child session id.

Each delegation ends in one of these statuses: `completed`, `budget_exhausted`, `killed`,
`stuck`, `error`, `failed` (with the error kind), `timeout`, or `interrupted`. The status comes
only from how the child's turn ended, never from its prose. A call
that is cancelled while running becomes `interrupted` at the next human turn or restart. There is
no watchdog for a stalled child; only the `agent` tool's own timeout applies. Delegations run one
at a time.

Delegations have no budget. While a record is live, the runtime opens a `running` entry before
each `agent` call and settles it after; the call always runs. A record holds at most 8 entries:
the oldest integrated entry is dropped to make room, and with 8 unintegrated entries a further
call still runs but is not recorded (a warning is logged). `integrate` records how the coordinator
used each result. An unintegrated delegation keeps the record live, sets the status to unknown if
every check passed, and is named in the evidence line.

## Persistence and recovery

The record is saved to `<state dir>/agents/<agent>/task.json`. The state dir is the project's
`.localharness/` folder when one applies, otherwise `~/.localharness`. The file is private
(mode 0600), written atomically, and at most 64 KiB. When the record would be larger, the oldest
human turns are dropped first; if it is still too large, the save is refused with "narrow the
record".

The record belongs to the folder it was started in. A record from another folder is not loaded,
and the file is left in place. Without a project `.localharness/`, however, every folder shares
`~/.localharness`, which holds one `task.json` per agent: starting a task in a different folder
replaces the saved record.

If the file cannot be read, it is moved to `task.json.corrupt` and the session prints "Task state
could not be read; starting without it". When a record is loaded again, running delegations
become `interrupted`. An artifact or reference whose saved and current hashes differ is noted as
"changed since last session"; one whose saved file can no longer be read is noted as "removed
since last session".

`/task` shows the record. `/task clear` deletes it and `task.json`.

## The example workflow

A synthetic research-note workflow ships in the
[GitHub repository](../examples/workflows/research-note/). It is not part of the PyPI package.
It contains an instruction file with stages and checks, two fictional sources, a voice sample, a
deterministic lint, and two specialist agents: a read-only reviewer and a writer.

Copy it outside the LocalHarness repository, then copy the specialists into
`.localharness/agents/`, which is where LocalHarness loads project agents from:

```bash
cp -r examples/workflows/research-note ~/research-note && cd ~/research-note
mkdir -p .localharness/agents && cp agents/*.yaml .localharness/agents/
localharness start
```

On the first start in that folder, a terminal session may ask once whether you trust the
workspace. Then send these messages in the normal session:

1. "Follow INSTRUCTIONS.md to prepare the Acme Analytics customer profile. Discuss the angle
   first." A model that uses the tool starts a record with INSTRUCTIONS.md, the voice sample and
   the sources as references.
2. "Use the second angle. Write the outline and stop for my review." The record shows your choice
   as a decision citing your turn, `outline.md` as an artifact, and a checkpoint that cites your
   turn.
3. Exit, run `localharness start` again, and type `/task`. The record is back, with any changed
   files noted.
4. "Outline approved. Write the draft and run the lint." The record shows `draft.md` as an
   artifact, the lint check, and its receipt from the actual run.
5. "Ask the reviewer to check source support, fix what it finds, then finish." The record shows a
   delegation entry with its runtime status, then how the result was integrated.

What happens at each step depends on the model. The record shows only what the model declared and
what the runtime observed.

## Programmatic callers

Code can pass a context to `AgentLoop(task_context=...)`. `localharness start` passes a
`TaskState`, the holder of the persisted record. A bare `TaskContext` still works and has no tool
or file:

```python
from pathlib import Path
from localharness.agent.task_context import Requirement, TaskContext

task = TaskContext(
    objective="Validate the current draft",
    artifacts={"draft": Path("draft.md")},
    requirements={
        "lint": Requirement(
            key="lint", description="Draft lint", tool="bash_exec",
            arguments={"command": "python lint.py draft.md"},
            result_field="exit_code", expected=0, dependencies=("draft",),
        ),
    },
)
```

A requirement matches a call with the same tool name whose arguments include every declared
argument with the same value. Receipts come from actual dispatch results. A command that runs but
exits nonzero does not pass an exit-code gate. Omitting `result_field` verifies only that the call
succeeded, never quality. `TaskComplete` still describes the end of the agent turn, not user
acceptance. Changed from 0.16.4: `TaskContext.finalize` appends the evidence line to the reply
instead of replacing the reply.

### Lower-level references (0.16.4)

Harness control messages carry private `_lh` origin metadata in history. Provider adapters render
a short origin marker and strip the metadata from the wire. These labels help attribution; they
are not a security boundary.

`tool_result_get` accepts optional `active_step` and `source` arguments, which keep up to four
stored snapshots in view for one step. Protection ends at the next human turn or when the step
changes. A handle names a fixed snapshot; after its source changes, the model must fetch and
declare the new handle. File-backed task references, described above, are re-read automatically.

## Limitations

- The model must choose to use the `task` tool. Nothing forces it to.
- Each request carries one more tool schema and one more sentence of role guidance.
- The act-guard is unchanged: with tools available, a first reply without a tool call still gets
  the existing one-time CONFIRMED nudge. This is the same in 0.16.4 and 0.16.5.
- Streaming text reaches you before the evidence line is appended to the final reply.
- The record reports facts; nothing is enforced on the model's behalf. A model can close as
  complete with a failed check; the record then stays live and the reply carries the evidence
  line.
- Evidence covers declared checks only. Judgments and HANDOFF text are not proof.
- There is no file watcher and no watchdog for a stalled child.
- References by handle do not survive a restart.
- Without a project `.localharness/`, a task started in one folder replaces the saved record of
  another.
- None of this guarantees the quality of the work.
