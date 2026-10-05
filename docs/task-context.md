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
| `update` | Changes the assignment, stop boundary, requested status, next action, open questions, or the revision and delegation budgets. |
| `decide` | Records a decision. |
| `artifact` | Names a workspace file the work produces, such as `draft.md`. |
| `reference` | Names a file (or a `tool_result_get` handle) the work depends on. |
| `check` | Declares a machine check: which tool call proves it and what result it must give. |
| `judge` | Records an editorial criterion and the model's assessment of it. |
| `waive` | Waives a check or judgment. This needs the human's words. |
| `integrate` | Records how a delegated result was used. |
| `close` | Closes as `complete`, `checkpoint`, `partial`, or `blocked`. |
| `show` | Prints the record. |

`start` and `update` also take `artifacts` (key to path) and `references` (a list of source and
path) in the same call. If any item is refused, the whole call is refused and the record stays as
it was.

**The human's words.** A decision, waiver, checkpoint, or budget raise counts as the human's only
when its `human_quote` appears in a human turn that the runtime recorded. The match ignores case,
extra spaces, and quote marks, and the quote must be at least 8 characters. Without a match, a
decision is recorded but labelled as an assumption. A waiver, checkpoint, or budget raise is
refused.

**What the runtime owns.** The model cannot write any of these: check receipts, artifact and
reference hashes, the human turns, delegation statuses, the revision count, and timestamps.

## References

A reference is a file path inside the workspace. Before every request of an active record, the
runtime reads each referenced file again and keeps its current text in view, even after
compaction. A record holds at most 4 references, each at most 200 KiB. A larger file is marked
"too large; narrow the reference".

There is no file watcher. A file that changes is picked up at the next request. A reference by
`tool_result_get` handle has no file behind it, so it does not survive a restart; it is then
marked "unavailable; read it again".

Task references share the four-reference limit with any set the model declares through
`tool_result_get`. A reference that does not fit is marked "unprotected: ..." with the reason.

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

A judgment is the model's editorial opinion. It is never evidence. `close complete` is refused
while any check is not passed or waived, any judgment is still open, or any delegation is not
integrated.

**Revision budget** (default 1). Editing an artifact after a check that depends on it has run
spends one revision, whether that run passed or failed. Once more revisions are used than the
budget allows, the packet tells the model to stop with the usable artifact and list the remaining
gaps. Raising the budget needs the human's words.

## Delegation handoff

The `agent` tool takes optional structured fields: `purpose`, `inputs`, `constraints`,
`expected_output`, `checks`, and `stop_condition`. The runtime builds the child's brief from these
fields. It never copies the conversation into the brief. A plain `agent` call works as before.

The child ends with a HANDOFF block, which is the child's own text, not proof. The runtime itself
observes only the status, the reason the child stopped, its tool-call count, the paths of files it
wrote, and the child session id.

Each delegation ends in one of these statuses: `completed`, `no_result`, `budget_exhausted`,
`killed`, `stuck`, `error`, `failed` (with the error kind), `timeout`, or `interrupted`. A call
that is cancelled while running becomes `interrupted` at the next human turn or restart. There is
no watchdog for a stalled child; only the `agent` tool's own timeout applies. Delegations run one
at a time.

The delegation budget (default 2) and the delegation entries apply only while a record is active.
`integrate` records how the coordinator used each result. If every check passes but a delegation
is not integrated, the turn ends with status unknown, the record stays active, and the reply names
the unresolved delegation.

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
become `interrupted`, and artifacts or references whose files changed are noted as "changed since
last session".

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
   as a decision in your words, `outline.md` as an artifact, and a checkpoint requested with your
   words.
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
acceptance.

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
- Streaming text can reach you before the final check rewrites the reply.
- Evidence covers declared checks only. Judgments and HANDOFF text are not proof.
- There is no file watcher and no watchdog for a stalled child.
- References by handle do not survive a restart.
- Without a project `.localharness/`, a task started in one folder replaces the saved record of
  another.
- None of this guarantees the quality of the work.
