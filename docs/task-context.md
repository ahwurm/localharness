# Task context and evidence (0.16.4)

Ordinary turns keep the existing inference path. Compact role guidance asks the model to follow
the latest human scope, stop at requested checkpoints, and report observed results faithfully.
No interview, planner, memory store, or preflight model call is added.

Harness control messages carry private `_lh` origin/subtype metadata in canonical history.
Provider adapters render a short origin marker and remove metadata from the wire. Quoting a
marker does not manufacture provenance. Compaction carries source lineage and labels origins
in the summarizer input. These labels help attribution; they are not a security boundary.

## References for the current step

The existing `tool_result_get` accepts optional `active_step` and `source` arguments:

```json
{"id": "<existing content handle>", "active_step": "compose", "source": "style notes"}
```

Up to four snapshots can be declared for one step. Reusing the same source label replaces its
revision; changing the step starts a new set. An empty `active_step` releases the set. Protection
also expires at the next human turn. Plain restores retain their existing behavior.

After packing, compaction, emergency truncation, and tool-pair repair, the context manager checks
the actual outgoing messages. Missing snapshots are restored as paired tool data. Already-visible
full content is reused. Missing storage entries or an oversized input plus the existing reply
reserve block the step with a narrowing notice. XML adapter expansion is checked at the loop's
final request boundary too. Content is never promoted to system instructions.

Handles identify immutable snapshots. A source label is not a file watcher: after a source changes,
the caller must declare its new handle. This mechanism only protects declared dependencies and
uses the configured token counter, which can be approximate on some runtimes. It cannot ensure
the model chooses the right references or uses them well.

## Optional caller evidence

Runtime callers can pass an internal `TaskContext` to `AgentLoop(task_context=...)`. This does
not add a model tool, configuration key, or public completion protocol. For example:

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

Requirements match exact tool names and argument dictionaries. Actual dispatch results supply
receipts, including tool success, the declared metadata result field, requirement revision, and
hashes of declared file dependencies before and after execution. A command that runs but exits
nonzero does not pass an exit-code gate. Absent metadata or a missing child result is unknown.
Changed requirements or relevant file contents make old evidence stale; unrelated changes and
compaction do not. Omitting `result_field` verifies successful execution only, never quality.

The caller owns requirement selection, revisions after human corrections, explicit human waivers,
and requested stopping status. The runtime retains the latest human text in a current request
packet; it does not semantically translate arbitrary corrections into machine predicates. A caller
must revise the contract or remove it when the task scope changes. Use `requested_status="checkpoint"`
and `stop_boundary` for an intentional partial handoff, then update them on continuation.

At natural completion, unresolved declared checks replace unsupported success with a compact
receipt summary, without another inference. Waivers stay labeled as waivers. A caller-requested
checkpoint/partial/blocked/unknown response remains reportable. Streaming text can precede this
final check; it is the final summary and internal task status that are checked. `TaskComplete`
still describes completion of the agent turn, not user acceptance or verified project success.
No automatic persistence or stalled-child watchdog is added.
