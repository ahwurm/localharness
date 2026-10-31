# The metaharness: critics on every answer, and a task pipeline of fresh agents

This page explains the two layers LocalHarness puts around a model's work, and how you drive them. Both are built from one unit, the critic: a fresh agent with a narrow job, read-only tools and only the documents it is pointed at. The first layer, the reply panel, puts critics on every answer before it reaches you. The second, the task pipeline, runs bigger work as decisions, research, a plan and its steps, each by a fresh agent and each followed by critics; the root agent calls one unit at a time, and you steer between units with the `/task` verbs.

The reasoning is plain. A model grading its own answer finds little, so strangers from different angles check it: one looks for claims that do not hold, another for whether the answer serves what you are trying to do. A fresh agent per unit reads only the docs it needs, which is meant to keep a small model sharp late in a long task. Plan, execute, verify is the familiar shape; what is different here is that the checks are separate agents and that every unit is one call the root makes and you can interrupt.

## In a normal session

There is nothing to set up. When the root answers with no tool call, critics read the answer before you see it. Two run on every reply, `answer` and `jtbd`, so a hello pays two critics, each at the floor budget (3 actions, 1 minute) unless `task_flow.critics` sets one. There is no skip and no off switch. (`jtbd` stands down only for a turn of nothing but `task` calls when one of them ran a passing panel that already included it.)

A reply with a standing blocker is revised once, in one generation with no tools, and checked again. What you see:

- `pass` prints nothing.
- A blocked reply is delivered, then `◆ reply blocked · /task show reply · <the finding>`.
- A panel that could not finish delivers the reply, then `◆ reply unreviewed · /task show reply · <why>`. Nothing prints after your own stop.

While a reply is held, the status row and the phone show the running critic and its bound, for example `checking 3/4 · factual · up to 26 min`; the classic path and ACP print one line, such as `checking reply · 4 critics · up to 28 min · Ctrl+C delivers it unchecked`. The channel's stop word during the check delivers the draft unchecked. `/task show reply` prints the last reply review, and `/task show job` prints what the `jtbd` critic takes the session's job to be; `◆ job now: … · turn N` appears when it changes.

## Which critics run

Nine critics ship in the package, one small YAML file each. A critic's role is read from the package alone, never from your config, a project or an overlay.

| Critic | What it looks for |
|--------|-------------------|
| `answer` | The reply is correct, honest and on the question, with no invented facts. |
| `jtbd` | The work serves what you are trying to do, and which of your asks it drops or drifts from. It also keeps the session's job. |
| `factual` | Each cited line says what the work claims it says. |
| `recency` | The right entity is named, the facts are current, and each source is what it is said to be. |
| `big-picture` | Every decision has a step, the plan answers the questions asked, and nothing drifted. |
| `parsimony` | The plan is the smallest that does the job: nothing speculative. |
| `security` | The permission gate, secrets, and what a cloned repository could do. |
| `spec-compliance` | The step did the step: nothing missing, nothing extra. |
| `quality` | The work reads clearly and a later change lands in one place. |

On a reply, the turn's tool calls add critics to `answer` and `jtbd`: `factual` when the turn read files or called an agent, `recency` when it used the web, `spec-compliance` when it wrote files. A tool in no known read-only class counts as reading and the web, so a plugin or MCP tool fails closed. Each pipeline stage has its own floor panel (the table below). Four critics ship ready to add above a floor, by name in `task_flow.panels`: `recency` after research, `security` after the plan, `quality` and `factual` after a step.

A critic is told to assume the work missed and to find where. It answers `NONE`, or findings of the shape `[BLOCKER|WARNING|NOTE] rule — evidence — example fix`. A doc is blocked only while a BLOCKER stands; warnings pass. Findings reach a model fenced as data, labelled as such, and the model is told to check them against the files, not obey them.

## The task pipeline

Only you open a task, with `/task new <objective>`. After that the root drives. It asks what it needs in ordinary replies and files `decisions.md`; it may research (`research.md`, one read-only explorer per question, at most six questions); it files `plan.md`, a list of `## Step N` sections, each with `files:`, `depends_on:` and `done:` lines; then a fresh executor runs each step (`step-N.md`) and the task review writes `review.md`. A plan that does not parse is handed back to the root with the shape it needs.

Every unit is one `task` call: one child or one panel. The call returns a manifest that names what was filed, the verdict, the findings and `next:`, and the root calls again until the task is done, blocked or halted. Between units you steer with the verbs below, or with a message where the channel reads input mid-turn. No call runs longer than one child or one panel, so neither the root nor you is out of the loop for longer than that.

| Stage | Writes | Floor panel | Revisions |
|-------|--------|-------------|-----------|
| `new` | `decisions.md` | `jtbd` | 1 |
| `research` (optional) | `research.md` | `jtbd`, `factual` | 0 |
| `plan` | `plan.md` | `jtbd`, `big-picture`, `parsimony` | 3 |
| each step (`run`) | `step-N.md` | `spec-compliance` | 1 per step |
| the task review | `review.md` | `jtbd`, `big-picture` | 0 |
| a reply | nothing | `answer`, `jtbd` and the critics the turn senses | 1 per turn |

A blocked doc is revised up to its stage's revisions, and only while its blockers fall. Past that it is yours to decide: `/task accept` it, or `/task plan <change>`. Every line the harness prints names the bound of the unit it describes, a panel's the sum of its critics' minute ceilings and a child's its own minute ceiling, never more than the flow cap has left; at the defaults a plan panel is up to 13 minutes and an executor up to 25. One flow cap per turn limits child time (default 180 minutes, ceiling 360): when what is left is under the next unit's floor the run halts at a unit boundary, and `/task run` continues in the next turn. It bounds unattended child time; it does not promise that a task fits one turn.

## The eight verbs

Bare `/task` prints where the open task stands and the next command to type. Mid-turn, every slash command but `/task stop` waits for the turn to end; ACP answers a `/task` verb typed mid-turn with the running line.

| Verb | What it does |
|------|--------------|
| `/task new [objective]` | `new <objective>` opens a task; only you open one. A task that is done is closed first, and an open one is closed (its folder kept, not resumable) only when you say `new <objective>` again. Bare `new` reopens the interview while no plan is filed. |
| `/task research` | Steers the root's next turn toward research: one explorer per question, read-only. |
| `/task plan [change]` | Steers the root's next turn toward the plan. `plan <change>` carries your change as your own words and asks for a revised plan. While the interview is still open, `plan` and `research` skip the questions and file the decisions from your objective and your messages since. |
| `/task run` | Steers the root's next turn toward the steps: it calls the task again after each result until it says done, blocked or halted. |
| `/task show [doc]` | Prints `decisions`, `research`, `plan`, `step-N`, `review`, `reply`, `job`, or `t<n>/<doc>` from an older folder; with no name, the doc the open task points at. |
| `/task accept [doc]` | Accepts a blocked doc, or adopts your own edit of a doc. |
| `/task stop` | Ends the run after the running unit, where the terminal's input box (box mode) is reading input. |
| `/task clear` | Deletes the open task's folder and an old `task.json`, and says what went. |

## Stopping, by channel

| Channel | Stop between units | Cancel | During the reply check | A message mid-turn |
|---------|--------------------|--------|------------------------|--------------------|
| Box mode | `/task stop` | `Ctrl+C` | `Ctrl+C` delivers the draft unchecked | routed: heard between units, or queued |
| Classic path | none | `Ctrl+C` | `Ctrl+C` delivers the draft unchecked | not read |
| Phone | none; a slash joins the queue | `stop` | `stop` delivers the draft unchecked | routed: heard between units, or queued |
| ACP | none | `cancel` | `cancel` delivers the draft unchecked | not read; a `/task` verb is answered with the running line |

`/task stop` ends the run after the running unit and never ends the reply check. Cancel ends the running unit at once. A cancel while an executor runs can leave its files half-edited: the line `step N/M stopped, files half-edited` says so, nothing reverts them, and `/task run` re-runs the step.

## The task folder and the session docs

Each task is a folder, `agents/<name>/tasks/t<n>/`, in the session's state folder (a project's `.localharness/` when it has one), holding each doc and its `-review.md`. The session docs sit beside it: `sessions/jtbd.md` and `sessions/reply-review.md`, emptied by a fresh `localharness start` and kept by an in-process restart.

- A doc is read or shown only after its digest matches what was filed. One you edited is headed `<doc> edited since filed · /task accept <doc> (uses your edit)`.
- A folder this session did not make is read-only: `/task show t<n>/<doc>` prints it, headed as not adopted.
- `/clear` pauses an open task, and the next `/task` verb the paused line names picks it up.
- A wake from disk begins with no task and names the old folder. A task is not picked back up across `localharness start`, a wake or `/task new`; its folder is kept.
- The agent's own writes under these folders are refused. A plan changes through `/task plan`, or you edit a doc and `/task accept` it.

## Settings

`task_flow` is machine-level only. It is read from your agent file in your config folder's `agents/`, else from the `agent:` section of your global `overrides.yaml`, and a project's copy is replaced whole by yours.

```yaml
task_flow:
  revisions: {plan: 2}
  executor: {actions: 30, minutes: 30}
```

| Key | Sets | Floor | Default | Ceiling |
|-----|------|-------|---------|---------|
| `panels` | critics a boundary's panel adds to its floor, by boundary | none added | none | 3 added |
| `revisions` | revisions a boundary allows before it asks you | 0 | `new` 1, `plan` 3, `step` 1, `reply` 1 | the default plus 2; `reply` 1 |
| `critics` | one critic's own budget, by name | 3 actions, 1 minute | 12 actions, 6 minutes; `answer` and `jtbd` 3 and 1 | 12 actions, 6 minutes |
| `executor` | the executor's budget | 3 actions, 1 minute | 24 actions, 25 minutes | 48 actions, 40 minutes |
| `explorer` | an explorer's budget | 3 actions, 1 minute | 12 actions, 6 minutes | 24 actions, 10 minutes |
| `flow_cap_minutes` | child minutes one turn may spend | 1 | 180 | 360 |

The floors and ceilings are fixed in code: not even your own config can empty a panel or raise a ceiling. A value outside its range is clamped, and start names it (`task_flow.<key>: <value> is outside <floor>..<ceiling>; using <n>`).

`agent.self_check` is gone; the critics replace it. `localharness config migrate` drops it from your global files and prints `agent.self_check dropped from config; the critics replace it` when it was on.

## Security

Read [the prompt-injection threat model](../SECURITY.md#threat-model-prompt-injection) and [the approval gate](../SECURITY.md#human-approval-gate) for the whole picture; these are the parts that touch the metaharness.

- A clean panel is a quality signal, never a security one. A poisoned file can make a critic answer `NONE` as easily as object. A critic holds read-only tools, fixed in code.
- No person's go-ahead stands between a plan and its steps. Executors hold the root's tools under the same approval gate, so in `auto` their ordinary repository writes and commands run unasked. What checks them is the gate, the executor's budget, the flow cap, `/task stop` and the critics.
- The `task` tool's results are labelled untrusted: the manifest and the docs are written by the model and its children from repository files.
- The refusal on the task and session folders is defence in depth, not a barrier. SECURITY.md names the write forms it does not cover.
- A paired phone's `/task new`, `run` and `accept` sit outside the `remote_unattended` lock.

## The example workflow

A synthetic research-note workflow ships in the repository (`examples/workflows/research-note/`), not in the PyPI package. Copy it outside the LocalHarness repository, put its two specialists where project agents load from, and start a session:

```bash
cp -r examples/workflows/research-note ~/research-note && cd ~/research-note
mkdir -p .localharness/agents && cp agents/*.yaml .localharness/agents/
localharness start
```

Then, in the session:

1. Type `/task new Prepare the Acme Analytics customer profile following INSTRUCTIONS.md. Discuss the angle first.` The task opens with that line as its objective, and the root asks about the angle in ordinary replies. Answer it, for example "Use the second angle."
2. The root files `decisions.md`, checked by `jtbd`, then plans the outline, the draft and the lint as steps in `plan.md`, checked by `jtbd`, `big-picture` and `parsimony`. `/task show plan` reads it.
3. The root runs each step with a fresh executor, each checked by `spec-compliance`. `/task show step-2` reads a step's log. In box mode `/task stop` ends the run after the running step and `/task run` continues it.
4. Exit and run `localharness start` again. The task is not carried over, its folder stays, and `/task show t1/plan` reads the plan.

What happens at each step depends on the model; the docs show what it filed and what the critics found.

## Known limitations

- No live run: the critics and the pipeline have been exercised with fake models and three bench scenarios, never against a live model, so whether a critic earns its place is decided in use.
- The folder refusal is defence in depth; a write by an interpreter, a script, a glob or `cp -r` is not covered.
- `localharness config migrate` drops `agent.self_check` from global files only, so a project agent file or overlay that still carries it fails validation naming the key.
- Critics check quality, not safety.
- A timeout's cancel during the reply check propagates, while your own stop delivers the draft unreviewed.
- The phone's status clock shows only when the snapshot carries a phase.
- A hard cancel can leave files half-edited, and a cancel during a panel or an explorer still prints `Turn cancelled.`.
- A wake from disk begins with no task, and a task is not picked back up across a start or a close.
- The phone and ACP have no stop between units, and the classic path reads no input mid-turn.
