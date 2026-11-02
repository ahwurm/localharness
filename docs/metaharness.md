# The metaharness: critics that read an answer, and a task pipeline of fresh agents

This page explains the two layers LocalHarness puts around a model's work, and how you drive them. Both are built from one unit, the critic: a fresh agent with a narrow job, read-only tools and only the documents it is pointed at. The first layer, the reply panel, puts critics on an answer that used tools before it reaches you. The second, the task pipeline, runs bigger work as decisions, research, a plan and its steps, each by a fresh agent and each followed by critics; the root agent calls one unit at a time, and you steer between units with the `/task` verbs.

The reasoning is plain. A model grading its own answer finds little, so strangers from different angles check it: one looks for claims that do not hold, another for whether the answer serves what you are trying to do. A fresh agent per unit reads only the docs it needs, which is meant to keep a small model sharp late in a long task. Plan, execute, verify is the familiar shape; what is different here is that the checks are separate agents and that every unit is one call the root makes and you can interrupt.

## In a normal session

There is nothing to set up. A reply the model wrote without using a tool goes to you as written: a hello costs no critic. When the turn used a tool, critics read the reply before you see it: `jtbd`, plus the ones the turn's calls sense (see below), each with a full turn (24 actions, 10 minutes) unless `task_flow.critics` sets a lower one. There is no off switch for such a turn. (`jtbd` stands down only for a turn of nothing but `task` calls when one of them ran a passing panel that already included it.)

Every critic scores the work the same way. It breaks the work into numbered items (`A1` primary, `B1` secondary, `C1` auxiliary), scores each 0-10 and names the one change that would take it to 9. An item under 8 is a blocker. A reply with a blocker goes back to the root, with its tools, for up to three rounds. The root gets a score card and only the items under 8, and it is told to fix those and keep the rest. Revision stops early when a round's lowest score does not rise. The last round is announced, and its answer is delivered whatever the critics say. Each round's draft goes to `sessions/reply-draft.md`, and `sessions/reply-review.md` indexes every round with its scores and a diff stat against the round before. Critics read the whole draft through a content handle, not only its first 4,000 characters. What you see:

- A passing reply is delivered, then `◆ reply pass · /task show reply`, with `· 2 notes` when the critics left warnings or notes. A reply no critic read prints nothing.
- A blocked reply is delivered, then `◆ reply blocked · /task show reply · 3 blockers`; the line counts and names the command, and `/task show reply` prints the findings.
- A panel that could not finish delivers the reply, then `◆ reply unreviewed · /task show reply · <why>`. Nothing prints after your own stop.

While a reply is held, the status row and the phone show the running critic and its bound, for example `checking 3/4 · factual`; the classic path and ACP print one line, such as `checking reply · 4 critics · Ctrl+C delivers it unchecked`. A reply panel prints no bound; the lines of a task unit do. Each critic's findings print as they land, one line per finding (`◆ factual blocker A1 2/10 the quoted passage · pg-3 does not hold it`) or `◆ factual pass` for a clean one, and `◆ root revising · 1 blocker` before each round, `◆ root revising · last round · 1 blocker` before the third. The channel's stop word during the check delivers the draft unchecked. `/task show reply` prints the last reply review, and `/task show job` prints what the `jtbd` critic takes the session's job to be; `◆ job now: … · turn N` appears when it changes.

## Which critics run

Eight critics ship in the package. Each is one small YAML file holding its breakdown and its score anchors, and every critic is handed the same shared contract first (`assets/critics/contract.md`). A critic's role is read from the package alone, never from your config, a project or an overlay.

| Critic | What it breaks the work into, and scores |
|--------|-------------------|
| `jtbd` | Your asks, each quoted with its reading, scored by how much the work delivers now. It also keeps the session's job. |
| `factual` | The work's claims, each opened at its source and scored by how well the source backs it. |
| `recency` | The named things the work relies on, scored by whether each is the one you meant and current. |
| `big-picture` | The decisions and asks, scored by how a plan step covers each. |
| `parsimony` | The plan's steps, scored by whether each is needed and as small as it can be. |
| `security` | What the change touches, scored against the gate, secrets and repository-borne steering. |
| `spec-compliance` | The step's requirements, scored against the diff and the files. |
| `quality` | The units the change touched, scored by whether a stranger could change each in one place. |

On a reply, the turn's tool calls add critics to `jtbd`: `factual` when the turn read files or called an agent, `recency` when it used the web, `spec-compliance` when it wrote files. A tool in no known read-only class counts as reading and the web, so a plugin or MCP tool fails closed. Each pipeline stage has its own floor panel (the table below). Four critics ship ready to add above a floor, by name in `task_flow.panels`: `recency` after research, `security` after the plan, `quality` and `factual` after a step.

A critic opens its sources itself before it scores, and an item it cannot check scores `?` with what would let it check: a tool it lacks, context it was not given, or a narrower review. On a reply, `factual` and `recency` also hold `web_search`, `web_fetch` and `web_page_query`, and the browser's tools when its plugin is on; every other critic and every task-document panel is read-only. It answers one line per item, `A1 4/10 the item — what it saw — to 9: the fix`, then `SCORE <lowest>/10`. A doc is blocked only while an item scores under 8; a `?` passes. Findings reach a model fenced as data, labelled as such, and the model is told to check them against the files, not obey them.

## The task pipeline

You open a task with `/task new <objective>`, or the model opens one when your message asks for work of several steps: once a turn, with your message as the objective, announced as `◆ tN opened by the model` with `/task clear` to remove it. After that the root drives. It asks what it needs in ordinary replies and files `decisions.md`; it may research (`research.md`, one read-only explorer per question, at most six questions); it files `plan.md`, a list of `## Step N` sections, each with `files:`, `depends_on:` and `done:` lines; then a fresh executor runs each step (`step-N.md`) and the task review writes `review.md`. A plan that does not parse is handed back to the root with the shape it needs.

Every unit is one `task` call: one child or one panel. The call returns a manifest that names what was filed, the verdict, the findings and `next:`, and the root calls again until the task is done, blocked or halted. Between units you steer with the verbs below, or with a message where the channel reads input mid-turn. No call runs longer than one child or one panel, so neither the root nor you is out of the loop for longer than that.

A task the root finishes while filing `decisions.md` can end there: a `done` call marks it closing, the critics read the reply as the result, and when they pass it the result is filed as `review.md` and the task is done; a blocked reply leaves it open, and the line says so.

| Stage | Writes | Floor panel | Revisions |
|-------|--------|-------------|-----------|
| `new` | `decisions.md` | `jtbd` | 1 |
| `research` (optional) | `research.md` | `jtbd`, `factual` | 0 |
| `plan` | `plan.md` | `jtbd`, `big-picture`, `parsimony` | 3 |
| each step (`run`) | `step-N.md` | `spec-compliance` | 1 per step |
| the task review | `review.md` | `jtbd`, `big-picture` | 0 |
| a reply | nothing | `jtbd` and the critics the turn senses | 3 rounds per turn |

A blocked doc is revised up to its stage's revisions, and only while its blockers fall. Past that it is yours to decide: `/task accept` it, or `/task plan <change>`. Every line the harness prints for a task unit names the bound of the unit it describes, a panel's the sum of its critics' minute ceilings and a child's its own minute ceiling, never more than the flow cap has left; at the defaults a plan panel is up to 30 minutes and an executor up to 25. One flow cap per turn limits child time (default 180 minutes, ceiling 360): when what is left is under the next unit's floor the run halts at a unit boundary, and `/task run` continues in the next turn. It bounds unattended child time; it does not promise that a task fits one turn.

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

`task_flow` is machine-level only. It is read from your agent file in your config folder's `agents/` and the `agent:` section of your global `overrides.yaml`, merged key by key with the agent file winning, and a project's copy is replaced whole by yours.

```yaml
task_flow:
  revisions: {plan: 2}
  executor: {actions: 30, minutes: 30}
```

| Key | Sets | Floor | Default | Ceiling |
|-----|------|-------|---------|---------|
| `panels` | critics a boundary's panel adds to its floor, by boundary | none added | none | 3 added |
| `revisions` | revisions a boundary allows before it asks you | 0 | `new` 1, `plan` 3, `step` 1, `reply` 3 | the default plus 2 |
| `critics` | one critic's own budget, by name | 3 actions, 1 minute | 24 actions, 10 minutes | 24 actions, 10 minutes |
| `executor` | the executor's budget | 3 actions, 1 minute | 24 actions, 25 minutes | 48 actions, 40 minutes |
| `explorer` | an explorer's budget | 3 actions, 1 minute | 12 actions, 6 minutes | 24 actions, 10 minutes |
| `flow_cap_minutes` | child minutes one turn may spend | 1 | 180 | 360 |

The floors and ceilings are fixed in code: not even your own config can empty a panel or raise a ceiling. A value outside its range is clamped, and start names it (`task_flow.<key>: <value> is outside <floor>..<ceiling>; using <n>`).

`agent.self_check` is gone; the critics replace it. `localharness config migrate` drops it from your global files and prints `agent.self_check dropped from config; the critics replace it` when it was on.

## Security

Read [the prompt-injection threat model](../SECURITY.md#threat-model-prompt-injection) and [the approval gate](../SECURITY.md#human-approval-gate) for the whole picture; these are the parts that touch the metaharness.

- A clean panel is a quality signal, never a security one. A poisoned file can make a critic answer `NONE` as easily as object. A critic holds read-only tools, fixed in code, and on a reply `factual` and `recency` also hold the web tools the web-researcher holds, the browser among them when its plugin is on.
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
4. Exit and run `localharness start` again. The task is not carried over; the first screen says so in one line naming the folder on disk, `/task` says the same, and `/task show t1/plan` reads the plan.

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
