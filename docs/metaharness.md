# The metaharness: critics on every answer, and a task pipeline of fresh agents

This page explains the two layers LocalHarness puts around a model's work, and how you drive them. Both are built from one unit, the critic: a fresh agent with a narrow job, read-only tools and only the documents it is pointed at. The first layer, the reply panel, puts critics on every answer before it reaches you. The second, the task pipeline, runs bigger work as decisions, research, a plan and its steps, each by a fresh agent and each followed by critics; the root agent calls one unit at a time, and you steer between units with the `/task` verbs.

The reasoning is plain. A model grading its own answer finds little, so strangers from different angles check it: one looks for claims that do not hold, another for whether the answer serves what you are trying to do. A fresh agent per unit reads only the docs it needs, which is meant to keep a small model sharp late in a long task. Plan, execute, verify is the familiar shape; what is different here is that the checks are separate agents and that every unit is one call the root makes and you can interrupt.

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
