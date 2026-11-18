# LocalHarness 0.28.3 validation

Gate build: the release tree, which is 0.28.2 plus the reply review panel change (one critic by default, the fact
check on a turn that fetched from outside the machine, the `recency` and `quality` critics removed) and the version
bump, tested from a clean worktree outside the home folder with the CI extras (`dev`, `dispatch`, `mobile`) and CI's
environment: no `COLORTERM` or `TERM`, an empty home directory and a PATH of the worktree's virtual environment and
the system folders.

## Unit tests

The feature-relevant unit tests, each group run on its own on the release tree:

| Group | Files | Result |
| --- | --- | --- |
| the panel, the fetch flag and delegation | `test_task_flow_panel.py`, `test_reply_panel.py`, `test_task_flow_record.py`, `test_critic_scorers.py`, `test_tool_declarations.py`, `test_agent_tool.py`, `test_subagent.py`, `test_subagent_children.py`, `test_repl_task.py`, `test_bench_runner.py`, `test_kept_list.py`, `test_global_only_sections.py`, `test_skills_discovery.py` | 1902 passed |
| the release pins, the release docs and the one version number | `test_docs_convergence.py`, `test_release_docs.py`, `test_version_source.py` | 126 passed, 1 skipped |

The gate was the full suite, the whole `tests/` directory, in CI's environment from a clean worktree, run the way
CI runs it (`pytest -q -p no:cacheprovider`). It printed
`24298 passed, 66 skipped, 4 xfailed, 101 warnings in 617.54s (0:10:17)`.

## Packaging

The sdist's file list, from `uv build --sdist` read with `tar -tzf`: 256 files, 250 under `src/localharness/` and
`.gitignore`, `CHANGELOG.md`, `LICENSE`, `PKG-INFO`, `README.md` and `pyproject.toml` outside it. 0.28.2's sdist held
258 and 252; the two fewer are the removed `recency.yaml` and `quality.yaml`. The critic files in it are
`big-picture.yaml`, `contract.md`, `factual.yaml`, `jtbd.yaml`, `parsimony.yaml`, `security.yaml` and
`spec-compliance.yaml`.

A wheel built from the release sdist (`uv build --wheel <sdist>`, as CI's publish builds it) and installed in a new
virtual environment printed `localharness 0.28.3` for `localharness --version`.

## Live checks

Both builds (0.28.2 and this one) ran on the same local Gemma model at review level 1, one run each, driving a
live session through the terminal. Timings vary from run to run.

| Scenario | 0.28.2 critics | 0.28.2 | 0.28.3 critics | 0.28.3 |
| --- | --- | --- | --- | --- |
| a question about local code | `jtbd`, `factual` | 95 s, pass | `jtbd` | 48 s, pass |
| a small edit: create a file | `jtbd`, `spec-compliance` | 74 s, pass | `jtbd` | 62 s, pass |
| a small edit: append a line | `jtbd`, `factual`, `spec-compliance` | 332 s, pass | `jtbd` | 32 s, pass |
| a device count from the shell | `jtbd`, `factual` | 344 s, blocked once and revised | `jtbd` | 100 s, pass |
| a web question, through the delegated researcher | `jtbd`, `factual`, `recency` | not finished at the 900 s cap | `jtbd`, `factual` | 387 s, pass |

The status row read `checking 1/1 · jtbd` on every local turn and `1/2 · jtbd` then `2/2 · factual` on the web
turn, so the fact check reaches the panel through a delegated agent's fetch.

On the web turn `factual` raised an identity finding of the kind `recency` raised: what the reply named was not
the entity asked about.

What the cut cost, seen in the device count: 0.28.2's `factual` blocked a count that did not match the command
output, and the revised reply was right. 0.28.3 checked that turn with `jtbd` alone and passed a reply that gave
the number of neighbor-table entries as the number of devices. The changelog lists it as a known limitation.

The runs in the table were made before `jtbd` gained the line that checks that what the work measured or found is
the thing asked for, not a stand-in passed off as the answer. The device-count question was run twice more on the
same model at level 1, with `jtbd` alone each time (`checking 1/1 · jtbd`):

| `jtbd` | Seconds | What the review did | The delivered reply |
| --- | --- | --- | --- |
| as in the table | 100 | passed | the number of neighbor-table entries, given as the number of devices |
| with the stand-in line | 200 | scored the scan and the count 5/10 as stand-ins and blocked on a section nobody asked for; one fix turn | the number of unique devices in the neighbor table, said to come from a cache, not a live scan |
| the release build | 224 | blocked on the scan (a stand-in, 2/10) and the count (an arithmetic error, 2/10), with an unasked guess as a warning; one fix turn | the number of neighbor-table entries, named as that |

The second run's block on unasked content is why the `jtbd` line about an "I did X" that no step backs was narrowed
to claims of work; in the third run unasked content did not block on its own. A stand-in passed off as the answer
blocked in the third run (the scan). At level 1 the fixed reply is not checked again: in the third run it said a
scanner was unavailable although the turn had not tried to run one.

Not run: review levels above 1, and a second model.
