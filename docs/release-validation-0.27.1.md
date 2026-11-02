# LocalHarness 0.27.1 validation

Candidate: the 0.27.1 content on top of `1978d682`, the commit `v0.27.0` is tagged on (the 0.27.0 release commit and two bench-test commits after it), validated before it was committed. This release changes what the harness prints after `/model` and what a swap prints; it changes nothing a swap does. There was no live gate: no person-emulating run, and no swap on a real server. What was watched, and what was not, is below.

## Watched in a live session

`localharness start` in a detached tmux session at 120 columns, in a throwaway workspace, against the vLLM server already serving `swift-qwen3.8-27b` on this box. Nothing was launched, stopped or swapped.

| Typed | What the screen gave |
| --- | --- |
| `/model` | `Models:`, one group row, `1. swift-qwen3.8-27b · serving · active · 26.7 t/s measured`, twelve `– <name> · not offered · <reason>` rows, and the footer `/model <name or number> switches · Enter picks from the menu`. The text is dim, the model's name and number are bright, the speed is yellow (26.7 sits in the 20–30 band), and the one row longer than the window wrapped with its second row two columns past the row's own indent. |
| `/model Qwen3.8-9B-Q4_K_M` | The whole sentence: `Qwen3.8-9B-Q4_K_M cannot run here: no llama-server — set server.binary`. |
| `/model no-such-model` | `Unknown model 'no-such-model'. /model lists what's available.` |

No row on this box is a RAM or GPU refusal, so the short fact with its advice row under it was not seen live; the unit tests and the render below cover it.

## Rendered, not watched

A swap was not run: the box is shared and a swap stops the serving model for minutes. The REPL's own swap code ran over `lifecycle.swap` with the server's strategy (its stop and its launch) stubbed, printing through the real terminal channel at 120 columns, with the names, speeds and window of the approved screenshot. The rows it printed are the approved ones:

```
◆ model launching gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M · a few minutes
◆ model stopped qwen-27b-swift-1.5 · GPU freed
◆ model ready gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M · 58,368 window · 50.9 t/s measured · 1.5 GiB of
  layers on the CPU, decode will be slow
◆ model switched · http://127.0.0.1:8083/v1 · tool calling native · context budget refit to 58,368
```

`tests/unit/test_model_swap_lines.py::test_the_screenshots_swap_prints_the_approved_rows_through_the_terminal_channel` pins them. Until a swap that must stop a GPU holder is watched on a real screen, the four lines are a candidate, unverified live.

## Deterministic checks

Unit suite on the 0.27.1 content (`uv` was not on the PATH of the shell that ran it, so the command was `python -m pytest -q tests/unit -p no:cacheprovider` in the repo's venv): `93 failed, 22644 passed, 4 skipped, 4 xfailed, 119 warnings in 712.80s (0:11:52)`.

Integration suite (`tests/integration`, which CI also runs), one file at a time: `348 passed, 42 skipped` over 69 files, none failed; the skips are the opt-in live tests. Run as one process, the directory climbs to about 35 GiB resident on this box and is stopped from outside (exit 143, on the unmodified tree too; the box runs `earlyoom`, whose log could not be read to confirm), so there is no whole-directory number. One test, `test_mobile_sleep_e2e.py::test_a_sleeping_server_holds_nothing_of_the_session`, fails when it runs alone (on a clean export of v0.27.0 as well) or inside a whole-directory run, and passes with the rest of its file.

The same 93 tests fail, by name, on the unmodified v0.27.0 tree on this box (a clean export of the tag fails the same 92; the last one needs this box's local `INDEX.md`, below), so none of them is this release's:

- 89 are in `test_git_read_confinement.py`, `test_git_read_exec.py`, `test_git_read_filters.py`, `test_git_read_registration.py` and `test_doctor_git_read_row.py`: the box's git, 2.43.0, does not support `--no-lazy-fetch`, and `git_read` refuses to register without it.
- 2 are `test_memory_store.py::test_shelf_relative_time_newest_first` and `::test_shelf_day_flip_changes_only_labels`, the daylight-saving pair the 0.27.0 page names; the suite ran on 2026-11-01.
- 2 are `test_mode_narrow_only_layers.py::test_a_workspace_root_outside_the_project_is_dropped` and `test_docs_convergence.py::test_index_md_lists_the_bundled_plugins`, which also fail on the unmodified tree here. The first fails because the warning shortens this box's long temporary path and the test looks for the whole path; the second because it reads `INDEX.md`, a git-ignored local file, and this box's copy is stale (the test skips where there is none, as in CI).

The feature tests for this release are `test_terminal_channel.py` (the hanging indent, the dim and bright styles, the two tests that moved), `test_model_swap_lines.py` (the four lines, one name to a line, the speed once, an inert server name), `test_model_rows.py` (the list rows and the advice row), `test_spill_report.py` (one placement fact) and the other eighteen existing test files the new text touched.
