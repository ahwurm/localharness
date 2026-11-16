# LocalHarness 0.28.0 validation

Gate build: the release commit on top of 0.27.7 (the stuck rule read from the chat, the stop wording, the
retired `stuck_detector` and `recovery_injection` settings, the reused `fetch_id` and the `skill` warning fix),
tested from a clean checkout with the CI extras (`dev`, `dispatch`, `mobile`) and an empty home directory.
Model under test: `qwen-27b-swift-1.5` on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_stuck_from_chat.py` (the fold, the pointer in the request, the
stop, cycles, writes and the person's message) and `test_superseded_failure_view.py` (the request view it
shares), `test_delegation_status.py`, `test_terminal_channel.py`, `test_stuck_single_notice.py`,
`channels/test_mobile_reference_page.py`, `channels/test_mobile_push.py` and `channels/test_mobile_protocol.py`
(a stop as the person sees it: the root headline, the receipt, the phone row and push), `test_subagent.py`
(the stopped child's head), `test_task_flow_panel.py` (a reviewer cut short), `test_global_only_sections.py`
and `test_config_models.py` (the retired settings and their notice), `test_content_store.py` and
`test_web_tool.py` (the reused `fetch_id`), and `test_docs_convergence.py`. The full suite on the release
tree: 23,288 passed, 66 skipped, 4 xfailed.

## Replay over logged sessions

The real code of 0.27.7 and of this rule, replayed over 1,646 turns (12,206 tool calls) from 13 local sessions,
each fire judged by hand against the turn it came from. The rule was replayed with cycles able to stop; the
release lets a cycle only point, and no cycle stopped a turn in these sessions.

| | fires | real | false | unclear | stops |
| --- | --- | --- | --- | --- | --- |
| 0.27.7 | 113 | 98 | 9 | 6 | 6 (3 real) |
| 0.28.0 | 180 | 143 | 2 | 35 | 16 (15 real) |

All nine false fires of 0.27.7 are gone: the scroll stops, the nine-call cap on a word count after each edit,
and the re-checks after an edit. The two false fires left are pointers on a word count that came back to an
earlier value after an edit. Neither rule catches a `task` plan resubmitted with new text and refused the same
way, and neither sees repeats whose results differ only by page noise.

## Live checks

A person's session driven through tmux on the build under test, asking for a note of exactly 120 words and
counting it:

| Build | Result |
| --- | --- |
| the release | 120 words in 51 s, no stuck line, no warning on screen |
| cycles able to stop | stopped at 124 words after 35 s while the model was still editing |
| no cycle rule | 379 tool calls over 465 s cycling through 23 counts before ending at 120 |
| 0.27.7 | 120 words in 54 s (two counts) |

A question about the project's files was answered correctly in 25 s with every reviewer passing and no stuck
fire. A request to keep checking for a file that never appears polled inside one shell command for about four
minutes: no call repeats, so no rule sees it (listed under Known limitations).
