# LocalHarness 0.28.1 validation

Gate build: the release commit on top of 0.28.0 and the two model-server fixes that followed it (the run folder and each session's scratch in it, private
sessions on the terminal and private chats on the phone with the gate rules that keep them private, mobile
protocol 13, the two fixes, the fixes made after review, and the example workflow's removal from the repository), tested from a clean worktree outside the home
folder with the CI extras (`dev`, `dispatch`, `mobile`) and an empty home directory. Models under test for
the live checks: `qwen-27b-swift-1.5` on llama.cpp for the first two runs below, and
`gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M` for the final private run.

## Unit tests

The feature-relevant unit tests, each group run on its own on the release tree:

| Group | Files | Result |
| --- | --- | --- |
| the run folder, its removal and a task's scratch | `test_run_folder.py`, `test_run_folder_gate.py`, `test_private_files_remover.py`, `test_private_files.py`, `test_task_scratch.py`, `test_told_line.py` | 614 passed |
| a private session on the terminal | `test_session_layers.py`, `test_private_row.py`, `test_private_start.py`, `test_private_signals.py`, `test_kept_list.py`, `test_private_terminal_text.py`, `test_workspace_discovery.py` | 346 passed |
| a private chat on the phone | `channels/test_mobile_private.py`, `channels/test_mobile_private_surfaces.py`, `channels/test_mobile_reference_page.py`, `channels/test_mobile_protocol.py` | 231 passed |
| `start` and `init --workspace` beside a run-only `.localharness` | `test_start_cmd.py`, `test_start_offers_workspace.py`, `test_init_workspace_scaffold.py` | 194 passed, 15 warnings |
| the two fixes (the saved-page "Always", a follow-up on a done task) | `test_saved_pages_gate.py`, `test_saved_pages_composed.py`, `test_downloads_gate.py`, `test_task_tool.py`, `test_task_flow_record.py`, `test_repl_task.py` | 1187 passed |
| the two model-server fixes (drafter and MTP-only GGUFs, a docker-launched server's memory) | `test_gguf_discovery.py`, `test_swap_floor.py` | 93 passed |
| the phone beside a private chat, and protocol 13 | `channels/test_mobile_server.py`, `channels/test_mobile_push.py`, `channels/test_mobile_sleep.py`, `channels/test_mobile_artifacts.py`, `test_image_hygiene.py` | 183 passed |
| the docs pins | `test_docs_convergence.py` | 114 passed, 1 skipped |
| the other test files the diff touches | `test_affordance.py`, `test_browser_plugin.py`, `test_subagent_children.py`, `test_web_fetch_save_to.py` | 470 passed |

The full unit suite (`tests/unit`) on the release tree is the backdrop: 23,942 passed, 15 skipped, 4 xfailed,
98 warnings. The 15 skips are the embeddings extra not being installed (7), Windows-only cases (4), local
planning files that a clone does not hold (3) and `INDEX.md`, which is git-ignored (1).

## Integration tests

`test_private_session_e2e.py`, `test_guardrails_from_global_dir_e2e.py`, `test_memory_surfaces_e2e.py` and
`test_dispatch_start_e2e.py` (every write of a private session, the org guardrails from the global folder only,
`/memory` on the real start, and `start --channel discord` offline): 15 passed. `test_private_session_e2e.py` run
alone, with the feature tests above: 9 passed.

## Live checks

A person's session on the terminal, driven through tmux with the `ux-emulate` skill and read from the
screen the way its user would. The two runs in this table used the qwen model:

| Run | Result |
| --- | --- |
| a reply that names a scratch file | usable: the told-about line appeared, and the run folder was gone after exit |
| `start --private`, asked to remember a fact (the second run, on the owner's install) | usable: the private refusal blocked a write to `MEMORY.md`, the run folder went with the session, only the files SECURITY.md lists were left beside the note the person asked for, and a fresh session did not know the fact |

The first private run is left out: it used a dev venv without the embeddings extra, which confounded it.
Both runs above came before the run folder moved into the project and before the text fixes, so they say
nothing of those.

The final private run was made on build `c4eabd06`, this release's code before its version bump and the fixes
made after it; the program has changed since only in its version number and the help text of `--private`. It ran
from the owner's install, whose banner still said v0.28.0; the checkout showed three uncommitted edits, in
`provider/refarch.py` and two tests. It was one run and was not repeated. The owner had switched the model
server, so it ran `gemma-4-31B-it-DECKARD-HERETIC-UNCENSORED-Thinking-Q4_K_M` and not the qwen model of the runs
above, and its replies and timings do not compare with theirs. The run's own verdict was not usable, for the
model's replies and for what it could not see.

What held on the harness's side: the run folder was in the project's `.localharness/run/` and was gone after
exit; only the files SECURITY.md lists remained, plus `.localharness/run/.gitignore` and two project files, the
one the person asked for and one the model wrote on its own; the reply named the project file that stays; a
fresh session did not know the fact; and no `UserWarning` text appeared on the screen.

What was worse on this model: it said the fact was saved to its persistent memory, which it was not; after a
reviewer line called the scratch folder a volatile temp directory, it wrote a name idea into a project file the
person had not asked for; and it named no scratch file in a reply, so the first Ctrl-D ended the session at
once, as designed.

Not exercised live: the told-about line with a path in the project's run folder, the new arm line and the armed
footer, since no reply named a scratch file. Unit tests pin them. A known cosmetic issue remains: with a very
long model name the footer's left hint is still cut into the status.

The phone was not run live. The owner's checklist, each item not run:

| Check | Result |
| --- | --- |
| a locked phone during End's undo | not run |
| End within 3 s of Stop | not run |
| a push tap on a sleeping private chat | not run |
