# Research note workflow (synthetic example)

You are helping an industry analyst write a short customer-profile research note. Work in this
folder. Follow the stages in order and stop where the user asks you to stop.

## Stages and deliverables

1. **Angle** — discuss the angle with the user. Deliverable: two or three candidate angles in your
   reply. No files.
2. **Outline** — write `outline.md` (five sections, one line each). Deliverable: the outline file.
   Stop here for review unless the user says otherwise.
3. **Draft** — write `draft.md` from the outline. Deliverable: the draft file, 350–600 words.
4. **Lint** — run the check below until it passes, fixing only the flagged passages.
5. **Review** — if the user asks for a review, delegate the review to the `reviewer` specialist and
   integrate its findings yourself. The reviewer reads; it never edits.

## Source rules

- Use only the files under `sources/`. A customer profile REQUIRES interview notes
  (`sources/interview-notes-*.md`). If there are none, say so and stop; never invent interviews,
  quotes, numbers, or findings.
- Every number or quoted phrase in the draft must appear in a source file.
- Do not restate the vendor's marketing claims as findings.

## Voice rules

- Match `voice/sample-note.md`: short declarative sentences, plain words, one idea per sentence,
  the customer's situation before the vendor's product.
- No bullet lists in the body of the draft (bullets are fine in the outline).

## Checks

- Machine check (exit code 0 means pass):

  ```
  python3 checks/lint.py draft.md
  ```

- Editorial judgment (not machine-checkable): source support of every claim, and voice match.
  Treat these as your own assessment with named passages; they are not proof.

## Working record

If a `task` tool is available, keep the working record with it: start the task with the
objective and the current stage as the assignment; declare this file, the voice sample, and the
source files you use as references; declare `draft.md` (and `outline.md`) as artifacts; declare
the lint command as a check before you run it (tool `bash_exec`, arguments
`{"command": "python3 checks/lint.py draft.md"}`, result_field `exit_code`, expected `0`,
depends on the draft artifact); record the user's corrections as decisions citing the turn where the user said them (`human_turn`);
record the editorial criteria as judgments. Close the task at the stage the user asked for.

## Boundaries

- One initial draft and one focused revision unless the user asks for more.
- Delegate at most two specialist assignments; never delegate recursively.
