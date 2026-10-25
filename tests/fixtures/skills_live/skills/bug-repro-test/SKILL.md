---
name: bug-repro-test
description: Writes a failing test that reproduces a reported bug before any fix is made. Use when asked to reproduce a bug, add a regression test or prove a bug with a test.
---
# Bug reproduction test

1. Restate the bug in one sentence: the input, the expected result and the actual result.
2. Find the nearest existing test file for the code involved with a read-only search.
3. Write the smallest test that sets up that input and asserts the expected result.
4. Run only that test and confirm it fails for the reason the report gives.
5. If it fails for another reason, fix the test, not the code.
6. Name the test after the behaviour that is wrong.
7. Stop there and leave the fix for a later step.
