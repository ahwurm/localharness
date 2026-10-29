---
name: commit-message
description: "Writes a commit message for the staged changes: a component prefix, a colon and one lowercase sentence saying what changed. Use when asked to write, draft, reword or fix a commit message."
---
# Commit message

1. Read the staged changes with `git diff --staged`.
2. Pick the component the changes touch, such as the folder or module name.
3. Write one lowercase sentence saying what changed and why it matters.
4. Put the component and a colon in front: `component: sentence`.
5. Keep the line under seventy-two characters.
6. Show the message in the reply and leave committing to the person.
