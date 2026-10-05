# Synthetic research-note example

A small, fully fictional workflow for trying LocalHarness on substantive writing work: an
instruction file with stages and checks, two synthetic sources, a voice sample, a deterministic
lint, and two specialist agents (a read-only reviewer and a writer). Start `localharness start` in
this folder and ask, for example: "Follow INSTRUCTIONS.md to prepare the Acme Analytics customer
profile. Discuss the angle first." Then: "Write the outline and stop for my review." Use `/task`
at any time to see the working record, and restart the session to see it resume.

## Setup

Copy this folder somewhere outside the LocalHarness repository first, so the session does not pick
up the repository's own `.localharness/`. LocalHarness loads project specialists only from
`.localharness/agents/` (or the global `~/.localharness/agents/`), not from `agents/`, so copy
them there:

```bash
cp -r examples/workflows/research-note ~/research-note && cd ~/research-note
mkdir -p .localharness/agents && cp agents/*.yaml .localharness/agents/
localharness start
```

On the first start in a folder, a terminal session may ask once whether you trust this
workspace; answer yes to let tools run there without asking, or no to run the session in guarded
mode. Because the agent files come from a project folder, the session may also print a line that it
ignores their `unattended` permission mode: a project layer may only tighten the mode. The
working record is saved in `.localharness/agents/<agent>/task.json` inside this folder.
