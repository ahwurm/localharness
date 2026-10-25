# The skill format

## Fields
- `name`: lowercase letters and digits joined by single hyphens, at most 64 characters, equal to the directory name.
- `description`: one line, at most 1024 characters, best at most 250. Avoid `: ` and ` #` inside it, or quote it.
- Every other key is ignored. `allowed-tools` grants nothing.

## Layout
- `<name>/SKILL.md`: frontmatter between `---` lines, then the body; under 200 KiB.
- `references/`: detail the body points at, one level deep, read with `read`.
- `scripts/`: helpers the body runs, reading only what the person named.
- `assets/`: templates and data.

## A worked example
```markdown
---
name: release-notes
description: Drafts release notes from the local git history since the last tag and the CHANGELOG. Use when the person says write release notes, draft the changelog or prepare a release.
---
# Release notes
1. Find the last tag with `git describe --tags --abbrev=0`; with no tag, use the first commit.
2. List the commits since it with `git log --oneline <tag>..HEAD`.
3. Read the top of `CHANGELOG.md` and copy its headings and its tone.
4. Group the commits under Added, Changed, Fixed and Removed.
5. Drop merge commits and commits that only touch tests or formatting.
6. Write one present-tense sentence per entry, naming what a user of the project sees.
7. Start any breaking change with "Breaking:".
8. Ask which version and date the heading takes.
9. Show the draft to the person before writing anything.
10. If the person asks for changes, redo steps 4 to 7 and show it again.
11. After approval, add the entry under the top heading of `CHANGELOG.md` with `edit`.
12. Run `git diff CHANGELOG.md` and show it.
13. Do not commit, tag or publish.
14. Stop when the person says the entry is right.
```

## Description checklist
- Third person.
- What it does, then when to use it.
- The words the person would type.
- One line, at most 250 characters.
- Distinct from the other listed skills.
- `localharness skills` marks a description the listing dropped.

## Where to put a skill
- `~/.localharness/skills/<name>/`: yours, every project (`<config dir>/skills/<name>/` in a `--config-dir` session).
- `.localharness/skills/<name>/`: this project's, shared with the repository, at the project root (the folder holding `.localharness/`).
- `.agents/skills/<name>/`: the cross-client folder, at the same root, read only beside a `.localharness/`.
A project skill shadows a machine one of the same name. A bundled name cannot be taken.

## Skip lines
`localharness skills` prints a line for a skill it did not load: the reason, then the fix.
- `skipped: <origin> holds more than 256 entries; the rest are ignored. remove or move skills you do not need.`
- `skipped: <path> is not a regular file. replace it with a regular file.`
- `skipped: <dir> resolves outside <origin>. replace the link with a directory.`
- `skipped: <path> holds a control character. rename it without control characters.`
- `skipped: <path> is over 200 KiB. move detail into files under references/.`
- `skipped: <path> is not readable UTF-8 text. save it as UTF-8.`
- `skipped: <path> has no frontmatter. begin it with a --- line, then name and description, then a --- line.`
- `skipped: <path> frontmatter is not valid YAML. fix the YAML between the --- lines.`
- `skipped: <path> frontmatter uses YAML anchors or aliases. write each value out in full.`
- `skipped: <path> frontmatter is not a mapping with string values. write name: and description: as plain text.`
- `skipped: <path> has no name. add a name: line equal to the directory name.`
- `skipped: <path> name <name> is not lowercase letters and digits joined by single hyphens. rename it, for example release-notes.`
- `skipped: <path> name <name> is over 64 characters. shorten the name and the directory to 64 characters.`
- `skipped: <path> name <name> differs from its directory. make the name and the directory name equal.`
- `skipped: <path> name <name> is a bundled skill's name. rename the skill and its directory.`
- `skipped: <path> has no description. add a description: line saying when to use the skill.`
- `skipped: <path> description is over 1024 characters. shorten it to 1024 characters.`
- `skipped: <path> description is more than one line. write it on one line.`
- `skipped: <path> description holds a control character. remove the control characters.`
- `skipped: <dir> is past the 32 project skills kept. remove project skills you do not need.`
- `skipped: <dir> is past the 64 skills kept. remove skills you do not need.`
- `skipped: <dir> is shadowed by <other>. rename yours to keep both.`
