---
name: release-notes
description: Drafts user-facing release notes from the changes since the last tag, grouped into features, fixes and breaking changes. Use when asked for release notes, a release announcement or what is new in a version.
---
# Release notes

1. Find the last tag with `git describe --tags --abbrev=0`.
2. List the changes since it with `git log <tag>..HEAD --oneline`.
3. Sort each change into features, fixes or breaking changes.
4. Write one plain line per change, in the words a user would use, not the commit's.
5. Put breaking changes first and say what the reader must do about each.
6. Leave out changes a user cannot see, such as refactors and test-only commits.
7. Show the draft in the reply before writing any file.
