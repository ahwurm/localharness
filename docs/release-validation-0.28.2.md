# LocalHarness 0.28.2 validation

Gate build: the release tree, which is 0.28.1's code plus the test fix that followed it, the sdist's include list
anchored to the repository root and the version bump, tested from a clean worktree outside the home folder with the
CI extras (`dev`, `dispatch`, `mobile`) and CI's environment: no `COLORTERM` or `TERM`, an empty home directory and
a PATH of the worktree's virtual environment and the system folders.

## Unit tests

The feature-relevant unit tests, each group run on its own on the release tree:

| Group | Files | Result |
| --- | --- | --- |
| the style tests the end-to-end test had broken | `test_cli_theme.py` | 20 passed |
| the release pins, the release docs and the one version number | `test_docs_convergence.py`, `test_release_docs.py`, `test_version_source.py` | 123 passed, 1 skipped |

The gate was the full suite, the whole `tests/` directory, in CI's environment from a clean worktree, run the way
CI runs it (`pytest -q -p no:cacheprovider`). It printed
`24302 passed, 66 skipped, 4 xfailed, 101 warnings in 643.88s (0:10:43)`. The 66 skips are the opt-in live tests,
off unless their variable is set (31 for the browser, 15 for a model server, 5 for the embedding model), the
embeddings extra not being installed (7), Windows-only cases (4), local planning files that a clone does not hold
(3) and `INDEX.md`, which is git-ignored (1).

## Integration tests

`test_private_session_e2e.py`, the test the color fix is in: 9 passed. Run in the suite's order, with
`test_cli_theme.py` after it: 29 passed.

## Packaging

The sdist's file list, from `uv build --sdist` read with `tar -tzf`, on the tree before the include list was
anchored to the repository root and on the release tree:

| | Files | Outside `src/localharness/` |
| --- | --- | --- |
| before | 262 | `.gitignore`, `CHANGELOG.md`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml`, and four nested READMEs: `docs/reference-architectures/README.md`, `examples/README.md`, `examples/plugin-template/README.md`, `tests/fixtures/bench/exploration_root/README.md` |
| after | 258 | `.gitignore`, `CHANGELOG.md`, `LICENSE`, `PKG-INFO`, `README.md`, `pyproject.toml` |

The 252 files under `src/localharness/` are the same in both. `.gitignore` is the one file the include list does
not choose: the build tool adds the repository's own to every sdist.

A wheel built from the release sdist (`uv build --wheel <sdist>`, as CI's publish builds it) and installed in a new
virtual environment printed `localharness 0.28.2` for `localharness --version`.

## Live checks

No live checks; the code is 0.28.1's plus a test fix and the packaging fix.
