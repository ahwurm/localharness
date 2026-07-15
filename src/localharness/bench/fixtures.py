"""Bench fixture staging — copies tests/fixtures/bench/* to the absolute path scenario prompts
hardcode (/tmp/bench_fixtures/...), which must stay literal (the prompt text is not code and is
never dynamically injected). Two staging targets on Windows, one everywhere else — see
stage_bench_fixtures.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

_NATIVE_STAGED_ROOT = Path("/tmp/bench_fixtures")


def _staging_roots() -> list[Path]:
    """Target roots for staged fixtures.

    Path("/tmp/bench_fixtures") resolves natively per-platform (POSIX: /tmp/bench_fixtures;
    Windows: <cwd-drive>:\\tmp\\bench_fixtures) and is what native-Python file tools (read/write/
    glob) resolve scenario prompts' hardcoded path against. On Windows that native root is NOT
    where git-bash's /tmp lives (git-bash mounts /tmp under %TEMP%), so bash_exec-driven scenario
    steps need fixtures staged there too — hence the second, Windows-only root.
    """
    roots = [_NATIVE_STAGED_ROOT]
    if sys.platform == "win32":
        roots.append(Path(os.environ.get("TEMP", tempfile.gettempdir())) / "bench_fixtures")
    return roots


def stage_bench_fixtures(source_dir: Path) -> list[Path]:
    """Copy non-YAML fixture data from source_dir into every staging root.

    Idempotent and overwrite-safe — safe to call on every bench/test run; re-running refreshes
    any changed files. Recursively copies subdirectories (e.g. exploration_root/) so multi-file
    fixture trees stage cleanly. Scenario YAMLs are skipped (they live in the corpus, not staged
    data). A missing source_dir is a no-op per root (still creates the root dir) so callers can
    treat "nothing to stage" as non-fatal. Returns the staging roots.
    """
    source_dir = Path(source_dir)
    roots = _staging_roots()
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
        if not source_dir.exists():
            continue
        for src in source_dir.iterdir():
            if src.suffix == ".yaml":
                continue
            dst = root / src.name
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            elif src.is_file():
                shutil.copy2(src, dst)
    return roots
