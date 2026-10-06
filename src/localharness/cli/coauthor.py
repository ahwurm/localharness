"""Per-project co-author consent prompt + git hook (spec 14).

Two integration surfaces:

1. **Startup prompt** (``establish_coauthor_consent``): the gate/channel asks the
   user once per project at session start. The answer is recorded in the global
   consent store.

2. **``prepare-commit-msg`` git hook** (``install_hook``): installed at session
   start into ``.git/hooks/``. Fires on every ``git commit`` the harness runs
   (scoped by ``LOCALHARNESS_COMMIT=1``). Reads the recorded consent; if granted,
   appends the trailer to the commit message file. If no record exists, prompts
   via ``/dev/tty`` (the controlling terminal) and records the answer.

The hook is the integration seam: it is the one place the trailer is applied.
A user's hand-run ``git commit`` never carries the env var, so the hook is a
no-op for commits the harness did not make.
"""
from __future__ import annotations

import logging
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Optional

from localharness.agent.gate_types import PermissionRequest
from localharness.config import coauthor

log = logging.getLogger(__name__)

COAUTHOR_QUESTION = (
    "Credit localharness as a co-author on commits in this project? "
    "This adds a 'Co-Authored-By: localharness <localharness.agent@gmail.com>' line to "
    "commit messages in {project_root}. It is attribution only — MIT license, no loss of "
    "ownership, just a shoutout."
)

COAUTHOR_QUESTION_DETAIL = (
    "Answering yes records this for this project and it is not asked again for this project. "
    "Answering no means no co-author line is ever added to commits in this project. You can "
    "change your answer any time by editing ~/.localharness/coauthor_consent.yaml."
)

COAUTHOR_OPTIONS_LEGEND = "[y]es, credit it   [n]o, no co-author line"

COAUTHOR_TOOL_NAME = "coauthor"

# --------------------------------------------------------------------------- hook

HOOK_SCRIPT = '''\
#!/usr/bin/env python3
"""prepare-commit-msg hook: appends Co-Authored-By trailer for harness commits.

Fires on every `git commit`. Checks LOCALHARNESS_COMMIT=1 to scope to harness
commits only. Reads the per-project consent from the global config dir.
If consent is granted, appends the trailer to the commit message file.
Fail closed: no record, no yaml, no repo — no trailer, no crash.
"""
import os
import sys
from pathlib import Path


def _repo_root() -> str | None:
    import subprocess
    try:
        return subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _read_consent(repo_root: str) -> bool | None:
    """Read the per-project consent. True / False / None (no record)."""
    config_dir = Path(os.environ.get("LOCALHARNESS_DIR", str(Path.home() / ".localharness")))
    consent_file = config_dir / "coauthor_consent.yaml"
    if not consent_file.exists():
        return None
    try:
        text = consent_file.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import yaml
        data = yaml.safe_load(text)
    except Exception:
        data = _minimal_parse(text)
    if not isinstance(data, dict):
        return None
    projects = data.get("projects", {})
    if not isinstance(projects, dict):
        return None
    root_resolved = str(Path(repo_root).resolve())
    entry = projects.get(root_resolved)
    if not isinstance(entry, dict):
        return None
    val = entry.get("co_author")
    if val is True:
        return True
    if val is False:
        return False
    return None


def _minimal_parse(text: str) -> dict:
    """Minimal YAML parse for our known format (no external deps)."""
    result: dict = {"projects": {}}
    current_key: str | None = None
    for line in text.split("\\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("projects:"):
            continue
        if line.startswith("  ") and not line.startswith("    "):
            current_key = line.strip().rstrip(":")
            result["projects"][current_key] = {}
        elif line.startswith("    ") and current_key:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                k, v = parts[0].strip(), parts[1].strip().strip('"')
                if v == "true":
                    v = True
                elif v == "false":
                    v = False
                result["projects"][current_key][k] = v
    return result


def main() -> int:
    # Scope: only harness commits carry this env var.
    if os.environ.get("LOCALHARNESS_COMMIT") != "1":
        return 0
    # The commit message file is argv[1].
    if len(sys.argv) < 2:
        return 0
    msg_file = Path(sys.argv[1])
    if not msg_file.exists():
        return 0
    repo_root = _repo_root()
    if repo_root is None:
        return 0
    consent = _read_consent(repo_root)
    if consent is not True:
        return 0  # declined or no record: no trailer (fail closed)
    trailer = "Co-Authored-By: localharness <localharness.agent@gmail.com>"
    try:
        content = msg_file.read_text(encoding="utf-8")
    except OSError:
        return 0
    if trailer in content:
        return 0  # idempotent
    body = content.rstrip("\\n")
    try:
        msg_file.write_text(f"{body}\\n\\n{trailer}\\n", encoding="utf-8")
    except OSError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def git_repo_root(start_dir: str | Path) -> Optional[str]:
    """The git repo root containing ``start_dir``, or None if it is not in a repo.

    Resolved via ``git rev-parse --show-toplevel`` — the same source of truth the
    hook uses, so the startup prompt and the hook agree on which project a commit
    belongs to."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=str(start_dir), timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    root = out.stdout.strip()
    return root or None


async def settle_coauthor_startup(gate: Any, workspace: Optional[str | Path], notice: Any = None) -> None:
    """Settle co-author consent + install the hook for this session's project.

    Called at session start, right after workspace trust is settled. Resolves the
    git repo root from the workspace; if there is one, settles consent (asking once
    per project) and installs the prepare-commit-msg hook. Best-effort: a failure
    here costs a log line, never the session."""
    if workspace is None:
        return
    repo_root = git_repo_root(workspace)
    if repo_root is None:
        return  # not a git repo: no commits to credit
    try:
        await establish_coauthor_consent(gate, repo_root, notice)
    except Exception:  # noqa: BLE001 — a broken consent store costs a notice, not the session
        log.warning("could not settle co-author consent", exc_info=True)
        return
    try:
        hook = install_hook(repo_root)
        if hook is not None:
            log.info("co-author hook installed at %s", hook)
    except Exception:  # noqa: BLE001
        log.warning("could not install co-author hook", exc_info=True)


def install_hook(repo_root: str) -> Optional[Path]:
    """Install the prepare-commit-msg hook into the repo's .git/hooks/.

    Returns the hook path, or None if the repo has no .git directory.
    Idempotent: overwrites an existing hook. The hook is scoped by
    LOCALHARNESS_COMMIT=1, so it is a no-op for hand-run commits.
    """
    git_dir = Path(repo_root) / ".git"
    if not git_dir.is_dir():
        return None
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    hook_path = hooks_dir / "prepare-commit-msg"
    hook_path.write_text(HOOK_SCRIPT, encoding="utf-8")
    hook_path.chmod(0o755)
    return hook_path


def _request(project_root: str) -> PermissionRequest:
    """The co-author question as the PermissionRequest every channel already renders.

    ``grantable=False`` so the channels offer the yes/no pair, not four options: there
    is no "always" to distinguish from "once" here, because yes IS always for this
    project (session_trust.py:120-139)."""
    root = coauthor._normalize_root(project_root)
    return PermissionRequest(
        tool_name=COAUTHOR_TOOL_NAME,
        tool_params={},
        klass="coauthor-consent",
        key=root,
        grantable=False,
        reason=COAUTHOR_QUESTION.format(project_root=root),
        display=f"{COAUTHOR_QUESTION.format(project_root=root)}\n{COAUTHOR_QUESTION_DETAIL}",
        options_legend=COAUTHOR_OPTIONS_LEGEND,
    )


async def establish_coauthor_consent(gate: Any, project_root: str, notice: Any = None) -> bool:
    """Settle co-author consent for one project. Returns the effective consent
    (True/False) for this project.

    The order is the point: a recorded decision beats asking; only a project with no
    record is worth a question; a run that cannot ask records NOTHING and returns
    False (fail closed)."""
    recorded = coauthor.consent(project_root)
    if recorded is not None:
        return recorded
    if getattr(gate, "asker", None) is None:
        # Fail closed, record nothing: nobody was asked, so nobody answered, and a
        # later interactive session still gets its one question for this project.
        # Safe default = not credited.
        log.info("coauthor consent: cannot ask for %s, failing closed (no trailer)", project_root)
        if notice is not None:
            notice("Co-author consent: could not ask (no interactive channel) — no co-author line.")
        return False
    answer = await gate.asker(_request(project_root))
    granted = bool(getattr(answer, "allowed", False))
    coauthor.record_consent(project_root, granted)
    log.info("coauthor consent: %s recorded for %s", "granted" if granted else "declined", project_root)
    if notice is not None:
        notice(
            "Co-author consent: granted — localharness will be credited on commits in this project."
            if granted
            else "Co-author consent: declined — no co-author line on commits in this project."
        )
    return granted
