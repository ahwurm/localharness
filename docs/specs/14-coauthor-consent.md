# Spec 14: Co-Author Consent (the localharness shoutout)

**Project:** LocalHarness
**Component:** `config/coauthor.py`, `cli/coauthor.py`, `config/defaults.py`
**Layer:** config (store) + CLI (prompt) + a single pure helper (the trailer)
**Status:** Authoritative — implement against this document
**Last updated:** 2026-10-06

---

## 1. Purpose

When the harness helps a user produce a commit, the user may want **localharness credited as a
co-author** on that commit — a small, permanent shoutout in the commit message, the way
`Co-Authored-By:` trailers work for other AI and human collaborators.

This spec scopes a **very narrow** feature:

1. A **yes/no consent prompt** asking the user, **once per project**, whether they want
   localharness included as a co-author on the commits the harness helps make in that project.
2. A **remembered answer per project** — the question is asked once per project root and not
   asked again for that project.
3. A **pure helper** that returns the co-author trailer string when consent was granted, and
   nothing when it was not.
4. **One clearly-bounded integration seam** where a commit message is prepared and the trailer
   is appended (or not).

The framing of the prompt is fixed by the owner: **MIT license, no loss of ownership, just a
shoutout.** A `Co-Authored-By:` trailer is attribution only. It transfers no intellectual
property, grants no rights, and changes no license. The user's code stays the user's; the
trailer is a credit line, nothing more. The prompt must say exactly that, so a "yes" is a
consent to be *named*, not a consent to give anything up.

The co-author identity is the dedicated account the owner set up:

* **Name:** `localharness`
* **Email:** `localharness.agent@gmail.com`
* **GitHub account:** the account bound to that email (the harness's own GitHub identity).

---

## 2. What this is — and is not

**This is a consent + attribution feature.** It decides, once per project, whether a credit
line appears on commits in that project, and it produces that credit line.

**This is NOT a git feature.** The harness does not own `git commit` today. There is no
commit code path in `src/` — commits happen because the agent runs `git commit` through
`bash_exec`. This spec does **not** add a `git_commit` tool, does not wrap or intercept the
shell, and does not rewrite `bash_exec`. The narrow, honest integration is a single pure
function that a commit-carrying command hands its message to. Owning the commit end-to-end is
**explicitly out of scope** for this release (see §9).

Keeping it this narrow is the point: the whole feature is one remembered yes/no per project,
one file on disk, and one pure function. Nothing about the agent loop, the gate, or the
permission verdict changes.

---

## 3. Grounding in the existing code

The feature is net-new, and it deliberately reuses the one existing pattern that is exactly
"ask a yes/no question once per workspace and remember it": **workspace trust**
(`src/localharness/cli/session_trust.py` + `src/localharness/config/trust.py`).

The per-project scope makes the reuse **direct**: `trusted_workspaces.yaml` is already a map
keyed by workspace root, and `trust.is_trusted_tree(path)` / `trust.record_trust(root, bool)`
already take a project root. The co-author store mirrors that shape one-for-one.

| Concern | Existing precedent | Reuse in this feature |
|---|---|---|
| The yes/no question shape | `PermissionRequest` (`agent/gate_types.py:126`), rendered with `grantable=False` and a custom `options_legend` (`session_trust.py:120-139`) | Build the same request shape; `grantable=False` so the channel offers a yes/no pair, not four options |
| Asking through the channel | `answer = await gate.asker(_request(root))` (`session_trust.py:182`) | `answer = await gate.asker(_request(project_root))` → `Decision` (`gate_types.py:197`) |
| Remembering the answer | `trust.record_trust(root, bool)` → `~/.localharness/trusted_workspaces.yaml` (`trust.py:162`) | `coauthor.record_consent(project_root, bool)` → `~/.localharness/coauthor_consent.yaml` |
| Reading the remembered answer | `trust.is_trusted_tree(path) -> Optional[bool]` (`trust.py:83`) | `coauthor.consent(project_root) -> Optional[bool]` |
| Per-project key | workspace root path, normalized (`trust.py:61-62`, `_key()`) | same: project root path, normalized |
| Fail-closed when it can't ask | `session_trust.py:177-180` — no `asker` → record nothing, run the safe mode | no `asker` → record nothing, **no co-author** (the safe default is "not credited") |
| Identity constant | `AGENT_NAME = "localharness"` (`channels/acp.py:88`) | `COAUTHOR_NAME` / `COAUTHOR_EMAIL` in `config/defaults.py` |

The license is **MIT** (`pyproject.toml:6`, `LICENSE`). The "no loss of ownership, just a
shoutout" framing is a property of the `Co-Authored-By:` trailer itself, not of the license —
the trailer is inert metadata that git stores in the commit message.

---

## 4. File layout

```
~/.localharness/
└── coauthor_consent.yaml          # NEW — per-project yes/no map, global dir, 0600

src/localharness/
├── config/
│   ├── defaults.py                # + COAUTHOR_NAME, COAUTHOR_EMAIL constants
│   └── coauthor.py                # NEW — consent store: consent(), record_consent()
├── cli/
│   └── coauthor.py                # NEW — the yes/no prompt: establish_coauthor_consent()
└── (integration)
    └── coauthor_trailer()         # NEW — pure helper, lives in config/coauthor.py
```

The consent file is in the **global config dir** (`~/.localharness/`), never inside a
workspace. A workspace that could vouch for its own co-author consent is not a consent
boundary — the same reason `trusted_workspaces.yaml` lives in the global dir, not the project
(`trust.py:41-53`). The file is a **map keyed by project root** (the git repo root), so each
project gets its own yes/no without polluting the project directory.

Example `coauthor_consent.yaml`:

```yaml
projects:
  /home/awurm/localharness:
    co_author: true
    recorded: "2026-10-06T13:00:00Z"
  /home/awurm/side-project:
    co_author: false
    recorded: "2026-10-06T14:00:00Z"
```

---

## 5. The consent store (`config/coauthor.py`)

A small, self-contained module. It mirrors `config/trust.py`'s shape (global YAML, atomic
write, `None` = never asked) but is deliberately far smaller — one key per project, one file.

```python
COAUTHOR_CONSENT_FILE = "coauthor_consent.yaml"

def consent_store_path() -> Path:
    """Always the GLOBAL dir, resolved at call time (tests change env)."""
    return global_config_dir() / COAUTHOR_CONSENT_FILE

def _normalize_root(project_root: str) -> str:
    """Normalize a project root to a canonical key: absolute, resolved, no trailing slash.
    Mirrors trust.py's `_key()` (trust.py:61-62) so the same project maps to the same key
    in both stores. `Path.resolve()` follows symlinks, so a symlinked checkout and its
    real path are one entry (same doctrine as trust.py:5-7)."""
    return str(Path(project_root).resolve())

def consent(project_root: str) -> Optional[bool]:
    """True / False / None (never asked for this project). ``project_root`` is the git repo
    root (from ``git rev-parse --show-toplevel``), not the workspace directory. None is the
    fail-closed case: no record for this project, so no co-author and (if interactive) the
    question is still owed for this project."""
    entry = _load(consent_store_path())
    projects = entry.get("projects", {})
    if not isinstance(projects, dict):
        return None
    root = _normalize_root(project_root)
    p = projects.get(root)
    if isinstance(p, dict) and isinstance(p.get("co_author"), bool):
        return p["co_author"]
    return None

def record_consent(project_root: str, granted: bool) -> None:
    """Persist a decision for one project. Permanent by design — changing the answer means
    hand-editing ~/.localharness/coauthor_consent.yaml, exactly as with workspace trust.
    Only a human answering the prompt gets here; a session that could not ask records
    nothing."""
    data = _load(consent_store_path())
    if not isinstance(data.get("projects"), dict):
        data["projects"] = {}
    root = _normalize_root(project_root)
    data["projects"][root] = {"co_author": granted, "recorded": _now()}
    atomic_write_overlay(consent_store_path(), data)
```

Rules, carried over from the trust store's doctrine:

* **`None` is not `False`.** A session that could not ask must not become a permanent "no"
  for that project. Only an answered prompt records anything (`trust.py:14`).
* **Atomic write, 0600.** Reuse `atomic_write_overlay` (`config/overlay.py`) so a crash never
  leaves a half-written consent file.
* **Per-project key, global file.** The file lives in the global config dir; the key is the
  normalized project root. No per-workspace files, no fingerprinting, no executables record.
  This is a preference, not a security boundary, and it must not grow entries that mean
  something else (the reason `declined_workspace_offers.yaml` is a sibling, not a second key,
  in `trust.py:43-47`).
* **Key normalization.** The project root is resolved to an absolute, canonical path (no
  symlinks, no trailing slash) so `/home/awurm/proj` and `/home/awurm/proj/` map to the same
  entry. This mirrors `trust.py`'s key handling.

---

## 6. The prompt (`cli/coauthor.py`)

Modeled line-for-line on `cli/session_trust.py`. The question is drawn where every other
permission question is drawn — inline in the terminal, as a dialog in Zed, as a message in
Discord — because it goes through the channel's own ask path.

```python
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
```

```python
def _request(project_root: str) -> PermissionRequest:
    """The co-author question as the PermissionRequest every channel already renders.
    grantable=False so the channels offer the yes/no pair, not four options: there is no
    'always' to distinguish from 'once' here, because yes IS always for this project
    (session_trust.py:120-139)."""
    root = _normalize_root(project_root)
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
    """Settle co-author consent for one project. Returns the effective consent (True/False)
    for this project. The order is the point: a recorded decision beats asking; only a
    project with no record is worth a question; a run that cannot ask records NOTHING and
    returns False."""
    recorded = consent(project_root)
    if recorded is not None:
        return recorded
    if getattr(gate, "asker", None) is None:
        # Fail closed, record nothing: nobody was asked, so nobody answered, and a later
        # interactive session still gets its one question for this project.
        # Safe default = not credited.
        return False
    answer = await gate.asker(_request(project_root))
    granted = bool(getattr(answer, "allowed", False))
    record_consent(project_root, granted)
    return granted
```

### When to ask

Two triggers, both per-project (project = git repo root):

1. **Startup (primary).** At session start, before the first turn, for the **current git repo
   root** (resolved via `git rev-parse --show-toplevel` from the workspace directory). This is
   the same slot `establish_session_trust` occupies (`cli/workspace.settle_startup_trust`). If
   the project already has a recorded answer, no prompt — the recorded value is returned
   silently. If the workspace is **not a git repo**, no prompt (there are no commits to
   credit; the consent is irrelevant).

2. **Lazy (fallback).** If a commit is prepared in a project that has **no recorded answer**
   (e.g. the workspace contains multiple git repos, and the agent works in one that was not
   the startup repo), the seam calls
   `establish_coauthor_consent(gate, project_b_root)` before applying the trailer. If the
   channel cannot ask (no `asker`), it fails closed: no trailer, no record.

The `notice` callable (the console's print, a list's append in a test) reports the outcome;
the decision must not depend on there being somewhere to print it (`session_trust.py:146-147`).

---

## 7. The trailer helper (the pure function)

The one thing the feature *produces*. Pure, trivially testable, no I/O:

```python
def coauthor_trailer(granted: bool) -> str | None:
    """The credit line when consent was granted, else None.

    'Co-Authored-By: localharness <localharness.agent@gmail.com>' — the exact trailer git and
    GitHub render as a co-author. None means 'append nothing'."""
    if not granted:
        return None
    return f"Co-Authored-By: {COAUTHOR_NAME} <{COAUTHOR_EMAIL}>"

def prepare_commit_message(message: str, granted: bool) -> str:
    """The single integration seam. If consent was granted and the message does not already
    carry the trailer, append it on its own line. Idempotent: a message that already has the
    trailer is returned unchanged."""
    trailer = coauthor_trailer(granted)
    if trailer is None:
        return message
    if trailer in message:
        return message
    body = message.rstrip("\n")
    return f"{body}\n\n{trailer}"
```

Constants live in `config/defaults.py`, named once so the name and email cannot drift:

```python
COAUTHOR_NAME: str = "localharness"
COAUTHOR_EMAIL: str = "localharness.agent@gmail.com"
```

---

## 8. The integration seam (and why it is this narrow)

Because the harness does not own `git commit`, the feature does not try to catch every commit.
The narrow, honest contract is:

* **`prepare_commit_message(message, granted)` is the one place the trailer is applied.**
  Whatever command carries a commit message hands its message to this function before the
  commit is made. Today that is the agent composing a `git commit -m "..."` via `bash_exec`;
  the agent is instructed (system prompt / tool doc) to run the message through this helper
  when it is preparing a commit on the user's behalf.
* **The consent value is read per-project** from `coauthor.consent(project_root)`, where
  `project_root` is the **git repo root** the commit is being made in (resolved via
  `git rev-parse --show-toplevel` from the commit's working directory). If the project has
  no record, the seam calls `establish_coauthor_consent(gate, project_root)` to settle it
  (the lazy trigger, §6). The helper itself never reads the store, so it stays pure and
  testable.
* **Idempotency is the safety property.** If the user (or the model) already wrote the
  trailer, it is not duplicated. If consent is `False` or `None`, nothing is appended.

What this deliberately does **not** do: it does not parse the shell, does not rewrite the
`bash_exec` command, does not add a `git_commit` tool, and does not force the trailer onto
commits the user made by hand. The seam is a value the commit-carrying command *chooses* to
consume. That is what keeps the feature narrow and reversible.

---

## 9. Out of scope (this release)

* **A `git_commit` tool / owning the commit end-to-end.** The harness still commits through
  `bash_exec`. Building a first-class git tool is a separate, larger feature.
* **Co-authoring pushes, PRs, or release notes.** The trailer is on the commit message only.
* **Changing the git author/committer identity.** The user's own `user.name`/`user.email` stay
  the author. localharness is a *co*-author trailer, not the primary author.
* **Any IP, license, or ownership change.** There is none. The trailer is inert attribution.
* **A machine-wide default consent.** There is no global "yes for all projects" toggle. Each
  project is asked independently. (A future release could add a machine-wide default that
  projects inherit; the store is designed so that is additive — a top-level `default` key
  that `consent()` falls back to when the project key is absent.)

---

## 10. Tests

Unit tests, no live git, no network:

1. `consent(project_root)` returns `None` with no file, `True`/`False` after
   `record_consent(project_root, …)`, and survives a corrupt file as `None` (never a crash).
2. `consent()` is **per-project**: recording `True` for `/a` does not affect `consent("/b")`
   (still `None`).
3. `record_consent` writes atomically to the global dir, 0600, and is the only thing that
   writes that file. Multiple projects coexist in the same file.
4. `establish_coauthor_consent` returns the recorded value without asking; asks exactly once
   when there is no record and an `asker`; records nothing and returns `False` when there is
   no `asker` (fail closed). Asking for project A does not settle project B.
5. `coauthor_trailer(True)` returns the exact string `Co-Authored-By: localharness
   <localharness.agent@gmail.com>`; `coauthor_trailer(False)` returns `None`.
6. `prepare_commit_message` appends the trailer on its own line when granted, returns the
   message unchanged when not granted, and is idempotent when the trailer is already present.
7. The prompt renders as a yes/no pair (`grantable=False`) through the terminal channel's
   `options_legend`, and a channel that ignores the legend still offers two options. The
   prompt text includes the project root.
8. Key normalization: `consent("/home/awurm/proj")` and `consent("/home/awurm/proj/")` return
   the same value (trailing slash stripped). A symlinked path and its resolved target return
   the same value (`Path.resolve()` follows symlinks).

---

## 11. Open questions

1. **How the agent is told to use the seam.** System-prompt instruction, a `bash_exec` doc
   note, or a tiny `commit_message` helper tool? The narrowest is a doc note + the pure
   function; a helper tool is more reliable but is a (small) new tool surface. **Decision
   needed.**
2. **GitHub account linkage.** The trailer uses the email; GitHub shows the co-author by
   resolving that email to the account. Do we document that the account must have
   `localharness.agent@gmail.com` as a (necessarily private) email for the avatar/name to
   resolve, or is the raw `Name <email>` line sufficient? **Likely: document it, no code.**
3. **Project root determination.** Resolved: the project root is the **git repo root**
   (via `git rev-parse --show-toplevel`), not the workspace directory. This makes the
   per-project scope meaningful (a workspace can contain multiple repos) and makes the
   lazy trigger reachable. If the workspace is not a git repo, no consent prompt is shown
   (there are no commits to credit).

---

## 12. Acceptance criteria

* [ ] `~/.localharness/coauthor_consent.yaml` is created only after a human answers the
      prompt for a project.
* [ ] The prompt is asked at most once **per project** and says "MIT license, no loss of
      ownership, just a shoutout."
* [ ] A granted consent for project A produces exactly `Co-Authored-By: localharness
      <localharness.agent@gmail.com>` on commit messages in project A, on its own line, never
      duplicated.
* [ ] A consent for project A does **not** affect commits in project B (project B is asked
      independently).
* [ ] A declined or absent consent appends nothing.
* [ ] No change to the agent loop, the gate, or the permission verdict.
* [ ] All §10 tests pass with no live git and no network.
