"""One question per new workspace: do you trust this place?

Owner ruling 2026-09-11: "default to auto mode so that it asks to trust the workspace, and
allows anything except a dangerous blacklist". ``auto`` runs almost everything without asking,
so the thing it rests on is not a per-call prompt — it is knowing that the person meant to be
working HERE. That is one question, asked once per workspace root, remembered forever.

Three ways a session gets past it, and only one of them is a prompt:

* **Recognized.** The workspace root already has a state store with sessions in it. Owner, same
  day: "it should recognize I've been in this environment before, used X tools etc." A place
  you have already worked in is not a place to be asked about; the record is written so it is
  explicit from now on, and one quiet line says what happened.
* **Recorded.** Somebody answered the question here, or above here — nested folders inherit
  (:func:`~localharness.config.trust.is_trusted_tree`). A "no" is remembered too, and runs the
  session in ``guarded``.
* **Answered.** A truly new root asks, through the channel's own ask path, so the question is
  drawn where every other permission question is drawn: inline in the terminal, as a dialog in
  Zed, as a message in Discord.

A session that cannot ask and has no record runs ``guarded`` — fail closed, and record nothing,
so the next interactive session in that directory still gets its one question. ``permissions.
mode`` set to anything other than ``auto`` skips all of it: the person already said what they
wanted, and ``unattended`` in config still bypasses everything exactly as it did before.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from localharness.agent.gate_types import PermissionRequest
from localharness.config import trust
from localharness.config.paths import WORKSPACE_DIR_NAME, global_config_dir

log = logging.getLogger(__name__)

TRUST_GATED_MODE = "auto"
"""The only mode that asks this question.

``auto`` is the mode whose safety rests on the answer. Every other mode was chosen explicitly —
``guarded`` and ``read-only`` already ask or refuse, ``trusted`` and ``unattended`` are
deliberate loosenings someone typed into a config — and asking a person to confirm a decision
they just made is the fatigue this release exists to remove."""

UNTRUSTED_MODE = "guarded"
"""Where a "no", and a session that could not ask, land.

Not ``read-only``: the person did not say "do nothing", they said "do not do it silently".
``guarded`` is the v0.14.0 default — it asks before a call crosses the workspace boundary or
looks destructive, and remembers the answer — which is exactly "run here, but tell me"."""

TRUST_QUESTION = (
    "Trust this workspace? LocalHarness will load its .localharness config and run tools here "
    "without asking, except for dangerous actions."
)
"""The one question. Both halves of trust in one sentence, because they are one decision (the
config layer and the permission mode share a record: :func:`trust.is_trusted_tree`)."""

TRUST_QUESTION_DETAIL = (
    "Answering yes records {root} and this is not asked here again. Answering no runs this "
    "session in guarded mode, where boundary-crossing and destructive calls ask first."
)
"""What the second line of the prompt says. A question about a permanent decision has to say
that it is permanent, and what the other answer costs — the terminal renders it under the
question and Zed renders it as the dialog body."""

TRUST_OPTIONS_LEGEND = "[y]es, trust it   [n]o, ask me (guarded)"
"""The terminal's key legend for this one question.

The ordinary ungrantable legend ends "(asks every time — cannot be remembered)", which is the
opposite of true here: this answer is the one that IS remembered."""

RECOGNIZED_NOTICE = "recognized this workspace ({count} earlier session{plural})"
"""The quiet line a recognized workspace prints instead of asking."""

DECLINED_NOTICE = (
    "this workspace is not trusted, so the session runs in {mode} mode — "
    "boundary-crossing and destructive calls will ask first"
)

CANNOT_ASK_NOTICE = (
    "this workspace has no trust record and this channel cannot ask, so the session runs in "
    "{mode} mode. Run localharness here interactively once to decide."
)

TRUST_TOOL_NAME = "workspace"
"""What the request names itself as. Not a real tool — the channels render an ASK against a tool
name, and "workspace" is what this question is about. It carries no tool call id, so a channel
that pairs questions with calls draws this one on its own."""


def trust_root(boundary: Optional[Path]) -> Path:
    """The directory this session's trust decision is about.

    The workspace boundary when there is one, and ``$HOME`` when there is not — a session
    started in your home directory or above has no project to trust, and ``$HOME`` is the
    honest name for "everywhere you keep things" (``verdict.derive_boundary`` returns None for
    exactly that case).
    """
    return Path(boundary).resolve() if boundary is not None else Path.home().resolve()


def state_store_for(boundary: Optional[Path]) -> Path:
    """Where to look for evidence that work has already happened at this root.

    A project with its own ``.localharness/`` keeps its sessions there. A session with no
    boundary keeps them in the global store, which is the same directory for every home-rooted
    run — which is why this only counts as evidence for the home-rooted case, and never lets one
    old home session vouch for a project directory nobody has opened.
    """
    if boundary is None:
        return global_config_dir()
    return Path(boundary) / WORKSPACE_DIR_NAME


def _request(root: Path) -> PermissionRequest:
    """The trust question as the :class:`PermissionRequest` every channel already knows how to
    render.

    ``grantable=False`` so the channels offer the yes/no pair rather than four options: there is
    no "always" to distinguish from "once" here, because yes IS always. The terminal replaces
    the legend through ``options_legend``; a channel that does not read that field falls back to
    its own two-option rendering, which is the same question with plainer buttons.
    """
    detail = TRUST_QUESTION_DETAIL.format(root=root)
    return PermissionRequest(
        tool_name=TRUST_TOOL_NAME,
        tool_params={"workspace": str(root)},
        klass="workspace-trust",
        key=str(root),
        grantable=False,
        reason=TRUST_QUESTION,
        display=f"{TRUST_QUESTION}\n{detail}",
        options_legend=TRUST_OPTIONS_LEGEND,
    )


async def establish_session_trust(gate: Any, notice: Any = None) -> str:
    """Settle this session's workspace trust, once, before the first turn.

    Returns the mode the session ends up in, so a caller can report it. ``notice`` is a callable
    taking one string — the console's print in a real session, a list's append in a test — and
    is optional because the decision must not depend on there being somewhere to print it.

    The order is the point: a recorded decision beats everything (it is what the person already
    said), evidence of prior use beats asking (it is what they already did), and only a root
    with neither is worth a question.
    """
    if getattr(gate, "mode", None) != TRUST_GATED_MODE:
        return getattr(gate, "mode", TRUST_GATED_MODE)

    root = trust_root(getattr(gate, "boundary", None))
    decision = trust.is_trusted_tree(root)
    if decision is True:
        return gate.mode
    if decision is False:
        return _decline(gate, notice, DECLINED_NOTICE)

    count = trust.prior_session_count(state_store_for(getattr(gate, "boundary", None)))
    if count > 0:
        # Recognized, and recorded so it is explicit from here on rather than re-derived from
        # whatever happens to be on disk next time.
        trust.record_trust(root, True)
        _say(notice, RECOGNIZED_NOTICE.format(count=count, plural="" if count == 1 else "s"))
        return gate.mode

    if getattr(gate, "asker", None) is None:
        # Fail closed, and record NOTHING: nobody was asked, so nobody answered, and a later
        # interactive session in this directory still gets its one question.
        return _decline(gate, notice, CANNOT_ASK_NOTICE)

    answer = await gate.asker(_request(root))
    if getattr(answer, "allowed", False):
        trust.record_trust(root, True)
        return gate.mode
    trust.record_trust(root, False)
    return _decline(gate, notice, DECLINED_NOTICE)


def _decline(gate: Any, notice: Any, template: str) -> str:
    gate.set_mode(UNTRUSTED_MODE)
    _say(notice, template.format(mode=UNTRUSTED_MODE))
    return gate.mode


def _say(notice: Any, text: str) -> None:
    log.info("workspace trust: %s", text)
    if notice is not None:
        notice(text)
