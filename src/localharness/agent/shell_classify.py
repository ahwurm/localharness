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

UNRESOLVABLE_TARGET_CHARS = ("$", "*", "?", "`")
"""PRD §3.2 step 8: a target containing any of these cannot be resolved to a path here, so
it is treated as outside the boundary."""

REDIRECTION_OPERATOR_CHARS = "<>&|"
"""Characters that may follow the first ``<``/``>`` of a redirection operator: ``>>``, ``>|``,
``>&``, ``<&``, ``<<<``, ``&>>``. Consumed as one unit by the splitter so an fd duplication is
never mistaken for the ``&`` separator, and by the redirection parser so ``N>&M`` is read as a
duplication rather than a write."""

REDIRECT_ONLY_SIGNATURE = ">"
"""Signature for a segment that is nothing but a redirection (``> file``). It is a write with
no command; step 8 still has to report its target."""

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

PATH_SEPARATORS = ("/", "\\")
"""A command may be spelled as a path. ``/bin/rm -rf x`` is ``rm -rf`` for signature purposes:
the leading directories are dropped so an absolute spelling cannot dodge the rule sets."""

WRAPPER_VALUE_FLAGS = frozenset({"-n", "-k", "-c", "-u", "-p", "--signal", "--kill-after"})
"""Wrapper flags that consume the next argument (``nice -n 10 cmd``, ``timeout -k 5 2 cmd``),
so the peel does not mistake the value for the wrapped command (PRD §3.2 step 4)."""

WRAPPER_NUMERIC_ARG = frozenset({"timeout"})
"""Wrappers whose first positional argument is a duration, not the wrapped command."""

SUBCOMMAND_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "git": frozenset({"-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path"}),
}
"""Global flags that take a value *before* the subcommand. Without this,
``git -c core.sshCommand=x push --force`` would read its subcommand as the ``-c`` value
(PRD §3.2 step 7)."""

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

SHELL_INTERPRETERS = frozenset({"sh", "bash", "zsh"})
"""Interpreters whose inline payload is *shell*, so PRD §3.2 step 5 recurses into it. A
``python3 -c`` payload is python and is not re-parsed."""

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

XARGS_VALUE_FLAGS = frozenset({"-n", "-I", "-i", "-P", "-d", "-a", "-E", "-s", "-L", "--max-args",
                               "--replace", "--max-procs", "--delimiter", "--arg-file", "--eof",
                               "--max-chars", "--max-lines"})
"""``xargs`` options that consume the next argument; the first token after them is the
payload command (PRD §3.2 step 5)."""

SED_MODE_FLAGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("-i", ("-i", "--in-place")),
    ("-n", ("-n", "--quiet", "--silent")),
)
"""``sed``'s mode is its flag: ``sed -n`` is the read-only signature listed in
``READ_ONLY_SIGNATURES_DEFAULT``, ``sed -i`` edits its file arguments in place. First match
wins, so ``sed -n -i`` is a write (PRD §3.2 steps 7-8)."""

SED_SCRIPT_FLAGS = frozenset({"-e", "--expression", "-f", "--file"})
"""When present, every positional argument of ``sed`` is a file; otherwise the first
positional is the script and the rest are files."""


# --------------------------------------------------------------------------- public API

def classify_shell(command: str, settings: GateSettings) -> ShellClassification:
    """Classify one shell command string. PRD §3.2 steps 1-8, in order.

    Pure: the same string and settings always produce the same classification.
    """
    text = _strip_heredocs(command)
    text = text.replace("\\\n", "")
    segments, dropped = _classify_text(text, settings)
    return ShellClassification(segments=tuple(segments), dropped=tuple(dropped))


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
    """Strip ``<<WORD`` / ``<<-WORD`` / ``<<'WORD'`` from one line, returning the delimiters."""
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
        if line[index : index + 2] == "<<" and line[index : index + 3] != "<<<":
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


def _group_inner(body: str) -> str | None:
    """Return the inside of a ``( … )`` or ``{ … }`` group, or None if this is not one."""
    if body.startswith("(") and body.endswith(")"):
        return body[1:-1]
    if body.startswith("{") and body.endswith("}") and body[1:2] in (" ", "\t", "\n"):
        return body[1:-1]
    return None


def _classify_text(text: str, settings: GateSettings) -> tuple[list[ShellSegment], list[str]]:
    """Steps 2-8 over a already-heredoc-stripped string; recursive for groups and payloads."""
    segments: list[ShellSegment] = []
    dropped: list[str] = []
    previous_head: str | None = None
    for raw in _split_top_level(text):
        body = raw.text.strip()
        if not body:
            previous_head = None
            continue
        inner = _group_inner(body)
        if inner is not None:
            group_segments, group_dropped = _classify_text(inner, settings)
            segments.extend(group_segments)
            dropped.extend(group_dropped)
            previous_head = None
            continue
        residual, lifted = _lift_substitutions(body, settings)
        segments.extend(lifted)
        host, extras, host_dropped, head = _classify_one(residual, settings)
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
    text: str, settings: GateSettings
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
        inner_segments, _ = _classify_text(inner, settings)
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
    text: str, settings: GateSettings
) -> tuple[ShellSegment | None, list[ShellSegment], list[str], str | None]:
    """One command: redirections, drop rule, wrapper peel, payload lift, signature, targets.

    Returns ``(host segment or None if dropped, lifted payload segments, dropped texts,
    head token)``. PRD §3.2 steps 4-8.
    """
    command_text, redirect_targets = _extract_redirections(text)
    argv = _strip_prefixes(_tokenize(command_text))
    if argv:
        argv = [_basename(argv[0]), *argv[1:]]
    if not argv:
        if not redirect_targets:
            return None, [], [text.strip()] if text.strip() else [], None
        return _make_segment(REDIRECT_ONLY_SIGNATURE, (), settings, redirect_targets), [], [], None

    # Step 6 — drop only the exact no-op segments; after step 2 no substitution survives here.
    if argv[0] in settings.dropped_commands and not redirect_targets:
        return None, [], [text.strip()], argv[0]

    argv = _peel(argv, settings)
    if not argv:
        return None, [], [text.strip()], None
    return _build(argv, settings, redirect_targets)


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
    argv: list[str], settings: GateSettings, redirect_targets: list[str]
) -> tuple[ShellSegment | None, list[ShellSegment], list[str], str | None]:
    """Signature + payload lifting for a peeled argv (PRD §3.2 steps 5, 7, 8)."""
    argv = [_basename(argv[0]), *argv[1:]]
    head = argv[0]
    extras: list[ShellSegment] = []

    runner = _composing_runner(argv, settings)
    if runner is not None:
        prefix, inner_argv = runner
        host, inner_extras, _, _ = _build(inner_argv, settings, redirect_targets)
        if host is None:  # pragma: no cover - inner argv is non-empty by construction
            host = _make_segment(prefix, tuple(argv), settings, redirect_targets)
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
        if head == "xargs":
            payload = _xargs_payload(argv)
            if payload:
                payload_argvs.append(payload)
        else:
            payload_argvs, find_delete = _find_payloads(argv)

    signature, inline = _signature(argv, settings, find_delete=find_delete)

    if head in SHELL_INTERPRETERS:
        payload_texts = _inline_payloads(argv, settings)

    for payload in payload_argvs:
        segment, more, _, _ = _build(payload, settings, [])
        if segment is not None:
            extras.append(segment)
        extras.extend(more)
    for payload_text in payload_texts:
        inner_segments, _ = _classify_text(payload_text, settings)
        extras.extend(inner_segments)

    host = _make_segment(
        signature,
        tuple(argv),
        settings,
        redirect_targets,
        inline=inline,
        payload_lifted=bool(payload_argvs or payload_texts),
    )
    return host, extras, [], head


def _make_segment(
    signature: str,
    argv: tuple[str, ...],
    settings: GateSettings,
    redirect_targets: list[str],
    *,
    inline: bool = False,
    payload_lifted: bool = False,
) -> ShellSegment:
    """Assemble the segment: destructive/read-only verdict plus step 8's write targets."""
    targets = list(redirect_targets) + _write_targets(signature, argv, settings)
    destructive = signature in settings.destructive_signatures
    read_only = (
        signature in settings.read_only_signatures and not destructive and not payload_lifted
    )
    return ShellSegment(
        signature=signature,
        argv=argv,
        read_only=read_only,
        destructive=destructive,
        inline_interpreter=inline,
        write_targets=tuple(targets),
        unresolvable_write=any(
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
        index = 0
        while index < len(rest):
            token = rest[index]
            if ASSIGNMENT_RE.match(token):
                index += 1
                continue
            if token.startswith("-") and token != "-":
                index += 2 if token in WRAPPER_VALUE_FLAGS else 1
                continue
            if head in WRAPPER_NUMERIC_ARG and DURATION_RE.match(token):
                index += 1
                continue
            break
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


def _xargs_payload(argv: list[str]) -> list[str]:
    """PRD §3.2 step 5: the command ``xargs`` will run (empty when it defaults to ``echo``)."""
    index = 1
    while index < len(argv):
        token = argv[index]
        if token.startswith("-"):
            index += 2 if token in XARGS_VALUE_FLAGS else 1
            continue
        break
    return argv[index:]


def _inline_payloads(argv: list[str], settings: GateSettings) -> list[str]:
    """PRD §3.2 step 5: the string after ``sh -c`` / ``bash -c`` / ``zsh -c``, to recurse into."""
    flags = settings.inline_code_flags.get(argv[0], ())
    payloads: list[str] = []
    for index, token in enumerate(argv[1:], start=1):
        if _inline_flag(token, flags) and index + 1 < len(argv):
            payloads.append(argv[index + 1])
    return payloads


def _inline_flag(token: str, flags: tuple[str, ...]) -> str | None:
    """The inline-code flag this token carries, or None.

    Short flags cluster: ``bash -lc "rm -rf x"`` is ``bash -c`` with a login shell, and reading
    only exact matches would file it as ``bash <script>`` and never recurse into the payload.
    """
    if token in flags:
        return token
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
        subcommand = _first_subcommand(head, rest)
        if subcommand:
            base = f"{head} {subcommand}"

    if head == "sed":
        for canonical, spellings in SED_MODE_FLAGS:
            if any(_matches_flag(token, spellings) for token in rest):
                return f"sed {canonical}", False

    if head in settings.inline_by_nature:
        return base, True

    if head in settings.interpreter_commands or base in settings.interpreter_commands:
        key = base if base in settings.interpreter_commands else head
        mode, inline = _interpreter_mode(key, rest, settings)
        if mode:
            return f"{key} {mode}", inline
        return key, False

    flags = _destructive_flags(base, rest, settings)
    return base + flags, False


def _interpreter_mode(
    key: str, rest: list[str], settings: GateSettings
) -> tuple[str, bool]:
    """``-c`` (inline), ``-m MOD``, ``<script>`` or nothing — the four keys of critic finding 5."""
    inline_flags = settings.inline_code_flags.get(key, ())
    index = 0
    while index < len(rest):
        token = rest[index]
        flag = _inline_flag(token, inline_flags)
        if flag:
            return flag, True
        if token == "-m" and index + 1 < len(rest):
            return f"-m {rest[index + 1]}", False
        if not token.startswith("-"):
            return "<script>", False
        index += 1
    return "", False


def _first_subcommand(head: str, rest: list[str]) -> str | None:
    """First positional argument, skipping global flags and their values (PRD §3.2 step 7)."""
    value_flags = SUBCOMMAND_VALUE_FLAGS.get(head, frozenset())
    index = 0
    while index < len(rest):
        token = rest[index]
        if token.startswith("-"):
            index += 2 if token in value_flags else 1
            continue
        return token
    return None


def _destructive_flags(base: str, rest: list[str], settings: GateSettings) -> str:
    """Canonical flag suffix for a verb in ``settings.destructive_flag_verbs``, else ``""``."""
    canonical = settings.destructive_flag_verbs.get(base)
    if not canonical:
        return ""
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


def _matches_flag(token: str, spellings: tuple[str, ...]) -> bool:
    """A flag matches its own spelling or a long form with an attached value/suffix."""
    return any(
        token == spelling or token.startswith(f"{spelling}=") or
        (len(spelling) == 2 and spelling == "-i" and token.startswith("-i"))
        for spelling in spellings
    )


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
    rest = list(argv[1:])
    positionals = [token for token in rest if not token.startswith("-")]
    if head in ("tee", "touch", "mkdir"):
        return positionals
    if head in ("cp", "mv", "install", "rsync"):
        return positionals[-1:] if len(positionals) >= 2 else positionals
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
