"""Phase 10 (SHELL-02): the write-shaped and exec-shaped spellings of the read-only tier.

``find -fprint*``, ``sort -o``, ``uniq IN OUT``, ``tree -o`` and ``sed w`` write a file; ``rg
--pre``, ``sort --compress-program``, ``git grep -O`` and ``sed e`` run a program the model
names. Three tables below: what writes (command -> targets), what runs a program (inline, never
read-only), and the ordinary reads that must keep their signature, their class and no target.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import GateSettings, Verdict
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import GateContext, evaluate
from localharness.tools.builtin.bash_tool import BashExecTool

SETTINGS = GateSettings()
BASH_META = tool_meta_from_schema(BashExecTool().info())


def _no_grants(workspace: Path, klass: str, key: str):
    return None


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with its own HOME and global config dir, so nothing touches the real ones."""
    home = (tmp_path / "home").resolve()
    (home / ".localharness").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCALHARNESS_DIR", str(home / ".localharness"))
    workspace = (home / "proj").resolve()
    workspace.mkdir()
    return workspace


def _verdict(command: str, workspace: Path, mode: str):
    ctx = GateContext(
        workspace=workspace, boundary=workspace, grants=_no_grants,
        has_review_surface=True, mode=mode,  # type: ignore[arg-type]
    )
    return evaluate("bash_exec", {"command": command}, BASH_META, ctx, SETTINGS)


def _segments(command: str, settings: GateSettings = SETTINGS):
    return classify_shell(command, settings).segments


def _targets(command: str, settings: GateSettings = SETTINGS) -> tuple[str, ...]:
    return tuple(t for segment in _segments(command, settings) for t in segment.write_targets)


# ------------------------------------------------------------------ write-shaped rows

WRITE_ROWS = [
    # find: -fprint / -fprint0 / -fprintf / -fls take the next token as the file
    ("find . -fprint out.txt", ("out.txt",)),
    ("find . -fprint0 o", ("o",)),
    ("find . -fprintf o '%p'", ("o",)),
    ("find . -fls o", ("o",)),
    ("find ~ -maxdepth 0 -fprintf ~/.bashrc 'x'", ("~/.bashrc",)),
    (r"find . -exec grep -fprint x {} \;", ()),
    # sort: -o in every spelling, abbreviations of --output, clusters, after other value flags
    ("sort -o out f", ("out",)),
    ("sort -oout f", ("out",)),
    ("sort --output=out f", ("out",)),
    ("sort --output out f", ("out",)),
    ("sort --o=out f", ("out",)),
    ("sort --outp=out f", ("out",)),
    ("sort -uo out f", ("out",)),
    ("sort -roout f", ("out",)),
    ("sort -k 2 -o out f", ("out",)),
    ("sort -t o f", ()),
    ("sort -- -o f", ()),
    # uniq: the second operand is the output file
    ("uniq in out", ("out",)),
    ("uniq -c in out", ("out",)),
    ("uniq -f 1 in out", ("out",)),
    ("uniq -f1 in out", ("out",)),
    ("uniq --skip-fields=1 in out", ("out",)),
    ("uniq --skip-f=1 in out", ("out",)),
    ("uniq --skip-fields 1 in out", ("out",)),
    ("uniq -- -a out", ("out",)),
    ("uniq - out", ("out",)),
    ("uniq f", ()),
    ("uniq -f 1 f", ()),
    # tree: -o FILE
    ("tree -o out", ("out",)),
    ("tree -ao out", ("out",)),
    ("tree -oout", ("out",)),
    ("tree -L 2", ()),
]


@pytest.mark.parametrize("command,expected", WRITE_ROWS)
def test_write_rows_report_their_targets(command, expected):
    assert _targets(command) == expected


@pytest.mark.parametrize("command", [
    "find . -fprint out", "sort -o out f", "uniq in out", "tree -o out",
])
def test_targets_survive_an_emptied_write_shaped_override(command):
    """D-56: these are computed before the write_shaped_commands gate, so a machine override of
    that list neither drops them nor records a loosening."""
    settings = GateSettings(write_shaped_commands=frozenset())
    assert _targets(command, settings) == ("out",)


def test_none_of_the_swept_commands_joined_the_write_shaped_defaults():
    from localharness.agent.gate_types import WRITE_SHAPED_COMMANDS_DEFAULT as defaults

    assert not {"find", "sort", "uniq", "sed", "tree"} & defaults


def test_find_fprintf_into_a_protected_path_asks_in_auto(ws):
    result = _verdict("find ~ -maxdepth 0 -fprintf ~/.bashrc 'x'", ws, "auto")
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "protected-path"


# ------------------------------------------------------------------ inline rows

INLINE_ROWS = [
    "rg --pre cat x",
    "rg --pre=cat x",
    "rg --hostname-bin x y",
    "rg --hostname-bin=x y",
    "sort --compress-program=gzip f",
    "sort --compress-program gzip f",
    "sort --compress=gzip f",
    "sort --co=gzip f",
    "git grep -O x",
    "git grep -Ovim x",
    "git grep -nO x",
    "git grep -nOvim x",
    "git grep --open-files-in-pager x",
    "git grep --open-files-in-pager=vim x",
    "git grep --op x",
    "git grep --open=vim x",
]


@pytest.mark.parametrize("command", INLINE_ROWS)
def test_program_naming_flags_are_inline_and_never_read_only(command):
    (segment,) = _segments(command)
    assert segment.inline_interpreter is True
    assert segment.read_only is False


def test_rg_pre_asks_interpreter_inline_in_guarded(ws):
    result = _verdict("rg --pre cat x", ws, "guarded")
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "interpreter-inline"


def test_rg_pre_is_denied_read_only(ws):
    assert _verdict("rg --pre cat x", ws, "read-only").verdict is Verdict.DENY


def test_an_inline_segment_is_never_read_only_even_with_a_read_only_signature():
    """D-55: `sort` is a default read-only signature; the exec flag still takes it out."""
    plain, flagged = _segments("sort f")[0], _segments("sort --compress-program=gzip f")[0]
    assert plain.signature == flagged.signature == "sort"
    assert plain.read_only is True and flagged.read_only is False


# ------------------------------------------------------------------ ordinary reads

ORDINARY_READS = [
    ("find . -name x", "find"),
    ("find . -type f -print", "find"),
    ("find . -printf '%p\\n'", "find"),
    ("sort f", "sort"),
    ("sort -n f", "sort"),
    ("sort -k 2 f", "sort"),
    ("sort -t o f", "sort"),
    ("uniq f", "uniq"),
    ("uniq -c f", "uniq"),
    ("uniq -f 1 f", "uniq"),
    ("rg x", "rg"),
    ("rg --pre-glob '*.gz' x", "rg"),
    ("tree", "tree"),
    ("tree -L 2", "tree"),
]


@pytest.mark.parametrize("command,signature", ORDINARY_READS)
def test_ordinary_reads_keep_their_class(command, signature):
    (segment,) = _segments(command)
    assert segment.signature == signature
    assert segment.read_only is True
    assert segment.write_targets == ()
    assert segment.inline_interpreter is False


def test_git_grep_without_a_pager_is_not_inline():
    (segment,) = _segments("git grep x")
    assert segment.inline_interpreter is False


@pytest.mark.parametrize("mode", ("auto", "guarded", "trusted", "read-only", "unattended"))
@pytest.mark.parametrize("command", [c for c, _ in ORDINARY_READS])
def test_ordinary_reads_stay_allow_in_every_mode(command, mode, ws):
    assert _verdict(command, ws, mode).verdict is Verdict.ALLOW


# ------------------------------------------------------------------ sed: the script is read

SED_TARGET_ROWS = [
    ("sed -n 'w out' f", ("out",)),
    ("sed -n 'W out' f", ("out",)),
    ("sed 's/a/b/w out' f", ("out",)),
    ("sed -n '/x/w out' f", ("out",)),
    ("sed -n -e 'w out' f", ("out",)),
    ("sed -n --expression='w out' f", ("out",)),
    ("sed -ne 'w out' f", ("out",)),
    ("sed -n -e p -e 'w out' f", ("out",)),
    ("sed -n 'w /dev/stdout' f", ("/dev/stdout",)),
    ("sed -i 's/a/b/w log' f", ("f", "log")),
    ("sed -i 's/a/b/' f", ("f",)),
    # WR-51: every getopt prefix of --in-place edits f in place, whatever -n says
    ("sed -n --in-pl d f", ("f",)),
    ("sed -n --in d f", ("f",)),
    ("sed -n --i p f", ("f",)),
    ("sed --in-place=.bak 's/a/b/' f", ("f",)),
    ("sed -n --in-place p f", ("f",)),
    ("sed 's/a/b/' f", ()),
    ("sed -n --quiet p f", ()),
    ("sed -n 'p;w out' f", ("out",)),
    ("sed -n '1d;$!{w out\n}' f", ("out",)),
    ("sed -n ': x w out' f", ("out",)),
]


@pytest.mark.parametrize("command,expected", SED_TARGET_ROWS)
def test_sed_script_targets(command, expected):
    assert _targets(command) == expected


SED_INLINE_ROWS = [
    "sed -n '1e cmd' f",
    "sed e f",
    "sed 's/a/b/e' f",
    "sed -n -f script.sed f",
    "sed --file=s.sed f",
    "sed -nf script.sed f",
    "sed -n --frobnicate p f",
    "sed -n 's/a' f",
    "sed -n 'k' f",
    "sed -n '/x' f",
    "sed -n 'y/abc/xy' f",
    "sed -n 'w' f",
    "sed --s p f",
]


@pytest.mark.parametrize("command", SED_INLINE_ROWS)
def test_sed_that_runs_a_command_or_cannot_be_read_is_inline(command):
    (segment,) = _segments(command)
    assert segment.inline_interpreter is True
    assert segment.read_only is False


SED_ORDINARY_READS = [
    "sed -n p f",
    "sed -n '/where/p' f",
    "sed -n -e p f",
    "sed -n '1,40p' f",
    "sed -n '10,20p;30q' f",
    "sed -n '/start/,/end/p' f",
    "sed -n '$p' f",
    "sed -n '5{p;q}' f",
    "sed -n '2~3p' f",
    "sed -n '0,/re/p' f",
    "sed -n '/a/!p' f",
    "sed -n 's/x/y/p' f",
    "sed -n 's|a/b|c|gp' f",
    r"sed -E -n 's/(a)/\1/p' f",
    "sed -n '/^#/d; p' f",
    "sed -n '/re/Ip' f",
    "sed -n 'l' f",
    "sed -n '/x/{=;p}' f",
    "sed -n 'y/abc/xyz/;p' f",
    "sed --quiet p f",
    "sed -n '2,+3p' f",
    "sed -n '/x/,~4p' f",
    "sed -n 'a hello' f",
    "sed -n 'q5' f",
    "sed -n -e '$!N' -e p f",
]


@pytest.mark.parametrize("command", SED_ORDINARY_READS)
def test_ordinary_sed_reads_keep_their_class(command):
    (segment,) = _segments(command)
    assert segment.signature == "sed -n"
    assert segment.read_only is True
    assert segment.write_targets == ()
    assert segment.inline_interpreter is False


def test_a_regex_that_contains_a_w_is_not_a_write_command():
    """Pitfall 11: the script is parsed, never substring-matched."""
    assert _targets("sed -n '/w out/p' f") == ()
    assert _targets("sed -n 's/w x/e y/p' f") == ()


def test_plain_sed_substitution_keeps_its_bare_signature():
    (segment,) = _segments("sed 's/a/b/' f")
    assert segment.signature == "sed"
    assert segment.write_targets == () and segment.inline_interpreter is False


def test_sed_e_asks_interpreter_inline_in_guarded(ws):
    result = _verdict("sed -n '1e cmd' f", ws, "guarded")
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "interpreter-inline"


def test_sed_e_is_denied_read_only(ws):
    assert _verdict("sed -n '1e cmd' f", ws, "read-only").verdict is Verdict.DENY


def test_sed_w_into_a_protected_path_asks_in_auto(ws):
    result = _verdict("sed -n 'w ~/.bashrc' f", ws, "auto")
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "protected-path"


@pytest.mark.parametrize("command", SED_ORDINARY_READS)
def test_ordinary_sed_reads_stay_allow_in_every_mode(command, ws):
    for mode in ("auto", "guarded", "trusted", "read-only", "unattended"):
        assert _verdict(command, ws, mode).verdict is Verdict.ALLOW, (command, mode)


def test_sed_append_text_continues_only_over_an_odd_backslash():
    """`a text\\` continues onto the next line (that line is text); `a text\\\\` does not (the next
    line is a command, and here a write)."""
    assert _targets("sed -n 'a foo\\\nw out' f") == ()
    assert _targets("sed -n 'a foo\\\\\nw out' f") == ("out",)


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
def test_an_abbreviated_in_place_sed_is_not_a_read(mode, ws):
    assert _verdict("sed -n --in-pl d /etc/hosts", ws, mode).verdict is not Verdict.ALLOW


def test_sed_in_place_suffix_ends_the_cluster():
    """`-ie` is `-i` with the suffix `e`, not an `-e` script; `-i.bak` likewise."""
    assert _targets("sed -ie 's/a/b/' f") == ("f",)
    assert _targets("sed -i.bak 's/a/b/' f") == ("f",)


@pytest.mark.parametrize("command,expected", [
    ("echo 'a\\\nb'", ("echo", "a\\\nb")),   # single quotes keep the pair
    ('echo "a\\\nb"', ("echo", "ab")),        # double quotes drop it
    ("echo a\\\nb", ("echo", "ab")),          # so does the bare shell
    ("echo a\\\\\nb", ("echo", "a\\", "b")),  # `\\` is a pair: the newline after it is real
])
def test_backslash_newline_is_joined_the_way_the_shell_joins_it(command, expected):
    """The sed scanner reads quoted scripts, so the pre-pass must not glue their lines together."""
    argv = tuple(
        token for segment in _segments(command) for token in segment.argv
    )
    assert argv[: len(expected)] == expected
