"""Phase 10 (SHELL-01): the git gaps the read-only tier never saw.

``git diff|log|show --output=F`` writes a file, ``git diff --no-index A B`` reads two arbitrary
filesystem paths, and four more git config key families run a program on a later, ordinary
command. The first block is the golden for what must NOT move: the plain reads stay ALLOW in every
mode (D-20), measured against the unmodified tree before the classifier was edited.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from localharness.agent.gate import tool_meta_from_schema
from localharness.agent.gate_types import (
    GIT_CONFIG_DANGEROUS_KEYS_DEFAULT,
    UNPARSED_SHELL_SIGNATURE,
    GateSettings,
    Verdict,
)
from localharness.agent.shell_classify import classify_shell
from localharness.agent.verdict import GateContext, evaluate
from localharness.tools.builtin.bash_tool import BashExecTool

SETTINGS = GateSettings()
MODES = ("auto", "guarded", "trusted", "read-only", "unattended")
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


def _segments(command: str):
    return classify_shell(command, SETTINGS).segments


def _targets(command: str) -> tuple[str, ...]:
    return tuple(t for segment in _segments(command) for t in segment.write_targets)


# ------------------------------------------------------------------ the plain-read golden

PLAIN_READS = (
    "git diff",
    "git diff HEAD~1 -- x",
    "git diff --stat",
    "git diff --cached",
    "git log -p -3",
    "git log --oneline -5",
    "git show HEAD",
    "git show HEAD:README.md",
)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("command", PLAIN_READS)
def test_plain_git_reads_stay_allow_in_every_mode(command, mode, ws):
    """D-20: written and run against the unmodified tree first; it must pass there and after."""
    assert _verdict(command, ws, mode).verdict is Verdict.ALLOW


@pytest.mark.parametrize("command", PLAIN_READS)
def test_plain_git_reads_are_one_read_only_segment_without_targets(command):
    (segment,) = _segments(command)
    assert segment.read_only is True
    assert segment.write_targets == ()
    assert segment.inline_interpreter is False


# ------------------------------------------------------------------ --output / --no-index targets

@pytest.mark.parametrize("command,expected", [
    ("git diff --output=out.patch", ("out.patch",)),
    ("git diff --output out.patch", ("out.patch",)),
    ("git log -p --output=l.txt", ("l.txt",)),
    ("git show HEAD --output=s.txt", ("s.txt",)),
    ("git -C sub diff --output=o", ("sub/o",)),  # CR-54: git applies -C before it opens the file
    ("git diff --no-index a b", ("a", "b")),
    ("git diff --no-index -- a b", ("a", "b")),
    ("git diff --stat --no-index a b", ("a", "b")),
    ("git diff -- --output", ()),
    ("git diff -- --output=x", ()),
    ("git diff --output-indicator-new=+", ()),
    ("git diff HEAD~1 -- x", ()),
])
def test_output_and_no_index_report_their_targets(command, expected):
    assert _targets(command) == expected


@pytest.mark.parametrize("command,expected", [
    ("git -C ~ log --output=.bashrc", ("~/.bashrc",)),
    ("git -C ~/.ssh show --output=authorized_keys HEAD", ("~/.ssh/authorized_keys",)),
    ("git -C /etc diff --output=cron.d/x", ("/etc/cron.d/x",)),
    ("git -C a -C b show --output=x", ("a/b/x",)),
    ("git -C a -C /abs show --output=x", ("/abs/x",)),
    ("git -C ~ diff --no-index --output=.bashrc a b", ("~/.bashrc", "~/a", "~/b")),
    ("git -C sub diff --output=/tmp/o", ("/tmp/o",)),
    ("git -C $D log --output=x", ("$D/x",)),
    ("git -c core.pager=cat -C sub log --output=x", ("sub/x",)),
])
def test_dash_c_moves_the_output_and_no_index_targets_into_its_directory(command, expected):
    assert _targets(command) == expected


@pytest.mark.parametrize("mode", ("auto", "guarded", "trusted"))
def test_output_through_dash_c_into_a_protected_path_asks(mode, ws):
    result = _verdict("git -C ~ log --output=.bashrc", ws, mode)
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "protected-path"


def test_output_through_dash_c_is_denied_read_only(ws):
    assert _verdict("git -C ~ diff --no-index --output=.bashrc a b", ws, "read-only").verdict is Verdict.DENY


def test_a_machine_override_of_write_shaped_commands_is_unchanged_for_git():
    """git stays in the write-shaped family: dropping it drops these too, as for clone/init."""
    settings = GateSettings(write_shaped_commands=frozenset())
    targets = tuple(
        t for s in classify_shell("git diff --output=o", settings).segments for t in s.write_targets
    )
    assert targets == ()


@pytest.mark.parametrize("mode", ("auto", "guarded", "trusted"))
def test_output_into_a_protected_path_asks(mode, ws):
    result = _verdict("git diff --output=~/.bashrc", ws, mode)
    assert result.verdict is Verdict.ASK
    assert result.request is not None and result.request.klass == "protected-path"


def test_output_into_a_protected_path_is_denied_read_only(ws):
    assert _verdict("git diff --output=~/.bashrc", ws, "read-only").verdict is Verdict.DENY


def test_output_into_a_protected_path_is_allowed_unattended(ws):
    assert _verdict("git diff --output=~/.bashrc", ws, "unattended").verdict is Verdict.ALLOW


def test_no_index_outside_the_boundary_asks_in_guarded(ws):
    result = _verdict("git diff --no-index /etc/hosts /etc/passwd", ws, "guarded")
    assert result.verdict is Verdict.ASK


# ------------------------------------------------------------------ the dangerous keys

NEW_KEY_WRITES = (
    "git config diff.tc.textconv ./x",
    "git config pager.log ./x",
    "git config --global pager.diff 'sh -c evil'",
    "git config credential.https://x.helper ./x",
    "git config filter.x.process ./x",
)


@pytest.mark.parametrize("command", NEW_KEY_WRITES)
def test_new_dangerous_key_writes_are_destructive(command):
    (segment,) = _segments(command)
    assert segment.destructive is True
    assert segment.signature.startswith("git config ")
    assert segment.read_only is False


def test_the_four_globs_are_defaults():
    for glob in ("diff.*.textconv", "pager.*", "credential.*.helper", "filter.*.process"):
        assert glob in GIT_CONFIG_DANGEROUS_KEYS_DEFAULT


def test_inline_dash_c_pager_adds_a_destructive_segment():
    segments = _segments("git -c pager.log=./x log")
    assert [s.signature for s in segments] == ["git log", "git config pager.log"]
    assert any(s.destructive for s in segments)


def test_reading_a_dangerous_key_stays_read_only():
    (segment,) = _segments("git config --get pager.log")
    assert segment.read_only is True and segment.destructive is False


def test_an_ordinary_inline_config_is_unchanged():
    assert [s.signature for s in _segments("git -c color.ui=always log")] == ["git log"]


def test_no_index_reports_operands_on_either_side_of_the_flag():
    """git permutes options, so `git diff /etc/shadow --no-index /dev/null` reads /etc/shadow."""
    assert _targets("git diff /etc/shadow --no-index /dev/null") == ("/etc/shadow", "/dev/null")


# ------------------------------------------------------------------ continuations (D-54, T-10-16)

def test_a_dangerous_word_split_by_a_continuation_still_classifies_dangerous():
    split = _segments("git conf\\\nig --global core.pager x")
    joined = _segments("git config --global core.pager x")
    assert [s.argv for s in split] == [s.argv for s in joined]
    assert split[0].argv[1] == "config"
    assert _targets("git diff --out\\\nput=/tmp/x") == _targets("git diff --output=/tmp/x") == ("/tmp/x",)


def test_ordinary_multiline_commands_classify_as_their_joined_form():
    assert [s.argv for s in _segments("git log \\\n  --oneline \\\n  -5")] == [
        s.argv for s in _segments("git log   --oneline   -5")
    ]
    assert [s.argv for s in _segments("ls -la\ngit status")] == [("ls", "-la"), ("git", "status")]


def test_a_continuation_inside_single_quotes_is_kept():
    assert _segments("echo 'a\\\nb'")[0].argv == ("echo", "a\\\nb")


# ------------------------------------------------------------------ payloads and comments (CR-01, CR-03)

PAYLOAD_ROWS = (
    ("bash -c 'r\\\nm -rf /etc/x'", "bash -c 'rm -rf /etc/x'"),
    ("eval 'r\\\nm -rf /etc/x'", "eval 'rm -rf /etc/x'"),
    ("sh -c 'g\\\nit push --force'", "sh -c 'git push --force'"),
    ("sh -c 'mk\\\nfs /dev/sda'", "sh -c 'mkfs /dev/sda'"),
    ("ls | xargs bash -c 'r\\\nm -rf /etc/x'", "ls | xargs bash -c 'rm -rf /etc/x'"),
)


@pytest.mark.parametrize("split,joined", PAYLOAD_ROWS)
def test_a_continuation_inside_a_lifted_payload_is_joined_before_it_is_classified(split, joined):
    """The inner shell joins `r\\<newline>m`, so the payload must classify as `rm`, not as `r` and `m`."""
    assert any(s.destructive for s in _segments(joined))
    assert [(s.signature, s.destructive) for s in _segments(split)] == [
        (s.signature, s.destructive) for s in _segments(joined)
    ]


@pytest.mark.parametrize("mode", ("auto", "guarded"))
@pytest.mark.parametrize("split,joined", PAYLOAD_ROWS)
def test_a_split_destructive_payload_is_not_allowed(split, joined, mode, ws):
    assert _verdict(split, ws, mode).verdict is _verdict(joined, ws, mode).verdict
    assert _verdict(split, ws, mode).verdict is not Verdict.ALLOW


COMMENT_ROWS = (
    "echo a # c\\\nrm -rf /etc/x",
    "ls # don't\nrm -rf /etc/x",
    'ls # "\nrm -rf /etc/x',
    "ls # (\nrm -rf /etc/x",
    "ls # {\nrm -rf /etc/x",
    "ls # note \\\nrm -rf /etc/x",
)


@pytest.mark.parametrize("command", COMMENT_ROWS)
def test_a_comment_never_hides_the_next_line(command):
    """Bash ends a comment at the newline, whatever the comment holds; the next line is a command."""
    assert [s.signature for s in _segments(command)][-1] == "rm -rf"
    assert any(s.destructive for s in _segments(command))


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", COMMENT_ROWS)
def test_a_comment_row_has_the_verdict_of_its_control(command, mode, ws):
    control = _verdict("echo a\nrm -rf /etc/x", ws, mode)
    assert _verdict(command, ws, mode).verdict is control.verdict
    assert control.verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command,joined", [
    ('echo "a # b"', 'echo "a # b"'),
    ("echo a#b", "echo a#b"),
    ("echo '#' x", "echo '#' x"),
    ("echo \\#x", "echo \\#x"),
    ("echo $# ${#HOME}", "echo $# ${#HOME}"),
    ("echo ${HOME/ #x/y} z", "echo ${HOME/ #x/y} z"),
    ("echo a\\\n#b", "echo a#b"),
    ("sed 's/a/#b/' f", "sed 's/a/#b/' f"),
    ("echo $((16#ff))", "echo $((16#ff))"),
    ("ls # a comment", "ls "),
    ("ls;# a comment\nls", "ls;\nls"),
])
def test_comment_stripping_leaves_non_comments_alone(command, joined):
    from localharness.agent.shell_classify import _join_continuations

    assert _join_continuations(command) == joined


# ------------------------------------------------------------------ invented comments (CR-01, CR-02, CR-51, CR-03; iteration 2)

NOT_A_COMMENT_ROWS = (
    "echo $(true)#c ; rm -rf /etc/x",
    "echo $((1))#c ; rm -rf /etc/x",
    "cat <(true)#c ; rm -rf /etc/x",
    "x=(a b)#c; rm -rf /etc/x",
    "echo $(ls)#x; rm -rf /etc/x",
    "echo a #c ; rm -rf /etc/x",
    "echo a #c ; rm -rf /etc/x",
    "echo a\u0085#c ; rm -rf /etc/x",
    "echo a\x1f#c ; rm -rf /etc/x",
    "echo a\x0b#c ; rm -rf /etc/x",
    "echo a\x0c#c ; rm -rf /etc/x",
    "echo a\r#c ; rm -rf /etc/x",
    "echo `ls # c` ; rm -rf /etc/x",
    "echo `ls # c`#d ; rm -rf /etc/x",
)
"""A ``#`` that bash does not read as a comment: the stripper used to drop it and the rest of the line,
`; rm -rf /etc/x` included. Each has the verdict of the same command with the comment removed."""


@pytest.mark.parametrize("command", NOT_A_COMMENT_ROWS)
def test_an_invented_comment_never_hides_the_command_after_it(command):
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", NOT_A_COMMENT_ROWS)
def test_an_invented_comment_row_is_not_allowed(command, mode, ws):
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


QUOTED_SUBSTITUTION_COMMENT_ROWS = (
    'echo "$(echo a # c\\\nrm -rf /etc/x)"',
    'echo "$(echo a # don\'t\nrm -rf /etc/x)"',
    'echo "$(echo a # c\nrm -rf /etc/x)"',
    "echo $(echo a # don't\nrm -rf /etc/x)",
    "echo `echo a # c\nrm -rf /etc/x`",
)


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", QUOTED_SUBSTITUTION_COMMENT_ROWS)
def test_a_comment_inside_a_substitution_never_hides_its_next_line(command, mode, ws):
    """A `$(...)` is a shell of its own, also inside double quotes: its comment ends at the newline."""
    assert any(s.signature == "rm -rf" for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command,joined", [
    ("echo $(true)#c", "echo $(true)#c"),
    ("echo $((1))#c ; ls", "echo $((1))#c ; ls"),
    ("echo a #c ; ls", "echo a #c ; ls"),
    ("echo a\r#c ; ls", "echo a\r#c ; ls"),
    ("(ls)#c\nrm x", "(ls)\nrm x"),  # the `)` of a bare group ends the word (WR-54, CR-02)
    ("(ls) #c\nrm x", "(ls) \nrm x"),
    ("(#c\nls)", "(\nls)"),
    ("echo `ls # c` x", "echo `ls ` x"),
    ("echo `ls #c\nls` x", "echo `ls \nls` x"),
    ("echo $(ls # c\nrm x)", "echo $(ls \nrm x)"),
    ('echo "$(ls # c\nrm x)"', 'echo "$(ls \nrm x)"'),
    ('echo "$(ls # c\\\nrm x)"', 'echo "$(ls \nrm x)"'),
    ("echo ${x:-a # b} ; ls", "echo ${x:-a # b} ; ls"),
    ("echo $((a # b)) ; ls", "echo $((a # b)) ; ls"),
    ("echo 'a\nb # c'", "echo 'a\nb # c'"),
])
def test_comment_stripping_follows_bashs_own_word_starts(command, joined):
    from localharness.agent.shell_classify import _join_continuations

    assert _join_continuations(command) == joined


@pytest.mark.parametrize("command", [
    "echo it's ; rm -rf /etc/x",
    "echo 'unterminated",
    'echo "unterminated ; rm -rf /etc/x',
    "echo $(ls ; rm -rf /etc/x",
    "echo `ls ; rm -rf /etc/x",
    "echo ${x:-a ; rm -rf /etc/x",
])
@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
def test_text_whose_quoting_does_not_balance_is_never_silent(command, mode, ws):
    """Bash refuses it, and the classifier cannot trust where its commands begin: ask in every mode."""
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


# ------------------------------------------------------------------ heredocs (CR-52, CR-55; iteration 2)

COMMENT_HEREDOC_ROWS = (
    ("ls # <<E\ntouch y\nE", "touch"),
    ("ls # <<'X'\ntouch y", "touch"),
    ("ls # note <<-EOF\ntouch y\nEOF", "touch"),
    ("ls ; # <<E\ntouch y\nE", "touch"),
    ('echo "a\\" <<E"\ntouch y', "touch"),
)


@pytest.mark.parametrize("command,expected", COMMENT_HEREDOC_ROWS)
def test_a_heredoc_operator_inside_a_comment_opens_no_heredoc(command, expected):
    """Bash never reads `<<E` in a comment, so the next line is a command and must be classified."""
    assert expected in [s.signature for s in _segments(command)], command


@pytest.mark.parametrize("mode", ("read-only", "guarded"))
@pytest.mark.parametrize("command,expected", COMMENT_HEREDOC_ROWS[:3])
def test_a_commented_heredoc_row_is_not_allowed(command, expected, mode, ws):
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


OPERATOR_AFTER_DELIMITER_ROWS = (
    ("cat <<E|sh\necho hi\nE", "sh"),
    ("cat <<E;touch y\nbody\nE", "touch"),
    ("cat <<E; touch y\nbody\nE", "touch"),
    ("cat <<E&&touch y\nbody\nE", "touch"),
    ("cat <<'E'|sh\nbody\nE", "sh"),
    ("cat <<-E;touch y\n\tbody\n\tE", "touch"),
    ("cat <<E\nbody\nE\ntouch y", "touch"),
)


@pytest.mark.parametrize("command,expected", OPERATOR_AFTER_DELIMITER_ROWS)
def test_the_heredoc_delimiter_stops_at_a_shell_operator(command, expected):
    """`cat <<E|sh` has the delimiter `E`; the old reader took `E|sh` and hid the sink."""
    assert expected in [s.signature for s in _segments(command)], command


@pytest.mark.parametrize("mode", ("read-only", "guarded"))
def test_a_heredoc_piped_to_a_shell_is_not_read_only(mode, ws):
    assert _verdict("cat <<E|sh\necho hi\nE", ws, mode).verdict is not Verdict.ALLOW


def test_a_heredoc_body_is_still_data():
    assert [s.signature for s in _segments("cat <<E\nrm -rf /etc/x\nE")] == ["cat"]
    assert [s.signature for s in _segments("cat <<E >f\nrm -rf /etc/x\nE")] == ["cat"]


# ------------------------------------------------------------------ a `#` inside a word is not a comment (CR-53; iteration 2)

def test_a_hash_inside_a_word_does_not_hide_the_flags_after_it():
    """shlex ended a word at any `#`; bash does not, and `--output`, `-delete`, `--force` were lost."""
    assert _targets("git diff a#b --output=x") == ("x",)
    assert _targets("git diff a#b --output=~/.bashrc") == ("~/.bashrc",)
    assert [s.signature for s in _segments("find . a#b -delete")] == ["find -delete"]
    assert [s.signature for s in _segments("git push origin feat#12 --force")] == ["git push --force"]
    assert [s.signature for s in _segments("rm a#b -rf ~/z")] == ["rm -rf"]


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", (
    "git diff a#b --output=~/.bashrc",
    "git push origin feat#12 --force",
    "cat a#b ~/.ssh/id_rsa",
))
def test_a_hash_inside_a_word_row_has_the_verdict_of_its_control(command, mode, ws):
    control = command.replace("a#b", "ab").replace("feat#12", "feat12")
    assert _verdict(command, ws, mode).verdict is _verdict(control, ws, mode).verdict


# ------------------------------------------------------------------ ANSI-C quoting and ${...} (CR-54, WR-01, CR-53; iteration 2)

ANSI_C_HIDING_ROWS = (
    "echo $'a\\'' ; rm -rf /etc/x",
    "echo $'a\\'b' ; rm -rf /etc/x",
    "echo $'it\\'s'\nrm -rf /etc/x",
    "echo $'it\\'s' ; rm -rf /etc/x",
    "echo ${x:-(} ; rm -rf /etc/x",
    "echo ${x:-{} ; rm -rf /etc/x",
    "echo ${x:-)} ; rm -rf /etc/x",
    "echo {a ; rm -rf /etc/x",
    "echo $'(' ; rm -rf /etc/x",
)


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", ANSI_C_HIDING_ROWS)
def test_ansi_c_quotes_and_odd_expansions_hide_nothing(command, mode, ws):
    """`$'a\\''` is one word, `${x:-(}` opens no group: the `rm -rf` after them is seen and asked about."""
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command,signature", [
    ("$'rm' -rf /etc/x", "rm -rf"),
    ("r$''m -rf /etc/x", "rm -rf"),
    ("git $'push' --force origin main", "git push --force"),
    ("$'\\x72m' -rf /etc/x", "rm -rf"),
    ("$'\\162m' -rf /etc/x", "rm -rf"),
    ("git $'p'$'ush' --force origin main", "git push --force"),
])
def test_an_ansi_c_spelled_program_signs_as_the_program(command, signature):
    assert [s.signature for s in _segments(command)] == [signature]


@pytest.mark.parametrize("command", ("cut -d$'\\t' -f1 f", "grep $'a\\tb' f", "echo $'x' \"$'y'\""))
def test_ordinary_ansi_c_words_stay_read_only(command):
    assert all(s.read_only for s in _segments(command))


@pytest.mark.parametrize("command,joined", [
    ("echo $'a\\'b' ; ls", "echo 'a'\\''b' ; ls"),
    ("echo $'a\\tb'", "echo 'a\tb'"),
    ("echo $'\\x41\\101'", "echo 'AA'"),
    ('echo "$\'x\'"', 'echo "$\'x\'"'),
    ("echo '$'\\''x'", "echo '$'\\''x'"),
    ('echo $"a"', 'echo "a"'),
])
def test_ansi_c_words_are_decoded_into_plain_single_quotes(command, joined):
    from localharness.agent.shell_classify import _join_continuations

    assert _join_continuations(command) == joined


def test_an_ansi_c_operator_is_not_a_heredoc():
    """`<<E` inside `$'…\\''…` is text; a heredoc scanner that closed the string at `\\'` read it as one."""
    command = "echo $'a\\'<<E'\ntouch y\nE"
    assert "touch" in [s.signature for s in _segments(command)]


def test_a_brace_that_is_not_a_group_does_not_swallow_the_rest():
    assert "touch" in [s.signature for s in _segments("{#;touch y")]
    assert "touch" in [s.signature for s in _segments("echo {a; touch y")]
    assert [s.signature for s in _segments("{ ls; }; touch y")] == ["ls", "touch"]
    assert [s.signature for s in _segments("f() { echo {a,b}; }; f")] == ["echo", "f"]


# ------------------------------------------------------------------ bash 5.3 funsubs (CR-56; iteration 2)

FUNSUB_ROWS = (
    "echo ${ rm -rf /etc/x; }",
    "echo ${| rm -rf /etc/x; }",
    'echo "${ rm -rf /etc/x; }"',
    "x=${ rm -rf /etc/x; }",
    "echo ${ { rm -rf /etc/x; }; }",
    "echo ${ ls # c\nrm -rf /etc/x\n}",
)


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", FUNSUB_ROWS)
def test_a_funsub_body_is_classified_like_a_command_substitution(command, mode, ws):
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


def test_a_funsub_of_a_read_is_read_only_and_a_plain_expansion_is_unchanged():
    assert all(s.read_only for s in _segments("echo ${ ls; } ${x:-a} ${#y}"))
    assert [s.signature for s in _segments("echo ${x:-$(rm -rf /etc/x)}")] == ["rm -rf", "echo"]


# ------------------------------------------------------------------ writes from an unknown directory (WR-53; iteration 2)

@pytest.mark.parametrize("command", (
    "cd $HOME && echo x > .bashrc",
    'cd "$X"; echo x > f',
    "cd $(pwd) && echo x > f",
    "popd; echo x > f",
    "cd - && echo x > f",
))
def test_a_write_after_a_cd_the_classifier_cannot_read_is_never_silent_in_guarded(command, ws):
    assert _verdict(command, ws, "guarded").verdict is Verdict.ASK


@pytest.mark.parametrize("command", ("cd sub && echo x > f", "cd ./sub/.. && echo x > f", "echo x > f"))
def test_a_write_after_a_readable_cd_inside_the_workspace_is_unchanged(command, ws):
    assert _verdict(command, ws, "guarded").verdict is Verdict.ALLOW


@pytest.mark.parametrize("command", (
    "f(){ rm -rf /etc/x; }; f",
    "f() { rm -rf /etc/x; }; f",
    "function f { rm -rf /etc/x; }; f",
    "{ rm -rf /etc/x;}",
    "true && { rm -rf /etc/x; touch y; }",
    "if true; then { rm -rf /etc/x; }; fi",
))
def test_a_brace_group_in_any_ordinary_spelling_is_recursed_into(command):
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command


# ------------------------------------------------------------------ the stripper can not hide a command (T-10-16; iteration 2)

CONTEXTS_WHERE_HASH_IS_NOT_A_COMMENT = (
    "[[ a =~ (a|#b) ]]; rm -rf /etc/x",
    "[[ a =~ a|#b ]] && rm -rf /etc/x",
    "shopt -s extglob\necho @(a|#b); rm -rf /etc/x",
    "shopt -s extglob\necho +(#b); rm -rf /etc/x",
    "(( 0 # )); rm -rf /etc/x",
    "(( 1 # x )) && rm -rf /etc/x",
    'bash -c "[[ a =~ (a|#b) ]]; rm -rf /etc/x"',
    "echo $(( 1 # x )); rm -rf /etc/x",
)
"""Real bash runs the `rm -rf` in each (verified on 5.3.9): the `#` is part of a regex, an extglob group or
an arithmetic expression, not a comment. The stripper reads it as one; the dropped text holds a `;` or `&`,
so the command is classified again with every `#` kept and the union is what the verdict sees."""


@pytest.mark.parametrize("mode", ("read-only", "guarded", "auto"))
@pytest.mark.parametrize("command", CONTEXTS_WHERE_HASH_IS_NOT_A_COMMENT)
def test_a_hash_in_a_context_the_stripper_does_not_know_still_shows_the_command_after_it(command, mode, ws):
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command", (
    "ls # list the directory",
    "git status # what changed? (short)",
    "ls -la # a < b > c",
    "echo a # don't",  # an apostrophe alone leaves the text balanced after the comment is dropped
    "cat f # the file\ngrep x f # and a match",
))
def test_an_ordinary_comment_is_dropped_and_changes_nothing(command):
    from localharness.agent.shell_classify import _join

    assert _join(command)[1] is False
    uncommented = "\n".join(line.split(" #")[0] for line in command.split("\n"))
    assert [s.signature for s in _segments(command)] == [s.signature for s in _segments(uncommented)]


def test_a_comment_that_holds_a_separator_is_also_read_with_the_comment_left_in():
    from localharness.agent.shell_classify import _join

    assert _join("ls # a; b")[1] is True
    assert _join("ls # a | b")[1] is True
    assert _join("ls # a `b`")[1] is True
    assert _join("ls # a $(b)")[1] is True
    assert _join("ls # ab")[1] is False
    assert _join("ls # a; b", strip_comments=False)[0] == "ls # a; b"


# ------------------------------------------------------------------ heredocs read inside the join scan (iteration 3)

X = "rm -rf /etc/x"
THREE_MODES = ("read-only", "guarded", "auto")

HEREDOC_LAYOUT_ROWS = (
    # CR-01 / CR-52: a continuation after the operator, a quote spanning lines, a quoted newline delimiter
    f"cat <<E \\\n; {X}\nhi\nE",
    f"cat <<E\\\n; {X}",
    f"cat <<E\\\n;{X}\nE\n",
    f"cat <<E&&\\\n{X}\necho",
    f'echo "multi\nline <<E text"\n{X}',
    f"echo 'multi\nline <<E text'\n{X}",
    f'echo "\n<<E "\n{X}\nE\n',
    f'echo "a\n<<E b"\n{X}\nE',
    f"cat <<'\n' ; {X}",
    # CR-02: `)#` opens a comment, so `<<E` after it is text
    f"(true)#<<E\n{X}\nE",
    f"(true)#<<'X'\n{X}",
    f"(true) #<<E\n{X}\nE",
    # CR-03: two delimiter spellings and the arithmetic that is not `$((`
    f"cat <<$'E'\nbody\nE\n{X}",
    f'cat <<$"E"\nbody\nE\n{X}',
    f"cat <<$'E\\x46'\nbody\nEF\n{X}",
    f"echo $[1<<2]\n{X}",
    f"a[1<<1]=3\n{X}",
    f"echo $(( 1 << 2 )); {X}",
    f"(( 1 << 2 )); {X}",
    # WR-54: a bare group is followed by a real comment
    f"({X})#c",
    f"ls\n({X})#c",
)


@pytest.mark.parametrize("mode", THREE_MODES)
@pytest.mark.parametrize("command", HEREDOC_LAYOUT_ROWS)
def test_a_heredoc_layout_bash_reads_differently_never_hides_the_command(command, mode, ws):
    """Each row ran its `rm` in bash 5.3.9 (marker files); the old line-at-a-time scanner took it for body."""
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


def test_a_continuation_after_a_heredoc_operator_joins_the_lines_first():
    assert "sh" in [s.signature for s in _segments("cat <<E \\\n| sh\nhi\nE")]
    # bash joins the lines: `cat <<E body`, and the next line `E` ends an empty body
    assert [s.signature for s in _segments("cat <<E \\\nbody\nE")] == ["cat"]


def test_an_unterminated_heredoc_is_classified_not_swallowed(ws):
    """Bash reads the rest as the body, but a delimiter this reader got wrong would look the same."""
    assert "rm -rf" in [s.signature for s in _segments(f"cat <<E\n{X}\n")]
    assert _verdict(f"cat <<'E\n{X}", ws, "auto").verdict is not Verdict.ALLOW  # an unclosed quote asks


@pytest.mark.parametrize("command", (
    "cat <<E\nbody\nE\nls",
    "cat <<-E\n\tbody\n\tE\nls",
    "cat <<'E' > f\nrm -rf /\nE\nls",
    "cat <<A <<B\none\nA\ntwo\nB\nls",
    "cat <<$'E'\nrm -rf /\nE\nls",
    'cat <<"E"\nrm -rf /\nE\nls',
    "cat <<\\E\nrm -rf /\nE\nls",
    "cat <<E\nrm -rf /\nE\nls # <<F",
    "echo $((1<<2)) $[1<<2]; ls",
    "a=$((1<<2)); b[2]=1; ls",
    "cat <<<'x' <<<'y'; ls",
))
def test_a_well_formed_heredoc_still_swallows_its_body_and_only_it(command):
    assert "rm -rf" not in [s.signature for s in _segments(command)], command
    assert not any(s.signature == UNPARSED_SHELL_SIGNATURE for s in _segments(command)), command


COMMIT_IDIOM_ROWS = (
    "git commit -m \"$(cat <<'EOF'\nfix: don't crash\nEOF\n)\"",
    "git add -A && git commit -m \"$(cat <<'EOF'\nfix: don't crash\nEOF\n)\"",
    "gh pr create --title \"x\" --body \"$(cat <<'EOF'\n- it's a `thing`; fine\nEOF\n)\"",
    "git commit -m \"$(cat <<'EOF'\nDon't (crash\nwith `x` and \"y\" and $(z)\nEOF\n)\"",
    "git commit -m \"$(cat <<'EOF'\nDon't crash )\nEOF\n)\"",
    "git commit -m \"$(cat <<'EOF'\nit's\nEOF\n)\" && git push",
)
PAYLOAD_HEREDOC_ROWS = (
    "bash -c 'cat <<EOF\nit'\"'\"'s fine\nEOF'",
    "ssh host 'cat > f <<EOF\nit'\"'\"'s fine\nEOF'",
    "docker exec c sh -c 'cat > f <<EOF\nit'\"'\"'s fine\nEOF'",
    "eval \"cat <<EOF\nit's fine\nEOF\"",
)


@pytest.mark.parametrize("command", COMMIT_IDIOM_ROWS + PAYLOAD_HEREDOC_ROWS)
def test_the_commit_idiom_and_heredoc_payloads_are_not_an_unparsed_ask(command, ws):
    """WR-01 / WR-51: iteration 2 made `don't` in a heredoc body an ungrantable ask in every mode."""
    assert not any(s.signature == UNPARSED_SHELL_SIGNATURE for s in _segments(command)), command
    assert _verdict(command, ws, "auto").verdict is not Verdict.DENY
    assert _verdict(command, ws, "guarded").verdict is not Verdict.DENY


@pytest.mark.parametrize("command", COMMIT_IDIOM_ROWS[:3])
def test_the_commit_idiom_classifies_as_the_same_command_without_the_apostrophe(command):
    control = command.replace("don't", "dont").replace("it's", "its")
    assert [s.signature for s in _segments(command)] == [s.signature for s in _segments(control)]


def test_the_heredoc_body_of_a_commit_message_is_data_in_auto(ws):
    command = "git commit -m \"$(cat <<'EOF'\nfix: don't crash\nEOF\n)\""
    assert _verdict(command, ws, "auto").verdict is Verdict.ALLOW


def test_a_newline_in_a_nested_context_while_a_heredoc_is_owed_is_never_silent(ws):
    """The body of `cat <<E $(echo a<nl>b)` starts after which newline? Not guessed: the text is flagged."""
    command = f"cat <<E $(echo a\nb)\nbody\nE\n{X}"
    assert any(s.signature == "rm -rf" for s in _segments(command))
    assert _verdict(command, ws, "auto").verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command,joined", [
    ("cat <<E\nbody\nE\nls", "cat  \nls"),
    ("cat <<E >f\nbody\nE", "cat   >f\n"),
    ("cat <<-E\n\tbody\n\tE\nls", "cat  \nls"),
    ("echo $(cat <<'E'\nit's\nE\n) ; ls", "echo $(cat  \n) ; ls"),
    ("cat <<E \\\n; ls\nbody\nE", "cat   ; ls\n"),
    ("echo $((1<<2)) ; ls", "echo $((1<<2)) ; ls"),
    ("(true)#c\nls", "(true)\nls"),
])
def test_the_join_strips_heredocs_with_the_text_it_reads(command, joined):
    from localharness.agent.shell_classify import _join_continuations

    assert _join_continuations(command) == joined


# ------------------------------------------------------------------ an unquoted heredoc body runs its substitutions (CR-51; iteration 3)

UNQUOTED_BODY_ROWS = (
    f"cat <<E\n$({X})\nE",
    f"cat <<E\nhello `{X}`\nE",
    f"read -r l <<E\n$({X})\nE\necho \"$l\"",
    f"cat <<E\n${{ {X}; }}\nE",
    f"cat <<E\nx ${{y:-$({X})}} z\nE",
    f"cat <<-E\n\t$({X})\n\tE",
    f"cat <<E > f\n$(echo $({X}))\nE",
    f"cat <<A <<'B'\n$({X})\nA\nbody\nB",
)


@pytest.mark.parametrize("mode", THREE_MODES)
@pytest.mark.parametrize("command", UNQUOTED_BODY_ROWS)
def test_the_substitutions_in_an_unquoted_heredoc_body_are_classified(command, mode, ws):
    """Bash expands the body of `<<E`: `read -r l <<E` / `$(rm x)` / `E` runs the `rm` (proof run)."""
    assert any(s.signature == "rm -rf" and s.destructive for s in _segments(command)), command
    assert _verdict(command, ws, mode).verdict is not Verdict.ALLOW


@pytest.mark.parametrize("command", (
    f"cat <<'E'\n$({X})\nE",
    f'cat <<"E"\n$({X})\nE',
    f"cat <<\\E\n$({X})\nE",
    f"cat <<$'E'\n$({X})\nE",
    f"cat <<E\n\\$({X})\nE",
    f"cat <<E\n\\`{X}\\`\nE",
    "cat <<E\n${HOME} $HOME ${x:-a} $((1+2))\nE",
    "cat <<'A' <<B\n$(rm -rf /)\nA\nplain\nB",
))
def test_a_quoted_or_escaped_heredoc_body_is_still_data(command):
    assert [s.signature for s in _segments(command)] == ["cat"], command


def test_an_ordinary_unquoted_body_adds_only_what_it_runs(ws):
    command = "cat > f <<EOF\nbuilt on $(date) in ${HOME}\nEOF"
    assert [s.signature for s in _segments(command)] == ["date", "cat"]
    assert _verdict(command, ws, "auto").verdict is Verdict.ALLOW
    assert _targets(command) == ("f",)
    assert _targets("cat <<E > f\n$(date)\nE") == ("f",)
