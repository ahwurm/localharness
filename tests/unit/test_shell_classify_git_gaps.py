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
from localharness.agent.gate_types import GIT_CONFIG_DANGEROUS_KEYS_DEFAULT, GateSettings, Verdict
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
    ("git -C sub diff --output=o", ("o",)),
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
