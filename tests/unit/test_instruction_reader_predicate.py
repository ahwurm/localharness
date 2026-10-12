"""The reader predicate behind the instructions naming rule (Phase 11, D-35, D-36, D-73, D-86).

``instruction_file_reads_only`` decides, for a ``bash_exec`` command that mentions one of the
instruction-file basenames, whether the command can only READ it. One classifier call; the
whole command is the unit; every unclear case fails toward marking (False).

Nothing here executes a command: every row goes through ``classify_shell`` in-process.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from localharness.agent import shell_classify
from localharness.agent.gate_types import GateSettings
from localharness.agent.shell_classify import (
    INSTRUCTION_FILE_READERS,
    INSTRUCTION_PIPELINE_READERS,
    classify_shell,
    instruction_file_reads_only,
)

NAMES = ("CLAUDE.md", "AGENTS.md", "INSTRUCTIONS.md")


def reads_only(command: str) -> bool:
    return instruction_file_reads_only(command, NAMES)


# --------------------------------------------------------------------------- the tracer rows


def test_a_plain_read_is_a_reader() -> None:
    assert reads_only("cat CLAUDE.md") is True


def test_a_redirect_into_the_file_is_not() -> None:
    assert reads_only("cat > CLAUDE.md") is False


def test_a_command_that_names_nothing_is_a_reader() -> None:
    assert reads_only("ls -la") is True


def test_naming_is_case_insensitive() -> None:
    assert reads_only("cat claude.MD") is True
    assert reads_only("cat > Claude.md") is False


def test_settings_none_and_explicit_settings_agree() -> None:
    assert instruction_file_reads_only("cat CLAUDE.md", NAMES, GateSettings()) is True
    assert instruction_file_reads_only("cat > CLAUDE.md", NAMES, GateSettings()) is False
    assert instruction_file_reads_only("cat CLAUDE.md", NAMES, None) is True


# --------------------------------------------------------------------------- the D-36 matrix, verbatim

NOT_MARKING = [
    "cat CLAUDE.md",
    "sed -n '1,40p' CLAUDE.md",
    "rg x CLAUDE.md",
    "grep -l x CLAUDE.md > out",
    "git add CLAUDE.md",
    'git commit -m "docs: CLAUDE.md"',
    "diff CLAUDE.md x",
    "find . -name CLAUDE.md",
    "git blame CLAUDE.md",
    "git grep x CLAUDE.md",
    "git ls-files CLAUDE.md",
    "cat CLAUDE.md | wc",
    "rg x CLAUDE.md | sort | uniq -c",
    'cat CLAUDE.md 2>/dev/null || echo "no CLAUDE.md"',
    "GIT_PAGER=cat git diff CLAUDE.md",
]

HEREDOC_INTO_FILE = "cat <<EOF > CLAUDE.md\nx\nEOF"

MARKING = [
    "cat > CLAUDE.md",
    "sed -i s/a/b/ CLAUDE.md",
    "tee CLAUDE.md",
    "git checkout -- CLAUDE.md",
    "mv x CLAUDE.md",
    "sort -o CLAUDE.md x",
    "uniq p CLAUDE.md",
    "sed -n 'w CLAUDE.md' x",
    "find . -fprintf CLAUDE.md x",
    HEREDOC_INTO_FILE,
    "python3 -c 'open(\"CLAUDE.md\",\"w\")'",
    "sh -c 'echo x > CLAUDE.md'",
    'eval "echo x > CLAUDE.md"',
    "GIT_EXTERNAL_DIFF=x git diff CLAUDE.md",
    "PATH=cat git diff CLAUDE.md",
    "PATH=cat; git diff CLAUDE.md",
    "awk '{print}' CLAUDE.md",
    "find . -name CLAUDE.md -execdir rm {} +",
    "git grep -O'printf x > CLAUDE.md' x CLAUDE.md",
    "git grep -O x CLAUDE.md",
    "git grep --open-files-in-pager=vi x CLAUDE.md",
    "rg x CLAUDE.md | sort --compress-program=x",
    "echo CLAUDE.md | xargs sed -i s/a/b/",
    'f=CLAUDE.md; sed -i s/a/b/ "$f"',
]

# research 4.2: shapes the literal spec would get wrong, and the accepted false positives
EXTRA_MARKING = [
    "cat CLAUDE.md > $OUT",
    'echo x > CLAU""DE.md',
    "sort CLAUDE.md",
    "[[ -f CLAUDE.md ]]",
    "sudo cat CLAUDE.md",
    "xargs cat < CLAUDE.md",
]

OTHER_READERS = [
    "echo CLAUDE.md",
    "printf '%s' CLAUDE.md",
    "test -f CLAUDE.md",
    "[ -f CLAUDE.md ]",
    "ls CLAUDE.md",
    "stat CLAUDE.md",
    "head CLAUDE.md",
    "tail CLAUDE.md",
    "wc -l CLAUDE.md",
    "cat CLAUDE.md | cut -c1-5 | tr a b",
]

SPOT_ROWS = [
    ("cat AGENTS.md", True),
    ("sed -i s/a/b/ AGENTS.md", False),
    ("cat .localharness/INSTRUCTIONS.md", True),
    ("tee .localharness/INSTRUCTIONS.md", False),
    ("cat > sub/CLAUDE.md", False),
]


@pytest.mark.parametrize("command", NOT_MARKING)
def test_matrix_non_marking_rows_are_readers(command: str) -> None:
    assert reads_only(command) is True


@pytest.mark.parametrize("command", MARKING)
def test_matrix_marking_rows_are_not_readers(command: str) -> None:
    assert reads_only(command) is False


@pytest.mark.parametrize("command", EXTRA_MARKING)
def test_extra_shapes_mark(command: str) -> None:
    assert reads_only(command) is False


@pytest.mark.parametrize("command", OTHER_READERS)
def test_other_listed_readers(command: str) -> None:
    assert reads_only(command) is True


@pytest.mark.parametrize(("command", "expected"), SPOT_ROWS)
def test_other_instruction_files_follow_the_same_rule(command: str, expected: bool) -> None:
    assert reads_only(command) is expected


def test_matrix_sizes() -> None:
    # D-36: 15 non-marking rows; 23 marking rows plus the --open-files-in-pager spelling.
    assert len(NOT_MARKING) == 15
    assert len(MARKING) == 24


@pytest.mark.parametrize(
    "command",
    [
        "git grep -O x CLAUDE.md",
        "git grep -Ovi x CLAUDE.md",
        "git grep --open-files-in-pager x CLAUDE.md",
        "git grep --open-files-in-pager=less x CLAUDE.md",
        "git grep --open-files-in-pa=less x CLAUDE.md",
    ],
)
def test_git_grep_pager_spellings_mark(command: str) -> None:
    assert reads_only(command) is False


def test_an_unparsed_command_naming_the_file_marks() -> None:
    assert reads_only("cat CLAUDE.md '") is False


# --------------------------------------------------------------------------- the lists (D-07, D-86)


def test_the_lists_are_exactly_as_decided() -> None:
    assert INSTRUCTION_FILE_READERS == frozenset(
        {
            "cat", "head", "tail", "wc", "grep", "ls", "stat", "sed -n", "rg", "diff", "find",
            "git show", "git log", "git diff", "git status", "git blame", "git ls-files",
            "git add", "git commit", "git grep", "echo", "printf", "test", "[",
        }
    )
    assert INSTRUCTION_PIPELINE_READERS == frozenset({"sort", "uniq", "cut", "tr"})


@pytest.mark.parametrize("verb", ["[[", "sudo", "xargs", "awk", "tee", "mv", "cp"])
def test_left_off_verbs_are_on_neither_list(verb: str) -> None:
    assert verb not in INSTRUCTION_FILE_READERS
    assert verb not in INSTRUCTION_PIPELINE_READERS


def test_the_three_added_git_reads_and_diff_find_are_listed() -> None:
    for sig in ("diff", "find", "git blame", "git grep", "git ls-files", "git commit"):
        assert sig in INSTRUCTION_FILE_READERS


def test_pipeline_only_verbs_are_readers_only_inside_a_pipeline() -> None:
    assert reads_only("sort CLAUDE.md") is False
    assert reads_only("cat CLAUDE.md | sort") is True
    assert reads_only("cat CLAUDE.md | uniq") is True


# --------------------------------------------------------------------------- each fact is load-bearing


def _forcing(monkeypatch: pytest.MonkeyPatch, **forced: object) -> None:
    """Make the module's classify_shell return the real classification with ``forced`` set on
    every segment (dataclasses.replace), so one deciding fact is turned off."""
    real = classify_shell

    def patched(command: str, settings: GateSettings):  # type: ignore[no-untyped-def]
        result = real(command, settings)
        segs = tuple(dataclasses.replace(s, **forced) for s in result.segments)
        return dataclasses.replace(result, segments=segs)

    monkeypatch.setattr(shell_classify, "classify_shell", patched)


def test_env_redirect_decides_the_external_diff_row(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "GIT_EXTERNAL_DIFF=x git diff CLAUDE.md"
    assert reads_only(command) is False
    _forcing(monkeypatch, env_redirect=False)
    assert reads_only(command) is True


def test_inline_interpreter_decides_the_git_grep_pager_row(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "git grep -O x CLAUDE.md"
    assert reads_only(command) is False
    _forcing(monkeypatch, inline_interpreter=False)
    assert reads_only(command) is True


def test_write_targets_decide_the_redirect_row(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "cat > CLAUDE.md"
    assert reads_only(command) is False
    # the target is gone from the facts and the name is gone from the argv: only the text has it,
    # so the text-but-not-argv rule still marks until the target is also restored in argv
    _forcing(monkeypatch, write_targets=(), argv=("cat", "CLAUDE.md"))
    assert reads_only(command) is True


def test_unresolvable_write_decides_the_variable_target_row(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "cat CLAUDE.md > $OUT"
    assert reads_only(command) is False
    _forcing(monkeypatch, unresolvable_write=False, write_targets=())
    assert reads_only(command) is True


def test_destructive_decides_the_checkout_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # the signature check would catch the verb, so also put the verb on the list for the proof
    command = "git checkout -- CLAUDE.md"
    assert reads_only(command) is False
    _forcing(monkeypatch, destructive=False, signature="cat")
    assert reads_only(command) is True


def test_find_read_only_decides_the_execdir_row(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "find . -name CLAUDE.md -execdir rm {} +"
    assert reads_only(command) is False
    real = classify_shell

    def patched(cmd: str, settings: GateSettings):  # type: ignore[no-untyped-def]
        result = real(cmd, settings)
        finds = tuple(
            dataclasses.replace(s, read_only=True) for s in result.segments if s.signature == "find"
        )
        assert finds, "the find segment must survive classification"
        return dataclasses.replace(result, segments=finds)

    monkeypatch.setattr(shell_classify, "classify_shell", patched)
    assert reads_only(command) is True


def test_the_payload_segment_alone_marks_the_execdir_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # whole command is the unit: with find's read_only left alone the row still marks
    command = "find . -name CLAUDE.md -execdir rm {} +"
    real = classify_shell

    def patched(cmd: str, settings: GateSettings):  # type: ignore[no-untyped-def]
        result = real(cmd, settings)
        segs = tuple(
            dataclasses.replace(s, read_only=True) if s.signature == "find" else s
            for s in result.segments
        )
        return dataclasses.replace(result, segments=segs)

    monkeypatch.setattr(shell_classify, "classify_shell", patched)
    assert reads_only(command) is False


# --------------------------------------------------------------------------- purity (INSTR-09)

ALLOWED_SEGMENT_FACTS = {
    "signature",
    "argv",
    "write_targets",
    "inline_interpreter",
    "destructive",
    "unresolvable_write",
    "env_redirect",
    "read_only",
}


def _function_ast() -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(shell_classify))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "instruction_file_reads_only":
            return node
    raise AssertionError("instruction_file_reads_only not found")


def test_the_predicate_reads_only_the_allowed_segment_facts() -> None:
    fn = _function_ast()
    read = {
        node.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "s"
    }
    assert read, "the loop variable of the predicate is expected to be named s"
    assert read <= ALLOWED_SEGMENT_FACTS, read - ALLOWED_SEGMENT_FACTS
    # it classifies exactly once
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "classify_shell"
    ]
    assert len(calls) == 1


def test_the_module_imports_nothing_from_config() -> None:
    path = Path(inspect.getsourcefile(shell_classify) or "")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "config" not in (node.module or "").split("."), node.module
            assert not (node.level and (node.module or "").startswith("config")), node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "localharness.config" not in alias.name
