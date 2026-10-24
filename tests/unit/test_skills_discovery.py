"""Skill discovery: the record a SKILL.md becomes, the rule data, and the import boundaries."""
from __future__ import annotations

import ast
import dataclasses
import os
import time
from pathlib import Path

import pytest

from localharness.agent import task_record
from localharness.config import skills
from localharness.config.skills import SkillSet, skip_line
from localharness.core import content

SRC = Path(skills.__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _empty_bundled_origin(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")


@pytest.fixture
def cfg(tmp_path):
    return tmp_path / "cfg"


@pytest.fixture
def proj(tmp_path):
    root = tmp_path / "proj"
    (root / ".localharness").mkdir(parents=True)
    return root


def _skill(origin_dir, entry, text=None):
    path = origin_dir / entry / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        text = f"---\nname: {entry}\ndescription: Use {entry} when asked.\n---\nBody of {entry}.\n"
    path.write_bytes(text if isinstance(text, bytes) else text.encode())
    return path


def test_discover_orders_project_before_machine_and_skips_a_linked_skill_md(tmp_path, cfg, proj):
    alpha = _skill(cfg / "skills", "alpha")
    _skill(proj / ".localharness" / "skills", "zeta")
    (tmp_path / "elsewhere.md").write_text("---\nname: linked\ndescription: x\n---\n")
    linked = cfg / "skills" / "linked" / "SKILL.md"
    linked.parent.mkdir()
    linked.symlink_to(tmp_path / "elsewhere.md")

    found = SkillSet.discover(cfg, proj)

    assert [s.name for s in found.skills] == ["zeta", "alpha"]
    zeta, alpha_skill = found.skills
    assert (zeta.origin, zeta.untrusted) == ("project", True)
    assert (alpha_skill.origin, alpha_skill.untrusted, alpha_skill.path) == ("machine", False, alpha)
    assert [skip_line(s) for s in found.skipped] == [
        f"skipped: {str(linked)!r} is not a regular file. replace it with a regular file."
    ]


def test_rules_are_the_data_in_order():
    assert skills.RULES == (
        ("<origin> holds more than 256 entries; the rest are ignored", "remove or move skills you do not need"),
        ("<path> is not a regular file", "replace it with a regular file"),
        ("<dir> resolves outside <origin>", "replace the link with a directory"),
        ("<path> holds a control character", "rename it without control characters"),
        ("<path> is over 200 KiB", "move detail into files under references/"),
        ("<path> is not readable UTF-8 text", "save it as UTF-8"),
        ("<path> has no frontmatter", "begin it with a --- line, then name and description, then a --- line"),
        ("<path> frontmatter is not valid YAML", "fix the YAML between the --- lines"),
        ("<path> frontmatter uses YAML anchors or aliases", "write each value out in full"),
        ("<path> frontmatter is not a mapping with string values", "write name: and description: as plain text"),
        ("<path> has no name", "add a name: line equal to the directory name"),
        ("<path> name <name> is not lowercase letters and digits joined by single hyphens", "rename it, for example release-notes"),
        ("<path> name <name> is over 64 characters", "shorten the name and the directory to 64 characters"),
        ("<path> name <name> differs from its directory", "make the name and the directory name equal"),
        ("<path> name <name> is a bundled skill's name", "rename the skill and its directory"),
        ("<path> has no description", "add a description: line saying when to use the skill"),
        ("<path> description is over 1024 characters", "shorten it to 1024 characters"),
        ("<path> description is more than one line", "write it on one line"),
        ("<path> description holds a control character", "remove the control characters"),
        ("<dir> is past the 32 project skills kept", "remove project skills you do not need"),
        ("<dir> is past the 64 skills kept", "remove skills you do not need"),
        ("<dir> is shadowed by <other>", "rename yours to keep both"),
    )
    assert len({fix for _, fix in skills.RULES}) == 22


def _tree(name):
    return ast.parse((SRC / name).read_text())


def test_max_reference_bytes_lives_in_core_content():
    assert task_record.MAX_REFERENCE_BYTES == content.MAX_REFERENCE_BYTES == 200 * 1024
    assert skills.MAX_REFERENCE_BYTES is content.MAX_REFERENCE_BYTES
    tree = _tree("agent/task_record.py")
    assert any(
        isinstance(n, ast.ImportFrom) and n.module == "localharness.core.content"
        and "MAX_REFERENCE_BYTES" in [a.name for a in n.names]
        for n in ast.walk(tree)
    )
    assert not any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAX_REFERENCE_BYTES" for t in n.targets)
        for n in ast.walk(tree)
    )


def test_skills_module_imports_neither_agent_nor_cli():
    names = []
    for node in ast.walk(_tree("config/skills.py")):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not [n for n in names if n.startswith(("localharness.agent", "localharness.cli")) or n == "hashlib"]


def _expect(pair, **fields):
    reason, fix = pair
    for key, value in fields.items():
        reason = reason.replace(f"<{key}>", repr(str(value)))
    return f"skipped: {reason}. {fix}."


def _fm(name="a", description="d", extra=""):
    return f"---\nname: {name}\ndescription: {description}\n{extra}---\nbody\n"


def _only(cfg, proj=None):
    found = SkillSet.discover(cfg, proj)
    return [skip_line(s) for s in found.skipped]


def _write(cfg, text, entry="a"):
    return _skill(cfg / "skills", entry, text)


def _not_regular(cfg, proj, pkg):
    path = _write(cfg, _fm())
    path.unlink()
    (path.parent / "other.md").write_text(_fm())
    path.symlink_to(path.parent / "other.md")
    return _expect(skills.NOT_REGULAR, path=path)


def _link(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    return link


def _outside(cfg, proj, pkg):
    _skill(cfg.parent / "dotfiles", "x")
    link = _link(cfg / "skills" / "x", cfg.parent / "dotfiles" / "x")
    return _expect(skills.OUTSIDE, dir=link, origin=cfg / "skills")


def _control_path(cfg, proj, pkg):
    return _expect(skills.CONTROL_PATH, path=_write(cfg, _fm(), entry="a\nb"))


def _too_large(cfg, proj, pkg):
    head = _fm().encode()
    return _expect(skills.TOO_LARGE, path=_write(cfg, head + b"x" * (204_801 - len(head))))


def _unreadable(cfg, proj, pkg):
    return _expect(skills.UNREADABLE, path=_write(cfg, b"---\nname: a\ndescription: \xff\xfe\n---\n"))


def _no_frontmatter(cfg, proj, pkg):
    return _expect(skills.NO_FRONTMATTER, path=_write(cfg, "name: a\n"))


def _unparseable(cfg, proj, pkg):
    return _expect(skills.UNPARSEABLE, path=_write(cfg, "---\nname: [a\n---\n"))


def _anchors(cfg, proj, pkg):
    return _expect(skills.ANCHORS, path=_write(cfg, "---\nname: &n a\ndescription: *n\n---\n"))


def _not_mapping(cfg, proj, pkg):
    return _expect(skills.NOT_MAPPING, path=_write(cfg, "---\n- a\n- b\n---\n"))


def _no_name(cfg, proj, pkg):
    return _expect(skills.NO_NAME, path=_write(cfg, "---\ndescription: d\n---\n"))


def _bad_name(cfg, proj, pkg):
    return _expect(skills.BAD_NAME, path=_write(cfg, _fm("Bad"), entry="Bad"), name="Bad")


def _long_name(cfg, proj, pkg):
    name = "a" * 65
    return _expect(skills.LONG_NAME, path=_write(cfg, _fm(name), entry=name), name="a" * 64)


def _name_dir(cfg, proj, pkg):
    return _expect(skills.NAME_DIR, path=_write(cfg, _fm("b")), name="b")


def _no_description(cfg, proj, pkg):
    return _expect(skills.NO_DESCRIPTION, path=_write(cfg, "---\nname: a\n---\n"))


def _long_description(cfg, proj, pkg):
    return _expect(skills.LONG_DESCRIPTION, path=_write(cfg, _fm(description="d" * 1025)))


def _multiline_description(cfg, proj, pkg):
    return _expect(
        skills.MULTILINE_DESCRIPTION,
        path=_write(cfg, "---\nname: a\ndescription: |\n  one\n  two\n---\n"),
    )


def _control_description(cfg, proj, pkg):
    return _expect(skills.CONTROL_DESCRIPTION, path=_write(cfg, _fm(description='"a\\eb"')))


CASES = [
    (skills.NOT_REGULAR, _not_regular),
    (skills.OUTSIDE, _outside),
    (skills.CONTROL_PATH, _control_path),
    (skills.TOO_LARGE, _too_large),
    (skills.UNREADABLE, _unreadable),
    (skills.NO_FRONTMATTER, _no_frontmatter),
    (skills.UNPARSEABLE, _unparseable),
    (skills.ANCHORS, _anchors),
    (skills.NOT_MAPPING, _not_mapping),
    (skills.NO_NAME, _no_name),
    (skills.BAD_NAME, _bad_name),
    (skills.LONG_NAME, _long_name),
    (skills.NAME_DIR, _name_dir),
    (skills.NO_DESCRIPTION, _no_description),
    (skills.LONG_DESCRIPTION, _long_description),
    (skills.MULTILINE_DESCRIPTION, _multiline_description),
    (skills.CONTROL_DESCRIPTION, _control_description),
]


@pytest.mark.parametrize("pair,build", CASES, ids=[build.__name__.lstrip("_") for _, build in CASES])
def test_each_per_skill_rule_skips_with_its_exact_line(pair, build, tmp_path, cfg, proj):
    expected = build(cfg, proj, tmp_path / "pkg")
    found = SkillSet.discover(cfg, proj)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == [expected]
    assert pair in skills.RULES and expected.endswith(f". {pair[1]}.")


def test_the_rules_run_in_rules_order():
    order = [skills.RULES.index(pair) for pair, _ in CASES]
    assert order == sorted(order)


def test_byte_description_and_name_caps_are_exact(cfg):
    head = _fm().encode()
    _write(cfg, head + b"x" * (204_800 - len(head)))
    _write(cfg, _fm("b" * 64, "d" * 1024), entry="b" * 64)
    found = SkillSet.discover(cfg, None)
    assert [s.name for s in found.skills] == ["a", "b" * 64]
    assert found.skipped == ()
    assert len(found.skills[1].description) == 1024


def test_too_large_text_is_the_cap():
    assert f"{skills.MAX_REFERENCE_BYTES // 1024} KiB" in skills.TOO_LARGE[0]
    assert skills.MAX_REFERENCE_BYTES // 1024 == 200


def test_a_run_of_open_brackets_is_skipped_quickly(cfg):
    path = _write(cfg, "---\n" + "[" * 100_000 + "\n---\n")
    start = time.monotonic()
    lines = _only(cfg)
    assert time.monotonic() - start < 5
    assert lines == [_expect(skills.UNPARSEABLE, path=path)]


def test_a_deep_flow_mapping_is_skipped(cfg):
    path = _write(cfg, "---\n" + "{a: " * 2000 + "}" * 2000 + "\n---\n")
    assert _only(cfg) == [_expect(skills.UNPARSEABLE, path=path)]


def test_quoted_ampersand_and_star_are_kept(cfg):
    _write(cfg, _fm(description='"fish & chips * 2"'))
    (kept,) = SkillSet.discover(cfg, None).skills
    assert kept.description == "fish & chips * 2"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_a_fifo_skill_md_is_skipped_without_blocking(cfg):
    path = cfg / "skills" / "a" / "SKILL.md"
    path.parent.mkdir(parents=True)
    os.mkfifo(path)
    assert _only(cfg) == [_expect(skills.NOT_REGULAR, path=path)]


def test_a_directory_and_a_dangling_link_are_not_regular_files(cfg):
    folder = cfg / "skills" / "a" / "SKILL.md"
    folder.mkdir(parents=True)
    dangling = cfg / "skills" / "b" / "SKILL.md"
    dangling.parent.mkdir()
    dangling.symlink_to(cfg / "nowhere")
    assert _only(cfg) == [
        _expect(skills.NOT_REGULAR, path=folder),
        _expect(skills.NOT_REGULAR, path=dangling),
    ]


def test_an_undecodable_entry_name_is_skipped_with_the_surrogate_escaped(cfg):
    base = os.fsencode(cfg / "skills")
    os.makedirs(base + b"/a\xff")
    with open(base + b"/a\xff/SKILL.md", "wb") as handle:
        handle.write(_fm().encode())
    path = cfg / "skills" / os.fsdecode(b"a\xff") / "SKILL.md"
    (line,) = _only(cfg)
    assert line == _expect(skills.CONTROL_PATH, path=path)
    assert "\\udcff" in line


def test_a_newline_in_the_directory_name_is_escaped(cfg):
    line = _control_path(cfg, None, None)
    assert "\\n" in line and "\n" not in line
    assert _only(cfg) == [line]


def test_an_escape_in_a_name_is_visible_in_the_quote(cfg):
    path = _write(cfg, _fm(name='"a\\e"'))
    (line,) = _only(cfg)
    assert line == _expect(skills.BAD_NAME, path=path, name="a\x1b")
    assert "\\x1b" in line


@pytest.mark.parametrize("char", [" ", "\u0085"])
def test_line_separators_in_a_description_are_multiline(cfg, char):
    path = _write(cfg, _fm(description=f'"a{char.encode("unicode_escape").decode()}b"'))
    assert _only(cfg) == [_expect(skills.MULTILINE_DESCRIPTION, path=path)]


@pytest.mark.parametrize("char", ["\\t", "\\u200b"])
def test_tab_and_zero_width_space_in_a_description_are_control_characters(cfg, char):
    path = _write(cfg, _fm(description=f'"a{char}b"'))
    assert _only(cfg) == [_expect(skills.CONTROL_DESCRIPTION, path=path)]


def test_a_folded_description_loses_its_one_trailing_newline(cfg):
    _write(cfg, "---\nname: a\ndescription: >\n  folded\n  text\n---\n")
    (kept,) = SkillSet.discover(cfg, None).skills
    assert kept.description == "folded text"


def test_crlf_line_ends_are_accepted(cfg):
    _write(cfg, _fm().replace("\n", "\r\n"))
    (kept,) = SkillSet.discover(cfg, None).skills
    assert kept.description == "d"


def test_a_byte_order_mark_and_a_zero_byte_file_have_no_frontmatter(cfg):
    bom = _write(cfg, b"\xef\xbb\xbf" + _fm().encode(), entry="a")
    empty = _write(cfg, b"", entry="b")
    assert _only(cfg) == [_expect(skills.NO_FRONTMATTER, path=bom), _expect(skills.NO_FRONTMATTER, path=empty)]


def test_four_hyphens_do_not_open_the_frontmatter(cfg):
    path = _write(cfg, "----\nname: a\ndescription: d\n---\n")
    assert _only(cfg) == [_expect(skills.NO_FRONTMATTER, path=path)]


def test_text_after_the_closing_line_is_never_parsed(cfg):
    _write(cfg, _fm() + "---\n[[[ not: yaml: {\n")
    assert [s.name for s in SkillSet.discover(cfg, None).skills] == ["a"]


def test_empty_frontmatter_is_not_a_mapping(cfg):
    path = _write(cfg, "---\n---\n")
    assert _only(cfg) == [_expect(skills.NOT_MAPPING, path=path)]


@pytest.mark.parametrize("text,pair", [
    ("---\nname:\ndescription: d\n---\n", skills.NO_NAME),
    ("---\nname: null\ndescription: d\n---\n", skills.NO_NAME),
    ("---\nname: a\ndescription:\n---\n", skills.NO_DESCRIPTION),
    ('---\nname: a\ndescription: "  "\n---\n', skills.NO_DESCRIPTION),
    ("---\nname: 7\ndescription: d\n---\n", skills.NOT_MAPPING),
    ("---\nname: true\ndescription: d\n---\n", skills.NOT_MAPPING),
    ("---\nname: a\ndescription: 2024-01-01\n---\n", skills.NOT_MAPPING),
    ("---\nname: a\ndescription: [x]\n---\n", skills.NOT_MAPPING),
])
def test_absent_and_non_string_values(cfg, text, pair):
    path = _write(cfg, text)
    assert _only(cfg) == [_expect(pair, path=path)]


def test_an_empty_name_is_still_visibly_quoted(cfg):
    path = _write(cfg, '---\nname: ""\ndescription: d\n---\n')
    (line,) = _only(cfg)
    assert line == _expect(skills.BAD_NAME, path=path, name="")
    assert "name '' is not" in line


def test_a_plain_file_and_a_directory_without_skill_md_are_not_skills(cfg):
    (cfg / "skills" / "empty").mkdir(parents=True)
    (cfg / "skills" / "notes.txt").write_text("hello")
    found = SkillSet.discover(cfg, None)
    assert found.skills == () and found.skipped == ()


def test_the_first_failing_rule_alone_is_reported(cfg):
    path = _write(cfg, '---\nname: BAD\ndescription: ""\n---\n', entry="BAD")
    assert _only(cfg) == [_expect(skills.BAD_NAME, path=path, name="BAD")]


def test_echoed_names_are_cut_to_64_and_echoed_paths_are_whole(cfg):
    name = "A" * 64 + "Z" * 236
    path = _write(cfg, _fm(name=name))
    (line,) = _only(cfg)
    assert line == _expect(skills.BAD_NAME, path=path, name="A" * 64)
    assert "ZZ" not in line
    long_entry = "a" * 100
    long_path = _write(cfg, _fm(long_entry), entry=long_entry)
    assert _only(cfg)[-1] == _expect(skills.LONG_NAME, path=long_path, name="a" * 64)
    assert long_entry in _only(cfg)[-1]


def test_unknown_keys_are_ignored_and_a_skill_has_five_fields(cfg):
    extra = "allowed-tools: [bash]\nmetadata:\n  k: v\nlicense: MIT\n"
    _write(cfg, _fm(extra=extra))
    (kept,) = SkillSet.discover(cfg, None).skills
    assert kept.name == "a"
    assert [f.name for f in dataclasses.fields(skills.Skill)] == ["name", "description", "origin", "path", "untrusted"]


def test_no_project_root_reads_no_project_origin(cfg, proj):
    _skill(proj / ".localharness" / "skills", "p")
    assert SkillSet.discover(cfg, None).skills == ()
    (kept,) = SkillSet.discover(cfg, proj).skills
    assert kept.name == "p"


def test_the_agents_origin_is_read_and_an_empty_tree_gives_empty_tuples(cfg, proj):
    assert SkillSet.discover(cfg, proj).skills == () and SkillSet.discover(cfg, proj).skipped == ()
    _skill(proj / ".agents" / "skills", "q")
    (kept,) = SkillSet.discover(cfg, proj).skills
    assert (kept.name, kept.origin, kept.untrusted) == ("q", "project", True)


def test_the_skill_path_is_lexical(tmp_path):
    real = tmp_path / "real"
    _skill(real / "skills", "a")
    link = tmp_path / "link"
    link.symlink_to(real)
    (kept,) = SkillSet.discover(link, None).skills
    assert kept.path == link / "skills" / "a" / "SKILL.md"


def _outside_lines(origin, *names):
    return [_expect(skills.OUTSIDE, dir=origin / name, origin=origin) for name in names]


def test_a_symlinked_agents_directory_skips_every_skill_even_when_the_target_is_inside_the_root(cfg, proj):
    for name in ("a", "b"):
        _skill(proj / "tools" / "agents" / "skills", name)
    _link(proj / ".agents", proj / "tools" / "agents")
    found = SkillSet.discover(cfg, proj)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == _outside_lines(proj / ".agents" / "skills", "a", "b")


def test_a_symlinked_agents_skills_directory_skips_every_skill_under_it(cfg, proj):
    _skill(proj / "shared" / "skills", "a")
    _link(proj / ".agents" / "skills", proj / "shared" / "skills")
    found = SkillSet.discover(cfg, proj)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == _outside_lines(proj / ".agents" / "skills", "a")


def test_a_project_skill_linked_into_another_project_is_skipped(tmp_path, cfg, proj):
    _skill(tmp_path / "other" / ".localharness" / "skills", "x")
    _link(proj / ".localharness" / "skills" / "x", tmp_path / "other" / ".localharness" / "skills" / "x")
    found = SkillSet.discover(cfg, proj)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == _outside_lines(proj / ".localharness" / "skills", "x")


def test_a_skill_linked_to_a_sibling_origin_is_skipped_and_the_real_one_kept(cfg, proj):
    real = _skill(proj / ".localharness" / "skills", "x")
    _link(proj / ".agents" / "skills" / "x", Path("../../.localharness/skills/x"))
    found = SkillSet.discover(cfg, proj)
    assert [s.path for s in found.skills] == [real]
    assert [skip_line(s) for s in found.skipped] == _outside_lines(proj / ".agents" / "skills", "x")


def test_a_skill_linked_inside_the_root_but_outside_its_origin_is_skipped(cfg, proj):
    _skill(proj / "docs", "x")
    _link(proj / ".localharness" / "skills" / "x", proj / "docs" / "x")
    found = SkillSet.discover(cfg, proj)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == _outside_lines(proj / ".localharness" / "skills", "x")


def test_a_skill_linked_to_another_directory_inside_its_origin_is_kept(cfg):
    _skill(cfg / "skills", "store", _fm("y"))
    _link(cfg / "skills" / "y", Path("store"))
    found = SkillSet.discover(cfg, None)
    assert [s.path for s in found.skills] == [cfg / "skills" / "y" / "SKILL.md"]
    assert [skip_line(s) for s in found.skipped] == [
        _expect(skills.NAME_DIR, path=cfg / "skills" / "store" / "SKILL.md", name="y")
    ]


def test_a_link_at_the_base_is_followed(tmp_path):
    _skill(tmp_path / "cfg-real" / "skills", "alpha")
    link = _link(tmp_path / "cfg-link", tmp_path / "cfg-real")
    found = SkillSet.discover(link, None)
    assert [s.path for s in found.skills] == [link / "skills" / "alpha" / "SKILL.md"]
    assert found.skipped == ()


def test_the_regular_file_test_runs_before_containment(tmp_path, cfg):
    real = tmp_path / "dotfiles" / "x"
    _skill(tmp_path / "dotfiles", "x")
    (real / "SKILL.md").unlink()
    (real / "other.md").write_text(_fm("x"))
    (real / "SKILL.md").symlink_to(real / "other.md")
    link = _link(cfg / "skills" / "x", real)
    assert _only(cfg) == [_expect(skills.NOT_REGULAR, path=link / "SKILL.md")]


def test_regular_file_test_is_lstat():
    tree = _tree("config/skills.py")
    assert "is_file" not in {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert {"os.lstat", "stat.S_ISREG"} <= calls
