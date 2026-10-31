"""Skill discovery: the record a SKILL.md becomes, the rule data, and the import boundaries."""
from __future__ import annotations

import ast
import dataclasses
import os
import signal
import time
from types import SimpleNamespace
from pathlib import Path

import pytest

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
        ("<path> purpose is not one line of plain text", "write purpose: on one line"),
        ("<path> tools is not a list of tool names", "write tools: as a list of tool names"),
        ("<dir> is past the 32 project skills kept", "remove project skills you do not need"),
        ("<dir> is past the 64 skills kept", "remove skills you do not need"),
        ("<dir> is shadowed by <other>", "rename yours to keep both"),
    )
    assert len({fix for _, fix in skills.RULES}) == 24


def _tree(name):
    return ast.parse((SRC / name).read_text())


def test_max_reference_bytes_lives_in_core_content():
    assert content.MAX_REFERENCE_BYTES == 200 * 1024
    assert skills.MAX_REFERENCE_BYTES is content.MAX_REFERENCE_BYTES
    tree = _tree("config/skills.py")
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


def _empty_dirs(base, count):
    for i in range(count):
        (base / f"d{i:03d}").mkdir(parents=True, exist_ok=True)


def _over_256(cfg, proj, pkg):
    _empty_dirs(cfg / "skills", 257)
    return _expect(skills.OVER_256, origin=cfg / "skills")


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


def _bundled_name(cfg, proj, pkg):
    _skill(pkg / "assets" / "skills", "skill-creator")
    return _expect(skills.BUNDLED_NAME, path=_write(cfg, None, entry="skill-creator"), name="skill-creator")


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


def _bad_purpose(cfg, proj, pkg):
    return _expect(skills.BAD_PURPOSE, path=_write(cfg, _fm(extra="purpose: 7\n")))


def _bad_tools(cfg, proj, pkg):
    return _expect(skills.BAD_TOOLS, path=_write(cfg, _fm(extra="tools: read\n")))


def _cap_project(cfg, proj, pkg):
    for i in range(33):
        last = _skill(proj / ".localharness" / "skills", f"s{i:02d}")
    return _expect(skills.CAP_PROJECT, dir=last.parent)


def _cap_total(cfg, proj, pkg):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
    for i in range(33):
        last = _write(cfg, None, entry=f"m{i:02d}")
    return _expect(skills.CAP_TOTAL, dir=last.parent)


def _shadowed(cfg, proj, pkg):
    kept = _skill(proj / ".localharness" / "skills", "x")
    return _expect(skills.SHADOWED, dir=_write(cfg, None, entry="x").parent, other=kept.parent)


CASES = [
    (skills.OVER_256, _over_256),
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
    (skills.BUNDLED_NAME, _bundled_name),
    (skills.NO_DESCRIPTION, _no_description),
    (skills.LONG_DESCRIPTION, _long_description),
    (skills.MULTILINE_DESCRIPTION, _multiline_description),
    (skills.CONTROL_DESCRIPTION, _control_description),
    (skills.BAD_PURPOSE, _bad_purpose),
    (skills.BAD_TOOLS, _bad_tools),
    (skills.CAP_PROJECT, _cap_project),
    (skills.CAP_TOTAL, _cap_total),
    (skills.SHADOWED, _shadowed),
]
KEPT = {skills.BUNDLED_NAME: 1, skills.CAP_PROJECT: 32, skills.CAP_TOTAL: 64, skills.SHADOWED: 1}


@pytest.mark.parametrize("pair,build", CASES, ids=[build.__name__.lstrip("_") for _, build in CASES])
def test_each_per_skill_rule_skips_with_its_exact_line(pair, build, tmp_path, cfg, proj):
    expected = build(cfg, proj, tmp_path / "pkg")
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == KEPT.get(pair, 0)
    assert [skip_line(s) for s in found.skipped] == [expected]
    assert pair in skills.RULES and expected.endswith(f". {pair[1]}.")


def _link_and_file(cfg, proj, pkg):
    real = cfg.parent / "dotfiles" / "x"
    _skill(cfg.parent / "dotfiles", "x")
    (real / "SKILL.md").unlink()
    (real / "other.md").write_text(_fm("x"))
    (real / "SKILL.md").symlink_to(real / "other.md")
    return _link(cfg / "skills" / "x", real) / "SKILL.md", {}


def _outside_and_control(cfg, proj, pkg):
    _skill(cfg.parent / "dotfiles", "x")
    link = _link(cfg / "skills" / "a\nb", cfg.parent / "dotfiles" / "x")
    return link, {"dir": link, "origin": cfg / "skills"}


def _bundled_and(entry, text):
    def build(cfg, proj, pkg):
        _skill(pkg / "assets" / "skills", "skill-creator")
        return _write(cfg, text, entry=entry), {"name": "skill-creator"}

    return build


def _project_cap_and_control(cfg, proj, pkg):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"s{i:02d}")
    return _skill(proj / ".localharness" / "skills", "s32", _fm("s32", extra="tools: read\n")), {}


def _total_cap_and_shadow(cfg, proj, pkg):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", "zz-dup" if i == 31 else f"p{i:02d}")
        _write(cfg, None, entry=f"m{i:02d}")
    return _write(cfg, None, entry="zz-dup"), {"dir": cfg / "skills" / "zz-dup"}


def _plain(text, entry="a", **fields):
    return lambda cfg, proj, pkg: (_write(cfg, text, entry=entry), fields)


LONG_A = "A" * 65
PAIRS = [
    (skills.NOT_REGULAR, skills.OUTSIDE, _link_and_file),
    (skills.OUTSIDE, skills.CONTROL_PATH, _outside_and_control),
    (skills.CONTROL_PATH, skills.TOO_LARGE, _plain(b"x" * 204_801, entry="a\nb")),
    (skills.TOO_LARGE, skills.UNREADABLE, _plain(b"\xff" * 204_801)),
    (skills.UNREADABLE, skills.NO_FRONTMATTER, _plain(b"\xff")),
    (skills.UNPARSEABLE, skills.ANCHORS, _plain("---\nname: &n a\ndescription: [x\n---\n")),
    (skills.ANCHORS, skills.NOT_MAPPING, _plain("---\n- &a x\n- *a\n---\n")),
    (skills.NOT_MAPPING, skills.NO_NAME, _plain("---\ndescription: 7\n---\n")),
    (skills.BAD_NAME, skills.LONG_NAME, _plain(_fm(LONG_A), entry=LONG_A, name="A" * 64)),
    (skills.LONG_NAME, skills.NAME_DIR, _plain(_fm("a" * 65), name="a" * 64)),
    (skills.NAME_DIR, skills.BUNDLED_NAME, _bundled_and("other", _fm("skill-creator"))),
    (skills.BUNDLED_NAME, skills.NO_DESCRIPTION, _bundled_and("skill-creator", "---\nname: skill-creator\n---\n")),
    (skills.NO_DESCRIPTION, skills.LONG_DESCRIPTION, _plain(_fm(description='"' + " " * 1025 + '"'))),
    (skills.LONG_DESCRIPTION, skills.MULTILINE_DESCRIPTION, _plain(_fm(description=f'"{"d" * 1025}\\nx"'))),
    (skills.MULTILINE_DESCRIPTION, skills.CONTROL_DESCRIPTION, _plain(_fm(description='"a\\nb"'))),
    (skills.CONTROL_DESCRIPTION, skills.BAD_PURPOSE, _plain(_fm(description='"a\\tb"', extra="purpose: 7\n"))),
    (skills.BAD_PURPOSE, skills.BAD_TOOLS, _plain(_fm(extra="purpose: 7\ntools: read\n"))),
    (skills.BAD_TOOLS, skills.CAP_PROJECT, _project_cap_and_control),
    (skills.CAP_TOTAL, skills.SHADOWED, _total_cap_and_shadow),
]
UNREACHABLE = [
    (skills.NO_FRONTMATTER, skills.UNPARSEABLE),
    (skills.OVER_256, skills.NOT_REGULAR),
    (skills.NO_NAME, skills.BAD_NAME),
    (skills.CAP_PROJECT, skills.CAP_TOTAL),
]


@pytest.mark.parametrize("earlier,later,build", PAIRS, ids=[f"rule{skills.RULES.index(e)}-before-rule{skills.RULES.index(nxt)}" for e, nxt, _ in PAIRS])
def test_a_skill_breaking_two_adjacent_rules_reports_the_earlier(earlier, later, build, cfg, proj, tmp_path):
    path, fields = build(cfg, proj, tmp_path / "pkg")
    fields.setdefault("path", path)
    expected = _expect(earlier, **fields)
    assert skills.RULES.index(later) == skills.RULES.index(earlier) + 1
    assert [skip_line(s) for s in SkillSet.discover(cfg, proj).skipped] == [expected]


def test_every_adjacent_pair_is_tested_or_cannot_hold_at_once():
    pairs = {(e, nxt) for e, nxt, _ in PAIRS} | set(UNREACHABLE)
    assert pairs == set(zip(skills.RULES, skills.RULES[1:]))


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


@pytest.mark.parametrize("char", ["\u2028", "\u0085"])
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


def test_a_byte_order_mark_is_read_past_and_a_zero_byte_file_has_no_frontmatter(cfg):
    _write(cfg, b"\xef\xbb\xbf" + _fm().encode(), entry="a")
    empty = _write(cfg, b"", entry="b")
    assert [s.name for s in SkillSet.discover(cfg, None).skills] == ["a"]
    assert _only(cfg) == [_expect(skills.NO_FRONTMATTER, path=empty)]


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


def test_unknown_keys_are_ignored_and_a_skill_has_seven_fields(cfg):
    extra = "allowed-tools: [bash]\nmetadata:\n  k: v\nlicense: MIT\n"
    _write(cfg, _fm(extra=extra))
    (kept,) = SkillSet.discover(cfg, None).skills
    assert kept.name == "a"
    assert [f.name for f in dataclasses.fields(skills.Skill)] == [
        "name", "description", "origin", "path", "untrusted", "purpose", "tools"]


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


def test_every_rule_has_a_case():
    assert {pair for pair, _ in CASES} == set(skills.RULES)
    assert len(CASES) == len(skills.RULES) == 24
    fixes = [fix for _, fix in skills.RULES]
    assert len(set(fixes)) == 24 and all(fixes)


def test_the_entry_bound_is_256_and_the_257th_sorted_name_is_never_read(cfg):
    base = cfg / "skills"
    _empty_dirs(base, 255)
    _skill(base, "zzz")
    found = SkillSet.discover(cfg, None)
    assert [s.name for s in found.skills] == ["zzz"] and found.skipped == ()
    (base / "d255").mkdir()
    found = SkillSet.discover(cfg, None)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == [_expect(skills.OVER_256, origin=base)]


def test_two_over_full_origins_give_two_notices_project_first(cfg, proj):
    _empty_dirs(cfg / "skills", 257)
    _empty_dirs(proj / ".localharness" / "skills", 257)
    assert _only(cfg, proj) == [
        _expect(skills.OVER_256, origin=proj / ".localharness" / "skills"),
        _expect(skills.OVER_256, origin=cfg / "skills"),
    ]


def test_entries_are_read_in_code_point_order(cfg):
    for name in ("a-2", "a-10", "a-1"):
        _skill(cfg / "skills", name)
    assert [s.name for s in SkillSet.discover(cfg, None).skills] == ["a-1", "a-10", "a-2"]


def test_a_non_bundled_skill_never_takes_a_bundled_name(tmp_path, cfg, proj):
    bundled = _skill(tmp_path / "pkg" / "assets" / "skills", "skill-creator")
    project = _skill(proj / ".localharness" / "skills", "skill-creator")
    machine = _write(cfg, None, entry="skill-creator")
    found = SkillSet.discover(cfg, proj)
    assert [(s.origin, s.path) for s in found.skills] == [("bundled", bundled)]
    assert [skip_line(s) for s in found.skipped] == [
        _expect(skills.BUNDLED_NAME, path=project, name="skill-creator"),
        _expect(skills.BUNDLED_NAME, path=machine, name="skill-creator"),
    ]


def test_an_invalid_bundled_directory_reserves_no_name(tmp_path, cfg, proj):
    _skill(tmp_path / "pkg" / "assets" / "skills", "broken", "name: broken\n")
    _skill(proj / ".localharness" / "skills", "broken")
    found = SkillSet.discover(cfg, proj)
    assert [(s.name, s.origin) for s in found.skills] == [("broken", "project")]
    assert len(found.skipped) == 1


def test_the_caps_keep_exactly_32_project_and_64_total(cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
        _write(cfg, None, entry=f"m{i:02d}")
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == 64 and found.skipped == ()


def test_a_shadowed_record_takes_no_project_slot(cfg, proj):
    for i in range(31):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
        _skill(proj / ".agents" / "skills", f"p{i:02d}")
    _skill(proj / ".agents" / "skills", "q")
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == 32 and found.skills[-1].name == "q"
    assert len(found.skipped) == 31
    assert all(" is shadowed by " in s.reason for s in found.skipped)


def test_a_full_cap_wins_over_a_shadow(cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
    dup = _skill(proj / ".agents" / "skills", "p00")
    found = SkillSet.discover(cfg, proj)
    assert [skip_line(s) for s in found.skipped] == [_expect(skills.CAP_PROJECT, dir=dup.parent)]


def test_a_machine_duplicate_past_the_total_cap_is_capped_not_shadowed(cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", "zz-dup" if i == 31 else f"p{i:02d}")
        _write(cfg, None, entry=f"m{i:02d}")
    dup = _write(cfg, None, entry="zz-dup")
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == 64
    assert [skip_line(s) for s in found.skipped] == [_expect(skills.CAP_TOTAL, dir=dup.parent)]


def test_the_highest_precedence_same_name_skill_is_kept(cfg, proj):
    top = _skill(proj / ".localharness" / "skills", "x")
    agents = _skill(proj / ".agents" / "skills", "x")
    machine = _write(cfg, None, entry="x")
    found = SkillSet.discover(cfg, proj)
    assert [s.path for s in found.skills] == [top]
    assert [skip_line(s) for s in found.skipped] == [
        _expect(skills.SHADOWED, dir=agents.parent, other=top.parent),
        _expect(skills.SHADOWED, dir=machine.parent, other=top.parent),
    ]


def test_no_project_root_reads_machine_and_bundled_only(tmp_path, cfg, proj):
    _skill(proj / ".localharness" / "skills", "p")
    _skill(proj / ".agents" / "skills", "q")
    _write(cfg, None, entry="m")
    _skill(tmp_path / "pkg" / "assets" / "skills", "b")
    assert [(s.name, s.origin) for s in SkillSet.discover(cfg, None).skills] == [("m", "machine"), ("b", "bundled")]


def test_agents_skills_without_a_localharness_directory_contribute_nothing(tmp_path, cfg):
    bare = tmp_path / "bare"
    _skill(bare / ".agents" / "skills", "a")
    found = SkillSet.discover(cfg, bare)
    assert found.skills == () and found.skipped == ()


def test_every_origin_absent_gives_empty_tuples(cfg):
    found = SkillSet.discover(cfg, None)
    assert found.skills == () and found.skipped == ()


def test_every_skill_carries_an_origin_label_in_precedence_then_name_order(tmp_path, cfg, proj):
    bases = {"project": proj, "machine": cfg, "bundled": tmp_path / "pkg"}
    dirs = [bases[label] / rel for label, rel in skills.ORIGINS]
    for directory, names in zip(dirs, (("z1", "z2"), ("y1", "y2"), ("x1", "x2"), ("w1", "w2"))):
        for name in names:
            _skill(directory, name)
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == 8 and found.skipped == ()
    assert {s.origin for s in found.skills} <= {label for label, _ in skills.ORIGINS}
    assert list(found.skills) == sorted(found.skills, key=lambda s: (dirs.index(s.path.parent.parent), s.name))
    assert [s.name for s in found.skills] == ["z1", "z2", "y1", "y2", "x1", "x2", "w1", "w2"]


def _one(cfg, text=None, entry="a"):
    path = _write(cfg, text, entry=entry)
    return path, SkillSet.discover(cfg, None)


def test_read_body_returns_the_raw_text_and_repeats(cfg):
    text = "---\nname: a\ndescription: d\n---\nBodyé\n"
    path, found = _one(cfg, text)
    record = found.skills[0]
    assert found.read_body(record) == text == path.read_bytes().decode()
    assert found.read_body(record) == found.read_body(record)


def test_read_body_keeps_crlf(cfg):
    text = "---\r\nname: a\r\ndescription: d\r\n---\r\nBody\r\n"
    _, found = _one(cfg, text)
    assert found.read_body(found.skills[0]) == text


def test_read_body_cap_is_exact(cfg):
    head = "---\nname: a\ndescription: d\n---\n"
    path, found = _one(cfg, head + "x" * (skills.MAX_REFERENCE_BYTES - len(head)))
    assert path.stat().st_size == skills.MAX_REFERENCE_BYTES
    assert len(found.read_body(found.skills[0])) == skills.MAX_REFERENCE_BYTES
    with path.open("ab") as handle:
        handle.write(b"x")
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.TOO_LARGE, path=path)


def test_read_body_refuses_a_file_swapped_for_a_link(tmp_path, cfg):
    path, found = _one(cfg)
    (tmp_path / "copy.md").write_text(_fm())
    path.unlink()
    path.symlink_to(tmp_path / "copy.md")
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.NOT_REGULAR, path=path)


def test_read_body_refuses_a_directory_swapped_for_a_link_out_of_its_origin(tmp_path, cfg):
    path, found = _one(cfg)
    _skill(tmp_path / "dotfiles", "a")
    path.parent.rename(tmp_path / "moved")
    path.parent.symlink_to(tmp_path / "dotfiles" / "a")
    assert skip_line(found.read_body(found.skills[0])) == _expect(
        skills.OUTSIDE, dir=path.parent, origin=cfg / "skills"
    )


def test_read_body_of_a_deleted_file_is_not_regular(cfg):
    path, found = _one(cfg)
    path.unlink()
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.NOT_REGULAR, path=path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_read_body_refuses_a_fifo_without_blocking(cfg):
    path, found = _one(cfg)
    path.unlink()
    os.mkfifo(path)
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.NOT_REGULAR, path=path)


def test_read_body_refuses_bytes_that_became_invalid_utf8(cfg):
    path, found = _one(cfg)
    path.write_bytes(b"\xff\xfe")
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.UNREADABLE, path=path)


def test_read_body_runs_the_checks_in_discovery_order(tmp_path, cfg):
    path, found = _one(cfg)
    path.write_bytes(b"\xff" * (skills.MAX_REFERENCE_BYTES + 1))
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.TOO_LARGE, path=path)


def test_read_body_refuses_a_record_under_no_origin(tmp_path, cfg):
    _, found = _one(cfg)
    stray = skills.Skill("s", "d", "machine", tmp_path / "stray" / "s" / "SKILL.md", False)
    assert skip_line(found.read_body(stray)) == _expect(
        skills.OUTSIDE, dir=stray.path.parent, origin=stray.path.parent.parent
    )


def test_read_body_leaves_the_set_unchanged(cfg):
    path, found = _one(cfg)
    before = (found.skills, found.skipped)
    path.unlink()
    for _ in range(3):
        found.read_body(found.skills[0])
    assert (found.skills, found.skipped) == before


BAD_NAMES = ["../x", "/etc", "X", "Alpha", "a\n", "", "a" * 65, "a--b", "-a", None]


@pytest.mark.parametrize("name", BAD_NAMES, ids=[repr(n)[:12] for n in BAD_NAMES])
def test_find_returns_none_for_a_bad_name_before_any_scan(name, cfg, monkeypatch):
    _write(cfg, None)
    found = SkillSet.discover(cfg, None)

    def boom(*args):
        raise AssertionError("scanned")

    monkeypatch.setattr(skills, "_scan", boom)
    monkeypatch.setattr(skills, "_hits", boom)
    assert found.find(name) is None


def test_find_returns_a_skill_added_since_start_once(cfg):
    _write(cfg, None, entry="old")
    found = SkillSet.discover(cfg, None)
    before = (found.skills, found.skipped)
    _write(cfg, None, entry="new")
    result = found.find("new")
    assert isinstance(result, skills.Skill) and result.origin in {label for label, _ in skills.ORIGINS}
    assert (result.origin, result.name) == ("machine", "new")
    assert found.skills[-1] is result and found.skills[:-1] == before[0] and found.skipped == before[1]
    assert found.find("new") is result and len(found.skills) == 2


def test_find_prefers_the_start_set_over_a_later_higher_origin(cfg, proj):
    machine = _write(cfg, None, entry="x")
    found = SkillSet.discover(cfg, proj)
    _skill(proj / ".localharness" / "skills", "x")
    assert found.find("x") is found.skills[0] and found.skills[0].path == machine


def test_find_applies_the_shadow_rule_to_a_fresh_pair(cfg, proj):
    found = SkillSet.discover(cfg, proj)
    top = _skill(proj / ".localharness" / "skills", "y")
    _skill(proj / ".agents" / "skills", "y")
    result = found.find("y")
    assert result.path == top and found.skills == (result,) and found.skipped == ()


def test_find_returns_the_first_skip_when_every_match_is_skipped(cfg, proj):
    found = SkillSet.discover(cfg, proj)
    first = _skill(proj / ".localharness" / "skills", "x", _fm("other"))
    _skill(proj / ".agents" / "skills", "x", _fm("other"))
    result = found.find("x")
    assert skip_line(result) == _expect(skills.NAME_DIR, path=first, name="other")
    assert found.skills == () and found.skipped == ()


def test_find_returns_the_skip_for_an_invalid_new_directory(cfg):
    found = SkillSet.discover(cfg, None)
    path = _write(cfg, _fm("other"), entry="bad")
    assert skip_line(found.find("bad")) == _expect(skills.NAME_DIR, path=path, name="other")
    assert found.skills == () and found.skipped == ()


def test_find_returns_none_for_a_name_no_origin_holds(cfg):
    _write(cfg, None, entry="a")
    (cfg / "skills" / "empty").mkdir()
    found = SkillSet.discover(cfg, None)
    assert found.find("nothing-here") is None and found.find("empty") is None


@pytest.mark.parametrize("added", ["zz-new", "a-new"])
def test_find_keeps_the_total_cap_on_the_live_set(added, cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
        _write(cfg, None, entry=f"m{i:02d}")
    found = SkillSet.discover(cfg, proj)
    new = _write(cfg, None, entry=added)
    assert skip_line(found.find(added)) == _expect(skills.CAP_TOTAL, dir=new.parent)
    assert len(found.skills) == 64 and found.skipped == ()


def test_find_keeps_the_project_cap_on_the_live_set(cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
    found = SkillSet.discover(cfg, proj)
    new = _skill(proj / ".localharness" / "skills", "p-new")
    assert skip_line(found.find("p-new")) == _expect(skills.CAP_PROJECT, dir=new.parent)
    assert len(found.skills) == 32


def test_find_reuses_the_start_time_workspace_decision(tmp_path, cfg):
    bare = tmp_path / "bare"
    _skill(bare / ".agents" / "skills", "y")
    found = SkillSet.discover(cfg, bare)
    (bare / ".localharness").mkdir()
    assert found.find("y") is None and found.skills == ()


def test_read_body_reuses_the_start_time_workspace_decision(cfg, proj):
    record = _skill(proj / ".localharness" / "skills", "p")
    found = SkillSet.discover(cfg, proj)
    (proj / ".localharness" / "skills" / "p").rename(proj / "moved")
    (proj / ".localharness" / "skills").rmdir()
    (proj / ".localharness").rmdir()
    assert skip_line(found.read_body(found.skills[0])) == _expect(skills.NOT_REGULAR, path=record)


def test_find_counts_the_live_total_not_the_fresh_scan(cfg, proj):
    for i in range(32):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
    for i in range(30):
        _write(cfg, None, entry=f"m{i:02d}")
    found = SkillSet.discover(cfg, proj)
    assert len(found.skills) == 62
    new = {name: _write(cfg, None, entry=name) for name in ("m30", "m31", "m32")}
    assert isinstance(found.find("m32"), skills.Skill) and isinstance(found.find("m31"), skills.Skill)
    assert skip_line(found.find("m30")) == _expect(skills.CAP_TOTAL, dir=new["m30"].parent)
    assert len(found.skills) == 64


def test_find_counts_the_live_project_skills_not_the_fresh_scan(cfg, proj):
    for i in range(30):
        _skill(proj / ".localharness" / "skills", f"p{i:02d}")
    found = SkillSet.discover(cfg, proj)
    new = {name: _skill(proj / ".localharness" / "skills", name) for name in ("p30", "p31", "p32")}
    assert isinstance(found.find("p32"), skills.Skill) and isinstance(found.find("p31"), skills.Skill)
    assert skip_line(found.find("p30")) == _expect(skills.CAP_PROJECT, dir=new["p30"].parent)
    assert len(found.skills) == 32


def test_find_validates_only_the_directories_named_like_the_typed_name(cfg, proj, monkeypatch):
    for entry in ("a", "b", "c"):
        _write(cfg, "not a skill", entry=entry)
        _skill(proj / ".localharness" / "skills", entry, "also not")
    _skill(proj / ".localharness" / "skills", "x")
    _write(cfg, None, entry="x")
    found = SkillSet.discover(cfg, proj)
    seen = []
    real = skills._check
    monkeypatch.setattr(skills, "_check", lambda *a: seen.append(a[3]) or real(*a))
    assert found.find("nothing") is None and seen == []
    assert found.find("a") is not None and set(seen) == {"a"}


@pytest.fixture
def alarm():
    def fire(*_):
        raise TimeoutError("blocked")

    previous = []

    def arm(seconds=3):
        previous.append(signal.signal(signal.SIGALRM, fire))
        signal.setitimer(signal.ITIMER_REAL, seconds)

    yield arm
    signal.setitimer(signal.ITIMER_REAL, 0)
    if previous:
        signal.signal(signal.SIGALRM, previous[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_read_opens_a_fifo_without_blocking_and_refuses_it(tmp_path, alarm):
    fifo = tmp_path / "f"
    os.mkfifo(fifo)
    alarm()
    assert skills._read(fifo) == skills.NOT_REGULAR


def test_read_refuses_a_directory_and_a_symlink(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "t").write_text("x")
    (tmp_path / "l").symlink_to(tmp_path / "t")
    assert skills._read(tmp_path / "d") == skills.NOT_REGULAR
    assert skills._read(tmp_path / "l") == skills.NOT_REGULAR
    assert skills._read(tmp_path / "gone") == skills.UNREADABLE


def test_read_refuses_a_file_over_the_cap_by_its_own_size(tmp_path):
    big = tmp_path / "big"
    big.write_bytes(b"x" * (skills.MAX_REFERENCE_BYTES + 1))
    assert skills._read(big) == skills.TOO_LARGE


def test_read_refuses_a_file_that_grew_after_the_size_check(tmp_path, monkeypatch):
    big = tmp_path / "big"
    big.write_bytes(b"x" * (skills.MAX_REFERENCE_BYTES + 10))
    real = os.fstat
    monkeypatch.setattr(os, "fstat", lambda fd: SimpleNamespace(st_mode=real(fd).st_mode, st_size=1))
    got = []
    real_read = os.read
    monkeypatch.setattr(os, "read", lambda fd, n: got.append(real_read(fd, n)) or got[-1])
    assert skills._read(big) == skills.TOO_LARGE
    assert sum(map(len, got)) == skills.MAX_REFERENCE_BYTES + 1


def _snapshot(root):
    return {
        str(p.relative_to(root)): (p.lstat().st_size, p.lstat().st_mtime_ns) for p in sorted(root.rglob("*"))
    }


def test_discover_find_and_read_body_write_and_print_nothing(tmp_path, cfg, proj, capsys):
    _write(cfg, None, entry="m")
    _skill(proj / ".localharness" / "skills", "p")
    _write(cfg, _fm("other"), entry="bad")
    _write(cfg, None, entry="late")
    before = _snapshot(tmp_path)
    found = SkillSet.discover(cfg, proj)
    found.find("late"), found.find("bad"), found.find("m"), found.find("../x"), found.find("none")
    for record in found.skills:
        found.read_body(record)
    assert _snapshot(tmp_path) == before
    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def _found(cfg, extra):
    _write(cfg, _fm(extra=extra))
    return SkillSet.discover(cfg, None)


def test_purpose_and_tools_parse_onto_the_skill(cfg):
    found = _found(cfg, "purpose: Drafts release notes\ntools: [read, grep, read]\n")
    assert [(s.purpose, s.tools) for s in found.skills] == [("Drafts release notes", ("read", "grep"))]
    assert found.skipped == ()


def test_a_skill_without_purpose_or_tools_has_empty_purpose_and_no_tools(cfg):
    found = _found(cfg, "")
    assert [(s.purpose, s.tools) for s in found.skills] == [("", None)]


def test_empty_tools_list_is_an_empty_tuple_and_null_values_are_absent(cfg):
    assert [s.tools for s in _found(cfg, "tools: []\n").skills] == [()]
    found = _found(cfg, "tools:\npurpose:\n")
    assert [(s.purpose, s.tools) for s in found.skills] == [("", None)]


@pytest.mark.parametrize("extra", [
    "purpose: 7\n", "purpose: [a]\n", 'purpose: "a\\nb"\n', 'purpose: "a\\u202eb"\n',
    'purpose: "a\\u200bb"\n', 'purpose: "a\\x07b"\n', "purpose: |\n  one\n  two\n",
    f"purpose: {'p' * 1025}\n",
])
def test_a_malformed_purpose_skips_the_skill_with_its_own_line(cfg, extra):
    found = _found(cfg, extra)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == [_expect(skills.BAD_PURPOSE, path=cfg / "skills" / "a" / "SKILL.md")]


def test_a_purpose_at_the_cap_is_kept(cfg):
    assert [s.purpose for s in _found(cfg, f"purpose: {'p' * 1024}\n").skills] == ["p" * 1024]


@pytest.mark.parametrize("extra", [
    "tools: read\n", "tools: {a: b}\n", "tools: [1]\n", "tools: [read, null]\n", 'tools: [""]\n',
    'tools: [" "]\n', 'tools: ["a b"]\n', 'tools: ["a\\u202eb"]\n', 'tools: ["a\\nb"]\n',
])
def test_malformed_tools_skip_the_skill_with_their_own_line(cfg, extra):
    found = _found(cfg, extra)
    assert found.skills == ()
    assert [skip_line(s) for s in found.skipped] == [_expect(skills.BAD_TOOLS, path=cfg / "skills" / "a" / "SKILL.md")]


def test_tools_keep_their_listed_order_with_duplicates_removed(cfg):
    found = _found(cfg, "tools: [z, a, z, m, a]\n")
    assert found.skills[0].tools == ("z", "a", "m")


def test_only_the_display_and_parse_modules_read_a_purpose():
    allowed = {"orchestrator/cards.py", "config/skills.py", "tools/builtin/agent_tool.py"}
    readers = {
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if any(isinstance(n, ast.Attribute) and n.attr == "purpose" for n in ast.walk(ast.parse(path.read_text())))
    }
    assert readers <= allowed
