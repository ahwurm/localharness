"""Skill discovery: the record a SKILL.md becomes, the rule data, and the import boundaries."""
from __future__ import annotations

import ast
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
