"""Skill-reference pinning and withdrawal live on the ContentStore, with no task record in sight."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import localharness
from localharness.agent.affordance import pinned_skills
from localharness.agent.context import MAX_ACTIVE_REFERENCES, ContentStore
from localharness.config import skills
from localharness.config.skills import SkillSet
from localharness.tools.builtin.skill_tool import SkillTool
from tests.unit.test_skills_discovery import _skill


def fill(store, step, *names):
    for name in names:
        store.declare_active_reference(step, store.put(name), name)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(skills, "PACKAGE_DIR", tmp_path / "pkg")
    (tmp_path / "proj" / ".localharness").mkdir(parents=True)
    return SimpleNamespace(cfg=tmp_path / "cfg", proj=tmp_path / "proj", bundled=tmp_path / "pkg" / "assets" / "skills")


async def test_a_skill_load_pins_through_the_store_alone(dirs):
    _skill(dirs.bundled, "notes")
    store = ContentStore()
    result = await SkillTool(SkillSet.discover(dirs.cfg, dirs.proj), store, lambda: 50_000).run(name="notes")
    (active,) = store.active_references
    assert (active.source, store.active_step) == ("skill notes", "skill notes")
    assert store.get(active.handle) == result.output


def test_an_empty_store_has_room_under_the_sources_own_step():
    assert ContentStore().room("skill x") == ("skill x", True)


def test_a_different_active_step_is_the_step_reported():
    store = ContentStore()
    fill(store, "build", "a")
    assert store.room("skill x") == ("build", True)


def test_the_fourth_source_in_a_step_fits_and_the_fifth_does_not():
    store = ContentStore()
    fill(store, "build", *"abc")
    assert store.room("skill x") == ("build", True)
    fill(store, "build", "d")
    assert MAX_ACTIVE_REFERENCES == 4 and store.room("skill x") == ("build", False)


def test_a_source_already_held_fits_a_full_step_and_sources_compare_exactly():
    store = ContentStore()
    fill(store, "build", "skill Release", "b", "c", "d")
    assert store.room("skill Release")[1] is True
    assert store.room("skill release")[1] is False


def test_room_changes_nothing():
    store = ContentStore()
    fill(store, "build", "a", "b")
    seen = (store.active_step, store.active_references)
    store.room("skill x")
    assert (store.active_step, store.active_references) == seen


def test_withdraw_reports_whether_anything_was_declared_and_leaves_nothing():
    store = ContentStore()
    assert store.withdraw() is False
    fill(store, "build", "a")
    assert store.withdraw() is True
    assert (store.active_step, store.active_references) == (None, ())
    assert store.withdraw() is False


def test_pinned_skills_reads_the_stores_skill_references():
    store = ContentStore()
    skills = [SimpleNamespace(name="notes"), SimpleNamespace(name="other")]
    assert pinned_skills(store, skills) == pinned_skills(None, skills) == set()
    fill(store, "build", "skill notes", "notes.md")
    assert pinned_skills(store, skills) == {"notes"}


@pytest.mark.parametrize("module", ["tools/builtin/skill_tool.py", "agent/affordance.py"])
def test_neither_the_skill_tool_nor_the_cue_names_the_task_record(module):
    source = (Path(localharness.__file__).parent / module).read_text()
    assert "task_record" not in source and "TaskState" not in source
