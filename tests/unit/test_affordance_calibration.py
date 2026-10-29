"""The shipped calibration record matches the prompts, descriptions and sequence cap in the tree."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pytest
import yaml

import localharness.agent.affordance as affordance
from localharness.agent.affordance import descriptions_digest, load_calibration
from localharness.config.skills import SkillSet
from localharness.memory.config import MemoryConfig
from localharness.memory.resonance import EMBED_MAX_TOKENS

pytestmark = pytest.mark.xfail(strict=True, reason="placeholder calibration record: run python scripts/gen_calibration.py, then delete this marker")

ROOT = Path(__file__).resolve().parents[2]
LABELLED = ROOT / "src" / "localharness" / "assets" / "affordance"
RECORD = Path(affordance.__file__).resolve().parent.parent / "assets" / "affordance" / "calibration.yaml"
STALE = "calibration is stale: run python scripts/gen_calibration.py"
DEFAULT = MemoryConfig.model_fields["embedding_model"].default


def entry() -> dict:
    raw = yaml.safe_load(RECORD.read_text(encoding="utf-8"))
    assert list(raw) == [DEFAULT], STALE
    return raw[DEFAULT]


def test_the_record_is_keyed_by_the_default_embedding_model_and_its_rows_load():
    rows = entry()["rows"]
    calibration = load_calibration(DEFAULT)
    assert calibration is not None, STALE
    barred = [s for item, label, top, s in rows if label != "matching" or top != item]
    assert all(s <= calibration.threshold for s in barred), STALE


def test_the_record_pins_the_prompts_and_the_descriptions_in_the_tree(tmp_path):
    shutil.copytree(LABELLED / "skills", tmp_path / "skills")
    skills = SkillSet.discover(tmp_path, None).skills
    assert entry()["prompts_sha256"] == hashlib.sha256((LABELLED / "prompts.yaml").read_bytes()).hexdigest(), STALE
    assert entry()["descriptions_sha256"] == descriptions_digest(skills), STALE


def test_the_record_pins_the_sequence_cap():
    assert entry()["max_seq"] == EMBED_MAX_TOKENS, STALE
