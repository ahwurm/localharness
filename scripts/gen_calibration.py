#!/usr/bin/env python3
"""Write the skill cue's calibration record from the local embedding model and the labelled prompts."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import subprocess
from pathlib import Path

import yaml

from localharness.agent.affordance import LABELS, calibrate, descriptions_digest, scores, snapshot_files
from localharness.config.skills import Skill, _check
from localharness.memory.resonance import DEFAULT_EMBEDDING_MODEL, ResonanceEngine

ROOT = Path(__file__).resolve().parent.parent


def read_skills(directory: Path, label: str, bundled: frozenset[str]) -> list[Skill]:
    found = [_check(label, directory, directory.resolve(), path.parent.name, bundled)
             for path in sorted(directory.glob("*/SKILL.md"))]
    if skipped := [s for s in found if not isinstance(s, Skill)]:
        raise SystemExit("\n".join(f"{s.path}: {s.reason}" for s in skipped))
    return found


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests" / "fixtures" / "skills_live" / "skills")
    parser.add_argument("--bundled", type=Path, default=ROOT / "src" / "localharness" / "assets" / "skills")
    parser.add_argument("--prompts", type=Path, default=ROOT / "tests" / "fixtures" / "skills_live" / "prompts.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "src" / "localharness" / "assets" / "affordance" / "calibration.yaml")
    args = parser.parse_args()
    prompts = yaml.safe_load(args.prompts.read_text(encoding="utf-8"))["skills"]
    bundled = read_skills(args.bundled, "bundled", frozenset())
    skills = read_skills(args.fixtures, "machine", frozenset(s.name for s in bundled)) + bundled
    if {s.name for s in skills} != set(prompts):
        raise SystemExit(f"skills {sorted(s.name for s in skills)} differ from the prompts' {sorted(prompts)}")
    engine = ResonanceEngine(DEFAULT_EMBEDDING_MODEL)
    folder = engine.ensure_local()
    if folder is None:
        raise SystemExit(f"{engine.model_name} is not complete in the local Hugging Face cache; "
                         f"`hf download {engine.model_name}` fetches it, this script never downloads")
    docs = engine.embed_docs([s.description for s in skills])
    rows = []
    for skill, group in prompts.items():
        for kind in LABELS:
            for text in group[kind]:
                s = scores(docs, engine.embed_query(text))
                i = int(s.argmax())
                rows.append([skill, kind, skills[i].name, float(s[i])])
    threshold = calibrate(rows)
    entry = {
        "rows": rows,
        "snapshot_files": snapshot_files(folder),
        "max_seq": engine._model.max_seq_length,
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "descriptions_sha256": descriptions_digest(skills),
        "date": datetime.date.today(),
        "commit": f"{git('rev-parse', 'HEAD')} ({'dirty' if git('status', '--porcelain') else 'clean'} tree)",
    }
    args.out.write_text(yaml.safe_dump({engine.model_name: entry}, sort_keys=False, default_flow_style=None,
                                       allow_unicode=True), encoding="utf-8")
    print(f"wrote {args.out}: {len(rows)} rows, threshold {threshold:.6f}")


if __name__ == "__main__":
    main()
