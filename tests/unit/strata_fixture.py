"""A fake Strata install tree and a Strata record over the recorded Flash-Next shards; nothing here runs anything."""
from __future__ import annotations

import json
import os
from pathlib import Path

from localharness.config.models import ManagedServerConfig
from tests.unit.gguf_support import load_fixture, write_recorded, write_sparse

CONFIG_NAME = "strata-swift-IQ3_XXS.json"
PROJECTOR = "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf"


def flash_next(directory: Path) -> tuple[Path, Path]:
    fixture = load_fixture("flash-next-iq3-xxs")
    shard1 = write_recorded(directory, fixture)[0]
    return shard1, write_sparse(directory / PROJECTOR, fixture["mmproj_bytes"])


def strata_config(model: Path | str, **over) -> dict:
    return {"model_name": "swift-1.5-iq3_xxs",
            "args": ["--native", str(model), "--max-context", "65536", "--kv", "q8_0", "--expert-cache", "auto",
                     "--spec", "4", "--spec-min-p", "0.5", "--mtp"],
            "parallel": 1, "gpu": 0, **over}


def _file(path: Path, mode: int, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def strata_tree(root: Path, *, config: dict | None = None, model: Path | str | None = None) -> Path:
    engine = root / "Strata"
    _file(engine / "serve" / "server.py", 0o644)
    _file(engine / ".venv" / "bin" / "python", 0o755)
    _file(engine / "engine" / "strata", 0o755)
    _file(engine / CONFIG_NAME, 0o644, json.dumps(config or strata_config(model)))
    for directory in (d for d, _, _ in os.walk(engine)):
        os.chmod(directory, 0o755)
    return engine


def strata_record(engine_dir: Path, model: Path, mmproj: Path, **over) -> ManagedServerConfig:
    return ManagedServerConfig(
        runtime="strata", engine_dir=str(engine_dir), engine_config=CONFIG_NAME, model=str(model),
        mmproj=str(mmproj), served_name="swift-1.5-iq3_xxs", port=8083, gpu=True, **over)
