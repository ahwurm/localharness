"""MEMP-08, composed: the memory plugin's doctor checks, reached through the real `localharness doctor`
(PAPI-08: off is off, not failed).

With memory on, doctor prints the plugin's `memory-db` row per `agents/*/memory.db` (opened
read-only) and one `memory-embedding` row (the local Hugging Face cache, nothing downloaded). With
memory off — by either key — the off row names the enable command, the memory checks do not run, and
memory adds no failure. The suite-wide conftest fixture that treats the embedding model as cached is
undone here where the real lookup is the point (`_real_embedding_check`).
"""
from __future__ import annotations

import asyncio
import importlib.util
import re
from pathlib import Path

import huggingface_hub
import pytest
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.memory import plugin as memory_plugin
from tests.unit.test_memory_config_surface import CONFIG
from tests.unit.test_memory_plugin_doctor import _real_db

runner = CliRunner()
MODEL = "Qwen/Qwen3-Embedding-0.6B"
REAL_EMBEDDING_CHECK = memory_plugin._embedding_check  # captured before conftest patches it per test
REAL_LOOKUP = huggingface_hub.try_to_load_from_cache


def _config_dir(tmp_path: Path, monkeypatch, extra: str = "") -> Path:
    d = tmp_path / "lh"
    d.mkdir()
    (d / "config.yaml").write_text(CONFIG + extra, encoding="utf-8")
    monkeypatch.setenv("LOCALHARNESS_DIR", str(d))
    monkeypatch.setenv("COLUMNS", "400")
    return d


def _real_embedding_check(monkeypatch) -> None:
    monkeypatch.setattr(memory_plugin, "_embedding_check", REAL_EMBEDDING_CHECK)


def _model_cached(monkeypatch) -> None:
    """The real check, against a cache lookup that finds the model and an importable package."""
    _real_embedding_check(monkeypatch)
    real = importlib.util.find_spec
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache",
                        lambda repo_id, filename, **_: f"/cache/{repo_id}/{filename}")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: object()
                        if name == "sentence_transformers" else real(name, *a))


def _doctor() -> tuple[str, int, int]:
    result = runner.invoke(app, ["doctor"])
    assert result.exception is None or isinstance(result.exception, SystemExit), result.output
    m = re.search(r"(\d+) issue\(s\) found", result.output)
    return result.output, result.exit_code, int(m.group(1)) if m else 0


def _lines(out: str, name: str) -> list[str]:
    return [line for line in out.splitlines() if re.search(rf"\b{re.escape(name)}\b", line)]


def test_doctor_runs_the_memory_checks_when_on(tmp_path, monkeypatch):
    d = _config_dir(tmp_path, monkeypatch)
    db = asyncio.run(_real_db(d))
    _model_cached(monkeypatch)
    out, _, _ = _doctor()
    (db_row,) = _lines(out, "memory-db")
    assert "✓" in db_row and str(db) in db_row, db_row
    (emb_row,) = _lines(out, "memory-embedding")
    assert "✓" in emb_row and MODEL in emb_row, emb_row


def test_doctor_embedding_missing_fails_with_hint(tmp_path, monkeypatch):
    """No patch on the lookup: HF_HOME/HF_HUB_CACHE point at an empty dir. huggingface_hub reads
    HF_HUB_CACHE once, at import, so the module constant is pointed there too — the env's effect,
    not a stand-in for the check."""
    d = _config_dir(tmp_path, monkeypatch)
    asyncio.run(_real_db(d))
    _model_cached(monkeypatch)
    _, _, issues_when_cached = _doctor()

    empty = tmp_path / "empty-hf"
    empty.mkdir()
    # The real lookup again — only the cache location is chosen. The package probe stays "found"
    # (`_model_cached`) so this is the model-missing row on any box, CI's without the package too;
    # test_memory_plugin_doctor.test_embedding_package_missing owns the package-missing row.
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", REAL_LOOKUP)
    monkeypatch.setenv("HF_HOME", str(empty))
    monkeypatch.setenv("HF_HUB_CACHE", str(empty))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(empty))
    out, code, issues = _doctor()
    (emb_row,) = _lines(out, "memory-embedding")
    assert "✗" in emb_row and MODEL in emb_row, emb_row
    assert f"hf download {MODEL}" in out, out
    assert code != 0 and issues == issues_when_cached + 1, (issues, issues_when_cached, out)


@pytest.mark.parametrize("extra", ["memory:\n  enabled: false\n", "org:\n  memory_enabled: false\n"],
                         ids=["memory.enabled", "org.memory_enabled"])
def test_doctor_memory_off_is_off_not_failed(tmp_path, monkeypatch, extra):
    """Off by either key: the off row with the enable command, no memory checks, no failure — even
    on a box where the embedding check WOULD fail (the real lookup, an empty cache)."""
    d = _config_dir(tmp_path, monkeypatch)
    asyncio.run(_real_db(d))
    _model_cached(monkeypatch)
    _, _, issues_when_on = _doctor()

    (d / "config.yaml").write_text(CONFIG + extra, encoding="utf-8")
    _real_embedding_check(monkeypatch)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    out, _, issues = _doctor()
    (row,) = [line for line in out.splitlines() if "memory: off" in line]
    assert "localharness plugins enable memory" in row and "✗" not in row, row
    assert not _lines(out, "memory-db") and not _lines(out, "memory-embedding"), out
    assert issues == issues_when_on, (issues, issues_when_on, out)


def test_doctor_no_database_yet_skips(tmp_path, monkeypatch):
    _config_dir(tmp_path, monkeypatch)
    _model_cached(monkeypatch)
    out, _, _ = _doctor()
    (db_row,) = _lines(out, "memory-db")
    assert "no memory database yet — created on first start" in db_row, db_row
    assert "✗" not in db_row, db_row
