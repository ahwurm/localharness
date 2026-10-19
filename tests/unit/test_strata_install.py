"""Strata install sensing: lstat and capped reads only, the roots, the family test, and strata_dir as a machine-only key."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from localharness.config.loader import ConfigLoader
from localharness.provider.model_library import flash_next_quant, strata_install, strata_roots
from tests.unit.gguf_support import write_gguf
from tests.unit.strata_fixture import CONFIG_NAME, flash_next, strata_config, strata_tree

PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m"}


@pytest.fixture(autouse=True)
def nothing_runs(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("sensing ran a program")
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _real(path: Path) -> str:
    return os.path.realpath(path)


def _setup(tmp_path: Path, parent: str = "") -> tuple[Path, Path, Path]:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    shard1, _ = flash_next(models)
    return strata_tree(tmp_path / parent, model=shard1), shard1, models


def _sense(engine: Path, shard1: Path):
    return strata_install((engine,), {_real(shard1)})


def _layers(tmp_path: Path, *, g_cfg=None, ws_cfg=None):
    g, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    ws.mkdir(parents=True)
    for path, data in ((g / "config.yaml", {"version": "1", "provider": PROVIDER, **(g_cfg or {})}),
                       (ws / "config.yaml", ws_cfg)):
        if data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.dump(data), encoding="utf-8")
    return ConfigLoader(config_dir=g, local_config_dir=ws), ws


def test_a_fake_install_at_strata_dir_is_sensed(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    loader, _ = _layers(tmp_path, g_cfg={"autosense": {"strata_dir": str(engine)}})
    harness = loader.load_harness()
    install = strata_install(strata_roots(harness.autosense), {_real(shard1)})
    assert install.engine_dir == engine
    assert install.configs == ((CONFIG_NAME, _real(shard1)),)


def test_a_symlinked_server_py_is_not_an_install(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    real = engine / "serve" / "real.py"
    real.write_text("")
    (engine / "serve" / "server.py").unlink()
    (engine / "serve" / "server.py").symlink_to(real)
    assert _sense(engine, shard1) is None


@pytest.mark.parametrize("directory, mode", [(".", 0o775), ("serve", 0o777), ("engine", 0o775), (".venv/bin", 0o757)])
def test_a_group_or_world_writable_directory_is_not_an_install(tmp_path, directory, mode):
    engine, shard1, _ = _setup(tmp_path)
    assert _sense(engine, shard1) is not None
    (engine / directory).chmod(mode)
    assert _sense(engine, shard1) is None


def test_a_missing_or_non_executable_engine_is_not_an_install(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    (engine / "engine" / "strata").chmod(0o644)
    assert _sense(engine, shard1) is None
    (engine / "engine" / "strata").unlink()
    assert _sense(engine, shard1) is None


def test_a_config_naming_an_unknown_file_is_not_an_install(tmp_path):
    engine, shard1, models = _setup(tmp_path)
    assert strata_install((engine,), {_real(models / "other.gguf")}) is None


def test_a_venv_python_linked_to_a_closed_executable_is_an_install(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    base = tmp_path / "base-python"
    base.write_text("")
    base.chmod(0o755)
    (engine / ".venv" / "bin" / "python").unlink()
    (engine / ".venv" / "bin" / "python").symlink_to(base)
    assert _sense(engine, shard1) is not None


def test_a_venv_python_linked_to_a_writable_file_or_nothing_is_not_an_install(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    python = engine / ".venv" / "bin" / "python"
    base = tmp_path / "base-python"
    base.write_text("")
    base.chmod(0o777)
    python.unlink()
    python.symlink_to(base)
    assert _sense(engine, shard1) is None
    base.unlink()
    assert _sense(engine, shard1) is None


def test_a_config_that_is_not_utf8_is_skipped(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    (engine / "strata-a.json").write_bytes(b"\xff\xfe{")
    (engine / "strata-a.json").chmod(0o644)
    assert _sense(engine, shard1).configs == ((CONFIG_NAME, _real(shard1)),)


def test_a_config_whose_native_has_a_nul_byte_is_skipped(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    (engine / "strata-a.json").write_text(json.dumps(strata_config(str(shard1) + "\x00")))
    (engine / "strata-a.json").chmod(0o644)
    assert _sense(engine, shard1).configs == ((CONFIG_NAME, _real(shard1)),)


def test_configs_are_listed_in_name_order_and_only_those_naming_a_known_file(tmp_path):
    engine, shard1, models = _setup(tmp_path)
    for name, model in (("strata-z.json", shard1), ("strata-a.json", shard1), ("strata-m.json", models / "x.gguf")):
        (engine / name).write_text(json.dumps(strata_config(model)))
        (engine / name).chmod(0o644)
    names = [n for n, _ in _sense(engine, shard1).configs]
    assert names == sorted(names) == ["strata-a.json", CONFIG_NAME, "strata-z.json"]


def test_a_relative_native_resolves_against_the_engine_dir(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    (engine / CONFIG_NAME).write_text(json.dumps(strata_config(os.path.relpath(shard1, engine))))
    assert _sense(engine, shard1).configs == ((CONFIG_NAME, _real(shard1)),)


def test_a_set_strata_dir_is_the_only_root_consulted(home):
    _, shard1, _ = _setup(home, "src")
    other = home / "elsewhere"
    other.mkdir()
    loader, _ = _layers(home, g_cfg={"autosense": {"strata_dir": str(other)}})
    roots = strata_roots(loader.load_harness().autosense)
    assert roots == (other,)
    assert strata_install(roots, {_real(shard1)}) is None


def test_the_default_roots_prefer_home_strata_then_src_strata(home):
    from localharness.config.models import AutosenseConfig
    roots = strata_roots(AutosenseConfig())
    assert roots == (home / "Strata", home / "src" / "Strata")
    assert strata_install(roots, set()) is None
    _, shard1, _ = _setup(home, "src")
    assert strata_install(roots, {_real(shard1)}).engine_dir == home / "src" / "Strata"
    _setup(home)
    assert strata_install(roots, {_real(shard1)}).engine_dir == home / "Strata"


def test_a_root_that_is_a_symlink_is_not_an_install(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(engine)
    assert strata_install((link,), {_real(shard1)}) is None


def test_sensing_caches_nothing(tmp_path):
    engine, shard1, _ = _setup(tmp_path)
    assert _sense(engine, shard1) is not None
    engine.chmod(0o777)
    assert _sense(engine, shard1) is None
    engine.chmod(0o755)
    assert _sense(engine, shard1) is not None


def test_flash_next_quant_names_the_swift_shard(tmp_path):
    shard1, _ = flash_next(tmp_path)
    assert flash_next_quant(shard1) == "IQ3_XXS"


def test_a_renamed_shard_or_another_architecture_is_not_flash_next(tmp_path):
    shard1, _ = flash_next(tmp_path)
    renamed = shard1.with_name("renamed.gguf")
    renamed.write_bytes(shard1.read_bytes()[:1 << 16])
    assert flash_next_quant(renamed) is None
    other = write_gguf(tmp_path / shard1.name.replace("IQ3_XXS", "Q4_K"), {"general.architecture": "qwen35"})
    assert flash_next_quant(other) is None
    assert flash_next_quant(tmp_path / "missing.gguf") is None


def test_a_workspace_strata_dir_is_dropped_and_the_global_stands(tmp_path):
    loader, ws = _layers(tmp_path, ws_cfg={"autosense": {"strata_dir": "/tmp/evil"}})
    assert loader.load_harness().autosense.strata_dir is None
    assert any("strata_dir" in w and str(ws / "config.yaml") in w for w in loader.harness_warnings)
    loader, _ = _layers(tmp_path / "b", g_cfg={"autosense": {"strata_dir": "/opt/Strata"}},
                        ws_cfg={"autosense": {"strata_dir": "/tmp/evil"}})
    assert loader.load_harness().autosense.strata_dir == "/opt/Strata"
