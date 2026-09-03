"""Bench + model-overlay resolution pinning (v0.13 critique amendments #2 and #3).

Both guarantees are byte-identical TODAY and load-bearing the moment phase 39 teaches
``resolve_config_dir`` to discover a workspace ``.localharness/`` up-tree:

- #3 bench pins NON-DISCOVERING resolution: bench's own ConfigLoader constructions name their
  config dir, so whatever CWD a run happens to start in cannot steer it. This repo's own sibling
  worktrees make that collision near-certain during v0.13's dogfood.
- #2 model-swap writes pin the GLOBAL overlay: there is ONE physical GPU daemon, so a workspace
  must never fork ``server.model`` against it.

These tests are the regression guard for phase 39/40 — if a later change makes either site follow
a workspace layer, they fail here rather than in a bench run or a GPU relaunch.
"""
from __future__ import annotations

import asyncio

import yaml

from localharness.config.loader import ConfigLoader
from localharness.config.paths import global_config_dir


def _write_config(directory, model: str, provider: str = "vllm") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "version": "1",
                "provider": {
                    "provider_type": provider,
                    "base_url": "http://localhost:8081/v1",
                    "default_model": model,
                    "available_models": [model],
                },
                "org": {"default_model": model},
            }
        ),
        encoding="utf-8",
    )


# --- Amendment #3: bench resolution is non-discovering ------------------------------------ #


def test_synthesize_default_entry_ignores_cwd_localharness(monkeypatch, tmp_path):
    """A ``.localharness/`` sitting in the process CWD has NO influence on what bench runs.

    The global dir says ``global-model``; the CWD-adjacent directory says ``worktree-model``.
    Bench must report the global one — running a sibling worktree's model while claiming the
    configured backend is exactly the silent mis-attribution amendment #3 exists to prevent.
    """
    from localharness.bench.orchestrator import _synthesize_default_entry

    global_home = tmp_path / "global" / ".localharness"
    _write_config(global_home, "global-model")
    project = tmp_path / "project"
    _write_config(project / ".localharness", "worktree-model", provider="ollama")

    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(global_home))
    monkeypatch.chdir(project)

    entry = _synthesize_default_entry()
    assert entry.model_id == "global-model"
    assert entry.provider == "vllm"


def test_bench_names_its_config_dir_explicitly(monkeypatch, tmp_path):
    """Structural half: BOTH bench sites pass an explicit ``config_dir``.

    Behavior alone can't prove this today (bare and explicit resolve identically); a
    construction that NAMES its dir is what makes discovery structurally unable to reach it.
    """
    from localharness.bench import orchestrator as bench_orch
    from localharness.bench.config import BenchConfig, MatrixEntry

    global_home = tmp_path / "global" / ".localharness"
    _write_config(global_home, "global-model")
    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(global_home))

    seen: list[object] = []
    real_loader = ConfigLoader

    class SpyLoader(ConfigLoader):
        def __init__(self, *args, config_dir=None, **kwargs):
            seen.append(config_dir)
            super().__init__(*args, config_dir=config_dir, **kwargs)

    monkeypatch.setattr("localharness.config.loader.ConfigLoader", SpyLoader)

    bench_orch._synthesize_default_entry()

    # _resolve_matrix's active-provider check: a stale matrix pinning a foreign provider must
    # re-read the ACTIVE config — through the same explicitly-named dir.
    stale = BenchConfig(
        corpus_path=str(tmp_path / "corpus"),
        results_path=str(tmp_path / "results"),
        matrix=[MatrixEntry(name="stale", provider="ollama", model_id="stale-model")],
    )
    bench_orch._resolve_matrix(stale, matrix=False, models=None)

    assert seen, "bench never constructed a ConfigLoader"
    assert all(d is not None for d in seen), f"a bench ConfigLoader was left bare: {seen}"
    assert all(real_loader(config_dir=d)._config_dir == global_config_dir() for d in seen)


# --- Amendment #2: the model overlay write targets the global layer ----------------------- #


def test_persist_default_model_writes_to_the_named_global_layer(tmp_path):
    """An explicit ``config_dir`` is a FULL REPLACEMENT (LAYR-02): the overlay lands there, and
    never in the real ``~/.localharness``."""
    from localharness.cli import model_ops

    home = tmp_path / ".localharness"
    _write_config(home, "model-a")
    harness = ConfigLoader(config_dir=home).load_harness()

    warning = asyncio.run(
        model_ops.persist_default_model(harness, "model-b", config_dir=home)
    )

    assert warning is None
    overlay = home / "overrides.yaml"
    assert overlay.exists(), "overlay did not land under the named config dir"
    assert yaml.safe_load(overlay.read_text())["provider"]["default_model"] == "model-b"
    assert overlay == global_config_dir(home) / "overrides.yaml"
    assert ConfigLoader(config_dir=home).load_harness().provider.default_model == "model-b"


def test_persist_active_endpoint_writes_to_the_named_global_layer(tmp_path):
    """Same pin for the cross-endpoint switch — it touches only ``active_endpoint``, but it must
    touch it in the GLOBAL overlay (one physical daemon, one machine-wide record)."""
    from types import SimpleNamespace

    from localharness.cli import model_ops

    home = tmp_path / ".localharness"
    _write_config(home, "model-a")
    harness = ConfigLoader(config_dir=home).load_harness()
    endpoint = SimpleNamespace(
        name="peer",
        base_url="http://localhost:11434/v1",
        provider_type="ollama",
        api_key="none",
    )

    asyncio.run(
        model_ops.persist_active_endpoint(harness, endpoint, "peer-model", config_dir=home)
    )

    overlay = global_config_dir(home) / "overrides.yaml"
    assert overlay.exists()
    record = yaml.safe_load(overlay.read_text())["active_endpoint"]
    assert record["model"] == "peer-model" and record["base_url"] == endpoint.base_url
    # provider/server identity stays pristine — this write is additive only.
    assert "provider" not in yaml.safe_load(overlay.read_text())
