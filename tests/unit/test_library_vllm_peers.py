"""A cached Hugging Face text-generation checkpoint is a vLLM peer line only when a `vllm` binary is found, the
binary and where it came from on the line the person is shown. The cache listing is faked at one seam over tmp
snapshots, PATH and the harness venv live under tmp_path, the interview is scripted."""
from __future__ import annotations

import io
import json
import os
import re
from types import SimpleNamespace

import pytest
import yaml
from rich.console import Console

from localharness.cli import library
from localharness.cli.workspace import shown_path
from localharness.config import trust
from localharness.provider import model_library
from tests.unit import test_init_discovery_records as world
from tests.unit.interview_support import scripted
from tests.unit.test_init_core_setup import _exited, _hermetic, _init  # noqa: F401  (autouse fixture)
from tests.unit.test_init_discovery_records import _peers

box = world.box
GB16 = 16 * 1024**3
QWEN = {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}


def cache_repo(tmp_path, repo_id, config=QWEN, *, size=GB16, refs=("main",), repo_type="model", snap="abc123",
               modified=1.0, raw=None):
    """A cache repo as `scan_cache_dir` lists it, over a real snapshot directory under tmp_path."""
    directory = tmp_path / "hfcache" / ("models--" + repo_id.replace("/", "--")) / "snapshots" / snap
    directory.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (directory / "config.json").write_bytes(raw)
    elif config is not None:
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rev = SimpleNamespace(commit_hash=snap, snapshot_path=directory, refs=frozenset(refs), last_modified=modified)
    return SimpleNamespace(repo_id=repo_id, repo_type=repo_type, size_on_disk=size, revisions=[rev])


def vllm_on_path(box, name="vllm"):
    exe = box.bin.parent / name
    exe.write_text("#!/bin/sh\n", encoding="utf-8")
    exe.chmod(0o755)
    return exe


@pytest.fixture
def vbox(box, monkeypatch):
    """The two-model box after its first run, a cache listing the test fills, and a console that holds the lines."""
    _exited(_init(box.cfg), 0)
    box.printed = io.StringIO()
    monkeypatch.setattr(library, "console", Console(file=box.printed, width=300))
    box.repos = []
    box.listed = []

    def repos():
        box.listed.append(1)
        return tuple(box.repos)

    monkeypatch.setattr(model_library, "_cache_repos", repos)
    return box


def sync(box, **kw) -> library.Sync:
    return library.sync(box.cfg, scope=kw.pop("scope", "new"), person=kw.pop("person", True), **kw)


def peer(box, name):
    return next((p for p in _peers(box.cfg) if p["name"] == name), None)


def test_a_cached_checkpoint_with_vllm_on_path_is_listed_with_its_binary_and_set_up_on_enter(vbox, monkeypatch, tmp_path):
    exe = vllm_on_path(vbox)
    vbox.repos.append(cache_repo(tmp_path, "org/Qwen3-8B"))
    asked = scripted(monkeypatch, "set-up")

    synced = sync(vbox)

    [bullet] = [line for line in vbox.printed.getvalue().splitlines() if line.startswith("• ")]
    assert re.fullmatch(r"• Qwen3-8B \(vLLM\) ← org/Qwen3-8B · 16\.0 GB · first seen \d{4}-\d\d-\d\d"
                        rf" · binary: {re.escape(shown_path(exe))} \(on PATH\)", bullet)
    [question] = asked.asked
    assert question.question == "New model Qwen3-8B — set it up?"
    record = peer(vbox, "Qwen3-8B")
    assert synced.appended == 1 and record["provider_type"] == "vllm"
    life = record["lifecycle"]
    assert {k: life[k] for k in ("runtime", "launch", "binary", "model", "gpu")} == {
        "runtime": "vllm", "launch": "binary", "binary": str(exe), "model": "org/Qwen3-8B", "gpu": True}
    assert "served_name" not in life
    taken = {p["lifecycle"]["port"] for p in _peers(vbox.cfg) if p["name"] != "Qwen3-8B"} | {8081}
    assert life["port"] not in taken and record["base_url"] == f"http://127.0.0.1:{life['port']}/v1"
    adopted = {e["name"]: e["shown"] for e in trust.recorded_machine(vbox.cfg)["entries"]}
    assert adopted["extra_endpoints.Qwen3-8B.lifecycle.binary"] == str(exe)
    assert adopted["extra_endpoints.Qwen3-8B.lifecycle.model"] == "org/Qwen3-8B"
    entries = json.loads(model_library.library_path(vbox.cfg).read_text(encoding="utf-8"))["entries"]
    assert os.path.realpath(vbox.repos[0].revisions[0].snapshot_path) not in entries


def test_a_second_start_asks_nothing_about_a_set_up_checkpoint(vbox, monkeypatch, tmp_path):
    vllm_on_path(vbox)
    vbox.repos.append(cache_repo(tmp_path, "org/Qwen3-8B"))
    scripted(monkeypatch, "set-up")
    sync(vbox)
    asked = scripted(monkeypatch)

    again = sync(vbox)

    assert asked.asked == [] and again.appended == 0 and len(_peers(vbox.cfg)) == 2
    assert yaml.safe_load((vbox.cfg / "overrides.yaml").read_text(encoding="utf-8"))
