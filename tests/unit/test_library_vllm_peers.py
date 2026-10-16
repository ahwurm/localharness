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
from localharness.provider import model_library, server, tuning
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


def test_without_a_vllm_binary_nothing_is_scanned_listed_or_asked(vbox, monkeypatch, tmp_path):
    vbox.repos.append(cache_repo(tmp_path, "org/Qwen3-8B"))
    asked = scripted(monkeypatch)

    synced = sync(vbox)
    sync(vbox, scope="all", choose=True)

    assert synced == library.Sync() and asked.asked == [] and vbox.listed == []
    assert "Qwen3-8B" not in vbox.printed.getvalue() and len(_peers(vbox.cfg)) == 1


def test_the_harness_venv_binary_is_named_with_its_source(vbox, monkeypatch, tmp_path):
    venv = server.venv_vllm_bin(library.global_config_dir(vbox.cfg))
    venv.parent.mkdir(parents=True)
    venv.write_text("#!/bin/sh\n", encoding="utf-8")
    vllm_on_path(vbox)
    vbox.repos.append(cache_repo(tmp_path, "org/Qwen3-8B"))
    scripted(monkeypatch, "set-up")

    sync(vbox)

    assert f"binary: {shown_path(venv)} (harness venv)" in vbox.printed.getvalue()
    assert peer(vbox, "Qwen3-8B")["lifecycle"]["binary"] == str(venv)


def test_a_repo_a_record_names_is_never_listed_and_two_revisions_are_one_entry(vbox, monkeypatch, tmp_path):
    exe = vllm_on_path(vbox)
    named = cache_repo(tmp_path, "org/Named")
    two = cache_repo(tmp_path, "org/Two", snap="old", refs=(), modified=9.0)
    two.revisions.append(cache_repo(tmp_path, "org/Two", snap="head", refs=("main",), modified=1.0).revisions[0])
    vbox.repos += [named, two]
    checkpoint = model_library.Checkpoint("org/Named", named.revisions[0].snapshot_path, "/x", 1)
    library.append_peers(vbox.cfg, [library.vllm_peer(checkpoint, name="Named", port=8090, binary=str(exe))],
                         adopt=True, say=lambda _line: None)
    asked = scripted(monkeypatch, "set-up")

    sync(vbox)

    assert [q.question for q in asked.asked] == ["New model Two — set it up?"]
    assert "Named" not in vbox.printed.getvalue()
    assert peer(vbox, "Two")["lifecycle"]["model"] == "org/Two" and peer(vbox, "Named")["lifecycle"]["port"] == 8090
    assert len([p for p in _peers(vbox.cfg) if p["name"].startswith("Two")]) == 1


def test_a_repo_id_with_a_control_character_prints_escaped(vbox, monkeypatch, tmp_path):
    vllm_on_path(vbox)
    vbox.repos.append(cache_repo(tmp_path, "org/x\x1bq"))
    scripted(monkeypatch, "not-now")

    sync(vbox)

    printed = vbox.printed.getvalue()
    assert "← org/x\\x1bq" in printed and "\x1b" not in printed


def test_a_first_run_with_only_a_checkpoint_serves_it_on_the_primary_port(box, monkeypatch, tmp_path):
    for gone in (box.swift, box.qwen, box.proj):
        gone.unlink()
    exe = vllm_on_path(box)
    repos = [cache_repo(tmp_path, "org/Qwen3-8B")]
    monkeypatch.setattr(model_library, "_cache_repos", lambda: tuple(repos))
    asked = scripted(monkeypatch, "set-up")

    result = _init(box.cfg)

    _exited(result, 0)
    [question] = asked.asked
    assert question.question == "Set up Qwen3-8B?"
    config = yaml.safe_load((box.cfg / "config.yaml").read_text(encoding="utf-8"))
    assert {k: config["server"][k] for k in ("runtime", "launch", "binary", "model", "port")} == {
        "runtime": "vllm", "launch": "binary", "binary": str(exe), "model": "org/Qwen3-8B", "port": 8081}
    assert config["extra_endpoints"] == [] and config["provider"]["default_model"] == "org/Qwen3-8B"
    adopted = {e["name"]: e["shown"] for e in trust.recorded_machine(box.cfg)["entries"]}
    assert adopted["server.binary"] == str(exe) and adopted["server.model"] == "org/Qwen3-8B"
    assert "Later: just run localharness start — it brings back your last choice." in result.output
    assert "/model" not in result.output.split("Later:")[1]


def test_a_gguf_serves_first_and_a_checkpoint_after_it_is_a_peer(box, monkeypatch, tmp_path):
    exe = vllm_on_path(box)
    monkeypatch.setattr(model_library, "_cache_repos", lambda: (cache_repo(tmp_path, "org/Qwen3-8B"),))

    _exited(_init(box.cfg), 0)

    config = yaml.safe_load((box.cfg / "config.yaml").read_text(encoding="utf-8"))
    assert config["server"]["model"] == str(box.qwen)
    [vllm] = [p for p in _peers(box.cfg) if p["lifecycle"]["runtime"] == "vllm"]
    assert vllm["lifecycle"]["binary"] == str(exe) and vllm["name"] == "Qwen3-8B"
    ports = [config["server"]["port"], *(p["lifecycle"]["port"] for p in _peers(box.cfg))]
    assert len(set(ports)) == 3


def test_without_a_person_a_first_run_writes_the_checkpoint_unconfirmed_and_a_later_run_nothing(box, monkeypatch, tmp_path):
    for gone in (box.swift, box.qwen, box.proj):
        gone.unlink()
    exe = vllm_on_path(box)
    repos = [cache_repo(tmp_path, "org/Qwen3-8B")]
    monkeypatch.setattr(model_library, "_cache_repos", lambda: tuple(repos))
    box.cfg.mkdir(exist_ok=True)

    first = library.sync(box.cfg, scope="all", person=False, first_run=True)

    assert first.records.server.runtime == "vllm" and first.records.server.binary == str(exe)
    assert first.records.confirmed is False
    assert library.sync(box.cfg, scope="new", person=False) == library.Sync()


def test_counts_and_sync_agree_a_hand_added_peer_is_set_up_and_a_removed_one_is_new_again(vbox, monkeypatch, tmp_path):
    exe = vllm_on_path(vbox)
    repo = cache_repo(tmp_path, "org/Qwen3-8B")
    vbox.repos.append(repo)
    scripted(monkeypatch, "not-now")
    sync(vbox)
    snapshot = os.path.realpath(repo.revisions[0].snapshot_path)
    entries = json.loads(model_library.library_path(vbox.cfg).read_text(encoding="utf-8"))["entries"]
    assert entries[snapshot]["repo"] == "org/Qwen3-8B" and entries[snapshot]["verdict"] == "model"
    assert library.counts(vbox.cfg) == library.Counts(set_up=2, found=1, declined=0)

    checkpoint = model_library.Checkpoint("org/Qwen3-8B", repo.revisions[0].snapshot_path, snapshot, GB16)
    library.append_peers(vbox.cfg, [library.vllm_peer(checkpoint, name="Qwen3-8B", port=8090, binary=str(exe))],
                         adopt=True, say=lambda _line: None)

    assert library.counts(vbox.cfg) == library.Counts(set_up=3, found=0, declined=0)
    asked = scripted(monkeypatch)
    sync(vbox)
    assert asked.asked == []

    overlay = vbox.cfg / "overrides.yaml"
    data = yaml.safe_load(overlay.read_text(encoding="utf-8"))
    data["extra_endpoints"] = [p for p in data["extra_endpoints"] if p["name"] != "Qwen3-8B"]
    overlay.write_text(yaml.safe_dump(data), encoding="utf-8")
    asked = scripted(monkeypatch, "not-now")
    sync(vbox)
    assert [q.question for q in asked.asked] == ["New model Qwen3-8B — set it up?"]


def test_a_checkpoint_the_missing_binary_cannot_serve_is_not_counted_as_found(vbox, monkeypatch, tmp_path):
    exe = vllm_on_path(vbox)
    vbox.repos.append(cache_repo(tmp_path, "org/Qwen3-8B"))
    scripted(monkeypatch, "not-now")
    sync(vbox)
    assert library.counts(vbox.cfg).found == 1

    exe.unlink()

    assert library.counts(vbox.cfg) == library.Counts(set_up=2, found=0, declined=0)


def _fifo(tmp_path):
    repo = cache_repo(tmp_path, "org/Fifo", config=None)
    os.mkfifo(repo.revisions[0].snapshot_path / "config.json")
    return repo


SKIPPED = {
    "no config.json": lambda t: cache_repo(t, "org/Bare", config=None),
    "an encoder": lambda t: cache_repo(t, "org/Bert", {"architectures": ["BertModel"]}),
    "over the cap": lambda t: cache_repo(t, "org/Big", raw=b'{"architectures":["XForCausalLM"]}'
                                         + b" " * (tuning.MODEL_CONFIG_CAP + 1)),
    "a fifo": _fifo,
    "nested": lambda t: cache_repo(t, "org/Deep", raw=b"[" * 2000 + b"]" * 2000),
    "malformed": lambda t: cache_repo(t, "org/Bad", raw=b'{"architectures": ["XForCausalLM"'),
    "architectures not a list": lambda t: cache_repo(t, "org/Str", {"architectures": "XForCausalLM"}),
    "a dataset": lambda t: cache_repo(t, "org/Data", repo_type="dataset"),
    "no bytes": lambda t: cache_repo(t, "org/Empty", size=0),
}


@pytest.mark.parametrize("kind", SKIPPED)
def test_a_repo_that_is_not_a_readable_text_generation_checkpoint_is_skipped(vbox, monkeypatch, tmp_path, kind):
    vllm_on_path(vbox)
    vbox.repos.append(SKIPPED[kind](tmp_path))
    asked = scripted(monkeypatch)

    assert model_library.checkpoints() == ()
    assert sync(vbox) == library.Sync() and asked.asked == []


def test_the_snapshot_listed_is_the_main_revision_else_the_most_recent(tmp_path, monkeypatch):
    pinned = cache_repo(tmp_path, "org/Pinned", snap="old", refs=("main",), modified=1.0)
    pinned.revisions.append(cache_repo(tmp_path, "org/Pinned", snap="new", refs=(), modified=9.0).revisions[0])
    loose = cache_repo(tmp_path, "org/Loose", snap="old", refs=(), modified=1.0)
    loose.revisions.append(cache_repo(tmp_path, "org/Loose", snap="new", refs=(), modified=9.0).revisions[0])
    monkeypatch.setattr(model_library, "_cache_repos", lambda: (pinned, loose))

    assert [(c.repo_id, c.snapshot.name) for c in model_library.checkpoints()] == [
        ("org/Loose", "new"), ("org/Pinned", "old")]
