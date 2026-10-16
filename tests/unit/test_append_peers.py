"""The one peer writer: `library.append_peers` appends to overrides.yaml under the overlay lock, validated,
audited after the lock is released, and adopted from exactly what was written."""
from __future__ import annotations

import ast
import fcntl
import json
import os
import stat
import threading
from pathlib import Path

import pytest
import yaml

from localharness.cli import library
from localharness.config import trust
from localharness.config.redact import reveal
from localharness.config.models import EndpointRef, ManagedServerConfig
from tests.unit.procfs_support import fake_proc  # noqa: F401  (fixture)

CONFIG = {"version": "1", "provider": {"provider_type": "llamacpp", "base_url": "http://127.0.0.1:8081/v1",
                                       "default_model": "qwen", "available_models": ["qwen"]},
          "org": {"audit_log_path": "audit.jsonl"}}


@pytest.fixture
def g() -> Path:
    return Path(os.environ["LOCALHARNESS_HOME"])


@pytest.fixture(autouse=True)
def _nothing_listens(fake_proc):  # noqa: F811
    return fake_proc


def model(g: Path, name: str) -> Path:
    path = g / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return path


def peer(g: Path, name: str, port: int, file: str | None = None) -> EndpointRef:
    return EndpointRef(
        name=name, base_url=f"http://127.0.0.1:{port}/v1", provider_type="llamacpp", gpu=True,
        lifecycle=ManagedServerConfig(runtime="llamacpp", model=str(model(g, file or f"{name}.gguf")),
                                      served_name=name, port=port, gpu=True))


def seed(g: Path, **extra) -> Path:
    server = {"runtime": "llamacpp", "model": str(model(g, "primary.gguf")), "served_name": "qwen", "port": 8081}
    path = g / "config.yaml"
    path.write_text(yaml.safe_dump({**CONFIG, "server": server, "extra_endpoints": [], **extra}), encoding="utf-8")
    return path


def overrides(g: Path) -> dict:
    return yaml.safe_load((g / "overrides.yaml").read_text(encoding="utf-8"))


def audited(g: Path) -> list[str]:
    path = g / "audit.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [json.loads(line)["path"] for line in lines if line.strip()]


def record(g: Path) -> None:
    trust.record_machine(g, sorted(trust.machine_snapshot(g), key=trust._entry_key),
                         kinds=frozenset({trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND}))


def launch_files(g: Path) -> dict[str, str]:
    rec = trust.recorded_machine(g)
    return {e["name"]: e["file"] for e in rec["entries"] if e["kind"] in (trust.LAUNCH_KIND, trust.LAUNCH_MODEL_KIND)}


def test_one_peer_lands_in_overrides_locked_audited_and_adopted(g):
    config = seed(g)
    record(g)
    before = config.read_bytes()
    one = peer(g, "alpha", 8082)

    written = library.append_peers(g, [one], adopt=True)

    assert [p["name"] for p in written] == ["alpha"] and overrides(g)["extra_endpoints"] == written
    assert config.read_bytes() == before and "server" not in overrides(g)
    lock = g / "overrides.yaml.lock"
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600
    files = launch_files(g)
    assert files["extra_endpoints.alpha.lifecycle.model"] == "overrides.yaml"
    assert files["server.model"] == "config.yaml"
    paths = audited(g)
    assert "extra_endpoints.alpha.lifecycle.model" in paths and "extra_endpoints.alpha.lifecycle.runtime" in paths


def test_the_lock_is_released_before_the_audit_publishes(g, monkeypatch):
    from localharness.core.bus import EventBus

    seed(g)
    free: list[bool] = []
    publish = EventBus.publish

    async def spy(self, event, *a, **k):
        fd = os.open(g / "overrides.yaml.lock", os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free.append(True)
        except BlockingIOError:
            free.append(False)
        finally:
            os.close(fd)
        return await publish(self, event, *a, **k)

    monkeypatch.setattr(EventBus, "publish", spy)

    assert library.append_peers(g, [peer(g, "alpha", 8082)], adopt=False)

    assert free and all(free)


def test_without_adopt_nothing_is_adopted(g):
    seed(g)
    record(g)
    before = trust.recorded_machine(g)

    assert library.append_peers(g, [peer(g, "alpha", 8082)], adopt=False)

    assert trust.recorded_machine(g) == before


def dicts(*peers: EndpointRef) -> list[dict]:
    return [reveal(p.model_dump(mode="python", exclude_none=True)) for p in peers]


def covered(g: Path) -> set[str]:
    return {e["name"] for e in trust.recorded_machine(g)["entries"] if e["file"] == "overrides.yaml"}


def test_a_taken_name_gets_the_next_free_suffix(g):
    seed(g)
    (g / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": dicts(peer(g, "qwen-2", 8082))}), encoding="utf-8")

    written = library.append_peers(g, [peer(g, "qwen", 8083)], adopt=False)

    assert [p["name"] for p in written] == ["qwen-3"] and written[0]["lifecycle"]["served_name"] == "qwen-3"


def test_a_recorded_or_listening_port_moves_the_peer_to_the_next_free_one(g, fake_proc):  # noqa: F811
    seed(g)
    (g / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": dicts(peer(g, "other", 8082))}), encoding="utf-8")
    fake_proc.listen(8083, 1, os.getuid())

    written = library.append_peers(g, [peer(g, "alpha", 8082)], adopt=False)

    assert written[0]["base_url"] == "http://127.0.0.1:8084/v1" and written[0]["lifecycle"]["port"] == 8084


def test_a_model_already_recorded_is_not_appended_even_through_a_symlink(g):
    seed(g)
    link = g / "models" / "link.gguf"
    link.symlink_to(g / "models" / "primary.gguf")
    one = peer(g, "alpha", 8082)
    one.lifecycle.model = str(link)

    assert library.append_peers(g, [one], adopt=False) is None
    assert not (g / "overrides.yaml").exists()


def test_the_same_peer_twice_is_appended_once(g):
    seed(g)
    one = peer(g, "alpha", 8082)

    assert library.append_peers(g, [one], adopt=False)
    assert library.append_peers(g, [one], adopt=False) is None

    assert [p["name"] for p in overrides(g)["extra_endpoints"]] == ["alpha"]


def test_the_server_block_is_never_touched(g):
    seed(g)
    server = {"port": 8085, "gpu": False}
    (g / "overrides.yaml").write_text(yaml.safe_dump({"server": server}), encoding="utf-8")

    assert library.append_peers(g, [peer(g, "alpha", 8082)], adopt=False)

    assert overrides(g)["server"] == server


def copied_setup(g: Path, *, with_record: bool) -> tuple[Path, list[EndpointRef]]:
    p1, p2 = peer(g, "p1", 8082), peer(g, "p2", 8083)
    config = seed(g, extra_endpoints=dicts(p1, p2))
    if with_record:
        record(g)
    edited = yaml.safe_load(config.read_text(encoding="utf-8"))
    edited["extra_endpoints"][1]["lifecycle"]["model"] = str(model(g, "p2-edited.gguf"))
    config.write_text(yaml.safe_dump(edited), encoding="utf-8")
    return config, [p1, p2]


def test_the_first_append_copies_a_0_18_list_and_readopts_only_covered_twins(g, capsys):
    config, _ = copied_setup(g, with_record=True)
    before = config.read_bytes()

    written = library.append_peers(g, [peer(g, "p3", 8084)], adopt=True)

    assert [p["name"] for p in written] == ["p3"]
    assert [p["name"] for p in overrides(g)["extra_endpoints"]] == ["p1", "p2", "p3"]
    assert config.read_bytes() == before
    held = covered(g)
    assert "extra_endpoints.p1.lifecycle.model" in held and "extra_endpoints.p1.lifecycle.port" in held
    assert "extra_endpoints.p2.lifecycle.model" not in held
    assert "extra_endpoints.p3.lifecycle.model" in held
    out = capsys.readouterr().out
    assert out.count(library.workspace.PEER_LIST_COPIED) == 1
    assert library.workspace.PEER_LIST_COPIED + " — 2 copied, 1 still confirmed" in out


def test_with_no_machine_record_no_copied_peer_is_adopted(g, capsys):
    copied_setup(g, with_record=False)

    library.append_peers(g, [peer(g, "p3", 8084)], adopt=True)

    held = covered(g)
    assert held and all(name.startswith("extra_endpoints.p3.") for name in held)
    assert library.workspace.PEER_LIST_COPIED + " — 2 copied, 0 still confirmed" in capsys.readouterr().out


def test_a_peer_no_copies_the_list_through_the_one_copier(g, monkeypatch):
    from localharness.cli import workspace

    config = seed(g, extra_endpoints=dicts(peer(g, "p1", 8082)))
    peers = yaml.safe_load(config.read_text(encoding="utf-8"))["extra_endpoints"]
    peers[0]["lifecycle"]["bind_all"] = True
    config.write_text(yaml.safe_dump({**yaml.safe_load(config.read_text(encoding="utf-8")), "extra_endpoints": peers}),
                      encoding="utf-8")
    calls: list[int] = []
    real = workspace.copy_peer_list
    monkeypatch.setattr(workspace, "copy_peer_list", lambda *a: calls.append(1) or real(*a))
    entry = {"file": "config.yaml", "kind": trust.LAUNCH_KIND, "name": "extra_endpoints.p1.lifecycle.bind_all",
             "shown": "llamacpp:true:open"}

    assert workspace._remember_posture_no(g, [entry], []) == [entry]

    assert calls == [1] and overrides(g)["extra_endpoints"][0]["lifecycle"]["bind_all"] is False


def test_only_copy_peer_list_deep_copies_a_peer_list_and_both_writers_call_it():
    root = Path(library.__file__).parent
    copying: set[str] = set()
    calling: set[tuple[str, str]] = set()
    for source in root.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                name = ast.unparse(call.func)
                if name.endswith("deepcopy") and any("extra_endpoints" in ast.unparse(a) for a in call.args):
                    copying.add(fn.name)
                if name.endswith("copy_peer_list"):
                    calling.add((source.stem, fn.name))
    assert copying == {"copy_peer_list"}
    assert {("workspace", "_remember_posture_no"), ("library", "mutate")} <= calling


def test_a_lock_held_elsewhere_writes_nothing_and_says_so(g, monkeypatch, capsys):
    seed(g)
    (g / "overrides.yaml").write_text(yaml.safe_dump({"extra_endpoints": []}), encoding="utf-8")
    before = (g / "overrides.yaml").read_bytes()
    monkeypatch.setattr("localharness.config.overlay.OVERLAY_LOCK_BUDGET_SECONDS", 0.1)
    fd = os.open(g / "overrides.yaml.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        assert library.append_peers(g, [peer(g, "alpha", 8082)], adopt=True) is None
    finally:
        os.close(fd)

    assert (g / "overrides.yaml").read_bytes() == before
    assert capsys.readouterr().out.strip() == library.OVERLAY_BUSY_LINE


def test_two_writers_racing_both_land(g):
    seed(g)
    barrier = threading.Barrier(2)
    results: list = []

    def append(one: EndpointRef) -> None:
        barrier.wait()
        results.append(library.append_peers(g, [one], adopt=False))

    threads = [threading.Thread(target=append, args=(peer(g, name, port),))
               for name, port in (("alpha", 8082), ("beta", 8083))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results)
    assert sorted(p["name"] for p in overrides(g)["extra_endpoints"]) == ["alpha", "beta"]
