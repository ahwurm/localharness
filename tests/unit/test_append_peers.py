"""The one peer writer: `library.append_peers` appends to overrides.yaml under the overlay lock, validated,
audited after the lock is released, and adopted from exactly what was written."""
from __future__ import annotations

import fcntl
import json
import os
import stat
from pathlib import Path

import pytest
import yaml

from localharness.cli import library
from localharness.config import trust
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
