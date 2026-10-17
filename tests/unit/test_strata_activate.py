"""A Strata record launches through its row on a fake engine tree; Popen and HTTP are faked and nothing runs."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from localharness.provider import server, strata
from tests.unit.launch_support import popen  # noqa: F401
from tests.unit.strata_fixture import flash_next, strata_record, strata_tree

BASE = "http://127.0.0.1:8083/v1"
SERVED = "swift-1.5-iq3_xxs"


class World:
    def __init__(self, tmp_path: Path, monkeypatch, started: list, *, config: dict | None = None):
        shard, projector = flash_next(tmp_path)
        self.shard, self.projector = shard, projector
        self.engine = strata_tree(tmp_path, config=config, model=shard)
        self.config_dir = tmp_path / "cfg"
        self.config_dir.mkdir()
        self.spec = strata_record(self.engine, shard, projector)
        self.health = {"loaded": True, "service": "strata", "max_context": 65536}
        self.models = {"data": [{"id": SERVED, "meta": {"n_ctx": 65536}}]}
        self.requests: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            self.requests.append(request.url.path)
            body = {"/health": self.health, "/v1/models": self.models}.get(request.url.path)
            return httpx.Response(200 if body is not None else 404, json=body)

        monkeypatch.setattr(server, "_TRANSPORT", httpx.MockTransport(answer))
        monkeypatch.setattr(server, "server_pid", lambda config_dir: 4242 if started else None)

    async def activate(self, **kw):
        return await strata.StrataStrategy().activate(self.spec, self.config_dir, BASE, **kw)

    @property
    def copy(self) -> dict:
        return json.loads(strata.copy_path(self.config_dir, 8083).read_text("utf-8"))


@pytest.fixture
def world(tmp_path, monkeypatch, popen):  # noqa: F811
    return World(tmp_path, monkeypatch, popen)


async def test_a_strata_record_launches_through_its_row_end_to_end(world, popen, monkeypatch):  # noqa: F811
    monkeypatch.setenv("STRATA_SKIP_SHA256", "1")
    live = await world.activate(autosense=None)
    engine, copy = str(world.engine), str(strata.copy_path(world.config_dir, 8083))
    assert popen[0]["cmd"] == [
        f"{engine}/.venv/bin/python", f"{engine}/serve/server.py", "--engine", "strata", "--config", copy,
        "--port", "8083", "--host", "127.0.0.1"]
    assert popen[0]["cwd"] == engine
    assert "STRATA_SKIP_SHA256" not in popen[0]["env"] and live.dropped_env == ("STRATA_SKIP_SHA256",)
    assert world.copy["model_name"] == SERVED
    assert world.copy["args"][world.copy["args"].index("--native") + 1] == str(world.shard)
    assert live.served_models == [SERVED] and live.handle == 4242 and "/health" in world.requests
