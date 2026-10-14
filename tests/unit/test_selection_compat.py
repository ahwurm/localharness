"""The release is additive (spec 20 §3 G6, COMPAT-01, D-30): a 0.18 install upgrades without a change it
did not ask for.

A 0.18 config (no `speculative`, none of the new stores), a 0.18 overlay `active_endpoint` (no effort), the
0.17 and 0.18 pidfile shapes and the plugin API version all still read and resolve. Offline: configs come
from `ConfigLoader` over a tmp dir, the pidfile is read through a faked `stat` seam; nothing is started and
no socket or service manager is touched. Test fakes take `**kw`.
"""
from __future__ import annotations

import pytest
import yaml

from localharness.cli.model_ops import RUNG_ACTIVE, RUNG_DEFAULT, Serving, resolve_serving
from localharness.cli.workspace import MachineTrust
from localharness.config.loader import ConfigLoader
from localharness.plugins.api import PLUGIN_API_VERSION
from localharness.provider import procfs
from tests.unit.procfs_support import stat_bytes

PRIMARY = "qwen-27b-swift-1.5"
PEER_URL = "http://127.0.0.1:8081/v1"


def _config(home, *, with_peer: bool) -> None:
    """A 0.18-shaped config.yaml: a llama.cpp `server:` with no `speculative`, and no store anywhere."""
    data = {
        "version": "1",
        "provider": {
            "provider_type": "llamacpp", "base_url": "http://127.0.0.1:8080/v1",
            "default_model": PRIMARY, "available_models": [PRIMARY],
        },
        "org": {"default_model": PRIMARY, "audit_log_path": str(home / "audit.jsonl")},
        "server": {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/m/a.gguf",
                   "served_name": PRIMARY, "port": 8080},
    }
    if with_peer:
        data["extra_endpoints"] = [{
            "name": "flash-next", "base_url": PEER_URL, "provider_type": "llamacpp",
            "gpu": True, "lifecycle": {"gpu": True, "runtime": "llamacpp", "model": "/m/f.gguf", "port": 8081},
        }]
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture
def home(tmp_path):
    h = tmp_path / ".localharness"
    h.mkdir()
    return h


def test_a_018_config_resolves_to_the_primary_on_server(home):
    _config(home, with_peer=False)
    harness = ConfigLoader(config_dir=home).load_harness()

    got = resolve_serving(harness, MachineTrust(), None, config_dir=home)

    assert isinstance(got, Serving)
    assert (got.record.prefix, got.rung, got.explicit, got.gone) == ("server", RUNG_DEFAULT, False, ())


def test_a_018_overlay_active_endpoint_without_effort_loads_and_resolves(home):
    _config(home, with_peer=True)
    (home / "overrides.yaml").write_text(yaml.safe_dump({"active_endpoint": {
        "name": "flash-next", "base_url": PEER_URL, "provider_type": "llamacpp", "model": "flash-next"}}),
        encoding="utf-8")
    harness = ConfigLoader(config_dir=home).load_harness()

    got = resolve_serving(harness, MachineTrust(), None, config_dir=home)

    assert isinstance(got, Serving)
    assert (got.record.name, got.record.base_url, got.rung) == ("flash-next", PEER_URL, RUNG_ACTIVE)


@pytest.mark.parametrize("text, start", [("4242 777", 777), ("4242", None)])
def test_a_017_or_018_pidfile_reads_with_no_port(tmp_path, text, start):
    directory = tmp_path / procfs.SERVER_DIR_NAME
    directory.mkdir(mode=0o700)
    pidfile = directory / procfs.PIDFILE_NAME
    pidfile.write_text(text, encoding="utf-8")
    pidfile.chmod(0o600)

    launch = procfs.read_launch(tmp_path, stat_reader=lambda pid, **kw: stat_bytes(pid, 777))

    assert launch is not None
    assert (launch.pid, launch.start, launch.port) == (4242, start, None)


def test_the_plugin_api_version_is_unchanged():
    assert PLUGIN_API_VERSION == "1"
