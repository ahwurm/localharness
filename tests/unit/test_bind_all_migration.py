"""An upgrade never takes a launched model server off the network someone uses.

0.16 launched a vLLM server on every interface. This release launches on 127.0.0.1 unless
`server.bind_all: true`, so the migration that already folds new deny defaults into config.yaml
also writes `bind_all: true` into each launched vLLM section that predates the setting — the
`server` section and each `extra_endpoints[i].lifecycle` — with the same backup and one receipt
line. It never moves the deny-defaults revision stamp (a bump would re-add deny patterns a user
deleted), never touches a section that already says `bind_all` either way, and never adds
`require_api_key` (off by default, absent stays absent).

Accepted (orchestrator ruling R19): a `server:` section written by hand after the upgrade without
`bind_all` looks exactly like a 0.16 one and is migrated the same way; the receipt and doctor's row
say how to keep the server on this machine. `init` writes `bind_all: false` explicitly, so a new
install is never migrated.
"""
from __future__ import annotations

import copy

import yaml
from typer.testing import CliRunner

from localharness.cli.app import app
from localharness.config import migrate
from localharness.config.defaults import CURRENT_DEFAULTS_REVISION
from localharness.config.models import HarnessConfig, ManagedServerConfig, ProviderConfig
from localharness.config.redact import reveal
from tests.unit.test_start_cmd import _capture_start_console, _stub_start_boundaries

runner = CliRunner()
VLLM = {"runtime": "vllm", "launch": "binary", "binary": "/usr/bin/vllm", "model": "m", "port": 8081}
PROVIDER = {"provider_type": "vllm", "base_url": "http://127.0.0.1:9/v1", "default_model": "m",
            "api_key": "none"}
KEPT = "kept the model server reachable from other machines (server.bind_all: true)"
DRY_RUN_LINE = "server.bind_all: true (keeps the model server reachable from other machines)"


def _stamped(**sections) -> dict:
    return {"version": "1", "provider": dict(PROVIDER),
            "org": {"permissions": {"defaults_revision": CURRENT_DEFAULTS_REVISION,
                                    "deny_patterns": ["write(/etc/*)"]}},
            **sections}


def _peer(lifecycle: dict) -> dict:
    return {"name": "peer-1", "base_url": "http://127.0.0.1:8001/v1", "provider_type": "vllm",
            "gpu": True, "lifecycle": lifecycle}


# ------------------------------------------------------------------------------- plan()


def test_a_launched_vllm_written_before_the_setting_keeps_its_reach():
    for server in (dict(VLLM), {k: v for k, v in VLLM.items() if k != "runtime"}):  # runtime defaults to vllm
        data = _stamped(server=server)
        before = copy.deepcopy(data)
        plan = migrate.plan(data)
        assert plan is not None and plan.bind_all_paths == ("server",)
        assert plan.updated["server"] == {**server, "bind_all": True}
        assert plan.added == [] and plan.from_revision == plan.to_revision == CURRENT_DEFAULTS_REVISION
        assert plan.config_unchanged is False
        assert plan.updated["org"] == before["org"]  # the stamp and the deny list are not rewritten
        assert "require_api_key" not in plan.updated["server"]
        assert data == before  # the input is never mutated


def test_a_stamped_config_without_a_deny_list_keeps_its_shipped_defaults():
    """A plan that only keeps a server's reach must not write `deny_patterns: []` (which would drop
    every shipped deny default) into a stamped config that never listed them."""
    data = {"version": "1", "provider": dict(PROVIDER),
            "org": {"permissions": {"defaults_revision": CURRENT_DEFAULTS_REVISION}}, "server": dict(VLLM)}
    plan = migrate.plan(data)
    assert plan is not None and "deny_patterns" not in plan.updated["org"]["permissions"]
    migrated = HarnessConfig.model_validate(plan.updated)
    assert migrated.org.permissions.deny_patterns == HarnessConfig.model_validate(data).org.permissions.deny_patterns


def test_each_launched_peer_keeps_its_reach_too():
    data = _stamped(extra_endpoints=[_peer(dict(VLLM)),
                                     {"name": "attach", "base_url": "http://127.0.0.1:11434/v1"},
                                     _peer({**VLLM, "bind_all": False})])
    plan = migrate.plan(data)
    assert plan.bind_all_paths == ("extra_endpoints[0].lifecycle",)
    assert plan.updated["extra_endpoints"][0]["lifecycle"]["bind_all"] is True
    assert plan.updated["extra_endpoints"][1:] == data["extra_endpoints"][1:]
    assert "bind_all" not in data["extra_endpoints"][0]["lifecycle"]


def test_what_was_loopback_already_or_says_bind_all_is_left_alone():
    llamacpp = {"runtime": "llamacpp", "binary": "/x/llama-server", "model": "/x/m.gguf"}
    ollama = {"runtime": "ollama", "model": "qwen2.5:0.5b"}
    for server in (llamacpp, ollama, {**VLLM, "bind_all": True}, {**VLLM, "bind_all": False}, None):
        assert migrate.plan(_stamped(server=server)) is None, server
    assert migrate.plan(_stamped()) is None


def test_below_the_revision_both_halves_land_in_one_plan_and_the_revision_is_the_shipped_one():
    data = _stamped(server=dict(VLLM))
    data["org"]["permissions"]["defaults_revision"] = 0
    plan = migrate.plan(data)
    assert plan.added and plan.bind_all_paths == ("server",)
    assert plan.to_revision == CURRENT_DEFAULTS_REVISION
    assert plan.updated["org"]["permissions"]["defaults_revision"] == CURRENT_DEFAULTS_REVISION
    assert plan.updated["server"]["bind_all"] is True


def test_a_new_install_is_never_migrated():
    """`init` writes every server key, `bind_all: false` included — so a fresh config plans nothing."""
    harness = HarnessConfig(version="1", provider=ProviderConfig(**PROVIDER),
                            server=ManagedServerConfig(launch="docker", docker_image="img", model="m"))
    data = yaml.safe_load(yaml.safe_dump(reveal(harness.model_dump(mode="python"))))
    data["org"]["permissions"]["defaults_revision"] = CURRENT_DEFAULTS_REVISION
    assert data["server"]["bind_all"] is False
    assert migrate.plan(data) is None


# ------------------------------------------------------------------------------- the surfaces


async def test_the_first_start_after_the_upgrade_says_so_once_with_a_backup(tmp_path, monkeypatch):
    from localharness.cli.start_cmd import _start_async

    _stub_start_boundaries(tmp_path, monkeypatch)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(_stamped(server=dict(VLLM))), encoding="utf-8")
    printed = _capture_start_console(monkeypatch)

    await _start_async(None, False, False, str(tmp_path))

    receipts = [line for line in printed if KEPT in line]
    assert len(receipts) == 1 and "localharness doctor" in receipts[0], printed
    assert "Config updated" in receipts[0]
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1
    assert yaml.safe_load((tmp_path / "config.yaml").read_text())["server"]["bind_all"] is True

    printed.clear()
    await _start_async(None, False, False, str(tmp_path))
    assert not [line for line in printed if "server.bind_all" in line or "Config updated" in line], printed
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1


def test_config_migrate_lists_it_on_a_dry_run_and_applies_it(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(_stamped(server=dict(VLLM))), encoding="utf-8")
    before = config.read_bytes()

    dry = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path), "--dry-run"])
    assert dry.exit_code == 0 and DRY_RUN_LINE in " ".join(dry.output.split()), dry.output
    assert config.read_bytes() == before and not list(tmp_path.glob("config.yaml.bak-*"))

    real = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert real.exit_code == 0 and DRY_RUN_LINE in " ".join(real.output.split()), real.output
    assert yaml.safe_load(config.read_text())["server"]["bind_all"] is True
    assert len(list(tmp_path.glob("config.yaml.bak-*"))) == 1
    again = runner.invoke(app, ["config", "migrate", "--config-dir", str(tmp_path)])
    assert again.exit_code == 0 and "bind_all" not in again.output
