"""`/effort`'s durable half in cli/model_ops.py (EFFORT-05, EFFORT-07; 03-CONTEXT D-05, D-12, D-13, D-14).

`persist_effort` writes ONE dict path through validate-then-atomic_write_overlay, audits it as the
person's channel action, and never lets a peer's key or a fingerprinted key reach the overlay.
`agents_with_effort` is the raw-yaml scan `/effort show` prints.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import SecretStr

from localharness.cli import model_ops
from localharness.config import trust
from localharness.config.overlay import load_overlay
from tests.unit.effort_support import PRIMARY_URL, harness_with

PEER_URL = "http://127.0.0.1:9090/v1"


def _persist(h, level, tmp_path, *, base_url=PRIMARY_URL, channel_id="terminal", **kw):
    return asyncio.run(model_ops.persist_effort(
        h, level, base_url=base_url, channel_id=channel_id, config_dir=tmp_path, **kw))


def _rows(tmp_path):
    log = tmp_path / "audit.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]


def _mutated(tmp_path):
    return [r for r in _rows(tmp_path) if r.get("event_type") == "ComponentMutated"]


def _seed_machine_files(tmp_path):
    """Files the machine trust snapshot reads, so 'byte-identical' compares something."""
    (tmp_path / "agents").mkdir(exist_ok=True)
    (tmp_path / "agents" / "scout.yaml").write_text(yaml.safe_dump({
        "name": "scout",
        "tools": {"mcp_servers": [{"name": "fs", "command": "npx", "args": ["x"],
                                   "env": {"TOKEN": "s3cret"}}]},
    }))
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump({
        "server": {"binary": "/usr/bin/llama-server", "port": 8081},
    }))


# ── tracer: the primary with a managed server ──────────────────────────────────────────────────

def test_primary_with_server_writes_server_effort_and_audits_one_channel_row(tmp_path):
    h = harness_with(server_effort=None)
    assert _persist(h, "low", tmp_path) is None

    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay == {"server": {"effort": "low"}}
    assert h.server.effort == "low"

    rows = _mutated(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert (row["path"], row["before_value"], row["after_value"]) == ("server.effort", None, "low")
    assert (row["layer"], row["actor"], row["actor_detail"]) == ("user", "channel", "terminal")


def test_trailing_slash_on_base_url_still_names_the_primary(tmp_path):
    h = harness_with()
    assert _persist(h, "max", tmp_path, base_url=PRIMARY_URL + "/") is None
    assert h.server.effort == "max"


def test_machine_snapshot_is_byte_identical_and_overlay_keeps_its_other_keys(tmp_path):
    _seed_machine_files(tmp_path)
    before = trust.machine_snapshot(tmp_path)
    assert before, "the seeded files must give the snapshot something to compare"

    assert _persist(harness_with(), "high", tmp_path) is None

    assert json.dumps(trust.machine_snapshot(tmp_path), sort_keys=True) == json.dumps(
        before, sort_keys=True)
    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay["server"] == {"binary": "/usr/bin/llama-server", "port": 8081, "effort": "high"}


# ── the other two dict paths ───────────────────────────────────────────────────────────────────

def test_active_peer_writes_active_endpoint_effort_only(tmp_path):
    h = harness_with(peer=PEER_URL, active=PEER_URL, active_effort="low")
    # the overlay already carries the selection; only its effort may change
    seeded = {"active_endpoint": {"name": "peer", "base_url": PEER_URL,
                                  "provider_type": "llamacpp", "model": "m"}}
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump(seeded))

    assert _persist(h, "max", tmp_path, base_url=PEER_URL, channel_id="mobile") is None

    overlay = load_overlay(tmp_path / "overrides.yaml")
    assert overlay == {"active_endpoint": {**seeded["active_endpoint"], "effort": "max"}}
    assert h.active_endpoint.effort == "max"
    assert h.server.effort is None, "the primary's record is not the serving one"
    (row,) = _mutated(tmp_path)
    assert (row["path"], row["before_value"], row["after_value"]) == (
        "active_endpoint.effort", "low", "max")
    assert row["actor_detail"] == "mobile"


def test_hand_run_primary_writes_org_default_effort(tmp_path):
    h = harness_with(hand_run=True)
    assert _persist(h, "off", tmp_path) is None

    assert load_overlay(tmp_path / "overrides.yaml") == {"org": {"default_effort": "off"}}
    assert h.org.default_effort == "off"
    (row,) = _mutated(tmp_path)
    assert row["path"] == "org.default_effort"


def test_peer_without_an_active_record_writes_nothing_and_says_session_only(tmp_path):
    h = harness_with(peer=PEER_URL)
    note = _persist(h, "low", tmp_path, base_url=PEER_URL)

    assert note == model_ops.EFFORT_SESSION_ONLY_NOTE
    assert "session only" in note
    assert not (tmp_path / "overrides.yaml").exists()
    assert _rows(tmp_path) == []


# ── prohibitions: no peer key, no extra_endpoints ──────────────────────────────────────────────

@pytest.mark.parametrize("which", ["server", "active", "org"])
def test_no_key_or_peer_record_reaches_the_overlay(tmp_path, which):
    if which == "server":
        h, url = harness_with(peer=PEER_URL), PRIMARY_URL
    elif which == "active":
        h, url = harness_with(peer=PEER_URL, active=PEER_URL), PEER_URL
    else:
        h, url = harness_with(peer=PEER_URL, hand_run=True), PRIMARY_URL
    h.extra_endpoints[0].api_key = SecretStr("peer-secret-key")
    h.extra_endpoints[0].extra_headers = {"X-Auth": SecretStr("peer-header-secret")}
    if h.active_endpoint is not None:
        h.active_endpoint.api_key = SecretStr("active-secret-key")

    assert _persist(h, "high", tmp_path, base_url=url) is None

    text = (tmp_path / "overrides.yaml").read_text()
    for forbidden in ("api_key", "extra_headers", "extra_endpoints",
                      "peer-secret-key", "peer-header-secret", "active-secret-key"):
        assert forbidden not in text, forbidden


# ── audit and validation ───────────────────────────────────────────────────────────────────────

def test_unwritable_audit_log_warns_after_the_overlay_was_written(tmp_path):
    h = harness_with()
    # The bus swallows an append failure but cannot make a log's parent under a regular file, which
    # is the real failure that reaches persist_effort's own try/except.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    h.org.audit_log_path = str(blocker / "audit.jsonl")

    warning = _persist(h, "low", tmp_path)

    assert warning is not None and warning.startswith("persisted, but the audit log")
    assert load_overlay(tmp_path / "overrides.yaml") == {"server": {"effort": "low"}}
    assert h.server.effort == "low"


def test_audit_record_follows_the_work_not_the_overlay(tmp_path):
    cfg, ws = tmp_path / "global", tmp_path / "ws"
    cfg.mkdir()
    ws.mkdir()
    assert _persist(harness_with(), "low", cfg, audit_base_dir=ws) is None
    assert (cfg / "overrides.yaml").exists() and not (cfg / "audit.jsonl").exists()
    assert (ws / "audit.jsonl").exists() and not (ws / "overrides.yaml").exists()


@pytest.mark.parametrize("bad", ["", "medium", "inherit", "HIGH"])
def test_a_level_outside_levels_raises_before_any_write(tmp_path, bad):
    with pytest.raises(ValueError):
        _persist(harness_with(), bad, tmp_path)
    assert not (tmp_path / "overrides.yaml").exists()
    assert _rows(tmp_path) == []


# ── idempotency and the empty case (EFFORT-05) ─────────────────────────────────────────────────

def test_persisting_the_same_level_twice_writes_the_same_overlay_and_a_no_change_row(tmp_path):
    h = harness_with()
    assert _persist(h, "high", tmp_path) is None
    first = (tmp_path / "overrides.yaml").read_text()
    assert _persist(h, "high", tmp_path) is None

    assert (tmp_path / "overrides.yaml").read_text() == first
    rows = _mutated(tmp_path)
    assert len(rows) == 2
    # persist_default_model emits unconditionally; this mirrors it: the repeat row is a no-change row
    assert rows[1]["before_value"] == rows[1]["after_value"] == "high"


def test_empty_overlay_and_no_active_endpoint_still_persist_the_server_level(tmp_path):
    (tmp_path / "overrides.yaml").write_text("")
    h = harness_with()
    assert h.active_endpoint is None

    assert _persist(h, "low", tmp_path) is None
    assert load_overlay(tmp_path / "overrides.yaml") == {"server": {"effort": "low"}}


# ── D-13: a switch drops the previous endpoint's effort and key ────────────────────────────────

def test_persist_active_endpoint_drops_api_key_and_effort_from_the_previous_record(tmp_path):
    h = harness_with(peer=PEER_URL, active=PEER_URL, active_effort="max")
    (tmp_path / "overrides.yaml").write_text(yaml.safe_dump({"active_endpoint": {
        "name": "old", "base_url": "http://old/v1", "provider_type": "llamacpp",
        "model": "old-m", "api_key": "x", "effort": "max"}}))
    endpoint = SimpleNamespace(name="peer", base_url=PEER_URL, provider_type="llamacpp",
                               api_key=SecretStr("k"))

    assert asyncio.run(model_ops.persist_active_endpoint(
        h, endpoint, "new-m", config_dir=tmp_path)) is None

    record = load_overlay(tmp_path / "overrides.yaml")["active_endpoint"]
    assert record == {"name": "peer", "base_url": PEER_URL, "provider_type": "llamacpp",
                      "model": "new-m"}
    assert h.active_endpoint.effort is None
    assert h.active_endpoint.model == "new-m"


# ── agents_with_effort: the raw-yaml list `/effort show` prints (D-05) ─────────────────────────

def _agent(base: Path, stem: str, body: str) -> None:
    (base / "agents").mkdir(parents=True, exist_ok=True)
    (base / "agents" / f"{stem}.yaml").write_text(body, encoding="utf-8")


def test_agents_with_effort_lists_levels_from_both_layers_sorted_by_name(tmp_path):
    glob, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _agent(glob, "a", "name: a\neffort: low\n")
    _agent(glob, "b", "name: b\neffort: inherit\n")
    _agent(glob, "c", "name: c\n")
    _agent(ws, "d", "name: d\neffort: max\n")

    assert model_ops.agents_with_effort(glob, local_config_dir=ws) == [("a", "low"), ("d", "max")]


def test_agents_with_effort_without_a_workspace_reads_only_the_global_layer(tmp_path):
    glob, ws = tmp_path / "global", tmp_path / "proj" / ".localharness"
    _agent(glob, "a", "name: a\neffort: low\n")
    _agent(ws, "d", "name: d\neffort: max\n")

    assert model_ops.agents_with_effort(glob) == [("a", "low")]


def test_agents_with_effort_skips_a_yaml_that_does_not_parse(tmp_path):
    glob = tmp_path / "global"
    _agent(glob, "broken", "name: [unclosed\neffort: high\n")
    _agent(glob, "ok", "name: ok\neffort: 'off'\n")
    _agent(glob, "scalar", "just a string\n")

    assert model_ops.agents_with_effort(glob) == [("ok", "off")]


def test_agents_with_effort_with_no_agents_dir_or_no_config_dir_is_empty(tmp_path):
    assert model_ops.agents_with_effort(tmp_path / "nothing") == []
    assert model_ops.agents_with_effort(None) == []


def test_agents_with_effort_lists_only_values_the_loader_would_accept(tmp_path):
    glob = tmp_path / "global"
    _agent(glob, "typo", "name: typo\neffort: turbo\n")
    _agent(glob, "unquoted", "name: unquoted\neffort: off\n")  # YAML 1.1 reads this as False
    _agent(glob, "fine", "name: fine\neffort: high\n")

    assert model_ops.agents_with_effort(glob) == [("fine", "high")]
