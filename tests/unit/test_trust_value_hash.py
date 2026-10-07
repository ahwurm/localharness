"""§7.5: value-hashing of env/header values in executables_snapshot and machine_snapshot.

A value-only change (swapping a TOKEN) now trips the fingerprint. The raw value is
never stored — the no-plaintext property is preserved.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from localharness.config import trust
from localharness.config.paths import WORKSPACE_DIR_NAME


def _ws(tmp_path: Path, name: str = "proj") -> Path:
    ws = tmp_path / name / WORKSPACE_DIR_NAME
    (ws / "agents").mkdir(parents=True)
    return ws


def _agent(ws: Path, file: str, *servers: dict) -> Path:
    path = ws / "agents" / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"name": Path(file).stem, "role": "R",
                        "tools": {"mcp_servers": list(servers)}}),
        encoding="utf-8",
    )
    return path


def _root(ws: Path) -> Path:
    return ws.resolve().parent


# --------------------------------------------------------------------------- _value_digests


def test_value_digests_returns_name_to_sha256_hex():
    d = trust._value_digests({"TOKEN": "secret123", "EMPTY": ""})
    assert d == {
        "TOKEN": hashlib.sha256(b"secret123").hexdigest(),
        "EMPTY": hashlib.sha256(b"").hexdigest(),
    }


def test_value_digests_empty_and_non_dict():
    assert trust._value_digests({}) == {}
    assert trust._value_digests(None) == {}
    assert trust._value_digests("not a dict") == {}


def test_value_digests_is_deterministic():
    a = trust._value_digests({"K": "V"})
    b = trust._value_digests({"K": "V"})
    assert a == b


def test_value_digests_differs_on_value_change():
    a = trust._value_digests({"K": "V1"})
    b = trust._value_digests({"K": "V2"})
    assert a != b


# --------------------------------------------------------------------------- snapshot


def test_snapshot_env_is_digest_dict_not_name_list(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", {"name": "s", "transport": "stdio", "command": "node",
                          "env": {"TOKEN": "abc"}})
    snap = trust.executables_snapshot(ws)
    entry = snap[0]
    assert isinstance(entry["env"], dict)
    assert entry["env"] == {"TOKEN": hashlib.sha256(b"abc").hexdigest()}
    # The raw value is never in the snapshot
    blob = str(snap)
    assert "abc" not in blob


def test_snapshot_header_is_digest_dict(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "b.yaml", {"name": "w", "transport": "streamable_http",
                          "url": "https://h.example/mcp",
                          "headers": {"Authorization": "Bearer tok"}})
    snap = trust.executables_snapshot(ws)
    entry = snap[0]
    assert entry["headers"] == {"Authorization": hashlib.sha256(b"Bearer tok").hexdigest()}
    assert "Bearer tok" not in str(snap)


# --------------------------------------------------------------------------- fingerprint


def test_fingerprint_moves_with_env_value_change(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", {"name": "s", "transport": "stdio", "command": "node",
                          "env": {"TOKEN": "old"}})
    before = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "a.yaml", {"name": "s", "transport": "stdio", "command": "node",
                          "env": {"TOKEN": "new"}})
    after = trust.fingerprint(trust.executables_snapshot(ws))
    assert before != after, "a value-only env change must trip the fingerprint"


def test_fingerprint_moves_with_header_value_change(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "b.yaml", {"name": "w", "transport": "streamable_http",
                          "url": "https://h.example/mcp",
                          "headers": {"Authorization": "Bearer old"}})
    before = trust.fingerprint(trust.executables_snapshot(ws))
    _agent(ws, "b.yaml", {"name": "w", "transport": "streamable_http",
                          "url": "https://h.example/mcp",
                          "headers": {"Authorization": "Bearer new"}})
    after = trust.fingerprint(trust.executables_snapshot(ws))
    assert before != after, "a value-only header change must trip the fingerprint"


def test_fingerprint_stable_for_same_values(tmp_path):
    ws = _ws(tmp_path)
    _agent(ws, "a.yaml", {"name": "s", "transport": "stdio", "command": "node",
                          "env": {"TOKEN": "same"}})
    f1 = trust.fingerprint(trust.executables_snapshot(ws))
    # Re-read (same file, same values)
    f2 = trust.fingerprint(trust.executables_snapshot(ws))
    assert f1 == f2


# --------------------------------------------------------------------------- machine_snapshot


def test_machine_snapshot_hashes_env_values(tmp_path, monkeypatch):
    """machine_snapshot (global path) also uses _value_digests for env/headers."""
    gdir = tmp_path / "global"
    (gdir / "agents").mkdir(parents=True)
    (gdir / "agents" / "bot.yaml").write_text(
        yaml.safe_dump({"name": "bot", "role": "R",
                        "tools": {"mcp_servers": [
                            {"name": "s", "transport": "stdio", "command": "node",
                             "env": {"TOKEN": "machine_secret"}}]}}),
        encoding="utf-8",
    )
    snap = trust.machine_snapshot(gdir)
    mcp_entries = [e for e in snap if e.get("kind") == "mcp_server"]
    assert len(mcp_entries) == 1
    assert mcp_entries[0]["env"] == {"TOKEN": hashlib.sha256(b"machine_secret").hexdigest()}
    assert "machine_secret" not in str(snap)
