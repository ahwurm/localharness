"""The artifact contract's core half (PAPI-10, decision 13): "plugins choose whether they have
artifacts, never where they live".

Core computes the root (`<state dir>/artifacts/<plugin>/`), mints every id in one shape, and allows
three image mimes. ArtifactRef is a typed reference, NOT an event: the web protocol snapshot must not
move (tests/unit/channels/test_mobile_protocol.py runs in the same verify command).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

import localharness.core.artifacts as artifacts
from localharness.channels.mobile.protocol import event_schemas
from localharness.core.artifacts import ARTIFACTS_DIR_NAME, artifact_root, mint_artifact_id, write_artifact
from localharness.core.events import ARTIFACT_ID_RE, ARTIFACT_MIMES, EVENT_TYPE_MAP, ArtifactRef, BaseEvent

PNG = b"\x89PNG\r\n\x1a\n" + bytes(16)


def _ref(**overrides) -> ArtifactRef:
    fields = {"plugin": "example", "kind": "image", "id": mint_artifact_id(), "mime": "image/png"}
    return ArtifactRef(**{**fields, **overrides})


def test_ref_constructs_and_is_frozen():
    ref = _ref()
    assert (ref.plugin, ref.kind, ref.mime) == ("example", "image", "image/png")
    with pytest.raises(ValidationError):
        ref.id = mint_artifact_id()


@pytest.mark.parametrize("bad", [
    "../x", "art-1", "art-20260929-120000-ABCDEF", "{minted}.png", "{minted}\n", "*",
    "art-20260929-120000-abcdef/../x", "art-٢٠٢٦٠٩٢٩-120000-abcdef",
])
def test_ref_refuses_an_id_core_did_not_mint(bad):
    """Last case: Arabic-Indic digits, which an unflagged Python `\\d` accepts."""
    with pytest.raises(ValidationError, match="core-minted"):
        _ref(id=bad.replace("{minted}", mint_artifact_id()))


@pytest.mark.parametrize("field,value", [
    ("kind", "audio"), ("mime", "image/gif"), ("mime", "text/html"),
    ("plugin", "../x"), ("plugin", "Example"), ("plugin", ""), ("plugin", 'x"><img'),
])
def test_ref_refuses_off_contract_fields(field, value):
    """`plugin` reaches the phone's artifact URL just like `id`, so it is held to the plugin-name rule."""
    with pytest.raises(ValidationError):
        _ref(**{field: value})


def test_the_allowlist_is_one_set():
    """The model's mime Literal and the table core writes suffixes from (and the route's 415 reads)
    are the same set."""
    literal = set(get_args(ArtifactRef.model_fields["mime"].annotation))
    assert literal == set(ARTIFACT_MIMES) == {"image/png", "image/jpeg", "image/webp"}


def test_minted_ids_fit_the_one_shape():
    assert all(ARTIFACT_ID_RE.fullmatch(mint_artifact_id()) for _ in range(1000))
    assert mint_artifact_id() != mint_artifact_id()
    got = mint_artifact_id(datetime(2026, 9, 29, 12, 0, 5, tzinfo=timezone.utc))
    assert got.startswith("art-20260929-120005-") and ARTIFACT_ID_RE.fullmatch(got)


def test_artifact_root_is_core_computed():
    assert ARTIFACTS_DIR_NAME == "artifacts"
    assert artifact_root(Path("/s"), "example") == Path("/s/artifacts/example")
    for bad in ("../etc", "a/b", ""):
        with pytest.raises(ValueError):
            artifact_root(Path("/s"), bad)


@pytest.mark.parametrize("mime,suffix", [("image/png", ".png"), ("image/jpeg", ".jpg"), ("image/webp", ".webp")])
def test_write_artifact_writes_exactly_one_named_file(tmp_path, mime, suffix):
    root = tmp_path / "state" / "artifacts" / "example"
    assert not root.exists()
    ref = write_artifact(root, "example", PNG, mime)
    assert [p.name for p in root.iterdir()] == [f"{ref.id}{suffix}"]
    assert (root / f"{ref.id}{suffix}").read_bytes() == PNG
    assert (ref.plugin, ref.kind, ref.mime) == ("example", "image", mime)


@pytest.mark.parametrize("plugin,mime,match", [
    ("example", "image/gif", "image/png"),  # the error names the allowlist
    ("../x", "image/png", "plugin name"),
])
def test_write_artifact_refuses_and_writes_nothing(tmp_path, plugin, mime, match):
    root = tmp_path / "artifacts" / "example"
    with pytest.raises(ValueError, match=match):
        write_artifact(root, plugin, b"GIF89a", mime)
    assert list(tmp_path.rglob("*")) == []


def test_write_artifact_never_overwrites(tmp_path, monkeypatch):
    """A repeated id (24 random bits within one second) fails loudly rather than replacing a file
    the phone may already hold under an immutable-cache header."""
    monkeypatch.setattr(artifacts, "mint_artifact_id", lambda now=None: "art-20260929-120000-abcdef")
    write_artifact(tmp_path, "example", b"first", "image/png")
    with pytest.raises(FileExistsError):
        write_artifact(tmp_path, "example", b"second", "image/png")
    assert (tmp_path / "art-20260929-120000-abcdef.png").read_bytes() == b"first"


def test_artifact_ref_is_not_an_event():
    assert "ArtifactRef" not in EVENT_TYPE_MAP
    assert not issubclass(ArtifactRef, BaseEvent)
    # Since protocol v4 exactly one event carries it: Observation.artifact (45-02). No other embeds it.
    schemas = event_schemas()
    assert [name for name, schema in schemas.items() if "ArtifactRef" in json.dumps(schema)] == ["Observation"]
    assert "ArtifactRef" in json.dumps(schemas["Observation"]["properties"]["artifact"])
