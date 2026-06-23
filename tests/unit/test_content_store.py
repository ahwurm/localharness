"""P1 — the unified per-agent content-addressable store (ContentStore).

Locks the keystone invariants the rest of the milestone rests on, all deterministic (no model):
- content-hash handles (same body -> same handle, across instances/agents)
- back-compat: EvictionStore IS ContentStore; the 1-arg put / get path is byte-identical
- sticky, monotonic origin taint (web/untrusted never relaunders; derived stays tainted)
- per-agent pg-N aliases (each store's own first fetch is pg-1) + web LRU bound
- the GRANT read-through: a child reads ONLY granted parent handles, never ambient
"""
from __future__ import annotations

from localharness.agent.context import (
    ContentStore,
    EvictionStore,
    _content_handle,
    _evict_id,
)


# --- content-hash handles + EvictionStore back-compat -------------------------

def test_handle_is_deterministic_content_hash():
    body = "deterministic body content"
    assert _content_handle(body) == _content_handle(body)
    assert _content_handle(body) != _content_handle(body + "!")
    # cross-instance / cross-agent: same body -> same handle (no per-store salt)
    assert ContentStore().put(body) == ContentStore().put(body)


def test_eviction_store_is_content_store_alias():
    # The old name resolves to the new class; the old import surface is preserved.
    assert EvictionStore is ContentStore
    assert _evict_id is _content_handle


def test_legacy_put_get_roundtrip_is_byte_identical():
    s = EvictionStore()
    big = "X" * 20_000
    rid = s.put(big)                 # 1-arg form (origin defaults trusted)
    assert rid == _content_handle(big)
    assert s.get(rid) == big
    assert s.get("deadbeef") is None  # unknown handle -> None (tool maps this to not_found)


# --- sticky, monotonic origin taint -------------------------------------------

def test_origin_defaults_trusted():
    s = ContentStore()
    h = s.put("plain local body")
    assert s.origin(h) == "trusted"


def test_untrusted_is_sticky_and_monotonic():
    s = ContentStore()
    body = "page body"
    h = s.put(body, origin="untrusted")
    assert s.origin(h) == "untrusted"
    # a later TRUSTED put of the SAME bytes must NOT relaunder it
    assert s.put(body, origin="trusted") == h
    assert s.origin(h) == "untrusted"


def test_derived_from_propagates_taint():
    s = ContentStore()
    src = s.put("untrusted source", origin="untrusted")
    deriv = s.put("a slice derived from the source", origin="trusted", derived_from=src)
    assert s.origin(deriv) == "untrusted"
    clean = s.put("independent trusted body", origin="trusted")
    assert s.origin(clean) == "trusted"


# --- per-agent pg-N aliases + web LRU -----------------------------------------

def test_put_web_is_untrusted_and_pg_aliased():
    s = ContentStore()
    alias = s.put_web("a fetched web page")
    assert alias == "pg-1"
    assert s.get("pg-1") == "a fetched web page"
    assert s.origin("pg-1") == "untrusted"


def test_pg_alias_is_per_store_deterministic():
    # Two independent agents each get pg-1 for their OWN first fetch — no global counter bleed.
    a, b = ContentStore(), ContentStore()
    assert a.put_web("A's page") == "pg-1"
    assert b.put_web("B's page") == "pg-1"
    assert a.get("pg-1") == "A's page"
    assert b.get("pg-1") == "B's page"


def test_reset_restarts_pg_sequence():
    s = ContentStore()
    s.put_web("first")
    assert s.put_web("second") == "pg-2"
    s.reset()
    assert s.get("pg-1") is None
    assert s.put_web("after reset") == "pg-1"


def test_web_bodies_are_lru_bounded_trusted_durable():
    s = ContentStore(max_web=2)
    a = s.put_web("web one")
    s.put_web("web two")
    s.put_web("web three")          # evicts the oldest web body (pg-1)
    assert s.get(a) is None          # aged out -> re-fetch lever
    assert s.get("pg-3") == "web three"
    # a durable (trusted) body is never dropped by the web LRU
    h = s.put("durable local result")
    s.put_web("web four")
    assert s.get(h) == "durable local result"


def test_stub_meta_reports_size_and_origin():
    s = ContentStore()
    h = s.put("x" * 1234, origin="trusted")
    assert s.stub_meta(h) == (1234, "trusted")
    s.put_web("y" * 50)
    assert s.stub_meta("pg-1") == (50, "untrusted")
    assert s.stub_meta("nope") is None


# --- the GRANT: read-through allow-set (the per-delegation capability) ---------

def test_grant_reads_only_granted_parent_handles():
    parent = ContentStore()
    granted = parent.put("granted body")
    secret = parent.put("ungranted secret body")

    child = ContentStore(parent=parent, granted=frozenset({granted}))
    assert child.get(granted) == "granted body"     # the capability
    assert child.get(secret) is None                # ambient read is impossible
    assert child.origin(granted) == "trusted"       # taint resolves through the grant


def test_leaf_with_no_grant_sees_nothing_of_parent():
    parent = ContentStore()
    h = parent.put("parent body")
    leaf = ContentStore(parent=None, granted=None)  # how a leaf child is constructed
    assert leaf.get(h) is None


def test_grant_preserves_parent_taint():
    parent = ContentStore()
    tainted = parent.put("web-derived body", origin="untrusted")
    child = ContentStore(parent=parent, granted=frozenset({tainted}))
    assert child.get(tainted) == "web-derived body"
    assert child.origin(tainted) == "untrusted"     # untrusted never launders across the grant
