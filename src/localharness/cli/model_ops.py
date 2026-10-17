"""Shared `/model` operations used by BOTH the REPL `/model` command and the
`localharness model` CLI.

The full REPL swap — live client mutation, TokenCounter rebind, managed-server restart,
channel I/O — stays in ``repl.py``; it is agent-loop-coupled and cannot run without a live
session. What is extracted here is the agent-loop-INDEPENDENT decision + persistence layer:
which models exist, and how a chosen default is durably + atomically recorded (mirroring the
`components set` overlay path, issue #22) and which loaded agents a persisted switch will NOT
reach because their per-agent yaml pins a concrete model.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from localharness.config.models import LEVELS, HarnessConfig, coerce_yaml_effort
from localharness.config.overlay import (
    _resolve_user_overlay_path,
    atomic_write_overlay,
    deep_merge,
    load_overlay,
)
from localharness.config import redact, trust
from localharness.config.paths import global_config_dir, resolve_runtime_path
from localharness.config.plugin_sections import core_harness_view
from localharness.config.redact import secret_values, validation_text
from localharness.config.trust import LAUNCH_KIND, LAUNCH_MODEL_KIND
from localharness.core.bus import EventBus
from localharness.core.events import ComponentMutated
from localharness.provider.runtimes import RUNTIMES, row_of
from localharness.provider.units import (
    PRIMARY_TARGET, TARGET_NAME, in_flight_target, read_target, remove_target,
)
from localharness.registry import set_value_in_dict

if TYPE_CHECKING:
    from localharness.cli.workspace import MachineTrust

_AGENT_KEY = "agent"

EFFORT_SESSION_ONLY_NOTE = (
    "effort applies to this session only (no active_endpoint record to persist it to)"
)


def _validated(harness: Any, harness_overlay: dict) -> None:
    """Check the config the next `start` reads, `harness` with `harness_overlay` merged in. A refusal
    is a ValueError whose text is each error's location and message, scrubbed (R16): pydantic's own
    text carries the merged config, a key stored raw in overrides.yaml included, and callers print
    this text as the reason."""
    merged = deep_merge(harness.model_dump(mode="python"), harness_overlay)
    try:
        # plugin sections are validated by their plugins (ENAB-01); unknown keys are rejected at load
        HarnessConfig.model_validate(core_harness_view(merged))
    except ValidationError as exc:
        raise ValueError(validation_text(exc, secret_values(HarnessConfig, merged))) from None


class MalformedModelListError(Exception):
    """The endpoint was REACHED but its ``/models`` reply wasn't an OpenAI-compatible model list
    (HTML error page, wrong API, unexpected JSON shape). Kept distinct from unreachable (#38) so
    callers stop rendering "is it running?" — the server IS running, base_url points at the
    wrong thing."""


def list_live_models(base_url: str, timeout: float = 3.0, *,
                     api_key: str | None = None) -> tuple[list[str], bool]:
    """Probe the OpenAI-compatible ``/models`` endpoint. Returns ``(model_ids, reachable)``.

    ``api_key`` is the key of a launched server set to require one (callers pass it only through
    ``provider.server.required_key_kwargs``); without it the probe sends no Authorization header.

    Three outcomes, three signals (#38 — previously one blanket ``except`` collapsed the last two
    into a bogus ``reachable=False``, so a live-but-wrong endpoint read as "is it running?"):
      - transport error (connection refused/DNS/timeout) -> ``([], False)`` = unreachable;
      - reached, valid OpenAI ``data`` shape -> ``(ids, True)`` (empty list stays reachable — a
        legit empty result, the #16 lesson);
      - reached but the body isn't an OpenAI model list -> raise ``MalformedModelListError``.

    Parses only the OpenAI ``data`` shape; Ollama/LM-Studio-aware listing (audit gap #2) is out
    of scope here — same shape the REPL /model list uses (which now delegates here).
    """
    import httpx

    from localharness.provider.server import auth_headers
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout, headers=auth_headers(api_key))
    except (httpx.RequestError, httpx.InvalidURL):
        # #1: httpx.InvalidURL is a SIBLING of RequestError (both subclass Exception directly), so a
        # typo'd base_url (non-numeric port, stray ':') would otherwise propagate and crash the whole
        # /model call. A malformed URL is, for our purposes, just an endpoint we can't reach.
        return [], False  # transport failure / malformed URL → genuinely unreachable
    try:
        return [m["id"] for m in resp.json()["data"]], True
    except (ValueError, KeyError, TypeError) as exc:
        # Reached, but not an OpenAI /models list (bad JSON, missing `data`, wrong item shape).
        raise MalformedModelListError(
            f"{base_url} responded, but not with an OpenAI-compatible model list ({exc})"
        ) from exc


def _merged_available_models(harness: Any, existing_overlay: dict, model: str) -> list[str]:
    """Union every known model name. The overlay's ``deep_merge`` REPLACES lists wholesale
    (config/overlay.py), so writing a bare ``[model]`` would DROP the rest — read-merge the
    in-memory harness list, the overlay's current list, and the new model, order-preserving."""
    out: list[str] = []
    overlay_avail = (existing_overlay.get("provider") or {}).get("available_models") or []
    for src in (list(harness.provider.available_models), overlay_avail, [model]):
        for m in src:
            if m and m not in out:
                out.append(m)
    return out


async def persist_default_model(
    harness: Any,
    model: str,
    *,
    actor: str = "cli",
    config_dir: Any = None,
    audit_base_dir: Any = None,
    local_operator: bool = False,
    write_server_model: bool = True,
) -> str | None:
    """Persist ``model`` as the org + provider default via the atomic USER OVERLAY — the same
    crash-safe, audited path ``components set`` uses (issue #22), replacing the prior full,
    non-atomic ``config.yaml`` rewrite.

    Flow: load overlay -> set ``provider.default_model`` + ``org.default_model`` +
    union(``available_models``) -> validate the merged HarnessConfig (raises on an invalid
    result) -> ``atomic_write_overlay``
    -> emit one ``ComponentMutated`` per path. Only ``provider.*`` / ``org.*`` are ever touched
    in the overlay, so an ``agent:`` slice (the kill-lever layer) is preserved untouched.
    Also mutates the in-memory ``harness`` so the live session's view stays consistent.

    TWO independent directory inputs, because the two writes answer to different layers (v0.13):

    - ``config_dir`` (#35): the GLOBAL overlay target — machine-wide truth. The same resolved dir
      the harness was loaded from, so a persist tracks ``--config-dir`` instead of leaking to
      ``~/.localharness``. Callers thread their loader's ``_config_dir``. There is ONE physical GPU
      daemon, so this value must never be a workspace layer (amendment #2, C2 invariant).
    - ``audit_base_dir`` (MEMS-04): where the audit RECORD lands — it follows the WORK, so in a
      workspace session callers pass the workspace dir. Defaults to ``config_dir``, which makes the
      call byte-identical for every caller that does not pass it.

    Name rationale — the name deliberately does NOT end in ``config_dir``. The structural guard
    ``tests/unit/test_provider_carveout_workspace.py::test_model_cmd_passes_the_global_config_dir_to_persist``
    asserts the literal ``"config_dir=loader._local_dir"`` is absent from ``model_cmd.py``, and it
    is a plain substring scan that does not respect Python token boundaries — so ANY parameter
    whose name ends in ``config_dir``, handed the workspace directly, would tail-match and trip a
    guard about a DIFFERENT parameter. ``audit_base_dir`` is also the more accurate name: the value
    is the BASE ``resolve_runtime_path`` roots a bare name under, not the audit file itself.

    ``local_operator`` (D-19, TRUST-09): True only when a person at this machine made the switch.
    The origin signal is ``ChannelAdapter.local_operator`` (the REPL) or the CLI's tty-without-
    ``LOCALHARNESS_COMMIT`` rule — never ``actor`` and never message content. When True and
    ``server.model`` was actually written, the one ``launch_model`` entry is re-recorded in the
    machine record, built from the value written here and never re-snapshotted, so a hand edit
    made since the last start still asks. False (the default) records nothing: the next terminal
    start asks about ``server.model (overrides.yaml)``.

    ``write_server_model`` (WR-54): False when ``model`` is the primary's served-name alias (a vLLM
    ``--served-model-name``), which is a default the record answers to and never the model it loads
    (D-70): ``server.model`` is then left as written.

    Returns None on clean success, or a warning string (#37) when ONLY a post-write step failed
    (the audit emit, or the ``local_operator`` re-record in the machine record) — the durable
    overlay write already succeeded, so callers surface this as a secondary note and still report
    the switch as done (never as a persist failure).
    """
    before_provider = harness.provider.default_model
    before_org = harness.org.default_model

    # Amendment #2 (v0.13): the model/endpoint overlay is MACHINE-WIDE truth — there is one physical
    # GPU daemon, so this write names the GLOBAL layer explicitly and never follows a workspace layer.
    # Byte-identical today (the global seam IS resolve_config_dir); load-bearing from phase 40 on.
    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    existing = load_overlay(overlay_path)
    merged_avail = _merged_available_models(harness, existing, model)

    new_overlay = dict(existing)
    set_value_in_dict(new_overlay, "provider.default_model", model)
    set_value_in_dict(new_overlay, "org.default_model", model)
    set_value_in_dict(new_overlay, "provider.available_models", merged_avail)
    # #34: with a managed server configured, ALSO persist server.model — else cold start rebuilds
    # `vllm serve <srv.model>` from the stale config value and relaunches the old model. Only when
    # a server block exists (never invent one: a bare server:{model} fails validation).
    # For vLLM, Ollama and LM Studio server.model IS the served name the strategies warm-load, so a
    # cold start relaunches the chosen model. For llama.cpp server.model is the GGUF path and the
    # alias is `served_name`, so the alias must never reach `-m`: it is written to neither the
    # overlay (here) nor the in-memory record (below).
    wrote_server_model = (write_server_model and harness.server is not None
                          and not RUNTIMES[harness.server.runtime].gguf)
    if wrote_server_model:
        set_value_in_dict(new_overlay, "server.model", model)

    # Validate the SAME cascade the next `start` sees: current config ⊕ new overlay. Exclude the
    # agent-scope `agent:` section (not a HarnessConfig field — mirrors components_cmd and
    # load_harness's overlay handling). Raises ValueError if the result is invalid.
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})

    atomic_write_overlay(overlay_path, new_overlay)

    # Keep the in-memory harness consistent with what was just persisted.
    harness.provider.default_model = model
    harness.org.default_model = model
    harness.provider.available_models = merged_avail
    if wrote_server_model:
        harness.server.model = model

    # D-19: a person's own switch re-records exactly the one entry it wrote. After the durable write
    # (a failed write adopts nothing) and before the audit emit (an audit failure still leaves the
    # entry adopted). A no-op when this machine has no record yet.
    # The switch is durable by here, so a trust-store write that fails (read-only dir, full disk) is a
    # secondary note like the audit emit's, never a failed persist: the next start simply asks (WR-04).
    adopt_warning: str | None = None
    if local_operator and wrote_server_model:
        try:
            trust.adopt_entries(
                global_config_dir(config_dir),
                [{"file": "overrides.yaml", "kind": trust.LAUNCH_MODEL_KIND,
                  "name": "server.model", "shown": model}],
            )
        except OSError as exc:
            adopt_warning = (f"persisted, but server.model could not be re-recorded in the machine "
                             f"record (the next start asks about it): {exc}")

    # Audit trail (mirrors components_cmd): one ComponentMutated per path written. A bare default
    # 'audit.jsonl' lands under the base dir given here (#35).
    # #37: scope the emit in its OWN try/except — it runs AFTER the durable overlay write, so a
    # failure here (unwritable audit log) must NOT surface as a persist failure. Return a warning
    # the callers show as a secondary note; the switch itself already succeeded.
    try:
        # MEMS-04 (v0.13): the audit log follows the WORK. In a workspace session this is the
        # workspace dir; everywhere else it falls back to `config_dir` so the call is unchanged for
        # every caller that does not pass it. Deliberately a SECOND parameter: `global_config_dir()`
        # above returns whatever it is GIVEN — its "global" guarantee is calling discipline, not a
        # property of the function — so widening the single `config_dir` to "workspace when present"
        # would move the overlay write too and fork the one physical GPU daemon's server.model.
        _audit_base = audit_base_dir if audit_base_dir is not None else config_dir
        audit_path = harness.org.audit_log_path
        bus = EventBus(
            persist_path=resolve_runtime_path(audit_path, _audit_base) if audit_path else None
        )
        for path, before in (
            ("provider.default_model", before_provider),
            ("org.default_model", before_org),
        ):
            await bus.publish(
                ComponentMutated(
                    path=path,
                    before_value=before,
                    after_value=model,
                    layer="user",
                    actor=actor,  # type: ignore[arg-type]
                )
            )
    except Exception as exc:  # noqa: BLE001 — the durable overlay write already succeeded
        audit_warning = f"persisted, but the audit log could not be written: {exc}"
        return f"{adopt_warning}; {audit_warning}" if adopt_warning else audit_warning
    return adopt_warning


_SERVED_NAME_FLAGS = ("-a", "--alias", "--served-model-name")


def _alias_in_args(args: Any) -> str | None:
    """The value of the first of llama.cpp ``-a``/``--alias`` or vLLM ``--served-model-name`` present
    in a launch-arg list, in either spelling (``--alias NAME`` or ``--alias=NAME``), else None (a flag
    with no value after it counts as absent)."""
    args = [a for a in (args or []) if isinstance(a, str)]
    for flag in _SERVED_NAME_FLAGS:
        for i, arg in enumerate(args):
            if arg == flag:
                if i + 1 < len(args):
                    return args[i + 1]
                break
            if flag.startswith("--") and arg.startswith(flag + "="):
                value = arg[len(flag) + 1:]
                if value:
                    return value
                break
    return None


def peer_served_name(ep: Any) -> str:
    """The model id a lifecycle peer serves: its record's ``served_name`` (the ``-a`` alias), else
    the alias in its launch ``extra_args``, else the peer's name. The same rule as the REPL's
    cold-peer label; the client still binds to what the peer reports once it is up."""
    lifecycle = getattr(ep, "lifecycle", None)
    if lifecycle is not None:
        served = getattr(lifecycle, "served_name", None)
        if served:
            return served
        alias = _alias_in_args(getattr(lifecycle, "extra_args", None))
        if alias:
            return alias
    return ep.name


def primary_served_name(server: Any, provider: Any) -> str | None:
    """The model id the primary answers to: ``server.served_name``, else the alias in
    ``server.extra_args``, else ``provider.default_model``. None when there is no managed server
    (an attach-only primary has no served-name identity to match a preset against)."""
    if server is None:
        return None
    served = getattr(server, "served_name", None)
    if served:
        return served
    alias = _alias_in_args(getattr(server, "extra_args", None))
    if alias:
        return alias
    return getattr(provider, "default_model", None)


def _launch_key(harness: Any, base_url: str, spec: Any = None) -> dict[str, str]:
    """What a launch of, or a probe to, the server at ``base_url`` sends as its key:
    ``{"api_key": <raw>}`` only when the server the harness launches there is set to require one
    (``spec``, else the primary's ``server`` or that peer's ``lifecycle``), with the primary
    provider's key or that peer's own; else ``{}`` — the keyless call it always was. A URL that is
    neither the primary nor a configured peer gets no key. A function of the harness, not a
    method: the swap's helpers are also driven with a bare stand-in for the session."""
    from localharness.provider.server import required_key_kwargs
    provider = getattr(harness, "provider", None)
    if spec is not None:
        # A peer that shares the primary's address is told apart by its own launch record: its spec
        # is its key's rule, and the peer's key is the one it sends (never the primary's).
        owner = next((e for e in getattr(harness, "extra_endpoints", None) or []
                      if getattr(e, "lifecycle", None) is spec), None)
        if owner is not None:
            return required_key_kwargs(spec, getattr(owner, "api_key", None))
    if provider is not None and base_url == provider.base_url:
        return required_key_kwargs(getattr(harness, "server", None) if spec is None else spec,
                                   provider.api_key)
    ep = next((e for e in getattr(harness, "extra_endpoints", None) or [] if e.base_url == base_url), None)
    if ep is None:
        return {}
    return required_key_kwargs(getattr(ep, "lifecycle", None) if spec is None else spec,
                               getattr(ep, "api_key", None))


def redact_for_channel(text: str, harness: Any, key_file_key: Any = None) -> str:
    """`text` as any channel (the REPL, Discord, the phone, start's console) may show a launch's tail:
    every secret the config holds, the key a key file supplied (a ``SecretStr`` or a str), ``Bearer``
    tokens, llama-server's ``api_keys: ****wxyz`` line and URL credentials all masked (D-22, SEC-04).
    The one channel-side redaction; ``lifecycle.swap`` takes it, bound, as its ``redact``. Redaction
    never raises into a channel: when the config cannot be read for its secrets, the key, the
    patterns and the URL credentials are still scrubbed."""
    from localharness.provider.server import strip_url_credentials
    raw = key_file_key.get_secret_value() if hasattr(key_file_key, "get_secret_value") else key_file_key
    held = [raw] if isinstance(raw, str) and raw else []
    try:
        held += secret_values(HarnessConfig, harness.model_dump(mode="python"))
    except Exception:  # noqa: BLE001 — a config that cannot be read for secrets: the key and patterns remain
        pass
    return strip_url_credentials(redact.scrub(text, held, patterns=redact.LOGGED_KEY_PATTERNS))


def heavy_for(serving: "Serving", harness: Any) -> Any:
    """The ``lifecycle.Heavy`` a swap launches for the record ``serving`` chose: its spec, address,
    peer name and launch key. ``serving`` comes from ``resolve_serving`` (trust already applied), so
    a record the machine withholds never reaches here. A record with no launch record (attach-only)
    cannot be launched: ValueError."""
    from localharness.provider.lifecycle import Heavy
    record = serving.record
    if record.spec is None:
        raise ValueError(f"{record.name} has no launch record — there is nothing to launch")
    return Heavy(record.spec, record.base_url, record.peer.name if record.peer else None,
                 _launch_key(harness, record.base_url, record.spec))


def incumbent_heavy(active: tuple[Any, str], harness: Any) -> Any:
    """The ``lifecycle.Heavy`` for what runs now, ``active`` being the ``(spec, base_url)`` the session
    tracks. The peer is named only when ``spec`` IS that peer's own lifecycle (identity, not
    equality), so a restore's refusal names ``extra_endpoints[<name>]`` exactly for a peer and
    ``server`` for the primary."""
    from localharness.provider.lifecycle import Heavy
    spec, base_url = active
    peer_ep = next((e for e in getattr(harness, "extra_endpoints", None) or []
                    if e.base_url == base_url), None)
    peer = peer_ep.name if peer_ep is not None and getattr(peer_ep, "lifecycle", None) is spec else None
    return Heavy(spec, base_url, peer, _launch_key(harness, base_url, spec))


def _peer_session_key(ep: Any) -> "tuple[Any, str | None]":
    """The key a cold peer's ``lifecycle.api_key_file`` holds, read once at its launch: ``(SecretStr,
    None)`` on success; ``(None, line)`` when the file fails the byte rule, with the one line to
    show (``extra_endpoints[<name>].lifecycle.api_key_file <path>: <reason> — this session sends no
    key``, or ``— the file is not used; this peer keeps sending the key it is configured with`` when
    the peer is configured with one); ``(None, None)`` when the lifecycle names no file or does not
    require a key. The caller runs it only for a lifecycle the loader kept (a withheld one is already
    None), so the file is opened only once the record covers the entry. The key stays in memory: it is
    never written, printed or logged, and the line carries the reason, never any of the file's content.

    The outcome is also recorded on the lifecycle (``mark_key_file``), so the launch decision keeps a
    refused file off argv and on loopback; a read that succeeds, or names no file, clears it."""
    from pydantic import SecretStr

    from localharness.cli.workspace import visible_text
    from localharness.provider.server import (
        KeyFileError,
        key_destination,
        key_kwargs,
        mark_key_file,
        read_key_file,
    )
    lifecycle = getattr(ep, "lifecycle", None)
    path = getattr(lifecycle, "api_key_file", None)
    if not path or getattr(lifecycle, "require_api_key", False) is not True:
        mark_key_file(lifecycle, False)
        return None, None

    def refused(reason: object) -> "tuple[None, str]":
        mark_key_file(lifecycle, True)
        # the peer's name, its path and a reason come from settings files: shown through visible_text
        head = (f"extra_endpoints[{visible_text(ep.name)}].lifecycle.api_key_file "
                f"{visible_text(path)}: {visible_text(reason)} — ")
        configured = getattr(ep, "configured_api_key", getattr(ep, "api_key", None))
        if key_kwargs(configured):
            return None, head + "the file is not used; this peer keeps sending the key it is configured with"
        return None, head + "this session sends no key"

    # The key goes to the peer's base_url; only a destination that is this machine gets one out of a
    # file (WR-06).
    here, shown = key_destination(getattr(ep, "base_url", ""))
    if not here:
        return refused(f"base_url {shown} is not this machine, so the file is not read")
    try:
        key = SecretStr(read_key_file(path))
    except KeyFileError as exc:
        return refused(exc)
    mark_key_file(lifecycle, False)
    return key, None


def _hold_peer_session_key(ep: Any, peer_key: Any) -> None:
    """Put this launch's key on the peer: the file's key when it was read, else the key the peer is
    CONFIGURED with - never the key an earlier launch's file left behind."""
    setter = getattr(ep, "set_session_key", None)
    if callable(setter):
        setter(peer_key)
    elif peer_key is not None:
        ep.api_key = peer_key


async def persist_active_endpoint(
    harness: Any, endpoint: Any, model: str, *, config_dir: Any = None
) -> str | None:
    """Persist a CROSS-ENDPOINT switch (to a peer, not the primary provider) into the atomic user
    overlay as a top-level ``active_endpoint`` record, so it can be resumed without ever corrupting
    the primary ``provider``/``server`` identity.

    Deliberately NOT ``persist_default_model``: that path rewrites ``provider.default_model``,
    unions ``provider.available_models``, and — with a managed server — sets ``server.model``,
    which would make the next ``start`` build a ``vllm serve <peer-model>`` command and try to
    serve a foreign model on the managed GPU. Here we touch ONLY the additive ``active_endpoint``
    key; ``provider`` and ``server`` stay pristine.

    Same crash-safe contract as ``persist_default_model``: validate the merged HarnessConfig BEFORE
    writing (so a bad record can never brick the next config load), then ``atomic_write_overlay``.
    Also mutates the in-memory ``harness`` so the live session view stays consistent.

    ``start`` reads ``active_endpoint`` back as the ladder's last-success rung (spec 20 §8.1); this
    write is the durable, reversible half. A landing on the peer the record already names keeps its
    ``effort``; any other switch drops it.
    """
    from localharness.config.models import ActiveSelection

    # No `audit_base_dir` sibling here ON PURPOSE (not an oversight): this function builds no
    # EventBus and emits no audit event, so it has no audit path to root anywhere. Adding the
    # parameter for symmetry with `persist_default_model` would be dead API surface.
    # Amendment #2 (v0.13): the model/endpoint overlay is MACHINE-WIDE truth — there is one physical
    # GPU daemon, so this write names the GLOBAL layer explicitly and never follows a workspace layer.
    # Byte-identical today (the global seam IS resolve_config_dir); load-bearing from phase 40 on.
    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    existing = load_overlay(overlay_path)
    new_overlay = dict(existing)
    # No key in this record: the key stays with the endpoint it belongs to, `active_endpoint` is not
    # read back at start, and a SecretStr cannot be written to YAML. A key an earlier version wrote
    # here would now sit beside another endpoint's address, so it goes.
    # Nor an `effort` from ANOTHER endpoint: the level belongs to the endpoint it was chosen on, so a
    # switch drops the previous one and the new endpoint starts from its own record's default (spec
    # §7.2). Landing again on the peer the record already names is not a switch: its level stays
    # (spec 20 §8.4, D-29), or a cold `start` would erase what `/effort` stored.
    prior = new_overlay.get("active_endpoint")
    keep_effort = (
        isinstance(prior, dict)
        and prior.get("name", "") == getattr(endpoint, "name", "")
        and _norm_url(prior.get("base_url")) == _norm_url(endpoint.base_url))
    kept_effort = prior.get("effort") if keep_effort else None
    if isinstance(prior, dict):
        new_overlay["active_endpoint"] = {
            k: v for k, v in prior.items()
            if k != "api_key" and (k != "effort" or keep_effort)}
    set_value_in_dict(new_overlay, "active_endpoint.name", getattr(endpoint, "name", ""))
    set_value_in_dict(new_overlay, "active_endpoint.base_url", endpoint.base_url)
    set_value_in_dict(new_overlay, "active_endpoint.provider_type", endpoint.provider_type)
    set_value_in_dict(new_overlay, "active_endpoint.model", model)

    # Validate the SAME cascade the next `start` sees (current config ⊕ new overlay), excluding the
    # agent-scope slice — mirrors persist_default_model. Raises ValueError on a bad result.
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})

    atomic_write_overlay(overlay_path, new_overlay)

    harness.active_endpoint = ActiveSelection(
        name=getattr(endpoint, "name", ""),
        base_url=endpoint.base_url,
        provider_type=endpoint.provider_type,
        model=model,
        api_key=getattr(endpoint, "api_key", "none"),
        effort=kept_effort,
    )
    return None


def clear_active_endpoint(harness: Any, *, config_dir: Any = None) -> None:
    """Drop the persisted cross-endpoint selection, so the next ``start`` and this process are back
    on the primary (D-58). Touches ONLY the ``active_endpoint`` key of the GLOBAL overlay, the same
    layer ``persist_active_endpoint`` writes (one physical GPU daemon, so never a workspace layer);
    an ``agent:`` slice and every other key stay as they were. Nothing is written when the overlay
    holds no ``active_endpoint`` (the harness is left as it was). Same crash-safe contract: validate the merged config before
    ``atomic_write_overlay``. Synchronous, it awaits nothing."""
    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    existing = load_overlay(overlay_path)
    if "active_endpoint" not in existing:
        return
    new_overlay = {k: v for k, v in existing.items() if k != "active_endpoint"}
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})
    atomic_write_overlay(overlay_path, new_overlay)
    harness.active_endpoint = None


# --------------------------------------------------------------------------- the selection ladder
# One ladder every surface reads (spec 20 §8.1, D-01): an explicit --preset, then the intent the shell
# recorded (`pending_preset`), then the unit's own target (`unit-target`), then the last success
# (`active_endpoint`), then the primary (`server:`). Every rung hands back the RECORD the loaded config
# holds, never a store's own fields: a store contributes a name (or, for `active_endpoint`, a match
# key) and nothing it says can become a URL or an argument (T-13-13).

RUNG_PRESET = "--preset"
RUNG_PENDING = "pending_preset"
RUNG_UNIT = TARGET_NAME  # the store's own name: units.py is the one place that spells it
RUNG_ACTIVE = "active_endpoint"
RUNG_DEFAULT = "server:"

UNKNOWN_PRESET_LINE = "unknown preset '{name}' — presets: {names}"
AMBIGUOUS_PRESET_LINE = "preset '{name}' is ambiguous — it matches {first} and {second}"
GONE_SELECTION_LINE = "{store} names {name}, which is no longer a preset — removed"


@dataclass(frozen=True)
class ServingRecord:
    """One launch record the loaded config holds: the primary (``peer`` None, ``prefix`` "server") or
    a peer (``prefix`` "extra_endpoints.<name>.lifecycle"). ``spec`` is its ``ManagedServerConfig``,
    None for an attach-only record. ``served`` is the model id it answers to."""
    name: str
    served: str
    base_url: str
    provider_type: str
    peer: Any
    spec: Any
    prefix: str


@dataclass(frozen=True)
class Serving:
    """What the ladder chose: the record, the rung that named it, and whether a person asked for it
    (``--preset`` and ``pending_preset`` are asks; the rest are what happened last). ``gone`` lists
    the stored selections that named nothing, for the caller to remove through ``clear_gone``."""
    record: ServingRecord
    rung: str
    explicit: bool
    gone: tuple = ()


@dataclass(frozen=True)
class ServingRefusal:
    """A name the ladder will not serve. ``kind`` is "unknown", "ambiguous" or "withheld"; ``line`` is
    the one line to show; ``exit_code`` is what a command exits with."""
    line: str
    kind: str
    exit_code: int = 2


@dataclass(frozen=True)
class GoneSelection:
    """A stored selection (``store`` "pending_preset" or "unit-target") naming no record."""
    store: str
    name: str
    line: str


def _norm_url(url: Any) -> str:
    return str(url or "").rstrip("/")


def _primary_record(harness: Any) -> ServingRecord:
    provider = harness.provider
    server = getattr(harness, "server", None)
    served = primary_served_name(server, provider) or getattr(provider, "default_model", "") or ""
    return ServingRecord(
        name=served, served=served, base_url=provider.base_url,
        provider_type=provider.provider_type, peer=None, spec=server, prefix="server")


def serving_records(harness: Any) -> tuple[ServingRecord, ...]:
    """Every record the loaded config can serve or attach to: the primary first, then each peer in
    config order. An attach-only record is here (an ``active_endpoint`` can name it) but is not a
    preset; see ``_preset_records``."""
    records = [_primary_record(harness)]
    for ep in getattr(harness, "extra_endpoints", None) or []:
        records.append(ServingRecord(
            name=ep.name, served=peer_served_name(ep), base_url=ep.base_url,
            provider_type=ep.provider_type, peer=ep, spec=getattr(ep, "lifecycle", None),
            prefix=f"extra_endpoints.{ep.name}.lifecycle"))
    return tuple(records)


def record_for_base_url(harness: Any, base_url: str) -> ServingRecord | None:
    """The record whose own ``base_url`` is ``base_url`` (a trailing slash does not matter)."""
    want = _norm_url(base_url)
    return next((r for r in serving_records(harness) if _norm_url(r.base_url) == want), None)


# Launch keys that narrow a record (a withheld one is applied as false/true) rather than drop it.
_NARROWING_KEYS = frozenset({"bind_all", "require_api_key"})


def _withheld_entries(trust: Any, record: ServingRecord) -> list[tuple[str, str]]:
    """The ``(file, name)`` of every withheld launch entry that drops ``record`` (R-13-10)."""
    out: list[tuple[str, str]] = []
    for file, items in sorted((getattr(trust, "withheld", None) or {}).items()):
        for kind, name in sorted(items):
            if kind not in (LAUNCH_KIND, LAUNCH_MODEL_KIND):
                continue
            if name != record.prefix and not name.startswith(record.prefix + "."):
                continue
            if name.rsplit(".", 1)[-1] in _NARROWING_KEYS:
                continue
            out.append((file, name))
    return out


def record_withheld(trust: Any, record: ServingRecord) -> bool:
    """True when start withheld a launch entry of ``record``: the loader dropped its launch spec, so
    it must not be served, and a selection naming it must not be cleared. A withheld ``bind_all`` or
    ``require_api_key`` narrows the record, it does not withhold it."""
    return bool(_withheld_entries(trust, record))


def _withheld_refusal(trust: Any, record: ServingRecord, config_dir: Any) -> ServingRefusal:
    """Spec 16's withheld line: ``trust.line`` when start composed one, else the same format built
    from this record's own withheld names (R-13-12)."""
    line = getattr(trust, "line", "")
    if not line:
        from localharness.cli.workspace import MACHINE_UNASKED, MACHINE_WITHHELD_LINE, visible_text

        entries = _withheld_entries(trust, record)
        names = ", ".join(f"{visible_text(n)} ({visible_text(f)})" for f, n in entries)
        line = MACHINE_WITHHELD_LINE.format(
            n=len(entries), global_dir=global_config_dir(config_dir), names=names, why=MACHINE_UNASKED)
    return ServingRefusal(line, "withheld")


def trust_as_recorded(config_dir: Any) -> "MachineTrust":
    """The machine record read, never made: every launch entry withheld while none exists, else the entries it does not cover."""
    from collections import defaultdict

    from localharness.cli.workspace import MachineTrust, uncovered

    cfg = global_config_dir(config_dir)
    snapshot, recorded = trust.machine_snapshot(cfg), trust.recorded_machine(cfg)
    held = (uncovered(recorded["entries"], snapshot) if recorded is not None
            else [e for e in snapshot if e.get("kind") in (LAUNCH_KIND, LAUNCH_MODEL_KIND)])
    by_file: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for entry in held:
        by_file[entry["file"]].add((entry["kind"], entry["name"]))
    return MachineTrust({file: frozenset(items) for file, items in by_file.items()})


def _preset_records(harness: Any, trust: Any) -> tuple[ServingRecord, ...]:
    """The records a name can select: those with a launch spec, plus a record start withheld (the
    loader kept it without a spec, and it resolves by name so it can be refused as withheld, R-13-09)."""
    return tuple(r for r in serving_records(harness) if r.spec is not None or record_withheld(trust, r))


def _candidates(record: ServingRecord) -> tuple[str, ...]:
    return (record.name,) if record.served == record.name else (record.name, record.served)


def resolve_preset(
    harness: Any, trust: Any, name: str, *, config_dir: Any = None
) -> ServingRecord | ServingRefusal:
    """The preset record ``name`` stands for (D-03, D-04): exact name or served name first, then
    case-folded, then a unique prefix (counted in records, not names). Two records at a tier is
    ``ambiguous`` and names both; none is ``unknown`` and lists the presets. Names resolve FIRST, then
    ``trust`` is checked: a withheld record is refused with spec 16's line (``withheld``)."""
    records = _preset_records(harness, trust)
    folded = name.casefold()
    tiers = (
        lambda c: c == name,
        lambda c: c.casefold() == folded,
        lambda c: bool(folded) and c.casefold().startswith(folded),
    )
    for matches in tiers:
        hits = [r for r in records if any(matches(c) for c in _candidates(r))]
        if len(hits) == 1:
            if record_withheld(trust, hits[0]):
                return _withheld_refusal(trust, hits[0], config_dir)
            return hits[0]
        if hits:
            return ServingRefusal(
                AMBIGUOUS_PRESET_LINE.format(name=name, first=hits[0].name, second=hits[1].name),
                "ambiguous")
    names = ", ".join(sorted({r.name for r in records}))
    return ServingRefusal(UNKNOWN_PRESET_LINE.format(name=name, names=names), "unknown")


def _stored(harness: Any, trust: Any, store: str, name: str, config_dir: Any) -> Any:
    """A stored name resolved: its record, a refusal, or a ``GoneSelection`` when it names nothing.
    When the primary is withheld its own names cannot be read, so a name that matches nothing is
    treated as the primary's and refused as withheld, never gone (R-13-10)."""
    got = resolve_preset(harness, trust, name, config_dir=config_dir)
    if isinstance(got, ServingRefusal) and got.kind == "unknown":
        primary = _primary_record(harness)
        if record_withheld(trust, primary):
            return _withheld_refusal(trust, primary, config_dir)
        return GoneSelection(store, name, GONE_SELECTION_LINE.format(store=store, name=name))
    return got


def _stored_target(harness: Any, trust: Any, name: str, config_dir: Any) -> Any:
    """``_stored`` for a ``unit-target`` name (either rung): the reserved ``server:`` token (what
    ``UnitStrategy`` writes for the primary) is the primary record, refused as withheld when the primary
    is; any other name resolves as a preset name does."""
    if name == PRIMARY_TARGET:
        primary = _primary_record(harness)
        if record_withheld(trust, primary):
            return _withheld_refusal(trust, primary, config_dir)
        return primary
    return _stored(harness, trust, RUNG_UNIT, name, config_dir)


def _active_record(harness: Any) -> ServingRecord | None:
    """The record ``active_endpoint`` names: by ``base_url``, then by a ``model`` exactly one record
    serves. None when it matches nothing (the rung is passed over; the store is left as it is)."""
    active = getattr(harness, "active_endpoint", None)
    if active is None:
        return None
    # A peer may share the primary's address; its own name tells them apart, so the name is asked first.
    by_name = next((r for r in serving_records(harness)
                    if r.peer is not None and active.name and r.name == active.name
                    and _norm_url(r.base_url) == _norm_url(active.base_url)), None)
    if by_name is not None:
        return by_name
    by_url = record_for_base_url(harness, active.base_url)
    if by_url is not None:
        return by_url
    by_model = [r for r in serving_records(harness) if r.served == active.model]
    return by_model[0] if len(by_model) == 1 else None


def resolve_serving(
    harness: Any, trust: MachineTrust, preset: str | None, *, config_dir: Any = None
) -> Serving | ServingRefusal:
    """Walk the ladder: ``--preset`` > ``unit-target`` in flight (its requester alive) > ``pending_preset`` >
    ``unit-target`` > ``active_endpoint`` > ``server:``. Writes nothing; a stored selection that names
    nothing comes back on ``Serving.gone`` for the caller to pass to ``clear_gone`` (R-13-11).

    The in-flight rung (R-14-38, spec 20 §8.1): while ``UnitStrategy.activate`` restarts the harness unit
    it writes ``unit-target`` as ``<name> <pid> <start>``; that name outranks ``pending_preset`` only while
    the requesting process still runs, so the restart a swap or a restore asked for serves what it asked
    for. Once ``activate`` ends, or the requester is gone (no requester survives a boot), the same file is
    the plain ``unit-target`` rung below ``pending_preset``, as the spec orders a boot. The ``server:``
    token names the primary record on both ``unit-target`` rungs."""
    gone: list[GoneSelection] = []
    if preset is not None:
        got = resolve_preset(harness, trust, preset, config_dir=config_dir)
        if isinstance(got, ServingRefusal):
            return got
        return Serving(got, RUNG_PRESET, True)
    target_dir = global_config_dir(config_dir)
    skip_plain = False
    in_flight = in_flight_target(target_dir)
    if in_flight:
        got = _stored_target(harness, trust, in_flight, config_dir)
        if isinstance(got, GoneSelection):
            gone.append(got)
            skip_plain = True  # the same file: reported once, never twice
        elif isinstance(got, ServingRefusal):
            return got
        else:
            return Serving(got, RUNG_UNIT, False, tuple(gone))
    pending = getattr(harness, "pending_preset", None)
    if pending:
        got = _stored(harness, trust, RUNG_PENDING, pending, config_dir)
        if isinstance(got, GoneSelection):
            gone.append(got)
        elif isinstance(got, ServingRefusal):
            return got
        else:
            return Serving(got, RUNG_PENDING, True, tuple(gone))
    target = None if skip_plain else read_target(target_dir)
    if target:
        got = _stored_target(harness, trust, target, config_dir)
        if isinstance(got, GoneSelection):
            gone.append(got)
        elif isinstance(got, ServingRefusal):
            return got
        else:
            return Serving(got, RUNG_UNIT, False, tuple(gone))
    record = _active_record(harness)
    if record is not None:
        if record_withheld(trust, record):
            return _withheld_refusal(trust, record, config_dir)
        return Serving(record, RUNG_ACTIVE, False, tuple(gone))
    return Serving(_primary_record(harness), RUNG_DEFAULT, False, tuple(gone))


def default_serving(harness: Any, gone: tuple = ()) -> Serving:
    """The ladder's last rung as a ``Serving``: the primary ``server:`` record, nothing asked for. For a
    caller whose chosen record turned out not launchable and which falls back to what a start always
    served (R-13-33); a ``Serving`` is built here and in ``resolve_serving`` only (D-11)."""
    return Serving(serving_records(harness)[0], RUNG_DEFAULT, False, gone)


def clear_gone(harness: Any, gone: Any, *, config_dir: Any = None) -> list[str]:
    """Remove each stored selection in ``gone`` (``Serving.gone``) and return their lines. Prints
    nothing; the caller shows the lines."""
    lines: list[str] = []
    for entry in gone:
        if entry.store == RUNG_PENDING:
            clear_pending_preset(harness, config_dir=config_dir)
        elif entry.store == RUNG_UNIT:
            remove_target(global_config_dir(config_dir))
        else:
            continue
        lines.append(entry.line)
    return lines


def write_pending_preset(harness: Any, name: str, *, config_dir: Any = None) -> None:
    """Record ``name`` as the preset the next start serves (D-31): the one writer of
    ``pending_preset``. Touches ONLY that key of the GLOBAL overlay (the machine's own file, never a
    workspace layer); the ``agent:`` slice and every other key stay. The merged config is validated
    BEFORE the atomic write, so a bad name (``ValueError``) writes nothing."""
    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    new_overlay = dict(load_overlay(overlay_path))
    new_overlay["pending_preset"] = name
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})
    atomic_write_overlay(overlay_path, new_overlay)
    harness.pending_preset = name


def clear_pending_preset(harness: Any, *, config_dir: Any = None) -> None:
    """Remove ``pending_preset`` from the GLOBAL overlay (the one remover), writing nothing when the
    overlay holds none; every other key and the ``agent:`` slice stay as they were."""
    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    existing = load_overlay(overlay_path)
    if "pending_preset" not in existing:
        harness.pending_preset = None
        return
    new_overlay = {k: v for k, v in existing.items() if k != "pending_preset"}
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})
    atomic_write_overlay(overlay_path, new_overlay)
    harness.pending_preset = None


# --------------------------------------------------------------------------- the sensor
# `sense_serving` is the one probe `start` and `/model` share (spec 20 §8.2, D-12): which launch record
# answers at its own address right now. It asks loopback addresses only (T-13-29): a record whose
# `base_url` is a remote host is never probed from here, so `start` cannot be turned into a way to
# reach an address a config merely names.

SENSE_TIMEOUT_SECONDS = 2.0
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


@dataclass(frozen=True)
class ServingSense:
    """What the sensor found: ``answering`` the records that answer at their own address (record
    order, each once), ``ids`` the model ids each answering address reported, keyed by the address
    probed and present only for addresses that answered (a reply that was not a model list answers
    with no ids)."""
    answering: tuple[ServingRecord, ...]
    ids: Mapping[str, tuple[str, ...]]


def is_loopback(base_url: Any) -> bool:
    """True when ``base_url`` names ``127.0.0.1``, ``::1`` or ``localhost``. Anything else, an
    unparseable value included, is not loopback."""
    try:
        host = urlsplit(str(base_url)).hostname
    except ValueError:
        return False
    return host is not None and host.lower() in _LOOPBACK_HOSTS


def launchable_loopback_others(harness: Any, trust: Any, selected: ServingRecord) -> tuple[ServingRecord, ...]:
    """The records other than ``selected`` the sensor probes: those the harness can launch (a launch
    spec, not withheld by ``trust``) whose address is loopback. An attach-only peer, a withheld one
    and a remote one are never in it."""
    return tuple(
        r for r in serving_records(harness)
        if r.prefix != selected.prefix and r.spec is not None
        and not record_withheld(trust, r) and is_loopback(r.base_url))


def shares_base_url(record: ServingRecord, records: Any) -> bool:
    """True when a record in ``records`` other than ``record`` itself has ``record``'s address."""
    want = _norm_url(record.base_url)
    return any(r.prefix != record.prefix and _norm_url(r.base_url) == want for r in records)


def _attribute(on_url: list[ServingRecord], ids: tuple[str, ...]) -> ServingRecord | None:
    """The record an answering address belongs to (R-13-27): the one whose served name is among the
    ``ids``; else the only record there; else the primary when it is there; else none."""
    named = next((r for r in on_url if r.served in ids), None)
    if named is not None:
        return named
    if len(on_url) == 1:
        return on_url[0]
    return next((r for r in on_url if r.peer is None), None)


async def sense_serving(
    harness: Any, records: Any, *, timeout: float = SENSE_TIMEOUT_SECONDS
) -> ServingSense:
    """Probe the address of every record in ``records`` once (records sharing an address share one
    probe), in parallel and off the event loop, with the launch key of the server there. A probe that
    cannot connect, times out or fails any other way answers nothing; a reply that is not an OpenAI
    model list answers with no ids (the server IS there). An answering address is attributed to one
    record by ``_attribute``. Writes nothing."""
    records = tuple(records)
    by_url: dict[str, list[ServingRecord]] = {}
    for record in records:
        by_url.setdefault(_norm_url(record.base_url), []).append(record)
    probes = [(group[0].base_url, group) for group in by_url.values()]

    async def _probe(url: str) -> tuple[str, ...] | None:
        try:
            ids, reachable = await asyncio.to_thread(
                list_live_models, url, timeout, **_launch_key(harness, url))
        except MalformedModelListError:
            return ()
        except Exception:  # noqa: BLE001 — a probe never raises into start: it answers nothing
            return None
        return tuple(i for i in ids if isinstance(i, str)) if reachable else None

    results = await asyncio.gather(*(_probe(url) for url, _ in probes))
    answering: dict[str, ServingRecord] = {}
    ids_by_url: dict[str, tuple[str, ...]] = {}
    for (url, group), got in zip(probes, results):
        if got is None:
            continue
        ids_by_url[url] = got
        owner = _attribute(group, got)
        if owner is not None:
            answering.setdefault(owner.prefix, owner)
    order = {r.prefix: i for i, r in enumerate(records)}
    return ServingSense(
        tuple(sorted(answering.values(), key=lambda r: order[r.prefix])),
        MappingProxyType(ids_by_url))


_UNSET: Any = object()


@dataclass(frozen=True)
class PlanEntry:
    """One entry; ``kind`` is "live" (served now), "cold" (a launch record not answering) or "setup"."""
    name: str
    kind: str
    runtime: str | None
    window: int | None = None
    where: str | None = None
    refusal: str | None = None
    record: ServingRecord | None = None
    endpoint: Any = None
    active: bool = False
    rate: str | None = None
    path: str | None = None
    setup: Any = None
    others: tuple[str, ...] = ()


@dataclass(frozen=True)
class EntryGroup:
    header: str
    entries: tuple[PlanEntry, ...]
    numbered: bool


@dataclass(frozen=True)
class Entries:
    """The list: groups in display order, the entries said with their reason, notes, and who else holds the primary's port."""
    groups: tuple[EntryGroup, ...]
    not_offered: tuple[PlanEntry, ...] = ()
    notes: tuple[str, ...] = ()
    foreign: str | None = None

    def numbered(self) -> tuple[PlanEntry, ...]:
        return tuple(e for g in self.groups if g.numbered for e in g.entries)

    def pick(self, arg: str) -> "PlanEntry | str | None":
        """A decimal is that numbered entry; a name is the entry it names, or, when that entry is not numbered, the sentence saying why it cannot be picked."""
        if arg.isascii() and arg.isdigit():
            shown = self.numbered()
            return shown[int(arg) - 1] if 1 <= int(arg) <= len(shown) else None
        every = [(e, g.numbered) for g in self.groups for e in g.entries] + [(e, False) for e in self.not_offered]
        for entry, numbered in every:
            names = {entry.name, *entry.others}
            if entry.record is not None:
                names |= {entry.record.name, entry.record.served}
            if arg in names:
                return entry if numbered else (entry.refusal or f"{entry.name} cannot be switched to now")
        return None


def _runtime_label(provider_type: Any) -> str:
    row = row_of(provider_type)
    return row.name if row is not None else str(provider_type or "server")


def _host(base_url: str) -> str:
    try:
        return urlsplit(base_url).netloc or base_url
    except ValueError:
        return base_url


def _primary_port(harness: Any) -> int | None:
    server = getattr(harness, "server", None)
    if server is not None:
        return server.port
    url = harness.provider.base_url
    try:
        return urlsplit(url).port if is_loopback(url) else None
    except ValueError:
        return None


def _model_file(spec: Any) -> str:
    """The realpath of a GGUF record's file, a checkpoint record's model string as it is."""
    model = str(spec.model)
    return os.path.realpath(os.path.expanduser(model)) if RUNTIMES[spec.runtime].gguf else model


def _unit_serves(config_dir: Any, specs: Any, main_pid: int | None) -> bool:
    """True when the harness unit's main process is the verified launch of one of ``specs``."""
    from localharness.provider import lifecycle, procfs
    if main_pid is None:
        return False
    launches = (procfs.read_launch(config_dir, expect=lifecycle.record_identity(s), main_pid=main_pid)
                for s in specs if s is not None)
    return any(x is not None and x.verified and x.owner == "unit" for x in launches)


def _medians(config_dir: Any) -> dict[str, float]:
    from localharness.provider import speed_stats
    try:
        return speed_stats.all_median_tps(speed_stats.default_speed_stats_path(config_dir))
    except Exception:  # noqa: BLE001 — a ledger that cannot be read shows no rates
        return {}


def _priced(harness: Any, record: ServingRecord, *, config_dir: Any, memory: Any, floor: int | None,
            unit: Any) -> tuple[int | None, str | None, str | None]:
    """``(window, where, refusal)`` of the next launch of ``record``'s spec against ``memory``."""
    from localharness.provider import lifecycle, server
    spec, auto = record.spec, getattr(harness, "autosense", None)
    key = _launch_key(harness, record.base_url, spec)
    decision = server.launch_decision(spec, bool(key), peer=record.peer.name if record.peer else None)
    if decision.refusal is not None:
        return None, None, decision.refusal
    if auto is None or not getattr(lifecycle.strategy_for(spec, unit=unit), "plans_launch", False):
        return None, None, None
    try:
        plan = lifecycle.plan_launch(
            spec, lifecycle.sense_launch(spec, config_dir, auto, memory=memory), decision,
            autosense=auto, api_key_held=bool(key), floor=floor)
    except server.LaunchRefused as exc:
        return None, None, str(exc)
    report = plan.report
    placement = report.placement if report is not None else None
    return (report.asked if report is not None else None), (placement.where if placement is not None else None), None


def plan_entries(
    harness: Any, *, live: Mapping[str, "tuple[str, ...] | None"], config_dir: Any, trust: Any,
    floor: int | None, library: Any = None, memory: Any = _UNSET, unit: Any = None,
    current: "tuple[str, str] | None" = None,
) -> Entries:
    """What this box can switch to now; ``live`` maps each probed ``_norm_url`` to its served ids (``()`` not a model list, None no answer). Reads memory once, takes no lock, writes nothing."""
    from localharness.cli import workspace
    from localharness.provider import lifecycle, speed_stats

    provider, records = harness.provider, serving_records(harness)
    cur_url, cur_model = current or (provider.base_url, provider.default_model)
    cur_url = _norm_url(cur_url)
    auto = getattr(harness, "autosense", None)
    specs = [r.spec for r in records]
    port = _primary_port(harness)
    holder = workspace.port_holder(port) if port is not None else None
    own = lifecycle.own_launch(config_dir, specs)
    foreign = holder if own is None else None
    who = None
    if foreign is not None:
        who = workspace.visible_text(foreign.unit) if foreign.unit else workspace.incumbent_subject(foreign, port=port)
    owner = None
    if own is not None:
        owner = "autostart" if _unit_serves(config_dir, specs, getattr(unit, "main_pid", None)) else "the harness"
    elif who is not None:
        owner = f"{who} (yours)"
    held = cur_model if cur_url == _norm_url(provider.base_url) else (
        primary_served_name(getattr(harness, "server", None), provider) or provider.default_model)
    if memory is _UNSET:
        memory = lifecycle.memory_after_launch(auto, own, stopping=foreign is not None) if auto is not None else None
    medians = _medians(config_dir)

    def rate(ptype: Any, name: str) -> str | None:
        v = medians.get(speed_stats.speed_key(ptype, name))
        return f"{v:.1f} t/s measured" if v is not None else None

    endpoints = {_norm_url(provider.base_url): (provider.provider_type, None)}
    for ep in getattr(harness, "extra_endpoints", None) or []:
        endpoints.setdefault(_norm_url(ep.base_url), (ep.provider_type, ep))

    groups: list[EntryGroup] = []
    notes: list[str] = []
    shown: dict[str, tuple[str, str]] = {}
    for url in sorted(endpoints, key=lambda u: u != cur_url):
        ptype, ep = endpoints[url]
        ids = live.get(url)
        if ids is None:
            continue
        label = ep.name if ep is not None else "primary"
        on_url = [r for r in records if _norm_url(r.base_url) == url]
        found = []
        for served in ids:
            if served in shown:
                first, first_url = shown[served]
                if cur_url not in (url, first_url):
                    notes.append(f"{label} ({url}) also serves {served!r} — hidden by {first} (first configured wins)")
                continue
            shown[served] = (label, url)
            found.append(PlanEntry(
                served, "live", ptype, record=next((r for r in on_url if served in (r.served, r.name)), None),
                endpoint=ep, active=url == cur_url and served == cur_model, rate=rate(ptype, served)))
        if found:
            suffix = f" · {owner}" if owner and url == _norm_url(provider.base_url) else ""
            groups.append(EntryGroup(f"{_runtime_label(ptype)} · {_host(url)}{suffix}", tuple(found), True))
        elif not ids and url != cur_url:
            notes.append(f"skipped {label} ({url}): not an OpenAI-compatible model list")
    for ep in getattr(harness, "extra_endpoints", None) or []:
        if ep.lifecycle is None and live.get(_norm_url(ep.base_url)) is None:
            notes.append(f"skipped {ep.name} ({ep.base_url}): unreachable")

    serving = set()
    for url, ids in live.items():
        owner = _attribute([r for r in records if _norm_url(r.base_url) == url], ids) if ids is not None else None
        if owner is not None:
            serving.add(owner.prefix)
    cold, refused = [], []
    last = speed_stats.last_served(speed_stats.default_speed_stats_path(config_dir))
    by_file: dict[str, list[ServingRecord]] = {}
    for record in records:
        if record.spec is None or record.prefix in serving:
            continue
        if record_withheld(trust, record):
            refused.append(PlanEntry(
                record.name, "cold", record.spec.runtime, record=record, endpoint=record.peer,
                refusal=f"{record.name} cannot run here: its record has changes you have not confirmed"))
        else:
            by_file.setdefault(_model_file(record.spec), []).append(record)
    for path, twins in by_file.items():
        record = next((r for r in twins if r.name == last), twins[0])
        try:
            window, where, refusal = _priced(harness, record, config_dir=config_dir, memory=memory, floor=floor, unit=unit)
        except Exception as exc:  # noqa: BLE001 — one record's failure is its own line, never the list's
            window, where, refusal = None, None, f"{record.name} cannot run here: could not plan it ({type(exc).__name__})"
        entry = PlanEntry(
            record.name, "cold", record.spec.runtime, window, where, refusal, record, record.peer,
            rate=rate(record.provider_type, record.served), path=path,
            others=tuple(r.name for r in twins if r is not record))
        (refused if refusal is not None else cold).append(entry)
    if cold and foreign is not None:
        cold = [replace(e, refusal=workspace.incumbent_sentence(
            foreign, port=port, form="imperative", served=held, target=e.name)) for e in cold]
        groups.append(EntryGroup(f"available once {who} is stopped", tuple(cold), False))
    elif cold:
        groups.append(EntryGroup("this box can launch", tuple(cold), True))
    return Entries(tuple(groups), tuple(refused), tuple(notes), who)


def not_offered_reason(sentence: str, name: str) -> str:
    """The reason in a refusal sentence: without the leading name and ``cannot run here:``, up to the first `` — ``, no final period."""
    text = sentence.removeprefix(f"{name} ").removeprefix("cannot run here: ")
    return text.split(" — ")[0].rstrip(".")


def entry_lines(entries: Entries) -> list[str]:
    """The list as the lines both surfaces print; every name and URL passes ``visible_text``."""
    from localharness.cli.workspace import visible_text as vt

    lines, n = ["Models:"], 0
    for group in entries.groups:
        lines.append(f"▸ {vt(group.header)}")
        for e in group.entries:
            if e.kind == "live":
                detail = "(serving)" + ("  ● [active]" if e.active else "")
            else:
                detail = _runtime_label(e.runtime)
                detail += " · experts in RAM" if e.where == "experts in RAM" else ""
                detail += f" · {e.window:,}" if e.window is not None else ""
            if group.numbered:
                n += 1
                lead = f"  {n}. "
            else:
                lead = "     "
            lines.append(f"{lead}{vt(e.name)}  {detail}" + (f"  [{e.rate}]" if e.rate else ""))
    for e in entries.not_offered:
        lines.append(f"  not offered: {vt(e.name)} — {vt(not_offered_reason(e.refusal or '', e.name))}")
    lines.extend(f"  · {vt(note)}" for note in entries.notes)
    return lines


def listing(harness: Any, **kw: Any) -> tuple[Entries, list[str]]:
    entries = plan_entries(harness, **kw)
    return entries, entry_lines(entries)


# --------------------------------------------------------------------------- landing / selection writers
# `persist_landed` is the ONE writer of a landing's outcome (D-07): start, the REPL wrapper and the
# autostart unit's immediate swap all call it. `select_pending` is the CLI's SELECTION write (D-10):
# the intent the next start's ladder reads, not an outcome.

LANDING_OUTCOMES = frozenset({"launch", "swap", "attach"})
SELECTED_NEXT_START_LINE = "Selected {name} — the next start serves it"
SELECTED_NEXT_START_OR_BOOT_LINE = "Selected {name} — the next start or boot serves it"


async def persist_landed(
    harness: Any,
    record: ServingRecord,
    *,
    outcome: str,
    rung: str | None = None,
    served: str | None = None,
    config_dir: Any = None,
    keep_selection: bool = False,
) -> None:
    """Write what a landing on ``record`` decided. ``outcome`` is ``"launch"`` or ``"swap"`` (this
    process made ``record`` the served one) or ``"attach"`` (it found ``record`` already serving).

    ``last_served`` is always recorded (best effort, like every ledger write). ``active_endpoint`` is
    written for a launch or swap only (a peer's, or cleared for the primary): an attach, including an
    attach to another record, never persists a selection (D-15). ``pending_preset`` is removed after a
    launch or swap, and after an attach that resolved on the ``pending_preset`` rung (D-08). Raises
    what the overlay writers raise; the caller words it.

    ``keep_selection`` leaves the stored ``active_endpoint`` as it is on a launch or swap: for a landing
    on the primary because the selected peer could not be served from here (an attach-only peer that is
    down), which must not make the person forget the peer they chose."""
    if outcome not in LANDING_OUTCOMES:
        raise ValueError(
            f"unknown landing outcome {outcome!r} (one of {', '.join(sorted(LANDING_OUTCOMES))})")
    moved = outcome in ("launch", "swap")
    if moved and not keep_selection:
        if record.peer is not None:
            await persist_active_endpoint(
                harness, record.peer, served or record.served or record.name, config_dir=config_dir)
        else:
            clear_active_endpoint(harness, config_dir=config_dir)
    from localharness.provider import speed_stats

    try:
        speed_stats.record_last_served(
            speed_stats.default_speed_stats_path(global_config_dir(config_dir)), record.name)
    except OSError:
        pass
    if moved or (outcome == "attach" and rung == RUNG_PENDING):
        clear_pending_preset(harness, config_dir=config_dir)


async def select_pending(
    harness: Any, record: ServingRecord, *, served: str | None = None, config_dir: Any = None
) -> None:
    """Record ``record`` as what the next start serves (D-10): a peer is persisted as the
    ``active_endpoint`` as ``/model`` does, the primary clears one, then ``pending_preset`` names the
    record. Global overlay only; raises what the overlay writers raise."""
    if record.peer is not None:
        await persist_active_endpoint(
            harness, record.peer, served or record.served or record.name, config_dir=config_dir)
    else:
        clear_active_endpoint(harness, config_dir=config_dir)
    write_pending_preset(harness, record.name, config_dir=config_dir)


async def persist_effort(
    harness: Any,
    level: str,
    *,
    base_url: str,
    channel_id: str,
    config_dir: Any = None,
    audit_base_dir: Any = None,
) -> str | None:
    """Persist a ``/effort`` choice for the endpoint serving ``base_url`` through the audited user
    overlay, and return None on clean success, ``EFFORT_SESSION_ONLY_NOTE`` when there is nowhere
    to record it, or a warning string when ONLY the post-write audit emit failed.

    Exactly ONE dict path is written, chosen by who serves ``base_url``:

    - the primary with a ``server:`` block -> ``server.effort``;
    - the active peer (``active_endpoint.base_url`` matches) -> ``active_endpoint.effort``;
    - the primary run by hand (no ``server:``) -> ``org.default_effort``;
    - any other peer -> nothing is written (it has no record of its own to carry the level).

    Never ``extra_endpoints`` and never a whole record: a peer's list entry holds its ``api_key`` /
    ``extra_headers``, and rewriting it would copy those into ``overrides.yaml`` (spec §7.2). Setting
    one dict path leaves every other key as the overlay already had it. ``effort`` is not a
    fingerprinted key (spec §9), so the machine trust snapshot is untouched by this write.

    The audit row is actor ``channel`` with the channel id as ``actor_detail``: the level was
    chosen by a person on a channel (terminal, phone or Discord), not by ``components set``. As in
    ``persist_default_model`` the row is emitted even when the level is unchanged
    (``before_value == after_value``). ``config_dir`` / ``audit_base_dir`` are the same two inputs
    as there: the overlay is machine-wide truth, the audit record follows the work.
    """
    if level not in LEVELS:
        raise ValueError(f"effort must be one of {', '.join(LEVELS)}; got {level!r}")

    def norm(url: Any) -> str:
        return (url or "").rstrip("/")

    primary = norm(base_url) == norm(harness.provider.base_url)
    active = getattr(harness, "active_endpoint", None)
    if primary and harness.server is not None:
        path, before = "server.effort", harness.server.effort

        def apply() -> None:
            harness.server.effort = level
    elif active is not None and norm(active.base_url) == norm(base_url):
        path, before = "active_endpoint.effort", active.effort

        def apply() -> None:
            active.effort = level
    elif primary:
        path, before = "org.default_effort", harness.org.default_effort

        def apply() -> None:
            harness.org.default_effort = level
    else:
        return EFFORT_SESSION_ONLY_NOTE

    overlay_path = _resolve_user_overlay_path(global_config_dir(config_dir))
    new_overlay = dict(load_overlay(overlay_path))
    set_value_in_dict(new_overlay, path, level)
    _validated(harness, {k: v for k, v in new_overlay.items() if k != _AGENT_KEY})
    atomic_write_overlay(overlay_path, new_overlay)
    apply()

    # The emit has its own try/except (#37): it runs after the durable write, so an unwritable
    # audit log is a warning to show, never a persist failure.
    try:
        _audit_base = audit_base_dir if audit_base_dir is not None else config_dir
        audit_path = harness.org.audit_log_path
        bus = EventBus(
            persist_path=resolve_runtime_path(audit_path, _audit_base) if audit_path else None
        )
        await bus.publish(
            ComponentMutated(
                path=path,
                before_value=before,
                after_value=level,
                layer="user",
                actor="channel",
                actor_detail=channel_id,
            )
        )
    except Exception as exc:  # noqa: BLE001 — the durable overlay write already succeeded
        return f"persisted, but the audit log could not be written: {exc}"
    return None


def pinned_agents(
    config_dir: Path | None, local_config_dir: Path | None = None
) -> list[tuple[str, str]]:
    """Agents a persisted org/provider ``default_model`` switch will NOT reach on the next
    ``start`` — because a concrete ``model:`` pins them ABOVE the org default in the resolution
    chain ``agent -> division -> org`` (``start_cmd`` resolves that chain).

    Two pin sources, both reported (#36 — the division source was previously missed, so a
    division-pinned agent silently ignored a switch with no warning):
      - the agent's own raw ``model:`` (an agent-level override lever) -> ``(name, model)``;
      - else, its division's ``model:`` (a division-wide pin) -> ``(name (via division X), model)``.
    An agent that inherits all the way to org is NOT listed — the switch DOES reach it.

    Reads RAW yaml via the loader's own reader (NOT ``load_agent().model``, which RESOLVES
    ``inherit`` to a concrete org default and would make every inheritor look pinned). Returns
    ``[(agent_label, pinned_model), ...]``.

    ``local_config_dir`` (v0.13, closes 39-05's named carve-out): the session's workspace layer,
    or None. Callers that have one pass it; omitting it keeps today's global-only roster.
    """
    out: list[tuple[str, str]] = []
    if config_dir is None:
        return out
    from localharness.config.loader import ConfigLoader, _load_yaml_file

    # v0.13 (closes 39-05's carve-out): with a workspace layer the pin warning must see the
    # workspace's own agents too — a workspace agent pinning a concrete model traps the switch
    # exactly like a global one, and warning about only half the roster is worse than not warning.
    loader = ConfigLoader(config_dir=Path(config_dir), local_config_dir=local_config_dir)
    for stem in loader.list_agents():
        path = loader._find_file("agents", stem)
        if path is None:
            continue
        try:
            raw = _load_yaml_file(path)  # the loader's own yaml reader (shared parse/error path)
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        label = raw.get("name") or stem
        m = raw.get("model")
        if isinstance(m, str) and m and m != "inherit":
            out.append((label, m))  # agent-level pin
            continue
        # Agent inherits at its own level — a division-level pin still traps it.
        div_name = raw.get("division")
        if not div_name:
            continue
        try:
            division = loader.load_division(div_name)
        except Exception:
            continue
        if division.model and division.model != "inherit":
            out.append((f"{label} (via division {div_name})", division.model))
    return out


def agents_with_effort(
    config_dir: Path | None, local_config_dir: Path | None = None
) -> list[tuple[str, str]]:
    """Agents whose own yaml sets a reasoning level, as ``[(agent_label, level), ...]`` sorted by
    name — the list ``/effort show`` prints so a person can see which agents a session's ``/effort``
    does not move on its own (D-05: a session choice is a rung above the agent file, but the file's
    value is still what that agent asks for).

    Reads RAW yaml through the loader's own reader, exactly as ``pinned_agents`` does, and for the
    same reason: a resolved ``AgentConfig.effort`` already has the division's rung collapsed in
    (spec §7.2), so every agent in a division that sets a level would look like it set one itself.
    An agent whose ``effort`` is absent, ``inherit`` or not a level is not listed; a yaml that will not parse is
    skipped, as ``pinned_agents`` skips it. ``local_config_dir`` is the session's workspace layer or
    None, with the same meaning as there.
    """
    out: list[tuple[str, str]] = []
    if config_dir is None:
        return out
    from localharness.config.loader import ConfigLoader, _load_yaml_file

    loader = ConfigLoader(config_dir=Path(config_dir), local_config_dir=local_config_dir)
    for stem in loader.list_agents():
        path = loader._find_file("agents", stem)
        if path is None:
            continue
        try:
            raw = _load_yaml_file(path)
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        # An unquoted `off` is the boolean False to YAML 1.1; the loader reads it as `off`, so this
        # listing does too (one coercion, shared). `on` is not a level and the loader refuses it.
        try:
            level = coerce_yaml_effort(raw.get("effort"))
        except ValueError:
            continue
        # Only a level the loader would accept: absent and "inherit" set nothing, and anything else
        # (a typo) is a file the loader refuses, which is reported there, not here.
        if not isinstance(level, str) or level not in LEVELS:
            continue
        out.append((raw.get("name") or stem, level))
    return sorted(out)
