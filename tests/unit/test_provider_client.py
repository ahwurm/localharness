

def test_local_client_disables_sdk_auto_retries():
    """The SDK's silent default (2 retries) turns one timed-out local generation into
    3x the wait — observed live as 30 min of dead air (3 x 600s). Local endpoints
    must fail fast; remote keeps the default."""
    from localharness.provider.client import LLMClient, LLMConfig

    local = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m",
                                timeout_seconds=600))
    assert local._client.max_retries == 0

    remote = LLMClient(LLMConfig(base_url="https://api.example.com/v1", model="m",
                                 timeout_seconds=120, is_local=False))
    assert remote._client.max_retries == 2


# ---------------------------------------------------------------------------
# True-streaming chunk assembly (_consume_native_stream)
# ---------------------------------------------------------------------------

import pytest
from types import SimpleNamespace as NS


def _chunk(content=None, tool_calls=None, usage=None):
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(usage=usage, choices=[NS(delta=delta)] if (content or tool_calls) or usage is None else [])


async def _aiter(chunks):
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_assembles_content_and_calls_on_token():
    from localharness.provider.client import LLMClient

    seen = []
    async def on_token(piece): seen.append(piece)

    chunks = [_chunk(content="Hel"), _chunk(content="lo "), _chunk(content="world")]
    msg, usage = await LLMClient._consume_native_stream(_aiter(chunks), on_token)
    assert msg.content == "Hello world"
    assert seen == ["Hel", "lo ", "world"]
    assert msg.tool_calls is None
    assert usage is None


@pytest.mark.asyncio
async def test_stream_assembles_fragmented_tool_calls():
    from localharness.provider.client import LLMClient

    chunks = [
        _chunk(tool_calls=[NS(index=0, id="tc-a", function=NS(name="web_search", arguments=""))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='{"que'))]),
        _chunk(tool_calls=[NS(index=1, id="tc-b", function=NS(name="agent", arguments='{"agent_id"'))]),
        _chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='ry": "x"}'))]),
        _chunk(tool_calls=[NS(index=1, id=None, function=NS(name=None, arguments=': "explore"}'))]),
    ]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.content is None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0] == {"id": "tc-a", "type": "function",
                                 "function": {"name": "web_search", "arguments": '{"query": "x"}'}}
    assert msg.tool_calls[1]["function"]["arguments"] == '{"agent_id": "explore"}'


@pytest.mark.asyncio
async def test_stream_captures_final_usage_chunk():
    from localharness.provider.client import LLMClient

    final = NS(usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15), choices=[])
    chunks = [_chunk(content="ok"), final]
    msg, usage = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.content == "ok"
    assert usage.completion_tokens == 5


def _chunk_fr(content=None, tool_calls=None, finish_reason=None):
    """Like _chunk but carries choices[0].finish_reason (the field OpenAI-style streams set
    on the final content chunk: 'stop' | 'length' | 'tool_calls')."""
    delta = NS(content=content, tool_calls=tool_calls)
    return NS(usage=None, choices=[NS(delta=delta, finish_reason=finish_reason)])


@pytest.mark.asyncio
async def test_stream_captures_finish_reason_length_midtoolcall():
    """#77: a completion cut at the output-token ceiling mid-tool-call carries
    finish_reason='length' on its final chunk. The assembled message MUST surface it so the
    loop can refuse to execute the truncated call. Native + xml share this one consumer, so
    capturing here covers both modes."""
    from localharness.provider.client import LLMClient

    chunks = [
        _chunk_fr(tool_calls=[NS(index=0, id="tc-a", function=NS(name="write", arguments='{"pa'))]),
        _chunk_fr(finish_reason="length"),
    ]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason == "length"
    assert msg.tool_calls is not None  # a (truncated) call was still assembled


@pytest.mark.asyncio
async def test_stream_captures_finish_reason_stop():
    """A clean completion surfaces finish_reason='stop' (guard must NOT fire)."""
    from localharness.provider.client import LLMClient

    chunks = [_chunk_fr(content="done"), _chunk_fr(finish_reason="stop")]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_finish_reason_absent_is_none():
    """Providers that never emit finish_reason (older/partial mocks) yield None, not an
    error — the guard simply stays dormant."""
    from localharness.provider.client import LLMClient

    chunks = [_chunk(content="hi")]  # legacy chunk, no finish_reason attribute
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.finish_reason is None


# ---------------------------------------------------------------------------
# #18 — XML tool-call mode must stream at the transport level. `_complete_xml` /
# `_complete_xml_fallback` accepted a `stream` parameter and IGNORED it, so any
# model whose capability probe falls back to XML mode silently issued whole-
# response requests for the entire agent loop (read timeout races the whole
# generation; a cancel leaves vLLM decoding into the void). No log signal.
# ---------------------------------------------------------------------------

from localharness.provider.client import LLMConfig


def _xml_cfg() -> LLMConfig:
    # is_local=False: skip the inference gate + the local-timeout floor; the subject
    # here is whether `stream=True` reaches the transport, not gating.
    return LLMConfig(base_url="http://127.0.0.1:9/v1", model="m", timeout_seconds=300.0,
                     tool_call_mode="xml", is_local=False)


class _StreamOrResp:
    """A create() return that behaves as BOTH a whole-response object (.choices/.usage)
    AND an async chunk stream — so the RED failure is a clean `stream=True` assertion,
    never an AttributeError from whichever branch the code happens to take."""

    def __init__(self, content: str):
        self._content = content
        self.choices = [NS(message=NS(content=content, tool_calls=None))]
        self.usage = None

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        yield NS(usage=None, choices=[NS(delta=NS(content=self._content, tool_calls=None))])


@pytest.mark.asyncio
async def test_stream_complete_xml_mode_passes_stream_true():
    """RED: stream_complete() in XML mode must request transport streaming. The dead
    `stream` param made XML mode silently non-streaming for the whole loop (#18)."""
    from localharness.provider.client import LLMClient

    client = LLMClient(_xml_cfg())
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        return _StreamOrResp("<tool_call>{}</tool_call>")

    client._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _usage = await client.stream_complete([{"role": "user", "content": "hi"}])
    assert captured and captured[0].get("stream") is True
    # Full text is buffered client-side BEFORE the XML parse still sees it.
    assert msg.content == "<tool_call>{}</tool_call>"


@pytest.mark.asyncio
async def test_complete_xml_fallback_streams_when_requested():
    """RED: the system-prompt-injection fallback honors stream too — a BadRequestError
    re-entry must not silently drop back to a whole-response request (#18)."""
    from localharness.provider.client import LLMClient

    client = LLMClient(_xml_cfg())
    captured: list[dict] = []

    async def fake_create(**kwargs):
        captured.append(kwargs)
        return _StreamOrResp("ok")

    client._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _usage = await client._complete_xml_fallback(
        [{"role": "user", "content": "hi"}], None, stream=True
    )
    assert captured and captured[0].get("stream") is True
    assert msg.content == "ok"


# ---------------------------------------------------------------------------
# rebind_endpoint — cross-endpoint /model swap (0.10.0 model tree)
# ---------------------------------------------------------------------------


def test_rebind_endpoint_rebuilds_client_and_resets_sticky():
    """Re-pointing at a different server REBUILDS the AsyncOpenAI (base_url is baked at
    construction) and resets the per-server sticky `_tools_param_rejected` — the new server may
    accept the `tools` param the old one rejected."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600))
    old = c._client
    c._tools_param_rejected = True  # a prior server rejected `tools`
    c.rebind_endpoint("http://127.0.0.1:11434/v1", api_key="k2", extra_headers={"X": "1"})
    assert c.config.base_url == "http://127.0.0.1:11434/v1"
    assert c.config.api_key == "k2"
    assert "11434" in str(c._client.base_url)   # rebuilt against the new server
    assert c._client is not old
    assert c._tools_param_rejected is False      # reset for the new server


@pytest.mark.asyncio
async def test_rebind_then_detect_refreshes_fn_converter():
    """After a rebind the caller MUST call detect_capabilities(), which re-derives the converter:
    non-None for an xml server, None for a native one — the single place mode + converter stay in
    lockstep (the SAME path the same-endpoint hot-swap already uses, so no separate fix needed)."""
    from localharness.provider.client import LLMClient, LLMConfig

    # is_local=False → the inference gate yields without a TCP probe (CPU-only, no network).
    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m",
                            timeout_seconds=600, is_local=False, tool_call_mode="native"))
    assert c._fn_converter is None
    c.rebind_endpoint("http://127.0.0.1:11434/v1")

    async def create_xml(**kw):
        return NS(choices=[NS(message=NS(tool_calls=None, content="<tool_call>x</tool_call>"))])

    async def models_empty():
        return NS(data=[])

    c._client = NS(chat=NS(completions=NS(create=create_xml)), models=NS(list=models_empty))
    cap = await c.detect_capabilities()
    assert cap.tool_call_mode == "xml"
    assert c._fn_converter is not None           # created for an xml/text server

    async def create_native(**kw):
        return NS(choices=[NS(message=NS(tool_calls=[NS(id="t")], content=None))])

    c._client = NS(chat=NS(completions=NS(create=create_native)), models=NS(list=models_empty))
    cap = await c.detect_capabilities()
    assert cap.tool_call_mode == "native"
    assert c._fn_converter is None               # cleared for a native server


def test_rebind_endpoint_failure_restores_prior_and_reraises(monkeypatch):
    """If the rebuild raises, the prior client + config are restored and the error re-raised, so a
    failed re-point never strands a half-configured client (mirrors TokenCounter.rebind #30)."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600))
    old_client = c._client
    old_url = c.config.base_url

    def boom(self):
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(LLMClient, "_build_client", boom)
    with pytest.raises(RuntimeError, match="cannot build client"):
        c.rebind_endpoint("http://127.0.0.1:11434/v1", api_key="k2")
    assert c.config.base_url == old_url          # restored (never left half-applied)
    assert c.config.api_key == "none"            # restored
    assert c._client is old_client               # restored


def test_rebind_to_empty_headers_clears_previous_headers():
    """#3: rebinding to an endpoint with {} headers after one with custom headers CLEARS them —
    'leave unchanged' is only for extra_headers=None. An endpoint's identity is exactly its OWN
    headers; the call site must pass {} (not `ep.extra_headers or None`, which would coerce the
    default {} to None and silently INHERIT the previous endpoint's headers)."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600,
                            extra_headers={"X-Custom": "secret"}))
    assert "X-Custom" in c._client.default_headers          # baked into the first client
    c.rebind_endpoint("http://127.0.0.1:11434/v1", extra_headers={})
    assert c.config.extra_headers == {}                     # explicitly cleared, not left as-is
    assert "X-Custom" not in c._client.default_headers      # NOT carried over to the new server


def test_rebind_none_headers_leaves_previous_unchanged():
    """The contract the {} case contrasts with: extra_headers=None means LEAVE UNCHANGED (the only
    correct use of None), so an explicit None keeps the prior headers on the rebuilt client."""
    from localharness.provider.client import LLMClient, LLMConfig

    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="m", timeout_seconds=600,
                            extra_headers={"X-Custom": "secret"}))
    c.rebind_endpoint("http://127.0.0.1:11434/v1", extra_headers=None)
    assert c.config.extra_headers == {"X-Custom": "secret"}  # unchanged
    assert "X-Custom" in c._client.default_headers


# ---------------------------------------------------------------------------
# Measured decode speed (speed_stats wiring)
# ---------------------------------------------------------------------------


def _mk_speed_client(tmp_path, monkeypatch, provider_type="llamacpp"):
    from localharness.provider.client import LLMClient, LLMConfig

    monkeypatch.setenv("LOCALHARNESS_DIR", str(tmp_path))  # ledger lands under tmp
    return LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1", model="m",
                               provider_type=provider_type))


@pytest.mark.asyncio
async def test_stream_progress_counts_payload_deltas_once_each():
    """Content, tool-call AND reasoning deltas all advance the live progress (a native
    tool-calling or thinking stream measures like a prose one); role-only and usage-only
    chunks are not payload."""
    from localharness.provider.client import LLMClient

    progress = {"first_at": None, "chunks": 0, "server_tps": None}
    chunks = [
        NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=None))]),  # role-only
        _chunk(content="a"),
        _chunk(tool_calls=[NS(index=0, id="t", function=NS(name="f", arguments="{}"))]),
        NS(usage=None,
           choices=[NS(delta=NS(content=None, tool_calls=None, reasoning_content="hm"))]),
        NS(usage=NS(prompt_tokens=1, completion_tokens=3, total_tokens=4), choices=[]),
    ]
    await LLMClient._consume_native_stream(_aiter(chunks), None, progress)
    assert progress["chunks"] == 3
    assert progress["first_at"] is not None


@pytest.mark.asyncio
async def test_stream_progress_captures_engine_reported_rate():
    """llama.cpp's timings.predicted_per_second (engine ground truth) rides the final chunk."""
    from localharness.provider.client import LLMClient

    progress = {"first_at": None, "chunks": 0, "server_tps": None}
    final = NS(usage=NS(prompt_tokens=8, completion_tokens=40, total_tokens=48), choices=[],
               timings={"predicted_per_second": 16.49})
    await LLMClient._consume_native_stream(_aiter([_chunk(content="x"), final]), None, progress)
    assert progress["server_tps"] == 16.49


def test_note_gen_speed_records_verified_wall_rate(tmp_path, monkeypatch):
    """Exact usage tokens over the measured first-delta→done window → last_gen_tps + a
    ledger sample under provider_type:model."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 13.0)
    c._note_gen_speed({"first_at": 10.0, "chunks": 31, "server_tps": None},
                      NS(completion_tokens=31))
    assert c.last_gen_tps == 10.0  # 30 intervals / 3s
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == 10.0


def test_note_gen_speed_prefers_engine_rate_and_needs_ptype_for_ledger(tmp_path, monkeypatch):
    from localharness.provider.speed_stats import default_speed_stats_path

    c = _mk_speed_client(tmp_path, monkeypatch, provider_type=None)
    c._note_gen_speed({"first_at": 10.0, "chunks": 5, "server_tps": 16.49},
                      NS(completion_tokens=40))
    assert c.last_gen_tps == 16.49  # engine truth wins over wall math
    assert not default_speed_stats_path().exists()  # no provider_type → no key → no write


def test_note_gen_speed_without_usage_or_engine_rate_records_nothing(tmp_path, monkeypatch):
    c = _mk_speed_client(tmp_path, monkeypatch)
    c._note_gen_speed({"first_at": 10.0, "chunks": 50, "server_tps": None}, None)
    assert c.last_gen_tps is None  # chunk counts alone are never a verified sample


def test_gen_speed_snapshot_states(tmp_path, monkeypatch):
    """None → live (~approximate) while a stream is active → last verified after.

    The live leg needs a known tokens-per-delta ratio. Until one is measured the live rate is
    SUPPRESSED rather than assuming 1.0: under speculative decoding a delta carries several
    accepted tokens, so that assumption reads ~1/3 of the real rate. The ratio is persisted per
    model, so the blind window is the first stream ever for a model, not every session."""
    from localharness.provider import client as client_mod

    c = _mk_speed_client(tmp_path, monkeypatch)
    assert c.gen_speed_snapshot() is None
    c._stream_progress = {"first_at": 10.0, "chunks": 21, "server_tps": None}
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 12.0)
    assert c.gen_speed_snapshot() is None, "no ratio measured yet -> no live number"

    c._tokens_per_chunk = 1.0  # measured: one token per delta (no speculative decoding)
    assert c.gen_speed_snapshot() == (10.0, False)  # 20 intervals / 2s, live

    c._tokens_per_chunk = 3.4  # measured: MTP bundles ~3.4 accepted tokens per delta
    live, verified = c.gen_speed_snapshot()
    assert verified is False and 33.0 < live < 36.0, live

    c._stream_progress = None
    c.last_gen_tps = 16.5
    assert c.gen_speed_snapshot() == (16.5, True)


@pytest.mark.asyncio
async def test_create_and_consume_stream_measures_and_records(tmp_path, monkeypatch):
    """Full stream path: include_usage requested, progress cleared after the stream, and a
    cleanly finished stream leaves a verified rate + a ledger sample."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)
    # 21 completion tokens over the 2s window below: a substantive sample (#136 floors), so the
    # wiring under test is exercised end to end rather than dropped at admission.
    final = NS(usage=NS(prompt_tokens=8, completion_tokens=21, total_tokens=29), choices=[])
    chunks = [_chunk(content="x"), _chunk(content="y"), _chunk(content="z"), final]

    async def fake_create(**kwargs):
        assert kwargs["stream_options"] == {"include_usage": True}
        return _aiter(chunks)

    c._client = NS(chat=NS(completions=NS(create=fake_create)))
    ticks = iter([100.0, 102.0])  # first payload delta at 100, note-time at 102
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: next(ticks, 102.0))
    msg, usage = await c._create_and_consume({}, stream=True)
    assert msg.content == "xyz"
    assert usage.completion_tokens == 21
    assert c._stream_progress is None  # cleared even though the stream succeeded
    assert c.last_gen_tps == 10.0  # 20 intervals / 2s
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == 10.0


def test_ledger_lands_in_the_sessions_config_dir_with_env_unset(tmp_path, monkeypatch):
    """`--config-dir` must reach the WRITER. Typer's envvar only READS $LOCALHARNESS_DIR — the
    flag never exports it — so a session started with an explicit dir has the env unset and the
    ambient path resolves to ~/.localharness: a ledger the REPL (which reads the session's dir)
    never sees, plus state leaking out of a deliberately isolated config dir."""
    from localharness.provider.client import LLMClient, LLMConfig
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    monkeypatch.delenv("LOCALHARNESS_DIR", raising=False)
    monkeypatch.delenv("LOCALHARNESS_HOME", raising=False)  # legacy leg of the #35 chain
    session_dir = tmp_path / "alt-config"
    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:9/v1", model="m",
                            provider_type="llamacpp", config_dir=session_dir))
    c._note_gen_speed({"first_at": 10.0, "chunks": 5, "server_tps": 16.49}, None)

    assert median_tps(default_speed_stats_path(session_dir), "llamacpp", "m") == 16.49
    assert default_speed_stats_path() != default_speed_stats_path(session_dir)  # ambient ≠ session


def test_llm_config_without_config_dir_keeps_ambient_ledger(tmp_path, monkeypatch):
    """Backward compatible: no config_dir (every existing caller until start passes one) keeps
    the ambient #35 precedence — $LOCALHARNESS_DIR else ~."""
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)  # sets LOCALHARNESS_DIR, no config_dir field
    assert c.config.config_dir is None
    c._note_gen_speed({"first_at": 10.0, "chunks": 5, "server_tps": 20.0}, None)
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == 20.0


def test_rebind_endpoint_clears_the_previous_endpoints_verified_rate(tmp_path, monkeypatch):
    """A verified rate belongs to the model that produced it. After a cross-endpoint swap the
    old server's tok/s must not be reported as the new model's measurement (the terminal renders
    verified rates in green, and _thinking_label shows ONLY verified ones)."""
    c = _mk_speed_client(tmp_path, monkeypatch)
    c.last_gen_tps = 31.0
    c.rebind_endpoint("http://127.0.0.1:11434/v1", provider_type="ollama")
    assert c.last_gen_tps is None
    assert c.gen_speed_snapshot() is None  # no number beats a stale one


def test_failed_rebind_keeps_the_still_current_rate(tmp_path, monkeypatch):
    """Mirror of the exception-safety contract: a rebind that raises leaves the client on the
    OLD endpoint, where the measured rate is still true — clearing it would lose a real sample."""
    from localharness.provider.client import LLMClient

    c = _mk_speed_client(tmp_path, monkeypatch)
    c.last_gen_tps = 31.0

    def boom(self):
        raise RuntimeError("cannot build client")

    monkeypatch.setattr(LLMClient, "_build_client", boom)
    with pytest.raises(RuntimeError, match="cannot build client"):
        c.rebind_endpoint("http://127.0.0.1:11434/v1")
    assert c.last_gen_tps == 31.0


@pytest.mark.asyncio
async def test_detect_capabilities_clears_the_previous_models_verified_rate():
    """Same-endpoint hot swap assigns config.model then re-probes — detect_capabilities is the
    one choke point every swap path crosses, so the stale rate dies there too."""
    from localharness.provider.client import LLMClient, LLMConfig

    # is_local=False → the inference gate yields without a TCP probe (CPU-only, no network).
    c = LLMClient(LLMConfig(base_url="http://127.0.0.1:8000/v1", model="fast-model",
                            timeout_seconds=600, is_local=False))
    c.last_gen_tps = 31.0          # measured on the OUTGOING model
    c.config.model = "slow-model"  # what the hot swap does, before probing

    async def create_native(**kw):
        return NS(choices=[NS(message=NS(tool_calls=[NS(id="t")], content=None))])

    async def models_empty():
        return NS(data=[])

    c._client = NS(chat=NS(completions=NS(create=create_native)), models=NS(list=models_empty))
    await c.detect_capabilities()
    assert c.last_gen_tps is None
    assert c.gen_speed_snapshot() is None  # no number beats the old model's number


def test_note_gen_speed_discards_implausible_rate_from_degenerate_window(tmp_path, monkeypatch):
    """#130: a sub-millisecond measured window makes decode_tps return tens of thousands of
    tok/s. The old guard only rejected tps<=0, so the artifact reached both the status line and
    the ledger median. Nothing is recorded and last_gen_tps stays untouched."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path

    c = _mk_speed_client(tmp_path, monkeypatch)
    # 5 tokens over a 0.1 ms window -> decode_tps == 40000.0
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 10.0001)
    c._note_gen_speed({"first_at": 10.0, "chunks": 5, "server_tps": None},
                      NS(completion_tokens=5))
    assert c.last_gen_tps is None
    assert not default_speed_stats_path().exists()


def test_note_gen_speed_still_records_a_fast_but_plausible_rate(tmp_path, monkeypatch):
    """The ceiling must not clip real hardware: a genuinely fast local decode still records.
    Measured over a substantive window (#136) — 301 tokens across a full second, not the same
    300 tok/s inferred from a 0.1s window that no longer counts as a measurement."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 11.0)
    c._note_gen_speed({"first_at": 10.0, "chunks": 301, "server_tps": None},
                      NS(completion_tokens=301))
    assert c.last_gen_tps == pytest.approx(300.0)  # 300 intervals / 1.0s
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == pytest.approx(300.0)


def test_note_gen_speed_discards_a_buffered_burst_the_ceiling_would_admit(tmp_path, monkeypatch):
    """#136 (the regression pin): 60 EXACT tokens whose deltas all landed inside 12ms — the
    coalesced-burst shape the short generations between tool calls produced on vLLM — measure
    4,916 tok/s. That is transport timing, not decoding, and it sails under #130's 10k ceiling:
    samples of 1,715..6,788 tok/s were recorded on a ~78 tok/s config that way."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 10.012)
    c._note_gen_speed({"first_at": 10.0, "chunks": 2, "server_tps": None},
                      NS(completion_tokens=60))
    assert c.last_gen_tps is None            # never reaches the status line either
    assert not default_speed_stats_path().exists()


def test_note_gen_speed_records_the_calibration_sample(tmp_path, monkeypatch):
    """The other half of #136's calibration: the honest 60-token / 1.5s generation (39.3 tok/s)
    the box actually produces must still be admitted — the floors reject artifacts, not work."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 11.5)
    c._note_gen_speed({"first_at": 10.0, "chunks": 60, "server_tps": None},
                      NS(completion_tokens=60))
    assert c.last_gen_tps == pytest.approx(39.33, abs=0.01)  # 59 intervals / 1.5s
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == pytest.approx(39.33, abs=0.01)


def test_note_gen_speed_keeps_the_ceiling_behind_the_substance_floors(tmp_path, monkeypatch):
    """#130's backstop stays live: a sample that CLEARS both floors (3,001 tokens over 0.29s)
    but states an impossible rate (10,344 tok/s) is still dropped."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 10.29)
    c._note_gen_speed({"first_at": 10.0, "chunks": 3001, "server_tps": None},
                      NS(completion_tokens=3001))
    assert c.last_gen_tps is None
    assert not default_speed_stats_path().exists()


def test_note_gen_speed_trusts_the_engines_own_rate_on_a_short_burst(tmp_path, monkeypatch):
    """The substance floors are for CLIENT-measured windows only. llama.cpp's
    timings.predicted_per_second is measured inside the engine's decode loop, so chunk-arrival
    timing cannot corrupt it — and it stayed clean across 25 live samples while vLLM's
    client-measured path produced the #136 garbage. A short burst still records."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client_mod.time, "monotonic", lambda: 10.012)
    c._note_gen_speed({"first_at": 10.0, "chunks": 2, "server_tps": 16.49},
                      NS(completion_tokens=5))
    assert c.last_gen_tps == 16.49
    assert median_tps(default_speed_stats_path(), "llamacpp", "m") == 16.49


# ---------------------------------------------------------------------------
# Reasoning-delta field spelling (#142)
#
# vLLM and llama.cpp name the thinking delta `reasoning_content`; Ollama's
# OpenAI-compatible endpoint names it `reasoning` and streams an EMPTY `content`
# alongside it while thinking. Reading only `reasoning_content` therefore left the
# whole thinking phase invisible on Ollama: the decode window opened at the first
# ANSWER token while usage counted the reasoning tokens too (inflated tok/s), and
# the reasoning text never reached anyone.
# ---------------------------------------------------------------------------


def _think_chunk(field, text, content=None):
    """A thinking delta spelled the way one runtime spells it, plus Ollama's empty
    `content` companion. `field` is "reasoning_content" (vLLM/llama.cpp) or "reasoning"."""
    delta = NS(content=content, tool_calls=None, **{field: text})
    return NS(usage=None, choices=[NS(delta=delta)])


class _TickClock:
    """A monotonic clock that advances with the STREAM rather than with calls, so WHICH
    delta stamps first_at changes the measured window — the whole consequence of #142."""

    def __init__(self, start=100.0, step=0.5):
        self.now, self.step = start, step

    def __call__(self):
        return self.now

    async def stream(self, chunks):
        for c in chunks:
            self.now += self.step
            yield c


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.asyncio
async def test_stream_window_opens_at_the_first_thinking_delta_either_spelling(field, monkeypatch):
    """first_at stamps on the arrival of ANY reasoning delta — including one carrying no text
    yet ("") — never waiting for the first answer token. Both spellings, same behavior."""
    from localharness.provider import client as client_mod

    clock = _TickClock()
    monkeypatch.setattr(client_mod.time, "monotonic", clock)
    progress = {"first_at": None, "chunks": 0, "server_tps": None}
    chunks = [
        _think_chunk(field, "", content=""),        # thinking started, no text in this delta
        _think_chunk(field, "hm", content=""),
        _chunk(content="42"),                       # answer begins
    ]
    await client_mod.LLMClient._consume_native_stream(clock.stream(chunks), None, progress)
    assert progress["first_at"] == 100.5            # the FIRST thinking delta, not the answer
    assert progress["chunks"] == 3                  # every thinking delta is payload too


@pytest.mark.asyncio
async def test_stream_role_only_delta_is_still_not_payload(monkeypatch):
    """The stamp rule widened to reasoning deltas only — a delta with no reasoning field at all
    and nothing else in it stays non-payload (a role-only chunk must not open the window)."""
    from localharness.provider import client as client_mod

    clock = _TickClock()
    monkeypatch.setattr(client_mod.time, "monotonic", clock)
    progress = {"first_at": None, "chunks": 0, "server_tps": None}
    chunks = [
        NS(usage=None, choices=[NS(delta=NS(content=None, tool_calls=None))]),
        _chunk(content="hi"),
    ]
    await client_mod.LLMClient._consume_native_stream(clock.stream(chunks), None, progress)
    assert progress["first_at"] == 101.0            # the content delta, not the role chunk
    assert progress["chunks"] == 1


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
@pytest.mark.asyncio
async def test_stream_surfaces_reasoning_text_under_either_spelling(field):
    """The assembled message exposes the thinking text on `reasoning_content` whatever the
    runtime called it on the wire — downstream readers ask for one field name."""
    from localharness.provider.client import LLMClient

    chunks = [
        _think_chunk(field, "Let me", content=""),
        _think_chunk(field, " think", content=""),
        _think_chunk(field, "", content=""),
        _chunk(content="42"),
    ]
    msg, _ = await LLMClient._consume_native_stream(_aiter(chunks), None)
    assert msg.reasoning_content == "Let me think"
    assert msg.content == "42"                      # answer text stays separate


@pytest.mark.asyncio
async def test_stream_without_reasoning_reports_none():
    """A plain prose stream carries no thinking: the field is None, not ""."""
    from localharness.provider.client import LLMClient

    msg, _ = await LLMClient._consume_native_stream(_aiter([_chunk(content="hi")]), None)
    assert msg.reasoning_content is None


@pytest.mark.asyncio
async def test_ollama_thinking_stream_measures_the_whole_generation(tmp_path, monkeypatch):
    """The tok/s consequence, end to end. 21 exact completion tokens are spent over 2.0s, of
    which 1.5s is thinking. Reading only `reasoning_content` opened the window at the answer
    delta — 0.5s — and reported 40 tok/s for a 10 tok/s generation."""
    from localharness.provider import client as client_mod
    from localharness.provider.speed_stats import default_speed_stats_path, median_tps

    c = _mk_speed_client(tmp_path, monkeypatch, provider_type="ollama")
    clock = _TickClock()
    monkeypatch.setattr(client_mod.time, "monotonic", clock)
    final = NS(usage=NS(prompt_tokens=8, completion_tokens=21, total_tokens=29), choices=[])
    chunks = [
        _think_chunk("reasoning", "Let me", content=""),   # t=100.5 — window opens here
        _think_chunk("reasoning", " think", content=""),   # t=101.0
        _think_chunk("reasoning", "", content=""),         # t=101.5
        _chunk(content="42"),                              # t=102.0 — answer
        final,                                             # t=102.5 — done
    ]

    async def fake_create(**kwargs):
        return clock.stream(chunks)

    c._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, usage = await c._create_and_consume({}, stream=True)
    assert msg.content == "42"
    assert msg.reasoning_content == "Let me think"
    assert c.last_gen_tps == 10.0                          # 20 intervals / 2.0s, not / 0.5s
    assert median_tps(default_speed_stats_path(), "ollama", "m") == 10.0


@pytest.mark.asyncio
async def test_non_streaming_normalizes_ollama_reasoning_onto_reasoning_content(tmp_path, monkeypatch):
    """Non-streaming: Ollama's `reasoning` is copied onto the field the rest of the code reads."""
    c = _mk_speed_client(tmp_path, monkeypatch, provider_type="ollama")
    message = NS(content="42", tool_calls=None, reasoning="Let me think")

    async def fake_create(**kwargs):
        return NS(choices=[NS(message=message, finish_reason="stop")], usage=None)

    c._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _ = await c._create_and_consume({}, stream=False)
    assert msg.reasoning_content == "Let me think"
    assert msg.content == "42"


@pytest.mark.asyncio
async def test_non_streaming_leaves_an_existing_reasoning_content_alone(tmp_path, monkeypatch):
    """vLLM/llama.cpp shape is untouched — normalization only fills an ABSENT field."""
    c = _mk_speed_client(tmp_path, monkeypatch)
    message = NS(content="42", tool_calls=None,
                 reasoning_content="native", reasoning="should not win")

    async def fake_create(**kwargs):
        return NS(choices=[NS(message=message, finish_reason="stop")], usage=None)

    c._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _ = await c._create_and_consume({}, stream=False)
    assert msg.reasoning_content == "native"


@pytest.mark.asyncio
async def test_non_streaming_normalization_survives_a_message_that_rejects_new_fields(
    tmp_path, monkeypatch
):
    """Same best-effort contract as the finish_reason surfacing (#77): the SDK message is a
    pydantic model that may refuse an unknown attribute. A refusal leaves the field unset —
    it never fails the completion."""
    class _Strict:
        __slots__ = ("content", "tool_calls", "reasoning")

        def __init__(self):
            self.content, self.tool_calls, self.reasoning = "42", None, "thinking"

    c = _mk_speed_client(tmp_path, monkeypatch, provider_type="ollama")
    message = _Strict()

    async def fake_create(**kwargs):
        return NS(choices=[NS(message=message, finish_reason="stop")], usage=None)

    c._client = NS(chat=NS(completions=NS(create=fake_create)))
    msg, _ = await c._create_and_consume({}, stream=False)
    assert msg.content == "42"
    assert getattr(msg, "reasoning_content", None) is None
